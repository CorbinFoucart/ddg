"""High-resolution flow past a cylinder using the BDM_k block-Schur
projection solver (:mod:`ddg.dg2d.projection_ns_bdm`).

A streamlined version of ``ns_bdm_flow_past_cylinder_2d.py``: same
geometry, BC classification, and snapshot/render machinery, but with
the time integrator replaced by the projection step instead of the
monolithic Newton-Picard with iterative inner solves. Designed for
remote GPU dispatch where we want shedding videos at production
resolution.

Per-step cost (rough numbers on a 16-vCPU Intel Xeon, BDM_2,
K = 432): monolithic Newton with n_newton=1 is ~150 ms / step, the
projection is ~5 ms / step (31x speedup; cf.\\
:func:`projection_vs_monolithic_cylinder.main`). On the L4 GPU at
production resolution K ~ 2000-5000, monolithic LU hits the 24 GB
VRAM ceiling; the projection's block factors stay well below.

Usage
-----
::

    # Local smoke test (~30 s)
    python applications/ns_bdm_cylinder_projection_2d.py \\
        --smoke-geometry --order 2 --kx 20 --ky 10 \\
        --dt 0.005 --t-final 0.1 --snapshot-stride 5

    # Production shedding video on L4 (vortex shedding at Re=100,
    # T_final = 8.0 captures ~12 shedding cycles at St ~ 0.16)
    python applications/ns_bdm_cylinder_projection_2d.py \\
        --order 2 --kx 22 --ky 4 --n-radial 4 --r-outer 0.08 \\
        --dt 0.005 --t-final 8.0 --t-ramp 0.5 \\
        --snapshot-stride 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "applications"))

from ddg.dg2d.bdm import (
    BDM, bdm_k_basis_at_points, bdm_k_divergence_apply,
    bdm_k_evaluate_boundary_velocity, bdm1_gather,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
)
from ddg.dg2d.curved import apply_circular_hole_2d
from ddg.dg2d.mesh import build_mesh_2d, channel_with_circle_mesh_2d
from ddg.dg2d.projection_ns_bdm import (
    make_projection_solvers, projection_step_bdm_k_2d,
)

# Reuse the geometry constants + helpers from the canonical app.
import ns_bdm_flow_past_cylinder_2d as _cyl
from ns_bdm_flow_past_cylinder_2d import (
    classify_boundary_faces, make_bc_fn, _select_geometry,
    _bdm_velocity_at_nodes, _nodal_vorticity, _ramp_factor,
)


def _bc_fn_at_time(t, T_ramp):
    """BC velocity field at a given physical time, with cosine ramp."""
    rf = float(_ramp_factor(t, T_ramp))
    return make_bc_fn(ramp_factor=rf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=22)
    ap.add_argument("--ky", type=int, default=4)
    ap.add_argument("--smoke-geometry", action="store_true",
                    help="Use a smaller mesh-friendly geometry "
                         "(channel 1.0x0.5) instead of Schaefer-Turek.")
    ap.add_argument("--n-radial", type=int, default=4,
                    help="annular layers between cylinder and bounding sq")
    ap.add_argument("--r-outer", type=float, default=0.08,
                    help="bounding-square half side around cylinder")
    ap.add_argument("--radial-growth", type=float, default=1.3)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--t-final", type=float, default=0.05,
                    help="smoke value -- bump up for production")
    ap.add_argument("--t-ramp", type=float, default=0.5)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    ap.add_argument("--snapshot-stride", type=int, default=0,
                    help="Save velocity at mesh nodes every N steps; "
                         "0 = disabled.")
    ap.add_argument("--checkpoint-stride", type=int, default=200,
                    help="Number of collected snapshots between "
                         "writes of the persistent snapshots.npz. "
                         "Survives crashes; replotting only needs "
                         "this file plus the mesh.npz.")
    ap.add_argument("--render-mp4", action="store_true",
                    help="Render the wake mp4 at end of run. Off by "
                         "default on remote runs; replotting uses the "
                         "saved snapshots.npz.")
    ap.add_argument("--nu", type=float, default=None,
                    help="Override the geometry-default viscosity. "
                         "Default = 1e-3 (Schaefer-Turek Re=100). "
                         "Set to 5e-4 for Re=200 etc.")
    ap.add_argument("--inflow-pert", type=float, default=0.0,
                    help="Asymmetric inflow perturbation magnitude. "
                         "When > 0, the inflow gets a small "
                         "**one-sided** y-velocity component "
                         "alpha * sin(pi y / H) (peaks at y = H/2, "
                         "same sign everywhere) added to the "
                         "parabolic profile -- breaks the up-down "
                         "channel symmetry across the cylinder. "
                         "Antisymmetric perturbations (sin 2 pi y/H) "
                         "do NOT work because they merely push fluid "
                         "symmetrically over and under the centered "
                         "cylinder. 0.05 is a good default for Re=100; "
                         "smaller for higher Re.")
    ap.add_argument("--stem", type=str, default="cylinder_projection",
                    help="Output file stem.")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    _select_geometry(args.smoke_geometry)

    # Geometry constants are now patched on the imported module; pull
    # the current values back from there.
    X_MIN = _cyl.X_MIN; X_MAX = _cyl.X_MAX
    Y_MIN = _cyl.Y_MIN; Y_MAX = _cyl.Y_MAX
    CX = _cyl.CX; CY = _cyl.CY; RADIUS = _cyl.RADIUS
    D_CYL = _cyl.D_CYL
    U_MAX = _cyl.U_MAX; U_MEAN = _cyl.U_MEAN
    NU = args.nu if args.nu is not None else _cyl.NU
    Re = U_MEAN * D_CYL / NU
    print(f"BDM_{args.order} cylinder (projection, T_final={args.t_final}): "
          f"Re={Re:.0f}, Kx={args.kx}, Ky={args.ky}, dt={args.dt}",
          flush=True)

    # --- Mesh + curving ---
    VX, VY, EToV = channel_with_circle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, args.kx, args.ky,
        cx=CX, cy=CY, radius=RADIUS,
        R_outer=args.r_outer, n_radial=args.n_radial,
        radial_growth=args.radial_growth,
    )
    mesh = build_mesh_2d(VX, VY, EToV, args.order)
    mesh = apply_circular_hole_2d(
        mesh, cx=CX, cy=CY, radius=RADIUS,
        blend_interior=True, detection_tol=0.05,
    )
    J = np.asarray(mesh.J)
    n_inv = int((J <= 0).sum())
    if n_inv > 0:
        raise RuntimeError(f"Mesh has {n_inv} inverted nodes after curving")
    V = BDM.from_mesh(mesh, order=args.order)
    n_u = V.total_dofs
    dim_p = args.order * (args.order + 1) // 2
    n_p = V.K * dim_p
    print(f"  mesh: K={mesh.K}  n_u={n_u}  n_p={n_p}", flush=True)

    is_inflow, is_outflow, is_wall_or_cyl, is_cyl = classify_boundary_faces(mesh)
    is_natural_face = is_outflow
    dirichlet_face = is_inflow | is_wall_or_cyl
    bc_mask = bdm1_boundary_dofs_mask(V, dirichlet_face=dirichlet_face)
    print(f"  Dirichlet DOFs: {int(bc_mask.sum())}/{n_u}", flush=True)

    # --- BC samplers at full ramp (the projection solvers are built
    # once with the at-T_ramp lift). Inflow is the only BC component
    # that varies with the ramp factor; for the smoke + shedding video
    # we just hold the lift fixed at the post-ramp profile and let the
    # BDF history absorb the start-up transient.
    # Optional asymmetric inflow perturbation: adds a small uy on
    # the inflow profile so the channel symmetry is broken. With
    # CY=Y/2 (centered cylinder) the symmetric solution is a
    # spurious attractor that hides the von-Karman shedding;
    # an O(0.01-0.05) perturbation is enough to seed the
    # instability and lets the wake grow to its natural amplitude.
    H = Y_MAX - Y_MIN
    Y_MIN_local = Y_MIN

    def _bc_with_pert(x, y):
        yfrac = (y - Y_MIN_local) / H
        parabolic_ux = 4.0 * U_MAX * yfrac * (1.0 - yfrac)
        on_inflow = jnp.abs(x - X_MIN) < 1e-3
        ux = jnp.where(on_inflow, parabolic_ux, 0.0)
        # One-sided perturbation (same sign for all y) so the kick
        # is asymmetric across the cylinder's symmetry plane y=H/2.
        # sin(pi yfrac) peaks at yfrac=0.5 and is positive throughout.
        uy_pert = args.inflow_pert * jnp.sin(jnp.pi * yfrac)
        uy = jnp.where(on_inflow, uy_pert, 0.0)
        return ux, uy

    bc_fn_full = (_bc_with_pert if args.inflow_pert > 0.0
                   else make_bc_fn(ramp_factor=1.0))
    u_bc_full = bdm1_evaluate_dirichlet_bc(V, bc_fn_full)
    u_bc_vec_full = bdm_k_evaluate_boundary_velocity(V, bc_fn_full)

    # --- Build projection solvers (BDF1 bootstrap + BDF2 production) ---
    print("  building BDF1 + BDF2 projection solvers ...", flush=True)
    t0 = time.time()
    pressure_solver_bdf1, h_inv_bdf1 = make_projection_solvers(
        V, dt=args.dt, nu=NU, order=1, tau_scale=args.tau_scale,
        bc_mask=bc_mask, is_natural_face=is_natural_face,
    )
    pressure_solver_bdf2, h_inv_bdf2 = make_projection_solvers(
        V, dt=args.dt, nu=NU, order=2, tau_scale=args.tau_scale,
        bc_mask=bc_mask, is_natural_face=is_natural_face,
    )
    print(f"  solver build: {time.time() - t0:.1f}s", flush=True)

    def make_step(order_eff):
        pressure_solver = (
            pressure_solver_bdf1 if order_eff == 1 else pressure_solver_bdf2
        )
        h_inv = h_inv_bdf1 if order_eff == 1 else h_inv_bdf2

        # u_bc / u_bc_vec are now traced step arguments so the time loop
        # can ramp the inflow (scale them by the cosine ramp factor).
        # The solver factorisations (H, S) are BC-independent, so this
        # adds no recompilation. The impulsive full-strength start
        # otherwise blows up the explicit-convection update regardless
        # of dt; ramping matches the monolithic driver's gentle start.
        def step(u_n, u_nm1, u_bc, u_bc_vec):
            u_hist = [u_n] if order_eff == 1 else [u_n, u_nm1]
            return projection_step_bdm_k_2d(
                V, u_hist, dt=args.dt, nu=NU, order=order_eff,
                pressure_solver=pressure_solver, h_inv=h_inv,
                u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
                is_natural_face=is_natural_face,
                tau_scale=args.tau_scale,
            )

        return jax.jit(step)

    step_bdf1 = make_step(1)
    step_bdf2 = make_step(2)

    # --- Persist mesh + geometry once, at run start ---
    mesh_path = args.out_dir / f"{args.stem}_mesh.npz"
    np.savez_compressed(
        mesh_path,
        x=np.asarray(mesh.x), y=np.asarray(mesh.y),
        VX=np.asarray(mesh.VX), VY=np.asarray(mesh.VY),
        EToV=np.asarray(mesh.EToV),
        rx=np.asarray(mesh.rx), ry=np.asarray(mesh.ry),
        sx=np.asarray(mesh.sx), sy=np.asarray(mesh.sy),
        J=np.asarray(mesh.J),
        Dr=np.asarray(mesh.Dr), Ds=np.asarray(mesh.Ds),
        is_cyl=np.asarray(is_cyl),
        CX=CX, CY=CY, RADIUS=RADIUS,
        X_MIN=X_MIN, X_MAX=X_MAX, Y_MIN=Y_MIN, Y_MAX=Y_MAX,
        order=args.order, kx=args.kx, ky=args.ky,
        dt=args.dt, t_final=args.t_final, Re=Re,
    )
    print(f"  wrote {mesh_path}", flush=True)

    def _dump_snapshots(snapshots, path):
        if not snapshots:
            return
        ts = np.asarray([s[0] for s in snapshots], dtype=np.float64)
        uxs = np.stack([s[1] for s in snapshots], axis=0).astype(np.float32)
        uys = np.stack([s[2] for s in snapshots], axis=0).astype(np.float32)
        np.savez_compressed(path, t=ts, ux=uxs, uy=uys)

    snap_path = args.out_dir / f"{args.stem}_snapshots.npz"

    # --- Time-stepping loop ---
    n_steps = int(round(args.t_final / args.dt))
    print(f"  marching {n_steps} steps ...", flush=True)
    u_n = jnp.zeros(n_u); u_nm1 = jnp.zeros(n_u)
    snapshots = []
    ke_series = []; div_series = []; times = []
    t_per_step = []
    t_start = time.time()
    for s in range(n_steps):
        t = (s + 1) * args.dt
        order_eff = 1 if s == 0 else 2
        step_fn = step_bdf1 if order_eff == 1 else step_bdf2
        # Cosine-ramp the inflow over [0, t_ramp] so the explicit-
        # convection update sees a gentle start (the BC scales linearly
        # with the ramp factor).
        rf = _ramp_factor(t, args.t_ramp)
        u_bc_t = rf * u_bc_full
        u_bc_vec_t = rf * u_bc_vec_full
        t_step = time.time()
        u_new, _ = step_fn(u_n, u_nm1, u_bc_t, u_bc_vec_t)
        u_new.block_until_ready()
        t_per_step.append(time.time() - t_step)
        u_nm1, u_n = u_n, u_new

        ke = float(0.5 * jnp.sum(u_n ** 2))
        divn = float(jnp.linalg.norm(bdm_k_divergence_apply(V, u_n)))
        ke_series.append(ke); div_series.append(divn); times.append(t)
        if not jnp.isfinite(jnp.linalg.norm(u_n)):
            print(f"  *** NaN at step {s}, t={t:.3f}", flush=True)
            break

        if args.snapshot_stride > 0 and (s % args.snapshot_stride == 0):
            ux_n, uy_n = _bdm_velocity_at_nodes(V, u_n)
            snapshots.append((t, np.asarray(ux_n), np.asarray(uy_n)))
            if (len(snapshots) % args.checkpoint_stride) == 0:
                _dump_snapshots(snapshots, snap_path)
                print(f"  checkpointed {len(snapshots)} snapshots -> "
                      f"{snap_path}", flush=True)

        if s % max(1, n_steps // 20) == 0:
            print(f"  step {s:5d}/{n_steps}  t={t:.3f}  "
                  f"||u||={float(jnp.linalg.norm(u_n)):.3e}  "
                  f"||Du||={divn:.2e}  "
                  f"ms/step={1e3 * t_per_step[-1]:.1f}",
                  flush=True)
    total_wall = time.time() - t_start
    print(f"  total wall: {total_wall:.1f}s  "
          f"steady ms/step: {1e3 * float(np.median(t_per_step[5:])):.2f}",
          flush=True)

    stem = args.stem
    # --- Diagnostics JSON ---
    json_path = args.out_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky,
            K=int(mesh.K), n_u=int(n_u),
            dt=args.dt, t_final=args.t_final, n_steps=n_steps,
            Re=Re, total_wall=total_wall,
            steady_per_step_ms=1e3 * float(np.median(t_per_step[5:])),
            t=times, ke=ke_series, div=div_series,
        ), f, indent=2)
    print(f"  wrote {json_path}", flush=True)

    # --- Diagnostics PNG ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(times, ke_series, "k-"); axes[0].set_yscale("log")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("kinetic energy")
    axes[0].grid(alpha=0.3, which="both")
    axes[1].semilogy(times, div_series, "k-")
    axes[1].set_xlabel("t"); axes[1].set_ylabel(r"$\|D u\|$")
    axes[1].set_title("discrete divergence (should be ~LU roundoff)")
    axes[1].grid(alpha=0.3, which="both")
    fig.suptitle(f"cylinder projection: BDM_{args.order}, K={int(mesh.K)}, "
                  f"Re={Re:.0f}, n_steps={n_steps}")
    out_png = args.out_dir / f"{stem}_diag.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)

    # --- Final snapshot dump ---
    if snapshots:
        _dump_snapshots(snapshots, snap_path)
        print(f"  wrote {snap_path}  ({len(snapshots)} snapshots)",
              flush=True)

    # --- Wake mp4 (optional; replot from snapshots.npz instead) ---
    if snapshots and args.render_mp4:
        print(f"  rendering wake mp4 from {len(snapshots)} snapshots ...",
              flush=True)
        from matplotlib.animation import FuncAnimation
        px = np.asarray(mesh.x).T.reshape(-1)
        py = np.asarray(mesh.y).T.reshape(-1)
        theta = np.linspace(0, 2 * np.pi, 80)
        cyl_x = CX + RADIUS * np.cos(theta)
        cyl_y = CY + RADIUS * np.sin(theta)
        omegas, speeds = [], []
        for (_t, ux, uy) in snapshots:
            omegas.append(np.asarray(_nodal_vorticity(mesh, ux, uy)))
            speeds.append(np.sqrt(ux ** 2 + uy ** 2))
        w_max = max(np.percentile(np.abs(w), 98) for w in omegas)
        s_max = max(np.percentile(s, 99) for s in speeds)
        levels_w = np.linspace(-w_max, w_max, 31)
        levels_s = np.linspace(0, s_max, 21)
        fig2, axes = plt.subplots(
            2, 1, figsize=(13, 5), sharex=True, constrained_layout=True,
        )

        def draw(i):
            for ax in axes:
                ax.clear()
            t_i, ux_i, uy_i = snapshots[i]
            w_i = omegas[i].T.reshape(-1)
            s_i = speeds[i].T.reshape(-1)
            axes[0].tricontourf(px, py, w_i, levels=levels_w,
                                cmap="RdBu_r", extend="both")
            axes[0].fill(cyl_x, cyl_y, color="black")
            axes[0].set_aspect("equal")
            axes[0].set_title(f"vorticity $\\omega$    t = {t_i:.2f}")
            axes[1].tricontourf(px, py, s_i, levels=levels_s,
                                cmap="viridis", extend="max")
            axes[1].fill(cyl_x, cyl_y, color="black")
            axes[1].set_aspect("equal")
            axes[1].set_title("speed $|U|$")
            return []

        anim = FuncAnimation(fig2, draw, frames=len(snapshots),
                              interval=60, blit=False)
        out_mp4 = args.out_dir / f"{stem}.mp4"
        anim.save(str(out_mp4), writer="ffmpeg", dpi=100)
        plt.close(fig2)
        print(f"  wrote {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
