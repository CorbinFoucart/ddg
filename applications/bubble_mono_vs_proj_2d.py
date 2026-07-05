"""Boussinesq rising-bubble: monolithic-Newton vs block-Schur projection.

Runs the SAME buoyancy-driven rising-bubble problem under BOTH BDM_k
velocity-pressure solvers on the SAME mesh, same order/nu/kappa/tau,
marching both from rest with the same initial blob. The scalar transport
and the buoyancy body force are identical between the two runs; only the
momentum solve differs:

  monolithic : transient coupled saddle-point Newton (dense LU),
               reused from ns_bdm_rising_bubble_2d._newton_momentum_step
  projection : Fehn-style block-Schur step (explicit-extrapolated
               convection), projection_step_bdm_k_2d, buoyancy as f_body.

Defaults MATCH the known-good rising-bubble configuration of
``ns_rising_bubble_2d.py`` (the config that produced the paper figure):
order 3, tall 1x2 domain, Atwood-scaled signed-scalar buoyancy
``fy = -A g c`` (c<0 light rises), signed cosine-bump IC, nu=kappa=1e-4,
dt=3e-3. The point is to show the patched projection reproduces the
monolithic bubble physics on a genuinely buoyant transient -- the bubble
analogue of the Schaefer-Turek cylinder comparison.

Use ``--solvers proj`` for a cheap projection-only stability smoke before
committing to the (expensive, order-3 dense-LU) monolithic side.

Run on the remote (needs float64):
    python applications/bubble_mono_vs_proj_2d.py --solvers proj \
        --kx 8 --ky 16 --tfinal 2.0 --snap-every 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "applications"))

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d  # noqa: E402
from ddg.dg2d.bdm import BDM  # noqa: E402
from ddg.dg2d.coupled_ns_bdm import BDF_COEFFS, bdf_history_k  # noqa: E402
from ddg.dg2d.scalar_transport import (  # noqa: E402
    build_scalar_operators, scalar_transport_step_2d,
)
from ddg.dg2d.incompressible_ns import BDF1, BDF2  # noqa: E402
from ddg.dg2d.sipg import BC_NEUMANN  # noqa: E402
from ddg.dg2d.projection_ns_bdm import (  # noqa: E402
    make_projection_solvers, projection_step_bdm_k_2d,
)
from ddg.dg2d.bdm_multigrid import make_coupled_velocity_mg_pc  # noqa: E402

# Reuse the bubble app's physics helpers verbatim (single source of truth).
from ns_bdm_rising_bubble_2d import (  # noqa: E402
    _build_space, _bdm_velocity_at_nodes, _buoyancy_body_force,
    _newton_momentum_step,
)
from projection_vs_monolithic_verification import divergence_norm  # noqa: E402
from ddg.dg2d.scalar_limiter import zhang_shu_limit  # noqa: E402
from ddg.dg2d.incompressible_ns import _div_2d  # noqa: E402  (diagnostic only)


def _cosine_bubble(mesh, cx, cy, rb):
    """Signed cosine-bump bubble matching ns_rising_bubble_2d:
    c = -0.5 (1 + cos(pi r / r_b)) inside r<r_b, 0 outside (c<0 = light)."""
    x = np.asarray(mesh.x); y = np.asarray(mesh.y)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    bump = -0.5 * (1.0 + np.cos(np.pi * np.clip(r, 0.0, rb) / rb))
    return jnp.asarray(np.where(r < rb, bump, 0.0))


def _centroid_y(c_nodal, y_nodal):
    """c-weighted mean height of the blob: sum(c*y)/sum(c). Both solvers
    share the identical nodal set, so this is a fair rise-height proxy.
    (c is signed; the weighting is by c, which is fine for a relative
    mono-vs-proj comparison on the same nodes.)"""
    c = np.asarray(c_nodal); y = np.asarray(y_nodal)
    s = float(c.sum())
    return float((c * y).sum() / s) if s != 0.0 else float("nan")


def _build_scalar_ops(mesh, *, dt, kappa):
    bc = jnp.full((mesh.K, 3), BC_NEUMANN, dtype=jnp.int32)
    return (build_scalar_operators(mesh, dt=dt, kappa=kappa, g0=BDF1.g0,
                                   diffusion_bc_types=bc, tau_scale=200.0),
            build_scalar_operators(mesh, dt=dt, kappa=kappa, g0=BDF2.g0,
                                   diffusion_bc_types=bc, tau_scale=200.0))


def run(args):
    solvers = (["mono", "proj"] if args.solvers == "both"
               else [args.solvers])
    VX, VY, EToV = uniform_rectangle_mesh_2d(
        0.0, args.W, 0.0, args.H, args.kx, args.ky)
    mesh = build_mesh_2d(VX, VY, EToV, args.order)
    V, u_bc, u_bc_vec, bc_mask = _build_space(mesh, args.order)
    dim_p = args.order * (args.order + 1) // 2
    n_u, n_p = V.total_dofs, V.K * dim_p
    buoy_scale = -args.atwood * args.gravity   # fy = -A g c ; _buoy gives (0,c)
    print(f"BDM_{args.order} bubble compare [{','.join(solvers)}]: "
          f"{args.kx}x{args.ky} on {args.W}x{args.H} (K={V.K}), "
          f"DOFs {n_u + n_p} ({n_u} u + {n_p} p), dt={args.dt}, "
          f"tfinal={args.tfinal}, nu={args.nu}, kappa={args.kappa}, "
          f"A={args.atwood}, g={args.gravity}", flush=True)

    ops_c_bdf1, ops_c_bdf2 = _build_scalar_ops(
        mesh, dt=args.dt, kappa=args.kappa)
    ps1, hi1 = make_projection_solvers(
        V, dt=args.dt, nu=args.nu, order=1, bc_mask=bc_mask,
        tau_scale=args.tau_scale)
    ps2, hi2 = make_projection_solvers(
        V, dt=args.dt, nu=args.nu, order=2, bc_mask=bc_mask,
        tau_scale=args.tau_scale)

    # Monolithic coupled-Newton uses a MATRIX-FREE JVP + GMRES solve
    # preconditioned by the coupled hp-MG block PC (velocity hp-MG +
    # P_{k-1} pressure Schur), built once per BDF order outside the time
    # loop -- the same path the standalone bubble app uses. This avoids the
    # dense jacfwd Jacobian (n=n_u+n_p), which would need O((n_u+n_p)^2)
    # memory and OOMs at 16x32/BDM_3 (~90 GB). The PC's velocity block
    # matches the transient operator alpha*M + nu*A (alpha = gamma0/dt).
    mono_pc_bdf1 = mono_pc_bdf2 = None
    if "mono" in solvers and not args.mono_lu:
        a1 = BDF_COEFFS[1]["gamma0"] / args.dt
        a2 = BDF_COEFFS[2]["gamma0"] / args.dt
        # sparse_fine=False: the dense-jax vertex-patch smoother is
        # traceable under gmres's lax.custom_linear_solve (which traces the
        # PC). The sparse (scipy lu_factor/np.asarray) smoother is NOT --
        # it raises TracerArrayConversionError inside the GMRES solve.
        #
        # cycle="V", nsm=1 (vs the steady default W/2): the W-cycle and
        # extra smoother sweeps are STATICALLY UNROLLED into the GMRES
        # while_loop body, and at 16x32/BDM_3 that fused module overflows
        # the CPU LLVM compiler ("Cannot allocate memory" / "Failed to
        # materialize dot_general kernel"). A single V-cycle with one
        # pre/post sweep keeps the module compilable. This is sufficient
        # here because the transient velocity block alpha*M + nu*A is
        # strongly mass-dominated at dt=3e-3 (alpha=gamma0/dt ~ 333-500),
        # hence very well conditioned -- a light PC still converges fast.
        mono_pc_bdf1 = make_coupled_velocity_mg_pc(
            V, mesh, n_h=0, nu=args.nu, tau_scale=args.tau_scale,
            nsm=1, cycle="V", sparse_fine=False, velocity_alpha=a1)
        mono_pc_bdf2 = make_coupled_velocity_mg_pc(
            V, mesh, n_h=0, nu=args.nu, tau_scale=args.tau_scale,
            nsm=1, cycle="V", sparse_fine=False, velocity_alpha=a2)
        print(f"  built coupled hp-MG PCs for monolithic GMRES "
              f"(BDF1 alpha={a1:.1f}, BDF2 alpha={a2:.1f})", flush=True)

    c0 = _cosine_bubble(mesh, args.cx, args.cy, args.rb)
    y_nodal = mesh.y
    # Zhang-Shu bound-preserving limiter (validated in test_scalar_limiter):
    # clamp the advected scalar to its physical range each step. Reference
    # mass M = inv(V V^T); bounds are the initial scalar's [min, max].
    lim_mass = jnp.linalg.inv(mesh.V @ mesh.V.T) if args.limiter else None
    c_lo = float(jnp.min(c0)); c_hi = float(jnp.max(c0))
    state = {
        s: dict(u=jnp.zeros(n_u), u_hist=[jnp.zeros(n_u), jnp.zeros(n_u)],
                p=jnp.zeros((V.K, dim_p)), c=c0, c_prev=c0,
                Nc=jnp.zeros_like(c0), wall=0.0, peak_div=0.0)
        for s in solvers
    }
    snaps = {s: [] for s in solvers}
    series = dict(t=[], rel_c_diff=[],
                  **{f"centroid_{s}": [] for s in solvers},
                  **{f"peak_div_{s}": [] for s in solvers})

    def snapshot(s, t):
        ux_n, uy_n = _bdm_velocity_at_nodes(V, state[s]["u"])
        umag = np.sqrt(np.asarray(ux_n) ** 2 + np.asarray(uy_n) ** 2)
        snaps[s].append(dict(t=float(t), c=np.asarray(state[s]["c"]),
                             u_mag=umag))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(steps_done):
        sd = max(1, steps_done)
        meta = dict(order=args.order, kx=args.kx, ky=args.ky, W=args.W,
                    H=args.H, K=int(V.K), n_u=n_u, n_p=n_p, dt=args.dt,
                    tfinal=args.tfinal, nu=args.nu, kappa=args.kappa,
                    atwood=args.atwood, gravity=args.gravity,
                    tau_scale=args.tau_scale, solvers=solvers,
                    n_steps=n_steps, steps_done=steps_done, series=series)
        for s in solvers:
            meta[f"per_step_ms_{s}"] = state[s]["wall"] / sd * 1e3
            meta[f"peak_div_{s}"] = float(state[s]["peak_div"])
        meta["final_rel_c_diff"] = (series["rel_c_diff"][-1]
                                    if series["rel_c_diff"] else None)
        (out_dir / "bubble_mono_vs_proj.json").write_text(
            json.dumps(meta, indent=2))
        npz = dict(x=np.asarray(mesh.x), y=np.asarray(mesh.y),
                   t=np.array([sn["t"] for sn in snaps[solvers[0]]]))
        for s in solvers:
            npz[f"c_{s}"] = np.stack([sn["c"] for sn in snaps[s]])
            npz[f"umag_{s}"] = np.stack([sn["u_mag"] for sn in snaps[s]])
        np.savez_compressed(out_dir / "bubble_mono_vs_proj_snapshots.npz", **npz)
        return meta

    for s in solvers:
        snapshot(s, 0.0)
    n_steps = int(np.ceil(args.tfinal / args.dt))
    t = 0.0
    for step in range(n_steps):
        order_eff = 1 if step == 0 else 2
        ops_c = ops_c_bdf1 if order_eff == 1 else ops_c_bdf2
        bdf_s = BDF1 if order_eff == 1 else BDF2
        velocity_alpha = BDF_COEFFS[order_eff]["gamma0"] / args.dt
        ps = ps1 if order_eff == 1 else ps2
        hi = hi1 if order_eff == 1 else hi2
        frozen = (args.freeze_vel_after is not None
                  and t >= args.freeze_vel_after)

        for s in solvers:
            st = state[s]
            ux_n, uy_n = _bdm_velocity_at_nodes(V, st["u"])
            c_new, Nc_new = scalar_transport_step_2d(
                mesh, st["c"], st["c_prev"], ux_n, uy_n, st["Nc"],
                dt=args.dt, kappa=args.kappa, bdf=bdf_s, ops=ops_c)
            if args.limiter:
                c_new = zhang_shu_limit(c_new, lim_mass, c_lo, c_hi)
            st["c_prev"], st["c"], st["Nc"] = st["c"], c_new, Nc_new

            # Experiment 1 (feedback isolation): once frozen, stop updating
            # the velocity -- the scalar then advects under a fixed, realistic
            # field with NO buoyancy feedback.
            if not frozen:
                f_buoy = buoy_scale * _buoyancy_body_force(V, st["c"])
                t0 = time.time()
                if s == "mono":
                    f_hist = bdf_history_k(V, st["u_hist"], dt=args.dt,
                                           order=order_eff)
                    pc_this = (None if args.mono_lu
                               else (mono_pc_bdf1 if order_eff == 1
                                     else mono_pc_bdf2))
                    u_new, p_new, _conv, _it = _newton_momentum_step(
                        V, nu=args.nu, tau_scale=args.tau_scale, bc_mask=bc_mask,
                        velocity_alpha=velocity_alpha,
                        f_velocity=f_hist + f_buoy,
                        u_prev=st["u"], p_prev=st["p"], coupled_pc=pc_this)
                else:
                    u_new, p_new = projection_step_bdm_k_2d(
                        V, st["u_hist"], dt=args.dt, nu=args.nu, order=order_eff,
                        pressure_solver=ps, h_inv=hi,
                        u_bc_global=None, u_bc_vec=None, bc_mask=bc_mask,
                        tau_scale=args.tau_scale, f_body=f_buoy)
                u_new.block_until_ready()
                st["wall"] += time.time() - t0
                st["u_hist"] = [u_new, st["u_hist"][0]]
                st["u"], st["p"] = u_new, p_new
                st["peak_div"] = max(st["peak_div"], divergence_norm(V, u_new))
        t += args.dt

        if (step + 1) % args.snap_every == 0 or step == n_steps - 1:
            series["t"].append(float(t))
            for s in solvers:
                series[f"centroid_{s}"].append(
                    _centroid_y(state[s]["c"], y_nodal))
                series[f"peak_div_{s}"].append(float(state[s]["peak_div"]))
            if len(solvers) == 2:
                cm = np.asarray(state["mono"]["c"])
                cp = np.asarray(state["proj"]["c"])
                rel = float(np.linalg.norm(cm - cp)
                            / (np.linalg.norm(cm) + 1e-30))
            else:
                rel = float("nan")
            series["rel_c_diff"].append(rel)
            for s in solvers:
                snapshot(s, t)
            dump(step + 1)
            def _veldiv(s):  # nodal divergence of the advecting velocity
                uxn, uyn = _bdm_velocity_at_nodes(V, state[s]["u"])
                return float(jnp.linalg.norm(_div_2d(mesh, uxn, uyn)))
            msg = (f"  step {step+1}/{n_steps} t={t:.3f}  "
                   + "  ".join(
                       f"{s}: cen={series[f'centroid_{s}'][-1]:.4f} "
                       f"|c|max={float(np.abs(np.asarray(state[s]['c'])).max()):.3f} "
                       f"pdiv={state[s]['peak_div']:.1e} "
                       f"veldiv={_veldiv(s):.2e}"
                       for s in solvers))
            if len(solvers) == 2:
                msg += f"  rel||dc||={rel:.2e}"
            print(msg, flush=True)

    meta = dump(n_steps)
    print(f"\nwrote {out_dir/'bubble_mono_vs_proj.json'}")
    print(f"wrote {out_dir/'bubble_mono_vs_proj_snapshots.npz'}")
    for s in solvers:
        print(f"  {s}: per-step {meta[f'per_step_ms_{s}']:.1f} ms  "
              f"peak_div {meta[f'peak_div_{s}']:.2e}")
    if len(solvers) == 2:
        print(f"  final rel ||c_mono - c_proj|| = {meta['final_rel_c_diff']:.3e}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=8)
    ap.add_argument("--ky", type=int, default=16)
    ap.add_argument("--W", type=float, default=1.0)
    ap.add_argument("--H", type=float, default=2.0)
    ap.add_argument("--nu", type=float, default=1.0e-4)
    ap.add_argument("--kappa", type=float, default=1.0e-4)
    ap.add_argument("--dt", type=float, default=3.0e-3)
    ap.add_argument("--tfinal", type=float, default=4.0)
    ap.add_argument("--atwood", type=float, default=0.5)
    ap.add_argument("--gravity", type=float, default=1.0)
    ap.add_argument("--rb", type=float, default=0.25)
    ap.add_argument("--cx", type=float, default=0.5)
    ap.add_argument("--cy", type=float, default=0.7)   # 0.35 * H
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--freeze-vel-after", type=float, default=None,
                    help="EXPERIMENT: after this sim time, stop the momentum "
                         "solve and hold the velocity fixed; the scalar then "
                         "advects passively (no buoyancy feedback).")
    ap.add_argument("--limiter", action="store_true",
                    help="apply the Zhang-Shu bound-preserving limiter to "
                         "the advected scalar each step (clamp to its "
                         "physical range, preserving cell means)")
    ap.add_argument("--solvers", choices=("both", "mono", "proj"),
                    default="both")
    ap.add_argument("--mono-lu", action="store_true", default=True,
                    help="monolithic uses a dense-LU Newton solve "
                         "(coupled_pc=None). Default True: the iterative "
                         "GMRES+hp-MG path fuses the unrolled V-cycle into "
                         "one kernel that overflows the CPU LLVM compiler "
                         "at order 3 ('Failed to materialize dot kernel').")
    ap.add_argument("--no-mono-lu", dest="mono_lu", action="store_false",
                    help="use the iterative GMRES+hp-MG monolithic path "
                         "instead (may OOM the compiler at order 3).")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "output" / "bubble_mono_vs_proj")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
