"""Flow past a circular cylinder: Schäfer-Turek 2D-2 benchmark.

The Schäfer-Turek 1996 ("Benchmark Computations of Laminar Flow Around
a Cylinder") problem 2D-2 is the canonical reference case for laminar
vortex shedding off a 2D cylinder. Published reference windows:

    C_{D,max} ∈ [3.2200, 3.2400]
    C_{L,max} ∈ [0.9800, 1.0200]
    Strouhal  ∈ [0.2950, 0.3050]

Setup:

    Channel:   [0, 2.2] × [0, 0.41]
    Cylinder:  centre (0.2, 0.2), radius 0.05
    Inflow:    parabolic U_x(y) = 4 · U_max · y · (H − y) / H²
               with U_max = 1.5 ⇒ U_mean = (2/3) · U_max = 1.0
    Outflow:   natural (zero traction)
    Walls / cylinder: no-slip
    ν = 0.001 ⇒ Re = U_mean · D / ν = 1.0 · 0.1 / 0.001 = 100

Mesh construction: :func:`channel_with_circle_mesh_2d` builds a
body-fitted O-grid around the cylinder — an annular ring of
``n_radial`` radial layers between the cylinder and a bounding
square of half-side ``R_outer``, plus the standard Cartesian
channel outside the bounding square. After meshing,
:func:`apply_circular_hole_2d` bends the innermost ring's faces
onto the true cylinder. With ``n_circ ≈ 56`` angular cells the
per-face bowing is ~0.5% of the cylinder radius — well below the
local triangle size — so no elements invert.

Note: the channel y-extent is 0.41 (Schäfer-Turek exact). The grid
must satisfy: ``hx`` divides ``cx = 0.2`` and ``R_outer = 0.1``,
and ``hy`` divides ``cy = 0.2`` and ``0.1``. With
``Kx = 88, Ky = 41``: ``hx = 0.025, hy = 0.01``. Cells outside the
annulus are anisotropic; inside the annulus they are body-fitted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, channel_with_circle_mesh_2d
from ddg.dg2d.curved import apply_circular_hole_2d
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OBSTACLE, BC_INS_OUTFLOW,
    BC_INS_WALL,
    BDF1, BDF2, build_ins_operators, make_penalty_config_dimensional,
    ns_advection_2d, ns_step_2d,
)
from ddg.dg2d.sipg import (
    IterativeConfig, make_helmholtz_block_jacobi_preconditioner,
    make_sipg_block_jacobi_preconditioner,
)
from ddg.dg2d.cubature import attach_cubature
from ddg.dg2d.ins_metrics import (
    drag_lift_coefficients, obstacle_forces_2d, recirculation_length_2d,
    strouhal_from_lift,
)


# ---------------------------------------------------------------------------
# Schäfer-Turek 2D-2 configuration
# ---------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 2.2
Y_MIN, Y_MAX = 0.0, 0.41
H = Y_MAX - Y_MIN

CX, CY = 0.2, 0.2
RADIUS = 0.05
D_CYL = 2.0 * RADIUS                     # diameter, used for C_D, C_L

# Body-fitted O-grid around the cylinder. R_outer is the half-side of
# the bounding square between the cylinder and the outer Cartesian
# mesh; it must be a multiple of both hx and hy.
R_OUTER = 0.1
N_RADIAL = 4
RADIAL_GROWTH = 1.3                      # mild clustering near cylinder

U_MAX = 1.5                              # peak of inflow parabola
U_MEAN = (2.0 / 3.0) * U_MAX             # = 1.0
U_REF = U_MEAN                           # used for Re, C_D, Strouhal
NU = 1.0e-3                              # ⇒ Re = U_mean · D / ν = 100

# Mesh — hx = 0.025 (divides 0.2 and 0.1), hy = 0.01 (divides 0.2 and
# 0.1). Channel y-range 0.41 then needs Ky = 41. Anisotropic cells
# (hx > hy) are fine — the body-fitted annulus near the cylinder uses
# its own (radial, angular) spacing.
KX, KY = 88, 41
N = 3

T_FINAL = 8.0
# CFL: the body-fitted O-grid has small annular cells near the
# cylinder (h_min ~ 4e-3). The dual-splitting scheme treats
# convection explicitly, so dt is convective-CFL limited — and for
# high-order DG the stable CFL is well below 1 (the (N+1)² factor).
# dt = 5e-3 blows up regardless of solver; dt = 1.5e-4 is the largest
# step verified stable at the production Kx=88, N=3 config (probe:
# 15 steps, max|U| bounded). T=8 covers ~24 shedding periods
# (Strouhal period ~ D/(St·U) ~ 0.33). ~5.3e4 steps — a GPU job.
DT = 1.5e-4
SAVE_EVERY = 100

# Smoothly ramp the inflow from 0 to U_max over t ∈ [0, T_RAMP] to
# damp the impulsive-start pressure transient. The Schäfer-Turek 2D-2
# spec is constant-inflow, but a brief ramp at startup costs nothing
# physically (the integration window already saturates after a few
# shedding periods) and removes the huge initial Cd spike that
# otherwise pollutes the early-time series.
T_RAMP = 0.5
def _ramp_factor(t):
    return jnp.where(
        t >= T_RAMP,
        1.0,
        0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(t, 0.0, T_RAMP) / T_RAMP)),
    )

# At Schäfer-Turek resolution (Kx=88, Ky=41, N=3 — ~70k DOFs/field)
# the dense SIPG LU factor is ~38 GB, infeasible. Run matrix-free:
# build_ins_operators(skip_factor=True) skips the dense assembly and
# ns_step_2d solves pressure + viscous with matrix-free CG,
# accelerated by the matrix-free block-Jacobi preconditioner.
USE_ITERATIVE = True
ITER_CFG = IterativeConfig(method="cg", tol=1.0e-8, atol=0.0, maxiter=3000)

# Divergence + continuity penalty post-projection (Fehn 2018), on by
# default: the robustness audit (report §3) and the benchmark-suite
# synthesis (§5) both identify it as the cure for the unstabilised-
# splitting wake instability — exactly the failure mode that blew up
# the cylinder during GPU calibration. Element-local dimensional
# scaling τ ~ PENALTY_SCALE · h · U. Matrix-free CG penalty path, so
# it composes with the iterative solver above.
USE_PENALTY = True
PENALTY_SCALE = 1.0

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# BC tagging — same logic as ns_flow_past_square_2d.py
# ---------------------------------------------------------------------------

def tag_boundary_faces(mesh):
    """Classify each boundary face as inflow / outflow / wall / cylinder.

    Cylinder = any boundary face whose midpoint sits within
    ``2·RADIUS`` of ``(CX, CY)``. Outer channel = any boundary face
    near the outer four edges. Body-fitted mesh: the cylinder faces
    are well-separated from outer-channel faces by the annulus.
    """
    Nfaces, K = mesh.Nfaces, mesh.K
    Fmask = np.asarray(mesh.Fmask)
    x_np = np.asarray(mesh.x); y_np = np.asarray(mesh.y)
    x_face = np.zeros((Nfaces, K)); y_face = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        x_face[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE_np = np.asarray(mesh.EToE)
    is_bdy = (EToE_np == np.arange(K)[:, None]).T

    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    edge_tol = 1e-3
    inflow = is_bdy & (x_face < X_MIN + edge_tol)
    outflow = is_bdy & (x_face > X_MAX - edge_tol)
    on_outer_y = is_bdy & (
        (y_face < Y_MIN + edge_tol) | (y_face > Y_MAX - edge_tol)
    )
    r_from_cyl = np.sqrt((x_face - CX) ** 2 + (y_face - CY) ** 2)
    on_cyl = is_bdy & (r_from_cyl < RADIUS * 1.05) & (~inflow) & (~outflow) & (~on_outer_y)

    bc = np.where(inflow, BC_INS_INFLOW, bc)
    bc = np.where(outflow, BC_INS_OUTFLOW, bc)
    bc = np.where(on_outer_y, BC_INS_WALL, bc)
    bc = np.where(on_cyl, BC_INS_OBSTACLE, bc)
    return jnp.asarray(bc.T)


def parabolic_inflow_field(mesh):
    """Schäfer-Turek inflow: U_x(y) = 4 · U_max · y(H − y) / H², U_y = 0."""
    yfrac = (mesh.y - Y_MIN) / H
    ux_field = 4.0 * U_MAX * yfrac * (1.0 - yfrac)
    uy_field = jnp.zeros_like(mesh.y)
    return ux_field, uy_field


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print(
        f"Schäfer-Turek 2D-2: Kx={KX}, Ky={KY}, N={N}, "
        f"cyl=({CX}, {CY}, r={RADIUS}), Re={U_MEAN * D_CYL / NU:.0f}"
    )
    VX, VY, EToV = channel_with_circle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, KX, KY,
        cx=CX, cy=CY, radius=RADIUS,
        R_outer=R_OUTER, n_radial=N_RADIAL, radial_growth=RADIAL_GROWTH,
    )
    mesh = build_mesh_2d(VX, VY, EToV, N)
    # Bend the inner-ring faces onto the cylinder. With n_circ ~ 56 the
    # per-face bowing is < 1% of the radius — no inversions.
    mesh = apply_circular_hole_2d(
        mesh, cx=CX, cy=CY, radius=RADIUS,
        blend_interior=True, detection_tol=0.05,
    )
    J = np.asarray(mesh.J)
    n_inv = int((J <= 0).sum())
    if n_inv > 0:
        raise RuntimeError(
            f"Mesh has {n_inv} inverted nodes after curving — refine "
            f"or reduce R_outer."
        )
    n_curved = int((J.max(axis=0) - J.min(axis=0) > 1e-12).sum())
    print(
        f"mesh: K={mesh.K} triangles, Np={mesh.Np}, "
        f"dofs/field={mesh.K * mesh.Np}, curved={n_curved}"
    )
    # High-order cubature for the SIPG operator: the curved inner-ring
    # elements need genuine quadrature, not nodal "diagonal-J".
    print("attaching cubature ...")
    cub = attach_cubature(mesh)
    print(f"  cubature: {cub.r.shape[0]} pts/triangle, degree>={4*N-2}")

    ins_bc_types = tag_boundary_faces(mesh)
    n_inflow = int((ins_bc_types == BC_INS_INFLOW).sum())
    n_outflow = int((ins_bc_types == BC_INS_OUTFLOW).sum())
    n_wall = int((ins_bc_types == BC_INS_WALL).sum())
    n_cyl = int((ins_bc_types == BC_INS_OBSTACLE).sum())
    print(
        f"BC face counts: inflow={n_inflow} outflow={n_outflow} "
        f"wall={n_wall} cylinder={n_cyl}"
    )

    bc_ux_full, bc_uy_full = parabolic_inflow_field(mesh)
    # Force the no-slip Dirichlet trace to zero on every wall / cylinder
    # face node.
    Fmask = np.asarray(mesh.Fmask)
    K, Np, Nfp, Nfaces = mesh.K, mesh.Np, mesh.Nfp, mesh.Nfaces
    bc_per_face = np.asarray(ins_bc_types).T
    wall_or_obstacle = (
        (bc_per_face == BC_INS_WALL) | (bc_per_face == BC_INS_OBSTACLE)
    )
    nodewise_force_zero = np.zeros((Np, K), dtype=bool)
    for f in range(Nfaces):
        for k in np.where(wall_or_obstacle[f, :])[0]:
            nodewise_force_zero[Fmask[:, f], k] = True
    nodewise_force_zero = jnp.asarray(nodewise_force_zero)
    bc_ux_full = jnp.where(nodewise_force_zero, 0.0, bc_ux_full)
    bc_uy_full = jnp.where(nodewise_force_zero, 0.0, bc_uy_full)

    # Start from rest; the ramped inflow drives the flow up.
    ux = jnp.zeros_like(bc_ux_full)
    uy = jnp.zeros_like(bc_uy_full)
    # bc_ux/bc_uy refreshed each step with the current ramp factor.
    bc_ux = jnp.zeros_like(bc_ux_full)
    bc_uy = jnp.zeros_like(bc_uy_full)

    iter_cfg = ITER_CFG if USE_ITERATIVE else None
    mode = "matrix-free CG" if USE_ITERATIVE else "direct LU"
    print(f"building INS operators for BDF1 ({mode}) ...")
    t0 = time.time()
    ops_bdf1 = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=BDF1.g0, ins_bc_types=ins_bc_types,
        skip_factor=USE_ITERATIVE, cub=cub,
    )

    def _make_pcs(ops):
        """Matrix-free block-Jacobi preconditioners for the pressure
        Poisson and the viscous Helmholtz of this operator set."""
        if not USE_ITERATIVE:
            return None, None
        p_pc = make_sipg_block_jacobi_preconditioner(
            mesh, bc_types=ops.pressure_bc_types, matfree=True,
        )
        v_pc = make_helmholtz_block_jacobi_preconditioner(
            mesh, alpha=ops.alpha, nu=NU,
            bc_types=ops.viscous_bc_types, matfree=True,
        )
        return p_pc, v_pc

    p_pc, v_pc = _make_pcs(ops_bdf1)
    print(f"  build took {time.time() - t0:.1f}s")

    pen_cfg = None
    if USE_PENALTY:
        pen_cfg = make_penalty_config_dimensional(
            mesh, u_char=U_MEAN,
            scale_div=PENALTY_SCALE, scale_cont=PENALTY_SCALE,
        )
        print(f"  divergence/continuity penalty ON (scale={PENALTY_SCALE})")

    n_steps = int(np.ceil(T_FINAL / DT))
    print(f"running {n_steps} steps with dt = {DT}")

    snapshots_ux, snapshots_uy, snapshots_p, snapshots_w, times = [], [], [], [], []

    from ddg.dg2d.maxwell import curl_z_2d
    def _vorticity(ux_, uy_): return curl_z_2d(mesh, ux_, uy_)

    ux_old = ux; uy_old = uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))

    # Obstacle face-node mask for the surface integral.
    bc_per_node = np.repeat(bc_per_face[None, :, :], Nfp, axis=0)
    bc_flat = bc_per_node.reshape(-1, order="F")
    obstacle_mapB_mask = jnp.asarray(
        bc_flat[np.asarray(mesh.mapB)] == BC_INS_OBSTACLE
    )
    Cd_series, Cl_series, force_times = [], [], []

    step_start = time.time()
    ops_current = ops_bdf1
    bdf_current = BDF1
    for step in range(1, n_steps + 1):
        # Refresh ramped inflow BC for this step.
        t_now = step * DT
        r = _ramp_factor(t_now)
        bc_ux = r * bc_ux_full
        bc_uy = r * bc_uy_full
        if step == 2:
            print("Rebuilding operators for BDF2 ...")
            ops_current = build_ins_operators(
                mesh, dt=DT, nu=NU, g0=BDF2.g0, ins_bc_types=ins_bc_types,
                skip_factor=USE_ITERATIVE, cub=cub,
            )
            # alpha changes BDF1->BDF2: the viscous preconditioner must
            # be rebuilt; the pressure Poisson PC is alpha-independent.
            p_pc, v_pc = _make_pcs(ops_current)
            bdf_current = BDF2

        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=DT, nu=NU, bdf=bdf_current, ops=ops_current,
            ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
            iterative=iter_cfg,
            pressure_preconditioner=p_pc, viscous_preconditioner=v_pc,
            penalty_iterative=pen_cfg,
            cub=cub,
        )
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n

        forces = obstacle_forces_2d(
            mesh, ux, uy, p, nu=NU, obstacle_mapB_mask=obstacle_mapB_mask,
        )
        Cd, Cl = drag_lift_coefficients(forces, U_inf=U_REF, D=D_CYL)
        Cd_series.append(Cd); Cl_series.append(Cl)
        force_times.append(step * DT)

        if step % 20 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            print(
                f"  step {step:5d}/{n_steps}  t={step*DT:5.2f}  "
                f"max|U|={max_u:.3f}  Cd={Cd:.3f}  Cl={Cl:+.3f}  "
                f"elapsed={elapsed:.1f}s"
            )
        if step % SAVE_EVERY == 0 or step == n_steps:
            snapshots_ux.append(np.asarray(ux))
            snapshots_uy.append(np.asarray(uy))
            snapshots_p.append(np.asarray(p))
            snapshots_w.append(np.asarray(_vorticity(ux, uy)))
            times.append(step * DT)

    # Post-processing.
    Cd_arr = np.array(Cd_series); Cl_arr = np.array(Cl_series)
    ft_arr = np.array(force_times)
    tail = len(Cd_arr) // 2
    Cd_tail = float(np.mean(Cd_arr[tail:]))
    Cl_tail = float(np.mean(Cl_arr[tail:]))
    Cd_max = float(np.max(Cd_arr[tail:]))
    Cl_max = float(np.max(Cl_arr[tail:]))
    St, freqs, spec = strouhal_from_lift(
        ft_arr, Cl_arr, D=D_CYL, U_inf=U_REF,
    )
    L_rec = recirculation_length_2d(
        mesh, ux,
        centerline_y=(Y_MAX + Y_MIN) / 2,
        obstacle_x_max=CX + RADIUS,
        sample_x=np.linspace(CX + RADIUS, X_MAX, 80),
    )
    # Schäfer-Turek 2D-2 reference windows (Turek group, FeatFlow web).
    ref = {
        "C_D_max": (3.2200, 3.2400),
        "C_L_max": (0.9800, 1.0200),
        "Strouhal": (0.2950, 0.3050),
    }
    print(
        f"\nCylinder metrics (tail = last {tail} steps):\n"
        f"  C_d̅       = {Cd_tail:.4f}    "
        f"C_D,max  = {Cd_max:.4f}    (ref ∈ {ref['C_D_max']})\n"
        f"  C_l̅       = {Cl_tail:+.4f}   "
        f"C_L,max  = {Cl_max:+.4f}   (ref ∈ {ref['C_L_max']})\n"
        f"  Strouhal = {St:.4f}              (ref ∈ {ref['Strouhal']})\n"
        f"  recirculation length L ≈ {L_rec:.4f}"
    )

    # Drag / lift time-series plot.
    fig_dl, ax_dl = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax_dl[0].plot(ft_arr, Cd_arr, label="$C_D$")
    ax_dl[0].axhline(Cd_tail, color="k", linestyle="--",
                     label=f"tail-avg = {Cd_tail:.3f}")
    ax_dl[0].axhspan(*ref["C_D_max"], color="g", alpha=0.15,
                     label="Schäfer-Turek $C_{D,\\max}$ window")
    ax_dl[0].set_ylabel("$C_D$"); ax_dl[0].legend(loc="lower right")
    ax_dl[0].grid(True, alpha=0.3)
    ax_dl[1].plot(ft_arr, Cl_arr, label="$C_L$", color="C1")
    ax_dl[1].axhline(Cl_tail, color="k", linestyle="--",
                     label=f"tail-avg = {Cl_tail:+.3f}")
    ax_dl[1].axhspan(*ref["C_L_max"], color="g", alpha=0.15)
    ax_dl[1].axhspan(-ref["C_L_max"][1], -ref["C_L_max"][0], color="g", alpha=0.15,
                     label="Schäfer-Turek $|C_{L,\\max}|$ window")
    ax_dl[1].set_xlabel("$t$"); ax_dl[1].set_ylabel("$C_L$")
    ax_dl[1].legend(loc="lower right"); ax_dl[1].grid(True, alpha=0.3)
    fig_dl.suptitle(
        f"Cylinder force coefficients  ·  Re = {U_REF*D_CYL/NU:.0f}  ·  "
        f"St = {St:.4f}"
    )
    fig_dl.tight_layout()
    dl_path = OUT_DIR / "ns_flow_past_cylinder_2d_forces.png"
    fig_dl.savefig(dl_path, dpi=100)
    plt.close(fig_dl)
    print(f"wrote {dl_path}")

    # Vorticity + speed animation.
    print("rendering animation ...")
    from matplotlib.animation import FuncAnimation
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    levels_w = np.linspace(-30, 30, 31)
    levels_s = np.linspace(0, 1.5 * U_MAX, 16)
    theta = np.linspace(0, 2 * np.pi, 80)
    cyl_x = CX + RADIUS * np.cos(theta)
    cyl_y = CY + RADIUS * np.sin(theta)

    def draw(i):
        for ax in axes:
            ax.clear()
        pz_w = snapshots_w[i].T.reshape(-1)
        axes[0].tricontourf(px, py, pz_w, levels=levels_w,
                            cmap="RdBu_r", extend="both")
        axes[0].fill(cyl_x, cyl_y, color="black")
        axes[0].set_aspect("equal")
        axes[0].set_title(f"vorticity $\\omega$    t = {times[i]:.2f}")
        speed = np.sqrt(snapshots_ux[i] ** 2 + snapshots_uy[i] ** 2)
        axes[1].tricontourf(px, py, speed.T.reshape(-1),
                            levels=levels_s, cmap="viridis", extend="max")
        axes[1].fill(cyl_x, cyl_y, color="black")
        axes[1].set_aspect("equal")
        axes[1].set_title("speed $|U|$")
        return []

    anim = FuncAnimation(fig, draw, frames=len(times), interval=60, blit=False)
    out_mp4 = OUT_DIR / "ns_flow_past_cylinder_2d.mp4"
    anim.save(str(out_mp4), writer="ffmpeg", dpi=100)
    plt.close(fig)
    print(f"wrote {out_mp4}")

    meta = {
        "benchmark": "Schaefer-Turek 2D-2",
        "X_MIN": X_MIN, "X_MAX": X_MAX,
        "Y_MIN": Y_MIN, "Y_MAX": Y_MAX,
        "CX": CX, "CY": CY, "RADIUS": RADIUS, "D": D_CYL,
        "R_OUTER": R_OUTER, "N_RADIAL": N_RADIAL,
        "KX": KX, "KY": KY, "N": N, "K": int(mesh.K), "Np": int(mesh.Np),
        "U_MAX": U_MAX, "U_MEAN": U_MEAN, "NU": NU,
        "Reynolds": U_REF * D_CYL / NU,
        "T_FINAL": T_FINAL, "DT": DT, "n_steps": n_steps,
        "Cd_tail_avg": Cd_tail, "Cl_tail_avg": Cl_tail,
        "Cd_max_tail": Cd_max, "Cl_max_tail": Cl_max,
        "Strouhal": St, "recirculation_length": L_rec,
        "reference_windows": ref,
    }
    meta_path = OUT_DIR / "ns_flow_past_cylinder_2d_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
