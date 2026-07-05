"""Flow past a square obstacle, velocity-pressure incompressible NS.

Domain: a rectangular channel ``[0, 6] × [0, 2]`` with an axis-aligned
square hole cut out at ``x ∈ [1.5, 2.0], y ∈ [0.75, 1.25]``.

Boundary conditions:
  * **Inflow** (``x = 0``): parabolic profile ``U_in(y) = 4·U·y(1−y/H)/H``
    with peak ``U`` at mid-height ``y = H/2``.
  * **Outflow** (``x = 6``): natural (zero traction in normal direction).
  * **Channel walls** (``y = 0``, ``y = 2``): no-slip.
  * **Square obstacle**: no-slip.

We integrate with the Hesthaven dual-splitting scheme: BDF1 for the
first time step (to bootstrap), BDF2 thereafter.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, rectangle_with_square_hole_mesh_2d
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OBSTACLE, BC_INS_OUTFLOW,
    BC_INS_WALL,
    BDF1, BDF2, build_ins_operators, ns_advection_2d, ns_step_2d,
)
from ddg.dg2d.ins_metrics import (
    drag_lift_coefficients, obstacle_forces_2d, recirculation_length_2d,
    strouhal_from_lift,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 6.0
Y_MIN, Y_MAX = 0.0, 2.0
H = Y_MAX - Y_MIN
KX, KY = 24, 8
N = 2
HOLE_X = (1.5, 2.0)
HOLE_Y = (0.75, 1.25)
U_INF = 1.0
NU = 0.02      # Reynolds ≈ U·D/ν = 1·0.5/0.02 = 25; mildly turbulent / steady wake
T_FINAL = 6.0
DT = 0.02
SAVE_EVERY = 1
OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Boundary tagging
# ---------------------------------------------------------------------------

def tag_boundary_faces(mesh, hole_x, hole_y):
    """Build (K, Nfaces) BC tags for the channel-with-square-hole demo."""
    Nfaces, K = mesh.Nfaces, mesh.K
    nx_face_avg = np.asarray(
        mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F")
    )[0]
    ny_face_avg = np.asarray(
        mesh.ny.reshape((mesh.Nfp, Nfaces, K), order="F")
    )[0]
    Fmask = np.asarray(mesh.Fmask)
    x_np = np.asarray(mesh.x)
    y_np = np.asarray(mesh.y)
    x_face = np.zeros((Nfaces, K))
    y_face = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        x_face[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE_np = np.asarray(mesh.EToE)
    is_bdy = (EToE_np == np.arange(K)[:, None]).T  # (Nfaces, K)

    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    # Inflow / outflow (only the x = X_MIN and x = X_MAX outer faces;
    # x-aligned faces inside the hole boundary are obstacle faces, not
    # inflow/outflow).
    inflow = is_bdy & (np.abs(nx_face_avg) > 0.5) & (x_face < X_MIN + 0.1)
    outflow = is_bdy & (np.abs(nx_face_avg) > 0.5) & (x_face > X_MAX - 0.1)
    # Channel walls (outer y = Y_MIN or Y_MAX, not the obstacle y faces)
    on_outer_y = is_bdy & (np.abs(ny_face_avg) > 0.5) & (
        (y_face < Y_MIN + 0.1) | (y_face > Y_MAX - 0.1)
    )
    # Obstacle: any other boundary face (lies on the hole's perimeter)
    on_hole = is_bdy & (~inflow) & (~outflow) & (~on_outer_y)

    bc = np.where(inflow, BC_INS_INFLOW, bc)
    bc = np.where(outflow, BC_INS_OUTFLOW, bc)
    bc = np.where(on_outer_y, BC_INS_WALL, bc)
    bc = np.where(on_hole, BC_INS_OBSTACLE, bc)
    return jnp.asarray(bc.T)


def parabolic_inflow_field(mesh):
    """Volume-shaped (Np, K) inflow profile ``U(y) = 4 U_inf · (y/H)(1 − y/H)``.

    Used as the Dirichlet BC for ux on the inflow face. It's zero at
    ``y = 0`` and ``y = 2``, so it also serves as the BC on the channel
    walls (no-slip). On the obstacle face nodes we override to zero.
    """
    yfrac = (mesh.y - Y_MIN) / H
    ux_field = 4.0 * U_INF * yfrac * (1.0 - yfrac)
    uy_field = jnp.zeros_like(mesh.y)
    return ux_field, uy_field


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print(f"Building mesh: Kx={KX}, Ky={KY}, N={N}, "
          f"hole [{HOLE_X[0]}, {HOLE_X[1]}] × [{HOLE_Y[0]}, {HOLE_Y[1]}]")
    VX, VY, EToV = rectangle_with_square_hole_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, KX, KY,
        HOLE_X[0], HOLE_X[1], HOLE_Y[0], HOLE_Y[1],
    )
    mesh = build_mesh_2d(VX, VY, EToV, N)
    print(f"mesh: {mesh.K} triangles, Np = {mesh.Np}, "
          f"dofs / field = {mesh.K * mesh.Np}, Re ≈ "
          f"{U_INF * (HOLE_Y[1] - HOLE_Y[0]) / NU:.0f}")

    ins_bc_types = tag_boundary_faces(mesh, HOLE_X, HOLE_Y)
    n_inflow = int((ins_bc_types == BC_INS_INFLOW).sum())
    n_outflow = int((ins_bc_types == BC_INS_OUTFLOW).sum())
    n_wall = int((ins_bc_types == BC_INS_WALL).sum())
    n_obstacle = int((ins_bc_types == BC_INS_OBSTACLE).sum())
    print(f"BC face counts: inflow={n_inflow} outflow={n_outflow} "
          f"wall={n_wall} obstacle={n_obstacle}")

    bc_ux, bc_uy = parabolic_inflow_field(mesh)
    # Override bc_ux to zero at every node touching a wall / obstacle
    # face, so the no-slip Dirichlet trace is 0 everywhere on those
    # faces (the parabolic profile is non-zero off y=0,2 in general).
    Fmask = np.asarray(mesh.Fmask)
    K, Np, Nfp, Nfaces = mesh.K, mesh.Np, mesh.Nfp, mesh.Nfaces
    bc_per_face = np.asarray(ins_bc_types).T  # (Nfaces, K)
    wall_or_obstacle = (
        (bc_per_face == BC_INS_WALL) | (bc_per_face == BC_INS_OBSTACLE)
    )
    nodewise_force_zero = np.zeros((Np, K), dtype=bool)
    for f in range(Nfaces):
        mask_f = wall_or_obstacle[f, :]
        for k in np.where(mask_f)[0]:
            nodewise_force_zero[Fmask[:, f], k] = True
    nodewise_force_zero = jnp.asarray(nodewise_force_zero)
    bc_ux = jnp.where(nodewise_force_zero, 0.0, bc_ux)
    bc_uy = jnp.where(nodewise_force_zero, 0.0, bc_uy)

    # Initial condition: parabolic profile everywhere (steady plane
    # Poiseuille away from the obstacle; the flow then evolves to
    # satisfy the no-slip boundary on the square).
    ux = bc_ux
    uy = bc_uy

    print("LU-factoring pressure + viscous matrices for BDF1 ...")
    t0 = time.time()
    ops_bdf1 = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=BDF1.g0, ins_bc_types=ins_bc_types,
    )
    print(f"  factor took {time.time() - t0:.1f}s")

    n_steps = int(np.ceil(T_FINAL / DT))
    print(f"running {n_steps} steps with dt = {DT}")

    snapshots_ux, snapshots_uy, snapshots_p, snapshots_w, times = [], [], [], [], []

    from ddg.dg2d.maxwell import curl_z_2d

    def _vorticity(ux_, uy_):
        return curl_z_2d(mesh, ux_, uy_)

    ux_old = ux
    uy_old = uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))

    # mapB-mask for obstacle faces (used to integrate forces).
    bc_per_face = np.asarray(ins_bc_types).T
    Nfp, Nfaces, K = mesh.Nfp, mesh.Nfaces, mesh.K
    bc_per_node = np.repeat(bc_per_face[None, :, :], Nfp, axis=0)
    bc_flat = bc_per_node.reshape(-1, order="F")
    obstacle_mapB_mask = jnp.asarray(
        bc_flat[np.asarray(mesh.mapB)] == BC_INS_OBSTACLE
    )
    D_obstacle = HOLE_X[1] - HOLE_X[0]
    Cd_series, Cl_series, force_times = [], [], []

    step_start = time.time()
    ops_current = ops_bdf1
    bdf_current = BDF1
    for step in range(1, n_steps + 1):
        if step == 2:
            print("Rebuilding operators for BDF2 ...")
            ops_current = build_ins_operators(
                mesh, dt=DT, nu=NU, g0=BDF2.g0, ins_bc_types=ins_bc_types,
            )
            bdf_current = BDF2

        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=DT, nu=NU, bdf=bdf_current, ops=ops_current,
            ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
        )
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n

        # Force coefficients every step (cheap; useful for FFT).
        forces = obstacle_forces_2d(
            mesh, ux, uy, p, nu=NU, obstacle_mapB_mask=obstacle_mapB_mask,
        )
        Cd, Cl = drag_lift_coefficients(forces, U_inf=U_INF, D=D_obstacle)
        Cd_series.append(Cd)
        Cl_series.append(Cl)
        force_times.append(step * DT)

        if step % 5 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            max_w = float(jnp.max(jnp.abs(_vorticity(ux, uy))))
            print(f"  step {step:4d}/{n_steps}  t={step*DT:5.2f}  "
                  f"max|U|={max_u:.3f}  max|ω|={max_w:.3f}  "
                  f"Cd={Cd:.3f}  Cl={Cl:+.3f}  "
                  f"elapsed={elapsed:.1f}s")
        if step % SAVE_EVERY == 0 or step == n_steps:
            snapshots_ux.append(np.asarray(ux))
            snapshots_uy.append(np.asarray(uy))
            snapshots_p.append(np.asarray(p))
            snapshots_w.append(np.asarray(_vorticity(ux, uy)))
            times.append(step * DT)

    # Post-process: tail-averaged drag/lift + Strouhal + recirculation.
    Cd_arr = np.array(Cd_series)
    Cl_arr = np.array(Cl_series)
    ft_arr = np.array(force_times)
    tail = len(Cd_arr) // 2
    Cd_tail = float(np.mean(Cd_arr[tail:]))
    Cl_tail = float(np.mean(Cl_arr[tail:]))
    St, freqs, spec = strouhal_from_lift(
        ft_arr, Cl_arr, D=D_obstacle, U_inf=U_INF,
    )
    L_rec = recirculation_length_2d(
        mesh, ux,
        centerline_y=(Y_MAX + Y_MIN) / 2,
        obstacle_x_max=HOLE_X[1],
        sample_x=np.linspace(HOLE_X[1], X_MAX, 80),
    )
    print(
        f"\nObstacle metrics (tail-averaged over last {tail} steps):\n"
        f"  C_d̅ = {Cd_tail:.3f}   C_l̅ = {Cl_tail:+.3f}\n"
        f"  Strouhal St ≈ {St:.3f}   (≈ 0 if no clear shedding peak)\n"
        f"  recirculation length L ≈ {L_rec:.3f}"
    )

    # Drag / lift time-series plot.
    fig_dl, ax_dl = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    ax_dl[0].plot(ft_arr, Cd_arr, label="C_d")
    ax_dl[0].axhline(Cd_tail, color="k", linestyle="--",
                     label=f"tail-avg = {Cd_tail:.3f}")
    ax_dl[0].set_ylabel("C_d")
    ax_dl[0].legend(); ax_dl[0].grid(True, alpha=0.3)
    ax_dl[1].plot(ft_arr, Cl_arr, label="C_l", color="C1")
    ax_dl[1].axhline(Cl_tail, color="k", linestyle="--",
                     label=f"tail-avg = {Cl_tail:+.3f}")
    ax_dl[1].set_xlabel("t"); ax_dl[1].set_ylabel("C_l")
    ax_dl[1].legend(); ax_dl[1].grid(True, alpha=0.3)
    fig_dl.suptitle(
        f"Forces on obstacle    Re ≈ "
        f"{U_INF*D_obstacle/NU:.0f}    Strouhal ≈ {St:.3f}"
    )
    fig_dl.tight_layout()
    dl_path = OUT_DIR / "ns_flow_past_square_2d_forces.png"
    fig_dl.savefig(dl_path, dpi=80)
    plt.close(fig_dl)
    print(f"wrote {dl_path}")

    print("rendering animation ...")
    from matplotlib.animation import FuncAnimation
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    levels_w = np.linspace(-5, 5, 31)
    levels_s = np.linspace(0, 1.5 * U_INF, 16)

    def draw(i):
        for ax in axes:
            ax.clear()
        pz_w = snapshots_w[i].T.reshape(-1)
        axes[0].tricontourf(px, py, pz_w, levels=levels_w, cmap="RdBu_r", extend="both")
        axes[0].add_patch(plt.Rectangle(
            (HOLE_X[0], HOLE_Y[0]),
            HOLE_X[1] - HOLE_X[0], HOLE_Y[1] - HOLE_Y[0],
            facecolor="black", edgecolor="black",
        ))
        axes[0].set_aspect("equal")
        axes[0].set_title(f"vorticity ω    t = {times[i]:.2f}")

        ux_i = snapshots_ux[i]
        uy_i = snapshots_uy[i]
        speed = np.sqrt(ux_i ** 2 + uy_i ** 2)
        pz_s = speed.T.reshape(-1)
        axes[1].tricontourf(px, py, pz_s, levels=levels_s, cmap="viridis", extend="max")
        axes[1].add_patch(plt.Rectangle(
            (HOLE_X[0], HOLE_Y[0]),
            HOLE_X[1] - HOLE_X[0], HOLE_Y[1] - HOLE_Y[0],
            facecolor="black", edgecolor="black",
        ))
        axes[1].set_aspect("equal")
        axes[1].set_title("speed |U|")
        return []

    anim = FuncAnimation(fig, draw, frames=len(times), interval=80, blit=False)
    out_mp4 = OUT_DIR / "ns_flow_past_square_2d.mp4"
    anim.save(str(out_mp4), writer="ffmpeg", dpi=80)
    plt.close(fig)
    print(f"wrote {out_mp4}")

    meta = {
        "X_MIN": X_MIN, "X_MAX": X_MAX,
        "Y_MIN": Y_MIN, "Y_MAX": Y_MAX,
        "HOLE_X": list(HOLE_X), "HOLE_Y": list(HOLE_Y),
        "KX": KX, "KY": KY, "N": N,
        "U_INF": U_INF, "NU": NU,
        "T_FINAL": T_FINAL, "DT": DT,
        "n_steps": n_steps,
        "Reynolds_approx": U_INF * (HOLE_Y[1] - HOLE_Y[0]) / NU,
        "Cd_tail_avg": Cd_tail,
        "Cl_tail_avg": Cl_tail,
        "Strouhal": St,
        "recirculation_length": L_rec,
    }
    meta_path = OUT_DIR / "ns_flow_past_square_2d_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
