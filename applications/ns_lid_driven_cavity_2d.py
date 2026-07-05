"""Lid-driven cavity: the Ghia-Ghia-Shin 1982 benchmark.

A unit square ``[0, 1]²`` of fluid, initially at rest. The top wall
("lid") slides in the ``+x`` direction at constant speed ``U_LID``;
the other three walls are stationary no-slip boundaries. The flow
develops a primary recirculating vortex plus secondary corner
vortices and settles to a steady state.

Reynolds number ``Re = U_LID · L / ν`` with ``L = 1``. The classic
reference is Ghia, Ghia & Shin (1982), "High-Re solutions for
incompressible flow using the Navier-Stokes equations and a
multigrid method", J. Comput. Phys. 48, 387-411 — which tabulates
the steady velocity profiles along the two centrelines. Those
tables are embedded below for Re = 100 and Re = 400.

Boundary conditions: all four walls are velocity-Dirichlet /
pressure-Neumann (:data:`BC_INS_WALL`). The moving lid is simply a
wall whose prescribed tangential velocity is ``U_LID`` instead of
zero. The two top corners carry the classic lid-cavity velocity
discontinuity; we resolve them in favour of the stationary side
walls (the "non-leaky" cavity), the convention behind Ghia's data.

The cavity is a *steady* benchmark — we march in time from rest
until the field stops changing, then compare centreline profiles.
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

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.incompressible_ns import (
    BC_INS_INTERIOR, BC_INS_WALL,
    BDF1, BDF2, build_ins_operators, make_penalty_config_dimensional,
    ns_advection_2d, ns_step_2d,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
L = X_MAX - X_MIN

KX, KY = 32, 32          # even -> centrelines fall on element boundaries
N = 4

U_LID = 1.0
RE = 100.0
NU = U_LID * L / RE

T_FINAL = 40.0           # long enough to reach steady state at Re=100
DT = 0.01
STEADY_TOL = 1.0e-6      # stop early once ||Δu||/dt drops below this

# Divergence + continuity penalty post-projection (Fehn 2018), on by
# default — the robustness audit identifies it as the best-performing
# stabilisation. Benign here (Re=100 stays well-resolved) but kept on
# for a uniform, stable default across the benchmark suite. Element-
# local dimensional scaling τ ~ PENALTY_SCALE · h · U.
USE_PENALTY = True
PENALTY_SCALE = 1.0

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Ghia-Ghia-Shin 1982 reference data
# ---------------------------------------------------------------------------
# u-velocity along the vertical centreline x = 0.5 (Table I).
GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000,
])
GHIA_U = {
    100: np.array([
        0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
        -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
        0.68717, 0.73722, 0.78871, 0.84123, 1.00000,
    ]),
    400: np.array([
        0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
        -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
        0.55892, 0.61756, 0.68439, 0.75837, 1.00000,
    ]),
}
# v-velocity along the horizontal centreline y = 0.5 (Table II).
GHIA_X = np.array([
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
    0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000,
])
GHIA_V = {
    100: np.array([
        0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
        0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313,
        -0.08864, -0.07391, -0.05906, 0.00000,
    ]),
    400: np.array([
        0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203,
        0.30174, 0.05186, -0.38598, -0.44993, -0.23827, -0.22847,
        -0.19254, -0.15663, -0.12146, 0.00000,
    ]),
}


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def tag_boundary_faces(mesh):
    """Every boundary face of the cavity is a no-slip / moving wall."""
    Nfaces, K = mesh.Nfaces, mesh.K
    EToE_np = np.asarray(mesh.EToE)
    is_bdy = (EToE_np == np.arange(K)[:, None])  # (K, Nfaces)
    bc = np.where(is_bdy, BC_INS_WALL, BC_INS_INTERIOR).astype(np.int32)
    return jnp.asarray(bc)


def lid_velocity_field(mesh):
    """Volume-shaped ``(Np, K)`` Dirichlet trace: ``u_x = U_LID`` on the
    lid (y = Y_MAX), zero on the other three walls and the interior.

    The two top corners are set to zero — they belong to the
    stationary side walls (non-leaky cavity, matching Ghia's BC).
    """
    y = np.asarray(mesh.y)
    x = np.asarray(mesh.x)
    tol = 1e-8
    on_lid = np.abs(y - Y_MAX) < tol
    on_side = (np.abs(x - X_MIN) < tol) | (np.abs(x - X_MAX) < tol)
    ux_field = np.where(on_lid & (~on_side), U_LID, 0.0)
    bc_ux = jnp.asarray(ux_field)
    bc_uy = jnp.zeros_like(mesh.y)
    return bc_ux, bc_uy


# ---------------------------------------------------------------------------
# Centreline extraction
# ---------------------------------------------------------------------------

def extract_centerline(mesh, field, *, axis, coord, tol=1e-6):
    """Pull ``(s, value)`` pairs for nodes on a centreline.

    ``axis='x'`` extracts the vertical centreline (constant
    ``x = coord``), returning ``(y, value)`` sorted by ``y``.
    ``axis='y'`` extracts the horizontal centreline. With even KX/KY
    the centreline coincides with element boundaries so nodes land
    exactly on it; duplicate nodes from the two adjacent elements are
    averaged out by the sort + unique pass.
    """
    x = np.asarray(mesh.x).reshape(-1)
    y = np.asarray(mesh.y).reshape(-1)
    f = np.asarray(field).reshape(-1)
    if axis == "x":
        on_line = np.abs(x - coord) < tol
        s = y[on_line]
    else:
        on_line = np.abs(y - coord) < tol
        s = x[on_line]
    vals = f[on_line]
    order = np.argsort(s)
    s_sorted, vals_sorted = s[order], vals[order]
    # Collapse duplicate coordinates (shared element-boundary nodes).
    s_unique, idx = np.unique(np.round(s_sorted, 10), return_index=True)
    out = np.zeros_like(s_unique)
    for i, sv in enumerate(s_unique):
        out[i] = vals_sorted[np.abs(s_sorted - sv) < tol].mean()
    return s_unique, out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    re_key = int(round(RE))
    print(
        f"Lid-driven cavity: Kx={KX}, Ky={KY}, N={N}, "
        f"Re={RE:.0f} (nu={NU:.4g})"
    )
    VX, VY, EToV = uniform_rectangle_mesh_2d(X_MIN, X_MAX, Y_MIN, Y_MAX, KX, KY)
    mesh = build_mesh_2d(VX, VY, EToV, N)
    print(f"mesh: K={mesh.K} triangles, Np={mesh.Np}, "
          f"dofs/field={mesh.K * mesh.Np}")

    ins_bc_types = tag_boundary_faces(mesh)
    bc_ux, bc_uy = lid_velocity_field(mesh)

    # Start from rest.
    ux = jnp.zeros((mesh.Np, mesh.K))
    uy = jnp.zeros((mesh.Np, mesh.K))

    print("LU-factoring pressure + viscous matrices for BDF1 ...")
    t0 = time.time()
    ops_bdf1 = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=BDF1.g0, ins_bc_types=ins_bc_types,
    )
    print(f"  factor took {time.time() - t0:.1f}s")

    n_steps = int(np.ceil(T_FINAL / DT))
    print(f"running up to {n_steps} steps with dt = {DT}")

    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))

    pen_cfg = None
    if USE_PENALTY:
        pen_cfg = make_penalty_config_dimensional(
            mesh, u_char=U_LID,
            scale_div=PENALTY_SCALE, scale_cont=PENALTY_SCALE,
        )
        print(f"  divergence/continuity penalty ON (scale={PENALTY_SCALE})")

    ops_current, bdf_current = ops_bdf1, BDF1
    step_start = time.time()
    residual = np.inf
    step_reached = 0
    for step in range(1, n_steps + 1):
        if step == 2:
            ops_current = build_ins_operators(
                mesh, dt=DT, nu=NU, g0=BDF2.g0, ins_bc_types=ins_bc_types,
            )
            bdf_current = BDF2

        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=DT, nu=NU, bdf=bdf_current, ops=ops_current,
            ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
            penalty_iterative=pen_cfg,
        )
        # Steady-state residual: ||Δu|| / dt, relative to U_LID.
        du = jnp.sqrt(jnp.mean((ux_new - ux) ** 2 + (uy_new - uy) ** 2))
        residual = float(du) / DT / U_LID

        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n
        step_reached = step

        if step % 100 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            print(f"  step {step:5d}/{n_steps}  t={step*DT:6.2f}  "
                  f"max|U|={max_u:.4f}  steady-resid={residual:.2e}  "
                  f"elapsed={elapsed:.1f}s")
        if residual < STEADY_TOL and step > 10:
            print(f"  steady state reached at step {step} "
                  f"(t={step*DT:.2f}, resid={residual:.2e})")
            break

    # ----- Centreline profiles + Ghia comparison -----
    y_c, u_c = extract_centerline(mesh, ux, axis="x", coord=0.5 * (X_MIN + X_MAX))
    x_c, v_c = extract_centerline(mesh, uy, axis="y", coord=0.5 * (Y_MIN + Y_MAX))

    have_ref = re_key in GHIA_U
    if have_ref:
        u_at_ghia = np.interp(GHIA_Y, y_c, u_c)
        v_at_ghia = np.interp(GHIA_X, x_c, v_c)
        u_err = float(np.sqrt(np.mean((u_at_ghia - GHIA_U[re_key]) ** 2)))
        v_err = float(np.sqrt(np.mean((v_at_ghia - GHIA_V[re_key]) ** 2)))
        print(
            f"\nGhia et al. (Re={re_key}) comparison:\n"
            f"  RMS error in u(0.5, y): {u_err:.4e}\n"
            f"  RMS error in v(x, 0.5): {v_err:.4e}"
        )
    else:
        u_err = v_err = None
        print(f"\n(no Ghia reference table for Re={re_key})")

    # ----- Plots -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(u_c, y_c, "-", color="C0", label="ddg")
    if have_ref:
        axes[0].plot(GHIA_U[re_key], GHIA_Y, "ks", ms=5,
                     label="Ghia et al.")
    axes[0].set_xlabel("$u_x$"); axes[0].set_ylabel("$y$")
    axes[0].set_title("$u_x$ on vertical centreline $x=0.5$")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_c, v_c, "-", color="C1", label="ddg")
    if have_ref:
        axes[1].plot(GHIA_X, GHIA_V[re_key], "ks", ms=5,
                     label="Ghia et al.")
    axes[1].set_xlabel("$x$"); axes[1].set_ylabel("$v_y$")
    axes[1].set_title("$v_y$ on horizontal centreline $y=0.5$")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Lid-driven cavity   Re = {re_key}   "
                 f"(N={N}, {KX}×{KY} cells)")
    fig.tight_layout()
    prof_path = OUT_DIR / f"ns_lid_driven_cavity_2d_Re{re_key}_profiles.png"
    fig.savefig(prof_path, dpi=100)
    plt.close(fig)
    print(f"wrote {prof_path}")

    # Streamline / speed plot of the steady field.
    fig2, ax2 = plt.subplots(figsize=(6, 5.5))
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    speed = np.sqrt(ux ** 2 + uy ** 2).T.reshape(-1)
    tpc = ax2.tricontourf(px, py, speed, levels=24, cmap="viridis")
    ax2.set_aspect("equal")
    ax2.set_title(f"speed $|U|$ at steady state   Re = {re_key}")
    fig2.colorbar(tpc, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    field_path = OUT_DIR / f"ns_lid_driven_cavity_2d_Re{re_key}_speed.png"
    fig2.savefig(field_path, dpi=100)
    plt.close(fig2)
    print(f"wrote {field_path}")

    meta = {
        "benchmark": "Ghia-Ghia-Shin 1982 lid-driven cavity",
        "Re": RE, "NU": NU, "U_LID": U_LID,
        "KX": KX, "KY": KY, "N": N,
        "K": int(mesh.K), "Np": int(mesh.Np),
        "T_FINAL": T_FINAL, "DT": DT,
        "steps_run": step_reached,
        "final_steady_residual": residual,
        "u_centerline_rms_error": u_err,
        "v_centerline_rms_error": v_err,
    }
    meta_path = OUT_DIR / f"ns_lid_driven_cavity_2d_Re{re_key}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
