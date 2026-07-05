"""Backward-facing step: the Armaly 1983 separated-flow benchmark.

Fluid enters a channel of height ``h`` and at ``x = L_in`` the bottom
wall drops by a step of height ``s``, opening into a taller channel
of height ``s + h``. The sudden expansion separates the flow off the
step corner; it recirculates and reattaches to the bottom wall some
distance downstream. The length of that primary recirculation bubble
``x_r`` (measured from the step) is the headline quantity — it grows
with Reynolds number in the laminar regime.

Reference: Armaly, Durst, Pereira & Schönung (1983), "Experimental
and theoretical investigation of backward-facing step flow",
J. Fluid Mech. 127, 473-496. Their Reynolds number is
``Re = ū · D / ν`` with ``D = 2h`` the inlet hydraulic diameter and
``ū`` the mean inlet velocity. Approximate experimental primary-
reattachment lengths ``x_r / s`` are embedded below.

Expansion ratio here is ``(s + h)/h = 2`` (``s = h``), the standard
laminar BFS configuration.

Boundary conditions:
  * Inflow (``x = 0``, the upper inlet strip): fully-developed
    parabolic channel profile.
  * Outflow (``x = L_in + L_out``): natural / zero-traction.
  * Every other boundary (top wall, downstream bottom wall, the step
    face, and the inlet-section bottom wall): no-slip.

A steady benchmark — march in time until the field stops changing.
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

from ddg.dg2d.mesh import build_mesh_2d, backward_facing_step_mesh_2d
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OUTFLOW, BC_INS_WALL,
    BDF1, BDF2, build_ins_operators, make_ins_mg_preconditioners,
    make_penalty_config_dimensional, ns_advection_2d, ns_step_2d,
)
from ddg.dg2d.sipg import IterativeConfig
from ddg.dg2d.ins_metrics import recirculation_length_2d


# ---------------------------------------------------------------------------
# Geometry (step height s sets the length scale; s = 1)
# ---------------------------------------------------------------------------
STEP = 1.0               # step height s
H_INLET = 1.0            # inlet channel height h  ->  expansion ratio 2
L_IN = 5.0               # inlet section length (upstream of the step)
L_OUT = 30.0             # downstream section length
X_STEP = L_IN            # x-coordinate of the step face
Y_STEP = STEP            # y-coordinate of the inlet-channel floor

KX, KY = 140, 8
N = 3

# ---------------------------------------------------------------------------
# Flow parameters
# ---------------------------------------------------------------------------
U_MAX = 1.0                          # peak of the inlet parabola
U_MEAN = (2.0 / 3.0) * U_MAX         # mean inlet velocity
RE = 200.0                           # Armaly Re = U_mean · 2h / ν
NU = U_MEAN * (2.0 * H_INLET) / RE

T_FINAL = 200.0                      # several flow-through times
DT = 0.02
STEADY_TOL = 5.0e-6

# Divergence + continuity penalty post-projection (Fehn 2018), on by
# default — the robustness audit's best-performing stabilisation.
# Element-local dimensional scaling τ ~ PENALTY_SCALE · h · U.
USE_PENALTY = True
PENALTY_SCALE = 1.0

# Solve the pressure / viscous systems with matrix-free CG preconditioned
# by the hp-multigrid V-cycle (ddg.dg2d.multigrid) instead of a dense LU
# factorisation. The backward-facing step has an outflow face, so its
# pressure Poisson is non-singular and the iterative path applies. Off
# by default — the dense path is faster at this benchmark's mesh size;
# multigrid is the route to the 1e5–1e6-DOF production regime.
USE_MULTIGRID = False

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Armaly et al. 1983 primary-reattachment reference (x_r / s vs Re)
# ---------------------------------------------------------------------------
ARMALY_RE = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
ARMALY_XR_OVER_S = np.array([2.9, 4.8, 6.9, 8.4, 9.7])  # approx, from Fig. 6


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def tag_boundary_faces(mesh):
    """inflow at x=0, outflow at x=L_in+L_out, no-slip everywhere else."""
    Nfaces, K = mesh.Nfaces, mesh.K
    Fmask = np.asarray(mesh.Fmask)
    x_np = np.asarray(mesh.x); y_np = np.asarray(mesh.y)
    x_face = np.zeros((Nfaces, K)); y_face = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        x_face[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE_np = np.asarray(mesh.EToE)
    is_bdy = (EToE_np == np.arange(K)[:, None]).T

    x_out = L_IN + L_OUT
    tol = 1e-3
    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    inflow = is_bdy & (x_face < tol)
    outflow = is_bdy & (x_face > x_out - tol)
    wall = is_bdy & (~inflow) & (~outflow)
    bc = np.where(inflow, BC_INS_INFLOW, bc)
    bc = np.where(outflow, BC_INS_OUTFLOW, bc)
    bc = np.where(wall, BC_INS_WALL, bc)
    return jnp.asarray(bc.T)


def inflow_field(mesh):
    """Parabolic channel profile over the inlet strip ``y ∈ [s, s+h]``,
    zero below the step. ``u_x`` is the BC value on the inflow face and
    doubles as the no-slip (zero) trace elsewhere.
    """
    y = np.asarray(mesh.y)
    eta = (y - Y_STEP) / H_INLET            # 0 at floor, 1 at ceiling
    prof = 4.0 * U_MAX * eta * (1.0 - eta)  # parabola, peak U_MAX
    # Only valid inside the inlet channel; clamp elsewhere to 0.
    ux_field = np.where((y >= Y_STEP - 1e-9), prof, 0.0)
    ux_field = np.where(ux_field > 0.0, ux_field, 0.0)
    bc_ux = jnp.asarray(ux_field)
    bc_uy = jnp.zeros_like(mesh.y)
    return bc_ux, bc_uy


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print(
        f"Backward-facing step: Kx={KX}, Ky={KY}, N={N}, "
        f"Re={RE:.0f} (nu={NU:.4g}), ER={(STEP+H_INLET)/H_INLET:.1f}"
    )
    VX, VY, EToV = backward_facing_step_mesh_2d(
        L_IN, L_OUT, H_INLET, STEP, KX, KY,
    )
    mesh = build_mesh_2d(VX, VY, EToV, N)
    print(f"mesh: K={mesh.K} triangles, Np={mesh.Np}, "
          f"dofs/field={mesh.K * mesh.Np}")

    ins_bc_types = tag_boundary_faces(mesh)
    n_in = int((ins_bc_types == BC_INS_INFLOW).sum())
    n_out = int((ins_bc_types == BC_INS_OUTFLOW).sum())
    n_wall = int((ins_bc_types == BC_INS_WALL).sum())
    print(f"BC face counts: inflow={n_in} outflow={n_out} wall={n_wall}")

    bc_ux, bc_uy = inflow_field(mesh)
    # Force the no-slip Dirichlet trace to zero on every wall node.
    Fmask = np.asarray(mesh.Fmask)
    bc_per_face = np.asarray(ins_bc_types).T
    wall_mask = bc_per_face == BC_INS_WALL
    nodewise_zero = np.zeros((mesh.Np, mesh.K), dtype=bool)
    for f in range(mesh.Nfaces):
        for k in np.where(wall_mask[f, :])[0]:
            nodewise_zero[Fmask[:, f], k] = True
    bc_ux = jnp.where(jnp.asarray(nodewise_zero), 0.0, bc_ux)
    bc_uy = jnp.where(jnp.asarray(nodewise_zero), 0.0, bc_uy)

    # Initial condition: inflow profile carried across the upper
    # channel, zero in the step-recirculation region.
    ux = bc_ux
    uy = bc_uy

    iter_cfg = IterativeConfig(tol=1.0e-9, maxiter=2000) if USE_MULTIGRID \
        else None
    print("building pressure + viscous operators for BDF1 ...")
    t0 = time.time()
    ops_bdf1 = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=BDF1.g0, ins_bc_types=ins_bc_types,
        skip_factor=USE_MULTIGRID,
    )
    mg_p = mg_v = None
    if USE_MULTIGRID:
        mg_p, mg_v = make_ins_mg_preconditioners(
            mesh, dt=DT, nu=NU, g0=BDF1.g0, ins_bc_types=ins_bc_types)
        print("  matrix-free CG + hp-multigrid preconditioning ON")
    print(f"  setup took {time.time() - t0:.1f}s")

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
            mesh, u_char=U_MAX,
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
                skip_factor=USE_MULTIGRID,
            )
            bdf_current = BDF2
            if USE_MULTIGRID:           # alpha changed -> rebuild the V-cycles
                mg_p, mg_v = make_ins_mg_preconditioners(
                    mesh, dt=DT, nu=NU, g0=BDF2.g0,
                    ins_bc_types=ins_bc_types)

        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=DT, nu=NU, bdf=bdf_current, ops=ops_current,
            ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
            penalty_iterative=pen_cfg,
            iterative=iter_cfg,
            pressure_preconditioner=mg_p, viscous_preconditioner=mg_v,
        )
        du = jnp.sqrt(jnp.mean((ux_new - ux) ** 2 + (uy_new - uy) ** 2))
        residual = float(du) / DT / U_MEAN

        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n
        step_reached = step

        if step % 200 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            print(f"  step {step:5d}/{n_steps}  t={step*DT:7.2f}  "
                  f"max|U|={max_u:.4f}  steady-resid={residual:.2e}  "
                  f"elapsed={elapsed:.1f}s")
        if residual < STEADY_TOL and step > 20:
            print(f"  steady state reached at step {step} "
                  f"(t={step*DT:.2f}, resid={residual:.2e})")
            break

    # ----- Primary reattachment length on the bottom wall -----
    # Sample u_x just above the downstream floor (y = 0) and find where
    # it first crosses from negative (recirculating) to positive.
    x_r = recirculation_length_2d(
        mesh, ux,
        centerline_y=0.5 * STEP * 0.15,        # near-wall sample band
        obstacle_x_max=X_STEP,
        sample_x=np.linspace(X_STEP, L_IN + L_OUT, 200),
    )
    xr_over_s = x_r / STEP
    re_key = float(RE)
    ref_xr = float(np.interp(re_key, ARMALY_RE, ARMALY_XR_OVER_S))
    print(
        f"\nReattachment (primary recirculation):\n"
        f"  x_r        = {x_r:.3f}\n"
        f"  x_r / s    = {xr_over_s:.3f}\n"
        f"  Armaly ref ≈ {ref_xr:.2f}  (Re={RE:.0f}, interpolated)"
    )

    # ----- Plots -----
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    speed = np.sqrt(ux ** 2 + uy ** 2).T.reshape(-1)
    ux_flat = np.asarray(ux).T.reshape(-1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    tpc0 = axes[0].tricontourf(px, py, speed, levels=24, cmap="viridis")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"speed |U|   Re={RE:.0f}")
    fig.colorbar(tpc0, ax=axes[0], fraction=0.02, pad=0.02)
    # u_x with the zero contour highlighting the recirculation bubble.
    tpc1 = axes[1].tricontourf(px, py, ux_flat, levels=np.linspace(-0.3, 1.2, 25),
                               cmap="RdBu_r", extend="both")
    axes[1].tricontour(px, py, ux_flat, levels=[0.0], colors="k", linewidths=1.0)
    axes[1].axvline(X_STEP + x_r, color="lime", ls="--",
                    label=f"reattach x_r/s={xr_over_s:.2f}")
    axes[1].set_aspect("equal")
    axes[1].set_title("$u_x$ (black = zero contour / separation line)")
    axes[1].legend(loc="upper right")
    fig.colorbar(tpc1, ax=axes[1], fraction=0.02, pad=0.02)
    fig.tight_layout()
    field_path = OUT_DIR / f"ns_backward_facing_step_2d_Re{int(RE)}.png"
    fig.savefig(field_path, dpi=110)
    plt.close(fig)
    print(f"wrote {field_path}")

    meta = {
        "benchmark": "Armaly 1983 backward-facing step",
        "STEP": STEP, "H_INLET": H_INLET, "L_IN": L_IN, "L_OUT": L_OUT,
        "expansion_ratio": (STEP + H_INLET) / H_INLET,
        "Re": RE, "NU": NU, "U_MAX": U_MAX, "U_MEAN": U_MEAN,
        "KX": KX, "KY": KY, "N": N,
        "K": int(mesh.K), "Np": int(mesh.Np),
        "T_FINAL": T_FINAL, "DT": DT, "steps_run": step_reached,
        "final_steady_residual": residual,
        "reattachment_x_r": x_r,
        "reattachment_x_r_over_s": xr_over_s,
        "armaly_reference_x_r_over_s": ref_xr,
    }
    meta_path = OUT_DIR / f"ns_backward_facing_step_2d_Re{int(RE)}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
