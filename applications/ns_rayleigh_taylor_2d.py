"""Rayleigh-Taylor instability via the Boussinesq incompressible NS.

A heavy fluid resting on top of a light fluid is gravitationally
unstable: the interface buckles, the heavy fluid sinks in "spikes"
and the light fluid rises in "bubbles", and a turbulent mixing layer
grows between them.

We model the two fluids with a single normalised density scalar
``c ∈ [-1, +1]`` — ``c = +1`` heavy, ``c = -1`` light — transported
by the flow (see :mod:`ddg.dg2d.scalar_transport`). Under the
Boussinesq approximation the density contrast enters the momentum
equation only through a buoyancy body force

    f = (0, -A g c)

with ``A`` the Atwood number and ``g`` gravity. Heavy fluid
(``c=+1``) feels a downward force; light fluid (``c=-1``) an upward
one. The constant background is absorbed into the pressure.

Reference structure: ExaDG ``incompressible_flow_with_transport``
(rayleigh_benard / rising_bubble) — same transport + Boussinesq
coupling, different IC/BC.

Setup:
  * domain ``[0, W] × [0, H]``, periodic in x, no-slip top/bottom
  * interface at ``y = H/2``, heavy above, single-mode cosine
    perturbation; tanh-smoothed over thickness ``δ``
  * scalar: no-flux (Neumann) on the top/bottom walls
  * diagnostics: spike / bubble front positions and the mixing-layer
    width versus time, plus a density-field animation

Time scale: ``τ = sqrt(W / (A g))``. The mixing layer grows
quadratically, ``h ~ α A g t²``, once past the linear-instability
phase.
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

from ddg.dg2d.mesh import (
    apply_rectangle_periodic_2d, build_mesh_2d, uniform_rectangle_mesh_2d,
)
from ddg.dg2d.incompressible_ns import (
    BC_INS_INTERIOR, BC_INS_WALL,
    BDF1, BDF2, build_ins_operators, make_penalty_config_dimensional,
    ns_advection_2d, ns_step_2d,
)
from ddg.dg2d.scalar_transport import (
    all_neumann_bc_types, build_scalar_operators, scalar_advection_2d,
    scalar_transport_step_2d,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
W, H = 1.0, 2.0          # domain width, height
KX, KY = 24, 48          # KY/KX ≈ H/W keeps cells near-square
N = 3

ATWOOD = 0.5             # Atwood number A = (ρ_heavy - ρ_light)/(ρ_h + ρ_l)
GRAVITY = 1.0            # gravitational acceleration magnitude
DELTA = 0.03             # interface thickness (tanh smoothing length)
PERT_AMP = 0.05          # interface perturbation amplitude

NU = 1.0e-3              # kinematic viscosity
KAPPA = 1.0e-3           # scalar diffusivity (Schmidt/Prandtl number 1)

# Time scale τ = sqrt(W / (A g)); run several τ.
TAU = np.sqrt(W / (ATWOOD * GRAVITY))
T_FINAL = 6.0 * TAU
DT = 5.0e-3
SAVE_EVERY = 10

# Divergence + continuity penalty post-projection (Fehn 2018), on by
# default — the robustness audit's best-performing stabilisation, and
# the cure for the under-resolved mixing layer this case stresses.
# Characteristic velocity is the buoyancy scale sqrt(A g W).
USE_PENALTY = True
PENALTY_SCALE = 1.0
U_CHAR = np.sqrt(ATWOOD * GRAVITY * W)

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Boundary tagging + initial condition
# ---------------------------------------------------------------------------

def tag_velocity_bc(mesh):
    """Top/bottom faces no-slip wall; sides are periodic (interior)."""
    K, Nfaces = mesh.K, mesh.Nfaces
    EToE = np.asarray(mesh.EToE)
    is_bdy = EToE == np.arange(K)[:, None]   # only top/bottom survive
    bc = np.where(is_bdy, BC_INS_WALL, BC_INS_INTERIOR).astype(np.int32)
    return jnp.asarray(bc)


def initial_density(mesh):
    """Heavy fluid (c=+1) above a perturbed interface, light (c=-1)
    below. Single-mode cosine perturbation, tanh-smoothed."""
    x = np.asarray(mesh.x)
    y = np.asarray(mesh.y)
    y_interface = 0.5 * H + PERT_AMP * np.cos(2.0 * np.pi * x / W)
    c = np.tanh((y - y_interface) / DELTA)
    return jnp.asarray(c)


# ---------------------------------------------------------------------------
# Mixing-layer diagnostics
# ---------------------------------------------------------------------------

def mixing_fronts(mesh, c, thresh=0.8):
    """Spike (heavy sinking) and bubble (light rising) front positions.

    spike front = lowest y carrying near-pure heavy fluid (c > thresh);
    bubble front = highest y carrying near-pure light fluid
    (c < -thresh). Mixing-layer width = bubble_y - spike_y.
    """
    y = np.asarray(mesh.y).reshape(-1)
    cf = np.asarray(c).reshape(-1)
    heavy = y[cf > thresh]
    light = y[cf < -thresh]
    spike_y = float(heavy.min()) if heavy.size else 0.5 * H
    bubble_y = float(light.max()) if light.size else 0.5 * H
    return spike_y, bubble_y


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print(
        f"Rayleigh-Taylor: Kx={KX}, Ky={KY}, N={N}, "
        f"A={ATWOOD}, g={GRAVITY}, nu={NU}, kappa={KAPPA}, tau={TAU:.3f}"
    )
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, W, 0.0, H, KX, KY)
    mesh = build_mesh_2d(VX, VY, EToV, N)
    mesh = apply_rectangle_periodic_2d(
        mesh, xmin=0.0, xmax=W, ymin=0.0, ymax=H,
        periodic_x=True, periodic_y=False,
    )
    print(f"mesh: K={mesh.K} triangles, Np={mesh.Np}, "
          f"dofs/field={mesh.K * mesh.Np}")

    ins_bc_types = tag_velocity_bc(mesh)
    scalar_diff_bc = all_neumann_bc_types(mesh)   # no-flux top/bottom

    # Velocity at rest; density = perturbed two-layer profile.
    ux = jnp.zeros((mesh.Np, mesh.K))
    uy = jnp.zeros((mesh.Np, mesh.K))
    bc_ux = jnp.zeros((mesh.Np, mesh.K))
    bc_uy = jnp.zeros((mesh.Np, mesh.K))
    c = initial_density(mesh)

    print("LU-factoring velocity + scalar operators for BDF1 ...")
    t0 = time.time()
    ops_v1 = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=BDF1.g0, ins_bc_types=ins_bc_types,
    )
    ops_c1 = build_scalar_operators(
        mesh, dt=DT, kappa=KAPPA, g0=BDF1.g0,
        diffusion_bc_types=scalar_diff_bc,
    )
    print(f"  factor took {time.time() - t0:.1f}s")

    n_steps = int(np.ceil(T_FINAL / DT))
    print(f"running {n_steps} steps with dt = {DT}")

    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    c_old = c
    Nc_old = scalar_advection_2d(mesh, c, ux, uy)

    pen_cfg = None
    if USE_PENALTY:
        pen_cfg = make_penalty_config_dimensional(
            mesh, u_char=U_CHAR,
            scale_div=PENALTY_SCALE, scale_cont=PENALTY_SCALE,
        )
        print(f"  divergence/continuity penalty ON (scale={PENALTY_SCALE})")

    times, spike_series, bubble_series = [], [], []
    snapshots_c = []

    ops_v, ops_c, bdf = ops_v1, ops_c1, BDF1
    step_start = time.time()
    for step in range(1, n_steps + 1):
        if step == 2:
            ops_v = build_ins_operators(
                mesh, dt=DT, nu=NU, g0=BDF2.g0, ins_bc_types=ins_bc_types,
            )
            ops_c = build_scalar_operators(
                mesh, dt=DT, kappa=KAPPA, g0=BDF2.g0,
                diffusion_bc_types=scalar_diff_bc,
            )
            bdf = BDF2

        # 1) advance the density scalar with the current velocity.
        c_new, Nc_n = scalar_transport_step_2d(
            mesh, c, c_old, ux, uy, Nc_old,
            dt=DT, kappa=KAPPA, bdf=bdf, ops=ops_c,
        )
        # 2) buoyancy body force from the *new* density: f = (0, -A g c).
        fy = -ATWOOD * GRAVITY * c_new
        fx = jnp.zeros_like(fy)
        # 3) advance velocity with the buoyancy force.
        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=DT, nu=NU, bdf=bdf, ops=ops_v,
            ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
            body_force=(fx, fy),
            penalty_iterative=pen_cfg,
        )
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n
        c_old, c, Nc_old = c, c_new, Nc_n

        spike_y, bubble_y = mixing_fronts(mesh, c)
        times.append(step * DT)
        spike_series.append(spike_y)
        bubble_series.append(bubble_y)

        if step % 50 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            cmin, cmax = float(jnp.min(c)), float(jnp.max(c))
            print(f"  step {step:5d}/{n_steps}  t={step*DT:6.3f}  "
                  f"({step*DT/TAU:.2f}τ)  max|U|={max_u:.4f}  "
                  f"c∈[{cmin:+.3f},{cmax:+.3f}]  "
                  f"mix-width={bubble_y-spike_y:.4f}  elapsed={elapsed:.1f}s")
        if step % SAVE_EVERY == 0 or step == n_steps:
            snapshots_c.append(np.asarray(c))

    # ----- Mixing-layer growth plot -----
    spike = np.array(spike_series); bubble = np.array(bubble_series)
    t_arr = np.array(times)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_arr / TAU, (0.5 * H - spike), label="spike penetration (heavy ↓)")
    ax.plot(t_arr / TAU, (bubble - 0.5 * H), label="bubble penetration (light ↑)")
    ax.plot(t_arr / TAU, bubble - spike, "k--", label="mixing-layer width")
    ax.set_xlabel(r"$t / \tau$"); ax.set_ylabel("penetration")
    ax.set_title(f"Rayleigh-Taylor mixing growth   A={ATWOOD}, g={GRAVITY}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    growth_path = OUT_DIR / "ns_rayleigh_taylor_2d_growth.png"
    fig.savefig(growth_path, dpi=110)
    plt.close(fig)
    print(f"wrote {growth_path}")

    # ----- Density-field animation -----
    print("rendering density animation ...")
    from matplotlib.animation import FuncAnimation
    fig2, ax2 = plt.subplots(figsize=(4.5, 8))
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    levels = np.linspace(-1.0, 1.0, 21)
    save_times = times[SAVE_EVERY - 1::SAVE_EVERY]

    def draw(i):
        ax2.clear()
        ax2.tricontourf(px, py, snapshots_c[i].T.reshape(-1),
                        levels=levels, cmap="RdBu", extend="both")
        ax2.set_aspect("equal")
        ax2.set_title(f"density c   t = {save_times[i]:.2f} "
                      f"({save_times[i]/TAU:.2f}τ)")
        return []

    anim = FuncAnimation(fig2, draw, frames=len(snapshots_c),
                         interval=60, blit=False)
    out_mp4 = OUT_DIR / "ns_rayleigh_taylor_2d.mp4"
    anim.save(str(out_mp4), writer="ffmpeg", dpi=100)
    plt.close(fig2)
    print(f"wrote {out_mp4}")

    meta = {
        "benchmark": "Rayleigh-Taylor instability (Boussinesq)",
        "W": W, "H": H, "KX": KX, "KY": KY, "N": N,
        "K": int(mesh.K), "Np": int(mesh.Np),
        "ATWOOD": ATWOOD, "GRAVITY": GRAVITY,
        "NU": NU, "KAPPA": KAPPA, "DELTA": DELTA, "PERT_AMP": PERT_AMP,
        "TAU": float(TAU), "T_FINAL": T_FINAL, "DT": DT, "n_steps": n_steps,
        "final_mixing_width": float(bubble[-1] - spike[-1]),
        "final_spike_penetration": float(0.5 * H - spike[-1]),
        "final_bubble_penetration": float(bubble[-1] - 0.5 * H),
    }
    meta_path = OUT_DIR / "ns_rayleigh_taylor_2d_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
