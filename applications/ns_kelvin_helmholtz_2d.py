"""Kelvin-Helmholtz instability: the doubly-periodic double shear layer.

A doubly-periodic unit square carrying two horizontal shear layers.
A small sinusoidal vertical-velocity perturbation seeds the
Kelvin-Helmholtz instability; each shear layer rolls up into a row
of "cat's-eye" vortices. This is the Bell-Colella-Glaz / Minion-Brown
double-shear-layer test, a standard probe of a convective scheme's
behaviour on under-resolved vortical flow.

Initial condition (divergence-free by construction — ``u_x`` depends
only on ``y``, ``u_y`` only on ``x``):

    u_x = tanh((y - 1/4) / ρ)     for y ≤ 1/2
    u_x = tanh((3/4 - y) / ρ)     for y > 1/2
    u_y = δ · sin(2π x)

with shear-layer thickness ``ρ`` and perturbation amplitude ``δ``.

There are no walls — every face is interior — so the pressure
Poisson is singular up to a constant; build_ins_operators detects
this and regularises the factorisation automatically.

Diagnostics: kinetic energy and enstrophy versus time (energy should
decay only slowly through viscosity; enstrophy spikes as the layers
roll up), plus a vorticity animation of the rollup.
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
    BC_INS_INTERIOR, BDF1, BDF2, build_ins_operators,
    make_penalty_config_dimensional, ns_advection_2d, ns_step_2d,
)
from ddg.dg2d.ins_metrics import kinetic_energy, enstrophy
from ddg.dg2d.maxwell import curl_z_2d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0

KX, KY = 32, 32          # even -> opposite edges discretised identically
N = 4

RHO = 1.0 / 30.0         # shear-layer thickness
DELTA = 0.05             # perturbation amplitude
RE = 1000.0              # Re = U·L/ν,  U = L = 1
NU = 1.0 / RE
U_CHAR = 1.0             # shear velocity scale (tanh amplitude)

# Divergence + continuity penalty post-projection (Fehn 2018), on by
# default — the robustness audit's best-performing stabilisation, and
# the cure for the under-resolved free-shear rollup that this case
# stresses. Element-local dimensional scaling τ ~ PENALTY_SCALE·h·U.
USE_PENALTY = True
PENALTY_SCALE = 1.0

T_FINAL = 2.0
DT = 2.0e-3
SAVE_EVERY = 5

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def double_shear_layer_ic(mesh):
    """Divergence-free double-shear-layer initial state."""
    x = np.asarray(mesh.x)
    y = np.asarray(mesh.y)
    lower = np.tanh((y - 0.25) / RHO)
    upper = np.tanh((0.75 - y) / RHO)
    ux = np.where(y <= 0.5, lower, upper)
    uy = DELTA * np.sin(2.0 * np.pi * x)
    return jnp.asarray(ux), jnp.asarray(uy)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print(
        f"Kelvin-Helmholtz double shear layer: Kx={KX}, Ky={KY}, "
        f"N={N}, Re={RE:.0f} (nu={NU:.4g})"
    )
    VX, VY, EToV = uniform_rectangle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, KX, KY,
    )
    mesh = build_mesh_2d(VX, VY, EToV, N)
    mesh = apply_rectangle_periodic_2d(
        mesh, xmin=X_MIN, xmax=X_MAX, ymin=Y_MIN, ymax=Y_MAX,
        periodic_x=True, periodic_y=True,
    )
    n_bdy = int(len(np.asarray(mesh.mapB)))
    print(f"mesh: K={mesh.K} triangles, Np={mesh.Np}, "
          f"dofs/field={mesh.K * mesh.Np}, boundary nodes={n_bdy} "
          f"(0 => fully periodic)")

    # Fully-periodic: every face interior, no Dirichlet velocity trace.
    ins_bc_types = jnp.full((mesh.K, mesh.Nfaces), BC_INS_INTERIOR,
                            dtype=jnp.int32)
    bc_ux = jnp.zeros((mesh.Np, mesh.K))
    bc_uy = jnp.zeros((mesh.Np, mesh.K))

    ux, uy = double_shear_layer_ic(mesh)
    div0 = float(jnp.max(jnp.abs(
        (mesh.rx * (mesh.Dr @ ux) + mesh.sx * (mesh.Ds @ ux))
        + (mesh.ry * (mesh.Dr @ uy) + mesh.sy * (mesh.Ds @ uy))
    )))
    print(f"initial max|div u| = {div0:.3e}")

    print("LU-factoring pressure + viscous matrices for BDF1 ...")
    t0 = time.time()
    ops_bdf1 = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=BDF1.g0, ins_bc_types=ins_bc_types,
    )
    print(f"  factor took {time.time() - t0:.1f}s")

    n_steps = int(np.ceil(T_FINAL / DT))
    print(f"running {n_steps} steps with dt = {DT}")

    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))

    ke0 = kinetic_energy(mesh, ux, uy)
    times, ke_series, ens_series = [], [], []
    snapshots_w = []

    pen_cfg = None
    if USE_PENALTY:
        pen_cfg = make_penalty_config_dimensional(
            mesh, u_char=U_CHAR,
            scale_div=PENALTY_SCALE, scale_cont=PENALTY_SCALE,
        )
        print(f"  divergence/continuity penalty ON (scale={PENALTY_SCALE})")

    ops_current, bdf_current = ops_bdf1, BDF1
    step_start = time.time()
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
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n

        ke = kinetic_energy(mesh, ux, uy)
        ens = enstrophy(mesh, ux, uy)
        times.append(step * DT)
        ke_series.append(float(ke))
        ens_series.append(float(ens))

        if step % 50 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            print(f"  step {step:5d}/{n_steps}  t={step*DT:5.2f}  "
                  f"max|U|={max_u:.4f}  KE/KE0={float(ke)/float(ke0):.5f}  "
                  f"enstrophy={float(ens):.2f}  elapsed={elapsed:.1f}s")
        if step % SAVE_EVERY == 0 or step == n_steps:
            snapshots_w.append(np.asarray(curl_z_2d(mesh, ux, uy)))

    # ----- Diagnostics plot -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, np.array(ke_series) / float(ke0))
    axes[0].set_xlabel("$t$"); axes[0].set_ylabel("$KE(t) / KE(0)$")
    axes[0].set_title("kinetic energy (slow viscous decay)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, ens_series, color="C1")
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel("enstrophy")
    axes[1].set_title("enstrophy (spikes as layers roll up)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Kelvin-Helmholtz   Re={RE:.0f}   N={N}, {KX}×{KY} cells")
    fig.tight_layout()
    diag_path = OUT_DIR / "ns_kelvin_helmholtz_2d_diagnostics.png"
    fig.savefig(diag_path, dpi=100)
    plt.close(fig)
    print(f"wrote {diag_path}")

    # ----- Vorticity animation -----
    print("rendering vorticity animation ...")
    from matplotlib.animation import FuncAnimation
    fig2, ax2 = plt.subplots(figsize=(6, 5.5))
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    wmax = float(np.percentile(np.abs(np.array(snapshots_w)), 99))
    levels = np.linspace(-wmax, wmax, 25)
    save_times = times[SAVE_EVERY - 1::SAVE_EVERY]

    def draw(i):
        ax2.clear()
        ax2.tricontourf(px, py, snapshots_w[i].T.reshape(-1),
                        levels=levels, cmap="RdBu_r", extend="both")
        ax2.set_aspect("equal")
        ax2.set_title(f"vorticity $\\omega$   t = {save_times[i]:.2f}")
        return []

    anim = FuncAnimation(fig2, draw, frames=len(snapshots_w),
                         interval=60, blit=False)
    out_mp4 = OUT_DIR / "ns_kelvin_helmholtz_2d.mp4"
    anim.save(str(out_mp4), writer="ffmpeg", dpi=100)
    plt.close(fig2)
    print(f"wrote {out_mp4}")

    meta = {
        "benchmark": "Kelvin-Helmholtz double shear layer",
        "X": [X_MIN, X_MAX], "Y": [Y_MIN, Y_MAX],
        "KX": KX, "KY": KY, "N": N,
        "K": int(mesh.K), "Np": int(mesh.Np),
        "RHO": RHO, "DELTA": DELTA, "Re": RE, "NU": NU,
        "T_FINAL": T_FINAL, "DT": DT, "n_steps": n_steps,
        "initial_max_div": div0,
        "KE_final_over_KE0": ke_series[-1] / float(ke0),
        "enstrophy_final": ens_series[-1],
        "enstrophy_max": max(ens_series),
    }
    meta_path = OUT_DIR / "ns_kelvin_helmholtz_2d_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
