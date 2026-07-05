"""1D channel flow (Poiseuille) — the natural "1D NS" demo.

Strictly speaking, 1D incompressible Navier-Stokes is degenerate:
``∇·u = 0`` in 1D forces ``u = const(x)``, so there's no flow at all.
The standard interpretation of "1D NS" is therefore a **channel-flow
reduction** — we assume the 2D velocity field is purely streamwise
``u(y, t) ê_x`` and depends only on the *cross-channel* coordinate
``y``. The momentum equation along ``x`` then reduces to

    ρ ∂u/∂t = -∂p/∂x + μ ∂²u/∂y²

i.e. a 1D heat equation with a **body-force / pressure-gradient
source** on the right-hand side. With a constant streamwise pressure
gradient ``G = -∂p/∂x``, no-slip walls ``u(0, t) = u(1, t) = 0``, and
the rod starting at rest, the steady-state velocity profile is the
textbook Poiseuille parabola

    u_ss(y) = G / (2 ν) · y (1 - y).

This is exactly ``heat_rhs(mesh, u, t, kappa=ν, forcing_fn=lambda t: G)``,
so no new solver is needed — we just exercise the ``forcing_fn``
extension to the existing heat code. The demo animates the transient
``u(y, t)`` developing from rest to the steady parabola.

Outputs (``output/``)
---------------------
* ``ins_poiseuille_1d.png``: u(y, t) at a series of times, with the
  analytical steady profile overlaid; |u_h(T) - u_ss| convergence.
* ``ins_poiseuille_1d.mp4`` / ``.gif``: transient animation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _anim_utils import save_field_animation  # noqa: E402

from ddg.dg1d import build_mesh_1d, uniform_mesh_1d  # noqa: E402
from ddg.dg1d.heat import solve_heat_trajectory  # noqa: E402


YMIN, YMAX = 0.0, 1.0
NU = 0.1            # kinematic viscosity
G_FORCING = 1.0     # constant pressure gradient: -dp/dx
T_FINAL = 5.0       # ~ 5 relaxation times (τ = L² / (π² ν) ≈ 1)
N, K = 5, 16


def u_steady_state(y: jax.Array) -> jax.Array:
    """Analytical Poiseuille profile ``u_ss(y) = G / (2 ν) y (1 - y)``."""
    return G_FORCING / (2.0 * NU) * y * (1.0 - y)


def main() -> Path:
    VX, EToV = uniform_mesh_1d(YMIN, YMAX, K)
    mesh = build_mesh_1d(VX, EToV, N=N)

    u0 = jnp.zeros_like(mesh.x)
    n_frames = 240
    u_traj, times = solve_heat_trajectory(
        mesh, u0, T_FINAL, n_frames,
        kappa=NU,
        forcing_fn=lambda t: G_FORCING,
    )

    # Final-time error vs analytical Poiseuille parabola.
    u_ss = u_steady_state(mesh.x)
    err = float(jnp.linalg.norm(u_traj[-1] - u_ss) / jnp.linalg.norm(u_ss))
    print(f"|u_h(T) - u_ss| / |u_ss| at T={T_FINAL}: {err:.3e}")
    print(f"steady-state peak u_ss(0.5) = G/(8ν) = {G_FORCING / (8 * NU):.4f}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    output_path = out_dir / "ins_poiseuille_1d.mp4"
    saved = save_field_animation(
        x=np.asarray(mesh.x),
        fields=[
            ("u_h(y, t)", np.asarray(u_traj), "#1f77b4"),
            ("u_ss = G/(2ν) y(1-y)",
             np.broadcast_to(np.asarray(u_ss)[None], np.asarray(u_traj).shape),
             "#d62728"),
        ],
        times=np.asarray(times),
        output_path=output_path,
        title=f"1D channel flow / Poiseuille  (ν={NU}, G={G_FORCING}, N={N}, K={K})",
        ylabel="u(y, t)",
        fps=30,
    )
    print(f"wrote {saved}")
    return saved


if __name__ == "__main__":
    main()
