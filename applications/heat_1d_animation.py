"""Animate 1D heat diffusion of a Gaussian on [0, 1] with Dirichlet u=0 BCs.

Writes ``output/heat_1d.mp4``. Run with:

    JAX_PLATFORMS=cpu conda run -n ddg python applications/heat_1d_animation.py
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

from ddg.dg1d import build_mesh_1d, solve_heat_trajectory, uniform_mesh_1d  # noqa: E402


def main() -> Path:
    N, K = 5, 16
    kappa = 1.0
    t_final = 0.15
    n_frames = 240

    VX, EToV = uniform_mesh_1d(0.0, 1.0, K)
    mesh = build_mesh_1d(VX, EToV, N=N)

    u0 = jnp.exp(-80.0 * (mesh.x - 0.5) ** 2)
    u_traj, times = solve_heat_trajectory(mesh, u0, t_final, n_frames, kappa=kappa)

    output_path = Path(__file__).resolve().parents[1] / "output" / "heat_1d.mp4"
    saved = save_field_animation(
        x=np.asarray(mesh.x),
        fields=[("u", np.asarray(u_traj), "#2ca02c")],
        times=np.asarray(times),
        output_path=output_path,
        title=f"1D heat diffusion (kappa={kappa}, Dirichlet u=0, N={N}, K={K})",
        ylabel="u(x, t)",
        fps=30,
    )
    print(f"wrote {saved}")
    return saved


if __name__ == "__main__":
    main()
