"""Animate 1D linear advection of a Gaussian pulse on a periodic domain.

Writes ``output/advection_1d.mp4``. Run with the project conda env:

    JAX_PLATFORMS=cpu conda run -n ddg python applications/advection_1d_animation.py
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

# Make the sibling helper importable when running this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _anim_utils import save_field_animation  # noqa: E402

from ddg.dg1d import (  # noqa: E402
    build_mesh_1d,
    solve_advection_trajectory,
    uniform_mesh_1d,
)


def main() -> Path:
    xmin, xmax = 0.0, 2.0 * np.pi
    N, K = 8, 16
    a = 2.0 * np.pi  # one period per unit time
    t_final = 2.0
    n_frames = 240

    VX, EToV = uniform_mesh_1d(xmin, xmax, K)
    mesh = build_mesh_1d(VX, EToV, N=N, periodic=True)

    # Gaussian centered at pi; clearer to follow than the analytical sin test.
    sigma = 0.4
    x_c = np.pi
    u0 = jnp.exp(-((mesh.x - x_c) ** 2) / (2.0 * sigma**2))

    u_traj, times = solve_advection_trajectory(mesh, u0, t_final, a, n_frames)

    output_path = Path(__file__).resolve().parents[1] / "output" / "advection_1d.mp4"
    saved = save_field_animation(
        x=np.asarray(mesh.x),
        fields=[("u", np.asarray(u_traj), "#1f77b4")],
        times=np.asarray(times),
        output_path=output_path,
        title=f"1D advection (a = 2π, periodic, N={N}, K={K})",
        ylabel="u(x, t)",
        fps=30,
    )
    print(f"wrote {saved}")
    return saved


if __name__ == "__main__":
    main()
