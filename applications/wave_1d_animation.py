"""Animate the 1D Maxwell-style wave system in a homogeneous medium with PEC walls.

The combined system ``E_t = -H_x``, ``H_t = -E_x`` gives the scalar wave
equation ``E_tt = E_xx`` with speed ``c = 1``. Writes
``output/wave_1d.mp4``. Run with:

    JAX_PLATFORMS=cpu conda run -n ddg python applications/wave_1d_animation.py
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

from ddg.dg1d import build_mesh_1d, solve_wave_trajectory, uniform_mesh_1d  # noqa: E402


def main() -> Path:
    N, K = 6, 80
    t_final = 10.0
    n_frames = 300

    VX, EToV = uniform_mesh_1d(-1.0, 1.0, K)
    mesh = build_mesh_1d(VX, EToV, N=N)  # PEC walls (non-periodic)

    # The Hesthaven driver IC: half-sine on the left half, zero elsewhere.
    E0 = jnp.sin(jnp.pi * mesh.x) * (mesh.x < 0.0)
    H0 = jnp.zeros_like(mesh.x)

    E_traj, H_traj, times = solve_wave_trajectory(mesh, E0, H0, t_final, n_frames)

    output_path = Path(__file__).resolve().parents[1] / "output" / "wave_1d.mp4"
    saved = save_field_animation(
        x=np.asarray(mesh.x),
        fields=[
            ("E", np.asarray(E_traj), "#d62728"),
            ("H", np.asarray(H_traj), "#1f77b4"),
        ],
        times=np.asarray(times),
        output_path=output_path,
        title=f"1D wave (Maxwell, PEC walls, N={N}, K={K})",
        ylabel="field amplitude",
        fps=30,
    )
    print(f"wrote {saved}")
    return saved


if __name__ == "__main__":
    main()
