"""Animate the inviscid 1D Burgers equation ``u_t + (u^2/2)_x = 0`` with a
slope-limited DG scheme on a periodic domain.

Initial condition ``u_0(x) = sin(x)`` on ``[0, 2*pi]``: characteristics
crossing at ``t = 1`` produce a stationary shock at ``x = pi`` that
persists for ``t > 1``. We show two solutions side by side so the
benefit of the limiter is visible:

* a Lax-Friedrichs DG run *without* the limiter, which develops Gibbs
  oscillations across the forming shock;
* the same run *with* the minmod slope limiter applied between time
  steps, which stays monotone.

Writes ``output/burgers_inviscid_1d.mp4``.
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
from ddg.dg1d.burgers import solve_burgers_1d_trajectory  # noqa: E402


def main() -> Path:
    N, K = 4, 32
    t_final = 1.5
    n_frames = 240

    VX, EToV = uniform_mesh_1d(0.0, 2.0 * np.pi, K)
    mesh = build_mesh_1d(VX, EToV, N=N, periodic=True)
    u0 = jnp.sin(mesh.x)

    u_no_lim, times = solve_burgers_1d_trajectory(
        mesh, u0, t_final, n_frames, nu=0.0, apply_limiter=False,
    )
    u_lim, _ = solve_burgers_1d_trajectory(
        mesh, u0, t_final, n_frames, nu=0.0, apply_limiter=True,
    )

    output_path = Path(__file__).resolve().parents[1] / "output" / "burgers_inviscid_1d.mp4"
    saved = save_field_animation(
        x=np.asarray(mesh.x),
        fields=[
            ("no limiter (Gibbs)", np.asarray(u_no_lim), "#1f77b4"),
            ("minmod-limited", np.asarray(u_lim), "#d62728"),
        ],
        times=np.asarray(times),
        output_path=output_path,
        title=f"1D inviscid Burgers, periodic sin IC (N={N}, K={K}, shock at t=1)",
        ylabel="u",
        fps=30,
    )
    print(f"wrote {saved}")
    return saved


if __name__ == "__main__":
    main()
