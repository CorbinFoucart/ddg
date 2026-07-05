"""Animate the 1D viscous Burgers traveling wave from the Hesthaven driver.

``u_t + (u^2/2)_x = nu u_xx`` on ``[-1, 1]`` with Dirichlet boundary
data set to the exact traveling-wave solution

    u_exact(x, t) = 1 - tanh((x + 0.5 - t) / (2 nu))

so the discrete wave travels at speed 1 and the front sits at
``x = -0.5 + t``. Writes ``output/burgers_viscous_1d.mp4``.
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
    nu = 0.1
    N, K = 4, 8
    t_final = 1.5
    n_frames = 240

    VX, EToV = uniform_mesh_1d(-1.0, 1.0, K)
    mesh = build_mesh_1d(VX, EToV, N=N)

    def u_exact(x, t):
        return 1.0 - jnp.tanh((x + 0.5 - t) / (2.0 * nu))

    inflow = lambda t: u_exact(jnp.array(-1.0), t)
    outflow = lambda t: u_exact(jnp.array(1.0), t)
    u0 = u_exact(mesh.x, 0.0)

    u_traj, times = solve_burgers_1d_trajectory(
        mesh, u0, t_final, n_frames,
        nu=nu, inflow_fn=inflow, outflow_fn=outflow,
    )

    output_path = Path(__file__).resolve().parents[1] / "output" / "burgers_viscous_1d.mp4"
    saved = save_field_animation(
        x=np.asarray(mesh.x),
        fields=[("u(x, t)", np.asarray(u_traj), "#8c564b")],
        times=np.asarray(times),
        output_path=output_path,
        title=f"1D viscous Burgers (nu={nu}, N={N}, K={K})",
        ylabel="u",
        fps=30,
    )
    print(f"wrote {saved}")
    return saved


if __name__ == "__main__":
    main()
