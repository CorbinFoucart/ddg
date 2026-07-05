"""Animate 2D TM Maxwell in a square PEC cavity.

Initial pulse: localised Gaussian for ``Ez`` with ``Hx = Hy = 0``.
PEC walls reflect the wave; the pulse splits, bounces, and re-coheres
into nontrivial standing-wave patterns over time.

Writes ``output/maxwell_2d.mp4`` and a per-snapshot energy series to
``output/maxwell_2d_energy.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _anim_utils_2d import save_scalar_field_animation_2d  # noqa: E402

from ddg.dg2d import (  # noqa: E402
    build_mesh_2d,
    solve_maxwell_2d_trajectory,
    uniform_rectangle_mesh_2d,
)


def main() -> Path:
    K_side = 24
    N = 4
    t_final = 3.0
    n_frames = 240

    VX, VY, EToV = uniform_rectangle_mesh_2d(-1.0, 1.0, -1.0, 1.0, K_side, K_side)
    mesh = build_mesh_2d(VX, VY, EToV, N=N)
    print(f"K = {mesh.K} triangles, N = {N}, Np = {mesh.Np}, dofs/field = {mesh.K * mesh.Np}")

    # Localised Gaussian for Ez, no initial magnetic field.
    sigma = 0.12
    Ez0 = jnp.exp(-((mesh.x - 0.2) ** 2 + (mesh.y - 0.0) ** 2) / (2.0 * sigma**2))
    Hx0 = jnp.zeros_like(mesh.x)
    Hy0 = jnp.zeros_like(mesh.x)

    Hx_t, Hy_t, Ez_t, times = solve_maxwell_2d_trajectory(
        mesh, Hx0, Hy0, Ez0, t_final, n_frames
    )

    # Energy diagnostic.
    M_ref = np.asarray(jnp.linalg.inv(mesh.V @ mesh.V.T))
    J_np = np.asarray(mesh.J)
    energies = []
    for i in range(Ez_t.shape[0]):
        Hx_i = np.asarray(Hx_t[i])
        Hy_i = np.asarray(Hy_t[i])
        Ez_i = np.asarray(Ez_t[i])
        e = 0.5 * np.sum(J_np * (Hx_i * (M_ref @ Hx_i) + Hy_i * (M_ref @ Hy_i) + Ez_i * (M_ref @ Ez_i)))
        energies.append(float(e))
    print(f"energy(t=0) = {energies[0]:.4e}  energy(t_final) = {energies[-1]:.4e}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "maxwell_2d.mp4"

    saved = save_scalar_field_animation_2d(
        mesh=mesh,
        field_traj=np.asarray(Ez_t),
        times=np.asarray(times),
        output_path=output_path,
        title=f"2D Maxwell TM: Ez field (N={N}, K={mesh.K} triangles, PEC walls)",
        cmap="RdBu_r",
        fps=30,
        symmetric=True,
        show_mesh=False,
    )
    print(f"wrote {saved}")

    energy_path = out_dir / "maxwell_2d_energy.json"
    with open(energy_path, "w") as fh:
        json.dump({"times": [float(t) for t in times], "energy": energies}, fh, indent=2)
    print(f"wrote {energy_path}")
    return saved


if __name__ == "__main__":
    main()
