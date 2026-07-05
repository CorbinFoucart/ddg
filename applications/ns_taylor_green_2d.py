"""2D incompressible Navier-Stokes: Taylor-Green vortex decay.

The textbook verification problem for 2D NS in vorticity-streamfunction
form. On the square ``[0, π]²`` with free-slip walls
(``ω = ψ = 0`` on ∂Ω), the initial condition

    ω(x, y, 0) = 2 sin(x) sin(y)

decays self-similarly under viscosity ν as

    ω(x, y, t) = 2 sin(x) sin(y) · exp(-2 ν t).

The flow keeps the same spatial shape forever; only the amplitude
shrinks at the predicted exponential rate. Comparing the simulated
amplitude vs ``exp(-2 ν t)`` over the entire run validates the whole
NS stack — vorticity transport, streamfunction Poisson solve, gradient
of ψ, all wired together.

Outputs (``output/``)
---------------------
* ``ns_taylor_green_2d.mp4`` / ``.gif``: animation of ω(x, y, t).
* ``ns_taylor_green_2d.png``: amplitude-decay diagnostic
  (simulated vs ``exp(-2 ν t)``).
* ``ns_taylor_green_2d.json``: numerical record.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _anim_utils_2d import save_scalar_field_animation_2d  # noqa: E402

from ddg.dg2d import build_mesh_2d, uniform_rectangle_mesh_2d  # noqa: E402
from ddg.dg2d.navier_stokes import (  # noqa: E402
    solve_vorticity_streamfunction_trajectory,
)
from ddg.dg2d.poisson import poisson_ip_matrix_2d  # noqa: E402


L = math.pi
NU = 0.05
T_FINAL = 1.0
N_TIME_STEPS = 10000
N_FRAMES = 200


def main() -> Path:
    N = 4
    K_side = 8

    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, L, 0.0, L, K_side, K_side)
    mesh = build_mesh_2d(VX, VY, EToV, N=N)
    print(f"{mesh.K} triangles, Np = {mesh.Np}, dofs = {mesh.K * mesh.Np}")

    omega0 = 2.0 * jnp.sin(mesh.x) * jnp.sin(mesh.y)
    A_poisson = poisson_ip_matrix_2d(mesh)

    omega_traj, times = solve_vorticity_streamfunction_trajectory(
        mesh, omega0, T_FINAL, N_FRAMES,
        nu=NU, n_time_steps=N_TIME_STEPS, A_poisson=A_poisson,
    )

    # Amplitude diagnostic.
    omega_traj_np = np.asarray(omega_traj)
    times_np = np.asarray(times)
    peaks_sim = np.array([float(np.max(np.abs(omega_traj_np[i])))
                           for i in range(N_FRAMES + 1)])
    peaks_analytic = 2.0 * np.exp(-2.0 * NU * times_np)
    rel_err = np.abs(peaks_sim - peaks_analytic) / peaks_analytic
    print(f"peak |ω| at T = {T_FINAL}: simulated {peaks_sim[-1]:.6f}, "
          f"analytical {peaks_analytic[-1]:.6f}, rel err {rel_err[-1]:.3e}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_np, peaks_sim, "o", color="#d62728", markersize=4,
            label="simulated max|ω|(t)")
    ax.plot(times_np, peaks_analytic, "-", color="k", lw=1.5,
            label=r"analytical $2 \cdot e^{-2 \nu t}$")
    ax.set_xlabel("t")
    ax.set_ylabel("max |ω(·, t)|")
    ax.set_title(f"Taylor-Green decay  (ν = {NU}, N = {N}, K = {mesh.K})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    png_path = out_dir / "ns_taylor_green_2d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "ns_taylor_green_2d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": int(mesh.K), "K_side": K_side, "T_final": T_FINAL,
                "nu": NU, "n_time_steps": N_TIME_STEPS, "n_frames": N_FRAMES,
                "times": times_np.tolist(),
                "peaks_simulated": peaks_sim.tolist(),
                "peaks_analytic": peaks_analytic.tolist(),
                "rel_err_final": float(rel_err[-1]),
            },
            fh, indent=2,
        )
    print(f"wrote {json_path}")

    mp4_path = out_dir / "ns_taylor_green_2d.mp4"
    save_scalar_field_animation_2d(
        mesh=mesh,
        field_traj=omega_traj_np,
        times=times_np,
        output_path=mp4_path,
        title=f"Taylor-Green vortex decay: ω(x, y, t)  (ν = {NU})",
        cmap="RdBu_r",
        fps=30,
        symmetric=True,
    )
    print(f"wrote {mp4_path}")
    return png_path


if __name__ == "__main__":
    main()
