"""Taylor-Green vortex decay with in-the-loop adaptive mesh refinement.

The AMR analogue of :mod:`ns_taylor_green_2d`. Same manufactured
problem — on ``[0, π]²`` with ``ω = ψ = 0`` walls, the vortex

    ω(x, y, t) = 2 sin(x) sin(y) · exp(-2 ν t)

decays self-similarly — but started on a *coarse* mesh and adaptively
refined while marching: every ``ADAPT_EVERY`` RK4 steps a Kelly
indicator on ω flags the under-resolved elements, the refinement
forest red-refines them, ω is transferred onto the new mesh
(:func:`~ddg.dg2d.refinement.transfer_field`, exact), and the IPDG
Poisson matrix is rebuilt.

Purpose — AMR verification against a manufactured solution. Because
the exact ω is known everywhere, the error is unambiguous. The
uniform-mesh run reproduces the ``exp(-2 ν t)`` decay; this run
confirms the AMR machinery (refine + solution transfer + operator
rebuild, *inside* the time loop) reproduces it too. No numerics
orthogonal to AMR change: identical RHS, identical RK4 integrator.
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

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d  # noqa: E402
from ddg.dg2d.navier_stokes import vorticity_streamfunction_rhs  # noqa: E402
from ddg.dg2d.poisson import poisson_ip_matrix_2d  # noqa: E402
from ddg.dg1d.integrator import low_storage_rk4_step  # noqa: E402
from ddg.dg2d.refinement import (  # noqa: E402
    RefinementForest, kelly_error_indicator,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
L = math.pi
NU = 0.05
T_FINAL = 1.0
N_TIME_STEPS = 10000          # dt = 1e-4 — sized for the finest level

N = 4
K_SIDE = 4                    # coarse base mesh; AMR refines from here

# AMR: Kelly indicator on ω, Dörfler bulk marking, hard depth cap.
# MAX_LEVEL = 1 keeps the finest element at the K_SIDE=8-equivalent
# resolution of the validated uniform run, so the fixed dt stays safe.
MAX_LEVEL = 1
ADAPT_EVERY = 2000
THETA = 0.6

N_FRAMES = 200


def _omega_exact(mesh, t):
    """The manufactured solution sampled on a mesh's nodes."""
    return (2.0 * jnp.sin(mesh.x) * jnp.sin(mesh.y)
            * math.exp(-2.0 * NU * t))


def _l2_rel_error(mesh, omega, t):
    """Relative L2 error of ω against the exact solution at time t."""
    exact = np.asarray(_omega_exact(mesh, t))
    V = np.asarray(mesh.V)
    M = np.linalg.inv(V @ V.T)
    Jc = np.asarray(mesh.J)[0]
    diff = np.asarray(omega) - exact
    num = np.sum(Jc * np.einsum("ik,ij,jk->k", diff, M, diff))
    den = np.sum(Jc * np.einsum("ik,ij,jk->k", exact, M, exact))
    return float(np.sqrt(num / den))


def main():
    print(f"Taylor-Green vortex (AMR): base K_side={K_SIDE}, N={N}, "
          f"nu={NU}, T={T_FINAL}")

    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, L, 0.0, L, K_SIDE, K_SIDE)
    base_mesh = build_mesh_2d(VX, VY, EToV, N)
    forest = RefinementForest(base_mesh, max_level=MAX_LEVEL)
    mesh, a2l = forest.active_mesh()
    print(f"base mesh: K={mesh.K} triangles, Np={mesh.Np}")

    omega = _omega_exact(mesh, 0.0)
    A_poisson = poisson_ip_matrix_2d(mesh)

    dt = T_FINAL / N_TIME_STEPS
    frame_every = max(1, N_TIME_STEPS // N_FRAMES)

    times = [0.0]
    peaks_sim = [float(jnp.max(jnp.abs(omega)))]
    peaks_exact = [2.0]
    n_adapts = 0
    t = 0.0

    for step in range(1, N_TIME_STEPS + 1):
        rhs = lambda u, tt: vorticity_streamfunction_rhs(
            mesh, u, tt, nu=NU, A_poisson=A_poisson)
        omega = low_storage_rk4_step(rhs, omega, t, dt)
        t = step * dt

        if step % frame_every == 0 or step == N_TIME_STEPS:
            times.append(t)
            peaks_sim.append(float(jnp.max(jnp.abs(omega))))
            peaks_exact.append(2.0 * math.exp(-2.0 * NU * t))

        # --- adaptation pass ---------------------------------------
        if step % ADAPT_EVERY == 0 and step < N_TIME_STEPS:
            eta = kelly_error_indicator(mesh, omega)
            new_mesh, fields, a2l, n = forest.adapt(
                mesh, a2l, {"omega": omega}, eta, theta=THETA)
            if n > 0:
                n_adapts += 1
                mesh = new_mesh
                omega = fields["omega"]
                A_poisson = poisson_ip_matrix_2d(mesh)
                top = int(forest.leaf_levels().max())
                print(f"  [adapt {n_adapts}] step {step}: refined {n} "
                      f"leaves, K={mesh.K}, max level={top}")

    rel_err_final = _l2_rel_error(mesh, omega, T_FINAL)
    peak_rel_err = abs(peaks_sim[-1] - peaks_exact[-1]) / peaks_exact[-1]
    print(f"\nfinal mesh: K={mesh.K} triangles ({n_adapts} adaptations)")
    print(f"peak |ω| at T={T_FINAL}: sim {peaks_sim[-1]:.6f}, "
          f"exact {peaks_exact[-1]:.6f}, rel err {peak_rel_err:.3e}")
    print(f"L2 relative error vs manufactured solution: "
          f"{rel_err_final:.3e}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decay diagnostic.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, peaks_sim, "o", color="#d62728", ms=3,
            label="simulated max|ω| (AMR)")
    ax.plot(times, peaks_exact, "-", color="k", lw=1.5,
            label=r"exact $2\,e^{-2\nu t}$")
    ax.set_xlabel("t"); ax.set_ylabel("max |ω(·, t)|")
    ax.set_title(f"Taylor-Green decay with AMR  "
                 f"(ν={NU}, N={N}, base K_side={K_SIDE}, K_final={mesh.K})")
    ax.grid(alpha=0.3); ax.legend(loc="upper right")
    fig.tight_layout()
    png_path = out_dir / "ns_taylor_green_amr_2d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    # Final ω field + the adapted mesh.
    fig2, ax2 = plt.subplots(figsize=(6, 5.5))
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    om = np.asarray(omega).T.reshape(-1)
    tpc = ax2.tricontourf(px, py, om, levels=24, cmap="RdBu_r")
    ax2.triplot(np.asarray(mesh.VX), np.asarray(mesh.VY),
                np.asarray(mesh.EToV), color="k", lw=0.25, alpha=0.4)
    ax2.set_aspect("equal")
    ax2.set_title(f"ω at T={T_FINAL} + adapted mesh")
    fig2.colorbar(tpc, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    field_path = out_dir / "ns_taylor_green_amr_2d_field.png"
    fig2.savefig(field_path, dpi=140)
    plt.close(fig2)
    print(f"wrote {field_path}")

    meta = {
        "benchmark": "Taylor-Green vortex decay (AMR)",
        "nu": NU, "T_final": T_FINAL, "n_time_steps": N_TIME_STEPS,
        "N": N, "base_K_side": K_SIDE,
        "max_level": MAX_LEVEL, "adapt_every": ADAPT_EVERY,
        "theta": THETA, "n_adapts": n_adapts,
        "K_base": int(base_mesh.K), "K_final": int(mesh.K),
        "peak_rel_err_final": peak_rel_err,
        "l2_rel_err_final": rel_err_final,
    }
    meta_path = out_dir / "ns_taylor_green_amr_2d.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
