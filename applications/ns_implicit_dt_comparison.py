"""Explicit RK4 vs implicit Newton-Krylov for the vorticity-streamfunction
NS solver: how much can we push ``dt`` before each scheme falls apart?

Setup is the Taylor-Green vortex on ``[0, π]²`` (same as
``ns_taylor_green_2d.py``), where the analytical decay rate
``max|ω(t)| = 2 exp(-2 ν t)`` lets us measure the actual error at the
final time as a function of the time step ``dt``. We sweep ``dt`` for
both schemes and plot:

* the final-time error (rel. to the analytical decay), and
* whether the run is stable at all (explicit blows up past a CFL-set
  threshold; implicit stays accurate at any dt).

The implicit step uses backward Euler with a Newton iteration; each
Newton update solves ``(I - dt · J_R) δ = -G`` via
``jax.scipy.sparse.linalg.gmres``, with the Jacobian-vector product
``J_R · v`` computed matrix-free using ``jax.jvp`` on the NS RHS. The
Jacobian is never materialised.

Outputs (``output/``)
---------------------
* ``ns_implicit_dt_comparison.png``: error-vs-dt log-log plot for the
  two schemes, with the explicit stability cliff annotated.
* ``ns_implicit_dt_comparison.json``: numerical record.
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

from ddg.dg2d import build_mesh_2d, uniform_rectangle_mesh_2d  # noqa: E402
from ddg.dg2d.navier_stokes import (  # noqa: E402
    solve_vorticity_streamfunction,
    solve_vorticity_streamfunction_implicit,
)
from ddg.dg2d.poisson import poisson_ip_matrix_2d  # noqa: E402


def main() -> Path:
    L = math.pi
    NU = 0.05
    T_FINAL = 0.5

    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, L, 0.0, L, 8, 8)
    mesh = build_mesh_2d(VX, VY, EToV, N=4)
    A_poisson = poisson_ip_matrix_2d(mesh)

    omega0 = 2.0 * jnp.sin(mesh.x) * jnp.sin(mesh.y)
    decay_target = 2.0 * math.exp(-2.0 * NU * T_FINAL)

    explicit_results: list[dict] = []
    explicit_n_steps = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    for n in explicit_n_steps:
        dt = T_FINAL / n
        try:
            omega = solve_vorticity_streamfunction(
                mesh, omega0, T_FINAL, nu=NU, n_time_steps=n, A_poisson=A_poisson,
            )
            peak = float(jnp.max(jnp.abs(omega)))
            stable = bool(jnp.all(jnp.isfinite(omega)))
            err = abs(peak - decay_target) / decay_target if stable else float("nan")
        except Exception:
            stable, peak, err = False, float("nan"), float("nan")
        explicit_results.append(
            {"n_steps": n, "dt": dt, "stable": stable,
             "peak": peak, "err": err}
        )
        print(f"explicit  n={n:>5} dt={dt:.2e}  stable={stable}  peak={peak:.4f}  err={err:.3e}")

    print()
    implicit_results: list[dict] = []
    implicit_n_steps = [5, 10, 25, 50, 100, 250, 500]
    for n in implicit_n_steps:
        dt = T_FINAL / n
        omega, diags = solve_vorticity_streamfunction_implicit(
            mesh, omega0, T_FINAL, nu=NU, n_time_steps=n, A_poisson=A_poisson,
        )
        peak = float(jnp.max(jnp.abs(omega)))
        err = abs(peak - decay_target) / decay_target
        avg_iters = sum(d["newton_iters"] for d in diags) / len(diags)
        implicit_results.append(
            {"n_steps": n, "dt": dt, "peak": peak, "err": err,
             "avg_newton_iters": float(avg_iters)}
        )
        print(f"implicit  n={n:>5} dt={dt:.2e}  peak={peak:.4f}  err={err:.3e}  Newton avg {avg_iters:.1f}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    dt_exp = [r["dt"] for r in explicit_results]
    err_exp = [r["err"] if r["stable"] else None for r in explicit_results]
    dt_exp_stable = [d for d, e in zip(dt_exp, err_exp) if e is not None]
    err_exp_stable = [e for e in err_exp if e is not None]
    dt_exp_unstable = [d for d, e in zip(dt_exp, err_exp) if e is None]
    if dt_exp_stable:
        ax.loglog(dt_exp_stable, err_exp_stable, "o-", color="#1f77b4",
                  markersize=8, lw=1.5, label="explicit RK4 (stable)")
    if dt_exp_unstable:
        # Mark unstable runs at top of plot.
        ax.scatter(dt_exp_unstable, [1.0] * len(dt_exp_unstable),
                   marker="x", s=120, color="#1f77b4",
                   label="explicit RK4 (NaN / blew up)")

    dt_imp = [r["dt"] for r in implicit_results]
    err_imp = [r["err"] for r in implicit_results]
    ax.loglog(dt_imp, err_imp, "s-", color="#d62728", markersize=8,
              lw=1.5, label="implicit backward-Euler + Newton-Krylov")

    ax.set_xlabel("time step dt")
    ax.set_ylabel("relative error in max|ω(T)| vs analytical decay")
    ax.set_title(
        f"Explicit RK4 vs implicit Newton-Krylov for Taylor-Green decay  "
        f"(ν={NU}, T={T_FINAL})"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    png_path = out_dir / "ns_implicit_dt_comparison.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png_path}")

    json_path = out_dir / "ns_implicit_dt_comparison.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "T_final": T_FINAL, "nu": NU,
                "explicit": explicit_results,
                "implicit": implicit_results,
            },
            fh, indent=2,
        )
    print(f"wrote {json_path}")
    return png_path


if __name__ == "__main__":
    main()
