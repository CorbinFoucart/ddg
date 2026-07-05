"""DIRK / SDIRK convergence rates on a scalar ODE.

Pulls out just the time-integration order verification — no spatial
discretisation involved, so each method shows its theoretical order
cleanly across the whole dt range. We run the four implicit schemes
(BE / CN / SDIRK-2 / SDIRK-3) on a stiff linear ODE and a nonlinear
ODE (where the per-stage Newton iteration is genuinely nonlinear) and
plot ``log(err) vs log(dt)`` for each. The dotted reference lines have
the theoretical order's slope.

The same verification is encoded as a pytest in
``test/test_dirk_convergence_ode.py``; this script just makes the
result visible.

Outputs (``output/``)
---------------------
* ``dirk_convergence_ode.png``: side-by-side log-log plots for the
  linear and nonlinear ODEs.
* ``dirk_convergence_ode.json``: numerical record (errors and slopes).
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

from ddg.dg2d.navier_stokes import DIRK_TABLEAUX, _dirk_step  # noqa: E402


METHOD_DISPLAY = {
    "be":     ("Backward Euler",   "#1f77b4", 1),
    "cn":     ("Crank-Nicolson",   "#2ca02c", 2),
    "sdirk2": ("SDIRK-2 Crouzeix", "#ff7f0e", 2),
    "sdirk3": ("SDIRK-3 Alexander", "#d62728", 3),
}


def _integrate(rhs_flat, u0, t_final, dt, method):
    n_steps = int(round(t_final / dt))
    dt_actual = t_final / n_steps
    u = u0
    for _ in range(n_steps):
        u, _ = _dirk_step(
            rhs_flat, u, dt_actual, DIRK_TABLEAUX[method],
            newton_tol=1e-12, newton_max_iter=20,
            gmres_tol=1e-13, gmres_restart=20,
        )
    return u


def _sweep(rhs_flat, u0, t_final, exact, dt_list):
    out: dict[str, dict] = {}
    for method in METHOD_DISPLAY:
        errs = []
        for dt in dt_list:
            u_final = _integrate(rhs_flat, u0, t_final, dt, method)
            errs.append(float(jnp.max(jnp.abs(u_final - exact))))
        log_dt = np.log10(dt_list)
        log_err = np.log10(np.maximum(errs, 1e-16))
        slope, _ = np.polyfit(log_dt, log_err, 1)
        out[method] = {"dt": dt_list, "errs": errs, "slope": float(slope)}
    return out


def main() -> Path:
    dt_list = [0.2, 0.1, 0.05, 0.025]

    # Linear decay
    lam = -2.0
    u0 = jnp.array([1.0])
    exact_linear = jnp.array([math.exp(lam * 1.0)])
    linear_results = _sweep(
        lambda u: lam * u, u0, 1.0, exact_linear, dt_list,
    )

    # Nonlinear du/dt = -u^2, u(0) = 1, exact 1/(1+t)
    exact_nonlinear = jnp.array([1.0 / 2.0])
    nonlinear_results = _sweep(
        lambda u: -u * u, u0, 1.0, exact_nonlinear, dt_list,
    )

    print("Linear ODE   du/dt = -2u:")
    for m, res in linear_results.items():
        print(f"  {m:7s} slope = {res['slope']:.3f}  (theory {METHOD_DISPLAY[m][2]})")
    print("Nonlinear ODE   du/dt = -u²:")
    for m, res in nonlinear_results.items():
        print(f"  {m:7s} slope = {res['slope']:.3f}  (theory {METHOD_DISPLAY[m][2]})")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, results, title in [
        (axes[0], linear_results,    "Linear ODE  du/dt = -2u"),
        (axes[1], nonlinear_results, "Nonlinear ODE  du/dt = -u²"),
    ]:
        for method, (name, color, order) in METHOD_DISPLAY.items():
            res = results[method]
            ax.loglog(res["dt"], res["errs"], "o-", color=color,
                      markersize=7, lw=1.4,
                      label=f"{name} (order {order})  slope {res['slope']:.2f}")
            # Theoretical-slope reference through the first data point.
            x0, y0 = res["dt"][0], res["errs"][0]
            x_ref = np.array([res["dt"][0], res["dt"][-1]])
            ax.loglog(x_ref, y0 * (x_ref / x0) ** order, ":",
                      color=color, lw=1.0, alpha=0.55)
        ax.set_xlabel("time step dt")
        ax.set_ylabel("error at t = 1")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "DIRK / SDIRK convergence orders (dotted lines = theoretical slope)",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = out_dir / "dirk_convergence_ode.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png_path}")

    json_path = out_dir / "dirk_convergence_ode.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "linear": {m: {"dt": res["dt"], "errs": res["errs"],
                                "slope": res["slope"]}
                            for m, res in linear_results.items()},
                "nonlinear": {m: {"dt": res["dt"], "errs": res["errs"],
                                   "slope": res["slope"]}
                               for m, res in nonlinear_results.items()},
                "theoretical_orders": {m: METHOD_DISPLAY[m][2] for m in METHOD_DISPLAY},
            },
            fh, indent=2,
        )
    print(f"wrote {json_path}")
    return png_path


if __name__ == "__main__":
    main()
