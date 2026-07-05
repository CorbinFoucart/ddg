"""DIRK / SDIRK time-integrator convergence orders on scalar ODEs.

The 2D Navier-Stokes problem couples time-discretisation error with the
spatial DG floor, which makes it hard to observe the time integrator's
theoretical order directly — once the time error drops below the
spatial floor, the convergence curve flattens. To verify the
integrators in isolation we run them on a couple of scalar ODEs where
the analytical solution is exact and the only source of error is time
discretisation.

Two test problems:

  1) Linear decay   : du/dt = λu,       u(0) = 1  →  u(t) = exp(λt)
  2) Nonlinear      : du/dt = -u^2,     u(0) = 1  →  u(t) = 1 / (1 + t)

The nonlinear problem actually exercises the Newton iteration inside
each implicit stage — for a linear ODE, Newton converges in a single
step regardless of method, so we'd be testing only the time
discretisation. The ``-u^2`` ODE makes the per-stage equation
genuinely nonlinear so the Newton-Krylov machinery has to work too.

We fit the slope of ``log(error) vs log(dt)`` and assert it lies
within ±0.15 of the theoretical order for each method.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg2d.navier_stokes import DIRK_TABLEAUX, _dirk_step


NEWTON_KWARGS = dict(
    newton_tol=1e-12,
    newton_max_iter=20,
    gmres_tol=1e-13,
    gmres_restart=20,
)


def _integrate(rhs_flat, u0, t_final, dt, method):
    """Integrate from 0 to t_final with the given DIRK method."""
    n_steps = int(round(t_final / dt))
    dt_actual = t_final / n_steps
    u = u0
    for _ in range(n_steps):
        u, _ = _dirk_step(
            rhs_flat, u, dt_actual, DIRK_TABLEAUX[method], **NEWTON_KWARGS,
        )
    return u


def _measure_slope(method, rhs_flat, u0, t_final, exact, dt_list):
    """Return the log-log slope of error-vs-dt for `method`."""
    errs = []
    for dt in dt_list:
        u_final = _integrate(rhs_flat, u0, t_final, dt, method)
        err = float(jnp.max(jnp.abs(u_final - exact)))
        errs.append(err)
    log_dt = np.log10(dt_list)
    log_err = np.log10(np.maximum(errs, 1e-16))
    slope, _ = np.polyfit(log_dt, log_err, 1)
    return float(slope), errs


# ---------------------------------------------------------------------------
# Test 1: linear decay du/dt = λu
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,order", [
    ("be",     1),
    ("cn",     2),
    ("sdirk2", 2),
    ("sdirk3", 3),
])
def test_linear_ode_convergence_order(method, order):
    """``du/dt = -2 u``, ``u(t) = exp(-2t)`` — verify each method's slope."""
    lam = -2.0
    u0 = jnp.array([1.0])
    t_final = 1.0
    exact = jnp.array([math.exp(lam * t_final)])

    def rhs_flat(u):
        return lam * u

    dt_list = [0.2, 0.1, 0.05, 0.025]
    slope, errs = _measure_slope(method, rhs_flat, u0, t_final, exact, dt_list)
    assert abs(slope - order) < 0.15, (
        f"{method}: measured slope {slope:.3f}, expected {order} (±0.15). "
        f"errors at dt={dt_list}: {errs}"
    )


# ---------------------------------------------------------------------------
# Test 2: nonlinear du/dt = -u^2 (Newton actually has to work)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,order", [
    ("be",     1),
    ("cn",     2),
    ("sdirk2", 2),
    ("sdirk3", 3),
])
def test_nonlinear_ode_convergence_order(method, order):
    """``du/dt = -u²``, ``u(t) = 1/(1 + t)`` — orders should match theory.

    The per-stage residual ``Y - Y_const + α Y² = 0`` is genuinely
    nonlinear, so this exercises both the integrator coefficients and
    the inner Newton-Krylov solve.
    """
    u0 = jnp.array([1.0])
    t_final = 1.0
    exact = jnp.array([1.0 / (1.0 + t_final)])

    def rhs_flat(u):
        return -u * u

    dt_list = [0.2, 0.1, 0.05, 0.025]
    slope, errs = _measure_slope(method, rhs_flat, u0, t_final, exact, dt_list)
    assert abs(slope - order) < 0.15, (
        f"{method}: measured slope {slope:.3f}, expected {order} (±0.15). "
        f"errors at dt={dt_list}: {errs}"
    )


# ---------------------------------------------------------------------------
# Sanity: explicit slopes are an order higher than predecessors
# ---------------------------------------------------------------------------

def test_higher_order_methods_beat_lower_order_at_same_dt():
    """On the nonlinear problem at a fixed moderate dt, error should
    decrease as we move BE -> CN -> SDIRK-2 -> SDIRK-3."""
    u0 = jnp.array([1.0])
    t_final = 1.0
    exact = jnp.array([1.0 / (1.0 + t_final)])
    dt = 0.05

    def rhs_flat(u):
        return -u * u

    errs = {}
    for method in ("be", "cn", "sdirk2", "sdirk3"):
        u_final = _integrate(rhs_flat, u0, t_final, dt, method)
        errs[method] = float(jnp.max(jnp.abs(u_final - exact)))

    assert errs["cn"]     < errs["be"],     f"CN err {errs['cn']:.2e} >= BE err {errs['be']:.2e}"
    assert errs["sdirk3"] < errs["sdirk2"], f"SDIRK3 err {errs['sdirk3']:.2e} >= SDIRK2 err {errs['sdirk2']:.2e}"
