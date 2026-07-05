"""2D Navier-Stokes (vorticity-streamfunction): explicit and implicit
solvers agree with the Taylor-Green analytical decay."""

import math

import jax.numpy as jnp
import pytest

from ddg.dg2d import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.navier_stokes import (
    solve_vorticity_streamfunction,
    solve_vorticity_streamfunction_implicit,
)
from ddg.dg2d.poisson import poisson_ip_matrix_2d


def _taylor_green_setup():
    L = math.pi
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, L, 0.0, L, 8, 8)
    mesh = build_mesh_2d(VX, VY, EToV, N=4)
    omega0 = 2.0 * jnp.sin(mesh.x) * jnp.sin(mesh.y)
    A = poisson_ip_matrix_2d(mesh)
    return mesh, omega0, A


def test_explicit_rk4_matches_taylor_green_decay():
    """Peak amplitude decays as ``2 exp(-2 ν t)`` (analytical)."""
    mesh, omega0, A = _taylor_green_setup()
    nu, T = 0.05, 0.5
    omega = solve_vorticity_streamfunction(
        mesh, omega0, T, nu=nu, n_time_steps=5000, A_poisson=A,
    )
    expected = 2.0 * math.exp(-2.0 * nu * T)
    peak = float(jnp.max(jnp.abs(omega)))
    rel_err = abs(peak - expected) / expected
    assert rel_err < 1e-5, f"explicit rel_err = {rel_err:.3e}"


def test_implicit_newton_krylov_matches_taylor_green_decay():
    """Implicit BE matches the analytical decay too (lower-order in time
    but still well-converged for these dt values)."""
    mesh, omega0, A = _taylor_green_setup()
    nu, T = 0.05, 0.5
    omega, diags = solve_vorticity_streamfunction_implicit(
        mesh, omega0, T, nu=nu, n_time_steps=100, A_poisson=A,
    )
    expected = 2.0 * math.exp(-2.0 * nu * T)
    peak = float(jnp.max(jnp.abs(omega)))
    rel_err = abs(peak - expected) / expected
    assert rel_err < 5e-5, f"implicit rel_err = {rel_err:.3e}"
    # Newton should converge in just a few iterations from this initial guess.
    avg_iters = sum(d["newton_iters"] for d in diags) / len(diags)
    assert avg_iters < 4.0, f"Newton averaged {avg_iters:.1f} iters/step (expected < 4)"


@pytest.mark.parametrize(
    "method,expected_max_err",
    [("be", 5e-4), ("cn", 1e-5), ("sdirk2", 1e-5), ("sdirk3", 1e-5)],
)
def test_dirk_methods_match_taylor_green_decay(method, expected_max_err):
    """Every DIRK/SDIRK method should match the analytical decay within
    its order's expected accuracy at a fixed moderate dt."""
    mesh, omega0, A = _taylor_green_setup()
    nu, T = 0.05, 0.5
    omega, diags = solve_vorticity_streamfunction_implicit(
        mesh, omega0, T, nu=nu, n_time_steps=50, A_poisson=A, method=method,
    )
    expected = 2.0 * math.exp(-2.0 * nu * T)
    peak = float(jnp.max(jnp.abs(omega)))
    rel_err = abs(peak - expected) / expected
    assert rel_err < expected_max_err, (
        f"{method}: rel_err = {rel_err:.3e}, allowed {expected_max_err:.0e}"
    )
    # Higher-order methods need more Newton iters per stage (more nonlinear
    # constants in the per-stage residual).
    avg_iters = sum(d["newton_iters"] for d in diags) / len(diags)
    max_expected_iters = {"be": 4, "cn": 4, "sdirk2": 6, "sdirk3": 10}
    assert avg_iters <= max_expected_iters[method], (
        f"{method}: Newton avg {avg_iters:.1f}, allowed {max_expected_iters[method]}"
    )


def test_implicit_stable_at_dt_that_explodes_explicit():
    """At dt = 1e-2, explicit RK4 produces NaN; implicit remains stable."""
    mesh, omega0, A = _taylor_green_setup()
    nu, T = 0.05, 0.5

    # Explicit fails at this dt (50-step run is too coarse for the CFL).
    omega_explicit = solve_vorticity_streamfunction(
        mesh, omega0, T, nu=nu, n_time_steps=50, A_poisson=A,
    )
    assert not jnp.all(jnp.isfinite(omega_explicit)), (
        "Explicit should NOT be stable at dt = 1e-2 for these parameters"
    )

    # Implicit handles the same dt cleanly.
    omega_implicit, _ = solve_vorticity_streamfunction_implicit(
        mesh, omega0, T, nu=nu, n_time_steps=50, A_poisson=A,
    )
    assert jnp.all(jnp.isfinite(omega_implicit))
    expected = 2.0 * math.exp(-2.0 * nu * T)
    peak = float(jnp.max(jnp.abs(omega_implicit)))
    assert abs(peak - expected) / expected < 1e-3
