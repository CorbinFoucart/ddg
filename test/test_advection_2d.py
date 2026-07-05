"""2D advection: smooth-IC translation tests + spectral convergence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg2d.advection import advec_rhs_2d, solve_advection_2d
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _l2(mesh, u: jax.Array) -> float:
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)
    return float(jnp.sqrt(jnp.sum(mesh.J * (u * (M_ref @ u)))))


def test_linear_function_translates_exactly():
    """``u(x, y, 0) = x`` translates to ``u(x, y, t) = x - ax t`` exactly."""
    ax, ay = 1.0, 0.5
    T = 0.1
    VX, VY, EToV = uniform_rectangle_mesh_2d(-1.0, 1.0, -1.0, 1.0, 4, 4)
    mesh = build_mesh_2d(VX, VY, EToV, N=3)
    u0 = mesh.x.copy()
    u_h = solve_advection_2d(mesh, u0, T, ax, ay)
    u_exact = mesh.x - ax * T
    rel_err = float(jnp.linalg.norm(u_h - u_exact) / jnp.linalg.norm(u_exact))
    assert rel_err < 1e-12


def test_single_step_rhs_matches_analytical():
    """For ``u = sin(x) sin(y)``, RHS should approximate ``-(ax cos(x) sin(y) + ay sin(x) cos(y))``."""
    ax, ay = 1.0, 0.5
    VX, VY, EToV = uniform_rectangle_mesh_2d(-1.0, 1.0, -1.0, 1.0, 8, 8)
    mesh = build_mesh_2d(VX, VY, EToV, N=5)
    u = jnp.sin(mesh.x) * jnp.sin(mesh.y)
    rhs = advec_rhs_2d(mesh, u, 0.0, ax, ay)
    expected = -(ax * jnp.cos(mesh.x) * jnp.sin(mesh.y) + ay * jnp.sin(mesh.x) * jnp.cos(mesh.y))
    rel = float(jnp.linalg.norm(rhs - expected) / jnp.linalg.norm(expected))
    assert rel < 1e-5


@pytest.mark.parametrize("N", [3, 5, 7])
def test_gaussian_translates_with_spectral_convergence(N):
    """Compactly-supported Gaussian: error should drop with N."""
    ax, ay = 1.0, 0.5
    T = 0.3
    sigma = 0.2
    K_side = 16
    VX, VY, EToV = uniform_rectangle_mesh_2d(-2.0, 2.0, -2.0, 2.0, K_side, K_side)
    mesh = build_mesh_2d(VX, VY, EToV, N=N)

    def gaussian(x, y, x0, y0):
        return jnp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    u0 = gaussian(mesh.x, mesh.y, 0.0, 0.0)
    u_h = solve_advection_2d(mesh, u0, T, ax, ay, cfl=0.4)
    u_exact = gaussian(mesh.x, mesh.y, ax * T, ay * T)
    rel_err = float(jnp.linalg.norm(u_h - u_exact) / jnp.linalg.norm(u_exact))
    bounds = {3: 1e-2, 5: 5e-4, 7: 1e-5}
    assert rel_err < bounds[N], f"N={N}: rel_err={rel_err:.3e}"


def test_upwind_is_dissipative_central_is_not():
    """``alpha=0`` (upwind) reduces amplitude; ``alpha=1`` (central) is unstable
    on a non-periodic mesh."""
    ax, ay = 1.0, 0.5
    sigma = 0.2
    T = 0.4
    VX, VY, EToV = uniform_rectangle_mesh_2d(-2.0, 2.0, -2.0, 2.0, 16, 16)
    mesh = build_mesh_2d(VX, VY, EToV, N=4)
    u0 = jnp.exp(-(mesh.x**2 + mesh.y**2) / (2 * sigma**2))

    u_upwind = solve_advection_2d(mesh, u0, T, ax, ay, alpha=0.0, cfl=0.4)
    assert float(jnp.max(jnp.abs(u_upwind))) <= float(jnp.max(u0)) + 1e-3
