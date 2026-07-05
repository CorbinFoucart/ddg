"""2D Poisson IPDG: operator structure + manufactured-solution convergence."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg2d import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.poisson import (
    poisson_ip_apply_2d,
    poisson_ip_matrix_2d,
    solve_poisson_2d,
)


def _l2_error(mesh, u_h, u_exact):
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)
    diff = u_h - u_exact
    return float(jnp.sqrt(jnp.sum(mesh.J * (diff * (M_ref @ diff)))))


def test_2d_poisson_recovers_manufactured_solution():
    """``-Δu = 2π² sin(πx) sin(πy)`` on [0,1]² gives ``u = sin(πx) sin(πy)``."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, 6, 6)
    mesh = build_mesh_2d(VX, VY, EToV, N=4)
    source = 2 * jnp.pi**2 * jnp.sin(jnp.pi * mesh.x) * jnp.sin(jnp.pi * mesh.y)
    u_h = solve_poisson_2d(mesh, source)
    u_exact = jnp.sin(jnp.pi * mesh.x) * jnp.sin(jnp.pi * mesh.y)
    err = _l2_error(mesh, u_h, u_exact) / _l2_error(mesh, u_exact, jnp.zeros_like(u_exact))
    assert err < 1e-4, f"rel L2 err = {err:.3e}"


def test_2d_poisson_matrix_is_symmetric_and_pd():
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, 4, 4)
    mesh = build_mesh_2d(VX, VY, EToV, N=3)
    A = poisson_ip_matrix_2d(mesh)
    assert float(jnp.linalg.norm(A - A.T) / jnp.linalg.norm(A)) < 1e-12
    eigs = jnp.linalg.eigvalsh((A + A.T) / 2)
    assert float(jnp.min(eigs)) > 0.0


@pytest.mark.parametrize("N", [2, 3, 4])
def test_2d_poisson_p_convergence(N):
    """Increasing N drives the manufactured-solution error down."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, 4, 4)
    mesh = build_mesh_2d(VX, VY, EToV, N=N)
    source = 2 * jnp.pi**2 * jnp.sin(jnp.pi * mesh.x) * jnp.sin(jnp.pi * mesh.y)
    u_h = solve_poisson_2d(mesh, source)
    u_exact = jnp.sin(jnp.pi * mesh.x) * jnp.sin(jnp.pi * mesh.y)
    err = _l2_error(mesh, u_h, u_exact) / _l2_error(mesh, u_exact, jnp.zeros_like(u_exact))
    bounds = {2: 1e-1, 3: 1e-2, 4: 1e-3}
    assert err < bounds[N], f"N={N}: rel L2 err = {err:.3e}"


def test_apply_matches_dense_matvec():
    """Matrix-free apply and the assembled matrix must agree."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, 3, 3)
    mesh = build_mesh_2d(VX, VY, EToV, N=3)
    A = poisson_ip_matrix_2d(mesh)

    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    via_apply = poisson_ip_apply_2d(mesh, u).T.reshape(-1)
    via_matvec = A @ u.T.reshape(-1)
    np.testing.assert_allclose(np.asarray(via_apply), np.asarray(via_matvec), atol=1e-12)
