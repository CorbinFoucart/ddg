"""End-to-end 1D advection: correctness + spectral convergence + autodiff."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg1d import build_mesh_1d, solve_advection, uniform_mesh_1d


def _periodic_mesh(N: int, K: int, xmin: float = 0.0, xmax: float = 2.0 * np.pi):
    VX, EToV = uniform_mesh_1d(xmin, xmax, K)
    return build_mesh_1d(VX, EToV, N=N, periodic=True)


def test_periodic_advection_returns_to_initial_state_after_one_period():
    """``u_t + 2*pi u_x = 0`` on [0, 2*pi] with u_0 = sin(x), T = 1 returns to u_0."""
    mesh = _periodic_mesh(N=8, K=8)
    a = 2.0 * np.pi
    u0 = jnp.sin(mesh.x)
    u_final = solve_advection(mesh, u0, t_final=1.0, a=a)
    rel_err = float(jnp.linalg.norm(u_final - u0) / jnp.linalg.norm(u0))
    assert rel_err < 1e-7


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_periodic_advection_error_bounds_per_order(N):
    """Fix K=4, sweep N: error must fall below an order-specific bound."""
    mesh = _periodic_mesh(N=N, K=4)
    a = 2.0 * np.pi
    u0 = jnp.sin(mesh.x)
    u_final = solve_advection(mesh, u0, t_final=1.0, a=a)
    rel_err = float(jnp.linalg.norm(u_final - u0) / jnp.linalg.norm(u0))
    bounds = {2: 1e-1, 4: 1e-3, 6: 1e-5, 8: 1e-7}
    assert rel_err < bounds[N], f"N={N}: rel_err={rel_err:.3e}"


def test_spectral_convergence_in_polynomial_order():
    """Error should fall by at least 10x for every step of +2 in N (exponential)."""
    a = 2.0 * np.pi
    errs = []
    for N in [2, 4, 6, 8]:
        mesh = _periodic_mesh(N=N, K=4)
        u0 = jnp.sin(mesh.x)
        u_final = solve_advection(mesh, u0, t_final=1.0, a=a)
        errs.append(float(jnp.linalg.norm(u_final - u0) / jnp.linalg.norm(u0)))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(r > 10 for r in ratios), f"errs={errs}  ratios={ratios}"


def test_advection_is_autodiffable():
    """`jax.value_and_grad` produces finite gradients through the trajectory."""
    mesh = _periodic_mesh(N=4, K=4)
    a = 2.0 * np.pi

    def loss(u0):
        u_final = solve_advection(mesh, u0, t_final=0.1, a=a)
        return jnp.sum(u_final**2)

    u0 = jnp.sin(mesh.x)
    val, grad = jax.value_and_grad(loss)(u0)
    assert jnp.isfinite(val)
    assert grad.shape == u0.shape
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.linalg.norm(grad)) > 0.0
