"""1D Poisson (IPDG): operator structure + manufactured-solution convergence."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg1d import build_mesh_1d, uniform_mesh_1d
from ddg.dg1d.poisson import poisson_ip_apply, poisson_ip_matrix, solve_poisson


def _periodic_l2_error(mesh, u_h, u_exact) -> float:
    """L2 norm of (u_h - u_exact) using local mass matrices."""
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)
    diff = u_h - u_exact
    return float(jnp.sqrt(jnp.sum(mesh.J * (diff * (M_ref @ diff)))))


@pytest.mark.parametrize("K,N", [(8, 3), (10, 4), (16, 5)])
def test_recovers_manufactured_solution(K, N):
    """``-u'' = sin(x)`` on [0, 2*pi] with u=0 BCs has u_exact = sin(x)."""
    VX, EToV = uniform_mesh_1d(0.0, 2.0 * np.pi, K)
    mesh = build_mesh_1d(VX, EToV, N=N)
    source = jnp.sin(mesh.x)
    u_h = solve_poisson(mesh, source)
    u_exact = jnp.sin(mesh.x)
    rel_err = _periodic_l2_error(mesh, u_h, u_exact) / _periodic_l2_error(mesh, u_exact, jnp.zeros_like(u_exact))
    assert rel_err < 1e-3


def test_stiffness_matrix_is_symmetric_and_positive_definite():
    VX, EToV = uniform_mesh_1d(0.0, 1.0, 6)
    mesh = build_mesh_1d(VX, EToV, N=4)
    A = poisson_ip_matrix(mesh)
    # Symmetric
    assert float(jnp.linalg.norm(A - A.T) / jnp.linalg.norm(A)) < 1e-12
    # Positive definite (all eigenvalues > 0)
    eigs = jnp.linalg.eigvalsh((A + A.T) / 2)
    assert float(jnp.min(eigs)) > 0.0


def test_h_convergence_at_fixed_N():
    """Halve h: error should drop by 2^(N+1) (algebraic h-convergence)."""
    N = 3
    Ks = [4, 8, 16, 32]
    errs = []
    for K in Ks:
        VX, EToV = uniform_mesh_1d(0.0, 2.0 * np.pi, K)
        mesh = build_mesh_1d(VX, EToV, N=N)
        u_h = solve_poisson(mesh, jnp.sin(mesh.x))
        errs.append(_periodic_l2_error(mesh, u_h, jnp.sin(mesh.x)))
    # consecutive convergence rates
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    # IPDG yields rate N+1 = 4; with small constants and rounding we allow ~3.7
    assert all(r > 3.5 for r in rates), f"errs={errs}  rates={rates}"


def test_p_convergence_at_fixed_K():
    """Hold K fixed, increase N: error must fall by >= 5x for each +1 in N."""
    K = 4
    Ns = [2, 3, 4, 5, 6]
    errs = []
    for N in Ns:
        VX, EToV = uniform_mesh_1d(0.0, 2.0 * np.pi, K)
        mesh = build_mesh_1d(VX, EToV, N=N)
        u_h = solve_poisson(mesh, jnp.sin(mesh.x))
        errs.append(_periodic_l2_error(mesh, u_h, jnp.sin(mesh.x)))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(r > 5 for r in ratios), f"errs={errs}  ratios={ratios}"


def test_apply_matches_matvec_against_dense_matrix():
    """The matrix-free apply and the assembled matrix must agree."""
    VX, EToV = uniform_mesh_1d(-1.0, 1.0, 5)
    mesh = build_mesh_1d(VX, EToV, N=3)
    A = poisson_ip_matrix(mesh)

    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    via_apply = poisson_ip_apply(mesh, u).T.reshape(-1)
    via_matvec = A @ u.T.reshape(-1)
    np.testing.assert_allclose(np.asarray(via_apply), np.asarray(via_matvec), atol=1e-12)
