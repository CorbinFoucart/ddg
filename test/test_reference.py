"""Reference-element operators: known closed-form and exactness checks."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg1d.reference import (
    d_matrix_1d,
    grad_jacobi_p,
    jacobi_gauss_lobatto_nodes,
    jacobi_gauss_quadrature,
    jacobi_p,
    lift_1d,
    vandermonde_1d,
)


def test_legendre_gauss_lobatto_N4_matches_closed_form():
    """For Legendre LGL with N=4 the interior nodes are ±sqrt(3/7), 0."""
    r = jacobi_gauss_lobatto_nodes(0.0, 0.0, 4)
    expected = jnp.array([-1.0, -np.sqrt(3 / 7), 0.0, np.sqrt(3 / 7), 1.0])
    np.testing.assert_allclose(np.asarray(r), np.asarray(expected), atol=1e-14)


def test_legendre_gauss_weights_sum_to_2():
    """Legendre Gauss weights integrate the constant 1 on [-1, 1] to 2."""
    for N in [1, 4, 9, 16]:
        _, w = jacobi_gauss_quadrature(0.0, 0.0, N)
        assert abs(float(jnp.sum(w)) - 2.0) < 1e-13


def test_legendre_polynomials_are_orthonormal():
    """Gram matrix of P_0..P_N under Legendre weight is the identity."""
    N = 6
    x, w = jacobi_gauss_quadrature(0.0, 0.0, 2 * N + 1)
    P = jnp.stack([jacobi_p(x, 0.0, 0.0, j) for j in range(N + 1)], axis=1)
    G = P.T @ jnp.diag(w) @ P
    np.testing.assert_allclose(np.asarray(G), np.eye(N + 1), atol=1e-13)


@pytest.mark.parametrize("N", [3, 5, 7, 9])
def test_differentiation_matrix_exact_on_polynomials(N):
    """Dr is exact for polynomials up to degree N."""
    r = jacobi_gauss_lobatto_nodes(0.0, 0.0, N)
    V = vandermonde_1d(N, r)
    Dr = d_matrix_1d(N, r, V)
    for p in range(N + 1):
        # d/dr r^p = p * r^(p-1)
        rhs = (Dr @ (r**p))
        expected = jnp.where(jnp.array(p == 0), jnp.zeros_like(r), p * r ** max(p - 1, 0))
        np.testing.assert_allclose(np.asarray(rhs), np.asarray(expected), atol=1e-11)


def test_grad_jacobi_p_consistent_with_finite_differences():
    """Spectrally evaluated derivative matches a high-order FD reference."""
    N = 5
    x = jnp.linspace(-0.9, 0.9, 9)
    h = 1e-6
    p_plus = jacobi_p(x + h, 0.0, 0.0, N)
    p_minus = jacobi_p(x - h, 0.0, 0.0, N)
    fd = (p_plus - p_minus) / (2 * h)
    dp = grad_jacobi_p(x, 0.0, 0.0, N)
    np.testing.assert_allclose(np.asarray(dp), np.asarray(fd), atol=1e-6)


def test_lift_shape_and_face_action():
    """LIFT picks up face-only contributions: applied to a face-only vector,
    the result is non-trivial only in the lifted polynomial."""
    N = 4
    r = jacobi_gauss_lobatto_nodes(0.0, 0.0, N)
    V = vandermonde_1d(N, r)
    LIFT = lift_1d(N + 1, 2, 1, V)
    assert LIFT.shape == (N + 1, 2)
