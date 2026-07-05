"""2D reference-element operators on the triangle."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg2d.reference import (
    d_matrices_2d,
    equilateral_nodes_2d,
    face_masks_2d,
    grad_simplex_2d_p,
    lift_2d,
    rs_to_ab,
    simplex_2d_p,
    vandermonde_2d,
    xy_to_rs,
)


def test_reference_triangle_area_via_mass_matrix():
    """1^T M 1 == 2 (area of the standard reference triangle)."""
    N = 5
    x_eq, y_eq = equilateral_nodes_2d(N)
    r, s = xy_to_rs(x_eq, y_eq)
    V = vandermonde_2d(N, r, s)
    M = jnp.linalg.inv(V @ V.T)
    Np = (N + 1) * (N + 2) // 2
    area = float(jnp.ones(Np) @ M @ jnp.ones(Np))
    assert abs(area - 2.0) < 1e-10


@pytest.mark.parametrize("N", [2, 3, 4, 6])
def test_d_matrices_exact_on_polynomials(N):
    """Dr, Ds are exact on all monomials of total degree <= N."""
    x_eq, y_eq = equilateral_nodes_2d(N)
    r, s = xy_to_rs(x_eq, y_eq)
    V = vandermonde_2d(N, r, s)
    Dr, Ds = d_matrices_2d(N, r, s, V)
    for p in range(N + 1):
        for q in range(N - p + 1):
            f = (r**p) * (s**q)
            df_dr = p * (r ** max(p - 1, 0)) * (s**q) if p > 0 else jnp.zeros_like(r)
            df_ds = q * (r**p) * (s ** max(q - 1, 0)) if q > 0 else jnp.zeros_like(r)
            assert float(jnp.max(jnp.abs(Dr @ f - df_dr))) < 1e-10
            assert float(jnp.max(jnp.abs(Ds @ f - df_ds))) < 1e-10


def test_face_masks_have_correct_count_and_position():
    """N+1 nodes per face, lying exactly on the three reference edges."""
    N = 5
    x_eq, y_eq = equilateral_nodes_2d(N)
    r, s = xy_to_rs(x_eq, y_eq)
    Fmask = face_masks_2d(N, r, s)
    assert Fmask.shape == (N + 1, 3)
    r_np = np.asarray(r)
    s_np = np.asarray(s)
    assert np.max(np.abs(s_np[Fmask[:, 0]] + 1.0)) < 1e-12  # s = -1
    assert np.max(np.abs(r_np[Fmask[:, 1]] + s_np[Fmask[:, 1]])) < 1e-12  # r + s = 0
    assert np.max(np.abs(r_np[Fmask[:, 2]] + 1.0)) < 1e-12  # r = -1


def test_lift_shape():
    """LIFT is (Np, 3*(N+1))."""
    N = 4
    x_eq, y_eq = equilateral_nodes_2d(N)
    r, s = xy_to_rs(x_eq, y_eq)
    V = vandermonde_2d(N, r, s)
    Fmask = face_masks_2d(N, r, s)
    LIFT = lift_2d(N, r, s, V, Fmask)
    Np = (N + 1) * (N + 2) // 2
    assert LIFT.shape == (Np, 3 * (N + 1))


def test_grad_simplex_consistent_with_central_differences():
    """Spectrally-evaluated derivative matches finite differences."""
    r = jnp.array([0.0, 0.1, -0.3, -0.2])
    s = jnp.array([0.0, -0.5, 0.0, 0.3])
    a, b = rs_to_ab(r, s)
    h = 1e-5
    for (i, j) in [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]:
        P_plus_r = simplex_2d_p(*rs_to_ab(r + h, s), i, j)
        P_minus_r = simplex_2d_p(*rs_to_ab(r - h, s), i, j)
        P_plus_s = simplex_2d_p(*rs_to_ab(r, s + h), i, j)
        P_minus_s = simplex_2d_p(*rs_to_ab(r, s - h), i, j)
        fd_dr = (P_plus_r - P_minus_r) / (2 * h)
        fd_ds = (P_plus_s - P_minus_s) / (2 * h)
        spec_dr, spec_ds = grad_simplex_2d_p(a, b, i, j)
        np.testing.assert_allclose(np.asarray(spec_dr), np.asarray(fd_dr), atol=1e-5)
        np.testing.assert_allclose(np.asarray(spec_ds), np.asarray(fd_ds), atol=1e-5)
