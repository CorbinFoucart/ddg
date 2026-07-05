"""Reference-element operators for 1D nodal DG.

Ports the dimension-agnostic routines from Hesthaven & Warburton's
nodal-dg ``Codes1D/`` directory: Jacobi polynomial recurrences,
Gauss / Gauss-Lobatto quadrature, Vandermonde matrices, the
differentiation matrix ``Dr``, and the surface lifting operator.

Everything is written with ``jax.numpy`` so the operators are
ready to drop into autodiff. Setup-time computations don't need
to be jitted, but staying in jnp keeps types consistent.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


def jacobi_p(x: jnp.ndarray, alpha: float, beta: float, N: int) -> jnp.ndarray:
    """Evaluate the orthonormal Jacobi polynomial P_N^{(alpha,beta)} at x.

    Mirrors ``JacobiP.m``: orthonormal on (-1, 1) w.r.t. the weight
    (1-x)^alpha (1+x)^beta. Uses the three-term recurrence in normalised
    form so that ``V = vandermonde_1d(N, r)`` is well-conditioned.
    """
    x = jnp.asarray(x).reshape(-1)

    gamma0 = (
        2.0 ** (alpha + beta + 1)
        / (alpha + beta + 1)
        * math.gamma(alpha + 1)
        * math.gamma(beta + 1)
        / math.gamma(alpha + beta + 1)
    )
    p_prev = jnp.full_like(x, 1.0 / math.sqrt(gamma0))
    if N == 0:
        return p_prev

    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    p_curr = ((alpha + beta + 2) * x / 2 + (alpha - beta) / 2) / math.sqrt(gamma1)
    if N == 1:
        return p_curr

    a_old = (
        2.0
        / (2 + alpha + beta)
        * math.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    )
    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        a_new = (
            2.0
            / (h1 + 2)
            * math.sqrt(
                (i + 1)
                * (i + 1 + alpha + beta)
                * (i + 1 + alpha)
                * (i + 1 + beta)
                / (h1 + 1)
                / (h1 + 3)
            )
        )
        b_new = -(alpha**2 - beta**2) / h1 / (h1 + 2)
        p_next = (1.0 / a_new) * (-a_old * p_prev + (x - b_new) * p_curr)
        p_prev, p_curr = p_curr, p_next
        a_old = a_new
    return p_curr


def grad_jacobi_p(r: jnp.ndarray, alpha: float, beta: float, N: int) -> jnp.ndarray:
    """Derivative of the orthonormal Jacobi polynomial of order N at r."""
    r = jnp.asarray(r).reshape(-1)
    if N == 0:
        return jnp.zeros_like(r)
    return math.sqrt(N * (N + alpha + beta + 1)) * jacobi_p(r, alpha + 1, beta + 1, N - 1)


def jacobi_gauss_quadrature(
    alpha: float, beta: float, N: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """N+1-point Gauss quadrature for the Jacobi weight (alpha, beta).

    Returns ``(x, w)``. Implemented as the eigenvalue solve of the
    symmetric Jacobi recurrence matrix (Golub-Welsch).
    """
    if N == 0:
        x = jnp.array([-(alpha - beta) / (alpha + beta + 2.0)])
        w = jnp.array([2.0])
        return x, w

    n = jnp.arange(N + 1, dtype=jnp.float64)
    h1 = 2.0 * n + alpha + beta

    diag = -0.5 * (alpha**2 - beta**2) / (h1 + 2.0) / h1
    # Patch the indeterminate 0/0 at n=0 for the Legendre case (alpha+beta=0).
    if abs(alpha + beta) < 10 * jnp.finfo(diag.dtype).eps:
        diag = diag.at[0].set(0.0)

    i = jnp.arange(1, N + 1, dtype=jnp.float64)
    h1_lo = h1[:N]
    offdiag = (
        2.0
        / (h1_lo + 2.0)
        * jnp.sqrt(
            i * (i + alpha + beta) * (i + alpha) * (i + beta)
            / (h1_lo + 1.0)
            / (h1_lo + 3.0)
        )
    )

    J = jnp.diag(diag) + jnp.diag(offdiag, 1)
    J = J + J.T

    eigvals, eigvecs = jnp.linalg.eigh(J)
    x = eigvals
    w = (
        eigvecs[0, :] ** 2
        * (2.0 ** (alpha + beta + 1) / (alpha + beta + 1))
        * math.gamma(alpha + 1)
        * math.gamma(beta + 1)
        / math.gamma(alpha + beta + 1)
    )
    return x, w


def jacobi_gauss_lobatto_nodes(alpha: float, beta: float, N: int) -> jnp.ndarray:
    """N+1 Gauss-Lobatto nodes for the Jacobi weight (alpha, beta).

    Endpoints -1 and 1 are exact; interior points come from the
    Jacobi-Gauss rule for (alpha+1, beta+1) of degree N-2.
    """
    if N == 1:
        return jnp.array([-1.0, 1.0])
    x_int, _ = jacobi_gauss_quadrature(alpha + 1, beta + 1, N - 2)
    return jnp.concatenate([jnp.array([-1.0]), x_int, jnp.array([1.0])])


def vandermonde_1d(N: int, r: jnp.ndarray) -> jnp.ndarray:
    """1D Legendre Vandermonde: V[i, j] = P_j(r_i) for j = 0..N."""
    r = jnp.asarray(r).reshape(-1)
    cols = [jacobi_p(r, 0.0, 0.0, j) for j in range(N + 1)]
    return jnp.stack(cols, axis=1)


def grad_vandermonde_1d(N: int, r: jnp.ndarray) -> jnp.ndarray:
    """Gradient of the 1D Legendre Vandermonde: Vr[i, j] = P'_j(r_i)."""
    r = jnp.asarray(r).reshape(-1)
    cols = [grad_jacobi_p(r, 0.0, 0.0, j) for j in range(N + 1)]
    return jnp.stack(cols, axis=1)


def d_matrix_1d(N: int, r: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Differentiation matrix Dr on reference nodes r: ``Dr = Vr @ inv(V)``."""
    Vr = grad_vandermonde_1d(N, r)
    return jnp.linalg.solve(V.T, Vr.T).T


def lift_1d(Np: int, Nfaces: int, Nfp: int, V: jnp.ndarray) -> jnp.ndarray:
    """Surface lifting operator ``LIFT = inv(M) @ E`` written via V @ V.T @ E.

    In 1D, ``E`` is a (Np, Nfaces*Nfp) = (Np, 2) matrix that picks the
    boundary nodes of the reference element ([-1, 1]).
    """
    Emat = jnp.zeros((Np, Nfaces * Nfp))
    Emat = Emat.at[0, 0].set(1.0)
    Emat = Emat.at[Np - 1, 1].set(1.0)
    return V @ (V.T @ Emat)
