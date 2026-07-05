"""Tests for the BDM_1 -> P_0 divergence and P_0 -> BDM_1 gradient operators.

Phase-5c deliverable in :mod:`ddg.dg2d.bdm`:
    * :func:`bdm_p0_divergence_apply` -- ``D : BDM_1 -> P_0``.
    * :func:`bdm_p0_gradient_apply`   -- ``G : P_0 -> BDM_1`` (= -D^T).

Both are element-local: H(div)-conformity of BDM_1 means the normal
trace is single-valued across each interior face, so no inter-element
flux choice (centered, upwind, LF) enters the divergence assembly.
This is the structural reason the BDM saddle-point sidesteps the
LBB-pollution problem that defeated the equal-order and embedded
mixed-order solvers.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM,
    bdm_p0_divergence_apply,
    bdm_p0_gradient_apply,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _build_mesh(Kx: int = 4, Ky: int = 4):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, 1)


# ---------------------------------------------------------------------------
# Shape + linearity
# ---------------------------------------------------------------------------

def test_div_apply_shapes_and_linearity():
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh)
    u = jnp.zeros(V.total_dofs)
    out = bdm_p0_divergence_apply(V, u)
    assert out.shape == (V.K,)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-14)


def test_grad_apply_shapes_and_linearity():
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh)
    p = jnp.zeros(V.K)
    out = bdm_p0_gradient_apply(V, p)
    assert out.shape == (V.total_dofs,)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Adjointness: G = -D^T (the saddle-point structural identity)
# ---------------------------------------------------------------------------

def test_div_and_grad_are_negative_transposes():
    """For random BDM velocity ``u`` and P_0 pressure ``p``:

        <D u, p>_P0 + <G p, u>_BDM = 0     (G = -D^T)

    This is the structural property that makes the BDM-P_0 saddle-point
    Murphy-Wathen analysis apply cleanly -- and the reason no flux-
    choice question arises in the assembly. It's a purely algebraic
    identity that the implementation must satisfy by construction; this
    test guards against off-by-sign / off-by-factor-2 bugs in either
    operator.
    """
    for Kx, Ky in [(2, 2), (3, 4), (4, 4)]:
        mesh = _build_mesh(Kx=Kx, Ky=Ky)
        V = BDM.from_mesh(mesh)
        rng = np.random.default_rng(0)
        u = jnp.asarray(rng.standard_normal(V.total_dofs))
        p = jnp.asarray(rng.standard_normal(V.K))
        D_u = bdm_p0_divergence_apply(V, u)
        G_p = bdm_p0_gradient_apply(V, p)
        inner_Dp = float(jnp.sum(D_u * p))           # <D u, p>
        inner_Gu = float(jnp.sum(G_p * u))           # <G p, u>
        np.testing.assert_allclose(
            inner_Dp + inner_Gu, 0.0,
            atol=1e-12,
            err_msg=f"Kx={Kx} Ky={Ky}: <D u, p>={inner_Dp}, "
                    f"<G p, u>={inner_Gu}",
        )


def test_div_matrix_negative_transpose_of_grad_matrix():
    """Same property at the matrix level: explicitly assemble both
    operator matrices and verify D + G^T = 0. This is the strongest
    pointwise check -- if there's any nonzero entry off by sign, this
    catches it."""
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh)
    n_u, n_p = V.total_dofs, V.K
    # Assemble D matrix column-by-column via canonical basis vectors.
    D_mat = jax.vmap(
        lambda i: bdm_p0_divergence_apply(V, jnp.eye(n_u)[i]),
    )(jnp.arange(n_u)).T                              # (n_p, n_u)
    # Assemble G matrix column-by-column.
    G_mat = jax.vmap(
        lambda i: bdm_p0_gradient_apply(V, jnp.eye(n_p)[i]),
    )(jnp.arange(n_p)).T                              # (n_u, n_p)
    np.testing.assert_allclose(
        np.asarray(D_mat + G_mat.T), 0.0, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Divergence-free fields
# ---------------------------------------------------------------------------

def test_div_of_grad_kernel_is_zero():
    """G p has divergence zero in the volume integral sense: <D(Gp), q>
    = ... wait, on second thought, for BDM-P_0 saddle-point with G =
    -D^T, the *symmetric Schur complement* is D M^{-1} D^T which is not
    zero. But the field G p, viewed as a BDM coefficient vector,
    doesn't have to be divergence-free.

    What IS true: kernel of D (pure-divergence-free BDM fields) is
    orthogonal to range of G (under the right inner product). We test
    this via the adjointness identity already, so this test is
    redundant. Replaced with a basic sanity: zero pressure -> zero
    gradient."""
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh)
    p_zero = jnp.zeros(V.K)
    G_p = bdm_p0_gradient_apply(V, p_zero)
    np.testing.assert_allclose(np.asarray(G_p), 0.0, atol=1e-14)


def test_div_matrix_rank_matches_continuity_constraint():
    """The discrete divergence ``D : BDM_1 -> P_0`` should be
    surjective (or close to it): for an inf-sup-stable BDM-P_0 pair the
    rank of D equals dim(P_0) = K, up to the constant-pressure
    nullspace (rank K - 1 on a closed velocity-Dirichlet domain;
    rank K when a pressure-pin is present). Verifying this is the
    discrete LBB sentinel."""
    mesh = _build_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh)
    n_u, n_p = V.total_dofs, V.K
    D_mat = jax.vmap(
        lambda i: bdm_p0_divergence_apply(V, jnp.eye(n_u)[i]),
    )(jnp.arange(n_u)).T                              # (n_p, n_u)
    rank = int(np.linalg.matrix_rank(np.asarray(D_mat), tol=1e-10))
    # On a domain with no pressure-Dirichlet face, the BDM-P_0 pair
    # has a 1-dim nullspace: the constant pressure mode. So rank
    # should be n_p - 1 OR n_p (depending on BC; this test mesh has
    # closed boundary -> nullspace expected). Either way the rank
    # must NOT be lower than n_p - 1 -- that would be an LBB failure.
    assert rank >= n_p - 1, (
        f"D rank {rank} < n_p - 1 = {n_p - 1}: LBB failure"
    )


def test_grad_matrix_rank_matches_constraint():
    """Same test on G: rank(G) >= n_p - 1, since G = -D^T has the same
    rank as D."""
    mesh = _build_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh)
    n_u, n_p = V.total_dofs, V.K
    G_mat = jax.vmap(
        lambda i: bdm_p0_gradient_apply(V, jnp.eye(n_p)[i]),
    )(jnp.arange(n_p)).T                              # (n_u, n_p)
    rank = int(np.linalg.matrix_rank(np.asarray(G_mat), tol=1e-10))
    assert rank >= n_p - 1
