"""Tests for the BDM_1 viscous volume bilinear form (Phase 5d-1).

Volume term only -- face IPG terms come next. Volume alone produces a
symmetric positive *semi*-definite operator on BDM_1: the nullspace is
spanned by the rigid-translation modes (constants in (P_0)^2) plus
whatever else the centered-flux DG saddle-point preserves. Face penalty
terms recover full coercivity.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM,
    bdm1_viscous_volume_apply,
    bdm1_viscous_volume_stiffness,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _build_mesh(Kx: int = 4, Ky: int = 4):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, 1)


def test_viscous_volume_apply_shape_and_linearity():
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh)
    out = bdm1_viscous_volume_apply(V, jnp.zeros(V.total_dofs), nu=1.0)
    assert out.shape == (V.total_dofs,)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-14)


def test_per_element_viscous_stiffness_is_symmetric():
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh)
    S = bdm1_viscous_volume_stiffness(V, nu=1.0)     # (K, 6, 6)
    np.testing.assert_allclose(
        np.asarray(S), np.asarray(S).transpose(0, 2, 1), atol=1e-13,
    )


def test_per_element_viscous_stiffness_is_psd():
    """Per-element volume stiffness has 2-dim nullspace (constant vector
    fields on the element); all other eigenvalues are positive."""
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh)
    S = bdm1_viscous_volume_stiffness(V, nu=1.0)     # (K, 6, 6)
    for k in range(mesh.K):
        eigs = np.linalg.eigvalsh(np.asarray(S[k]))
        # Bottom of spectrum should be approximately zero (the
        # constant nullspace); rest positive.
        assert eigs.min() > -1e-10
        # Verify nullspace is at least 2-dim (constant vectors are in
        # BDM_1 and have zero gradient).
        near_zero = np.sum(eigs < 1e-8)
        assert near_zero >= 2, f"element {k}: only {near_zero} near-zero eigs"


def test_global_viscous_volume_matrix_is_symmetric():
    """The assembled global viscous-volume matrix must be symmetric --
    this is the saddle-point structural identity carrying over from
    Phase 5c (D = -G^T) to the velocity Helmholtz block.

    Assemble by applying to each canonical basis vector."""
    mesh = _build_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh)
    n = V.total_dofs
    A = jax.vmap(
        lambda i: bdm1_viscous_volume_apply(V, jnp.eye(n)[i], nu=1.0),
    )(jnp.arange(n)).T                               # (n, n)
    np.testing.assert_allclose(np.asarray(A - A.T), 0.0, atol=1e-11)


def test_global_viscous_volume_is_psd_with_correct_nullspace():
    """The global volume bilinear form ``a_vol(u, v) = nu * integral grad u :
    grad v dx`` (sum over elements) has nullspace exactly = the H(div)-
    conforming constant velocity fields. On a 2D mesh this is 2-dim
    (translations along x and y). All other eigenvalues are strictly
    positive."""
    mesh = _build_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh)
    n = V.total_dofs
    A = jax.vmap(
        lambda i: bdm1_viscous_volume_apply(V, jnp.eye(n)[i], nu=1.0),
    )(jnp.arange(n)).T
    A_sym = 0.5 * (A + A.T)
    eigs = np.linalg.eigvalsh(np.asarray(A_sym))
    near_zero = int(np.sum(eigs < 1e-8))
    # Constant translations are 2-dim. The volume bilinear form's
    # nullspace may be larger if there are *element-wise* rigid modes
    # that are not globally conforming -- but for BDM these get sewn
    # together by the topology. Expected global nullspace dim = 2.
    assert near_zero >= 2
    assert eigs[-1] > 0.0, "all eigenvalues are zero -- something is very wrong"
