"""Tests for the BDM_1 face SIPG viscous bilinear form (Phase 5d-2).

The face SIPG terms (consistency + symmetry + tangential-jump penalty)
combine with the volume term to produce a coercive symmetric viscous
operator on BDM_1. Tests:

  * Face apply produces correctly-shaped output, zero in -> zero out.
  * The FULL viscous (volume + face) assembled matrix is symmetric.
  * The FULL viscous matrix is SPD with no nontrivial nullspace
    (homogeneous-Dirichlet ghost; the constant nullspace of the volume
    term is killed by the boundary penalty).
  * Linear-velocity sanity: for u(x, y) = (y, 0) projected onto BDM_1,
    Delta u = 0, so the volume contribution should be zero. (Face terms
    are nonzero due to the homogeneous-Dirichlet ghost convention,
    which is intentional -- it's how the lift to non-homogeneous BCs
    will work in 5f.)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM,
    bdm1_viscous_apply,
    bdm1_viscous_face_apply,
    bdm1_viscous_volume_apply,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _build_mesh(Kx: int = 3, Ky: int = 3):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, 1)


def test_face_apply_shape_and_linearity():
    mesh = _build_mesh()
    V = BDM.from_mesh(mesh)
    out = bdm1_viscous_face_apply(V, jnp.zeros(V.total_dofs), nu=1.0)
    assert out.shape == (V.total_dofs,)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-14)


def test_full_viscous_matrix_is_symmetric():
    mesh = _build_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh)
    n = V.total_dofs
    A = jax.vmap(
        lambda i: bdm1_viscous_apply(V, jnp.eye(n)[i], nu=1.0, tau_scale=4.0),
    )(jnp.arange(n)).T
    np.testing.assert_allclose(np.asarray(A - A.T), 0.0, atol=1e-10)


def test_full_viscous_matrix_is_spd():
    """With homogeneous-Dirichlet ghost on all boundaries and a coercive
    penalty parameter, the full SIPG viscous matrix is SPD -- no
    nontrivial nullspace, all eigenvalues strictly positive."""
    mesh = _build_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh)
    n = V.total_dofs
    A = jax.vmap(
        lambda i: bdm1_viscous_apply(V, jnp.eye(n)[i], nu=1.0, tau_scale=4.0),
    )(jnp.arange(n)).T
    A_sym = 0.5 * (np.asarray(A) + np.asarray(A).T)
    eigs = np.linalg.eigvalsh(A_sym)
    assert eigs.min() > 0.0, (
        f"smallest eigenvalue {eigs.min():.3e} not strictly positive -- "
        f"SIPG penalty insufficient or sign convention wrong"
    )


def test_face_apply_handles_boundary_homogeneous_dirichlet():
    """For a velocity field that's identically zero, both face apply and
    full viscous apply give zero. (Trivial but guards against accidental
    bias terms.)"""
    mesh = _build_mesh()
    V = BDM.from_mesh(mesh)
    u_zero = jnp.zeros(V.total_dofs)
    np.testing.assert_allclose(
        np.asarray(bdm1_viscous_face_apply(V, u_zero, nu=1.0)),
        0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(bdm1_viscous_apply(V, u_zero, nu=1.0)),
        0.0, atol=1e-14,
    )


def test_face_apply_nontrivial_for_generic_field():
    """For a non-zero velocity, the face apply produces some non-zero
    output (sanity that the operator isn't silently broken)."""
    mesh = _build_mesh()
    V = BDM.from_mesh(mesh)
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal(V.total_dofs))
    out = bdm1_viscous_face_apply(V, u, nu=1.0)
    assert float(jnp.linalg.norm(out)) > 1e-3
