"""Tests for the BDM-P_0 Stokes saddle-point solver (Phase 5e).

Validates the assembled monolithic operator on two canonical sanity
checks before convection / time stepping are added:

  * **Saddle-point structure**: solving with RHS ``(G p_test, 0)``
    recovers ``(u ≈ 0, p ≈ p_test + const)`` -- the test that
    pressure-gradient loads produce zero velocity (after killing the
    nullspace), which is the discrete analogue of "pressure-robustness
    for free" that motivates BDM in the first place.

  * **Operator structure**: the assembled saddle-point matrix is the
    expected ``[νA G; -D εM_p]`` block layout (verified by direct
    sub-block extraction).

The Kovasznay sentinel (Phase 5g) is the next milestone; this test is
its precursor -- if the Stokes saddle-point doesn't solve correctly,
the NS solver won't either.
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
    bdm1_viscous_apply,
)
from ddg.dg2d.coupled_ns_bdm import (
    stokes_apply_bdm_2d,
    stokes_matrix_bdm_2d,
    stokes_step_bdm_2d,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _build_mesh(Kx: int = 3, Ky: int = 3):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, 1)


def test_stokes_matrix_has_expected_block_structure():
    """The assembled matrix should match the block layout

        A_full[:n_u, :n_u]   = nu * A_visc       (SPD viscous block)
        A_full[:n_u, n_u:]   = G                  (pressure-gradient)
        A_full[n_u:, :n_u]   = -D                 (continuity)
        A_full[n_u:, n_u:]   = eps * M_p          (diagonal stab)

    by direct sub-block extraction and comparison with the
    individually-applied operators.
    """
    mesh = _build_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh)
    n_u, n_p = V.total_dofs, V.K
    nu, eps = 1.5, 1e-3
    A_full = stokes_matrix_bdm_2d(V, nu=nu, pressure_eps=eps, tau_scale=4.0)

    # Velocity-velocity block: should equal nu * (viscous matrix).
    A_visc = jax.vmap(
        lambda i: bdm1_viscous_apply(V, jnp.eye(n_u)[i], nu=nu, tau_scale=4.0),
    )(jnp.arange(n_u)).T
    np.testing.assert_allclose(
        np.asarray(A_full[:n_u, :n_u]), np.asarray(A_visc), atol=1e-11,
    )

    # Velocity-pressure block: G.
    G_mat = jax.vmap(
        lambda i: bdm_p0_gradient_apply(V, jnp.eye(n_p)[i]),
    )(jnp.arange(n_p)).T
    np.testing.assert_allclose(
        np.asarray(A_full[:n_u, n_u:]), np.asarray(G_mat), atol=1e-12,
    )

    # Pressure-velocity block: -D.
    D_mat = jax.vmap(
        lambda i: bdm_p0_divergence_apply(V, jnp.eye(n_u)[i]),
    )(jnp.arange(n_u)).T
    np.testing.assert_allclose(
        np.asarray(A_full[n_u:, :n_u]), -np.asarray(D_mat), atol=1e-12,
    )

    # Pressure-pressure block: eps * diag(area(K)).
    area_K = 2.0 * np.asarray(V.mesh.J[0])
    expected_pp = eps * np.diag(area_K)
    np.testing.assert_allclose(
        np.asarray(A_full[n_u:, n_u:]), expected_pp, atol=1e-13,
    )


def test_stokes_solves_pressure_gradient_load():
    """The canonical sanity check: solving with RHS = (G p_test, 0)
    recovers (u ≈ 0, p ≈ p_test + const).

    The pressure-gradient load is the discrete analogue of an
    irrotational body force -- a pressure-robust solver should NOT
    transmit any of it to the velocity. For BDM-P_0, this is exact at
    the discrete level: the velocity stays at machine zero.
    """
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh)
    n_p = V.K

    rng = np.random.default_rng(0)
    p_test = jnp.asarray(rng.standard_normal(n_p))
    # Centre p_test (mean zero) so it lives in the orthogonal complement
    # of the constant-pressure nullspace -- expected solution then has
    # mean-zero pressure too.
    p_test = p_test - jnp.mean(p_test)

    f_u = bdm_p0_gradient_apply(V, p_test)              # = G p_test
    u_sol, p_sol = stokes_step_bdm_2d(
        V, nu=1.0, f_u=f_u, pressure_eps=1e-10, tau_scale=4.0,
    )
    # Velocity should be ~zero (modulo the eps-scale of the regularisation).
    u_norm = float(jnp.linalg.norm(u_sol))
    print(f"||u_sol|| = {u_norm:.3e}")
    assert u_norm < 1e-5, f"Velocity not ~zero: ||u|| = {u_norm:.3e}"

    # Pressure: should match p_test up to a constant shift.
    p_sol_centred = p_sol - jnp.mean(p_sol)
    p_err = float(jnp.linalg.norm(p_sol_centred - p_test))
    print(f"||p_sol_centred - p_test|| = {p_err:.3e}")
    assert p_err < 1e-5, f"Pressure error: {p_err:.3e}"


def test_stokes_apply_zero_input_returns_zero():
    """Linearity sanity."""
    mesh = _build_mesh()
    V = BDM.from_mesh(mesh)
    u = jnp.zeros(V.total_dofs)
    p = jnp.zeros(V.K)
    out_u, out_p = stokes_apply_bdm_2d(V, u, p, nu=1.0, pressure_eps=1e-3)
    np.testing.assert_allclose(np.asarray(out_u), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(out_p), 0.0, atol=1e-14)


def test_stokes_matrix_invertible_with_pressure_stab():
    """With ε > 0, the saddle-point matrix is invertible (no nullspace
    -- ε kills the constant pressure mode). Verified via condition
    number; for a small mesh this should be well-conditioned."""
    mesh = _build_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh)
    A = stokes_matrix_bdm_2d(V, nu=1.0, pressure_eps=1e-3, tau_scale=4.0)
    cond = float(np.linalg.cond(np.asarray(A)))
    print(f"Stokes matrix condition number: {cond:.3e}")
    assert cond < 1e8, (
        f"Condition number {cond:.3e} suggests near-singularity; "
        f"expected well-conditioned with pressure_eps regularisation"
    )
