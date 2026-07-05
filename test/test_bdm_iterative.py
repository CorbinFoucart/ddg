"""Iterative-solver validation for the BDM coupled NS solver.

Validates the matrix-free GMRES + block-diagonal preconditioner path
against the existing direct-LU path on the cavity Re=100 problem. The
two paths must produce the same physical solution to numerical
precision; the iterative path is the scalable production route, the LU
path is the reference.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import BDM, bdm1_evaluate_boundary_velocity
from ddg.dg2d.coupled_ns import IterativeSolverConfig
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask,
    bdm1_evaluate_dirichlet_bc,
    make_bdm_block_diag_pc,
    make_bdm_cahouet_chabard_pc,
    make_bdm_pressure_mass_inverse,
    make_bdm_velocity_block_jacobi_pc,
    ns_newton_solve_bdm_2d,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


U_LID = 1.0


def _lid_bc_fn(x, y):
    on_lid = jnp.abs(y - 1.0) < 1e-9
    return jnp.where(on_lid, U_LID, 0.0), jnp.zeros_like(x)


def _build_cavity_setup(Kx: int = 8):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Kx)
    mesh = build_mesh_2d(VX, VY, EToV, 1)
    V = BDM.from_mesh(mesh)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _lid_bc_fn)
    u_bc_vec = bdm1_evaluate_boundary_velocity(V, _lid_bc_fn)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


# ---------------------------------------------------------------------------
# Component PCs in isolation
# ---------------------------------------------------------------------------

def test_velocity_block_jacobi_pc_shape_and_linearity():
    """The block-Jacobi velocity PC must produce a vector of the
    velocity-DOF dimension, be linear, and zero in -> zero out."""
    V, _, _, _ = _build_cavity_setup(Kx=6)
    pc = make_bdm_velocity_block_jacobi_pc(V, alpha=1.0, nu=0.01)
    n_u = V.total_dofs
    z = jnp.zeros(n_u)
    assert pc(z).shape == (n_u,)
    np.testing.assert_allclose(np.asarray(pc(z)), 0.0, atol=1e-14)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(n_u))
    y = jnp.asarray(rng.standard_normal(n_u))
    a, b = 1.5, -0.7
    np.testing.assert_allclose(
        np.asarray(pc(a * x + b * y)),
        a * np.asarray(pc(x)) + b * np.asarray(pc(y)),
        atol=1e-10,
    )


def test_pressure_mass_inverse_is_diagonal_division():
    """For BDM-P_0 the pressure mass is exactly diagonal
    (M_p[K,K] = 2*J_K). The inverse PC should just divide by that."""
    V, _, _, _ = _build_cavity_setup(Kx=4)
    nu = 0.01
    pc = make_bdm_pressure_mass_inverse(V, nu=nu, scale=1.0)
    n_p = V.K
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.standard_normal(n_p))
    # Expected: p[K] / (nu * 2 * J_K)
    J = V.mesh.J[0]
    expected = p / (nu * 2.0 * J)
    np.testing.assert_allclose(np.asarray(pc(p)), np.asarray(expected),
                                atol=1e-13)


def test_cahouet_chabard_pc_is_finite_and_acts_correctly():
    """The combined Cahouet-Chabard PC is finite + non-trivial on a
    random pressure vector. Smoke test."""
    V, _, _, _ = _build_cavity_setup(Kx=4)
    pc = make_bdm_cahouet_chabard_pc(V, nu=0.01, dt=0.05, gamma0=1.0)
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.standard_normal(V.K))
    out = pc(p)
    assert out.shape == (V.K,)
    assert jnp.all(jnp.isfinite(out))
    assert float(jnp.linalg.norm(out)) > 0


# ---------------------------------------------------------------------------
# End-to-end: GMRES + block-PC vs direct LU on cavity Re=100
# ---------------------------------------------------------------------------

def test_iterative_matvec_matches_lu_on_cavity_re100_8x8():
    """The matvec correctness check. Solve cavity Re=100 on an 8x8
    mesh both ways: direct LU as the reference, then GMRES on the same
    operator (no PC, large iteration budget). The two solutions must
    match to Krylov tolerance.

    This isolates "is the matvec right" from "is the PC any good" --
    if the matvec is consistent with the matrix the LU path assembles,
    no-PC GMRES converges (slowly) to the same solution.
    """
    V, u_bc, u_bc_vec, bc_mask = _build_cavity_setup(Kx=8)
    nu = 0.01

    rng = np.random.default_rng(0)
    interior_noise = jnp.asarray(rng.standard_normal(V.total_dofs)) * 0.1
    u_init = u_bc + jnp.where(bc_mask, 0.0, interior_noise)

    u_lu, p_lu, _ = ns_newton_solve_bdm_2d(
        V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        u_init=u_init, p_init=jnp.zeros(V.K),
        pressure_eps=1e-10, tau_scale=4.0,
        max_iters=40, tol=1e-9, atol=1e-12,
        linearisation="picard",
    )

    iter_cfg = IterativeSolverConfig(
        method="gmres", tol=1e-10, atol=1e-12,
        maxiter=2000, restart=100,
    )
    inner_hist = []
    u_it, p_it, _ = ns_newton_solve_bdm_2d(
        V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        u_init=u_init, p_init=jnp.zeros(V.K),
        pressure_eps=1e-10, tau_scale=4.0,
        max_iters=40, tol=1e-9, atol=1e-12,
        linearisation="picard",
        iterative=iter_cfg,
        # No preconditioner -- this is the "matvec correctness" test.
        coupled_preconditioner=None,
        inner_residual_history=inner_hist,
    )
    print(f"\nInner GMRES final-residuals per Newton step: "
          f"{[f'{h:.2e}' for h in inner_hist[:5]]}")
    diff_u = float(jnp.linalg.norm(u_it - u_lu) /
                    (jnp.linalg.norm(u_lu) + 1e-30))
    diff_p = float(jnp.linalg.norm(p_it - p_lu) /
                    (jnp.linalg.norm(p_lu) + 1e-30))
    print(f"||u_it - u_lu|| / ||u_lu|| = {diff_u:.3e}")
    print(f"||p_it - p_lu|| / ||p_lu|| = {diff_p:.3e}")
    assert diff_u < 1e-6, f"velocity matvec inconsistent: {diff_u:.3e}"
    assert diff_p < 1e-4, f"pressure matvec inconsistent: {diff_p:.3e}"


def test_pc_smoke_block_diag_at_least_runs():
    """Smoke test: the block-diagonal PC (block-Jacobi velocity +
    diagonal Schur) runs without error. Quality of convergence is a
    separate question -- block-Jacobi-of-volume-only is known to be
    weak for steady Stokes-cavity (no mass-time term, SIPG penalty
    dominates the velocity block). Production-quality PCs (full
    velocity LU, AMG, augmented-Lagrangian) are follow-ups."""
    V, u_bc, u_bc_vec, bc_mask = _build_cavity_setup(Kx=6)
    nu = 0.01

    rng = np.random.default_rng(0)
    interior_noise = jnp.asarray(rng.standard_normal(V.total_dofs)) * 0.1
    u_init = u_bc + jnp.where(bc_mask, 0.0, interior_noise)

    vel_pc = make_bdm_velocity_block_jacobi_pc(V, alpha=0.0, nu=nu)
    schur_pc = make_bdm_pressure_mass_inverse(V, nu=nu)
    pc = make_bdm_block_diag_pc(V, velocity_pc=vel_pc, pressure_pc=schur_pc)
    iter_cfg = IterativeSolverConfig(
        method="gmres", tol=1e-6, atol=1e-10,
        maxiter=100, restart=50,
    )
    u_pc, p_pc, _ = ns_newton_solve_bdm_2d(
        V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        u_init=u_init, p_init=jnp.zeros(V.K),
        pressure_eps=1e-10, max_iters=5, tol=1e-3, atol=1e-6,
        linearisation="picard",
        iterative=iter_cfg, coupled_preconditioner=pc,
    )
    # Just check finite + boundary-respecting; not asking for accuracy.
    assert jnp.all(jnp.isfinite(u_pc))
    assert jnp.all(jnp.isfinite(p_pc))
    bc_diff = float(jnp.linalg.norm(
        jnp.where(bc_mask, u_pc - u_bc, 0.0)
    ))
    assert bc_diff < 1e-8, f"BC pin broken: {bc_diff:.3e}"
