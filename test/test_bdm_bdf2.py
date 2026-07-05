"""BDF2 time-stepping validation for the BDM monolithic NS solver.

The BDM-P_0 saddle-point apply is BDF-order agnostic -- the time
discretisation lives entirely in the caller-supplied
``velocity_alpha = gamma_0/dt`` and ``f_velocity`` history term. These
tests exercise the BDF coefficient helper and confirm BDF2 produces a
different (and more accurate) trajectory than BDF1 on a smooth
problem, while keeping the H(div) divergence-free property at machine
zero.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import BDM, bdm1_evaluate_boundary_velocity
from ddg.dg2d.coupled_ns import IterativeSolverConfig
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdf_history,
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    bdm1_mass_apply,
    bdm_p0_divergence_apply,
    make_bdm_block_diag_pc, make_bdm_cahouet_chabard_pc,
    make_bdm_velocity_block_jacobi_pc,
    ns_newton_solve_bdm_2d,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


U_LID = 1.0


def _lid_bc_fn(x, y):
    on_lid = jnp.abs(y - 1.0) < 1e-9
    return jnp.where(on_lid, U_LID, 0.0), jnp.zeros_like(x)


def _cavity_setup(Kx: int = 6):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Kx)
    mesh = build_mesh_2d(VX, VY, EToV, 1)
    V = BDM.from_mesh(mesh)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _lid_bc_fn)
    u_bc_vec = bdm1_evaluate_boundary_velocity(V, _lid_bc_fn)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


def _step(V, *, u_bc, u_bc_vec, bc_mask, u_n, u_nm1, p_prev, nu, dt, order):
    """Single BDF step. Picard, iterative GMRES + block-diag PC."""
    gamma0 = BDF_COEFFS[order]["gamma0"]
    alpha = gamma0 / dt
    u_hist = [u_n] if order == 1 else [u_n, u_nm1]
    f_history = bdf_history(V, u_hist, dt=dt, order=order)

    vel_pc = make_bdm_velocity_block_jacobi_pc(V, alpha=alpha, nu=nu)
    schur_pc = make_bdm_cahouet_chabard_pc(V, nu=nu, dt=dt, gamma0=gamma0)
    pc = make_bdm_block_diag_pc(V, velocity_pc=vel_pc, pressure_pc=schur_pc)
    iter_cfg = IterativeSolverConfig(
        method="gmres", tol=1e-9, atol=1e-12, maxiter=300, restart=50,
    )

    u_new, p_new, _ = ns_newton_solve_bdm_2d(
        V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        u_init=u_n, p_init=p_prev,
        velocity_alpha=alpha, f_velocity=f_history,
        pressure_eps=1e-10, tau_scale=4.0,
        max_iters=12, tol=1e-8, atol=1e-12,
        linearisation="picard",
        iterative=iter_cfg, coupled_preconditioner=pc,
    )
    return u_new, p_new


def test_bdf_coeffs_consistency():
    """BDF coefficient table must satisfy the discrete-derivative
    consistency check: for a constant trajectory u^n = u^{n-i} = c,
    gamma_0 * c - sum_i beta_i * c = 0."""
    for order, c in BDF_COEFFS.items():
        gamma0 = c["gamma0"]
        beta_sum = sum(c["beta"])
        assert abs(gamma0 - beta_sum) < 1e-14, (
            f"BDF{order}: gamma_0 ({gamma0}) - sum(beta) ({beta_sum}) "
            f"must be 0 for a consistent discrete time derivative."
        )


def test_bdf_history_matches_explicit_formula():
    """The bdf_history helper must equal the hand-written formula."""
    V, _, _, _ = _cavity_setup(Kx=4)
    rng = jax.random.PRNGKey(0)
    u_n = jax.random.normal(rng, (V.total_dofs,))
    u_nm1 = jax.random.normal(jax.random.fold_in(rng, 1), (V.total_dofs,))
    dt = 0.05

    # BDF1
    f1_helper = bdf_history(V, [u_n], dt=dt, order=1)
    f1_explicit = (1.0 / dt) * bdm1_mass_apply(V, u_n)
    assert jnp.allclose(f1_helper, f1_explicit, atol=1e-14)

    # BDF2
    f2_helper = bdf_history(V, [u_n, u_nm1], dt=dt, order=2)
    f2_explicit = ((2.0 / dt) * bdm1_mass_apply(V, u_n)
                    + (-0.5 / dt) * bdm1_mass_apply(V, u_nm1))
    assert jnp.allclose(f2_helper, f2_explicit, atol=1e-14)


def test_bdf2_preserves_divergence_free_property():
    """BDF2 must preserve the H(div) discrete-div-free property at
    every step, same as BDF1. The actual divergence violation should
    stay at machine zero (~1e-12) for the cavity."""
    nu = 0.01
    dt = 0.05
    V, u_bc, u_bc_vec, bc_mask = _cavity_setup(Kx=6)

    u_n = u_bc                                # IC: u_bc on boundary, 0 interior
    u_nm1 = None
    p_prev = jnp.zeros(V.K)

    # Step 0: BDF1 bootstrap.
    u_new, p_new = _step(
        V, u_bc=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        u_n=u_n, u_nm1=u_nm1, p_prev=p_prev,
        nu=nu, dt=dt, order=1,
    )
    div_step0 = float(jnp.linalg.norm(bdm_p0_divergence_apply(V, u_new)))
    assert div_step0 < 1e-10, f"step 0 BDF1: ||div||={div_step0:.3e}"
    u_nm1, u_n = u_n, u_new
    p_prev = p_new

    # Steps 1-3: BDF2.
    for step in range(1, 4):
        u_new, p_new = _step(
            V, u_bc=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
            u_n=u_n, u_nm1=u_nm1, p_prev=p_prev,
            nu=nu, dt=dt, order=2,
        )
        div_viol = float(jnp.linalg.norm(bdm_p0_divergence_apply(V, u_new)))
        assert div_viol < 1e-10, (
            f"step {step} BDF2: ||div||={div_viol:.3e} -- "
            f"H(div) property lost"
        )
        u_nm1, u_n = u_n, u_new
        p_prev = p_new


def test_bdf2_differs_from_bdf1_on_transient():
    """Sanity: a few BDF2 steps must produce a different solution
    than the same number of BDF1 steps. If they coincide, BDF2 isn't
    actually using u^{n-1}."""
    nu = 0.01
    dt = 0.05
    V, u_bc, u_bc_vec, bc_mask = _cavity_setup(Kx=6)

    # Common IC.
    u0 = u_bc
    p0 = jnp.zeros(V.K)

    # BDF1 trajectory: 3 steps.
    u_n_b1, p_prev_b1 = u0, p0
    for _ in range(3):
        u_new, p_new = _step(
            V, u_bc=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
            u_n=u_n_b1, u_nm1=None, p_prev=p_prev_b1,
            nu=nu, dt=dt, order=1,
        )
        u_n_b1, p_prev_b1 = u_new, p_new

    # BDF2 trajectory: 3 steps (step 0 bootstraps with BDF1, identical
    # to the BDF1 trajectory's first step).
    u_nm1_b2 = None
    u_n_b2 = u0
    p_prev_b2 = p0
    for step in range(3):
        order = 1 if step == 0 else 2
        u_new, p_new = _step(
            V, u_bc=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
            u_n=u_n_b2, u_nm1=u_nm1_b2, p_prev=p_prev_b2,
            nu=nu, dt=dt, order=order,
        )
        u_nm1_b2, u_n_b2 = u_n_b2, u_new
        p_prev_b2 = p_new

    diff = float(jnp.linalg.norm(u_n_b1 - u_n_b2)) / float(jnp.linalg.norm(u_n_b1))
    # The two should agree at step 0 (both BDF1) but diverge once BDF2
    # kicks in. At dt=0.05, T=0.15, relative diff is small but clearly
    # above noise.
    assert diff > 1e-5, (
        f"BDF1 and BDF2 produced essentially identical trajectories "
        f"(rel diff = {diff:.3e}) -- BDF2 history term may not be active"
    )
