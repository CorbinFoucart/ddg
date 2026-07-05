"""Sentinels for the Fehn-style dual-splitting BDM_k projection solver.

The projection solver lives in :mod:`ddg.dg2d.projection_ns_bdm`.

Tests here:

* :func:`test_smoke_steps_run`: bring-up. Runs three BDF1 + BDF2 steps on
  a tiny Kovasznay slice and asserts no NaN / no blow-up. This is the
  weakest sentinel; passes if the assembly + LU + step driver wire up.

* :func:`test_kovasznay_fixed_point_bdf1`: start from the exact Kovasznay
  steady state, take a single BDF1 step at small ``dt``. The truth is a
  (continuous-PDE) fixed point of the time-stepping, so the discrete
  update should reproduce it up to the splitting error which scales as
  ``dt``. Verifies all sign conventions in one shot -- if any term has
  a wrong sign, the "fixed-point" iterate will diverge from truth at
  ``O(1)`` instead of ``O(dt)``.

* :func:`test_kovasznay_steady_via_splitting`: run many BDF2 steps with
  larger ``dt`` from the truth as initial condition; verify the
  steady-state L^2 velocity error stays bounded by the same tolerance
  the monolithic Newton hits on the same mesh.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM, bdm_k_basis_at_points, bdm_k_evaluate_boundary_velocity,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc, bdm1_gather,
    bdf_history_k, ns_picard_solve_bdm_k_2d, ns_newton_solve_bdm_k_2d,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.projection_ns_bdm import (
    make_projection_solvers,
    projection_step_bdm_k_2d,
)


RE = 40.0
NU = 1.0 / RE
LAMBDA = RE / 2.0 - np.sqrt(RE ** 2 / 4.0 + 4.0 * np.pi ** 2)


def _kovasznay_uv(x, y):
    ux = 1.0 - jnp.exp(LAMBDA * x) * jnp.cos(2.0 * jnp.pi * y)
    uy = (LAMBDA / (2.0 * jnp.pi)) * jnp.exp(LAMBDA * x) * jnp.sin(2.0 * jnp.pi * y)
    return ux, uy


def _build_setup(Kx: int, Ky: int, order: int):
    VX, VY, EToV = uniform_rectangle_mesh_2d(-0.5, 1.0, -0.5, 1.5, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, 1)
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _kovasznay_uv)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _kovasznay_uv)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


def _l2_velocity_error(V: BDM, u_global) -> float:
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 2)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum("kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    u_h_at_q = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)

    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX)
    VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    r_np = np.asarray(r_q); s_np = np.asarray(s_q)
    x_q = (
        Vx0[:, None]
        + 0.5 * (r_np[None, :] + 1.0) * (Vx1 - Vx0)[:, None]
        + 0.5 * (s_np[None, :] + 1.0) * (Vx2 - Vx0)[:, None]
    )
    y_q = (
        Vy0[:, None]
        + 0.5 * (r_np[None, :] + 1.0) * (Vy1 - Vy0)[:, None]
        + 0.5 * (s_np[None, :] + 1.0) * (Vy2 - Vy0)[:, None]
    )
    u_ex, v_ex = _kovasznay_uv(jnp.asarray(x_q), jnp.asarray(y_q))
    diff_x = u_h_at_q[..., 0] - u_ex
    diff_y = u_h_at_q[..., 1] - v_ex
    sq = diff_x ** 2 + diff_y ** 2
    return float(jnp.sqrt(jnp.einsum("q,k,kq->", w_q, J, sq)))


def test_smoke_steps_run():
    """One BDF1 step + two BDF2 steps on a tiny Kovasznay slice.

    Asserts no NaN / no blow-up. This is the cheapest possible test that
    the dual-splitting wires up at all (operators, BC mask, LU factors).
    """
    V, u_bc, u_bc_vec, bc_mask = _build_setup(Kx=4, Ky=4, order=1)
    dt = 0.01

    pressure_solver_bdf1, h_inv_bdf1 = make_projection_solvers(
        V, dt=dt, nu=NU, order=1, bc_mask=bc_mask,
    )
    pressure_solver_bdf2, h_inv_bdf2 = make_projection_solvers(
        V, dt=dt, nu=NU, order=2, bc_mask=bc_mask,
    )

    u_n = u_bc
    u_nm1 = u_bc

    # BDF1 bootstrap
    u_new, p_new = projection_step_bdm_k_2d(
        V, [u_n], dt=dt, nu=NU, order=1,
        pressure_solver=pressure_solver_bdf1, h_inv=h_inv_bdf1,
        u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
    )
    assert jnp.all(jnp.isfinite(u_new))
    assert jnp.all(jnp.isfinite(p_new))
    u_nm1, u_n = u_n, u_new

    # Two BDF2 steps
    for _ in range(2):
        u_new, p_new = projection_step_bdm_k_2d(
            V, [u_n, u_nm1], dt=dt, nu=NU, order=2,
            pressure_solver=pressure_solver_bdf2, h_inv=h_inv_bdf2,
            u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        )
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(p_new))
        # No strong-BC assertion: the projection matches the monolithic
        # convention where the boundary trace is encoded implicitly via
        # the SIPG-lift on the RHS and the BC velocity DOFs are not
        # stored in ``u_new``. Strong overwrite is a caller-side option.
        u_nm1, u_n = u_n, u_new


def test_projection_block_schur_factorisation():
    """The projection block-Schur solve must EXACTLY factorise the
    projection's own (convection-free) saddle to LU roundoff.

    Subtlety -- this is NOT the same as "one Picard iteration of the
    monolithic". The projection is a semi-implicit (IMEX) scheme: its
    velocity Helmholtz block is ``H = alpha M + nu A_visc`` with NO
    convective term -- convection is treated explicitly, entering only
    through the extrapolated residual on the RHS. That is exactly why
    ``H`` can be factored once and reused across all steps (the per-step
    speedup). The fully-implicit monolithic, by contrast, carries the
    convective Jacobian ``C(u_star)`` in its LHS. The two coincide only
    at ``u_star = 0`` (rest), where ``C(0) = 0``; with a non-zero
    linearisation point they differ by the convective LHS block. The
    earlier version of this test compared against the monolithic at
    rest, which masked this distinction.

    The correct invariant to unit-test is that the block-Schur
    elimination is an exact factorisation of the saddle the projection
    assembles, ``[[H, G], [-D, -eps I]]`` with the strong-Dirichlet
    interior reduction. We verify that against a dense LU of the same
    saddle.
    """
    from ddg.dg2d.bdm import (
        bdm_k_mass_apply, bdm_k_viscous_apply,
        bdm_k_viscous_dirichlet_lift,
        bdm_k_gradient_apply, bdm_k_divergence_apply,
    )
    from ddg.dg2d.coupled_ns_bdm import ns_picard_apply_bdm_k_2d

    V, u_bc, u_bc_vec, bc_mask = _build_setup(Kx=4, Ky=4, order=1)
    n_u = V.total_dofs
    dim_p = 1
    n_p = V.K * dim_p
    dt = 0.05
    alpha = 1.0 / dt
    eps = 1.0e-10

    # Non-trivial linearisation point: BC trace injected on the
    # boundary (so u_star != 0 and the convective term is active --
    # the regime the old test never exercised).
    u_init = jnp.zeros(n_u)
    u_extrap = jnp.where(bc_mask, u_bc, u_init)
    M_f_hist = (1.0 / dt) * bdm_k_mass_apply(V, u_init)
    lift_visc = bdm_k_viscous_dirichlet_lift(V, u_bc_vec, nu=NU, tau_scale=4.0)

    # Projection residual at the linearisation point (same call the
    # projection step makes internally). Use the HOMOGENEOUS form
    # (u_bc_global=None -> a_homog) and subtract lift_visc, matching the
    # corrected projection_step convention; the Dirichlet trace is carried
    # by the BC-injection into u_extrap and by lift_visc (passing
    # u_bc_global here would select a_nh and double-count the lift).
    F_u, F_p = ns_picard_apply_bdm_k_2d(
        V, u_extrap, jnp.zeros((V.K, dim_p)),
        u_star_global=u_extrap, nu=NU,
        pressure_eps=0.0, tau_scale=4.0,
        velocity_alpha=alpha, f_velocity=M_f_hist,
        u_bc_global=None,
    )
    F_u = F_u - lift_visc
    rhs_aux = jnp.where(bc_mask, 0.0, -F_u)   # h_inv zeros BC rows
    rhs_p = -F_p.reshape(-1)

    # --- Dense reference: the projection's convection-free saddle ---
    def helm(u):
        return alpha * bdm_k_mass_apply(V, u) + bdm_k_viscous_apply(
            V, u, nu=NU, tau_scale=4.0)
    H = jax.vmap(helm, in_axes=1, out_axes=1)(jnp.eye(n_u))
    I_u = jnp.eye(n_u)
    H_c = jnp.where(bc_mask[:, None], I_u,
                    jnp.where(bc_mask[None, :], 0.0, H))
    G = jax.vmap(lambda pc: bdm_k_gradient_apply(V, pc.reshape(V.K, dim_p)),
                 in_axes=1, out_axes=1)(jnp.eye(n_p))          # (n_u, n_p)
    G_r = jnp.where(bc_mask[:, None], 0.0, G)                  # zero BC rows
    D = jax.vmap(lambda u: bdm_k_divergence_apply(V, u).reshape(-1),
                 in_axes=1, out_axes=1)(jnp.eye(n_u))          # (n_p, n_u)
    A_proj = jnp.block([[H_c, G_r],
                        [-D, -eps * jnp.eye(n_p)]])
    rhs = jnp.concatenate([rhs_aux, rhs_p])
    delta = jnp.linalg.solve(A_proj, rhs)
    u_ref = u_extrap + delta[:n_u]
    p_ref = delta[n_u:].reshape(V.K, dim_p)

    # --- Projection block-Schur step ---
    pressure_solver, h_inv = make_projection_solvers(
        V, dt=dt, nu=NU, order=1, bc_mask=bc_mask, pressure_eps=eps,
    )
    u_proj, p_proj = projection_step_bdm_k_2d(
        V, [u_init], dt=dt, nu=NU, order=1,
        pressure_solver=pressure_solver, h_inv=h_inv,
        u_bc_global=u_extrap, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
    )

    p_ref_def = p_ref.at[:, 0].add(-jnp.mean(p_ref[:, 0]))
    p_proj_def = p_proj.at[:, 0].add(-jnp.mean(p_proj[:, 0]))

    rel_u = float(jnp.linalg.norm(u_proj - u_ref)
                  / max(float(jnp.linalg.norm(u_ref)), 1e-30))
    rel_p = float(jnp.linalg.norm(p_proj_def - p_ref_def)
                  / max(float(jnp.linalg.norm(p_ref_def)), 1e-30))
    print(f"  block-Schur vs dense-LU of own saddle: "
          f"rel_u={rel_u:.3e}, rel_p={rel_p:.3e}")
    assert rel_u < 1.0e-9, (
        f"block-Schur disagrees with dense LU of its own saddle: "
        f"rel_u={rel_u:.3e}"
    )
    assert rel_p < 1.0e-7, f"pressure disagreement {rel_p:.3e}"


def test_projection_preserves_dirichlet_bc_trace():
    """Regression test: after ONE projection step from rest with a
    non-zero Dirichlet velocity boundary condition, ``u_new`` must
    actually carry the prescribed boundary values on the Dirichlet
    DOFs. The original projection step left BC DOFs at zero (the
    initial value) because the interior-reduction in ``h_inv`` zeroes
    BC rows of the delta, so ``u_new[bdy] = u_extrap[bdy]`` and
    ``u_extrap[bdy] = 0`` from rest -- the BC trace was never
    injected. The fix injects ``u_bc_global`` into the BC rows of
    ``u_extrap`` inside ``projection_step_bdm_k_2d``.

    Without this fix the Schaefer-Turek cylinder produces zero
    downstream velocity (the inflow is never seen by the solver).
    """
    V, u_bc, u_bc_vec, bc_mask = _build_setup(Kx=4, Ky=4, order=1)
    n_u = V.total_dofs
    dt = 0.05
    pressure_solver, h_inv = make_projection_solvers(
        V, dt=dt, nu=NU, order=1, bc_mask=bc_mask,
    )
    u_init = jnp.zeros(n_u)
    u_new, _ = projection_step_bdm_k_2d(
        V, [u_init], dt=dt, nu=NU, order=1,
        pressure_solver=pressure_solver, h_inv=h_inv,
        u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
    )
    # BC trace agreement on Dirichlet DOFs.
    u_new_bc = jnp.where(bc_mask, u_new, 0.0)
    u_bc_only = jnp.where(bc_mask, u_bc, 0.0)
    err = float(jnp.linalg.norm(u_new_bc - u_bc_only))
    norm = float(jnp.linalg.norm(u_bc_only))
    rel = err / max(norm, 1e-30)
    print(f"  BC-trace error: ||u_new[bc] - u_bc|| / ||u_bc|| = {rel:.3e}")
    assert rel < 1.0e-12, (
        f"projection step lost the Dirichlet BC trace: "
        f"||u_new[bc] - u_bc|| / ||u_bc|| = {rel:.3e}"
    )


def test_kovasznay_steady_via_splitting():
    """Initialise from truth, time-march, verify steady-state error stays
    bounded by the monolithic spatial error tolerance.

    Validates that the dual-splitting drift over many time steps does not
    accumulate into a quantity comparable to the spatial discretisation
    error. At BDF2 / small dt, this is a strong test of the time-stepping
    stability and the projection accuracy together.
    """
    V, u_bc, u_bc_vec, bc_mask = _build_setup(Kx=6, Ky=6, order=1)
    # Reference monolithic solution
    u_mono, _ = ns_picard_solve_bdm_k_2d(
        V, nu=NU,
        u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        u_init=u_bc, pressure_eps=1e-10, tau_scale=4.0,
        max_iters=40, tol=1e-9,
    )
    mono_err = _l2_velocity_error(V, u_mono)

    # dt must respect the projection's explicit-convection stability
    # limit. A dt-sweep on this Kovasznay slice (Kx=Ky=6, BDM_1) shows
    # the IMEX scheme NaNs at dt=0.01 (step ~10) but is stable and
    # accurate (split error 0.17-0.43x the monolithic reference) for
    # dt <= 0.005 -- the CFL-type bound characterised in the
    # projection-vs-monolithic CFL study. Use dt=0.005.
    dt = 0.005
    pressure_solver_bdf1, h_inv_bdf1 = make_projection_solvers(
        V, dt=dt, nu=NU, order=1, bc_mask=bc_mask,
    )
    pressure_solver_bdf2, h_inv_bdf2 = make_projection_solvers(
        V, dt=dt, nu=NU, order=2, bc_mask=bc_mask,
    )

    u_n = u_mono
    u_nm1 = u_mono
    # BDF1 bootstrap
    u_new, _ = projection_step_bdm_k_2d(
        V, [u_n], dt=dt, nu=NU, order=1,
        pressure_solver=pressure_solver_bdf1, h_inv=h_inv_bdf1,
        u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
    )
    u_nm1, u_n = u_n, u_new
    # 20 BDF2 steps
    for s in range(20):
        u_new, _ = projection_step_bdm_k_2d(
            V, [u_n, u_nm1], dt=dt, nu=NU, order=2,
            pressure_solver=pressure_solver_bdf2, h_inv=h_inv_bdf2,
            u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        )
        u_nm1, u_n = u_n, u_new
        assert jnp.all(jnp.isfinite(u_new)), f"NaN at step {s}"

    split_err = _l2_velocity_error(V, u_n)
    print(f"  monolithic L2 err = {mono_err:.3e}")
    print(f"  splitting  L2 err = {split_err:.3e}")
    # Splitting drift should sit within 5x of the monolithic spatial error.
    # ``mono_err`` includes the Picard-overwrite divergence pollution
    # so it's a generous reference -- we mainly want stability and
    # boundedness here.
    assert split_err < 5.0 * mono_err, (
        f"splitting drift {split_err:.3e} >> 5 * monolithic err {mono_err:.3e}"
    )
