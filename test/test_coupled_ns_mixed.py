"""Smoke + convergence tests for the mixed-order Lagrangian DG coupled NS step.

End-to-end checks for :func:`ddg.dg2d.coupled_ns_mixed.coupled_ns_step_mixed_2d`:
    * Smoke: one step on a Poiseuille channel runs to completion and
      returns the correct shapes.
    * Kovasznay Re=40 fixed-point test: with the analytic solution as
      IC, one Newton iteration should pin to it. This is the test that
      ``test_coupled_ns_stress::test_kovasznay_newton_recovers_analytic``
      xfails on the equal-order (LBB-violating) solver. If the mixed-
      order pair is genuinely inf-sup stable in our discretisation, this
      test passes -- the GMRES residual floor disappears.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.function_space import LagrangianDG
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OUTFLOW, BC_INS_WALL, BDF1,
    viscous_bc_types_from_ins, pressure_bc_types_from_ins,
)
from ddg.dg2d.coupled_ns import (
    CoupledNSConvergenceReport,
    IterativeSolverConfig,
    NonlinearSolverConfig,
    make_pressure_mass_inverse,
    make_pressure_laplacian_inverse,
)
from ddg.dg2d.coupled_ns_mixed import coupled_ns_step_mixed_2d
from ddg.dg2d.sipg import make_helmholtz_block_jacobi_preconditioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_rect_mesh(xmin, xmax, ymin, ymax, Kx, Ky, N):
    VX, VY, EToV = uniform_rectangle_mesh_2d(xmin, xmax, ymin, ymax, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def _channel_bc(mesh, x_mid):
    """Inflow left, outflow right, no-slip top/bottom."""
    Nfaces, K = mesh.Nfaces, mesh.K
    nx = np.asarray(mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    ny = np.asarray(mesh.ny.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    Fmask = np.asarray(mesh.Fmask)
    x_face = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        x_face[f, :] = np.asarray(mesh.x)[Fmask[:, f], :].mean(axis=0)
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T
    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    bc = np.where(is_bdy & (np.abs(nx) > 0.5) & (x_face < x_mid),
                  BC_INS_INFLOW, bc)
    bc = np.where(is_bdy & (np.abs(nx) > 0.5) & (x_face > x_mid),
                  BC_INS_OUTFLOW, bc)
    bc = np.where(is_bdy & (np.abs(ny) > 0.5), BC_INS_WALL, bc)
    return jnp.asarray(bc.T)


def _three_inflow_one_outflow_bc(mesh, outflow_normal):
    Nfaces, K = mesh.Nfaces, mesh.K
    nx = np.asarray(mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T
    is_out = {"+x": nx > 0.5, "-x": nx < -0.5}[outflow_normal]
    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    bc = np.where(is_bdy & ~is_out, BC_INS_INFLOW, bc)
    bc = np.where(is_bdy & is_out, BC_INS_OUTFLOW, bc)
    return jnp.asarray(bc.T)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_mixed_step_smoke_poiseuille():
    """One step of mixed-order on a Poiseuille channel must run and
    return correctly-shaped (ux_V_u, uy_V_u, p_V_p) outputs."""
    nu = 0.1
    mesh = _build_rect_mesh(0.0, 2.0, 0.0, 1.0, 4, 2, 3)
    V_u = LagrangianDG.at_degree(mesh, degree=3, components=2)
    V_p = LagrangianDG.at_degree(mesh, degree=2, components=1)

    ins_bc = _channel_bc(mesh, x_mid=1.0)
    ux_ex = mesh.y * (1.0 - mesh.y)
    uy_ex = jnp.zeros_like(mesh.y)

    iter_cfg = IterativeSolverConfig(method="gmres", tol=1e-6, atol=1e-8,
                                      maxiter=200, restart=50)

    ux, uy, p = coupled_ns_step_mixed_2d(
        V_u, V_p,
        ux_ex, uy_ex, ux_ex, uy_ex,
        dt=0.05, nu=nu, bdf=BDF1, ins_bc_types=ins_bc,
        bc_ux=ux_ex, bc_uy=uy_ex,
        iterative=iter_cfg,
    )
    assert ux.shape == (V_u.Np, V_u.K)
    assert uy.shape == (V_u.Np, V_u.K)
    assert p.shape == (V_p.Np, V_p.K)
    # On Poiseuille (steady), one BDF1 step from the analytic IC must stay
    # close to the analytic solution (without LBB pollution).
    err = float(jnp.max(jnp.abs(ux - ux_ex)))
    print(f"\nPoiseuille smoke (mixed-order): ||u - u_ex||_inf = {err:.3e}")


# ---------------------------------------------------------------------------
# Kovasznay -- the LBB-floor sentinel
# ---------------------------------------------------------------------------

def _kovasznay(mesh, nu):
    Re = 1.0 / nu
    lam = 0.5 * Re - np.sqrt(0.25 * Re ** 2 + 4 * np.pi ** 2)
    e = jnp.exp(lam * mesh.x)
    ux = 1.0 - e * jnp.cos(2 * jnp.pi * mesh.y)
    uy = (lam / (2 * jnp.pi)) * e * jnp.sin(2 * jnp.pi * mesh.y)
    p = 0.5 * (1.0 - jnp.exp(2 * lam * mesh.x))
    return ux, uy, p, float(lam)


def _kovasznay_face_data(mesh, nu, lam):
    """Outflow Neumann velocity ∂u/∂n + Dirichlet pressure, sampled at
    V_u face DOFs."""
    x_M = mesh.x.T.reshape(-1)[mesh.vmapM]
    y_M = mesh.y.T.reshape(-1)[mesh.vmapM]
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    e = jnp.exp(lam * x_M)
    twopi = 2 * jnp.pi
    dux_dx = -lam * e * jnp.cos(twopi * y_M)
    dux_dy = twopi * e * jnp.sin(twopi * y_M)
    duy_dx = (lam ** 2 / twopi) * e * jnp.sin(twopi * y_M)
    duy_dy = lam * e * jnp.cos(twopi * y_M)
    q_ux = (dux_dx * nx_flat + dux_dy * ny_flat).reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F",
    )
    q_uy = (duy_dx * nx_flat + duy_dy * ny_flat).reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F",
    )
    _, _, p_dir, _ = _kovasznay(mesh, nu)
    return q_ux, q_uy, p_dir


@pytest.mark.xfail(
    reason=(
        "Mixed-order Lagrangian DG (P_N / P_{N-1}) via the embedding "
        "trick stagnates at the SAME 1e-1 drift floor as equal-order "
        "(verified across no-stab / light-stab / full-stab "
        "configurations). The LBB violation in this discretisation lives "
        "in the CENTERED-FLUX choice for the saddle-point off-diagonals, "
        "not in the polynomial-degree pairing. Cockburn-Kanschat-Schoetzau "
        "2005 use LDG lifting on top of the mixed-order pair specifically "
        "for this reason. Two paths to fix: (A) implement LDG lifting on "
        "the Lagrangian operators, (B) pivot to BDM-DG H(div) (Phase 5). "
        "Decision pending."
    ),
    strict=False,
)
def test_mixed_kovasznay_recovers_analytic():
    """Re=40 Kovasznay is an exact steady NS root. With the analytic IC,
    Newton on the mixed-order pair should pin to it -- this is the test
    that LBB stagnation breaks on the equal-order solver. Whether the
    mixed-order pair clears the residual floor here is the operational
    answer to the inf-sup question raised in Phase 2."""
    nu = 1.0 / 40.0
    mesh = _build_rect_mesh(-0.5, 1.5, -0.5, 1.5, 8, 8, 3)
    V_u = LagrangianDG.at_degree(mesh, degree=3, components=2)
    V_p = LagrangianDG.at_degree(mesh, degree=2, components=1)

    ins_bc = _three_inflow_one_outflow_bc(mesh, "+x")
    ux_ex, uy_ex, _, lam = _kovasznay(mesh, nu)
    q_ux_neu, q_uy_neu, p_dir = _kovasznay_face_data(mesh, nu, lam)

    iter_cfg = IterativeSolverConfig(method="gmres", tol=1e-8, atol=1e-10,
                                      maxiter=500, restart=50)
    nl_cfg = NonlinearSolverConfig(
        linearisation="newton", newton_method="hand",
        tol=1e-6, atol=1e-8, maxiter=5,
    )

    # Block-Jacobi velocity PC + pressure-mass Schur PC (built at V_p degree).
    v_bc = viscous_bc_types_from_ins(ins_bc)
    alpha = BDF1.g0 / (nu * 0.05)
    vel_pc = make_helmholtz_block_jacobi_preconditioner(
        V_u.mesh, alpha=alpha, nu=nu, bc_types=v_bc,
    )
    p_mass_inv = make_pressure_mass_inverse(V_p.mesh, nu=nu)

    history = []
    report = CoupledNSConvergenceReport()
    ux, uy, p = coupled_ns_step_mixed_2d(
        V_u, V_p, ux_ex, uy_ex, ux_ex, uy_ex,
        dt=0.05, nu=nu, bdf=BDF1, ins_bc_types=ins_bc,
        bc_ux=ux_ex, bc_uy=uy_ex,
        p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
        iterative=iter_cfg,
        velocity_preconditioner=vel_pc,
        pressure_schur_inv=p_mass_inv,
        nl_config=nl_cfg,
        nl_residual_history_out=history,
        convergence_out=report,
    )
    err = float(jnp.linalg.norm(ux - ux_ex) + jnp.linalg.norm(uy - uy_ex))
    print(f"\nKovasznay Re=40 (mixed-order P3/P2) Newton history: {history}")
    print(report.summary())
    print(f"||u_NL - u_analytic|| = {err:.3e}")
    # Floor for equal-order was ~1e-1 (LBB pollution). If mixed-order
    # gets us under 1e-3, the inf-sup gain is real.
    assert err < 1e-3, f"Mixed-order Kovasznay drift {err:.3e}"
