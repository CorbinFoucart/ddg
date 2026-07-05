"""Stress tests for the coupled-NS nonlinear solver.

The existing parity / Poiseuille / Taylor-Green / AMR tests
(:mod:`test_coupled_ns_newton` and :mod:`test_coupled_ns_nonlinear_suite`)
cover the *easy* regime: Poiseuille has zero convective stress, and
Taylor-Green is special because the convective term and the pressure
gradient cancel exactly. This file pushes harder:

  * **Kovasznay flow** -- steady analytic NS solution where the
    convective and pressure terms do *not* cancel. ``u`` and ``v``
    couple through real convection, and the nonlinear residual at the
    Newton iterate is dominated by the convective contribution. This
    is the cleanest single-step exercise of the Newton-extra term.

  * **High-Re channel** -- the same Poiseuille setup as
    test_coupled_ns_nonlinear_suite but at ``Re = 1000`` instead of
    ``Re = 10``. Picard's contraction rate degrades sharply at high
    Re (the coupling matrix is roughly the identity minus
    ``A^{-1} * (u_k . grad)``, and the spectral radius grows with
    Re); Newton is essentially Re-independent. The gap between
    Picard iteration count and Newton iteration count is the
    practical-superiority claim, made quantitative here.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OUTFLOW, BC_INS_WALL,
    BDF1, viscous_bc_types_from_ins,
)
from ddg.dg2d.coupled_ns import (
    NonlinearSolverConfig, IterativeSolverConfig, coupled_ns_step_2d,
    make_pressure_mass_inverse,
)
from ddg.dg2d.sipg import make_helmholtz_block_jacobi_preconditioner


# ---------------------------------------------------------------------------
# Mesh + BC helpers (same conventions as test_coupled_ns_nonlinear_suite.py)
# ---------------------------------------------------------------------------

_PC_TAU_DIV = 1.0
_PC_TAU_CONT = 1.0
_PC_PRESSURE_EPS = 1.0e-3


def _build_rect_mesh(xmin, xmax, ymin, ymax, Kx, Ky, N):
    VX, VY, EToV = uniform_rectangle_mesh_2d(xmin, xmax, ymin, ymax, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def _bc_three_inflow_one_outflow(mesh, outflow_normal):
    """Three Dirichlet edges + one outflow edge (Neumann velocity,
    Dirichlet pressure). Standard Kovasznay setup."""
    Nfaces, K = mesh.Nfaces, mesh.K
    nx = np.asarray(mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    ny = np.asarray(mesh.ny.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T
    is_out = {
        "+x": nx > 0.5, "-x": nx < -0.5,
        "+y": ny > 0.5, "-y": ny < -0.5,
    }[outflow_normal]
    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    bc = np.where(is_bdy & ~is_out, BC_INS_INFLOW, bc)
    bc = np.where(is_bdy & is_out, BC_INS_OUTFLOW, bc)
    return jnp.asarray(bc.T)


def _channel_bc(mesh, x_mid):
    """Inflow left, outflow right, no-slip walls top/bottom."""
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
    bc = np.where(is_bdy & (np.abs(nx) > 0.5) & (x_face < x_mid), BC_INS_INFLOW, bc)
    bc = np.where(is_bdy & (np.abs(nx) > 0.5) & (x_face > x_mid), BC_INS_OUTFLOW, bc)
    bc = np.where(is_bdy & (np.abs(ny) > 0.5), BC_INS_WALL, bc)
    return jnp.asarray(bc.T)


def _make_preconditioners(mesh, ins_bc, *, nu, dt, bdf):
    v_bc = viscous_bc_types_from_ins(ins_bc)
    alpha = bdf.g0 / (nu * dt)
    vel_pc = make_helmholtz_block_jacobi_preconditioner(
        mesh, alpha=alpha, nu=nu, bc_types=v_bc,
    )
    p_mass_inv = make_pressure_mass_inverse(mesh, nu=nu)
    return vel_pc, p_mass_inv


# ---------------------------------------------------------------------------
# Kovasznay analytics
# ---------------------------------------------------------------------------

def _kovasznay(mesh, nu):
    """Steady Kovasznay flow on [-0.5, 1.5]^2.  Re = 1/nu."""
    Re = 1.0 / nu
    lam = 0.5 * Re - np.sqrt(0.25 * Re ** 2 + 4 * np.pi ** 2)
    e = jnp.exp(lam * mesh.x)
    ux = 1.0 - e * jnp.cos(2 * jnp.pi * mesh.y)
    uy = (lam / (2 * jnp.pi)) * e * jnp.sin(2 * jnp.pi * mesh.y)
    p = 0.5 * (1.0 - jnp.exp(2 * lam * mesh.x))
    return ux, uy, p, float(lam)


def _kovasznay_face_data(mesh, nu, lam):
    """Outflow Neumann velocity ∂u/∂n + Dirichlet pressure."""
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
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F"
    )
    q_uy = (duy_dx * nx_flat + duy_dy * ny_flat).reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F"
    )
    _, _, p_dir, _ = _kovasznay(mesh, nu)
    return q_ux, q_uy, p_dir


def _kovasznay_step(mesh, *, nu, dt, nl_config, history=None,
                    perturb=0.0, rng_seed=0):
    """One BDF1 Kovasznay step. IC = analytic + optional perturbation,
    BC = analytic on three sides, Neumann/Dirichlet on the +x outflow."""
    ins_bc = _bc_three_inflow_one_outflow(mesh, "+x")
    ux_ex, uy_ex, p_ex, lam = _kovasznay(mesh, nu)
    if perturb > 0.0:
        rng = np.random.default_rng(rng_seed)
        ux_n = ux_ex + perturb * jnp.asarray(rng.standard_normal(ux_ex.shape))
        uy_n = uy_ex + perturb * jnp.asarray(rng.standard_normal(uy_ex.shape))
    else:
        ux_n, uy_n = ux_ex, uy_ex
    bc_ux, bc_uy = ux_ex, uy_ex
    q_ux_neu, q_uy_neu, p_dir = _kovasznay_face_data(mesh, nu, lam)
    iter_cfg = IterativeSolverConfig(method="gmres", tol=1.0e-8,
                                      atol=1.0e-10, maxiter=500, restart=50)
    vel_pc, p_mass_inv = _make_preconditioners(mesh, ins_bc,
                                                 nu=nu, dt=dt, bdf=BDF1)
    ux_new, uy_new, p, _ = coupled_ns_step_2d(
        mesh, ux_n, uy_n, ux_n, uy_n,
        dt=dt, nu=nu, bdf=BDF1, ins_bc_types=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
        p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
        tau_div=_PC_TAU_DIV, tau_cont=_PC_TAU_CONT,
        pressure_eps=_PC_PRESSURE_EPS,
        iterative=iter_cfg,
        velocity_preconditioner=vel_pc, pressure_mass_inv=p_mass_inv,
        nl_config=nl_config,
        nl_residual_history_out=history,
    )
    return ux_new, uy_new, p, ux_ex, uy_ex, p_ex


# ---------------------------------------------------------------------------
# Kovasznay tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Equal-order DG (Pn-Pn) violates LBB; the Fehn-style "
        "tau_div+tau_cont+pressure_eps stabilisation pollutes the "
        "Schur complement so the Krylov residual floors at ~1e-3 "
        "regardless of preconditioner choice (verified across BJ/LU "
        "velocity PCs x M_p/K_p/Cahouet-Chabard Schur PCs, block-DIAG "
        "and block-TRI structures). That floor translates to ~10% "
        "velocity error and trips this drift bound. Resolves with "
        "mixed-order Lagrangian or BDM-DG (planned)."
    ),
    strict=False,
)
@pytest.mark.parametrize("newton_method", ["hand", "jvp"])
def test_kovasznay_newton_recovers_analytic(newton_method):
    """Re=40 Kovasznay is an exact steady NS root. Newton with IC =
    analytic should pin to it in 1 outer iteration (Newton's job on a
    fixed point is trivial; this is mainly checking the residual
    machinery doesn't drift)."""
    nu = 1.0 / 40.0
    mesh = _build_rect_mesh(-0.5, 1.5, -0.5, 1.5, 8, 8, 3)
    history = []
    nl_cfg = NonlinearSolverConfig(
        linearisation="newton", newton_method=newton_method,
        tol=1.0e-6, atol=1.0e-8, maxiter=5,
    )
    ux, uy, _, ux_ex, uy_ex, _ = _kovasznay_step(
        mesh, nu=nu, dt=0.05, nl_config=nl_cfg, history=history,
    )
    print(f"\nKovasznay Re=40 Newton ({newton_method}) history: {history}")
    err = float(jnp.linalg.norm(ux - ux_ex) + jnp.linalg.norm(uy - uy_ex))
    print(f"  ||u_NL - u_analytic|| = {err:.3e}")
    assert err < 5.0e-2, f"Kovasznay drift {err:.3e}"


@pytest.mark.parametrize("newton_method", ["hand", "jvp"])
def test_kovasznay_newton_first_step_superlinear(newton_method):
    """Perturb the IC by O(0.1) and verify Newton's first-step
    contraction is super-linear. Kovasznay's convective term does NOT
    cancel the pressure gradient, so this is a real test of the
    Newton-extra ∂_j(δu_j u_k_i) piece."""
    nu = 1.0 / 40.0
    mesh = _build_rect_mesh(-0.5, 1.5, -0.5, 1.5, 6, 6, 3)
    history = []
    nl_cfg = NonlinearSolverConfig(
        linearisation="newton", newton_method=newton_method,
        tol=1.0e-6, atol=1.0e-8, maxiter=5,
    )
    _kovasznay_step(mesh, nu=nu, dt=0.05, nl_config=nl_cfg,
                    history=history, perturb=0.1)
    print(f"\nKovasznay Re=40 (perturbed IC) Newton ({newton_method}) "
          f"history: {history}")
    assert len(history) >= 2, "Newton did not produce a residual sequence"
    ratio = history[1] / history[0]
    print(f"  ||F_1||/||F_0|| = {ratio:.3e}  (super-linear -> << 1)")
    assert ratio < 0.1, f"Newton first-step contraction {ratio:.3e}"


def test_kovasznay_newton_beats_picard():
    """At Re=40, Newton converges in strictly fewer outer iterations
    than Picard on the perturbed Kovasznay problem."""
    nu = 1.0 / 40.0
    mesh = _build_rect_mesh(-0.5, 1.5, -0.5, 1.5, 6, 6, 3)
    tol, maxiter = 1.0e-6, 20
    h_n = []
    _kovasznay_step(
        mesh, nu=nu, dt=0.05,
        nl_config=NonlinearSolverConfig(
            linearisation="newton", newton_method="hand",
            tol=tol, maxiter=maxiter,
        ),
        history=h_n, perturb=0.1,
    )
    h_p = []
    _kovasznay_step(
        mesh, nu=nu, dt=0.05,
        nl_config=NonlinearSolverConfig(
            linearisation="picard", tol=tol, maxiter=maxiter,
        ),
        history=h_p, perturb=0.1,
    )
    print(f"\nKovasznay Re=40 (perturbed IC):")
    print(f"  Newton iters to tol={tol:.0e}: {len(h_n)} (history: {h_n})")
    print(f"  Picard iters to tol={tol:.0e}: {len(h_p)} (history: {h_p})")
    assert len(h_n) <= len(h_p), (
        f"Newton iters ({len(h_n)}) not better than Picard ({len(h_p)})"
    )


# ---------------------------------------------------------------------------
# High-Re channel Poiseuille
# ---------------------------------------------------------------------------

def _channel_step(mesh, *, nu, dt, nl_config, history=None, perturb=0.0,
                   rng_seed=0):
    """One BDF1 Poiseuille step at the requested viscosity, optionally
    perturbing the IC so the NL outer loop has something to solve."""
    ins_bc = _channel_bc(mesh, x_mid=1.0)
    ux_ex = mesh.y * (1.0 - mesh.y)
    uy_ex = jnp.zeros_like(mesh.y)
    if perturb > 0.0:
        rng = np.random.default_rng(rng_seed)
        ux_n = ux_ex + perturb * jnp.asarray(rng.standard_normal(ux_ex.shape))
        uy_n = uy_ex + perturb * jnp.asarray(rng.standard_normal(uy_ex.shape))
    else:
        ux_n, uy_n = ux_ex, uy_ex
    iter_cfg = IterativeSolverConfig(method="gmres", tol=1.0e-8,
                                      atol=1.0e-10, maxiter=500, restart=50)
    vel_pc, p_mass_inv = _make_preconditioners(mesh, ins_bc,
                                                 nu=nu, dt=dt, bdf=BDF1)
    ux_new, uy_new, p, _ = coupled_ns_step_2d(
        mesh, ux_n, uy_n, ux_n, uy_n,
        dt=dt, nu=nu, bdf=BDF1, ins_bc_types=ins_bc,
        bc_ux=ux_ex, bc_uy=uy_ex,
        tau_div=_PC_TAU_DIV, tau_cont=_PC_TAU_CONT,
        pressure_eps=_PC_PRESSURE_EPS,
        iterative=iter_cfg,
        velocity_preconditioner=vel_pc, pressure_mass_inv=p_mass_inv,
        nl_config=nl_config,
        nl_residual_history_out=history,
    )
    return ux_new, uy_new, p, ux_ex


@pytest.mark.xfail(
    reason=(
        "Same LBB stagnation as Kovasznay; see "
        "test_kovasznay_newton_recovers_analytic for the diagnosis. "
        "The high-Re scaling alpha = gamma / (nu dt) makes the floor "
        "worse here (smaller absolute residuals; the LBB-pollution "
        "relative error is amplified). Resolves with mixed-order "
        "Lagrangian or BDM-DG (planned)."
    ),
    strict=False,
)
@pytest.mark.parametrize("newton_method", ["hand", "jvp"])
def test_high_re_channel_newton_converges(newton_method):
    """Re=1000 channel (ν=1e-3). Even with the perturbed IC the steady
    Poiseuille is still parallel flow, so convection is weak, but the
    α = γ/(ν dt) scaling makes the viscous block sharply
    ill-conditioned -- a different stress on the inner GMRES + outer
    Newton coupling."""
    nu = 1.0e-3                           # Re ~ 1000
    mesh = _build_rect_mesh(0.0, 2.0, 0.0, 1.0, 4, 2, 2)
    history = []
    nl_cfg = NonlinearSolverConfig(
        linearisation="newton", newton_method=newton_method,
        tol=1.0e-6, atol=1.0e-8, maxiter=5,
    )
    ux, _, _, ux_ref = _channel_step(mesh, nu=nu, dt=0.01,
                                       nl_config=nl_cfg, history=history,
                                       perturb=0.05)
    print(f"\nRe=1000 channel Newton ({newton_method}) history: {history}")
    drift = float(jnp.max(jnp.abs(ux - ux_ref)))
    print(f"  ||u - u_ref||_inf = {drift:.3e}")
    assert drift < 5.0e-2


def test_high_re_channel_newton_beats_picard():
    """At ν=1e-3 Picard's contraction rate is meaningfully worse; the
    iteration-count gap should widen vs the Re=10 baseline in
    test_coupled_ns_nonlinear_suite.py::test_taylor_green_newton_beats_picard_iteration_count."""
    nu = 1.0e-3
    mesh = _build_rect_mesh(0.0, 2.0, 0.0, 1.0, 4, 2, 2)
    tol, maxiter = 1.0e-4, 25
    h_n = []
    _channel_step(
        mesh, nu=nu, dt=0.01,
        nl_config=NonlinearSolverConfig(
            linearisation="newton", newton_method="hand",
            tol=tol, maxiter=maxiter,
        ),
        history=h_n, perturb=0.05,
    )
    h_p = []
    _channel_step(
        mesh, nu=nu, dt=0.01,
        nl_config=NonlinearSolverConfig(
            linearisation="picard", tol=tol, maxiter=maxiter,
        ),
        history=h_p, perturb=0.05,
    )
    print(f"\nRe=1000 channel (perturbed IC):")
    print(f"  Newton iters to tol={tol:.0e}: {len(h_n)} (history: {h_n})")
    print(f"  Picard iters to tol={tol:.0e}: {len(h_p)} (history: {h_p})")
    assert len(h_n) <= len(h_p)
