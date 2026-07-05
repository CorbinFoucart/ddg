"""Monolithic coupled incompressible Navier-Stokes (exadg ``BDFCoupledSolution``).

In contrast to the projection schemes in :mod:`ddg.dg2d.incompressible_ns`,
this module solves velocity and pressure **together** as a saddle-point
system each time step:

    [ A_M(u*)   G ] [ u ]   [ b_u ]
    [   D      0 ] [ p ] = [ b_p ]

where:

  * ``A_M(u*) = (γ/dt) M  +  ν·A_sipg  +  N'(·; u*)`` — the linearised
    velocity-momentum operator. Time-derivative (BDF mass), viscous
    Laplacian (SIPG, per component), and convective term linearised by
    freezing the transport velocity to the BDF-extrapolated value
    ``u* = β_0 u_n + β_1 u_{n−1}`` (Picard linearisation /
    "linearly-implicit" convection in exadg's taxonomy).

  * ``G`` is the weak pressure-gradient operator
    ``(G p)·v = −∫ p (∇·v) + ∫_∂ {{p}} [v·n]`` (with appropriate
    boundary terms).

  * ``D = −G^T`` (up to BC contributions) is the weak velocity
    divergence operator.

The system is symmetric indefinite. We assemble the full dense matrix
via ``jax.jacobian`` on a vectorised apply, then LU-factor and solve.
For our mesh sizes (~ 10⁴ DOFs total) this is feasible without
iterative solvers — the natural follow-up is block preconditioners +
Krylov methods (we cite this as a deferred TODO).

This module is **opt-in** alongside the projection schemes — it shares
nothing with :mod:`incompressible_ns` besides the BC tagging enum and
SIPG infrastructure.

Boundary conditions follow the same INS BC tags
(:data:`BC_INS_INFLOW` / :data:`BC_INS_OUTFLOW` / :data:`BC_INS_WALL`
/ :data:`BC_INS_OBSTACLE`):

  * Inflow / Wall / Obstacle: velocity Dirichlet ``u = u_BC``; pressure
    natural Neumann ``∂p/∂n = 0`` (no contribution to the system).
  * Outflow: velocity natural ``∂u/∂n = 0``; pressure Dirichlet
    ``p = p_BC`` (defaults to zero).

**Stabilisation** (Fehn et al. JCP 2018, arXiv:1801.08103 + exadg's
``operator_coupled.cpp``): equal-order DG saddle-point systems are
inf-sup unstable without help. We follow exadg's recipe exactly:

  1. **Symmetric saddle-point**: scale the gradient block by
     ``+sc`` and the divergence block by ``−sc`` with
     ``sc = γ/dt``. Because ``D = −G^T`` already (verified to 1e-15
     adjointness), this gives a symmetric matrix
     ``[A, sc·G; sc·G^T, 0]``.

  2. **Velocity-block penalties**: add ``τ_d D_pen + τ_c C_pen``
     (divergence + normal-continuity penalty operators from
     :mod:`incompressible_ns`) to the (0,0) block. These provide the
     inf-sup-stabilising effect without adding any pressure-pressure
     term. exadg adds them when
     ``apply_penalty_terms_in_postprocessing_step == false``.

  3. **Pressure-mass regularisation** (default ``ε = 0``): only if the
     pressure is globally undetermined (no outflow face). For the
     default outflow Dirichlet ``p = 0`` BC the matrix is non-singular
     already and ``ε = 0`` is fine.

With (1) + (2) the equal-order coupled scheme is stable and the
accuracy ceiling drops to the genuine spatial / temporal-discretisation
error.

**Nonlinear treatment** (this module supports several levels):

The continuous semi-discrete NS residual after BDF time integration is

    F_u(u, p) = (γ/dt) M u + N(u) + ν A_sipg u + G p − b_u  = 0
    F_p(u)   = −D u + b_p                                    = 0

with the convective term ``N(u)_i = ∂_j(u_j u_i)`` in divergence form.
``N`` is the only nonlinearity. Its Fréchet derivative at ``u_k`` along
direction ``δu`` is::

    N'(u_k) · δu = ∂_j(u_k_j δu_i) + ∂_j(δu_j u_k_i)
                   ── Picard term ──   ── Newton extra ──

The Picard linearisation freezes the transport velocity (``u_k_j``)
and keeps only the first term. Newton's full Jacobian also includes
the linearisation w.r.t. the transport velocity.

This module offers three nonlinear levels, in order of increasing
cost and accuracy:

  1. **1-sweep Picard** (``nl_config=None``, the default). The
     transport velocity is frozen at the BDF extrapolation
     ``u* = β₀ uⁿ + β₁ uⁿ⁻¹``; one linear saddle-point solve gives
     ``uⁿ⁺¹``. Cheap, accuracy bounded by the BDF-extrapolation
     error.

  2. **Outer Picard to convergence**
     (``nl_config=NonlinearSolverConfig(linearisation="picard")``).
     Iterate ``u_{k+1} = A_Picard(u_k)⁻¹ RHS`` until
     ``||u_{k+1} − u_k|| < tol``. Linear convergence; converges to
     the actual fixed point of the lagged-velocity formulation. No
     Jacobian assembly needed.

  3. **Newton-Krylov** (``linearisation="newton"``). Build the full
     Jacobian-vector product including the extra
     ``∂_j(δu_j u_k_i)`` term, then run inner GMRES on ``J(u_k) δ
     = −F(u_k)`` and update ``u_{k+1} = u_k + δ``. Quadratic local
     convergence.

Two implementations of the Newton-extra term are provided and tested
for parity: a **hand-derived** version
(:func:`_conv_apply_newton_extra_2d`, transparent in source) and a
**JAX-AD** version using :func:`jax.jvp` on the full nonlinear
convection (one of the main motivations for the differentiable
library --- the Jacobian comes for free). Their relative wall-time
is part of the benchmark suite.

Caveat: the outer loop is wired through the **matrix-free GMRES**
path only; the direct-LU path would require re-factoring the matrix
at every Newton iteration and is not the regime we benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.mesh import Mesh2D
from ddg.dg2d.sipg import (
    BC_DIRICHLET, BC_INTERIOR, BC_NEUMANN,
    _face_1d_mass_matrices, _local_mass,
    factor_sipg_2d, factor_sipg_helmholtz_2d,
    sipg_apply_2d, sipg_dirichlet_load_2d, sipg_neumann_load_2d,
)
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OBSTACLE, BC_INS_OUTFLOW,
    BC_INS_WALL,
    BDFCoeffs, BDF1, BDF2,
    _ins_bc_facenodes_flat,
    continuity_penalty_apply_2d, divergence_penalty_apply_2d,
    pressure_bc_types_from_ins, viscous_bc_types_from_ins,
)


# ---------------------------------------------------------------------------
# Operator primitives
# ---------------------------------------------------------------------------

def _grad_apply_2d(
    mesh: Mesh2D,
    p: jax.Array,
    *,
    ins_bc_types: jax.Array,
    M_ref: jax.Array,
    p_dir: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Weak pressure-gradient operator applied to velocity test functions.

    Returns ``(G_x p, G_y p)`` representing
    ``(G p · e_x, G p · e_y) = ∫ ∇p · v``. In weak / IBP form:

        ∫ ∇p · v = −∫ p ∇·v + ∫_∂ {{p}} v·n

    Interior faces use the centered flux ``{{p}}``; pressure-Dirichlet
    faces (outflow) use ``p_M = p_BC``; pressure-Neumann faces
    (inflow/wall/obstacle) use ``p_P = p_M`` so the boundary integral
    reduces to ``∫_∂_N p_M (v·n)`` — this is exactly what makes the
    pair ``(G, D)`` adjoint.
    """
    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K
    # Volume: −∫ p ∇·v = −p · (Dx^T M v_x + Dy^T M v_y)  →  in the apply
    # form, the velocity load is −(Dx^T M p)·e_x − (Dy^T M p)·e_y (per
    # element). With the J scaling already accounted for via mesh.J.
    M_p = M_ref @ p
    vol_x = -mesh.J * (mesh.rx * (mesh.Dr.T @ M_p) + mesh.sx * (mesh.Ds.T @ M_p))
    vol_y = -mesh.J * (mesh.ry * (mesh.Dr.T @ M_p) + mesh.sy * (mesh.Ds.T @ M_p))

    # Face term: ∫_∂ {{p}} v·n
    p_flat = p.T.reshape(-1)
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")

    p_M = p_flat[mesh.vmapM]
    p_P_raw = p_flat[mesh.vmapP]

    bc_per_facenode = _ins_bc_facenodes_flat(mesh, ins_bc_types)
    is_outflow = bc_per_facenode == BC_INS_OUTFLOW
    if p_dir is None:
        p_dir_flat_M = jnp.zeros_like(p_M)
    else:
        p_dir_flat = p_dir.T.reshape(-1)
        p_dir_flat_M = p_dir_flat[mesh.vmapM]
    # Pressure-Dirichlet (outflow) face: p_P = p_BC ghost gives {{p}} = (p_M + p_BC)/2.
    # Pressure-Neumann (inflow/wall/obstacle) face: p_P = p_M → {{p}} = p_M.
    # Interior face: p_P from neighbour.
    p_P = jnp.where(is_outflow, 2.0 * p_dir_flat_M - p_M, p_P_raw)
    # Boundary faces have vmapP = vmapM so p_P_raw = p_M already.
    avg_p = 0.5 * (p_M + p_P)

    flux_x = (avg_p * nx_flat).reshape((Nfp_Nf, K), order="F")
    flux_y = (avg_p * ny_flat).reshape((Nfp_Nf, K), order="F")

    surf_x = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * flux_x)))
    surf_y = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * flux_y)))
    return vol_x + surf_x, vol_y + surf_y


def _div_apply_2d(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    *,
    ins_bc_types: jax.Array,
    M_ref: jax.Array,
    bc_ux: jax.Array | None = None,
    bc_uy: jax.Array | None = None,
) -> jax.Array:
    """Weak velocity-divergence operator applied to pressure test functions.

    Returns ``D u = ∫ (∇·u) q`` (per pressure test q). In weak form:

        ∫ (∇·u) q = −∫ u · ∇q + ∫_∂ {{u·n}} q

    Velocity-Dirichlet faces use ``u·n = u_BC·n``; outflow uses
    ``u·n = u_M·n`` (one-sided trace).
    """
    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K
    # Volume: −∫ u · ∇q = −∫ (u_x ∂q/∂x + u_y ∂q/∂y)
    M_ux = M_ref @ ux
    M_uy = M_ref @ uy
    vol = -mesh.J * (
        mesh.rx * (mesh.Dr.T @ M_ux) + mesh.sx * (mesh.Ds.T @ M_ux)
        + mesh.ry * (mesh.Dr.T @ M_uy) + mesh.sy * (mesh.Ds.T @ M_uy)
    )

    # Face: ∫_∂ {{u·n}} q
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")

    bc_per_facenode = _ins_bc_facenodes_flat(mesh, ins_bc_types)
    is_vel_dir = (
        (bc_per_facenode == BC_INS_INFLOW)
        | (bc_per_facenode == BC_INS_WALL)
        | (bc_per_facenode == BC_INS_OBSTACLE)
    )

    ux_M = ux_flat[mesh.vmapM]
    uy_M = uy_flat[mesh.vmapM]
    ux_P_raw = ux_flat[mesh.vmapP]
    uy_P_raw = uy_flat[mesh.vmapP]
    if bc_ux is not None:
        bc_ux_flat = bc_ux.T.reshape(-1)
        ux_BC = bc_ux_flat[mesh.vmapM]
    else:
        ux_BC = jnp.zeros_like(ux_M)
    if bc_uy is not None:
        bc_uy_flat = bc_uy.T.reshape(-1)
        uy_BC = bc_uy_flat[mesh.vmapM]
    else:
        uy_BC = jnp.zeros_like(uy_M)
    # Velocity-Dirichlet ghost: u_P = 2 u_BC − u_M  (mirror), so {{u}} = u_BC.
    # Outflow / interior: standard {{u}} = 0.5(u_M + u_P_raw).
    ux_P = jnp.where(is_vel_dir, 2.0 * ux_BC - ux_M, ux_P_raw)
    uy_P = jnp.where(is_vel_dir, 2.0 * uy_BC - uy_M, uy_P_raw)
    avg_un = 0.5 * (ux_M + ux_P) * nx_flat + 0.5 * (uy_M + uy_P) * ny_flat

    flux = avg_un.reshape((Nfp_Nf, K), order="F")
    surf = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * flux)))
    return vol + surf


def _conv_apply_2d(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    *,
    u_star_x: jax.Array,
    u_star_y: jax.Array,
    ins_bc_types: jax.Array,
    M_ref: jax.Array,
    bc_ux: jax.Array | None = None,
    bc_uy: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Linearised convection ``N'(u; u*) = ∇·(u* ⊗ u)`` with LF flux.

    Returns ``(M·N_x, M·N_y)``. The transport velocity ``u*`` is
    frozen (no nonlinearity in ``u``) so the operator is linear in
    ``u`` and can be assembled as part of the saddle-point matrix.

    Volume term in divergence form: ``-∫ (u* ⊗ u) : ∇v = -∫ (u*·∇v_i) u_i``
    per component. Surface term: local Lax-Friedrichs with
    ``λ_max = max(|u*·n|_M, |u*·n|_P)``.
    """
    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K

    # Volume: ∫ (u* ⊗ u) : ∇v contributes (u·u*_x) · ∂v/∂x + (u·u*_y) · ∂v/∂y
    # Per-component: ∂(u*_x · u_i)/∂x + ∂(u*_y · u_i)/∂y, integrated against v_i.
    # In weak form: -∫ (u*_x u_i ∂v_i/∂x + u*_y u_i ∂v_i/∂y)
    M_fx_ux = M_ref @ (u_star_x * ux)
    M_fy_ux = M_ref @ (u_star_y * ux)
    M_fx_uy = M_ref @ (u_star_x * uy)
    M_fy_uy = M_ref @ (u_star_y * uy)
    vol_x = -mesh.J * (
        mesh.rx * (mesh.Dr.T @ M_fx_ux) + mesh.sx * (mesh.Ds.T @ M_fx_ux)
        + mesh.ry * (mesh.Dr.T @ M_fy_ux) + mesh.sy * (mesh.Ds.T @ M_fy_ux)
    )
    vol_y = -mesh.J * (
        mesh.rx * (mesh.Dr.T @ M_fx_uy) + mesh.sx * (mesh.Ds.T @ M_fx_uy)
        + mesh.ry * (mesh.Dr.T @ M_fy_uy) + mesh.sy * (mesh.Ds.T @ M_fy_uy)
    )

    # Face LF flux
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)
    usx_flat = u_star_x.T.reshape(-1)
    usy_flat = u_star_y.T.reshape(-1)
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")

    bc_per_facenode = _ins_bc_facenodes_flat(mesh, ins_bc_types)
    is_vel_dir = (
        (bc_per_facenode == BC_INS_INFLOW)
        | (bc_per_facenode == BC_INS_WALL)
        | (bc_per_facenode == BC_INS_OBSTACLE)
    )

    ux_M = ux_flat[mesh.vmapM]
    uy_M = uy_flat[mesh.vmapM]
    ux_P_raw = ux_flat[mesh.vmapP]
    uy_P_raw = uy_flat[mesh.vmapP]
    if bc_ux is not None:
        bc_ux_flat = bc_ux.T.reshape(-1)
        bc_uy_flat = bc_uy.T.reshape(-1)
        ux_BC = bc_ux_flat[mesh.vmapM]
        uy_BC = bc_uy_flat[mesh.vmapM]
    else:
        ux_BC = jnp.zeros_like(ux_M)
        uy_BC = jnp.zeros_like(ux_M)
    ux_P = jnp.where(is_vel_dir, ux_BC, ux_P_raw)
    uy_P = jnp.where(is_vel_dir, uy_BC, uy_P_raw)

    us_n = usx_flat[mesh.vmapM] * nx_flat + usy_flat[mesh.vmapM] * ny_flat
    lam = jnp.abs(us_n)
    # Standard LF flux: F* = 0.5(F_M + F_P) − 0.5 λ_max (u_P − u_M)
    # → upwind in the limit. For u*·n > 0, F* reduces to (u*·n) u_M.
    flux_x = 0.5 * us_n * (ux_M + ux_P) - 0.5 * lam * (ux_P - ux_M)
    flux_y = 0.5 * us_n * (uy_M + uy_P) - 0.5 * lam * (uy_P - uy_M)
    flux_x_2d = flux_x.reshape((Nfp_Nf, K), order="F")
    flux_y_2d = flux_y.reshape((Nfp_Nf, K), order="F")
    surf_x = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * flux_x_2d)))
    surf_y = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * flux_y_2d)))
    return vol_x + surf_x, vol_y + surf_y


def _mass_apply(mesh: Mesh2D, u: jax.Array, M_ref: jax.Array) -> jax.Array:
    return mesh.J * (M_ref @ u)


def _conv_apply_nonlinear_2d(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    *,
    ins_bc_types: jax.Array,
    M_ref: jax.Array,
    bc_ux: jax.Array | None = None,
    bc_uy: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Full nonlinear convection ``N(u) = ∇·(u ⊗ u)`` (= Picard apply at u_star=u).

    Used for the nonlinear residual and as the function we differentiate
    with ``jax.jvp`` to obtain the Newton Jacobian-vector product.
    """
    return _conv_apply_2d(
        mesh, ux, uy,
        u_star_x=ux, u_star_y=uy,
        ins_bc_types=ins_bc_types, M_ref=M_ref,
        bc_ux=bc_ux, bc_uy=bc_uy,
    )


def _conv_apply_newton_extra_2d(
    mesh: Mesh2D,
    delta_ux: jax.Array,
    delta_uy: jax.Array,
    u_k_x: jax.Array,
    u_k_y: jax.Array,
    *,
    ins_bc_types: jax.Array,
    M_ref: jax.Array,
    bc_ux: jax.Array | None = None,
    bc_uy: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Hand-derived Newton **extra** term beyond the Picard linearisation.

    The Fréchet derivative of ``N(u) = ∂_j(u_j u_i)`` at ``u = u_k`` along
    direction ``δu`` is::

        N'(u_k) · δu = ∂_j(u_k_j · δu_i)   +   ∂_j(δu_j · u_k_i)
                       ─── Picard term ───   ─── Newton extra ───

    The Picard term is exactly what :func:`_conv_apply_2d` computes with
    ``u_star = u_k`` and ``u = δu``. This function returns the Newton
    extra term: divergence of ``(δu ⊗ u_k)`` plus the LF surface-flux
    contribution from differentiating ``|u·n|`` and ``(u·n)`` w.r.t. the
    transport velocity. The two pieces summed give the full
    Jacobian-vector product ``N'(u_k) · δu``.

    Volume term is symmetric in ``(δu, u_k)`` (the integrand
    ``(δu·u_k) ∂v/∂x`` is bilinear in ``δu, u_k``), so the formula
    mirrors :func:`_conv_apply_2d` with the roles swapped.

    Surface (Lax-Friedrichs) at face ``f``:

        Picard part        :   0.5 (u_k·n)(δu_M + δu_P)
                             - 0.5 |u_k·n|(δu_P - δu_M)
        Newton extra part  :   0.5 (δu·n)(u_k_M + u_k_P)
                             - 0.5 sign(u_k·n)(δu·n)(u_k_P - u_k_M)

    where ``δu`` has homogeneous Dirichlet BC (it is an increment of an
    iterate that already satisfies ``u_BC``).
    """
    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K

    # Volume: same shape as _conv_apply_2d's volume but with δu and u_k
    # playing the roles of (u_star, u). The integrand u_star·u is
    # symmetric in its two arguments.
    M_fx_ux = M_ref @ (delta_ux * u_k_x)
    M_fy_ux = M_ref @ (delta_uy * u_k_x)
    M_fx_uy = M_ref @ (delta_ux * u_k_y)
    M_fy_uy = M_ref @ (delta_uy * u_k_y)
    vol_x = -mesh.J * (
        mesh.rx * (mesh.Dr.T @ M_fx_ux) + mesh.sx * (mesh.Ds.T @ M_fx_ux)
        + mesh.ry * (mesh.Dr.T @ M_fy_ux) + mesh.sy * (mesh.Ds.T @ M_fy_ux)
    )
    vol_y = -mesh.J * (
        mesh.rx * (mesh.Dr.T @ M_fx_uy) + mesh.sx * (mesh.Ds.T @ M_fx_uy)
        + mesh.ry * (mesh.Dr.T @ M_fy_uy) + mesh.sy * (mesh.Ds.T @ M_fy_uy)
    )

    # Surface: Newton extra flux derivative.
    delta_ux_flat = delta_ux.T.reshape(-1)
    delta_uy_flat = delta_uy.T.reshape(-1)
    ukx_flat = u_k_x.T.reshape(-1)
    uky_flat = u_k_y.T.reshape(-1)
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")

    bc_per_facenode = _ins_bc_facenodes_flat(mesh, ins_bc_types)
    is_vel_dir = (
        (bc_per_facenode == BC_INS_INFLOW)
        | (bc_per_facenode == BC_INS_WALL)
        | (bc_per_facenode == BC_INS_OBSTACLE)
    )

    # u_k boundary handling: u_k satisfies the full Dirichlet u_BC.
    if bc_ux is not None:
        bc_ux_flat = bc_ux.T.reshape(-1)
        bc_uy_flat = bc_uy.T.reshape(-1)
        ukx_BC = bc_ux_flat[mesh.vmapM]
        uky_BC = bc_uy_flat[mesh.vmapM]
    else:
        ukx_BC = jnp.zeros_like(ukx_flat[mesh.vmapM])
        uky_BC = jnp.zeros_like(uky_flat[mesh.vmapM])
    ukx_M = ukx_flat[mesh.vmapM]
    uky_M = uky_flat[mesh.vmapM]
    ukx_P_raw = ukx_flat[mesh.vmapP]
    uky_P_raw = uky_flat[mesh.vmapP]
    ukx_P = jnp.where(is_vel_dir, ukx_BC, ukx_P_raw)
    uky_P = jnp.where(is_vel_dir, uky_BC, uky_P_raw)

    # δu · n (M side only — only ∂(u·n)/∂u along δu enters the Newton extra).
    delta_ux_M = delta_ux_flat[mesh.vmapM]
    delta_uy_M = delta_uy_flat[mesh.vmapM]
    delta_u_n = delta_ux_M * nx_flat + delta_uy_M * ny_flat
    uk_n = ukx_M * nx_flat + uky_M * ny_flat
    sgn_uk_n = jnp.sign(uk_n)        # 0 at exact zero — JAX subgradient of |·|

    # Newton extra flux:
    #   F_x_extra = 0.5 (δu·n)(u_k_x_M + u_k_x_P) - 0.5 sign(u_k·n)(δu·n)(u_k_x_P - u_k_x_M)
    flux_x_extra = (
        0.5 * delta_u_n * (ukx_M + ukx_P)
        - 0.5 * sgn_uk_n * delta_u_n * (ukx_P - ukx_M)
    )
    flux_y_extra = (
        0.5 * delta_u_n * (uky_M + uky_P)
        - 0.5 * sgn_uk_n * delta_u_n * (uky_P - uky_M)
    )
    flux_x_2d = flux_x_extra.reshape((Nfp_Nf, K), order="F")
    flux_y_2d = flux_y_extra.reshape((Nfp_Nf, K), order="F")
    surf_x = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * flux_x_2d)))
    surf_y = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * flux_y_2d)))
    return vol_x + surf_x, vol_y + surf_y


# ---------------------------------------------------------------------------
# Block-matrix assembly
# ---------------------------------------------------------------------------

@dataclass
class CoupledNSOperators:
    """Pre-factored monolithic operators for one BDF order + dt + u_star."""
    A_full: jax.Array
    lu: jax.Array
    piv: jax.Array
    viscous_bc_types: jax.Array
    pressure_eps: float


@dataclass
class CoupledNSConvergenceReport:
    """Per-step convergence diagnostics emitted by :func:`coupled_ns_step_2d`.

    Pass an empty instance via ``convergence_out=`` and read fields back
    after the step. All fields are populated whenever the iterative
    (GMRES / BiCGStab) path is taken; outer-loop fields are populated
    only when ``nl_config`` is supplied.

    The single most diagnostic field is ``inner_converged``: a False
    here means the inner Krylov solver hit ``maxiter`` without reaching
    its tolerance, which is the most common reason an outer Newton /
    Picard loop "looks like" it's not progressing. Pair with
    ``inner_residuals_achieved`` to see *how* far off it was.
    """
    # Outer-loop diagnostics (populated only when nl_config is supplied).
    outer_residuals: list = field(default_factory=list)
    outer_converged: bool = False
    # Inner-solver diagnostics (populated on every iterative call).
    # Length 1 for the 1-sweep default; len = (#outer iters) for nl_config.
    inner_residuals_initial: list = field(default_factory=list)
    inner_residuals_achieved: list = field(default_factory=list)
    inner_converged: list = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        if self.outer_residuals:
            lines.append(
                f"outer: {len(self.outer_residuals)} iters, "
                f"converged={self.outer_converged}, "
                f"history={[f'{r:.2e}' for r in self.outer_residuals]}"
            )
        for k, (r0, r1, ok) in enumerate(zip(
            self.inner_residuals_initial,
            self.inner_residuals_achieved,
            self.inner_converged,
        )):
            tag = "ok" if ok else "FAIL"
            lines.append(
                f"  inner[{k}]: ||rhs||={r0:.2e} -> ||res||={r1:.2e}  "
                f"[{tag}]"
            )
        return "\n".join(lines) if lines else "(no diagnostics captured)"


@dataclass(frozen=True)
class NonlinearSolverConfig:
    """Outer nonlinear iteration for the coupled NS saddle-point.

    The 1-sweep Picard default (``nl_config=None``) freezes the convective
    transport at the BDF-extrapolated ``u* = β₀ uⁿ + β₁ uⁿ⁻¹`` and solves
    the resulting linear saddle-point system once. Setting an
    ``NonlinearSolverConfig`` activates an outer loop that converges the
    actual nonlinear residual at the new time level.

    Unified δ-formulation:

        for k = 0, 1, ...
            F(u_k, p_k) = full_nonlinear_apply(u_k, p_k) - RHS
            stop if  ||F|| <= tol·||F_0|| + atol
            δ        = J(u_k)⁻¹ · (-F)
            (u, p)  ← (u_k, p_k) + δ

    ``linearisation`` selects ``J``:

      * ``"picard"``: ``J = A_Picard(u_k)`` — linear in the unknowns,
        no Newton extra. Linear convergence (fixed-point iteration).
      * ``"newton"``: ``J = A_Picard(u_k) + N'_extra(u_k)`` where
        ``N'_extra(u_k)·δu = ∂_j(δu_j u_k_i)`` is the linearisation of
        the convective term w.r.t. the *transport* velocity --- the
        piece omitted by Picard. Quadratic convergence near smooth
        roots.

    ``newton_method``: ``"hand"`` or ``"jvp"``. Mathematically equivalent
    (verified by parity test); the choice is between an explicit
    hand-coded derivation and automatic differentiation via
    :func:`jax.jvp` on the full nonlinear convection. The performance
    of the two is a benchmark in itself.

    ``tol`` / ``atol``: relative / absolute outer-iteration tolerance.

    ``maxiter``: outer-iteration cap.
    """
    linearisation: str = "newton"             # "picard" | "newton"
    newton_method: str = "hand"               # "hand" | "jvp"
    tol: float = 1.0e-8
    atol: float = 0.0
    maxiter: int = 20


@dataclass(frozen=True)
class IterativeSolverConfig:
    """Configuration for matrix-free iterative solves.

    ``method``:
      * ``"gmres"`` — generalised minimal residual (good for the indefinite
        saddle-point system; default for coupled NS).
      * ``"bicgstab"`` — biconjugate-gradient stabilised.

    ``tol`` / ``atol``: relative / absolute convergence tolerance.
    ``maxiter``: hard cap on Krylov iterations.
    ``restart``: GMRES restart length (ignored by bicgstab).
    """
    method: str = "gmres"
    tol: float = 1e-8
    atol: float = 0.0
    maxiter: int = 1000
    restart: int = 30


def _flat3(ux: jax.Array, uy: jax.Array, p: jax.Array) -> jax.Array:
    return jnp.concatenate([
        ux.T.reshape(-1), uy.T.reshape(-1), p.T.reshape(-1)
    ])


def _unflat3(
    v: jax.Array, Np: int, K: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n = Np * K
    return (
        v[:n].reshape(K, Np).T,
        v[n:2 * n].reshape(K, Np).T,
        v[2 * n:].reshape(K, Np).T,
    )


def coupled_ns_apply_2d(
    mesh: Mesh2D,
    ux: jax.Array, uy: jax.Array, p: jax.Array,
    *,
    nu: float,
    gamma0_over_dt: float,
    u_star_x: jax.Array,
    u_star_y: jax.Array,
    ins_bc_types: jax.Array,
    viscous_bc_types_sipg: jax.Array,
    M_ref: jax.Array,
    m1d_per_face: jax.Array,
    tau_scale: float = 200.0,
    tau_div: float = 0.0,
    tau_cont: float = 0.0,
    pressure_eps: float = 0.0,
    newton_u_k: tuple[jax.Array, jax.Array] | None = None,
    newton_method: str = "hand",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply the stabilised coupled operator to ``(u, p)``.

    The pressure variable here is the *physical* pressure. We achieve
    symmetric saddle-point structure by flipping the sign of the
    divergence equation (since ``D = -G^T``, this gives ``-D = G^T`` =
    the gradient transpose). No additional scaling on the off-diagonal
    blocks.

        out_x = (γ/dt) M ux + ν A_sipg ux + M·N_x(u; u*)
                + τ_d (D_pen)_x + τ_c (C_pen)_x  + G_x p
        out_y = (γ/dt) M uy + ν A_sipg uy + M·N_y(u; u*)
                + τ_d (D_pen)_y + τ_c (C_pen)_y  + G_y p
        out_p = − D(ux, uy)  + pressure_eps · M p

    Velocity-block penalties (Fehn 2018) stabilise the inf-sup constant
    for equal-order DG.
    """
    M_ux = _mass_apply(mesh, ux, M_ref)
    M_uy = _mass_apply(mesh, uy, M_ref)
    visc_x = sipg_apply_2d(
        mesh, ux, bc_types=viscous_bc_types_sipg,
        tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d_per_face,
    )
    visc_y = sipg_apply_2d(
        mesh, uy, bc_types=viscous_bc_types_sipg,
        tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d_per_face,
    )
    # Picard linearisation at u_star: ∇·(u_star ⊗ u). For Newton, add the
    # extra Frechet term ∂_j(δu_j · u_k_i), either hand-derived or via
    # jax.jvp on the full nonlinear convection. For Newton, (ux, uy) play
    # the role of the increment δu; newton_u_k is the current outer iterate.
    conv_x, conv_y = _conv_apply_2d(
        mesh, ux, uy,
        u_star_x=u_star_x, u_star_y=u_star_y,
        ins_bc_types=ins_bc_types, M_ref=M_ref,
    )
    if newton_u_k is not None:
        u_k_x, u_k_y = newton_u_k
        if newton_method == "hand":
            extra_x, extra_y = _conv_apply_newton_extra_2d(
                mesh, ux, uy, u_k_x, u_k_y,
                ins_bc_types=ins_bc_types, M_ref=M_ref,
            )
        elif newton_method == "jvp":
            # J·δu via jax.jvp on the full nonlinear apply; subtract the
            # Picard piece to leave the Newton extra (so the assembly below
            # remains the simple ``Picard + extra`` structure).
            def _nonlin(u_pair):
                return _conv_apply_nonlinear_2d(
                    mesh, u_pair[0], u_pair[1],
                    ins_bc_types=ins_bc_types, M_ref=M_ref,
                )
            full_x, full_y = jax.jvp(_nonlin, ((u_k_x, u_k_y),), ((ux, uy),))[1]
            # Picard part at u_k along δu (homog δu BC).
            picard_x_at_uk, picard_y_at_uk = _conv_apply_2d(
                mesh, ux, uy,
                u_star_x=u_k_x, u_star_y=u_k_y,
                ins_bc_types=ins_bc_types, M_ref=M_ref,
            )
            extra_x = full_x - picard_x_at_uk
            extra_y = full_y - picard_y_at_uk
        else:
            raise ValueError(f"unknown newton_method {newton_method!r}")
        conv_x = conv_x + extra_x
        conv_y = conv_y + extra_y
    Gx_p, Gy_p = _grad_apply_2d(
        mesh, p, ins_bc_types=ins_bc_types, M_ref=M_ref,
    )
    Dp = _div_apply_2d(
        mesh, ux, uy, ins_bc_types=ins_bc_types, M_ref=M_ref,
    )

    out_x = gamma0_over_dt * M_ux + nu * visc_x + conv_x + Gx_p
    out_y = gamma0_over_dt * M_uy + nu * visc_y + conv_y + Gy_p

    if tau_div != 0.0:
        dpx, dpy = divergence_penalty_apply_2d(mesh, ux, uy)
        out_x = out_x + tau_div * dpx
        out_y = out_y + tau_div * dpy
    if tau_cont != 0.0:
        cpx, cpy = continuity_penalty_apply_2d(mesh, ux, uy)
        out_x = out_x + tau_cont * cpx
        out_y = out_y + tau_cont * cpy

    # Sign-flipped divergence equation. Brezzi-Pitkäranta-style mass
    # stabilisation of the pressure subtracted (matches the sign flip).
    out_p = -Dp
    if pressure_eps != 0.0:
        out_p = out_p - pressure_eps * _mass_apply(mesh, p, M_ref)
    return out_x, out_y, out_p


def coupled_ns_matrix_2d(
    mesh: Mesh2D,
    *,
    nu: float,
    gamma0_over_dt: float,
    u_star_x: jax.Array,
    u_star_y: jax.Array,
    ins_bc_types: jax.Array,
    tau_scale: float = 200.0,
    tau_div: float = 0.0,
    tau_cont: float = 0.0,
    pressure_eps: float = 0.0,
) -> jax.Array:
    """Assemble the dense ``(3 Np K) × (3 Np K)`` coupled NS matrix.

    The matrix layout is column-major flat per field, stacked
    ``[ux; uy; p]``.
    """
    Np, K = mesh.Np, mesh.K
    M_ref = _local_mass(mesh)
    m1d_per_face = _face_1d_mass_matrices(mesh)
    viscous_bc_sipg = viscous_bc_types_from_ins(ins_bc_types)

    def op_flat(v: jax.Array) -> jax.Array:
        ux, uy, p = _unflat3(v, Np, K)
        out_x, out_y, out_p = coupled_ns_apply_2d(
            mesh, ux, uy, p,
            nu=nu, gamma0_over_dt=gamma0_over_dt,
            u_star_x=u_star_x, u_star_y=u_star_y,
            ins_bc_types=ins_bc_types,
            viscous_bc_types_sipg=viscous_bc_sipg,
            M_ref=M_ref, m1d_per_face=m1d_per_face,
            tau_scale=tau_scale,
            tau_div=tau_div, tau_cont=tau_cont,
            pressure_eps=pressure_eps,
        )
        return _flat3(out_x, out_y, out_p)

    n_dof = 3 * Np * K
    return jax.jacobian(op_flat)(jnp.zeros(n_dof))


def factor_coupled_ns_2d(
    mesh: Mesh2D,
    *,
    nu: float,
    gamma0_over_dt: float,
    u_star_x: jax.Array,
    u_star_y: jax.Array,
    ins_bc_types: jax.Array,
    tau_scale: float = 200.0,
    tau_div: float = 0.0,
    tau_cont: float = 0.0,
    pressure_eps: float = 0.0,
) -> CoupledNSOperators:
    """Assemble + LU-factor the coupled system for one ``(dt, γ, u*)`` setup.

    Re-call this whenever ``u*`` changes (every step for linearly-implicit
    convection) — the matrix isn't time-invariant.

    Set ``tau_div``, ``tau_cont`` > 0 for inf-sup stabilisation (Fehn
    2018). Sensible scaling from dimensional analysis:
    ``τ_d, τ_c ~ O(h · |u*|)`` with ``O(1)`` constants.
    """
    A = coupled_ns_matrix_2d(
        mesh, nu=nu, gamma0_over_dt=gamma0_over_dt,
        u_star_x=u_star_x, u_star_y=u_star_y,
        ins_bc_types=ins_bc_types, tau_scale=tau_scale,
        tau_div=tau_div, tau_cont=tau_cont,
        pressure_eps=pressure_eps,
    )
    lu, piv = jax.scipy.linalg.lu_factor(A)
    return CoupledNSOperators(
        A_full=A, lu=lu, piv=piv,
        viscous_bc_types=viscous_bc_types_from_ins(ins_bc_types),
        pressure_eps=pressure_eps,
    )


# ---------------------------------------------------------------------------
# Right-hand side + step driver
# ---------------------------------------------------------------------------

def _coupled_ns_rhs_2d(
    mesh: Mesh2D,
    *,
    nu: float,
    dt: float,
    bdf: BDFCoeffs,
    ux_n: jax.Array, uy_n: jax.Array,
    ux_old: jax.Array, uy_old: jax.Array,
    ins_bc_types: jax.Array,
    bc_ux: jax.Array, bc_uy: jax.Array,
    p_dir: jax.Array | None,
    q_ux_neu: jax.Array | None,
    q_uy_neu: jax.Array | None,
    M_ref: jax.Array,
    m1d_per_face: jax.Array,
    viscous_bc_types_sipg: jax.Array,
    tau_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the (ux, uy, p) right-hand side fields for the coupled step.

    Matches the sign-flipped saddle-point: divergence equation
    multiplied by ``-1`` for symmetry; no other scaling on the
    off-diagonal blocks (pressure variable is physical).
    """
    u_hist_x = (bdf.a0 * ux_n + bdf.a1 * ux_old) / dt
    u_hist_y = (bdf.a0 * uy_n + bdf.a1 * uy_old) / dt
    rhs_x = _mass_apply(mesh, u_hist_x, M_ref)
    rhs_y = _mass_apply(mesh, u_hist_y, M_ref)

    # Viscous Dirichlet load (u_BC weakly imposed) — subtracted because
    # the SIPG framework's convention is A·u = M·f − L_D + L_N.
    visc_load_x = sipg_dirichlet_load_2d(
        mesh, bc_ux, bc_types=viscous_bc_types_sipg, tau_scale=tau_scale,
    )
    visc_load_y = sipg_dirichlet_load_2d(
        mesh, bc_uy, bc_types=viscous_bc_types_sipg, tau_scale=tau_scale,
    )
    rhs_x = rhs_x - nu * visc_load_x
    rhs_y = rhs_y - nu * visc_load_y
    # Viscous Neumann load (non-homog ∂u/∂n at outflow): added.
    if q_ux_neu is not None:
        rhs_x = rhs_x + nu * sipg_neumann_load_2d(
            mesh, q_ux_neu, bc_types=viscous_bc_types_sipg,
        )
    if q_uy_neu is not None:
        rhs_y = rhs_y + nu * sipg_neumann_load_2d(
            mesh, q_uy_neu, bc_types=viscous_bc_types_sipg,
        )

    # Pressure-Dirichlet load → velocity equations via G.
    if p_dir is not None:
        Gx_p_dir, Gy_p_dir = _grad_apply_2d(
            mesh, jnp.zeros_like(p_dir), ins_bc_types=ins_bc_types,
            M_ref=M_ref, p_dir=p_dir,
        )
        rhs_x = rhs_x - Gx_p_dir
        rhs_y = rhs_y - Gy_p_dir

    # Velocity-Dirichlet load on the divergence equation, then sign-flipped.
    Dp_load = _div_apply_2d(
        mesh, jnp.zeros_like(ux_n), jnp.zeros_like(uy_n),
        ins_bc_types=ins_bc_types, M_ref=M_ref,
        bc_ux=bc_ux, bc_uy=bc_uy,
    )
    rhs_p = Dp_load
    return rhs_x, rhs_y, rhs_p


def make_velocity_lu_preconditioner(
    mesh: Mesh2D,
    *,
    alpha: float,
    nu: float,
    bc_types: jax.Array,
    tau_scale: float = 200.0,
):
    """Direct-LU preconditioner for the velocity Helmholtz block.

    Returns a callable ``M_inv(v_flat) -> v_flat`` suitable as the
    ``velocity_preconditioner`` argument of :func:`coupled_ns_step_2d`.
    Much stronger than :func:`ddg.dg2d.sipg.make_helmholtz_block_jacobi_preconditioner`
    because it includes element-element coupling -- per-element
    block-Jacobi is a poor approximation of the SIPG-coupled Helmholtz
    once face flux contributions matter (i.e. moderately fine meshes
    or moderately strong convection).

    Pays an ``O(n^3)`` one-time factorisation and ``O(n^2)`` per
    preconditioner apply, where ``n = N_p * K`` is the velocity-block
    dimension. Worth it whenever the velocity Helmholtz dominates the
    saddle-point spectrum -- the typical case at ``alpha = gamma /
    (nu * dt)``. Note the convective term is *not* in this PC; with
    Newton or outer Picard the convective contribution is recovered
    by the outer iteration anyway.

    The same factor preconditions both the x- and y-momentum blocks
    (they share the same Helmholtz operator).
    """
    _, lu, piv = factor_sipg_helmholtz_2d(
        mesh, alpha=alpha, nu=nu, bc_types=bc_types, tau_scale=tau_scale,
    )

    def M_inv(v_flat):
        return jax.scipy.linalg.lu_solve((lu, piv), v_flat)

    return M_inv


def make_pressure_laplacian_inverse(
    mesh: Mesh2D,
    *,
    p_bc_types: jax.Array,
    scale: float = 1.0,
    tau_scale: float = 200.0,
    regularize: bool = True,
    eps: float = 1.0e-8,
):
    """Pressure-Laplacian inverse as a Schur preconditioner -- the
    *unsteady* (small-Δt) limit of the Cahouet--Chabard approximation.

    The Schur complement for the coupled NS saddle-point is
    ``S = D · A^{-1} · G - ε·M_p`` where ``A = (γ/Δt) M_v + ν A_v^{SIPG} +
    convective``. Two asymptotic regimes give clean approximations:

      * **Viscous limit** (large Δt, ``ν·A_v`` dominates ``A``):
        ``S^{-1} ≈ ν^{-1} M_p^{-1}`` -- the
        :func:`make_pressure_mass_inverse` (Cahouet--Chabard) PC.
      * **Unsteady limit** (small Δt, ``(γ/Δt) M_v`` dominates ``A``):
        ``S^{-1} ≈ (Δt/γ) K_p^{-1}`` where ``K_p`` is the *pressure*
        Laplacian -- this function. Pass ``scale = Δt/γ``.

    For our typical coupled runs at ``α = γ/(ν Δt) ≫ 1`` the unsteady
    limit dominates and this PC is the right one. Use both in
    combination (``S^{-1} ≈ ν^{-1} M_p^{-1} + (Δt/γ) K_p^{-1}``) for a
    full Cahouet--Chabard if the regime is in between.

    The pressure-Poisson is singular when no pressure-Dirichlet face is
    present (closed velocity-Dirichlet box, all-periodic, etc.).
    ``regularize=True`` lifts the constant nullspace by an ε-shift on the
    diagonal so the LU factor exists and is well-conditioned.
    """
    _, lu, piv = factor_sipg_2d(
        mesh, bc_types=p_bc_types, tau_scale=tau_scale,
        regularize=regularize, eps=eps,
    )

    def M_inv(v_flat):
        return scale * jax.scipy.linalg.lu_solve((lu, piv), v_flat)

    return M_inv


def make_cahouet_chabard_full_inverse(
    mesh: Mesh2D,
    *,
    nu: float,
    dt: float,
    gamma0: float,
    p_bc_types: jax.Array,
    tau_scale: float = 200.0,
    regularize: bool | None = None,
    eps: float = 1.0e-8,
    mass_scale: float = 1.0,
    laplacian_scale: float = 1.0,
):
    """Full Cahouet--Chabard Schur preconditioner:

        S^{-1} \\approx  \\nu^{-1} M_p^{-1}  +  (\\Delta t / \\gamma_0) K_p^{-1}

    The two terms cover the *viscous* (large ``Δt``) and *unsteady*
    (small ``Δt``) asymptotic limits respectively. Combining them
    (rather than picking one) removes the regime-mismatch sensitivity
    that the single-term :func:`make_pressure_mass_inverse` and
    :func:`make_pressure_laplacian_inverse` PCs suffer from, and is
    the form used in production-grade coupled NS codes (exadg /
    Kay-Loghin-Wathen).

    The pressure-Laplacian factor is built via
    :func:`factor_sipg_2d`. ``regularize`` lifts the constant
    nullspace when the pressure Poisson is singular (no
    pressure-Dirichlet face); by default we auto-detect from
    ``p_bc_types``.

    ``mass_scale`` / ``laplacian_scale`` are diagnostic knobs for
    tuning the relative weight of the two terms; leave at 1.0 in
    production.
    """
    if regularize is None:
        regularize = not bool(jnp.any(p_bc_types == BC_DIRICHLET))
    m_inv = make_pressure_mass_inverse(mesh, nu=nu, scale=mass_scale)
    k_inv = make_pressure_laplacian_inverse(
        mesh, p_bc_types=p_bc_types,
        scale=laplacian_scale * dt / gamma0,
        tau_scale=tau_scale, regularize=regularize, eps=eps,
    )

    def S_inv(v_flat):
        return m_inv(v_flat) + k_inv(v_flat)

    return S_inv


def make_block_triangular_pc(
    mesh: Mesh2D,
    *,
    velocity_pc: "Callable[[jax.Array], jax.Array]",
    pressure_schur_pc: "Callable[[jax.Array], jax.Array]",
    ins_bc_types: jax.Array,
    M_ref: jax.Array | None = None,
):
    """Silvester--Wathen block upper-triangular preconditioner for the
    coupled NS saddle-point.

    For the linearised system

        [  A    G   ] [u]   [b_u]
        [ -D   epsM ] [p] = [b_p]

    this preconditioner approximates the inverse via the block-upper-
    triangular factor ``P_U = [A, G; 0, S]`` and applies it as

        p_out = pressure_schur_pc(b_p)
        u_out = velocity_pc(b_u - G * p_out)
        v_out = velocity_pc(b_v - G * p_out)

    The off-diagonal coupling that block-diagonal PCs ignore is here
    folded into the velocity solve. For *symmetric* saddle-points the
    block-diagonal Murphy--Wathen PC is theoretically optimal, but
    once the convective term enters the (0,0) block (Picard at
    ``u_star = u_k``, or full Newton) the system is no longer
    symmetric and block-triangular becomes the more robust choice --
    this is the structure used by the exadg / Fehn coupled NS solver.

    Pass the returned callable via the ``coupled_preconditioner=``
    argument of :func:`coupled_ns_step_2d`; it supersedes the
    component-wise ``velocity_preconditioner`` /
    ``pressure_mass_inv`` arguments which build a block-diagonal PC.
    """
    if M_ref is None:
        M_ref = _local_mass(mesh)
    Np, K = mesh.Np, mesh.K
    n = Np * K

    def M_inv(v_flat):
        v_ux = v_flat[:n]
        v_uy = v_flat[n:2 * n]
        v_p_flat = v_flat[2 * n:]
        # Solve Schur first.
        p_out_flat = pressure_schur_pc(v_p_flat)
        p_out = p_out_flat.reshape(K, Np).T
        # Subtract G * p from the velocity RHS, then velocity solve.
        Gx_p, Gy_p = _grad_apply_2d(
            mesh, p_out, ins_bc_types=ins_bc_types, M_ref=M_ref,
        )
        v_u_adj = v_ux - Gx_p.T.reshape(-1)
        v_v_adj = v_uy - Gy_p.T.reshape(-1)
        u_out = velocity_pc(v_u_adj)
        v_out = velocity_pc(v_v_adj)
        return jnp.concatenate([u_out, v_out, p_out_flat])

    return M_inv


def make_pressure_mass_inverse(
    mesh: Mesh2D,
    *,
    nu: float,
    scale: float = 1.0,
):
    """Cheap Schur-complement approximation: ``scale · (1/ν) M_p^{-1}``.

    The pressure mass matrix in our DG discretisation is element-local
    block-diagonal ``J_K M_ref``; its inverse is one ``Np × Np`` solve
    per element. This is the classical Cahouet-Chabard
    (1988) cheap approximation to the pressure Schur complement
    ``S^{-1} ≈ (1/ν) M_p^{-1}`` for incompressible NS.

    Pass the returned callable as ``pressure_mass_inv`` to
    :func:`coupled_ns_step_2d` in iterative mode.
    """
    M_ref = _local_mass(mesh)              # (Np, Np)
    M_ref_inv = jnp.linalg.inv(M_ref)      # (Np, Np)
    Np, K = mesh.Np, mesh.K
    # Per-element scaling: 1 / (nu * J_K). For affine elements J is
    # constant per element; sample at node 0.
    J_K = mesh.J[0]                        # (K,)
    factor = scale / (nu * J_K)            # (K,)

    def M_inv(v_flat):
        v_per_elem = v_flat.reshape(K, Np)               # (K, Np)
        out = jnp.einsum("ij,kj->ki", M_ref_inv, v_per_elem)
        return (factor[:, None] * out).reshape(-1)

    return M_inv


def coupled_ns_step_2d(
    mesh: Mesh2D,
    ux_n: jax.Array, uy_n: jax.Array,
    ux_old: jax.Array, uy_old: jax.Array,
    *,
    dt: float,
    nu: float,
    bdf: BDFCoeffs,
    ins_bc_types: jax.Array,
    bc_ux: jax.Array, bc_uy: jax.Array,
    p_dir: jax.Array | None = None,
    q_ux_neu: jax.Array | None = None,
    q_uy_neu: jax.Array | None = None,
    tau_scale: float = 200.0,
    tau_div: float = 0.0,
    tau_cont: float = 0.0,
    pressure_eps: float = 0.0,
    ops: CoupledNSOperators | None = None,
    iterative: IterativeSolverConfig | None = None,
    x0_iterative: tuple | None = None,
    velocity_preconditioner=None,
    pressure_mass_inv=None,
    coupled_preconditioner=None,
    nl_config: NonlinearSolverConfig | None = None,
    nl_residual_history_out: list | None = None,
    convergence_out: CoupledNSConvergenceReport | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, CoupledNSOperators | None]:
    """One monolithic coupled NS step. Returns ``(ux, uy, p, ops_used)``.

    The convective term is linearised by freezing the transport
    velocity to the BDF-extrapolated value ``u* = β₀ u_n + β₁ u_{n−1}``.

    Two solver modes:

      * **Direct LU** (``iterative=None``, default). Assembles the
        coupled matrix, LU-factors it, and back-solves. ``ops``
        skips re-factorisation across calls if non-None.

      * **Matrix-free iterative** (``iterative=IterativeSolverConfig(...)``).
        Calls ``jax.scipy.sparse.linalg.gmres`` (or ``bicgstab``) on a
        closure over the apply function — no dense matrix is ever
        materialised. Critical for large problems / GPU since LU
        cost is ``O(n³)`` and memory is ``O(n²)``. Returns ``ops=None``
        in this mode.

    ``x0_iterative``: optional warm-start ``(ux_init, uy_init, p_init)``
    for the Krylov solver (usually the previous step's solution).
    """
    Np, K = mesh.Np, mesh.K
    M_ref = _local_mass(mesh)
    m1d_per_face = _face_1d_mass_matrices(mesh)
    viscous_bc_sipg = viscous_bc_types_from_ins(ins_bc_types)

    if bdf is BDF1:
        u_star_x = ux_n
        u_star_y = uy_n
    else:
        u_star_x = 2.0 * ux_n - ux_old
        u_star_y = 2.0 * uy_n - uy_old

    rhs_x, rhs_y, rhs_p = _coupled_ns_rhs_2d(
        mesh, nu=nu, dt=dt, bdf=bdf,
        ux_n=ux_n, uy_n=uy_n, ux_old=ux_old, uy_old=uy_old,
        ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
        p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
        M_ref=M_ref, m1d_per_face=m1d_per_face,
        viscous_bc_types_sipg=viscous_bc_sipg,
        tau_scale=tau_scale,
    )
    rhs_flat = _flat3(rhs_x, rhs_y, rhs_p)

    if iterative is None:
        if ops is None:
            ops = factor_coupled_ns_2d(
                mesh, nu=nu, gamma0_over_dt=bdf.g0 / dt,
                u_star_x=u_star_x, u_star_y=u_star_y,
                ins_bc_types=ins_bc_types,
                tau_scale=tau_scale,
                tau_div=tau_div, tau_cont=tau_cont,
                pressure_eps=pressure_eps,
            )
        sol = jax.scipy.linalg.lu_solve((ops.lu, ops.piv), rhs_flat)
        ux_new, uy_new, p_new = _unflat3(sol, Np, K)
        return ux_new, uy_new, p_new, ops

    # Matrix-free iterative path.
    def matvec(v: jax.Array) -> jax.Array:
        ux, uy, p = _unflat3(v, Np, K)
        out_x, out_y, out_p = coupled_ns_apply_2d(
            mesh, ux, uy, p,
            nu=nu, gamma0_over_dt=bdf.g0 / dt,
            u_star_x=u_star_x, u_star_y=u_star_y,
            ins_bc_types=ins_bc_types,
            viscous_bc_types_sipg=viscous_bc_sipg,
            M_ref=M_ref, m1d_per_face=m1d_per_face,
            tau_scale=tau_scale,
            tau_div=tau_div, tau_cont=tau_cont,
            pressure_eps=pressure_eps,
        )
        return _flat3(out_x, out_y, out_p)

    if x0_iterative is not None:
        x0 = _flat3(*x0_iterative)
    else:
        x0 = jnp.concatenate([
            ux_n.T.reshape(-1), uy_n.T.reshape(-1),
            jnp.zeros(Np * K),  # zero pressure guess
        ])

    # Block-diagonal preconditioner. The coupled system is the saddle-point
    #   [ A   G  ] [u]   [f_u]
    #   [ D   εI ] [p] = [f_p]
    # with A the velocity Helmholtz block, G/D the (sign-flipped) grad/div
    # blocks, ε a small pressure-stabilisation. The classical block-diag
    # PC is diag(A^{-1}, S^{-1}) where the Schur complement is
    #   S = D A^{-1} G + εI ≈ M_p / nu  (Cahouet-Chabard 1988).
    # We accept the velocity Helmholtz PC and pressure-mass inverse from
    # the caller (typically block-Jacobi for each) and bundle them.
    # Three PC modes:
    #   - coupled_preconditioner explicitly given  -> use it as-is
    #     (e.g. make_block_triangular_pc; supersedes component-wise args)
    #   - velocity_preconditioner / pressure_mass_inv given -> build a
    #     block-DIAGONAL PC (the classical Murphy-Wathen structure for
    #     symmetric indefinite saddle-points)
    #   - neither -> unpreconditioned GMRES (typically too slow)
    if coupled_preconditioner is not None:
        M = coupled_preconditioner
    elif velocity_preconditioner is not None or pressure_mass_inv is not None:
        n = Np * K

        def M_inv(v):
            v_ux = v[:n]
            v_uy = v[n:2 * n]
            v_p  = v[2 * n:]
            if velocity_preconditioner is not None:
                v_ux = velocity_preconditioner(v_ux)
                v_uy = velocity_preconditioner(v_uy)
            if pressure_mass_inv is not None:
                v_p = pressure_mass_inv(v_p)
            return jnp.concatenate([v_ux, v_uy, v_p])
        M = M_inv
    else:
        M = None

    def _solve_and_record(matvec_fn, b, x0_):
        """Run the chosen Krylov method on ``matvec_fn x = b``, with x0_.
        Returns x. Populates convergence_out (if provided) with the
        initial and achieved residual + a converged flag based on the
        achieved-vs-tol check (so callers don't have to interpret the
        JAX-internal ``info`` integer)."""
        if iterative.method == "gmres":
            x, _ = jax.scipy.sparse.linalg.gmres(
                matvec_fn, b, x0=x0_,
                tol=iterative.tol, atol=iterative.atol,
                maxiter=iterative.maxiter, restart=iterative.restart,
                M=M,
            )
        elif iterative.method == "bicgstab":
            x, _ = jax.scipy.sparse.linalg.bicgstab(
                matvec_fn, b, x0=x0_,
                tol=iterative.tol, atol=iterative.atol,
                maxiter=iterative.maxiter,
                M=M,
            )
        else:
            raise ValueError(f"unknown iterative method {iterative.method!r}")
        if convergence_out is not None:
            r0 = float(jnp.linalg.norm(b - matvec_fn(x0_)))
            r1 = float(jnp.linalg.norm(b - matvec_fn(x)))
            target = iterative.tol * r0 + iterative.atol
            convergence_out.inner_residuals_initial.append(r0)
            convergence_out.inner_residuals_achieved.append(r1)
            convergence_out.inner_converged.append(bool(r1 <= target * 1.1))
        return x

    if nl_config is None:
        # Default: one Picard sweep with u_star = BDF extrapolation.
        sol = _solve_and_record(matvec, rhs_flat, x0)
        ux_new, uy_new, p_new = _unflat3(sol, Np, K)
        return ux_new, uy_new, p_new, None

    # Outer nonlinear loop (Picard or Newton, both in δ-form).
    if nl_config.linearisation not in ("picard", "newton"):
        raise ValueError(
            f"unknown linearisation {nl_config.linearisation!r}; "
            "expected 'picard' or 'newton'"
        )
    use_newton = nl_config.linearisation == "newton"

    u_k_x, u_k_y = u_star_x, u_star_y
    p_k = jnp.zeros((Np, K))
    if x0_iterative is not None:
        u_k_x, u_k_y, p_k = x0_iterative

    def _full_apply_at(u_x_, u_y_, p_):
        return coupled_ns_apply_2d(
            mesh, u_x_, u_y_, p_,
            nu=nu, gamma0_over_dt=bdf.g0 / dt,
            u_star_x=u_x_, u_star_y=u_y_,             # nonlinear: u_star = u
            ins_bc_types=ins_bc_types,
            viscous_bc_types_sipg=viscous_bc_sipg,
            M_ref=M_ref, m1d_per_face=m1d_per_face,
            tau_scale=tau_scale,
            tau_div=tau_div, tau_cont=tau_cont,
            pressure_eps=pressure_eps,
        )

    norm_F0 = None
    for k in range(nl_config.maxiter):
        F_u_x, F_u_y, F_p_minus = _full_apply_at(u_k_x, u_k_y, p_k)
        F_x = F_u_x - rhs_x
        F_y = F_u_y - rhs_y
        F_p = F_p_minus - rhs_p
        F_flat = _flat3(F_x, F_y, F_p)
        norm_F = jnp.linalg.norm(F_flat)
        if nl_residual_history_out is not None:
            nl_residual_history_out.append(float(norm_F))
        if convergence_out is not None:
            convergence_out.outer_residuals.append(float(norm_F))
        if norm_F0 is None:
            norm_F0 = norm_F
        if bool(norm_F <= nl_config.tol * norm_F0 + nl_config.atol):
            break

        newton_u_k = (u_k_x, u_k_y) if use_newton else None
        def J_matvec(v):
            ux_, uy_, p_ = _unflat3(v, Np, K)
            out_x, out_y, out_p = coupled_ns_apply_2d(
                mesh, ux_, uy_, p_,
                nu=nu, gamma0_over_dt=bdf.g0 / dt,
                u_star_x=u_k_x, u_star_y=u_k_y,
                ins_bc_types=ins_bc_types,
                viscous_bc_types_sipg=viscous_bc_sipg,
                M_ref=M_ref, m1d_per_face=m1d_per_face,
                tau_scale=tau_scale,
                tau_div=tau_div, tau_cont=tau_cont,
                pressure_eps=pressure_eps,
                newton_u_k=newton_u_k,
                newton_method=nl_config.newton_method,
            )
            return _flat3(out_x, out_y, out_p)

        delta_flat = _solve_and_record(J_matvec, -F_flat,
                                        jnp.zeros_like(F_flat))
        delta_x, delta_y, delta_p = _unflat3(delta_flat, Np, K)
        u_k_x = u_k_x + delta_x
        u_k_y = u_k_y + delta_y
        p_k = p_k + delta_p

    if convergence_out is not None and norm_F0 is not None:
        target = nl_config.tol * norm_F0 + nl_config.atol
        convergence_out.outer_converged = bool(norm_F <= target)
    return u_k_x, u_k_y, p_k, None
