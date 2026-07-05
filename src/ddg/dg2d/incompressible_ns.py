"""2D incompressible Navier-Stokes in primitive variables (velocity-pressure)
via Hesthaven's dual-splitting scheme.

Each time step is operator-split into three pieces:

  1. **Advection** (explicit). Evaluate ``N(U) = ∇·(U ⊗ U)`` (the
     nonlinear convective term) with a local-Lax-Friedrichs (Rusanov)
     surface flux. Extrapolate from previous steps using BDF1/BDF2
     coefficients to obtain ``Ũ``:

         g0 Ũ = a0 U + a1 U_old − dt (b0 N(U) + b1 N(U_old))

     BDF1 (first step): g0=1, a0=1, a1=0, b0=1, b1=0.
     BDF2 (subsequent): g0=3/2, a0=2, a1=-1/2, b0=2, b1=-1.

  2. **Pressure**. Solve

         −Δ p = (g0/dt) ∇·Ũ

     with mixed BCs: Dirichlet ``p = 0`` on outflow; consistent Neumann
     ``∂p/∂n = −n·(N(U) + ν ∇×∇×U)`` on inflow / walls / obstacle
     (matches Hesthaven's INSPressure2D.m). Then update:

         Ũ̃ = Ũ − (dt/g0) ∇p

  3. **Viscous**. Solve a Helmholtz system per velocity component:

         (g0/(ν dt)) M U_new − Δ U_new = (g0/(ν dt)) M Ũ̃

     i.e. ``α M − Δ`` with ``α = g0/(ν dt)``. Mixed BCs flip: Dirichlet
     on inflow / walls / obstacle (no-slip / inflow profile); Neumann
     ``∂U/∂n = 0`` on outflow.

Boundary tagging is per-face via integer codes (:data:`BC_INS_INTERIOR`
/ :data:`BC_INS_INFLOW` / :data:`BC_INS_OUTFLOW` / :data:`BC_INS_WALL`
/ :data:`BC_INS_OBSTACLE`) in a ``(K, Nfaces)`` array. The mapping from
INS BC types to SIPG Dirichlet/Neumann codes for the pressure and
viscous solves is built by :func:`pressure_bc_types_from_ins` and
:func:`viscous_bc_types_from_ins`.

Port of ``reference/nodal-dg/Codes1.1/CFD2D/{CurvedINS2D, INSAdvection2D,
INSPressure2D, CurvedINSViscous2D}.m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.maxwell import curl_z_2d, grad_2d
from ddg.dg2d.mesh import Mesh2D
from ddg.dg2d.sipg import (
    BC_DIRICHLET, BC_INTERIOR, BC_NEUMANN, IterativeConfig,
    _face_1d_mass_matrices, _local_mass,
    factor_sipg_2d, factor_sipg_helmholtz_2d,
    sipg_apply_2d, sipg_dirichlet_load_2d, sipg_neumann_load_2d,
)
from ddg.dg2d.multigrid import make_sipg_mg_preconditioner


# ---------------------------------------------------------------------------
# BC type codes
# ---------------------------------------------------------------------------
BC_INS_INTERIOR = 0
BC_INS_INFLOW = 1
BC_INS_OUTFLOW = 2
BC_INS_WALL = 3
BC_INS_OBSTACLE = 4


def pressure_bc_types_from_ins(ins_bc_types: jax.Array) -> jax.Array:
    """Inflow / wall / obstacle → Neumann; outflow → Dirichlet."""
    is_neu = (
        (ins_bc_types == BC_INS_INFLOW)
        | (ins_bc_types == BC_INS_WALL)
        | (ins_bc_types == BC_INS_OBSTACLE)
    )
    is_dir = ins_bc_types == BC_INS_OUTFLOW
    return jnp.where(is_dir, BC_DIRICHLET, jnp.where(is_neu, BC_NEUMANN, BC_INTERIOR))


def viscous_bc_types_from_ins(ins_bc_types: jax.Array) -> jax.Array:
    """Inflow / wall / obstacle → Dirichlet; outflow → Neumann."""
    is_dir = (
        (ins_bc_types == BC_INS_INFLOW)
        | (ins_bc_types == BC_INS_WALL)
        | (ins_bc_types == BC_INS_OBSTACLE)
    )
    is_neu = ins_bc_types == BC_INS_OUTFLOW
    return jnp.where(is_dir, BC_DIRICHLET, jnp.where(is_neu, BC_NEUMANN, BC_INTERIOR))


# ---------------------------------------------------------------------------
# Volume divergence and physical-gradient operators
# ---------------------------------------------------------------------------

def _div_2d(mesh: Mesh2D, fx: jax.Array, fy: jax.Array) -> jax.Array:
    """Strong-form divergence ``∂fx/∂x + ∂fy/∂y`` on each element."""
    fxr = mesh.Dr @ fx
    fxs = mesh.Ds @ fx
    fyr = mesh.Dr @ fy
    fys = mesh.Ds @ fy
    return (
        mesh.rx * fxr + mesh.sx * fxs
        + mesh.ry * fyr + mesh.sy * fys
    )


# ---------------------------------------------------------------------------
# Advection step: N(U) = ∇·(U⊗U) with Lax-Friedrichs flux
# ---------------------------------------------------------------------------

def ns_advection_2d(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    *,
    ins_bc_types: jax.Array,
    bc_ux: jax.Array | None = None,
    bc_uy: jax.Array | None = None,
    velocity_reconstruction=None,
    convective_form: str = "conservative",
) -> tuple[jax.Array, jax.Array]:
    """Compute the nonlinear advection ``(N_x, N_y) = ∇·(U⊗U)``.

    Volume term: weak ``∇·`` (here in strong form). Surface term: local
    Lax-Friedrichs flux ``F* = 0.5(F_M + F_P) − 0.5 |λ_max| (U_P − U_M)``
    with ``λ_max = max(|U·n|_M, |U·n|_P)``.

    ``bc_ux``, ``bc_uy`` are volume-shaped ``(Np, K)`` velocity BC
    fields; only boundary-face traces matter. On Inflow / Wall /
    Obstacle faces, the ``+`` trace of velocity is replaced by these BC
    values (sets the upwind/Rusanov flux to the prescribed velocity).
    On Outflow / interior faces the ``+`` trace is the usual ``vmapP``
    lookup.

    ``velocity_reconstruction`` (optional): callable
    ``f(mesh, ux, uy, *, bc_normal_overrides=None) -> (ux_rec, uy_rec)``
    applied to the velocity before computing N(u). When given (e.g.
    :func:`ddg.dg2d.velocity_reconstruction.reconstruct_velocity_rt0_2d`),
    yields a pressure-robust convective term per Linke 2014 / Fehn 2025
    §4.7. Default ``None`` leaves the original behaviour.

    ``convective_form`` chooses the volume-term form:

      * ``"conservative"`` (default): ``N = ∇·(U⊗U)``. Standard for
        DG; what every L²-DG NS paper uses by default.

      * ``"skew_symmetric"``: ``N = ½[∇·(U⊗U) + (U·∇) U]``. Discretely
        energy-conserving for the convective term at the volume level
        (Gassner 2013 for compressible NS, applied to incompressible
        by analogy — Fehn 2025 Remark 4.14 notes this direction is
        "surprisingly under-explored" for incompressible). The surface
        flux stays Lax-Friedrichs; a fully entropy-conservative version
        would need a Tadmor / EC two-point face flux as well.
    """
    if velocity_reconstruction is not None:
        bc_pair = (bc_ux, bc_uy) if (bc_ux is not None and bc_uy is not None) else None
        ux, uy = velocity_reconstruction(mesh, ux, uy, bc_normal_overrides=bc_pair)

    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K

    # Volume term per chosen form.
    if convective_form == "conservative":
        # N = ∇·(U⊗U)
        Nux_vol = _div_2d(mesh, ux * ux, ux * uy)
        Nuy_vol = _div_2d(mesh, ux * uy, uy * uy)
    elif convective_form == "skew_symmetric":
        # N = 0.5 [∇·(U⊗U) + (U·∇) U]. The split between the conservative
        # and the convective form gives a volume operator V such that
        # ∫ U · V(U) dx = 0 on each element K — discrete energy
        # conservation of the volume contribution.
        div_uu_x = _div_2d(mesh, ux * ux, ux * uy)
        div_uu_y = _div_2d(mesh, ux * uy, uy * uy)
        # (U·∇) U component-wise: build per-node ∂ux/∂x, ∂ux/∂y, etc.
        ux_r = mesh.Dr @ ux
        ux_s = mesh.Ds @ ux
        uy_r = mesh.Dr @ uy
        uy_s = mesh.Ds @ uy
        dux_dx = mesh.rx * ux_r + mesh.sx * ux_s
        dux_dy = mesh.ry * ux_r + mesh.sy * ux_s
        duy_dx = mesh.rx * uy_r + mesh.sx * uy_s
        duy_dy = mesh.ry * uy_r + mesh.sy * uy_s
        adv_x = ux * dux_dx + uy * dux_dy
        adv_y = ux * duy_dx + uy * duy_dy
        Nux_vol = 0.5 * (div_uu_x + adv_x)
        Nuy_vol = 0.5 * (div_uu_y + adv_y)
    else:
        raise ValueError(
            f"convective_form must be 'conservative' or 'skew_symmetric', "
            f"got {convective_form!r}"
        )

    # Surface contributions
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)
    ux_M = ux_flat[mesh.vmapM]
    ux_P_raw = ux_flat[mesh.vmapP]
    uy_M = uy_flat[mesh.vmapM]
    uy_P_raw = uy_flat[mesh.vmapP]

    # Override + trace at inflow / wall / obstacle with BC values.
    bc_per_facenode = _ins_bc_facenodes_flat(mesh, ins_bc_types)
    use_bc = (
        (bc_per_facenode == BC_INS_INFLOW)
        | (bc_per_facenode == BC_INS_WALL)
        | (bc_per_facenode == BC_INS_OBSTACLE)
    )
    if bc_ux is not None:
        bc_ux_flat = bc_ux.T.reshape(-1)
        ux_P = jnp.where(use_bc, bc_ux_flat[mesh.vmapM], ux_P_raw)
    else:
        ux_P = jnp.where(use_bc, 0.0, ux_P_raw)
    if bc_uy is not None:
        bc_uy_flat = bc_uy.T.reshape(-1)
        uy_P = jnp.where(use_bc, bc_uy_flat[mesh.vmapM], uy_P_raw)
    else:
        uy_P = jnp.where(use_bc, 0.0, uy_P_raw)

    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")

    # Lax-Friedrichs flux:
    # flux = 0.5 (F_M + F_P) − 0.5 λ_max (U_P − U_M)
    # where F is the convective flux vector. Hesthaven's form jumps the
    # full F directly:
    #   flux_Ux = 0.5(−nx(uxux_M − uxux_P) − ny(uxuy_M − uxuy_P) − λmax(ux_P − ux_M))
    fxUxM = ux_M * ux_M
    fyUxM = ux_M * uy_M
    fxUyM = ux_M * uy_M
    fyUyM = uy_M * uy_M
    fxUxP = ux_P * ux_P
    fyUxP = ux_P * uy_P
    fxUyP = ux_P * uy_P
    fyUyP = uy_P * uy_P

    UDotN_M = ux_M * nx_flat + uy_M * ny_flat
    UDotN_P = ux_P * nx_flat + uy_P * ny_flat
    lam_max_node = jnp.maximum(jnp.abs(UDotN_M), jnp.abs(UDotN_P))
    # Hesthaven reduces λ_max over the Nfp nodes of each face (one
    # value per face), to avoid grossly under-dissipating at single
    # nodes. Mirror that:
    lam_face = lam_max_node.reshape((mesh.Nfp, mesh.Nfaces * K), order="F")
    lam_face_max = jnp.max(lam_face, axis=0, keepdims=True)
    lam_max = jnp.broadcast_to(lam_face_max, lam_face.shape).reshape(-1, order="F")

    flux_ux = 0.5 * (
        -nx_flat * (fxUxM - fxUxP)
        - ny_flat * (fyUxM - fyUxP)
        - lam_max * (ux_P - ux_M)
    )
    flux_uy = 0.5 * (
        -nx_flat * (fxUyM - fxUyP)
        - ny_flat * (fyUyM - fyUyP)
        - lam_max * (uy_P - uy_M)
    )

    flux_ux_face = flux_ux.reshape((Nfp_Nf, K), order="F")
    flux_uy_face = flux_uy.reshape((Nfp_Nf, K), order="F")

    Nux = Nux_vol + mesh.LIFT @ (mesh.Fscale * flux_ux_face)
    Nuy = Nuy_vol + mesh.LIFT @ (mesh.Fscale * flux_uy_face)
    return Nux, Nuy


def _ins_bc_facenodes_flat(
    mesh: Mesh2D, ins_bc_types: jax.Array
) -> jax.Array:
    """Expand ``(K, Nfaces)`` BC codes to ``(Nfp*Nfaces*K,)`` face-node flat."""
    bc = ins_bc_types.T.astype(jnp.int32)  # (Nfaces, K)
    bc_expanded = jnp.broadcast_to(
        bc[None, :, :], (mesh.Nfp, mesh.Nfaces, mesh.K)
    )
    return bc_expanded.reshape(-1, order="F")


# ---------------------------------------------------------------------------
# Pressure step
# ---------------------------------------------------------------------------

def ns_pressure_neumann_data(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    Nux: jax.Array,
    Nuy: jax.Array,
    *,
    nu: float,
) -> jax.Array:
    """Consistent Neumann data ``∂p/∂n = −n·(N(U) + ν ∇×∇×U)``.

    Returns face-trace shape ``(Nfp*Nfaces, K)`` with values at every
    face node (only flagged-Neumann faces are read downstream).

    Identity used: in 2D, ``∇·U = 0`` gives ``∇×∇×U = (−∂_y ω, ∂_x ω)``
    where ``ω = ∂_x u_y − ∂_y u_x``.
    """
    omega = curl_z_2d(mesh, ux, uy)
    dwdx, dwdy = grad_2d(mesh, omega)
    res_x = -Nux - nu * dwdy
    res_y = -Nuy + nu * dwdx
    res_x_flat = res_x.T.reshape(-1)
    res_y_flat = res_y.T.reshape(-1)
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    # Use the K-side trace of (res_x, res_y) for the boundary data.
    dpdn_flat = (
        nx_flat * res_x_flat[mesh.vmapM]
        + ny_flat * res_y_flat[mesh.vmapM]
    )
    return dpdn_flat.reshape((mesh.Nfp * mesh.Nfaces, mesh.K), order="F")


# ---------------------------------------------------------------------------
# BDF coefficients
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BDFCoeffs:
    g0: float
    a0: float
    a1: float
    b0: float
    b1: float


BDF1 = BDFCoeffs(g0=1.0, a0=1.0, a1=0.0, b0=1.0, b1=0.0)
BDF2 = BDFCoeffs(g0=1.5, a0=2.0, a1=-0.5, b0=2.0, b1=-1.0)


@dataclass(frozen=True)
class PenaltyConfig:
    """Parameters for the divergence + continuity penalty post-projection.

    ``tau_div`` / ``tau_cont`` may be **scalars** (uniform) or **per-element
    arrays of shape ``(K,)``** (dimensionally consistent scaling). Fehn,
    Kronbichler & Lube 2025, Remark 4.10 recommend the element-local
    scaling ``τ ~ O(h · |u*|)`` — see
    :func:`make_penalty_config_dimensional` for a helper that builds it.

    Used as a matrix-free alternative to
    :func:`factor_penalty_projection_2d`. Required at problem sizes past
    ~15k DOFs/field where the dense LU factor exceeds memory.
    """
    tau_div: float | jax.Array
    tau_cont: float | jax.Array
    tol: float = 1e-9
    atol: float = 0.0
    maxiter: int = 1000


def make_penalty_config_dimensional(
    mesh: Mesh2D,
    u_char: float | jax.Array = 1.0,
    *,
    scale_div: float = 1.0,
    scale_cont: float = 1.0,
    tol: float = 1e-9,
    maxiter: int = 1000,
) -> "PenaltyConfig":
    """Element-local penalty parameters ``τ_K = scale · h_K · |u*|_K``.

    Fehn / Kronbichler / Lube 2025 Remark 4.10: ``L²``-conforming
    penalty stabilisation needs ``τ`` of the right physical units
    (``length × velocity``) for the scheme to be uniformly robust
    across mesh sizes. A single constant ``τ`` (the current default
    of ``tau_div = tau_cont = 1``) over-stabilises at coarse mesh
    (the A_small ``dual_lu+pen`` blow-up in the audit) and
    under-stabilises at fine mesh.

    ``h_K`` is taken as the element-area length scale
    ``\\sqrt{2 J_K}`` (the reference triangle has area 2; physical
    element area is ``2 J_K`` for affine elements).

    ``u_char`` may be a scalar (e.g. ``U_inf``) or a ``(K,)`` array
    if a per-element characteristic velocity is available (typically
    the local ``max|u*|`` from the BDF history).
    """
    # Affine elements: J is constant per element; take node-0 sample.
    J_K = mesh.J[0]                                       # (K,)
    h_K = jnp.sqrt(2.0 * J_K)                             # (K,)
    # u_char may already be (K,), or scalar — broadcast against h_K.
    u_K = jnp.broadcast_to(jnp.asarray(u_char), h_K.shape)
    tau_K = h_K * u_K                                     # (K,)
    return PenaltyConfig(
        tau_div=scale_div * tau_K,
        tau_cont=scale_cont * tau_K,
        tol=tol, maxiter=maxiter,
    )


# ---------------------------------------------------------------------------
# INS step driver
# ---------------------------------------------------------------------------

@dataclass
class INSOperators:
    """Precomputed operators for the pressure and viscous steps.

    These are LU-factored matrices that depend on the time step size
    ``dt`` (the viscous Helmholtz parameter ``α = g0/(ν dt)``) and the
    BC topology, so they're rebuilt whenever ``dt`` or ``g0`` changes
    (BDF1 → BDF2 transition between steps 1 and 2 in particular).
    """
    pressure_factored: tuple
    viscous_factored: tuple
    pressure_bc_types: jax.Array
    viscous_bc_types: jax.Array
    alpha: float


def build_ins_operators(
    mesh: Mesh2D,
    *,
    dt: float,
    nu: float,
    g0: float,
    ins_bc_types: jax.Array,
    tau_scale: float = 200.0,
    skip_factor: bool = False,
    cub: "MeshCubature | None" = None,
) -> INSOperators:
    """Assemble + factor the pressure Poisson and viscous Helmholtz.

    Pass ``skip_factor=True`` to skip the LU factorisation step
    (returns ``pressure_factored = viscous_factored = None``). Use
    this when running with the matrix-free iterative path —
    factorising would waste the GPU memory savings.
    """
    p_bc = pressure_bc_types_from_ins(ins_bc_types)
    v_bc = viscous_bc_types_from_ins(ins_bc_types)
    alpha = g0 / (nu * dt)
    # With no Dirichlet pressure face (fully-periodic domain, or all
    # walls) the pressure Poisson is singular up to a constant —
    # regularise so the LU factorisation is well-posed.
    p_singular = not bool(jnp.any(p_bc == BC_DIRICHLET))
    if skip_factor:
        p_fact = None
        v_fact = None
    else:
        p_fact = factor_sipg_2d(
            mesh, bc_types=p_bc, tau_scale=tau_scale,
            regularize=p_singular, cub=cub,
        )
        v_fact = factor_sipg_helmholtz_2d(
            mesh, alpha=alpha, nu=nu, bc_types=v_bc, tau_scale=tau_scale,
            cub=cub,
        )
    return INSOperators(
        pressure_factored=p_fact,
        viscous_factored=v_fact,
        pressure_bc_types=p_bc,
        viscous_bc_types=v_bc,
        alpha=alpha,
    )


def make_ins_mg_preconditioners(
    mesh: Mesh2D,
    *,
    dt: float,
    nu: float,
    g0: float,
    ins_bc_types: jax.Array,
    tau_scale: float = 200.0,
    p_coarse: int = 1,
    nu_smooth: int = 2,
    omega: float = 0.7,
):
    """Build p-multigrid preconditioners for the INS pressure + viscous
    solves.

    Returns ``(pressure_preconditioner, viscous_preconditioner)`` —
    callables for the matching arguments of :func:`ns_step_2d` on its
    matrix-free iterative path. Each is a p-multigrid V-cycle
    (:func:`ddg.dg2d.multigrid.make_sipg_mg_preconditioner`) for the
    corresponding SIPG operator: the pressure Poisson ``M(-Δ)`` and the
    viscous Helmholtz ``α M + M(-Δ)`` with ``α = g0/(ν dt)``. Multigrid
    keeps the inner-CG iteration count near degree-independent, where
    block-Jacobi's count climbs with the polynomial order.

    The preconditioners depend on ``dt`` and ``g0`` through ``α``, so
    rebuild them whenever those change — in particular at the
    BDF1 → BDF2 switch between steps 1 and 2.

    For the full hp-multigrid (also mesh-independent) use
    :func:`ddg.dg2d.multigrid.make_sipg_hpmg_preconditioner` directly;
    it needs the solve mesh built from a forest hierarchy.
    """
    p_bc = pressure_bc_types_from_ins(ins_bc_types)
    v_bc = viscous_bc_types_from_ins(ins_bc_types)
    alpha = g0 / (nu * dt)
    p_singular = not bool(jnp.any(p_bc == BC_DIRICHLET))
    pressure_pc = make_sipg_mg_preconditioner(
        mesh, bc_types=p_bc, tau_scale=tau_scale, alpha=0.0,
        p_coarse=p_coarse, nu_smooth=nu_smooth, omega=omega,
        regularize=p_singular,
    )
    viscous_pc = make_sipg_mg_preconditioner(
        mesh, bc_types=v_bc, tau_scale=tau_scale, alpha=alpha,
        p_coarse=p_coarse, nu_smooth=nu_smooth, omega=omega,
    )
    return pressure_pc, viscous_pc


def ns_step_2d(
    mesh: Mesh2D,
    ux_n: jax.Array, uy_n: jax.Array,
    ux_old: jax.Array, uy_old: jax.Array,
    Nux_old: jax.Array, Nuy_old: jax.Array,
    dpdn_old: jax.Array,
    *,
    dt: float,
    nu: float,
    bdf: BDFCoeffs,
    ops: INSOperators,
    ins_bc_types: jax.Array,
    bc_ux: jax.Array,
    bc_uy: jax.Array,
    p_dir: jax.Array | None = None,
    q_ux_neu: jax.Array | None = None,
    q_uy_neu: jax.Array | None = None,
    tau_scale: float = 200.0,
    penalty_factored: tuple | None = None,
    penalty_iterative: "PenaltyConfig | None" = None,
    iterative: IterativeConfig | None = None,
    pressure_preconditioner=None,
    viscous_preconditioner=None,
    velocity_reconstruction=None,
    convective_form: str = "conservative",
    body_force: tuple[jax.Array, jax.Array] | None = None,
    cub: "MeshCubature | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One dual-splitting NS step. Returns ``(ux, uy, p, Nux, Nuy, dpdn)``.

    Pass ``penalty_factored = factor_penalty_projection_2d(...)`` to apply
    the divergence + continuity penalty post-projection after the viscous
    step (Fehn 2018, arXiv:1801.08103). Off by default.

    Pass ``penalty_iterative = PenaltyConfig(...)`` for a matrix-free CG
    version of the penalty projection — required past ~15k DOFs/field
    where the dense LU factor exceeds memory. If both are given,
    ``penalty_iterative`` takes precedence.

    Pass ``iterative=IterativeConfig(...)`` to swap the LU-factored inner
    solves for matrix-free CG. Important for GPU scaling: avoids the
    O(n²) memory of dense matrix assembly.

    ``dpdn_old`` is the previous step's pressure-Neumann data, needed
    for the BDF2 multistep extrapolation ``b0 dpdn_n + b1 dpdn_old`` of
    the pressure boundary forcing (Hesthaven INSPressure2D.m). For the
    first BDF1 step pass an array of zeros.

    Optional non-homogeneous BC fields (only relevant for validation
    against analytical solutions like Taylor-Green / Kovasznay where
    the canonical ``p=0`` outflow / ``∂u/∂n=0`` outflow are wrong):

      * ``p_dir`` (``(Np, K)``): non-homog Dirichlet for pressure on
        outflow faces. Default ``None`` means ``p = 0``.
      * ``q_ux_neu``, ``q_uy_neu`` (``(Nfp*Nfaces, K)``): non-homog
        Neumann for ``∂u/∂n`` on outflow faces. Default ``None`` means
        zero traction.

    ``body_force`` (optional ``(f_x, f_y)``, each ``(Np, K)``): an
    explicit momentum body force added to the transport step — the
    Boussinesq buoyancy term ``f = −β(T−T_ref) g`` for a
    temperature/density-coupled flow. Treated explicitly at the
    current step (the buoyancy field evolves slowly relative to dt).
    """
    g0 = bdf.g0
    # 1) Advection: evaluate N(U^n) and extrapolate.
    Nux_n, Nuy_n = ns_advection_2d(
        mesh, ux_n, uy_n,
        ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
        velocity_reconstruction=velocity_reconstruction,
        convective_form=convective_form,
    )
    ux_T = (
        bdf.a0 * ux_n + bdf.a1 * ux_old
        - dt * (bdf.b0 * Nux_n + bdf.b1 * Nux_old)
    ) / g0
    uy_T = (
        bdf.a0 * uy_n + bdf.a1 * uy_old
        - dt * (bdf.b0 * Nuy_n + bdf.b1 * Nuy_old)
    ) / g0
    # Explicit body force (Boussinesq buoyancy). Enters the transport
    # step alongside −N(u); the pressure projection then balances it
    # against incompressibility.
    if body_force is not None:
        fx, fy = body_force
        ux_T = ux_T + (dt / g0) * fx
        uy_T = uy_T + (dt / g0) * fy

    # 2) Pressure step. Multistep-extrapolate the boundary Neumann data
    # ``b0 dpdn_n + b1 dpdn_old`` (matches Hesthaven INSPressure2D.m).
    # When a velocity-reconstruction Π is supplied (Linke 2014), use
    # ∇·(Π u_T) instead of ∇·u_T for pressure-robustness. The pressure
    # gradient that ends up in u^TT is the same — but the source for
    # the Poisson solve no longer carries the spurious 1/ν contribution.
    if velocity_reconstruction is not None:
        ux_T_div, uy_T_div = velocity_reconstruction(
            mesh, ux_T, uy_T,
            bc_normal_overrides=(bc_ux, bc_uy),
        )
    else:
        ux_T_div, uy_T_div = ux_T, uy_T
    div_U_T = _div_2d(mesh, ux_T_div, uy_T_div)
    p_source = -(g0 / dt) * div_U_T
    dpdn_n = ns_pressure_neumann_data(
        mesh, ux_n, uy_n, Nux_n, Nuy_n, nu=nu,
    )
    dpdn_extrap = bdf.b0 * dpdn_n + bdf.b1 * dpdn_old
    p = _pressure_solve(
        mesh, p_source,
        bc_types=ops.pressure_bc_types,
        q_neu=dpdn_extrap, u_dir=p_dir,
        A_factored=ops.pressure_factored,
        tau_scale=tau_scale, iterative=iterative,
        preconditioner=pressure_preconditioner,
        cub=cub,
    )
    dpdx, dpdy = grad_2d(mesh, p)
    ux_TT = ux_T - (dt / g0) * dpdx
    uy_TT = uy_T - (dt / g0) * dpdy

    # 3) Viscous step.
    alpha = ops.alpha
    ux_new = _viscous_solve(
        mesh, alpha * ux_TT,
        bc_types=ops.viscous_bc_types,
        u_dir=bc_ux, q_neu=q_ux_neu,
        alpha=alpha, nu=nu,
        A_factored=ops.viscous_factored,
        iterative=iterative, x0=ux_n,
        preconditioner=viscous_preconditioner,
        cub=cub,
    )
    uy_new = _viscous_solve(
        mesh, alpha * uy_TT,
        bc_types=ops.viscous_bc_types,
        u_dir=bc_uy, q_neu=q_uy_neu,
        alpha=alpha, nu=nu,
        A_factored=ops.viscous_factored,
        iterative=iterative, x0=uy_n,
        preconditioner=viscous_preconditioner,
        cub=cub,
    )

    if penalty_iterative is not None:
        ux_new, uy_new = penalty_projection_step_cg_2d(
            mesh, ux_new, uy_new,
            tau_div=penalty_iterative.tau_div,
            tau_cont=penalty_iterative.tau_cont,
            tol=penalty_iterative.tol,
            atol=penalty_iterative.atol,
            maxiter=penalty_iterative.maxiter,
        )
    elif penalty_factored is not None:
        ux_new, uy_new = penalty_projection_step_2d(
            mesh, ux_new, uy_new, factored=penalty_factored,
        )

    return ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n


def _pressure_solve(
    mesh: Mesh2D,
    source: jax.Array,
    *,
    bc_types: jax.Array,
    q_neu: jax.Array,
    u_dir: jax.Array | None,
    A_factored: tuple | None,
    tau_scale: float = 200.0,
    iterative: IterativeConfig | None = None,
    x0: jax.Array | None = None,
    preconditioner=None,
    cub: "MeshCubature | None" = None,
) -> jax.Array:
    """Solve the pressure Poisson. Direct LU (with ``A_factored``) by
    default; matrix-free CG via ``iterative=IterativeConfig(...)`` for
    GPU-scalable runs.

    Pass ``preconditioner`` (e.g. from
    :func:`ddg.dg2d.sipg.make_sipg_block_jacobi_preconditioner`) to
    accelerate CG; ignored in direct-LU mode. Typical 1.5-2x wall
    speedup on the SIPG Poisson CG path.
    """
    M_ref = _local_mass(mesh)
    if cub is None:
        Mf = mesh.J * (M_ref @ source)
    else:
        from ddg.dg2d.cubature import cubature_mass_apply
        Mf = cubature_mass_apply(cub, source)
    rhs = Mf + sipg_neumann_load_2d(mesh, q_neu, bc_types=bc_types)
    if u_dir is not None:
        rhs = rhs - sipg_dirichlet_load_2d(
            mesh, u_dir, bc_types=bc_types, tau_scale=tau_scale,
        )
    rhs_flat = rhs.T.reshape(-1)

    if iterative is None:
        _, lu, piv = A_factored
        p_flat = jax.scipy.linalg.lu_solve((lu, piv), rhs_flat)
    else:
        m1d_per_face = _face_1d_mass_matrices(mesh)
        Np, K = mesh.Np, mesh.K
        def raw_matvec(v):
            u = v.reshape(K, Np).T
            return sipg_apply_2d(
                mesh, u, bc_types=bc_types, u_dir=None,
                tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d_per_face,
                cub=cub,
            ).T.reshape(-1)
        # Closed / fully-periodic domain: with no Dirichlet face the
        # SIPG Poisson is singular, its nullspace the constant fields.
        # Deflate it -- project the RHS onto range(A) (mean-zero) so it
        # is consistent, and ε-shift the operator so it is strictly SPD
        # and CG cannot stall on a zero pivot. Only ∇p feeds the
        # velocity update, so the residual constant gauge is immaterial.
        singular = not bool(jnp.any(bc_types == BC_DIRICHLET))
        if singular:
            idx = jnp.arange(Np * K, dtype=jnp.float64)
            probe = jnp.sin(idx)
            probe = probe - jnp.mean(probe)
            scale = (jnp.linalg.norm(raw_matvec(probe))
                     / jnp.linalg.norm(probe))
            shift = 1.0e-6 * scale
            def matvec(v):
                return raw_matvec(v) + shift * v
            rhs_flat = rhs_flat - jnp.mean(rhs_flat)
        else:
            matvec = raw_matvec
        x0_flat = x0.T.reshape(-1) if x0 is not None else jnp.zeros(Np * K)
        p_flat, _ = jax.scipy.sparse.linalg.cg(
            matvec, rhs_flat, x0=x0_flat,
            tol=iterative.tol, atol=iterative.atol, maxiter=iterative.maxiter,
            M=preconditioner,
        )
    return p_flat.reshape(mesh.K, mesh.Np).T


def _viscous_solve(
    mesh: Mesh2D,
    source: jax.Array,
    *,
    bc_types: jax.Array,
    u_dir: jax.Array,
    q_neu: jax.Array | None,
    alpha: float,
    nu: float,
    A_factored: tuple | None,
    tau_scale: float = 200.0,
    iterative: IterativeConfig | None = None,
    x0: jax.Array | None = None,
    preconditioner=None,
    cub: "MeshCubature | None" = None,
) -> jax.Array:
    """Solve the viscous Helmholtz step. Direct LU by default; matrix-
    free CG via ``iterative``.

    Pass ``preconditioner`` (e.g. from
    :func:`ddg.dg2d.sipg.make_helmholtz_block_jacobi_preconditioner`)
    to accelerate the matrix-free CG.
    """
    M_ref = _local_mass(mesh)
    if cub is None:
        Mf = mesh.J * (M_ref @ source)
    else:
        from ddg.dg2d.cubature import cubature_mass_apply
        Mf = cubature_mass_apply(cub, source)
    rhs = Mf - sipg_dirichlet_load_2d(
        mesh, u_dir, bc_types=bc_types, tau_scale=tau_scale,
    )
    if q_neu is not None:
        rhs = rhs + sipg_neumann_load_2d(mesh, q_neu, bc_types=bc_types)
    rhs_flat = rhs.T.reshape(-1)

    if iterative is None:
        _, lu, piv = A_factored
        u_flat = jax.scipy.linalg.lu_solve((lu, piv), rhs_flat)
    else:
        m1d_per_face = _face_1d_mass_matrices(mesh)
        Np, K = mesh.Np, mesh.K
        def matvec(v):
            u = v.reshape(K, Np).T
            sipg_part = sipg_apply_2d(
                mesh, u, bc_types=bc_types, u_dir=None,
                tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d_per_face,
                cub=cub,
            )
            if cub is None:
                mass_part = mesh.J * (M_ref @ u)
            else:
                from ddg.dg2d.cubature import cubature_mass_apply
                mass_part = cubature_mass_apply(cub, u)
            return (alpha * mass_part + sipg_part).T.reshape(-1)
        x0_flat = x0.T.reshape(-1) if x0 is not None else jnp.zeros(Np * K)
        u_flat, _ = jax.scipy.sparse.linalg.cg(
            matvec, rhs_flat, x0=x0_flat,
            tol=iterative.tol, atol=iterative.atol, maxiter=iterative.maxiter,
            M=preconditioner,
        )
    return u_flat.reshape(mesh.K, mesh.Np).T


# ---------------------------------------------------------------------------
# Consistent splitting scheme (Fehn / Wall / Kronbichler, JCP 2018)
#
# Differences vs the dual-splitting :func:`ns_step_2d`:
#
#   1. **Pressure equation**: ``−Δp = −∇·N(u)`` (in weak form), as opposed
#      to dual splitting's ``−Δp = (g0/dt) ∇·Ũ``. Integrating by parts:
#
#          ∫ ∇p · ∇v = − ∫ N(u_extrap) · ∇v
#                      + ∫_∂_int  {{N(u)·n}} [v]
#                      + ∫_Γ_neu  (N(u)·n) v_M
#                      − ∫_Γ_dir  ((γ/dt) g_D · n + ν curl-curl(u) · n) v
#
#      The last term is the *consistent* Neumann boundary condition that
#      replaces the convective contribution on the velocity-Dirichlet
#      boundary. Crucially, ``N`` doesn't appear in the BC — only in the
#      volume / interior-face contributions.
#
#   2. **Momentum step**: solves the full Helmholtz system
#      ``(γ/(ν dt)) M − Δ) u_{n+1} = (γ/(ν dt)) (Σ_i α_i u_{n-i}) − N_extrap − ∇p_{n+1}``
#      with Dirichlet ``u = g_D`` on inflow/wall/obstacle and Neumann
#      ``∂u/∂n = 0`` on outflow — same Helmholtz operator as the dual-
#      splitting viscous step, only the RHS differs.
#
#   3. **Step order**: pressure first, then momentum. Dual splitting does
#      advection → pressure → viscous.
#
# References:
#   * Fehn, Wall, Kronbichler. "On the stability of projection methods…"
#     JCP 2018. arXiv:1706.09252. The paper that introduced this scheme
#     specifically to fix the small-dt instability of DG dual splitting.
#   * exadg `operator_consistent_splitting.cpp` / `time_int_bdf_consistent_splitting.cpp`.
# ---------------------------------------------------------------------------


def consistent_splitting_pressure_rhs_2d(
    mesh: Mesh2D,
    ux_extrap: jax.Array,
    uy_extrap: jax.Array,
    *,
    nu: float,
    dt: float,
    bdf: "BDFCoeffs",
    ins_bc_types: jax.Array,
    bc_ux: jax.Array,
    bc_uy: jax.Array,
    ux_n: jax.Array,
    uy_n: jax.Array,
    ux_old: jax.Array,
    uy_old: jax.Array,
) -> jax.Array:
    """Assemble the right-hand-side load for the consistent-splitting Poisson.

    Returns an ``(Np, K)`` field ``b`` such that solving the SIPG
    Poisson with ``A p = b`` (using the same pressure BC types as dual
    splitting: Neumann on inflow/wall/obstacle, Dirichlet on outflow)
    gives the consistent-splitting pressure ``p_{n+1}``.

    The consistent Neumann BC discretises the time derivative
    ``∂_t u ≈ (γ u^{n+1} − a_0 u^n − a_1 u^{n−1}) / dt`` and
    contributes ``−n·∂_t u_{bc} − ν n·curl-curl(u_extrap)`` on the
    velocity-Dirichlet boundary. For a steady BC (``g_D^{n+1} = g_D^n =
    g_D^{n−1}``) the BDF combination ``γ − a_0 − a_1 = 0`` so the
    time-derivative term vanishes automatically — that's how steady
    flows like Poiseuille remain exact fixed points.

    ``bc_ux, bc_uy`` are the new-time prescribed Dirichlet velocity
    fields (``g_D^{n+1}``); ``ux_n, uy_n`` and ``ux_old, uy_old`` carry
    the velocity history (only boundary face traces matter for the BC,
    but we accept volume fields for consistency with the calling code).
    """
    Np, K, Nfp, Nfaces = mesh.Np, mesh.K, mesh.Nfp, mesh.Nfaces
    M_ref = _local_mass(mesh)

    # --- Convective term N(u) = u · ∇u, evaluated nodally ---
    ur = mesh.Dr @ ux_extrap
    us = mesh.Ds @ ux_extrap
    vr = mesh.Dr @ uy_extrap
    vs = mesh.Ds @ uy_extrap
    dudx = mesh.rx * ur + mesh.sx * us
    dudy = mesh.ry * ur + mesh.sy * us
    dvdx = mesh.rx * vr + mesh.sx * vs
    dvdy = mesh.ry * vr + mesh.sy * vs
    Nx = ux_extrap * dudx + uy_extrap * dudy
    Ny = ux_extrap * dvdx + uy_extrap * dvdy

    # --- Volume contribution: −∫_Ω N · ∇v ---
    # = −J_K · (Dx^T · M_ref · Nx + Dy^T · M_ref · Ny) per element.
    M_Nx = M_ref @ Nx
    M_Ny = M_ref @ Ny
    rhs_vol = -mesh.J * (
        mesh.rx * (mesh.Dr.T @ M_Nx) + mesh.sx * (mesh.Ds.T @ M_Nx)
        + mesh.ry * (mesh.Dr.T @ M_Ny) + mesh.sy * (mesh.Ds.T @ M_Ny)
    )

    # --- Face contributions ---
    Nx_flat = Nx.T.reshape(-1)
    Ny_flat = Ny.T.reshape(-1)
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")

    Nx_M = Nx_flat[mesh.vmapM]
    Ny_M = Ny_flat[mesh.vmapM]
    Nx_P = Nx_flat[mesh.vmapP]
    Ny_P = Ny_flat[mesh.vmapP]

    flux_n = 0.5 * (Nx_M + Nx_P) * nx_flat + 0.5 * (Ny_M + Ny_P) * ny_flat

    bc_per_facenode_flat = _ins_bc_facenodes_flat(mesh, ins_bc_types)
    is_vel_dir = (
        (bc_per_facenode_flat == BC_INS_INFLOW)
        | (bc_per_facenode_flat == BC_INS_WALL)
        | (bc_per_facenode_flat == BC_INS_OBSTACLE)
    )
    is_vel_neu = bc_per_facenode_flat == BC_INS_OUTFLOW
    # Velocity-Dirichlet faces: the convective contribution cancels with
    # the consistent NBC — set to 0 here, then add the NBC term below.
    flux_n = jnp.where(is_vel_dir, 0.0, flux_n)
    # Interior and velocity-Neumann (outflow) faces: keep the flux.
    # On boundary outflow faces, vmapP == vmapM so the "average" reduces
    # to the K-side trace, matching exadg's boundary treatment.

    flux_face = flux_n.reshape((Nfp * Nfaces, K), order="F")
    surf_int = mesh.LIFT @ (mesh.Fscale * flux_face)
    rhs_face = mesh.J * (M_ref @ surf_int)

    # --- Consistent Neumann BC contribution ---
    # ∂p/∂n = −(γ/dt) g_D · n − ν curl_curl(u) · n
    # In 2D: curl_curl(u) = (∂y ω, −∂x ω) where ω = ∂x v − ∂y u.
    omega = curl_z_2d(mesh, ux_extrap, uy_extrap)
    dwdx, dwdy = grad_2d(mesh, omega)
    dwdx_flat = dwdx.T.reshape(-1)
    dwdy_flat = dwdy.T.reshape(-1)
    # BDF approximation to ∂_t g_D at the boundary:
    #   ∂_t g_D ≈ (γ g_D^{n+1} − a_0 g_D^n − a_1 g_D^{n−1}) / dt
    # Use bc_ux as g_D^{n+1} and the velocity history (= BC at previous
    # times for nodes on a velocity-Dirichlet boundary) for the others.
    bc_ux_flat = bc_ux.T.reshape(-1)
    bc_uy_flat = bc_uy.T.reshape(-1)
    ux_n_flat = ux_n.T.reshape(-1)
    uy_n_flat = uy_n.T.reshape(-1)
    ux_old_flat = ux_old.T.reshape(-1)
    uy_old_flat = uy_old.T.reshape(-1)

    g_np1_dot_n = (
        bc_ux_flat[mesh.vmapM] * nx_flat
        + bc_uy_flat[mesh.vmapM] * ny_flat
    )
    g_n_dot_n = (
        ux_n_flat[mesh.vmapM] * nx_flat
        + uy_n_flat[mesh.vmapM] * ny_flat
    )
    g_old_dot_n = (
        ux_old_flat[mesh.vmapM] * nx_flat
        + uy_old_flat[mesh.vmapM] * ny_flat
    )
    dt_g_dot_n = (
        bdf.g0 * g_np1_dot_n - bdf.a0 * g_n_dot_n - bdf.a1 * g_old_dot_n
    )
    h_dudt = -dt_g_dot_n / dt
    # curl_curl · n = nx · ∂y ω − ny · ∂x ω
    cc_dot_n = (
        nx_flat * dwdy_flat[mesh.vmapM]
        - ny_flat * dwdx_flat[mesh.vmapM]
    )
    h_viscous = -nu * cc_dot_n
    h_NBC = jnp.where(is_vel_dir, h_dudt + h_viscous, 0.0)

    h_face = h_NBC.reshape((Nfp * Nfaces, K), order="F")
    rhs_NBC = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * h_face)))

    return rhs_vol + rhs_face + rhs_NBC


def _pressure_dirichlet_load_consistent(
    mesh: Mesh2D,
    p_dir: jax.Array,
    *,
    pressure_bc_types: jax.Array,
    tau_scale: float = 200.0,
) -> jax.Array:
    """Standard SIPG Dirichlet-load contribution for non-homog pressure BC."""
    return sipg_dirichlet_load_2d(
        mesh, p_dir, bc_types=pressure_bc_types, tau_scale=tau_scale,
    )


def ns_step_consistent_splitting_2d(
    mesh: Mesh2D,
    ux_n: jax.Array, uy_n: jax.Array,
    ux_old: jax.Array, uy_old: jax.Array,
    *,
    dt: float,
    nu: float,
    bdf: BDFCoeffs,
    ops: INSOperators,
    ins_bc_types: jax.Array,
    bc_ux: jax.Array,
    bc_uy: jax.Array,
    p_dir: jax.Array | None = None,
    q_ux_neu: jax.Array | None = None,
    q_uy_neu: jax.Array | None = None,
    tau_scale: float = 200.0,
    penalty_factored: tuple | None = None,
    penalty_iterative: "PenaltyConfig | None" = None,
    iterative: IterativeConfig | None = None,
    pressure_preconditioner=None,
    viscous_preconditioner=None,
    velocity_reconstruction=None,
    convective_form: str = "conservative",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One consistent-splitting step. Returns ``(ux, uy, p)``.

    Note: unlike :func:`ns_step_2d` this does *not* need ``Nux_old / Nuy_old``
    or ``dpdn_old`` history fields — the extrapolation is built directly
    from the velocity history ``(ux_n, ux_old)`` since the convective
    term is computed inline.

    Pass ``penalty_factored`` to apply the divergence + continuity
    penalty post-projection after the momentum step (Fehn 2018,
    arXiv:1801.08103).

    Pass ``penalty_iterative = PenaltyConfig(...)`` for the matrix-free
    CG version of the penalty projection. Required past ~15k DOFs/field
    where the dense LU factor exceeds memory. Takes precedence over
    ``penalty_factored`` if both are given.

    Pass ``iterative=IterativeConfig(...)`` to swap the LU-factored
    pressure / momentum inner solves for matrix-free CG.
    """
    g0 = bdf.g0

    # Time-extrapolated velocity (for the pressure RHS only).
    # BDF1: u_extrap = u^n. BDF2: u_extrap = 2 u^n − u^{n−1}.
    if bdf is BDF1:
        ux_extrap = ux_n
        uy_extrap = uy_n
    else:
        ux_extrap = 2.0 * ux_n - ux_old
        uy_extrap = 2.0 * uy_n - uy_old

    # 1) Pressure step.
    rhs_p = consistent_splitting_pressure_rhs_2d(
        mesh, ux_extrap, uy_extrap,
        nu=nu, dt=dt, bdf=bdf,
        ins_bc_types=ins_bc_types,
        bc_ux=bc_ux, bc_uy=bc_uy,
        ux_n=ux_n, uy_n=uy_n,
        ux_old=ux_old, uy_old=uy_old,
    )
    if p_dir is not None:
        rhs_p = rhs_p - _pressure_dirichlet_load_consistent(
            mesh, p_dir, pressure_bc_types=ops.pressure_bc_types,
            tau_scale=tau_scale,
        )
    rhs_flat = rhs_p.T.reshape(-1)
    Np, K = mesh.Np, mesh.K
    if iterative is None:
        _, lu_p, piv_p = ops.pressure_factored
        p_flat = jax.scipy.linalg.lu_solve((lu_p, piv_p), rhs_flat)
    else:
        from ddg.dg2d.sipg import (
            _face_1d_mass_matrices as _f1d, sipg_apply_2d as _sipg_app,
        )
        M_ref_p = _local_mass(mesh)
        m1d_p = _f1d(mesh)
        def _matvec_p(v):
            u = v.reshape(K, Np).T
            return _sipg_app(
                mesh, u, bc_types=ops.pressure_bc_types, u_dir=None,
                tau_scale=tau_scale, M_ref=M_ref_p, m1d_per_face=m1d_p,
            ).T.reshape(-1)
        p_flat, _ = jax.scipy.sparse.linalg.cg(
            _matvec_p, rhs_flat, x0=jnp.zeros(Np * K),
            tol=iterative.tol, atol=iterative.atol, maxiter=iterative.maxiter,
            M=pressure_preconditioner,
        )
    p = p_flat.reshape(K, Np).T

    # 2) Momentum step. Helmholtz: (α M − Δ) u_{n+1} = α·Σ α_i u_{n-i}
    # − N_extrap − ∇p_{n+1}, with α = g0 / (ν dt).
    Nux_extrap, Nuy_extrap = ns_advection_2d(
        mesh, ux_extrap, uy_extrap,
        ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
        velocity_reconstruction=velocity_reconstruction,
        convective_form=convective_form,
    )
    dpdx, dpdy = grad_2d(mesh, p)
    bdf_velocity_sum_x = (bdf.a0 * ux_n + bdf.a1 * ux_old)
    bdf_velocity_sum_y = (bdf.a0 * uy_n + bdf.a1 * uy_old)
    # The momentum equation: γ ∂_t u + N(u) + ∇p = ν Δ u
    #   → γ (u_{n+1} − Σ α_i u_{n-i} / γ) / dt + N_extrap + ∇p = ν Δ u_{n+1}
    #   → (γ/(ν dt)) (M u_{n+1} − M · Σ α_i u_{n-i} / γ) = Δ u_{n+1} − (1/ν)(N + ∇p)
    #   → ((γ/(ν dt)) M − Δ) u_{n+1} = (γ/(ν dt)) M Σ α_i u_{n-i} / γ
    #                                    − (1/ν) M (N_extrap + ∇p)
    # In our solver convention (RHS includes the M f part):
    alpha = ops.alpha  # = g0 / (ν dt)
    # (α / g0) = 1 / (ν dt); BDF α_i = a0 = g0, a1 = etc.; sum α_i u_i = g0·u + a1·u_old.
    # Wait: BDF coefficients (α_0=γ, α_1=−4/3·..., etc.) — exadg uses
    # alpha for the sum. In our convention BDFCoeffs has a0, a1 as the
    # BDF extrapolation coefficients (a0=γ for BDF1, =2 for BDF2). So
    # γ·u_{n+1} − a0·u_n − a1·u_old = dt·(rhs). a0 = g0 in our convention
    # only for BDF1 (a0=1=g0). For BDF2 a0=2 ≠ g0=1.5.
    #
    # Rearranging the BDF time discretization:
    #   (g0 u_{n+1} − a0 u_n − a1 u_old)/dt + N_extrap + ∇p = ν Δ u_{n+1}
    # → (g0/(ν dt)) M u_{n+1} − M·Δ u_{n+1}
    #     = (1/(ν dt)) M (a0 u_n + a1 u_old) − (1/ν) M (N_extrap + ∇p)
    # The Helmholtz operator we factored is α M − Δ_h with α = g0/(ν dt),
    # so its RHS source field f satisfies (α M − Δ_h) u = M f i.e.
    #   f = (1/(ν dt))(a0 u_n + a1 u_old) − (1/ν)(N_extrap + ∇p)
    source_x = (
        (bdf_velocity_sum_x) / (nu * dt)
        - (Nux_extrap + dpdx) / nu
    )
    source_y = (
        (bdf_velocity_sum_y) / (nu * dt)
        - (Nuy_extrap + dpdy) / nu
    )

    ux_new = _viscous_solve(
        mesh, source_x,
        bc_types=ops.viscous_bc_types,
        u_dir=bc_ux, q_neu=q_ux_neu,
        alpha=alpha, nu=nu,
        A_factored=ops.viscous_factored,
        iterative=iterative, x0=ux_n,
        preconditioner=viscous_preconditioner,
    )
    uy_new = _viscous_solve(
        mesh, source_y,
        bc_types=ops.viscous_bc_types,
        u_dir=bc_uy, q_neu=q_uy_neu,
        alpha=alpha, nu=nu,
        A_factored=ops.viscous_factored,
        iterative=iterative, x0=uy_n,
        preconditioner=viscous_preconditioner,
    )

    if penalty_iterative is not None:
        ux_new, uy_new = penalty_projection_step_cg_2d(
            mesh, ux_new, uy_new,
            tau_div=penalty_iterative.tau_div,
            tau_cont=penalty_iterative.tau_cont,
            tol=penalty_iterative.tol,
            atol=penalty_iterative.atol,
            maxiter=penalty_iterative.maxiter,
        )
    elif penalty_factored is not None:
        ux_new, uy_new = penalty_projection_step_2d(
            mesh, ux_new, uy_new, factored=penalty_factored,
        )

    return ux_new, uy_new, p


# ---------------------------------------------------------------------------
# Divergence + continuity penalty post-projection
# (Fehn / Wall / Kronbichler, JCP 2018; arXiv:1801.08103)
#
# Adds two consistent penalty terms to the velocity field via a separate
# post-projection solve after the momentum step. They keep the scheme
# robust on under-resolved turbulent flows by weakly enforcing:
#
#   a_D(U, V) = τ_d ∫_K (∇·U)(∇·V) dx          (divergence — within element)
#   a_C(U, V) = τ_c ∫_∂_int [U·n][V·n] dS      (continuity — across interfaces,
#                                               normal component only)
#
# The projection step solves ``(M + τ_d D + τ_c C) U_new = M U_int``. Both
# penalties are symmetric positive-semi-definite, so the operator is SPD
# and admits a Cholesky / LU factorisation.
# ---------------------------------------------------------------------------


def _div_velocity_2d(
    mesh: Mesh2D, ux: jax.Array, uy: jax.Array
) -> jax.Array:
    """Nodal strong-form divergence ``∂x ux + ∂y uy`` per element."""
    ur = mesh.Dr @ ux
    us = mesh.Ds @ ux
    vr = mesh.Dr @ uy
    vs = mesh.Ds @ uy
    return (
        mesh.rx * ur + mesh.sx * us
        + mesh.ry * vr + mesh.sy * vs
    )


def divergence_penalty_apply_2d(
    mesh: Mesh2D, ux: jax.Array, uy: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply the divergence-penalty operator (per-element, no τ scaling).

    Bilinear form: ``a_D(U, V) = ∫_K (∇·U)(∇·V) dx``. The action on
    ``(U_x, U_y)`` returns ``(D U)_x = ∂(∇·U)/∂x``-side test integral,
    ``(D U)_y = ∂(∇·U)/∂y``-side. With block-diagonal element structure.
    """
    M_ref = _local_mass(mesh)
    div_U = _div_velocity_2d(mesh, ux, uy)
    M_div = M_ref @ div_U
    out_x = mesh.J * (
        mesh.rx * (mesh.Dr.T @ M_div) + mesh.sx * (mesh.Ds.T @ M_div)
    )
    out_y = mesh.J * (
        mesh.ry * (mesh.Dr.T @ M_div) + mesh.sy * (mesh.Ds.T @ M_div)
    )
    return out_x, out_y


def continuity_penalty_apply_2d(
    mesh: Mesh2D, ux: jax.Array, uy: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply the (normal-only) continuity-penalty operator, no τ scaling.

    Bilinear form: ``a_C(U, V) = ∫_∂_int [U·n][V·n] dS`` summed over
    interior faces. Returns ``(C U)_x, (C U)_y`` as ``(Np, K)`` arrays.

    On boundary faces ``vmapP = vmapM`` automatically zeros the jump, so
    only true interior faces contribute.
    """
    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")

    UM_dot_n = ux_flat[mesh.vmapM] * nx_flat + uy_flat[mesh.vmapM] * ny_flat
    UP_dot_n = ux_flat[mesh.vmapP] * nx_flat + uy_flat[mesh.vmapP] * ny_flat
    jump_dot_n = UM_dot_n - UP_dot_n

    flux_x = (jump_dot_n * nx_flat).reshape((Nfp_Nf, K), order="F")
    flux_y = (jump_dot_n * ny_flat).reshape((Nfp_Nf, K), order="F")

    M_ref = _local_mass(mesh)
    out_x = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * flux_x)))
    out_y = mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * flux_y)))
    return out_x, out_y


def _mass_apply_2d(
    mesh: Mesh2D, ux: jax.Array, uy: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply the velocity mass matrix per element: ``M_K · U``."""
    M_ref = _local_mass(mesh)
    return mesh.J * (M_ref @ ux), mesh.J * (M_ref @ uy)


def penalty_projection_matrix_2d(
    mesh: Mesh2D,
    *,
    tau_div: float,
    tau_cont: float,
) -> jax.Array:
    """Dense ``(2 Np K) × (2 Np K)`` matrix for ``M + τ_d D + τ_c C``.

    Layout: column-major flatten of ``(Np, K)`` per component, with
    ``ux`` first ``(Np * K,)``, then ``uy``. Use the same flatten /
    unflatten convention throughout.
    """
    Np, K = mesh.Np, mesh.K
    n_dof = Np * K

    def op_flat(uvec: jax.Array) -> jax.Array:
        ux = uvec[:n_dof].reshape(K, Np).T
        uy = uvec[n_dof:].reshape(K, Np).T
        mx, my = _mass_apply_2d(mesh, ux, uy)
        dx, dy = divergence_penalty_apply_2d(mesh, ux, uy)
        cx, cy = continuity_penalty_apply_2d(mesh, ux, uy)
        out_x = mx + tau_div * dx + tau_cont * cx
        out_y = my + tau_div * dy + tau_cont * cy
        return jnp.concatenate([out_x.T.reshape(-1), out_y.T.reshape(-1)])

    return jax.jacobian(op_flat)(jnp.zeros(2 * n_dof))


def factor_penalty_projection_2d(
    mesh: Mesh2D,
    *,
    tau_div: float,
    tau_cont: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Assemble + LU-factor the penalty-projection matrix."""
    A = penalty_projection_matrix_2d(mesh, tau_div=tau_div, tau_cont=tau_cont)
    lu, piv = jax.scipy.linalg.lu_factor(A)
    return A, lu, piv


def penalty_projection_step_2d(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    *,
    factored: tuple,
) -> tuple[jax.Array, jax.Array]:
    """Apply one post-projection: ``(M + τ_d D + τ_c C) U_new = M U``.

    Given a pre-factored matrix from :func:`factor_penalty_projection_2d`,
    take the intermediate velocity ``(ux, uy)`` and return the projected
    velocity. Penalty parameters were baked into the factored matrix.
    """
    Np, K = mesh.Np, mesh.K
    n_dof = Np * K
    mx, my = _mass_apply_2d(mesh, ux, uy)
    rhs = jnp.concatenate([mx.T.reshape(-1), my.T.reshape(-1)])
    _, lu, piv = factored
    u_flat = jax.scipy.linalg.lu_solve((lu, piv), rhs)
    ux_new = u_flat[:n_dof].reshape(K, Np).T
    uy_new = u_flat[n_dof:].reshape(K, Np).T
    return ux_new, uy_new


def penalty_projection_step_cg_2d(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    *,
    tau_div: float,
    tau_cont: float,
    tol: float = 1e-9,
    atol: float = 0.0,
    maxiter: int = 1000,
) -> tuple[jax.Array, jax.Array]:
    """Matrix-free CG version of :func:`penalty_projection_step_2d`.

    Solves the same SPD system ``(M + τ_d D + τ_c C) U_new = M U``
    without ever assembling the ``(2 Np K)^2`` dense matrix. Required
    for problem sizes past ~15k DOFs/field, where the LU factor blows
    out host or device memory.

    The operator is symmetric positive-definite (sum of a mass matrix
    and two SPD penalty terms), so CG converges. The matrix-free
    applies of ``M``, ``D`` and ``C`` are reused from the existing
    primitives.
    """
    Np, K = mesh.Np, mesh.K
    n_dof = Np * K

    def matvec(v_flat):
        ux_v = v_flat[:n_dof].reshape(K, Np).T
        uy_v = v_flat[n_dof:].reshape(K, Np).T
        mx, my = _mass_apply_2d(mesh, ux_v, uy_v)
        dx, dy = divergence_penalty_apply_2d(mesh, ux_v, uy_v)
        cx, cy = continuity_penalty_apply_2d(mesh, ux_v, uy_v)
        out_x = mx + tau_div * dx + tau_cont * cx
        out_y = my + tau_div * dy + tau_cont * cy
        return jnp.concatenate([out_x.T.reshape(-1), out_y.T.reshape(-1)])

    mx, my = _mass_apply_2d(mesh, ux, uy)
    rhs = jnp.concatenate([mx.T.reshape(-1), my.T.reshape(-1)])
    x0 = jnp.concatenate([ux.T.reshape(-1), uy.T.reshape(-1)])
    u_flat, _ = jax.scipy.sparse.linalg.cg(
        matvec, rhs, x0=x0, tol=tol, atol=atol, maxiter=maxiter,
    )
    ux_new = u_flat[:n_dof].reshape(K, Np).T
    uy_new = u_flat[n_dof:].reshape(K, Np).T
    return ux_new, uy_new
