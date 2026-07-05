"""Monolithic coupled Stokes / Navier-Stokes solver on the BDM_1 / P_0 pair.

H(div)-conforming Brezzi-Douglas-Marini velocity at degree 1 paired with
element-piecewise-constant pressure. The saddle-point block structure
is

    [ ν A_visc   G    ] [u]   [b_u]
    [ -D         εM_p ] [p] = [b_p]

where

    A_visc : BDM_1 -> BDM_1     SPD viscous Helmholtz block (Fehn-style
                                 H(div) IPG; volume + face SIPG terms).
    D      : BDM_1 -> P_0       element-local divergence
                                 (∇·BDM_1 ⊆ P_0 by construction).
    G      : P_0   -> BDM_1     -D^T -- saddle-point skew adjoint.
    M_p    : P_0   -> P_0       diagonal pressure mass for the
                                 Brezzi-Pitkaranta nullspace lift.

ε > 0 lifts the constant-pressure nullspace on closed-boundary problems
(small enough that the lift is negligible; default 0 i.e. add only when
needed). For Stokes-only the system is linear; the convective term
enters as an additional Newton/Picard linearisation in NS (Phase 5g).

The LBB sentinel from :mod:`bdm` (rank(D) >= K - 1) is the structural
reason this saddle-point is well-posed without LBB stabilisation knobs.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.bdm import (
    BDM,
    _bdm1_face_quadrature_data,
    _bdm1_physical_gradient_matrices,
    _element_jacobian_matrices,
    bdm1_basis_at_points,
    bdm1_element_local_mass,
    bdm1_evaluate_boundary_velocity,
    bdm1_gather,
    bdm1_scatter_add,
    bdm1_viscous_apply,
    bdm1_viscous_dirichlet_lift,
    bdm1_viscous_volume_stiffness,
    bdm_p0_divergence_apply,
    bdm_p0_gradient_apply,
)
from ddg.dg2d.coupled_ns import IterativeSolverConfig


# ---------------------------------------------------------------------------
# Convective term: linearised at u_star with Lax-Friedrichs flux
# ---------------------------------------------------------------------------
#
# Continuous: N(u; u*) = ∇·(u* ⊗ u). For Picard, u* = u (lagged); for
# Newton the Jacobian adds the extra term ∇·(u ⊗ u*) (symmetric in u*, u
# via the chain rule on N(u) = ∇·(u⊗u)). Both are linear in u for fixed
# u*, so they assemble as part of the saddle-point matrix.
#
# Volume term (weak form, IBP): -∫_K (u* ⊗ u) : ∇v dx
#   In index notation: -∫_K (u*_a u_b)(∂_b v_a) dx.
#   For BDM_1 + affine elements: u and u* are linear, ∇v is constant per
#   element; the integrand is degree 2 in (r, s) and Hammer 3-point
#   cubature is exact.
#
# Face term (Lax-Friedrichs): ∫_F F*·v_M dS where
#   F* = 0.5(F_M + F_P) - 0.5 λ (u_P - u_M)
#   F = u* ⊗ u  ->  F·n = (u*·n) u
#   λ = max(|u*_M·n|, |u*_P·n|).
#
# For BDM, u·n is continuous across interior faces (u_M·n = u_P·n), so
# the jump (u_P - u_M) is purely tangential -- the LF dissipation only
# damps the tangential component, as it should.


_HAMMER_R = jnp.array([0.0, -1.0, 0.0])
_HAMMER_S = jnp.array([-1.0, 0.0, 0.0])
_HAMMER_W = jnp.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])


def _bdm1_unsigned_basis_at_hammer_points(V: BDM) -> jax.Array:
    """Unsigned physical basis at the Hammer 3-point quadrature.

    Returns ``(K, Q=3, 6, 2)``. Used by the convective volume term to
    evaluate ``u`` and ``u*`` at quadrature points.
    """
    phi_ref = bdm1_basis_at_points(_HAMMER_R, _HAMMER_S)  # (3, 6, 2)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    return jnp.einsum("kab,qlb->kqla", DF, phi_ref) / J[:, None, None, None]


def bdm1_convective_lam_face(
    V_u: BDM,
    u_star_global: jax.Array,
    *,
    u_bc_global: jax.Array | None = None,
) -> jax.Array:
    """Precompute the LF speed envelope ``lam_face`` from a linearisation
    point. Counterpart of :func:`bdm_k_convective_lam_face` for BDM_1;
    pass the result back as ``lam_face`` to :func:`bdm1_convective_apply`.
    """
    u_star_local = bdm1_gather(V_u.topology, u_star_global)
    fdata = _bdm1_face_quadrature_data(V_u)
    phi_face = fdata["phi_at_face_dofs"]
    n_face = fdata["n_face"]
    K_nbr = fdata["K_nbr"]; f_nbr = fdata["f_nbr"]; q_nbr = fdata["q_nbr"]
    is_boundary = fdata["is_boundary"]
    u_star_M = jnp.einsum("kl,kfqlc->kfqc", u_star_local, phi_face)
    u_star_P = u_star_M[K_nbr[:, :, None], f_nbr[:, :, None], q_nbr]
    u_star_P = jnp.where(is_boundary[:, :, None, None], u_star_M, u_star_P)
    u_star_n_M = jnp.einsum("kfqc,kfc->kfq", u_star_M, n_face)
    u_star_n_P = jnp.einsum("kfqc,kfc->kfq", u_star_P, n_face)
    return jnp.maximum(jnp.abs(u_star_n_M), jnp.abs(u_star_n_P))


def bdm1_convective_apply(
    V_u: BDM,
    u_global: jax.Array,
    *,
    u_star_global: jax.Array,
    u_bc_global: jax.Array | None = None,
    lam_face: jax.Array | None = None,
) -> jax.Array:
    """Apply the linearised convective operator ``∇·(u* ⊗ u)`` with
    Lax-Friedrichs face flux to a global BDM_1 DOF vector.

    For Picard, the caller passes ``u_star_global = u_global``; for
    Newton, the operator is part of the Jacobian applied to a search
    direction ``δu`` with ``u_star_global`` the current outer iterate.

    Boundary handling: when ``u_bc_global is not None``, boundary face
    ghost values use ``u_P = u_bc_global`` (the prescribed Dirichlet
    value); otherwise the operator uses ``u_P = u_M`` (homogeneous
    natural / "outflow" condition, i.e. no incoming flux).
    """
    u_local = bdm1_gather(V_u.topology, u_global)
    u_star_local = bdm1_gather(V_u.topology, u_star_global)
    M_grad = _bdm1_physical_gradient_matrices(V_u)              # (K, 6, 2, 2)
    J = V_u.mesh.J[0]

    # ---- Volume term: -integral_K (u*_a u_b)(d_b v_a) dx.
    # The Hammer cubature on the reference triangle gives
    #     integral_K f dx = J * sum_q w_q f(F_K(r_q, s_q)).
    # The PHYSICAL test gradient is (1/J) M_grad (since M_grad =
    # DF*grad_ref*DF_inv is J-scaled per the convention). The two J
    # factors cancel exactly, so the cubature carries no explicit J:
    #     vol_K[ell] = -sum_q w_q * (u*_a u_b)_q * M_grad[K, ell, a, b].
    # (The earlier version of this code carried an extra explicit J
    # factor -- a bug that broke the integration-by-parts cancellation
    # with the face apply and degraded the convergence rate to slope-1.
    # Fixed in tandem with bdm1_viscous_face_apply.)
    phi_phys_hammer = _bdm1_unsigned_basis_at_hammer_points(V_u)  # (K, Q, 6, 2)
    u_at_q = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys_hammer)
    u_star_at_q = jnp.einsum("kl,kqlc->kqc", u_star_local, phi_phys_hammer)
    vol_local = -jnp.einsum(
        "q,kqa,kqb,klab->kl",
        _HAMMER_W, u_star_at_q, u_at_q, M_grad,
    )                                                           # (K, 6)

    # ---- Face term: Lax-Friedrichs on the tangential trace
    fdata = _bdm1_face_quadrature_data(V_u)
    phi_face = fdata["phi_at_face_dofs"]      # (K, Nfaces, Q=2, 6, 2), unsigned
    n_face = fdata["n_face"]                   # (K, Nfaces, 2)
    edge_jac = fdata["edge_jac"]               # (K, Nfaces)
    K_nbr = fdata["K_nbr"]
    f_nbr = fdata["f_nbr"]
    q_nbr = fdata["q_nbr"]                     # (K, Nfaces, 2)
    is_boundary = fdata["is_boundary"]         # (K, Nfaces)
    w_q_face = fdata["w_q"]                    # (2,)

    # Trace evaluations at face quad points.
    u_M = jnp.einsum("kl,kfqlc->kfqc", u_local, phi_face)        # (K, Nfaces, Q, 2)
    u_star_M = jnp.einsum("kl,kfqlc->kfqc", u_star_local, phi_face)

    # u_P / u_star_P from neighbour, with q-swap for traversal direction.
    u_P = u_M[K_nbr[:, :, None], f_nbr[:, :, None], q_nbr]
    u_star_P = u_star_M[K_nbr[:, :, None], f_nbr[:, :, None], q_nbr]

    # Boundary handling: u_P = u_bc_global trace (Dirichlet) or u_P = u_M
    # (natural). u_star_P always = u_star_M on boundary (Dirichlet u_star
    # ghost = strong; natural = use M-trace).
    if u_bc_global is not None:
        u_bc_local = bdm1_gather(V_u.topology, u_bc_global)
        u_bc_M = jnp.einsum("kl,kfqlc->kfqc", u_bc_local, phi_face)
        u_P = jnp.where(is_boundary[:, :, None, None], u_bc_M, u_P)
        u_star_P = jnp.where(is_boundary[:, :, None, None], u_star_M, u_star_P)
    else:
        u_P = jnp.where(is_boundary[:, :, None, None], u_M, u_P)
        u_star_P = jnp.where(is_boundary[:, :, None, None], u_star_M, u_star_P)

    # u*·n at quadrature points (per face, per quad pt).
    u_star_n_M = jnp.einsum("kfqc,kfc->kfq", u_star_M, n_face)  # (K, Nfaces, Q)
    u_star_n_P = jnp.einsum("kfqc,kfc->kfq", u_star_P, n_face)
    # ``lam`` depends only on u_star; when the caller is a linear
    # matvec for GMRES they pass it in precomputed. See note in
    # bdm_k_convective_apply.
    if lam_face is None:
        lam = jnp.maximum(jnp.abs(u_star_n_M), jnp.abs(u_star_n_P))
    else:
        lam = lam_face                                            # (K, Nfaces, Q)

    # F·n on each side:  (F·n)_a = (u*·n) * u_a    (this is the conservative form).
    F_n_M = u_star_n_M[:, :, :, None] * u_M                      # (K, Nfaces, Q, 2)
    F_n_P = u_star_n_P[:, :, :, None] * u_P
    # LF numerical flux:
    F_star = 0.5 * (F_n_M + F_n_P) - 0.5 * lam[:, :, :, None] * (u_P - u_M)

    # Face contribution to local test ℓ: integral_F F_star · phi_M_ℓ dS.
    face_local = jnp.einsum(
        "q,kf,kfqc,kfqlc->kl",
        w_q_face, edge_jac, F_star, phi_face,
    )

    return bdm1_scatter_add(V_u.topology, vol_local + face_local)


# ---------------------------------------------------------------------------
# Boundary DOF identification + u_BC evaluation
# ---------------------------------------------------------------------------

def bdm1_boundary_dofs_mask(
    V_u: BDM, dirichlet_face: jax.Array | None = None,
) -> jax.Array:
    """Boolean mask of shape ``(V_u.total_dofs,)`` marking BDM_k face
    DOFs on the domain boundary. Order-parametric: at order k, each
    boundary face contributes its ``k+1`` local DOFs to the mask;
    interior (bubble) DOFs are never on the boundary.

    ``dirichlet_face`` (optional ``(K, Nfaces)`` bool): when provided,
    only boundary faces flagged True contribute DOFs to the mask. The
    complement (Natural / free-stress faces) leaves its normal-trace
    DOFs FREE -- the right behaviour for an outflow BC. Default
    (``None``) treats every boundary face as Dirichlet, preserving the
    all-walls cavity / bubble semantics.
    """
    EToE = jnp.asarray(V_u.mesh.EToE)               # (K, Nfaces)
    K_idx = jnp.arange(V_u.K)
    is_boundary_face = EToE == K_idx[:, None]       # (K, Nfaces)
    if dirichlet_face is not None:
        is_boundary_face = is_boundary_face & dirichlet_face
    k = V_u.order
    n_per_face = k + 1
    n_face_total = 3 * n_per_face
    n_interior = (k + 1) * (k - 1)
    bdry_face_local = jnp.repeat(is_boundary_face, n_per_face, axis=1)  # (K, n_face_total)
    bdry_interior_local = jnp.zeros((V_u.K, n_interior), dtype=jnp.bool_)
    bdry_local = jnp.concatenate([bdry_face_local, bdry_interior_local], axis=1)  # (K, Np)
    mask = jnp.zeros(V_u.total_dofs, dtype=jnp.bool_)
    mask = mask.at[V_u.topology.local_to_global.reshape(-1)].max(
        bdry_local.reshape(-1)
    )
    return mask


def bdm1_evaluate_dirichlet_bc(
    V_u: BDM,
    u_bc_fn,
) -> jax.Array:
    """Evaluate a vector-valued Dirichlet boundary condition on BDM_k.

    Order-parametric: ``u_bc_fn(x, y) -> (u_x, u_y)`` is sampled at the
    ``k+1`` face GL points of every boundary face. The DOF at point j
    is ``(u_BC . n_K_out)(GL point j)`` -- a scalar per DOF. Returns a
    global DOF vector with prescribed values on boundary face DOFs
    and zeros on interior face DOFs and interior bubble DOFs.

    Function name is kept for back-compat with k=1 callers; despite
    the prefix it handles arbitrary order.
    """
    from ddg.dg2d.bdm import (
        _bdm_k_face_quadrature_data, bdm_k_face_dof_coordinates,
    )
    fdata = _bdm_k_face_quadrature_data(V_u)
    n_face = fdata["n_face"]                   # (K, Nfaces, 2)
    is_boundary = fdata["is_boundary"]         # (K, Nfaces)
    k = V_u.order
    Q = k + 1
    K = V_u.K

    # Physical positions of face GL points (k+1 per face). Kept in JAX
    # for differentiable-mesh AD through VX/VY.
    face_coords = bdm_k_face_dof_coordinates(k)        # (3 * Q, 2)
    r_dofs = face_coords[:, 0]
    s_dofs = face_coords[:, 1]
    EToV = jnp.asarray(V_u.mesh.EToV)
    VX = jnp.asarray(V_u.mesh.VX)
    VY = jnp.asarray(V_u.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    x_at = (
        Vx0[:, None]
        + 0.5 * (r_dofs[None, :] + 1.0) * (Vx1 - Vx0)[:, None]
        + 0.5 * (s_dofs[None, :] + 1.0) * (Vx2 - Vx0)[:, None]
    )                                                   # (K, 3*Q)
    y_at = (
        Vy0[:, None]
        + 0.5 * (r_dofs[None, :] + 1.0) * (Vy1 - Vy0)[:, None]
        + 0.5 * (s_dofs[None, :] + 1.0) * (Vy2 - Vy0)[:, None]
    )
    x_face = x_at.reshape(K, 3, Q)
    y_face = y_at.reshape(K, 3, Q)

    u_bc_x, u_bc_y = u_bc_fn(x_face, y_face)
    u_bc_x = jnp.asarray(u_bc_x); u_bc_y = jnp.asarray(u_bc_y)
    un_bc_phys = u_bc_x * n_face[:, :, 0:1] + u_bc_y * n_face[:, :, 1:2]
    un_bc_masked = jnp.where(is_boundary[:, :, None], un_bc_phys, 0.0)

    # Reshape to per-element face block (K, 3*Q) and pad with zeros for
    # the interior DOFs (n_interior = (k+1)(k-1)).
    n_face_total = 3 * Q
    n_interior = (k + 1) * (k - 1)
    u_local_face = un_bc_masked.reshape(K, n_face_total)
    u_local_interior = jnp.zeros((K, n_interior))
    u_local = jnp.concatenate([u_local_face, u_local_interior], axis=1)  # (K, Np)

    lg = V_u.topology.local_to_global               # (K, Np)
    u_bc_global = jnp.zeros(V_u.total_dofs)
    u_bc_global = u_bc_global.at[lg.reshape(-1)].set(u_local.reshape(-1))
    return u_bc_global


# ---------------------------------------------------------------------------
# Saddle-point apply
# ---------------------------------------------------------------------------

def stokes_apply_bdm_2d(
    V_u: BDM,
    u_global: jax.Array,
    p_per_elem: jax.Array,
    *,
    nu: float,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
    velocity_alpha: float = 0.0,
    u_bc_global: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Apply the BDM-P_0 Stokes saddle-point operator to ``(u, p)``.

    Returns ``(out_u, out_p)`` matching the input shapes:
        out_u = velocity_alpha * M_u u + ν A_visc u + G p
        out_p = -D u                                 + ε M_p p.

    ``velocity_alpha`` lets the same routine assemble a time-discretised
    velocity Helmholtz block ``(γ/Δt) M_u + ν A_visc`` -- set to
    ``bdf.g0 / dt`` for an unsteady step, leave at 0 for steady Stokes.

    ``pressure_eps`` adds the Brezzi-Pitkäranta mass stabilisation to
    lift the constant-pressure nullspace on closed boundaries; set to a
    small positive number (e.g. 1e-8) when no pressure-pinning BC is
    present, or 0 otherwise. The P_0 element mass is diagonal:
    ``M_p[K, K] = ∫_K 1 dx = |K| = 2 J_K`` on affine triangles.
    """
    # Velocity Helmholtz block
    out_u = nu * bdm1_viscous_apply(
        V_u, u_global, nu=1.0, tau_scale=tau_scale,
        u_bc_global=u_bc_global,
    )
    if velocity_alpha != 0.0:
        # Mass M_u: per-element 6x6 unsigned mass via the standard apply
        # pattern (gather, matvec, scatter).
        M_K_unsigned = bdm1_element_local_mass(V_u)              # (K, 6, 6)
        u_local = bdm1_gather(V_u.topology, u_global)
        Mu_local = jnp.einsum("kij,kj->ki", M_K_unsigned, u_local)
        out_u = out_u + velocity_alpha * bdm1_scatter_add(
            V_u.topology, Mu_local,
        )
    # G p contribution
    out_u = out_u + bdm_p0_gradient_apply(V_u, p_per_elem)

    # Continuity block (sign-flipped divergence as in the equal-order code).
    out_p = -bdm_p0_divergence_apply(V_u, u_global)
    if pressure_eps != 0.0:
        # P_0 mass is diagonal: M_p[K, K] = area(K) = 2 * J_K on affine.
        area_K = 2.0 * V_u.mesh.J[0]                              # (K,)
        out_p = out_p + pressure_eps * area_K * p_per_elem
    return out_u, out_p


# ---------------------------------------------------------------------------
# Preconditioners for the matrix-free Krylov path
# ---------------------------------------------------------------------------
#
# The BDM-P_0 saddle-point is
#     [  alpha M_u + nu A_visc + conv     G    ]
#     [  -D                                S    ]
# where S = eps M_p (Brezzi-Pitkäranta lift only, since BDM-P_0 has
# structural inf-sup; eps is small).
#
# Block-diagonal Murphy-Wathen preconditioner:
#     P_BD = diag(A_v_pc^{-1}, S_pc^{-1})
# with
#  * A_v_pc = block-Jacobi of (alpha M_u + nu A_visc_volume) per element
#    (cheap; ignores SIPG face coupling on velocity but exact on the
#    block-diagonal volume contributions, which dominate at mass-time-
#    dominated regimes).
#  * S_pc = the Cahouet-Chabard mix
#      S^{-1} ~ (1/nu) M_p^{-1}  +  (1/alpha) K_p^{-1}
#    where M_p is the P_0 mass (exactly diagonal: M_p[K,K] = 2 J_K) and
#    K_p is a pressure Laplacian. For BDM-P_0 the simplest K_p
#    approximation is the diagonal mass; richer choices (D M_u^{-1} D^T)
#    are sparse-block.


def make_bdm_velocity_block_jacobi_pc(
    V_u: BDM, *, alpha: float, nu: float,
):
    """Per-element block-Jacobi PC for the BDM velocity Helmholtz block.

    Inverts ``alpha * M_K + nu * S_K_vol`` per element (6x6 symmetric
    SPD). The viscous *face* (SIPG IPG penalty / consistency / symmetry)
    couples elements through shared boundary DOFs -- this PC ignores
    that coupling. Adequate when mass and viscous-volume dominate
    (low-to-moderate Re; large time-mass coefficient alpha = gamma0/dt).

    Returns a callable ``M_inv(u_global) -> v_global`` operating on the
    Krylov-flat velocity sub-vector.
    """
    M_K = bdm1_element_local_mass(V_u)                       # (K, 6, 6)
    S_K = bdm1_viscous_volume_stiffness(V_u, nu=1.0)         # (K, 6, 6)
    A_K = alpha * M_K + nu * S_K                              # (K, 6, 6)
    A_K_inv = jnp.linalg.inv(A_K)                             # (K, 6, 6)

    def M_inv(u_global: jax.Array) -> jax.Array:
        u_local = bdm1_gather(V_u.topology, u_global)
        out_local = jnp.einsum("kij,kj->ki", A_K_inv, u_local)
        return bdm1_scatter_add(V_u.topology, out_local)

    return M_inv


def make_bdm_pressure_mass_inverse(
    V_u: BDM, *, nu: float, scale: float = 1.0,
):
    """Diagonal Schur PC: ``S^{-1} ~ scale * (1/nu) * M_p^{-1}`` --
    Cahouet-Chabard viscous-limit on the BDM-P_0 pressure space.

    For BDM-P_0 the pressure mass is *exactly* diagonal,
    ``M_p[K, K] = 2 J_K`` (the P_0 mass on an affine triangle).
    ``M_p^{-1}`` is therefore one division per element -- the cheapest
    possible Schur PC.
    """
    J_per_elem = V_u.mesh.J[0]                                # (K,)
    factor = scale / (nu * 2.0 * J_per_elem)                  # (K,)

    def M_inv(p_per_elem: jax.Array) -> jax.Array:
        return factor * p_per_elem

    return M_inv


def make_bdm_pressure_laplacian_inverse(
    V_u: BDM, *, scale: float = 1.0,
):
    """Schur PC unsteady limit: ``S^{-1} ~ scale * K_p^{-1}`` where
    ``K_p = D M_u_diag^{-1} D^T`` is the pressure-Laplacian approximated
    via the *diagonal* velocity mass.

    For BDM-P_0, the divergence ``D`` is element-local: each pressure
    test (one per element) is paired with the 6 velocity DOFs of that
    element. ``K_p`` is then SPARSE -- one row per element with at
    most 4 column neighbours via the edge-shared DOFs. We assemble it
    densely here (small K); for larger meshes, switch to a sparse
    factorisation.

    Use this PC when ``alpha = gamma0/dt`` is large (small timestep);
    pair with :func:`make_bdm_pressure_mass_inverse` for steady /
    large-Δt regimes. ``scale`` lets the caller supply ``Δt/gamma0`` to
    get the proper dimensional unsteady-limit weighting.
    """
    # Build K_p = D M_u_diag^{-1} D^T densely.
    n_p = V_u.K

    # Diagonal velocity mass inverse: diag(M_u) ≈ trace per row of the
    # block-diagonal element mass. For BDM each global DOF's diagonal
    # entry is sum_K (sign^2 piola^2 M_K_unsigned[ell, ell]) over the
    # (at most 2) elements that share it. Compute via scatter.
    M_K = bdm1_element_local_mass(V_u)                        # (K, 6, 6)
    diag_local = jnp.diagonal(M_K, axis1=1, axis2=2)           # (K, 6)
    diag_global = bdm1_scatter_add(V_u.topology, diag_local)   # (n_u,)
    # Avoid division by zero on unused global indices.
    inv_diag_global = jnp.where(
        diag_global > 1e-30, 1.0 / diag_global, 0.0,
    )

    # Build D and apply D inv(M_diag) D^T column-by-column.
    def apply_D(u_global):
        return bdm_p0_divergence_apply(V_u, u_global)

    def apply_Dt(p_per_elem):
        # G = -D^T, so D^T = -G.
        return -bdm_p0_gradient_apply(V_u, p_per_elem)

    def K_p_apply(p_per_elem):
        v = apply_Dt(p_per_elem)                              # (n_u,)
        return apply_D(inv_diag_global * v)                   # (K,)

    # Dense assembly via vmap on canonical basis (K columns).
    K_p = jax.vmap(
        lambda i: K_p_apply(jnp.eye(n_p)[i])
    )(jnp.arange(n_p)).T                                       # (K, K)

    # K_p has a constant-pressure nullspace on closed-boundary problems;
    # add a tiny regulariser before factorisation.
    K_p_reg = K_p + 1e-12 * jnp.eye(n_p)
    lu, piv = jax.scipy.linalg.lu_factor(K_p_reg)

    def M_inv(p_per_elem: jax.Array) -> jax.Array:
        return scale * jax.scipy.linalg.lu_solve((lu, piv), p_per_elem)

    return M_inv


def bdm1_mass_apply(V_u: BDM, u_global: jax.Array) -> jax.Array:
    """Apply the BDM_1 velocity mass matrix ``M_u`` to a global DOF vector.

    Used by the BDF time-stepper to build the history term
    ``f_velocity = (gamma0 / dt) M u_n``. Cheap: per-element 6x6 mass
    times the gather-scatter-applied DOFs.
    """
    M_K = bdm1_element_local_mass(V_u)                   # (K, 6, 6)
    u_local = bdm1_gather(V_u.topology, u_global)
    out_local = jnp.einsum("kij,kj->ki", M_K, u_local)
    return bdm1_scatter_add(V_u.topology, out_local)


# BDF time-stepping coefficients. The implicit time-derivative
# discretisation is
#     (gamma0/dt) u^{n+1} - sum_i (beta_i/dt) u^{n-i} = RHS,
# so the solver sees velocity_alpha = gamma0/dt on the LHS and
# f_velocity = sum_i (beta_i/dt) M u^{n-i} on the RHS. Order 2 needs a
# bootstrap step at order 1 because u^{n-1} doesn't yet exist.
BDF_COEFFS = {
    1: {"gamma0": 1.0, "beta": (1.0,)},
    2: {"gamma0": 1.5, "beta": (2.0, -0.5)},
}


def bdf_history(
    V_u: BDM,
    u_hist,
    *,
    dt: float,
    order: int,
) -> jax.Array:
    """Build the BDF history vector ``f_velocity = sum_i (beta_i/dt) M u^{n-i}``.

    ``u_hist[0]`` is the most recent state ``u^n``, ``u_hist[1]`` is
    ``u^{n-1}``, etc. Caller must supply at least ``order`` past states.
    """
    coeffs = BDF_COEFFS[order]
    betas = coeffs["beta"]
    if len(u_hist) < order:
        raise ValueError(
            f"BDF{order} needs {order} history states, got {len(u_hist)}"
        )
    f = (betas[0] / dt) * bdm1_mass_apply(V_u, u_hist[0])
    for i in range(1, order):
        f = f + (betas[i] / dt) * bdm1_mass_apply(V_u, u_hist[i])
    return f


def make_bdm_block_diag_pc(
    V_u: BDM, *,
    velocity_pc, pressure_pc,
):
    """Murphy-Wathen block-diagonal coupled PC.

    Combines ``velocity_pc`` (acting on the n_u velocity DOFs) and
    ``pressure_pc`` (acting on the K pressure DOFs) into a single
    coupled apply operating on a Krylov-flat ``(u, p)`` vector of size
    ``n_u + K``. Block-diagonal: no off-diagonal coupling.

    Pass to :func:`ns_newton_solve_bdm_2d` via ``coupled_preconditioner=``.
    """
    n_u = V_u.total_dofs
    n_p = V_u.K

    def M_inv(v_flat: jax.Array) -> jax.Array:
        v_u = v_flat[:n_u]
        v_p = v_flat[n_u:n_u + n_p]
        out_u = velocity_pc(v_u)
        out_p = pressure_pc(v_p)
        return jnp.concatenate([out_u, out_p])

    return M_inv


def make_bdm_cahouet_chabard_pc(
    V_u: BDM, *, nu: float, dt: float, gamma0: float,
):
    """Full Cahouet-Chabard combined Schur PC for BDM-P_0:

        S^{-1} ~ (1/nu) M_p^{-1} + (dt/gamma0) K_p^{-1}.

    Covers both the viscous (large dt) and unsteady (small dt) limits.
    Recommended default Schur PC for the BDM coupled solver.
    """
    mass_pc = make_bdm_pressure_mass_inverse(V_u, nu=nu, scale=1.0)
    lapl_pc = make_bdm_pressure_laplacian_inverse(
        V_u, scale=dt / gamma0,
    )

    def M_inv(p_per_elem: jax.Array) -> jax.Array:
        return mass_pc(p_per_elem) + lapl_pc(p_per_elem)

    return M_inv


# ---------------------------------------------------------------------------
# Order-parametric block preconditioners (BDM_k / P_{k-1})
# ---------------------------------------------------------------------------
#
# The P_0 PCs above hardcode a single pressure DOF per element. These
# generalise to arbitrary k: the velocity block-Jacobi inverts the
# per-element (Np x Np) Helmholtz block, and the pressure-mass Schur PC
# inverts the per-element (dim_p x dim_p) monomial Gram matrix. Both are
# block-diagonal (element-local), so building/inverting them is cheap.


def make_bdm_k_velocity_block_jacobi_pc(
    V_u: BDM, *, nu: float, alpha: float = 0.0, tau_scale: float = 4.0,
    gamma_al: float = 0.0,
):
    """Per-element IPG block-Jacobi PC for the BDM_k velocity block.

    Inverts the full per-element viscous Helmholtz block

        A_K = alpha * M_K + nu * S_K_vol + (SIPG face self-block)
            + gamma_al * D_K^T M_p_K^{-1} D_K,

    where the face self-block is the element-isolated penalty +
    consistency + symmetry (:func:`bdm_k_viscous_face_self_block`).
    Including the face penalty is essential for the *steady* problem:
    the volume stiffness alone has a per-element rigid nullspace and is
    a poor approximation of the true velocity block (the penalty term
    dominates), so a volume-only block-Jacobi preconditions badly and
    GMRES stalls. The penalty makes ``A_K`` SPD even at ``alpha = 0``;
    ``alpha > 0`` adds a mass floor (e.g. ``gamma0/dt`` for transient).
    The convective coupling is omitted (it is not symmetric and the PC
    need not be exact).

    ``gamma_al > 0`` adds the augmented-Lagrangian grad-div
    stabilisation per element. Since D, M_p are element-local on BDM,
    this is a pure per-element matrix update.
    """
    from ddg.dg2d.bdm import (
        bdm_k_element_local_mass, bdm_k_viscous_volume_stiffness,
        bdm_k_viscous_face_self_block, bdm1_gather, bdm1_scatter_add,
        _bdm_k_reference_divergence_matrix, bdm_k_pkm1_basis_at_points,
    )
    S_vol = bdm_k_viscous_volume_stiffness(V_u, nu=nu)            # (K, Np, Np)
    S_face = bdm_k_viscous_face_self_block(
        V_u, nu=nu, tau_scale=tau_scale)                          # (K, Np, Np)
    A_K = S_vol + S_face
    if alpha != 0.0:
        A_K = A_K + alpha * bdm_k_element_local_mass(V_u)
    if gamma_al != 0.0:
        from ddg.dg2d.cubature import triangle_cubature
        k = V_u.order
        r_q, s_q, w_q = triangle_cubature(max(2 * (k - 1), 1))
        q_at = bdm_k_pkm1_basis_at_points(k, r_q, s_q)
        M_p_ref_inv = jnp.linalg.inv(
            jnp.einsum("q,qm,qn->mn", w_q, q_at, q_at))           # (dim_p, dim_p)
        D_ref = _bdm_k_reference_divergence_matrix(k)             # (dim_p, Np)
        # AL_K = gamma * D_ref^T @ (M_p_ref_inv / J_K) @ D_ref
        AL_ref = D_ref.T @ M_p_ref_inv @ D_ref                    # (Np, Np)
        inv_J = 1.0 / V_u.mesh.J[0]                               # (K,)
        AL_K = gamma_al * inv_J[:, None, None] * AL_ref[None]
        A_K = A_K + AL_K
    A_K_inv = jnp.linalg.inv(A_K)                                 # (K, Np, Np)

    def M_inv(u_global: jax.Array) -> jax.Array:
        u_local = bdm1_gather(V_u.topology, u_global)
        out_local = jnp.einsum("kij,kj->ki", A_K_inv, u_local)
        return bdm1_scatter_add(V_u.topology, out_local)

    return M_inv


def make_bdm_k_pressure_mass_inverse(V_u: BDM, *, nu: float, scale: float = 1.0):
    """Diagonal-block Schur PC ``S^{-1} ~ scale * (1/nu) * M_p^{-1}`` on
    the P_{k-1} pressure space (Cahouet-Chabard viscous limit).

    The pressure DOFs are monomial coefficients, so the pressure mass is
    the monomial Gram matrix ``M_p_K[m, n] = J_K * integral_T q_m q_n``,
    block-diagonal across elements. ``M_p_ref`` is the same for every
    affine element; only ``J_K`` differs, so we invert the reference Gram
    once and scale by ``1/J_K``. For k=1 this reduces to
    ``1/(2 J_K)`` -- identical to :func:`make_bdm_pressure_mass_inverse`.
    """
    from ddg.dg2d.bdm import bdm_k_pkm1_basis_at_points
    from ddg.dg2d.cubature import triangle_cubature
    k = V_u.order
    r_q, s_q, w_q = triangle_cubature(max(2 * (k - 1), 1))
    q_at = bdm_k_pkm1_basis_at_points(k, r_q, s_q)       # (Q, dim_p)
    M_p_ref_inv = jnp.linalg.inv(
        jnp.einsum("q,qm,qn->mn", w_q, q_at, q_at))      # (dim_p, dim_p)
    inv_J = 1.0 / V_u.mesh.J[0]                           # (K,)

    def M_inv(p_per_elem: jax.Array) -> jax.Array:        # (K, dim_p)
        out = jnp.einsum("mn,kn->km", M_p_ref_inv, p_per_elem)
        return (scale / nu) * out * inv_J[:, None]

    return M_inv


def make_bdm_k_pressure_laplacian_inverse(
    V_u: BDM, *, scale: float = 1.0,
):
    """Schur PC unsteady limit at arbitrary order: ``S^{-1} ~ scale *
    K_p^{-1}`` where ``K_p = D M_u_diag^{-1} D^T`` is the pressure
    Laplacian built on the order-parametric BDM_k / P_{k-1} pair.
    Order-parametric generalisation of
    :func:`make_bdm_pressure_laplacian_inverse` (BDM-P_0 only); the
    matrix size grows with the pressure DOF count ``K * dim_p``,
    ``dim_p = k(k+1)/2``.

    Build cost: assembles K_p densely via one
    ``jax.vmap`` on the (K * dim_p) pressure basis, then LU. Amortised
    over time steps that reuse the same ``scale`` (i.e., ``dt /
    gamma0``); only re-build when the BDF order or ``dt`` change.

    Use in transient BDM_k coupled solvers where ``alpha = gamma_0 /
    dt`` is large; pair with :func:`make_bdm_k_pressure_mass_inverse`
    for the combined Cahouet-Chabard PC via
    :func:`make_bdm_k_cahouet_chabard_pc`.
    """
    from ddg.dg2d.bdm import (
        bdm_k_divergence_apply, bdm_k_gradient_apply, bdm_k_element_local_mass,
        bdm1_scatter_add,
    )
    k = V_u.order
    dim_p = k * (k + 1) // 2
    n_p = V_u.K * dim_p

    # Diagonal-of-mass inverse with a relative-magnitude clamp. The
    # standard Cahouet-Chabard form is K_p = D M_diag^{-1} D^T; at
    # BDM_k the per-DOF diagonal of the assembled mass varies widely
    # (face DOFs vs interior bubble DOFs), and a literal 1/diag with
    # an absolute floor (1e-30) lets machine-epsilon-positive entries
    # pass through, producing ``inv_diag`` values around 10^15 that
    # poison the LU. A relative floor (1e-10 * max(diag)) drops those
    # phantom entries without affecting well-resolved DOFs.
    #
    # NOTE: this is a hardening, not a fix. The fundamental issue at
    # order > 1 is that K_p = D M_diag^{-1} D^T does not yield a
    # well-conditioned pressure-Laplacian approximation -- the proper
    # k>1 implementation is a P_{k-1} SIPG-DG pressure Laplacian
    # assembled as a real bilinear form (deferred).
    M_K = bdm_k_element_local_mass(V_u)                  # (K, Np, Np)
    diag_local = jnp.diagonal(M_K, axis1=1, axis2=2)     # (K, Np)
    diag_global = bdm1_scatter_add(V_u.topology, diag_local)
    diag_threshold = 1e-10 * float(jnp.max(diag_global))
    inv_diag_global = jnp.where(
        diag_global > diag_threshold, 1.0 / diag_global, 0.0,
    )

    def K_p_apply(p_per_elem):
        # K_p p = D ( M_u_lumped^{-1} ( -G p ) ).
        v = -bdm_k_gradient_apply(V_u, p_per_elem)      # (n_u,) == D^T p
        return bdm_k_divergence_apply(V_u, inv_diag_global * v)

    # Dense assembly: one column per pressure DOF. At BDM_3 K=1000
    # this is a 6000 x 6000 dense matrix (~280 MB f64), ~1 min to
    # build + factor on CPU. Acceptable as a per-PC-rebuild cost.
    def column(idx):
        p_basis = jnp.zeros(n_p).at[idx].set(1.0).reshape(V_u.K, dim_p)
        out = K_p_apply(p_basis)
        return out.reshape(-1)

    K_p_dense = jax.vmap(column)(jnp.arange(n_p)).T      # (n_p, n_p)
    # The pressure Poisson is singular on a closed Neumann domain
    # (constant nullspace). Brezzi-Pitkaranta-style lift: add a tiny
    # diagonal so LU works. On mixed-BC problems (cylinder etc.) the
    # outflow pins pressure so this isn't strictly needed, but the
    # epsilon protects the LU stability either way.
    K_p_dense = K_p_dense + 1e-12 * jnp.eye(n_p)
    K_p_lu = jax.scipy.linalg.lu_factor(K_p_dense)

    def M_inv(p_per_elem: jax.Array) -> jax.Array:
        p_flat = p_per_elem.reshape(-1)
        x = jax.scipy.linalg.lu_solve(K_p_lu, p_flat)
        return scale * x.reshape(V_u.K, dim_p)

    return M_inv


def make_bdm_k_cahouet_chabard_pc(
    V_u: BDM, *, nu: float, dt: float, gamma0: float,
    sipg_sigma: float = 10.0, legacy_kp: bool = False,
):
    """Full Cahouet-Chabard Schur PC at arbitrary order:

        S^{-1} ~ (1/nu) * K_p^{-1} + (dt / gamma0) * M_p^{-1}

    matching the limiting cases of the discrete Schur
    ``S = nu L_p + (gamma0/dt) M_p``: K_p^{-1} dominates in steady
    Stokes, M_p^{-1} dominates in the transient/mass-dominated limit.

    Order-parametric generalisation of
    :func:`make_bdm_cahouet_chabard_pc`. Combines the viscous Laplace
    inverse and the mass inverse. By default uses the proper P_{k-1}
    SIPG-DG Laplacian
    (:func:`make_bdm_k_sipg_pressure_laplacian_inverse`), which is
    well-conditioned at every order; pass ``legacy_kp=True`` to use
    the ``D M_diag^{-1} D^T`` approximation
    (:func:`make_bdm_k_pressure_laplacian_inverse`) for back-
    compatibility / debugging.

    Use as the pressure block in :func:`make_bdm_k_block_diag_pc` for
    the transient BDM_k coupled solver.
    """
    # ``make_bdm_k_pressure_mass_inverse`` returns ``(scale/nu) * M_p^{-1}``.
    # We want ``(dt/gamma0) * M_p^{-1}`` for the transient term, so we
    # set its ``nu`` argument to ``gamma0/dt`` (a bit hacky but avoids
    # changing the existing API).
    mass_pc = make_bdm_k_pressure_mass_inverse(
        V_u, nu=gamma0 / dt, scale=1.0,
    )
    if legacy_kp:
        lapl_pc = make_bdm_k_pressure_laplacian_inverse(
            V_u, scale=1.0 / nu,
        )
    else:
        lapl_pc = make_bdm_k_sipg_pressure_laplacian_inverse(
            V_u, scale=1.0 / nu, sigma=sipg_sigma,
        )

    def M_inv(p_per_elem: jax.Array) -> jax.Array:
        return mass_pc(p_per_elem) + lapl_pc(p_per_elem)

    return M_inv


def make_bdm_k_sipg_pressure_laplacian_inverse(
    V_u: BDM, *, scale: float = 1.0, sigma: float = 10.0,
):
    """Schur PC unsteady limit at arbitrary order: ``S^{-1} ~ scale *
    K_p^{-1}`` where K_p is the **SIPG-DG Laplacian on P_{k-1}**:

        a_K_p(p, q) = sum_K int_K grad(p) . grad(q) dx
                    + sum_{F interior} int_F (
                        -{grad(p).n}[q] - {grad(q).n}[p] + (sigma/h_F)[p][q]
                      ) dS

    where n is the M-to-P normal across F, {.} the average, [.] the jump.
    Affine-element assembly: uses constant J_K, invJ_K, h_F per element/
    face (the cylinder mesh has a handful of curved annular elements;
    those get a slightly-off PC, which is acceptable for preconditioning).

    Replaces :func:`make_bdm_k_pressure_laplacian_inverse` (the
    ``D M_diag^{-1} D^T`` approximation that is ill-conditioned at
    order > 1). At BDM_1, both K_p forms differ from each other but
    both work as PCs; the SIPG form is the production-aligned choice
    at all orders.
    """
    from ddg.dg2d.bdm import (
        bdm_k_pkm1_basis_at_points, bdm_k_pkm1_basis_grad_at_points,
    )
    from ddg.dg2d.cubature import triangle_cubature

    k = V_u.order
    dim_p = k * (k + 1) // 2
    K = V_u.K
    n_p = K * dim_p
    mesh = V_u.mesh

    # ---- Volume term: per-element block J * G^{-1}_{ab} S_{ab}[m, n] ----
    deg_vol = max(2 * (k - 1), 1)
    r_vq, s_vq, w_vq = triangle_cubature(deg_vol)
    grad_v = np.asarray(
        bdm_k_pkm1_basis_grad_at_points(k, r_vq, s_vq),
    )                                                    # (Q, dim_p, 2)
    w_vq_np = np.asarray(w_vq)
    # S[a, b, m, n] = sum_q w_q (∂_a q_m)(∂_b q_n)
    S = np.einsum("q,qma,qnb->abmn", w_vq_np, grad_v, grad_v)

    rx = np.asarray(mesh.rx[0]); sx = np.asarray(mesh.sx[0])
    ry = np.asarray(mesh.ry[0]); sy = np.asarray(mesh.sy[0])
    J_per = np.asarray(mesh.J[0])                        # (K,)
    G00 = rx * rx + ry * ry
    G01 = rx * sx + ry * sy
    G11 = sx * sx + sy * sy

    # Volume blocks: shape (K, dim_p, dim_p)
    K_vol = (
        J_per[:, None, None] * (
            G00[:, None, None] * S[0, 0]
            + G01[:, None, None] * (S[0, 1] + S[1, 0])
            + G11[:, None, None] * S[1, 1]
        )
    )

    # ---- Face term: precompute reference traces per face index ----
    n_gl = max(k + 1, 2)
    xi_gl, w_gl = np.polynomial.legendre.leggauss(n_gl)

    def _face_ref_pts(f: int, xi: np.ndarray):
        # Hesthaven parameterisation, xi in [-1, 1]; matches
        # bdm_k_face_dof_coordinates.
        if f == 0:
            return xi, -np.ones_like(xi)                 # s = -1
        if f == 1:
            return -xi, xi                                # r + s = 0
        if f == 2:
            return -np.ones_like(xi), -xi                 # r = -1
        raise ValueError(f)

    q_face = []                                          # [f][Q, dim_p]
    grad_face = []                                       # [f][Q, dim_p, 2]
    for f in range(3):
        r_f, s_f = _face_ref_pts(f, xi_gl)
        q_face.append(np.asarray(
            bdm_k_pkm1_basis_at_points(k, jnp.asarray(r_f), jnp.asarray(s_f)),
        ))
        grad_face.append(np.asarray(
            bdm_k_pkm1_basis_grad_at_points(
                k, jnp.asarray(r_f), jnp.asarray(s_f),
            ),
        ))

    # Per-face geometry (affine: take first nodal entry per face).
    nx_arr = np.asarray(mesh.nx); ny_arr = np.asarray(mesh.ny)
    sJ_arr = np.asarray(mesh.sJ)
    Nfp = mesh.Nfp
    nx_KF = nx_arr.reshape(3, Nfp, K).transpose(2, 0, 1)[:, :, 0]  # (K, 3)
    ny_KF = ny_arr.reshape(3, Nfp, K).transpose(2, 0, 1)[:, :, 0]
    sJ_KF = sJ_arr.reshape(3, Nfp, K).transpose(2, 0, 1)[:, :, 0]
    # Reference edge spans 2 (xi in [-1, 1]); physical face length = 2 sJ.
    h_F = 2.0 * sJ_KF                                    # (K, 3)

    EToE = np.asarray(mesh.EToE)                         # (K, 3)
    EToF = np.asarray(mesh.EToF)                         # (K, 3)

    K_p_full = np.zeros((n_p, n_p))
    # Volume blocks.
    for kk in range(K):
        i0 = kk * dim_p
        K_p_full[i0:i0 + dim_p, i0:i0 + dim_p] += K_vol[kk]

    # Face blocks. Each interior face visited once.
    visited = set()
    for kM in range(K):
        for fM in range(3):
            kP = int(EToE[kM, fM]); fP = int(EToF[kM, fM])
            if kP == kM:
                continue                                   # boundary
            key = (min((kM, fM), (kP, fP)), max((kM, fM), (kP, fP)))
            if key in visited:
                continue
            visited.add(key)

            wsJ = w_gl * sJ_KF[kM, fM]                   # (Q,)
            nvec = np.array([nx_KF[kM, fM], ny_KF[kM, fM]])
            inv_h = sigma / h_F[kM, fM]

            # M-side traces (use fM as-is).
            q_M = q_face[fM]                             # (Q, dim_p)
            grad_M_ref = grad_face[fM]                   # (Q, dim_p, 2)
            invJ_M = np.array(
                [[rx[kM], sx[kM]], [ry[kM], sy[kM]]],
            )                                            # row i: dxi/dx_i row
            grad_M_x = grad_M_ref @ invJ_M.T             # (Q, dim_p, 2)
            ngrad_M = grad_M_x @ nvec                    # (Q, dim_p)

            # P-side traces. Same physical points correspond to face fP
            # of K_P, but the edge is traversed in the opposite
            # direction -> reverse the xi ordering.
            r_P, s_P = _face_ref_pts(fP, xi_gl[::-1])
            q_P = np.asarray(bdm_k_pkm1_basis_at_points(
                k, jnp.asarray(r_P), jnp.asarray(s_P),
            ))                                           # (Q, dim_p)
            grad_P_ref = np.asarray(bdm_k_pkm1_basis_grad_at_points(
                k, jnp.asarray(r_P), jnp.asarray(s_P),
            ))                                           # (Q, dim_p, 2)
            invJ_P = np.array(
                [[rx[kP], sx[kP]], [ry[kP], sy[kP]]],
            )
            grad_P_x = grad_P_ref @ invJ_P.T
            ngrad_P = grad_P_x @ nvec                    # using SAME n (M->P)

            # Quadrature primitives.
            qq_MM = np.einsum("q,qa,qb->ab", wsJ, q_M, q_M)
            qq_MP = np.einsum("q,qa,qb->ab", wsJ, q_M, q_P)
            qq_PM = np.einsum("q,qa,qb->ab", wsJ, q_P, q_M)
            qq_PP = np.einsum("q,qa,qb->ab", wsJ, q_P, q_P)
            ng_MM = np.einsum("q,qa,qb->ab", wsJ, ngrad_M, q_M)
            ng_MP = np.einsum("q,qa,qb->ab", wsJ, ngrad_M, q_P)
            ng_PM = np.einsum("q,qa,qb->ab", wsJ, ngrad_P, q_M)
            ng_PP = np.einsum("q,qa,qb->ab", wsJ, ngrad_P, q_P)

            # Block convention: rows = test (q) side, cols = trial (p) side.
            # Off-diagonal: (M, P) gets terms from (p_P, q_M); the two
            # consistency contributions land on ng_MP (q test grad,
            # symmetry) and ng_PM.T (p trial grad, consistency).
            blk_MM = -0.5 * (ng_MM + ng_MM.T) + inv_h * qq_MM
            blk_MP = +0.5 * ng_MP - 0.5 * ng_PM.T - inv_h * qq_MP
            blk_PM = blk_MP.T
            blk_PP = +0.5 * (ng_PP + ng_PP.T) + inv_h * qq_PP

            iM = kM * dim_p; iP = kP * dim_p
            K_p_full[iM:iM + dim_p, iM:iM + dim_p] += blk_MM
            K_p_full[iM:iM + dim_p, iP:iP + dim_p] += blk_MP
            K_p_full[iP:iP + dim_p, iM:iM + dim_p] += blk_PM
            K_p_full[iP:iP + dim_p, iP:iP + dim_p] += blk_PP

    # Closed-domain Neumann nullspace lift. K_p has a 1D nullspace
    # (constant pressure: monomial-0 coef = 1 in every element). The
    # 1e-12 regularisation lets LU work, but inverts to ~1e12 in the
    # nullspace direction -- catastrophic if residuals carry any
    # spurious constant. We project the constant mode out of the
    # input and the result.
    K_p_full = K_p_full + 1e-12 * np.eye(n_p)
    K_p_lu = jax.scipy.linalg.lu_factor(jnp.asarray(K_p_full))

    def M_inv(p_per_elem: jax.Array) -> jax.Array:
        # Project constant pressure out.
        mean0 = jnp.mean(p_per_elem[:, 0])
        p_proj = p_per_elem.at[:, 0].add(-mean0)
        x = jax.scipy.linalg.lu_solve(
            K_p_lu, p_proj.reshape(-1),
        ).reshape(K, dim_p)
        x = x.at[:, 0].add(-jnp.mean(x[:, 0]))
        return scale * x

    return M_inv


def make_bdm_k_iterative_inner_solvers(
    V_u: BDM, *,
    iterative: "IterativeSolverConfig",
    coupled_preconditioner,
    nu: float,
    pressure_eps: float = 1.0e-10,
    tau_scale: float = 4.0,
    velocity_alpha: float = 0.0,
    is_natural_face=None,
    bc_mask=None,
    u_bc_global=None,
    pressure_pin=None,
    project_pressure_mean: bool = False,
    gamma_al: float = 0.0,
    build_newton: bool = True,
):
    """Construct the JIT'd Picard / Newton inner-iterative-solvers once.

    Returns ``(inner_solve_picard, inner_solve_newton)`` where each is a
    ``@jax.jit`` function of ``(u_k, rhs_full)`` returning
    ``(delta, r_final)``. Pass them into successive calls of
    :func:`ns_newton_solve_bdm_k_2d` (via the ``inner_solvers`` kwarg)
    so the gmres+matvec+PC trace cache survives across BDF time steps
    -- the canonical "build once, reuse" pattern for matrix-free
    transient INS solvers.

    All non-state arguments (mesh, nu, alpha, PC, etc.) must remain
    constant across the lifetime of the returned solvers; any change
    triggers a JIT retrace on the first call after the change.
    """
    from ddg.dg2d.bdm import (
        bdm_k_convective_apply, bdm_k_convective_lam_face,
    )
    n_u = V_u.total_dofs
    dim_p = V_u.order * (V_u.order + 1) // 2
    n_p = V_u.K * dim_p

    def _build(_use_newton: bool):
        @jax.jit
        def _solve(u_k_arg, rhs_full_arg):
            lam_face_uk = bdm_k_convective_lam_face(
                V_u, u_k_arg, u_bc_global=u_bc_global,
                is_natural_face=is_natural_face,
            )

            def linear_matvec(v_flat):
                du_raw = v_flat[:n_u]
                dp_flat = v_flat[n_u:n_u + n_p]
                dp = dp_flat.reshape(V_u.K, dim_p)
                du = jnp.where(bc_mask, 0.0, du_raw)
                out_u, out_p = ns_picard_apply_bdm_k_2d(
                    V_u, du, dp,
                    u_star_global=u_k_arg, nu=nu,
                    pressure_eps=pressure_eps, tau_scale=tau_scale,
                    velocity_alpha=velocity_alpha,
                    is_natural_face=is_natural_face,
                    lam_face=lam_face_uk,
                    pressure_pin=pressure_pin,
                    project_pressure_mean=project_pressure_mean,
                    gamma_al=gamma_al,
                )
                if _use_newton:
                    out_u = out_u + bdm_k_convective_apply(
                        V_u, u_k_arg, u_star_global=du,
                        is_natural_face=is_natural_face,
                        lam_face=lam_face_uk,
                    )
                out_u = jnp.where(bc_mask, du_raw, out_u)
                return jnp.concatenate([out_u, out_p.reshape(-1)])

            x0 = jnp.zeros_like(rhs_full_arg)
            if iterative.method == "gmres":
                delta_, _info = jax.scipy.sparse.linalg.gmres(
                    linear_matvec, rhs_full_arg, x0=x0,
                    tol=iterative.tol, atol=iterative.atol,
                    maxiter=iterative.maxiter,
                    restart=iterative.restart,
                    M=coupled_preconditioner,
                )
            else:
                delta_, _info = jax.scipy.sparse.linalg.bicgstab(
                    linear_matvec, rhs_full_arg, x0=x0,
                    tol=iterative.tol, atol=iterative.atol,
                    maxiter=iterative.maxiter,
                    M=coupled_preconditioner,
                )
            r_final_ = jnp.linalg.norm(
                rhs_full_arg - linear_matvec(delta_)
            )
            return delta_, r_final_
        return _solve

    inner_picard = _build(False)
    inner_newton = _build(True) if build_newton else None
    return inner_picard, inner_newton


def make_bdm_k_block_diag_pc(V_u: BDM, *, velocity_pc, pressure_pc):
    """Murphy-Wathen block-diagonal coupled PC for BDM_k / P_{k-1}.

    Like :func:`make_bdm_block_diag_pc` but the pressure sub-vector has
    ``K * dim_p`` entries (``dim_p = k(k+1)/2``) instead of ``K``.
    Operates on a Krylov-flat ``(u, p)`` vector of size
    ``n_u + K * dim_p``.
    """
    n_u = V_u.total_dofs
    dim_p = V_u.order * (V_u.order + 1) // 2
    n_p = V_u.K * dim_p

    def M_inv(v_flat: jax.Array) -> jax.Array:
        v_u = v_flat[:n_u]
        v_p = v_flat[n_u:n_u + n_p].reshape(V_u.K, dim_p)
        out_u = velocity_pc(v_u)
        out_p = pressure_pc(v_p).reshape(-1)
        return jnp.concatenate([out_u, out_p])

    return M_inv


def make_bdm_k_block_triangular_pc(
    V_u: BDM, *, velocity_pc, pressure_pc, bc_mask=None,
):
    """Block-upper-triangular saddle PC for BDM_k / P_{k-1}.

    The PC inverts (in order):
      1. pressure block: ``v_p_pc = pressure_pc(v_p)``
      2. velocity block with pressure-coupling correction:
         ``v_u_pc = velocity_pc(v_u - B^T v_p_pc)``
    Equivalent to applying the inverse of the block-upper-triangular
    factor ``[A B^T; 0 S]`` of the saddle. In primitives, ``B^T = G``
    (gradient), so the correction is ``v_u - G(v_p_pc)``.

    Pairs well with the augmented-Lagrangian transformation: the
    upper-tri PC with AL gives K-independent iteration counts per
    Benzi-Olshanskii. When ``bc_mask`` is given, the velocity input
    correction is zero'd on boundary DOFs (identity rows are
    unaffected).
    """
    from ddg.dg2d.bdm import bdm_k_gradient_apply
    n_u = V_u.total_dofs
    dim_p = V_u.order * (V_u.order + 1) // 2
    n_p = V_u.K * dim_p

    def M_inv(v_flat: jax.Array) -> jax.Array:
        v_u = v_flat[:n_u]
        v_p = v_flat[n_u:n_u + n_p].reshape(V_u.K, dim_p)
        # 1. pressure
        p_pc = pressure_pc(v_p)
        # 2. correct velocity: v_u -= B^T p_pc = G p_pc
        Gp = bdm_k_gradient_apply(V_u, p_pc)
        if bc_mask is not None:
            Gp = jnp.where(bc_mask, 0.0, Gp)
        u_corrected = v_u - Gp
        u_pc = velocity_pc(u_corrected)
        return jnp.concatenate([u_pc, p_pc.reshape(-1)])

    return M_inv


# ---------------------------------------------------------------------------
# Dense matrix assembly (for direct LU on small problems)
# ---------------------------------------------------------------------------

def stokes_matrix_bdm_2d(
    V_u: BDM,
    *,
    nu: float,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
    velocity_alpha: float = 0.0,
) -> jax.Array:
    """Assemble the full ``(n_u + n_p) × (n_u + n_p)`` saddle-point matrix
    for the BDM Stokes problem via :func:`stokes_apply_bdm_2d` on canonical
    basis vectors.

    Intended for small debugging problems and validation tests where a
    direct LU solve is preferred over GMRES. The matrix is sparse but
    we store it densely -- if profiling shows this is a bottleneck, the
    apply route is the production path.
    """
    n_u, n_p = V_u.total_dofs, V_u.K
    n_total = n_u + n_p

    def op_flat(v: jax.Array) -> jax.Array:
        u = v[:n_u]
        p = v[n_u:]
        out_u, out_p = stokes_apply_bdm_2d(
            V_u, u, p,
            nu=nu, pressure_eps=pressure_eps,
            tau_scale=tau_scale, velocity_alpha=velocity_alpha,
        )
        return jnp.concatenate([out_u, out_p])

    return jax.vmap(
        lambda i: op_flat(jnp.eye(n_total)[i]),
    )(jnp.arange(n_total)).T


# ---------------------------------------------------------------------------
# Stokes step driver (direct LU)
# ---------------------------------------------------------------------------

def stokes_step_bdm_2d(
    V_u: BDM,
    *,
    nu: float,
    f_u: jax.Array | None = None,
    f_p: jax.Array | None = None,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
) -> tuple[jax.Array, jax.Array]:
    """Solve the steady Stokes saddle-point with given RHS via direct LU.

    Returns ``(u_global, p_per_elem)``. Both fields default to zero
    RHS if not supplied.

    For homogeneous-Dirichlet velocity BCs (the only BC convention
    currently wired through the operator), the system has a 1-dim
    constant-pressure nullspace iff no pressure-pinning is present.
    Pass ``pressure_eps > 0`` (e.g. 1e-8) to lift the nullspace via
    Brezzi-Pitkäranta stabilisation.
    """
    n_u, n_p = V_u.total_dofs, V_u.K
    if f_u is None:
        f_u = jnp.zeros(n_u)
    if f_p is None:
        f_p = jnp.zeros(n_p)
    A = stokes_matrix_bdm_2d(
        V_u, nu=nu, pressure_eps=pressure_eps, tau_scale=tau_scale,
    )
    rhs = jnp.concatenate([f_u, f_p])
    sol = jnp.linalg.solve(A, rhs)
    return sol[:n_u], sol[n_u:]


# ---------------------------------------------------------------------------
# Order-parametric Stokes solver for BDM_k / P_{k-1}
# ---------------------------------------------------------------------------
#
# Companion to the BDM_1-only stokes_apply_bdm_2d / stokes_step_bdm_2d
# above. Uses the bdm_k_* operators end-to-end. The k=1 path of this
# new solver agrees with stokes_step_bdm_2d to numerical zero
# (regression-verified); the new path is the only way to get k=2, 3.


def stokes_apply_bdm_k_2d(
    V_u: BDM,
    u_global: jax.Array,
    p_per_elem: jax.Array,
    *,
    nu: float,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
) -> tuple[jax.Array, jax.Array]:
    """Apply the (n_u + K*dim_p) saddle-point matrix to (u, p).

    ``p_per_elem`` has shape ``(K, dim_p)`` where ``dim_p = k(k+1)/2``.
    At k=1 this is ``(K, 1)`` and squeezes naturally to scalar P_0.

    Returns ``(out_u, out_p)`` with ``out_u`` of shape ``(n_u,)`` and
    ``out_p`` of shape ``(K, dim_p)``.

    Block structure:
        out_u = A u + G p,   A = nu * viscous_apply (homog ghost)
        out_p = -D u - pressure_eps * p
    """
    from ddg.dg2d.bdm import (
        bdm_k_viscous_apply, bdm_k_gradient_apply, bdm_k_divergence_apply,
    )
    Au = bdm_k_viscous_apply(V_u, u_global, nu=nu, tau_scale=tau_scale)
    Gp = bdm_k_gradient_apply(V_u, p_per_elem)
    Du = bdm_k_divergence_apply(V_u, u_global)
    out_u = Au + Gp
    out_p = -Du - pressure_eps * p_per_elem
    return out_u, out_p


def stokes_matrix_bdm_k_2d(
    V_u: BDM,
    *,
    nu: float,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
) -> jax.Array:
    """Assemble the full saddle-point matrix at order k via apply on
    canonical basis vectors. Returns ``(n_u + n_p, n_u + n_p)`` dense
    matrix with ``n_p = K * dim_p``. Pressure DOFs are flattened
    row-major.
    """
    k = V_u.order
    dim_p = k * (k + 1) // 2
    n_u = V_u.total_dofs
    n_p = V_u.K * dim_p
    n = n_u + n_p

    def matvec(x_flat):
        u = x_flat[:n_u]
        p = x_flat[n_u:].reshape(V_u.K, dim_p)
        out_u, out_p = stokes_apply_bdm_k_2d(
            V_u, u, p, nu=nu, pressure_eps=pressure_eps, tau_scale=tau_scale,
        )
        return jnp.concatenate([out_u, out_p.reshape(-1)])

    cols = jax.vmap(matvec, in_axes=1, out_axes=1)(jnp.eye(n))
    return cols


def stokes_step_bdm_k_2d(
    V_u: BDM,
    *,
    nu: float,
    f_u: jax.Array | None = None,
    f_p: jax.Array | None = None,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
) -> tuple[jax.Array, jax.Array]:
    """Direct LU solve of the order-parametric Stokes saddle-point.

    Returns ``(u_global, p_per_elem)`` with ``p_per_elem`` of shape
    ``(K, dim_p)``.
    """
    k = V_u.order
    dim_p = k * (k + 1) // 2
    n_u = V_u.total_dofs
    n_p = V_u.K * dim_p
    if f_u is None:
        f_u = jnp.zeros(n_u)
    if f_p is None:
        f_p = jnp.zeros((V_u.K, dim_p))
    A = stokes_matrix_bdm_k_2d(
        V_u, nu=nu, pressure_eps=pressure_eps, tau_scale=tau_scale,
    )
    rhs = jnp.concatenate([f_u, f_p.reshape(-1)])
    sol = jnp.linalg.solve(A, rhs)
    u_sol = sol[:n_u]
    p_sol = sol[n_u:].reshape(V_u.K, dim_p)
    return u_sol, p_sol


# ---------------------------------------------------------------------------
# Order-parametric steady NS via Picard + direct LU
# ---------------------------------------------------------------------------


def bdm_k_mass_apply(V_u: BDM, u_global: jax.Array) -> jax.Array:
    """Order-parametric BDM velocity mass-matrix apply ``M u``. Used by
    the BDF history (``f_velocity = sum_i (beta_i / dt) M u^{n-i}``) and
    by the transient-Helmholtz LHS (``(gamma0 / dt) M u + nu A_visc u``).
    Generalises :func:`bdm1_mass_apply` to arbitrary order.
    """
    from ddg.dg2d.bdm import (
        bdm_k_element_local_mass, bdm1_gather, bdm1_scatter_add,
    )
    M_K = bdm_k_element_local_mass(V_u)                    # (K, Np, Np)
    u_local = bdm1_gather(V_u.topology, u_global)
    out_local = jnp.einsum("kij,kj->ki", M_K, u_local)
    return bdm1_scatter_add(V_u.topology, out_local)


def bdf_history_k(
    V_u: BDM, u_hist, *, dt: float, order: int,
) -> jax.Array:
    """Order-parametric BDF history: ``sum_i (beta_i / dt) M u^{n-i}``.
    Same as :func:`bdf_history` but uses :func:`bdm_k_mass_apply`."""
    coeffs = BDF_COEFFS[order]
    betas = coeffs["beta"]
    if len(u_hist) < order:
        raise ValueError(
            f"BDF{order} needs {order} history states, got {len(u_hist)}"
        )
    f = (betas[0] / dt) * bdm_k_mass_apply(V_u, u_hist[0])
    for i in range(1, order):
        f = f + (betas[i] / dt) * bdm_k_mass_apply(V_u, u_hist[i])
    return f


def ns_picard_apply_bdm_k_2d(
    V_u: BDM,
    u_global: jax.Array,
    p_per_elem: jax.Array,
    *,
    u_star_global: jax.Array,
    nu: float,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
    u_bc_global: jax.Array | None = None,
    velocity_alpha: float = 0.0,
    f_velocity: jax.Array | None = None,
    is_natural_face: jax.Array | None = None,
    lam_face: jax.Array | None = None,
    pressure_pin: tuple[int, int] | None = None,
    project_pressure_mean: bool = False,    # ignored; kept for API compat
    gamma_al: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Picard saddle-point apply at arbitrary order ``k``:

        out_u = velocity_alpha * M u + (A + C(u*)) u + G p - f_velocity
        out_p = -D u - pressure_eps * p

    ``velocity_alpha`` lets the same routine assemble a time-discretised
    Helmholtz LHS (set ``velocity_alpha = gamma0 / dt`` for BDF). The BDF
    history goes on the RHS through ``f_velocity`` (build via
    :func:`bdf_history_k`); a Boussinesq buoyancy body force adds to the
    same ``f_velocity`` slot.

    Both viscous and convective use NON-HOMOGENEOUS ghost when
    ``u_bc_global`` is provided -- the discrete bilinear form is
    self-consistent (apply represents a_nh directly, no separate
    lift required on the residual).

    Nullspace handling -- two options for the constant-pressure mode
    that the BDM-DG saddle has on closed / weakly-pinned domains:

    * ``project_pressure_mean=True`` (recommended): subtract the mean
      of ``out_p[:, 0]`` from itself. The matrix's nullspace
      ``(0, p_const)`` then maps to zero exactly; combined with the
      same projection on the RHS, GMRES converges on the operator
      restricted to the orthogonal complement of the nullspace. Does
      not perturb the operator (unlike a Brezzi-Pitkaranta lift via
      ``pressure_eps``) and is numerically robust (unlike pinning a
      single DOF).
    * ``pressure_pin=(k_idx, m_idx)``: hard identity-row pin replacing
      the divergence equation at that DOF. Kept as an option for
      diagnostic use; mean projection is preferred.
    """
    from ddg.dg2d.bdm import (
        bdm_k_viscous_apply, bdm_k_convective_apply,
        bdm_k_gradient_apply, bdm_k_divergence_apply,
    )
    Au = bdm_k_viscous_apply(
        V_u, u_global, nu=nu, tau_scale=tau_scale,
        u_bc_global=u_bc_global,
        is_natural_face=is_natural_face,
    )
    Cu = bdm_k_convective_apply(
        V_u, u_global, u_star_global=u_star_global,
        u_bc_global=u_bc_global,
        is_natural_face=is_natural_face,
        lam_face=lam_face,
    )
    Gp = bdm_k_gradient_apply(V_u, p_per_elem)
    Du = bdm_k_divergence_apply(V_u, u_global)
    out_u = Au + Cu + Gp
    # Always add the mass term -- when ``velocity_alpha`` is 0 the
    # contribution is exactly zero, and this keeps the function
    # JAX-traceable when ``velocity_alpha`` is a tracer (e.g. when
    # differentiating w.r.t. dt via alpha = gamma0 / dt).
    out_u = out_u + velocity_alpha * bdm_k_mass_apply(V_u, u_global)
    # Augmented Lagrangian term: A_gamma u = A u + gamma * D^T M_p^{-1} D u
    # adds grad-div stabilisation to the velocity block, making the
    # discrete Schur S_gamma = B A_gamma^{-1} B^T tend to (M_p / gamma)
    # for large gamma. Element-local on BDM (D, M_p, G are local).
    # In primitives: D^T q = -G q, so gamma D^T M_p^{-1} D u =
    # -gamma G(M_p^{-1}(D u)).
    if gamma_al != 0.0:
        from ddg.dg2d.bdm import bdm_k_pkm1_basis_at_points
        from ddg.dg2d.cubature import triangle_cubature
        k = V_u.order
        r_q, s_q, w_q = triangle_cubature(max(2 * (k - 1), 1))
        q_at = bdm_k_pkm1_basis_at_points(k, r_q, s_q)        # (Q, dim_p)
        M_p_ref_inv = jnp.linalg.inv(
            jnp.einsum("q,qm,qn->mn", w_q, q_at, q_at))       # (dim_p, dim_p)
        inv_J = 1.0 / V_u.mesh.J[0]                           # (K,)
        M_p_inv_Du = jnp.einsum("mn,kn->km", M_p_ref_inv, Du) * inv_J[:, None]
        out_u = out_u - gamma_al * bdm_k_gradient_apply(V_u, M_p_inv_Du)
    if f_velocity is not None:
        out_u = out_u - f_velocity
    out_p = -Du - pressure_eps * p_per_elem
    if pressure_pin is not None:
        # Identity row in the pressure equation at the pinned DOF.
        k_idx, m_idx = pressure_pin
        out_p = out_p.at[k_idx, m_idx].set(p_per_elem[k_idx, m_idx])
    # ``project_pressure_mean`` is handled by the caller (subtracts
    # mean from the pressure RHS, ensuring consistency along the
    # constant-pressure left-nullspace direction). We do NOT modify
    # the matvec here: deflation on both ends introduces a new rank
    # deficiency; output-only projection introduces another one.
    # Singular-but-consistent GMRES handles the original operator.
    return out_u, out_p


def ns_picard_matrix_bdm_k_2d(
    V_u: BDM,
    *,
    u_star_global: jax.Array,
    nu: float,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
    velocity_alpha: float = 0.0,
    is_natural_face: jax.Array | None = None,
    pressure_pin: tuple[int, int] | None = None,
    project_pressure_mean: bool = False,
    gamma_al: float = 0.0,
) -> jax.Array:
    """Dense saddle-point matrix at the current Picard linearisation
    point ``u_star_global``. Set ``velocity_alpha = gamma0 / dt`` to
    include the BDF mass term ``alpha * M_u`` in the LHS for the
    transient solver. ``is_natural_face`` flags free-stress outflow
    faces; see :func:`ns_picard_apply_bdm_k_2d`."""
    k = V_u.order
    dim_p = k * (k + 1) // 2
    n_u = V_u.total_dofs
    n_p = V_u.K * dim_p
    n = n_u + n_p

    def matvec(x_flat):
        u = x_flat[:n_u]
        p = x_flat[n_u:].reshape(V_u.K, dim_p)
        out_u, out_p = ns_picard_apply_bdm_k_2d(
            V_u, u, p,
            u_star_global=u_star_global, nu=nu,
            pressure_eps=pressure_eps, tau_scale=tau_scale,
            velocity_alpha=velocity_alpha,
            is_natural_face=is_natural_face,
            pressure_pin=pressure_pin,
            project_pressure_mean=project_pressure_mean,
            gamma_al=gamma_al,
        )
        return jnp.concatenate([out_u, out_p.reshape(-1)])

    return jax.vmap(matvec, in_axes=1, out_axes=1)(jnp.eye(n))


def ns_picard_solve_bdm_k_2d(
    V_u: BDM,
    *,
    nu: float,
    u_bc_global: jax.Array,
    u_bc_vec: jax.Array,
    bc_mask: jax.Array,
    u_init: jax.Array | None = None,
    p_init: jax.Array | None = None,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
    max_iters: int = 30,
    tol: float = 1e-8,
    atol: float = 1e-12,
    residual_history: list | None = None,
    verbose: bool = False,
):
    """Picard iteration for steady NS at arbitrary order k.

    Returns ``(u_sol, p_sol)``. Each iteration freezes ``u_star`` at
    the current ``u_k``, assembles the dense saddle-point matrix
    ``(A + C(u_k))`` plus the gradient/divergence blocks, and solves
    via direct LU. Strong BC pinning: boundary DOFs are overwritten
    with ``u_bc_global`` between Picard steps. Suitable for small
    problems (Kovasznay sentinel, ~hundreds of velocity DOFs).
    """
    from ddg.dg2d.bdm import (
        bdm_k_viscous_dirichlet_lift,
    )
    k = V_u.order
    dim_p = k * (k + 1) // 2
    n_u = V_u.total_dofs
    n_p = V_u.K * dim_p

    if u_init is None:
        u_init = u_bc_global
    if p_init is None:
        p_init = jnp.zeros((V_u.K, dim_p))

    # Viscous Dirichlet lift is constant -- assemble once.
    lift = bdm_k_viscous_dirichlet_lift(
        V_u, u_bc_vec, nu=nu, tau_scale=tau_scale,
    )

    u_k = u_init
    p_k = p_init
    init_residual = None
    for it in range(max_iters):
        A = ns_picard_matrix_bdm_k_2d(
            V_u, u_star_global=u_k, nu=nu,
            pressure_eps=pressure_eps, tau_scale=tau_scale,
        )
        # RHS: viscous Dirichlet lift only (no body force here).
        rhs = jnp.concatenate([lift, jnp.zeros(n_p)])
        sol = jnp.linalg.solve(A, rhs)
        u_new = sol[:n_u]
        p_new = sol[n_u:].reshape(V_u.K, dim_p)
        # Strong-BC: overwrite boundary DOFs with the prescribed values.
        u_new = jnp.where(bc_mask, u_bc_global, u_new)

        # Picard residual = ||u_new - u_k||.
        delta = float(jnp.linalg.norm(u_new - u_k))
        if residual_history is not None:
            residual_history.append(delta)
        if init_residual is None:
            init_residual = max(delta, 1e-30)
        if verbose:
            print(f"  iter {it+1:3d}: ||du||={delta:.3e}")
        if delta < tol or delta < atol * init_residual:
            u_k, p_k = u_new, p_new
            break
        u_k, p_k = u_new, p_new
    return u_k, p_k


# ---------------------------------------------------------------------------
# NS saddle-point apply + nonlinear residual
# ---------------------------------------------------------------------------

def ns_apply_bdm_2d(
    V_u: BDM,
    u_global: jax.Array,
    p_per_elem: jax.Array,
    *,
    nu: float,
    u_star_global: jax.Array,
    velocity_alpha: float = 0.0,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
    u_bc_global: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Linearised NS saddle-point apply at frozen transport velocity ``u*``.

    ``u_bc_global`` (when provided) is used as the Dirichlet boundary
    ghost in the viscous and convective face fluxes -- essential for
    non-homogeneous BCs (the Newton outer loop passes the current
    iterate's boundary values here, which match the prescribed
    Dirichlet data and so kill the boundary jump on the analytic
    solution).
    """
    out_u, out_p = stokes_apply_bdm_2d(
        V_u, u_global, p_per_elem,
        nu=nu, pressure_eps=pressure_eps,
        tau_scale=tau_scale, velocity_alpha=velocity_alpha,
        u_bc_global=u_bc_global,
    )
    out_u = out_u + bdm1_convective_apply(
        V_u, u_global, u_star_global=u_star_global,
        u_bc_global=u_bc_global,
    )
    return out_u, out_p


def ns_residual_bdm_2d(
    V_u: BDM,
    u_global: jax.Array,
    p_per_elem: jax.Array,
    *,
    nu: float,
    f_u: jax.Array | None = None,
    f_p: jax.Array | None = None,
    velocity_alpha: float = 0.0,
    pressure_eps: float = 0.0,
    tau_scale: float = 4.0,
    u_bc_global: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Full nonlinear NS residual ``F(u, p) = N(u) + ν A_visc u + G p - f``.

    Uses the full nonlinear convection ``N(u) = ∇·(u⊗u)`` (i.e.,
    ``u_star_global = u_global``). Returns ``(F_u, F_p)`` -- pass these
    to the outer Newton loop. Pass ``u_bc_global`` so the viscous and
    convective boundary fluxes use the non-homogeneous Dirichlet ghost.
    """
    out_u, out_p = ns_apply_bdm_2d(
        V_u, u_global, p_per_elem,
        nu=nu, u_star_global=u_global,
        velocity_alpha=velocity_alpha,
        pressure_eps=pressure_eps, tau_scale=tau_scale,
        u_bc_global=u_bc_global,
    )
    if f_u is not None:
        out_u = out_u - f_u
    if f_p is not None:
        out_p = out_p - f_p
    return out_u, out_p


# ---------------------------------------------------------------------------
# Newton outer loop with strong Dirichlet BC handling
# ---------------------------------------------------------------------------

def ns_newton_solve_bdm_2d(
    V_u: BDM,
    *,
    nu: float,
    u_bc_global: jax.Array,
    bc_mask: jax.Array,
    u_bc_vec: jax.Array | None = None,
    u_init: jax.Array | None = None,
    p_init: jax.Array | None = None,
    velocity_alpha: float = 0.0,
    pressure_eps: float = 1.0e-10,
    tau_scale: float = 4.0,
    max_iters: int = 10,
    tol: float = 1.0e-8,
    atol: float = 1.0e-10,
    residual_history: list | None = None,
    linearisation: str = "picard",
    iterative: IterativeSolverConfig | None = None,
    coupled_preconditioner=None,
    inner_residual_history: list | None = None,
    f_velocity: jax.Array | None = None,
    f_pressure: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, bool]:
    """Steady NS Newton solve on BDM-P_0 with strong Dirichlet BCs.

    The boundary DOFs of ``u_global`` are *pinned* to ``u_bc_global``
    via the ``bc_mask`` -- the linear solve at every Newton step only
    updates interior DOFs. Initial guess defaults to ``u_init = u_bc_global``
    (BC-respecting) and ``p_init = 0``.

    ``linearisation`` chooses ``"picard"`` (frozen convective ``u*``) or
    ``"newton"`` (full Frechet derivative, adding the extra Jacobian
    term ``∇·(δu⊗u_k)`` on top of the Picard ``∇·(u_k⊗δu)``). Newton
    gives super-linear local convergence near a fixed point; Picard
    gives geometric linear convergence with rate proportional to the
    convective contraction.

    Returns ``(u_global, p_per_elem, converged)``. The residual history
    is appended to ``residual_history`` (if provided).
    """
    if linearisation not in ("picard", "newton", "hybrid"):
        raise ValueError(
            f"linearisation must be 'picard', 'newton', or 'hybrid'; "
            f"got {linearisation!r}"
        )
    n_u, n_p = V_u.total_dofs, V_u.K

    if u_init is None:
        u_k = u_bc_global
    else:
        # Enforce the BC on the initial guess.
        u_k = jnp.where(bc_mask, u_bc_global, u_init)
    if p_init is None:
        p_k = jnp.zeros(n_p)
    else:
        p_k = p_init

    # Proper SIPG Dirichlet lift: boundary-only integral that depends
    # ONLY on the full vector u_BC (not on the volume / interior-face
    # structure of any "lift function"). The BDM DOFs in u_bc_global
    # only carry the normal-trace component; for the SIPG penalty +
    # symmetry terms we need the full vector at boundary GL points,
    # passed in as u_bc_vec (or implicitly zero if not supplied).
    if u_bc_vec is None:
        # Fall back to "BC vector = the normal-trace value times normal"
        # -- correct when the user only set normal-direction BCs (rare).
        # For TANGENTIAL BCs like the lid-driven cavity, callers must
        # pass u_bc_vec explicitly (built via bdm1_evaluate_boundary_velocity).
        u_bc_vec = jnp.zeros(
            (V_u.K, 3, 2, 2)
        )
    lift_u = bdm1_viscous_dirichlet_lift(
        V_u, u_bc_vec, nu=nu, tau_scale=tau_scale,
    )
    lift_p = jnp.zeros(n_p)

    norm_F0 = None
    converged = False
    for k in range(max_iters):
        # Per-iter linearisation: 'hybrid' = first iter Picard, then Newton.
        use_newton = (
            linearisation == "newton"
            or (linearisation == "hybrid" and k > 0)
        )
        F_u, F_p = ns_residual_bdm_2d(
            V_u, u_k, p_k,
            nu=nu, velocity_alpha=velocity_alpha,
            pressure_eps=pressure_eps, tau_scale=tau_scale,
            f_u=f_velocity, f_p=f_pressure,
        )
        F_u = F_u - lift_u
        F_p = F_p - lift_p
        # Project: residual on boundary DOFs is meaningless (those are
        # pinned), so zero them for the convergence check.
        F_u_inner = jnp.where(bc_mask, 0.0, F_u)
        norm_F = float(jnp.linalg.norm(F_u_inner) + jnp.linalg.norm(F_p))
        if residual_history is not None:
            residual_history.append(norm_F)
        if norm_F0 is None:
            norm_F0 = norm_F
        if norm_F <= tol * norm_F0 + atol:
            converged = True
            break

        # Solve the linearised system for δ.
        mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])
        rhs_full = jnp.concatenate([-F_u_inner, -F_p])

        if iterative is None:
            # Direct-LU path: build the full matrix and zero out
            # boundary rows/cols, replacing diagonal with 1. Only viable
            # at small mesh sizes (n_total ~ 1000 or less); above that
            # pass iterative=IterativeSolverConfig(...) to use GMRES
            # with the block-PCs in this module.
            A = stokes_matrix_bdm_2d(
                V_u, nu=nu, pressure_eps=pressure_eps,
                tau_scale=tau_scale, velocity_alpha=velocity_alpha,
            )

            def conv_col(i):
                e = jnp.eye(n_u)[i]
                return bdm1_convective_apply(
                    V_u, e, u_star_global=u_k,
                )
            C = jax.vmap(conv_col)(jnp.arange(n_u)).T
            if use_newton:
                def newton_extra_col(i):
                    e = jnp.eye(n_u)[i]
                    return bdm1_convective_apply(
                        V_u, u_k, u_star_global=e,
                    )
                N_ext = jax.vmap(newton_extra_col)(jnp.arange(n_u)).T
                C = C + N_ext
            A_uu = A[:n_u, :n_u] + C
            A_full = A.at[:n_u, :n_u].set(A_uu)
            I_n = jnp.eye(n_u + n_p)
            A_constrained = jnp.where(
                mask_full[:, None],
                I_n,
                jnp.where(mask_full[None, :], 0.0, A_full),
            )
            delta = jnp.linalg.solve(A_constrained, rhs_full)
        else:
            # Matrix-free iterative path. The matvec applies the full
            # linearised saddle-point operator (Stokes + Picard / Newton
            # convective contributions) and then zeros out boundary
            # rows so the constraint δ_boundary = 0 is enforced.
            # Precompute lam_face outside the matvec to lift the
            # non-transposable jnp.maximum out of the operator (see
            # the BDM_k transient solver for the same idiom).
            lam_face_uk = bdm1_convective_lam_face(V_u, u_k)

            def linear_matvec(v_flat: jax.Array) -> jax.Array:
                du_raw = v_flat[:n_u]
                dp = v_flat[n_u:n_u + n_p]
                # Boundary entries of δu must stay zero; enforce in
                # input so they don't contaminate the apply.
                du = jnp.where(bc_mask, 0.0, du_raw)
                out_u, out_p = stokes_apply_bdm_2d(
                    V_u, du, dp,
                    nu=nu, pressure_eps=pressure_eps,
                    tau_scale=tau_scale, velocity_alpha=velocity_alpha,
                )
                # Picard convective Jacobian: ∇·(u_k ⊗ δu).
                out_u = out_u + bdm1_convective_apply(
                    V_u, du, u_star_global=u_k,
                    lam_face=lam_face_uk,
                )
                if use_newton:
                    # Newton extra: ∇·(δu ⊗ u_k). Freeze lam at u_k.
                    out_u = out_u + bdm1_convective_apply(
                        V_u, u_k, u_star_global=du,
                        lam_face=lam_face_uk,
                    )
                # Identity row on boundary DOFs: ``du_raw`` (NOT the
                # zeroed ``du``) so the boundary equation is
                # ``I delta_u_bnd = 0`` -> delta_u_bnd = 0. Using
                # the masked ``du`` here puts 0 in the row, leaving
                # boundary delta_u as a free variable and stalling
                # GMRES.
                out_u = jnp.where(bc_mask, du_raw, out_u)
                return jnp.concatenate([out_u, out_p])

            x0 = jnp.zeros_like(rhs_full)
            if iterative.method == "gmres":
                delta, info = jax.scipy.sparse.linalg.gmres(
                    linear_matvec, rhs_full, x0=x0,
                    tol=iterative.tol, atol=iterative.atol,
                    maxiter=iterative.maxiter,
                    restart=iterative.restart,
                    M=coupled_preconditioner,
                )
            elif iterative.method == "bicgstab":
                delta, info = jax.scipy.sparse.linalg.bicgstab(
                    linear_matvec, rhs_full, x0=x0,
                    tol=iterative.tol, atol=iterative.atol,
                    maxiter=iterative.maxiter,
                    M=coupled_preconditioner,
                )
            else:
                raise ValueError(
                    f"unknown iterative method {iterative.method!r}"
                )
            if inner_residual_history is not None:
                r_final = float(jnp.linalg.norm(
                    rhs_full - linear_matvec(delta)
                ))
                inner_residual_history.append(r_final)
        delta_u = delta[:n_u]
        delta_p = delta[n_u:]
        # Boundary δ is 0 by construction; double-check:
        delta_u = jnp.where(bc_mask, 0.0, delta_u)

        u_k = u_k + delta_u
        p_k = p_k + delta_p

    return u_k, p_k, converged


def ns_newton_solve_bdm_k_2d(
    V_u: BDM,
    *,
    nu: float,
    u_bc_global: jax.Array,
    bc_mask: jax.Array,
    u_bc_vec: jax.Array | None = None,
    u_init: jax.Array | None = None,
    p_init: jax.Array | None = None,
    velocity_alpha: float = 0.0,
    pressure_eps: float = 1.0e-10,
    tau_scale: float = 4.0,
    max_iters: int = 10,
    tol: float = 1.0e-8,
    atol: float = 1.0e-10,
    residual_history: list | None = None,
    linearisation: str = "picard",
    iterative: IterativeSolverConfig | None = None,
    coupled_preconditioner=None,
    inner_residual_history: list | None = None,
    f_velocity: jax.Array | None = None,
    f_pressure: jax.Array | None = None,
    is_natural_face: jax.Array | None = None,
    pressure_pin: tuple[int, int] | None = None,
    project_pressure_mean: bool = False,
    gamma_al: float = 0.0,
    inner_solvers: tuple | None = None,
) -> tuple[jax.Array, jax.Array, bool]:
    """Order-parametric transient NS Newton/Picard solver on BDM_k /
    P_{k-1} with strong Dirichlet BCs. Generalises
    :func:`ns_newton_solve_bdm_2d` (BDM_1 / P_0 only) to arbitrary order
    via the order-k primitives in :mod:`ddg.dg2d.bdm`.

    Differences from the BDM_1 path:
    * pressure DOFs are ``(K, dim_p)`` with ``dim_p = k(k+1)/2``; the
      linear solve internally flattens / reshapes around the BDF/BDM-1
      ``(K,)`` convention;
    * the residual is computed via :func:`ns_picard_apply_bdm_k_2d`
      (already transient-aware via ``velocity_alpha`` + ``f_velocity``);
    * the dense matrix path uses :func:`ns_picard_matrix_bdm_k_2d` plus
      a jvp-built convective column (avoiding building it column-by-column
      against a (n_u, n_u) identity, which is too large for k>1);
    * the Dirichlet lift uses :func:`bdm_k_viscous_dirichlet_lift`;
    * the Newton extra term uses :func:`bdm_k_convective_apply` with
      ``u_star`` and ``u`` swapped.

    Returns ``(u_global, p_per_elem, converged)``. The pressure return
    is reshaped to ``(K, dim_p)`` for caller convenience.
    """
    from ddg.dg2d.bdm import (
        bdm_k_viscous_dirichlet_lift, bdm_k_convective_apply,
    )

    if linearisation not in ("picard", "newton", "hybrid"):
        raise ValueError(
            f"linearisation must be 'picard', 'newton', or 'hybrid'; "
            f"got {linearisation!r}"
        )
    # 'hybrid' = first iter Picard (wide basin), then Newton (quadratic
    # local rate). Standard globalisation pattern -- the first iter
    # absorbs the BDF-step transient while the basin shifts, then
    # Newton finishes; in practice this halves outer iterations vs
    # pure Picard with no robustness loss when the initial guess is
    # the previous time step's solution.
    k = V_u.order
    dim_p = k * (k + 1) // 2
    n_u = V_u.total_dofs
    n_p = V_u.K * dim_p

    if u_init is None:
        u_k = u_bc_global
    else:
        u_k = jnp.where(bc_mask, u_bc_global, u_init)
    if p_init is None:
        p_k = jnp.zeros((V_u.K, dim_p))
    else:
        p_k = p_init if p_init.ndim == 2 else p_init.reshape(V_u.K, dim_p)

    if u_bc_vec is None:
        # No tangential BC contribution. Lift is identically zero;
        # legitimate only when all Dirichlet data is normal-trace.
        lift_u = jnp.zeros(n_u)
    else:
        lift_u = bdm_k_viscous_dirichlet_lift(
            V_u, u_bc_vec, nu=nu, tau_scale=tau_scale,
            is_natural_face=is_natural_face,
        )
    lift_p_flat = jnp.zeros(n_p)

    # Hoist the inner-iterative-solve out of the Newton loop. ``u_k``
    # is captured as an explicit tracer argument so the gmres+matvec+PC
    # trace hits the JAX cache across Newton iters. Two variants
    # (Picard / Newton) are dispatched per-iter by ``use_newton``.
    # Callers can pass prebuilt ``inner_solvers`` to share the JIT
    # cache ACROSS BDF time steps (the cache is keyed on the Python
    # function identity, which a fresh ``ns_newton_solve_bdm_k_2d``
    # call would otherwise rebuild).
    inner_solve_picard = None
    inner_solve_newton = None
    if iterative is not None:
        if inner_solvers is not None:
            inner_solve_picard, inner_solve_newton = inner_solvers
        else:
            built = make_bdm_k_iterative_inner_solvers(
                V_u, iterative=iterative,
                coupled_preconditioner=coupled_preconditioner,
                nu=nu, pressure_eps=pressure_eps, tau_scale=tau_scale,
                velocity_alpha=velocity_alpha,
                is_natural_face=is_natural_face,
                bc_mask=bc_mask, u_bc_global=u_bc_global,
                pressure_pin=pressure_pin,
                project_pressure_mean=project_pressure_mean,
                gamma_al=gamma_al,
                build_newton=linearisation in ("newton", "hybrid"),
            )
            inner_solve_picard, inner_solve_newton = built

    norm_F0 = None
    converged = False
    for it in range(max_iters):
        # Per-iter linearisation: 'hybrid' = first iter Picard, then Newton.
        use_newton = (
            linearisation == "newton"
            or (linearisation == "hybrid" and it > 0)
        )
        # Residual uses the ORIGINAL (un-augmented) operator. The AL
        # augmentation only enters the linearised Newton update equation
        # (matvec + RHS), not the nonlinear residual measure.
        F_u, F_p = ns_picard_apply_bdm_k_2d(
            V_u, u_k, p_k,
            u_star_global=u_k, nu=nu,
            pressure_eps=pressure_eps, tau_scale=tau_scale,
            velocity_alpha=velocity_alpha,
            f_velocity=f_velocity,
            is_natural_face=is_natural_face,
            pressure_pin=pressure_pin,
            project_pressure_mean=project_pressure_mean,
            gamma_al=0.0,
        )
        F_u = F_u - lift_u
        F_p_flat = F_p.reshape(-1) - lift_p_flat
        if f_pressure is not None:
            F_p_flat = F_p_flat - (
                f_pressure.reshape(-1) if f_pressure.ndim == 2
                else f_pressure
            )
        if project_pressure_mean:
            # Project the constant-pressure component out of the
            # pressure RHS, ensuring consistency along the saddle's
            # constant-pressure left-nullspace direction. GMRES on
            # the singular but now-consistent system converges; the
            # constant-pressure component of the Newton update
            # ``delta_p`` floats freely (we re-fix it after solve).
            F_p_arr = F_p_flat.reshape(V_u.K, dim_p)
            F_p_flat = F_p_arr.at[:, 0].add(-jnp.mean(F_p_arr[:, 0])).reshape(-1)
        F_u_inner = jnp.where(bc_mask, 0.0, F_u)
        norm_F = float(
            jnp.linalg.norm(F_u_inner) + jnp.linalg.norm(F_p_flat)
        )
        if residual_history is not None:
            residual_history.append(norm_F)
        if norm_F0 is None:
            norm_F0 = norm_F
        if norm_F <= tol * norm_F0 + atol:
            converged = True
            break

        mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])
        # Augmented Lagrangian RHS: rhs_u_AL = -F_u + gamma B^T W^{-1} (-F_p).
        # With B = -D, B^T = G = -D^T, W = M_p (per-element pressure mass):
        #   rhs_u_AL = -F_u - gamma G(M_p^{-1}(F_p))
        # The matvec carries the matching A_gamma = A + gamma D^T M_p^{-1} D
        # augmentation when gamma_al is plumbed through.
        if gamma_al != 0.0:
            from ddg.dg2d.bdm import bdm_k_pkm1_basis_at_points
            from ddg.dg2d.cubature import triangle_cubature
            r_q, s_q, w_q = triangle_cubature(max(2 * (V_u.order - 1), 1))
            q_at = bdm_k_pkm1_basis_at_points(V_u.order, r_q, s_q)
            M_p_ref_inv = jnp.linalg.inv(
                jnp.einsum("q,qm,qn->mn", w_q, q_at, q_at))
            inv_J_ = 1.0 / V_u.mesh.J[0]
            F_p_arr = F_p_flat.reshape(V_u.K, dim_p)
            M_p_inv_Fp = (jnp.einsum("mn,kn->km", M_p_ref_inv, F_p_arr)
                          * inv_J_[:, None])
            from ddg.dg2d.bdm import bdm_k_gradient_apply as _G
            AL_rhs = gamma_al * _G(V_u, M_p_inv_Fp)
            AL_rhs = jnp.where(bc_mask, 0.0, AL_rhs)
            rhs_full = jnp.concatenate([-F_u_inner - AL_rhs, -F_p_flat])
        else:
            rhs_full = jnp.concatenate([-F_u_inner, -F_p_flat])

        if iterative is None:
            # Direct LU: build the dense (n_u+n_p) saddle-point matrix
            # at the current Picard linearisation, optionally with the
            # Newton extra term. ns_picard_matrix_bdm_k_2d already
            # includes velocity_alpha * M when requested, so the LHS is
            # the full transient implicit operator.
            A_full = ns_picard_matrix_bdm_k_2d(
                V_u, u_star_global=u_k, nu=nu,
                pressure_eps=pressure_eps, tau_scale=tau_scale,
                velocity_alpha=velocity_alpha,
                is_natural_face=is_natural_face,
                pressure_pin=pressure_pin,
                project_pressure_mean=project_pressure_mean,
                gamma_al=gamma_al,
            )
            if use_newton:
                # Newton adds the column-derivative through u_star:
                # the extra Jacobian column ``i`` is
                # bdm_k_convective_apply(V_u, u_k, u_star=e_i).
                def newton_extra_col(i):
                    e = jnp.eye(n_u)[i]
                    return bdm_k_convective_apply(
                        V_u, u_k, u_star_global=e,
                        is_natural_face=is_natural_face,
                    )
                N_ext = jax.vmap(newton_extra_col)(jnp.arange(n_u)).T
                A_full = A_full.at[:n_u, :n_u].add(N_ext)
            I_n = jnp.eye(n_u + n_p)
            A_constrained = jnp.where(
                mask_full[:, None],
                I_n,
                jnp.where(mask_full[None, :], 0.0, A_full),
            )
            delta = jnp.linalg.solve(A_constrained, rhs_full)
        else:
            # Use the pre-JIT'd inner solve (constructed once outside
            # the Newton loop). u_k is an explicit tracer argument so
            # the gmres+matvec+PC stack is traced once and re-used.
            if iterative.method not in ("gmres", "bicgstab"):
                raise ValueError(
                    f"unknown iterative method {iterative.method!r}"
                )
            solver = (inner_solve_newton if use_newton else inner_solve_picard)
            delta, r_final_iter = solver(u_k, rhs_full)
            if inner_residual_history is not None:
                inner_residual_history.append(float(r_final_iter))

        delta_u = delta[:n_u]
        delta_p_flat = delta[n_u:]
        delta_u = jnp.where(bc_mask, 0.0, delta_u)
        delta_p = delta_p_flat.reshape(V_u.K, dim_p)
        if project_pressure_mean:
            # GMRES on the singular saddle (which has a constant-
            # pressure null direction) returns a delta whose constant-
            # pressure component is freely-drifting. Subtract its mean
            # so the Newton update doesn't accumulate spurious constant-
            # pressure offsets.
            delta_p = delta_p.at[:, 0].add(-jnp.mean(delta_p[:, 0]))

        u_k = u_k + delta_u
        p_k = p_k + delta_p

    if project_pressure_mean:
        # Final canonical normalisation: pressure has zero mean over
        # the monomial-0 component.
        p_k = p_k.at[:, 0].add(-jnp.mean(p_k[:, 0]))

    return u_k, p_k, converged

