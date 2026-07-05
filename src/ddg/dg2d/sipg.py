"""2D Poisson operator in SIPG (symmetric interior-penalty Galerkin)
primal form, with native support for mixed Dirichlet / Neumann BCs.

This module exists alongside :mod:`ddg.dg2d.poisson` because the LDG-form
operator there does not extend cleanly to mixed BCs — its per-element
LIFT correction from Dirichlet faces spills into the auxiliary ``q``
at Neumann face nodes of the same element, breaking symmetry of the
bilinear form. The SIPG primal form here is per-face-local, which lets
each face carry its own BC handling without cross-talk.

Port of Hesthaven's ``Codes2D/PoissonIPDG2D.m`` and
``Codes2D/PoissonIPDGbc2D.m``.

The bilinear form is

    a(u, v) = ∫_Ω ∇u·∇v
            − ∫_Γ {{∇u·n}} [v]
            − ∫_Γ [u] {{∇v·n}}
            + ∫_Γ τ [u] [v]

with the standard SIPG ghost-value conventions per face type:

  * **Interior face**: ``u_P = u_neighbour``, ``∇u_P = ∇u_neighbour``;
    each face is processed twice (once from each side), so its
    contribution is halved here.

  * **Dirichlet face** (homog ``u = 0``): ``u_P = 0`` (one-sided jump),
    ``∇u_P = ∇u_M``; processed once with full weight.

  * **Neumann face** (homog ``∂u/∂n = 0``): ``u_P = u_M`` (no jump),
    ``∇u_P = ∇u_M``; surface contribution vanishes automatically.

Non-homogeneous BCs are linear functionals — the matrix is the same;
only the right-hand-side load changes.

The operator returns ``M·(−Δ_h) u`` and is intended to be paired with
``M f`` on the right (so the equation ``-Δu = f`` becomes ``A u = M f``
plus boundary loads).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg1d.reference import vandermonde_1d
from ddg.dg2d.mesh import Mesh2D


# Public BC type codes (per-face integer flags).
BC_INTERIOR = 0
BC_DIRICHLET = 1
BC_NEUMANN = 2


@dataclass(frozen=True)
class IterativeConfig:
    """Iterative solver configuration for SPD systems (Poisson, Helmholtz).

    The SIPG-form ``M(-Δ)`` operator and the Helmholtz ``αM + A_sipg`` are
    both symmetric positive-definite, so conjugate gradient is the
    right Krylov method. Matrix-free — the dense matrix is never
    assembled, only the apply function. Critical for GPU scaling.
    """
    method: str = "cg"          # "cg" only for SPD
    tol: float = 1e-9
    atol: float = 0.0
    maxiter: int = 1000


# ---------------------------------------------------------------------------
# Reference-element pre-computation
# ---------------------------------------------------------------------------

def _local_mass(mesh: Mesh2D) -> jax.Array:
    """Reference-element volume mass matrix ``M_ref = inv(V V^T)``."""
    return jnp.linalg.inv(mesh.V @ mesh.V.T)


def _face_param_for_face(mesh: Mesh2D, f: int) -> jax.Array:
    """1D parameter along the reference triangle's face ``f``.

    For face 0 (s=-1) and face 1 (r+s=0), ``r`` of the face nodes serves
    as the parameter; for face 2 (r=-1) it's ``s``. In all cases the
    parameter spans ``[-1, 1]`` (possibly in reverse order). The 1D
    Vandermonde / face mass matrix built from this parameter is what
    Hesthaven uses in ``PoissonIPDG2D.m``.
    """
    Fm = np.asarray(mesh.Fmask)[:, f]
    if f < 2:
        return mesh.r[Fm]
    return mesh.s[Fm]


def _face_1d_mass_matrices(mesh: Mesh2D) -> jax.Array:
    """Per-reference-face 1D mass matrix.

    Returns ``(Nfaces, Nfp, Nfp)`` array: ``m1d[f]`` is the 1D mass on
    face ``f``'s nodes in the ordering they appear in ``Fmask[:, f]``.
    Computed as ``inv(V1D V1D^T)`` on the face-trace parameter.
    """
    mats = []
    for f in range(mesh.Nfaces):
        param = _face_param_for_face(mesh, f)
        V1D = vandermonde_1d(mesh.N, param)
        mats.append(jnp.linalg.inv(V1D @ V1D.T))
    return jnp.stack(mats, axis=0)


# ---------------------------------------------------------------------------
# Bilinear form a(u, v) and apply
# ---------------------------------------------------------------------------

def _bc_face_arrays(mesh: Mesh2D, bc_types: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Expand per-(K, Nfaces) BC tags into face-node, per-(f, K), and flat forms.

    Returns ``(bc_per_facenode_flat, bc_per_face, w_per_face)``:
      * ``bc_per_facenode_flat`` : shape ``(Nfp*Nfaces*K,)``, integer flags.
      * ``bc_per_face`` : shape ``(Nfaces, K)``, integer flags.
      * ``w_per_face`` : shape ``(Nfaces, K)`` float weights — 0.5 on
        interior faces (because in our face-trace layout each interior
        face is hit twice, once from each side), 1.0 on Dirichlet,
        0.0 on Neumann.
    """
    # bc_types is (K, Nfaces). Transpose to (Nfaces, K) for column-major
    # consistency with the face layout below.
    bc_per_face = bc_types.T.astype(jnp.int32)
    bc_per_facenode = jnp.broadcast_to(
        bc_per_face[None, :, :], (mesh.Nfp, mesh.Nfaces, mesh.K)
    )
    bc_per_facenode_flat = bc_per_facenode.reshape(-1, order="F")
    w_per_face = jnp.where(
        bc_per_face == BC_DIRICHLET, 1.0,
        jnp.where(bc_per_face == BC_NEUMANN, 0.0, 0.5),
    )
    return bc_per_facenode_flat, bc_per_face, w_per_face


def sipg_bilinear_form(
    mesh: Mesh2D,
    u: jax.Array,
    v: jax.Array,
    *,
    bc_types: jax.Array,
    u_dir: jax.Array | None = None,
    tau_scale: float = 200.0,
    M_ref: jax.Array | None = None,
    m1d_per_face: jax.Array | None = None,
    cub: "MeshCubature | None" = None,
) -> jax.Array:
    """SIPG bilinear form ``a(u, v)`` with mixed BCs.

    ``bc_types`` is a ``(K, Nfaces)`` integer array using the codes
    :data:`BC_INTERIOR` / :data:`BC_DIRICHLET` / :data:`BC_NEUMANN`.
    ``u_dir`` (optional, ``(Np, K)``) supplies non-homogeneous Dirichlet
    BC values — only boundary face traces matter; everything else is
    ignored. Non-homogeneous Neumann ``q = h`` is handled separately by
    :func:`sipg_neumann_load_2d` since it doesn't enter the operator
    (only the RHS).

    Returns a scalar. The matrix is recovered via
    ``A = jax.grad(jax.grad(form, argnums=0), argnums=1)`` style
    auto-diff (see :func:`sipg_apply_2d` and :func:`sipg_matrix_2d`).
    """
    Np, K, Nfp, Nfaces = mesh.Np, mesh.K, mesh.Nfp, mesh.Nfaces
    if M_ref is None:
        M_ref = _local_mass(mesh)
    if m1d_per_face is None:
        m1d_per_face = _face_1d_mass_matrices(mesh)

    # Gradients at NODES (used by the surface trace below).
    ur = mesh.Dr @ u
    us = mesh.Ds @ u
    ux = mesh.rx * ur + mesh.sx * us
    uy = mesh.ry * ur + mesh.sy * us
    vr = mesh.Dr @ v
    vs = mesh.Ds @ v
    vx = mesh.rx * vr + mesh.sx * vs
    vy = mesh.ry * vr + mesh.sy * vs

    # Volume term: ∫_Ω ∇u·∇v J dr ds.
    # Nodal "diagonal-J" quadrature (default) is exact for affine
    # elements; for curved elements (J varying across the element)
    # supply a `cub: MeshCubature` for proper high-order cubature.
    if cub is None:
        vol = jnp.sum(mesh.J * (ux * (M_ref @ vx) + uy * (M_ref @ vy)))
    else:
        ur_q = cub.Drq @ u                              # (Ncub, K)
        us_q = cub.Dsq @ u
        vr_q = cub.Drq @ v
        vs_q = cub.Dsq @ v
        ux_q = cub.rx * ur_q + cub.sx * us_q
        uy_q = cub.ry * ur_q + cub.sy * us_q
        vx_q = cub.rx * vr_q + cub.sx * vs_q
        vy_q = cub.ry * vr_q + cub.sy * vs_q
        vol = jnp.sum(
            cub.w[:, None] * cub.J * (ux_q * vx_q + uy_q * vy_q)
        )

    # Surface term: face traces of u, v and their normal derivatives.
    u_flat = u.T.reshape(-1)
    v_flat = v.T.reshape(-1)
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)
    vx_flat = vx.T.reshape(-1)
    vy_flat = vy.T.reshape(-1)

    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    sJ_flat = mesh.sJ.reshape(-1, order="F")
    Fscale_flat = mesh.Fscale.reshape(-1, order="F")

    u_M = u_flat[mesh.vmapM]
    u_P_raw = u_flat[mesh.vmapP]
    v_M = v_flat[mesh.vmapM]
    v_P_raw = v_flat[mesh.vmapP]
    Dn_u_M = nx_flat * ux_flat[mesh.vmapM] + ny_flat * uy_flat[mesh.vmapM]
    Dn_u_P_raw = nx_flat * ux_flat[mesh.vmapP] + ny_flat * uy_flat[mesh.vmapP]
    Dn_v_M = nx_flat * vx_flat[mesh.vmapM] + ny_flat * vy_flat[mesh.vmapM]
    Dn_v_P_raw = nx_flat * vx_flat[mesh.vmapP] + ny_flat * vy_flat[mesh.vmapP]

    bc_per_facenode_flat, bc_per_face, w_per_face = _bc_face_arrays(mesh, bc_types)
    is_dir = bc_per_facenode_flat == BC_DIRICHLET
    is_neu = bc_per_facenode_flat == BC_NEUMANN
    is_bdy = is_dir | is_neu

    # Ghost-value convention (SIPG, one-sided jump):
    #   Dirichlet: u_P = u_dir (or 0 if not supplied), ∇u_P = ∇u_M
    #   Neumann:   u_P = u_M,                          ∇u_P = ∇u_M
    #   Interior:  u_P = real neighbour,               ∇u_P = real neighbour
    if u_dir is None:
        u_P_dir = jnp.zeros_like(u_M)
        v_P_dir = jnp.zeros_like(v_M)
    else:
        u_dir_flat = u_dir.T.reshape(-1)
        u_P_dir = u_dir_flat[mesh.vmapM]
        v_P_dir = jnp.zeros_like(v_M)  # test functions have homog ghost
    u_P_eff = jnp.where(is_dir, u_P_dir, jnp.where(is_neu, u_M, u_P_raw))
    v_P_eff = jnp.where(is_dir, v_P_dir, jnp.where(is_neu, v_M, v_P_raw))
    Dn_u_P_eff = jnp.where(is_bdy, Dn_u_M, Dn_u_P_raw)
    Dn_v_P_eff = jnp.where(is_bdy, Dn_v_M, Dn_v_P_raw)

    avg_Dn_u = 0.5 * (Dn_u_M + Dn_u_P_eff)
    avg_Dn_v = 0.5 * (Dn_v_M + Dn_v_P_eff)
    jmp_u = u_M - u_P_eff
    jmp_v = v_M - v_P_eff

    # Reshape to (Nfp, Nfaces, K) for per-face integration
    def _to_face(x):
        return x.reshape((Nfp, Nfaces, K), order="F")
    avg_Dn_u_face = _to_face(avg_Dn_u)
    avg_Dn_v_face = _to_face(avg_Dn_v)
    jmp_u_face = _to_face(jmp_u)
    jmp_v_face = _to_face(jmp_v)
    sJ_face = _to_face(sJ_flat)
    Fscale_face = _to_face(Fscale_flat)

    sJ_per_face = sJ_face[0, :, :]            # (Nfaces, K)
    Fscale_per_face = Fscale_face[0, :, :]    # (Nfaces, K)
    tau_per_face = tau_scale * (mesh.N + 1) * (mesh.N + 1) * Fscale_per_face

    if cub is None:
        # Affine-face quadrature: ∫_F X Y dS = sJ_F · X_face^T m1d_F Y_face
        # Total = Σ_F w_F · sJ_F · X^T m1d_F Y
        #   = einsum('fk, fk, pfk, fpq, qfk ->', w, sJ, X, m1d, Y)
        w_sJ = w_per_face * sJ_per_face
        cons_int = -jnp.einsum(
            "fk,pfk,fpq,qfk->", w_sJ, avg_Dn_u_face, m1d_per_face, jmp_v_face,
        )
        adj_int = -jnp.einsum(
            "fk,pfk,fpq,qfk->", w_sJ, jmp_u_face, m1d_per_face, avg_Dn_v_face,
        )
        pen_int = jnp.einsum(
            "fk,pfk,fpq,qfk->",
            w_per_face * tau_per_face * sJ_per_face,
            jmp_u_face, m1d_per_face, jmp_v_face,
        )
    else:
        # 1D face cubature: interpolate face-nodal traces to face cubature
        # via cub.face_L1d, integrate with cub.face_sJ at cubature points.
        def _to_cub(face_vals):
            # (Nfp, Nfaces, K) -> (Nfcub, Nfaces, K)
            return jnp.einsum("fqp,pfk->qfk", cub.face_L1d, face_vals)
        avg_Dn_u_q = _to_cub(avg_Dn_u_face)
        avg_Dn_v_q = _to_cub(avg_Dn_v_face)
        jmp_u_q = _to_cub(jmp_u_face)
        jmp_v_q = _to_cub(jmp_v_face)
        sJ_q = jnp.transpose(cub.face_sJ, (1, 0, 2))             # (Nfcub, Nfaces, K)
        weights = (cub.face_w[:, None, None]
                   * sJ_q
                   * w_per_face[None, :, :])                     # (Nfcub, Nfaces, K)
        cons_int = -jnp.sum(weights * avg_Dn_u_q * jmp_v_q)
        adj_int = -jnp.sum(weights * jmp_u_q * avg_Dn_v_q)
        pen_int = jnp.sum(
            weights * tau_per_face[None, :, :] * jmp_u_q * jmp_v_q
        )

    return vol + cons_int + adj_int + pen_int


def sipg_apply_2d(
    mesh: Mesh2D,
    u: jax.Array,
    *,
    bc_types: jax.Array,
    u_dir: jax.Array | None = None,
    tau_scale: float = 200.0,
    M_ref: jax.Array | None = None,
    m1d_per_face: jax.Array | None = None,
    cub: "MeshCubature | None" = None,
) -> jax.Array:
    """Apply the SIPG operator ``M·(−Δ_h) u`` (with the BC-load offset).

    With ``u_dir`` non-zero the result includes the Dirichlet load
    automatically (since the bilinear form is affine in ``u`` through
    the ghost convention). To isolate the homogeneous LHS operator,
    pass ``u_dir=None``. Used by :func:`sipg_matrix_2d` (via
    ``jax.jacobian`` with ``u_dir=None``) to materialise the matrix, and
    by :func:`solve_sipg_poisson_2d` (with ``u_dir=...``) to assemble
    the boundary load.
    """
    if M_ref is None:
        M_ref = _local_mass(mesh)
    if m1d_per_face is None:
        m1d_per_face = _face_1d_mass_matrices(mesh)
    v_zero = jnp.zeros_like(u)

    def form(v):
        return sipg_bilinear_form(
            mesh, u, v, bc_types=bc_types, u_dir=u_dir,
            tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d_per_face,
            cub=cub,
        )
    return jax.grad(form)(v_zero)


# ---------------------------------------------------------------------------
# Matrix assembly + LU factor
# ---------------------------------------------------------------------------

def sipg_matrix_2d(
    mesh: Mesh2D,
    *,
    bc_types: jax.Array,
    tau_scale: float = 200.0,
    cub: "MeshCubature | None" = None,
) -> jax.Array:
    """Assemble the dense ``(Np*K, Np*K)`` SIPG stiffness matrix."""
    Np, K = mesh.Np, mesh.K
    M_ref = _local_mass(mesh)
    m1d_per_face = _face_1d_mass_matrices(mesh)

    def op_flat(u_flat):
        u = u_flat.reshape(K, Np).T
        return sipg_apply_2d(
            mesh, u, bc_types=bc_types, u_dir=None,
            tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d_per_face,
            cub=cub,
        ).T.reshape(-1)

    return jax.jacobian(op_flat)(jnp.zeros(Np * K))


def factor_sipg_2d(
    mesh: Mesh2D,
    *,
    bc_types: jax.Array,
    tau_scale: float = 200.0,
    regularize: bool = False,
    eps: float = 1.0e-8,
    cub: "MeshCubature | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Assemble + LU-factor the SIPG matrix; returns ``(A, lu, piv)``.

    With no Dirichlet face the pure-Neumann (or fully-periodic) SIPG
    Laplacian is singular — its nullspace is the constants. Pass
    ``regularize=True`` to add ``eps · max(diag(A)) · I`` before
    factoring, which breaks the nullspace without an RHS change. The
    regularised solution converges to the minimum-norm (mean-near-
    zero) field as ``eps → 0``; for a pressure Poisson that is exactly
    the right gauge, since only ``∇p`` feeds the velocity update.
    """
    A = sipg_matrix_2d(mesh, bc_types=bc_types, tau_scale=tau_scale, cub=cub)
    if regularize:
        shift = eps * jnp.max(jnp.abs(jnp.diag(A)))
        A = A + shift * jnp.eye(A.shape[0])
    lu, piv = jax.scipy.linalg.lu_factor(A)
    return A, lu, piv


# ---------------------------------------------------------------------------
# Boundary loads (RHS contributions)
# ---------------------------------------------------------------------------

def sipg_dirichlet_load_2d(
    mesh: Mesh2D,
    u_dir: jax.Array,
    *,
    bc_types: jax.Array,
    tau_scale: float = 200.0,
) -> jax.Array:
    """RHS load from non-homogeneous Dirichlet BC ``u = u_dir``.

    Returned as an ``(Np, K)`` field. Solving ``A u = M f - L_D + L_N``
    enforces the BC weakly. Computed by applying the SIPG operator to a
    zero solution but with the non-zero Dirichlet ghost, which extracts
    the affine offset from the bilinear form.
    """
    M_ref = _local_mass(mesh)
    m1d_per_face = _face_1d_mass_matrices(mesh)
    return sipg_apply_2d(
        mesh, jnp.zeros_like(u_dir), bc_types=bc_types, u_dir=u_dir,
        tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d_per_face,
    )


def sipg_neumann_load_2d(
    mesh: Mesh2D,
    q_neu: jax.Array,
    *,
    bc_types: jax.Array,
) -> jax.Array:
    """RHS load from non-homogeneous Neumann BC ``∂u/∂n = q_neu``.

    The Neumann data enters the right-hand side as ``∫_Γ_N q v dS`` —
    the surface mass matrix times the Neumann values, evaluated only on
    Neumann faces. ``q_neu`` has shape ``(Nfp*Nfaces, K)``; only Neumann
    face entries are read.

    Built directly using the per-face 1D mass: for a Neumann face F,
    the load contribution at face node p is ``sJ_F · (m1d_F q_neu)[p]``,
    embedded back to the volume by scatter into Fmask positions.
    """
    Np, K, Nfp, Nfaces = mesh.Np, mesh.K, mesh.Nfp, mesh.Nfaces
    m1d_per_face = _face_1d_mass_matrices(mesh)
    _, bc_per_face, _ = _bc_face_arrays(mesh, bc_types)
    is_neu_face = (bc_per_face == BC_NEUMANN).astype(q_neu.dtype)  # (Nfaces, K)

    q_face = q_neu.reshape((Nfp, Nfaces, K), order="F")
    sJ_per_face = mesh.sJ.reshape((Nfp, Nfaces, K), order="F")[0, :, :]
    # m1d_F @ q_face[:, f, k]  →  (Nfp, Nfaces, K)
    m1d_q = jnp.einsum("fpq,qfk->pfk", m1d_per_face, q_face)
    # Mask out non-Neumann faces and scale by sJ
    contribution = m1d_q * (sJ_per_face * is_neu_face)[None, :, :]
    # Scatter back from face-node ordering (Nfp, Nfaces, K) to volume (Np, K).
    Fmask = np.asarray(mesh.Fmask, dtype=np.int64)
    out = jnp.zeros((Np, K))
    for f in range(Nfaces):
        out = out.at[Fmask[:, f], :].add(contribution[:, f, :])
    return out


# ---------------------------------------------------------------------------
# Helmholtz operator: αM + (−ν Δ) for the viscous step of dual-splitting INS
# ---------------------------------------------------------------------------

def sipg_helmholtz_matrix_2d(
    mesh: Mesh2D,
    *,
    alpha: float,
    nu: float,
    bc_types: jax.Array,
    tau_scale: float = 200.0,
    cub: "MeshCubature | None" = None,
) -> jax.Array:
    """Assemble the Helmholtz operator ``α M + ν · A_sipg`` as a dense matrix.

    This is the LHS for the viscous step of Hesthaven's dual-splitting
    incompressible Navier-Stokes scheme: ``(g0/(ν dt)) u_new - Δ u_new
    = (g0/(ν dt)) u_TT`` rewrites as ``(α M + ν A_sipg) u_new = α M
    u_TT + ν · (boundary loads)`` with ``α = g0/(ν dt)`` (after
    multiplying both sides by ``ν`` to keep ``A_sipg`` as our
    ``M(-Δ_h)`` operator). We multiply by ``ν`` rather than absorbing
    into ``α`` to keep the boundary-load convention unchanged.

    Concretely:  α · M_K + A_sipg  →  this returns that.
    The viscous solve becomes ``(αM + A) u = αM u_TT + L_D + L_N``.
    """
    A = sipg_matrix_2d(mesh, bc_types=bc_types, tau_scale=tau_scale, cub=cub)
    # Volume mass matrix block-diagonal in Np × Np per element, scaled by J.
    M_ref = _local_mass(mesh)
    if cub is None:
        M_block = _block_diag_mass(mesh, M_ref)
    else:
        from ddg.dg2d.cubature import per_element_mass
        M_K = per_element_mass(cub)                                   # (K, Np, Np)
        Np, K = mesh.Np, mesh.K
        M_block = jnp.zeros((Np * K, Np * K))
        for k in range(K):
            M_block = M_block.at[
                k * Np:(k + 1) * Np, k * Np:(k + 1) * Np
            ].set(M_K[k])
    return alpha * M_block + A


def _block_diag_mass(mesh: Mesh2D, M_ref: jax.Array) -> jax.Array:
    """Block-diagonal physical mass matrix of shape ``(Np K, Np K)``.

    Layout matches the column-major flattening used everywhere else
    (``u_flat = u.T.reshape(-1)``), so the block for element ``k``
    occupies rows / cols ``k·Np : (k+1)·Np``.
    """
    Np, K = mesh.Np, mesh.K
    J_per_elem = mesh.J[0, :]  # (K,) — constant for affine
    blocks = J_per_elem[:, None, None] * M_ref[None, :, :]  # (K, Np, Np)
    # scipy.linalg.block_diag would work, but doing it in JAX:
    M_global = jnp.zeros((Np * K, Np * K))
    for k in range(K):
        M_global = M_global.at[k * Np:(k + 1) * Np, k * Np:(k + 1) * Np].set(
            blocks[k]
        )
    return M_global


def factor_sipg_helmholtz_2d(
    mesh: Mesh2D,
    *,
    alpha: float,
    nu: float,
    bc_types: jax.Array,
    tau_scale: float = 200.0,
    cub: "MeshCubature | None" = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Assemble + LU-factor the Helmholtz matrix; returns ``(A, lu, piv)``."""
    A = sipg_helmholtz_matrix_2d(
        mesh, alpha=alpha, nu=nu, bc_types=bc_types, tau_scale=tau_scale,
        cub=cub,
    )
    lu, piv = jax.scipy.linalg.lu_factor(A)
    return A, lu, piv


def solve_sipg_helmholtz_2d(
    mesh: Mesh2D,
    rhs_field: jax.Array,
    *,
    alpha: float,
    nu: float,
    bc_types: jax.Array,
    u_dir: jax.Array | None = None,
    q_neu: jax.Array | None = None,
    tau_scale: float = 200.0,
    A_factored: tuple | None = None,
    iterative: IterativeConfig | None = None,
    x0: jax.Array | None = None,
    preconditioner=None,
    cub: "MeshCubature | None" = None,
) -> jax.Array:
    """Solve ``(αM − Δ) u = rhs_field`` with mixed BCs.

    Direct LU by default; pass ``iterative=IterativeConfig(...)`` for
    matrix-free CG (the Helmholtz operator is SPD).

    Pass ``preconditioner`` (a callable ``M^{-1}(v_flat) -> v_flat``,
    e.g. from :func:`make_helmholtz_block_jacobi_preconditioner`) to
    accelerate CG. Typical 1.5-2x wall-time speedup on
    moderate-and-larger meshes; ignored when ``iterative is None``.
    """
    M_ref = _local_mass(mesh)
    m1d_per_face = _face_1d_mass_matrices(mesh)
    if cub is None:
        Mf = mesh.J * (M_ref @ rhs_field)
    else:
        from ddg.dg2d.cubature import cubature_mass_apply
        Mf = cubature_mass_apply(cub, rhs_field)
    rhs = Mf
    if u_dir is not None:
        rhs = rhs - sipg_dirichlet_load_2d(
            mesh, u_dir, bc_types=bc_types, tau_scale=tau_scale,
        )
    if q_neu is not None:
        rhs = rhs + sipg_neumann_load_2d(mesh, q_neu, bc_types=bc_types)
    rhs_flat = rhs.T.reshape(-1)

    if iterative is None:
        if A_factored is None:
            A_factored = factor_sipg_helmholtz_2d(
                mesh, alpha=alpha, nu=nu, bc_types=bc_types, tau_scale=tau_scale,
                cub=cub,
            )
        _, lu, piv = A_factored
        u_flat = jax.scipy.linalg.lu_solve((lu, piv), rhs_flat)
        return u_flat.reshape(mesh.K, mesh.Np).T

    Np, K = mesh.Np, mesh.K
    def matvec(u_flat: jax.Array) -> jax.Array:
        u = u_flat.reshape(K, Np).T
        out_sipg = sipg_apply_2d(
            mesh, u, bc_types=bc_types, u_dir=None,
            tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d_per_face,
            cub=cub,
        )
        if cub is None:
            out_mass = mesh.J * (M_ref @ u)
        else:
            from ddg.dg2d.cubature import cubature_mass_apply
            out_mass = cubature_mass_apply(cub, u)
        return (alpha * out_mass + out_sipg).T.reshape(-1)
    x0_flat = x0.T.reshape(-1) if x0 is not None else jnp.zeros(Np * K)
    if iterative.method == "cg":
        u_flat, _ = jax.scipy.sparse.linalg.cg(
            matvec, rhs_flat, x0=x0_flat,
            tol=iterative.tol, atol=iterative.atol,
            maxiter=iterative.maxiter,
            M=preconditioner,
        )
    else:
        raise ValueError(f"iterative method {iterative.method!r} not supported for SPD")
    return u_flat.reshape(K, Np).T


# ---------------------------------------------------------------------------
# Block-Jacobi preconditioner
# ---------------------------------------------------------------------------
#
# At ~1e5-1e6 DOFs/field the unpreconditioned SIPG CG iteration count grows
# as O(h^-1) — order 10^3 iterations per solve. The block-diagonal of the
# SIPG (or Helmholtz) matrix, taken per element, is a good cheap
# preconditioner: it captures the dominant local stiffness (volume +
# face-self penalty) while ignoring the element-element coupling.
#
# We store the (K, Np, Np) per-element block inverses explicitly and apply
# the preconditioner as a single batched einsum — 10x faster on CPU than
# vmap'd cho_solve at our typical Np values (6-15).
#
# IMPLEMENTATION NOTE: The current path EXTRACTS diagonal blocks from the
# dense matrix build. Works up to ~5e4 DOFs/field (above that the dense
# matrix itself blows out memory). A closed-form element-local builder
# that lifts this ceiling is a TODO; left here as `_extract_block_diag_dense`
# vs the planned `_assemble_diag_blocks_local`.


def _extract_block_diag_dense(
    A_dense: jax.Array, Np: int, K: int,
) -> jax.Array:
    """Per-element ``(Np, Np)`` diagonal blocks from a dense ``(Np K, Np K)`` matrix.

    Layout assumption: the global flatten of ``(Np, K)`` field via
    ``u.T.reshape(-1)`` places element ``k``'s ``Np`` dofs contiguously at
    indices ``[k Np, k Np + Np)``.
    """
    rows = jnp.arange(K)[:, None] * Np + jnp.arange(Np)[None, :]   # (K, Np)
    cols = rows                                                     # (K, Np)
    # blocks[k, i, j] = A_dense[k*Np + i, k*Np + j]
    return A_dense[rows[:, :, None], cols[:, None, :]]              # (K, Np, Np)


def _color_elements(mesh: Mesh2D) -> tuple[np.ndarray, int]:
    """Greedy graph-colouring of the element adjacency.

    Returns ``(colour, ncolours)`` with ``colour`` a ``(K,)`` array
    such that any two face-adjacent elements get different colours.
    A 2D triangulation has at most three face neighbours per element,
    so greedy colouring needs at most four colours.
    """
    K = mesh.K
    EToE = np.asarray(mesh.EToE)
    colour = -np.ones(K, dtype=np.int64)
    for k in range(K):
        used = {colour[nb] for nb in EToE[k] if nb != k and colour[nb] >= 0}
        c = 0
        while c in used:
            c += 1
        colour[k] = c
    return colour, int(colour.max()) + 1


def extract_block_diag_matfree(
    apply_flat, mesh: Mesh2D,
) -> jax.Array:
    """Per-element ``(Np, Np)`` diagonal blocks of a SIPG-type operator
    *without* assembling the dense matrix.

    ``apply_flat`` is the operator's matrix-free apply on the
    ``(K·Np,)`` column-major-by-element flat vector (the same layout
    the CG solver uses: ``u.T.reshape(-1)`` of a ``(Np, K)`` field).

    Method: colour the element graph so no two face-adjacent elements
    share a colour. For each colour ``c`` and local node ``j``, probe
    with the vector that is ``1`` at node ``j`` of every colour-``c``
    element and ``0`` elsewhere. Because face-neighbours have a
    different colour, the probe is zero on every neighbour, so the
    operator's off-diagonal coupling contributes nothing — the result
    restricted to a colour-``c`` element ``k`` is exactly column ``j``
    of that element's self-block ``A_kk``. Total cost is
    ``ncolours × Np`` operator applies, independent of ``K``.
    """
    Np, K = mesh.Np, mesh.K
    colour_np, ncolours = _color_elements(mesh)
    colour = jnp.asarray(colour_np)
    blocks = jnp.zeros((K, Np, Np))
    for c in range(ncolours):
        mask = (colour == c).astype(jnp.float64)            # (K,)
        for j in range(Np):
            probe = jnp.zeros((K, Np)).at[:, j].set(mask)   # (K, Np)
            r = apply_flat(probe.reshape(-1)).reshape(K, Np)
            # r[k] = A_kk e_j for colour-c elements; column j of block k.
            blocks = blocks.at[:, :, j].add(
                jnp.where(mask[:, None] > 0.0, r, 0.0)
            )
    return blocks


def _block_jacobi_apply(blocks: jax.Array, Np: int, K: int):
    """Build the ``M_inv(v_flat)`` apply from per-element ``(Np,Np)`` blocks."""
    B_inv = jax.vmap(jnp.linalg.inv)(blocks)                         # (K, Np, Np)

    def M_inv(v_flat):
        v_per_elem = v_flat.reshape(K, Np)
        return jnp.einsum("kij,kj->ki", B_inv, v_per_elem).reshape(-1)

    return M_inv


def make_sipg_block_jacobi_preconditioner(
    mesh: Mesh2D,
    *,
    bc_types: jax.Array,
    tau_scale: float = 200.0,
    A_dense: jax.Array | None = None,
    matfree: bool = False,
):
    """Build a block-Jacobi preconditioner ``M^{-1}`` for the SIPG matrix.

    Returns a JAX-pure ``matvec(v_flat)`` that the JAX ``cg`` solver
    accepts as its ``M`` parameter.

    Two ways to obtain the per-element diagonal blocks:

      * **Dense** (default): extract them from the assembled
        ``(Np K)²`` matrix — pass ``A_dense`` to reuse one you already
        have, otherwise it is built here. Memory-bound at fine mesh.
      * **Matrix-free** (``matfree=True``): get the blocks with
        :func:`extract_block_diag_matfree` — ``ncolours × Np`` operator
        applies, no dense matrix. The only viable route past the
        ~5e4-DOF dense-matrix ceiling.
    """
    Np, K = mesh.Np, mesh.K
    if matfree:
        M_ref = _local_mass(mesh)
        m1d = _face_1d_mass_matrices(mesh)

        def apply_flat(u_flat):
            u = u_flat.reshape(K, Np).T
            out = sipg_apply_2d(
                mesh, u, bc_types=bc_types, u_dir=None,
                tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d,
            )
            return out.T.reshape(-1)

        blocks = extract_block_diag_matfree(apply_flat, mesh)
    else:
        if A_dense is None:
            A_dense = sipg_matrix_2d(
                mesh, bc_types=bc_types, tau_scale=tau_scale,
            )
        blocks = _extract_block_diag_dense(A_dense, Np, K)
    return _block_jacobi_apply(blocks, Np, K)


def make_helmholtz_block_jacobi_preconditioner(
    mesh: Mesh2D,
    *,
    alpha: float,
    nu: float,
    bc_types: jax.Array,
    tau_scale: float = 200.0,
    A_dense: jax.Array | None = None,
    matfree: bool = False,
):
    """Same as :func:`make_sipg_block_jacobi_preconditioner` for the
    Helmholtz operator ``α M + A_sipg``.

    ``matfree=True`` extracts the blocks with the colored-probe
    method; the matrix-free apply is the SIPG operator plus the
    element-local mass term ``α M``.
    """
    Np, K = mesh.Np, mesh.K
    if matfree:
        M_ref = _local_mass(mesh)
        m1d = _face_1d_mass_matrices(mesh)
        J_per_elem = mesh.J[0, :]                          # (K,) affine

        def apply_flat(u_flat):
            u = u_flat.reshape(K, Np).T                    # (Np, K)
            a_part = sipg_apply_2d(
                mesh, u, bc_types=bc_types, u_dir=None,
                tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d,
            )
            # alpha * M u, M block-diagonal = J_K · M_ref per element.
            m_part = alpha * J_per_elem[None, :] * (M_ref @ u)
            return (a_part + m_part).T.reshape(-1)

        blocks = extract_block_diag_matfree(apply_flat, mesh)
    else:
        if A_dense is None:
            A_dense = sipg_helmholtz_matrix_2d(
                mesh, alpha=alpha, nu=nu,
                bc_types=bc_types, tau_scale=tau_scale,
            )
        blocks = _extract_block_diag_dense(A_dense, Np, K)
    return _block_jacobi_apply(blocks, Np, K)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_sipg_poisson_2d(
    mesh: Mesh2D,
    source: jax.Array,
    *,
    bc_types: jax.Array,
    u_dir: jax.Array | None = None,
    q_neu: jax.Array | None = None,
    tau_scale: float = 200.0,
    A_factored: tuple | None = None,
    iterative: IterativeConfig | None = None,
    x0: jax.Array | None = None,
    preconditioner=None,
    cub: "MeshCubature | None" = None,
) -> jax.Array:
    """Solve ``-Δu = source`` with mixed Dirichlet / Neumann BCs.

    ``bc_types`` is ``(K, Nfaces)`` with codes :data:`BC_INTERIOR` /
    :data:`BC_DIRICHLET` / :data:`BC_NEUMANN`. ``u_dir`` (``(Np, K)``)
    supplies Dirichlet values; only boundary face traces matter.
    ``q_neu`` (``(Nfp*Nfaces, K)``) supplies Neumann values; only
    Neumann face entries are read.

    Two solver modes:

      * **Direct LU** (default): pass ``A_factored = (A, lu, piv)``
        from :func:`factor_sipg_2d` to skip re-factorisation across
        calls.
      * **Matrix-free CG** (``iterative=IterativeConfig(...)``):
        runs ``jax.scipy.sparse.linalg.cg`` on the SIPG apply.
        No dense matrix is assembled — critical for large-scale GPU
        runs. ``x0`` warm-starts the Krylov solver.

        Pass ``preconditioner`` (e.g. from
        :func:`make_sipg_block_jacobi_preconditioner`) to accelerate
        CG. Typical 1.5-2x wall-time speedup; ignored when running in
        direct-LU mode.
    """
    M_ref = _local_mass(mesh)
    m1d_per_face = _face_1d_mass_matrices(mesh)
    # Mass-weighted source: cubature on curved meshes, nodal-quad on affine.
    if cub is None:
        Mf = mesh.J * (M_ref @ source)
    else:
        from ddg.dg2d.cubature import cubature_mass_apply
        Mf = cubature_mass_apply(cub, source)
    rhs = Mf
    if u_dir is not None:
        rhs = rhs - sipg_dirichlet_load_2d(
            mesh, u_dir, bc_types=bc_types, tau_scale=tau_scale,
        )
    if q_neu is not None:
        rhs = rhs + sipg_neumann_load_2d(mesh, q_neu, bc_types=bc_types)
    rhs_flat = rhs.T.reshape(-1)

    if iterative is None:
        if A_factored is None:
            A_factored = factor_sipg_2d(
                mesh, bc_types=bc_types, tau_scale=tau_scale, cub=cub,
            )
        _, lu, piv = A_factored
        u_flat = jax.scipy.linalg.lu_solve((lu, piv), rhs_flat)
        return u_flat.reshape(mesh.K, mesh.Np).T

    Np, K = mesh.Np, mesh.K
    def raw_matvec(u_flat: jax.Array) -> jax.Array:
        u = u_flat.reshape(K, Np).T
        out = sipg_apply_2d(
            mesh, u, bc_types=bc_types, u_dir=None,
            tau_scale=tau_scale, M_ref=M_ref, m1d_per_face=m1d_per_face,
            cub=cub,
        )
        return out.T.reshape(-1)

    # Closed / fully-periodic domain: with no Dirichlet face the SIPG
    # Poisson is singular (constant nullspace) and matrix-free CG
    # diverges. Mirror the dense-path regularisation of factor_sipg_2d
    # with an eps-shift on the matrix-free operator. The shift scale is
    # estimated from one apply on a deterministic mean-zero probe
    # (A·constant = 0, so the probe must not be constant).
    singular = not bool(jnp.any(bc_types == BC_DIRICHLET))
    if singular:
        idx = jnp.arange(Np * K, dtype=jnp.float64)
        probe = jnp.sin(idx)
        probe = probe - jnp.mean(probe)
        scale = (jnp.linalg.norm(raw_matvec(probe))
                 / jnp.linalg.norm(probe))
        shift = 1.0e-6 * scale
        def matvec(u_flat: jax.Array) -> jax.Array:
            return raw_matvec(u_flat) + shift * u_flat
        # Deflate the constant nullspace: project the RHS onto
        # range(A) so the system is consistent, keeping the solution's
        # gauge bounded (the ε-shift alone would leave an O(b̄/ε) offset).
        rhs_flat = rhs_flat - jnp.mean(rhs_flat)
    else:
        matvec = raw_matvec

    x0_flat = x0.T.reshape(-1) if x0 is not None else jnp.zeros(Np * K)
    if iterative.method == "cg":
        u_flat, _ = jax.scipy.sparse.linalg.cg(
            matvec, rhs_flat, x0=x0_flat,
            tol=iterative.tol, atol=iterative.atol,
            maxiter=iterative.maxiter,
            M=preconditioner,
        )
    else:
        raise ValueError(f"iterative method {iterative.method!r} not supported for SPD")
    return u_flat.reshape(K, Np).T
