"""Brezzi-Douglas-Marini (BDM) basis on triangles for H(div)-conforming DG.

This module is the *foundation* for an H(div)-conforming incompressible
NS solver: it provides the reference-element BDM_k basis evaluation
and the geometric pieces needed to assemble the divergence / gradient /
mass operators in the H(div) velocity space.

Lowest order, BDM_1 (the only order implemented here):

  * Velocity space on each triangle: ``(P_1)^2``                — 6 scalar coeffs
  * Pressure pairing (downstream):   ``P_0``  (constant per element)
  * Velocity DOFs per element:       6   (2 normal-flux per edge × 3 edges)
  * Normal trace on each edge:       linear function ∈ ``P_1(edge)``
  * Element-local divergence:        constant ∈ ``P_0(K)``

The key property — and what motivates this entire module — is that
when a globally-conforming H(div) velocity is paired with the
matching pressure space, ``∇·u_h ≡ 0`` *pointwise* (not just in
some weak average sense). No divergence or continuity penalty is
needed for mass conservation, and the scheme is pressure-robust by
construction. Fehn-Kronbichler-Lube 2025, §4.8 explains the role
of H(div) in this context vs the penalty-stabilised L^2-DG approach
in §4.7.

This file contains:

  * :func:`bdm1_basis_at_points`  — evaluate the 6 BDM_1 basis
    functions on the reference triangle at arbitrary ``(r, s)`` points.
  * :func:`bdm1_reference_dof_coordinates`  — the canonical reference-edge
    nodal points where DOFs are defined (2 per edge).
  * :func:`bdm1_divergence_basis`  — divergence of each BDM_1 basis
    function on the reference triangle (constants).
  * :class:`BDM1Topology` + :func:`build_bdm1_topology` — global
    edge-shared DOF indexing with the per-element sign convention that
    makes normal traces H(div)-conforming across element boundaries.

The next steps (separate follow-up work):
  * Mass matrix in BDM_1 with Piola-mapped basis (element-local).
  * Divergence operator from BDM_1 → P_0.
  * Gradient operator from L^2-DG (pressure space) → BDM_1.
  * Lax-Friedrichs convective flux on the tangential trace
    (normal trace is automatic).
  * Splitting / coupled time integrators that use these operators.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.mesh import Mesh2D


# ---------------------------------------------------------------------------
# Reference-triangle geometry (Hesthaven convention)
# ---------------------------------------------------------------------------
#
# Vertices:   v0 = (-1, -1)
#             v1 = ( 1, -1)
#             v2 = (-1,  1)
#
# Edges (CCW around the reference triangle, with the outward unit normal
# we use for normal-trace DOFs):
#   edge 0:  v0 → v1   (s = -1, bottom)                    n_0 = ( 0, -1)
#   edge 1:  v1 → v2   (r + s = 0, hypotenuse)             n_1 = (1/√2)·(1, 1)
#   edge 2:  v2 → v0   (r = -1, left)                      n_2 = (-1,  0)
#
# Edge length in the reference (r, s) plane:
#   |edge 0| = 2          (bottom: r ∈ [-1, 1])
#   |edge 1| = 2√2        (hypotenuse: from (1,-1) to (-1,1))
#   |edge 2| = 2          (left: s ∈ [-1, 1])


_REF_VERTICES = jnp.array([[-1.0, -1.0],
                            [ 1.0, -1.0],
                            [-1.0,  1.0]])

_REF_FACE_NORMALS = jnp.array([
    [0.0, -1.0],                           # edge 0 (bottom)
    [1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)],  # edge 1 (hypotenuse)
    [-1.0, 0.0],                           # edge 2 (left)
])


# ---------------------------------------------------------------------------
# BDM_1 DOF locations on each edge
# ---------------------------------------------------------------------------
#
# BDM_1 has 2 DOFs per edge — values of the normal-trace polynomial at
# 2 Gauss-Legendre points along the edge. We use the standard
# ξ_GL = ±1/√3, parameterising each edge from one endpoint to the other.

_GL_NODES = jnp.array([-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)])


# Euclidean (NOT parameterised) length of each reference-triangle edge.
# Face 0 and 2 are legs of length 2; face 1 is the hypotenuse of length 2*sqrt(2).
# This is the length used by Nanson's formula on the unit normal:
#   |J * DF^{-T} n_ref| = L_phys / L_ref_euclidean.
# The mesh's stored ``sJ`` equals ``L_phys / 2`` (the *parameterised* Jacobian
# along an isoparametric face), which agrees with the Euclidean ratio for
# legs but is off by 1/sqrt(2) for the hypotenuse. Downstream BDM
# coefficient calculations need the Euclidean version, so we rescale
# explicitly.
_REF_FACE_EUCLIDEAN_LENGTH = jnp.array([2.0, 2.0 * jnp.sqrt(2.0), 2.0])
_FACE_JAC_RESCALE = 2.0 / _REF_FACE_EUCLIDEAN_LENGTH    # (1, 1/sqrt(2), 1)


def bdm1_reference_dof_coordinates() -> jnp.ndarray:
    """Reference-element (r, s) coords of the 6 BDM_1 DOFs.

    Returns an ``(6, 2)`` array. DOF ``(i, j)`` (face ``i``, node ``j``)
    is at row ``2*i + j``. Order:
      (face 0, node 0), (face 0, node 1),
      (face 1, node 0), (face 1, node 1),
      (face 2, node 0), (face 2, node 1).
    """
    # Edge 0: from v0=(-1,-1) to v1=(1,-1).  Parameter ξ ∈ [-1, 1].
    #   r = ξ, s = -1
    pts0 = jnp.stack([_GL_NODES, -jnp.ones_like(_GL_NODES)], axis=1)
    # Edge 1: from v1=(1,-1) to v2=(-1,1).  Parameter ξ ∈ [-1, 1].
    #   r = -ξ, s = ξ        (so at ξ=-1 we are at v1, at ξ=1 we are at v2)
    pts1 = jnp.stack([-_GL_NODES, _GL_NODES], axis=1)
    # Edge 2: from v2=(-1,1) to v0=(-1,-1).  Parameter ξ ∈ [-1, 1].
    #   r = -1, s = -ξ        (so ξ=-1 → v2, ξ=1 → v0)
    pts2 = jnp.stack([-jnp.ones_like(_GL_NODES), -_GL_NODES], axis=1)
    return jnp.concatenate([pts0, pts1, pts2], axis=0)


# ---------------------------------------------------------------------------
# BDM_1 basis on the reference triangle
# ---------------------------------------------------------------------------
#
# Each basis function is a 2-vector field in (P_1)^2, written as
#     φ(r, s) = (a + b r + c s,  d + e r + f s)
# i.e. 6 scalar coefficients ∈ R^6. The 6 DOFs are linear functionals
# evaluating the normal trace at the 6 reference DOF points. The
# basis matrix is thus the inverse of the 6×6 DOF-evaluation matrix.

def _bdm1_dof_matrix() -> jnp.ndarray:
    """Build the 6×6 matrix ``D`` whose entry ``D[ℓ, m]`` is the m-th
    monomial evaluated under the ℓ-th DOF functional.

    Monomial ordering for the 6 coefficients (a, b, c, d, e, f):
      φ_x = a + b r + c s
      φ_y = d + e r + f s

    DOF functional ℓ = (face i, node j): take ``(φ_x, φ_y) · n_i``
    at the j-th DOF point on face i.
    """
    dof_pts = bdm1_reference_dof_coordinates()        # (6, 2)
    face_idx = jnp.repeat(jnp.arange(3), 2)            # [0,0,1,1,2,2]
    normals = _REF_FACE_NORMALS[face_idx]              # (6, 2)
    r, s = dof_pts[:, 0], dof_pts[:, 1]
    # Six monomials evaluated as 2-vectors:
    #   m_a = (1, 0),   m_b = (r, 0),   m_c = (s, 0)
    #   m_d = (0, 1),   m_e = (0, r),   m_f = (0, s)
    # D[ℓ, m] = m_m(dof_pts[ℓ]) · n_ℓ
    nx, ny = normals[:, 0], normals[:, 1]
    D = jnp.stack([
        nx * 1.0,                                       # coeff a
        nx * r,                                         # coeff b
        nx * s,                                         # coeff c
        ny * 1.0,                                       # coeff d
        ny * r,                                         # coeff e
        ny * s,                                         # coeff f
    ], axis=1)
    return D


_BDM1_COEFFS = jnp.linalg.inv(_bdm1_dof_matrix())     # (6, 6)
# _BDM1_COEFFS[:, ℓ] holds the (a, b, c, d, e, f) coefficients of
# basis function ℓ. Each column is a basis function.


def bdm1_basis_at_points(r: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the 6 BDM_1 basis functions at points ``(r, s)``.

    Returns shape ``(npts, 6, 2)`` — for each point, 6 basis functions,
    each a 2-vector ``(φ_x, φ_y)``.
    """
    r = jnp.atleast_1d(r); s = jnp.atleast_1d(s)
    # Monomial matrix at points: shape (npts, 6), columns (1, r, s, 0, 0, 0)
    # for the x-component contributions and (0, 0, 0, 1, r, s) for y.
    npts = r.shape[0]
    one = jnp.ones_like(r); zero = jnp.zeros_like(r)
    # Pack each basis function (column of _BDM1_COEFFS) into a 2-vector
    # at every point.
    # Basis ℓ:  φ_x = a_ℓ + b_ℓ r + c_ℓ s
    #          φ_y = d_ℓ + e_ℓ r + f_ℓ s
    # _BDM1_COEFFS has shape (6, 6); rows are coeffs (a..f), cols are basis.
    coeffs = _BDM1_COEFFS                             # (6, 6)
    a, b, c, d, e, f = coeffs                          # each (6,)
    phi_x = (a[None, :] + b[None, :] * r[:, None]
             + c[None, :] * s[:, None])               # (npts, 6)
    phi_y = (d[None, :] + e[None, :] * r[:, None]
             + f[None, :] * s[:, None])               # (npts, 6)
    return jnp.stack([phi_x, phi_y], axis=2)           # (npts, 6, 2)


def bdm1_divergence_basis() -> jnp.ndarray:
    """Divergence of each BDM_1 basis function in reference coordinates.

    Since each basis function is ``(a + b r + c s, d + e r + f s)``,
    ``∇·φ = b + f`` — a constant in reference coordinates. Returns shape
    ``(6,)`` — one divergence value per basis function. To get the
    physical divergence on element ``K``, apply the chain rule with the
    affine map (i.e. multiply by ``1/J_K`` and account for ``r_x, s_x``,
    ``r_y, s_y``).
    """
    coeffs = _BDM1_COEFFS
    b, f = coeffs[1], coeffs[5]                        # each (6,)
    return b + f


# ---------------------------------------------------------------------------
# Mesh-level BDM_1 DOF topology
# ---------------------------------------------------------------------------
#
# Each triangle K carries 6 local BDM_1 DOFs (2 per edge × 3 edges). On a
# shared interior edge between K and K', both elements see *the same 2
# physical DOFs* — the normal-trace values at the 2 Gauss-Legendre points
# along the edge. We index those DOFs globally so the saddle-point
# assembly produces a single coupled system, and we attach a per-element
# sign to encode that K's "outward normal" and K's "outward normal" on
# the shared edge point in opposite directions.
#
# Canonical-owner convention: of the two triangles sharing an edge, the
# one with the *smaller element index* owns the edge. The owner's local
# DOFs get sign +1, the other side gets sign -1. Boundary edges have a
# single owner with sign +1.
#
# DOF ordering on a shared edge: the 2 GL points are at the same physical
# positions from both sides, but the two triangles parameterise the edge
# in opposite directions (CCW traversal). When the non-owner triangle K'
# looks at the edge, its local DOF index 0 corresponds to the owner's
# local DOF index 1, and vice versa. We detect this from the vertex
# orientation (EToV).


@dataclass(frozen=True)
class BDMTopology:
    """Global BDM_k DOF indexing for a triangular mesh.

    Per-element local DOF layout: ``[face_0_DOFs (k+1), face_1_DOFs
    (k+1), face_2_DOFs (k+1), interior_DOFs ((k+1)(k-1))]`` for total
    ``(k+1)(k+2)``. Face DOFs are shared with the neighbouring element
    (with sign convention and possible per-face permutation when the
    two elements traverse the edge in opposite directions); interior
    DOFs are owned exclusively by their element.

    Attributes
    ----------
    order : int
        Polynomial order ``k`` of the BDM space.
    n_global_dofs : int
        Total number of distinct DOFs across the mesh.
    local_to_global : jax.Array, shape ``(K, (k+1)(k+2))``
        (element, local-DOF) -> global DOF index.
    local_sign : jax.Array, shape ``(K, (k+1)(k+2))``
        +/-1 sign per local DOF. Face DOFs: +1 if this element is the
        canonical owner of the edge (smaller element index), -1 if not.
        Interior DOFs are always +1 (no sharing).
    piola_scale : jax.Array, shape ``(K, (k+1)(k+2))``
        ``L_phys / L_ref_euclidean`` per face DOF; 1.0 per interior
        DOF. Multiplied through gather to convert global normal-trace
        DOFs to the unsigned Piola-mapped basis coefficient.
    """
    order: int
    n_global_dofs: int
    local_to_global: jax.Array
    local_sign: jax.Array
    piola_scale: jax.Array


# Back-compat alias for code/tests that imported the original
# ``BDM1Topology`` name. The generalised ``BDMTopology`` carries the
# extra ``order`` field but is otherwise field-compatible.
BDM1Topology = BDMTopology


def build_bdm_topology(mesh: Mesh2D, order: int) -> BDMTopology:
    """Assemble the global BDM_k DOF indexing for ``mesh``.

    For each element, walks the 3 faces in order and assigns ``k+1``
    DOFs per face. When a face has no neighbour with smaller index
    (boundary, or this side is the canonical owner) we emit ``k+1``
    fresh global DOFs. Otherwise we borrow from the neighbour with
    sign ``-1`` and -- in the typical CCW-reversed case -- reverse the
    per-face DOF ordering so K's node ``j`` is the same physical DOF
    as K's node ``k+1-1-j`` from the neighbour's perspective.

    After the face pass, each element gets its own ``(k+1)(k-1)``
    interior DOFs (no sharing).
    """
    K = int(mesh.K)
    n_per_face = order + 1
    n_face_total = 3 * n_per_face
    n_interior = (order + 1) * (order - 1)
    n_local = (order + 1) * (order + 2)

    EToE = np.asarray(mesh.EToE)
    EToF = np.asarray(mesh.EToF)
    EToV = np.asarray(mesh.EToV)

    local_to_global = -np.ones((K, n_local), dtype=np.int64)
    local_sign = np.zeros((K, n_local), dtype=np.int64)
    next_g = 0

    # Pass 1: face DOFs.
    for k in range(K):
        for f in range(3):
            k_nbr = int(EToE[k, f])
            f_nbr = int(EToF[k, f])
            base = n_per_face * f

            if k_nbr == k or k < k_nbr:
                # Canonical owner: emit n_per_face fresh global DOFs.
                for j in range(n_per_face):
                    local_to_global[k, base + j] = next_g + j
                    local_sign[k, base + j] = +1
                next_g += n_per_face
            else:
                nbr_base = n_per_face * f_nbr
                v_K_start = int(EToV[k, f])
                v_K_end = int(EToV[k, (f + 1) % 3])
                v_Kp_start = int(EToV[k_nbr, f_nbr])
                v_Kp_end = int(EToV[k_nbr, (f_nbr + 1) % 3])
                if v_Kp_start == v_K_end and v_Kp_end == v_K_start:
                    # Reversed traversal: K node j <-> K' node (n-1-j).
                    for j in range(n_per_face):
                        local_to_global[k, base + j] = local_to_global[
                            k_nbr, nbr_base + (n_per_face - 1 - j)
                        ]
                elif v_Kp_start == v_K_start and v_Kp_end == v_K_end:
                    # Same direction.
                    for j in range(n_per_face):
                        local_to_global[k, base + j] = local_to_global[
                            k_nbr, nbr_base + j
                        ]
                else:
                    raise RuntimeError(
                        f"BDM topology: shared edge between elements {k} "
                        f"and {k_nbr} has inconsistent vertex labels "
                        f"K=({v_K_start},{v_K_end}), "
                        f"K'=({v_Kp_start},{v_Kp_end})"
                    )
                for j in range(n_per_face):
                    local_sign[k, base + j] = -1

    # Pass 2: interior DOFs (no sharing, sign +1).
    for k in range(K):
        for j in range(n_interior):
            local_to_global[k, n_face_total + j] = next_g
            local_sign[k, n_face_total + j] = +1
            next_g += 1

    # Piola scaling: L_phys / L_ref_euclidean per face DOF; 1.0 for
    # interior DOFs. Kept in JAX so gradients flow back through
    # ``mesh.sJ`` for differentiable-mesh applications.
    sJ_face = mesh.sJ.reshape((mesh.Nfp, mesh.Nfaces, K), order="F")[0]    # (Nfaces, K)
    edge_jac = sJ_face.T                                                    # (K, Nfaces) JAX
    rescale = jnp.asarray(_FACE_JAC_RESCALE)                                # (3,)
    piola_per_face = edge_jac * rescale[None, :]                            # (K, 3)
    piola_face_dofs = jnp.repeat(piola_per_face, n_per_face, axis=1)        # (K, n_face_total)
    piola_interior = jnp.ones((K, n_interior))
    piola_scale = jnp.concatenate([piola_face_dofs, piola_interior], axis=1)  # (K, n_local)

    return BDMTopology(
        order=int(order),
        n_global_dofs=int(next_g),
        local_to_global=jnp.asarray(local_to_global),
        local_sign=jnp.asarray(local_sign),
        piola_scale=piola_scale,
    )


def refresh_piola_scale(
    topology: BDMTopology, mesh: Mesh2D,
) -> BDMTopology:
    """Return a copy of ``topology`` with ``piola_scale`` recomputed
    from ``mesh.sJ``. ``local_to_global`` and ``local_sign`` are kept
    untouched -- they depend only on integer connectivity and don't
    change as the mesh is deformed.

    For differentiable-mesh applications where the integer connectivity
    is invariant but the geometry shifts, cache the BDMTopology once
    host-side at the nominal mesh and call this helper inside the
    jit'd forward to avoid the ``np.asarray(mesh.EToE)`` etc.\\
    conversions in :func:`build_bdm_topology` that fail under
    tracing.
    """
    import dataclasses
    k = topology.order
    n_per_face = k + 1                               # k+1 DOFs per face
    n_face_total = 3 * n_per_face
    n_local = (k + 1) * (k + 2)
    n_interior = n_local - n_face_total
    K = mesh.K
    sJ_face = mesh.sJ.reshape(
        (mesh.Nfp, mesh.Nfaces, K), order="F",
    )[0]                                              # (Nfaces, K)
    edge_jac = sJ_face.T                              # (K, Nfaces)
    rescale = jnp.asarray(_FACE_JAC_RESCALE)
    piola_per_face = edge_jac * rescale[None, :]      # (K, 3)
    piola_face_dofs = jnp.repeat(piola_per_face, n_per_face, axis=1)
    piola_interior = jnp.ones((K, n_interior))
    piola_scale = jnp.concatenate(
        [piola_face_dofs, piola_interior], axis=1)
    return dataclasses.replace(topology, piola_scale=piola_scale)


def build_bdm1_topology(mesh: Mesh2D) -> BDMTopology:
    """BDM_1 topology -- thin wrapper around the order-parametric
    :func:`build_bdm_topology`. Kept for backward compatibility with
    pre-refactor call sites; new code should call ``build_bdm_topology``
    directly with an explicit order.
    """
    return build_bdm_topology(mesh, order=1)


def bdm1_gather(topology: BDM1Topology, u_global: jax.Array) -> jax.Array:
    """Scatter global BDM_1 DOFs to per-element local form.

    ``u_local[K, ℓ] = local_sign[K, ℓ] * piola_scale[K, ℓ] *
    u_global[local_to_global[K, ℓ]]``. Returns shape ``(K, 6)``.

    The ``piola_scale`` factor rescales the global basis (normalised so
    its physical normal trace at the matching GL point is 1) to the
    unsigned Piola-mapped basis ``(1/J_K) DF_K φ^ref`` used by the
    per-element apply matrices. Without it, the BDM space wouldn't be
    H(div)-conforming when adjacent elements map the shared edge from
    different reference face indices.
    """
    weight = topology.local_sign * topology.piola_scale
    return weight * u_global[topology.local_to_global]


def bdm1_scatter_add(
    topology: BDM1Topology, u_local: jax.Array
) -> jax.Array:
    """Sum per-element local contributions into a global DOF vector.

    The adjoint of :func:`bdm1_gather`. For each global DOF ``g``,
    accumulate ``sum_{(K,ℓ): local_to_global[K,ℓ]==g} local_sign[K, ℓ] *
    piola_scale[K, ℓ] * u_local[K, ℓ]``. Interior DOFs receive two
    contributions; the signed sum encodes H(div)-conforming assembly.

    Used in the test-functional adjoint when going from per-element apply
    outputs back to global DOF residuals.
    """
    out = jnp.zeros(topology.n_global_dofs, dtype=u_local.dtype)
    weight = topology.local_sign * topology.piola_scale
    contributions = (weight * u_local).reshape(-1)
    indices = topology.local_to_global.reshape(-1)
    return out.at[indices].add(contributions)


# ---------------------------------------------------------------------------
# BDM FunctionSpace: Piola-mapped basis + element-local mass
# ---------------------------------------------------------------------------
#
# Piola contravariant map for H(div):
#     φ^phys(x) = sign_K_ℓ · (1/J_K) · DF_K · φ^ref(F_K^{-1}(x))
#
# This map preserves normal traces (up to the sign convention), so
# H(div) functions in physical space correspond to H(div) functions on
# the reference triangle. For affine elements ``DF_K``, ``J_K`` are
# constant per element. Mass matrix on element K:
#
#     M_K[i, j] = ∫_K φ^phys_i · φ^phys_j dx
#              = sign_i sign_j / J_K · ∫_T φ^ref_i^T G_K φ^ref_j dr ds
#
# where ``G_K = DF_K^T DF_K`` is the (symmetric, constant-per-element)
# metric tensor. We pre-compute four reference 6×6 tensors
# ``M_ref_xx, M_ref_xy, M_ref_yx, M_ref_yy`` and assemble M_K from them.


# Hammer 3-point cubature on the reference triangle, exact for degree 2.
# Reference triangle [(-1,-1), (1,-1), (-1,1)]; edge midpoints; w = 2/3.
_HAMMER_PTS = jnp.array([
    [0.0, -1.0],   # midpoint of edge 0 (bottom)
    [-1.0, 0.0],   # midpoint of edge 2 (left)
    [0.0, 0.0],   # midpoint of edge 1 (hypotenuse)
])
_HAMMER_W = jnp.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])


def _bdm1_reference_mass_tensors():
    """Compute the four reference-triangle 6×6 tensors
    ``M_ref_xx, M_ref_xy, M_ref_yx, M_ref_yy``:

        M_ref_ab[i, j] = ∫_T φ^ref_i_a (r,s) · φ^ref_j_b (r,s) dr ds

    Used to assemble per-element ``M_K`` via the metric ``G_K``.
    """
    r, s = _HAMMER_PTS[:, 0], _HAMMER_PTS[:, 1]
    w = _HAMMER_W
    phi = bdm1_basis_at_points(r, s)        # (Q=3, 6, 2)
    phi_x = phi[:, :, 0]                     # (Q, 6)
    phi_y = phi[:, :, 1]                     # (Q, 6)
    # M_ab[i, j] = sum_q w_q * phi_a[q, i] * phi_b[q, j]
    M_xx = jnp.einsum("q,qi,qj->ij", w, phi_x, phi_x)
    M_xy = jnp.einsum("q,qi,qj->ij", w, phi_x, phi_y)
    M_yx = jnp.einsum("q,qi,qj->ij", w, phi_y, phi_x)
    M_yy = jnp.einsum("q,qi,qj->ij", w, phi_y, phi_y)
    return M_xx, M_xy, M_yx, M_yy


_BDM1_M_REF_XX, _BDM1_M_REF_XY, _BDM1_M_REF_YX, _BDM1_M_REF_YY = (
    _bdm1_reference_mass_tensors()
)


def _element_jacobian_matrices(mesh: Mesh2D) -> tuple[jax.Array, jax.Array]:
    """Per-element affine Jacobian ``DF_K`` (2x2) and ``G_K = DF_K^T DF_K``.

    For affine triangles, ``rx, sx, ry, sy`` are constant per element. We
    sample at node 0 and reconstruct DF_K from the relation
    ``DF_K^{-1} = [[rx, ry], [sx, sy]]`` (each row is a gradient in
    physical space), giving ``DF_K = J_K · [[sy, -ry], [-sx, rx]]``.

    Returns
    -------
    DF_K : ``(K, 2, 2)``
    G_K  : ``(K, 2, 2)``, ``DF_K.T @ DF_K``
    """
    rx, sx, ry, sy = mesh.rx[0], mesh.sx[0], mesh.ry[0], mesh.sy[0]
    J = mesh.J[0]  # (K,)
    DF = jnp.stack(
        [
            jnp.stack([J * sy, -J * ry], axis=-1),
            jnp.stack([-J * sx, J * rx], axis=-1),
        ],
        axis=-2,
    )                                            # (K, 2, 2)
    G = jnp.einsum("kab,kac->kbc", DF, DF)       # DF^T DF, (K, 2, 2)
    return DF, G


@dataclass(frozen=True)
class BDM:
    """BDM_k H(div)-conforming function space on a triangular mesh.

    Wraps a :class:`Mesh2D` (geometry only -- the BDM basis is built
    from the reference machinery in this module, not the mesh's
    Lagrangian Vandermonde) plus the global DOF :class:`BDMTopology`.
    ``Np = (order+1)(order+2)`` is the per-element local DOF count;
    the assembled-system dimension is ``total_dofs =
    topology.n_global_dofs``.

    Many downstream operators (``bdm1_*`` names) currently still
    require ``order=1``; only the reference-basis machinery
    (``bdm_k_basis_at_points`` etc.) is fully order-parametric for now.

    Constructed via :meth:`BDM.from_mesh` for ergonomics.
    """

    mesh: Mesh2D
    topology: BDMTopology
    order: int = 1

    @property
    def Np(self) -> int:
        k = self.order
        return (k + 1) * (k + 2)

    @property
    def K(self) -> int:
        return self.mesh.K

    @property
    def components(self) -> int:
        return 2  # vector-valued by construction

    @property
    def total_dofs(self) -> int:
        return self.topology.n_global_dofs

    @classmethod
    def from_mesh(cls, mesh: Mesh2D, *, order: int = 1) -> "BDM":
        topology = build_bdm_topology(mesh, order=order)
        return cls(mesh=mesh, topology=topology, order=order)


def bdm1_element_local_mass(V: BDM) -> jax.Array:
    """Per-element BDM_1 mass matrix on the *unsigned* Piola-mapped basis.

    Returns ``(K, 6, 6)`` -- one symmetric SPD 6×6 matrix per element.
    No topology signs are folded in: this is the intrinsic per-element
    operator on the unsigned basis ``(1/J_K) DF_K φ^ref``. Signs enter
    only via :func:`bdm1_gather` / :func:`bdm1_scatter_add` when this
    matrix is applied to global DOFs (the convention used by all apply
    functions in this module).
    """
    _, G = _element_jacobian_matrices(V.mesh)   # (K, 2, 2)
    J = V.mesh.J[0]                              # (K,)
    g00 = G[:, 0, 0]                             # (K,)
    g01 = G[:, 0, 1]                             # (K,) (== G[:, 1, 0])
    g11 = G[:, 1, 1]                             # (K,)
    inv_J = 1.0 / J                              # (K,)
    # M_K_unsigned[k, i, j] = (1/J_k) * (g00 * M_xx + g01 * (M_xy + M_yx)
    #                                    + g11 * M_yy)[i, j]
    return (
        _BDM1_M_REF_XX[None, :, :] * (inv_J * g00)[:, None, None]
        + (_BDM1_M_REF_XY + _BDM1_M_REF_YX)[None, :, :]
        * (inv_J * g01)[:, None, None]
        + _BDM1_M_REF_YY[None, :, :] * (inv_J * g11)[:, None, None]
    )


def bdm1_physical_basis_at_ref_points(
    V: BDM, r: jax.Array, s: jax.Array
) -> jax.Array:
    """Evaluate the *physical* (Piola-mapped, signed) BDM_1 basis at
    reference-element points ``(r, s)``, on every element.

    Returns shape ``(K, npts, 6, 2)`` -- for each element K and each
    reference-element evaluation point, the 6 physical basis functions
    as 2-vectors:

        φ^phys_K_ℓ(F_K(r, s)) = sign_K_ℓ · (1/J_K) · DF_K · φ^ref_ℓ(r, s)

    Used by downstream operator builders (convective trace, divergence,
    cross-space pressure pairing).
    """
    DF, _ = _element_jacobian_matrices(V.mesh)   # (K, 2, 2)
    J = V.mesh.J[0]                              # (K,)
    phi_ref = bdm1_basis_at_points(r, s)         # (npts, 6, 2)

    # Multiply by DF_K and divide by J_K, per element. Broadcasting:
    #   DF[k, a, b] · phi_ref[q, ℓ, b] -> phi_phys_unsigned[k, q, ℓ, a]
    phi_phys = jnp.einsum("kab,qlb->kqla", DF, phi_ref) / J[:, None, None, None]

    # Per-DOF sign:
    sign = V.topology.local_sign                 # (K, 6)
    return phi_phys * sign[:, None, :, None]


# ---------------------------------------------------------------------------
# Divergence and gradient against P_0 pressure
# ---------------------------------------------------------------------------
#
# For BDM_1 paired with element-piecewise-constant pressure P_0, the
# divergence of a velocity ``u ∈ V_u`` is *constant per element* (one of
# the defining properties of the BDM-P_0 pair: ∇·BDM_1 ⊆ P_0). The
# action on a P_0 test function ``q = sum_K q_K * 1_K`` is therefore an
# element-local sum:
#
#     [D u]_K = ∫_K (∇·u) dx
#             = (∇·u)|_K · |K|
#             = (sum_ℓ u_local[K, ℓ] · (b_ℓ + f_ℓ) / J_K) · (2 · J_K)
#             = 2 · sum_ℓ u_local[K, ℓ] · div_ref[ℓ]
#
# where ``div_ref[ℓ] = b_ℓ + f_ℓ`` from :func:`bdm1_divergence_basis` and
# the factor 2 comes from the reference-triangle area. ``J_K`` cancels,
# so the operator is purely topological-algebraic.
#
# The gradient operator G : P_0 → V_u is defined as the *negative
# transpose* of D in the saddle-point sense (G = -D^T), matching the
# standard Stokes / Navier-Stokes block structure. Concretely:
#
#     [G p]_global = -D^T · p   in global-DOF coordinates


def bdm_p0_divergence_apply(
    V_u: BDM, u_global: jax.Array
) -> jax.Array:
    """Apply ``D : BDM_1 → P_0`` to a global BDM_1 DOF vector.

    Returns ``D u`` of shape ``(K,)`` -- the per-element integrated
    divergence ``∫_K ∇·u dx``. By H(div)-conformity the inter-element
    flux terms vanish *automatically* (no centered/upwind/Lax-Friedrichs
    choice needed): the normal trace is single-valued across each
    interior face, and the divergence is a per-element computation.
    """
    u_local = bdm1_gather(V_u.topology, u_global)        # (K, 6), signed
    div_ref = bdm1_divergence_basis()                    # (6,)
    return 2.0 * jnp.einsum("kl,l->k", u_local, div_ref)


def bdm_p0_gradient_apply(
    V_u: BDM, p_per_elem: jax.Array
) -> jax.Array:
    """Apply ``G : P_0 → BDM_1`` (defined as ``-D^T``) to a per-element
    pressure DOF vector ``p_per_elem`` of shape ``(K,)``.

    Returns a global BDM_1 DOF vector ``G p`` of shape
    ``(V_u.total_dofs,)``. Adjoint of :func:`bdm_p0_divergence_apply` in
    the saddle-point sign convention: for the system

        [ A   G ] [u]   [b_u]
        [ -D  S ] [p] = [b_p]

    we set ``G = -D^T``, so ``<G p, u>_BDM = -<p, D u>_P0`` for any
    matched pair (u, p) -- this is what makes the off-diagonal blocks
    of the saddle-point exactly skew-adjoint. The H(div) conformity
    guarantees no boundary correction is needed.
    """
    div_ref = bdm1_divergence_basis()                    # (6,)
    # Local-frame contributions: for element K with pressure p_K,
    # [G p]_local[K, ℓ] = -2 * p_K * div_ref[ℓ]. The sign per DOF is
    # encoded by the topology's scatter, which handles the +/- on the
    # two sides of each interior face.
    G_local = -2.0 * p_per_elem[:, None] * div_ref[None, :]    # (K, 6)
    return bdm1_scatter_add(V_u.topology, G_local)


# ---------------------------------------------------------------------------
# Viscous bilinear form: volume term
# ---------------------------------------------------------------------------
#
# For Fehn-style H(div) IPG on the BDM_1 velocity space, the symmetric
# viscous bilinear form is
#
#   a_visc(u, v) = ν ∫_Ω ∇u : ∇v dx
#                − ν Σ_F ∫_F {{∇u·n}} · [v]_t dS    (consistency)
#                − ν Σ_F ∫_F [u]_t · {{∇v·n}} dS    (symmetry)
#                + ν Σ_F ∫_F τ [u]_t · [v]_t dS    (penalty).
#
# Since BDM_1 is H(div)-conforming, [u·n] ≡ 0 -- the inter-element jump
# [u] = u_+ - u_- is *automatically* tangential. The face form is
# therefore unchanged from the standard SIPG layout, just on a different
# velocity space.
#
# This section implements the volume term only:
#   [A_vol u]_K_ell = 2 ν / J_K * sign_K_ell * sum_m sign_K_m
#                       * (M_K_ell : M_K_m) * u_local[K, m]
# where M_K_ell = DF_K · ∇φ_ell^ref · DF_K^{-1} is the constant-per-
# affine-element physical gradient of basis ell. The face terms are in
# the next code block.


def _bdm1_reference_gradients() -> jax.Array:
    """Reference-element gradient of each BDM_1 basis function.

    Each basis is ``(a + b r + c s, d + e r + f s)`` with reference
    gradient (constant in r, s):
        ∇φ_ref[ell, i, j] = ∂_j φ_ref_ell_i
        -> row i=0: (b_ell, c_ell), row i=1: (e_ell, f_ell).
    Returns shape ``(6, 2, 2)``.
    """
    b = _BDM1_COEFFS[1]
    c = _BDM1_COEFFS[2]
    e = _BDM1_COEFFS[4]
    f = _BDM1_COEFFS[5]
    return jnp.stack(
        [jnp.stack([b, c], axis=1), jnp.stack([e, f], axis=1)],
        axis=1,
    )  # (6, 2, 2)


_BDM1_GRAD_REF = _bdm1_reference_gradients()


def _bdm1_physical_gradient_matrices(V: BDM) -> jax.Array:
    """Per-element, per-basis physical-gradient matrices
    ``M_K_ell = DF_K · ∇φ_ell^ref · DF_K^{-1}`` (unsigned, sign convention
    folded in by callers).

    Returns shape ``(K, 6, 2, 2)``.
    """
    DF, _ = _element_jacobian_matrices(V.mesh)       # (K, 2, 2)
    DF_inv = jnp.linalg.inv(DF)                       # (K, 2, 2)
    grad_ref = _BDM1_GRAD_REF                         # (6, 2, 2)
    # M_K_ell = DF_K @ grad_ref_ell @ DF_K^{-1}
    # einsum: M[k, l, i, j] = DF[k, i, a] grad_ref[l, a, b] DF_inv[k, b, j]
    return jnp.einsum("kia,lab,kbj->klij", DF, grad_ref, DF_inv)


def bdm1_viscous_volume_stiffness(V: BDM, nu: float) -> jax.Array:
    """Per-element 6×6 viscous-volume stiffness on the *unsigned* basis:

        S_K[i, j] = ν · ∫_K (∇φ_i^unsigned : ∇φ_j^unsigned) dx
                  = 2 ν / J_K · (M_i : M_j)        (on affine triangles)

    Returns ``(K, 6, 6)``. Signs are applied at apply time via
    gather/scatter, *not* baked into the per-element matrix -- same
    convention as :func:`bdm1_element_local_mass`.
    """
    M = _bdm1_physical_gradient_matrices(V)           # (K, 6, 2, 2)
    J = V.mesh.J[0]                                    # (K,)
    # Frobenius inner product (M_i : M_j) per element:
    S_unsigned = jnp.einsum("kiab,kjab->kij", M, M)    # (K, 6, 6)
    return (2.0 * nu / J)[:, None, None] * S_unsigned


def bdm1_viscous_volume_apply(
    V: BDM, u_global: jax.Array, *, nu: float
) -> jax.Array:
    """Apply the volume viscous operator ``A_vol u`` to a global BDM_1
    DOF vector. Returns a global DOF vector.

    No face/jump contributions -- those are added by
    :func:`bdm1_viscous_face_apply`. Separating the two lets us assemble
    and test the volume part in isolation: it must be symmetric SPD on
    its own (the eigenvalues' nullspace is the rigid-translation modes
    of the velocity space).
    """
    S = bdm1_viscous_volume_stiffness(V, nu)           # (K, 6, 6)
    u_local = bdm1_gather(V.topology, u_global)        # (K, 6)
    out_local = jnp.einsum("kij,kj->ki", S, u_local)
    return bdm1_scatter_add(V.topology, out_local)


# ---------------------------------------------------------------------------
# Viscous bilinear form: face terms (SIPG on H(div))
# ---------------------------------------------------------------------------
#
# Per-face SIPG bilinear form contributions:
#
#   a_F(u, v) = -ν ∫_F {{∇u·n}} · [v] dS    (consistency)
#             -ν ∫_F [u] · {{∇v·n}} dS      (symmetry)
#             +ν ∫_F τ [u] · [v] dS         (penalty)
#
# For BDM the inter-element jump [u] is automatically tangential
# (normal trace continuous), so [u]_t = [u].
#
# Per-face quadrature: BDM_1 face DOFs sit at the 2 Gauss-Legendre
# points on each edge. We reuse the same 2-point rule (degree 3 exact)
# for the face integral; one quadrature node per BDM DOF.
#
# Boundary face convention: homogeneous-Dirichlet ghost
#     u_P = -u_M,   ∇u_P · n = ∇u_M · n
# i.e. mirror the value, replicate the gradient. Non-homogeneous u_BC
# contributes only to the RHS (handled by the lift in Phase 5f); from
# the operator's view, the apply uses this homogeneous ghost.


def _bdm1_face_quadrature_data(V: BDM):
    """Precompute per-(element, face) data for the SIPG face apply.

    Returns a dict with keys (each a jax array):

      ``phi_at_face_dofs`` : ``(K, Nfaces, Q=2, 6, 2)``
          Unsigned physical basis at the 2 face GL points (same as the
          BDM DOF coordinates on that face). Used to evaluate the
          element-side trace u_M.
      ``n_face`` : ``(K, Nfaces, 2)``
          Outward unit normal per (element, face), constant for affine.
      ``edge_jac`` : ``(K, Nfaces,)``
          Half the physical edge length -- the Jacobian of the face
          parameterization from reference (length 2) to physical.
      ``K_nbr`` / ``f_nbr`` : ``(K, Nfaces)`` int -- neighbour links.
      ``q_nbr`` : ``(K, Nfaces, Q)`` int -- node-index swap due to
          opposite traversal direction on the shared edge.
      ``is_boundary`` : ``(K, Nfaces)`` bool.
      ``w_q`` : ``(Q,)`` 1D Gauss-Legendre weights on [-1, 1] summing to 2.
    """
    mesh = V.mesh
    Nfaces = 3
    Q = 2
    K = mesh.K

    # Reference DOF points (= face quad points for BDM_1):
    ref_dofs = bdm1_reference_dof_coordinates()      # (6, 2)
    # Unsigned reference basis at those points.
    phi_ref = bdm1_basis_at_points(ref_dofs[:, 0], ref_dofs[:, 1])
    # phi_ref shape: (6 = ref points, 6 = basis, 2 = comp)

    # Physical basis at each ref-DOF point, per element (unsigned Piola).
    DF, _ = _element_jacobian_matrices(mesh)         # (K, 2, 2)
    J = mesh.J[0]                                     # (K,)
    phi_phys = jnp.einsum("kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]
    # shape (K, 6 ref pts, 6 basis, 2 comp). Reshape into face-major form.
    phi_at_face_dofs = phi_phys.reshape(K, Nfaces, Q, 6, 2)

    # Outward normals and edge Jacobians: constant per face on affine.
    Nfp = mesh.Nfp
    nx_face = mesh.nx.reshape((Nfp, Nfaces, K), order="F")[0]      # (Nfaces, K)
    ny_face = mesh.ny.reshape((Nfp, Nfaces, K), order="F")[0]
    sJ_face = mesh.sJ.reshape((Nfp, Nfaces, K), order="F")[0]      # = L_phys / 2

    n_face = jnp.stack([nx_face, ny_face], axis=-1).transpose(1, 0, 2)  # (K, Nfaces, 2)
    # edge_jac is the *parameterised* Jacobian dS_phys / dxi for face
    # quadrature on Gauss-Legendre nodes in xi in [-1, 1]: integral_F f dS =
    # sum_q w_q * edge_jac * f(xi_q). This equals L_phys / 2 for every
    # face regardless of whether it's a leg or the hypotenuse, since the
    # parameterisation interval is [-1, 1] (length 2) in all cases.
    # mesh.sJ stores this exactly.
    edge_jac = sJ_face.T                                                # (K, Nfaces)
    # piola_edge_jac is the *Euclidean* ratio L_phys / L_ref_euclidean,
    # which differs from edge_jac on the hypotenuse face (L_ref_euclidean
    # = 2*sqrt(2) there, not 2). Used by Nanson's formula to convert
    # physical normal traces to BDM coefficients on the unsigned basis.
    piola_edge_jac = edge_jac * _FACE_JAC_RESCALE[None, :]              # (K, Nfaces)

    # Connectivity for u_P lookup.
    EToE = jnp.asarray(mesh.EToE)
    EToF = jnp.asarray(mesh.EToF)
    EToV = jnp.asarray(mesh.EToV)
    K_idx = jnp.arange(K)
    is_boundary = EToE == K_idx[:, None]                # (K, Nfaces)

    # Traversal direction: reversed (typical case) -> q_nbr = 1 - q.
    v_K_end = EToV[K_idx[:, None], (jnp.arange(Nfaces)[None, :] + 1) % 3]
    v_Kp_start = EToV[EToE, EToF]                       # (K, Nfaces)
    reversed_ = (v_Kp_start == v_K_end)                 # (K, Nfaces)

    q_idx = jnp.arange(Q)
    q_nbr = jnp.where(
        reversed_[:, :, None],
        (Q - 1) - q_idx[None, None, :],
        q_idx[None, None, :],
    )                                                   # (K, Nfaces, Q)

    w_q = jnp.array([1.0, 1.0])                         # 2-point GL weights

    return dict(
        phi_at_face_dofs=phi_at_face_dofs,
        n_face=n_face,
        edge_jac=edge_jac,
        piola_edge_jac=piola_edge_jac,
        K_nbr=EToE,
        f_nbr=EToF,
        q_nbr=q_nbr,
        is_boundary=is_boundary,
        w_q=w_q,
    )


def bdm1_viscous_face_apply(
    V: BDM, u_global: jax.Array, *,
    nu: float, tau_scale: float = 4.0,
    u_bc_global: jax.Array | None = None,
) -> jax.Array:
    """Apply the SIPG face viscous operator to a global BDM_1 DOF vector.

    Adds the consistency, symmetry, and tangential-jump penalty
    contributions on every (interior + boundary) face. Boundary face
    ghost:

    * ``u_bc_global is None``: homogeneous Dirichlet ``u_P = -u_M``
      (used by the Stokes pressure-robustness sentinel, where u_BC = 0).
    * ``u_bc_global is not None``: non-homogeneous Dirichlet
      ``u_P = 2 u_BC - u_M`` so that the boundary jump ``[u] = 2 (u_M -
      u_BC)`` vanishes when ``u_M`` matches the prescribed BC. This
      is essential for the SIPG residual to be zero at a BC-respecting
      analytic solution (e.g. the Kovasznay sentinel).

    Penalty parameter (Hesthaven-style): ``tau = tau_scale * (N+1)^2 / h_F``
    where ``h_F`` is the physical edge length.
    """
    data = _bdm1_face_quadrature_data(V)
    phi_face = data["phi_at_face_dofs"]           # (K, Nfaces, Q, 6, 2)
    n_face = data["n_face"]                        # (K, Nfaces, 2)
    edge_jac = data["edge_jac"]                    # (K, Nfaces)
    K_nbr = data["K_nbr"]                          # (K, Nfaces)
    f_nbr = data["f_nbr"]
    q_nbr = data["q_nbr"]                          # (K, Nfaces, Q)
    is_boundary = data["is_boundary"]              # (K, Nfaces)
    w_q = data["w_q"]                              # (Q,)
    h_F = 2.0 * edge_jac                           # physical edge length
    tau = tau_scale * 4.0 / h_F                    # (K, Nfaces), (N+1)^2 = 4

    u_local = bdm1_gather(V.topology, u_global)    # (K, 6)

    # u_M at face quad points: contract u_local with the face-basis trace.
    u_M = jnp.einsum("kl,kfqlc->kfqc", u_local, phi_face)   # (K, Nfaces, Q, 2)

    # u_P: trace from the neighbour at the matching face quad point.
    K_idx = jnp.arange(V.K)
    u_P = u_M[K_nbr[:, :, None], f_nbr[:, :, None], q_nbr]  # (K, Nfaces, Q, 2)
    # Boundary ghost:
    if u_bc_global is not None:
        u_bc_local = bdm1_gather(V.topology, u_bc_global)
        u_bc_M = jnp.einsum("kl,kfqlc->kfqc", u_bc_local, phi_face)
        u_P = jnp.where(is_boundary[:, :, None, None],
                        2.0 * u_bc_M - u_M, u_P)
    else:
        u_P = jnp.where(is_boundary[:, :, None, None], -u_M, u_P)

    # True physical gradient: phi_phys = (sign/J) DF phi_ref, so
    #   grad_phi_phys = (sign/J) DF grad_ref DF_inv = (sign/J) M_grad.
    # The unsigned per-element gradient _bdm1_physical_gradient_matrices
    # returns just ``DF grad_ref DF_inv`` (no 1/J -- the docstring is
    # misleading on this point), so we divide explicitly. Both grad_u
    # and grad_test_n use the true physical gradient here; the volume
    # stiffness assembled in bdm1_viscous_volume_stiffness applies its
    # own (1/J) scaling, and these two pieces must use consistent
    # bookkeeping or the SIPG bilinear form is INCONSISTENT (the
    # volume-vs-face cancellation in integration by parts fails by a
    # factor of J, killing the convergence rate).
    M_grad = _bdm1_physical_gradient_matrices(V)        # (K, 6, 2, 2), J-scaled
    inv_J = 1.0 / V.mesh.J[0]                            # (K,)
    grad_u_M = jnp.einsum(
        "kl,klij,k->kij", u_local, M_grad, inv_J,
    )                                                    # (K, 2, 2)  -- true grad u_M

    # ∇u_P (neighbour's TRUE gradient); index by K_nbr to pick up
    # neighbour's u_local @ M_grad / J_nbr correctly.
    grad_u_P = grad_u_M[K_nbr]                           # (K, Nfaces, 2, 2)
    grad_u_P = jnp.where(
        is_boundary[:, :, None, None], grad_u_M[:, None, :, :], grad_u_P,
    )

    grad_u_n_M = jnp.einsum("kij,kfj->kfi", grad_u_M, n_face)
    grad_u_n_P = jnp.einsum("kfij,kfj->kfi", grad_u_P, n_face)
    avg_grad_un = 0.5 * (grad_u_n_M + grad_u_n_P)
    avg_grad_un_q = jnp.broadcast_to(avg_grad_un[:, :, None, :],
                                      (V.K, 3, 2, 2))

    jump_u = u_M - u_P
    test_at_face_q = phi_face

    # ∇φ_test^phys · n at face quad points, true physical gradient.
    grad_test_n = jnp.einsum(
        "klij,kfj,k->kfli", M_grad, n_face, inv_J,
    )                                                    # (K, Nfaces, 6, 2)

    # K's local face-DOF contribution: contract over (f, q, c).
    # term 1: -ν * avg_grad_un · test
    consistency = -nu * jnp.einsum(
        "q,kf,kfqc,kfqlc->kl",
        w_q, edge_jac, avg_grad_un_q, test_at_face_q,
    )
    # term 2: -ν * jump · 0.5 * ∇test·n
    symmetry = -0.5 * nu * jnp.einsum(
        "q,kf,kfqc,kflc->kl",
        w_q, edge_jac, jump_u, grad_test_n,
    )
    # term 3: +ν * τ * jump · test
    penalty = nu * jnp.einsum(
        "q,kf,kf,kfqc,kfqlc->kl",
        w_q, edge_jac, tau, jump_u, test_at_face_q,
    )
    out_local = consistency + symmetry + penalty                    # (K, 6)
    return bdm1_scatter_add(V.topology, out_local)


def bdm1_viscous_apply(
    V: BDM, u_global: jax.Array, *,
    nu: float, tau_scale: float = 4.0,
    u_bc_global: jax.Array | None = None,
) -> jax.Array:
    """Full SIPG viscous apply on BDM_1 = volume + face contributions.

    Convenience wrapper combining :func:`bdm1_viscous_volume_apply` and
    :func:`bdm1_viscous_face_apply`. Pass ``u_bc_global`` for the non-
    homogeneous Dirichlet ghost (essential when the linear operator is
    applied to a u that already carries non-zero BC values, e.g. the
    Kovasznay Newton iterate).
    """
    return (
        bdm1_viscous_volume_apply(V, u_global, nu=nu)
        + bdm1_viscous_face_apply(
            V, u_global,
            nu=nu, tau_scale=tau_scale,
            u_bc_global=u_bc_global,
        )
    )


def bdm1_evaluate_boundary_velocity(V: BDM, u_bc_fn) -> jax.Array:
    """Sample the user-supplied vector velocity BC ``u_bc_fn(x, y) -> (u_x, u_y)``
    at every boundary face GL point.

    Returns ``(K, Nfaces, Q=2, 2)`` -- the full *vector* u_BC. Zero on
    interior faces. Used by :func:`bdm1_viscous_dirichlet_lift`, which
    needs the full vector (including the tangential component); BDM DOFs
    by themselves only carry the normal trace ``u · n`` and lose the
    tangential information needed for the SIPG penalty / symmetry lift.
    """
    fdata = _bdm1_face_quadrature_data(V)
    is_boundary = fdata["is_boundary"]
    Nfaces = 3

    # Physical positions of face GL points (= BDM_1 DOF reference coords
    # mapped through the affine F_K).
    mesh = V.mesh
    # Keep arrays in JAX so AD flows back to VX, VY through this BC
    # sampling step (needed for differentiable-mesh applications).
    EToV = jnp.asarray(mesh.EToV)
    VX = jnp.asarray(mesh.VX)
    VY = jnp.asarray(mesh.VY)
    K = mesh.K
    ref_dofs = bdm1_reference_dof_coordinates()
    r = ref_dofs[:, 0]
    s = ref_dofs[:, 1]
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    x_at = (
        Vx0[:, None]
        + 0.5 * (r[None, :] + 1.0) * (Vx1 - Vx0)[:, None]
        + 0.5 * (s[None, :] + 1.0) * (Vx2 - Vx0)[:, None]
    )
    y_at = (
        Vy0[:, None]
        + 0.5 * (r[None, :] + 1.0) * (Vy1 - Vy0)[:, None]
        + 0.5 * (s[None, :] + 1.0) * (Vy2 - Vy0)[:, None]
    )
    x_face = x_at.reshape(K, Nfaces, 2)
    y_face = y_at.reshape(K, Nfaces, 2)
    u_bc_x, u_bc_y = u_bc_fn(x_face, y_face)
    u_bc_vec = jnp.stack(
        [jnp.asarray(u_bc_x), jnp.asarray(u_bc_y)], axis=-1
    )  # (K, Nfaces, Q, 2)
    return jnp.where(is_boundary[:, :, None, None], u_bc_vec, 0.0)


def bdm1_viscous_dirichlet_lift(
    V: BDM, u_bc_vec: jax.Array, *,
    nu: float, tau_scale: float = 4.0,
) -> jax.Array:
    """Boundary-only SIPG lift from a non-homogeneous Dirichlet velocity BC.

    The SIPG viscous bilinear form with homogeneous Dirichlet ghost
    ``u_P = -u_M`` implicitly assumes ``u_BC = 0``. For non-zero
    ``u_BC``, the correct (non-homogeneous-ghost) bilinear form differs
    from the homogeneous one by a boundary integral that is entirely
    independent of ``u``:

        a_nonhom(u, v; u_BC) - a_hom(u, v) =
              + 2 nu integral_F u_BC . (grad v_M . n) dS    (symmetry)
              - 4 nu tau integral_F u_BC . v_M dS           (penalty)

    To use the homog-ghost apply with a non-homog Dirichlet problem we
    SUBTRACT this lift from the residual: ``F = a_hom(u, v) - rhs - lift``.

    ``u_bc_vec`` is the *full vector* u_BC sampled at face GL points,
    shape ``(K, Nfaces, Q, 2)``. Use
    :func:`bdm1_evaluate_boundary_velocity` to build it from a callable.
    The tangential component matters: a lid-driven cavity has u_BC
    tangential to the lid (so ``u_BC . n_lid = 0``), but the penalty
    term still drives the tangential trace of u toward u_BC.
    """
    fdata = _bdm1_face_quadrature_data(V)
    phi_face = fdata["phi_at_face_dofs"]              # (K, Nfaces, Q, 6, 2)
    n_face = fdata["n_face"]                           # (K, Nfaces, 2)
    edge_jac = fdata["edge_jac"]                       # (K, Nfaces)
    is_boundary = fdata["is_boundary"]                 # (K, Nfaces)
    w_q = fdata["w_q"]                                 # (Q,)
    h_F = 2.0 * edge_jac
    tau = tau_scale * 4.0 / h_F                        # (K, Nfaces)

    # u_BC vector at boundary GL points (caller already masked interior to 0
    # via bdm1_evaluate_boundary_velocity; doing it again is idempotent).
    u_bc_M = jnp.where(is_boundary[:, :, None, None], u_bc_vec, 0.0)

    # Test gradient on the M-side, dot with face normal. The 1/J factor
    # makes this the TRUE physical gradient (consistent with the face
    # apply -- the original code stored M_grad = DF grad_ref DF_inv
    # without the 1/J, which is the inconsistency that caused the
    # slope-1 convergence bug).
    M_grad = _bdm1_physical_gradient_matrices(V)      # (K, 6, 2, 2)
    inv_J = 1.0 / V.mesh.J[0]
    grad_test_n = jnp.einsum(
        "klij,kfj,k->kfli", M_grad, n_face, inv_J,
    )                                                  # (K, Nfaces, 6, 2)

    # Matching factors to the actual SIPG apply (which uses the
    # asymmetric-ghost convention: u_P = 2 u_BC - u_M for the trial
    # field, v_P = 0 for the test). Working through:
    #   a_face_bdy(u, v; u_BC) - a_face_bdy(u, v; 0) =
    #         + nu integral u_BC . (grad_v . n) - 2 nu tau integral u_BC . v_M
    # Caller subtracts this lift, so:
    #   lift = -nu integral u_BC . (grad_v . n) + 2 nu tau integral u_BC . v_M
    symmetry = -nu * jnp.einsum(
        "q,kf,kfqc,kflc->kl",
        w_q, edge_jac, u_bc_M, grad_test_n,
    )
    penalty = 2.0 * nu * jnp.einsum(
        "q,kf,kf,kfqc,kfqlc->kl",
        w_q, edge_jac, tau, u_bc_M, phi_face,
    )
    out_local = symmetry + penalty
    return bdm1_scatter_add(V.topology, out_local)


# ---------------------------------------------------------------------------
# BDM_k order-parametric reference basis
# ---------------------------------------------------------------------------
#
# The block below is the foundation for higher-order BDM. The DOFs of
# BDM_k = P_k(K)^2 on a triangle:
#
#   * Face DOFs ((k+1) per face x 3 faces = 3(k+1) total): values of
#     the normal trace ``u . n`` at the k+1 Gauss-Legendre points on
#     each edge. These guarantee H(div)-conformity across element
#     interfaces.
#
#   * Interior DOFs ((k+1)(k-1) total, zero for k=1): moments of u
#     against a basis of an "interior test space" -- by Brezzi-Marini
#     this is grad(P_{k-1} \ R) extended with curl-bubble moments to
#     make up the full dimension. These DOFs annihilate the edge DOFs
#     and pick up purely-internal modes.
#
# Construction (Vandermonde / moment-matrix inversion):
#   1. Pick a primal monomial basis of P_k(K)^2 of dimension (k+1)(k+2).
#   2. Build the DOF matrix ``V[i, j] = DOF_i(monomial_j)``.
#   3. Invert: the BDM basis coefficients are the columns of V^{-1}.
#
# At k=1 there are no interior DOFs, and the constructions below
# reproduce the hard-coded ``_BDM1_COEFFS`` / ``bdm1_basis_at_points``
# results to machine precision (regression-verified in
# ``test_bdm_k_reference``). For k>=2 the interior moments are still
# TODO; the entry points raise ``NotImplementedError`` until those
# land.


def _gauss_legendre_nodes(n: int) -> jnp.ndarray:
    """``n`` Gauss-Legendre nodes on [-1, 1].

    Roots of the Legendre polynomial P_n. Computed once at module
    import time via numpy's tabulated polynomial routines (Golub-Welsch
    under the hood) -- not a hot path.
    """
    nodes, _ = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(nodes)


def bdm_k_face_dof_coordinates(k: int) -> jnp.ndarray:
    """Reference-triangle (r, s) coords of the 3(k+1) face DOFs of BDM_k.

    On each edge, the k+1 Gauss-Legendre points are taken in the
    parameterisation that maps xi=-1 to the starting vertex of the
    edge (Hesthaven convention: edge 0 = v0->v1, edge 1 = v1->v2,
    edge 2 = v2->v0). Returns an ``(3*(k+1), 2)`` array; DOF index
    ``(k+1)*f + j`` is the j-th GL node on face f.
    """
    xi = _gauss_legendre_nodes(k + 1)            # (k+1,)
    ones = jnp.ones_like(xi)
    # Edge 0: r = xi, s = -1.
    pts0 = jnp.stack([xi, -ones], axis=1)
    # Edge 1: r = -xi, s = xi (xi=-1 -> v1, xi=1 -> v2).
    pts1 = jnp.stack([-xi, xi], axis=1)
    # Edge 2: r = -1, s = -xi (xi=-1 -> v2, xi=1 -> v0).
    pts2 = jnp.stack([-ones, -xi], axis=1)
    return jnp.concatenate([pts0, pts1, pts2], axis=0)


def _p_k_monomial_exponents(k: int):
    """Exponent pairs ``(i, j)`` with ``i + j <= k`` listing the scalar
    monomial basis of ``P_k(K)``. Order: lexicographic in (j, i) so the
    first ``dim P_{k-1}`` entries form P_{k-1} (useful for hierarchical
    constructions).
    """
    out = []
    for total in range(k + 1):
        for j in range(total + 1):
            i = total - j
            out.append((i, j))
    return out


def _bdm_k_primal_basis_at_points(
    k: int, r: jnp.ndarray, s: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the vector monomial primal basis of ``P_k(K)^2`` at
    points ``(r, s)``. Dimension is ``(k+1)(k+2)``: each scalar
    monomial in P_k(K) is placed in the x-component (with zero y) and
    again in the y-component (with zero x).

    Returns shape ``(npts, dim, 2)`` where ``dim = (k+1)(k+2)``. The
    ordering pairs each scalar monomial first as an x-only monomial
    then as a y-only monomial: ``[(m_0, 0), (0, m_0), (m_1, 0), (0,
    m_1), ...]``. This ordering makes the divergence computation
    cleaner downstream.
    """
    r = jnp.atleast_1d(r); s = jnp.atleast_1d(s)
    exponents = _p_k_monomial_exponents(k)
    npts = r.shape[0]
    cols = []
    for (i, j) in exponents:
        m = (r ** i) * (s ** j)                   # (npts,)
        # Pair as x-component then y-component.
        cols.append(jnp.stack([m, jnp.zeros_like(m)], axis=1))
        cols.append(jnp.stack([jnp.zeros_like(m), m], axis=1))
    # Stack along the basis dimension.
    return jnp.stack(cols, axis=1)                # (npts, dim, 2)


def _bdm_k_face_dof_matrix(k: int) -> jnp.ndarray:
    """DOF matrix block for the 3(k+1) face DOFs of BDM_k.

    Row i corresponds to a face DOF (a normal-trace point evaluator);
    column j corresponds to a primal-basis monomial. Entry is
    ``(monomial_j . n_face) (evaluated at DOF coord i)``. Used as the
    upper block of the full DOF matrix; for k=1 it IS the full DOF
    matrix.
    """
    dof_pts = bdm_k_face_dof_coordinates(k)       # (3(k+1), 2)
    face_idx = jnp.repeat(jnp.arange(3), k + 1)    # [0,0,...,1,1,...,2,2,...]
    normals = _REF_FACE_NORMALS[face_idx]          # (3(k+1), 2)
    # Evaluate every primal-basis monomial at every face DOF point,
    # then dot with the corresponding face normal.
    phi_at = _bdm_k_primal_basis_at_points(
        k, dof_pts[:, 0], dof_pts[:, 1],
    )                                              # (3(k+1), dim, 2)
    # D[i, j] = phi_at[i, j, :] . normals[i, :]
    return jnp.einsum("idc,ic->id", phi_at, normals)


def _bubble_b3_and_derivs(r: jnp.ndarray, s: jnp.ndarray):
    """The reference-triangle cubic bubble
    ``b_3(r, s) = -(r+1)(s+1)(r+s)``, which vanishes on every edge of
    the reference triangle, together with its first partial
    derivatives. ``curl(b_3 q) = (∂_s(b_3 q), -∂_r(b_3 q))`` has zero
    normal trace on every edge -- the key property that makes
    curl(b_3 q) moments "purely interior".
    """
    b = -(r + 1.0) * (s + 1.0) * (r + s)
    # ∂_r b_3 = -(s+1)(r+s) - (r+1)(s+1) = -(s+1)(2r + s + 1)
    db_dr = -(s + 1.0) * (2.0 * r + s + 1.0)
    # ∂_s b_3 = -(r+1)(r+s) - (r+1)(s+1) = -(r+1)(r + 2s + 1)
    db_ds = -(r + 1.0) * (r + 2.0 * s + 1.0)
    return b, db_dr, db_ds


def _bdm_k_interior_test_functions(k: int):
    """Interior moment test functions for BDM_k using the
    Brezzi-Marini convention: ``∇p`` for ``p`` in a basis of
    ``P_{k-1}(K) \\ R`` plus ``curl(b_3 q)`` for ``q`` in a basis of
    ``P_{k-2}(K)``.

    Returns a list of *evaluators* ``w(r, s) -> (w_x, w_y)``; each
    returned tuple is the value of the vector test function at the
    given point.

    Dimensions match the required interior DOF count ``(k+1)(k-1)``:
    ``dim P_{k-1} - 1 = k(k+1)/2 - 1``, ``dim P_{k-2} = (k-1)k/2``,
    sum ``= k^2 - 1 = (k+1)(k-1)``. The ``∇p`` block captures the
    divergence content of u (since ``∫ u . ∇p = -∫ (∇.u) p`` for
    bubble u); the ``curl(b_3 q)`` block captures the curl/rotational
    content (gradient and curl give an L^2-orthogonal decomposition of
    bubble vector fields).
    """
    if k == 1:
        return []
    tests = []

    # ∇p moments: for each scalar monomial m = r^i s^j with 1 <= i+j <= k-1.
    # (The constant monomial m=1 has ∇m = 0 so we skip i+j == 0.)
    for (i, j) in _p_k_monomial_exponents(k - 1):
        if i + j == 0:
            continue
        # ∇(r^i s^j) = (i*r^(i-1)*s^j, j*r^i*s^(j-1))
        def make_grad(ii, jj):
            def w(r, s):
                wx = (ii * r ** (ii - 1) * s ** jj) if ii > 0 else jnp.zeros_like(r)
                wy = (jj * r ** ii * s ** (jj - 1)) if jj > 0 else jnp.zeros_like(r)
                return wx, wy
            return w
        tests.append(make_grad(i, j))

    # curl(b_3 q) moments: for each scalar monomial q in P_{k-2}.
    # curl(b_3 q) = (∂_s(b_3 q), -∂_r(b_3 q))
    #            = (b_3 ∂_s q + q ∂_s b_3, -(b_3 ∂_r q + q ∂_r b_3)).
    for (i, j) in _p_k_monomial_exponents(k - 2):
        def make_curl(ii, jj):
            def w(r, s):
                b, db_dr, db_ds = _bubble_b3_and_derivs(r, s)
                q = (r ** ii) * (s ** jj)
                dq_dr = (ii * r ** (ii - 1) * s ** jj) if ii > 0 else jnp.zeros_like(r)
                dq_ds = (jj * r ** ii * s ** (jj - 1)) if jj > 0 else jnp.zeros_like(r)
                wx = b * dq_ds + q * db_ds
                wy = -(b * dq_dr + q * db_dr)
                return wx, wy
            return w
        tests.append(make_curl(i, j))

    n_expected = (k + 1) * (k - 1)
    if len(tests) != n_expected:
        raise AssertionError(
            f"BDM_{k} interior test count: got {len(tests)}, expected {n_expected}"
        )
    return tests


def _eval_interior_test(w_callable, r: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Evaluate an interior test function from
    :func:`_bdm_k_interior_test_functions` at points ``(r, s)``.
    Returns shape ``(npts, 2)``.
    """
    wx, wy = w_callable(r, s)
    # Broadcast scalars (zero) to the shape of r.
    wx = wx if jnp.shape(wx) else jnp.full_like(r, wx)
    wy = wy if jnp.shape(wy) else jnp.full_like(r, wy)
    return jnp.stack([jnp.broadcast_to(wx, r.shape),
                       jnp.broadcast_to(wy, r.shape)], axis=1)


def _bdm_k_interior_dof_matrix(k: int) -> jnp.ndarray:
    """DOF-matrix block for the ``(k+1)(k-1)`` interior moments.

    Row i is the i-th interior test function; column j is the j-th
    primal-basis vector monomial. Entry is
    ``∫_K (primal_j . test_i) dr ds`` over the reference triangle.

    Cubature: Stroud rule exact for total degree ``2k`` (so primal
    times test, both at most degree k-1 in each component, integrates
    exactly).
    """
    from ddg.dg2d.cubature import triangle_cubature

    tests = _bdm_k_interior_test_functions(k)
    if not tests:
        # Empty block (k=1).
        dim = (k + 1) * (k + 2)
        return jnp.zeros((0, dim))

    # Cubature degree >= 2k is sufficient: primal is P_k, test is at
    # most P_{k-1}, product P_{2k-1}.
    r_q, s_q, w_q = triangle_cubature(2 * k)        # (Q,), (Q,), (Q,)

    primal_at = _bdm_k_primal_basis_at_points(k, r_q, s_q)  # (Q, dim, 2)
    # Evaluate every test at the cubature points.
    test_vals = jnp.stack(
        [_eval_interior_test(t, r_q, s_q) for t in tests],
        axis=0,
    )                                                # (n_interior, Q, 2)
    # Block[i, j] = sum_q w_q * primal_at[q, j, c] * test_vals[i, q, c]
    return jnp.einsum("q,qjc,iqc->ij", w_q, primal_at, test_vals)


def _bdm_k_dof_matrix(k: int) -> jnp.ndarray:
    """Full DOF matrix for BDM_k = face block stacked on interior
    block. Shape ``((k+1)(k+2), (k+1)(k+2))``.

    At k=1 the interior block is empty (no interior DOFs) and the
    face block is the full matrix -- agreeing with the original
    hard-coded ``_bdm1_dof_matrix``.
    """
    face_block = _bdm_k_face_dof_matrix(k)         # (3(k+1), dim)
    interior_block = _bdm_k_interior_dof_matrix(k) # ((k+1)(k-1), dim)
    return jnp.concatenate([face_block, interior_block], axis=0)


def _bdm_k_basis_coefficients(k: int) -> jnp.ndarray:
    """``(dim, dim)`` matrix whose column ``ell`` holds the
    primal-monomial coefficients of BDM_k basis function ``ell``.
    Computed as ``inv(_bdm_k_dof_matrix(k))``.
    """
    return jnp.linalg.inv(_bdm_k_dof_matrix(k))


def bdm_k_basis_at_points(
    k: int, r: jnp.ndarray, s: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the BDM_k basis at points ``(r, s)``.

    Returns shape ``(npts, dim, 2)`` where ``dim = (k+1)(k+2)``. At
    k=1 this is the order-parametric route to the same values produced
    by :func:`bdm1_basis_at_points`; regression test in
    ``test_bdm_k_reference`` enforces machine-precision agreement.
    """
    coeffs = _bdm_k_basis_coefficients(k)          # (dim, dim)
    primal = _bdm_k_primal_basis_at_points(k, r, s)  # (npts, dim, 2)
    # Linear combination over the primal dim. Per point, per
    # component: psi_ell(r) = sum_j coeffs[j, ell] * primal[r, j].
    return jnp.einsum("nja,jl->nla", primal, coeffs)


def _bdm_k_primal_divergence_at_points(
    k: int, r: jnp.ndarray, s: jnp.ndarray,
) -> jnp.ndarray:
    """Divergence of each primal vector monomial at points ``(r, s)``.

    Primal basis (from :func:`_bdm_k_primal_basis_at_points`) pairs
    each scalar monomial ``m = r**i s**j`` first as ``(m, 0)`` then as
    ``(0, m)``. So the divergences are ``∂_r m`` and ``∂_s m``
    respectively.

    Returns shape ``(npts, dim)``.
    """
    r = jnp.atleast_1d(r); s = jnp.atleast_1d(s)
    exponents = _p_k_monomial_exponents(k)
    cols = []
    for (i, j) in exponents:
        # (m, 0): div = ∂_r m = i * r^(i-1) * s^j
        div_x = (i * r ** (i - 1) * s ** j) if i > 0 else jnp.zeros_like(r)
        # (0, m): div = ∂_s m = j * r^i * s^(j-1)
        div_y = (j * r ** i * s ** (j - 1)) if j > 0 else jnp.zeros_like(r)
        cols.append(div_x)
        cols.append(div_y)
    return jnp.stack(cols, axis=1)                  # (npts, dim)


def _bdm_k_primal_grad_at_points(
    k: int, r: jnp.ndarray, s: jnp.ndarray,
) -> jnp.ndarray:
    """Reference gradient of each primal vector monomial.

    For primal vector ``(m, 0)``, ``grad = [[∂_r m, ∂_s m], [0, 0]]``.
    For ``(0, m)``, ``grad = [[0, 0], [∂_r m, ∂_s m]]``.

    Returns shape ``(npts, dim, 2, 2)`` where the last two indices are
    (component-of-vector, gradient-direction): entry
    ``[q, ell, c, d] = ∂_{(r,s)_d} (primal_ell)_c``.
    """
    r = jnp.atleast_1d(r); s = jnp.atleast_1d(s)
    exponents = _p_k_monomial_exponents(k)
    grads = []
    for (i, j) in exponents:
        dm_dr = (i * r ** (i - 1) * s ** j) if i > 0 else jnp.zeros_like(r)
        dm_ds = (j * r ** i * s ** (j - 1)) if j > 0 else jnp.zeros_like(r)
        # (m, 0): grad_x_r = dm_dr, grad_x_s = dm_ds, grad_y = 0.
        g_x = jnp.stack([
            jnp.stack([dm_dr, dm_ds], axis=1),
            jnp.zeros((r.shape[0], 2)),
        ], axis=1)                                  # (npts, 2, 2): [(comp), (dir)]
        # (0, m): grad_y_r = dm_dr, grad_y_s = dm_ds.
        g_y = jnp.stack([
            jnp.zeros((r.shape[0], 2)),
            jnp.stack([dm_dr, dm_ds], axis=1),
        ], axis=1)
        grads.append(g_x); grads.append(g_y)
    return jnp.stack(grads, axis=1)                 # (npts, dim, 2, 2)


def bdm_k_basis_grad_at_points(
    k: int, r: jnp.ndarray, s: jnp.ndarray,
) -> jnp.ndarray:
    """Reference gradient of each BDM_k basis function at points ``(r, s)``.

    Returns shape ``(npts, dim, 2, 2)``: ``[q, ell, c, d] =
    ∂_{(r,s)_d} (basis_ell)_c``. Same component / direction convention
    as :func:`_bdm_k_primal_grad_at_points`.
    """
    coeffs = _bdm_k_basis_coefficients(k)           # (dim, dim)
    primal_grad = _bdm_k_primal_grad_at_points(k, r, s)  # (npts, dim, 2, 2)
    # Linear combination over the primal dim:
    # basis_grad[q, ell, c, d] = sum_j coeffs[j, ell] * primal_grad[q, j, c, d]
    return jnp.einsum("qjcd,jl->qlcd", primal_grad, coeffs)


def bdm_k_divergence_at_points(
    k: int, r: jnp.ndarray, s: jnp.ndarray,
) -> jnp.ndarray:
    """Divergence of each BDM_k basis function evaluated at points
    ``(r, s)``. Returns shape ``(npts, dim)``.

    For k=1 the divergences are constant; we return them broadcast to
    the requested points to keep the shape uniform across orders.
    """
    coeffs = _bdm_k_basis_coefficients(k)           # (dim, dim)
    primal_div = _bdm_k_primal_divergence_at_points(k, r, s)  # (npts, dim)
    # Divergence is linear, so div(psi_ell) = sum_j coeffs[j, ell] * div(primal_j).
    return primal_div @ coeffs                       # (npts, dim)


def bdm_k_divergence_basis(k: int) -> jnp.ndarray:
    """Constant divergence values for BDM_k basis (k=1 case only).

    At k=1 every BDM basis function has constant divergence; this is
    the helper kept for backward compatibility with the k=1 codepath
    and the existing :func:`bdm1_divergence_basis`. For k>=2 the
    divergence is not constant -- use
    :func:`bdm_k_divergence_at_points` instead.
    """
    if k != 1:
        raise ValueError(
            f"bdm_k_divergence_basis: only k=1 has constant divergences; "
            f"for k>=2 use bdm_k_divergence_at_points(k, r, s)."
        )
    coeffs = _bdm_k_basis_coefficients(k)          # (6, 6)
    return coeffs[2] + coeffs[5]


# ---------------------------------------------------------------------------
# BDM_k order-parametric operators
# ---------------------------------------------------------------------------
#
# These build the per-element matrices via cubature integration in the
# reference triangle. The Piola map ``phi_phys = sign/J DF phi_ref`` is
# the same at every order; only the basis itself and the cubature
# degree change. Sign conventions stay the same as the k=1 path:
# per-element matrices are computed on the *unsigned* Piola-mapped
# basis; topology signs are applied via gather/scatter at apply time.


def bdm_k_element_local_mass(V: BDM) -> jax.Array:
    """Per-element mass matrix on the unsigned Piola-mapped BDM_k
    basis. Returns ``(K, Np, Np)`` with ``Np = (k+1)(k+2)``.

    On an affine triangle, ``DF_K``, ``J_K`` are constant; the
    metric-weighted reference inner product simplifies to one
    cubature pass. Cubature degree chosen as ``2k`` (Stroud conical
    rule integrates degree-``2k`` polynomials exactly).
    """
    from ddg.dg2d.cubature import triangle_cubature
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k)        # (Q,), (Q,), (Q,)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)    # (Q, Np, 2)
    _, G = _element_jacobian_matrices(V.mesh)        # (K, 2, 2)
    J = V.mesh.J[0]                                   # (K,)
    # M_K[i, j] = (1/J_K) sum_q w_q sum_{a, b} G_K[a, b] phi_ref[q, i, a] phi_ref[q, j, b]
    inv_J = 1.0 / J
    return jnp.einsum(
        "q,qia,kab,qjb,k->kij", w_q, phi_ref, G, phi_ref, inv_J,
    )


def bdm_k_physical_grad_at_cubature(
    V: BDM, r_q: jax.Array, s_q: jax.Array,
) -> jax.Array:
    """``J``-scaled physical gradient of each BDM_k basis at cubature
    points, per element. Matches the k=1 convention used by
    :func:`_bdm1_physical_gradient_matrices`: this returns ``DF *
    grad_ref * DF_inv`` *without* the ``1/J`` factor that the true
    physical gradient would carry.

    Returns shape ``(K, Q, Np, 2, 2)``. Callers multiply by ``1/J``
    when integrating against the volume element ``dx = J dr ds``, so
    the net effect is correct. This bookkeeping convention is kept to
    stay bit-compatible with the BDM_1 code path and the validated
    Stokes / Kovasznay / cavity sentinels.
    """
    k = V.order
    grad_ref = bdm_k_basis_grad_at_points(k, r_q, s_q)  # (Q, Np, 2, 2)
    DF, _ = _element_jacobian_matrices(V.mesh)           # (K, 2, 2)
    DF_inv = jnp.linalg.inv(DF)                          # (K, 2, 2)
    # M_grad[k, q, ell, c, d] = DF[k, c, a] * grad_ref[q, ell, a, e] * DF_inv[k, e, d]
    # Same as old _bdm1_physical_gradient_matrices but at every q.
    return jnp.einsum(
        "kca,qlae,ked->kqlcd", DF, grad_ref, DF_inv,
    )


def bdm_k_viscous_volume_stiffness(V: BDM, nu: float) -> jax.Array:
    """Per-element viscous volume stiffness on the unsigned Piola
    basis. Returns ``(K, Np, Np)``:

        S_K[i, j] = nu * integral_K (nabla phi^phys_i : nabla phi^phys_j) dx.

    At k=1 the physical gradient is constant per element and this
    reduces to ``(2 nu / J_K) * (M_i : M_j)`` as in
    :func:`bdm1_viscous_volume_stiffness`. At higher k the gradient
    varies in (r, s) and the integration is via cubature.

    Using the J-scaled convention from
    :func:`bdm_k_physical_grad_at_cubature`: ``M_grad = DF grad_ref DF_inv =
    J * true_grad``. The true integral is

        S_K = nu * integral_K (true_grad : true_grad) dx
            = nu * integral_T (1/J^2)(M_grad : M_grad) * J dr ds
            = (nu / J_K) * sum_q w_q (M_grad_q : M_grad_q).
    """
    from ddg.dg2d.cubature import triangle_cubature
    k = V.order
    # M_grad is degree k-1; product is degree 2(k-1).
    deg = max(2 * (k - 1), 1)
    r_q, s_q, w_q = triangle_cubature(deg)
    M_grad = bdm_k_physical_grad_at_cubature(V, r_q, s_q)  # (K, Q, Np, 2, 2) -- J-scaled
    J = V.mesh.J[0]                                           # (K,)
    inv_J = 1.0 / J
    return nu * jnp.einsum(
        "q,k,kqicd,kqjcd->kij", w_q, inv_J, M_grad, M_grad,
    )


def bdm_k_viscous_volume_apply(
    V: BDM, u_global: jax.Array, *, nu: float,
) -> jax.Array:
    """Apply the volume viscous operator. No face contributions."""
    S = bdm_k_viscous_volume_stiffness(V, nu)
    u_local = bdm1_gather(V.topology, u_global)
    out_local = jnp.einsum("kij,kj->ki", S, u_local)
    return bdm1_scatter_add(V.topology, out_local)


def bdm_k_mass_apply(V: BDM, u_global: jax.Array) -> jax.Array:
    """Apply the BDM_k velocity mass matrix to a global DOF vector.
    Order-parametric replacement for :func:`bdm1_mass_apply`."""
    M_K = bdm_k_element_local_mass(V)
    u_local = bdm1_gather(V.topology, u_global)
    out_local = jnp.einsum("kij,kj->ki", M_K, u_local)
    return bdm1_scatter_add(V.topology, out_local)


# ---------------------------------------------------------------------------
# P_{k-1} pressure space + cross-space divergence / gradient
# ---------------------------------------------------------------------------
#
# Pairing: BDM_k = P_k(K)^2 velocity / P_{k-1}(K) discontinuous
# pressure. At k=1 the pressure space is P_0 (one DOF per element);
# at k=2 it is P_1 (3 DOFs per element); at k=3 it is P_2 (6 DOFs per
# element). DOFs are coefficients against the monomial basis of
# P_{k-1} in *reference* coords -- the same _p_k_monomial_exponents
# ordering used elsewhere.
#
# Reference-coord pressure has a key advantage with the Piola
# contravariant map: the per-element divergence "matrix" D_local is
# element-independent. The J factor in the volume element and the
# 1/J in the Piola divergence cancel exactly, so D_local is just one
# (dim_p x Np) matrix computed once per order.


def bdm_k_pkm1_pressure_dim(k: int) -> int:
    """DOFs per element for the P_{k-1}(K) discontinuous pressure
    space paired with BDM_k. Equals ``k(k+1)/2``."""
    return k * (k + 1) // 2


def bdm_k_pkm1_basis_at_points(
    k: int, r: jnp.ndarray, s: jnp.ndarray,
) -> jnp.ndarray:
    """Monomial basis of P_{k-1}(K) at reference points ``(r, s)``.

    Returns shape ``(npts, dim_p)`` with ``dim_p = k(k+1)/2``. The
    ordering matches ``_p_k_monomial_exponents(k-1)``.
    """
    r = jnp.atleast_1d(r); s = jnp.atleast_1d(s)
    exponents = _p_k_monomial_exponents(k - 1)
    cols = [(r ** i) * (s ** j) for (i, j) in exponents]
    return jnp.stack(cols, axis=1)


def bdm_k_pkm1_basis_grad_at_points(
    k: int, r: jnp.ndarray, s: jnp.ndarray,
) -> jnp.ndarray:
    """Reference-element gradient of the P_{k-1}(K) monomial basis at
    points ``(r, s)``. Returns shape ``(npts, dim_p, 2)`` where the
    trailing axis is ``(d/dr, d/ds)``.
    """
    r = jnp.atleast_1d(r); s = jnp.atleast_1d(s)
    exponents = _p_k_monomial_exponents(k - 1)
    cols_r, cols_s = [], []
    for (i, j) in exponents:
        cols_r.append((i * r ** (i - 1) if i > 0 else jnp.zeros_like(r))
                      * (s ** j))
        cols_s.append((r ** i)
                      * (j * s ** (j - 1) if j > 0 else jnp.zeros_like(s)))
    return jnp.stack(
        [jnp.stack(cols_r, axis=1), jnp.stack(cols_s, axis=1)], axis=2,
    )


def _bdm_k_reference_divergence_matrix(k: int) -> jnp.ndarray:
    """``D_ref[m, ell] = integral_T (div phi^ref_ell)(r, s) * q_m(r, s) dr ds``
    where ``q_m`` is the m-th monomial of P_{k-1}(K) and ``phi^ref_ell``
    is the ell-th BDM_k reference basis. Same matrix for every element
    (J cancels under the Piola map).

    Returns ``(dim_p, Np)``.
    """
    from ddg.dg2d.cubature import triangle_cubature
    # div_ref has degree k-1; q has degree k-1; product 2(k-1).
    deg = max(2 * (k - 1), 1)
    r_q, s_q, w_q = triangle_cubature(deg)
    div_ref = bdm_k_divergence_at_points(k, r_q, s_q)   # (Q, Np)
    q_at = bdm_k_pkm1_basis_at_points(k, r_q, s_q)      # (Q, dim_p)
    return jnp.einsum("q,ql,qm->ml", w_q, div_ref, q_at)


def bdm_k_divergence_apply(V_u: BDM, u_global: jax.Array) -> jnp.ndarray:
    """Apply ``D : BDM_k -> P_{k-1}`` to a global velocity DOF vector.

    Returns the per-element pressure-DOF array of shape ``(K, dim_p)``
    where each row holds the test-function moments ``integral_K
    (div u) q_m dx`` for ``m = 0 ... dim_p - 1``.
    """
    k = V_u.order
    D_ref = _bdm_k_reference_divergence_matrix(k)        # (dim_p, Np)
    u_local = bdm1_gather(V_u.topology, u_global)         # (K, Np)
    return jnp.einsum("ml,kl->km", D_ref, u_local)        # (K, dim_p)


def bdm_k_gradient_apply(
    V_u: BDM, p_per_elem: jax.Array,
) -> jnp.ndarray:
    """Apply ``G : P_{k-1} -> BDM_k`` (defined as ``-D^T``) to a
    per-element pressure DOF array of shape ``(K, dim_p)``.

    Returns a global velocity DOF vector. Saddle-point sign convention:
    ``<G p, u>_BDM = -<p, D u>_P`` for any matched pair.
    """
    k = V_u.order
    D_ref = _bdm_k_reference_divergence_matrix(k)        # (dim_p, Np)
    # G_local[K, ell] = -sum_m D_ref[m, ell] * p_per_elem[K, m]
    G_local = -jnp.einsum("ml,km->kl", D_ref, p_per_elem)  # (K, Np)
    return bdm1_scatter_add(V_u.topology, G_local)


# ---------------------------------------------------------------------------
# Face quadrature data + SIPG face apply for arbitrary order
# ---------------------------------------------------------------------------


def _bdm_k_face_quadrature_data(V: BDM):
    """Order-parametric face quadrature data. Generalises
    :func:`_bdm1_face_quadrature_data` to k+1 GL points per face and
    evaluates the (per-element-constant on affine, but per-q on
    higher-order) basis trace + reference gradient at those points.

    Returns a dict with keys:
      ``phi_at_face`` : ``(K, Nfaces, Q, Np, 2)`` -- Piola-mapped
          unsigned physical basis at the Q = k+1 face GL points.
      ``grad_ref_at_face`` : ``(Nfaces, Q, Np, 2, 2)`` -- reference
          gradient of each basis at the face GL points. Same for
          every element (geometry-independent).
      ``n_face`` : ``(K, Nfaces, 2)`` outward normals.
      ``edge_jac`` : ``(K, Nfaces)`` parameterised face Jacobian
          (L_phys / 2 -- the surface measure for Gauss-Legendre
          quadrature on xi in [-1, 1]).
      ``K_nbr`` / ``f_nbr`` : ``(K, Nfaces)`` neighbour links.
      ``q_nbr`` : ``(K, Nfaces, Q)`` per-face GL-point permutation
          when adjacent elements traverse the edge in opposite
          directions: K's node j corresponds to K's node (Q-1-j).
      ``is_boundary`` : ``(K, Nfaces)`` bool.
      ``w_q`` : ``(Q,)`` Gauss-Legendre weights on ``[-1, 1]``
          summing to 2.
    """
    mesh = V.mesh
    Nfaces = 3
    k = V.order
    Q = k + 1
    K = mesh.K

    # Face DOF coords (= face quadrature points for BDM_k by construction).
    face_dof_coords = bdm_k_face_dof_coordinates(k)        # (Nfaces*Q, 2)
    r_at = face_dof_coords[:, 0]
    s_at = face_dof_coords[:, 1]

    # Unsigned reference basis + reference gradient at face quad points.
    phi_ref = bdm_k_basis_at_points(k, r_at, s_at)         # (Nfaces*Q, Np, 2)
    grad_ref = bdm_k_basis_grad_at_points(k, r_at, s_at)   # (Nfaces*Q, Np, 2, 2)
    # Piola-map the basis per element: phi_phys = (1/J) DF phi_ref.
    DF, _ = _element_jacobian_matrices(mesh)               # (K, 2, 2)
    J = mesh.J[0]                                            # (K,)
    phi_phys = jnp.einsum("kca,nla->knlc", DF, phi_ref) / J[:, None, None, None]
    # Reshape to face-major: (K, Nfaces, Q, Np, 2).
    Np = (k + 1) * (k + 2)
    phi_at_face = phi_phys.reshape(K, Nfaces, Q, Np, 2)
    # grad_ref is element-independent; reshape to (Nfaces, Q, Np, 2, 2).
    grad_ref_face = grad_ref.reshape(Nfaces, Q, Np, 2, 2)

    # Outward normals + edge Jacobians: constant per face on affine.
    Nfp = mesh.Nfp
    nx_face = mesh.nx.reshape((Nfp, Nfaces, K), order="F")[0]
    ny_face = mesh.ny.reshape((Nfp, Nfaces, K), order="F")[0]
    sJ_face = mesh.sJ.reshape((Nfp, Nfaces, K), order="F")[0]
    n_face = jnp.stack([nx_face, ny_face], axis=-1).transpose(1, 0, 2)  # (K, Nfaces, 2)
    edge_jac = sJ_face.T                                                  # (K, Nfaces)

    EToE = jnp.asarray(mesh.EToE)
    EToF = jnp.asarray(mesh.EToF)
    EToV = jnp.asarray(mesh.EToV)
    K_idx = jnp.arange(K)
    is_boundary = EToE == K_idx[:, None]

    # Traversal direction: reversed -> q_nbr[j] = Q - 1 - j.
    v_K_end = EToV[K_idx[:, None], (jnp.arange(Nfaces)[None, :] + 1) % 3]
    v_Kp_start = EToV[EToE, EToF]
    reversed_ = (v_Kp_start == v_K_end)
    q_idx = jnp.arange(Q)
    q_nbr = jnp.where(
        reversed_[:, :, None],
        (Q - 1) - q_idx[None, None, :],
        q_idx[None, None, :],
    )                                                       # (K, Nfaces, Q)

    # Gauss-Legendre weights on [-1, 1].
    _, w = np.polynomial.legendre.leggauss(Q)
    w_q = jnp.asarray(w)

    return dict(
        phi_at_face=phi_at_face,
        grad_ref_at_face=grad_ref_face,
        n_face=n_face,
        edge_jac=edge_jac,
        K_nbr=EToE,
        f_nbr=EToF,
        q_nbr=q_nbr,
        is_boundary=is_boundary,
        w_q=w_q,
    )


def bdm_k_viscous_face_apply(
    V: BDM, u_global: jax.Array, *,
    nu: float, tau_scale: float = 4.0,
    u_bc_global: jax.Array | None = None,
    is_natural_face: jax.Array | None = None,
) -> jax.Array:
    """SIPG face viscous apply for BDM_k. Generalises
    :func:`bdm1_viscous_face_apply` to arbitrary order; differences
    from k=1 are: the basis gradient at face quadrature points varies
    with the quad point, and there are k+1 quad points per face
    instead of 2.

    Penalty parameter: ``tau = tau_scale * (k+1)^2 / h_F`` (Hesthaven
    style, scaled with the *trial* order k since both u and v live in
    BDM_k = P_k(K)^2).

    ``is_natural_face`` (optional ``(K, Nf)`` bool): when True the face
    carries a free-stress (Neumann) BC and the SIPG bilinear form has
    \\emph{no} face contribution there (the natural BC is absorbed by
    integration by parts in the volume form; no boundary source).
    Implemented by zeroing ``edge_jac`` at those faces, which makes
    all three SIPG terms vanish via their shared ``edge_jac`` factor.
    """
    data = _bdm_k_face_quadrature_data(V)
    phi_face = data["phi_at_face"]                # (K, Nf, Q, Np, 2)
    grad_ref_face = data["grad_ref_at_face"]      # (Nf, Q, Np, 2, 2)
    n_face = data["n_face"]                        # (K, Nf, 2)
    edge_jac = data["edge_jac"]                    # (K, Nf)
    K_nbr = data["K_nbr"]                          # (K, Nf)
    f_nbr = data["f_nbr"]
    q_nbr = data["q_nbr"]                          # (K, Nf, Q)
    is_boundary = data["is_boundary"]              # (K, Nf)
    w_q = data["w_q"]                              # (Q,)

    k_order = V.order
    Nf = 3
    Q = k_order + 1
    Np = (k_order + 1) * (k_order + 2)

    h_F = 2.0 * edge_jac                            # physical edge length
    tau = tau_scale * (k_order + 1) ** 2 / h_F      # (K, Nf); tau_scale
                                                    # may be scalar or (K, Nf)
    if is_natural_face is not None:
        # Zero out the integration weight on natural-BC faces; this
        # cleanly drops the consistency, symmetry, and penalty terms
        # (all carry ``edge_jac``) without touching the per-face u_P /
        # grad_u_P computations.
        edge_jac = jnp.where(is_natural_face, 0.0, edge_jac)

    u_local = bdm1_gather(V.topology, u_global)    # (K, Np)

    # u_M at face quad points: contract u_local with the face-basis trace.
    u_M = jnp.einsum("kl,kfqlc->kfqc", u_local, phi_face)   # (K, Nf, Q, 2)

    # u_P at face quad points: trace from the neighbour at the matching point.
    u_P = u_M[K_nbr[:, :, None], f_nbr[:, :, None], q_nbr]  # (K, Nf, Q, 2)
    if u_bc_global is not None:
        u_bc_local = bdm1_gather(V.topology, u_bc_global)
        u_bc_M = jnp.einsum("kl,kfqlc->kfqc", u_bc_local, phi_face)
        u_P = jnp.where(is_boundary[:, :, None, None],
                        2.0 * u_bc_M - u_M, u_P)
    else:
        u_P = jnp.where(is_boundary[:, :, None, None], -u_M, u_P)

    # True physical gradient of basis at face quad points. Note: the
    # piola contravariant map gives grad(phi_phys) = (1/J) DF grad_ref
    # DF_inv. The 1/J factor is essential for consistency between
    # volume and face contributions in the SIPG bilinear form -- without
    # it, the integration-by-parts cancellation that gives SIPG its
    # consistency fails by a factor of J, killing the convergence rate.
    DF, _ = _element_jacobian_matrices(V.mesh)             # (K, 2, 2)
    DF_inv = jnp.linalg.inv(DF)
    inv_J = 1.0 / V.mesh.J[0]                               # (K,)
    grad_phys_face = jnp.einsum(
        "k,kca,fqlae,ked->kfqlcd",
        inv_J, DF, grad_ref_face, DF_inv,
    )                                                       # (K, Nf, Q, Np, 2, 2)
    grad_u_M_face = jnp.einsum(
        "kl,kfqlcd->kfqcd", u_local, grad_phys_face,
    )                                                       # (K, Nf, Q, 2, 2)

    grad_u_P_face = grad_u_M_face[
        K_nbr[:, :, None], f_nbr[:, :, None], q_nbr,
    ]
    grad_u_P_face = jnp.where(
        is_boundary[:, :, None, None, None],
        grad_u_M_face,
        grad_u_P_face,
    )
    grad_u_n_M = jnp.einsum("kfqcd,kfd->kfqc", grad_u_M_face, n_face)
    grad_u_n_P = jnp.einsum("kfqcd,kfd->kfqc", grad_u_P_face, n_face)
    avg_grad_un = 0.5 * (grad_u_n_M + grad_u_n_P)
    jump_u = u_M - u_P

    grad_test_n = jnp.einsum(
        "kfqlcd,kfd->kfqlc", grad_phys_face, n_face,
    )                                                         # (K, Nf, Q, Np, 2)

    # Three SIPG terms.
    # consistency: -nu * sum_(f, q) w_q * edge_jac * avg_grad_un . test_at_face_q
    consistency = -nu * jnp.einsum(
        "q,kf,kfqc,kfqlc->kl",
        w_q, edge_jac, avg_grad_un, phi_face,
    )
    # symmetry: -0.5 * nu * sum jump_u . grad_test_n
    symmetry = -0.5 * nu * jnp.einsum(
        "q,kf,kfqc,kfqlc->kl",
        w_q, edge_jac, jump_u, grad_test_n,
    )
    # penalty: +nu * tau * sum jump_u . test
    penalty = nu * jnp.einsum(
        "q,kf,kf,kfqc,kfqlc->kl",
        w_q, edge_jac, tau, jump_u, phi_face,
    )
    out_local = consistency + symmetry + penalty
    return bdm1_scatter_add(V.topology, out_local)


def bdm_k_viscous_face_self_block(
    V: BDM, *, nu: float, tau_scale: float = 4.0,
) -> jax.Array:
    """Per-element SIPG face *self*-block in the local unsigned-Piola
    basis. Returns ``(K, Np, Np)``.

    This is the exact diagonal (M-M) block of the assembled viscous
    operator :func:`bdm_k_viscous_face_apply`: the contribution to an
    element's own DOFs from its own DOFs, dropping the neighbour
    coupling. It is **interior/boundary-face aware**, matching the
    operator it preconditions:

    * interior faces use the ``1/2`` gradient averaging and the M-part
      of the jump (``u_M``): consistency/symmetry carry a ``1/2`` factor
      and the penalty a ``1`` factor;
    * boundary faces use the homogeneous Dirichlet ghost ``u_P = -u_M``
      (full gradient, jump ``2 u_M``): consistency/symmetry carry a
      ``1`` factor and the penalty a ``2`` factor.

    Using the Dirichlet (boundary) treatment on interior faces -- as a
    naive "isolated element" block would -- over-weights the penalty on
    the (majority) interior faces by ~4x, so the block-Jacobi smoother
    becomes wildly over-damped and useless inside multigrid. Together
    with :func:`bdm_k_viscous_volume_stiffness` this forms the true
    per-element viscous Helmholtz block.

    ``block[k, i, j]`` is the coefficient of trial DOF ``j`` in the
    residual of test DOF ``i`` on element ``k``."""
    data = _bdm_k_face_quadrature_data(V)
    phi_face = data["phi_at_face"]                 # (K, Nf, Q, Np, 2)
    grad_ref_face = data["grad_ref_at_face"]       # (Nf, Q, Np, 2, 2)
    n_face = data["n_face"]                         # (K, Nf, 2)
    edge_jac = data["edge_jac"]                    # (K, Nf)
    is_boundary = data["is_boundary"]              # (K, Nf)
    w_q = data["w_q"]                              # (Q,)

    k_order = V.order
    h_F = 2.0 * edge_jac
    tau = tau_scale * (k_order + 1) ** 2 / h_F     # (K, Nf)
    # Per-face factors: interior -> (1/2 avg, 1x penalty);
    # boundary (Dirichlet ghost) -> (full grad, 2x penalty).
    cs_fac = jnp.where(is_boundary, 1.0, 0.5)      # (K, Nf)
    pen_fac = jnp.where(is_boundary, 2.0, 1.0)     # (K, Nf)

    DF, _ = _element_jacobian_matrices(V.mesh)
    DF_inv = jnp.linalg.inv(DF)
    inv_J = 1.0 / V.mesh.J[0]
    grad_phys_face = jnp.einsum(
        "k,kca,fqlae,ked->kfqlcd", inv_J, DF, grad_ref_face, DF_inv,
    )                                              # (K, Nf, Q, Np, 2, 2)
    grad_n = jnp.einsum(
        "kfqlcd,kfd->kfqlc", grad_phys_face, n_face,
    )                                              # (K, Nf, Q, Np, 2)

    consistency = -nu * jnp.einsum(
        "q,kf,kf,kfqjc,kfqic->kij", w_q, edge_jac, cs_fac, grad_n, phi_face,
    )
    symmetry = -nu * jnp.einsum(
        "q,kf,kf,kfqjc,kfqic->kij", w_q, edge_jac, cs_fac, phi_face, grad_n,
    )
    penalty = nu * jnp.einsum(
        "q,kf,kf,kf,kfqjc,kfqic->kij",
        w_q, edge_jac, tau, pen_fac, phi_face, phi_face,
    )
    return consistency + symmetry + penalty         # (K, Np, Np)


def bdm_k_viscous_apply(
    V: BDM, u_global: jax.Array, *,
    nu: float, tau_scale: float = 4.0,
    u_bc_global: jax.Array | None = None,
    is_natural_face: jax.Array | None = None,
) -> jax.Array:
    """Full SIPG viscous apply = volume + face contributions, for
    arbitrary BDM_k. ``tau_scale`` may be a scalar or a ``(K, Nf)``
    array broadcastable onto the per-face penalty.

    ``is_natural_face`` (optional ``(K, Nf)`` bool): faces flagged True
    carry a free-stress (Neumann) BC and contribute nothing to the
    face term (see :func:`bdm_k_viscous_face_apply`)."""
    return (
        bdm_k_viscous_volume_apply(V, u_global, nu=nu)
        + bdm_k_viscous_face_apply(
            V, u_global,
            nu=nu, tau_scale=tau_scale,
            u_bc_global=u_bc_global,
            is_natural_face=is_natural_face,
        )
    )


# ---------------------------------------------------------------------------
# Non-homogeneous Dirichlet: boundary velocity sampler + SIPG lift
# ---------------------------------------------------------------------------


def bdm_k_evaluate_boundary_velocity(V: BDM, u_bc_fn) -> jax.Array:
    """Sample the user-supplied vector BC ``u_bc_fn(x, y) -> (u_x, u_y)``
    at every boundary face's k+1 Gauss-Legendre points.

    Returns shape ``(K, Nfaces, Q, 2)`` where ``Q = k + 1``. Zero on
    interior faces. The full vector (including the tangential
    component) is needed by :func:`bdm_k_viscous_dirichlet_lift`,
    since BDM DOFs by themselves only carry the normal trace.
    """
    data = _bdm_k_face_quadrature_data(V)
    is_boundary = data["is_boundary"]
    k = V.order
    Nfaces = 3
    Q = k + 1

    mesh = V.mesh
    EToV = jnp.asarray(mesh.EToV)
    VX = jnp.asarray(mesh.VX)
    VY = jnp.asarray(mesh.VY)
    K = mesh.K
    face_coords = bdm_k_face_dof_coordinates(k)        # (Nfaces*Q, 2)
    r = face_coords[:, 0]
    s = face_coords[:, 1]
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    x_at = (
        Vx0[:, None]
        + 0.5 * (r[None, :] + 1.0) * (Vx1 - Vx0)[:, None]
        + 0.5 * (s[None, :] + 1.0) * (Vx2 - Vx0)[:, None]
    )
    y_at = (
        Vy0[:, None]
        + 0.5 * (r[None, :] + 1.0) * (Vy1 - Vy0)[:, None]
        + 0.5 * (s[None, :] + 1.0) * (Vy2 - Vy0)[:, None]
    )
    x_face = x_at.reshape(K, Nfaces, Q)
    y_face = y_at.reshape(K, Nfaces, Q)
    u_bc_x, u_bc_y = u_bc_fn(x_face, y_face)
    u_bc_vec = jnp.stack(
        [jnp.asarray(u_bc_x), jnp.asarray(u_bc_y)], axis=-1
    )                                                   # (K, Nfaces, Q, 2)
    return jnp.where(is_boundary[:, :, None, None], u_bc_vec, 0.0)


def _bdm_k_overintegration_face_data(V: BDM, n_pts: int):
    """Build face quadrature data with ``n_pts`` Gauss-Legendre points
    per face -- decoupled from the BDM DOF positions. Used by the
    convective face flux, which is a degree-3k integrand and so
    requires more than the standard k+1 GL points (which only
    integrate degree 2k+1 exactly).

    Returns a dict with the same keys as
    :func:`_bdm_k_face_quadrature_data` but with ``Q = n_pts`` quad
    points instead of ``Q = k + 1``.
    """
    mesh = V.mesh
    Nfaces = 3
    k = V.order
    K = mesh.K
    Q = n_pts

    # GL nodes on [-1, 1] and the per-face parameterisation:
    #   face 0:  r = xi,  s = -1
    #   face 1:  r = -xi, s = xi
    #   face 2:  r = -1,  s = -xi
    nodes_np, weights_np = np.polynomial.legendre.leggauss(Q)
    xi = jnp.asarray(nodes_np)
    ones = jnp.ones_like(xi)
    pts0 = jnp.stack([xi, -ones], axis=1)
    pts1 = jnp.stack([-xi, xi], axis=1)
    pts2 = jnp.stack([-ones, -xi], axis=1)
    face_coords = jnp.concatenate([pts0, pts1, pts2], axis=0)   # (3*Q, 2)
    r_at = face_coords[:, 0]
    s_at = face_coords[:, 1]

    # Basis at the new quadrature points (unsigned Piola-mapped).
    phi_ref = bdm_k_basis_at_points(k, r_at, s_at)              # (3*Q, Np, 2)
    DF, _ = _element_jacobian_matrices(mesh)                     # (K, 2, 2)
    J = mesh.J[0]                                                 # (K,)
    phi_phys = jnp.einsum("kca,nla->knlc", DF, phi_ref) / J[:, None, None, None]
    Np = (k + 1) * (k + 2)
    phi_at_face = phi_phys.reshape(K, Nfaces, Q, Np, 2)

    Nfp = mesh.Nfp
    nx_face = mesh.nx.reshape((Nfp, Nfaces, K), order="F")[0]
    ny_face = mesh.ny.reshape((Nfp, Nfaces, K), order="F")[0]
    sJ_face = mesh.sJ.reshape((Nfp, Nfaces, K), order="F")[0]
    n_face = jnp.stack([nx_face, ny_face], axis=-1).transpose(1, 0, 2)
    edge_jac = sJ_face.T

    EToE = jnp.asarray(mesh.EToE)
    EToF = jnp.asarray(mesh.EToF)
    EToV = jnp.asarray(mesh.EToV)
    K_idx = jnp.arange(K)
    is_boundary = EToE == K_idx[:, None]

    # Reversed traversal: q_nbr[j] = Q-1-j; same logic as the standard
    # face quadrature but with the new Q.
    v_K_end = EToV[K_idx[:, None], (jnp.arange(Nfaces)[None, :] + 1) % 3]
    v_Kp_start = EToV[EToE, EToF]
    reversed_ = (v_Kp_start == v_K_end)
    q_idx = jnp.arange(Q)
    q_nbr = jnp.where(
        reversed_[:, :, None],
        (Q - 1) - q_idx[None, None, :],
        q_idx[None, None, :],
    )
    w_q = jnp.asarray(weights_np)

    return dict(
        phi_at_face=phi_at_face,
        n_face=n_face,
        edge_jac=edge_jac,
        K_nbr=EToE,
        f_nbr=EToF,
        q_nbr=q_nbr,
        is_boundary=is_boundary,
        w_q=w_q,
    )


def bdm_k_convective_lam_face(
    V_u: BDM,
    u_star_global: jax.Array,
    *,
    u_bc_global: jax.Array | None = None,
    n_face_quad_pts: int | None = None,
    is_natural_face: jax.Array | None = None,
) -> jax.Array:
    """Precompute the LF speed envelope ``lam_face`` from a linearisation
    point ``u_star_global``. Mirrors the face-trace logic of
    :func:`bdm_k_convective_apply`; the result can be passed back as
    ``lam_face`` to the same function, letting the caller hoist the
    non-transposable ``jnp.maximum`` outside a linear matvec.
    """
    from ddg.dg2d.cubature import triangle_cubature  # noqa: F401  (parity)
    k = V_u.order
    u_star_local = bdm1_gather(V_u.topology, u_star_global)
    if n_face_quad_pts is None:
        n_face_quad_pts = max(k + 1, (3 * k + 2) // 2)
    fdata = _bdm_k_overintegration_face_data(V_u, n_face_quad_pts)
    phi_face = fdata["phi_at_face"]
    n_face = fdata["n_face"]
    K_nbr = fdata["K_nbr"]
    f_nbr = fdata["f_nbr"]
    q_nbr = fdata["q_nbr"]
    is_boundary = fdata["is_boundary"]
    u_star_M = jnp.einsum("kl,kfqlc->kfqc", u_star_local, phi_face)
    u_star_P = u_star_M[K_nbr[:, :, None], f_nbr[:, :, None], q_nbr]
    # On Dirichlet/natural boundary faces, u_star_P collapses to u_star_M
    # (no neighbour contribution).
    u_star_P = jnp.where(is_boundary[:, :, None, None], u_star_M, u_star_P)
    if is_natural_face is not None:
        u_star_P = jnp.where(
            is_natural_face[:, :, None, None], u_star_M, u_star_P,
        )
    u_star_n_M = jnp.einsum("kfqc,kfc->kfq", u_star_M, n_face)
    u_star_n_P = jnp.einsum("kfqc,kfc->kfq", u_star_P, n_face)
    return jnp.maximum(jnp.abs(u_star_n_M), jnp.abs(u_star_n_P))


def bdm_k_convective_apply(
    V_u: BDM,
    u_global: jax.Array,
    *,
    u_star_global: jax.Array,
    u_bc_global: jax.Array | None = None,
    n_face_quad_pts: int | None = None,
    is_natural_face: jax.Array | None = None,
    lam_face: jax.Array | None = None,
) -> jax.Array:
    """Order-parametric convective apply: volume ``-int (u*_a u_b)(d_b v_a) dx``
    plus Lax-Friedrichs face flux on the tangential trace.

    Volume cubature: Stroud rule of degree ``3k - 1`` (integrand
    ``u_star * u * grad_v`` is exactly degree ``3k - 1``).

    Face cubature: by default ``n_face_quad_pts = ceil((3k + 1) / 2)``,
    chosen so that the face flux integrand ``(u_star * u) * v`` (degree
    ``3k`` on the face) is integrated exactly. This is the "32k
    overintegration" used in ExaDG (see ExaDG
    QuadratureRuleLinearization::Overintegration32k,
    spatial_operator_base.cpp:345-348). With only the standard ``k+1``
    GL points the face flux is under-integrated by ``k-1`` degrees,
    which destroys the optimal ``h^(k+1)`` convergence rate of BDM_k
    on Navier-Stokes (verified empirically: BDM_2 Kovasznay collapses
    to slope 1 without this overintegration).

    For Picard, the caller passes ``u_star_global = u_global``. For
    Newton, this is part of the Jacobian with ``u_star_global`` the
    current outer iterate.

    Boundary handling: when ``u_bc_global is not None``, boundary
    face ghost uses ``u_P = u_bc`` trace (Dirichlet); otherwise
    ``u_P = u_M`` (natural / outflow). ``u_star_P`` is always
    ``u_star_M`` on boundary faces.
    """
    from ddg.dg2d.cubature import triangle_cubature
    k = V_u.order

    u_local = bdm1_gather(V_u.topology, u_global)
    u_star_local = bdm1_gather(V_u.topology, u_star_global)

    # ----- Volume term -----
    deg_vol = max(3 * k - 1, 2)
    r_q, s_q, w_q = triangle_cubature(deg_vol)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V_u.mesh)
    J = V_u.mesh.J[0]
    phi_phys_q = jnp.einsum("kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]
    grad_ref = bdm_k_basis_grad_at_points(k, r_q, s_q)
    DF_inv = jnp.linalg.inv(DF)
    M_grad_q = jnp.einsum("kca,qlae,ked->kqlcd", DF, grad_ref, DF_inv)
    u_at_q = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys_q)
    u_star_at_q = jnp.einsum("kl,kqlc->kqc", u_star_local, phi_phys_q)
    # Volume integrand: -int_K (u*_a u_b)(d_b v_a) dx.  Physical test
    # gradient is (1/J) M_grad_q (since M_grad is J-scaled per the
    # convention) and the volume element brings a J. The two J factors
    # cancel; no explicit J in the cubature sum.
    vol_local = -jnp.einsum(
        "q,kqa,kqb,kqlab->kl",
        w_q, u_star_at_q, u_at_q, M_grad_q,
    )

    # ----- Face term: Lax-Friedrichs on the tangential trace -----
    # Overintegration: ceil((3k + 1) / 2) GL points so the degree-3k
    # face flux integrand is exactly integrated.
    if n_face_quad_pts is None:
        n_face_quad_pts = max(k + 1, (3 * k + 2) // 2)
    fdata = _bdm_k_overintegration_face_data(V_u, n_face_quad_pts)
    phi_face = fdata["phi_at_face"]       # (K, Nf, Q, Np, 2)
    n_face = fdata["n_face"]              # (K, Nf, 2)
    edge_jac = fdata["edge_jac"]          # (K, Nf)
    K_nbr = fdata["K_nbr"]
    f_nbr = fdata["f_nbr"]
    q_nbr = fdata["q_nbr"]
    is_boundary = fdata["is_boundary"]
    w_q_face = fdata["w_q"]

    u_M = jnp.einsum("kl,kfqlc->kfqc", u_local, phi_face)
    u_star_M = jnp.einsum("kl,kfqlc->kfqc", u_star_local, phi_face)
    u_P = u_M[K_nbr[:, :, None], f_nbr[:, :, None], q_nbr]
    u_star_P = u_star_M[K_nbr[:, :, None], f_nbr[:, :, None], q_nbr]
    if u_bc_global is not None:
        u_bc_local = bdm1_gather(V_u.topology, u_bc_global)
        u_bc_M = jnp.einsum("kl,kfqlc->kfqc", u_bc_local, phi_face)
        u_P = jnp.where(is_boundary[:, :, None, None], u_bc_M, u_P)
        u_star_P = jnp.where(is_boundary[:, :, None, None], u_star_M, u_star_P)
    else:
        u_P = jnp.where(is_boundary[:, :, None, None], u_M, u_P)
        u_star_P = jnp.where(is_boundary[:, :, None, None], u_star_M, u_star_P)
    if is_natural_face is not None:
        # On free-stress (Neumann) faces, override the Dirichlet ghost
        # with u_P = u_M so the LF flux reduces to (u_star . n) u_M --
        # the natural outflow flux carrying whatever the local trace
        # provides, no incoming-flux contribution.
        u_P = jnp.where(is_natural_face[:, :, None, None], u_M, u_P)
        u_star_P = jnp.where(
            is_natural_face[:, :, None, None], u_star_M, u_star_P)

    u_star_n_M = jnp.einsum("kfqc,kfc->kfq", u_star_M, n_face)
    u_star_n_P = jnp.einsum("kfqc,kfc->kfq", u_star_P, n_face)
    # ``lam`` is the LF speed envelope. It depends only on u_star (the
    # linearisation point), so when this apply sits inside a linear
    # matvec for GMRES the caller can precompute ``lam_face`` once
    # OUTSIDE the matvec closure; that lifts the non-transposable
    # ``jnp.maximum`` out of the operator and lets JAX's
    # custom_linear_solve transpose check pass.
    if lam_face is None:
        lam = jnp.maximum(jnp.abs(u_star_n_M), jnp.abs(u_star_n_P))
    else:
        lam = lam_face
    F_n_M = u_star_n_M[:, :, :, None] * u_M
    F_n_P = u_star_n_P[:, :, :, None] * u_P
    F_star = 0.5 * (F_n_M + F_n_P) - 0.5 * lam[:, :, :, None] * (u_P - u_M)

    face_local = jnp.einsum(
        "q,kf,kfqc,kfqlc->kl",
        w_q_face, edge_jac, F_star, phi_face,
    )
    return bdm1_scatter_add(V_u.topology, vol_local + face_local)


def bdm_k_viscous_dirichlet_lift(
    V: BDM, u_bc_vec: jax.Array, *,
    nu: float, tau_scale: float = 4.0,
    is_natural_face: jax.Array | None = None,
) -> jax.Array:
    """Boundary-only SIPG lift for the asymmetric-ghost convention.
    Order-parametric replacement for :func:`bdm1_viscous_dirichlet_lift`.

    The non-homogeneous-ghost vs homogeneous-ghost discrepancy is the
    boundary integral
        + 2 nu integral_F u_BC . (grad v . n) dS    (symmetry)
        - 4 nu tau integral_F u_BC . v dS           (penalty)
    so the caller subtracts this lift from the residual.

    ``u_bc_vec`` has shape ``(K, Nfaces, Q, 2)`` -- the full vector
    BC sampled at face GL points. Tangential component matters for
    lid-driven cavity etc.

    ``is_natural_face`` (optional ``(K, Nf)`` bool): faces flagged
    True carry a Neumann BC and the lift integrand vanishes at them
    (no boundary source). Zeroing ``edge_jac`` at those faces is
    sufficient since both integrands share it as a factor."""
    fdata = _bdm_k_face_quadrature_data(V)
    phi_face = fdata["phi_at_face"]               # (K, Nf, Q, Np, 2)
    grad_ref_face = fdata["grad_ref_at_face"]     # (Nf, Q, Np, 2, 2)
    n_face = fdata["n_face"]                       # (K, Nf, 2)
    edge_jac = fdata["edge_jac"]                   # (K, Nf)
    is_boundary = fdata["is_boundary"]             # (K, Nf)
    w_q = fdata["w_q"]                             # (Q,)
    k_order = V.order
    h_F = 2.0 * edge_jac
    tau = tau_scale * (k_order + 1) ** 2 / h_F     # (K, Nf)
    if is_natural_face is not None:
        edge_jac = jnp.where(is_natural_face, 0.0, edge_jac)

    u_bc_M = jnp.where(is_boundary[:, :, None, None], u_bc_vec, 0.0)

    # True physical test gradient . n (1/J factor included, same
    # convention as the corrected face apply).
    DF, _ = _element_jacobian_matrices(V.mesh)
    DF_inv = jnp.linalg.inv(DF)
    inv_J = 1.0 / V.mesh.J[0]
    grad_phys_face = jnp.einsum(
        "k,kca,fqlae,ked->kfqlcd", inv_J, DF, grad_ref_face, DF_inv,
    )
    grad_test_n = jnp.einsum(
        "kfqlcd,kfd->kfqlc", grad_phys_face, n_face,
    )                                              # (K, Nf, Q, Np, 2)

    # Asymmetric-ghost factors -- same as bdm1_viscous_dirichlet_lift.
    symmetry = -nu * jnp.einsum(
        "q,kf,kfqc,kfqlc->kl",
        w_q, edge_jac, u_bc_M, grad_test_n,
    )
    penalty = 2.0 * nu * jnp.einsum(
        "q,kf,kf,kfqc,kfqlc->kl",
        w_q, edge_jac, tau, u_bc_M, phi_face,
    )
    out_local = symmetry + penalty
    return bdm1_scatter_add(V.topology, out_local)


# ---------------------------------------------------------------------------
# BDM_k canonical interpolant + point evaluation
# ---------------------------------------------------------------------------

def bdm_k_evaluate_at_points(V, u_global, x_query, y_query):
    """Brute-force point evaluation of a BDM_k velocity field at
    ``(x_query, y_query)`` of equal shape. Returns ``(u_x, u_y)`` matching
    that shape. The element-containment search is a Python loop over
    ``V.K``; fine for tens-to-hundreds of points (AMR transfer,
    centreline plots) but not performance-critical."""
    k = V.order
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    K = V.K
    u_local = bdm1_gather(V.topology, u_global)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    x_q = np.asarray(x_query).reshape(-1)
    y_q = np.asarray(y_query).reshape(-1)
    out_x = np.zeros_like(x_q)
    out_y = np.zeros_like(y_q)
    for qi, (xq, yq) in enumerate(zip(x_q, y_q)):
        for kk in range(K):
            v0 = (VX[EToV[kk, 0]], VY[EToV[kk, 0]])
            v1 = (VX[EToV[kk, 1]], VY[EToV[kk, 1]])
            v2 = (VX[EToV[kk, 2]], VY[EToV[kk, 2]])
            det = ((v1[0] - v0[0]) * (v2[1] - v0[1])
                   - (v2[0] - v0[0]) * (v1[1] - v0[1]))
            a = ((v1[0] - xq) * (v2[1] - yq)
                 - (v2[0] - xq) * (v1[1] - yq)) / det
            b = ((v2[0] - xq) * (v0[1] - yq)
                 - (v0[0] - xq) * (v2[1] - yq)) / det
            c = 1.0 - a - b
            if a >= -1e-10 and b >= -1e-10 and c >= -1e-10:
                r = 2.0 * b - 1.0
                s = 2.0 * c - 1.0
                phi_ref = bdm_k_basis_at_points(
                    k, jnp.array([r]), jnp.array([s]))
                phi_phys = (DF[kk] @ phi_ref[0].T).T / J[kk]
                u_xy = np.asarray(
                    jnp.einsum("l,lc->c", u_local[kk], phi_phys))
                out_x[qi] = u_xy[0]
                out_y[qi] = u_xy[1]
                break
    return out_x.reshape(np.shape(x_query)), out_y.reshape(np.shape(y_query))


def bdm_k_canonical_interpolant(V, u_fn):
    """BDM_k canonical interpolant of ``u_fn``: face DOFs from
    ``u_fn . n`` at face GL points, interior DOFs from reference-coord
    moments ``int_T u_ref . w_ref dr ds`` with the inverse-Piola
    transformation ``u_ref = J * DF^{-1} * u_phys`` so the interior
    moments correspond to the BDM basis coefficient duality.
    ``u_fn(x, y)`` must accept JAX arrays of equal shape and return a
    tuple ``(u_x, u_y)``. Used for initial conditions and for AMR
    transfer (pass ``u_fn = lambda x, y: bdm_k_evaluate_at_points(
    V_old, u_old, x, y)``)."""
    from ddg.dg2d.cubature import triangle_cubature
    k = V.order; K = V.K; Q = k + 1
    n_face_total = 3 * Q
    n_interior = (k + 1) * (k - 1)

    fdata = _bdm_k_face_quadrature_data(V)
    n_face = fdata['n_face']
    coords = bdm_k_face_dof_coordinates(k)
    r = np.asarray(coords[:, 0]); s = np.asarray(coords[:, 1])
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    x_at = (Vx0[:, None] + 0.5 * (r + 1.0)[None, :] * (Vx1 - Vx0)[:, None]
            + 0.5 * (s + 1.0)[None, :] * (Vx2 - Vx0)[:, None])
    y_at = (Vy0[:, None] + 0.5 * (r + 1.0)[None, :] * (Vy1 - Vy0)[:, None]
            + 0.5 * (s + 1.0)[None, :] * (Vy2 - Vy0)[:, None])
    x_face = jnp.asarray(x_at).reshape(K, 3, Q)
    y_face = jnp.asarray(y_at).reshape(K, 3, Q)
    ux, uy = u_fn(x_face, y_face)
    un = ux * n_face[:, :, 0:1] + uy * n_face[:, :, 1:2]
    u_local_face = un.reshape(K, n_face_total)

    u_local_interior = jnp.zeros((K, n_interior))
    if n_interior > 0:
        interior_tests = _bdm_k_interior_test_functions(k)
        r_q, s_q, w_q = triangle_cubature(2 * k + 2)
        DF, _ = _element_jacobian_matrices(V.mesh)
        DF_inv = jnp.linalg.inv(DF)
        J = V.mesh.J[0]
        rqn = np.asarray(r_q); sqn = np.asarray(s_q)
        x_q = (Vx0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None]
               + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None])
        y_q = (Vy0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None]
               + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None])
        ux_q, uy_q = u_fn(jnp.asarray(x_q), jnp.asarray(y_q))
        u_phys_q = jnp.stack([ux_q, uy_q], axis=-1)
        u_ref_q = jnp.einsum('k,kab,kqb->kqa', J, DF_inv, u_phys_q)
        for i, w_fn in enumerate(interior_tests):
            w_val = _eval_interior_test(w_fn, r_q, s_q)
            mom = jnp.einsum('q,kqc,qc->k', w_q, u_ref_q, w_val)
            u_local_interior = u_local_interior.at[:, i].set(mom)

    u_local = jnp.concatenate([u_local_face, u_local_interior], axis=1)
    sign = V.topology.local_sign
    owner_mask = sign > 0
    lg = V.topology.local_to_global
    u_global = jnp.zeros(V.total_dofs)
    u_global = u_global.at[lg.reshape(-1)].add(
        jnp.where(owner_mask, u_local, 0).reshape(-1)
    )
    return u_global
