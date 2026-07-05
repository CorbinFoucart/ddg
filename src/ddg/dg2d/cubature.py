"""High-order cubature on the reference triangle for non-affine
(curved-element) integration in the SIPG operator.

The nodal-DG mass/stiffness assembly elsewhere in :mod:`ddg.dg2d`
uses *diagonal-J* nodal quadrature -- ``mesh.J * (M_ref @ f)`` -- which
is exact for affine elements (constant ``J`` per element) but only
first-order accurate when the element is curved (``J`` varies across
the element). Curved boundary elements produced by
:mod:`ddg.dg2d.curved` therefore need a genuine cubature rule.

Strategy: Stroud conical product of 1D Gauss-Legendre (in ``a``) and
1D Gauss-Jacobi(α=1, β=0) (in ``b``), with the affine map
``r = (1+a)(1-b)/2 - 1, s = b`` from the unit square ``[-1, 1]²`` to
the H&W reference triangle with vertices ``(-1,-1), (1,-1), (-1,1)``.
An ``n``-by-``n`` product integrates polynomials of total degree
``2n-1`` exactly; the Gauss-Jacobi weight absorbs the ``(1-b)/2``
Jacobian of the square→triangle map.

For nodal-DG degree ``N`` with curved boundary elements the mass
integrand ``φ_i φ_j J`` has total degree ``4N-2`` (basis functions
contribute ``2N``; the curved-element Jacobian is the determinant of
a degree-``(N-1)`` map and so contributes ``2(N-1)``). The default
``degree = 4*N - 2`` rule handles this exactly; affine elements
remain exact at any rule degree ``>= 2N``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi

from ddg.dg2d.mesh import Mesh2D
from ddg.dg2d.reference import (
    grad_vandermonde_2d, vandermonde_1d, vandermonde_2d,
)


# ---------------------------------------------------------------------------
# Reference-triangle cubature
# ---------------------------------------------------------------------------

def triangle_cubature(degree: int):
    """Stroud conical rule on the reference triangle ``(-1,-1),(1,-1),(-1,1)``.

    Integrates polynomials of total degree ``<= degree`` exactly.
    Uses ``n = ceil((degree+1)/2)`` points per direction (``n²`` total).
    Weights sum to ``2`` (the triangle area).
    """
    n = max(1, (degree + 1 + 1) // 2)               # ceil((degree+1)/2)
    a_nodes, a_w = leggauss(n)                      # Gauss-Legendre on [-1, 1]
    b_nodes, b_w = roots_jacobi(n, 1, 0)            # Gauss-Jacobi, weight (1-b)
    a_grid, b_grid = np.meshgrid(a_nodes, b_nodes, indexing="ij")
    r = (1.0 + a_grid) * (1.0 - b_grid) / 2.0 - 1.0
    s = b_grid
    w = np.outer(a_w, b_w) / 2.0                    # (1-b)/2 absorbed via Jacobi
    return (jnp.asarray(r.flatten()),
            jnp.asarray(s.flatten()),
            jnp.asarray(w.flatten()))


# ---------------------------------------------------------------------------
# Mesh-attached cubature data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeshCubature:
    """Per-mesh cubature data: reference-element evaluation matrices and
    per-element values of the geometric factors at the cubature points.

    Use to integrate ``∫_K f(r,s) J(r,s) dr ds`` on each element ``K``
    when the element is curved (``J`` non-constant). For an affine mesh
    cubature integration equals the nodal-quadrature `mesh.J * M_ref` to
    machine precision when ``degree >= 2N``.

    Face cubature: ``face_Lq[f]`` interpolates from *volume* nodal
    values ``(Np, K)`` to face-trace values at face-cubature points
    on face ``f``. ``face_sJ[f]``, ``face_nx[f]``, ``face_ny[f]``
    are surface Jacobian and outward-unit-normal components at the
    face cubature points -- both vary along curved faces.
    """
    # Reference-triangle cubature points + weights
    r: jax.Array                 # (Ncub,)
    s: jax.Array                 # (Ncub,)
    w: jax.Array                 # (Ncub,)
    # Lagrange basis + reference derivatives evaluated at cubature points
    Lq: jax.Array                # (Ncub, Np)   -- nodal_value -> cub_value
    Drq: jax.Array               # (Ncub, Np)   -- ∂φ_i/∂r at cubature
    Dsq: jax.Array               # (Ncub, Np)   -- ∂φ_i/∂s at cubature
    # Per-element geometry at cubature points
    J: jax.Array                 # (Ncub, K)
    rx: jax.Array                # (Ncub, K)
    sx: jax.Array                # (Ncub, K)
    ry: jax.Array                # (Ncub, K)
    sy: jax.Array                # (Ncub, K)
    # 1D face cubature (per face, evaluated for each of the 3 reference faces)
    face_w: jax.Array            # (Nfcub,)
    face_Lq: jax.Array           # (Nfaces, Nfcub, Np)
    face_L1d: jax.Array          # (Nfaces, Nfcub, Nfp) face-nodal -> face-cub
    face_sJ: jax.Array           # (Nfaces, Nfcub, K)
    face_nx: jax.Array           # (Nfaces, Nfcub, K)
    face_ny: jax.Array           # (Nfaces, Nfcub, K)


def attach_cubature(
    mesh: Mesh2D,
    degree: Optional[int] = None,
    face_degree: Optional[int] = None,
) -> MeshCubature:
    """Build the cubature data for ``mesh``.

    ``degree`` defaults to ``4N - 2`` -- the degree of the mass-with-
    curved-J integrand for nodal-DG order ``N``. ``face_degree`` defaults
    to ``2N + 1`` (face integrals involve products of degree-N basis
    functions times the surface Jacobian, polynomial degree ≤ 2N on
    affine faces; curved-face sJ adds up to N-1 more).
    """
    N = int(mesh.N)
    if degree is None:
        degree = 4 * N - 2
    if face_degree is None:
        face_degree = max(2 * N + 1, 2 * N)
    r_q, s_q, w_q = triangle_cubature(max(degree, 2 * N))

    # Lagrange basis at cubature: L_i(r_q, s_q) = (V_at_cub @ invV)_{q, i}.
    V_at_cub = vandermonde_2d(N, r_q, s_q)                       # (Ncub, Np)
    Lq = V_at_cub @ mesh.invV                                    # (Ncub, Np)
    # Reference derivatives.
    Vr_at_cub, Vs_at_cub = grad_vandermonde_2d(N, r_q, s_q)
    Drq = Vr_at_cub @ mesh.invV                                  # (Ncub, Np)
    Dsq = Vs_at_cub @ mesh.invV                                  # (Ncub, Np)

    # Per-element x_r, x_s, y_r, y_s, J at volume cubature points.
    x_r = Drq @ mesh.x
    x_s = Dsq @ mesh.x
    y_r = Drq @ mesh.y
    y_s = Dsq @ mesh.y
    J_q = x_r * y_s - x_s * y_r
    rx_q = y_s / J_q
    sx_q = -y_r / J_q
    ry_q = -x_s / J_q
    sy_q = x_r / J_q

    # 1D face cubature.
    n_face = max(1, (face_degree + 2) // 2)
    t_q, fw = leggauss(n_face)
    t_q = jnp.asarray(t_q)
    fw = jnp.asarray(fw)
    Nfcub = t_q.shape[0]

    # Per-face (r, s) at cubature + parameter-direction (dr/dt, ds/dt).
    # Traversal: face f goes from vertex f to vertex (f+1)%3 with CCW
    # orientation; outward normal is then tangent rotated -90°.
    ones = jnp.ones_like(t_q)
    face_rs = [
        ( t_q, -ones),                              # face 0: v0->v1, (r,s)=(t,-1)
        (-t_q,  t_q),                               # face 1: v1->v2, (r,s)=(-t,t)
        (-ones, -t_q),                              # face 2: v2->v0, (r,s)=(-1,-t)
    ]
    drsdt = [(1.0, 0.0), (-1.0, 1.0), (0.0, -1.0)]

    # Face-nodal t values per face (parameterization consistent with face_rs):
    # face 0: t = r at face nodes; face 1: t = -r; face 2: t = -s.
    Fmask_np = np.asarray(mesh.Fmask)
    r_np = np.asarray(mesh.r)
    s_np = np.asarray(mesh.s)
    t_face_nodes = [
        jnp.asarray(r_np[Fmask_np[:, 0]]),
        jnp.asarray(-r_np[Fmask_np[:, 1]]),
        jnp.asarray(-s_np[Fmask_np[:, 2]]),
    ]

    face_Lq_list, face_L1d_list = [], []
    face_sJ_list, face_nx_list, face_ny_list = [], [], []
    for f in range(3):
        r_f, s_f = face_rs[f]
        V_f = vandermonde_2d(N, r_f, s_f)                        # (Nfcub, Np)
        face_Lq_list.append(V_f @ mesh.invV)
        # 1D face-nodal -> face-cub interpolation.
        V1D_face = vandermonde_1d(N, t_face_nodes[f])            # (Nfp, Nfp)
        V1D_cub = vandermonde_1d(N, t_q)                         # (Nfcub, Nfp)
        face_L1d_list.append(V1D_cub @ jnp.linalg.inv(V1D_face))
        # Per-face metric derivatives -> tangent/normal at face cub.
        Vr_f, Vs_f = grad_vandermonde_2d(N, r_f, s_f)
        Drq_f = Vr_f @ mesh.invV
        Dsq_f = Vs_f @ mesh.invV
        dr_dt, ds_dt = drsdt[f]
        xr_f = Drq_f @ mesh.x
        xs_f = Dsq_f @ mesh.x
        yr_f = Drq_f @ mesh.y
        ys_f = Dsq_f @ mesh.y
        x_t = dr_dt * xr_f + ds_dt * xs_f
        y_t = dr_dt * yr_f + ds_dt * ys_f
        sJ_f = jnp.sqrt(x_t ** 2 + y_t ** 2)
        nx_f = y_t / sJ_f
        ny_f = -x_t / sJ_f
        face_sJ_list.append(sJ_f)
        face_nx_list.append(nx_f)
        face_ny_list.append(ny_f)

    return MeshCubature(
        r=r_q, s=s_q, w=w_q,
        Lq=Lq, Drq=Drq, Dsq=Dsq,
        J=J_q, rx=rx_q, sx=sx_q, ry=ry_q, sy=sy_q,
        face_w=fw,
        face_Lq=jnp.stack(face_Lq_list, axis=0),                 # (Nfaces, Nfcub, Np)
        face_L1d=jnp.stack(face_L1d_list, axis=0),               # (Nfaces, Nfcub, Nfp)
        face_sJ=jnp.stack(face_sJ_list, axis=0),                 # (Nfaces, Nfcub, K)
        face_nx=jnp.stack(face_nx_list, axis=0),
        face_ny=jnp.stack(face_ny_list, axis=0),
    )


# ---------------------------------------------------------------------------
# Per-element mass matrix M_K = ∫_K φ_i φ_j dx
# ---------------------------------------------------------------------------

def per_element_mass(cub: MeshCubature) -> jax.Array:
    """``M_K[k, i, j] = Σ_q w_q · J_q[q, k] · L_i(r_q, s_q) · L_j(r_q, s_q)``.

    Returns ``(K, Np, Np)``. For affine elements this equals
    ``J_K · M_ref`` to machine precision (``M_ref = Σ_q w_q Lq^T Lq``);
    for curved elements it is the exact (per chosen cubature) mass
    matrix.
    """
    # einsum: q over Ncub, k over K, i, j over Np.
    return jnp.einsum("q,qk,qi,qj->kij", cub.w, cub.J, cub.Lq, cub.Lq)


def cubature_mass_apply(cub: MeshCubature, f: jax.Array) -> jax.Array:
    """``(M_K @ f)[i, k] = Σ_q w_q · J_q[q, k] · L_i(r_q, s_q) · f_q[q, k]``.

    Drop-in replacement for ``mesh.J * (M_ref @ f)`` on curved meshes.
    ``f`` is the nodal field ``(Np, K)``; result is the same shape, and
    represents ``(test, M f)`` for each nodal basis function. Avoids
    materialising the per-element mass matrix.
    """
    f_q = cub.Lq @ f                                       # (Ncub, K)
    weighted = cub.w[:, None] * cub.J * f_q                # (Ncub, K)
    return cub.Lq.T @ weighted                             # (Np, K)
