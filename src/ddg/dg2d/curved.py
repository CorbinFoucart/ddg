"""Optional curved-boundary support for the 2D nodal-DG mesh.

By default our meshes use affine triangles (straight edges, constant
geometric factors per element). For problems with smooth curved
boundaries — circular cylinders, airfoils, etc. — projecting the
boundary face nodes onto the true curve and recomputing geometric
factors yields a higher-order isoparametric mapping with much better
boundary accuracy.

This module is opt-in: nothing here is invoked by the default
``build_mesh_2d`` path. Pass an already-built mesh through
:func:`project_to_curve_2d` (or the convenience :func:`apply_circular_hole_2d`)
to get a new mesh with curved boundary elements.

Implementation strategy — minimal viable curvature:

  1. For each boundary face flagged by the user-supplied predicate,
     replace the (Np, K) physical coordinates of the face nodes with
     ``project_fn(x, y)`` — typically the perpendicular projection onto
     the curve.

  2. Apply a per-element Gordon-Hall-style blending so interior nodes
     follow the boundary motion smoothly: each interior node moves
     by ``(1 − distance_to_curved_face) · offset_at_face``, which
     equals the face offset on the curved face and decays to zero at
     the opposite vertex. Prevents element inversion under modest
     curvature.

  3. Recompute all geometric factors (``rx``, ``sx``, ``ry``, ``sy``,
     ``J``, ``nx``, ``ny``, ``sJ``, ``Fscale``) from the new node
     positions — the existing ``_geometric_factors_2d`` and
     ``_normals_2d`` machinery in :mod:`ddg.dg2d.mesh` already handle
     spatially-varying factors (the ``(Np, K)`` shape we already store
     is the right home for them).

Caveat — accuracy: this gets you N-th-order isoparametric boundary
representation but does **not** introduce cubature for volume / face
integrals. For Hesthaven-style high-Re problems where the volume
integral on curved elements is the dominant error source, a cubature
upgrade (per ``CubatureVolumeMesh2D`` + ``GaussFaceMesh2D`` in the
reference) is the natural next step. For low-to-moderate Re problems
the isoparametric approach here is sufficient.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.mesh import Mesh2D, _geometric_factors_2d, _normals_2d


# ---------------------------------------------------------------------------
# Reference-element blending functions for Gordon-Hall interior smoothing
# ---------------------------------------------------------------------------

def _blending_weight_for_face(
    r: jax.Array, s: jax.Array, face: int
) -> jax.Array:
    """Reference-triangle Gordon-Hall blending weight.

    Returns the weight that should multiply a face-offset displacement
    when smoothing it into the interior. The weight is 1 on face
    ``face`` and 0 on the opposite vertex; in between it varies
    linearly.

    Reference-triangle convention used here (Hesthaven 2D nodal-DG):
      * Vertices at ``(-1, -1)``, ``(1, -1)``, ``(-1, 1)``.
      * Face 0 = bottom edge ``s = -1``; opposite vertex ``(-1, 1)``.
      * Face 1 = hypotenuse ``r + s = 0``; opposite vertex ``(-1, -1)``.
      * Face 2 = left edge ``r = -1``; opposite vertex ``(1, -1)``.
    """
    if face == 0:
        return (1.0 - s) / 2.0          # 1 on s=-1, 0 on s=1
    if face == 1:
        return (r + s + 2.0) / 2.0      # 1 on r+s=0, 0 on r+s=-2
    if face == 2:
        return (1.0 - r) / 2.0          # 1 on r=-1, 0 on r=1
    raise ValueError(f"face must be 0, 1, or 2, got {face}")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def project_to_curve_2d(
    mesh: Mesh2D,
    *,
    face_on_curve: Callable[[np.ndarray, np.ndarray], np.ndarray],
    project_fn: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    blend_interior: bool = True,
) -> Mesh2D:
    """Bend boundary faces onto a user-specified curve.

    Parameters
    ----------
    mesh
        Base affine mesh built by ``build_mesh_2d`` (or one already
        partially curved).
    face_on_curve
        Predicate ``(x_face_avg, y_face_avg) -> bool[Nfaces, K]``
        selecting which boundary faces should be projected onto the
        curve. Receives the per-(Nfaces, K) average physical position
        of each face. Only faces marked True AND on the boundary
        (``EToE == self``) are projected.
    project_fn
        ``(x_nodes, y_nodes) -> (x_new, y_new)`` projection — typically
        the orthogonal projection onto the curve. Both inputs and
        outputs are arrays of the same shape; project_fn must be
        traceable through JAX if you intend to differentiate through
        the curved geometry.
    blend_interior
        If True (default), use a Gordon-Hall-style blending so interior
        nodes pick up a fraction of the face-projection offset,
        avoiding element inversion. If False, only the face-trace
        nodes move; interior nodes stay at their affine position. The
        latter is faster but degrades element quality.

    Returns
    -------
    new_mesh
        A new ``Mesh2D`` with updated ``x``, ``y`` and geometric
        factors. Connectivity (``EToV``, ``vmapM``, etc.) is
        unchanged.
    """
    Nfaces = mesh.Nfaces
    K = mesh.K
    Np = mesh.Np
    Nfp = mesh.Nfp

    # Identify curved-boundary faces: predicate AND on the boundary.
    Fmask = np.asarray(mesh.Fmask)
    x_np = np.asarray(mesh.x)
    y_np = np.asarray(mesh.y)
    x_face_avg = np.zeros((Nfaces, K))
    y_face_avg = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        x_face_avg[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face_avg[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T  # (Nfaces, K)
    is_curved = np.asarray(face_on_curve(x_face_avg, y_face_avg)) & is_bdy

    # Start from current node positions.
    x_new = jnp.asarray(mesh.x)
    y_new = jnp.asarray(mesh.y)

    # For each (face, element) marked curved, project the face nodes and
    # (optionally) propagate the offset to interior nodes via blending.
    r = mesh.r
    s = mesh.s
    for f in range(Nfaces):
        ks = np.where(is_curved[f, :])[0]
        if ks.size == 0:
            continue
        Fm = Fmask[:, f]                            # (Nfp,) local node indices
        x_old_face = jnp.asarray(mesh.x)[Fm[:, None], ks[None, :]]  # (Nfp, len(ks))
        y_old_face = jnp.asarray(mesh.y)[Fm[:, None], ks[None, :]]
        x_proj_face, y_proj_face = project_fn(x_old_face, y_old_face)
        dx_face = x_proj_face - x_old_face          # (Nfp, len(ks))
        dy_face = y_proj_face - y_old_face

        # Update the face nodes directly.
        x_new = x_new.at[Fm[:, None], ks[None, :]].set(x_proj_face)
        y_new = y_new.at[Fm[:, None], ks[None, :]].set(y_proj_face)

        if not blend_interior:
            continue

        # Gordon-Hall blending into the interior:
        # For each face node p (parameter t_p on the face), and each
        # interior node (r_i, s_i), find the projection of (r_i, s_i)
        # onto face f and the blending weight. Bilinear basis: the
        # interior offset is φ_f(r_i, s_i) · Δ(t_face(r_i, s_i)).
        weight = _blending_weight_for_face(r, s, f)  # (Np,)
        # Express interior-node offset via the face-trace 1D interp.
        # The face parameter:
        if f == 0:
            t = r                          # face 0 along r, s=-1
            t_face = r[Fm]                 # face nodes' r-coords
        elif f == 1:
            # Face 1: hypotenuse from (1,-1) to (-1,1). Parameter along it:
            # arbitrary monotone choice. Use t = r (decreasing from 1 to -1).
            t = r
            t_face = r[Fm]
        else:  # f == 2
            t = s                          # face 2 along s, r=-1
            t_face = s[Fm]

        # 1D Vandermonde on face nodes (size Nfp×Nfp), then evaluate
        # at all interior reference points to get the interpolation
        # matrix: I[i, p] = ψ_p(t_i) where ψ_p is the Lagrange basis on
        # the face-node parameters.
        from ddg.dg1d.reference import vandermonde_1d
        V1D_face = vandermonde_1d(mesh.N, t_face)  # (Nfp, Nfp)
        V1D_vol = vandermonde_1d(mesh.N, t)        # (Np, Nfp)
        I_face_to_vol = V1D_vol @ jnp.linalg.inv(V1D_face)  # (Np, Nfp)

        dx_interp = I_face_to_vol @ dx_face          # (Np, len(ks))
        dy_interp = I_face_to_vol @ dy_face
        x_new = x_new.at[:, ks].add(weight[:, None] * dx_interp)
        y_new = y_new.at[:, ks].add(weight[:, None] * dy_interp)
        # The face nodes themselves now get the blending added on top
        # of the direct projection. Undo this — set them back to the
        # exact projected positions (blending weight at face 1 should
        # be exactly 1, but Vandermonde inversion can have tiny noise).
        x_new = x_new.at[Fm[:, None], ks[None, :]].set(x_proj_face)
        y_new = y_new.at[Fm[:, None], ks[None, :]].set(y_proj_face)

    # Recompute geometric factors from the new node positions.
    rx, sx, ry, sy, J = _geometric_factors_2d(x_new, y_new, mesh.Dr, mesh.Ds)
    nx, ny, sJ = _normals_2d(
        x_new, y_new, mesh.Dr, mesh.Ds, np.asarray(mesh.Fmask), mesh.Nfp, mesh.K
    )
    Fmask_flat = np.asarray(mesh.Fmask).flatten(order="F")
    Fscale = sJ / J[Fmask_flat, :]

    return dataclasses.replace(
        mesh,
        x=x_new, y=y_new,
        rx=rx, sx=sx, ry=ry, sy=sy, J=J,
        nx=nx, ny=ny, sJ=sJ, Fscale=Fscale,
    )


# ---------------------------------------------------------------------------
# Convenience: circle projection
# ---------------------------------------------------------------------------

def circle_projection(cx: float, cy: float, radius: float):
    """Orthogonal projection onto a circle centred at ``(cx, cy)``."""
    def project(x, y):
        dx, dy = x - cx, y - cy
        r = jnp.sqrt(dx ** 2 + dy ** 2)
        # Avoid div-by-zero at the centre (won't happen for boundary nodes
        # of a hole-bounded mesh, but be safe):
        scale = radius / jnp.where(r > 1e-14, r, 1.0)
        return cx + dx * scale, cy + dy * scale
    return project


def apply_circular_hole_2d(
    mesh: Mesh2D,
    *,
    cx: float, cy: float, radius: float,
    blend_interior: bool = True,
    detection_tol: float = 0.1,
) -> Mesh2D:
    """Convenience: bend the inner boundary of a hole-bounded mesh onto a circle.

    Identifies boundary faces whose midpoint distance to the centre is
    within ``radius * (1 + detection_tol)`` and projects them onto the
    circle of radius ``radius``. Use after building a square-hole mesh
    where the square is inscribed in the desired circle.
    """
    def face_pred(xf, yf):
        d2 = (xf - cx) ** 2 + (yf - cy) ** 2
        return d2 < (radius * (1 + detection_tol)) ** 2
    return project_to_curve_2d(
        mesh,
        face_on_curve=face_pred,
        project_fn=circle_projection(cx, cy, radius),
        blend_interior=blend_interior,
    )
