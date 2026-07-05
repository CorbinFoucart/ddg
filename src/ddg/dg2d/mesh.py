"""2D mesh + StartUp2D state on triangles, packaged as a pytree.

Port of ``Codes1.1/Codes2D/StartUp2D.m`` plus its connectivity
machinery (``tiConnect2D``, ``BuildMaps2D``, ``Normals2D``,
``GeometricFactors2D``). The integer connectivity (``EToV``, ``EToE``,
``EToF``, ``vmapM``, ``vmapP``, ``Fmask``) is computed once in numpy
since it is purely topological; the geometric fields (``x``, ``y``,
``rx``, ``sx``, ``ry``, ``sy``, ``J``, ``nx``, ``ny``, ``sJ``,
``Fscale``) live in jax so ``jax.grad`` flows through vertex
positions.

Layout convention (mirrors the MATLAB reference):
    Volume fields: ``(Np, K)`` arrays.
    Face fields:   ``(Nfp * Nfaces, K) = ((N+1)*3, K)`` arrays.
    ``vmapM, vmapP`` index into ``u.flatten(order='F')`` (column-major).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.reference import (
    d_matrices_2d,
    equilateral_nodes_2d,
    face_masks_2d,
    lift_2d,
    vandermonde_2d,
    xy_to_rs,
)


_ARRAY_FIELDS = (
    "VX", "VY", "EToV",
    "r", "s", "V", "invV", "Dr", "Ds", "LIFT",
    "Fmask",
    "x", "y",
    "rx", "sx", "ry", "sy", "J",
    "nx", "ny", "sJ", "Fscale",
    "EToE", "EToF",
    "vmapM", "vmapP", "vmapB", "mapB",
)
_META_FIELDS = ("N", "Np", "K", "Nfp", "Nfaces")


@dataclass(frozen=True)
class Mesh2D:
    """All operators + connectivity for a 2D nodal DG mesh on triangles."""

    # --- meta (static) ---
    N: int
    Np: int
    K: int
    Nfp: int
    Nfaces: int

    # --- mesh ---
    VX: jax.Array
    VY: jax.Array
    EToV: jax.Array  # (K, 3)

    # --- reference element ---
    r: jax.Array
    s: jax.Array
    V: jax.Array
    invV: jax.Array
    Dr: jax.Array
    Ds: jax.Array
    LIFT: jax.Array  # (Np, 3*Nfp)

    Fmask: jax.Array  # (Nfp, 3)

    # --- physical grid ---
    x: jax.Array  # (Np, K)
    y: jax.Array  # (Np, K)

    # --- geometric factors ---
    rx: jax.Array
    sx: jax.Array
    ry: jax.Array
    sy: jax.Array
    J: jax.Array

    # --- face data ---
    nx: jax.Array  # (3*Nfp, K)
    ny: jax.Array
    sJ: jax.Array
    Fscale: jax.Array

    # --- connectivity ---
    EToE: jax.Array  # (K, 3)
    EToF: jax.Array  # (K, 3)

    vmapM: jax.Array
    vmapP: jax.Array
    vmapB: jax.Array
    mapB: jax.Array

    # pytree registration.
    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in _ARRAY_FIELDS)
        aux = tuple(getattr(self, name) for name in _META_FIELDS)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        kwargs = dict(zip(_META_FIELDS, aux))
        kwargs.update(dict(zip(_ARRAY_FIELDS, children)))
        return cls(**kwargs)


jax.tree_util.register_pytree_node(
    Mesh2D, Mesh2D.tree_flatten, Mesh2D.tree_unflatten
)


# ----------------------------------------------------------------------
# Programmatic mesh builders
# ----------------------------------------------------------------------

def uniform_rectangle_mesh_2d(
    xmin: float, xmax: float, ymin: float, ymax: float, Kx: int, Ky: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Triangulate ``[xmin, xmax] x [ymin, ymax]`` into ``2 * Kx * Ky`` triangles.

    Each rectangular cell is split along its lower-left -> upper-right
    diagonal: lower-right triangle (V0 V1 V3), upper-left triangle
    (V0 V3 V2), where V0..V3 are the cell corners
    (bottom-left, bottom-right, top-left, top-right).

    Returns ``(VX, VY, EToV)`` with ``EToV`` of shape ``(2*Kx*Ky, 3)``.
    """
    xs = np.linspace(xmin, xmax, Kx + 1)
    ys = np.linspace(ymin, ymax, Ky + 1)
    VX = np.tile(xs, Ky + 1)
    VY = np.repeat(ys, Kx + 1)

    def vid(i: int, j: int) -> int:
        return j * (Kx + 1) + i

    triangles = []
    for j in range(Ky):
        for i in range(Kx):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            # Lower-right triangle (V00, V10, V11) and upper-left (V00, V11, V01).
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))
    EToV = np.array(triangles, dtype=np.int64)
    return jnp.asarray(VX), jnp.asarray(VY), jnp.asarray(EToV)


def rectangle_with_square_hole_mesh_2d(
    xmin: float, xmax: float, ymin: float, ymax: float,
    Kx: int, Ky: int,
    hole_xmin: float, hole_xmax: float,
    hole_ymin: float, hole_ymax: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Structured rectangle triangulation with an axis-aligned square hole.

    Generates the same grid as ``uniform_rectangle_mesh_2d`` but omits
    every cell whose centroid lies inside the box
    ``[hole_xmin, hole_xmax] × [hole_ymin, hole_ymax]``. The hole
    bounds must coincide with cell faces of the structured grid for an
    exact representation (otherwise the omitted cells extend slightly
    beyond the requested hole). Unused vertices are pruned and
    triangle indices renumbered.

    The resulting mesh has two distinct boundary components — the outer
    rectangle and the inner square's perimeter — both axis-aligned, so
    classifying boundary nodes by location is trivial.
    """
    xs = np.linspace(xmin, xmax, Kx + 1)
    ys = np.linspace(ymin, ymax, Ky + 1)
    VX_full = np.tile(xs, Ky + 1)
    VY_full = np.repeat(ys, Kx + 1)

    def vid(i, j):
        return j * (Kx + 1) + i

    triangles = []
    for j in range(Ky):
        for i in range(Kx):
            cx = 0.5 * (xs[i] + xs[i + 1])
            cy = 0.5 * (ys[j] + ys[j + 1])
            inside_hole = (
                hole_xmin <= cx <= hole_xmax and
                hole_ymin <= cy <= hole_ymax
            )
            if inside_hole:
                continue
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    EToV_full = np.array(triangles, dtype=np.int64)
    used = np.unique(EToV_full.flatten())
    remap = -np.ones(VX_full.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    EToV = remap[EToV_full]
    VX = VX_full[used]
    VY = VY_full[used]
    return jnp.asarray(VX), jnp.asarray(VY), jnp.asarray(EToV)


def backward_facing_step_mesh_2d(
    L_in: float, L_out: float, h_inlet: float, step: float,
    Kx: int, Ky: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """L-shaped backward-facing-step domain.

    The full bounding box is ``[0, L_in + L_out] × [0, step + h_inlet]``.
    The bottom-left corner block ``[0, L_in] × [0, step]`` is removed,
    leaving an L-shape: a thin inlet channel of height ``h_inlet`` for
    ``x < L_in``, opening into the full-height channel
    (``step + h_inlet``) for ``x > L_in``. The vertical wall at
    ``x = L_in, y ∈ [0, step]`` is the backward-facing step.

    Inflow is along ``x = 0`` (only the upper strip ``y ∈ [step,
    step+h_inlet]`` is fluid there); outflow is the full-height face
    ``x = L_in + L_out``.

    Cell omission requires the notch corner ``(L_in, step)`` to land on
    a grid intersection — verified with a tolerance check.

    Returns ``(VX, VY, EToV)`` like the other mesh constructors.
    """
    Lx = L_in + L_out
    Ly = step + h_inlet
    hx = Lx / Kx
    hy = Ly / Ky
    i_step = L_in / hx
    j_step = step / hy
    if abs(i_step - round(i_step)) > 1e-9 or abs(j_step - round(j_step)) > 1e-9:
        raise ValueError(
            f"Step corner ({L_in}, {step}) not grid-aligned: "
            f"L_in/hx={i_step}, step/hy={j_step} must be integral. "
            f"Adjust Kx, Ky."
        )
    i_step, j_step = int(round(i_step)), int(round(j_step))

    xs = np.linspace(0.0, Lx, Kx + 1)
    ys = np.linspace(0.0, Ly, Ky + 1)
    VX_full = np.tile(xs, Ky + 1)
    VY_full = np.repeat(ys, Kx + 1)

    def vid(i, j):
        return j * (Kx + 1) + i

    triangles = []
    for j in range(Ky):
        for i in range(Kx):
            # Omit the bottom-left corner block (the removed step region).
            if i < i_step and j < j_step:
                continue
            v00 = vid(i, j); v10 = vid(i + 1, j)
            v01 = vid(i, j + 1); v11 = vid(i + 1, j + 1)
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    EToV_full = np.array(triangles, dtype=np.int64)
    used = np.unique(EToV_full.flatten())
    remap = -np.ones(VX_full.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    EToV = remap[EToV_full]
    VX = VX_full[used]
    VY = VY_full[used]
    return jnp.asarray(VX), jnp.asarray(VY), jnp.asarray(EToV)


def _square_perimeter_ccw_indices(
    Kx: int, Ky: int,
    i_lo: int, i_hi: int, j_lo: int, j_hi: int,
) -> np.ndarray:
    """Return the vertex IDs walking CCW around an axis-aligned rectangle
    sub-grid in the structured (Kx+1)×(Ky+1) vertex array.

    The rectangle's grid corners are at ``(i_lo, j_lo)`` (bottom-left)
    through ``(i_hi, j_hi)`` (top-right). Walks: bottom edge LR, right
    edge BT, top edge RL, left edge TB. The starting corner appears
    only once (not duplicated at the end). The ordering matches the
    angular CCW direction around the rectangle's centre.
    """
    def vid(i: int, j: int) -> int:
        return j * (Kx + 1) + i
    ids = []
    for i in range(i_lo, i_hi):        # bottom edge (without top-right corner)
        ids.append(vid(i, j_lo))
    for j in range(j_lo, j_hi):        # right edge
        ids.append(vid(i_hi, j))
    for i in range(i_hi, i_lo, -1):    # top edge
        ids.append(vid(i, j_hi))
    for j in range(j_hi, j_lo, -1):    # left edge
        ids.append(vid(i_lo, j))
    return np.asarray(ids, dtype=np.int64)


def channel_with_circle_mesh_2d(
    xmin: float, xmax: float, ymin: float, ymax: float,
    Kx: int, Ky: int,
    *,
    cx: float, cy: float, radius: float,
    R_outer: float,
    n_radial: int = 4,
    radial_growth: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Channel ``[xmin, xmax] × [ymin, ymax]`` with a circular hole at
    ``(cx, cy)`` of radius ``radius``, body-fitted via an O-grid.

    Topology: an inner annulus of ``n_radial`` radial layers fills the
    region between the cylinder and a *bounding square* of half-side
    ``R_outer`` centred at ``(cx, cy)``. Outside that bounding square
    is the standard Cartesian channel mesh with that square excised.
    The two regions share vertices on the bounding-square perimeter.

    The outermost annular ring's outer edge follows the bounding
    square exactly (vertices at the grid intersections on its
    perimeter); the inner edge is the cylinder. Radial layers
    interpolate between these two boundaries. The inner edge is a
    polygon — vertices lie on the cylinder, but faces are straight
    chords. Call :func:`apply_circular_hole_2d` afterwards to bend
    these inner faces onto the true arc; with ``n_circ`` typically 16-
    64 the bowing per face is small enough not to invert neighbours.

    Constraints:
      * The bounding square ``[cx-R_outer, cx+R_outer] × [cy-R_outer,
        cy+R_outer]`` must align with the structured grid (its edges
        must coincide with cell faces). This is verified.
      * ``radius < R_outer < min(channel half-extents)``.

    Parameters
    ----------
    n_radial
        Number of radial layers in the O-grid. Layer thickness
        between cylinder and bounding square is split into
        ``n_radial`` cells (linearly by default, or geometrically if
        ``radial_growth != 1.0`` — the ratio of successive layer
        widths going outward).
    radial_growth
        Geometric growth ratio for radial-layer thickness. ``1.0``
        means uniform; ``>1`` puts more cells near the cylinder.

    Returns ``(VX, VY, EToV)`` like the other mesh constructors.
    """
    if not (radius < R_outer):
        raise ValueError(f"radius={radius} must be < R_outer={R_outer}")
    if not (cx - R_outer > xmin and cx + R_outer < xmax
            and cy - R_outer > ymin and cy + R_outer < ymax):
        raise ValueError(
            "Bounding square must lie strictly inside the channel"
        )

    # ----- 1. Outer Cartesian mesh with a *square* hole at the box ----
    hxmin, hxmax = cx - R_outer, cx + R_outer
    hymin, hymax = cy - R_outer, cy + R_outer

    hx = (xmax - xmin) / Kx
    hy = (ymax - ymin) / Ky
    # snap-to-grid check
    i_lo = (hxmin - xmin) / hx
    i_hi = (hxmax - xmin) / hx
    j_lo = (hymin - ymin) / hy
    j_hi = (hymax - ymin) / hy
    for name, val in (("i_lo", i_lo), ("i_hi", i_hi),
                      ("j_lo", j_lo), ("j_hi", j_hi)):
        if abs(val - round(val)) > 1e-9:
            raise ValueError(
                f"Bounding square not grid-aligned: {name}={val} "
                f"(must be integral). Pick R_outer compatible with "
                f"hx={hx}, hy={hy}, or adjust Kx, Ky."
            )
    i_lo, i_hi = int(round(i_lo)), int(round(i_hi))
    j_lo, j_hi = int(round(j_lo)), int(round(j_hi))

    # Full grid vertices.
    xs = np.linspace(xmin, xmax, Kx + 1)
    ys = np.linspace(ymin, ymax, Ky + 1)
    VX_full = np.tile(xs, Ky + 1)
    VY_full = np.repeat(ys, Kx + 1)

    def vid(i, j):
        return j * (Kx + 1) + i

    triangles = []
    for j in range(Ky):
        for i in range(Kx):
            # Skip cells entirely inside the bounding square.
            if i_lo <= i < i_hi and j_lo <= j < j_hi:
                continue
            v00 = vid(i, j); v10 = vid(i + 1, j)
            v01 = vid(i, j + 1); v11 = vid(i + 1, j + 1)
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    # ----- 2. Inner O-grid: cylinder <-> bounding-square perimeter ----
    # Walk vertices of the bounding-square perimeter CCW. The starting
    # vertex is the bottom-right corner of the square (angle 0 from
    # the cylinder centre when y=cy).
    #
    # Actually we want the CCW start to be along the +x ray from the
    # centre so that angular indexing is natural (θ=0 -> +x). The
    # bottom-right corner of the square has angle -π/4; the midpoint
    # of the right edge (i_hi, (j_lo+j_hi)/2) has angle 0 *only* if
    # the square is centred at (cx, cy) and even (j_hi - j_lo) — true
    # by construction. To keep things simple, walk starting from
    # (i_hi, j_lo) (BR corner) and don't insist on θ=0 alignment.
    outer_ids = _square_perimeter_ccw_indices(Kx, Ky, i_lo, i_hi, j_lo, j_hi)
    n_circ = outer_ids.shape[0]
    if n_circ < 8:
        raise ValueError(
            f"Bounding square has only {n_circ} perimeter vertices; "
            f"need at least 8 (refine Kx, Ky or shrink R_outer)."
        )

    # Outer perimeter coordinates (matches the existing grid vertices).
    OUT_X = VX_full[outer_ids]
    OUT_Y = VY_full[outer_ids]
    # Angular position of each outer vertex around the cylinder centre.
    theta_outer = np.arctan2(OUT_Y - cy, OUT_X - cx)

    # Inner-ring (cylinder) vertices: place at the SAME angles as the
    # outer ring. This keeps each O-grid wedge "radial" — the four
    # corners of each wedge share an angle pair, so the cell is
    # quad-like in (r, θ).
    IN_X = cx + radius * np.cos(theta_outer)
    IN_Y = cy + radius * np.sin(theta_outer)

    # Radial spacing: linear, or geometric with ratio radial_growth.
    if abs(radial_growth - 1.0) < 1e-12:
        radial_frac = np.linspace(0.0, 1.0, n_radial + 1)
    else:
        # Geometric series: layer widths w_k = w_0 * radial_growth**k,
        # summed to 1 over n_radial layers. Solve for w_0.
        g = radial_growth
        # sum_{k=0}^{n-1} g^k = (g^n - 1)/(g - 1) for g != 1.
        denom = (g ** n_radial - 1.0) / (g - 1.0)
        w0 = 1.0 / denom
        widths = w0 * g ** np.arange(n_radial)
        radial_frac = np.concatenate(([0.0], np.cumsum(widths)))

    # Build inner mesh vertices. Layer 0 = on the cylinder (inner ring),
    # layer n_radial = on the bounding square (outer ring — already
    # part of VX_full!). Interior layers are new vertices.
    inner_layers_x = []
    inner_layers_y = []
    inner_layers_ids = []          # global IDs per layer per angular slot
    n_full_verts = VX_full.shape[0]

    next_id = n_full_verts
    for L in range(n_radial + 1):
        if L == n_radial:
            # Use the existing outer-ring IDs.
            inner_layers_ids.append(outer_ids)
            inner_layers_x.append(OUT_X)
            inner_layers_y.append(OUT_Y)
        else:
            frac = radial_frac[L]
            xL = IN_X + frac * (OUT_X - IN_X)
            yL = IN_Y + frac * (OUT_Y - IN_Y)
            ids = np.arange(next_id, next_id + n_circ, dtype=np.int64)
            inner_layers_x.append(xL)
            inner_layers_y.append(yL)
            inner_layers_ids.append(ids)
            next_id += n_circ

    # Wedge triangulation: for each angular slot i and each radial
    # layer L (0 <= L < n_radial), the wedge has corners
    #   A = (L,   i)     B = (L,   i+1)
    #   D = (L+1, i)     C = (L+1, i+1)
    # CCW around the wedge interior is D → C → B → A (the cylinder is
    # the *hole*, so walking around the wedge with its interior on the
    # left requires going CCW on the outer ring and CW on the inner
    # ring — opposite of CCW-around-cylinder). Triangulate as
    # (A, D, C) and (A, C, B).
    for L in range(n_radial):
        for i in range(n_circ):
            ip1 = (i + 1) % n_circ
            A = int(inner_layers_ids[L][i])
            B = int(inner_layers_ids[L][ip1])
            C = int(inner_layers_ids[L + 1][ip1])
            D = int(inner_layers_ids[L + 1][i])
            triangles.append((A, D, C))
            triangles.append((A, C, B))

    EToV_full = np.array(triangles, dtype=np.int64)

    # Combine all vertex arrays into one. Outer-ring IDs already
    # reference VX_full; inner-ring IDs start at next_id.
    all_x = [VX_full]
    all_y = [VY_full]
    for L in range(n_radial):                          # skip outer ring
        all_x.append(np.asarray(inner_layers_x[L]))
        all_y.append(np.asarray(inner_layers_y[L]))
    VX_combined = np.concatenate(all_x)
    VY_combined = np.concatenate(all_y)

    # Prune unused vertices and renumber.
    used = np.unique(EToV_full.flatten())
    remap = -np.ones(VX_combined.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    EToV = remap[EToV_full]
    VX = VX_combined[used]
    VY = VY_combined[used]
    return jnp.asarray(VX), jnp.asarray(VY), jnp.asarray(EToV)


# ----------------------------------------------------------------------
# Connectivity (topology only — pure numpy)
# ----------------------------------------------------------------------

def _ti_connect_2d(EToV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangle face connectivity (Toby Isaac's algorithm).

    Returns ``(EToE, EToF)`` of shape ``(K, 3)`` each, 0-indexed.
    Boundary faces point to themselves.
    """
    K = EToV.shape[0]
    Nfaces = 3
    n_nodes = int(EToV.max()) + 1

    # Face k.f -> the pair of vertex IDs on that face (sorted, 0-indexed).
    fnodes = np.vstack([
        EToV[:, [0, 1]],
        EToV[:, [1, 2]],
        EToV[:, [2, 0]],
    ])
    fnodes = np.sort(fnodes, axis=1)

    EToE = np.tile(np.arange(K).reshape(-1, 1), (1, Nfaces))
    EToF = np.tile(np.arange(Nfaces).reshape(1, -1), (K, 1))

    # Unique face id = v_min * n_nodes + v_max.
    face_id = fnodes[:, 0] * n_nodes + fnodes[:, 1]
    # Each face has a global index k * Nfaces + f? In MATLAB the layout is
    # face f for each k, stacked. To match (1:Nfaces*K) we need ordering
    # by face then by element (face 1 of all k, then face 2 of all k, ...).
    # face 0 indices: 0..K-1, face 1: K..2K-1, face 2: 2K..3K-1.
    # Build (face_id, global_face_idx, element, face_within_element):
    n_total = Nfaces * K
    global_idx = np.arange(n_total)
    elements = np.tile(np.arange(K), Nfaces)
    faces_within = np.repeat(np.arange(Nfaces), K)

    # Sort by face_id so matching faces become adjacent.
    order = np.argsort(face_id, kind="stable")
    face_id_sorted = face_id[order]
    elem_sorted = elements[order]
    face_within_sorted = faces_within[order]
    global_idx_sorted = global_idx[order]

    # Find adjacent equal-id pairs.
    matches = np.where(face_id_sorted[:-1] == face_id_sorted[1:])[0]

    # For each match (i, i+1), tell each side about the other.
    for i in matches:
        k1, f1, g1 = elem_sorted[i], face_within_sorted[i], global_idx_sorted[i]
        k2, f2, g2 = elem_sorted[i + 1], face_within_sorted[i + 1], global_idx_sorted[i + 1]
        EToE[k1, f1] = k2
        EToF[k1, f1] = f2
        EToE[k2, f2] = k1
        EToF[k2, f2] = f1
    return EToE, EToF


def _build_maps_2d(
    Np: int, Nfp: int, Nfaces: int, K: int,
    Fmask: np.ndarray,
    x: np.ndarray, y: np.ndarray,
    EToE: np.ndarray, EToF: np.ndarray,
    EToV: np.ndarray, VX: np.ndarray, VY: np.ndarray,
    node_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build node-level connectivity ``(vmapM, vmapP, vmapB, mapB)``.

    For each face pair, finds the permutation matching ``-``-side nodes
    to ``+``-side nodes by spatial proximity in physical coordinates
    (the two faces may have nodes in reversed order). ``vmapM``,
    ``vmapP`` index into the column-major flat ``(Np*K,)`` field array.
    """
    node_ids = np.arange(K * Np).reshape(K, Np).T  # (Np, K), F-order

    vmapM = np.zeros((Nfp, Nfaces, K), dtype=np.int64)
    vmapP = np.zeros((Nfp, Nfaces, K), dtype=np.int64)

    for k in range(K):
        for f in range(Nfaces):
            vmapM[:, f, k] = node_ids[Fmask[:, f], k]

    x_flat = x.flatten(order="F")
    y_flat = y.flatten(order="F")
    for k1 in range(K):
        for f1 in range(Nfaces):
            k2 = EToE[k1, f1]
            f2 = EToF[k1, f1]

            # Reference edge length for tolerance scaling.
            v1 = EToV[k1, f1]
            v2 = EToV[k1, (f1 + 1) % Nfaces]
            ref_d = np.sqrt((VX[v1] - VX[v2]) ** 2 + (VY[v1] - VY[v2]) ** 2)

            vidM = vmapM[:, f1, k1]
            vidP = vmapM[:, f2, k2]
            xM, yM = x_flat[vidM], y_flat[vidM]
            xP, yP = x_flat[vidP], y_flat[vidP]
            D2 = (xM[:, None] - xP[None, :]) ** 2 + (yM[:, None] - yP[None, :]) ** 2
            matched = np.sqrt(D2) < node_tol * ref_d
            for fp in range(Nfp):
                idx = np.where(matched[fp])[0]
                if idx.size > 0:
                    vmapP[fp, f1, k1] = vidP[idx[0]]
                else:
                    vmapP[fp, f1, k1] = vidM[fp]  # boundary fallback

    vmapM = vmapM.flatten(order="F")
    vmapP = vmapP.flatten(order="F")
    mapB = np.where(vmapP == vmapM)[0]
    vmapB = vmapM[mapB]
    return vmapM, vmapP, vmapB, mapB


# ----------------------------------------------------------------------
# Geometric factors and normals (JAX-traceable through (VX, VY))
# ----------------------------------------------------------------------

def _geometric_factors_2d(x: jax.Array, y: jax.Array, Dr: jax.Array, Ds: jax.Array):
    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y
    J = xr * ys - xs * yr  # MATLAB Normals2D form; same magnitude as -xs*yr + xr*ys
    rx = ys / J
    sx = -yr / J
    ry = -xs / J
    sy = xr / J
    return rx, sx, ry, sy, J


def _normals_2d(
    x: jax.Array, y: jax.Array, Dr: jax.Array, Ds: jax.Array,
    Fmask: np.ndarray, Nfp: int, K: int,
):
    """Outward unit normals + surface Jacobians for each face."""
    xr = Dr @ x
    xs = Ds @ x
    yr = Dr @ y
    ys = Ds @ y

    Fmask_flat = jnp.asarray(Fmask.flatten(order="F"))
    fxr = xr[Fmask_flat, :]
    fxs = xs[Fmask_flat, :]
    fyr = yr[Fmask_flat, :]
    fys = ys[Fmask_flat, :]

    fid0 = jnp.arange(0, Nfp)
    fid1 = jnp.arange(Nfp, 2 * Nfp)
    fid2 = jnp.arange(2 * Nfp, 3 * Nfp)

    nx = jnp.zeros((3 * Nfp, K))
    ny = jnp.zeros((3 * Nfp, K))

    # Face 0 (s = -1): unit normal in (r,s) is (0, -1) -> in (x,y) is (yr, -xr).
    nx = nx.at[fid0, :].set(fyr[fid0, :])
    ny = ny.at[fid0, :].set(-fxr[fid0, :])

    # Face 1 (r + s = 0): normal in (r,s) is (1, 1)/sqrt(2); push through
    # the inverse Jacobian to get the (x,y) form (Hesthaven's expressions).
    nx = nx.at[fid1, :].set(fys[fid1, :] - fyr[fid1, :])
    ny = ny.at[fid1, :].set(-fxs[fid1, :] + fxr[fid1, :])

    # Face 2 (r = -1).
    nx = nx.at[fid2, :].set(-fys[fid2, :])
    ny = ny.at[fid2, :].set(fxs[fid2, :])

    sJ = jnp.sqrt(nx**2 + ny**2)
    return nx / sJ, ny / sJ, sJ


# ----------------------------------------------------------------------
# Public assembly
# ----------------------------------------------------------------------

def build_mesh_2d(VX: jax.Array, VY: jax.Array, EToV: jax.Array, N: int) -> Mesh2D:
    """Assemble a Mesh2D from vertex coordinates + element-to-vertex map.

    Equivalent of ``StartUp2D``.
    """
    EToV_np = np.asarray(EToV, dtype=np.int64)
    VX_np = np.asarray(VX, dtype=np.float64)
    VY_np = np.asarray(VY, dtype=np.float64)
    K = EToV_np.shape[0]
    Nfaces = 3
    Nfp = N + 1
    Np = (N + 1) * (N + 2) // 2

    # Reference element (depends only on N).
    x_eq, y_eq = equilateral_nodes_2d(N)
    r, s = xy_to_rs(x_eq, y_eq)
    V = vandermonde_2d(N, r, s)
    invV = jnp.linalg.inv(V)
    Dr, Ds = d_matrices_2d(N, r, s, V)
    Fmask = face_masks_2d(N, r, s)
    LIFT = lift_2d(N, r, s, V, Fmask)

    # Physical nodes: x = 0.5 * (-(r+s)*VX[va] + (1+r)*VX[vb] + (1+s)*VX[vc]).
    va = EToV_np[:, 0]
    vb = EToV_np[:, 1]
    vc = EToV_np[:, 2]
    r_np = np.asarray(r)
    s_np = np.asarray(s)
    x_np = 0.5 * (
        np.outer(-(r_np + s_np), VX_np[va])
        + np.outer(1.0 + r_np, VX_np[vb])
        + np.outer(1.0 + s_np, VX_np[vc])
    )
    y_np = 0.5 * (
        np.outer(-(r_np + s_np), VY_np[va])
        + np.outer(1.0 + r_np, VY_np[vb])
        + np.outer(1.0 + s_np, VY_np[vc])
    )

    rx, sx, ry, sy, J = _geometric_factors_2d(
        jnp.asarray(x_np), jnp.asarray(y_np), Dr, Ds
    )
    nx, ny, sJ = _normals_2d(
        jnp.asarray(x_np), jnp.asarray(y_np), Dr, Ds, Fmask, Nfp, K
    )

    # Fscale = sJ / J[Fmask, :]
    Fmask_flat = Fmask.flatten(order="F")
    Fscale = sJ / J[Fmask_flat, :]

    # Connectivity (topology + node maps).
    EToE_np, EToF_np = _ti_connect_2d(EToV_np)
    vmapM_np, vmapP_np, vmapB_np, mapB_np = _build_maps_2d(
        Np, Nfp, Nfaces, K, Fmask, np.asarray(x_np), np.asarray(y_np),
        EToE_np, EToF_np, EToV_np, VX_np, VY_np,
    )

    return Mesh2D(
        N=N, Np=Np, K=K, Nfp=Nfp, Nfaces=Nfaces,
        VX=jnp.asarray(VX_np), VY=jnp.asarray(VY_np),
        EToV=jnp.asarray(EToV_np),
        r=r, s=s, V=V, invV=invV, Dr=Dr, Ds=Ds, LIFT=LIFT,
        Fmask=jnp.asarray(Fmask),
        x=jnp.asarray(x_np), y=jnp.asarray(y_np),
        rx=rx, sx=sx, ry=ry, sy=sy, J=J,
        nx=nx, ny=ny, sJ=sJ, Fscale=Fscale,
        EToE=jnp.asarray(EToE_np), EToF=jnp.asarray(EToF_np),
        vmapM=jnp.asarray(vmapM_np), vmapP=jnp.asarray(vmapP_np),
        vmapB=jnp.asarray(vmapB_np), mapB=jnp.asarray(mapB_np),
    )


def apply_rectangle_periodic_2d(
    mesh: Mesh2D,
    *,
    xmin: float, xmax: float, ymin: float, ymax: float,
    periodic_x: bool = True,
    periodic_y: bool = True,
    tol: float = 1e-8,
) -> Mesh2D:
    """Make an axis-aligned rectangular mesh periodic.

    Post-processes a mesh built by :func:`build_mesh_2d`: rewires the
    topological connectivity (``EToE``, ``EToF``) and the node maps
    (``vmapP``, ``vmapB``, ``mapB``) so that boundary faces on opposite
    edges of the rectangle are treated as interior face pairs.

    A left-edge face (``x = xmin``) is paired with the right-edge face
    at the same ``y``; a bottom-edge face (``y = ymin``) with the
    top-edge face at the same ``x``. Node-level partners are matched on
    the *shared* coordinate (``y`` for an x-periodic pair), since the
    other coordinate differs by exactly the domain period.

    ``vmapM`` and the geometric fields are unchanged — periodicity is
    purely a connectivity relabelling, so a DG flux ``u(vmapP) -
    u(vmapM)`` wraps around automatically. Corner elements with two
    boundary faces are handled face-by-face, so doubly-periodic
    domains work.

    Constraints: the two edges of each periodic pair must carry the
    same number of faces, and the meshes must line up (matching
    node coordinates on the shared axis). Both are checked.
    """
    Nfp, Nfaces, K = mesh.Nfp, mesh.Nfaces, mesh.K
    EToE = np.asarray(mesh.EToE).copy()
    EToF = np.asarray(mesh.EToF).copy()
    vmapM = np.asarray(mesh.vmapM).reshape((Nfp, Nfaces, K), order="F").copy()
    vmapP = np.asarray(mesh.vmapP).reshape((Nfp, Nfaces, K), order="F").copy()
    x_flat = np.asarray(mesh.x).flatten(order="F")
    y_flat = np.asarray(mesh.y).flatten(order="F")

    is_bdy = np.asarray(mesh.EToE) == np.arange(K)[:, None]  # (K, Nfaces)

    # Collect boundary faces on each edge, tagged by the coordinate
    # that is *shared* with the periodic partner.
    left, right, bottom, top = [], [], [], []
    for k in range(K):
        for f in range(Nfaces):
            if not is_bdy[k, f]:
                continue
            ids = vmapM[:, f, k]
            xm = float(x_flat[ids].mean())
            ym = float(y_flat[ids].mean())
            if abs(xm - xmin) < tol:
                left.append((k, f, ym))
            elif abs(xm - xmax) < tol:
                right.append((k, f, ym))
            elif abs(ym - ymin) < tol:
                bottom.append((k, f, xm))
            elif abs(ym - ymax) < tol:
                top.append((k, f, xm))

    def _stitch(side_a, side_b, match_axis):
        """Pair faces of side_a with side_b, sorted by their shared
        coordinate. ``match_axis`` is 'y' (x-periodic) or 'x'."""
        if len(side_a) != len(side_b):
            raise ValueError(
                f"Periodic edges have {len(side_a)} vs {len(side_b)} "
                f"faces — they must match. Use a structured mesh whose "
                f"opposite edges are discretised identically."
            )
        a_sorted = sorted(side_a, key=lambda t: t[2])
        b_sorted = sorted(side_b, key=lambda t: t[2])
        coord = x_flat if match_axis == "x" else y_flat
        for (k1, f1, c1), (k2, f2, c2) in zip(a_sorted, b_sorted):
            if abs(c1 - c2) > 1e-6:
                raise ValueError(
                    f"Periodic faces fail to line up: shared coord "
                    f"{c1} vs {c2}. Opposite edges must have matching "
                    f"node positions."
                )
            # Topology.
            EToE[k1, f1] = k2
            EToF[k1, f1] = f2
            EToE[k2, f2] = k1
            EToF[k2, f2] = f1
            # Node maps — match on the shared coordinate.
            idsM = vmapM[:, f1, k1]
            idsP = vmapM[:, f2, k2]
            cM = coord[idsM]
            cP = coord[idsP]
            for fp in range(Nfp):
                j = int(np.argmin(np.abs(cP - cM[fp])))
                vmapP[fp, f1, k1] = idsP[j]
            for fp in range(Nfp):
                j = int(np.argmin(np.abs(cM - cP[fp])))
                vmapP[fp, f2, k2] = idsM[j]

    if periodic_x:
        _stitch(left, right, match_axis="y")
    if periodic_y:
        _stitch(bottom, top, match_axis="x")

    vmapM_flat = vmapM.flatten(order="F")
    vmapP_flat = vmapP.flatten(order="F")
    mapB = np.where(vmapP_flat == vmapM_flat)[0]
    vmapB = vmapM_flat[mapB]

    return dataclasses.replace(
        mesh,
        EToE=jnp.asarray(EToE), EToF=jnp.asarray(EToF),
        vmapM=jnp.asarray(vmapM_flat), vmapP=jnp.asarray(vmapP_flat),
        vmapB=jnp.asarray(vmapB), mapB=jnp.asarray(mapB),
    )


def update_mesh_2d_vertices(
    mesh: Mesh2D, VX: jax.Array, VY: jax.Array
) -> Mesh2D:
    """Return a new ``Mesh2D`` with updated vertex positions; topology fixed.

    Connectivity arrays are reused; ``x, y, rx, sx, ry, sy, J, nx, ny,
    sJ, Fscale`` are recomputed from ``VX, VY`` in JAX so ``jax.grad``
    flows through vertex coordinates.
    """
    VX = jnp.asarray(VX, dtype=jnp.float64)
    VY = jnp.asarray(VY, dtype=jnp.float64)
    EToV_np = np.asarray(mesh.EToV, dtype=np.int64)
    va = EToV_np[:, 0]
    vb = EToV_np[:, 1]
    vc = EToV_np[:, 2]

    r = mesh.r
    s = mesh.s
    x = 0.5 * (
        jnp.outer(-(r + s), VX[va])
        + jnp.outer(1.0 + r, VX[vb])
        + jnp.outer(1.0 + s, VX[vc])
    )
    y = 0.5 * (
        jnp.outer(-(r + s), VY[va])
        + jnp.outer(1.0 + r, VY[vb])
        + jnp.outer(1.0 + s, VY[vc])
    )

    rx, sx, ry, sy, J = _geometric_factors_2d(x, y, mesh.Dr, mesh.Ds)
    nx, ny, sJ = _normals_2d(
        x, y, mesh.Dr, mesh.Ds, np.asarray(mesh.Fmask), mesh.Nfp, mesh.K
    )
    Fmask_flat = np.asarray(mesh.Fmask).flatten(order="F")
    Fscale = sJ / J[Fmask_flat, :]

    return dataclasses.replace(
        mesh,
        VX=VX, VY=VY,
        x=x, y=y,
        rx=rx, sx=sx, ry=ry, sy=sy, J=J,
        nx=nx, ny=ny, sJ=sJ, Fscale=Fscale,
    )
