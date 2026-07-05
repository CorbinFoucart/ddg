"""Adaptive mesh refinement for 2D triangle meshes.

Phase 1 — the refinement *machinery* and its data structure. The
*strategy* that decides which elements to mark is deliberately kept
out of this module: refinement takes the marked-element set as a
plain input, so a gradient indicator, an error estimator, or
(eventually) a learned model are all interchangeable producers of
that set.

Technique and data structure are deliberately the common, robust,
textbook choices, picked for extensibility rather than peak
performance:

  * **Data structure — a hierarchical refinement forest**
    (:class:`RefinementForest`). The original macro-elements are the
    level-0 roots; red-refining a leaf adds four children; the
    *leaves* are the active mesh. This is the standard AMR structure
    (deal.II, p4est, libMesh, ...). It is what makes the machinery
    extensible: refinement levels, multi-level refinement, coarsening
    (drop a parent's children) and ML-driven marking all operate on
    the forest without touching the solver.

  * **Technique — conforming red--green ``h``-refinement.** The
    forest stores only *red* refinement (the clean 1 -> 4 quadtree
    split via edge midpoints). Conformity is restored at mesh-
    extraction time by *green* closure — an element left with a
    single hanging node is bisected 1 -> 2. Green "transition" cells
    are not tree nodes; they are regenerated each time the active
    mesh is built. This separation (persistent red tree + stateless
    green closure) is the standard way to keep the hierarchy clean.

Because the extracted active mesh is conforming, it is an ordinary
``(VX, VY, EToV)`` triple rebuilt with :func:`build_mesh_2d` — so the
whole existing solver stack runs on it unchanged.

Refinement marks are closed under two standard rules before the tree
is grown:

  * **2:1 balance** — face-adjacent leaves differ by at most one
    refinement level; refining a leaf pulls in any coarser neighbour.
  * **green-closability** — a leaf that would be left with two or
    three hanging nodes cannot be closed by a single bisection, so it
    is promoted to a full red refinement.

Both are resolved by a fixed-point iteration over the marked set.

Limitations (intentional, for Phase 1):

  * Refinement only; :class:`RefinementForest` is built to support
    coarsening (the tree makes it natural) but ``coarsen`` is left
    for a later pass.
  * Straight-sided — curvature of an input mesh
    (:func:`apply_circular_hole_2d`) is not carried to children.
  * Repeatedly green-refining the same green cell can erode angle
    quality; the forest tracks levels so a quality-preserving
    multi-level scheme can be layered on later.

Solution transfer onto the refined mesh is :func:`transfer_field`: a
fine element's nodal values come from evaluating its parent's
degree-``N`` polynomial at the fine node locations. This is *exact* —
every fine triangle is geometrically nested in its parent, so the
transfer just re-represents the same piecewise polynomial on a finer
partition. :meth:`RefinementForest.adapt` ties indicator-driven
refinement and field transfer together into one step usable inside a
time-stepping loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.mesh import Mesh2D, build_mesh_2d


def _edge(a: int, b: int) -> tuple[int, int]:
    """Undirected edge key — the vertex pair, smaller id first."""
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class RefinementResult:
    """Outcome of a one-shot :func:`refine_mesh_2d` call.

    Attributes
    ----------
    mesh
        The refined :class:`Mesh2D` (same polynomial order as input).
    parent
        ``(K_new,)`` int array — ``parent[i]`` is the index, in the
        *original* mesh, of the macro-element that active element
        ``i`` lies in. The hook for Phase-2 solution transfer.
    n_refined
        Number of original elements that were red-refined (after the
        2:1-balance / green-closability closure of the marked set).
    forest
        The :class:`RefinementForest` — kept so the caller can
        refine further, query levels, etc.
    """
    mesh: Mesh2D
    parent: np.ndarray
    n_refined: int
    forest: "RefinementForest"


class RefinementForest:
    """Hierarchical refinement forest over a 2D triangle mesh.

    Roots are the macro-elements of the seed mesh. Red-refining a
    leaf appends four children. The leaves are the active mesh;
    :meth:`active_mesh` extracts a conforming :class:`Mesh2D` from
    them via green closure.

    Node arrays (indexed by node id, roots first, in seed-element
    order — so a freshly built forest has leaf id == seed element
    index):

      ``_verts``     (n_nodes, 3)  vertex ids of each node's triangle
      ``_level``     (n_nodes,)    refinement level, 0 at the roots
      ``_parent``    (n_nodes,)    parent node id, -1 for roots
      ``_children``  list          4 child ids per red node, () if leaf
    """

    def __init__(self, mesh: Mesh2D, *, max_level: int | None = None,
                 periodic: dict | None = None):
        EToV = np.asarray(mesh.EToV, dtype=np.int64)
        self.N = int(mesh.N)
        # Periodic boundary identification. ``periodic`` maps an axis
        # (``"x"`` / ``"y"``) to the (lo, hi) coordinates of the two
        # boundaries identified along it, e.g.
        # ``periodic={"x": (0.0, 1.0)}``. Opposite boundaries are then
        # refined in lockstep (see ``_close_marks``) so the extracted
        # active mesh stays consistent for ``apply_rectangle_periodic_2d``.
        self._periodic: dict = dict(periodic) if periodic else {}
        # Hard cap on the refinement level. A leaf at ``max_level`` is
        # never marked; with the 2:1 balance kept by green closure this
        # also means it is never *promoted* by the closure, so the
        # realised maximum level is exactly ``max_level``.
        self.max_level = max_level
        self._vx: list[float] = [float(v) for v in np.asarray(mesh.VX)]
        self._vy: list[float] = [float(v) for v in np.asarray(mesh.VY)]
        self._verts: list[tuple[int, int, int]] = [
            (int(a), int(b), int(c)) for a, b, c in EToV
        ]
        K = EToV.shape[0]
        self._level: list[int] = [0] * K
        self._parent: list[int] = [-1] * K
        self._children: list[tuple[int, ...]] = [()] * K
        # edge -> midpoint vertex id (dedup; shared across elements)
        self._edge_mid: dict[tuple[int, int], int] = {}

    # ----- queries -------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self._verts)

    def leaves(self) -> np.ndarray:
        """Node ids of the current leaves (the active elements)."""
        return np.asarray(
            [i for i in range(self.n_nodes) if not self._children[i]],
            dtype=np.int64,
        )

    def leaf_levels(self) -> np.ndarray:
        """Refinement level of each leaf, aligned with :meth:`leaves`."""
        return np.asarray(
            [self._level[i] for i in range(self.n_nodes)
             if not self._children[i]],
            dtype=np.int64,
        )

    def _node_edges(self, i: int) -> tuple[tuple[int, int], ...]:
        v0, v1, v2 = self._verts[i]
        return (_edge(v0, v1), _edge(v1, v2), _edge(v2, v0))

    def _periodic_edge_key(self, a: int, b: int):
        """Canonical key of a periodic-boundary edge, or ``None``.

        An edge lying along a periodic boundary line is keyed on the
        *shared-axis* span of its endpoints; the edge on the opposite
        boundary with the same span gets the identical key, so the two
        are recognised as periodic partners regardless of side. As an
        edge is bisected the half-edge spans change identically on both
        sides (the boundaries refine in lockstep), so the keys keep
        matching.
        """
        if not self._periodic:
            return None
        tol = 1e-7
        for axis, (lo, hi) in self._periodic.items():
            if axis == "x":
                ca, cb = self._vx[a], self._vx[b]
                sa, sb = self._vy[a], self._vy[b]
            else:
                ca, cb = self._vy[a], self._vy[b]
                sa, sb = self._vx[a], self._vx[b]
            on_lo = abs(ca - lo) < tol and abs(cb - lo) < tol
            on_hi = abs(ca - hi) < tol and abs(cb - hi) < tol
            if on_lo or on_hi:
                span = (round(min(sa, sb), 6), round(max(sa, sb), 6))
                return (axis, span)
        return None

    # ----- refinement ----------------------------------------------------

    def refine(self, marked) -> int:
        """Red-refine the marked leaves (in place). Returns the number
        of leaves actually refined after closure.

        ``marked`` is either leaf *node ids* (array-like of ints) or a
        boolean mask aligned with :meth:`leaves`. The marked set is
        closed under 2:1 balance and green-closability before the tree
        is grown, so the leaves always stay green-closable.
        """
        leaf_ids = self.leaves()
        leaf_set = set(int(i) for i in leaf_ids)
        marked = np.asarray(marked)
        if marked.dtype == bool:
            if marked.shape[0] != leaf_ids.shape[0]:
                raise ValueError(
                    f"boolean mask length {marked.shape[0]} != "
                    f"n_leaves {leaf_ids.shape[0]}"
                )
            to_red = set(int(leaf_ids[k]) for k in np.where(marked)[0])
        else:
            to_red = set(int(k) for k in marked.reshape(-1))
        for k in to_red:
            if k not in leaf_set:
                raise ValueError(f"{k} is not a current leaf node")

        # Honour the hard depth cap: a leaf already at max_level is
        # silently dropped from the marked set.
        if self.max_level is not None:
            to_red = {k for k in to_red
                      if self._level[k] < self.max_level}

        to_red = self._close_marks(leaf_set, to_red)
        for node in sorted(to_red):
            self._red_refine_node(node)
        return len(to_red)

    def _close_marks(
        self, leaf_set: set[int], to_red: set[int],
    ) -> set[int]:
        """Grow ``to_red`` until 2:1 balance and green-closability hold.

        Hanging interfaces must be detected *across* refinement
        levels: a coarse leaf owns a full edge ``(a, b)`` while the
        finer leaves across it own the half-edges ``(a, m)``,
        ``(m, b)``. Adjacency therefore bridges the edge midpoint --
        matching full-edge keys alone (as an earlier version did)
        misses hanging nodes left by *previous* refinement passes, so
        a leaf can silently accumulate two of them under iterative
        refinement.

        Works with *effective* levels (a leaf in ``to_red`` counts one
        level finer). A leaf already at ``max_level`` is never
        promoted; since the realised mesh then never exceeds
        ``max_level``, such a leaf has no finer neighbour and so never
        needs promotion anyway -- the guard is belt-and-braces.
        """
        cap = self.max_level
        # edge key -> leaves owning that exact edge. Full edges and
        # the half-edges of finer leaves both appear as keys.
        edge_owners: dict[tuple[int, int], list[int]] = {}
        for i in leaf_set:
            for e in self._node_edges(i):
                edge_owners.setdefault(e, []).append(i)

        # Periodic edge groups: leaves owning periodically-identified
        # boundary edges (one per side) are co-marked so opposite
        # boundaries always refine in lockstep.
        periodic_groups: dict = {}
        if self._periodic:
            for i in leaf_set:
                for e in self._node_edges(i):
                    k = self._periodic_edge_key(*e)
                    if k is not None:
                        periodic_groups.setdefault(k, []).append(i)

        def finer_across(e: tuple[int, int]) -> list[int]:
            """Current leaves on the finer side of edge ``e``."""
            m = self._edge_mid.get(e)
            if m is None:
                return []
            a, b = e
            return (edge_owners.get(_edge(a, m), [])
                    + edge_owners.get(_edge(m, b), []))

        def full_nbr(i: int, e: tuple[int, int]) -> int | None:
            """The same-level leaf sharing the exact edge ``e``."""
            for j in edge_owners.get(e, ()):
                if j != i:
                    return j
            return None

        while True:
            changed = False
            eff = {i: self._level[i] + (1 if i in to_red else 0)
                   for i in leaf_set}
            for i in leaf_set:
                if i in to_red:
                    continue
                if cap is not None and self._level[i] >= cap:
                    continue
                hanging = 0
                need = False
                for e in self._node_edges(i):
                    finer = finer_across(e)
                    if finer:
                        # An already-subdivided edge is a hanging
                        # interface; 2:1 balance also demands the
                        # finer side stay within one level.
                        hanging += 1
                        for j in finer:
                            if eff[j] - eff[i] >= 2:
                                need = True
                    else:
                        fn = full_nbr(i, e)
                        if fn is not None and fn in to_red:
                            # same-level neighbour about to refine
                            hanging += 1
                # >=2 hanging edges cannot be closed by one bisection
                if hanging >= 2:
                    need = True
                if need:
                    to_red.add(i)
                    changed = True
            # Periodic co-marking: if either leaf of a periodic edge
            # pair refines, so must the other.
            for members in periodic_groups.values():
                if not any(m in to_red for m in members):
                    continue
                for m in members:
                    if m in to_red:
                        continue
                    if cap is not None and self._level[m] >= cap:
                        continue
                    to_red.add(m)
                    changed = True
            if not changed:
                return to_red

    def _get_midpoint(self, a: int, b: int) -> int:
        """Vertex id of edge ``(a, b)``'s midpoint, creating it once."""
        e = _edge(a, b)
        m = self._edge_mid.get(e)
        if m is None:
            m = len(self._vx)
            self._vx.append(0.5 * (self._vx[a] + self._vx[b]))
            self._vy.append(0.5 * (self._vy[a] + self._vy[b]))
            self._edge_mid[e] = m
        return m

    def _red_refine_node(self, node: int) -> None:
        """Split a leaf into four red children (preserving CCW)."""
        v0, v1, v2 = self._verts[node]
        m01 = self._get_midpoint(v0, v1)
        m12 = self._get_midpoint(v1, v2)
        m20 = self._get_midpoint(v2, v0)
        lvl = self._level[node] + 1
        child_tris = [
            (v0, m01, m20),
            (m01, v1, m12),
            (m20, m12, v2),
            (m01, m12, m20),
        ]
        kids = []
        for tri in child_tris:
            cid = len(self._verts)
            self._verts.append(tri)
            self._level.append(lvl)
            self._parent.append(node)
            self._children.append(())
            kids.append(cid)
        self._children[node] = tuple(kids)

    # ----- active mesh extraction ---------------------------------------

    def _root_of(self, node: int) -> int:
        while self._parent[node] != -1:
            node = self._parent[node]
        return node

    def active_mesh(self) -> tuple[Mesh2D, np.ndarray]:
        """Extract the conforming active :class:`Mesh2D`.

        Applies green closure to the leaves, prunes unused vertices,
        and rebuilds. Returns ``(mesh, active_to_leaf)`` where
        ``active_to_leaf[i]`` is the *leaf node id* that active
        element ``i`` belongs to (a green-bisected leaf yields two
        active elements that share its node id). This map is the hook
        both for solution transfer and for routing a per-element
        error indicator back to the leaf it should refine.
        """
        leaf_ids = self.leaves()
        # A leaf edge has a hanging node when its midpoint exists and
        # is a vertex of some (finer) leaf.
        used_verts: set[int] = set()
        for i in leaf_ids:
            used_verts.update(self._verts[i])

        tris: list[tuple[int, int, int]] = []
        active_to_leaf: list[int] = []
        for i in leaf_ids:
            v0, v1, v2 = self._verts[i]
            e01, e12, e20 = self._node_edges(i)
            m01 = self._edge_mid.get(e01)
            m12 = self._edge_mid.get(e12)
            m20 = self._edge_mid.get(e20)
            h0 = m01 is not None and m01 in used_verts
            h1 = m12 is not None and m12 in used_verts
            h2 = m20 is not None and m20 in used_verts
            nh = int(h0) + int(h1) + int(h2)
            if nh == 0:
                subs = [(v0, v1, v2)]
            elif nh == 1:
                # green closure — bisect from the hanging midpoint to
                # the opposite vertex, preserving CCW winding.
                if h0:
                    subs = [(v0, m01, v2), (m01, v1, v2)]
                elif h1:
                    subs = [(v1, m12, v0), (m12, v2, v0)]
                else:
                    subs = [(v2, m20, v1), (m20, v0, v1)]
            else:
                # Should not happen: the mark closure promotes any
                # leaf with >=2 hanging edges to a red refinement.
                raise RuntimeError(
                    f"leaf {i} has {nh} hanging nodes after closure — "
                    f"this is a bug in _close_marks"
                )
            for tri in subs:
                tris.append(tri)
                active_to_leaf.append(int(i))

        # Prune unused vertices, renumber.
        EToV_full = np.asarray(tris, dtype=np.int64)
        used = np.unique(EToV_full.reshape(-1))
        remap = -np.ones(len(self._vx), dtype=np.int64)
        remap[used] = np.arange(used.shape[0])
        EToV = remap[EToV_full]
        VX = np.asarray(self._vx, dtype=np.float64)[used]
        VY = np.asarray(self._vy, dtype=np.float64)[used]

        mesh = build_mesh_2d(
            jnp.asarray(VX), jnp.asarray(VY), jnp.asarray(EToV), self.N,
        )
        return mesh, np.asarray(active_to_leaf, dtype=np.int64)

    # ----- indicator-driven refinement ----------------------------------

    def refine_by_indicator(
        self, indicator, active_to_leaf, *, theta: float = 0.5,
    ) -> int:
        """Refine the leaves flagged by a per-element error indicator.

        ``indicator`` is a per-active-element error indicator (any
        non-negative array, e.g. from :func:`kelly_error_indicator`);
        ``active_to_leaf`` is the map returned by :meth:`active_mesh`.
        The per-element indicator is first aggregated to a per-leaf
        value (the max over the active elements of each leaf), then
        Dörfler bulk marking selects the leaves carrying the top
        ``theta`` fraction of the total. Leaves at ``max_level`` are
        excluded. Returns the number of leaves refined (after the
        2:1 / green-closability closure).
        """
        indicator = np.asarray(indicator, dtype=np.float64)
        active_to_leaf = np.asarray(active_to_leaf, dtype=np.int64)
        leaf_ids = self.leaves()
        pos = {int(n): i for i, n in enumerate(leaf_ids)}

        # Aggregate the per-active-element indicator onto leaves.
        per_leaf = np.zeros(leaf_ids.shape[0], dtype=np.float64)
        leaf_pos = np.array([pos[int(l)] for l in active_to_leaf],
                            dtype=np.int64)
        np.maximum.at(per_leaf, leaf_pos, indicator)

        # Depth cap: a maxed-out leaf contributes zero, so Dörfler
        # never selects it.
        if self.max_level is not None:
            per_leaf = np.where(
                self.leaf_levels() < self.max_level, per_leaf, 0.0)

        marked_pos = _dorfler_mark(per_leaf, theta)
        if marked_pos.size == 0:
            return 0
        return self.refine(leaf_ids[marked_pos])

    # ----- solution transfer --------------------------------------------

    def _parent_in_old_active(
        self, new_mesh, new_a2l, old_leaves, old_mesh, old_a2l,
    ) -> np.ndarray:
        """Map each new active element to the old active element that
        geometrically contains it.

        After one :meth:`refine` pass a new leaf is either an old leaf
        (unrefined) or a red child of an old leaf, so its old-leaf
        ancestor is at most one ``_parent`` hop away. That old leaf
        owns one or two old active elements (green closure); when two,
        the containing one is picked by the new element's centroid.
        """
        old_a2l = np.asarray(old_a2l, dtype=np.int64)
        leaf_to_old: dict[int, list[int]] = {}
        for ae, leaf in enumerate(old_a2l):
            leaf_to_old.setdefault(int(leaf), []).append(ae)

        old_EToV = np.asarray(old_mesh.EToV, dtype=np.int64)
        old_VX = np.asarray(old_mesh.VX, dtype=np.float64)
        old_VY = np.asarray(old_mesh.VY, dtype=np.float64)
        xc = np.asarray(new_mesh.x, dtype=np.float64).mean(0)
        yc = np.asarray(new_mesh.y, dtype=np.float64).mean(0)

        parent = np.empty(int(new_mesh.K), dtype=np.int64)
        for j in range(int(new_mesh.K)):
            lp = int(new_a2l[j])
            leaf = lp if lp in old_leaves else self._parent[lp]
            cands = leaf_to_old[leaf]
            if len(cands) == 1:
                parent[j] = cands[0]
                continue
            best, best_score = cands[0], -np.inf
            for c in cands:
                v = old_EToV[c]
                score = _point_in_triangle_score(
                    old_VX[v], old_VY[v], xc[j], yc[j])
                if score > best_score:
                    best_score, best = score, c
            parent[j] = best
        return parent

    def adapt(self, mesh_active, active_to_leaf, fields, indicator,
              *, theta: float = 0.5):
        """Refine by an error indicator and transfer fields in one step.

        ``fields`` is a mapping ``name -> (Np, K_active)`` array of
        solution fields living on ``mesh_active``. The forest is
        refined in place via :meth:`refine_by_indicator`; each field is
        then carried onto the new active mesh by :func:`transfer_field`
        (exact polynomial restriction onto the nested children).

        Returns ``(new_mesh, new_fields, new_active_to_leaf,
        n_refined)``. When nothing is refined the inputs are returned
        unchanged, so this is safe to call unconditionally in a loop.
        """
        old_leaves = set(int(i) for i in self.leaves())
        n_refined = self.refine_by_indicator(
            indicator, active_to_leaf, theta=theta)
        if n_refined == 0:
            return (mesh_active, dict(fields),
                    np.asarray(active_to_leaf, dtype=np.int64), 0)
        new_mesh, new_a2l = self.active_mesh()
        parent = self._parent_in_old_active(
            new_mesh, new_a2l, old_leaves, mesh_active, active_to_leaf)
        new_fields = {
            name: transfer_field(mesh_active, f, new_mesh, parent)
            for name, f in fields.items()
        }
        return new_mesh, new_fields, new_a2l, n_refined


def refine_mesh_2d(mesh: Mesh2D, marked_elements) -> RefinementResult:
    """One-shot convenience: refine ``mesh`` at the marked elements.

    Builds a :class:`RefinementForest`, red-refines the marked
    elements (with the standard 2:1-balance / green-closability
    closure), and extracts the conforming active mesh.

    ``marked_elements`` is a 1-D array-like of element indices into
    ``mesh``, or a length-``K`` boolean mask. On a freshly built
    forest the leaf node ids equal the seed element indices, so the
    indices are passed straight through.

    Returns a :class:`RefinementResult`; use ``result.forest`` to
    refine further.
    """
    forest = RefinementForest(mesh)
    marked = np.asarray(marked_elements)
    if marked.dtype == bool:
        if marked.shape[0] != mesh.K:
            raise ValueError(
                f"boolean mask length {marked.shape[0]} != K={mesh.K}"
            )
        marked = np.where(marked)[0]
    n_refined = forest.refine(marked)
    new_mesh, active_to_leaf = forest.active_mesh()
    # RefinementResult.parent reports the level-0 macro-element.
    parent = np.asarray(
        [forest._root_of(int(l)) for l in active_to_leaf],
        dtype=np.int64,
    )
    return RefinementResult(
        mesh=new_mesh, parent=parent, n_refined=n_refined, forest=forest,
    )


# ---------------------------------------------------------------------------
# Refinement strategy: error indicator + marking
# ---------------------------------------------------------------------------
# The strategy is kept separate from the forest machinery: an indicator
# is any function (mesh, field) -> per-element non-negative array, and
# marking is Dörfler bulk-chasing on that array. A learned strategy
# would simply be another producer of the per-element array.


def kelly_error_indicator(
    mesh: Mesh2D, field: jax.Array,
) -> jax.Array:
    """Kelly-type a-posteriori error indicator for a scalar field.

    For each element ``K`` the indicator is the ``h_K``-weighted sum,
    over the element's faces, of the integrated squared jump of the
    field's normal derivative,

        η_K  =  h_K · Σ_{e ⊂ ∂K} ∫_e ⟦∂field/∂n⟧² ds .

    The normal-derivative jump across interior faces is the classic
    Kelly 1983 signal: it is large exactly where the discrete
    solution fails to be smooth, i.e. where the mesh under-resolves.
    Boundary faces contribute nothing (``vmapP = vmapM`` there). The
    field is any scalar ``(Np, K)`` array — a velocity component,
    vorticity, or a transported scalar; the caller chooses what drives
    refinement.

    Returns a non-negative ``(K,)`` array; only the relative ranking
    matters for marking, so the overall constant is unimportant.
    """
    # Physical gradient of the field.
    fr = mesh.Dr @ field
    fs = mesh.Ds @ field
    dfdx = mesh.rx * fr + mesh.sx * fs
    dfdy = mesh.ry * fr + mesh.sy * fs

    dfdx_flat = dfdx.T.reshape(-1)
    dfdy_flat = dfdy.T.reshape(-1)
    nx = mesh.nx.reshape(-1, order="F")
    ny = mesh.ny.reshape(-1, order="F")

    # Normal derivative on the minus / plus face traces.
    dn_m = dfdx_flat[mesh.vmapM] * nx + dfdy_flat[mesh.vmapM] * ny
    dn_p = dfdx_flat[mesh.vmapP] * nx + dfdy_flat[mesh.vmapP] * ny
    jump = dn_m - dn_p                       # zero on boundary faces

    # Quadrature-weighted face integral of the squared jump: weight
    # each face node by the surface Jacobian, sum over the element's
    # Nfp*Nfaces face nodes.
    sJ = mesh.sJ.reshape(-1, order="F")
    contrib = (jump ** 2) * sJ
    per_face_node = contrib.reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F")
    face_sum = jnp.sum(per_face_node, axis=0)            # (K,)

    h_K = jnp.sqrt(2.0 * mesh.J[0])                       # (K,)
    return h_K * face_sum


def _dorfler_mark(indicator: np.ndarray, theta: float) -> np.ndarray:
    """Dörfler (bulk) marking: indices of the smallest set of elements
    whose indicators sum to at least ``theta`` of the total.

    ``theta`` in (0, 1] — larger marks more elements per pass.
    """
    eta = np.asarray(indicator, dtype=np.float64)
    total = float(eta.sum())
    if total <= 0.0:
        return np.array([], dtype=np.int64)
    order = np.argsort(eta)[::-1]                # largest first
    csum = np.cumsum(eta[order])
    k = int(np.searchsorted(csum, theta * total)) + 1
    return np.sort(order[:k])


# ---------------------------------------------------------------------------
# Solution transfer
# ---------------------------------------------------------------------------


def _point_in_triangle_score(vx3, vy3, x: float, y: float) -> float:
    """Smallest barycentric coordinate of ``(x, y)`` in the triangle
    with vertices ``(vx3, vy3)`` — non-negative iff the point is inside.

    Used to decide which of a leaf's two green sub-elements a refined
    child's centroid falls in.
    """
    x0, x1, x2 = float(vx3[0]), float(vx3[1]), float(vx3[2])
    y0, y1, y2 = float(vy3[0]), float(vy3[1]), float(vy3[2])
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    l0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
    l1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / det
    l2 = 1.0 - l0 - l1
    return float(min(l0, l1, l2))


def transfer_field(coarse: Mesh2D, field_coarse: jax.Array,
                    fine: Mesh2D, parent) -> jax.Array:
    """Transfer a scalar field from ``coarse`` onto the refined ``fine``.

    ``parent[j]`` is the index, in ``coarse``, of the element that
    geometrically contains ``fine`` element ``j``. Because every fine
    triangle is nested in its parent, the transfer is *exact*: the
    field on a fine element is the parent's degree-``N`` polynomial
    restricted to it. Each fine node is mapped into the parent's
    reference triangle (barycentric coordinates give ``(r, s)``) and
    the parent's nodal values are evaluated there via the Vandermonde,
    yielding a per-element ``(Np, Np)`` interpolation matrix.

    ``field_coarse`` is ``(Np, K_coarse)``; returns ``(Np, K_fine)``.
    """
    from ddg.dg2d.reference import vandermonde_2d

    p = int(coarse.N)
    EToV_c = np.asarray(coarse.EToV, dtype=np.int64)
    VX_c = np.asarray(coarse.VX, dtype=np.float64)
    VY_c = np.asarray(coarse.VY, dtype=np.float64)
    invV = np.asarray(coarse.invV, dtype=np.float64)
    fc = np.asarray(field_coarse, dtype=np.float64)
    xf = np.asarray(fine.x, dtype=np.float64)
    yf = np.asarray(fine.y, dtype=np.float64)
    Np, Kf = xf.shape
    parent = np.asarray(parent, dtype=np.int64)

    out = np.zeros((Np, Kf), dtype=np.float64)
    for j in range(Kf):
        c = int(parent[j])
        v0, v1, v2 = EToV_c[c]
        p0x, p0y = VX_c[v0], VY_c[v0]
        # Solve [x - p0] = lam1 (p1 - p0) + lam2 (p2 - p0) at every node.
        A = np.array([[VX_c[v1] - p0x, VX_c[v2] - p0x],
                      [VY_c[v1] - p0y, VY_c[v2] - p0y]])
        rhs = np.stack([xf[:, j] - p0x, yf[:, j] - p0y])      # (2, Np)
        lam = np.linalg.solve(A, rhs)                          # (2, Np)
        r = 2.0 * lam[0] - 1.0
        s = 2.0 * lam[1] - 1.0
        Veval = np.asarray(
            vandermonde_2d(p, jnp.asarray(r), jnp.asarray(s)))
        interp = Veval @ invV                                  # (Np, Np)
        out[:, j] = interp @ fc[:, c]
    return jnp.asarray(out)
