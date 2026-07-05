"""Adaptive mesh refinement machinery — conforming red-green h-refinement.

Checks the Phase-1 machinery:
  1. Marking every element quadruples the count and lifts all leaves
     to level 1 (pure red, no green needed).
  2. Refinement conserves total area — children tile the parent with
     no gaps or overlaps.
  3. The active mesh is conforming — its only count-1 edges are the
     true domain boundary (a hanging node would leave count-1 edges
     stranded in the interior).
  4. Every active element is CCW (build_mesh_2d gives J > 0).
  5. An empty mark set is the identity.
  6. The child -> macro-element parent map is well-formed.
  7. Multi-level refinement keeps the 2:1 balance (face-adjacent
     leaves differ by at most one level) and stays conforming.
  8. The refined mesh is solver-valid — a SIPG Poisson matvec runs.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.refinement import RefinementForest, _edge, refine_mesh_2d


def _base_mesh(N=2, K=4):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, K, K)
    return build_mesh_2d(VX, VY, EToV, N)


def _total_area(mesh):
    """Sum of element areas from the corner vertices of each element."""
    VX = np.asarray(mesh.VX); VY = np.asarray(mesh.VY)
    EToV = np.asarray(mesh.EToV)
    a = 0.0
    for v0, v1, v2 in EToV:
        a += 0.5 * abs((VX[v1] - VX[v0]) * (VY[v2] - VY[v0])
                       - (VX[v2] - VX[v0]) * (VY[v1] - VY[v0]))
    return a


def _count1_edge_length(mesh):
    """Total length of edges that belong to exactly one element.

    For a conforming mesh that is exactly the domain boundary; a
    hanging node strands extra count-1 edges in the interior.
    """
    VX = np.asarray(mesh.VX); VY = np.asarray(mesh.VY)
    EToV = np.asarray(mesh.EToV)
    counter: Counter = Counter()
    for v0, v1, v2 in EToV:
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            counter[_edge(int(a), int(b))] += 1
    total = 0.0
    for (a, b), n in counter.items():
        assert n <= 2, f"edge {(a, b)} shared by {n} elements"
        if n == 1:
            total += float(np.hypot(VX[a] - VX[b], VY[a] - VY[b]))
    return total


def test_uniform_refinement_quadruples():
    """Marking every element: pure red split, K -> 4K, all level 1."""
    mesh = _base_mesh(N=2, K=4)
    res = refine_mesh_2d(mesh, np.arange(mesh.K))
    print(f"  K {mesh.K} -> {res.mesh.K}")
    assert res.mesh.K == 4 * mesh.K
    levels = res.forest.leaf_levels()
    assert np.all(levels == 1)
    assert res.n_refined == mesh.K


def test_refinement_conserves_area():
    """Children tile their parent exactly."""
    mesh = _base_mesh(N=3, K=6)
    a0 = _total_area(mesh)
    res = refine_mesh_2d(mesh, [0, 7, 18, 25])
    a1 = _total_area(res.mesh)
    print(f"  area {a0:.6f} -> {a1:.6f}")
    assert abs(a1 - a0) < 1e-12
    assert abs(a1 - 1.0) < 1e-12


def test_active_mesh_is_conforming():
    """The only count-1 edges are the domain boundary (perimeter 4)."""
    mesh = _base_mesh(N=2, K=5)
    res = refine_mesh_2d(mesh, [0, 3, 12, 27, 40])
    bdy = _count1_edge_length(res.mesh)
    print(f"  count-1 edge length = {bdy:.6f}  (perimeter = 4)")
    assert abs(bdy - 4.0) < 1e-9


def test_all_elements_ccw():
    """Every active element is CCW => build_mesh_2d gives J > 0."""
    mesh = _base_mesh(N=3, K=4)
    res = refine_mesh_2d(mesh, [1, 5, 9, 14, 22])
    J = np.asarray(res.mesh.J)
    print(f"  J range [{J.min():.3e}, {J.max():.3e}]")
    assert np.all(J > 0.0)


def test_empty_marks_is_identity():
    """No marks -> same element count."""
    mesh = _base_mesh(N=2, K=4)
    res = refine_mesh_2d(mesh, [])
    print(f"  K {mesh.K} -> {res.mesh.K}")
    assert res.mesh.K == mesh.K
    assert res.n_refined == 0


def test_parent_map_well_formed():
    """parent has one entry per active element, all in range."""
    mesh = _base_mesh(N=2, K=5)
    res = refine_mesh_2d(mesh, [11, 12, 13])
    assert res.parent.shape[0] == res.mesh.K
    assert res.parent.min() >= 0 and res.parent.max() < mesh.K
    print(f"  parent map: {res.mesh.K} entries, "
          f"range [{res.parent.min()}, {res.parent.max()}]")


def test_multilevel_keeps_2to1_balance():
    """Refine, then refine inside the fine region — face-adjacent
    leaves must stay within one refinement level, and the mesh stays
    conforming.

    Adjacency must bridge the hanging-node interface: a level-L leaf
    owns edge ``(a, b)`` while the level-(L+1) leaves across it own
    the *halves* ``(a, m)`` and ``(m, b)``, so same-edge-key adjacency
    alone cannot see the cross-level contact.
    """
    mesh = _base_mesh(N=2, K=6)
    forest = RefinementForest(mesh)
    forest.refine(np.arange(mesh.K))            # uniform -> all level 1
    forest.refine(forest.leaves()[:6])          # refine a clutch
    leaf_ids = [int(i) for i in forest.leaves()]
    level = {i: int(forest._level[i]) for i in leaf_ids}
    levels_seen = sorted(set(level.values()))
    print(f"  leaf levels present: {levels_seen}")
    assert levels_seen == [1, 2]                # genuinely multi-level

    edge_owner: dict = {}
    for i in leaf_ids:
        for e in forest._node_edges(i):
            edge_owner.setdefault(e, []).append(i)

    max_gap = 0
    for i in leaf_ids:
        for e in forest._node_edges(i):
            # same-edge neighbour (equal-level contact)
            for j in edge_owner.get(e, ()):
                if j != i:
                    max_gap = max(max_gap, abs(level[i] - level[j]))
            # cross-level: a finer neighbour owns a half of this edge
            m = forest._edge_mid.get(e)
            if m is not None:
                a, b = e
                for half in (_edge(a, m), _edge(m, b)):
                    for j in edge_owner.get(half, ()):
                        max_gap = max(max_gap, abs(level[i] - level[j]))
    print(f"  max face-adjacent level gap (incl. cross-level) = {max_gap}")
    assert max_gap == 1                         # 2:1 balance, genuinely tested

    active, _ = forest.active_mesh()
    bdy = _count1_edge_length(active)
    print(f"  multilevel conforming: count-1 length = {bdy:.6f}")
    assert abs(bdy - 4.0) < 1e-9
    assert np.all(np.asarray(active.J) > 0.0)


def test_refined_mesh_is_solver_valid():
    """A SIPG Poisson operator applies on the refined mesh."""
    from ddg.dg2d.sipg import BC_DIRICHLET, BC_INTERIOR, sipg_apply_2d
    mesh = _base_mesh(N=2, K=4)
    res = refine_mesh_2d(mesh, [0, 6, 17])
    rm = res.mesh
    EToE = np.asarray(rm.EToE)
    is_bdy = EToE == np.arange(rm.K)[:, None]
    bc = jnp.asarray(np.where(is_bdy, BC_DIRICHLET, BC_INTERIOR).astype(np.int32))
    u = jnp.asarray(np.random.default_rng(0).standard_normal((rm.Np, rm.K)))
    out = sipg_apply_2d(rm, u, bc_types=bc, u_dir=None)
    assert out.shape == (rm.Np, rm.K)
    assert bool(jnp.all(jnp.isfinite(out)))
    print(f"  SIPG apply on refined mesh OK: {rm.K} elements")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("all ok")
