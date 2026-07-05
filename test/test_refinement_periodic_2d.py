"""Periodic-consistent AMR — opposite boundaries refine in lockstep.

On a periodic domain the refinement forest must keep periodically
identified boundaries refined identically, or
``apply_rectangle_periodic_2d`` cannot pair their faces (it requires
the two edges of each periodic pair to carry the same number of
faces). These tests verify the periodic co-marking closure:

  1. on an x-periodic mesh, refining a scattered subset still yields
     an active mesh that accepts x-periodic pairing;
  2. refining a single left-boundary leaf pulls in its right-boundary
     partner (lockstep);
  3. a doubly-periodic mesh stays consistent on both axes;
  4. indicator-driven refinement keeps the boundaries consistent.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import (
    apply_rectangle_periodic_2d, build_mesh_2d, uniform_rectangle_mesh_2d,
)
from ddg.dg2d.refinement import RefinementForest, kelly_error_indicator


def _mesh(K=6, N=2):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, K, K)
    return build_mesh_2d(VX, VY, EToV, N)


def _edge_on(forest, node, axis, coord, tol=1e-7):
    """True if leaf `node` owns an edge along axis=coord."""
    pick = forest._vx if axis == "x" else forest._vy
    for a, b in forest._node_edges(int(node)):
        if abs(pick[a] - coord) < tol and abs(pick[b] - coord) < tol:
            return True
    return False


# 1 -----------------------------------------------------------------------

def test_x_periodic_active_mesh_pairs():
    """An x-periodic forest yields a mesh that accepts x-periodic pairing."""
    forest = RefinementForest(_mesh(K=6), periodic={"x": (0.0, 1.0)})
    leaves = forest.leaves()
    forest.refine(leaves[[0, 7, 15, 22, 30]])
    amesh, _ = forest.active_mesh()
    per = apply_rectangle_periodic_2d(
        amesh, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
        periodic_x=True, periodic_y=False)
    print(f"  x-periodic: K {forest.leaves().size} leaves -> "
          f"mesh K={amesh.K}, periodic pairing OK")
    assert per.K == amesh.K


# 2 -----------------------------------------------------------------------

def test_left_leaf_refinement_pulls_right_partner():
    """Refining one left-boundary leaf co-marks its right partner."""
    forest = RefinementForest(_mesh(K=6), periodic={"x": (0.0, 1.0)})
    left = next(int(i) for i in forest.leaves()
                if _edge_on(forest, i, "x", 0.0))
    n = forest.refine([left])
    print(f"  refined left leaf {left} -> {n} leaves refined after closure")
    assert n >= 2                                   # left + right partner
    # some refined leaf now sits on the right boundary
    right_refined = any(
        lvl >= 1 and _edge_on(forest, node, "x", 1.0)
        for node, lvl in zip(forest.leaves(), forest.leaf_levels()))
    assert right_refined
    # and the extracted mesh still pairs
    amesh, _ = forest.active_mesh()
    apply_rectangle_periodic_2d(
        amesh, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
        periodic_x=True, periodic_y=False)


# 3 -----------------------------------------------------------------------

def test_doubly_periodic_active_mesh_pairs():
    """A doubly-periodic forest stays consistent on both axes."""
    forest = RefinementForest(
        _mesh(K=6), periodic={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    leaves = forest.leaves()
    forest.refine(leaves[[1, 9, 14, 28, 33]])
    amesh, _ = forest.active_mesh()
    per = apply_rectangle_periodic_2d(
        amesh, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
        periodic_x=True, periodic_y=True)
    print(f"  doubly-periodic: mesh K={amesh.K}, both axes pair OK")
    assert per.K == amesh.K


# 4 -----------------------------------------------------------------------

def test_indicator_driven_keeps_periodic_consistency():
    """Iterative indicator-driven refinement preserves periodic pairing."""
    forest = RefinementForest(
        _mesh(K=8, N=3), periodic={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        max_level=3)
    for it in range(3):
        mesh, a2l = forest.active_mesh()
        # a feature crossing the periodic boundary at x = 0 / x = 1
        field = jnp.tanh((np.asarray(mesh.x) - 0.5) / 0.05)
        eta = kelly_error_indicator(mesh, field)
        n = forest.refine_by_indicator(eta, a2l, theta=0.5)
        amesh, _ = forest.active_mesh()
        per = apply_rectangle_periodic_2d(
            amesh, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
            periodic_x=True, periodic_y=True)
        print(f"  round {it}: refined {n}, mesh K={amesh.K}, pairing OK")
        assert per.K == amesh.K


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("all ok")
