"""AMR refinement strategy — Kelly error indicator + Dörfler marking.

Phase 2 of the adaptive-mesh-refinement work: a *strategy* that turns
a discrete field into a set of marked elements. The forest machinery
(Phase 1) is exercised in test_refinement_2d.py; here we verify the
indicator and the marking, on progressively harder fields:

  1. the Dörfler marking picks the right bulk;
  2. the Kelly indicator localises an axis-aligned tanh layer;
  3. one indicator-driven refinement step concentrates the mesh on
     that layer and leaves the smooth region untouched;
  4. iterative refinement honours the hard depth cap exactly;
  5. on a harder, curved (ring) feature, iterative refinement tracks
     the feature and drives the worst-element indicator down.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.refinement import (
    RefinementForest, _dorfler_mark, kelly_error_indicator,
)


def _uniform(N=3, K=8):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, K, K)
    return build_mesh_2d(VX, VY, EToV, N)


def _tanh_layer(mesh, y0=0.5, delta=0.05):
    """A sharp horizontal shear-layer-like field."""
    return jnp.tanh((np.asarray(mesh.y) - y0) / delta)


def _ring(mesh, r0=0.3, delta=0.05, cx=0.5, cy=0.5):
    """A sharp circular-ring feature — not axis-aligned."""
    r = np.sqrt((np.asarray(mesh.x) - cx) ** 2
                + (np.asarray(mesh.y) - cy) ** 2)
    return jnp.tanh((r - r0) / delta)


def _centroids(mesh):
    return (np.asarray(mesh.x).mean(0), np.asarray(mesh.y).mean(0))


# 1 -----------------------------------------------------------------------

def test_dorfler_marking_picks_the_bulk():
    """Dörfler marks the smallest set carrying >= theta of the total."""
    eta = np.array([1.0, 1.0, 1.0, 1.0, 6.0])     # total 10
    m_half = _dorfler_mark(eta, theta=0.5)         # need >= 5
    print(f"  theta=0.5 -> {m_half}")
    assert list(m_half) == [4]                      # the lone big one
    m_big = _dorfler_mark(eta, theta=0.8)          # need >= 8
    print(f"  theta=0.8 -> {m_big}")
    assert len(m_big) == 3 and 4 in m_big
    assert _dorfler_mark(np.zeros(5), theta=0.5).size == 0


# 2 -----------------------------------------------------------------------

def test_kelly_indicator_localises_a_layer():
    """The Kelly indicator is large at a tanh layer, ~zero far from it."""
    mesh = _uniform(N=3, K=8)
    eta = np.asarray(kelly_error_indicator(mesh, _tanh_layer(mesh)))
    _, yc = _centroids(mesh)
    near = eta[np.abs(yc - 0.5) < 0.13]
    far = eta[np.abs(yc - 0.5) > 0.35]
    print(f"  max eta near layer = {near.max():.3e}, "
          f"far = {far.max():.3e}")
    assert near.max() > 0.0
    assert far.max() < 1e-6 * near.max()           # smooth region: flat
    # the single worst element straddles the layer
    yc_worst = yc[int(np.argmax(eta))]
    assert abs(yc_worst - 0.5) < 0.13


# 3 -----------------------------------------------------------------------

def test_one_step_concentrates_on_the_feature():
    """refine_by_indicator refines the layer, not the smooth bulk."""
    mesh = _uniform(N=2, K=8)
    forest = RefinementForest(mesh)
    mesh_a, a2l = forest.active_mesh()
    eta = kelly_error_indicator(mesh_a, _tanh_layer(mesh_a))
    n = forest.refine_by_indicator(eta, a2l, theta=0.4)
    print(f"  refined {n} leaves")
    assert n > 0

    levels = forest.leaf_levels()
    # centroid of every leaf, from the forest's own vertex table
    vx = np.asarray(forest._vx); vy = np.asarray(forest._vy)
    refined_y, untouched_y = [], []
    for node, lvl in zip(forest.leaves(), levels):
        v = forest._verts[int(node)]
        yc = float(np.mean([vy[v[0]], vy[v[1]], vy[v[2]]]))
        (refined_y if lvl >= 1 else untouched_y).append(yc)
    refined_y = np.array(refined_y)
    print(f"  refined leaves: {refined_y.size}, "
          f"max |y_c - 0.5| = {np.abs(refined_y - 0.5).max():.3f}")
    # every refined leaf sits near the layer
    assert np.all(np.abs(refined_y - 0.5) < 0.30)
    # the smooth far-field is left at level 0
    assert np.any(np.abs(np.array(untouched_y) - 0.5) > 0.35)


# 4 -----------------------------------------------------------------------

def test_iterative_refinement_honours_depth_cap():
    """With max_level set, iterative refinement never exceeds it."""
    max_level = 2
    forest = RefinementForest(_uniform(N=2, K=8), max_level=max_level)
    for it in range(6):
        mesh_a, a2l = forest.active_mesh()
        eta = kelly_error_indicator(mesh_a, _tanh_layer(mesh_a))
        forest.refine_by_indicator(eta, a2l, theta=0.5)
        top = int(forest.leaf_levels().max())
        print(f"  iter {it}: max leaf level = {top}")
        assert top <= max_level
    assert int(forest.leaf_levels().max()) == max_level   # cap reached


# 5 -----------------------------------------------------------------------

def test_curved_feature_iterative_refinement():
    """A harder, curved (ring) feature: iterative refinement tracks the
    ring and drives the worst-element indicator down."""
    forest = RefinementForest(_uniform(N=3, K=8), max_level=4)
    eta_max = []
    for it in range(4):
        mesh_a, a2l = forest.active_mesh()
        eta = np.asarray(kelly_error_indicator(mesh_a, _ring(mesh_a)))
        eta_max.append(float(eta.max()))
        forest.refine_by_indicator(eta, a2l, theta=0.5)
    print(f"  worst-element indicator by round: "
          f"{[f'{e:.2e}' for e in eta_max]}")
    # refinement resolves the feature -> the worst indicator falls
    assert eta_max[-1] < 0.5 * eta_max[0]

    # every refined leaf sits on the ring r ~ 0.3
    vx = np.asarray(forest._vx); vy = np.asarray(forest._vy)
    off_ring = 0
    for node, lvl in zip(forest.leaves(), forest.leaf_levels()):
        if lvl == 0:
            continue
        v = forest._verts[int(node)]
        xc = float(np.mean([vx[v[0]], vx[v[1]], vx[v[2]]]))
        yc = float(np.mean([vy[v[0]], vy[v[1]], vy[v[2]]]))
        r = np.hypot(xc - 0.5, yc - 0.5)
        if not (0.10 < r < 0.50):
            off_ring += 1
    print(f"  refined leaves off the ring band: {off_ring}")
    assert off_ring == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("all ok")
