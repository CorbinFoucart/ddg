"""AMR solution transfer — Phase 2, the field-carrying piece.

The forest machinery (Phase 1, test_refinement_2d.py) and the
indicator/marking strategy (Phase 2, test_refinement_strategy_2d.py)
turn a field into a refined mesh. This file verifies the remaining
piece: carrying the *solution* onto that refined mesh.

  1. transfer of a degree-<=p polynomial is exact under uniform
     refinement (machine precision — every child re-represents the
     same polynomial);
  2. ... and under adaptive (partial) refinement, where green-closure
     transition cells are also exercised;
  3. RefinementForest.adapt (refine-by-indicator + transfer) preserves
     the integral of a smooth field — transfer adds no error.

Verification of AMR inside an actual flow solve is done separately,
against the existing uniform-mesh benchmark applications, so that no
new numerics (time integrator, flux) are entangled with the AMR test.
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
    RefinementForest, kelly_error_indicator, transfer_field,
)


def _uniform(N=3, K=6):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, K, K)
    return build_mesh_2d(VX, VY, EToV, N)


def _poly(mesh, p):
    """A generic degree-p polynomial sampled on the mesh nodes."""
    x = np.asarray(mesh.x)
    y = np.asarray(mesh.y)
    f = np.zeros_like(x)
    rng = np.random.default_rng(0)
    for i in range(p + 1):
        for j in range(p + 1 - i):
            f += rng.uniform(-1.0, 1.0) * (x ** i) * (y ** j)
    return jnp.asarray(f)


def _blob(mesh, cx, cy, sig):
    x = np.asarray(mesh.x)
    y = np.asarray(mesh.y)
    return jnp.asarray(np.exp(-((x - cx) ** 2 + (y - cy) ** 2)
                              / (2.0 * sig ** 2)))


def _mass_ref(mesh):
    """Reference-element mass matrix M = (V V^T)^{-1}."""
    V = np.asarray(mesh.V)
    return np.linalg.inv(V @ V.T)


def _integral(mesh, field):
    """Domain integral of a scalar field (straight elements)."""
    M = _mass_ref(mesh)
    f = np.asarray(field)
    J = np.asarray(mesh.J)[0]                          # const per elem
    return float(np.sum(J * (np.ones(f.shape[0]) @ (M @ f))))


# 1 -----------------------------------------------------------------------

def test_transfer_polynomial_exact_uniform():
    """A degree-<=p polynomial transfers exactly under uniform refine."""
    p = 3
    forest = RefinementForest(_uniform(N=p, K=4))
    old_mesh, old_a2l = forest.active_mesh()
    field = _poly(old_mesh, p)

    old_leaves = set(int(i) for i in forest.leaves())
    forest.refine(forest.leaves())                     # refine all
    new_mesh, new_a2l = forest.active_mesh()
    parent = forest._parent_in_old_active(
        new_mesh, new_a2l, old_leaves, old_mesh, old_a2l)

    transferred = transfer_field(old_mesh, field, new_mesh, parent)
    expected = _poly_on(new_mesh, p, old_mesh)
    err = float(np.max(np.abs(np.asarray(transferred) - expected)))
    print(f"  uniform-refine transfer error = {err:.2e}")
    assert err < 1e-10


def _poly_on(mesh, p, ref_mesh):
    """Re-evaluate the same _poly coefficients on another mesh."""
    x = np.asarray(mesh.x)
    y = np.asarray(mesh.y)
    f = np.zeros_like(x)
    rng = np.random.default_rng(0)
    for i in range(p + 1):
        for j in range(p + 1 - i):
            f += rng.uniform(-1.0, 1.0) * (x ** i) * (y ** j)
    return f


# 2 -----------------------------------------------------------------------

def test_transfer_polynomial_exact_adaptive():
    """Exact transfer under partial refinement (green cells included)."""
    p = 3
    forest = RefinementForest(_uniform(N=p, K=6))
    old_mesh, old_a2l = forest.active_mesh()
    field = _poly(old_mesh, p)

    old_leaves = set(int(i) for i in forest.leaves())
    # refine a scattered subset -> green closure around it
    leaves = forest.leaves()
    forest.refine(leaves[[3, 10, 11, 22, 30]])
    new_mesh, new_a2l = forest.active_mesh()
    parent = forest._parent_in_old_active(
        new_mesh, new_a2l, old_leaves, old_mesh, old_a2l)

    transferred = transfer_field(old_mesh, field, new_mesh, parent)
    expected = _poly_on(new_mesh, p, old_mesh)
    err = float(np.max(np.abs(np.asarray(transferred) - expected)))
    print(f"  adaptive-refine transfer error = {err:.2e} "
          f"(K {old_mesh.K} -> {new_mesh.K})")
    assert err < 1e-10


# 3 -----------------------------------------------------------------------

def test_adapt_preserves_the_integral():
    """forest.adapt (refine-by-indicator + transfer) preserves the
    integral of a smooth field — transfer introduces no error."""
    forest = RefinementForest(_uniform(N=3, K=8), max_level=3)
    mesh, a2l = forest.active_mesh()
    field = _blob(mesh, 0.5, 0.5, 0.12)
    I0 = _integral(mesh, field)

    for it in range(3):
        eta = kelly_error_indicator(mesh, field)
        mesh, fields, a2l, n = forest.adapt(
            mesh, a2l, {"c": field}, eta, theta=0.6)
        field = fields["c"]
        I = _integral(mesh, field)
        print(f"  round {it}: refined {n}, K={mesh.K}, "
              f"integral drift = {abs(I - I0):.2e}")
        assert n > 0
        assert abs(I - I0) < 1e-9 * abs(I0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("all ok")
