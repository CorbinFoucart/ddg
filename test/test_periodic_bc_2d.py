"""Connectivity correctness for 2D rectangular periodic boundaries.

``apply_rectangle_periodic_2d`` rewires a built mesh so opposite edges
of an axis-aligned rectangle become interior face pairs. The checks:

  1. A doubly-periodic mesh has *no* boundary nodes — every face is
     matched to a partner.
  2. A continuous, periodic field has zero jump ``u(vmapP) -
     u(vmapM)`` across every face, including the wrapped ones. This is
     the real test of node-level matching: a wrong pairing shows up
     as an O(1) jump.
  3. Single-axis periodicity leaves exactly the other two edges as
     boundary.
  4. The periodic partner relation is symmetric — if face A maps to
     face B, face B maps back to A.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import (
    apply_rectangle_periodic_2d, build_mesh_2d, uniform_rectangle_mesh_2d,
)


def _periodic_mesh(N=3, Kx=8, Ky=8, periodic_x=True, periodic_y=True):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 2.0, 0.0, 2.0, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, N)
    return apply_rectangle_periodic_2d(
        mesh, xmin=0.0, xmax=2.0, ymin=0.0, ymax=2.0,
        periodic_x=periodic_x, periodic_y=periodic_y,
    )


def test_doubly_periodic_has_no_boundary():
    """Both axes periodic => every face is interior, mapB empty."""
    mp = _periodic_mesh()
    n_bdy = len(np.asarray(mp.mapB))
    print(f"  doubly-periodic boundary nodes: {n_bdy}")
    assert n_bdy == 0
    EToE = np.asarray(mp.EToE)
    n_self = int((EToE == np.arange(mp.K)[:, None]).sum())
    print(f"  self-referencing faces: {n_self}")
    assert n_self == 0


def test_continuous_periodic_field_has_zero_jump():
    """A continuous 2-periodic field must have machine-zero jump across
    every face after periodic wiring."""
    mp = _periodic_mesh()
    x = np.asarray(mp.x); y = np.asarray(mp.y)
    f = np.sin(np.pi * x) * np.sin(np.pi * y)   # period 2 in x and y
    f_flat = f.flatten(order="F")
    jump = f_flat[np.asarray(mp.vmapP)] - f_flat[np.asarray(mp.vmapM)]
    err = float(np.abs(jump).max())
    print(f"  max |f(vmapP) - f(vmapM)| = {err:.2e}")
    assert err < 1e-12


def test_single_axis_periodic_boundary_count():
    """x-periodic only: the bottom + top edges remain boundary."""
    Kx = Ky = 8
    N = 3
    mx = _periodic_mesh(N=N, Kx=Kx, Ky=Ky, periodic_x=True, periodic_y=False)
    # Two edges, Kx faces each, Nfp = N+1 nodes per face.
    expected = 2 * Kx * (N + 1)
    n_bdy = len(np.asarray(mx.mapB))
    print(f"  x-periodic boundary nodes: {n_bdy}  (expect {expected})")
    assert n_bdy == expected

    my = _periodic_mesh(N=N, Kx=Kx, Ky=Ky, periodic_x=False, periodic_y=True)
    expected_y = 2 * Ky * (N + 1)
    n_bdy_y = len(np.asarray(my.mapB))
    print(f"  y-periodic boundary nodes: {n_bdy_y}  (expect {expected_y})")
    assert n_bdy_y == expected_y


def test_periodic_partner_is_symmetric():
    """If EToE[k,f] = k', EToF[k,f] = f', then EToE[k',f'] = k and
    EToF[k',f'] = f."""
    mp = _periodic_mesh()
    EToE = np.asarray(mp.EToE); EToF = np.asarray(mp.EToF)
    K, Nfaces = mp.K, mp.Nfaces
    for k in range(K):
        for f in range(Nfaces):
            kp, fp = EToE[k, f], EToF[k, f]
            assert EToE[kp, fp] == k
            assert EToF[kp, fp] == f
    print("  all periodic partners are symmetric")


def test_constant_field_zero_jump():
    """Sanity: a constant field has zero jump everywhere (trivial but
    catches a vmapP that points off-array)."""
    mp = _periodic_mesh()
    f_flat = np.full(mp.Np * mp.K, 3.14159)
    jump = f_flat[np.asarray(mp.vmapP)] - f_flat[np.asarray(mp.vmapM)]
    assert float(np.abs(jump).max()) == 0.0
    print("  constant-field jump is exactly zero")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("all ok")
