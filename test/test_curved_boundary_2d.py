"""Tests for the optional curved-boundary toggle.

We build a base axis-aligned mesh with a square hole, then project the
inner boundary onto a circle using :func:`apply_circular_hole_2d`.
Verify that:

  1. The projection moves boundary nodes onto the circle (radii equal
     the target ``r``).
  2. The mesh stays valid (positive Jacobian everywhere) at a mesh
     resolution where the corner-projection offset is comfortably
     smaller than the adjacent cell width.
  3. The SIPG Poisson solve on the curved mesh still converges to a
     manufactured solution with reasonable L2 error.

The toggle is opt-in: the default ``build_mesh_2d`` path stays affine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, rectangle_with_square_hole_mesh_2d
from ddg.dg2d.curved import apply_circular_hole_2d
from ddg.dg2d.sipg import (
    BC_DIRICHLET, BC_INTERIOR, solve_sipg_poisson_2d,
)


def _bc_all_dirichlet(mesh):
    Nfaces, K = mesh.Nfaces, mesh.K
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T  # (Nfaces, K)
    bc = np.full((Nfaces, K), BC_INTERIOR, dtype=np.int32)
    bc = np.where(is_bdy, BC_DIRICHLET, bc)
    return jnp.asarray(bc.T)


def _build_annulus_mesh(N, Kx, Ky, *, curved: bool):
    """Square-hole mesh in [-2,2]², hole [-0.5, 0.5]². If ``curved``, project
    the inner boundary onto a circle of radius √2·0.5/√2 = 0.5 (the
    square's inscribed circle, so we just use radius 0.5)."""
    VX, VY, EToV = rectangle_with_square_hole_mesh_2d(
        -2.0, 2.0, -2.0, 2.0, Kx, Ky,
        -0.5, 0.5, -0.5, 0.5,
    )
    mesh = build_mesh_2d(VX, VY, EToV, N)
    if curved:
        # Project the inner square (max |xy| = 0.5) onto a circle of radius 0.5.
        # The square's corners sit at √2·0.5 ≈ 0.707 from origin, so the
        # "detection_tol" needs to admit those midpoint distances.
        mesh = apply_circular_hole_2d(
            mesh, cx=0.0, cy=0.0, radius=0.5, detection_tol=0.5,
        )
    return mesh


def test_curved_mesh_is_distinct_from_affine():
    """Sanity: the curve projection actually moves nodes onto the circle."""
    mesh_aff = _build_annulus_mesh(N=3, Kx=24, Ky=24, curved=False)
    mesh_cur = _build_annulus_mesh(N=3, Kx=24, Ky=24, curved=True)
    diff_x = float(jnp.max(jnp.abs(mesh_cur.x - mesh_aff.x)))
    print(f"  max |Δx| = {diff_x:.4f}")
    assert diff_x > 1e-3

    x_b = np.asarray(mesh_cur.x).T.reshape(-1)[mesh_cur.vmapB]
    y_b = np.asarray(mesh_cur.y).T.reshape(-1)[mesh_cur.vmapB]
    r_b = np.sqrt(x_b ** 2 + y_b ** 2)
    inner_r = r_b[r_b < 1.0]  # inner-boundary nodes only
    print(f"  inner-boundary radii: min={inner_r.min():.4f}, "
          f"max={inner_r.max():.4f}")
    assert np.allclose(inner_r, 0.5, atol=1e-10)


def test_curved_mesh_has_positive_jacobian():
    """At a mesh resolution where the corner offset is small compared to
    the adjacent cell width, the curved mesh stays valid (positive J).

    For ``[-2, 2]²`` with ``Kx = Ky = 24``, ``h = 4/24 ≈ 0.167``. The
    square-corner-to-circle offset is ``(√2 − 1)·0.5 ≈ 0.207`` for a
    side-length-1 square inscribed in a radius-0.707 circle, but here
    the square is the ``side length 1`` hole and we project to radius
    0.5 — so the corner moves inward by ``√2·0.5 − 0.5 ≈ 0.207``.
    The blending damps that to a fraction of the adjacent cell, keeping
    J positive.
    """
    mesh = _build_annulus_mesh(N=3, Kx=24, Ky=24, curved=True)
    Jmin = float(jnp.min(mesh.J))
    Jmax = float(jnp.max(mesh.J))
    print(f"  curved mesh: min(J) = {Jmin:.3e}, max(J) = {Jmax:.3e}")
    assert Jmin > 0.0, f"mesh inverted: min(J) = {Jmin}"


def test_curved_poisson_solve_well_behaved():
    """Solve −Δu = f on the rectangle-minus-disk with curved inner
    boundary. We use a manufactured solution and check L2 error is
    finite + reasonable. We don't compare against an affine variant —
    they discretise different domains, so a head-to-head L2 number is
    misleading.
    """
    mesh = _build_annulus_mesh(N=3, Kx=24, Ky=24, curved=True)
    pi = jnp.pi
    u_exact = jnp.sin(pi * mesh.x / 4) * jnp.sin(pi * mesh.y / 4)
    source = 2 * (pi / 4) ** 2 * u_exact
    bc = _bc_all_dirichlet(mesh)
    u = solve_sipg_poisson_2d(mesh, source, bc_types=bc, u_dir=u_exact)
    err_l2 = float(jnp.sqrt(
        jnp.sum(mesh.J * (u - u_exact) ** 2)
        / jnp.sum(mesh.J * u_exact ** 2)
    ))
    print(f"  curved Poisson: rel L2 err = {err_l2:.3e}")
    assert err_l2 < 2e-2


if __name__ == "__main__":
    test_curved_mesh_is_distinct_from_affine()
    test_curved_mesh_has_positive_jacobian()
    test_curved_poisson_solve_well_behaved()
