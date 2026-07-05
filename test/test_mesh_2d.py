"""2D Mesh2D builder + connectivity tests."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg2d.mesh import (
    build_mesh_2d,
    uniform_rectangle_mesh_2d,
    update_mesh_2d_vertices,
)


def test_uniform_rectangle_mesh_shapes():
    Kx, Ky = 3, 2
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 3.0, 0.0, 2.0, Kx, Ky)
    assert VX.shape == ((Kx + 1) * (Ky + 1),)
    assert VY.shape == ((Kx + 1) * (Ky + 1),)
    assert EToV.shape == (2 * Kx * Ky, 3)


def test_mesh_basic_shapes():
    N, Kx, Ky = 4, 3, 3
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, N=N)
    Np = (N + 1) * (N + 2) // 2
    assert mesh.Np == Np
    assert mesh.K == 2 * Kx * Ky
    assert mesh.Nfaces == 3
    assert mesh.Nfp == N + 1
    assert mesh.x.shape == (Np, mesh.K)
    assert mesh.J.shape == (Np, mesh.K)
    assert mesh.nx.shape == (3 * (N + 1), mesh.K)


def test_jacobian_equals_physical_over_reference_area():
    """For an axis-aligned right triangle of area A: J = A / 2 (reference area is 2)."""
    Kx, Ky = 4, 4
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, N=3)
    expected_J = (1.0 / Kx) * (1.0 / Ky) / 2.0 / 2.0
    np.testing.assert_allclose(np.asarray(mesh.J), expected_J, atol=1e-12)


def test_internal_face_matching_machine_precision():
    """Across every internal face, x[vmapM] == x[vmapP] and y[...] match."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(-1.0, 1.0, -1.0, 1.0, 3, 3)
    mesh = build_mesh_2d(VX, VY, EToV, N=4)
    x_flat = np.asarray(mesh.x).flatten(order="F")
    y_flat = np.asarray(mesh.y).flatten(order="F")
    dx = x_flat[np.asarray(mesh.vmapM)] - x_flat[np.asarray(mesh.vmapP)]
    dy = y_flat[np.asarray(mesh.vmapM)] - y_flat[np.asarray(mesh.vmapP)]
    internal = ~np.isin(np.arange(len(mesh.vmapM)), np.asarray(mesh.mapB))
    assert float(np.max(np.sqrt(dx[internal] ** 2 + dy[internal] ** 2))) < 1e-13


def test_unit_normals():
    """Surface normals have unit magnitude."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, 4, 4)
    mesh = build_mesh_2d(VX, VY, EToV, N=4)
    mag = jnp.sqrt(mesh.nx**2 + mesh.ny**2)
    assert float(jnp.max(jnp.abs(mag - 1.0))) < 1e-13


def test_connectivity_is_symmetric():
    """If element a's face f points to (b, g), then b's face g points back to a."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, 3, 3)
    mesh = build_mesh_2d(VX, VY, EToV, N=3)
    EToE = np.asarray(mesh.EToE)
    EToF = np.asarray(mesh.EToF)
    for k in range(mesh.K):
        for f in range(3):
            kp, fp = int(EToE[k, f]), int(EToF[k, f])
            if kp == k and fp == f:
                continue  # boundary
            assert int(EToE[kp, fp]) == k
            assert int(EToF[kp, fp]) == f


def test_boundary_node_count_for_rectangle():
    """Outer boundary has 4 sides * Kx (or Ky) faces * Nfp nodes."""
    N = 3
    Kx = Ky = 4
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, N=N)
    expected = (N + 1) * (2 * Kx + 2 * Ky)
    assert len(np.asarray(mesh.mapB)) == expected


def test_update_mesh_vertices_matches_full_build():
    """update_mesh_2d_vertices reproduces build_mesh_2d output bit-for-bit."""
    Kx = Ky = 3
    VX0, VY0, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    mesh0 = build_mesh_2d(VX0, VY0, EToV, N=4)

    # Slightly perturbed vertex positions (still non-degenerate).
    rng = np.random.default_rng(0)
    VX1_np = np.asarray(VX0) + 0.02 * rng.standard_normal(VX0.shape[0])
    VY1_np = np.asarray(VY0) + 0.02 * rng.standard_normal(VY0.shape[0])
    VX1 = jnp.asarray(VX1_np)
    VY1 = jnp.asarray(VY1_np)

    mesh_updated = update_mesh_2d_vertices(mesh0, VX1, VY1)
    mesh_full = build_mesh_2d(VX1, VY1, EToV, N=4)

    for field in ("x", "y", "rx", "sx", "ry", "sy", "J", "nx", "ny", "sJ", "Fscale"):
        a = np.asarray(getattr(mesh_updated, field))
        b = np.asarray(getattr(mesh_full, field))
        assert np.max(np.abs(a - b)) < 1e-12, f"{field} mismatch"
