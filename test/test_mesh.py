"""Mesh + connectivity tests for the 1D builder."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg1d.mesh import build_mesh_1d, uniform_mesh_1d


def test_uniform_mesh_shapes_and_vertex_layout():
    K = 6
    VX, EToV = uniform_mesh_1d(-1.0, 1.0, K)
    assert VX.shape == (K + 1,)
    assert EToV.shape == (K, 2)
    np.testing.assert_allclose(np.asarray(VX[0]), -1.0)
    np.testing.assert_allclose(np.asarray(VX[-1]), 1.0)


def test_mesh1d_basic_shapes_and_geometric_factors():
    N, K = 5, 4
    VX, EToV = uniform_mesh_1d(0.0, 2.0, K)
    mesh = build_mesh_1d(VX, EToV, N=N)

    Np = N + 1
    assert mesh.Np == Np and mesh.K == K and mesh.Nfaces == 2 and mesh.Nfp == 1
    assert mesh.x.shape == (Np, K)
    assert mesh.J.shape == (Np, K)
    assert mesh.rx.shape == (Np, K)
    # For a uniform mesh of [0, 2] split into K equal pieces, each element
    # has physical length 2/K, so J = (2/K) / 2 = 1/K.
    expected_J = 2.0 / K / 2.0
    np.testing.assert_allclose(np.asarray(mesh.J), expected_J, atol=1e-13)
    np.testing.assert_allclose(np.asarray(mesh.rx), 1.0 / expected_J, atol=1e-13)


def test_internal_faces_match_exactly():
    """At interior faces, x[vmapM] must equal x[vmapP]."""
    N, K = 4, 10
    VX, EToV = uniform_mesh_1d(0.0, 1.0, K)
    mesh = build_mesh_1d(VX, EToV, N=N)
    x_flat = np.asarray(mesh.x).flatten(order="F")
    diffs = x_flat[np.asarray(mesh.vmapM)] - x_flat[np.asarray(mesh.vmapP)]
    internal = ~np.isin(np.arange(len(mesh.vmapM)), np.asarray(mesh.mapB))
    assert np.max(np.abs(diffs[internal])) < 1e-13


def test_boundary_maps_for_non_periodic_mesh():
    """A non-periodic mesh has exactly two boundary nodes: the two endpoints."""
    N, K = 4, 5
    VX, EToV = uniform_mesh_1d(-1.0, 1.0, K)
    mesh = build_mesh_1d(VX, EToV, N=N, periodic=False)
    mapB = np.asarray(mesh.mapB)
    vmapB = np.asarray(mesh.vmapB)
    assert mapB.shape == (2,)
    assert vmapB.shape == (2,)
    # vmapB[0] is the left endpoint (node 0 of element 0)
    # vmapB[1] is the right endpoint (node Np-1 of element K-1)
    assert int(vmapB[0]) == 0
    assert int(vmapB[1]) == K * (N + 1) - 1


def test_periodic_wrap():
    """Periodic mesh sends the left endpoint to the right endpoint."""
    N, K = 3, 4
    VX, EToV = uniform_mesh_1d(0.0, 1.0, K)
    mesh = build_mesh_1d(VX, EToV, N=N, periodic=True)
    vmapM = np.asarray(mesh.vmapM)
    vmapP = np.asarray(mesh.vmapP)
    # left-most face wraps to the right-most face and vice versa
    assert int(vmapP[0]) == int(vmapM[-1])
    assert int(vmapP[-1]) == int(vmapM[0])
    assert np.asarray(mesh.mapB).size == 0


@pytest.mark.parametrize("K", [2, 4, 8])
def test_connectivity_is_symmetric(K):
    """If element a's face f points to (b, g), then b's face g points back to a."""
    VX, EToV = uniform_mesh_1d(0.0, 1.0, K)
    mesh = build_mesh_1d(VX, EToV, N=3)
    EToE = np.asarray(mesh.EToE)
    EToF = np.asarray(mesh.EToF)
    for k in range(K):
        for f in range(2):
            kp, fp = int(EToE[k, f]), int(EToF[k, f])
            # boundary face points to itself
            if kp == k and fp == f:
                continue
            assert int(EToE[kp, fp]) == k
            assert int(EToF[kp, fp]) == f
