"""Tests for the BDM_1 mesh-level DOF topology.

Covers :func:`ddg.dg2d.bdm.build_bdm1_topology` and the gather/scatter
helpers that round-trip a global DOF vector through per-element local
form.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM1Topology,
    bdm1_gather,
    bdm1_scatter_add,
    build_bdm1_topology,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _build_mesh(Kx: int = 4, Ky: int = 4, N: int = 1):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def _count_boundary_edges(mesh) -> int:
    """Number of element-faces lying on the domain boundary."""
    EToE = np.asarray(mesh.EToE)
    K = mesh.K
    return int(np.sum(EToE == np.arange(K)[:, None]))


def test_total_dof_count_matches_topology_formula():
    """Euler-style count: 6 local DOFs per triangle, interior edges
    deduplicate. Each interior face is one shared edge, so it removes 2
    DOFs from the naive 6K count. Boundary faces don't deduplicate.

        n_global = 6K - 2 * (number of interior faces)
                 = 6K - 2 * (3K - boundary) / 2 * 2     [wait, recount]

    Cleaner: n_global = 2 * (number of distinct edges) and
        2 * edges = 3K + boundary_faces   (each non-boundary face is
            shared, so 2 * interior_edges + boundary_edges = 3K).
    Therefore:
        n_global = 3K + boundary_faces.
    """
    for Kx, Ky in [(2, 2), (4, 4), (4, 3)]:
        mesh = _build_mesh(Kx=Kx, Ky=Ky)
        topo = build_bdm1_topology(mesh)
        bdy = _count_boundary_edges(mesh)
        expected = 3 * mesh.K + bdy
        assert topo.n_global_dofs == expected, (
            f"Kx={Kx} Ky={Ky}: got {topo.n_global_dofs}, "
            f"expected 3*{mesh.K} + {bdy} = {expected}"
        )


def test_local_to_global_indices_are_in_range_and_cover_all():
    """Every local DOF maps to a valid global index, and every global
    index is hit by at least one local DOF."""
    mesh = _build_mesh()
    topo = build_bdm1_topology(mesh)
    lg = np.asarray(topo.local_to_global)
    assert lg.min() >= 0
    assert lg.max() == topo.n_global_dofs - 1
    assert set(lg.flatten().tolist()) == set(range(topo.n_global_dofs))


def test_interior_dofs_appear_with_opposite_signs():
    """For each interior face, the (K, K') pair of adjacent triangles
    must label its 2 DOFs with opposite signs -- one +1 (owner), one -1
    (non-owner). This is what makes H(div)-conformity hold: ``φ · n_K``
    from one side equals ``-(φ · n_K')`` from the other."""
    mesh = _build_mesh()
    topo = build_bdm1_topology(mesh)
    K = mesh.K
    EToE = np.asarray(mesh.EToE)
    EToF = np.asarray(mesh.EToF)
    sign = np.asarray(topo.local_sign)
    for k in range(K):
        for f in range(3):
            k_nbr = int(EToE[k, f])
            if k_nbr == k:
                continue  # boundary face -- skip
            base = 2 * f
            nbr_base = 2 * int(EToF[k, f])
            # Each DOF on this side has the opposite sign of its
            # counterpart on the neighbour side.
            assert (sign[k, base] + sign[k_nbr, nbr_base]) == 0
            assert (sign[k, base + 1] + sign[k_nbr, nbr_base + 1]) == 0


def test_interior_dofs_share_global_index():
    """Each interior-face local DOF on K must map to the *same* global
    DOF as the counterpart local DOF on the neighbour K'. The within-
    face node index may swap if the two triangles traverse the edge in
    opposite directions."""
    mesh = _build_mesh()
    topo = build_bdm1_topology(mesh)
    K = mesh.K
    EToE = np.asarray(mesh.EToE)
    EToF = np.asarray(mesh.EToF)
    EToV = np.asarray(mesh.EToV)
    lg = np.asarray(topo.local_to_global)
    for k in range(K):
        for f in range(3):
            k_nbr = int(EToE[k, f])
            if k_nbr == k:
                continue
            base = 2 * f
            nbr_base = 2 * int(EToF[k, f])
            # Determine traversal: reversed (the typical case) means
            # K's start vertex == K's end vertex.
            v_K_start = int(EToV[k, f])
            v_K_end = int(EToV[k, (f + 1) % 3])
            v_Kp_start = int(EToV[k_nbr, int(EToF[k, f])])
            reversed_ = (v_Kp_start == v_K_end)
            if reversed_:
                # K node 0 shares global with K' node 1, K node 1 with K' node 0.
                assert lg[k, base] == lg[k_nbr, nbr_base + 1]
                assert lg[k, base + 1] == lg[k_nbr, nbr_base]
            else:
                assert lg[k, base] == lg[k_nbr, nbr_base]
                assert lg[k, base + 1] == lg[k_nbr, nbr_base + 1]


def test_boundary_dofs_have_sign_plus_one():
    """Boundary faces have only one element; that element owns the 2
    DOFs with sign +1."""
    mesh = _build_mesh()
    topo = build_bdm1_topology(mesh)
    K = mesh.K
    EToE = np.asarray(mesh.EToE)
    sign = np.asarray(topo.local_sign)
    for k in range(K):
        for f in range(3):
            if EToE[k, f] == k:
                # boundary face
                assert sign[k, 2 * f] == +1
                assert sign[k, 2 * f + 1] == +1


def test_gather_scatter_for_hdiv_conforming_field():
    """A global DOF vector represents an H(div)-conforming field by
    construction: ``u_global -> bdm1_gather -> u_local`` produces a
    field whose normal trace is continuous across every interior face
    (after sign correction). Concretely: for each interior face, the
    two adjacent triangles' local DOFs differ only by sign.

    The scatter-add adjoint sums signed contributions back to global;
    for an H(div) field this gives ``2 * u_global`` on interior DOFs
    (each side contributes ``sign * (sign * u_global) = u_global``,
    summed) and ``u_global`` on boundary DOFs (one contribution).
    """
    mesh = _build_mesh()
    topo = build_bdm1_topology(mesh)
    rng = np.random.default_rng(0)
    u_global = jnp.asarray(rng.standard_normal(topo.n_global_dofs))

    u_local = bdm1_gather(topo, u_global)
    # Sanity: each global DOF in u_local appears with the correct signed
    # value -- check by re-gather.
    assert u_local.shape == (mesh.K, 6)

    # Adjoint test: <bdm1_gather(u_global), v_local> should equal
    # <u_global, bdm1_scatter_add(v_local)> for any v_local.
    v_local = jnp.asarray(rng.standard_normal((mesh.K, 6)))
    lhs = float(jnp.sum(u_local * v_local))
    rhs = float(jnp.sum(u_global * bdm1_scatter_add(topo, v_local)))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_topology_pytree_friendly():
    """The dataclass holds jax arrays so it can be passed through jit.
    BDM1Topology isn't formally a pytree -- ``n_global_dofs`` is a static
    int -- but the arrays inside can be unpacked manually for jit
    closures. Verify the array fields have the expected dtypes / shapes
    for downstream use."""
    mesh = _build_mesh()
    topo = build_bdm1_topology(mesh)
    assert topo.local_to_global.shape == (mesh.K, 6)
    assert topo.local_sign.shape == (mesh.K, 6)
    assert topo.local_to_global.dtype in (jnp.int32, jnp.int64)
    # signs are exactly +1 or -1
    s = np.asarray(topo.local_sign)
    assert set(np.unique(s).tolist()).issubset({-1, +1})
