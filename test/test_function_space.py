"""Smoke tests for the FunctionSpace abstraction.

The protocol is in :mod:`ddg.dg2d.function_space`. These tests cover the
:class:`LagrangianDG` concrete impl: degree-preserving wrap of an existing
mesh, lower-degree pressure space sharing geometry with a higher-degree
velocity mesh, pytree round-trip (so spaces can be passed through
``jax.jit``), and an end-to-end jit example.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.function_space import FunctionSpace, LagrangianDG


def _build_mesh(N: int):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, 4, 4)
    return build_mesh_2d(VX, VY, EToV, N)


def test_lagrangian_dg_wraps_existing_mesh_at_its_degree():
    """``at_degree(mesh, mesh.N)`` reuses the mesh object as-is."""
    mesh = _build_mesh(N=3)
    V = LagrangianDG.at_degree(mesh, degree=3, components=2)
    assert V.mesh is mesh
    assert V.degree == 3
    assert V.Np == mesh.Np
    assert V.K == mesh.K
    assert V.components == 2
    assert V.total_dofs == V.Np * V.K * 2


def test_lagrangian_dg_at_lower_degree_shares_geometry():
    """A pressure space at degree N-1 paired with an N velocity mesh: the
    pressure space's Mesh2D shares the same vertex coordinates and
    element-vertex topology, but has its own reference-element operators
    at the lower degree."""
    mesh_u = _build_mesh(N=3)
    V_p = LagrangianDG.at_degree(mesh_u, degree=2, components=1)

    # New mesh object (Mesh2D was rebuilt at degree 2).
    assert V_p.mesh is not mesh_u
    # Same triangulation.
    np.testing.assert_array_equal(np.asarray(V_p.mesh.VX), np.asarray(mesh_u.VX))
    np.testing.assert_array_equal(np.asarray(V_p.mesh.VY), np.asarray(mesh_u.VY))
    np.testing.assert_array_equal(
        np.asarray(V_p.mesh.EToV), np.asarray(mesh_u.EToV)
    )
    # Different basis: degree 2 has Np = 6, degree 3 has Np = 10.
    assert V_p.degree == 2
    assert V_p.Np == 6
    assert mesh_u.Np == 10
    # Element count unchanged.
    assert V_p.K == mesh_u.K


def test_lagrangian_dg_satisfies_function_space_protocol():
    """Runtime check that LagrangianDG conforms to the FunctionSpace
    Protocol. Important for the pairwise-dispatch design of cross-space
    operators: the dispatcher inspects the spaces and needs to know they're
    FunctionSpaces."""
    mesh = _build_mesh(N=2)
    V = LagrangianDG(mesh=mesh, components=1)
    assert isinstance(V, FunctionSpace)


def test_lagrangian_dg_pytree_roundtrip():
    """jax.tree_util can flatten/unflatten the space, so it can be passed
    through jit. Without this the space would have to live as a closure or
    a static arg."""
    mesh = _build_mesh(N=2)
    V = LagrangianDG(mesh=mesh, components=2)
    leaves, treedef = jax.tree_util.tree_flatten(V)
    V_round = jax.tree_util.tree_unflatten(treedef, leaves)
    assert V_round.components == V.components
    assert V_round.degree == V.degree
    assert V_round.Np == V.Np
    assert V_round.K == V.K


def test_lagrangian_dg_jit_passthrough():
    """End-to-end: pass a space into a jit'd function that reads its mesh
    and does something elementwise. Validates the pytree registration is
    correct for real jit use."""
    mesh = _build_mesh(N=2)
    V = LagrangianDG(mesh=mesh, components=1)

    @jax.jit
    def field_norm(space: LagrangianDG) -> jax.Array:
        x, y = space.mesh.x, space.mesh.y
        return jnp.sum(x * x + y * y)

    out = field_norm(V)
    expected = float(jnp.sum(mesh.x * mesh.x + mesh.y * mesh.y))
    np.testing.assert_allclose(float(out), expected, rtol=1e-12)


def test_lagrangian_dg_velocity_and_pressure_pair():
    """The canonical mixed-order pair: V_u at degree N, V_p at degree N-1,
    sharing geometry. Both must be valid LagrangianDG objects with the
    correct shapes for the saddle-point assembly to come in Phase 2."""
    mesh = _build_mesh(N=3)
    V_u = LagrangianDG.at_degree(mesh, degree=3, components=2)
    V_p = LagrangianDG.at_degree(mesh, degree=2, components=1)

    assert V_u.K == V_p.K
    # Mixed-order: velocity has more DOFs per element than pressure.
    assert V_u.Np > V_p.Np
    # Total DOFs in the saddle-point linear system:
    #   2 * (Np_u * K)  velocity DOFs (one block per component)
    # + (Np_p * K)      pressure DOFs.
    total = V_u.total_dofs + V_p.total_dofs
    assert total == 2 * V_u.Np * V_u.K + V_p.Np * V_p.K
