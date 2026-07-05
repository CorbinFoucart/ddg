"""Function-space abstraction for mixed FEM on 2D triangular meshes.

A :class:`FunctionSpace` bundles a polynomial basis (degree, conformity,
scalar-vs-vector) over a shared mesh geometry. It exposes the per-element
operators that cross-space saddle-point assembly and Schur-PC machinery need.
Element-family specifics live on the concrete impl.

The protocol is intentionally minimal. Cross-space operators
(``D : V_u -> V_p``, ``G : V_p -> V_u``, ``Helmholtz : V_u``, etc.) are
free functions in sibling modules that dispatch on the pair
``(type(V_trial), type(V_test))`` -- no single impl needs to know about the
others.

Concrete impls:
    * :class:`LagrangianDG` -- nodal discontinuous Lagrange basis at a chosen
      polynomial degree on each triangle. Wraps a :class:`Mesh2D` carrying
      the basis-specific operators.

Planned:
    * ``BDM`` -- H(div)-conforming Brezzi-Douglas-Marini basis with edge-DOF
      sign convention for normal-trace continuity. Distinct ``Np`` and
      operator surface from LagrangianDG; will exercise the
      pairwise-dispatch design above.
    * ``LagrangianCG`` -- continuous nodal Lagrange basis (Taylor-Hood).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from ddg.dg2d.mesh import Mesh2D, build_mesh_2d


@runtime_checkable
class FunctionSpace(Protocol):
    """Polynomial function space over a 2D triangular mesh.

    Implementations carry the basis-specific reference-element operators
    and expose just enough surface for cross-space operators to work
    without reaching into element-family internals.

    Attributes accessed by cross-space code:
        ``mesh``      shared geometry / connectivity / face data.
        ``Np``        DOFs per element for ONE scalar component.
        ``K``         number of elements (== ``mesh.K``).
        ``components``  1 (scalar field) or 2 (2D vector field).
        ``total_dofs``  ``Np * K * components``.
    """

    mesh: Mesh2D
    Np: int
    K: int
    components: int
    total_dofs: int


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LagrangianDG:
    """Nodal discontinuous-Lagrange function space at degree ``mesh.N``.

    Wraps a :class:`Mesh2D` that already carries the basis-specific
    reference-element operators (``r``, ``s``, ``V``, ``Dr``, ``Ds``,
    ``LIFT``, ``Fmask``) and the derived physical / connectivity fields at
    those nodes. Use :meth:`at_degree` to construct a space at a polynomial
    degree different from the underlying mesh, e.g. the ``N-1`` pressure
    space paired with an ``N`` velocity mesh.

    ``components`` selects scalar (1) vs 2D-vector (2) -- the latter
    multiplies ``total_dofs`` by 2 but does not change the per-element
    operators, which act on one scalar component at a time.
    """

    mesh: Mesh2D
    components: int = 1

    # ------------------------------------------------------------------ ctors
    @classmethod
    def at_degree(
        cls,
        mesh: Mesh2D,
        degree: int,
        *,
        components: int = 1,
    ) -> "LagrangianDG":
        """Build a LagrangianDG at the requested polynomial degree.

        If ``degree == mesh.N`` the existing mesh is reused unchanged; otherwise
        a fresh :class:`Mesh2D` is built from the same
        ``VX, VY, EToV`` at the new degree.
        """
        if degree == mesh.N:
            return cls(mesh=mesh, components=components)
        sub_mesh = build_mesh_2d(mesh.VX, mesh.VY, mesh.EToV, degree)
        return cls(mesh=sub_mesh, components=components)

    # ---------------------------------------------------------------- accessors
    @property
    def degree(self) -> int:
        return self.mesh.N

    @property
    def Np(self) -> int:
        return self.mesh.Np

    @property
    def K(self) -> int:
        return self.mesh.K

    @property
    def total_dofs(self) -> int:
        return self.Np * self.K * self.components

    # ------------------------------------------------------------------ pytree
    def tree_flatten(self):
        # ``components`` is a Python int (static); the mesh's leaves flow.
        return (self.mesh,), (self.components,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (mesh,) = children
        (components,) = aux
        return cls(mesh=mesh, components=components)
