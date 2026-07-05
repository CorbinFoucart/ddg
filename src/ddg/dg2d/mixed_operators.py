"""Cross-space mixed-FEM operators for the velocity-pressure saddle point.

For an inf-sup stable pair ``(V_u, V_p)`` (velocity polynomial degree
strictly higher than pressure on each element, both nodal-Lagrange DG),
this module assembles the off-diagonal blocks of the coupled NS system:

    [  A     G  ] [u]   [b_u]
    [ -D    -S  ] [p] = [b_p]

where

    G : V_p -> V_u    pressure-gradient operator (action on velocity test)
    D : V_u -> V_p    velocity-divergence operator (action on pressure test)

For LagrangianDG / LagrangianDG pairs the implementation uses the
**embedding trick**: since the pressure space is a strict subspace of the
velocity space (``P_{N-1} ⊂ P_N`` on each triangle), we

    1. lift the pressure DOFs ``p ∈ V_p`` into ``V_u`` via the interpolation
       matrix ``P`` (``Np_u × Np_p``) and apply the existing equal-order
       operators :func:`coupled_ns._grad_apply_2d` /
       :func:`coupled_ns._div_apply_2d` at velocity degree;
    2. project the divergence result back to ``V_p`` via ``P.T``.

The embedding is *exact* (no quadrature error introduced) because every
``V_p`` function has a unique representation in ``V_u`` -- the
interpolation matrix is the change-of-basis between the two nodal sets.
This is what gives us correctness without writing any new low-level
operator code; only ``build_interpolation`` is new.

When BDM joins as a second :class:`FunctionSpace` impl, the dispatch in
:func:`mixed_grad_apply` / :func:`mixed_div_apply` grows a branch on the
``type(V_u)`` axis -- the LagrangianDG branch stays unchanged.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ddg.dg2d.coupled_ns import _div_apply_2d, _grad_apply_2d
from ddg.dg2d.function_space import LagrangianDG
from ddg.dg2d.reference import vandermonde_2d
from ddg.dg2d.sipg import _local_mass


def build_interpolation(V_from: LagrangianDG, V_to: LagrangianDG) -> jax.Array:
    """Lagrangian nodal interpolation matrix from ``V_from`` to ``V_to``.

    For LagrangianDG spaces sharing the same reference triangle and mesh
    geometry, this returns a matrix ``P`` of shape ``(V_to.Np, V_from.Np)``
    such that for any field with V_from-nodal DOFs ``c_from``, the V_to-
    nodal DOFs of the *same polynomial* are ``c_to = P @ c_from``.

    Built from the modal (Koornwinder-Dubiner) basis: ``P = V_modal_at_to
    @ V_from.mesh.invV``, where ``V_modal_at_to`` evaluates the V_from
    *modal* basis at the V_to *nodal* points. For ``V_from.degree ==
    V_to.degree`` this returns the identity.

    Requires ``V_from.degree <= V_to.degree`` for the interpolation to be
    exact (otherwise it's a lossy projection -- supported but flagged via
    a runtime check).
    """
    if V_from.degree > V_to.degree:
        raise ValueError(
            f"build_interpolation: V_from.degree={V_from.degree} > "
            f"V_to.degree={V_to.degree}; this would be a lossy projection. "
            f"Build it explicitly via least-squares if intended."
        )
    if V_from.degree == V_to.degree:
        return jnp.eye(V_from.Np)
    V_modal_at_to = vandermonde_2d(V_from.degree, V_to.mesh.r, V_to.mesh.s)
    return V_modal_at_to @ V_from.mesh.invV


def mixed_grad_apply(
    V_u: LagrangianDG,
    V_p: LagrangianDG,
    p: jax.Array,
    *,
    ins_bc_types: jax.Array,
    M_ref_u: jax.Array,
    P_emb: jax.Array,
    p_dir: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Cross-space pressure-gradient operator ``G p`` applied to velocity tests.

    Returns ``(Gx p, Gy p)`` with shapes ``(V_u.Np, V_u.K)`` -- the
    velocity-component RHS contributions ``∫ ∇p · v_i`` for ``v_i`` the
    nodal velocity test functions.

    ``p`` is provided in V_p coordinates (shape ``(V_p.Np, V_p.K)``); the
    operator embeds it into V_u via ``P_emb`` and calls the existing
    equal-order :func:`coupled_ns._grad_apply_2d` on the embedding.

    Parameters
    ----------
    p_dir : optional
        Outflow Dirichlet pressure boundary values, in V_u face-DOF
        layout (shape ``(V_u.Np, V_u.K)``). Already at velocity degree;
        no embedding applied. Pass ``None`` for problems without outflow.
    """
    p_emb = P_emb @ p
    return _grad_apply_2d(
        V_u.mesh, p_emb,
        ins_bc_types=ins_bc_types,
        M_ref=M_ref_u,
        p_dir=p_dir,
    )


def mixed_div_apply(
    V_u: LagrangianDG,
    V_p: LagrangianDG,
    ux: jax.Array,
    uy: jax.Array,
    *,
    ins_bc_types: jax.Array,
    M_ref_u: jax.Array,
    P_emb: jax.Array,
    bc_ux: jax.Array | None = None,
    bc_uy: jax.Array | None = None,
) -> jax.Array:
    """Cross-space velocity-divergence operator ``D u`` against pressure tests.

    Returns ``D u`` with shape ``(V_p.Np, V_p.K)`` -- the pressure-test RHS
    contribution ``∫ q (∇·u)`` for ``q`` the nodal pressure test functions.

    Computed by applying the existing equal-order
    :func:`coupled_ns._div_apply_2d` (which yields a V_u-test-indexed
    result) and projecting back to V_p via ``P_emb.T``:

        D_mixed u = P_emb.T (D_eq u)

    This identity follows from the embedding: for any ``q ∈ V_p`` with
    DOFs ``c``, ``q`` has V_u DOFs ``P_emb @ c``, so
    ``∫ q (∇·u) = c.T @ P_emb.T @ (D_eq u)``.
    """
    div_eq = _div_apply_2d(
        V_u.mesh, ux, uy,
        ins_bc_types=ins_bc_types,
        M_ref=M_ref_u,
        bc_ux=bc_ux, bc_uy=bc_uy,
    )
    return P_emb.T @ div_eq


def local_pressure_mass(V_p: LagrangianDG) -> jax.Array:
    """Per-element pressure mass matrix at V_p degree.

    Thin wrapper for clarity at call sites -- the existing
    :func:`sipg._local_mass` already returns the reference mass for any
    Mesh2D, so we just call it on ``V_p.mesh``. Returns shape
    ``(V_p.Np, V_p.Np)``; the physical element mass is ``J * M_ref``.
    """
    return _local_mass(V_p.mesh)
