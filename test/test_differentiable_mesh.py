"""Tests for the JAX-traceable mesh layer in
``ddg.dg2d.differentiable_mesh``.

Three things to check:

1. **Numerical equivalence at the nominal vertices.** The geometric
   quantities ``(x, y, J, rx, sx, ry, sy, nx, ny, sJ, Fscale)``
   produced by :func:`differentiable_mesh_quantities` must match the
   host-built :class:`Mesh2D` from ``build_mesh_2d`` to machine
   precision.
2. **AD vs FD consistency for the affine map.** ``jax.grad`` of any
   scalar functional of ``(x, y)`` w.r.t. a chosen vertex coordinate
   must agree with a central finite difference.
3. **AD vs FD for the geometric factors.** Same check but for ``J``
   so the chain rule through ``_geometric_factors_2d`` is exercised.

The tests use a small uniform-square mesh so the assertions are
cheap; ``test_meshes`` already covers the host ``build_mesh_2d`` so
we don't repeat that here.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.differentiable_mesh import (
    make_template, differentiable_mesh_quantities,
)


def _build_small():
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, 4, 4)
    mesh = build_mesh_2d(VX, VY, EToV, N=3)
    template = make_template(mesh)
    return mesh, template


def test_nominal_equivalence_matches_host_to_machine():
    """At ``VX_nominal, VY_nominal`` the JAX path must reproduce the
    host ``Mesh2D`` element-for-element."""
    mesh, template = _build_small()
    diff = differentiable_mesh_quantities(
        template.VX_nominal, template.VY_nominal, template,
    )
    for name in ("x", "y", "rx", "sx", "ry", "sy", "J",
                 "nx", "ny", "sJ", "Fscale"):
        host = np.asarray(getattr(mesh, name))
        jax_v = np.asarray(getattr(diff, name))
        max_err = float(np.max(np.abs(host - jax_v)))
        assert max_err < 1e-13, (
            f"{name}: host vs diff disagrees by {max_err:.3e}"
        )


def test_x_y_gradient_matches_fd():
    """A scalar functional of ``(x, y)`` differentiates analytically
    in ``VX``; AD must match central FD on a randomly chosen vertex."""
    _, template = _build_small()
    VX_t = template.VX_nominal
    VY_t = template.VY_nominal

    def loss(VX):
        diff = differentiable_mesh_quantities(VX, VY_t, template)
        # Some smooth functional that mixes (x, y, J).
        return jnp.sum(diff.x * diff.J) + jnp.sum(diff.y * diff.J)

    grad_loss = jax.grad(loss)(VX_t)
    # FD on vertex index 5 (arbitrary interior vertex).
    idx = 5
    h = 1e-5
    plus = VX_t.at[idx].add(h)
    minus = VX_t.at[idx].add(-h)
    fd = (loss(plus) - loss(minus)) / (2.0 * h)
    ad = float(grad_loss[idx])
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-30)
    assert rel < 5e-7, (
        f"AD {ad:+.4e} vs FD {fd:+.4e} relative diff {rel:.2e}"
    )


def test_J_sensitivity_matches_fd():
    """Differentiation of ``J`` w.r.t. ``VY``: probes the chain through
    ``_geometric_factors_2d`` and is the calculation BDM-via-mesh
    inverse problems will actually exercise."""
    _, template = _build_small()
    VX_t = template.VX_nominal
    VY_t = template.VY_nominal

    def total_J(VY):
        diff = differentiable_mesh_quantities(VX_t, VY, template)
        # sum(J) is area-conserving so its gradient vanishes; use
        # sum(J**2) which is not.
        return jnp.sum(diff.J ** 2)

    grad_J = jax.grad(total_J)(VY_t)
    # Vertex 1 is on the bottom boundary; uniform-mesh symmetry zeroes
    # the gradient at interior vertices, so we test a boundary one.
    idx = 1
    h = 1e-5
    plus = VY_t.at[idx].add(h)
    minus = VY_t.at[idx].add(-h)
    fd = (total_J(plus) - total_J(minus)) / (2.0 * h)
    ad = float(grad_J[idx])
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-30)
    assert rel < 5e-7, (
        f"AD {ad:+.4e} vs FD {fd:+.4e} relative diff {rel:.2e}"
    )


def test_normals_sensitivity_matches_fd():
    """Gradient flow through ``_normals_2d``: a scalar functional of
    ``(nx, ny, sJ)`` w.r.t. a vertex."""
    _, template = _build_small()
    VX_t = template.VX_nominal
    VY_t = template.VY_nominal

    def loss(VX):
        diff = differentiable_mesh_quantities(VX, VY_t, template)
        # sum(sJ) over interior faces is conserved when an interior
        # vertex moves; mix in sJ**2 + nx*ny*x to break the symmetry.
        return (jnp.sum(diff.sJ ** 2)
                + jnp.sum(diff.nx * diff.ny * diff.x[diff.Fmask].reshape(-1, diff.K)))

    g = jax.grad(loss)(VX_t)
    idx = 12
    h = 1e-5
    plus = VX_t.at[idx].add(h)
    minus = VX_t.at[idx].add(-h)
    fd = (loss(plus) - loss(minus)) / (2.0 * h)
    ad = float(g[idx])
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-30)
    assert rel < 5e-7, (
        f"AD {ad:+.4e} vs FD {fd:+.4e} relative diff {rel:.2e}"
    )


# ---------------------------------------------------------------------------
# Phase 2 / Phase 3: differentiable cylinder curving + gradient validation
# ---------------------------------------------------------------------------

from ddg.dg2d.mesh import channel_with_circle_mesh_2d
from ddg.dg2d.curved import apply_circular_hole_2d
from ddg.dg2d.differentiable_mesh import (
    make_curving_info, differentiable_curve, differentiable_mesh_with_circle,
)


def _build_cylinder_template(cx=0.5, cy=0.5, radius=0.1):
    """Tiny cylinder-in-channel mesh for the curving tests."""
    VX, VY, EToV = channel_with_circle_mesh_2d(
        0.0, 1.0, 0.0, 1.0, 10, 10,
        cx=cx, cy=cy, radius=radius,
        R_outer=0.2, n_radial=2, radial_growth=1.3,
    )
    mesh = build_mesh_2d(VX, VY, EToV, N=2)
    template = make_template(mesh)
    return mesh, template


def test_curving_matches_host_at_nominal():
    """At the nominal (cx, cy, radius) the JAX curving must reproduce
    the host ``apply_circular_hole_2d`` output."""
    CX, CY, R = 0.5, 0.5, 0.1
    base_mesh, template = _build_cylinder_template(CX, CY, R)
    curved_host = apply_circular_hole_2d(
        base_mesh, cx=CX, cy=CY, radius=R,
        blend_interior=True, detection_tol=0.05,
    )
    info = make_curving_info(
        base_mesh, cx=CX, cy=CY, radius=R, detection_tol=0.05,
    )
    diff = differentiable_mesh_with_circle(
        template.VX_nominal, template.VY_nominal, template, info,
        jnp.asarray(CX), jnp.asarray(CY), jnp.asarray(R),
    )
    for name in ("x", "y", "J", "sJ"):
        host = np.asarray(getattr(curved_host, name))
        jax_v = np.asarray(getattr(diff, name))
        err = float(np.max(np.abs(host - jax_v)))
        assert err < 1e-12, (
            f"{name}: host vs diff disagrees by {err:.3e}"
        )


def test_curving_gradient_matches_fd():
    """``jax.grad`` w.r.t. cy of an x*sJ-style functional must match
    central FD."""
    CX, CY, R = 0.5, 0.5, 0.1
    base_mesh, template = _build_cylinder_template(CX, CY, R)
    info = make_curving_info(
        base_mesh, cx=CX, cy=CY, radius=R, detection_tol=0.05,
    )

    def loss(cy_val):
        diff = differentiable_mesh_with_circle(
            template.VX_nominal, template.VY_nominal, template, info,
            jnp.asarray(CX), cy_val, jnp.asarray(R),
        )
        # Use diff.y * diff.J --- the y-position-weighted area integral.
        # Reflection symmetry across the centred cylinder zeroes any
        # even functional of (x, y); the asymmetric one below is
        # sensitive to a CY shift.
        return jnp.sum(diff.y * diff.J)

    ad = float(jax.grad(loss)(jnp.asarray(CY)))
    h = 1e-5
    fd = (float(loss(jnp.asarray(CY + h))) - float(loss(jnp.asarray(CY - h)))) / (2 * h)
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-30)
    assert rel < 1e-5, (
        f"AD {ad:+.4e} vs FD {fd:+.4e} relative diff {rel:.2e}"
    )


def test_radius_gradient_matches_fd():
    """``jax.grad`` w.r.t. the radius."""
    CX, CY, R = 0.5, 0.5, 0.1
    base_mesh, template = _build_cylinder_template(CX, CY, R)
    info = make_curving_info(
        base_mesh, cx=CX, cy=CY, radius=R, detection_tol=0.05,
    )

    def loss(R_val):
        diff = differentiable_mesh_with_circle(
            template.VX_nominal, template.VY_nominal, template, info,
            jnp.asarray(CX), jnp.asarray(CY), R_val,
        )
        return jnp.sum(diff.J ** 2)

    ad = float(jax.grad(loss)(jnp.asarray(R)))
    h = 1e-5
    fd = (float(loss(jnp.asarray(R + h))) - float(loss(jnp.asarray(R - h)))) / (2 * h)
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-30)
    assert rel < 1e-5, (
        f"AD {ad:+.4e} vs FD {fd:+.4e} relative diff {rel:.2e}"
    )
