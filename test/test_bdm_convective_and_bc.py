"""Tests for the BDM convective term (Phase 5f-1) and boundary handling
(Phase 5f-2).

The convective term must satisfy basic linearity and identity tests
before being wired into the Newton outer loop:

  * Linear in u (for fixed u*), so zero u -> zero output.
  * Zero u* -> zero output (no advection without a transport velocity).
  * For u = u* = constant in BDM_1, ∇·(u*⊗u) = 0 -- the output is zero
    modulo the boundary contributions from the LF flux at boundary
    faces (which use the natural u_P = u_M convention by default).

Boundary handling helpers:
  * bdm1_boundary_dofs_mask: marks the DOFs that live on boundary faces.
  * bdm1_evaluate_dirichlet_bc: builds a global DOF vector with the
    prescribed normal-trace values at boundary GL points.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import BDM
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask,
    bdm1_convective_apply,
    bdm1_evaluate_dirichlet_bc,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _build_mesh(Kx: int = 3, Ky: int = 3):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, 1)


# ---------------------------------------------------------------------------
# Convective apply
# ---------------------------------------------------------------------------

def test_convective_apply_zero_velocity_yields_zero():
    mesh = _build_mesh()
    V = BDM.from_mesh(mesh)
    z = jnp.zeros(V.total_dofs)
    rng = np.random.default_rng(0)
    u_star = jnp.asarray(rng.standard_normal(V.total_dofs))
    out = bdm1_convective_apply(V, z, u_star_global=u_star)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-14)


def test_convective_apply_zero_u_star_yields_zero():
    mesh = _build_mesh()
    V = BDM.from_mesh(mesh)
    z = jnp.zeros(V.total_dofs)
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal(V.total_dofs))
    out = bdm1_convective_apply(V, u, u_star_global=z)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-14)


def test_convective_linear_in_u_for_fixed_u_star():
    """For fixed u*, the operator must be linear in u: ``conv(a*u, u*) ==
    a * conv(u, u*)``."""
    mesh = _build_mesh()
    V = BDM.from_mesh(mesh)
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal(V.total_dofs))
    u_star = jnp.asarray(rng.standard_normal(V.total_dofs))
    a = 2.5
    out_a = bdm1_convective_apply(V, a * u, u_star_global=u_star)
    out_scaled = a * bdm1_convective_apply(V, u, u_star_global=u_star)
    np.testing.assert_allclose(
        np.asarray(out_a), np.asarray(out_scaled), atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Boundary handling
# ---------------------------------------------------------------------------

def test_boundary_mask_marks_only_boundary_dofs():
    """For a 3x3 unit-square mesh: the boundary mask should mark every
    DOF that lives on a boundary face (i.e., not shared with a
    neighbour) and only those."""
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh)
    mask = np.asarray(bdm1_boundary_dofs_mask(V))
    assert mask.shape == (V.total_dofs,)
    assert mask.dtype == bool

    # Count: each boundary face contributes 2 DOFs (none shared).
    # For a 3x3 mesh (2*9 = 18 triangles): 3 boundary edges per side x
    # 4 sides x 1 triangle each = 12 boundary faces; 12 * 2 = 24 DOFs.
    EToE = np.asarray(mesh.EToE)
    n_boundary_faces = int(np.sum(EToE == np.arange(mesh.K)[:, None]))
    expected = 2 * n_boundary_faces
    actual = int(np.sum(mask))
    assert actual == expected, (
        f"Boundary DOF count {actual} != expected {expected}"
    )


def test_dirichlet_bc_evaluation_constant_velocity():
    """For a CONSTANT u_BC = (a, b), the BDM normal-trace DOF at each
    boundary face is just ``a * n_x + b * n_y`` (with the appropriate
    face-outward normal). Test this against direct evaluation."""
    mesh = _build_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh)
    a, b = 1.5, -0.7

    def u_bc_fn(x, y):
        return a * jnp.ones_like(x), b * jnp.ones_like(y)

    u_bc_global = bdm1_evaluate_dirichlet_bc(V, u_bc_fn)
    mask = bdm1_boundary_dofs_mask(V)
    # u_bc_global is zero at interior DOFs.
    np.testing.assert_allclose(
        np.asarray(u_bc_global)[~np.asarray(mask)], 0.0, atol=1e-14,
    )
    # On boundary DOFs, it equals sJ_K * (a*nx + b*ny) at the
    # corresponding face quadrature point -- the Piola edge-Jacobian
    # factor converts physical normal trace to the BDM coefficient on
    # the unsigned basis.
    Nfp = mesh.Nfp
    nx_face = np.asarray(mesh.nx).reshape((Nfp, mesh.Nfaces, mesh.K),
                                           order="F")[0]   # (Nfaces, K)
    ny_face = np.asarray(mesh.ny).reshape((Nfp, mesh.Nfaces, mesh.K),
                                           order="F")[0]
    EToE = np.asarray(mesh.EToE)
    lg = np.asarray(V.topology.local_to_global)
    u_bc_np = np.asarray(u_bc_global)
    for k in range(mesh.K):
        for f in range(3):
            if EToE[k, f] != k:
                continue  # not boundary
            # Global DOF stores the physical normal trace at the GL pt;
            # the Piola factor is absorbed into gather/scatter, not the
            # stored value.
            expected = a * nx_face[f, k] + b * ny_face[f, k]
            for j in range(2):
                g = lg[k, 2 * f + j]
                np.testing.assert_allclose(u_bc_np[g], expected, atol=1e-13)


def test_dirichlet_bc_for_zero_function_is_zero():
    mesh = _build_mesh()
    V = BDM.from_mesh(mesh)

    def u_bc_fn(x, y):
        return jnp.zeros_like(x), jnp.zeros_like(y)

    u_bc_global = bdm1_evaluate_dirichlet_bc(V, u_bc_fn)
    np.testing.assert_allclose(np.asarray(u_bc_global), 0.0, atol=1e-14)


def test_convective_with_dirichlet_bc_runs_and_shape():
    """Smoke test that bdm1_convective_apply accepts u_bc_global without
    error and produces correctly-shaped output."""
    mesh = _build_mesh()
    V = BDM.from_mesh(mesh)
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal(V.total_dofs))
    u_star = jnp.asarray(rng.standard_normal(V.total_dofs))

    def u_bc_fn(x, y):
        return 0.5 * jnp.ones_like(x), 0.2 * jnp.ones_like(y)
    u_bc = bdm1_evaluate_dirichlet_bc(V, u_bc_fn)
    out = bdm1_convective_apply(V, u, u_star_global=u_star, u_bc_global=u_bc)
    assert out.shape == (V.total_dofs,)
