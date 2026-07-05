"""Matrix-free block-Jacobi preconditioner for the SIPG operators.

The block-Jacobi preconditioner needs each element's ``(Np, Np)``
self-coupling block. The dense path extracts these from the assembled
``(Np K)²`` matrix — memory-bound at fine mesh. The matrix-free path
(:func:`extract_block_diag_matfree`) gets the same blocks with
``ncolours × Np`` operator applies and never assembles the matrix.

Checks:
  1. The element graph-colouring is valid — no two face-adjacent
     elements share a colour (else the colored-probe extraction
     picks up off-diagonal pollution).
  2. The matrix-free SIPG preconditioner applies *identically* to
     the dense-extracted one (machine precision).
  3. Same for the Helmholtz preconditioner ``α M + A_sipg``.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.sipg import (
    BC_DIRICHLET, BC_INTERIOR, BC_NEUMANN,
    _color_elements, make_helmholtz_block_jacobi_preconditioner,
    make_sipg_block_jacobi_preconditioner,
)


def _mesh(N=3, K=8):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, K, K)
    return build_mesh_2d(VX, VY, EToV, N)


def _mixed_bc(mesh):
    """Dirichlet on the boundary, interior elsewhere."""
    EToE = np.asarray(mesh.EToE)
    is_bdy = EToE == np.arange(mesh.K)[:, None]
    return jnp.asarray(np.where(is_bdy, BC_DIRICHLET, BC_INTERIOR).astype(np.int32))


def test_element_colouring_is_valid():
    """No two face-adjacent elements share a colour."""
    mesh = _mesh()
    colour, ncolours = _color_elements(mesh)
    EToE = np.asarray(mesh.EToE)
    violations = 0
    for k in range(mesh.K):
        for nb in EToE[k]:
            if nb != k and colour[nb] == colour[k]:
                violations += 1
    print(f"  {ncolours} colours, {violations} adjacency violations")
    assert violations == 0
    assert ncolours <= 4   # planar triangulation, <=3 neighbours


def test_matfree_sipg_preconditioner_matches_dense():
    """Matrix-free SIPG block-Jacobi == dense-extracted, to machine eps."""
    mesh = _mesh()
    bc = _mixed_bc(mesh)
    pc_dense = make_sipg_block_jacobi_preconditioner(
        mesh, bc_types=bc, matfree=False)
    pc_matfree = make_sipg_block_jacobi_preconditioner(
        mesh, bc_types=bc, matfree=True)
    v = jax.random.normal(jax.random.PRNGKey(0), (mesh.Np * mesh.K,))
    err = float(jnp.max(jnp.abs(pc_dense(v) - pc_matfree(v))))
    print(f"  max|dense - matfree| = {err:.2e}")
    assert err < 1e-11


def test_matfree_helmholtz_preconditioner_matches_dense():
    """Matrix-free Helmholtz block-Jacobi == dense-extracted."""
    mesh = _mesh()
    bc = _mixed_bc(mesh)
    pc_dense = make_helmholtz_block_jacobi_preconditioner(
        mesh, alpha=37.0, nu=0.01, bc_types=bc, matfree=False)
    pc_matfree = make_helmholtz_block_jacobi_preconditioner(
        mesh, alpha=37.0, nu=0.01, bc_types=bc, matfree=True)
    v = jax.random.normal(jax.random.PRNGKey(1), (mesh.Np * mesh.K,))
    err = float(jnp.max(jnp.abs(pc_dense(v) - pc_matfree(v))))
    print(f"  max|dense - matfree| = {err:.2e}")
    assert err < 1e-11


def test_matfree_preconditioner_is_spd_like():
    """The block-Jacobi apply is SPD: <v, M_inv v> > 0 for v != 0
    (each element block of an SPD operator is SPD, so is its inverse)."""
    mesh = _mesh()
    bc = _mixed_bc(mesh)
    pc = make_sipg_block_jacobi_preconditioner(mesh, bc_types=bc, matfree=True)
    v = jax.random.normal(jax.random.PRNGKey(2), (mesh.Np * mesh.K,))
    quad = float(jnp.dot(v, pc(v)))
    print(f"  <v, M_inv v> = {quad:.4e}")
    assert quad > 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("all ok")
