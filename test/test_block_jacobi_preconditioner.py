"""Block-Jacobi preconditioner for SIPG / Helmholtz CG.

Equivalence checks: solving the same SPD system with and without the
preconditioner must give the same answer (to CG tolerance). Also
verifies the preconditioner is differentiable through ``jax.grad`` —
the differentiability invariant for the matfree CG path.
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
    BC_DIRICHLET, BC_NEUMANN, BC_INTERIOR, IterativeConfig,
    solve_sipg_poisson_2d, solve_sipg_helmholtz_2d,
    factor_sipg_2d, factor_sipg_helmholtz_2d,
    make_sipg_block_jacobi_preconditioner,
    make_helmholtz_block_jacobi_preconditioner,
)


def _mesh_and_bcs(N=3, Kx=4, Ky=4):
    """Tiny mesh with all-Dirichlet boundary."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, N)
    Nfaces, K = mesh.Nfaces, mesh.K
    # Boundary BC tagging: interior face if EToE[k,f] != k else Dirichlet.
    EToE = np.asarray(mesh.EToE)
    bc = np.full((K, Nfaces), BC_INTERIOR, dtype=np.int32)
    for k in range(K):
        for f in range(Nfaces):
            if EToE[k, f] == k:
                bc[k, f] = BC_DIRICHLET
    return mesh, jnp.asarray(bc)


def test_sipg_pc_matches_unpreconditioned():
    mesh, bc_types = _mesh_and_bcs(N=2, Kx=4, Ky=4)
    rng = np.random.default_rng(0)
    source = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))

    cfg = IterativeConfig(method="cg", tol=1e-11, maxiter=5000)
    u_unprec = solve_sipg_poisson_2d(
        mesh, source, bc_types=bc_types, iterative=cfg,
    )
    pc = make_sipg_block_jacobi_preconditioner(mesh, bc_types=bc_types)
    u_prec = solve_sipg_poisson_2d(
        mesh, source, bc_types=bc_types, iterative=cfg, preconditioner=pc,
    )
    err = float(jnp.max(jnp.abs(u_unprec - u_prec)))
    rel = err / float(jnp.max(jnp.abs(u_unprec)))
    print(f"  ||u_unprec - u_prec||_∞ = {err:.2e}  (rel {rel:.2e})")
    assert rel < 1e-7


def test_sipg_pc_matches_lu():
    mesh, bc_types = _mesh_and_bcs(N=2, Kx=4, Ky=4)
    rng = np.random.default_rng(1)
    source = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))

    A_factored = factor_sipg_2d(mesh, bc_types=bc_types)
    u_lu = solve_sipg_poisson_2d(
        mesh, source, bc_types=bc_types, A_factored=A_factored,
    )
    pc = make_sipg_block_jacobi_preconditioner(mesh, bc_types=bc_types)
    cfg = IterativeConfig(method="cg", tol=1e-11, maxiter=5000)
    u_pc = solve_sipg_poisson_2d(
        mesh, source, bc_types=bc_types, iterative=cfg, preconditioner=pc,
    )
    err = float(jnp.max(jnp.abs(u_lu - u_pc)))
    rel = err / float(jnp.max(jnp.abs(u_lu)))
    print(f"  ||u_lu - u_pc||_∞ = {err:.2e}  (rel {rel:.2e})")
    assert rel < 1e-7


def test_helmholtz_pc_matches_lu():
    mesh, bc_types = _mesh_and_bcs(N=2, Kx=4, Ky=4)
    rng = np.random.default_rng(2)
    source = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    alpha, nu = 50.0, 0.01

    A_factored = factor_sipg_helmholtz_2d(
        mesh, alpha=alpha, nu=nu, bc_types=bc_types,
    )
    u_lu = solve_sipg_helmholtz_2d(
        mesh, source, alpha=alpha, nu=nu, bc_types=bc_types,
        A_factored=A_factored,
    )
    pc = make_helmholtz_block_jacobi_preconditioner(
        mesh, alpha=alpha, nu=nu, bc_types=bc_types,
    )
    cfg = IterativeConfig(method="cg", tol=1e-11, maxiter=5000)
    u_pc = solve_sipg_helmholtz_2d(
        mesh, source, alpha=alpha, nu=nu, bc_types=bc_types,
        iterative=cfg, preconditioner=pc,
    )
    err = float(jnp.max(jnp.abs(u_lu - u_pc)))
    rel = err / float(jnp.max(jnp.abs(u_lu)))
    print(f"  ||u_lu - u_pc||_∞ = {err:.2e}  (rel {rel:.2e})")
    assert rel < 1e-7


def test_grad_through_pc_matches_grad_through_lu():
    """Differentiability invariant: gradient of a scalar loss through the
    preconditioned CG solve agrees with gradient through the LU solve.
    """
    mesh, bc_types = _mesh_and_bcs(N=2, Kx=3, Ky=3)

    def loss_lu(source):
        A_factored = factor_sipg_2d(mesh, bc_types=bc_types)
        u = solve_sipg_poisson_2d(
            mesh, source, bc_types=bc_types, A_factored=A_factored,
        )
        return jnp.sum(u * u)

    def loss_pc(source):
        pc = make_sipg_block_jacobi_preconditioner(mesh, bc_types=bc_types)
        cfg = IterativeConfig(method="cg", tol=1e-11, maxiter=5000)
        u = solve_sipg_poisson_2d(
            mesh, source, bc_types=bc_types, iterative=cfg, preconditioner=pc,
        )
        return jnp.sum(u * u)

    rng = np.random.default_rng(3)
    source = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    g_lu = jax.grad(loss_lu)(source)
    g_pc = jax.grad(loss_pc)(source)
    err = float(jnp.max(jnp.abs(g_lu - g_pc)))
    rel = err / float(jnp.max(jnp.abs(g_lu)))
    print(f"  ||grad_lu - grad_pc||_∞ = {err:.2e}  (rel {rel:.2e})")
    assert rel < 1e-6


if __name__ == "__main__":
    print("test_sipg_pc_matches_unpreconditioned"); test_sipg_pc_matches_unpreconditioned()
    print("test_sipg_pc_matches_lu"); test_sipg_pc_matches_lu()
    print("test_helmholtz_pc_matches_lu"); test_helmholtz_pc_matches_lu()
    print("test_grad_through_pc_matches_grad_through_lu"); test_grad_through_pc_matches_grad_through_lu()
    print("all ok")
