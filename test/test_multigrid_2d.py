"""p-multigrid preconditioning for the SIPG operators.

A multigrid V-cycle should drive the preconditioned-CG iteration count
far below block-Jacobi's and keep it near degree-independent, since it
adds the coarse-level corrections that block-Jacobi (high-frequency
damping only) misses. These tests instrument a plain PCG to count
iterations and check:

  1. the p-MG-preconditioned solve reaches the same answer as a direct
     LU solve (the V-cycle is a correct preconditioner);
  2. on the SIPG Poisson operator, p-MG needs far fewer CG iterations
     than block-Jacobi at high degree, and its count barely grows with
     degree while block-Jacobi's climbs;
  3. it works for the Helmholtz operator (alpha > 0) too.
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
    BC_DIRICHLET, BC_INTERIOR, make_sipg_block_jacobi_preconditioner,
)
from ddg.dg2d.multigrid import (
    _operator_apply, make_sipg_hpmg_preconditioner,
    make_sipg_mg_preconditioner,
)


def _mesh(p, K=8):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, K, K)
    return build_mesh_2d(VX, VY, EToV, p)


def _all_dirichlet(mesh):
    EToE = np.asarray(mesh.EToE)
    is_bdy = EToE == np.arange(mesh.K)[:, None]
    bc = np.where(is_bdy, BC_DIRICHLET, BC_INTERIOR).astype(np.int32)
    return jnp.asarray(bc)


def _pcg(A, b, Minv, *, tol=1e-8, maxiter=2000):
    """Preconditioned CG; returns (solution, iterations)."""
    bnorm = float(jnp.linalg.norm(b))
    x = jnp.zeros_like(b)
    r = b
    z = Minv(r)
    p = z
    rz = float(jnp.vdot(r, z))
    for it in range(1, maxiter + 1):
        Ap = A(p)
        alpha = rz / float(jnp.vdot(p, Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        if float(jnp.linalg.norm(r)) / bnorm < tol:
            return x, it
        z = Minv(r)
        rz_new = float(jnp.vdot(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxiter


def _identity(v):
    return v


# 1 -----------------------------------------------------------------------

def test_pmg_preconditioned_solve_is_correct():
    """A p-MG-preconditioned CG solve matches a direct LU solve."""
    mesh = _mesh(p=4, K=8)
    bc = _all_dirichlet(mesh)
    A = _operator_apply(mesh, bc, 200.0, 0.0)
    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.standard_normal(mesh.Np * mesh.K))

    A_dense = jax.jacobian(A)(jnp.zeros(mesh.Np * mesh.K))
    x_direct = jnp.linalg.solve(A_dense, b)

    mg = make_sipg_mg_preconditioner(mesh, bc_types=bc, alpha=0.0)
    x_mg, iters = _pcg(A, b, mg, tol=1e-10)
    err = float(jnp.linalg.norm(x_mg - x_direct)
                / jnp.linalg.norm(x_direct))
    print(f"  p-MG solve: {iters} iters, rel err vs direct = {err:.2e}")
    assert err < 1e-7


# 2 -----------------------------------------------------------------------

def test_pmg_beats_block_jacobi_and_is_degree_robust():
    """p-MG needs far fewer CG iterations than block-Jacobi, and its
    count barely grows with polynomial degree."""
    pmg_iters, bj_iters, plain_iters = {}, {}, {}
    for p in (2, 3, 4, 5):
        mesh = _mesh(p=p, K=8)
        bc = _all_dirichlet(mesh)
        A = _operator_apply(mesh, bc, 200.0, 0.0)
        rng = np.random.default_rng(1)
        b = jnp.asarray(rng.standard_normal(mesh.Np * mesh.K))

        bj = make_sipg_block_jacobi_preconditioner(
            mesh, bc_types=bc, matfree=True)
        mg = make_sipg_mg_preconditioner(mesh, bc_types=bc, alpha=0.0)

        _, plain_iters[p] = _pcg(A, b, _identity)
        _, bj_iters[p] = _pcg(A, b, bj)
        _, pmg_iters[p] = _pcg(A, b, mg)
        print(f"  p={p}: CG iters  none={plain_iters[p]:4d}  "
              f"block-Jacobi={bj_iters[p]:4d}  p-MG={pmg_iters[p]:3d}")

    # multigrid clearly wins at high degree
    assert pmg_iters[5] < 0.5 * bj_iters[5]
    assert pmg_iters[4] < 0.5 * bj_iters[4]
    # near degree-independence: p-MG count stays bounded as p grows,
    # while block-Jacobi's climbs substantially
    assert pmg_iters[5] < 2.0 * pmg_iters[2]
    assert bj_iters[5] > 1.5 * bj_iters[2]


# 3 -----------------------------------------------------------------------

def test_pmg_handles_the_helmholtz_operator():
    """The same builder preconditions the Helmholtz operator (alpha>0)."""
    mesh = _mesh(p=4, K=8)
    bc = _all_dirichlet(mesh)
    alpha = 50.0
    A = _operator_apply(mesh, bc, 200.0, alpha)
    rng = np.random.default_rng(2)
    b = jnp.asarray(rng.standard_normal(mesh.Np * mesh.K))

    A_dense = jax.jacobian(A)(jnp.zeros(mesh.Np * mesh.K))
    x_direct = jnp.linalg.solve(A_dense, b)

    mg = make_sipg_mg_preconditioner(mesh, bc_types=bc, alpha=alpha)
    bj = make_sipg_block_jacobi_preconditioner(
        mesh, bc_types=bc, matfree=True)
    x_mg, mg_iters = _pcg(A, b, mg, tol=1e-10)
    _, bj_iters = _pcg(A, b, bj, tol=1e-10)
    err = float(jnp.linalg.norm(x_mg - x_direct)
                / jnp.linalg.norm(x_direct))
    print(f"  Helmholtz: p-MG {mg_iters} iters, block-Jacobi {bj_iters}, "
          f"rel err = {err:.2e}")
    assert err < 1e-7
    assert mg_iters <= bj_iters


# 4 -----------------------------------------------------------------------

def test_hpmg_preconditioned_solve_is_correct():
    """An hp-MG-preconditioned CG solve matches a direct LU solve."""
    base = _mesh(p=3, K=2)
    mg, finest = make_sipg_hpmg_preconditioner(
        base, 2, bc_fn=_all_dirichlet, alpha=0.0)
    bc = _all_dirichlet(finest)
    A = _operator_apply(finest, bc, 200.0, 0.0)
    rng = np.random.default_rng(4)
    b = jnp.asarray(rng.standard_normal(finest.Np * finest.K))

    A_dense = jax.jacobian(A)(jnp.zeros(finest.Np * finest.K))
    x_direct = jnp.linalg.solve(A_dense, b)
    x_mg, it = _pcg(A, b, mg, tol=1e-10)
    err = float(jnp.linalg.norm(x_mg - x_direct)
                / jnp.linalg.norm(x_direct))
    print(f"  hp-MG: K_finest={finest.K}, {it} iters, rel err = {err:.2e}")
    assert err < 1e-7


# 5 -----------------------------------------------------------------------

def test_hpmg_is_mesh_independent():
    """hp-MG keeps the CG iteration count near-flat as the mesh refines,
    where block-Jacobi's count climbs with every refinement."""
    hp_iters, bj_iters = {}, {}
    for n_h in (1, 2, 3):
        base = _mesh(p=3, K=2)
        mg, finest = make_sipg_hpmg_preconditioner(
            base, n_h, bc_fn=_all_dirichlet, alpha=0.0)
        bc = _all_dirichlet(finest)
        A = _operator_apply(finest, bc, 200.0, 0.0)
        rng = np.random.default_rng(5)
        b = jnp.asarray(rng.standard_normal(finest.Np * finest.K))

        bj = make_sipg_block_jacobi_preconditioner(
            finest, bc_types=bc, matfree=True)
        _, hp_iters[n_h] = _pcg(A, b, mg)
        _, bj_iters[n_h] = _pcg(A, b, bj)
        print(f"  n_h={n_h}: K_finest={finest.K:5d}  "
              f"hp-MG iters={hp_iters[n_h]:3d}  "
              f"block-Jacobi iters={bj_iters[n_h]:4d}")

    # hp-MG count stays bounded as the mesh refines; block-Jacobi grows
    assert hp_iters[3] < 1.6 * hp_iters[1]
    assert bj_iters[3] > 1.8 * bj_iters[1]
    assert hp_iters[3] < 0.3 * bj_iters[3]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("all ok")
