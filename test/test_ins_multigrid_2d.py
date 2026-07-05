"""Multigrid-preconditioned INS step matches the direct-LU step.

Phase C of the multigrid work: the p-multigrid V-cycle is wired into
the incompressible-NS step as the pressure / viscous preconditioner on
the matrix-free iterative path. This test takes one ``ns_step_2d`` two
ways --- the default dense LU factorisation, and the matrix-free CG
path preconditioned by ``make_ins_mg_preconditioners`` --- and checks
the fields agree (they solve the same linear systems, so to CG
tolerance the answers coincide).

The domain has an outflow face, so the pressure Poisson is
non-singular: the matrix-free CG path is meant for exactly such
open-flow problems (the production flow-past-a-cylinder run is one).
A fully closed domain gives an all-Neumann, singular pressure Poisson
that the iterative path does not yet handle --- a separate, pre-
existing limitation, independent of multigrid.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.sipg import IterativeConfig
from ddg.dg2d.incompressible_ns import (
    BC_INS_INTERIOR, BC_INS_OUTFLOW, BC_INS_WALL, BDF1, build_ins_operators,
    make_ins_mg_preconditioners, ns_advection_2d, ns_step_2d,
)


def _channel_setup(p=3, K=6):
    """A box with walls on three sides and an outflow on x = 1, so the
    pressure Poisson has a Dirichlet face and is non-singular."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, K, K)
    mesh = build_mesh_2d(VX, VY, EToV, p)
    EToE = np.asarray(mesh.EToE)
    Fmask = np.asarray(mesh.Fmask)
    x = np.asarray(mesh.x)
    bc = np.full((mesh.K, mesh.Nfaces), BC_INS_INTERIOR, dtype=np.int32)
    for k in range(mesh.K):
        for f in range(mesh.Nfaces):
            if EToE[k, f] != k:
                continue
            xf = x[Fmask[:, f], k]
            bc[k, f] = (BC_INS_OUTFLOW if np.all(np.abs(xf - 1.0) < 1e-7)
                        else BC_INS_WALL)
    return mesh, jnp.asarray(bc)


def test_mg_preconditioned_ins_step_matches_direct():
    """One INS step: iterative + p-MG path == dense-LU path."""
    mesh, ins_bc = _channel_setup(p=3, K=6)
    dt, nu = 0.01, 0.02
    bc_ux = jnp.zeros((mesh.Np, mesh.K))
    bc_uy = jnp.zeros((mesh.Np, mesh.K))

    # a smooth non-trivial initial velocity
    x, y = mesh.x, mesh.y
    ux = 0.2 * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
    uy = -0.2 * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy)
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    args = (mesh, ux, uy, ux, uy, Nux_old, Nuy_old, dpdn_old)
    kw = dict(dt=dt, nu=nu, bdf=BDF1, ins_bc_types=ins_bc,
              bc_ux=bc_ux, bc_uy=bc_uy)

    # --- direct LU ---
    ops_lu = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc)
    ux_lu, uy_lu, p_lu, *_ = ns_step_2d(*args, ops=ops_lu, **kw)

    # --- matrix-free CG + p-multigrid preconditioner ---
    ops_mf = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc,
        skip_factor=True)
    pre_p, pre_v = make_ins_mg_preconditioners(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc)
    ux_mg, uy_mg, p_mg, *_ = ns_step_2d(
        *args, ops=ops_mf,
        iterative=IterativeConfig(tol=1e-11, maxiter=3000),
        pressure_preconditioner=pre_p, viscous_preconditioner=pre_v, **kw)

    eu = float(jnp.linalg.norm(ux_mg - ux_lu) / jnp.linalg.norm(ux_lu))
    ev = float(jnp.linalg.norm(uy_mg - uy_lu) / jnp.linalg.norm(uy_lu))
    ep = float(jnp.linalg.norm(p_mg - p_lu) / jnp.linalg.norm(p_lu))
    print(f"  rel diff vs direct LU:  ux={eu:.2e}  uy={ev:.2e}  p={ep:.2e}")
    assert eu < 1e-6 and ev < 1e-6 and ep < 1e-6


def _closed_setup(p=3, K=6):
    """A fully closed box --- all walls --- so the pressure Poisson is
    all-Neumann and singular (nullspace = constants)."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, K, K)
    mesh = build_mesh_2d(VX, VY, EToV, p)
    EToE = np.asarray(mesh.EToE)
    is_bdy = EToE == np.arange(mesh.K)[:, None]
    bc = jnp.asarray(
        np.where(is_bdy, BC_INS_WALL, BC_INS_INTERIOR).astype(np.int32))
    return mesh, bc


def test_mg_ins_step_closed_singular_pressure():
    """Closed domain: the singular all-Neumann pressure Poisson. With
    the nullspace deflated (mean-zero RHS + ε-shift) the iterative +
    p-MG INS step is finite and tracks the dense-LU step --- before the
    deflation the matrix-free CG diverged to ~1e9."""
    mesh, ins_bc = _closed_setup(p=3, K=6)
    dt, nu = 0.01, 0.02
    bc_ux = jnp.zeros((mesh.Np, mesh.K))
    bc_uy = jnp.zeros((mesh.Np, mesh.K))
    x, y = mesh.x, mesh.y
    ux = 0.2 * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
    uy = -0.2 * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy)
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    args = (mesh, ux, uy, ux, uy, Nux_old, Nuy_old, dpdn_old)
    kw = dict(dt=dt, nu=nu, bdf=BDF1, ins_bc_types=ins_bc,
              bc_ux=bc_ux, bc_uy=bc_uy)

    ops_lu = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc)
    ux_lu, uy_lu, *_ = ns_step_2d(*args, ops=ops_lu, **kw)

    ops_mf = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc,
        skip_factor=True)
    pre_p, pre_v = make_ins_mg_preconditioners(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc)
    ux_mg, uy_mg, *_ = ns_step_2d(
        *args, ops=ops_mf,
        iterative=IterativeConfig(tol=1e-11, maxiter=3000),
        pressure_preconditioner=pre_p, viscous_preconditioner=pre_v, **kw)

    assert jnp.all(jnp.isfinite(ux_mg)) and jnp.all(jnp.isfinite(uy_mg))
    mx = float(jnp.max(jnp.abs(ux_mg)))
    eu = float(jnp.linalg.norm(ux_mg - ux_lu) / jnp.linalg.norm(ux_lu))
    ev = float(jnp.linalg.norm(uy_mg - uy_lu) / jnp.linalg.norm(uy_lu))
    print(f"  closed domain: max|ux_mg|={mx:.3f} (finite, not 1e9), "
          f"rel diff vs LU  ux={eu:.2e}  uy={ev:.2e}")
    # the two paths regularise the singular system differently (dense:
    # ε-shifted matrix; iterative: ε-shifted operator + deflated RHS),
    # so they agree up to that O(ε) gauge, not to CG tolerance.
    assert mx < 5.0
    assert eu < 2e-2 and ev < 2e-2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("all ok")
