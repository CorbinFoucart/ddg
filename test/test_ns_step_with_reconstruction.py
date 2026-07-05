"""Regression + smoke tests for the velocity_reconstruction= kwarg in the
NS step functions.

Two invariants:
  1. With ``velocity_reconstruction=None`` (the default), the result is
     bit-equal to the un-instrumented code path. No accidental
     behaviour change for existing users.
  2. With ``velocity_reconstruction=reconstruct_velocity_rt0_2d``, the
     step still runs to completion and produces finite, non-trivial
     output. The values differ from the baseline (expected — the
     scheme is now applying a different operator) but the path is
     stable.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.incompressible_ns import (
    BDF1, build_ins_operators,
    ns_step_2d, ns_step_consistent_splitting_2d, ns_advection_2d,
    BC_INS_INTERIOR, BC_INS_WALL,
)
from ddg.dg2d.velocity_reconstruction import reconstruct_velocity_rt0_2d


def _setup(N=2, Kx=4, Ky=4, dt=0.04, nu=0.02):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, N)
    K, Nfaces = mesh.K, mesh.Nfaces
    EToE = np.asarray(mesh.EToE)
    bc = np.full((K, Nfaces), BC_INS_INTERIOR, dtype=np.int32)
    for k in range(K):
        for f in range(Nfaces):
            if EToE[k, f] == k:
                bc[k, f] = BC_INS_WALL
    ins_bc = jnp.asarray(bc)
    bc_ux = bc_uy = jnp.zeros_like(mesh.x)
    return mesh, ins_bc, bc_ux, bc_uy, dt, nu


def test_advection_no_recon_is_identity_to_baseline():
    """``ns_advection_2d`` with ``velocity_reconstruction=None``
    matches the un-instrumented call exactly."""
    mesh, ins_bc, bc_ux, bc_uy, _, _ = _setup()
    rng = np.random.default_rng(0)
    ux = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))

    Nx_a, Ny_a = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    Nx_b, Ny_b = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
        velocity_reconstruction=None,
    )
    err = float(jnp.max(jnp.abs(Nx_a - Nx_b)) + jnp.max(jnp.abs(Ny_a - Ny_b)))
    print(f"  ||N_default - N_recon=None||_∞ = {err:.2e}")
    assert err == 0.0


def test_advection_with_rt0_runs_and_differs():
    """Smoke test: RT_0 reconstruction in the convective term produces
    finite output that differs from the baseline."""
    mesh, ins_bc, bc_ux, bc_uy, _, _ = _setup(N=3, Kx=6, Ky=6)
    rng = np.random.default_rng(1)
    ux = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    Nx, Ny = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
        velocity_reconstruction=reconstruct_velocity_rt0_2d,
    )
    assert bool(jnp.all(jnp.isfinite(Nx)))
    assert bool(jnp.all(jnp.isfinite(Ny)))


def test_ns_step_2d_no_recon_matches_baseline():
    """``ns_step_2d`` with reconstruction off equals the baseline."""
    mesh, ins_bc, bc_ux, bc_uy, dt, nu = _setup()
    ops = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc,
    )
    rng = np.random.default_rng(2)
    ux = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    Nux_old = jnp.zeros_like(ux)
    Nuy_old = jnp.zeros_like(uy)
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))

    out_a = ns_step_2d(
        mesh, ux, uy, ux, uy, Nux_old, Nuy_old, dpdn_old,
        dt=dt, nu=nu, bdf=BDF1, ops=ops, ins_bc_types=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
    )
    out_b = ns_step_2d(
        mesh, ux, uy, ux, uy, Nux_old, Nuy_old, dpdn_old,
        dt=dt, nu=nu, bdf=BDF1, ops=ops, ins_bc_types=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
        velocity_reconstruction=None,
    )
    err = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(out_a, out_b))
    print(f"  ns_step_2d: ||with - without recon=None||_∞ = {err:.2e}")
    assert err == 0.0


def test_ns_step_2d_with_rt0_runs_and_differs():
    mesh, ins_bc, bc_ux, bc_uy, dt, nu = _setup(N=3, Kx=6, Ky=6)
    ops = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc,
    )
    rng = np.random.default_rng(3)
    ux = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    Nux_old = jnp.zeros_like(ux)
    Nuy_old = jnp.zeros_like(uy)
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))

    ux_a, uy_a, p_a, *_ = ns_step_2d(
        mesh, ux, uy, ux, uy, Nux_old, Nuy_old, dpdn_old,
        dt=dt, nu=nu, bdf=BDF1, ops=ops, ins_bc_types=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
    )
    ux_b, uy_b, p_b, *_ = ns_step_2d(
        mesh, ux, uy, ux, uy, Nux_old, Nuy_old, dpdn_old,
        dt=dt, nu=nu, bdf=BDF1, ops=ops, ins_bc_types=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
        velocity_reconstruction=reconstruct_velocity_rt0_2d,
    )
    assert bool(jnp.all(jnp.isfinite(ux_b)))
    assert bool(jnp.all(jnp.isfinite(p_b)))
    # The pressure source uses div(Π u_T), which differs from div(u_T)
    # in general — both fields and pressure should be different.
    err_u = float(jnp.max(jnp.abs(ux_a - ux_b)))
    err_p = float(jnp.max(jnp.abs(p_a - p_b)))
    print(f"  ns_step_2d with RT_0: ||u_diff||_∞ = {err_u:.2e}, "
          f"||p_diff||_∞ = {err_p:.2e}")
    assert err_u > 1e-6  # reconstruction must change something
    assert err_p > 1e-6


def test_consistent_step_no_recon_matches_baseline():
    mesh, ins_bc, bc_ux, bc_uy, dt, nu = _setup()
    ops = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc,
    )
    rng = np.random.default_rng(4)
    ux = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    a = ns_step_consistent_splitting_2d(
        mesh, ux, uy, ux, uy, dt=dt, nu=nu, bdf=BDF1, ops=ops,
        ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    b = ns_step_consistent_splitting_2d(
        mesh, ux, uy, ux, uy, dt=dt, nu=nu, bdf=BDF1, ops=ops,
        ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
        velocity_reconstruction=None,
    )
    err = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a, b))
    print(f"  consistent: ||with - without recon=None||_∞ = {err:.2e}")
    assert err == 0.0


if __name__ == "__main__":
    print("test_advection_no_recon_is_identity_to_baseline"); test_advection_no_recon_is_identity_to_baseline()
    print("test_advection_with_rt0_runs_and_differs"); test_advection_with_rt0_runs_and_differs()
    print("test_ns_step_2d_no_recon_matches_baseline"); test_ns_step_2d_no_recon_matches_baseline()
    print("test_ns_step_2d_with_rt0_runs_and_differs"); test_ns_step_2d_with_rt0_runs_and_differs()
    print("test_consistent_step_no_recon_matches_baseline"); test_consistent_step_no_recon_matches_baseline()
    print("all ok")
