"""Correctness checks for the DG scalar advection-diffusion solver.

Three properties:

  1. Pure diffusion of a Fourier mode decays at the analytic rate
     ``exp(-κ k² t)`` — verifies the implicit SIPG diffusion solve and
     the BDF time integration.
  2. The advection operator ``∇·(u c)`` integrates to zero over a
     periodic domain — discrete conservation of the total scalar
     under a divergence-free transport velocity.
  3. A uniform scalar is left exactly uniform by a transport step
     (advection + diffusion of a constant is zero).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import (
    apply_rectangle_periodic_2d, build_mesh_2d, uniform_rectangle_mesh_2d,
)
from ddg.dg2d.incompressible_ns import BDF1, BDF2, _div_2d
from ddg.dg2d.sipg import _local_mass
from ddg.dg2d.scalar_transport import (
    build_scalar_operators, scalar_advection_2d, scalar_transport_step_2d,
)


def _scalar_advection_signed(mesh, c, ux, uy, surf_sign):
    """Mirror of ``scalar_advection_2d`` with the surface-flux sign exposed.

    ``surf_sign = -1`` reproduces the (correct) production operator bitwise;
    ``surf_sign = +1`` is the pre-fix anti-diffusive sign that made the
    semi-discrete operator L²-unstable. Used by the regression + xfail tests
    below to pin both the correct behaviour and the historical bug.
    """
    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K
    vol = _div_2d(mesh, ux * c, uy * c)
    c_flat = c.T.reshape(-1)
    ux_flat = ux.T.reshape(-1); uy_flat = uy.T.reshape(-1)
    c_M = c_flat[mesh.vmapM]; c_P = c_flat[mesh.vmapP]
    ux_M = ux_flat[mesh.vmapM]; ux_P = ux_flat[mesh.vmapP]
    uy_M = uy_flat[mesh.vmapM]; uy_P = uy_flat[mesh.vmapP]
    nx_flat = mesh.nx.reshape(-1, order="F"); ny_flat = mesh.ny.reshape(-1, order="F")
    udotn_M = ux_M * nx_flat + uy_M * ny_flat
    udotn_P = ux_P * nx_flat + uy_P * ny_flat
    fM = udotn_M * c_M; fP = udotn_P * c_P
    lam_node = jnp.maximum(jnp.abs(udotn_M), jnp.abs(udotn_P))
    lam_face = lam_node.reshape((mesh.Nfp, mesh.Nfaces * K), order="F")
    lam_face_max = jnp.max(lam_face, axis=0, keepdims=True)
    lam_max = jnp.broadcast_to(lam_face_max, lam_face.shape).reshape(-1, order="F")
    flux = surf_sign * (0.5 * (fM - fP) + 0.5 * lam_max * (c_P - c_M))
    flux_face = flux.reshape((Nfp_Nf, K), order="F")
    return vol + mesh.LIFT @ (mesh.Fscale * flux_face)


def _rk4_advect_l2_ratio(mesh, M_ref, ux, uy, c0, dt, n_steps, advect):
    """March ``advect`` explicitly with RK4 (no diffusion) and return
    ‖c‖₂(final)/‖c‖₂(0). A correct upwind operator keeps this near 1."""
    def l2(cc):
        return float(jnp.sqrt(jnp.sum(mesh.J * (M_ref @ (cc * cc)))))
    c = c0; l0 = l2(c)
    for _ in range(n_steps):
        k1 = advect(c)
        k2 = advect(c - 0.5 * dt * k1)
        k3 = advect(c - 0.5 * dt * k2)
        k4 = advect(c - dt * k3)
        c = c - dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return l2(c) / l0


def _periodic_mesh(N=4, K=16):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, K, K)
    mesh = build_mesh_2d(VX, VY, EToV, N)
    return apply_rectangle_periodic_2d(
        mesh, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
    )


def test_pure_diffusion_decay_rate():
    """c = sin(2πx) under pure diffusion decays as exp(-κ(2π)²t)."""
    mesh = _periodic_mesh()
    kappa, dt, n_steps = 0.01, 2.0e-3, 50
    x = np.asarray(mesh.x)
    c = jnp.asarray(np.sin(2 * np.pi * x))
    bc = jnp.zeros((mesh.K, mesh.Nfaces), dtype=jnp.int32)
    ops1 = build_scalar_operators(mesh, dt=dt, kappa=kappa, g0=BDF1.g0,
                                  diffusion_bc_types=bc)
    ops2 = build_scalar_operators(mesh, dt=dt, kappa=kappa, g0=BDF2.g0,
                                  diffusion_bc_types=bc)
    zero_u = jnp.zeros((mesh.Np, mesh.K))
    c_old = c
    Nc_old = jnp.zeros((mesh.Np, mesh.K))
    for step in range(1, n_steps + 1):
        ops = ops1 if step == 1 else ops2
        bdf = BDF1 if step == 1 else BDF2
        c_new, Nc_n = scalar_transport_step_2d(
            mesh, c, c_old, zero_u, zero_u, Nc_old,
            dt=dt, kappa=kappa, bdf=bdf, ops=ops,
        )
        c_old, c, Nc_old = c, c_new, Nc_n
    t = n_steps * dt
    decay = np.exp(-kappa * (2 * np.pi) ** 2 * t)
    c_exact = decay * np.sin(2 * np.pi * x)
    err = float(jnp.max(jnp.abs(c - jnp.asarray(c_exact))))
    print(f"  analytic decay={decay:.5f}  max|c-c_exact|={err:.2e}")
    assert err < 1e-5


def test_advection_conserves_total_scalar():
    """∫ ∇·(u c) dx = 0 on a periodic domain (divergence-free u)."""
    mesh = _periodic_mesh()
    M_ref = _local_mass(mesh)
    x = np.asarray(mesh.x); y = np.asarray(mesh.y)
    c = jnp.asarray(np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))
    ux = jnp.ones((mesh.Np, mesh.K))
    uy = 0.5 * jnp.ones((mesh.Np, mesh.K))
    Nc = scalar_advection_2d(mesh, c, ux, uy)
    total = float(jnp.sum(mesh.J * (M_ref @ Nc)))
    print(f"  ∫ div(u c) = {total:.2e}")
    assert abs(total) < 1e-12


def test_uniform_scalar_stays_uniform():
    """A constant scalar is unchanged by a transport step."""
    mesh = _periodic_mesh()
    kappa, dt = 0.05, 1.0e-3
    bc = jnp.zeros((mesh.K, mesh.Nfaces), dtype=jnp.int32)
    ops = build_scalar_operators(mesh, dt=dt, kappa=kappa, g0=BDF1.g0,
                                 diffusion_bc_types=bc)
    c = jnp.full((mesh.Np, mesh.K), 2.71828)
    ux = 0.3 * jnp.ones((mesh.Np, mesh.K))
    uy = -0.7 * jnp.ones((mesh.Np, mesh.K))
    Nc_old = jnp.zeros((mesh.Np, mesh.K))
    c_new, _ = scalar_transport_step_2d(
        mesh, c, c, ux, uy, Nc_old,
        dt=dt, kappa=kappa, bdf=BDF1, ops=ops,
    )
    err = float(jnp.max(jnp.abs(c_new - c)))
    print(f"  max|c_new - c_const| = {err:.2e}")
    assert err < 1e-10


def _strained_flow_case():
    """Periodic mesh + strained (Taylor-Green cellular) div-free flow + a
    smooth scalar — the configuration that exposes the advection sign bug."""
    mesh = _periodic_mesh()
    M_ref = _local_mass(mesh)
    x = np.asarray(mesh.x); y = np.asarray(mesh.y)
    U = 0.3
    ux = jnp.asarray(U * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))
    uy = jnp.asarray(-U * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y))
    c = jnp.asarray(np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))
    return mesh, M_ref, ux, uy, c


def test_signed_helper_matches_production_operator():
    """``_scalar_advection_signed(surf_sign=-1)`` must equal the production
    ``scalar_advection_2d`` bitwise — so the xfail below genuinely pins the
    historical sign of the *real* operator, not a detached copy."""
    mesh, _, ux, uy, c = _strained_flow_case()
    a_lib = scalar_advection_2d(mesh, c, ux, uy)
    a_helper = _scalar_advection_signed(mesh, c, ux, uy, surf_sign=-1)
    err = float(jnp.max(jnp.abs(a_lib - a_helper)))
    print(f"  max|lib - helper(-1)| = {err:.2e}")
    assert err < 1e-13


def test_advection_is_L2_stable_under_strained_flow():
    """Multi-step advection must NOT grow ‖c‖₂ under a steady div-free flow.

    Regression for the surface-flux sign bug in ``scalar_advection_2d``:
    the upwind Lax-Friedrichs correction was added with the wrong sign,
    making the semi-discrete operator anti-dissipative (L²-unstable). The
    three older tests are all blind to it — conservation telescopes
    regardless of sign, the constant/uniform fields zero the jump, and the
    diffusion test has no advection. Here we march the bare operator
    explicitly (RK4, NO diffusion) under a strained Taylor-Green cellular
    flow; a correct upwind operator is L²-non-increasing, so ‖c‖₂ must stay
    bounded near its initial value.
    """
    mesh, M_ref, ux, uy, c = _strained_flow_case()
    ratio = _rk4_advect_l2_ratio(
        mesh, M_ref, ux, uy, c, dt=1.0e-3, n_steps=800,
        advect=lambda cc: scalar_advection_2d(mesh, cc, ux, uy),
    )
    print(f"  ‖c‖₂ ratio after 800 RK4 steps = {ratio:.4f}")
    assert np.isfinite(ratio) and ratio < 1.05


@pytest.mark.xfail(strict=True, reason="pre-fix anti-diffusive surface-flux "
                   "sign (surf_sign=+1) is L²-unstable: documents the bug")
def test_wrong_sign_advection_is_unstable():
    """The historical (wrong) surface-flux sign blows ‖c‖₂ up — kept as a
    strict-xfail record of the bug. If this ever stops failing (i.e. +1
    becomes stable), pytest flags it so the regression is revisited."""
    mesh, M_ref, ux, uy, c = _strained_flow_case()
    ratio = _rk4_advect_l2_ratio(
        mesh, M_ref, ux, uy, c, dt=1.0e-3, n_steps=800,
        advect=lambda cc: _scalar_advection_signed(mesh, cc, ux, uy, surf_sign=+1),
    )
    print(f"  ‖c‖₂ ratio (wrong sign) after 800 RK4 steps = {ratio:.4f}")
    assert np.isfinite(ratio) and ratio < 1.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("all ok")
