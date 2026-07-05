"""Tests for the H(div)-style velocity reconstructions.

Two reconstruction operators are tested:

  - ``reconstruct_velocity_rt0_2d``           (proper Linke / RT_0)
  - ``reconstruct_velocity_face_normal_avg_2d`` (cheap face-node averaging)

Key correctness criterion for the RT_0 operator: the *integrated*
normal-flux jump on every interior face must be machine zero. The
pointwise jump is not annihilated (RT_0 has a linear normal trace; our
P_N basis has more DOFs), and the cheap version doesn't annihilate
either — both are kept as baselines for comparison.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.velocity_reconstruction import (
    reconstruct_velocity_rt0_2d,
    reconstruct_velocity_face_normal_avg_2d,
    face_normal_jump_l2_2d,
    face_normal_flux_moment_jump_2d,
)


def _mesh(N=3, Kx=4, Ky=4):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


# ---- RT_0 reconstruction tests ------------------------------------------

def test_rt0_constant_field_invariant():
    """A globally constant velocity has zero face-flux moments — RT_0
    projection must leave it unchanged."""
    mesh = _mesh(N=2, Kx=4, Ky=4)
    ux = jnp.ones_like(mesh.x) * 0.7
    uy = jnp.ones_like(mesh.x) * -0.3
    ux_rt, uy_rt = reconstruct_velocity_rt0_2d(mesh, ux, uy)
    err_x = float(jnp.max(jnp.abs(ux_rt - ux)))
    err_y = float(jnp.max(jnp.abs(uy_rt - uy)))
    print(f"  constant: ||u_rt − u||_∞ = ({err_x:.2e}, {err_y:.2e})")
    # Constant fields lie in RT_0 — projection is identity, but only
    # up to discretisation: the RT_0 representative of a constant is
    # (-y, x)-type? Actually: u = const is divergence-free, so RT_0
    # must reproduce it. But our nodal basis evaluates Π u at
    # warp-blend nodes, not at the original const value. The
    # disagreement here measures how well RT_0 represents the
    # constant. For affine elements it should be exact.
    assert err_x < 1e-10
    assert err_y < 1e-10


def test_rt0_kills_face_flux_moment_jump():
    """An arbitrary discontinuous field has non-zero face-flux moments
    on interior faces; RT_0 projection must annihilate them.
    """
    mesh = _mesh(N=2, Kx=6, Ky=6)
    rng = np.random.default_rng(0)
    per_elem_x = jnp.asarray(rng.standard_normal(mesh.K))
    per_elem_y = jnp.asarray(rng.standard_normal(mesh.K))
    ux = jnp.broadcast_to(per_elem_x[None, :], (mesh.Np, mesh.K))
    uy = jnp.broadcast_to(per_elem_y[None, :], (mesh.Np, mesh.K))

    moment_before = face_normal_flux_moment_jump_2d(mesh, ux, uy)
    ux_rt, uy_rt = reconstruct_velocity_rt0_2d(mesh, ux, uy)
    moment_after = face_normal_flux_moment_jump_2d(mesh, ux_rt, uy_rt)
    print(f"  flux-moment jump²: {moment_before:.4e} -> {moment_after:.4e}")
    # The RT_0 projection's defining property — integrated normal-flux
    # jumps go to machine zero.
    assert moment_after < 1e-18
    assert moment_before > 1e-3  # sanity: input had real jumps


def test_rt0_is_differentiable():
    """``jax.grad`` of a scalar functional through RT_0 reconstruction
    agrees with finite differences.
    """
    mesh = _mesh(N=2, Kx=3, Ky=3)
    rng = np.random.default_rng(2)
    ux0 = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy0 = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))

    def loss(ux, uy):
        ux_r, uy_r = reconstruct_velocity_rt0_2d(mesh, ux, uy)
        return jnp.sum(ux_r ** 2 + uy_r ** 2)

    g_ux, g_uy = jax.grad(loss, argnums=(0, 1))(ux0, uy0)
    eps = 1e-5
    i, k = 0, 0
    fd = (loss(ux0.at[i, k].add(eps), uy0)
          - loss(ux0.at[i, k].add(-eps), uy0)) / (2 * eps)
    ad = float(g_ux[i, k])
    err = float(jnp.abs(ad - fd))
    print(f"  ad = {ad:.4e}  fd = {float(fd):.4e}  err = {err:.2e}")
    assert err < 1e-6


# ---- Face-normal-average reconstruction (cheap baseline) ------------------

def test_face_avg_reduces_pointwise_jump():
    """The cheap face-normal-average reduces (but does not annihilate)
    the pointwise jump. Used here as a baseline for the comparison
    against the proper RT_0 reconstruction.
    """
    mesh = _mesh(N=2, Kx=6, Ky=6)
    rng = np.random.default_rng(0)
    per_elem = jnp.asarray(rng.standard_normal(mesh.K))
    ux = jnp.broadcast_to(per_elem[None, :], (mesh.Np, mesh.K))
    uy = jnp.zeros_like(ux)

    jump_before = face_normal_jump_l2_2d(mesh, ux, uy)
    ux_a, uy_a = reconstruct_velocity_face_normal_avg_2d(mesh, ux, uy)
    jump_after = face_normal_jump_l2_2d(mesh, ux_a, uy_a)
    print(f"  pointwise jump²: {jump_before:.4f} -> {jump_after:.4e}  "
          f"(reduction {jump_before / max(jump_after, 1e-30):.1f}x)")
    # Not machine zero (corner-node ambiguity) but should be reduced.
    assert jump_after < 0.5 * jump_before


# ---- Comparison ----------------------------------------------------------

def test_rt0_vs_face_avg_on_moment_metric():
    """The metric that actually matters for pressure-robustness — the
    flux moment — is what RT_0 annihilates and face-average does not.
    Side-by-side comparison."""
    mesh = _mesh(N=3, Kx=6, Ky=6)
    rng = np.random.default_rng(7)
    ux = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))

    moment_orig = face_normal_flux_moment_jump_2d(mesh, ux, uy)
    ux_avg, uy_avg = reconstruct_velocity_face_normal_avg_2d(mesh, ux, uy)
    moment_avg = face_normal_flux_moment_jump_2d(mesh, ux_avg, uy_avg)
    ux_rt, uy_rt = reconstruct_velocity_rt0_2d(mesh, ux, uy)
    moment_rt = face_normal_flux_moment_jump_2d(mesh, ux_rt, uy_rt)
    print(f"  flux-moment jump²:")
    print(f"    original:           {moment_orig:.4e}")
    print(f"    face-normal-avg:    {moment_avg:.4e}")
    print(f"    RT_0 reconstruction:{moment_rt:.4e}")
    assert moment_rt < 1e-18
    assert moment_avg > 1e-6   # face-average does NOT annihilate moments


if __name__ == "__main__":
    print("test_rt0_constant_field_invariant"); test_rt0_constant_field_invariant()
    print("test_rt0_kills_face_flux_moment_jump"); test_rt0_kills_face_flux_moment_jump()
    print("test_rt0_is_differentiable"); test_rt0_is_differentiable()
    print("test_face_avg_reduces_pointwise_jump"); test_face_avg_reduces_pointwise_jump()
    print("test_rt0_vs_face_avg_on_moment_metric"); test_rt0_vs_face_avg_on_moment_metric()
    print("all ok")
