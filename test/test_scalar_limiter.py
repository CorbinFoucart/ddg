"""Unit tests for the QUARANTINED Zhang--Shu scalar limiter.

These validate the limiter in isolation before it is wired into the
scalar-transport step:
  1. conservation  -- the mass-weighted cell mean is preserved exactly;
  2. bound enforcement -- a deliberately overshooting field is brought
     back inside [c_lo, c_hi] at every node;
  3. identity on in-bounds data -- a field already within bounds is
     returned unchanged (theta = 1), so the limiter cannot degrade the
     high-order convergence rate on smooth data.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.scalar_limiter import cell_means, zhang_shu_limit


def _mesh(order=3, n=3):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, n, n)
    return build_mesh_2d(VX, VY, EToV, order)


def _mass_ref(mesh):
    # Nodal DG reference mass matrix M = inv(V V^T).
    V = mesh.V
    return jnp.linalg.inv(V @ V.T)


def test_conserves_cell_mean():
    mesh = _mesh()
    M = _mass_ref(mesh)
    rng = np.random.default_rng(0)
    # Field with big per-element oscillation -> limiter is active.
    c = jnp.asarray(0.5 + 3.0 * rng.standard_normal((mesh.Np, mesh.K)))
    c_lim = zhang_shu_limit(c, M, 0.0, 1.0)
    before = cell_means(c, M)
    after = cell_means(c_lim, M)
    assert float(jnp.max(jnp.abs(after - before))) < 1e-12


def test_enforces_bounds():
    mesh = _mesh()
    M = _mass_ref(mesh)
    rng = np.random.default_rng(1)
    c_lo, c_hi = 0.0, 1.0
    # Build a field whose mass-weighted cell mean is in-bounds (~0.5) but
    # whose nodal extrema overshoot: c = mu + d, d mass-mean-zero, large.
    d0 = jnp.asarray(5.0 * rng.standard_normal((mesh.Np, mesh.K)))
    d = d0 - cell_means(d0, M)[None, :]          # mass-mean-zero per element
    c = 0.5 + d
    # Confirm the test is non-trivial: there really is an overshoot.
    assert float(jnp.max(c)) > c_hi and float(jnp.min(c)) < c_lo
    c_lim = zhang_shu_limit(c, M, c_lo, c_hi)
    tol = 1e-12
    assert float(jnp.max(c_lim)) <= c_hi + tol
    assert float(jnp.min(c_lim)) >= c_lo - tol
    # mean still preserved
    assert float(jnp.max(jnp.abs(cell_means(c_lim, M) - cell_means(c, M)))) < 1e-12


def test_identity_on_in_bounds():
    mesh = _mesh()
    M = _mass_ref(mesh)
    rng = np.random.default_rng(2)
    # All nodal values strictly inside [0, 1] -> theta = 1 -> identity.
    c = jnp.asarray(0.2 + 0.6 * rng.random((mesh.Np, mesh.K)))
    assert float(jnp.max(c)) < 1.0 and float(jnp.min(c)) > 0.0
    c_lim = zhang_shu_limit(c, M, 0.0, 1.0)
    assert float(jnp.max(jnp.abs(c_lim - c))) < 1e-13


def test_smooth_field_unchanged():
    """A smooth in-range field (cosine bump) must pass through untouched
    so spatial accuracy is preserved."""
    mesh = _mesh(order=3, n=4)
    M = _mass_ref(mesh)
    x = np.asarray(mesh.x); y = np.asarray(mesh.y)
    c = jnp.asarray(0.5 + 0.4 * np.cos(np.pi * x) * np.cos(np.pi * y))  # in [0.1,0.9]
    c_lim = zhang_shu_limit(c, M, 0.0, 1.0)
    assert float(jnp.max(jnp.abs(c_lim - c))) < 1e-13


@pytest.mark.parametrize("signed", [False, True])
def test_signed_and_unsigned_ranges(signed):
    """Works for c in [0,1] (unsigned blob) and c in [-1,0] (the signed
    cosine-bump bubble scalar)."""
    mesh = _mesh()
    M = _mass_ref(mesh)
    rng = np.random.default_rng(3)
    if signed:
        c_lo, c_hi, mu = -1.0, 0.0, -0.5
    else:
        c_lo, c_hi, mu = 0.0, 1.0, 0.5
    d = jnp.asarray(4.0 * rng.standard_normal((mesh.Np, mesh.K)))
    d = d - cell_means(d, M)[None, :]
    c = mu + d
    c_lim = zhang_shu_limit(c, M, c_lo, c_hi)
    tol = 1e-12
    assert float(jnp.max(c_lim)) <= c_hi + tol
    assert float(jnp.min(c_lim)) >= c_lo - tol
    assert float(jnp.max(jnp.abs(cell_means(c_lim, M) - cell_means(c, M)))) < 1e-12
