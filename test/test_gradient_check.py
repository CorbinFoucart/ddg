"""Finite-difference gradient checks: validate the autodiff is consistent
with the discretisation.

These tests exercise the same machinery the optimisation-style demos rely
on (``jax.value_and_grad`` through ``lax.scan`` of an RK4 step, through
``solve_poisson``'s linear solve, through ``update_mesh_vertices``, ...)
and check the resulting gradients against centred finite differences at
randomly-chosen directions.

For a smooth pipeline of size ``n``, the directional derivative
``(L(x + h v) - L(x - h v)) / (2 h)`` should equal ``g · v`` to
``O(h²)``. We tolerate ~1e-6 relative error at ``h = 1e-4`` — well above
floating-point noise but tight enough to catch any sign/scaling bug.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg1d import build_mesh_1d, uniform_mesh_1d, update_mesh_vertices
from ddg.dg1d.advection import advec_rhs
from ddg.dg1d.heat import heat_rhs
from ddg.dg1d.integrator import low_storage_rk4_step
from ddg.dg1d.poisson import poisson_ip_matrix


def _check_grad_against_fd(loss_fn, x0, *, h=1e-4, rng_seed=0, tol=1e-6):
    """Check ``jax.grad(loss_fn)(x0)`` against a centred FD directional
    derivative along a random direction."""
    rng = np.random.default_rng(rng_seed)
    v_np = rng.standard_normal(x0.shape)
    v_np /= np.linalg.norm(v_np)
    v = jnp.asarray(v_np)
    g = jax.grad(loss_fn)(x0)
    dot_autodiff = float(jnp.sum(g * v))
    plus = float(loss_fn(x0 + h * v))
    minus = float(loss_fn(x0 - h * v))
    dot_fd = (plus - minus) / (2 * h)
    rel_err = abs(dot_autodiff - dot_fd) / max(abs(dot_autodiff), 1e-12)
    return dot_autodiff, dot_fd, rel_err, tol


# ---------------------------------------------------------------------------
# Gradient through an RK4-scan forward of 1D advection (final-state QoI)
# ---------------------------------------------------------------------------

def test_grad_advection_wrt_u0_matches_fd():
    """``QoI = sum(u(T)^2)`` for periodic advection, gradient w.r.t. u0."""
    VX, EToV = uniform_mesh_1d(0.0, 2 * np.pi, 8)
    mesh = build_mesh_1d(VX, EToV, N=4, periodic=True)
    a = 2.0 * np.pi
    n_steps = 80
    dt = 0.2 / n_steps

    def forward(u0):
        rhs = partial(advec_rhs, mesh, a=a)
        def body(carry, _):
            u, t = carry
            u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
            return (u_new, t + dt), None
        (u_final, _), _ = jax.lax.scan(body, (u0, 0.0), jnp.arange(n_steps))
        return u_final

    def loss(u0):
        return jnp.sum(forward(u0) ** 2)

    u0 = jnp.sin(mesh.x)
    da, dfd, rel, tol = _check_grad_against_fd(loss, u0, h=1e-4, tol=1e-6)
    assert rel < tol, f"autodiff={da:.6e}  fd={dfd:.6e}  rel_err={rel:.2e}"


# ---------------------------------------------------------------------------
# Gradient through heat with time-varying Dirichlet BC
# ---------------------------------------------------------------------------

def test_grad_heat_wrt_bc_schedule_matches_fd():
    """``QoI = sum(u(T)^2)``, gradient w.r.t. piecewise-linear bc_left(t)."""
    VX, EToV = uniform_mesh_1d(0.0, 1.0, 8)
    mesh = build_mesh_1d(VX, EToV, N=3)
    n_steps = 200
    T = 0.05
    dt = T / n_steps
    t_control = jnp.linspace(0.0, T, 5)

    def forward(g_values):
        bc_left = lambda t: jnp.interp(t, t_control, g_values)
        rhs = partial(heat_rhs, mesh, kappa=1.0, bc_left_fn=bc_left)
        u0 = jnp.zeros_like(mesh.x)
        def body(carry, _):
            u, t = carry
            u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
            return (u_new, t + dt), None
        (u_final, _), _ = jax.lax.scan(body, (u0, 0.0), jnp.arange(n_steps))
        return u_final

    def loss(g):
        return jnp.sum(forward(g) ** 2)

    g0 = jnp.array([0.0, 0.3, 0.6, 0.4, 0.0])
    da, dfd, rel, tol = _check_grad_against_fd(loss, g0, h=1e-5, tol=1e-5)
    assert rel < tol, f"autodiff={da:.6e}  fd={dfd:.6e}  rel_err={rel:.2e}"


# ---------------------------------------------------------------------------
# Gradient through Poisson assembly + linear solve, w.r.t. source f
# ---------------------------------------------------------------------------

def test_grad_poisson_wrt_source_matches_fd():
    """``QoI = sum(u^2)``, gradient w.r.t. source ``f`` in -u'' = f."""
    VX, EToV = uniform_mesh_1d(0.0, 1.0, 6)
    mesh = build_mesh_1d(VX, EToV, N=3)

    A = poisson_ip_matrix(mesh)
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)

    def loss(f_field):
        Mf = mesh.J * (M_ref @ f_field)
        u_flat = jnp.linalg.solve(A, Mf.T.reshape(-1))
        u = u_flat.reshape(mesh.K, mesh.Np).T
        return jnp.sum(u ** 2)

    f0 = jnp.exp(-((mesh.x - 0.5) ** 2) * 30.0)
    da, dfd, rel, tol = _check_grad_against_fd(loss, f0, h=1e-5, tol=1e-7)
    assert rel < tol, f"autodiff={da:.6e}  fd={dfd:.6e}  rel_err={rel:.2e}"


# ---------------------------------------------------------------------------
# Gradient through update_mesh_vertices: shape-opt-style sensitivity
# ---------------------------------------------------------------------------

def test_grad_through_mesh_vertices_matches_fd():
    """``QoI = sum(J^2)``, gradient w.r.t. interior vertex positions."""
    VX0, EToV = uniform_mesh_1d(-1.0, 1.0, 6)
    mesh_init = build_mesh_1d(VX0, EToV, N=3)

    def loss(vx_interior):
        VX = jnp.concatenate([
            jnp.array([-1.0]),
            vx_interior,
            jnp.array([1.0]),
        ])
        mesh = update_mesh_vertices(mesh_init, VX)
        return jnp.sum(mesh.J ** 2)

    # Slightly perturbed interior vertices.
    vx0 = jnp.linspace(-1.0, 1.0, 7)[1:-1] + 0.01 * jnp.array([1, -1, 1, -1, 1])
    da, dfd, rel, tol = _check_grad_against_fd(loss, vx0, h=1e-5, tol=1e-7)
    assert rel < tol, f"autodiff={da:.6e}  fd={dfd:.6e}  rel_err={rel:.2e}"
