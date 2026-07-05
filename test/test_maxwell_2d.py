"""2D Maxwell TM mode: standing-wave eigenmode + energy conservation."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg2d.maxwell import solve_maxwell_2d
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _energy(mesh, Hx, Hy, Ez) -> float:
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)
    return 0.5 * float(
        jnp.sum(mesh.J * (Hx * (M_ref @ Hx) + Hy * (M_ref @ Hy) + Ez * (M_ref @ Ez)))
    )


def _make_eigenmode_mesh(K: int, N: int):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, math.pi, 0.0, math.pi, K, K)
    return build_mesh_2d(VX, VY, EToV, N=N)


@pytest.mark.parametrize("K,N,bound", [(8, 4, 1e-5), (16, 4, 1e-6), (16, 6, 1e-10)])
def test_standing_wave_eigenmode(K, N, bound):
    """``Ez = sin(x)sin(y)`` standing wave returns to ``-Ez_0`` after a half period."""
    mesh = _make_eigenmode_mesh(K, N)
    Ez0 = jnp.sin(mesh.x) * jnp.sin(mesh.y)
    Hx0 = jnp.zeros_like(mesh.x)
    Hy0 = jnp.zeros_like(mesh.x)
    T_half = math.pi / math.sqrt(2.0)
    _, _, Ez = solve_maxwell_2d(mesh, Hx0, Hy0, Ez0, T_half)
    Ez_exact = -Ez0
    rel_err = float(jnp.linalg.norm(Ez - Ez_exact) / jnp.linalg.norm(Ez_exact))
    assert rel_err < bound, f"K={K} N={N}: rel_err = {rel_err:.3e}"


def test_full_period_returns_to_initial_state():
    """Ez(t = 2pi/omega) = Ez_0."""
    mesh = _make_eigenmode_mesh(K=16, N=4)
    Ez0 = jnp.sin(mesh.x) * jnp.sin(mesh.y)
    Hx0 = jnp.zeros_like(mesh.x)
    Hy0 = jnp.zeros_like(mesh.x)
    T_full = 2.0 * math.pi / math.sqrt(2.0)
    _, _, Ez = solve_maxwell_2d(mesh, Hx0, Hy0, Ez0, T_full)
    rel_err = float(jnp.linalg.norm(Ez - Ez0) / jnp.linalg.norm(Ez0))
    assert rel_err < 1e-5


def test_energy_drift_is_small_over_many_periods():
    """Upwind flux is mildly dissipative; energy should drift slowly."""
    mesh = _make_eigenmode_mesh(K=12, N=4)
    Ez0 = jnp.sin(mesh.x) * jnp.sin(mesh.y)
    Hx0 = jnp.zeros_like(mesh.x)
    Hy0 = jnp.zeros_like(mesh.x)
    e_start = _energy(mesh, Hx0, Hy0, Ez0)
    Hx, Hy, Ez = solve_maxwell_2d(mesh, Hx0, Hy0, Ez0, t_final=4 * math.pi / math.sqrt(2.0))
    e_end = _energy(mesh, Hx, Hy, Ez)
    rel_drift = (e_start - e_end) / e_start
    # Mildly dissipative — energy decreases but not too fast.
    assert 0.0 <= rel_drift < 0.05, f"drift = {rel_drift:.3e}"
