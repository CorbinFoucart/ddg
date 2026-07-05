"""1D heat equation: Fourier mode decay + maximum principle."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg1d import build_mesh_1d, uniform_mesh_1d
from ddg.dg1d.heat import solve_heat


def _periodic_l2_error(mesh, u_h, u_exact) -> float:
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)
    diff = u_h - u_exact
    return float(jnp.sqrt(jnp.sum(mesh.J * (diff * (M_ref @ diff)))))


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_fourier_mode_decays_at_analytical_rate(mode):
    """sin(m*pi*x) decays as exp(-(m*pi)^2 kappa t) on (0, 1) with u=0 BCs."""
    VX, EToV = uniform_mesh_1d(0.0, 1.0, 16)
    mesh = build_mesh_1d(VX, EToV, N=5)
    kappa = 1.0
    t_final = 0.05

    u0 = jnp.sin(mode * jnp.pi * mesh.x)
    u_h = solve_heat(mesh, u0, t_final, kappa=kappa)
    u_exact = jnp.exp(-((mode * jnp.pi) ** 2) * kappa * t_final) * jnp.sin(
        mode * jnp.pi * mesh.x
    )

    norm_exact = _periodic_l2_error(mesh, u_exact, jnp.zeros_like(u_exact))
    rel_err = _periodic_l2_error(mesh, u_h, u_exact) / norm_exact
    assert rel_err < 1e-4, f"mode {mode}: rel_err = {rel_err:.3e}"


def test_max_principle_on_positive_initial_data():
    """Heat eq with Dirichlet u=0 and non-negative IC: solution stays non-negative
    and bounded above by initial max."""
    VX, EToV = uniform_mesh_1d(0.0, 1.0, 16)
    mesh = build_mesh_1d(VX, EToV, N=4)
    u0 = jnp.exp(-50.0 * (mesh.x - 0.5) ** 2)  # Gaussian, positive

    u_h = solve_heat(mesh, u0, t_final=0.05, kappa=1.0)
    assert float(jnp.min(u_h)) > -1e-3  # tiny Gibbs-style undershoot allowed
    assert float(jnp.max(u_h)) <= float(jnp.max(u0)) + 1e-10


def test_steady_state_is_zero_for_dirichlet_bcs():
    """Long time => solution decays to 0 with homogeneous Dirichlet."""
    VX, EToV = uniform_mesh_1d(0.0, 1.0, 8)
    mesh = build_mesh_1d(VX, EToV, N=4)
    u0 = jnp.sin(jnp.pi * mesh.x) + 0.5 * jnp.sin(3.0 * jnp.pi * mesh.x)

    u_h = solve_heat(mesh, u0, t_final=2.0, kappa=1.0)
    assert float(jnp.max(jnp.abs(u_h))) < 1e-6


def test_kappa_scales_decay_rate():
    """Doubling kappa should give the same end state at half the time."""
    VX, EToV = uniform_mesh_1d(0.0, 1.0, 16)
    mesh = build_mesh_1d(VX, EToV, N=4)
    u0 = jnp.sin(jnp.pi * mesh.x)

    u1 = solve_heat(mesh, u0, t_final=0.05, kappa=1.0)
    u2 = solve_heat(mesh, u0, t_final=0.025, kappa=2.0)
    err = float(jnp.linalg.norm(u1 - u2) / jnp.linalg.norm(u1))
    assert err < 1e-6
