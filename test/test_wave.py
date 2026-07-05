"""1D wave equation (Maxwell-style first-order system) correctness."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg1d import build_mesh_1d, uniform_mesh_1d
from ddg.dg1d.wave import solve_wave, solve_wave_trajectory


def _energy(mesh, E, H, eps=1.0, mu=1.0) -> float:
    """``0.5 * sum_K J * (eps*E^T M E + mu*H^T M H)`` with reference mass M."""
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)
    return 0.5 * float(
        jnp.sum(mesh.J * (eps * (M_ref @ (E**2)) + mu * (M_ref @ (H**2))))
    )


def test_pec_wave_conserves_energy():
    """Wave in homogeneous medium with PEC walls preserves total energy."""
    VX, EToV = uniform_mesh_1d(-1.0, 1.0, 80)
    mesh = build_mesh_1d(VX, EToV, N=6)

    E0 = jnp.sin(jnp.pi * mesh.x) * (mesh.x < 0.0)
    H0 = jnp.zeros_like(mesh.x)

    E1, H1 = solve_wave(mesh, E0, H0, t_final=10.0)
    e_start = _energy(mesh, E0, H0)
    e_end = _energy(mesh, E1, H1)
    rel_drift = abs(e_end - e_start) / e_start
    assert rel_drift < 1e-5, f"energy drift = {rel_drift:.2e}"


def test_pec_wave_returns_to_initial_state_after_round_trip():
    """Round-trip time in [-1, 1] with c=1 is 2; the wave should refocus on E."""
    VX, EToV = uniform_mesh_1d(-1.0, 1.0, 80)
    mesh = build_mesh_1d(VX, EToV, N=6)

    E0 = jnp.sin(jnp.pi * mesh.x) * (mesh.x < 0.0)
    H0 = jnp.zeros_like(mesh.x)

    E_traj, H_traj, times = solve_wave_trajectory(mesh, E0, H0, t_final=2.0, n_frames=4)
    # At t = 2 (one round trip) the wave has refocused: max|H| ~ 0.
    assert float(jnp.max(jnp.abs(H_traj[-1]))) < 1e-3
    # And E recovers most of its initial amplitude.
    assert float(jnp.max(jnp.abs(E_traj[-1]))) > 0.99


@pytest.mark.parametrize("eps_ratio", [1.0, 2.0, 4.0])
def test_pec_wave_bounded_with_varying_permittivity(eps_ratio):
    """Heterogeneous medium (left eps=1, right eps=eps_ratio) stays bounded.

    Upwind flux is dissipative at the material interface, so energy is
    NOT conserved — but the simulation must remain stable and the field
    must not grow above the initial amplitude.
    """
    K = 60
    VX, EToV = uniform_mesh_1d(-1.0, 1.0, K)
    mesh = build_mesh_1d(VX, EToV, N=5)
    eps = jnp.where(mesh.x < 0.0, 1.0, eps_ratio)
    mu = jnp.ones_like(mesh.x)

    E0 = jnp.exp(-20.0 * (mesh.x + 0.5) ** 2)
    H0 = jnp.zeros_like(mesh.x)

    E1, H1 = solve_wave(mesh, E0, H0, t_final=2.0, eps=eps, mu=mu, cfl=0.5)
    amp_initial = float(jnp.max(jnp.abs(E0)))
    assert float(jnp.max(jnp.abs(E1))) <= 1.05 * amp_initial
    assert float(jnp.max(jnp.abs(H1))) <= 1.05 * amp_initial
