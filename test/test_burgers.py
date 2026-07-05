"""1D Burgers: viscous convergence + inviscid monotonicity."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg1d import build_mesh_1d, uniform_mesh_1d
from ddg.dg1d.burgers import (
    _minmod_3,
    slope_limit_n,
    solve_burgers_1d,
    solve_burgers_1d_trajectory,
)


def _periodic_l2(mesh, u, u_exact):
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)
    diff = u - u_exact
    return float(jnp.sqrt(jnp.sum(mesh.J * (diff * (M_ref @ diff)))))


# ---------------------------------------------------------------------------
# minmod helper
# ---------------------------------------------------------------------------

def test_minmod_three_args():
    a = jnp.array([1.0, 1.0, -1.0, 2.0])
    b = jnp.array([2.0, 1.0, -2.0, -1.0])
    c = jnp.array([3.0, -0.5, -0.5, 3.0])
    expected = jnp.array([1.0, 0.0, -0.5, 0.0])  # last two cases mixed signs -> 0
    out = _minmod_3(a, b, c)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), atol=1e-12)


# ---------------------------------------------------------------------------
# Viscous Burgers: textbook traveling wave (Hesthaven driver)
# ---------------------------------------------------------------------------

def _hesthaven_traveling_wave(nu: float):
    """Returns ``u_exact(x, t)``, ``inflow_fn(t)``, ``outflow_fn(t)`` for the
    viscous Burgers traveling wave with x boundaries at -1 and +1."""
    def u_exact(x, t):
        return -jnp.tanh((x + 0.5 - t) / (2 * nu)) + 1.0
    inflow = lambda t: u_exact(jnp.array(-1.0), t)
    outflow = lambda t: u_exact(jnp.array(1.0), t)
    return u_exact, inflow, outflow


def test_viscous_burgers_recovers_traveling_wave():
    """Hesthaven test: N=4, K=8, nu=0.1, T=1.5."""
    nu = 0.1
    VX, EToV = uniform_mesh_1d(-1.0, 1.0, 8)
    mesh = build_mesh_1d(VX, EToV, N=4)
    u_exact, inflow, outflow = _hesthaven_traveling_wave(nu)
    u0 = u_exact(mesh.x, 0.0)
    u_h = solve_burgers_1d(
        mesh, u0, t_final=1.5, nu=nu, inflow_fn=inflow, outflow_fn=outflow,
    )
    rel_err = float(jnp.linalg.norm(u_h - u_exact(mesh.x, 1.5))
                    / jnp.linalg.norm(u_exact(mesh.x, 1.5)))
    assert rel_err < 1e-3


@pytest.mark.parametrize("N", [2, 4, 6])
def test_viscous_burgers_p_convergence(N):
    """Increasing N drives the traveling-wave error down rapidly."""
    nu = 0.1
    VX, EToV = uniform_mesh_1d(-1.0, 1.0, 8)
    mesh = build_mesh_1d(VX, EToV, N=N)
    u_exact, inflow, outflow = _hesthaven_traveling_wave(nu)
    u0 = u_exact(mesh.x, 0.0)
    u_h = solve_burgers_1d(
        mesh, u0, t_final=1.5, nu=nu, inflow_fn=inflow, outflow_fn=outflow,
    )
    rel_err = float(jnp.linalg.norm(u_h - u_exact(mesh.x, 1.5))
                    / jnp.linalg.norm(u_exact(mesh.x, 1.5)))
    bounds = {2: 5e-3, 4: 1e-4, 6: 1e-5}
    assert rel_err < bounds[N], f"N={N}: rel_err={rel_err:.3e}"


# ---------------------------------------------------------------------------
# Inviscid Burgers: sin(x) shock with limiter
# ---------------------------------------------------------------------------

def test_inviscid_burgers_limited_is_monotonicity_preserving():
    """``u_0 = sin(x)`` on a periodic domain; after shock formation the
    limited solution must not overshoot the initial extrema."""
    VX, EToV = uniform_mesh_1d(0.0, 2.0 * jnp.pi, 32)
    mesh = build_mesh_1d(VX, EToV, N=4, periodic=True)
    u0 = jnp.sin(mesh.x)
    u_h = solve_burgers_1d(mesh, u0, t_final=1.5, nu=0.0, apply_limiter=True)
    u_max_init = float(jnp.max(jnp.abs(u0)))
    u_max_final = float(jnp.max(jnp.abs(u_h)))
    # Allow a small (~1%) numerical slack.
    assert u_max_final <= 1.01 * u_max_init
    # Solution stays finite.
    assert jnp.all(jnp.isfinite(u_h))


def test_inviscid_burgers_unlimited_overshoots_after_shock():
    """Without the limiter, ``u_0 = sin(x)`` shows Gibbs oscillations after
    the shock forms; amplitude exceeds the IC."""
    VX, EToV = uniform_mesh_1d(0.0, 2.0 * jnp.pi, 32)
    mesh = build_mesh_1d(VX, EToV, N=4, periodic=True)
    u0 = jnp.sin(mesh.x)
    u_h = solve_burgers_1d(mesh, u0, t_final=1.5, nu=0.0, apply_limiter=False)
    assert float(jnp.max(jnp.abs(u_h))) > 1.05  # Gibbs


def test_inviscid_burgers_conserves_mass_on_periodic_mesh():
    """Mass ``∫ u dx`` should be conserved exactly (up to round-off) on a
    periodic mesh — the slope limiter preserves cell averages."""
    VX, EToV = uniform_mesh_1d(0.0, 2.0 * jnp.pi, 24)
    mesh = build_mesh_1d(VX, EToV, N=3, periodic=True)
    u0 = jnp.sin(mesh.x) + 0.3  # nonzero mean so the diagnostic is meaningful
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)
    mass0 = float(jnp.sum(mesh.J * (M_ref @ u0)))
    u_h = solve_burgers_1d(mesh, u0, t_final=1.0, nu=0.0, apply_limiter=True)
    mass_final = float(jnp.sum(mesh.J * (M_ref @ u_h)))
    rel_drift = abs(mass_final - mass0) / abs(mass0)
    assert rel_drift < 1e-10


def test_burgers_trajectory_shape():
    """``solve_burgers_1d_trajectory`` returns ``(n_frames+1, Np, K)``."""
    VX, EToV = uniform_mesh_1d(-1.0, 1.0, 8)
    mesh = build_mesh_1d(VX, EToV, N=4)
    u_exact, inflow, outflow = _hesthaven_traveling_wave(0.1)
    u0 = u_exact(mesh.x, 0.0)
    traj, times = solve_burgers_1d_trajectory(
        mesh, u0, t_final=0.5, n_frames=10,
        nu=0.1, inflow_fn=inflow, outflow_fn=outflow,
    )
    assert traj.shape == (11, mesh.Np, mesh.K)
    assert times.shape == (11,)
