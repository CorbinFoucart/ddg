"""1D wave equation written as a Maxwell-style first-order system.

We solve
    E_t = -(1/eps) H_x
    H_t = -(1/mu)  E_x
which (combining) gives the scalar wave equation
    E_tt = (1/(eps*mu)) E_xx,
with wave speed ``c = 1/sqrt(eps*mu)``. Boundary handling defaults to
PEC walls (``E = 0`` at domain boundary); for a periodic mesh the
boundary updates are no-ops.

Direct port of ``Codes1.1/Codes1D/MaxwellRHS1D.m`` and ``Maxwell1D.m``
from the nodal-dg reference.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ddg.dg1d.integrator import low_storage_rk4, low_storage_rk4_trajectory
from ddg.dg1d.mesh import Mesh1D


def _flatten_f(a: jax.Array) -> jax.Array:
    """Column-major flatten of (Np, K) into (Np*K,), matching MATLAB."""
    return a.T.reshape(-1)


def _to_field(value: float | jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Broadcast a scalar or array material parameter to ``shape`` (Np, K)."""
    return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), shape)


def wave_rhs(
    mesh: Mesh1D,
    E: jax.Array,
    H: jax.Array,
    t: float,
    eps: float | jax.Array = 1.0,
    mu: float | jax.Array = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Semi-discrete RHS for the 1D Maxwell-style wave system.

    ``E``, ``H`` have shape ``(Np, K)``. ``eps`` and ``mu`` are either
    scalars or arrays broadcastable to ``(Np, K)`` for materials that
    vary in space (or jump across elements). Returns ``(rhsE, rhsH)``.
    """
    eps_field = _to_field(eps, E.shape)
    mu_field = _to_field(mu, E.shape)

    Z = jnp.sqrt(mu_field / eps_field)  # impedance, (Np, K)
    Y = 1.0 / Z                          # admittance

    E_flat = _flatten_f(E)
    H_flat = _flatten_f(H)
    Z_flat = _flatten_f(Z)
    Y_flat = _flatten_f(Y)

    Em = E_flat[mesh.vmapM]
    Ep = E_flat[mesh.vmapP]
    Hm = H_flat[mesh.vmapM]
    Hp = H_flat[mesh.vmapP]
    Zm = Z_flat[mesh.vmapM]
    Zp = Z_flat[mesh.vmapP]
    Ym = Y_flat[mesh.vmapM]
    Yp = Y_flat[mesh.vmapP]

    dE = Em - Ep
    dH = Hm - Hp

    # PEC boundary: Ebc = -E(vmapB) (Dirichlet E=0), Hbc = H(vmapB) (Neumann).
    # => dE(mapB) = 2 * E(vmapB);  dH(mapB) = 0.
    if mesh.mapB.size > 0:
        E_bdy = E_flat[mesh.vmapB]
        dE = dE.at[mesh.mapB].set(2.0 * E_bdy)
        dH = dH.at[mesh.mapB].set(0.0)

    nx_flat = mesh.nx.reshape(-1, order="F")
    flux_E = (nx_flat * Zp * dH - dE) / (Zm + Zp)
    flux_H = (nx_flat * Yp * dE - dH) / (Ym + Yp)

    Nfp, Nfaces, K = mesh.Nfp, mesh.Nfaces, mesh.K
    flux_E_face = flux_E.reshape((Nfp * Nfaces, K), order="F")
    flux_H_face = flux_H.reshape((Nfp * Nfaces, K), order="F")

    rhs_E = (-mesh.rx * (mesh.Dr @ H) + mesh.LIFT @ (mesh.Fscale * flux_E_face)) / eps_field
    rhs_H = (-mesh.rx * (mesh.Dr @ E) + mesh.LIFT @ (mesh.Fscale * flux_H_face)) / mu_field
    return rhs_E, rhs_H


def _wave_dt(mesh: Mesh1D, eps: float | jax.Array, mu: float | jax.Array, cfl: float) -> float:
    """CFL-limited time step for the wave system."""
    xmin_spacing = float(jnp.min(jnp.abs(mesh.x[1, :] - mesh.x[0, :])))
    c_max = 1.0 / float(jnp.sqrt(jnp.min(_to_field(eps, mesh.x.shape) * _to_field(mu, mesh.x.shape))))
    return cfl * xmin_spacing / c_max


def solve_wave(
    mesh: Mesh1D,
    E0: jax.Array,
    H0: jax.Array,
    t_final: float,
    *,
    eps: float | jax.Array = 1.0,
    mu: float | jax.Array = 1.0,
    cfl: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Integrate the 1D wave system to ``t_final``. Returns ``(E, H)``."""
    dt = _wave_dt(mesh, eps, mu, cfl)
    rhs = lambda state, t: wave_rhs(mesh, state[0], state[1], t, eps=eps, mu=mu)
    E_final, H_final = low_storage_rk4(rhs, (E0, H0), 0.0, t_final, dt)
    return E_final, H_final


def solve_wave_trajectory(
    mesh: Mesh1D,
    E0: jax.Array,
    H0: jax.Array,
    t_final: float,
    n_frames: int,
    *,
    eps: float | jax.Array = 1.0,
    mu: float | jax.Array = 1.0,
    cfl: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Integrate the wave system and return ``(E_traj, H_traj, times)``.

    ``E_traj`` and ``H_traj`` each have shape ``(n_frames+1, Np, K)``.
    """
    dt = _wave_dt(mesh, eps, mu, cfl)
    rhs = lambda state, t: wave_rhs(mesh, state[0], state[1], t, eps=eps, mu=mu)
    (E_traj, H_traj), times = low_storage_rk4_trajectory(
        rhs, (E0, H0), 0.0, t_final, dt, n_frames
    )
    return E_traj, H_traj, times
