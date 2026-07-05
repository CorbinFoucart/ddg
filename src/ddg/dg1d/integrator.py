"""Low-storage 5-stage 4th-order Runge-Kutta (Carpenter & Kennedy).

Works for any pytree-shaped state (e.g. a single array, or a
``(E, H)`` tuple for coupled systems): arithmetic is dispatched through
``jax.tree.map`` so the same integrator powers scalar advection and the
two-field Maxwell wave system.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import jax
import jax.numpy as jnp


RK4A = jnp.array([
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
])
RK4B = jnp.array([
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
])
RK4C = jnp.array([
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0,
])


State = TypeVar("State")  # any pytree of jax arrays


def _scaled_add(a: float, x: State, b: float, y: State) -> State:
    return jax.tree.map(lambda xi, yi: a * xi + b * yi, x, y)


def low_storage_rk4_step(
    rhs: Callable[[State, float], State],
    u: State,
    t: float,
    dt: float,
) -> State:
    """One outer time step of the 5-stage low-storage RK4 scheme."""
    res = jax.tree.map(jnp.zeros_like, u)

    def stage(carry, k):
        u, res = carry
        t_local = t + RK4C[k] * dt
        rhs_out = rhs(u, t_local)
        res = _scaled_add(RK4A[k], res, dt, rhs_out)
        u = jax.tree.map(lambda u_, r_: u_ + RK4B[k] * r_, u, res)
        return (u, res), None

    (u, _), _ = jax.lax.scan(stage, (u, res), jnp.arange(5))
    return u


def low_storage_rk4(
    rhs: Callable[[State, float], State],
    u0: State,
    t0: float,
    t_final: float,
    dt: float,
) -> State:
    """Integrate from ``t0`` to ``t_final`` with fixed-step low-storage RK4.

    The step size is rescaled so the trajectory lands exactly on
    ``t_final``. Returns the final state. ``lax.scan`` fuses the entire
    trajectory into one XLA program, so the integrator is autodiffable
    end-to-end.
    """
    duration = t_final - t0
    n_steps = max(int(jnp.ceil(duration / dt)), 1)
    dt_actual = duration / n_steps

    def body(carry, k):
        u, t = carry
        u_new = low_storage_rk4_step(rhs, u, t, dt_actual)
        return (u_new, t + dt_actual), None

    (u_final, _), _ = jax.lax.scan(body, (u0, t0), jnp.arange(n_steps))
    return u_final


def low_storage_rk4_trajectory(
    rhs: Callable[[State, float], State],
    u0: State,
    t0: float,
    t_final: float,
    dt_max: float,
    n_frames: int,
) -> tuple[State, jax.Array]:
    """Integrate while recording ``n_frames+1`` evenly-spaced snapshots.

    Returns ``(snapshots, times)`` where ``snapshots`` is a pytree shaped
    like ``u0`` but with a leading axis of size ``n_frames+1`` on every
    leaf, and ``times[i]`` is the time at which ``snapshots[i]`` was
    sampled. The chunks between snapshots are integrated with
    ``low_storage_rk4`` using a step ``<= dt_max``.

    This is the building block for animations: pick ``n_frames`` to
    control the movie length and ``dt_max`` to keep the per-step error
    bounded.
    """
    times = jnp.linspace(t0, t_final, n_frames + 1)
    states = [u0]
    u = u0
    for k in range(n_frames):
        u = low_storage_rk4(rhs, u, float(times[k]), float(times[k + 1]), dt_max)
        states.append(u)
    snapshots = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)
    return snapshots, times
