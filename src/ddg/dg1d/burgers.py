"""1D Burgers equation ``u_t + (u^2/2)_x = nu u_xx`` (viscous and inviscid).

Direct port of ``Codes1.1/Codes1D/BurgersRHS1D.m`` with two extras:

* ``nu = 0`` works out of the box (the viscous terms vanish), turning
  the solver into pure inviscid Burgers with a Lax-Friedrichs flux.
* A nodal slope limiter (``slope_limit_n``, the SlopeLimitN port from
  ``CFD1D/``) keeps the inviscid solution monotone across shocks. It's
  applied between time steps in ``solve_burgers_1d`` when
  ``apply_limiter=True``.

Boundary handling: the user can pass ``inflow_fn(t)`` / ``outflow_fn(t)``
callbacks for non-periodic problems with prescribed Dirichlet boundary
values (matches the MATLAB driver's traveling-wave setup). On a
periodic mesh, those callbacks should stay ``None`` and the standard
wrap-around ``vmapP`` takes care of itself.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from ddg.dg1d.integrator import (
    low_storage_rk4_step,
    low_storage_rk4_trajectory,
)
from ddg.dg1d.mesh import Mesh1D


def _flatten_f(a: jax.Array) -> jax.Array:
    return a.T.reshape(-1)


# ---------------------------------------------------------------------------
# Burgers RHS (handles nu = 0 case smoothly via sqrt_nu = jnp.sqrt(nu))
# ---------------------------------------------------------------------------

def burgers_rhs_1d(
    mesh: Mesh1D,
    u: jax.Array,
    t: float,
    *,
    nu: float = 0.0,
    inflow_fn: Callable[[float], jax.Array] | None = None,
    outflow_fn: Callable[[float], jax.Array] | None = None,
) -> jax.Array:
    """Semi-discrete RHS for ``u_t + (u^2/2)_x = nu u_xx``."""
    Nfp, Nfaces, K = mesh.Nfp, mesh.Nfaces, mesh.K
    sqrt_nu = jnp.sqrt(nu)

    u_flat = _flatten_f(u)
    uM = u_flat[mesh.vmapM]
    uP = u_flat[mesh.vmapP]
    du = uM - uP
    du2 = (uM**2 - uP**2) / 2.0

    have_bc = (inflow_fn is not None or outflow_fn is not None) and not mesh.periodic
    if have_bc:
        # MATLAB convention: vmapI/mapI = first volume-flat index / first
        # face-flat index; vmapO/mapO = last. For our 1D mesh, mapB sorted
        # in ascending order is exactly [mapI, mapO].
        mapI, mapO = 0, K * Nfaces - 1
        if inflow_fn is not None:
            u_in = inflow_fn(t)
            du = du.at[mapI].set(2.0 * (uM[mapI] - u_in))
            du2 = du2.at[mapI].set(uM[mapI] ** 2 - u_in ** 2)
        if outflow_fn is not None:
            u_out = outflow_fn(t)
            du = du.at[mapO].set(2.0 * (uM[mapO] - u_out))
            du2 = du2.at[mapO].set(uM[mapO] ** 2 - u_out ** 2)

    nx_flat = mesh.nx.reshape(-1, order="F")

    # Auxiliary q ~ sqrt(nu) * u_x (zero when nu = 0).
    flux_u = (nx_flat * du / 2.0).reshape((Nfp * Nfaces, K), order="F")
    q = sqrt_nu * (mesh.rx * (mesh.Dr @ u) - mesh.LIFT @ (mesh.Fscale * flux_u))

    q_flat = _flatten_f(q)
    dq = (q_flat[mesh.vmapM] - q_flat[mesh.vmapP]) / 2.0
    if have_bc:
        if inflow_fn is not None:
            dq = dq.at[mapI].set(0.0)
        if outflow_fn is not None:
            dq = dq.at[mapO].set(0.0)

    # Lax-Friedrichs penalty for the convective flux.
    maxvel = jnp.max(jnp.abs(u))
    flux = nx_flat * (du2 / 2.0 - sqrt_nu * dq) - 0.5 * maxvel * du
    flux_face = flux.reshape((Nfp * Nfaces, K), order="F")

    # Volume strong-form: f(u) = u^2/2 - sqrt(nu) * q.
    dfdr = mesh.Dr @ (u**2 / 2.0 - sqrt_nu * q)
    return -(mesh.rx * dfdr - mesh.LIFT @ (mesh.Fscale * flux_face))


# ---------------------------------------------------------------------------
# Slope limiter (port of CFD1D/SlopeLimitN.m + SlopeLimitLin.m + minmod.m)
# ---------------------------------------------------------------------------

def _minmod_3(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """Three-argument ``minmod``: returns the argument of smallest magnitude
    when all three arguments share a sign, zero otherwise."""
    s = jnp.sign(a) + jnp.sign(b) + jnp.sign(c)
    same_sign = jnp.abs(s) > 2.5  # all three same sign => |sum| == 3
    mag = jnp.minimum(jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))
    return jnp.where(same_sign, jnp.sign(a) * mag, 0.0)


def slope_limit_n(mesh: Mesh1D, u: jax.Array, *, eps0: float = 1e-8) -> jax.Array:
    """Generalised slope limiter for nodal DG of order N.

    Each element is checked: if the modal end-values disagree with the
    minmod-reconstructed end values, the element's polynomial is
    truncated to its linear projection and the linear slope is bounded
    by the local neighbour differences. Otherwise the element's
    polynomial is left untouched.
    """
    # Cell average per element (constant mode of the modal expansion).
    u_modes = mesh.invV @ u
    avg_modal = u_modes.at[1:, :].set(0.0)
    v = (mesh.V @ avg_modal)[0, :]

    ue1 = u[0, :]
    ue2 = u[-1, :]

    # Neumann replication at the domain ends so the boundary cell's slope
    # comparison yields zero (avoids spurious limiting there).
    vkm1 = jnp.concatenate([v[:1], v[:-1]])
    vkp1 = jnp.concatenate([v[1:], v[-1:]])

    # Troubled-cell indicator.
    ve1 = v - _minmod_3(v - ue1, v - vkm1, vkp1 - v)
    ve2 = v + _minmod_3(ue2 - v, v - vkm1, vkp1 - v)
    needs_limit = (jnp.abs(ve1 - ue1) > eps0) | (jnp.abs(ve2 - ue2) > eps0)

    # Linear projection of the original polynomial.
    if mesh.Np > 2:
        lin_modal = u_modes.at[2:, :].set(0.0)
    else:
        lin_modal = u_modes
    u_lin = mesh.V @ lin_modal

    # Reference -> physical slope: u_x = (2/h) (Dr u_lin), constant per cell
    # since u_lin is linear. Take the first row.
    h = mesh.x[-1, :] - mesh.x[0, :]
    x0 = mesh.x[0, :] + h / 2.0
    slope_current = ((2.0 / h)[None, :] * (mesh.Dr @ u_lin))[0, :]
    slope_left = (v - vkm1) / h
    slope_right = (vkp1 - v) / h
    slope_limited = _minmod_3(slope_current, slope_left, slope_right)

    u_limited = v[None, :] + (mesh.x - x0[None, :]) * slope_limited[None, :]
    return jnp.where(needs_limit[None, :], u_limited, u)


# ---------------------------------------------------------------------------
# Time integration with optional limiter
# ---------------------------------------------------------------------------

def _burgers_dt(mesh: Mesh1D, umax: float, nu: float, cfl: float) -> float:
    """CFL: ``cfl * min(h / u_max, h^2 / nu)`` (matches the MATLAB driver)."""
    xmin = float(jnp.min(jnp.abs(mesh.x[1, :] - mesh.x[0, :])))
    inv_dt_conv = umax / max(xmin, 1e-12)
    inv_dt_visc = nu / max(xmin**2, 1e-24)
    return cfl / max(inv_dt_conv + inv_dt_visc, 1e-12)


def _make_step_fn(rhs_fn, limiter_fn):
    """Compose one RK4 step with an optional post-step limiter."""
    if limiter_fn is None:
        return lambda u, t, dt: low_storage_rk4_step(rhs_fn, u, t, dt)
    return lambda u, t, dt: limiter_fn(low_storage_rk4_step(rhs_fn, u, t, dt))


def solve_burgers_1d(
    mesh: Mesh1D,
    u0: jax.Array,
    t_final: float,
    *,
    nu: float = 0.0,
    cfl: float = 0.25,
    apply_limiter: bool = False,
    inflow_fn: Callable[[float], jax.Array] | None = None,
    outflow_fn: Callable[[float], jax.Array] | None = None,
) -> jax.Array:
    """Integrate Burgers to ``t_final``. Returns the final ``u``."""
    rhs = partial(
        burgers_rhs_1d, mesh, nu=nu,
        inflow_fn=inflow_fn, outflow_fn=outflow_fn,
    )
    limiter = (lambda u: slope_limit_n(mesh, u)) if apply_limiter else None
    step_fn = _make_step_fn(lambda u, t: rhs(u, t), limiter)

    umax = float(jnp.max(jnp.abs(u0)))
    dt = _burgers_dt(mesh, umax, nu, cfl)
    n_steps = max(int(jnp.ceil(t_final / dt)), 1)
    dt_actual = t_final / n_steps

    def body(carry, k):
        u, t = carry
        u_new = step_fn(u, t, dt_actual)
        return (u_new, t + dt_actual), None

    (u_final, _), _ = jax.lax.scan(body, (u0, 0.0), jnp.arange(n_steps))
    return u_final


def solve_burgers_1d_trajectory(
    mesh: Mesh1D,
    u0: jax.Array,
    t_final: float,
    n_frames: int,
    *,
    nu: float = 0.0,
    cfl: float = 0.25,
    apply_limiter: bool = False,
    inflow_fn: Callable[[float], jax.Array] | None = None,
    outflow_fn: Callable[[float], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run Burgers and return ``(u_traj, times)`` with ``n_frames+1`` snapshots."""
    rhs = partial(
        burgers_rhs_1d, mesh, nu=nu,
        inflow_fn=inflow_fn, outflow_fn=outflow_fn,
    )
    limiter = (lambda u: slope_limit_n(mesh, u)) if apply_limiter else None

    umax = float(jnp.max(jnp.abs(u0)))
    dt = _burgers_dt(mesh, umax, nu, cfl)

    if limiter is None:
        return low_storage_rk4_trajectory(
            lambda u, t: rhs(u, t), u0, 0.0, t_final, dt, n_frames,
        )

    # Custom trajectory loop that interleaves the limiter.
    times = jnp.linspace(0.0, t_final, n_frames + 1)
    snapshots = [u0]
    u = u0
    step_fn = _make_step_fn(lambda u, t: rhs(u, t), limiter)
    for k in range(n_frames):
        t0 = float(times[k])
        t1 = float(times[k + 1])
        n_inner = max(int(jnp.ceil((t1 - t0) / dt)), 1)
        dt_inner = (t1 - t0) / n_inner

        def body(carry, _):
            u, t = carry
            u_new = step_fn(u, t, dt_inner)
            return (u_new, t + dt_inner), None

        (u, _), _ = jax.lax.scan(body, (u, t0), jnp.arange(n_inner))
        snapshots.append(u)
    return jnp.stack(snapshots, axis=0), times
