"""1D linear advection ``u_t + a u_x = 0`` with upwind numerical flux."""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from ddg.dg1d.integrator import low_storage_rk4, low_storage_rk4_trajectory
from ddg.dg1d.mesh import Mesh1D


def _gather_face_values(u: jax.Array, idx: jax.Array, K: int) -> jax.Array:
    """Pick values from ``u`` (Np, K) using indices into Fortran-flattened u."""
    # u.flatten(order='F') == u.T.reshape(-1)
    return u.T.reshape(-1)[idx]


def advec_rhs(
    mesh: Mesh1D,
    u: jax.Array,
    t: float,
    a: float,
    *,
    inflow_fn: Callable[[float], jax.Array] | None = None,
) -> jax.Array:
    """Semi-discrete RHS for ``u_t + a u_x = 0`` with pure upwind flux.

    For periodic meshes pass ``inflow_fn=None``. For an open inflow
    boundary at x=xmin (the MATLAB ``AdvecDriver1D`` setup), pass a
    callable ``inflow_fn(t)`` returning the scalar inflow value;
    the outflow boundary then has zero flux contribution.
    """
    Nfp, Nfaces, K = mesh.Nfp, mesh.Nfaces, mesh.K

    uM = _gather_face_values(u, mesh.vmapM, K)
    uP = _gather_face_values(u, mesh.vmapP, K)

    nx_flat = mesh.nx.reshape(-1, order="F")
    # Full upwinding (alpha=1 in Hesthaven): du = (uM - uP) * a*nx / 2
    du = (uM - uP) * (a * nx_flat) / 2.0

    if inflow_fn is not None and not mesh.periodic:
        # mapI = 0, vmapI = 0 (left endpoint); mapO = K*Nfaces - 1, vmapO = K*Np - 1
        mapI, mapO = 0, K * Nfaces - 1
        u_inflow = inflow_fn(t)
        n_inflow = nx_flat[mapI]
        du = du.at[mapI].set((uM[mapI] - u_inflow) * (a * n_inflow) / 2.0)
        du = du.at[mapO].set(0.0)

    du_face = du.reshape((Nfp * Nfaces, K), order="F")

    return -a * mesh.rx * (mesh.Dr @ u) + mesh.LIFT @ (mesh.Fscale * du_face)


def solve_advection(
    mesh: Mesh1D,
    u0: jax.Array,
    t_final: float,
    a: float,
    *,
    cfl: float = 0.75,
    inflow_fn: Callable[[float], jax.Array] | None = None,
) -> jax.Array:
    """Integrate the 1D advection equation to ``t_final``.

    Time-step is set from the local minimum node spacing and the
    advection speed, then halved (matching the MATLAB reference).
    """
    rhs = partial(advec_rhs, mesh, a=a, inflow_fn=inflow_fn)
    dt = _advection_dt(mesh, a, cfl)
    return low_storage_rk4(lambda u, t: rhs(u, t), u0, 0.0, t_final, dt)


def _advection_dt(mesh: Mesh1D, a: float, cfl: float) -> float:
    """Time step from local node spacing and advection speed (matches MATLAB)."""
    xmin_spacing = float(jnp.min(jnp.abs(mesh.x[1, :] - mesh.x[0, :])))
    return 0.5 * cfl / abs(a) * xmin_spacing


def solve_advection_trajectory(
    mesh: Mesh1D,
    u0: jax.Array,
    t_final: float,
    a: float,
    n_frames: int,
    *,
    cfl: float = 0.75,
    inflow_fn: Callable[[float], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run 1D advection and return ``(u_traj, times)``.

    ``u_traj`` has shape ``(n_frames+1, Np, K)``.
    """
    rhs = partial(advec_rhs, mesh, a=a, inflow_fn=inflow_fn)
    dt = _advection_dt(mesh, a, cfl)
    return low_storage_rk4_trajectory(
        lambda u, t: rhs(u, t), u0, 0.0, t_final, dt, n_frames
    )
