"""2D linear advection ``u_t + a . grad(u) = 0`` with upwind flux.

Strong-form DG on triangles. For a flow that doesn't reach the domain
boundary within the simulated window, the simple gather-from-neighbour
form ``du = u(vmapM) - u(vmapP)`` works as-is on a non-periodic mesh;
the trivial fallback at boundary faces (``vmapP = vmapM``) yields zero
boundary jump, equivalent to letting whatever's already there propagate
out. For long-time runs, use ``apply_rectangle_periodic_2d`` on the
mesh to wrap the boundary node maps before constructing ``rhs``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from ddg.dg1d.integrator import low_storage_rk4, low_storage_rk4_trajectory
from ddg.dg2d.mesh import Mesh2D


def _flatten_f(a: jax.Array) -> jax.Array:
    """Column-major flatten ``(Np, K) -> (Np*K,)``."""
    return a.T.reshape(-1)


def advec_rhs_2d(
    mesh: Mesh2D,
    u: jax.Array,
    t: float,
    ax: float,
    ay: float,
    *,
    alpha: float = 0.0,
) -> jax.Array:
    """Semi-discrete RHS for ``u_t + (ax, ay) . grad(u) = 0``.

    Numerical flux follows Hesthaven's convention
    ``du = (u_M - u_P) (a.n - (1 - alpha) |a.n|) / 2``:
        * ``alpha = 0`` (default): full upwind (dissipative; stable for
          non-periodic meshes).
        * ``alpha = 1``: central flux (energy-conserving; only stable
          on a periodic mesh).
    """
    u_flat = _flatten_f(u)
    uM = u_flat[mesh.vmapM]
    uP = u_flat[mesh.vmapP]

    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    a_dot_n = ax * nx_flat + ay * ny_flat
    du = (uM - uP) * (a_dot_n - (1.0 - alpha) * jnp.abs(a_dot_n)) / 2.0

    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K
    du_face = du.reshape((Nfp_Nf, K), order="F")

    ur = mesh.Dr @ u
    us = mesh.Ds @ u
    u_x = mesh.rx * ur + mesh.sx * us
    u_y = mesh.ry * ur + mesh.sy * us

    return -(ax * u_x + ay * u_y) + mesh.LIFT @ (mesh.Fscale * du_face)


def _advection_dt_2d(mesh: Mesh2D, ax: float, ay: float, cfl: float) -> float:
    """CFL ``dt = cfl * min(r_inscribed) * dr_LGL_min / |a|``.

    Matches the MATLAB ``dtscale2D`` recipe: inscribed-circle radius of
    each triangle gives the geometric scale; ``rLGL[1] - rLGL[0]`` (the
    smallest LGL node spacing) accounts for the order N.
    """
    speed = float(jnp.sqrt(ax * ax + ay * ay))
    r_inscribed = float(jnp.min(_inscribed_radius_per_element(mesh)))
    # Smallest LGL spacing on [-1, 1] is ~ 1 / N^2.
    from ddg.dg1d.reference import jacobi_gauss_lobatto_nodes
    lgl = jacobi_gauss_lobatto_nodes(0.0, 0.0, mesh.N)
    r_min_lgl = float(jnp.min(jnp.diff(lgl)))
    return cfl * r_inscribed * r_min_lgl / max(speed, 1e-12)


def _inscribed_radius_per_element(mesh: Mesh2D) -> jax.Array:
    """Inscribed-circle radius of each physical triangle."""
    VX = mesh.VX
    VY = mesh.VY
    v1 = mesh.EToV[:, 0]
    v2 = mesh.EToV[:, 1]
    v3 = mesh.EToV[:, 2]
    l12 = jnp.sqrt((VX[v1] - VX[v2]) ** 2 + (VY[v1] - VY[v2]) ** 2)
    l23 = jnp.sqrt((VX[v2] - VX[v3]) ** 2 + (VY[v2] - VY[v3]) ** 2)
    l31 = jnp.sqrt((VX[v3] - VX[v1]) ** 2 + (VY[v3] - VY[v1]) ** 2)
    sper = (l12 + l23 + l31) / 2.0
    area = jnp.sqrt(sper * (sper - l12) * (sper - l23) * (sper - l31))
    return area / sper


def solve_advection_2d(
    mesh: Mesh2D,
    u0: jax.Array,
    t_final: float,
    ax: float,
    ay: float,
    *,
    cfl: float = 0.75,
    alpha: float = 0.0,
) -> jax.Array:
    """Integrate 2D advection to ``t_final``."""
    rhs = partial(advec_rhs_2d, mesh, ax=ax, ay=ay, alpha=alpha)
    dt = _advection_dt_2d(mesh, ax, ay, cfl)
    return low_storage_rk4(lambda u, t: rhs(u, t), u0, 0.0, t_final, dt)


def solve_advection_2d_trajectory(
    mesh: Mesh2D,
    u0: jax.Array,
    t_final: float,
    ax: float,
    ay: float,
    n_frames: int,
    *,
    cfl: float = 0.75,
    alpha: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Run 2D advection and return ``(u_traj, times)``."""
    rhs = partial(advec_rhs_2d, mesh, ax=ax, ay=ay, alpha=alpha)
    dt = _advection_dt_2d(mesh, ax, ay, cfl)
    return low_storage_rk4_trajectory(
        lambda u, t: rhs(u, t), u0, 0.0, t_final, dt, n_frames
    )
