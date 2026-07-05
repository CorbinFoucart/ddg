"""2D Maxwell equations in TM polarisation with PEC walls.

Fields ``(Hx, Hy, Ez)`` evolve under

    ∂Hx/∂t = -∂Ez/∂y
    ∂Hy/∂t =  ∂Ez/∂x
    ∂Ez/∂t =  ∂Hy/∂x - ∂Hx/∂y         (wave speed c = 1)

with upwind Riemann flux derived in §6.6 of Hesthaven & Warburton.
Boundary handling is the perfect electric conductor (PEC): mirror Ez
across the wall (``Ez_+ = -Ez_-``) and copy H (``H_+ = H_-``).

Port of ``Codes1.1/Codes2D/{Maxwell2D, MaxwellRHS2D}.m``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ddg.dg1d.integrator import low_storage_rk4, low_storage_rk4_trajectory
from ddg.dg2d.mesh import Mesh2D


def _flatten_f(a: jax.Array) -> jax.Array:
    return a.T.reshape(-1)


def grad_2d(mesh: Mesh2D, u: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Physical gradient ``(∂u/∂x, ∂u/∂y)``."""
    ur = mesh.Dr @ u
    us = mesh.Ds @ u
    return mesh.rx * ur + mesh.sx * us, mesh.ry * ur + mesh.sy * us


def curl_z_2d(mesh: Mesh2D, ux: jax.Array, uy: jax.Array) -> jax.Array:
    """Out-of-plane component of the 2D curl: ``∂uy/∂x - ∂ux/∂y``."""
    ux_y = mesh.ry * (mesh.Dr @ ux) + mesh.sy * (mesh.Ds @ ux)
    uy_x = mesh.rx * (mesh.Dr @ uy) + mesh.sx * (mesh.Ds @ uy)
    return uy_x - ux_y


def maxwell_rhs_2d(
    mesh: Mesh2D,
    Hx: jax.Array,
    Hy: jax.Array,
    Ez: jax.Array,
    t: float,
    *,
    eps: float | jax.Array = 1.0,
    mu: float | jax.Array = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Semi-discrete RHS for 2D TM Maxwell with PEC walls.

    ``eps`` and ``mu`` may be scalars or arrays of shape ``(Np, K)``
    for spatially-varying permittivity / permeability. The flux is the
    simple Riemann form from Hesthaven (constant-coefficient impedance
    matching) divided pointwise by ``eps`` (for Ez) and ``mu`` (for H);
    this is consistent for smooth heterogeneous media but not the
    exact Riemann solver for material jumps.
    """
    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K

    Hx_flat = _flatten_f(Hx)
    Hy_flat = _flatten_f(Hy)
    Ez_flat = _flatten_f(Ez)

    dHx = Hx_flat[mesh.vmapM] - Hx_flat[mesh.vmapP]
    dHy = Hy_flat[mesh.vmapM] - Hy_flat[mesh.vmapP]
    dEz = Ez_flat[mesh.vmapM] - Ez_flat[mesh.vmapP]

    # PEC: Ez_+ = -Ez_- (mirror), H_+ = H_- (no jump).
    if mesh.mapB.size > 0:
        Ez_bdy = Ez_flat[mesh.vmapB]
        dHx = dHx.at[mesh.mapB].set(0.0)
        dHy = dHy.at[mesh.mapB].set(0.0)
        dEz = dEz.at[mesh.mapB].set(2.0 * Ez_bdy)

    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")

    n_dot_dH = nx_flat * dHx + ny_flat * dHy
    flux_Hx = ny_flat * dEz + (n_dot_dH * nx_flat - dHx)
    flux_Hy = -nx_flat * dEz + (n_dot_dH * ny_flat - dHy)
    flux_Ez = -nx_flat * dHy + ny_flat * dHx - dEz

    flux_Hx_face = flux_Hx.reshape((Nfp_Nf, K), order="F")
    flux_Hy_face = flux_Hy.reshape((Nfp_Nf, K), order="F")
    flux_Ez_face = flux_Ez.reshape((Nfp_Nf, K), order="F")

    Ezx, Ezy = grad_2d(mesh, Ez)
    cuHz = curl_z_2d(mesh, Hx, Hy)

    rhsHx = (-Ezy + 0.5 * mesh.LIFT @ (mesh.Fscale * flux_Hx_face)) / mu
    rhsHy = (Ezx + 0.5 * mesh.LIFT @ (mesh.Fscale * flux_Hy_face)) / mu
    rhsEz = (cuHz + 0.5 * mesh.LIFT @ (mesh.Fscale * flux_Ez_face)) / eps
    return rhsHx, rhsHy, rhsEz


def _maxwell_dt(mesh: Mesh2D, cfl: float) -> float:
    """CFL recipe from MATLAB ``Maxwell2D``: ``cfl * r_inscribed * dr_LGL_min * 2/3``."""
    from ddg.dg1d.reference import jacobi_gauss_quadrature
    from ddg.dg2d.advection import _inscribed_radius_per_element

    rLGL, _ = jacobi_gauss_quadrature(0.0, 0.0, mesh.N)
    r_min_lgl = float(jnp.min(jnp.diff(rLGL)))
    r_in = float(jnp.min(_inscribed_radius_per_element(mesh)))
    return cfl * r_in * r_min_lgl * 2.0 / 3.0


def solve_maxwell_2d(
    mesh: Mesh2D,
    Hx0: jax.Array,
    Hy0: jax.Array,
    Ez0: jax.Array,
    t_final: float,
    *,
    cfl: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Integrate 2D Maxwell TM to ``t_final``. Returns ``(Hx, Hy, Ez)``."""
    rhs = lambda state, t: maxwell_rhs_2d(mesh, state[0], state[1], state[2], t)
    dt = _maxwell_dt(mesh, cfl)
    Hx, Hy, Ez = low_storage_rk4(rhs, (Hx0, Hy0, Ez0), 0.0, t_final, dt)
    return Hx, Hy, Ez


def solve_maxwell_2d_trajectory(
    mesh: Mesh2D,
    Hx0: jax.Array,
    Hy0: jax.Array,
    Ez0: jax.Array,
    t_final: float,
    n_frames: int,
    *,
    cfl: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run 2D Maxwell TM and return ``(Hx_traj, Hy_traj, Ez_traj, times)``.

    Each trajectory array has shape ``(n_frames + 1, Np, K)``.
    """
    rhs = lambda state, t: maxwell_rhs_2d(mesh, state[0], state[1], state[2], t)
    dt = _maxwell_dt(mesh, cfl)
    (Hx_traj, Hy_traj, Ez_traj), times = low_storage_rk4_trajectory(
        rhs, (Hx0, Hy0, Ez0), 0.0, t_final, dt, n_frames
    )
    return Hx_traj, Hy_traj, Ez_traj, times
