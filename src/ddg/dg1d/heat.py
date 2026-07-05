"""1D heat equation ``u_t = kappa u_xx`` via central-flux LDG-style DG.

Port of ``Codes1.1/Codes1D/HeatCRHS1D.m`` and ``Heat1D.m``. Uses an
auxiliary variable ``q ≈ u_x`` computed with central flux, then takes
its discrete derivative for the strong-form spatial residual. Boundary
conditions are homogeneous Dirichlet on ``u`` (mirror reflection across
the boundary) and Neumann zero on ``q`` (no jump at boundary).

The right-hand side is in *strong* form — no mass-matrix inverse
required — so the standard explicit RK4 in ``ddg.dg1d.integrator``
drives it directly. The parabolic CFL ``dt ~ h^2`` makes long-time
runs slow, but for the textbook problem sizes it's fine.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ddg.dg1d.integrator import low_storage_rk4, low_storage_rk4_trajectory
from ddg.dg1d.mesh import Mesh1D


def _flatten_f(a: jax.Array) -> jax.Array:
    return a.T.reshape(-1)


def heat_rhs(
    mesh: Mesh1D,
    u: jax.Array,
    t: float,
    *,
    kappa: float = 1.0,
    bc_left_fn: "Callable[[float], jax.Array] | None" = None,
    bc_right_fn: "Callable[[float], jax.Array] | None" = None,
    forcing_fn: "Callable[[float], jax.Array] | None" = None,
) -> jax.Array:
    """Strong-form RHS for ``u_t = kappa u_xx + f(x, t)``.

    Boundary conditions are homogeneous Dirichlet ``u = 0`` by default
    (mirror reflection across each wall). Passing ``bc_left_fn(t)`` or
    ``bc_right_fn(t)`` switches the corresponding wall to a
    time-varying Dirichlet value ``g(t)`` — the standard tool for
    boundary control / inverse heat conduction problems.

    Passing ``forcing_fn(t)`` adds a body-force / source term to the
    strong-form RHS. The callable must return either a scalar or an
    array broadcastable to ``(Np, K)``. This is the natural extension
    that turns the heat solver into a Poiseuille-style channel-flow
    solver: ``u_t = -dp/dx + ν u_yy`` is exactly ``heat_rhs`` with
    ``kappa = ν`` and ``forcing_fn(t) = -dp/dx``.
    """
    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K
    u_flat = _flatten_f(u)

    # 1) Half-jump of u at faces. Dirichlet ghost u_ghost = 2 g - u_M gives
    #    du = (u_M - u_ghost)/2 = u_M - g. With g = 0 this collapses to
    #    the original mirror form.
    du = (u_flat[mesh.vmapM] - u_flat[mesh.vmapP]) / 2.0
    if mesh.mapB.size > 0:
        g_left = bc_left_fn(t) if bc_left_fn is not None else 0.0
        g_right = bc_right_fn(t) if bc_right_fn is not None else 0.0
        # mapB for a non-periodic 1D mesh is [left_endpoint, right_endpoint].
        g_bc = jnp.array([g_left, g_right])
        du = du.at[mesh.mapB].set(u_flat[mesh.vmapB] - g_bc)

    nx_flat = mesh.nx.reshape(-1, order="F")
    flux_u = (nx_flat * du).reshape((Nfp_Nf, K), order="F")

    # 2) Auxiliary q ≈ u_x.
    q = mesh.rx * (mesh.Dr @ u) - mesh.LIFT @ (mesh.Fscale * flux_u)
    q_flat = _flatten_f(q)

    # 3) Half-jump of q; Neumann BC -> dq = 0 at boundary.
    dq = (q_flat[mesh.vmapM] - q_flat[mesh.vmapP]) / 2.0
    if mesh.mapB.size > 0:
        dq = dq.at[mesh.mapB].set(0.0)

    flux_q = (nx_flat * dq).reshape((Nfp_Nf, K), order="F")

    # 4) RHS ≈ q_x = u_xx (in strong form).
    laplacian = mesh.rx * (mesh.Dr @ q) - mesh.LIFT @ (mesh.Fscale * flux_q)
    rhs = kappa * laplacian
    if forcing_fn is not None:
        rhs = rhs + forcing_fn(t)
    return rhs


def _heat_dt(mesh: Mesh1D, kappa: float, cfl: float) -> float:
    """Parabolic CFL: dt ~ h_min^2 / kappa."""
    xmin = float(jnp.min(jnp.abs(mesh.x[1, :] - mesh.x[0, :])))
    return cfl * xmin**2 / max(kappa, 1e-12)


def solve_heat(
    mesh: Mesh1D,
    u0: jax.Array,
    t_final: float,
    *,
    kappa: float = 1.0,
    cfl: float = 0.25,
    bc_left_fn=None,
    bc_right_fn=None,
    forcing_fn=None,
) -> jax.Array:
    """Integrate the heat equation to ``t_final`` with explicit RK4."""
    rhs = partial(heat_rhs, mesh, kappa=kappa,
                  bc_left_fn=bc_left_fn, bc_right_fn=bc_right_fn,
                  forcing_fn=forcing_fn)
    dt = _heat_dt(mesh, kappa, cfl)
    return low_storage_rk4(lambda u, t: rhs(u, t), u0, 0.0, t_final, dt)


def solve_heat_trajectory(
    mesh: Mesh1D,
    u0: jax.Array,
    t_final: float,
    n_frames: int,
    *,
    kappa: float = 1.0,
    cfl: float = 0.25,
    bc_left_fn=None,
    bc_right_fn=None,
    forcing_fn=None,
) -> tuple[jax.Array, jax.Array]:
    """Run the heat equation and return ``(u_traj, times)`` for animation.

    ``u_traj`` has shape ``(n_frames+1, Np, K)``.
    """
    rhs = partial(heat_rhs, mesh, kappa=kappa,
                  bc_left_fn=bc_left_fn, bc_right_fn=bc_right_fn,
                  forcing_fn=forcing_fn)
    dt = _heat_dt(mesh, kappa, cfl)
    return low_storage_rk4_trajectory(
        lambda u, t: rhs(u, t), u0, 0.0, t_final, dt, n_frames
    )
