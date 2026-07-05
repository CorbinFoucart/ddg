"""Diagnostic metrics for incompressible NS runs.

Pure functions over ``(mesh, ux, uy[, p])``. Each returns a Python float so
the result can go straight into JSON / CSV / a markdown table without
JAX-tree fuss.

The metrics fall into three buckets:

  * **Physics state** (per-step or per-snapshot): kinetic energy,
    enstrophy, discrete-divergence L2 norm, mass-flux balance through
    inflow + outflow, max-norms of velocity and vorticity.

  * **Validation** (against an analytical reference): relative L2
    velocity / pressure errors. Used by the test suite and the
    convergence-study script.

  * **Obstacle forces** (for bluff-body flows): drag and lift
    coefficients from the surface integral of the stress tensor over
    flagged obstacle faces; Strouhal number from a lift time series
    via FFT.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.maxwell import curl_z_2d, grad_2d
from ddg.dg2d.mesh import Mesh2D
from ddg.dg2d.sipg import _local_mass


# ---------------------------------------------------------------------------
# Physics-state metrics (per snapshot)
# ---------------------------------------------------------------------------

def kinetic_energy(mesh: Mesh2D, ux: jax.Array, uy: jax.Array) -> float:
    """Total kinetic energy ``½ ∫_Ω |U|² dV``."""
    M_ref = _local_mass(mesh)
    ke_dens = 0.5 * (ux ** 2 + uy ** 2)
    return float(jnp.sum(mesh.J * (M_ref @ ke_dens)))


def enstrophy(mesh: Mesh2D, ux: jax.Array, uy: jax.Array) -> float:
    """Total enstrophy ``∫_Ω ω² dV``."""
    M_ref = _local_mass(mesh)
    omega = curl_z_2d(mesh, ux, uy)
    return float(jnp.sum(mesh.J * (M_ref @ (omega ** 2))))


def divergence_l2(mesh: Mesh2D, ux: jax.Array, uy: jax.Array) -> float:
    """``‖∇·U‖_{L²(Ω)}``, the absolute incompressibility violation."""
    div_u = _divergence(mesh, ux, uy)
    M_ref = _local_mass(mesh)
    return float(jnp.sqrt(jnp.sum(mesh.J * (M_ref @ (div_u ** 2)))))


def divergence_l2_relative(
    mesh: Mesh2D, ux: jax.Array, uy: jax.Array,
) -> float:
    """``‖∇·U‖ / ‖U‖`` — relative incompressibility violation."""
    div_u = _divergence(mesh, ux, uy)
    M_ref = _local_mass(mesh)
    num = jnp.sum(mesh.J * (M_ref @ (div_u ** 2)))
    den = jnp.sum(mesh.J * (M_ref @ (ux ** 2 + uy ** 2))) + 1e-30
    return float(jnp.sqrt(num / den))


def max_speed(ux: jax.Array, uy: jax.Array) -> float:
    return float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))


def max_vorticity(mesh: Mesh2D, ux: jax.Array, uy: jax.Array) -> float:
    return float(jnp.max(jnp.abs(curl_z_2d(mesh, ux, uy))))


def mass_flux_through(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    *,
    face_mapB_mask: jax.Array,
) -> float:
    """Net outward mass flux ``∫_Γ U·n dS`` through tagged boundary nodes.

    ``face_mapB_mask`` is a boolean over ``mapB`` selecting which
    boundary nodes to integrate (e.g. all inflow nodes or all outflow
    nodes). For incompressible flow the SUM of inflow and outflow mass
    fluxes should be ~ 0 (mass conservation).

    Implementation: surface integral via the LIFT operator. For face-
    trace data ``h``, ``∫_∂Ω h dS = Σ_i (M_K · LIFT · h)_i`` where the
    sum is over all volume nodes — the test function ``v = 1`` projects
    onto the basis ``L_i`` with ``Σ_i L_i = 1``.
    """
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)
    Un_M = ux_flat[mesh.vmapM] * nx_flat + uy_flat[mesh.vmapM] * ny_flat
    # Embed: zero out everything except the selected boundary nodes.
    n_face = mesh.Nfp * mesh.Nfaces * mesh.K
    face_data_flat = jnp.zeros(n_face)
    face_data_flat = face_data_flat.at[mesh.mapB].set(
        jnp.where(face_mapB_mask, Un_M[mesh.mapB], 0.0)
    )
    face_data = face_data_flat.reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F"
    )
    M_ref = _local_mass(mesh)
    surf = mesh.LIFT @ (mesh.Fscale * face_data)
    lifted = mesh.J * (M_ref @ surf)   # M_K · LIFT · face_data
    return float(jnp.sum(lifted))


def _divergence(
    mesh: Mesh2D, ux: jax.Array, uy: jax.Array,
) -> jax.Array:
    """Strong-form pointwise ``∂x ux + ∂y uy`` per node."""
    ur = mesh.Dr @ ux
    us = mesh.Ds @ ux
    vr = mesh.Dr @ uy
    vs = mesh.Ds @ uy
    return (
        mesh.rx * ur + mesh.sx * us
        + mesh.ry * vr + mesh.sy * vs
    )


# ---------------------------------------------------------------------------
# Validation metrics (against analytical reference)
# ---------------------------------------------------------------------------

def rel_l2_velocity_error(
    mesh: Mesh2D,
    ux: jax.Array, uy: jax.Array,
    ux_exact: jax.Array, uy_exact: jax.Array,
) -> float:
    """``‖U − U_exact‖ / ‖U_exact‖`` — relative L2 velocity error."""
    num = jnp.sum(mesh.J * ((ux - ux_exact) ** 2 + (uy - uy_exact) ** 2))
    den = jnp.sum(mesh.J * (ux_exact ** 2 + uy_exact ** 2)) + 1e-30
    return float(jnp.sqrt(num / den))


def rel_l2_pressure_error(
    mesh: Mesh2D,
    p: jax.Array, p_exact: jax.Array,
    *, subtract_mean: bool = True,
) -> float:
    """``‖p − p_exact‖ / ‖p_exact‖`` — relative L2 pressure error.

    Pressure is determined only up to a constant for many flows; pass
    ``subtract_mean=True`` (default) to factor that out before comparison.
    """
    if subtract_mean:
        M_ref = _local_mass(mesh)
        mean_p = jnp.sum(mesh.J * (M_ref @ p)) / jnp.sum(mesh.J * (M_ref @ jnp.ones_like(p)))
        mean_pe = jnp.sum(mesh.J * (M_ref @ p_exact)) / jnp.sum(mesh.J * (M_ref @ jnp.ones_like(p_exact)))
        p = p - mean_p
        p_exact = p_exact - mean_pe
    num = jnp.sum(mesh.J * (p - p_exact) ** 2)
    den = jnp.sum(mesh.J * p_exact ** 2) + 1e-30
    return float(jnp.sqrt(num / den))


# ---------------------------------------------------------------------------
# Obstacle forces (for bluff-body flows)
# ---------------------------------------------------------------------------

@dataclass
class ObstacleForces:
    F_x: float       # streamwise force per unit depth
    F_y: float       # cross-stream force per unit depth
    p_part: tuple    # (F_x_pressure, F_y_pressure)
    v_part: tuple    # (F_x_viscous,  F_y_viscous)


def obstacle_forces_2d(
    mesh: Mesh2D,
    ux: jax.Array, uy: jax.Array, p: jax.Array,
    *,
    nu: float,
    obstacle_mapB_mask: jax.Array,
) -> ObstacleForces:
    """Net surface force on the obstacle: ``F = ∫_∂_obs (-p I + ν ∇U) n dS``.

    Returns components and pressure / viscous breakdown. Use these to
    form drag and lift coefficients
    ``C_D = 2 F_x / (ρ U_inf² D)`` and analogous ``C_L``.

    Implementation: standard SAT-style surface integration on the
    flagged boundary nodes. The mask is over ``mapB`` and selects only
    the obstacle face nodes.
    """
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)
    p_flat = p.T.reshape(-1)

    # ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y at boundary nodes
    ur = mesh.Dr @ ux; us = mesh.Ds @ ux
    vr = mesh.Dr @ uy; vs = mesh.Ds @ uy
    dudx = (mesh.rx * ur + mesh.sx * us).T.reshape(-1)
    dudy = (mesh.ry * ur + mesh.sy * us).T.reshape(-1)
    dvdx = (mesh.rx * vr + mesh.sx * vs).T.reshape(-1)
    dvdy = (mesh.ry * vr + mesh.sy * vs).T.reshape(-1)

    # Stress · n at boundary face nodes:
    #   σ·n = (-p I + ν ∇U) · n
    #   σ_x · n = -p n_x + ν (dudx n_x + dudy n_y)
    #   σ_y · n = -p n_y + ν (dvdx n_x + dvdy n_y)
    p_at = p_flat[mesh.vmapM]
    dudx_at = dudx[mesh.vmapM]
    dudy_at = dudy[mesh.vmapM]
    dvdx_at = dvdx[mesh.vmapM]
    dvdy_at = dvdy[mesh.vmapM]

    sig_x_n = -p_at * nx_flat + nu * (dudx_at * nx_flat + dudy_at * ny_flat)
    sig_y_n = -p_at * ny_flat + nu * (dvdx_at * nx_flat + dvdy_at * ny_flat)

    # Separately track the pressure and viscous parts for diagnostics.
    sig_x_n_p = -p_at * nx_flat
    sig_x_n_v = nu * (dudx_at * nx_flat + dudy_at * ny_flat)
    sig_y_n_p = -p_at * ny_flat
    sig_y_n_v = nu * (dvdx_at * nx_flat + dvdy_at * ny_flat)

    # NOTE: the force on the FLUID-side of the obstacle is what we just
    # computed (outward normal pointing away from fluid into the body).
    # By Newton's 3rd law the force on the body is the negative.

    def _surface_int(face_vec_flat):
        n_face = mesh.Nfp * mesh.Nfaces * mesh.K
        f = jnp.zeros(n_face)
        f = f.at[mesh.mapB].set(
            jnp.where(obstacle_mapB_mask, face_vec_flat[mesh.mapB], 0.0)
        )
        f_2d = f.reshape((mesh.Nfp * mesh.Nfaces, mesh.K), order="F")
        M_ref = _local_mass(mesh)
        return float(jnp.sum(mesh.J * (M_ref @ (mesh.LIFT @ (mesh.Fscale * f_2d)))))

    F_x = -_surface_int(sig_x_n)
    F_y = -_surface_int(sig_y_n)
    F_x_p = -_surface_int(sig_x_n_p)
    F_x_v = -_surface_int(sig_x_n_v)
    F_y_p = -_surface_int(sig_y_n_p)
    F_y_v = -_surface_int(sig_y_n_v)

    return ObstacleForces(
        F_x=F_x, F_y=F_y,
        p_part=(F_x_p, F_y_p), v_part=(F_x_v, F_y_v),
    )


def drag_lift_coefficients(
    forces: ObstacleForces, *, U_inf: float, D: float, rho: float = 1.0,
) -> tuple[float, float]:
    """Standard non-dimensionalisation ``2 F / (ρ U² D)`` per unit depth."""
    denom = 0.5 * rho * U_inf ** 2 * D
    return forces.F_x / denom, forces.F_y / denom


def strouhal_from_lift(
    times: np.ndarray, C_L: np.ndarray, *, D: float, U_inf: float,
    skip_initial_fraction: float = 0.3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate Strouhal number from the lift-coefficient time series.

    Drops the first ``skip_initial_fraction`` of the signal (transient)
    and computes the FFT of the remaining record. Picks the dominant
    non-DC peak as the shedding frequency and forms ``St = f · D / U_inf``.

    Returns ``(St, freqs, abs_spectrum)``. If the series is too short
    or too noisy to find a peak, returns ``St = 0.0``.
    """
    n = len(times)
    skip = int(skip_initial_fraction * n)
    times = np.asarray(times[skip:])
    C_L = np.asarray(C_L[skip:])
    if len(times) < 16:
        return 0.0, np.array([]), np.array([])
    dt = float(np.mean(np.diff(times)))
    sig = C_L - np.mean(C_L)
    # Hann window to suppress sidelobes from finite-window truncation.
    win = np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(len(sig), d=dt)
    # Pick dominant non-DC peak.
    if len(spec) < 2:
        return 0.0, freqs, spec
    peak_idx = int(np.argmax(spec[1:]) + 1)
    f_peak = float(freqs[peak_idx])
    St = f_peak * D / U_inf
    return St, freqs, spec


def recirculation_length_2d(
    mesh: Mesh2D,
    ux: jax.Array,
    *,
    centerline_y: float,
    obstacle_x_max: float,
    sample_x: np.ndarray,
) -> float:
    """Distance from the obstacle's downstream face to the point on the
    centerline where ``u_x`` first crosses zero from negative to positive.

    Sample ``u_x`` on the y = ``centerline_y`` line at the given
    ``sample_x`` positions (typically `np.linspace(x_obs_end, x_outlet)`)
    via nearest-node interpolation. Returns ``L`` such that the
    recirculation zone extends from ``obstacle_x_max`` to
    ``obstacle_x_max + L``. Returns ``0.0`` if no reattachment is found
    within the sampled window (flow stays attached / sample range too
    short).
    """
    x_np = np.asarray(mesh.x).T.reshape(-1)
    y_np = np.asarray(mesh.y).T.reshape(-1)
    ux_np = np.asarray(ux).T.reshape(-1)
    # Find candidate sample-line nodes (y within tol of centerline)
    tol = 0.05  # mesh-dependent; OK for our typical scale
    on_line = np.where(np.abs(y_np - centerline_y) < tol)[0]
    if on_line.size == 0:
        return 0.0
    xs = x_np[on_line]
    uxs = ux_np[on_line]
    order = np.argsort(xs)
    xs_sorted = xs[order]
    uxs_sorted = uxs[order]
    # Restrict to downstream of obstacle
    downstream = xs_sorted >= obstacle_x_max
    if not downstream.any():
        return 0.0
    xs_d = xs_sorted[downstream]
    uxs_d = uxs_sorted[downstream]
    # Find first sign change from negative to positive
    signs = np.sign(uxs_d)
    crossings = np.where(np.diff(signs) > 0)[0]
    if crossings.size == 0:
        return 0.0
    idx = int(crossings[0])
    # Linear interpolation for the zero crossing
    x0, x1 = xs_d[idx], xs_d[idx + 1]
    u0, u1 = uxs_d[idx], uxs_d[idx + 1]
    x_cross = x0 - u0 * (x1 - x0) / (u1 - u0 + 1e-30)
    return float(x_cross - obstacle_x_max)
