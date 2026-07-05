"""DG advection-diffusion for a passive scalar.

Solves the unsteady convection-diffusion equation

    ∂c/∂t + u·∇c = κ ∇²c

for a scalar ``c`` transported by a (typically divergence-free)
velocity field ``u``. This is the transport half of a Boussinesq
incompressible-flow solver: ``c`` is a temperature or a normalised
density, and it feeds back into the momentum equation through a
buoyancy body force (see :mod:`ddg.dg2d.incompressible_ns`).

Discretisation mirrors the velocity solver:

  * Convection ``∇·(u c)`` — strong-form volume term plus a
    Lax-Friedrichs surface flux, exactly as :func:`ns_advection_2d`
    does for momentum.
  * Diffusion ``κ ∇²c`` — treated implicitly with the SIPG operator
    from :mod:`ddg.dg2d.sipg`.
  * Time stepping — BDF1 then BDF2, sharing the :class:`BDFCoeffs`
    from the velocity solver so the two march in lockstep.

The implicit diffusion solve is a Helmholtz problem ``(α − Δ)c =
…`` with ``α = g0/(κ dt) > 0`` — always non-singular thanks to the
mass term, so (unlike the pressure Poisson) no regularisation is
needed even for an all-Neumann no-flux scalar.

Boundary conditions reuse the SIPG codes (:data:`BC_INTERIOR` /
:data:`BC_DIRICHLET` / :data:`BC_NEUMANN`). For the convective term,
boundary faces fall back to a zero-jump trace; at a no-slip wall the
velocity is zero there so the advective boundary flux vanishes
anyway, and periodic faces are handled by the mesh connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ddg.dg2d.mesh import Mesh2D
from ddg.dg2d.incompressible_ns import _div_2d, BDFCoeffs
from ddg.dg2d.sipg import (
    BC_NEUMANN, IterativeConfig, factor_sipg_helmholtz_2d, solve_sipg_helmholtz_2d,
)


@dataclass(frozen=True)
class ScalarOperators:
    """Factored Helmholtz operator for the implicit diffusion solve."""
    helmholtz_factored: tuple | None
    diffusion_bc_types: jax.Array
    alpha: float
    kappa: float


def build_scalar_operators(
    mesh: Mesh2D,
    *,
    dt: float,
    kappa: float,
    g0: float,
    diffusion_bc_types: jax.Array,
    tau_scale: float = 200.0,
    skip_factor: bool = False,
) -> ScalarOperators:
    """Assemble + LU-factor the scalar diffusion Helmholtz operator.

    ``α = g0 / (κ dt)`` — the BDF Helmholtz shift. Pass
    ``skip_factor=True`` for the matrix-free iterative path.
    """
    alpha = g0 / (kappa * dt)
    if skip_factor:
        fact = None
    else:
        fact = factor_sipg_helmholtz_2d(
            mesh, alpha=alpha, nu=kappa,
            bc_types=diffusion_bc_types, tau_scale=tau_scale,
        )
    return ScalarOperators(
        helmholtz_factored=fact,
        diffusion_bc_types=diffusion_bc_types,
        alpha=alpha, kappa=kappa,
    )


def scalar_advection_2d(
    mesh: Mesh2D,
    c: jax.Array,
    ux: jax.Array,
    uy: jax.Array,
) -> jax.Array:
    """Nonlinear transport term ``N_c = ∇·(u c)``.

    Strong-form DG: ``N = (∇·F)_vol − LIFT(Fscale ((F·n)_M − (F·n)*))``
    with the local Lax-Friedrichs numerical flux
    ``(F·n)* = ½((F·n)_M + (F·n)_P) − ½ λ_max (c_P − c_M)`` and
    ``λ_max = max(|u·n|_M, |u·n|_P)``. The surface correction enters with
    a **minus** sign (same convention as :func:`ns_advection_2d`); adding
    it with the wrong sign turns the upwind LF dissipation into
    anti-diffusion and the operator becomes L²-unstable. Boundary faces
    read ``vmapP``; on a true boundary that returns the M-side value
    (zero jump), and on periodic faces it returns the wrapped partner —
    both correct.
    """
    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K

    # Volume: ∇·(u c).
    vol = _div_2d(mesh, ux * c, uy * c)

    # Surface: Lax-Friedrichs.
    c_flat = c.T.reshape(-1)
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)
    c_M = c_flat[mesh.vmapM]
    c_P = c_flat[mesh.vmapP]
    ux_M = ux_flat[mesh.vmapM]
    ux_P = ux_flat[mesh.vmapP]
    uy_M = uy_flat[mesh.vmapM]
    uy_P = uy_flat[mesh.vmapP]

    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")

    # Convective flux F = u c; normal flux F·n = (u·n) c.
    udotn_M = ux_M * nx_flat + uy_M * ny_flat
    udotn_P = ux_P * nx_flat + uy_P * ny_flat
    fM = udotn_M * c_M
    fP = udotn_P * c_P

    lam_node = jnp.maximum(jnp.abs(udotn_M), jnp.abs(udotn_P))
    lam_face = lam_node.reshape((mesh.Nfp, mesh.Nfaces * K), order="F")
    lam_face_max = jnp.max(lam_face, axis=0, keepdims=True)
    lam_max = jnp.broadcast_to(lam_face_max, lam_face.shape).reshape(-1, order="F")

    # DG surface contribution: N = vol − LIFT((F·n)_M − (F·n)*) with the
    # LF flux (F·n)* = ½((F·n)_M + (F·n)_P) − ½ λ_max (c_P − c_M). The
    # correction enters with a MINUS sign — same convention as
    # ns_advection_2d. (The opposite sign turns the upwind dissipation
    # into anti-diffusion and the operator becomes L²-unstable.)
    flux = -(0.5 * (fM - fP) + 0.5 * lam_max * (c_P - c_M))
    flux_face = flux.reshape((Nfp_Nf, K), order="F")
    return vol + mesh.LIFT @ (mesh.Fscale * flux_face)


def scalar_transport_step_2d(
    mesh: Mesh2D,
    c_n: jax.Array,
    c_old: jax.Array,
    ux: jax.Array,
    uy: jax.Array,
    Nc_old: jax.Array,
    *,
    dt: float,
    kappa: float,
    bdf: BDFCoeffs,
    ops: ScalarOperators,
    c_dir: jax.Array | None = None,
    q_neu: jax.Array | None = None,
    tau_scale: float = 200.0,
    iterative: IterativeConfig | None = None,
    preconditioner=None,
) -> tuple[jax.Array, jax.Array]:
    """One BDF advection-diffusion step. Returns ``(c_new, Nc_n)``.

    Convection is explicit (BDF-extrapolated, like the velocity
    solver's treatment of ``N(u)``); diffusion is implicit. ``Nc_old``
    is the previous step's advection term — pass zeros for the first
    BDF1 step.

    ``c_dir`` supplies Dirichlet scalar values on faces flagged
    Dirichlet in ``ops.diffusion_bc_types``; ``q_neu`` supplies the
    Neumann flux (zero for the no-flux case, which is also the
    default when omitted).
    """
    g0 = bdf.g0
    Nc_n = scalar_advection_2d(mesh, c_n, ux, uy)
    # Explicit accumulation, mirroring the velocity transport step.
    c_T = (
        bdf.a0 * c_n + bdf.a1 * c_old
        - dt * (bdf.b0 * Nc_n + bdf.b1 * Nc_old)
    ) / g0
    # Implicit diffusion: (α − Δ) c = α c_T with α = g0/(κ dt).
    alpha = ops.alpha
    c_new = solve_sipg_helmholtz_2d(
        mesh, alpha * c_T,
        alpha=alpha, nu=kappa,
        bc_types=ops.diffusion_bc_types,
        u_dir=c_dir, q_neu=q_neu,
        tau_scale=tau_scale,
        A_factored=ops.helmholtz_factored,
        iterative=iterative,
        preconditioner=preconditioner,
    )
    return c_new, Nc_n


def all_neumann_bc_types(mesh: Mesh2D) -> jax.Array:
    """``(K, Nfaces)`` SIPG BC array: every boundary face no-flux Neumann.

    The right default for a closed / no-flux scalar (Rayleigh-Taylor,
    Rayleigh-Bénard side walls). Interior and periodic faces stay
    :data:`BC_INTERIOR`; only true boundary faces become Neumann.
    """
    import numpy as np
    K, Nfaces = mesh.K, mesh.Nfaces
    EToE = np.asarray(mesh.EToE)
    is_bdy = EToE == np.arange(K)[:, None]
    bc = np.where(is_bdy, BC_NEUMANN, 0).astype(np.int32)
    return jnp.asarray(bc)
