"""2D incompressible Navier-Stokes via the vorticity-streamfunction
formulation.

We solve the coupled system

    ω_t + u · ∇ω = ν Δω         (vorticity transport)
       -Δψ      = ω             (streamfunction Poisson)
        u       = (∂_y ψ, -∂_x ψ)

on a square domain with homogeneous Dirichlet boundaries
(``ω = ψ = 0``) — i.e. free-slip / impermeable walls. This avoids the
mixed velocity-pressure problem of the primitive-variable form: at
each time step we solve a scalar Poisson equation for ψ (using
``ddg.dg2d.poisson``), recover the velocity from ψ via the curl, and
advance ω with an explicit RK4 step.

This is the natural endpoint of the differentiable-solver stack: each
RHS evaluation runs ``solve_poisson_2d`` (a linear solve through
``jax.numpy.linalg.solve``), takes two ``grad_2d`` derivatives, and
applies the IPDG Laplacian — all pure JAX, all autodiff-clean.

The current implementation uses a *volume-only* advection update
(no upwind face flux) and a Laplacian via
``-mass_inverse @ poisson_ip_apply_2d``. That choice trades some of
the dissipative robustness of a proper Riemann flux for code that
fits in 50 lines on top of the existing solvers. It works for smooth
flows (Taylor-Green, vortex decay); for problems with sharp gradients
you'd want a proper upwind flux on the vorticity transport equation.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg1d.integrator import low_storage_rk4_step, low_storage_rk4_trajectory
from ddg.dg2d.maxwell import grad_2d
from ddg.dg2d.mesh import Mesh2D
from ddg.dg2d.poisson import _local_mass, poisson_ip_apply_2d, solve_poisson_2d


def _apply_mass_inverse_2d(mesh: Mesh2D, u: jax.Array) -> jax.Array:
    """Apply ``M^{-1} u`` where ``M = block_diag(J_k * M_ref)``."""
    M_ref_inv = mesh.V @ mesh.V.T   # since M_ref = inv(V V^T)
    return M_ref_inv @ (u / mesh.J)


def laplacian_via_poisson(
    mesh: Mesh2D, omega: jax.Array, A_poisson: jax.Array | None = None,
) -> jax.Array:
    """Discrete ``Δω`` via the assembled IPDG Poisson operator.

    The IPDG apply returns ``M · (-Δ_h ω)`` (positive-definite by
    construction). Negating and applying ``M^{-1}`` gives the desired
    ``Δ_h ω``. Cheap; reuses whatever assembly the caller already has.
    """
    pos_op = poisson_ip_apply_2d(mesh, omega)
    return -_apply_mass_inverse_2d(mesh, pos_op)


def vorticity_streamfunction_rhs(
    mesh: Mesh2D,
    omega: jax.Array,
    t: float,
    *,
    nu: float,
    A_poisson: jax.Array,
) -> jax.Array:
    """Semi-discrete RHS for the vorticity-streamfunction NS system.

    ``A_poisson`` is the assembled IPDG stiffness matrix; pass it in
    rather than re-assembling each call (assembly is the expensive
    part). Returns ``∂ω/∂t``.
    """
    # 1) Streamfunction from -Δψ = ω.
    psi = solve_poisson_2d(mesh, omega, A_factored=(A_poisson,))

    # 2) Velocity from u = (∂_y ψ, -∂_x ψ).
    psi_x, psi_y = grad_2d(mesh, psi)
    u_vel = psi_y
    v_vel = -psi_x

    # 3) Advection: -(u · ∇)ω, volume-only.
    om_x, om_y = grad_2d(mesh, omega)
    advection = -(u_vel * om_x + v_vel * om_y)

    # 4) Diffusion: ν Δω via the Poisson operator.
    diffusion = nu * laplacian_via_poisson(mesh, omega, A_poisson)

    return advection + diffusion


def solve_vorticity_streamfunction(
    mesh: Mesh2D,
    omega0: jax.Array,
    t_final: float,
    *,
    nu: float,
    n_time_steps: int,
    A_poisson: jax.Array | None = None,
) -> jax.Array:
    """Integrate the vorticity-streamfunction NS system to ``t_final``.

    Uses a fixed-step RK4 with ``n_time_steps`` steps (the dt is picked
    by the caller — there's no automatic CFL because the velocity scale
    depends on the solution itself, which makes the dt selection a
    problem-dependent design choice).
    """
    from ddg.dg2d.poisson import poisson_ip_matrix_2d
    if A_poisson is None:
        A_poisson = poisson_ip_matrix_2d(mesh)

    dt = t_final / n_time_steps
    rhs = partial(vorticity_streamfunction_rhs, mesh, nu=nu, A_poisson=A_poisson)

    def body(carry, _):
        u, t = carry
        u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
        return (u_new, t + dt), None

    (omega_final, _), _ = jax.lax.scan(body, (omega0, 0.0), jnp.arange(n_time_steps))
    return omega_final


def solve_vorticity_streamfunction_trajectory(
    mesh: Mesh2D,
    omega0: jax.Array,
    t_final: float,
    n_frames: int,
    *,
    nu: float,
    n_time_steps: int,
    A_poisson: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run NS and return ``n_frames+1`` snapshots of vorticity over time."""
    from ddg.dg2d.poisson import poisson_ip_matrix_2d
    if A_poisson is None:
        A_poisson = poisson_ip_matrix_2d(mesh)

    dt = t_final / n_time_steps
    rhs = partial(vorticity_streamfunction_rhs, mesh, nu=nu, A_poisson=A_poisson)

    return low_storage_rk4_trajectory(
        lambda u, t: rhs(u, t), omega0, 0.0, t_final, dt, n_frames,
    )


# ---------------------------------------------------------------------------
# Implicit time stepping: DIRK / SDIRK methods via Newton-Krylov
# ---------------------------------------------------------------------------
#
# We support a family of diagonally-implicit Runge-Kutta integrators via a
# generic Butcher-tableau driver. Each implicit stage i is a per-stage
# nonlinear system of the form
#
#     Y_i = u_n + dt * sum_{j<i} a_{ij} K_j  +  dt * a_{ii} R(Y_i)
#
# (with K_i := R(Y_i)), which we solve by Newton-Krylov: a single
# substep ``_newton_implicit_substep(rhs_flat, Y_const, alpha)`` solves
# the equation ``Y - Y_const - alpha * R(Y) = 0`` via Newton's method
# with a matrix-free GMRES inner solve. The Jacobian-vector product
# ``J_G v = v - alpha * J_R v`` is computed by ``jax.jvp`` on the NS
# right-hand side — autodiff supplies the Jacobian; nothing is ever
# materialised.
#
# Available methods (Butcher tableaux defined below):
#   - "be"     : backward Euler                     1st order, L-stable
#   - "cn"     : Crank-Nicolson (ESDIRK form)        2nd order, A-stable
#   - "sdirk2" : Crouzeix 2-stage SDIRK              2nd order, L-stable
#   - "sdirk3" : Alexander 3-stage SDIRK             3rd order, L-stable

def _flatten(a: jax.Array) -> jax.Array:
    """Column-major flatten of ``(Np, K)`` -> ``(Np*K,)``."""
    return a.T.reshape(-1)


def _unflatten(a_flat: jax.Array, Np: int, K: int) -> jax.Array:
    return a_flat.reshape(K, Np).T


# Butcher tableaux ----------------------------------------------------------

_GAMMA_SDIRK2 = 1.0 - 1.0 / math.sqrt(2.0)            # ~0.2929, L-stable
_GAMMA_SDIRK3 = 0.43586652150845899941601945           # root of 6γ³-18γ²+9γ-1 = 0
_B1_SDIRK3 = (-6 * _GAMMA_SDIRK3 ** 2 + 16 * _GAMMA_SDIRK3 - 1) / 4
_B2_SDIRK3 = (6 * _GAMMA_SDIRK3 ** 2 - 20 * _GAMMA_SDIRK3 + 5) / 4

DIRK_TABLEAUX: dict[str, dict] = {
    "be": {
        "name": "backward Euler",
        "order": 1,
        "s": 1,
        "a": ((1.0,),),
        "b": (1.0,),
        "c": (1.0,),
    },
    "cn": {
        "name": "Crank-Nicolson (ESDIRK)",
        "order": 2,
        "s": 2,
        "a": ((0.0, 0.0), (0.5, 0.5)),
        "b": (0.5, 0.5),
        "c": (0.0, 1.0),
    },
    "sdirk2": {
        "name": "Crouzeix SDIRK-2",
        "order": 2,
        "s": 2,
        "a": (
            (_GAMMA_SDIRK2, 0.0),
            (1.0 - _GAMMA_SDIRK2, _GAMMA_SDIRK2),
        ),
        "b": (1.0 - _GAMMA_SDIRK2, _GAMMA_SDIRK2),
        "c": (_GAMMA_SDIRK2, 1.0),
    },
    "sdirk3": {
        "name": "Alexander SDIRK-3",
        "order": 3,
        "s": 3,
        "a": (
            (_GAMMA_SDIRK3, 0.0, 0.0),
            ((1.0 - _GAMMA_SDIRK3) / 2.0, _GAMMA_SDIRK3, 0.0),
            (_B1_SDIRK3, _B2_SDIRK3, _GAMMA_SDIRK3),
        ),
        "b": (_B1_SDIRK3, _B2_SDIRK3, _GAMMA_SDIRK3),
        "c": (
            _GAMMA_SDIRK3,
            (1.0 + _GAMMA_SDIRK3) / 2.0,
            1.0,
        ),
    },
}


def _newton_implicit_substep(
    rhs_flat,
    Y_const: jax.Array,
    alpha: float,
    *,
    newton_tol: float,
    newton_max_iter: int,
    gmres_tol: float,
    gmres_restart: int,
) -> tuple[jax.Array, dict]:
    """Newton-Krylov solve of ``Y - Y_const - α R(Y) = 0``.

    The DIRK building block. ``Y_const`` is whatever ``u_n + dt
    Σ_{j<i} a_{ij} K_j`` evaluates to at this stage; ``alpha`` is
    ``dt · a_{ii}``. Returns ``(Y, info)``.
    """
    Y = Y_const  # initial guess
    final_resid = float("inf")
    iters_used = 0
    for k in range(newton_max_iter):
        R_Y = rhs_flat(Y)
        G = Y - Y_const - alpha * R_Y
        resid = float(jnp.linalg.norm(G))
        final_resid = resid
        iters_used = k
        if resid < newton_tol:
            break

        def jac_op(v: jax.Array) -> jax.Array:
            _, jr_v = jax.jvp(rhs_flat, (Y,), (v,))
            return v - alpha * jr_v

        delta, _ = jax.scipy.sparse.linalg.gmres(
            jac_op, -G, tol=gmres_tol, restart=gmres_restart, maxiter=200,
        )
        Y = Y + delta

    return Y, {"iters": iters_used + 1, "residual": final_resid}


def _dirk_step(
    rhs_flat,
    u_n_flat: jax.Array,
    dt: float,
    tableau: dict,
    *,
    newton_tol: float,
    newton_max_iter: int,
    gmres_tol: float,
    gmres_restart: int,
) -> tuple[jax.Array, dict]:
    """One step of a generic DIRK / SDIRK method.

    Driven entirely off the Butcher tableau dict so the same code
    handles BE, CN, SDIRK-2, SDIRK-3, etc. Returns the next state plus
    aggregate Newton diagnostics for the step (total iterations, max
    residual across stages, max GMRES residual, etc.).
    """
    s = tableau["s"]
    a = tableau["a"]
    b = tableau["b"]

    K_stages: list[jax.Array] = []
    total_newton_iters = 0
    max_residual = 0.0

    for i in range(s):
        Y_const = u_n_flat
        for j in range(i):
            Y_const = Y_const + dt * a[i][j] * K_stages[j]
        a_ii = a[i][i]

        if a_ii == 0.0:
            # Explicit stage (e.g. CN's first stage).
            Y_i = Y_const
        else:
            Y_i, info = _newton_implicit_substep(
                rhs_flat, Y_const, dt * a_ii,
                newton_tol=newton_tol, newton_max_iter=newton_max_iter,
                gmres_tol=gmres_tol, gmres_restart=gmres_restart,
            )
            total_newton_iters += info["iters"]
            max_residual = max(max_residual, info["residual"])

        K_stages.append(rhs_flat(Y_i))

    u_next = u_n_flat
    for i in range(s):
        u_next = u_next + dt * b[i] * K_stages[i]
    return u_next, {
        "newton_iters": total_newton_iters,
        "final_residual": max_residual,
        "stages": s,
    }


def newton_implicit_step(
    mesh: Mesh2D,
    omega_n: jax.Array,
    dt: float,
    *,
    nu: float,
    A_poisson: jax.Array,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 12,
    gmres_tol: float = 1e-9,
    gmres_restart: int = 30,
    method: str = "be",
) -> tuple[jax.Array, dict]:
    """One DIRK step on the vorticity-streamfunction NS system.

    Method defaults to ``"be"`` (backward Euler, 1st order, L-stable).
    Pass ``method="cn" | "sdirk2" | "sdirk3"`` for higher-order
    integrators. See ``DIRK_TABLEAUX`` for available methods.

    Solves the per-stage nonlinear system with matrix-free
    Newton-Krylov: ``J_R · v`` via ``jax.jvp``, linear solve via
    ``jax.scipy.sparse.linalg.gmres``. Returns ``(ω^{n+1}, info)`` with
    Newton diagnostics aggregated across all stages.
    """
    if method not in DIRK_TABLEAUX:
        raise ValueError(
            f"unknown method {method!r}; choose from {list(DIRK_TABLEAUX)}"
        )
    tableau = DIRK_TABLEAUX[method]

    Np, K = mesh.Np, mesh.K
    omega_n_flat = _flatten(omega_n)

    def rhs_flat(omega_flat: jax.Array) -> jax.Array:
        omega = _unflatten(omega_flat, Np, K)
        rhs = vorticity_streamfunction_rhs(
            mesh, omega, 0.0, nu=nu, A_poisson=A_poisson,
        )
        return _flatten(rhs)

    omega_next_flat, info = _dirk_step(
        rhs_flat, omega_n_flat, dt, tableau,
        newton_tol=newton_tol, newton_max_iter=newton_max_iter,
        gmres_tol=gmres_tol, gmres_restart=gmres_restart,
    )
    info["method"] = method
    return _unflatten(omega_next_flat, Np, K), info


def solve_vorticity_streamfunction_implicit(
    mesh: Mesh2D,
    omega0: jax.Array,
    t_final: float,
    *,
    nu: float,
    n_time_steps: int,
    A_poisson: jax.Array | None = None,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 12,
    method: str = "be",
) -> tuple[jax.Array, list[dict]]:
    """Integrate NS to ``t_final`` with a DIRK / SDIRK method.

    Defaults to backward Euler (``method="be"``). Pass
    ``method="cn"``, ``"sdirk2"``, or ``"sdirk3"`` for higher-order
    integrators (see ``DIRK_TABLEAUX``). All are unconditionally stable
    (A-stable; the SDIRK ones are also L-stable) so ``n_time_steps``
    can be picked for accuracy alone, not stability.

    Returns the final state and a per-step list of Newton diagnostics.
    """
    from ddg.dg2d.poisson import poisson_ip_matrix_2d
    if A_poisson is None:
        A_poisson = poisson_ip_matrix_2d(mesh)

    dt = t_final / n_time_steps
    omega = omega0
    diagnostics: list[dict] = []
    for step in range(n_time_steps):
        omega, info = newton_implicit_step(
            mesh, omega, dt,
            nu=nu, A_poisson=A_poisson,
            newton_tol=newton_tol, newton_max_iter=newton_max_iter,
            method=method,
        )
        diagnostics.append(info)
    return omega, diagnostics


def solve_vorticity_streamfunction_implicit_trajectory(
    mesh: Mesh2D,
    omega0: jax.Array,
    t_final: float,
    n_frames: int,
    *,
    nu: float,
    n_time_steps: int,
    A_poisson: jax.Array | None = None,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 12,
    method: str = "be",
) -> tuple[jax.Array, jax.Array, list[dict]]:
    """Implicit driver that records ``n_frames+1`` snapshots over the run.

    ``n_time_steps`` is the total number of backward-Euler steps; the
    snapshots come from sampling every ``n_time_steps // n_frames``
    steps. Returns ``(omega_traj, times, diagnostics)``.
    """
    from ddg.dg2d.poisson import poisson_ip_matrix_2d
    if A_poisson is None:
        A_poisson = poisson_ip_matrix_2d(mesh)

    if n_time_steps < n_frames:
        raise ValueError("n_time_steps must be >= n_frames")

    dt = t_final / n_time_steps
    step_per_frame = n_time_steps // n_frames

    omega = omega0
    snapshots = [np.asarray(omega)]
    times = [0.0]
    diagnostics: list[dict] = []

    for step in range(n_time_steps):
        omega, info = newton_implicit_step(
            mesh, omega, dt,
            nu=nu, A_poisson=A_poisson,
            newton_tol=newton_tol, newton_max_iter=newton_max_iter,
            method=method,
        )
        diagnostics.append(info)
        if (step + 1) % step_per_frame == 0 and len(snapshots) <= n_frames:
            snapshots.append(np.asarray(omega))
            times.append((step + 1) * dt)

    return jnp.stack([jnp.asarray(s) for s in snapshots]), jnp.array(times), diagnostics
