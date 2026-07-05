"""Coupled monolithic NS solver on a mixed-order Lagrangian DG pair.

Phase 3 of the mixed-FEM coupled NS work. Mirrors the equal-order solver
in :mod:`ddg.dg2d.coupled_ns` but parameterised by two FunctionSpaces:

    V_u  -- velocity space, LagrangianDG at degree N (vector, 2 components)
    V_p  -- pressure space, LagrangianDG at degree N-1 (scalar)

paired via the embedding trick from :mod:`ddg.dg2d.mixed_operators` so the
saddle-point off-diagonal blocks (D, G) reuse the existing equal-order
operators with an interpolation matrix.

Default ``tau_div = tau_cont = pressure_eps = 0``: with the inf-sup-stable
mixed-order pair the LBB-violation pollution that forced these knobs in
equal-order should be gone. The knobs are still exposed for diagnostic
purposes -- if Phase-3 numerics show residual stagnation we'll know
whether a small residual stabilisation closes it (then this is still
inf-sup-stable in practice, just centered-flux DG-noisy) or whether full
LDG lifting is needed (Phase 5 escalation).

Only the matrix-free iterative path is provided. The direct-LU path (for
small debugging problems) can be added later if needed.

Outer nonlinear loop and convergence-monitor wiring mirror the equal-
order :func:`coupled_ns.coupled_ns_step_2d`; the same
:class:`coupled_ns.CoupledNSConvergenceReport` is populated.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ddg.dg2d.coupled_ns import (
    CoupledNSConvergenceReport,
    IterativeSolverConfig,
    NonlinearSolverConfig,
    _conv_apply_2d,
    _conv_apply_newton_extra_2d,
    _conv_apply_nonlinear_2d,
    _grad_apply_2d,
    _mass_apply,
)
from ddg.dg2d.function_space import LagrangianDG
from ddg.dg2d.incompressible_ns import (
    BDFCoeffs, BDF1,
    continuity_penalty_apply_2d, divergence_penalty_apply_2d,
    viscous_bc_types_from_ins,
)
from ddg.dg2d.mixed_operators import (
    build_interpolation,
    mixed_div_apply,
    mixed_grad_apply,
)
from ddg.dg2d.sipg import (
    _face_1d_mass_matrices, _local_mass,
    sipg_apply_2d, sipg_dirichlet_load_2d, sipg_neumann_load_2d,
)


# ---------------------------------------------------------------------------
# Saddle-point apply
# ---------------------------------------------------------------------------

def coupled_ns_apply_mixed_2d(
    V_u: LagrangianDG, V_p: LagrangianDG,
    ux: jax.Array, uy: jax.Array, p: jax.Array,
    *,
    nu: float,
    gamma0_over_dt: float,
    u_star_x: jax.Array, u_star_y: jax.Array,
    ins_bc_types: jax.Array,
    viscous_bc_types_sipg: jax.Array,
    M_ref_u: jax.Array,
    m1d_per_face_u: jax.Array,
    P_emb: jax.Array,
    tau_scale: float = 200.0,
    tau_div: float = 0.0,
    tau_cont: float = 0.0,
    pressure_eps: float = 0.0,
    newton_u_k: tuple[jax.Array, jax.Array] | None = None,
    newton_method: str = "hand",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply the mixed-order saddle-point operator to ``(u, p)``.

    Shapes:
        ux, uy : ``(V_u.Np, V_u.K)``
        p      : ``(V_p.Np, V_p.K)``
        outputs match the inputs componentwise.

    Block structure (same sign convention as the equal-order version):

        out_x = (γ/dt) M_u ux + ν A_sipg ux + M·N_x(u; u*) + G_x p
        out_y = (γ/dt) M_u uy + ν A_sipg uy + M·N_y(u; u*) + G_y p
        out_p = −D u    (+ optional pressure_eps · M_p p)

    The velocity-block operators (mass, SIPG viscous, convection) act on
    V_u and are *identical* to the equal-order versions -- they don't see
    the pressure space at all. Only G and D cross spaces, and they use
    the embedding-trick wrappers in :mod:`mixed_operators`.

    ``tau_div / tau_cont`` retained for diagnostic use only; default 0
    because the inf-sup-stable pair shouldn't need them.
    """
    M_ux = _mass_apply(V_u.mesh, ux, M_ref_u)
    M_uy = _mass_apply(V_u.mesh, uy, M_ref_u)
    visc_x = sipg_apply_2d(
        V_u.mesh, ux, bc_types=viscous_bc_types_sipg,
        tau_scale=tau_scale, M_ref=M_ref_u, m1d_per_face=m1d_per_face_u,
    )
    visc_y = sipg_apply_2d(
        V_u.mesh, uy, bc_types=viscous_bc_types_sipg,
        tau_scale=tau_scale, M_ref=M_ref_u, m1d_per_face=m1d_per_face_u,
    )
    conv_x, conv_y = _conv_apply_2d(
        V_u.mesh, ux, uy,
        u_star_x=u_star_x, u_star_y=u_star_y,
        ins_bc_types=ins_bc_types, M_ref=M_ref_u,
    )
    if newton_u_k is not None:
        u_k_x, u_k_y = newton_u_k
        if newton_method == "hand":
            extra_x, extra_y = _conv_apply_newton_extra_2d(
                V_u.mesh, ux, uy, u_k_x, u_k_y,
                ins_bc_types=ins_bc_types, M_ref=M_ref_u,
            )
        elif newton_method == "jvp":
            def _nonlin(u_pair):
                return _conv_apply_nonlinear_2d(
                    V_u.mesh, u_pair[0], u_pair[1],
                    ins_bc_types=ins_bc_types, M_ref=M_ref_u,
                )
            full_x, full_y = jax.jvp(_nonlin, ((u_k_x, u_k_y),),
                                      ((ux, uy),))[1]
            picard_x_at_uk, picard_y_at_uk = _conv_apply_2d(
                V_u.mesh, ux, uy,
                u_star_x=u_k_x, u_star_y=u_k_y,
                ins_bc_types=ins_bc_types, M_ref=M_ref_u,
            )
            extra_x = full_x - picard_x_at_uk
            extra_y = full_y - picard_y_at_uk
        else:
            raise ValueError(f"unknown newton_method {newton_method!r}")
        conv_x = conv_x + extra_x
        conv_y = conv_y + extra_y

    # Cross-space pressure-gradient and velocity-divergence.
    Gx_p, Gy_p = mixed_grad_apply(
        V_u, V_p, p,
        ins_bc_types=ins_bc_types, M_ref_u=M_ref_u, P_emb=P_emb,
    )
    Dp = mixed_div_apply(
        V_u, V_p, ux, uy,
        ins_bc_types=ins_bc_types, M_ref_u=M_ref_u, P_emb=P_emb,
    )

    out_x = gamma0_over_dt * M_ux + nu * visc_x + conv_x + Gx_p
    out_y = gamma0_over_dt * M_uy + nu * visc_y + conv_y + Gy_p

    if tau_div != 0.0:
        dpx, dpy = divergence_penalty_apply_2d(V_u.mesh, ux, uy)
        out_x = out_x + tau_div * dpx
        out_y = out_y + tau_div * dpy
    if tau_cont != 0.0:
        cpx, cpy = continuity_penalty_apply_2d(V_u.mesh, ux, uy)
        out_x = out_x + tau_cont * cpx
        out_y = out_y + tau_cont * cpy

    out_p = -Dp
    if pressure_eps != 0.0:
        # Pressure mass at V_p degree (NOT V_u): M_p p evaluated at V_p nodes.
        M_ref_p = _local_mass(V_p.mesh)
        out_p = out_p - pressure_eps * _mass_apply(V_p.mesh, p, M_ref_p)
    return out_x, out_y, out_p


# ---------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------

def _coupled_ns_rhs_mixed_2d(
    V_u: LagrangianDG, V_p: LagrangianDG,
    *,
    nu: float, dt: float, bdf: BDFCoeffs,
    ux_n: jax.Array, uy_n: jax.Array,
    ux_old: jax.Array, uy_old: jax.Array,
    ins_bc_types: jax.Array,
    bc_ux: jax.Array, bc_uy: jax.Array,
    p_dir: jax.Array | None,
    q_ux_neu: jax.Array | None,
    q_uy_neu: jax.Array | None,
    M_ref_u: jax.Array,
    m1d_per_face_u: jax.Array,
    viscous_bc_types_sipg: jax.Array,
    P_emb: jax.Array,
    tau_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the ``(rhs_x, rhs_y, rhs_p)`` saddle-point right-hand side.

    Mirrors :func:`coupled_ns._coupled_ns_rhs_2d` but the pressure-side
    contribution (velocity-Dirichlet load on the divergence equation)
    is projected to ``V_p`` via ``P_emb.T``.

    All velocity-side fields (bc_ux, bc_uy, p_dir, q_*_neu) are at V_u
    shapes -- BC data is naturally specified on the velocity-space face
    DOFs (where the physical BCs live).
    """
    u_hist_x = (bdf.a0 * ux_n + bdf.a1 * ux_old) / dt
    u_hist_y = (bdf.a0 * uy_n + bdf.a1 * uy_old) / dt
    rhs_x = _mass_apply(V_u.mesh, u_hist_x, M_ref_u)
    rhs_y = _mass_apply(V_u.mesh, u_hist_y, M_ref_u)

    visc_load_x = sipg_dirichlet_load_2d(
        V_u.mesh, bc_ux, bc_types=viscous_bc_types_sipg, tau_scale=tau_scale,
    )
    visc_load_y = sipg_dirichlet_load_2d(
        V_u.mesh, bc_uy, bc_types=viscous_bc_types_sipg, tau_scale=tau_scale,
    )
    rhs_x = rhs_x - nu * visc_load_x
    rhs_y = rhs_y - nu * visc_load_y
    if q_ux_neu is not None:
        rhs_x = rhs_x + nu * sipg_neumann_load_2d(
            V_u.mesh, q_ux_neu, bc_types=viscous_bc_types_sipg,
        )
    if q_uy_neu is not None:
        rhs_y = rhs_y + nu * sipg_neumann_load_2d(
            V_u.mesh, q_uy_neu, bc_types=viscous_bc_types_sipg,
        )

    # Pressure-Dirichlet (outflow) load -> velocity equations via G_eq
    # (p_dir is already at V_u face shape; no embedding needed).
    if p_dir is not None:
        Gx_p_dir, Gy_p_dir = _grad_apply_2d(
            V_u.mesh, jnp.zeros_like(p_dir),
            ins_bc_types=ins_bc_types, M_ref=M_ref_u, p_dir=p_dir,
        )
        rhs_x = rhs_x - Gx_p_dir
        rhs_y = rhs_y - Gy_p_dir

    # Velocity-Dirichlet load on the divergence equation, sign-flipped
    # and projected from V_u test space to V_p test space.
    from ddg.dg2d.coupled_ns import _div_apply_2d as _div_apply_eq
    Dp_load_u = _div_apply_eq(
        V_u.mesh, jnp.zeros_like(ux_n), jnp.zeros_like(uy_n),
        ins_bc_types=ins_bc_types, M_ref=M_ref_u,
        bc_ux=bc_ux, bc_uy=bc_uy,
    )
    rhs_p = P_emb.T @ Dp_load_u
    return rhs_x, rhs_y, rhs_p


# ---------------------------------------------------------------------------
# Flat layout helpers (mixed degrees)
# ---------------------------------------------------------------------------

def _flat3_mixed(
    ux: jax.Array, uy: jax.Array, p: jax.Array,
) -> jax.Array:
    """Pack ``(ux, uy, p)`` into a single flat vector for Krylov. The
    velocity blocks (V_u-degree) and pressure block (V_p-degree) have
    different sizes."""
    return jnp.concatenate([
        ux.T.reshape(-1), uy.T.reshape(-1), p.T.reshape(-1),
    ])


def _unflat3_mixed(
    v: jax.Array, V_u: LagrangianDG, V_p: LagrangianDG,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Inverse of :func:`_flat3_mixed`."""
    n_u = V_u.Np * V_u.K
    n_p = V_p.Np * V_p.K
    ux = v[:n_u].reshape(V_u.K, V_u.Np).T
    uy = v[n_u:2 * n_u].reshape(V_u.K, V_u.Np).T
    p = v[2 * n_u:2 * n_u + n_p].reshape(V_p.K, V_p.Np).T
    return ux, uy, p


# ---------------------------------------------------------------------------
# Main step driver
# ---------------------------------------------------------------------------

def coupled_ns_step_mixed_2d(
    V_u: LagrangianDG, V_p: LagrangianDG,
    ux_n: jax.Array, uy_n: jax.Array,
    ux_old: jax.Array, uy_old: jax.Array,
    *,
    dt: float,
    nu: float,
    bdf: BDFCoeffs,
    ins_bc_types: jax.Array,
    bc_ux: jax.Array, bc_uy: jax.Array,
    p_dir: jax.Array | None = None,
    q_ux_neu: jax.Array | None = None,
    q_uy_neu: jax.Array | None = None,
    tau_scale: float = 200.0,
    tau_div: float = 0.0,
    tau_cont: float = 0.0,
    pressure_eps: float = 0.0,
    iterative: IterativeSolverConfig | None = None,
    x0_iterative: tuple | None = None,
    velocity_preconditioner=None,
    pressure_schur_inv=None,
    coupled_preconditioner=None,
    nl_config: NonlinearSolverConfig | None = None,
    nl_residual_history_out: list | None = None,
    convergence_out: CoupledNSConvergenceReport | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One monolithic mixed-order coupled NS step. Returns ``(ux, uy, p)``.

    ``V_u, V_p`` -- the velocity and pressure function spaces (typically
    ``LagrangianDG.at_degree(mesh, N)`` and
    ``LagrangianDG.at_degree(mesh, N-1)``).

    Iterative path only (no direct LU). Otherwise the surface matches
    :func:`coupled_ns.coupled_ns_step_2d`; see that docstring for the
    individual argument meanings.

    Preconditioner contract (block-diagonal mode):
        ``velocity_preconditioner`` acts on a flat V_u block of size
        ``V_u.Np * V_u.K``; ``pressure_schur_inv`` acts on a flat V_p
        block of size ``V_p.Np * V_p.K``. Build them with the existing
        :mod:`coupled_ns` helpers passing ``V_u.mesh`` /
        ``V_p.mesh`` respectively.

    Defaults ``tau_div = tau_cont = pressure_eps = 0`` -- mixed-order
    pair is inf-sup stable, no LBB stabilisation needed.
    """
    if iterative is None:
        raise NotImplementedError(
            "Direct-LU path not implemented for the mixed-order solver yet; "
            "pass iterative=IterativeSolverConfig(...)."
        )

    M_ref_u = _local_mass(V_u.mesh)
    m1d_per_face_u = _face_1d_mass_matrices(V_u.mesh)
    viscous_bc_sipg = viscous_bc_types_from_ins(ins_bc_types)
    P_emb = build_interpolation(V_p, V_u)

    if bdf is BDF1:
        u_star_x, u_star_y = ux_n, uy_n
    else:
        u_star_x = 2.0 * ux_n - ux_old
        u_star_y = 2.0 * uy_n - uy_old

    rhs_x, rhs_y, rhs_p = _coupled_ns_rhs_mixed_2d(
        V_u, V_p, nu=nu, dt=dt, bdf=bdf,
        ux_n=ux_n, uy_n=uy_n, ux_old=ux_old, uy_old=uy_old,
        ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
        p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
        M_ref_u=M_ref_u, m1d_per_face_u=m1d_per_face_u,
        viscous_bc_types_sipg=viscous_bc_sipg,
        P_emb=P_emb, tau_scale=tau_scale,
    )
    rhs_flat = _flat3_mixed(rhs_x, rhs_y, rhs_p)

    def matvec(v: jax.Array) -> jax.Array:
        ux, uy, p = _unflat3_mixed(v, V_u, V_p)
        out_x, out_y, out_p = coupled_ns_apply_mixed_2d(
            V_u, V_p, ux, uy, p,
            nu=nu, gamma0_over_dt=bdf.g0 / dt,
            u_star_x=u_star_x, u_star_y=u_star_y,
            ins_bc_types=ins_bc_types,
            viscous_bc_types_sipg=viscous_bc_sipg,
            M_ref_u=M_ref_u, m1d_per_face_u=m1d_per_face_u,
            P_emb=P_emb, tau_scale=tau_scale,
            tau_div=tau_div, tau_cont=tau_cont,
            pressure_eps=pressure_eps,
        )
        return _flat3_mixed(out_x, out_y, out_p)

    if x0_iterative is not None:
        x0 = _flat3_mixed(*x0_iterative)
    else:
        x0 = jnp.concatenate([
            ux_n.T.reshape(-1), uy_n.T.reshape(-1),
            jnp.zeros(V_p.Np * V_p.K),  # zero pressure guess
        ])

    n_u = V_u.Np * V_u.K
    n_p = V_p.Np * V_p.K
    if coupled_preconditioner is not None:
        M = coupled_preconditioner
    elif velocity_preconditioner is not None or pressure_schur_inv is not None:
        def M_inv(v):
            v_ux = v[:n_u]
            v_uy = v[n_u:2 * n_u]
            v_p = v[2 * n_u:2 * n_u + n_p]
            if velocity_preconditioner is not None:
                v_ux = velocity_preconditioner(v_ux)
                v_uy = velocity_preconditioner(v_uy)
            if pressure_schur_inv is not None:
                v_p = pressure_schur_inv(v_p)
            return jnp.concatenate([v_ux, v_uy, v_p])
        M = M_inv
    else:
        M = None

    def _solve_and_record(matvec_fn, b, x0_):
        if iterative.method == "gmres":
            x, _ = jax.scipy.sparse.linalg.gmres(
                matvec_fn, b, x0=x0_,
                tol=iterative.tol, atol=iterative.atol,
                maxiter=iterative.maxiter, restart=iterative.restart,
                M=M,
            )
        elif iterative.method == "bicgstab":
            x, _ = jax.scipy.sparse.linalg.bicgstab(
                matvec_fn, b, x0=x0_,
                tol=iterative.tol, atol=iterative.atol,
                maxiter=iterative.maxiter,
                M=M,
            )
        else:
            raise ValueError(f"unknown iterative method {iterative.method!r}")
        if convergence_out is not None:
            r0 = float(jnp.linalg.norm(b - matvec_fn(x0_)))
            r1 = float(jnp.linalg.norm(b - matvec_fn(x)))
            target = iterative.tol * r0 + iterative.atol
            convergence_out.inner_residuals_initial.append(r0)
            convergence_out.inner_residuals_achieved.append(r1)
            convergence_out.inner_converged.append(bool(r1 <= target * 1.1))
        return x

    if nl_config is None:
        sol = _solve_and_record(matvec, rhs_flat, x0)
        return _unflat3_mixed(sol, V_u, V_p)

    if nl_config.linearisation not in ("picard", "newton"):
        raise ValueError(
            f"unknown linearisation {nl_config.linearisation!r}; "
            "expected 'picard' or 'newton'"
        )
    use_newton = nl_config.linearisation == "newton"

    u_k_x, u_k_y = u_star_x, u_star_y
    p_k = jnp.zeros((V_p.Np, V_p.K))
    if x0_iterative is not None:
        u_k_x, u_k_y, p_k = x0_iterative

    def _full_apply_at(u_x_, u_y_, p_):
        return coupled_ns_apply_mixed_2d(
            V_u, V_p, u_x_, u_y_, p_,
            nu=nu, gamma0_over_dt=bdf.g0 / dt,
            u_star_x=u_x_, u_star_y=u_y_,
            ins_bc_types=ins_bc_types,
            viscous_bc_types_sipg=viscous_bc_sipg,
            M_ref_u=M_ref_u, m1d_per_face_u=m1d_per_face_u,
            P_emb=P_emb, tau_scale=tau_scale,
            tau_div=tau_div, tau_cont=tau_cont,
            pressure_eps=pressure_eps,
        )

    norm_F0 = None
    for k in range(nl_config.maxiter):
        F_u_x, F_u_y, F_p_minus = _full_apply_at(u_k_x, u_k_y, p_k)
        F_x = F_u_x - rhs_x
        F_y = F_u_y - rhs_y
        F_p = F_p_minus - rhs_p
        F_flat = _flat3_mixed(F_x, F_y, F_p)
        norm_F = jnp.linalg.norm(F_flat)
        if nl_residual_history_out is not None:
            nl_residual_history_out.append(float(norm_F))
        if convergence_out is not None:
            convergence_out.outer_residuals.append(float(norm_F))
        if norm_F0 is None:
            norm_F0 = norm_F
        if bool(norm_F <= nl_config.tol * norm_F0 + nl_config.atol):
            break

        newton_u_k = (u_k_x, u_k_y) if use_newton else None
        def J_matvec(v):
            ux_, uy_, p_ = _unflat3_mixed(v, V_u, V_p)
            out_x, out_y, out_p = coupled_ns_apply_mixed_2d(
                V_u, V_p, ux_, uy_, p_,
                nu=nu, gamma0_over_dt=bdf.g0 / dt,
                u_star_x=u_k_x, u_star_y=u_k_y,
                ins_bc_types=ins_bc_types,
                viscous_bc_types_sipg=viscous_bc_sipg,
                M_ref_u=M_ref_u, m1d_per_face_u=m1d_per_face_u,
                P_emb=P_emb, tau_scale=tau_scale,
                tau_div=tau_div, tau_cont=tau_cont,
                pressure_eps=pressure_eps,
                newton_u_k=newton_u_k,
                newton_method=nl_config.newton_method,
            )
            return _flat3_mixed(out_x, out_y, out_p)

        delta_flat = _solve_and_record(J_matvec, -F_flat,
                                        jnp.zeros_like(F_flat))
        delta_x, delta_y, delta_p = _unflat3_mixed(delta_flat, V_u, V_p)
        u_k_x = u_k_x + delta_x
        u_k_y = u_k_y + delta_y
        p_k = p_k + delta_p

    if convergence_out is not None and norm_F0 is not None:
        target = nl_config.tol * norm_F0 + nl_config.atol
        convergence_out.outer_converged = bool(norm_F <= target)
    return u_k_x, u_k_y, p_k
