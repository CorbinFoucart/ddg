"""Quantitative head-to-head between the BDM_k block-Schur projection
solver (``ddg.dg2d.projection_ns_bdm``) and the monolithic BDM Newton
solver (``ns_picard_apply_bdm_k_2d`` + dense LU) on the verification
suite. Local-only, smallish meshes; writes a JSON record + a plain-
text summary to ``output/projection_verification/``.

Sentinels
---------
1. **Kovasznay drift** (Re=40, BDM_1).
   Start at the canonical BDM interpolant of the analytic Kovasznay
   solution. March each solver with BDF2 for ``T_final`` and report
   the L^2 velocity drift from the analytic truth at the final time.
   Same problem on three mesh sizes -- comparable accuracy + the
   monolithic / projection drift ratio.

2. **Lid-driven cavity Re=100** (BDM_1).
   Standard Ghia benchmark. March from rest to near-steady, sample
   u along the vertical centreline at Ghia y points, report RMSE
   against the Ghia 1982 table.

3. **Scaling sweep** (BDM_1, cavity setup).
   Per-step wall as a function of mesh size K. Verifies the
   block-Schur speedup holds (and scales) as K grows.

Each sentinel reports for both solvers:
  - accuracy metric (analytic L^2 error or Ghia RMSE)
  - wall-clock per step
  - peak ||Du|| over the rollout (divergence health)

Run
---
    python applications/projection_vs_monolithic_verification.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))

from ddg.dg2d.bdm import (
    BDM, bdm_k_basis_at_points, bdm_k_canonical_interpolant,
    bdm_k_divergence_apply,
    bdm_k_evaluate_boundary_velocity, bdm_k_mass_apply,
    bdm_k_viscous_dirichlet_lift, bdm1_gather,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    ns_picard_apply_bdm_k_2d, ns_picard_matrix_bdm_k_2d,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.projection_ns_bdm import (
    make_projection_solvers, projection_step_bdm_k_2d,
)


# ---------------------------------------------------------------------------
# Shared step driver: monolithic Picard (n_newton=1, dense LU).
# ---------------------------------------------------------------------------

def make_monolithic_step(V, u_bc, u_bc_vec, bc_mask, is_natural_face,
                          *, dt, nu, order, tau_scale=4.0):
    """One BDF{order} monolithic Picard step (single iter, dense LU).

    Same structure as
    :func:`scripts.inverse_nu_cylinder_2d.make_step_nu_param` with
    ``n_newton = 1``.
    """
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n_full = n_u + n_p
    coeffs = BDF_COEFFS[order]
    alpha = coeffs["gamma0"] / dt
    lift_u = bdm_k_viscous_dirichlet_lift(
        V, u_bc_vec, nu=nu, tau_scale=tau_scale,
        is_natural_face=is_natural_face,
    )
    mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])
    I_n = jnp.eye(n_full)

    def step(u_n, u_nm1):
        if order == 1:
            f_velocity = (coeffs["beta"][0] / dt) * bdm_k_mass_apply(V, u_n)
        else:
            f_velocity = (
                (coeffs["beta"][0] / dt) * bdm_k_mass_apply(V, u_n)
                + (coeffs["beta"][1] / dt) * bdm_k_mass_apply(V, u_nm1)
            )
        u_extrap = u_n if order == 1 else 2.0 * u_n - u_nm1
        # Homogeneous form (u_bc_global=None -> a_homog) + explicit lift,
        # consistent with the Jacobian ns_picard_matrix below (also a_homog)
        # and with the fixed projection_step. Passing u_bc_global here gave
        # a_nh and, with -lift_u, double-counted the boundary (same bug as
        # the projection had) -- residual/matrix inconsistency.
        F_u, F_p = ns_picard_apply_bdm_k_2d(
            V, u_extrap, jnp.zeros((V.K, dim_p)),
            u_star_global=u_extrap, nu=nu,
            pressure_eps=1.0e-10, tau_scale=tau_scale,
            velocity_alpha=alpha,
            f_velocity=f_velocity,
            u_bc_global=None,
            is_natural_face=is_natural_face,
        )
        F_u = F_u - lift_u
        F_u_inner = jnp.where(bc_mask, 0.0, F_u)
        rhs_full = jnp.concatenate([-F_u_inner, -F_p.reshape(-1)])
        A_full = ns_picard_matrix_bdm_k_2d(
            V, u_star_global=u_extrap, nu=nu,
            pressure_eps=1.0e-10, tau_scale=tau_scale,
            velocity_alpha=alpha,
            is_natural_face=is_natural_face,
        )
        A_c = jnp.where(
            mask_full[:, None], I_n,
            jnp.where(mask_full[None, :], 0.0, A_full),
        )
        delta = jnp.linalg.solve(A_c, rhs_full)
        u_new = u_extrap + delta[:n_u]
        p_new = delta[n_u:].reshape(V.K, dim_p)
        return u_new, p_new

    return jax.jit(step)


def make_projection_step(V, u_bc, u_bc_vec, bc_mask, is_natural_face,
                          *, dt, nu, order, tau_scale=4.0):
    pressure_solver, h_inv = make_projection_solvers(
        V, dt=dt, nu=nu, order=order, tau_scale=tau_scale,
        bc_mask=bc_mask, is_natural_face=is_natural_face,
    )

    def step(u_n, u_nm1):
        u_hist = [u_n] if order == 1 else [u_n, u_nm1]
        return projection_step_bdm_k_2d(
            V, u_hist, dt=dt, nu=nu, order=order,
            pressure_solver=pressure_solver, h_inv=h_inv,
            u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
            is_natural_face=is_natural_face,
            tau_scale=tau_scale,
        )

    return jax.jit(step)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def l2_velocity_error(V: BDM, u_global, ux_fn):
    """L^2 error against an analytic velocity ``ux_fn(x, y) -> (ux, uy)``."""
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 2)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum("kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    u_h = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)

    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    r_np = np.asarray(r_q); s_np = np.asarray(s_q)
    x_q = (
        Vx0[:, None]
        + 0.5 * (r_np[None, :] + 1.0) * (Vx1 - Vx0)[:, None]
        + 0.5 * (s_np[None, :] + 1.0) * (Vx2 - Vx0)[:, None]
    )
    y_q = (
        Vy0[:, None]
        + 0.5 * (r_np[None, :] + 1.0) * (Vy1 - Vy0)[:, None]
        + 0.5 * (s_np[None, :] + 1.0) * (Vy2 - Vy0)[:, None]
    )
    ux_e, uy_e = ux_fn(jnp.asarray(x_q), jnp.asarray(y_q))
    dx = u_h[..., 0] - ux_e
    dy = u_h[..., 1] - uy_e
    sq = dx * dx + dy * dy
    return float(jnp.sqrt(jnp.einsum("q,k,kq->", w_q, J, sq)))


def rollout(step_fn, u_init, n_steps, order):
    """Run ``n_steps`` BDF{order} steps, return (u_n, peak_div_norm,
    per_step_wall, total_wall)."""
    u_n = u_init; u_nm1 = u_init
    peak_div = 0.0
    times = []
    # Warm-up JIT (don't count toward wall)
    u_new, _ = step_fn(u_n, u_nm1)
    u_new.block_until_ready()
    u_nm1, u_n = u_n, u_new
    peak_div = max(peak_div,
                    float(jnp.linalg.norm(bdm_k_divergence_apply(
                        step_fn.__self__ if False else None, u_new
                    ))) if False else 0.0)
    for s in range(1, n_steps):
        t0 = time.time()
        u_new, _ = step_fn(u_n, u_nm1)
        u_new.block_until_ready()
        times.append(time.time() - t0)
        u_nm1, u_n = u_n, u_new
    total = float(np.sum(times))
    per_step = total / max(1, len(times))
    return u_n, per_step, total


def divergence_norm(V, u):
    return float(jnp.linalg.norm(bdm_k_divergence_apply(V, u)))


# ---------------------------------------------------------------------------
# Sentinel 1: Kovasznay spatial convergence (BDM_1)
# ---------------------------------------------------------------------------

RE_KOV = 40.0; NU_KOV = 1.0 / RE_KOV
LAMBDA_KOV = RE_KOV / 2.0 - np.sqrt(RE_KOV ** 2 / 4.0 + 4.0 * np.pi ** 2)


def kov_uv(x, y):
    ux = 1.0 - jnp.exp(LAMBDA_KOV * x) * jnp.cos(2.0 * jnp.pi * y)
    uy = (LAMBDA_KOV / (2.0 * jnp.pi)) * jnp.exp(LAMBDA_KOV * x) * jnp.sin(
        2.0 * jnp.pi * y
    )
    return ux, uy


def run_kovasznay_convergence(out_dir: Path, *, order=1, ns=(4, 6, 8),
                                T_final=1.0, dt=0.025):
    print(f"\n=== Kovasznay drift (BDM_{order}) ===", flush=True)
    n_steps = int(round(T_final / dt))
    results = []
    for n in ns:
        VX, VY, EToV = uniform_rectangle_mesh_2d(
            -0.5, 1.0, -0.5, 1.5, n, n,
        )
        mesh = build_mesh_2d(VX, VY, EToV, 1)
        V = BDM.from_mesh(mesh, order=order)
        u_bc = bdm1_evaluate_dirichlet_bc(V, kov_uv)
        u_bc_vec = bdm_k_evaluate_boundary_velocity(V, kov_uv)
        bc_mask = bdm1_boundary_dofs_mask(V)

        # IC: BDM canonical interpolant of the analytic Kovasznay
        # (matches both boundary and interior DOFs to the truth, so any
        # drift over the integration window is purely the solver's
        # transient behaviour, not an IC-relaxation transient).
        u_init = bdm_k_canonical_interpolant(V, kov_uv)
        h = 1.5 / n
        rec = dict(n=n, h=h, K=int(V.K), n_u=int(V.total_dofs))
        for solver_name, make_step in [
            ("monolithic", make_monolithic_step),
            ("projection", make_projection_step),
        ]:
            step = make_step(V, u_bc, u_bc_vec, bc_mask, None,
                              dt=dt, nu=NU_KOV, order=2)
            # bootstrap with one BDF1 step
            step_bdf1 = make_step(V, u_bc, u_bc_vec, bc_mask, None,
                                   dt=dt, nu=NU_KOV, order=1)
            u_n = u_init; u_nm1 = u_init
            u_new, _ = step_bdf1(u_n, u_nm1)
            u_new.block_until_ready()
            u_nm1, u_n = u_n, u_new
            peak_div = divergence_norm(V, u_n)
            t0 = time.time()
            for _ in range(n_steps - 1):
                u_new, _ = step(u_n, u_nm1)
                u_new.block_until_ready()
                u_nm1, u_n = u_n, u_new
                peak_div = max(peak_div, divergence_norm(V, u_n))
            total_wall = time.time() - t0
            err = l2_velocity_error(V, u_n, kov_uv)
            rec[solver_name] = dict(
                L2_err=err, wall=total_wall,
                wall_per_step=total_wall / max(1, n_steps - 1),
                peak_div=peak_div,
            )
            print(f"  n={n:2d} {solver_name:11s}: L2={err:.3e}  "
                  f"per-step={rec[solver_name]['wall_per_step']*1e3:.1f}ms  "
                  f"peak||Du||={peak_div:.2e}",
                  flush=True)
        results.append(rec)
    # Convergence slopes
    for solver_name in ["monolithic", "projection"]:
        errs = [r[solver_name]["L2_err"] for r in results]
        hs = [r["h"] for r in results]
        slopes = []
        for i in range(len(hs) - 1):
            slopes.append(
                (np.log(errs[i]) - np.log(errs[i + 1])) /
                (np.log(hs[i]) - np.log(hs[i + 1]))
            )
        print(f"  {solver_name} slopes: {[f'{s:.2f}' for s in slopes]}",
              flush=True)
    out = dict(sentinel="kovasznay", order=order, T_final=T_final, dt=dt,
                results=results)
    with open(out_dir / "kovasznay.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Taylor-Green temporal-rate sentinel with time-varying BC
# ---------------------------------------------------------------------------

NU_TG = 0.1


def tg_uv_at(t, nu=NU_TG):
    e = np.exp(-2.0 * nu * t)

    def f(x, y):
        ux = -jnp.cos(x) * jnp.sin(y) * e
        uy = jnp.sin(x) * jnp.cos(y) * e
        return ux, uy

    return f


def run_taylor_green(out_dir: Path, *, order=1, n=8,
                      dts=(0.10, 0.05, 0.025), T_final=0.5):
    """Time-varying-BC Taylor-Green temporal-rate test.

    Rebuilds the LU factors at each step (cheap-ish at small mesh)
    because the lift depends on time; the per-step wall is dominated
    by factor construction so the absolute timing isn't apples-to-
    apples with the cached-LU sentinels. Goal is purely the temporal
    convergence rate of L^2 error vs ``dt`` for each solver.
    """
    print(f"\n=== Taylor-Green temporal rate (BDM_{order}, n={n}) ===",
          flush=True)
    L = 2.0 * float(np.pi)
    results = []
    for dt in dts:
        VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, L, 0.0, L, n, n)
        mesh = build_mesh_2d(VX, VY, EToV, 1)
        V = BDM.from_mesh(mesh, order=order)
        bc_mask = bdm1_boundary_dofs_mask(V)
        n_steps = int(round(T_final / dt))
        rec = dict(dt=dt, n_steps=n_steps, n=n,
                    K=int(V.K), n_u=int(V.total_dofs))

        # IC: canonical interpolant at t=0.
        u_init = bdm_k_canonical_interpolant(V, tg_uv_at(0.0))
        u_truth_fn = tg_uv_at(T_final)

        for solver_name, make_step in [
            ("monolithic", make_monolithic_step),
            ("projection", make_projection_step),
        ]:
            u_n = u_init; u_nm1 = u_init
            t0 = time.time()
            for s in range(n_steps):
                t_target = (s + 1) * dt
                order_eff = 1 if s == 0 else 2
                u_bc_t = bdm1_evaluate_dirichlet_bc(V, tg_uv_at(t_target))
                u_bc_vec_t = bdm_k_evaluate_boundary_velocity(
                    V, tg_uv_at(t_target),
                )
                # New step closure per timestep -- LU rebuilt.
                step = make_step(V, u_bc_t, u_bc_vec_t, bc_mask, None,
                                  dt=dt, nu=NU_TG, order=order_eff)
                u_new, _ = step(u_n, u_nm1)
                u_new.block_until_ready()
                u_nm1, u_n = u_n, u_new
            wall = time.time() - t0
            err = l2_velocity_error(V, u_n, u_truth_fn)
            rec[solver_name] = dict(L2_err=err, wall=wall)
            print(f"  dt={dt:.4f} {solver_name:11s}: L2={err:.3e}",
                  flush=True)
        results.append(rec)
    for solver_name in ["monolithic", "projection"]:
        errs = [r[solver_name]["L2_err"] for r in results]
        dts_l = [r["dt"] for r in results]
        slopes = []
        for i in range(len(dts_l) - 1):
            slopes.append(
                (np.log(errs[i]) - np.log(errs[i + 1])) /
                (np.log(dts_l[i]) - np.log(dts_l[i + 1]))
            )
        print(f"  {solver_name} slopes: {[f'{s:.2f}' for s in slopes]}",
              flush=True)
    out = dict(sentinel="taylor_green", order=order, T_final=T_final,
                results=results)
    with open(out_dir / f"taylor_green_k{order}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Sentinel 3: Scaling sweep
# ---------------------------------------------------------------------------

def run_scaling_sweep(out_dir: Path, *, order=1, ns=(6, 10, 14),
                        T_final=0.5, dt=0.05):
    """Per-step wall time as a function of mesh size, using the cavity
    Re=100 setup. Verifies the block-Schur speedup holds (and grows)
    as K increases."""
    print(f"\n=== Scaling sweep (BDM_{order}, cavity setup) ===",
          flush=True)
    n_steps = int(round(T_final / dt))
    NU = 0.01
    results = []
    for n in ns:
        VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, n, n)
        mesh = build_mesh_2d(VX, VY, EToV, 1)
        V = BDM.from_mesh(mesh, order=order)
        u_bc = bdm1_evaluate_dirichlet_bc(V, cavity_bc)
        u_bc_vec = bdm_k_evaluate_boundary_velocity(V, cavity_bc)
        bc_mask = bdm1_boundary_dofs_mask(V)
        rec = dict(n=n, K=int(V.K), n_u=int(V.total_dofs))
        for solver_name, make_step in [
            ("monolithic", make_monolithic_step),
            ("projection", make_projection_step),
        ]:
            step_bdf1 = make_step(V, u_bc, u_bc_vec, bc_mask, None,
                                   dt=dt, nu=NU, order=1)
            step_bdf2 = make_step(V, u_bc, u_bc_vec, bc_mask, None,
                                   dt=dt, nu=NU, order=2)
            u_n = jnp.zeros(V.total_dofs); u_nm1 = jnp.zeros(V.total_dofs)
            # Warm-up + bootstrap
            u_new, _ = step_bdf1(u_n, u_nm1)
            u_new.block_until_ready()
            u_nm1, u_n = u_n, u_new
            t0 = time.time()
            for _ in range(n_steps - 1):
                u_new, _ = step_bdf2(u_n, u_nm1)
                u_new.block_until_ready()
                u_nm1, u_n = u_n, u_new
            wall = time.time() - t0
            rec[solver_name] = dict(
                wall=wall, wall_per_step=wall / max(1, n_steps - 1),
            )
            print(f"  n={n:2d} K={int(V.K):4d} n_u={int(V.total_dofs):5d}  "
                  f"{solver_name:11s}: per-step="
                  f"{rec[solver_name]['wall_per_step']*1e3:7.1f}ms",
                  flush=True)
        speedup = (
            rec["monolithic"]["wall_per_step"]
            / rec["projection"]["wall_per_step"]
        )
        rec["speedup"] = speedup
        print(f"        speedup = {speedup:.2f}x", flush=True)
        results.append(rec)
    out = dict(sentinel="scaling", order=order, dt=dt, T_final=T_final,
                results=results)
    with open(out_dir / "scaling.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Sentinel 3: Lid-driven cavity Re=100
# ---------------------------------------------------------------------------

GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000,
])
GHIA_U_RE100 = np.array([
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
    -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
    0.68717, 0.73722, 0.78871, 0.84123, 1.00000,
])
GHIA_U_RE400 = np.array([
    0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
    -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
    0.55892, 0.61756, 0.68439, 0.75837, 1.00000,
])
GHIA_TABLES = {100: GHIA_U_RE100, 400: GHIA_U_RE400}


def cavity_bc(x, y):
    on_lid = y > 1.0 - 1.0e-6
    ux = jnp.where(on_lid, 1.0, 0.0)
    uy = jnp.zeros_like(y)
    return ux, uy


def run_cavity(out_dir: Path, *, order=1, n=8, T_final=10.0,
                 dt=0.05, Re=100.0, ghia_table=None):
    print(f"\n=== Lid-driven cavity Re={int(Re)} (BDM_{order}, n={n}) ===",
          flush=True)
    n_steps = int(round(T_final / dt))
    NU_CAV = 1.0 / Re
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, n, n)
    mesh = build_mesh_2d(VX, VY, EToV, 1)
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, cavity_bc)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, cavity_bc)
    bc_mask = bdm1_boundary_dofs_mask(V)
    rec = dict(order=order, n=n, dt=dt, T_final=T_final, Re=Re,
                K=int(V.K), n_u=int(V.total_dofs))

    # Sample u_x along the vertical centreline x = 0.5 at Ghia y points
    # via the BDM evaluator.
    from ddg.dg2d.bdm import bdm_k_evaluate_at_points
    x_query = 0.5 * np.ones_like(GHIA_Y)
    y_query = GHIA_Y

    for solver_name, make_step in [
        ("monolithic", make_monolithic_step),
        ("projection", make_projection_step),
    ]:
        step_bdf1 = make_step(V, u_bc, u_bc_vec, bc_mask, None,
                                dt=dt, nu=NU_CAV, order=1)
        step_bdf2 = make_step(V, u_bc, u_bc_vec, bc_mask, None,
                                dt=dt, nu=NU_CAV, order=2)
        u_n = jnp.zeros(V.total_dofs); u_nm1 = jnp.zeros(V.total_dofs)
        u_new, _ = step_bdf1(u_n, u_nm1)
        u_new.block_until_ready()
        u_nm1, u_n = u_n, u_new
        peak_div = divergence_norm(V, u_n)
        t0 = time.time()
        for _ in range(n_steps - 1):
            u_new, _ = step_bdf2(u_n, u_nm1)
            u_new.block_until_ready()
            u_nm1, u_n = u_n, u_new
            peak_div = max(peak_div, divergence_norm(V, u_n))
        wall = time.time() - t0
        ux, _ = bdm_k_evaluate_at_points(V, u_n,
                                          jnp.asarray(x_query),
                                          jnp.asarray(y_query))
        ux_np = np.asarray(ux)
        ghia_ref = ghia_table if ghia_table is not None else (
            GHIA_TABLES.get(int(Re))
        )
        if ghia_ref is None:
            ghia_rmse = float("nan")
        else:
            ghia_rmse = float(np.sqrt(np.mean((ux_np - ghia_ref) ** 2)))
        rec[solver_name] = dict(
            ghia_rmse=ghia_rmse, wall=wall,
            wall_per_step=wall / max(1, n_steps - 1),
            peak_div=peak_div,
            ux_centreline=ux_np.tolist(),
        )
        print(f"  {solver_name:11s}: Ghia RMSE={ghia_rmse:.3e}  "
              f"per-step={rec[solver_name]['wall_per_step']*1e3:.1f}ms  "
              f"peak||Du||={peak_div:.2e}",
              flush=True)
    with open(out_dir / f"cavity_re{int(Re)}_k{order}.json", "w") as f:
        json.dump(rec, f, indent=2)
    return rec


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(out_dir: Path, sections):
    lines = []
    lines.append("Projection (block-Schur) vs Monolithic (Picard n_newton=1)")
    lines.append("=" * 64)
    for title, table in sections:
        lines.append("")
        lines.append(f"== {title} ==")
        lines.extend(table)
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(text + "\n")


def _kov_table(kov):
    lines = [
        f"{'n':>4} {'h':>8} | {'mono L2':>10} {'mono ms/step':>13} | "
        f"{'proj L2':>10} {'proj ms/step':>13} | {'speedup':>8}"
    ]
    for r in kov["results"]:
        em = r["monolithic"]["L2_err"]; ep = r["projection"]["L2_err"]
        mm = r["monolithic"]["wall_per_step"] * 1e3
        pm = r["projection"]["wall_per_step"] * 1e3
        lines.append(
            f"{r['n']:>4} {r['h']:>8.4f} | {em:>10.3e} {mm:>13.1f} | "
            f"{ep:>10.3e} {pm:>13.1f} | {mm/pm:>7.2f}x"
        )
    return lines


def _cav_table(cav):
    mm = cav["monolithic"]["wall_per_step"] * 1e3
    pm = cav["projection"]["wall_per_step"] * 1e3
    return [
        f"  monolithic: Ghia RMSE = {cav['monolithic']['ghia_rmse']:.3e}  "
        f"per-step = {mm:7.1f} ms  peak||Du|| = "
        f"{cav['monolithic']['peak_div']:.2e}",
        f"  projection: Ghia RMSE = {cav['projection']['ghia_rmse']:.3e}  "
        f"per-step = {pm:7.1f} ms  peak||Du|| = "
        f"{cav['projection']['peak_div']:.2e}",
        f"  speedup: {mm/pm:.2f}x",
    ]


def _tg_table(tg):
    lines = [
        f"{'dt':>8} | {'mono L2':>10} {'mono slope':>11} | "
        f"{'proj L2':>10} {'proj slope':>11}"
    ]
    last_em, last_ep, last_dt = None, None, None
    for r in tg["results"]:
        em = r["monolithic"]["L2_err"]; ep = r["projection"]["L2_err"]
        sm = "  -  " if last_em is None else (
            f"{(np.log(last_em)-np.log(em))/(np.log(last_dt)-np.log(r['dt'])):.2f}"
        )
        sp = "  -  " if last_ep is None else (
            f"{(np.log(last_ep)-np.log(ep))/(np.log(last_dt)-np.log(r['dt'])):.2f}"
        )
        lines.append(
            f"{r['dt']:>8.4f} | {em:>10.3e} {sm:>11} | {ep:>10.3e} {sp:>11}"
        )
        last_em, last_ep, last_dt = em, ep, r["dt"]
    return lines


def _scale_table(scale):
    lines = [
        f"{'n':>4} {'K':>5} {'n_u':>6} | "
        f"{'mono ms/step':>13} {'proj ms/step':>13} | {'speedup':>8}"
    ]
    for r in scale["results"]:
        mm = r["monolithic"]["wall_per_step"] * 1e3
        pm = r["projection"]["wall_per_step"] * 1e3
        lines.append(
            f"{r['n']:>4} {r['K']:>5} {r['n_u']:>6} | "
            f"{mm:>13.1f} {pm:>13.1f} | {r['speedup']:>7.2f}x"
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                     default=ROOT / "output" / "projection_verification")
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--skip-tg", action="store_true",
                     help="skip the (slow) time-varying-BC Taylor-Green")
    ap.add_argument("--re-list", type=float, nargs="+", default=[100, 400])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    for order in args.orders:
        kov = run_kovasznay_convergence(args.out_dir, order=order)
        sections.append(
            (f"Kovasznay drift (BDM_{order}, Re=40)", _kov_table(kov))
        )
        for Re in args.re_list:
            # Bigger n for Re=400 to capture the secondary vortex.
            n = 8 if Re <= 100 else 12
            T = 10.0 if Re <= 100 else 20.0
            cav = run_cavity(args.out_dir, order=order, n=n,
                              T_final=T, dt=0.05, Re=float(Re))
            sections.append(
                (f"Cavity Re={int(Re)} (BDM_{order}, n={n})",
                 _cav_table(cav))
            )
        if not args.skip_tg:
            tg = run_taylor_green(args.out_dir, order=order)
            sections.append(
                (f"Taylor-Green temporal rate (BDM_{order}, BDF2)",
                 _tg_table(tg))
            )
        scale = run_scaling_sweep(args.out_dir, order=order)
        sections.append(
            (f"Scaling sweep (BDM_{order}, cavity setup)",
             _scale_table(scale))
        )

    write_summary(args.out_dir, sections)


if __name__ == "__main__":
    main()
