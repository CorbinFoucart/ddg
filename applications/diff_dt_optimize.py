"""Differentiable dt optimization on the BDM_3 cavity Re=10000 case.

The finite-difference scan (``diff_stabilize_cavity.py``) revealed that
tau has zero leverage on the dt=2.0 CFL-limited blowup and that the
stability boundary at Re=10000, Kx=8, BDM_3 lies somewhere in
[0.5, 2.0]. This script puts a smooth differentiable loss

  L(dt) = log( max_t  ||u(t; dt)||_inf )
                + alpha * (-log dt)               (efficiency push)

over a fixed-shape JIT-able trajectory (N fixed BDF2 + fixed-Newton
inner iters, no early stop) and uses ``jax.grad`` + Adam to find the
maximum dt that keeps ||u||_inf bounded.

Design choices for differentiability:
* Fixed Newton iter count per BDF step (4) -- no convergence-based
  break, so the entire trajectory is one fixed-shape jax expression.
* Linearised Newton step: build the dense (n_u + n_p) saddle-point
  matrix at the current Picard point u_star = u_k, then solve via
  jax.scipy.linalg.solve. Newton extra term added column-wise via
  vmap. Strong BC pinning via the bc_mask -- boundary rows replaced
  by identity, boundary RHS zeroed.
* The whole trajectory is a python for-loop (not lax.scan) so JAX
  unrolls it -- simpler reverse-mode autograd, no scan-related pytree
  contortions.

Optimization: log(dt) is the trainable, so dt stays positive without
projection. Adam with a small lr.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "applications"))

from ddg.dg2d.bdm import (
    BDM, bdm_k_convective_apply, bdm_k_viscous_dirichlet_lift,
    bdm_k_evaluate_boundary_velocity,
)
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdf_history_k,
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    ns_picard_apply_bdm_k_2d, ns_picard_matrix_bdm_k_2d,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


U_LID = 1.0


def _lid_bc_fn(x, y):
    on_lid = jnp.abs(y - 1.0) < 1e-9
    return jnp.where(on_lid, U_LID, 0.0), jnp.zeros_like(x)


def build_static(Kx, order):
    """Build the order-k cavity setup -- everything not depending on dt."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Kx)
    mesh = build_mesh_2d(VX, VY, EToV, order)
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _lid_bc_fn)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _lid_bc_fn)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


def make_step(V, u_bc, u_bc_vec, bc_mask, *, nu, tau_scale, n_newton):
    """Build a fixed-Newton-iter BDF step function.

    Returns ``step(u_hist, dt, order_eff) -> u_new``, where ``u_hist``
    is a list of past velocity fields (length ``order_eff``). Pure
    JAX -- everything inside is JIT-compatible. ``order_eff`` is
    static (Python int) so we can branch on it for BDF1 vs BDF2.
    """
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n = n_u + n_p

    lift_u = bdm_k_viscous_dirichlet_lift(V, u_bc_vec, nu=nu, tau_scale=tau_scale)

    def step(u_n, u_nm1, dt, order_eff: int):
        coeffs = BDF_COEFFS[order_eff]
        alpha = coeffs["gamma0"] / dt
        # BDF source term inline (avoid bdf_history_k Python overhead).
        u_hist = (u_n, u_nm1)
        f_velocity = (coeffs["beta"][0] / dt) * _mass_apply(V, u_hist[0])
        if order_eff == 2:
            f_velocity = f_velocity + (
                coeffs["beta"][1] / dt
            ) * _mass_apply(V, u_hist[1])

        u_k = u_n
        p_k = jnp.zeros((V.K, dim_p))
        mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])

        for it in range(n_newton):
            # Newton residual at (u_k, p_k).
            F_u, F_p = ns_picard_apply_bdm_k_2d(
                V, u_k, p_k,
                u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=alpha,
                f_velocity=f_velocity,
            )
            F_u = F_u - lift_u
            F_u_inner = jnp.where(bc_mask, 0.0, F_u)
            F_p_flat = F_p.reshape(-1)
            rhs_full = jnp.concatenate([-F_u_inner, -F_p_flat])

            # Picard Jacobian (already includes alpha*M via velocity_alpha
            # in ns_picard_apply_bdm_k_2d -> ns_picard_matrix_bdm_k_2d).
            A_full = ns_picard_matrix_bdm_k_2d(
                V, u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=alpha,
            )

            # Newton extra column-by-column (the bilinear form
            # d/du_star at u_k applied to canonical basis e_i).
            def newton_col(i):
                e = jnp.zeros(n_u).at[i].set(1.0)
                return bdm_k_convective_apply(V, u_k, u_star_global=e)
            N_ext = jax.vmap(newton_col)(jnp.arange(n_u)).T
            A_full = A_full.at[:n_u, :n_u].add(N_ext)

            # Constrained solve: identity on boundary rows/cols.
            I_n = jnp.eye(n)
            A_c = jnp.where(
                mask_full[:, None], I_n,
                jnp.where(mask_full[None, :], 0.0, A_full),
            )
            delta = jnp.linalg.solve(A_c, rhs_full)
            delta_u = delta[:n_u]
            delta_p_flat = delta[n_u:]
            delta_u = jnp.where(bc_mask, 0.0, delta_u)
            u_k = u_k + delta_u
            p_k = p_k + delta_p_flat.reshape(V.K, dim_p)
        return u_k

    return step


def _mass_apply(V, u_global):
    """Inline BDM_k mass apply (avoid extra import indirection)."""
    from ddg.dg2d.coupled_ns_bdm import bdm_k_mass_apply
    return bdm_k_mass_apply(V, u_global)


def make_trajectory_loss(V, u_bc, u_bc_vec, bc_mask, *, nu, tau_scale,
                          n_newton, n_steps):
    """Build ``loss(dt) -> log(max_t ||u||_inf)`` over an n-step run.

    The trajectory is a python for-loop with the BDF1 bootstrap on
    step 0 and BDF2 thereafter. JAX unrolls into a single fixed-shape
    expression. The loss uses ``||u||_inf`` rather than KE to stay
    monotone in the blowup amplitude (KE is squared, so it grows
    even more dramatically and overflows in float64 sooner).
    """
    step1 = make_step(
        V, u_bc, u_bc_vec, bc_mask, nu=nu, tau_scale=tau_scale,
        n_newton=n_newton,
    )

    def trajectory(dt):
        u_n = u_bc
        u_nm1 = u_bc
        peak = jnp.linalg.norm(u_n, ord=jnp.inf)
        for s in range(n_steps):
            order_eff = 1 if s == 0 else 2
            u_new = step1(u_n, u_nm1, dt, order_eff)
            peak = jnp.maximum(peak, jnp.linalg.norm(u_new, ord=jnp.inf))
            u_nm1 = u_n
            u_n = u_new
        return peak

    # Loss design:
    #   L = (1/beta) * softplus(beta * (log(peak) - log(U_target)))
    #         - gamma * log(dt)
    #
    # Softplus barrier: smooth, non-saturating. Derivative goes from
    # 0 (deep bounded) through 1/2 (right at U_target) to ~1 (blown).
    # So gradient *keeps* pointing toward smaller dt no matter how far
    # into the blowup we are -- fixes the sigmoid-saturation issue
    # where Adam couldn't escape the blown regime.
    #
    # Equilibrium: -gamma + d(barrier)/d(log_dt) = 0, i.e. where the
    # barrier's local slope balances the efficiency push. Sits just
    # above the boundary at the chosen gamma.
    U_target = 1.5
    beta = 4.0
    gamma = 0.1

    def loss(log_dt):
        dt = jnp.exp(log_dt)
        peak = trajectory(dt)
        log_peak = jnp.log(peak + 1e-30)
        z = beta * (log_peak - jnp.log(U_target))
        barrier = jax.nn.softplus(z) / beta
        return barrier - gamma * log_dt

    return loss, trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=10000.0)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=8)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--n-newton", type=int, default=4,
                    help="fixed Newton iters per BDF step")
    ap.add_argument("--n-steps", type=int, default=4,
                    help="BDF2 steps in the trajectory")
    ap.add_argument("--log-dt-init", type=float, default=np.log(2.0),
                    help="initial log(dt) -- start in the blown regime")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    nu = U_LID * 1.0 / args.re
    V, u_bc, u_bc_vec, bc_mask = build_static(args.kx, args.order)
    n_u = V.total_dofs
    n_p = V.K * (args.order * (args.order + 1) // 2)
    print(f"BDM_{args.order} cavity Re={args.re}: K_x={args.kx}, "
          f"n_u={n_u}, n_p={n_p}, n={n_u + n_p}, nu={nu:.2e}", flush=True)

    loss_fn, traj_fn = make_trajectory_loss(
        V, u_bc, u_bc_vec, bc_mask, nu=nu, tau_scale=args.tau_scale,
        n_newton=args.n_newton, n_steps=args.n_steps,
    )
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    # Baseline: scan the loss landscape over a fine log(dt) grid first.
    log_dts = np.linspace(np.log(0.05), np.log(4.0), 12)
    print("\n=== Loss landscape (log_dt grid) ===", flush=True)
    landscape = []
    for ld in log_dts:
        t0 = time.time()
        L = float(loss_jit(jnp.asarray(ld)))
        landscape.append((float(ld), float(np.exp(ld)), L))
        print(f"  dt={np.exp(ld):6.3f}  log(dt)={ld:+.3f}  "
              f"loss={L:+.3f}  wall={time.time() - t0:.1f}s",
              flush=True)

    # Adam on log(dt) starting from the user's init point.
    log_dt = jnp.asarray(float(args.log_dt_init))
    m = jnp.zeros_like(log_dt); v = jnp.zeros_like(log_dt)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    print(f"\n=== Adam on log(dt) ===", flush=True)
    print(f"  init: dt={float(jnp.exp(log_dt)):.3f}, "
          f"loss={float(loss_jit(log_dt)):.3f}", flush=True)

    history = [(float(log_dt), float(jnp.exp(log_dt)),
                float(loss_jit(log_dt)))]
    t0 = time.time()
    # Gradient clip: at sharp cliffs the autograd through near-Inf
    # peak values yields O(1e10) gradients that poison Adam's
    # second-moment estimate for many iterations afterward. Clipping
    # to ±g_clip preserves the sign signal (which is all we need to
    # know whether to go up or down) without contaminating ``v``.
    g_clip = 1.0
    for it in range(1, args.n_train + 1):
        g = grad_fn(log_dt)
        g = jnp.clip(g, -g_clip, g_clip)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** it)
        v_hat = v / (1 - beta2 ** it)
        log_dt = log_dt - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        L = float(loss_jit(log_dt))
        history.append((float(log_dt), float(jnp.exp(log_dt)), L))
        print(f"  iter {it:3d}: dt={float(jnp.exp(log_dt)):6.3f}  "
              f"loss={L:+.3f}  grad={float(g):+.3f}",
              flush=True)
    print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)

    # Plot the loss landscape + Adam trajectory.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    dts_grid = np.array([d for (_, d, _) in landscape])
    losses_grid = np.array([L for (_, _, L) in landscape])
    axes[0].semilogx(dts_grid, losses_grid, "o-", ms=4)
    axes[0].set_xlabel(r"$\Delta t$")
    axes[0].set_ylabel(r"$\log \,\|u\|_\infty$ over trajectory")
    axes[0].set_title(
        f"Loss landscape, Re={args.re}, BDM_{args.order}, K_x={args.kx}"
    )
    axes[0].grid(alpha=0.3, which="both")
    axes[0].axhline(0.0, color="k", ls="--", alpha=0.5,
                     label=r"$\|u\|_\infty \sim 1$ (bounded)")
    axes[0].legend(fontsize=9)

    hist_dts = np.array([d for (_, d, _) in history])
    hist_loss = np.array([L for (_, _, L) in history])
    axes[1].semilogy(np.arange(len(hist_dts)), hist_dts, "o-", ms=3)
    axes[1].set_xlabel("Adam iter"); axes[1].set_ylabel(r"$\Delta t$")
    axes[1].set_title("Adam trajectory on $\\log(\\Delta t)$")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = args.out_dir / "diff_dt_optimize.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}", flush=True)
    summary = dict(
        re=float(args.re), order=int(args.order), kx=int(args.kx),
        tau_scale=float(args.tau_scale),
        n_newton=int(args.n_newton), n_steps=int(args.n_steps),
        landscape=[dict(log_dt=ld, dt=d, loss=L)
                   for (ld, d, L) in landscape],
        adam_history=[dict(log_dt=ld, dt=d, loss=L)
                      for (ld, d, L) in history],
    )
    with open(args.out_dir / "diff_dt_optimize.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'diff_dt_optimize.json'}", flush=True)


if __name__ == "__main__":
    main()
