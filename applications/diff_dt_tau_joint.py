"""Joint differentiable optimization of (Delta t, tau_scale) on the
BDM_3 cavity at Re=10000.

The dt-only experiment (``diff_dt_optimize.py``) localised the
stability boundary at Delta t ~ 1.6. The FD scan
(``diff_stabilize_cavity.py``) showed tau across 4 orders of
magnitude had zero leverage at Delta t = 2.0 -- but that was the
deeply-blown regime where the gradient signal saturates. Near the
cliff, tau *might* carry some signal that shifts the boundary.

This script jointly trains (log Delta t, log tau_scale) through the
same JIT-able BDF2 + fixed-Newton trajectory and the same softplus
barrier + dt-efficiency-push loss as ``diff_dt_optimize.py``. Two
scalars, Adam descent, gradient clipping at the cliff. The question
is whether Adam co-adapts tau to push the dt-boundary higher, or
whether tau drifts independently while dt does all the work.

Loss:
    L(log_dt, log_tau) = (1/beta) * softplus(beta * (log peak - log U_target))
                         - gamma_dt * log_dt

Note: no -gamma_tau * log_tau term. We don't want tau pushed in any
particular direction by the loss; tau should *only* move if Adam
discovers it helps the barrier. Equilibrium thus sits at
d(barrier)/d(log_tau) = 0, which on a flat-in-tau landscape means
tau just drifts diffusively (informative null).
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
    BDF_COEFFS,
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    bdm_k_mass_apply,
    ns_picard_apply_bdm_k_2d, ns_picard_matrix_bdm_k_2d,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


U_LID = 1.0


def _lid_bc_fn(x, y):
    on_lid = jnp.abs(y - 1.0) < 1e-9
    return jnp.where(on_lid, U_LID, 0.0), jnp.zeros_like(x)


def build_static(Kx, order):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Kx)
    mesh = build_mesh_2d(VX, VY, EToV, order)
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _lid_bc_fn)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _lid_bc_fn)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


def make_step(V, u_bc, u_bc_vec, bc_mask, *, nu, n_newton):
    """Fixed-Newton-iter BDF step with TRAINABLE tau_scale.

    tau_scale enters both the viscous matvec and the SIPG Dirichlet
    lift, so both depend on it via JAX tracing.
    """
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n = n_u + n_p

    def step(u_n, u_nm1, dt, tau_scale, order_eff: int):
        coeffs = BDF_COEFFS[order_eff]
        alpha = coeffs["gamma0"] / dt
        u_hist = (u_n, u_nm1)
        f_velocity = (coeffs["beta"][0] / dt) * bdm_k_mass_apply(V, u_hist[0])
        if order_eff == 2:
            f_velocity = f_velocity + (coeffs["beta"][1] / dt) * bdm_k_mass_apply(
                V, u_hist[1])

        lift_u = bdm_k_viscous_dirichlet_lift(
            V, u_bc_vec, nu=nu, tau_scale=tau_scale)

        u_k = u_n
        p_k = jnp.zeros((V.K, dim_p))
        mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])

        for it in range(n_newton):
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

            A_full = ns_picard_matrix_bdm_k_2d(
                V, u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=alpha,
            )
            def newton_col(i):
                e = jnp.zeros(n_u).at[i].set(1.0)
                return bdm_k_convective_apply(V, u_k, u_star_global=e)
            N_ext = jax.vmap(newton_col)(jnp.arange(n_u)).T
            A_full = A_full.at[:n_u, :n_u].add(N_ext)

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


def make_loss(V, u_bc, u_bc_vec, bc_mask, *, nu, n_newton, n_steps):
    step_fn = make_step(V, u_bc, u_bc_vec, bc_mask, nu=nu, n_newton=n_newton)

    def trajectory(dt, tau_scale):
        u_n = u_bc
        u_nm1 = u_bc
        peak = jnp.linalg.norm(u_n, ord=jnp.inf)
        for s in range(n_steps):
            order_eff = 1 if s == 0 else 2
            u_new = step_fn(u_n, u_nm1, dt, tau_scale, order_eff)
            peak = jnp.maximum(peak, jnp.linalg.norm(u_new, ord=jnp.inf))
            u_nm1 = u_n
            u_n = u_new
        return peak

    U_target = 1.5
    beta = 4.0
    gamma_dt = 0.1

    def loss(params):
        log_dt = params["log_dt"]
        log_tau = params["log_tau"]
        dt = jnp.exp(log_dt)
        tau_scale = jnp.exp(log_tau)
        peak = trajectory(dt, tau_scale)
        log_peak = jnp.log(peak + 1e-30)
        z = beta * (log_peak - jnp.log(U_target))
        barrier = jax.nn.softplus(z) / beta
        return barrier - gamma_dt * log_dt

    return loss, trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=10000.0)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=8)
    ap.add_argument("--n-newton", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--log-dt-init", type=float, default=-0.693)
    ap.add_argument("--log-tau-init", type=float, default=np.log(4.0))
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    nu = U_LID * 1.0 / args.re
    V, u_bc, u_bc_vec, bc_mask = build_static(args.kx, args.order)
    n_u = V.total_dofs
    n_p = V.K * (args.order * (args.order + 1) // 2)
    print(f"BDM_{args.order} cavity Re={args.re}: K_x={args.kx}, "
          f"n_u={n_u}, n_p={n_p}, n={n_u + n_p}, nu={nu:.2e}", flush=True)

    loss_fn, traj_fn = make_loss(
        V, u_bc, u_bc_vec, bc_mask, nu=nu,
        n_newton=args.n_newton, n_steps=args.n_steps,
    )
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    params = {
        "log_dt": jnp.asarray(float(args.log_dt_init)),
        "log_tau": jnp.asarray(float(args.log_tau_init)),
    }
    print(f"  init: dt={float(jnp.exp(params['log_dt'])):.3f}, "
          f"tau={float(jnp.exp(params['log_tau'])):.3f}, "
          f"loss={float(loss_jit(params)):.4f}", flush=True)

    # Adam state per param.
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v_st = {k: jnp.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    g_clip = 1.0

    history = [dict(
        iter=0,
        dt=float(jnp.exp(params["log_dt"])),
        tau=float(jnp.exp(params["log_tau"])),
        loss=float(loss_jit(params)),
        grad_log_dt=0.0, grad_log_tau=0.0,
    )]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(params)
        # Clip both per-param.
        g_clipped = {k: jnp.clip(v, -g_clip, g_clip) for k, v in g.items()}
        for k in params:
            m[k] = beta1 * m[k] + (1 - beta1) * g_clipped[k]
            v_st[k] = beta2 * v_st[k] + (1 - beta2) * g_clipped[k] ** 2
            m_hat = m[k] / (1 - beta1 ** it)
            v_hat = v_st[k] / (1 - beta2 ** it)
            params[k] = params[k] - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        L = float(loss_jit(params))
        history.append(dict(
            iter=int(it),
            dt=float(jnp.exp(params["log_dt"])),
            tau=float(jnp.exp(params["log_tau"])),
            loss=L,
            grad_log_dt=float(g["log_dt"]),
            grad_log_tau=float(g["log_tau"]),
        ))
        print(f"  iter {it:3d}: dt={float(jnp.exp(params['log_dt'])):5.3f}  "
              f"tau={float(jnp.exp(params['log_tau'])):6.3f}  "
              f"loss={L:+.3f}  g_dt={float(g['log_dt']):+.2e}  "
              f"g_tau={float(g['log_tau']):+.2e}",
              flush=True)
    print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)

    # Plot: trajectory in (dt, tau) plane + per-iter dt & tau curves.
    dts = [r["dt"] for r in history]
    taus = [r["tau"] for r in history]
    losses = [r["loss"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(dts, taus, "o-", ms=3, alpha=0.7)
    axes[0].plot([dts[0]], [taus[0]], "go", ms=10, label="init")
    axes[0].plot([dts[-1]], [taus[-1]], "rs", ms=10, label="final")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\Delta t$"); axes[0].set_ylabel(r"$\tau_{\rm scale}$")
    axes[0].set_title(
        f"Joint trajectory (Re={args.re}, BDM_{args.order}, K_x={args.kx})"
    )
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend()
    axes[1].plot(range(len(dts)), dts, "-o", ms=2, label=r"$\Delta t$")
    axes[1].plot(range(len(taus)), taus, "-s", ms=2, label=r"$\tau$")
    axes[1].set_yscale("log"); axes[1].set_xlabel("Adam iter")
    axes[1].set_title("Per-iter trainables"); axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    axes[2].plot(range(len(losses)), losses, "-o", ms=2)
    axes[2].set_xlabel("Adam iter"); axes[2].set_ylabel("loss")
    axes[2].set_title("Loss"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    out = args.out_dir / "diff_dt_tau_joint.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}", flush=True)
    summary = dict(
        re=float(args.re), order=int(args.order), kx=int(args.kx),
        n_newton=int(args.n_newton), n_steps=int(args.n_steps),
        history=history,
    )
    with open(args.out_dir / "diff_dt_tau_joint.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'diff_dt_tau_joint.json'}", flush=True)


if __name__ == "__main__":
    main()
