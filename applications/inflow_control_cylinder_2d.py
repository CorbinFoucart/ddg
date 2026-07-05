"""Optimal inflow control for the BDM cylinder wake.

2-D analogue of ``applications/inflow_control_burgers_1d.py``. The control
variable is a time-varying inflow multiplier ``alpha(t)`` applied to the
canonical parabolic profile at ``x = X_MIN``; the loss is the integrated
squared vorticity in a downstream window at ``t = T_final`` plus a
mass-flux regulariser ``lambda_mass * (mean(alpha) - 1)**2`` that
prevents the degenerate ``alpha = 0`` solution.

Setup
-----
* BDM_2 cylinder smoke (Re=100, K=432); same geometry as
  ``scripts/diff_cylinder_dt_tau.py``.
* Trajectory: fixed-shape ``n_steps`` BDF2 transient from rest, with
  ``n_newton`` Newton iters per step. JIT'd once via ``jax.jit``;
  ``jax.grad`` backprops through the whole thing.
* Control: piecewise-constant ``alpha_s`` for ``s = 0, ..., n_steps-1``,
  bounded via sigmoid: ``alpha_s = alpha_min + (alpha_max - alpha_min)
  * sigmoid(theta_s)``.
* Loss:

      L(theta) = integral_{Omega_down} omega(x, T)**2 dx
               + lambda_mass * (mean(alpha) - 1)**2

  where ``Omega_down`` is the post-cylinder window
  ``[CX + 1.5 D, CX + 6 D] x [Y_MIN, Y_MAX]``.

Outputs
-------
``output/inflow_control_cylinder_k2.{png,json}`` -- alpha(t) history,
loss curve, baseline (alpha=1 throughout) vs optimised wake fields.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diff_cylinder_dt_tau import (
    build_static, _bc_fn,
    X_MIN, X_MAX, Y_MIN, Y_MAX, CX, CY, RADIUS, D_CYL,
    U_MAX, U_MEAN, NU, H_CHANNEL,
)
from ddg.dg2d.bdm import (
    bdm_k_basis_at_points, bdm_k_convective_apply,
    bdm_k_viscous_dirichlet_lift, bdm1_gather,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdm_k_mass_apply,
    ns_picard_apply_bdm_k_2d, ns_picard_matrix_bdm_k_2d,
)


def make_controlled_step(V, u_bc, u_bc_vec, bc_mask, is_natural_face,
                          *, nu, tau_scale, n_newton):
    """Same as ``diff_cylinder_dt_tau.make_step`` but with an inflow
    multiplier ``alpha`` threaded through ``u_bc`` and ``u_bc_vec``."""
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n_full = n_u + n_p

    def step(u_n, u_nm1, dt, alpha, order_eff: int):
        coeffs = BDF_COEFFS[order_eff]
        velocity_alpha = coeffs["gamma0"] / dt
        f_velocity = (coeffs["beta"][0] / dt) * bdm_k_mass_apply(V, u_n)
        if order_eff == 2:
            f_velocity = f_velocity + (
                coeffs["beta"][1] / dt
            ) * bdm_k_mass_apply(V, u_nm1)

        u_bc_t = alpha * u_bc
        u_bc_vec_t = alpha * u_bc_vec

        lift_u = bdm_k_viscous_dirichlet_lift(
            V, u_bc_vec_t, nu=nu, tau_scale=tau_scale,
            is_natural_face=is_natural_face,
        )

        u_k = alpha * u_n   # warm-start scale (small effect; for stability)
        p_k = jnp.zeros((V.K, dim_p))
        mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])

        for _it in range(n_newton):
            F_u, F_p = ns_picard_apply_bdm_k_2d(
                V, u_k, p_k,
                u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=velocity_alpha,
                f_velocity=f_velocity,
                u_bc_global=u_bc_t,
                is_natural_face=is_natural_face,
            )
            F_u = F_u - lift_u
            F_u_inner = jnp.where(bc_mask, 0.0, F_u)
            F_p_flat = F_p.reshape(-1)
            rhs_full = jnp.concatenate([-F_u_inner, -F_p_flat])

            A_full = ns_picard_matrix_bdm_k_2d(
                V, u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=velocity_alpha,
                is_natural_face=is_natural_face,
            )

            def newton_col(i):
                e = jnp.zeros(n_u).at[i].set(1.0)
                return bdm_k_convective_apply(
                    V, u_k, u_star_global=e,
                    is_natural_face=is_natural_face,
                )
            N_ext = jax.vmap(newton_col)(jnp.arange(n_u)).T
            A_full = A_full.at[:n_u, :n_u].add(N_ext)

            I_n = jnp.eye(n_full)
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
        return u_k, p_k
    return step


def _make_velocity_at_nodes_fn(V):
    """Return (u_global,) -> (ux_nodal, uy_nodal) each (Np, K)."""
    k = V.order
    mesh = V.mesh
    r, s = mesh.r, mesh.s
    phi_ref = bdm_k_basis_at_points(k, r, s)
    DF, _ = _element_jacobian_matrices(mesh)
    J = mesh.J[0]
    def fn(u_global):
        phi_phys = jnp.einsum(
            "kca,nla->knlc", DF, phi_ref) / J[:, None, None, None]
        u_local = bdm1_gather(V.topology, u_global)
        u_nodes = jnp.einsum("kl,knlc->knc", u_local, phi_phys)
        return u_nodes[..., 0].T, u_nodes[..., 1].T
    return fn


def make_vorticity_at_nodes_fn(V):
    """Build a closure that maps (u_global,) -> nodal vorticity (Np, K)."""
    k = V.order
    mesh = V.mesh
    r, s = mesh.r, mesh.s
    phi_ref = bdm_k_basis_at_points(k, r, s)        # (Np, dim, 2)
    DF, _ = _element_jacobian_matrices(mesh)        # (K, 2, 2)
    J = mesh.J[0]                                   # (K,)
    Dr, Ds = mesh.Dr, mesh.Ds
    rx, ry, sx, sy = mesh.rx, mesh.ry, mesh.sx, mesh.sy

    def vorticity(u_global):
        phi_phys = jnp.einsum(
            "kca,nla->knlc", DF, phi_ref) / J[:, None, None, None]
        u_local = bdm1_gather(V.topology, u_global)
        u_nodes = jnp.einsum("kl,knlc->knc", u_local, phi_phys)
        ux = u_nodes[..., 0].T        # (Np, K)
        uy = u_nodes[..., 1].T
        ux_r = Dr @ ux; ux_s = Ds @ ux
        uy_r = Dr @ uy; uy_s = Ds @ uy
        duydx = rx * uy_r + sx * uy_s
        duxdy = ry * ux_r + sy * ux_s
        return duydx - duxdy
    return vorticity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--alpha-min", type=float, default=0.3)
    ap.add_argument("--alpha-max", type=float, default=1.7)
    ap.add_argument("--lambda-mass", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    V, u_bc, u_bc_vec, bc_mask, is_natural_face, _is_cyl = build_static(
        args.kx, args.ky, args.order,
    )
    Re = U_MEAN * D_CYL / NU
    print(f"BDM_{args.order} cylinder smoke: Re={Re:.0f}, K={V.K}, "
          f"n_u={V.total_dofs}, n_steps={args.n_steps}, "
          f"dt={args.dt}, T_final={args.n_steps * args.dt}",
          flush=True)
    print(f"  control: alpha in [{args.alpha_min}, {args.alpha_max}], "
          f"piecewise constant, {args.n_steps} knots", flush=True)

    step_fn = make_controlled_step(
        V, u_bc, u_bc_vec, bc_mask, is_natural_face,
        nu=NU, tau_scale=args.tau_scale, n_newton=args.n_newton,
    )
    vort_fn = make_vorticity_at_nodes_fn(V)

    # Downstream window mask (precomputed once on host).
    x_np = np.asarray(V.mesh.x); y_np = np.asarray(V.mesh.y)
    in_window = (
        (x_np > CX + 1.5 * D_CYL) & (x_np < CX + 6.0 * D_CYL)
        & (y_np > Y_MIN + 0.02) & (y_np < Y_MAX - 0.02)
    )
    mass_node = jnp.asarray(in_window.astype(np.float64))
    # Nodal mass weights via element Jacobian (constant on affine elements).
    J_node = jnp.asarray(np.tile(V.mesh.J[0][None, :], (V.mesh.Np, 1)))
    weight = mass_node * J_node                        # (Np, K)
    weight_total = float(weight.sum())
    print(f"  downstream window: {int(in_window.sum())} nodes, "
          f"weight {weight_total:.3f}", flush=True)

    def alphas_from_theta(theta):
        return args.alpha_min + (args.alpha_max - args.alpha_min) * jax.nn.sigmoid(theta)

    def trajectory(alphas):
        u_n = jnp.zeros(V.total_dofs)
        u_nm1 = jnp.zeros(V.total_dofs)
        p_last = jnp.zeros((V.K, args.order * (args.order + 1) // 2))
        for s in range(args.n_steps):
            order_eff = 1 if s == 0 else 2
            u_new, p_new = step_fn(u_n, u_nm1, args.dt, alphas[s], order_eff)
            u_nm1, u_n, p_last = u_n, u_new, p_new
        return u_n, p_last

    def trajectory_full(alphas):
        """Same forward map but returns all (u_step, p_step) states.

        Used post-Adam to dump the full time history for movie rendering.
        Not jit-compiled to keep the JIT cache for the AD path clean.
        """
        u_n = jnp.zeros(V.total_dofs)
        u_nm1 = jnp.zeros(V.total_dofs)
        u_steps = []
        p_steps = []
        for s in range(args.n_steps):
            order_eff = 1 if s == 0 else 2
            u_new, p_new = step_fn(u_n, u_nm1, args.dt, alphas[s], order_eff)
            u_steps.append(u_new)
            p_steps.append(p_new)
            u_nm1, u_n = u_n, u_new
        return u_steps, p_steps

    def loss_fn(theta):
        alphas = alphas_from_theta(theta)
        u_final, _ = trajectory(alphas)
        omega = vort_fn(u_final)                       # (Np, K)
        L_vort = jnp.sum(weight * omega**2)
        L_mass = (jnp.mean(alphas) - 1.0)**2
        return L_vort + args.lambda_mass * L_mass

    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)
    traj_jit = jax.jit(trajectory)
    vort_jit = jax.jit(vort_fn)

    theta = jnp.zeros(args.n_steps)
    alphas0 = alphas_from_theta(theta)
    print(f"  init alphas: {np.asarray(alphas0)}", flush=True)

    t0 = time.time()
    L0 = float(loss_jit(theta))
    print(f"  loss = {L0:.4e}  (jit compile: {time.time() - t0:.1f}s)",
          flush=True)

    # Baseline: alpha = 1 throughout (theta = sigmoid^{-1}((1 - amin)/(amax - amin))).
    a1_sigm = (1.0 - args.alpha_min) / (args.alpha_max - args.alpha_min)
    theta1 = jnp.log(a1_sigm / (1.0 - a1_sigm)) * jnp.ones_like(theta)
    L_baseline = float(loss_jit(theta1))
    print(f"  baseline (alpha=1 throughout): loss = {L_baseline:.4e}",
          flush=True)

    m = jnp.zeros_like(theta); v = jnp.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    history = [dict(iter=0, alphas=alphas0.tolist(), loss=L0, gnorm=0.0)]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**it)
        v_hat = v / (1 - beta2**it)
        theta = theta - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        L = float(loss_jit(theta))
        alphas = alphas_from_theta(theta)
        gnorm = float(jnp.linalg.norm(g))
        history.append(dict(iter=int(it), alphas=alphas.tolist(),
                            loss=L, gnorm=gnorm))
        print(f"  iter {it:3d}: loss={L:.4e}  |g|={gnorm:.2e}  "
              f"alpha=[{', '.join(f'{a:.2f}' for a in alphas)}]",
              flush=True)
    print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)

    # Final and baseline wake fields for the report figure.
    alphas_final = alphas_from_theta(theta)
    u_opt, _ = traj_jit(alphas_final)
    u_base, _ = traj_jit(alphas_from_theta(theta1))
    omega_opt = np.asarray(vort_jit(u_opt))
    omega_base = np.asarray(vort_jit(u_base))

    # Full per-BDF-step trajectories for the wake-control movie. Both
    # baseline and optimised, evaluated at the Lagrangian mesh nodes.
    print("collecting full trajectories for mp4 ...", flush=True)
    velocity_at_nodes_jit = jax.jit(_make_velocity_at_nodes_fn(V))
    u_steps_opt, _ = trajectory_full(alphas_final)
    u_steps_base, _ = trajectory_full(alphas_from_theta(theta1))
    ux_opt_traj = []; uy_opt_traj = []
    ux_base_traj = []; uy_base_traj = []
    w_opt_traj = []; w_base_traj = []
    for u_s in u_steps_opt:
        ux_n, uy_n = velocity_at_nodes_jit(u_s)
        ux_opt_traj.append(np.asarray(ux_n))
        uy_opt_traj.append(np.asarray(uy_n))
        w_opt_traj.append(np.asarray(vort_jit(u_s)))
    for u_s in u_steps_base:
        ux_n, uy_n = velocity_at_nodes_jit(u_s)
        ux_base_traj.append(np.asarray(ux_n))
        uy_base_traj.append(np.asarray(uy_n))
        w_base_traj.append(np.asarray(vort_jit(u_s)))
    ux_opt_traj = np.stack(ux_opt_traj)        # (n_steps, Np, K)
    uy_opt_traj = np.stack(uy_opt_traj)
    ux_base_traj = np.stack(ux_base_traj)
    uy_base_traj = np.stack(uy_base_traj)
    w_opt_traj = np.stack(w_opt_traj)
    w_base_traj = np.stack(w_base_traj)
    px = np.asarray(V.mesh.x); py = np.asarray(V.mesh.y)
    t_axis = (np.arange(args.n_steps) + 1) * args.dt

    stem = f"inflow_control_cylinder_k{args.order}"
    out_npz = args.out_dir / f"{stem}.npz"
    np.savez_compressed(
        out_npz,
        # Control + diagnostics
        alphas_optimised=np.asarray(alphas_final),
        alphas_baseline=np.asarray(alphas_from_theta(theta1)),
        times=t_axis,
        loss_history=np.array([r["loss"] for r in history]),
        # Mesh node coordinates (Np, K)
        mesh_x=px, mesh_y=py,
        # Per-step trajectories (n_steps, Np, K)
        ux_optimised=ux_opt_traj, uy_optimised=uy_opt_traj,
        ux_baseline=ux_base_traj, uy_baseline=uy_base_traj,
        omega_optimised=w_opt_traj, omega_baseline=w_base_traj,
        # Geometry
        CX=CX, CY=CY, RADIUS=RADIUS,
        X_MIN=X_MIN, X_MAX=X_MAX, Y_MIN=Y_MIN, Y_MAX=Y_MAX,
        D_CYL=D_CYL, U_MAX=U_MAX, U_MEAN=U_MEAN, NU=NU,
    )
    print(f"  wrote {out_npz} ({out_npz.stat().st_size/1024:.1f} KB)",
          flush=True)

    out_json = args.out_dir / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky, K=int(V.K),
            re=Re, dt=args.dt, n_steps=args.n_steps,
            n_newton=args.n_newton, tau_scale=args.tau_scale,
            alpha_min=args.alpha_min, alpha_max=args.alpha_max,
            lambda_mass=args.lambda_mass, lr=args.lr,
            n_train=args.n_train,
            loss_baseline=L_baseline,
            history=history,
        ), f, indent=2)
    print(f"  wrote {out_json}", flush=True)

    # ---- mp4: baseline vs optimised, side by side over t ----
    print("rendering wake-control mp4 ...", flush=True)
    from matplotlib.animation import FuncAnimation
    px_flat = px.T.reshape(-1)
    py_flat = py.T.reshape(-1)
    theta_arc = np.linspace(0, 2 * np.pi, 80)
    cyl_x = CX + RADIUS * np.cos(theta_arc)
    cyl_y = CY + RADIUS * np.sin(theta_arc)
    w_max = max(np.percentile(np.abs(w_base_traj), 98),
                np.percentile(np.abs(w_opt_traj), 98))
    levels = np.linspace(-w_max, w_max, 21)
    x0, x1 = CX + 1.5 * D_CYL, CX + 6.0 * D_CYL
    y0, y1 = Y_MIN + 0.02, Y_MAX - 0.02
    fig2, axes2 = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, constrained_layout=True,
    )
    def draw(i):
        for ax in axes2:
            ax.clear()
        wb = w_base_traj[i].T.reshape(-1)
        wo = w_opt_traj[i].T.reshape(-1)
        axes2[0].tricontourf(px_flat, py_flat, wb, levels=levels,
                             cmap="RdBu_r", extend="both")
        axes2[0].fill(cyl_x, cyl_y, color="black")
        axes2[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                      "k-", lw=1, alpha=0.5)
        axes2[0].set_aspect("equal"); axes2[0].set_xlim(X_MIN, X_MAX)
        axes2[0].set_title(
            f"Baseline (alpha=1)  t={t_axis[i]:.2f}  "
            f"alpha_t={float(alphas_from_theta(theta1)[i]):.2f}")
        axes2[1].tricontourf(px_flat, py_flat, wo, levels=levels,
                             cmap="RdBu_r", extend="both")
        axes2[1].fill(cyl_x, cyl_y, color="black")
        axes2[1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                      "k-", lw=1, alpha=0.5)
        axes2[1].set_aspect("equal"); axes2[1].set_xlim(X_MIN, X_MAX)
        axes2[1].set_title(
            f"Optimised  t={t_axis[i]:.2f}  "
            f"alpha_t={float(alphas_final[i]):.2f}")
        return []
    anim = FuncAnimation(fig2, draw, frames=args.n_steps,
                         interval=400, blit=False)
    out_mp4 = args.out_dir / f"{stem}.mp4"
    anim.save(str(out_mp4), writer="ffmpeg", dpi=110)
    plt.close(fig2)
    print(f"  wrote {out_mp4}", flush=True)

    # Plot: alpha(t) trajectory, loss curve, vorticity fields.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    t_axis = np.arange(1, args.n_steps + 1) * args.dt
    axes[0, 0].step(t_axis, alphas_final, "b-", where="post",
                    lw=2, label="optimised")
    axes[0, 0].step(t_axis, np.ones_like(t_axis), "k--", where="post",
                    lw=1, label="baseline (alpha=1)")
    axes[0, 0].axhline(args.alpha_min, color="gray", ls=":", lw=0.7)
    axes[0, 0].axhline(args.alpha_max, color="gray", ls=":", lw=0.7)
    axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel(r"$\alpha(t)$")
    axes[0, 0].set_title("Inflow multiplier")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    iters = [r["iter"] for r in history]
    losses = [r["loss"] for r in history]
    axes[0, 1].semilogy(iters, losses, "k-")
    axes[0, 1].axhline(L_baseline, color="r", ls="--", lw=1,
                       label="baseline")
    axes[0, 1].set_xlabel("Adam iter")
    axes[0, 1].set_ylabel("loss")
    axes[0, 1].set_title("Adam loss")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3, which="both")

    # Vorticity panels.
    px = np.asarray(V.mesh.x).T.reshape(-1)
    py = np.asarray(V.mesh.y).T.reshape(-1)
    theta_arc = np.linspace(0, 2 * np.pi, 80)
    cyl_x = CX + RADIUS * np.cos(theta_arc)
    cyl_y = CY + RADIUS * np.sin(theta_arc)
    vmax = max(np.percentile(np.abs(omega_base), 98),
               np.percentile(np.abs(omega_opt), 98))
    levels = np.linspace(-vmax, vmax, 21)
    for ax, omega, title in (
        (axes[1, 0], omega_base, "Baseline wake (alpha=1)"),
        (axes[1, 1], omega_opt, "Optimised wake"),
    ):
        pz = omega.T.reshape(-1)
        ax.tricontourf(px, py, pz, levels=levels, cmap="RdBu_r",
                       extend="both")
        ax.fill(cyl_x, cyl_y, color="black")
        # Outline the downstream window.
        x0, x1 = CX + 1.5 * D_CYL, CX + 6.0 * D_CYL
        y0, y1 = Y_MIN + 0.02, Y_MAX - 0.02
        ax.plot([x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
                "k-", lw=1, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlim(X_MIN, X_MAX)

    fig.suptitle(
        f"Inflow control: BDM_{args.order} cylinder, Re={Re:.0f}, "
        f"T_final={args.n_steps * args.dt:.2f}"
    )
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
