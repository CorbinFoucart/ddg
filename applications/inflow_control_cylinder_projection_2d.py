"""Optimal inflow control of the BDM cylinder wake -- projection solver.

Projection-powered counterpart to ``inflow_control_cylinder_2d.py`` (which
uses the monolithic coupled-Newton forward solve). Here the forward model
is the block-Schur projection step ``projection_step_bdm_k_2d``: the
Helmholtz and pressure-Schur LU factors are built ONCE (they do not depend
on the control), so each time step is two cheap LU back-solves, and the
whole trajectory is cheap enough to differentiate end-to-end and to render
at high resolution. ``jax.grad`` backprops through ``jax.scipy.linalg.lu_solve``
(differentiable); the control ``alpha(t)`` enters only through the inflow
Dirichlet trace ``alpha * u_bc``.

Because the projection treats convection explicitly it carries a CFL limit
(unlike the implicit monolithic), so ``dt`` is small and ``n_steps`` large;
the trajectory is a ``lax.scan`` with per-step ``jax.checkpoint`` to bound
reverse-mode memory. The control is piecewise-constant over ``n_control``
knots (decoupled from ``n_steps``) and sigmoid-bounded.

  L(theta) = integral_{Omega_down} omega(x, T)^2 dx
           + lambda_mass * (mean(alpha) - 1)^2

Outputs ``output/inflow_control_cyl_proj_k{order}.{npz,json,png,mp4}``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diff_cylinder_dt_tau import (
    build_static, X_MIN, X_MAX, Y_MIN, Y_MAX, CX, CY, RADIUS, D_CYL,
    U_MAX, U_MEAN, NU,
)
from inflow_control_cylinder_2d import (
    make_vorticity_at_nodes_fn, _make_velocity_at_nodes_fn,
)
from ddg.dg2d.projection_ns_bdm import (
    make_projection_solvers, projection_step_bdm_k_2d,
)


def make_proj_step(V, u_bc, u_bc_vec, bc_mask, is_natural_face, *,
                   nu, tau_scale, dt):
    """Build BDF1/BDF2 projection solvers once; return a controlled step.

    step(u_n, u_nm1, alpha, order_eff) -> (u_new, p_new). The inflow
    multiplier ``alpha`` scales the Dirichlet trace only; the LU factors
    are alpha-independent so grad flows purely through the back-solves.
    """
    ps1, hinv1 = make_projection_solvers(
        V, dt=dt, nu=nu, order=1, tau_scale=tau_scale,
        bc_mask=bc_mask, is_natural_face=is_natural_face)
    ps2, hinv2 = make_projection_solvers(
        V, dt=dt, nu=nu, order=2, tau_scale=tau_scale,
        bc_mask=bc_mask, is_natural_face=is_natural_face)

    def step(u_n, u_nm1, alpha, order_eff):
        u_bc_t = alpha * u_bc
        u_bc_vec_t = alpha * u_bc_vec
        if order_eff == 1:
            ps, hinv, u_hist = ps1, hinv1, [u_n]
        else:
            ps, hinv, u_hist = ps2, hinv2, [u_n, u_nm1]
        return projection_step_bdm_k_2d(
            V, u_hist, dt=dt, nu=nu, order=order_eff,
            pressure_solver=ps, h_inv=hinv,
            u_bc_global=u_bc_t, u_bc_vec=u_bc_vec_t,
            bc_mask=bc_mask, is_natural_face=is_natural_face,
            tau_scale=tau_scale)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-radial", type=int, default=2)
    ap.add_argument("--r-outer", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--n-control", type=int, default=10)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--alpha-min", type=float, default=0.3)
    ap.add_argument("--alpha-max", type=float, default=1.7)
    ap.add_argument("--lambda-mass", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-train", type=int, default=20,
                    help="max Adam iters (early-stop ends sooner on flatline)")
    ap.add_argument("--tol", type=float, default=1.5e-3,
                    help="relative-loss-change flatline threshold")
    ap.add_argument("--patience", type=int, default=8,
                    help="consecutive sub-tol iters before declaring flatline")
    ap.add_argument("--remat", action="store_true",
                    help="jax.checkpoint each scanned step (bounds backprop "
                         "memory for long trajectories)")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--field-every", type=int, default=5,
                    help="dump baseline+controlled vorticity fields every N "
                         "Adam iters (state-as-it-goes for mid-run contours)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    V, u_bc, u_bc_vec, bc_mask, is_natural_face, _is_cyl = build_static(
        args.kx, args.ky, args.order,
        n_radial=args.n_radial, r_outer=args.r_outer)
    Re = U_MEAN * D_CYL / NU
    T_final = args.n_steps * args.dt
    print(f"[projection] BDM_{args.order} cylinder: Re={Re:.0f}, K={V.K}, "
          f"n_u={V.total_dofs}, n_steps={args.n_steps}, dt={args.dt}, "
          f"T_final={T_final:.3f}, n_control={args.n_control}", flush=True)

    step_fn = make_proj_step(
        V, u_bc, u_bc_vec, bc_mask, is_natural_face,
        nu=NU, tau_scale=args.tau_scale, dt=args.dt)
    vort_fn = make_vorticity_at_nodes_fn(V)

    # control-knot -> step map (piecewise constant over n_control intervals)
    knot_of_step = np.minimum(
        (np.arange(args.n_steps) * args.n_control) // args.n_steps,
        args.n_control - 1).astype(np.int32)
    knot_of_step = jnp.asarray(knot_of_step)

    # downstream vorticity window
    x_np = np.asarray(V.mesh.x); y_np = np.asarray(V.mesh.y)
    in_window = ((x_np > CX + 1.5 * D_CYL) & (x_np < CX + 6.0 * D_CYL)
                 & (y_np > Y_MIN + 0.02) & (y_np < Y_MAX - 0.02))
    J_node = jnp.asarray(np.tile(V.mesh.J[0][None, :], (V.mesh.Np, 1)))
    weight = jnp.asarray(in_window.astype(np.float64)) * J_node
    print(f"  downstream window: {int(in_window.sum())} nodes", flush=True)

    n_u = V.total_dofs
    dim_p = args.order * (args.order + 1) // 2

    def alphas_from_theta(theta):                      # theta: (n_control,)
        a = args.alpha_min + (args.alpha_max - args.alpha_min) * jax.nn.sigmoid(theta)
        return a[knot_of_step]                          # (n_steps,)

    bdf2_body = step_fn
    if args.remat:
        bdf2_body = jax.checkpoint(
            lambda u_n, u_nm1, alpha: step_fn(u_n, u_nm1, alpha, 2))

    def trajectory(alphas_step):
        # BDF1 first step (python), then lax.scan BDF2 over the rest.
        u0, _ = step_fn(jnp.zeros(n_u), jnp.zeros(n_u), alphas_step[0], 1)
        def scan_body(carry, alpha):
            u_n, u_nm1 = carry
            if args.remat:
                u_new, _ = bdf2_body(u_n, u_nm1, alpha)
            else:
                u_new, _ = step_fn(u_n, u_nm1, alpha, 2)
            return (u_new, u_n), u_new
        (u_last, _), _ = jax.lax.scan(
            scan_body, (u0, jnp.zeros(n_u)), alphas_step[1:])
        return u_last

    def loss_fn(theta):
        alphas = alphas_from_theta(theta)
        u_final = trajectory(alphas)
        omega = vort_fn(u_final)
        L_vort = jnp.sum(weight * omega**2)
        L_mass = (jnp.mean(alphas) - 1.0)**2
        return L_vort + args.lambda_mass * L_mass

    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn))

    theta = jnp.zeros(args.n_control)
    t0 = time.time()
    L0 = float(loss_jit(theta))
    print(f"  loss(init) = {L0:.4e}  (compile {time.time()-t0:.1f}s)", flush=True)
    if not np.isfinite(L0):
        print("  -> non-finite loss (CFL/blowup?); reduce dt", flush=True)
        return
    t0 = time.time()
    g0 = grad_jit(theta)
    print(f"  |grad(init)| = {float(jnp.linalg.norm(g0)):.4e}  "
          f"(grad compile {time.time()-t0:.1f}s)", flush=True)

    # baseline alpha=1
    a1 = (1.0 - args.alpha_min) / (args.alpha_max - args.alpha_min)
    theta1 = jnp.log(a1 / (1.0 - a1)) * jnp.ones_like(theta)
    L_base = float(loss_jit(theta1))
    print(f"  loss(baseline alpha=1) = {L_base:.4e}", flush=True)

    stem = f"inflow_control_cyl_proj_k{args.order}"
    def save_json(theta_now, history, complete):
        # Persist loss history + current control EVERY iter (atomic) so an
        # interrupted-but-successful run is fully recoverable. The loss
        # history drives the convergence plot; alpha_knots (theta) lets the
        # controlled field be reconstructed with a single forward solve.
        ak = np.asarray(args.alpha_min + (args.alpha_max - args.alpha_min)
                        * jax.nn.sigmoid(theta_now))
        payload = dict(solver="projection", order=args.order, kx=args.kx,
                       ky=args.ky, K=int(V.K), n_u=int(n_u), re=Re, dt=args.dt,
                       n_steps=args.n_steps, n_control=args.n_control,
                       tau_scale=args.tau_scale, alpha_min=args.alpha_min,
                       alpha_max=args.alpha_max, lambda_mass=args.lambda_mass,
                       lr=args.lr, n_train=args.n_train, loss_baseline=L_base,
                       loss_init=L0, loss_final=history[-1]["loss"],
                       complete=complete, alpha_knots=ak.tolist(),
                       history=history)
        tmp = args.out_dir / f"{stem}.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(args.out_dir / f"{stem}.json")     # atomic
        return ak

    # --- incremental STATE dump ---------------------------------------
    # Write the baseline + current-controlled final-time vorticity FIELDS
    # every --field-every iters (atomic). This lets the wake contours be
    # inspected MID-RUN (not only at the end) and makes an interrupted run
    # recoverable, so we never wait on a run whose intermediates are bad.
    final_vort = jax.jit(lambda th: vort_fn(trajectory(alphas_from_theta(th))))
    px_n = np.asarray(V.mesh.x); py_n = np.asarray(V.mesh.y)
    w_base_final = np.asarray(final_vort(theta1))

    def dump_fields(theta_now, it):
        w_opt = np.asarray(final_vort(theta_now))
        ak = np.asarray(args.alpha_min + (args.alpha_max - args.alpha_min)
                        * jax.nn.sigmoid(theta_now))
        tmp = args.out_dir / f"_{stem}_fields_tmp.npz"
        np.savez_compressed(
            tmp, mesh_x=px_n, mesh_y=py_n,
            omega_baseline=w_base_final, omega_optimised=w_opt,
            alpha_knots=ak, iter=it, t_final=args.n_steps * args.dt,
            CX=CX, CY=CY, RADIUS=RADIUS, X_MIN=X_MIN, X_MAX=X_MAX,
            Y_MIN=Y_MIN, Y_MAX=Y_MAX, D_CYL=D_CYL)
        tmp.replace(args.out_dir / f"{stem}_fields.npz")    # atomic

    m = jnp.zeros_like(theta); v = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = [dict(iter=0, loss=L0, gnorm=float(jnp.linalg.norm(g0)))]
    save_json(theta, history, complete=False)
    dump_fields(theta, 0)
    t0 = time.time()
    L_prev = L0
    n_flat = 0
    for it in range(1, args.n_train + 1):
        g = grad_jit(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        theta = theta - args.lr * (m / (1 - b1**it)) / (jnp.sqrt(v / (1 - b2**it)) + eps)
        L = float(loss_jit(theta))
        gn = float(jnp.linalg.norm(g))
        rel = abs(L_prev - L) / max(abs(L_prev), 1e-30)
        history.append(dict(iter=it, loss=L, gnorm=gn, rel_change=rel))
        print(f"  iter {it:3d}: loss={L:.4e}  |g|={gn:.2e}  rel_dL={rel:.2e}",
              flush=True)
        save_json(theta, history, complete=False)        # save AS IT GOES
        if it % args.field_every == 0:                    # state AS IT GOES
            dump_fields(theta, it)
        n_flat = n_flat + 1 if rel < args.tol else 0
        if n_flat >= args.patience:
            print(f"  flatlined: rel_dL < {args.tol:.1e} for {args.patience} "
                  f"iters -> stop at iter {it}", flush=True)
            break
        L_prev = L
    print(f"  Adam wall {time.time()-t0:.1f}s", flush=True)

    alpha_knots = save_json(theta, history, complete=True)
    print(f"  wrote {args.out_dir / (stem + '.json')}  "
          f"(loss {L0:.3e} -> {history[-1]['loss']:.3e}, "
          f"baseline {L_base:.3e})", flush=True)

    # ---- forward-only trajectories for the figure (no backprop) ----
    print("collecting baseline + optimised trajectories for the figure ...",
          flush=True)
    vel_fn = jax.jit(_make_velocity_at_nodes_fn(V))
    vort_jit = jax.jit(vort_fn)

    def collect(theta_):
        alphas = alphas_from_theta(theta_)
        u_n = jnp.zeros(n_u); u_nm1 = jnp.zeros(n_u)
        W = []
        for s in range(args.n_steps):
            u_new, _ = step_fn(u_n, u_nm1, alphas[s], 1 if s == 0 else 2)
            u_nm1, u_n = u_n, u_new
            W.append(np.asarray(vort_jit(u_n)))
        return np.stack(W)               # (n_steps, Np, K)

    w_base = collect(theta1)
    w_opt = collect(theta)
    px = np.asarray(V.mesh.x); py = np.asarray(V.mesh.y)
    np.savez_compressed(
        args.out_dir / f"{stem}.npz",
        mesh_x=px, mesh_y=py, omega_baseline=w_base, omega_optimised=w_opt,
        alpha_knots=alpha_knots, times=(np.arange(args.n_steps) + 1) * args.dt,
        CX=CX, CY=CY, RADIUS=RADIUS, X_MIN=X_MIN, X_MAX=X_MAX,
        Y_MIN=Y_MIN, Y_MAX=Y_MAX, D_CYL=D_CYL)

    # ---- still figure: baseline vs optimised vorticity at final time ----
    pxf = px.T.reshape(-1); pyf = py.T.reshape(-1)
    arc = np.linspace(0, 2 * np.pi, 80)
    cyl_x = CX + RADIUS * np.cos(arc); cyl_y = CY + RADIUS * np.sin(arc)
    vmax = max(np.percentile(np.abs(w_base[-1]), 98),
               np.percentile(np.abs(w_opt[-1]), 98), 1e-6)
    levels = np.linspace(-vmax, vmax, 25)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True,
                             sharex=True)
    for ax, w, ttl in ((axes[0], w_base[-1], "baseline ($\\alpha=1$)"),
                       (axes[1], w_opt[-1], "optimised inflow control")):
        ax.tricontourf(pxf, pyf, w.T.reshape(-1), levels=levels,
                       cmap="RdBu_r", extend="both")
        ax.fill(cyl_x, cyl_y, color="black")
        ax.set_aspect("equal"); ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX); ax.set_title(ttl, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Projection BDM$_{args.order}$ inflow control, Re={Re:.0f}, "
                 f"$t={args.n_steps*args.dt:.2f}$  "
                 f"(loss {L_base:.2e} $\\to$ {history[-1]['loss']:.2e})")
    fig.savefig(args.out_dir / f"{stem}.png", dpi=140)
    print(f"  wrote {args.out_dir / (stem + '.png')}", flush=True)


if __name__ == "__main__":
    main()
