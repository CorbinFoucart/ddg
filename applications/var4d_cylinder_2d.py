"""4D-VAR-style initial-condition recovery via jax.grad.

Given sparse-in-time velocity observations along a downstream probe
rake, recover the BDM_k initial-condition vector ``u_0`` (dim ~3000
on the smoke geometry) that, when evolved by the forward BDF2
Newton solver, best matches the observations. The recovered ``u_0``
will differ from the ground-truth one because the inverse problem is
severely underdetermined (n_obs ~ probes * time_steps * 2 << n_u),
but Adam should still drive the trace mismatch monotonically toward
zero --- this is the operative test of the gradient path on a
vector unknown.

Setup
-----
* BDM_2 cylinder smoke (Re=100, K=432, n_u ~ 3354). Same forward as
  ``scripts/inverse_nu_cylinder_2d.py``.
* Ground truth: ``u_0_truth`` is the state of the system after
  ``n_warmup`` BDF2 steps from rest at the canonical inflow ramp ---
  a non-trivial, partially-developed flow.
* Observations: forward solve ``n_obs`` more BDF2 steps from
  ``u_0_truth`` and record ``(u_x, u_y)`` at six probe points along
  ``y = CY`` at every BDF step. That is ``n_obs * 6 * 2`` scalar
  observations; the unknown ``u_0`` has dimension ``n_u``, so the
  inverse problem is heavily underdetermined.
* Loss = L2 trace mismatch + ``lambda_bg * ||u_0 - u_0_init||^2``
  background regularisation, with ``u_0_init = 0`` (rest).

Cost
----
Forward + reverse pass per Adam iter; cost is independent of the
parameter count (reverse mode), so a 3354-dim u_0 inverse is the
same wall as the scalar ν-recovery: ~4 min/iter steady on a single
Apple-silicon CPU thread.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diff_cylinder_dt_tau import (
    build_static, X_MIN, Y_MIN, Y_MAX, CX, CY, RADIUS, D_CYL,
    U_MAX, U_MEAN, NU,
)
from inverse_nu_cylinder_2d import (
    make_step_nu_param, make_probe_fn,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=2)
    ap.add_argument("--n-warmup", type=int, default=4,
                    help="BDF2 steps to develop u_0_truth from rest")
    ap.add_argument("--n-obs", type=int, default=4,
                    help="BDF2 steps over which probes are observed")
    ap.add_argument("--probe-nx", type=int, default=16,
                    help="probe grid columns (x)")
    ap.add_argument("--probe-ny", type=int, default=8,
                    help="probe grid rows (y)")
    ap.add_argument("--probe-x0", type=float, default=CX - 0.05)
    ap.add_argument("--probe-x1", type=float, default=0.92)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--lambda-bg", type=float, default=1e-3,
                    help="L2 regularisation toward u_0_init")
    ap.add_argument("--lr", type=float, default=5e-3,
                    help="initial learning rate (also the floor during "
                         "the from-rest warmup)")
    ap.add_argument("--lr-max", type=float, default=None,
                    help="peak learning rate; lr ramps lr->lr_max over "
                         "--lr-warmup iters then holds. Default: =lr "
                         "(no schedule)")
    ap.add_argument("--lr-warmup", type=int, default=0,
                    help="linear-warmup length in iters; small steps near "
                         "rest (off the divergence cliff), large steps once "
                         "u_0 is in the stable basin")
    ap.add_argument("--remat", action="store_true",
                    help="gradient-checkpoint each trajectory step "
                         "(jax.checkpoint) so reverse-mode memory is O(1) in "
                         "n_obs rather than O(n_obs); needed for long "
                         "observation windows that otherwise OOM. Costs ~2x "
                         "forward work in the backward pass.")
    ap.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam",
                    help="adam: SGD-style (needs lr tuning, slow on this "
                         "smooth deterministic inverse). lbfgs: scipy "
                         "L-BFGS-B driven by the JAX gradient, the standard "
                         "4D-VAR optimizer (curvature-aware, no lr).")
    ap.add_argument("--g-clip", type=float, default=1.0,
                    help="per-component gradient clip; protects "
                         "against the Newton-divergence cliff that "
                         "non-physical u_0 perturbations trigger")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    V, u_bc, u_bc_vec, bc_mask, is_natural_face, _ = build_static(
        args.kx, args.ky, args.order,
    )
    n_u = V.total_dofs
    print(f"BDM_{args.order} cylinder smoke 4D-VAR: K={V.K}, "
          f"n_u={n_u}", flush=True)
    print(f"  n_warmup = {args.n_warmup}  (develop u_0_truth)",
          flush=True)
    print(f"  n_obs    = {args.n_obs}  (observation horizon)",
          flush=True)
    print(f"  lr       = {args.lr}  lambda_bg = {args.lambda_bg}",
          flush=True)

    step_fn = make_step_nu_param(
        V, u_bc, u_bc_vec, bc_mask, is_natural_face,
        tau_scale=args.tau_scale, n_newton=args.n_newton,
    )
    step_jit = jax.jit(step_fn, static_argnames=("order_eff",))

    # observation probe grid over the wake region; drop points inside the
    # cylinder. A dense rake makes the inverse problem well-posed (cf. the
    # 1D twin experiment), unlike the 6-point centreline rake.
    _pxg = np.linspace(args.probe_x0, args.probe_x1, args.probe_nx)
    _pyg = np.linspace(Y_MIN + 0.03, Y_MAX - 0.03, args.probe_ny)
    _PX, _PY = np.meshgrid(_pxg, _pyg)
    _pts = np.column_stack([_PX.ravel(), _PY.ravel()])
    _inside = (_pts[:, 0] - CX) ** 2 + (_pts[:, 1] - CY) ** 2 < (RADIUS * 1.15) ** 2
    probe_xy = np.ascontiguousarray(_pts[~_inside])
    n_probe = int(probe_xy.shape[0])
    print(f"  probe grid {args.probe_nx}x{args.probe_ny} -> {n_probe} probes "
          f"(cylinder excluded)", flush=True)
    probe_fn = make_probe_fn(V, probe_xy)

    # --- develop u_0_truth from rest ---
    print("developing u_0_truth from rest ...", flush=True)
    t0 = time.time()
    u_n = jnp.zeros(n_u); u_nm1 = jnp.zeros(n_u)
    for s in range(args.n_warmup):
        order_eff = 1 if s == 0 else 2
        u_new, _ = step_jit(u_n, u_nm1, args.dt, NU, order_eff)
        u_nm1, u_n = u_n, u_new
    u_0_truth = u_n
    u_0_truth.block_until_ready()
    print(f"  done in {time.time() - t0:.1f}s   "
          f"||u_0_truth|| = {float(jnp.linalg.norm(u_0_truth)):.4e}",
          flush=True)

    # --- generate y_obs from u_0_truth ---
    # gradient-checkpoint the step so the reverse pass rematerialises each
    # forward step instead of storing all n_obs intermediates (bounds memory).
    step_for_traj = (jax.checkpoint(step_fn, static_argnums=(4,))
                     if args.remat else step_fn)

    def trajectory(u_init):
        """Forward map: BDF2 from u_init for n_obs steps; return probe
        traces at each step."""
        u_n = u_init
        u_nm1 = u_init   # BDF1 bootstrap (one-step history reuse)
        traces = []
        for s in range(args.n_obs):
            order_eff = 1 if s == 0 else 2
            u_new, _ = step_for_traj(u_n, u_nm1, args.dt, NU, order_eff)
            traces.append(probe_fn(u_new))
            u_nm1, u_n = u_n, u_new
        return jnp.stack(traces)        # (n_obs, n_probe, 2)

    print("generating y_obs ...", flush=True)
    traj_jit = jax.jit(trajectory)
    y_obs = traj_jit(u_0_truth)
    y_obs.block_until_ready()

    # --- 4D-VAR loss ---
    u_0_init = jnp.zeros(n_u)            # rest is the background guess

    def loss_fn(u_init):
        y_pred = trajectory(u_init)
        L_obs = jnp.sum((y_pred - y_obs) ** 2)
        L_bg = args.lambda_bg * jnp.sum((u_init - u_0_init) ** 2)
        return L_obs + L_bg

    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    u = jnp.zeros_like(u_0_truth)
    L0 = float(loss_jit(u))
    L_truth = float(loss_jit(u_0_truth))
    print(f"  initial loss = {L0:.4e}", flush=True)
    print(f"  loss at u_0_truth = {L_truth:.4e}  (baseline)",
          flush=True)

    m_t = jnp.zeros_like(u); v_t = jnp.zeros_like(u)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_loss = L0; best_u = u; best_iter = 0
    history = [dict(iter=0, loss=L0, gnorm=0.0,
                    u_err=float(jnp.linalg.norm(u - u_0_truth)))]

    stem = f"var4d_cylinder_k{args.order}"

    def save_outputs(u_best, hist, tag="progress"):
        """Atomic dump-as-you-go: write to a temp name then os.replace so a
        reader (or a crash) never sees a half-written file."""
        out_npz = args.out_dir / f"{stem}.npz"
        tmp_npz = args.out_dir / f"{stem}.tmp.npz"
        np.savez_compressed(
            tmp_npz,
            n_warmup=args.n_warmup, n_obs=args.n_obs, dt=args.dt,
            u_0_truth=np.asarray(u_0_truth),
            u_0_recovered=np.asarray(u_best),
            u_0_init=np.zeros(n_u),
            y_obs=np.asarray(y_obs),
            loss_hist=np.array([r["loss"] for r in hist]),
            u_err_hist=np.array([r["u_err"] for r in hist]),
        )
        os.replace(tmp_npz, out_npz)
        out_json = args.out_dir / f"{stem}.json"
        tmp_json = args.out_dir / f"{stem}.json.tmp"
        with open(tmp_json, "w") as f:
            json.dump(dict(
                order=args.order, kx=args.kx, ky=args.ky, K=int(V.K),
                n_u=int(n_u), n_warmup=args.n_warmup, n_obs=args.n_obs,
                dt=args.dt, n_newton=args.n_newton,
                tau_scale=args.tau_scale, lambda_bg=args.lambda_bg,
                lr=args.lr, n_train=args.n_train, n_probe=int(n_probe),
                loss_at_truth=L_truth, best_iter=int(best_iter),
                best_loss=float(best_loss), tag=tag, history=hist,
            ), f, indent=2)
        os.replace(tmp_json, out_json)

    lr_max = args.lr if args.lr_max is None else args.lr_max
    truth_norm = float(jnp.linalg.norm(u_0_truth))
    t0 = time.time()
    if args.optimizer == "adam":
        for it in range(1, args.n_train + 1):
            if args.lr_warmup > 0:
                frac = min(1.0, (it - 1) / float(args.lr_warmup))
                lr_t = args.lr + (lr_max - args.lr) * frac
            else:
                lr_t = args.lr
            g = grad_fn(u)
            g = jnp.clip(g, -args.g_clip, args.g_clip)
            m_t = beta1 * m_t + (1 - beta1) * g
            v_t = beta2 * v_t + (1 - beta2) * g * g
            m_hat = m_t / (1 - beta1 ** it)
            v_hat = v_t / (1 - beta2 ** it)
            u = u - lr_t * m_hat / (jnp.sqrt(v_hat) + eps)
            L = float(loss_jit(u))
            gnorm = float(jnp.linalg.norm(g))
            u_err = float(jnp.linalg.norm(u - u_0_truth))
            if L < best_loss:
                best_loss = L; best_u = u; best_iter = it
            u_norm = float(jnp.linalg.norm(u))
            rel_u_err = u_err / truth_norm
            print(f"  iter {it:3d}: loss = {L:.4e}  lr={lr_t:.1e}  "
                  f"|g| = {gnorm:.2e}  "
                  f"|u| = {u_norm:.3e}  ||u-u_truth||/||u_truth|| = "
                  f"{rel_u_err:.3f}",
                  flush=True)
            history.append(dict(iter=int(it), loss=L, gnorm=gnorm,
                                u_err=u_err))
            if it % 5 == 0 or it == 1:
                save_outputs(best_u, history, tag=f"iter{it}")
        print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)
    else:   # lbfgs: scipy L-BFGS-B on the JAX gradient (the 4D-VAR standard)
        from scipy.optimize import minimize
        st = dict(best_loss=best_loss, best_u=best_u, best_iter=best_iter,
                  it=0, bad=0)

        def fun_grad(x):
            xj = jnp.asarray(x)
            L = float(loss_jit(xj))
            g = np.asarray(grad_fn(xj), dtype=np.float64)
            if not (np.isfinite(L) and np.all(np.isfinite(g))):
                st["bad"] += 1            # off the Newton cliff: backtrack
                return 1e8, np.zeros_like(x)
            return L, g

        def cb(xk):
            st["it"] += 1
            it = st["it"]
            xj = jnp.asarray(xk)
            L = float(loss_jit(xj))
            u_err = float(jnp.linalg.norm(xj - u_0_truth))
            u_norm = float(jnp.linalg.norm(xj))
            if L < st["best_loss"]:
                st["best_loss"] = L; st["best_u"] = xj; st["best_iter"] = it
            history.append(dict(iter=int(it), loss=L, gnorm=0.0,
                                u_err=u_err))
            print(f"  lbfgs it {it:3d}: loss = {L:.4e}  |u| = {u_norm:.3e}  "
                  f"||u-u_truth||/||u_truth|| = {u_err / truth_norm:.3f}",
                  flush=True)
            if it % 5 == 0 or it == 1:
                save_outputs(st["best_u"], history, tag=f"lbfgs{it}")

        res = minimize(fun_grad, np.zeros(n_u, dtype=np.float64), jac=True,
                       method="L-BFGS-B", callback=cb,
                       options=dict(maxiter=args.n_train,
                                    maxfun=args.n_train * 5,
                                    ftol=1e-15, gtol=1e-12, maxcor=20))
        best_loss = st["best_loss"]; best_u = st["best_u"]
        best_iter = st["best_iter"]
        print(f"\ntotal L-BFGS wall: {time.time() - t0:.1f}s  "
              f"(nit={res.nit}, nfev={res.nfev}, bad_evals={st['bad']})",
              flush=True)
    print(f"\nbest loss = {best_loss:.4e} at iter {best_iter}",
          flush=True)

    save_outputs(best_u, history, tag="final")
    print(f"  wrote {args.out_dir / f'{stem}.npz'} (+ .json)", flush=True)

    iters = [r["iter"] for r in history]
    losses = [r["loss"] for r in history]
    u_errs = [r["u_err"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].semilogy(iters, losses, "k-")
    axes[0].axhline(L_truth, color="r", ls="--", lw=1, label="loss at truth")
    axes[0].set_xlabel("Adam iter")
    axes[0].set_ylabel(r"$\|y_{\rm pred} - y_{\rm obs}\|^2 + \lambda \|u_0\|^2$")
    axes[0].set_title("4D-VAR loss (probe trace + bg regularisation)")
    axes[0].legend(); axes[0].grid(alpha=0.3, which="both")
    axes[1].plot(iters, u_errs, "b-")
    axes[1].axhline(float(jnp.linalg.norm(u_0_truth)),
                    color="gray", ls=":", lw=1,
                    label=r"$\|u_0^{\rm truth}\|$")
    axes[1].set_xlabel("Adam iter")
    axes[1].set_ylabel(r"$\|u_0^{\rm rec} - u_0^{\rm truth}\|_2$")
    axes[1].set_title("Recovered-IC distance to truth")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(
        f"4D-VAR initial-condition recovery on BDM_{args.order} "
        f"cylinder smoke  (n_u = {n_u}, "
        f"obs = {args.n_obs * n_probe * 2})"
    )
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
