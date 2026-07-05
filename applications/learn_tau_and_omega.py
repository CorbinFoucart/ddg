"""Joint differentiable training of SIPG ``tau_scale`` + per-level
smoother damping ``omegas``. Closes the loop opened by
``scripts/learn_smoother_omega.py`` (omegas only, fixed tau) and
``scripts/sweep_tau_scale.py`` (tau, auto omegas): instead of two
separate optimisations, take ``jax.grad`` through the *full* hp-MG
hierarchy build with respect to both at once.

The library's ``make_vertex_patch_smoother`` calls ``np.linalg.inv``
on the patch submatrices, which breaks differentiability in
``tau_scale`` (the operator -- and therefore the patch matrices --
depend on tau). This script rebuilds the patch smoother inline using
``jnp.linalg.inv`` so the entire hierarchy (operator + Galerkin coarse
+ patch factorisations + coarse LU) is one JAX-traceable expression,
and ``jax.grad`` of the V-cycle log-rate produces gradients with
respect to ``tau_scale``.
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

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.bdm import BDM, bdm_k_viscous_apply
from ddg.dg2d.bdm_multigrid import make_p_prolong, vertex_patches


NU = 1.0 / 100.0


def build_static(mesh, order):
    """tau-independent pieces (built ONCE, outside any JIT trace).
    ``BDM.from_mesh`` and ``vertex_patches`` use numpy on the mesh
    arrays, so they can't live inside a jit-traced loss."""
    Vs = [BDM.from_mesh(mesh, order=k) for k in range(order, 0, -1)]
    prolongs = [make_p_prolong(Vs[li + 1], Vs[li])
                for li in range(len(Vs) - 1)]
    restricts = [jax.linear_transpose(prolongs[li],
                                       jnp.zeros(Vs[li + 1].total_dofs))
                 for li in range(len(Vs) - 1)]
    patches_idx = [[jnp.asarray(p) for p in vertex_patches(Vs[li])]
                   for li in range(len(Vs) - 1)]
    return Vs, prolongs, restricts, patches_idx


def build_diff(Vs, prolongs, patches_idx, tau_scale):
    """tau-dependent JAX-traceable build: operators, Galerkin coarse
    matrices, patch smoothers (jnp.linalg.inv-based), coarse LU."""
    applies = [(lambda u, V=V: bdm_k_viscous_apply(
        V, u, nu=NU, tau_scale=tau_scale)) for V in Vs]
    A_levels = [jax.jacobian(applies[0])(jnp.zeros(Vs[0].total_dofs))]
    for li in range(1, len(Vs)):
        Pl = jax.jacobian(prolongs[li - 1])(jnp.zeros(Vs[li].total_dofs))
        A_levels.append(Pl.T @ A_levels[-1] @ Pl)
    def make_smoother(A_l, idx_list):
        blocks_inv = [(idx, jnp.linalg.inv(A_l[idx[:, None], idx[None, :]]))
                      for idx in idx_list]
        def M_inv(r):
            out = jnp.zeros_like(r)
            for idx, binv in blocks_inv:
                out = out.at[idx].add(binv @ r[idx])
            return out
        return M_inv
    smoothers = [make_smoother(A_levels[li], patches_idx[li])
                 for li in range(len(Vs) - 1)]
    coarse_lu = jax.scipy.linalg.lu_factor(A_levels[-1])
    return applies, smoothers, coarse_lu


def make_vcycle(applies, smoothers, prolongs, restricts, coarse_lu,
                 *, nsm=2, gamma=2):
    nlev = len(applies)
    def cyc(li, b, omegas):
        if li == nlev - 1:
            return jax.scipy.linalg.lu_solve(coarse_lu, b)
        A, M, om = applies[li], smoothers[li], omegas[li]
        x = jnp.zeros_like(b)
        for _ in range(nsm):
            x = x + om * M(b - A(x))
        rc = restricts[li](b - A(x))[0]
        e = cyc(li + 1, rc, omegas)
        for _ in range(gamma - 1):
            e = e + cyc(li + 1, rc - applies[li + 1](e), omegas)
        x = x + prolongs[li](e)
        for _ in range(nsm):
            x = x + om * M(b - A(x))
        return x
    return lambda omegas, b: cyc(0, b, omegas)


def loss_fn_factory(Vs, prolongs, restricts, patches_idx, n_fine,
                    *, n_rhs=4, n_cycles=4, gamma=2, seed=0):
    rng = jax.random.PRNGKey(seed)
    bs = jax.random.normal(rng, (n_rhs, n_fine))
    def loss(params):
        tau_scale = params["tau_scale"]
        omegas = params["omegas"]
        applies, smoothers, coarse_lu = build_diff(
            Vs, prolongs, patches_idx, tau_scale)
        vcycle = make_vcycle(applies, smoothers, prolongs, restricts,
                              coarse_lu, gamma=gamma)
        Aop = applies[0]
        def per_b(b):
            r0 = jnp.linalg.norm(b)
            x = jnp.zeros_like(b)
            for _ in range(n_cycles):
                x = x + vcycle(omegas, b - Aop(x))
            rN = jnp.linalg.norm(Aop(x) - b)
            return (jnp.log(rN) - jnp.log(r0)) / n_cycles
        return jnp.mean(jax.vmap(per_b)(bs))
    return loss


def auto_omegas(applies, smoothers, Vs):
    rng = np.random.default_rng(0)
    out = []
    for li in range(len(Vs) - 1):
        v = jnp.asarray(rng.standard_normal(Vs[li].total_dofs))
        v = v / jnp.linalg.norm(v); lam = 1.0
        for _ in range(40):
            w = smoothers[li](applies[li](v))
            lam = float(jnp.linalg.norm(w)); v = w / lam
        out.append(1.0 / lam)
    return jnp.asarray(out)


def stationary_iters(vcycle, Aop, omegas, b, tol=1e-9, maxiter=200):
    x = jnp.zeros_like(b); nb = float(jnp.linalg.norm(b))
    for k in range(maxiter):
        r = b - Aop(x)
        if float(jnp.linalg.norm(r)) / nb < tol:
            return k
        x = x + vcycle(omegas, r)
    return maxiter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=4)
    ap.add_argument("--tau-init", type=float, default=4.0,
                    help="initial tau_scale (Hesthaven default)")
    ap.add_argument("--n-rhs", type=int, default=4)
    ap.add_argument("--n-cycles", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=40,
                    help="Adam iters (joint training is heavier per step)")
    ap.add_argument("--lr-tau", type=float, default=0.5)
    ap.add_argument("--lr-omega", type=float, default=0.05)
    ap.add_argument("--cycle", choices=("V", "W"), default="W")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    gamma = 1 if args.cycle == "V" else 2

    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, args.kx, args.kx)
    mesh = build_mesh_2d(VX, VY, EToV, args.order)
    # Build the tau-independent static pieces ONCE.
    Vs, prolongs, restricts, patches_idx = build_static(mesh, args.order)
    # Seed omegas from the heuristic at tau_init.
    applies0, smoothers0, _ = build_diff(Vs, prolongs, patches_idx, args.tau_init)
    om0 = auto_omegas(applies0, smoothers0, Vs)
    n_fine = Vs[0].total_dofs
    print(f"BDM_{args.order} {args.kx}x{args.kx}: n_fine={n_fine}, "
          f"cycle={args.cycle}", flush=True)
    print(f"  init: tau_scale={args.tau_init}, omegas={[float(x) for x in om0]}",
          flush=True)

    loss_fn = loss_fn_factory(
        Vs, prolongs, restricts, patches_idx, n_fine,
        n_rhs=args.n_rhs, n_cycles=args.n_cycles, gamma=gamma)
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    params = {"tau_scale": jnp.asarray(args.tau_init), "omegas": om0}
    L0 = float(loss_jit(params))
    print(f"  initial loss (log-rate): {L0:.4f}  -> rate {float(jnp.exp(L0)):.4f}",
          flush=True)

    # Adam with per-key learning rates (tau learns faster than omegas).
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v_st = {k: jnp.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lrs = {"tau_scale": args.lr_tau, "omegas": args.lr_omega}
    losses = [L0]; tau_trace = [float(params["tau_scale"])]
    omega_trace = [np.asarray(params["omegas"])]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(params)
        for k in params:
            m[k] = beta1 * m[k] + (1 - beta1) * g[k]
            v_st[k] = beta2 * v_st[k] + (1 - beta2) * g[k] ** 2
            m_hat = m[k] / (1 - beta1 ** it)
            v_hat = v_st[k] / (1 - beta2 ** it)
            params[k] = params[k] - lrs[k] * m_hat / (jnp.sqrt(v_hat) + eps)
        # Project tau >= 0.7 to stay above the coercivity floor.
        params["tau_scale"] = jnp.maximum(params["tau_scale"], 0.7)
        L = float(loss_jit(params))
        losses.append(L); tau_trace.append(float(params["tau_scale"]))
        omega_trace.append(np.asarray(params["omegas"]))
        if it % 5 == 0 or it == args.n_train:
            print(f"  iter {it:3d}: loss={L:.4f} rate={float(jnp.exp(L)):.4f}  "
                  f"tau={float(params['tau_scale']):.3f}  "
                  f"omegas={[f'{float(x):.4f}' for x in params['omegas']]}",
                  flush=True)
    print(f"\ntotal Adam time: {time.time() - t0:.1f}s", flush=True)

    # Final evaluation: build hierarchy at learned params, measure iters.
    tau_final = float(params["tau_scale"])
    om_final = params["omegas"]
    applies_f, smoothers_f, lu_f = build_diff(
        Vs, prolongs, patches_idx, tau_final)
    vcycle_f = make_vcycle(applies_f, smoothers_f, prolongs, restricts,
                            lu_f, gamma=gamma)
    b = jnp.asarray(np.random.default_rng(99).standard_normal(n_fine))
    it_final = stationary_iters(vcycle_f, applies_f[0], om_final, b)
    # Compare to (a) Hesthaven baseline tau=4 + auto omegas;
    # (b) sweep optimum tau=1 + auto omegas.
    refs = []
    for tau_ref in (4.0, 1.0):
        ap_r, sm_r, lu_r = build_diff(Vs, prolongs, patches_idx, tau_ref)
        om_r = auto_omegas(ap_r, sm_r, Vs)
        vc_r = make_vcycle(ap_r, sm_r, prolongs, restricts, lu_r,
                            gamma=gamma)
        L_r = float(loss_jit({"tau_scale": jnp.asarray(tau_ref),
                              "omegas": om_r}))
        it_r = stationary_iters(vc_r, ap_r[0], om_r, b)
        refs.append((tau_ref, [float(x) for x in om_r], float(jnp.exp(L_r)),
                     int(it_r)))
        print(f"  reference: tau={tau_ref} + auto omegas: "
              f"rate={float(jnp.exp(L_r)):.4f}, iters to 1e-9: {it_r}",
              flush=True)
    print(f"  LEARNED: tau={tau_final:.3f} + learned omegas: "
          f"rate={float(jnp.exp(losses[-1])):.4f}, iters to 1e-9: {it_final}",
          flush=True)

    # Plot.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(losses, "o-", ms=3); axes[0].set_xlabel("Adam iter")
    axes[0].set_ylabel("log V-cycle rate")
    axes[0].axhline(refs[0][2] and np.log(refs[0][2]), color="r", ls="--",
                     label=f"heuristic (tau=4, auto om): {refs[0][2]:.3f}")
    axes[0].axhline(np.log(refs[1][2]), color="g", ls="--",
                     label=f"tau=1 + auto om: {refs[1][2]:.3f}")
    axes[0].set_title("joint Adam training"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(tau_trace, "o-", ms=3, color="purple")
    axes[1].set_xlabel("Adam iter"); axes[1].set_ylabel("tau_scale")
    axes[1].axhline(4.0, color="r", ls=":", label="Hesthaven default")
    axes[1].axhline(1.0, color="g", ls=":", label="sweep optimum")
    axes[1].set_title("learned tau_scale"); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    om_arr = np.array(omega_trace)
    for i in range(om_arr.shape[1]):
        axes[2].plot(om_arr[:, i], "-", label=f"level {i} (BDM_{args.order-i})")
    axes[2].set_xlabel("Adam iter"); axes[2].set_ylabel("omega")
    axes[2].set_title("learned omegas"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
    fig.tight_layout()
    out = args.out_dir / "learn_tau_and_omega.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}", flush=True)
    summary = dict(
        order=args.order, kx=args.kx, n_fine=int(n_fine),
        cycle=args.cycle,
        init_tau=float(args.tau_init),
        init_omegas=[float(x) for x in om0],
        init_rate=float(jnp.exp(L0)),
        learned_tau=tau_final,
        learned_omegas=[float(x) for x in om_final],
        learned_rate=float(jnp.exp(losses[-1])),
        learned_iters_to_1e_9=int(it_final),
        baselines=[
            {"tau": t, "omegas": om, "rate": r, "iters_to_1e_9": it}
            for (t, om, r, it) in refs
        ],
        losses=losses, tau_trace=tau_trace,
    )
    with open(args.out_dir / "learn_tau_and_omega.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'learn_tau_and_omega.json'}", flush=True)


if __name__ == "__main__":
    main()
