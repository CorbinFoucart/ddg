"""Per-patch smoother damping omegas via Adam.

Extension of ``learn_smoother_omega.py``: instead of one scalar omega
per multigrid level, give each *vertex patch* its own scalar weight.
This is the most expressive per-level smoother that stays additive
(no patch-to-patch coupling). The expected pay-off: the V-cycle can
adapt to the spatial structure of the operator -- corner / boundary
patches and interior patches typically need different damping, which
a single per-level omega cannot represent.

Parameter count: for BDM_3 on a $K_x \times K_x$ mesh, vertices on the
finest level number $(K_x + 1)^2$; the next p-level (BDM_2) sees the
same vertex count. With two smoothable levels (BDM_3, BDM_2) and the
BDM_1 coarse grid solved directly, the trainable parameter count is
$2 (K_x + 1)^2$. For $K_x = 4$ this is 50 scalars.

Loss is unchanged from the per-level experiment: mean log-rate of the
stationary V/W-cycle on a small batch of random RHS.
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


def build_hierarchy(mesh, order, tau_scale):
    Vs = [BDM.from_mesh(mesh, order=k) for k in range(order, 0, -1)]
    applies = [(lambda u, V=V: bdm_k_viscous_apply(V, u, nu=NU, tau_scale=tau_scale))
               for V in Vs]
    prolongs = [make_p_prolong(Vs[li + 1], Vs[li]) for li in range(len(Vs) - 1)]
    restricts = [jax.linear_transpose(prolongs[li], jnp.zeros(Vs[li + 1].total_dofs))
                  for li in range(len(Vs) - 1)]
    # Galerkin coarse operators (dense, numpy for memory).
    A0 = np.asarray(jax.jacobian(applies[0])(jnp.zeros(Vs[0].total_dofs)))
    A_dense = [A0]
    for li in range(1, len(Vs)):
        Pl = np.asarray(jax.jacobian(prolongs[li - 1])(jnp.zeros(Vs[li].total_dofs)))
        A_dense.append(Pl.T @ A_dense[-1] @ Pl)
    patches_idx = [[jnp.asarray(p) for p in vertex_patches(Vs[li])]
                   for li in range(len(Vs) - 1)]
    # Per-patch inverses (precomputed, JAX arrays).
    patch_blocks_inv = []
    for li in range(len(Vs) - 1):
        blocks = [(idx, jnp.asarray(np.linalg.inv(
                    A_dense[li][np.asarray(idx)[:, None],
                                np.asarray(idx)[None, :]])))
                  for idx in patches_idx[li]]
        patch_blocks_inv.append(blocks)
    coarse_lu = jax.scipy.linalg.lu_factor(jnp.asarray(A_dense[-1]))
    return Vs, applies, patch_blocks_inv, prolongs, restricts, coarse_lu


def make_per_patch_smoother(patch_blocks):
    """Returns ``M_inv(omegas_vec, r) -> y`` where ``omegas_vec`` has
    one entry per patch on this level."""
    def M_inv(omegas_vec, r):
        out = jnp.zeros_like(r)
        for p, (idx, binv) in enumerate(patch_blocks):
            out = out.at[idx].add(omegas_vec[p] * (binv @ r[idx]))
        return out
    return M_inv


def make_vcycle_per_patch(applies, patch_blocks_per_level, prolongs,
                           restricts, coarse_lu, *, nsm=2, gamma=2):
    """V/W-cycle with per-patch damping. ``omegas_per_level`` is a list
    of arrays, one per non-coarsest level, each of shape
    ``(n_patches_at_that_level,)``."""
    nlev = len(applies)
    smoothers = [make_per_patch_smoother(pb)
                 for pb in patch_blocks_per_level]

    def cyc(li, b, omegas):
        if li == nlev - 1:
            return jax.scipy.linalg.lu_solve(coarse_lu, b)
        A, M = applies[li], smoothers[li]
        om_l = omegas[li]
        x = jnp.zeros_like(b)
        for _ in range(nsm):
            x = x + M(om_l, b - A(x))
        rc = restricts[li](b - A(x))[0]
        e = cyc(li + 1, rc, omegas)
        for _ in range(gamma - 1):
            e = e + cyc(li + 1, rc - applies[li + 1](e), omegas)
        x = x + prolongs[li](e)
        for _ in range(nsm):
            x = x + M(om_l, b - A(x))
        return x
    return lambda omegas, b: cyc(0, b, omegas)


def auto_per_patch_omegas(applies, patch_blocks_per_level, Vs):
    """Initialise each patch's omega via the per-level ``1 / lam_max``
    heuristic, then broadcast to all patches. (We don't have a per-patch
    spectral estimate; the global per-level lam_max is the closest
    equivalent.)"""
    rng = np.random.default_rng(0)
    out = []
    for li in range(len(Vs) - 1):
        # Build a single-omega smoother to compute lam_max.
        blocks = patch_blocks_per_level[li]
        def Minv_unit(r):
            o = jnp.zeros_like(r)
            for idx, binv in blocks:
                o = o.at[idx].add(binv @ r[idx])
            return o
        v = jnp.asarray(rng.standard_normal(Vs[li].total_dofs))
        v = v / jnp.linalg.norm(v); lam = 1.0
        for _ in range(40):
            w = Minv_unit(applies[li](v))
            lam = float(jnp.linalg.norm(w)); v = w / lam
        out.append(jnp.full(len(blocks), 1.0 / lam))
    return out


def make_rate_loss(applies, vcycle, n_fine, *, n_rhs, n_cycles, seed=0):
    rng = jax.random.PRNGKey(seed)
    bs = jax.random.normal(rng, (n_rhs, n_fine))
    Aop = applies[0]
    def loss(omegas):
        def per_b(b):
            r0 = jnp.linalg.norm(b)
            x = jnp.zeros_like(b)
            for _ in range(n_cycles):
                x = x + vcycle(omegas, b - Aop(x))
            rN = jnp.linalg.norm(Aop(x) - b)
            return (jnp.log(rN) - jnp.log(r0)) / n_cycles
        return jnp.mean(jax.vmap(per_b)(bs))
    return loss


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
    ap.add_argument("--tau-scale", type=float, default=1.0,
                    help="default to the sweep-optimum tau=1")
    ap.add_argument("--n-rhs", type=int, default=4)
    ap.add_argument("--n-cycles", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--cycle", choices=("V", "W"), default="W")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    gamma = 1 if args.cycle == "V" else 2

    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, args.kx, args.kx)
    mesh = build_mesh_2d(VX, VY, EToV, args.order)
    Vs, applies, patch_blocks, prolongs, restricts, coarse_lu = build_hierarchy(
        mesh, args.order, args.tau_scale)
    n_fine = Vs[0].total_dofs
    patches_per_level = [len(pb) for pb in patch_blocks]
    print(f"BDM_{args.order} {args.kx}x{args.kx}: n_fine={n_fine}, "
          f"patches per level: {patches_per_level}, "
          f"total trainable omegas: {sum(patches_per_level)}, "
          f"tau_scale={args.tau_scale}", flush=True)

    om0 = auto_per_patch_omegas(applies, patch_blocks, Vs)
    print(f"  initial per-patch omegas (broadcast 1/lam_max):", flush=True)
    for li, om in enumerate(om0):
        print(f"    level {li} (BDM_{args.order-li}): {len(om)} patches at "
              f"{float(om[0]):.4f}", flush=True)

    vcycle = make_vcycle_per_patch(applies, patch_blocks, prolongs,
                                     restricts, coarse_lu, gamma=gamma)
    loss_fn = make_rate_loss(applies, vcycle, n_fine,
                              n_rhs=args.n_rhs, n_cycles=args.n_cycles)
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    L0 = float(loss_jit(om0))
    print(f"  initial loss (log-rate): {L0:.4f}  -> rate {float(jnp.exp(L0)):.4f}",
          flush=True)

    # Adam over the pytree of per-level per-patch omega vectors.
    omegas = list(om0)
    m = [jnp.zeros_like(o) for o in omegas]
    v = [jnp.zeros_like(o) for o in omegas]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    losses = [L0]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(omegas)
        for li in range(len(omegas)):
            m[li] = beta1 * m[li] + (1 - beta1) * g[li]
            v[li] = beta2 * v[li] + (1 - beta2) * g[li] ** 2
            m_hat = m[li] / (1 - beta1 ** it)
            v_hat = v[li] / (1 - beta2 ** it)
            omegas[li] = omegas[li] - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        L = float(loss_jit(omegas))
        losses.append(L)
        if it % 10 == 0 or it == args.n_train:
            stats = [(float(jnp.min(om)), float(jnp.mean(om)), float(jnp.max(om)))
                     for om in omegas]
            stat_str = ", ".join(f"level{li} [{lo:.3f}, {mn:.3f}, {hi:.3f}]"
                                  for li, (lo, mn, hi) in enumerate(stats))
            print(f"  iter {it:3d}: loss={L:.4f} rate={float(jnp.exp(L)):.4f}  "
                  f"omegas (min, mean, max): {stat_str}", flush=True)
    print(f"\ntotal Adam time: {time.time() - t0:.1f}s", flush=True)

    # Final iter count.
    b = jnp.asarray(np.random.default_rng(99).standard_normal(n_fine))
    it_final = stationary_iters(vcycle, applies[0], omegas, b)
    it_auto = stationary_iters(vcycle, applies[0], om0, b)
    print(f"\nstationary cycles to rel-resid 1e-9 at tau={args.tau_scale}:", flush=True)
    print(f"  per-patch auto (lam_max, broadcast): {it_auto}", flush=True)
    print(f"  per-patch learned:                   {it_final}  "
          f"({it_auto/max(it_final,1):.2f}x speedup)", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(losses, "o-", ms=3)
    axes[0].set_xlabel("Adam iter"); axes[0].set_ylabel("log V-cycle rate")
    axes[0].axhline(L0, color="r", ls="--",
                     label=f"per-patch auto rate {float(jnp.exp(L0)):.3f}")
    axes[0].set_title(f"per-patch omegas, BDM_{args.order}, tau={args.tau_scale}")
    axes[0].grid(alpha=0.3); axes[0].legend()
    # Per-patch omega histograms at the finest level.
    om_arr = np.asarray(omegas[0])
    axes[1].hist(om_arr, bins=15, alpha=0.7, color="C0",
                  label=f"learned (BDM_{args.order})")
    axes[1].axvline(float(om0[0][0]), color="r", ls="--",
                     label=f"auto {float(om0[0][0]):.3f}")
    axes[1].set_xlabel("omega"); axes[1].set_ylabel("count")
    axes[1].set_title("per-patch omega distribution"); axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = args.out_dir / "learn_per_patch_omega.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}", flush=True)
    summary = dict(
        order=args.order, kx=args.kx, tau_scale=args.tau_scale,
        n_fine=int(n_fine),
        patches_per_level=patches_per_level,
        n_params=int(sum(patches_per_level)),
        rate_init=float(jnp.exp(L0)),
        rate_final=float(jnp.exp(losses[-1])),
        iters_auto_to_1e_9=int(it_auto),
        iters_learned_to_1e_9=int(it_final),
        losses=losses,
    )
    with open(args.out_dir / "learn_per_patch_omega.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'learn_per_patch_omega.json'}", flush=True)


if __name__ == "__main__":
    main()
