"""Per-element local SIPG penalty scale via Adam.

Extension of ``scripts/learn_tau_and_omega.py`` (one global scalar
``tau_scale``, joint with smoother omegas) and ``sweep_tau_scale.py``
(global tau, no learning): instead of one knob for the entire mesh,
give *each element* its own scalar penalty multiplier
``tau_K``. To keep the bilinear form symmetric, the per-face penalty
used at face ``f`` between elements ``K_a, K_b`` is the average
``(tau_K[K_a] + tau_K[K_b]) / 2`` --- on boundary faces the neighbour
is the element itself, so the average collapses to ``tau_K[K]``. The
geometric factor ``(k+1)^2 / h_F`` is unchanged.

For BDM_3 on $K_x \\times K_x$ the trainable count is $K_x^2 \\cdot 2$
elements (two right triangles per quad). On the 4x4 mesh that is 32
scalars (cf. 50 for per-patch ω, 1 for global tau_scale).

The smoother ω are held fixed at the per-level 1/lam_max heuristic
computed at the *initial* operator (tau_K = ``args.tau_init``).
This isolates the contribution of spatial tau structure --- joint
ω + tau_K would couple to the per-patch ω experiment.

Initial guess: tau_K = ``args.tau_init`` everywhere (default 1.0, the
global sweep optimum). The honest question is whether Adam finds
useful spatial variation on top of that.
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
    """tau-independent pieces (built ONCE outside JIT).

    Includes ``EToE`` so we can form the symmetric face-tau inside the
    JAX-traced loss without re-querying the topology each call."""
    Vs = [BDM.from_mesh(mesh, order=k) for k in range(order, 0, -1)]
    prolongs = [make_p_prolong(Vs[li + 1], Vs[li])
                for li in range(len(Vs) - 1)]
    restricts = [jax.linear_transpose(prolongs[li],
                                       jnp.zeros(Vs[li + 1].total_dofs))
                 for li in range(len(Vs) - 1)]
    patches_idx = [[jnp.asarray(p) for p in vertex_patches(Vs[li])]
                   for li in range(len(Vs) - 1)]
    EToE = jnp.asarray(mesh.EToE)               # (K, Nf)
    return Vs, prolongs, restricts, patches_idx, EToE


def face_tau_from_per_element(tau_K, EToE):
    """Symmetric face penalty: ``0.5 * (tau_K[K] + tau_K[EToE[K, f]])``.

    Boundary faces have ``EToE[K, f] == K`` so the average collapses
    to ``tau_K[K]``; no special case needed."""
    K = tau_K.shape[0]
    tau_self = tau_K[:, None]                       # (K, 1)
    tau_nbr = tau_K[EToE]                           # (K, Nf)
    return 0.5 * (tau_self + tau_nbr)               # (K, Nf)


def build_diff(Vs, prolongs, patches_idx, tau_K, EToE):
    """tau_K-dependent JAX-traceable build: fine-level matvec uses the
    symmetric per-face tau; coarse operators are Galerkin
    (so they inherit tau_K structure automatically). Patch smoothers
    use ``jnp.linalg.inv`` so they're differentiable in tau_K."""
    tau_face = face_tau_from_per_element(tau_K, EToE)  # (K, Nf)
    applies = [
        (lambda u: bdm_k_viscous_apply(
            Vs[0], u, nu=NU, tau_scale=tau_face)),
    ]
    # Coarser-level matvecs: same geometric tau_face structure rides
    # through Galerkin (P^T A P). We still need callable applies[li]
    # at each coarser level for the smoother+vcycle to use.
    A_levels = [jax.jacobian(applies[0])(jnp.zeros(Vs[0].total_dofs))]
    for li in range(1, len(Vs)):
        Pl = jax.jacobian(prolongs[li - 1])(jnp.zeros(Vs[li].total_dofs))
        A_levels.append(Pl.T @ A_levels[-1] @ Pl)
    # Coarse-level applies are matrix-vector products against the
    # Galerkin operator (treated as a function of u only --- still
    # differentiable in tau_K because A_levels[li] is).
    for li in range(1, len(Vs)):
        A_l = A_levels[li]
        applies.append(lambda u, A=A_l: A @ u)
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


def loss_fn_factory(Vs, prolongs, restricts, patches_idx, EToE,
                     omegas_fixed, n_fine, *, n_rhs=4, n_cycles=4,
                     gamma=2, seed=0):
    rng = jax.random.PRNGKey(seed)
    bs = jax.random.normal(rng, (n_rhs, n_fine))
    def loss(tau_K):
        applies, smoothers, coarse_lu = build_diff(
            Vs, prolongs, patches_idx, tau_K, EToE)
        vcycle = make_vcycle(applies, smoothers, prolongs, restricts,
                              coarse_lu, gamma=gamma)
        Aop = applies[0]
        def per_b(b):
            r0 = jnp.linalg.norm(b)
            x = jnp.zeros_like(b)
            for _ in range(n_cycles):
                x = x + vcycle(omegas_fixed, b - Aop(x))
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
    ap.add_argument("--tau-init", type=float, default=1.0,
                    help="initial per-element tau (sweep optimum)")
    ap.add_argument("--n-rhs", type=int, default=4)
    ap.add_argument("--n-cycles", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tau-floor", type=float, default=0.5,
                    help="lower clamp on each tau_K to stay coercive")
    ap.add_argument("--cycle", choices=("V", "W"), default="W")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    gamma = 1 if args.cycle == "V" else 2

    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, args.kx, args.kx)
    mesh = build_mesh_2d(VX, VY, EToV, args.order)
    Vs, prolongs, restricts, patches_idx, EToE = build_static(mesh, args.order)
    n_fine = Vs[0].total_dofs
    K = int(Vs[0].K)
    print(f"BDM_{args.order} {args.kx}x{args.kx}: n_fine={n_fine}, "
          f"K_elements={K}, trainable tau_K={K}, cycle={args.cycle}",
          flush=True)

    # Initial tau_K = uniform tau_init. Seed omegas from per-level
    # heuristic at this initial operator; omegas are HELD FIXED.
    tau_K = jnp.full(K, args.tau_init)
    applies0, smoothers0, _ = build_diff(
        Vs, prolongs, patches_idx, tau_K, EToE)
    om0 = auto_omegas(applies0, smoothers0, Vs)
    print(f"  init: tau_K={float(args.tau_init):.3f} (uniform), "
          f"omegas={[float(x) for x in om0]} (fixed)", flush=True)

    loss_fn = loss_fn_factory(
        Vs, prolongs, restricts, patches_idx, EToE, om0, n_fine,
        n_rhs=args.n_rhs, n_cycles=args.n_cycles, gamma=gamma)
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    L0 = float(loss_jit(tau_K))
    print(f"  initial loss (log-rate): {L0:.4f}  -> rate {float(jnp.exp(L0)):.4f}",
          flush=True)

    # Adam on the (K,) tau_K vector.
    m = jnp.zeros_like(tau_K); v_st = jnp.zeros_like(tau_K)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    losses = [L0]; tau_trace = [np.asarray(tau_K)]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(tau_K)
        m = beta1 * m + (1 - beta1) * g
        v_st = beta2 * v_st + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** it)
        v_hat = v_st / (1 - beta2 ** it)
        tau_K = tau_K - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        tau_K = jnp.maximum(tau_K, args.tau_floor)
        L = float(loss_jit(tau_K))
        losses.append(L); tau_trace.append(np.asarray(tau_K))
        if it % 10 == 0 or it == args.n_train:
            print(f"  iter {it:3d}: loss={L:.4f} rate={float(jnp.exp(L)):.4f}  "
                  f"tau_K (min, mean, max): "
                  f"[{float(jnp.min(tau_K)):.3f}, "
                  f"{float(jnp.mean(tau_K)):.3f}, "
                  f"{float(jnp.max(tau_K)):.3f}]", flush=True)
    print(f"\ntotal Adam time: {time.time() - t0:.1f}s", flush=True)

    # Final eval: iterations to 1e-9 (learned vs uniform baseline).
    applies_f, smoothers_f, lu_f = build_diff(
        Vs, prolongs, patches_idx, tau_K, EToE)
    vcycle_f = make_vcycle(applies_f, smoothers_f, prolongs, restricts,
                            lu_f, gamma=gamma)
    b = jnp.asarray(np.random.default_rng(99).standard_normal(n_fine))
    it_learned = stationary_iters(vcycle_f, applies_f[0], om0, b)
    # Baseline at uniform tau_K = tau_init.
    tau0 = jnp.full(K, args.tau_init)
    ap0, sm0, lu0 = build_diff(Vs, prolongs, patches_idx, tau0, EToE)
    vc0 = make_vcycle(ap0, sm0, prolongs, restricts, lu0, gamma=gamma)
    it_uniform = stationary_iters(vc0, ap0[0], om0, b)
    print(f"\nstationary cycles to rel-resid 1e-9:", flush=True)
    print(f"  uniform tau={args.tau_init}: {it_uniform}", flush=True)
    print(f"  learned per-element tau_K: {it_learned}  "
          f"({it_uniform/max(it_learned,1):.2f}x speedup)", flush=True)

    # Plots: loss curve + tau_K spatial map.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(losses, "o-", ms=3)
    axes[0].set_xlabel("Adam iter"); axes[0].set_ylabel("log V-cycle rate")
    axes[0].axhline(L0, color="r", ls="--",
                     label=f"uniform tau={args.tau_init} rate {float(jnp.exp(L0)):.4f}")
    axes[0].set_title(f"per-element tau_K, BDM_{args.order}")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    tau_arr = np.asarray(tau_K)
    axes[1].hist(tau_arr, bins=15, alpha=0.7, color="C2",
                  label=f"learned tau_K (K={K})")
    axes[1].axvline(args.tau_init, color="r", ls="--",
                     label=f"uniform {args.tau_init}")
    axes[1].set_xlabel(r"$\tau_K$"); axes[1].set_ylabel("element count")
    axes[1].set_title("learned per-element tau distribution")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    # Spatial scatter: plot each element at its centroid, coloured by tau.
    cx = mesh.VX[mesh.EToV].mean(axis=1)  # (K,)
    cy = mesh.VY[mesh.EToV].mean(axis=1)
    sc = axes[2].scatter(cx, cy, c=tau_arr, s=120, cmap="coolwarm",
                          edgecolor="k", linewidth=0.4)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    axes[2].set_title("spatial tau_K")
    plt.colorbar(sc, ax=axes[2], shrink=0.8)
    fig.tight_layout()
    out = args.out_dir / "learn_per_element_tau.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}", flush=True)
    summary = dict(
        order=args.order, kx=args.kx, K=K, n_fine=int(n_fine),
        cycle=args.cycle, tau_init=float(args.tau_init),
        tau_floor=float(args.tau_floor),
        omegas_fixed=[float(x) for x in om0],
        init_rate=float(jnp.exp(L0)),
        learned_rate=float(jnp.exp(losses[-1])),
        learned_tau_K=[float(x) for x in tau_K],
        learned_iters_to_1e_9=int(it_learned),
        uniform_iters_to_1e_9=int(it_uniform),
        losses=losses,
    )
    with open(args.out_dir / "learn_per_element_tau.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'learn_per_element_tau.json'}", flush=True)


if __name__ == "__main__":
    main()
