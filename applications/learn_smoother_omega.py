"""Experiment A: learn per-level MG smoother damping omegas via Adam.

The vertex-patch additive-Schwarz smoother in the BDM velocity hp-MG
currently uses a hand-tuned damping ``omega = 1/lam_max(M^{-1} A)``
(power-iteration estimate, one per level). This is a heuristic; the
optimal omegas depend on the spectrum of the *preconditioned* smoothing
operator and the way the coarse-grid correction handles the modes the
smoother leaves behind, neither of which the lam_max heuristic captures.

This experiment makes the omegas **trainable** scalars and minimises the
*measured* V/W-cycle log-rate on a small batch of random RHS, via Adam.
The whole solver is one JAX function, so ``jax.grad`` flows through the
recursive cycle, the patch-Jacobi smoother, the Galerkin coarse
correction, and the inter-level transfers without writing any adjoint
code. The only differentiable parameters are the few per-level damping
scalars; the operator + hierarchy are fixed.

Setup: BDM_3 viscous block on a uniform mesh, p-multigrid hierarchy
(BDM_3 -> BDM_2 -> BDM_1) on the same mesh (n_h = 0). For BDM_3 on a
small mesh the W-cycle has two smoothable levels (BDM_3 and BDM_2), so
``omegas`` is a 2-vector.
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
from ddg.dg2d.bdm_multigrid import (
    make_p_prolong, make_vertex_patch_smoother, vertex_patches,
)


NU = 1.0 / 100.0


def build_hierarchy(mesh, order, tau_scale):
    """Build the level operators + smoothers + transfers + coarse LU for
    a p-only hp-MG (BDM_order down to BDM_1 on the same mesh). Returns
    everything the V/W-cycle needs except ``omegas``."""
    Vs = [BDM.from_mesh(mesh, order=k) for k in range(order, 0, -1)]
    applies = [
        (lambda u, V=V: bdm_k_viscous_apply(V, u, nu=NU, tau_scale=tau_scale))
        for V in Vs
    ]
    prolongs = [make_p_prolong(Vs[li + 1], Vs[li]) for li in range(len(Vs) - 1)]
    restricts = [
        jax.linear_transpose(prolongs[li], jnp.zeros(Vs[li + 1].total_dofs))
        for li in range(len(Vs) - 1)
    ]
    # Galerkin coarse operators (dense).
    A0 = np.asarray(jax.jacobian(applies[0])(jnp.zeros(Vs[0].total_dofs)))
    A_dense = [A0]
    for li in range(1, len(Vs)):
        Pl = np.asarray(jax.jacobian(prolongs[li - 1])(jnp.zeros(Vs[li].total_dofs)))
        A_dense.append(Pl.T @ A_dense[-1] @ Pl)
    smoothers = [make_vertex_patch_smoother(A_dense[li], vertex_patches(Vs[li]))
                 for li in range(len(Vs) - 1)]
    coarse_lu = jax.scipy.linalg.lu_factor(jnp.asarray(A_dense[-1]))
    return Vs, applies, smoothers, prolongs, restricts, coarse_lu


def auto_omegas(applies, smoothers, Vs):
    """Hand-tuned baseline: omega_l = 1 / lam_max(M_l^{-1} A_l) via power
    iteration. One scalar per smoothable level."""
    rng = np.random.default_rng(0)
    out = []
    for li in range(len(Vs) - 1):
        v = jnp.asarray(rng.standard_normal(Vs[li].total_dofs))
        v = v / jnp.linalg.norm(v)
        lam = 1.0
        for _ in range(40):
            w = smoothers[li](applies[li](v))
            lam = float(jnp.linalg.norm(w))
            v = w / lam
        out.append(1.0 / lam)
    return jnp.asarray(out)


def make_vcycle(applies, smoothers, prolongs, restricts, coarse_lu,
                *, nsm=2, gamma=2):
    """Returns ``vcycle(omegas, b) -> x``. Differentiable in ``omegas``;
    ``gamma=2`` is a W-cycle, ``gamma=1`` a V-cycle."""
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


def make_rate_loss(applies, vcycle, n_fine, *, n_rhs, n_cycles, seed=0):
    """Loss = mean over a batch of random RHS of ``log(||r_K||/||r_0||) /
    K``, with the V-cycle used as a stationary iterator. ``log`` of the
    contraction factor: lower = better PC."""
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


def count_stationary(applies, vcycle, omegas, *, seed=99, tol=1e-9, maxiter=200):
    rng = np.random.default_rng(seed)
    n = applies[0](jnp.zeros(1)).shape[0] if False else None
    # Just generate one RHS of the right shape:
    # We need n_fine; infer by calling the apply on a probe.
    # Easier: caller passes b explicitly.
    raise NotImplementedError  # use the dedicated helper below


def stationary_iters(vcycle, Aop, omegas, b, tol=1e-9, maxiter=200):
    x = jnp.zeros_like(b)
    nb = float(jnp.linalg.norm(b))
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
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--n-rhs", type=int, default=4)
    ap.add_argument("--n-cycles", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--cycle", choices=("V", "W"), default="W")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, args.kx, args.kx)
    mesh = build_mesh_2d(VX, VY, EToV, args.order)
    Vs, applies, smoothers, prolongs, restricts, coarse_lu = build_hierarchy(
        mesh, args.order, args.tau_scale)
    n_fine = Vs[0].total_dofs
    print(f"BDM_{args.order} {args.kx}x{args.kx}: n_fine={n_fine}, "
          f"sublevels={[V.total_dofs for V in Vs]}, "
          f"cycle={args.cycle}, tau_scale={args.tau_scale}", flush=True)
    om0 = auto_omegas(applies, smoothers, Vs)
    print(f"  auto omegas (1/lam_max): "
          f"{[float(x) for x in om0]}", flush=True)

    gamma = 1 if args.cycle == "V" else 2
    vcycle = make_vcycle(applies, smoothers, prolongs, restricts, coarse_lu,
                          gamma=gamma)
    loss_fn = make_rate_loss(
        applies, vcycle, n_fine,
        n_rhs=args.n_rhs, n_cycles=args.n_cycles)
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    L0 = float(loss_jit(om0))
    print(f"  initial loss (log-rate): {L0:.4f}  -> rate {float(jnp.exp(L0)):.4f}",
          flush=True)

    # Adam.
    omegas = jnp.asarray(om0)
    m = jnp.zeros_like(omegas); v = jnp.zeros_like(omegas)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = args.lr
    losses = [L0]
    omegas_trace = [np.asarray(omegas)]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(omegas)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** it)
        v_hat = v / (1 - beta2 ** it)
        omegas = omegas - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        L = float(loss_jit(omegas))
        losses.append(L)
        omegas_trace.append(np.asarray(omegas))
        if it % 10 == 0 or it == args.n_train:
            print(f"  iter {it:3d}: loss={L:.4f}  rate={float(jnp.exp(L)):.4f}  "
                  f"omegas={[f'{float(x):.4f}' for x in omegas]}", flush=True)
    print(f"total Adam time: {time.time() - t0:.1f}s", flush=True)

    # Stationary-iteration counts.
    rng = np.random.default_rng(99)
    b = jnp.asarray(rng.standard_normal(n_fine))
    Aop = applies[0]
    it_auto = stationary_iters(vcycle, Aop, om0, b)
    it_learned = stationary_iters(vcycle, Aop, omegas, b)
    print(f"\nstationary cycles to rel-resid 1e-9:  "
          f"auto={it_auto}   learned={it_learned}   "
          f"({it_auto/max(it_learned,1):.2f}x speedup)", flush=True)

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(losses, "o-", ms=3); axes[0].set_xlabel("Adam iter")
    axes[0].set_ylabel("log V-cycle rate"); axes[0].grid(alpha=0.3)
    axes[0].axhline(L0, color="r", ls="--",
                     label=f"auto rate {float(jnp.exp(L0)):.3f}")
    axes[0].set_title(f"BDM_{args.order} hp-MG, log-rate per cycle")
    axes[0].legend()
    omegas_arr = np.array(omegas_trace)
    for i in range(omegas_arr.shape[1]):
        axes[1].plot(omegas_arr[:, i], "-", label=f"level {i} (BDM_{args.order-i})")
        axes[1].axhline(float(om0[i]), color=f"C{i}", ls=":", alpha=0.5,
                         label=f"  auto level {i}: {float(om0[i]):.3f}")
    axes[1].set_xlabel("Adam iter"); axes[1].set_ylabel("omega")
    axes[1].set_title("learned omegas"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = args.out_dir / "learn_smoother_omega.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}", flush=True)
    summary = dict(
        order=args.order, kx=args.kx, n_fine=int(n_fine),
        tau_scale=float(args.tau_scale), cycle=args.cycle,
        omegas_auto=[float(x) for x in om0],
        omegas_learned=[float(x) for x in omegas],
        rate_auto=float(jnp.exp(L0)),
        rate_learned=float(jnp.exp(losses[-1])),
        iters_auto_to_1e_9=int(it_auto),
        iters_learned_to_1e_9=int(it_learned),
        losses=losses,
    )
    with open(args.out_dir / "learn_smoother_omega.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'learn_smoother_omega.json'}", flush=True)


if __name__ == "__main__":
    main()
