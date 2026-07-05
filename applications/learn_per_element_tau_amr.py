"""Per-element local SIPG penalty on a non-uniform (AMR) mesh.

Direct test of the hypothesis from
``scripts/learn_per_element_tau.py``: on a uniform mesh with uniform
nu the per-element parameterisation finds spread but no iteration
speedup, because there is no spatial structure to localise against.
A red-refined mesh has variable face length ``h_F`` -- finer
elements have a much smaller penalty ``tau_scale * (k+1)^2 / h_F``
*per face* than coarse elements. The natural question: does the
``tau_K`` field that Adam learns now correlate with that
heterogeneity, and does the resulting V-cycle beat uniform
``tau_K = 1``?

Mesh: start from a uniform ``Kx0 x Kx0`` mesh and red-refine a
chosen sub-region (default: bottom-left quadrant) ``levels`` times.
Closure may bring in extra green refinements. The active mesh is
then non-uniform with up to a 2^levels ratio in element size.

Loss + smoother + Adam loop are identical to the uniform
experiment; the only thing that changes is the operator.
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
from ddg.dg2d.refinement import RefinementForest


NU = 1.0 / 100.0


def build_non_uniform_mesh(kx0, order, levels, region):
    """Red-refine ``levels`` times within ``region = (x_lo, x_hi, y_lo, y_hi)``
    centroid box. Returns the active conforming :class:`Mesh2D`."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, kx0, kx0)
    seed = build_mesh_2d(VX, VY, EToV, order)
    forest = RefinementForest(seed, max_level=levels)
    x_lo, x_hi, y_lo, y_hi = region
    for _ in range(levels):
        active, active_to_leaf = forest.active_mesh()
        VXa = np.asarray(active.VX); VYa = np.asarray(active.VY)
        EToVa = np.asarray(active.EToV)
        cx = VXa[EToVa].mean(axis=1)
        cy = VYa[EToVa].mean(axis=1)
        in_region = (cx >= x_lo) & (cx <= x_hi) & (cy >= y_lo) & (cy <= y_hi)
        # active->leaf may map multiple active elements (green sib pair)
        # to the same leaf id; mark the union as a set of unique leaves.
        marked = np.unique(np.asarray(active_to_leaf)[in_region])
        if marked.size == 0:
            break
        forest.refine(marked)
    active, _ = forest.active_mesh()
    return build_mesh_2d(np.asarray(active.VX), np.asarray(active.VY),
                          np.asarray(active.EToV), order)


def face_tau_from_per_element(tau_K, EToE):
    K = tau_K.shape[0]
    tau_self = tau_K[:, None]
    tau_nbr = tau_K[EToE]
    return 0.5 * (tau_self + tau_nbr)


def build_static(mesh, order):
    Vs = [BDM.from_mesh(mesh, order=k) for k in range(order, 0, -1)]
    prolongs = [make_p_prolong(Vs[li + 1], Vs[li])
                for li in range(len(Vs) - 1)]
    restricts = [jax.linear_transpose(prolongs[li],
                                       jnp.zeros(Vs[li + 1].total_dofs))
                 for li in range(len(Vs) - 1)]
    patches_idx = [[jnp.asarray(p) for p in vertex_patches(Vs[li])]
                   for li in range(len(Vs) - 1)]
    EToE = jnp.asarray(mesh.EToE)
    return Vs, prolongs, restricts, patches_idx, EToE


def build_diff(Vs, prolongs, patches_idx, tau_K, EToE):
    tau_face = face_tau_from_per_element(tau_K, EToE)
    applies = [
        (lambda u: bdm_k_viscous_apply(
            Vs[0], u, nu=NU, tau_scale=tau_face)),
    ]
    A_levels = [jax.jacobian(applies[0])(jnp.zeros(Vs[0].total_dofs))]
    for li in range(1, len(Vs)):
        Pl = jax.jacobian(prolongs[li - 1])(jnp.zeros(Vs[li].total_dofs))
        A_levels.append(Pl.T @ A_levels[-1] @ Pl)
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
                     omegas_fixed, n_fine, *, n_rhs=4, n_cycles=6,
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


def element_h(mesh):
    """Per-element representative size: longest edge length (K,)."""
    VX = np.asarray(mesh.VX); VY = np.asarray(mesh.VY)
    EToV = np.asarray(mesh.EToV)
    xs = VX[EToV]; ys = VY[EToV]  # (K, 3)
    e01 = np.hypot(xs[:, 1] - xs[:, 0], ys[:, 1] - ys[:, 0])
    e12 = np.hypot(xs[:, 2] - xs[:, 1], ys[:, 2] - ys[:, 1])
    e20 = np.hypot(xs[:, 0] - xs[:, 2], ys[:, 0] - ys[:, 2])
    return np.maximum(np.maximum(e01, e12), e20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx0", type=int, default=4,
                    help="seed mesh: kx0 x kx0 squares")
    ap.add_argument("--levels", type=int, default=2,
                    help="red-refinement levels inside the region")
    ap.add_argument("--region", type=float, nargs=4,
                    default=[0.0, 0.5, 0.0, 0.5],
                    help="x_lo x_hi y_lo y_hi centroid box to refine")
    ap.add_argument("--tau-init", type=float, default=1.0)
    ap.add_argument("--n-rhs", type=int, default=4)
    ap.add_argument("--n-cycles", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tau-floor", type=float, default=0.3)
    ap.add_argument("--cycle", choices=("V", "W"), default="W")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    gamma = 1 if args.cycle == "V" else 2

    mesh = build_non_uniform_mesh(
        args.kx0, args.order, args.levels, tuple(args.region))
    Vs, prolongs, restricts, patches_idx, EToE = build_static(mesh, args.order)
    n_fine = Vs[0].total_dofs
    K = int(Vs[0].K)
    h = element_h(mesh)
    print(f"BDM_{args.order} non-uniform: seed {args.kx0}x{args.kx0}, "
          f"{args.levels} levels in region {args.region}", flush=True)
    print(f"  K={K}, n_fine={n_fine}, "
          f"h range [{h.min():.4f}, {h.max():.4f}] "
          f"(ratio {h.max()/h.min():.2f}x)", flush=True)

    tau_K = jnp.full(K, args.tau_init)
    applies0, smoothers0, _ = build_diff(
        Vs, prolongs, patches_idx, tau_K, EToE)
    om0 = auto_omegas(applies0, smoothers0, Vs)
    print(f"  init: tau_K={args.tau_init} uniform, "
          f"omegas={[float(x) for x in om0]} (fixed)", flush=True)

    loss_fn = loss_fn_factory(
        Vs, prolongs, restricts, patches_idx, EToE, om0, n_fine,
        n_rhs=args.n_rhs, n_cycles=args.n_cycles, gamma=gamma)
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    L0 = float(loss_jit(tau_K))
    print(f"  initial loss (log-rate): {L0:.4f}  -> rate {float(jnp.exp(L0)):.4f}",
          flush=True)

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
                  f"tau_K [{float(jnp.min(tau_K)):.3f}, "
                  f"{float(jnp.mean(tau_K)):.3f}, "
                  f"{float(jnp.max(tau_K)):.3f}]", flush=True)
    print(f"\ntotal Adam time: {time.time() - t0:.1f}s", flush=True)

    # Final eval.
    applies_f, smoothers_f, lu_f = build_diff(
        Vs, prolongs, patches_idx, tau_K, EToE)
    vcycle_f = make_vcycle(applies_f, smoothers_f, prolongs, restricts,
                            lu_f, gamma=gamma)
    b = jnp.asarray(np.random.default_rng(99).standard_normal(n_fine))
    it_learned = stationary_iters(vcycle_f, applies_f[0], om0, b)
    tau0 = jnp.full(K, args.tau_init)
    ap0, sm0, lu0 = build_diff(Vs, prolongs, patches_idx, tau0, EToE)
    vc0 = make_vcycle(ap0, sm0, prolongs, restricts, lu0, gamma=gamma)
    it_uniform = stationary_iters(vc0, ap0[0], om0, b)
    print(f"\nstationary cycles to rel-resid 1e-9:", flush=True)
    print(f"  uniform tau={args.tau_init}: {it_uniform}", flush=True)
    print(f"  learned per-element tau_K: {it_learned}  "
          f"({it_uniform/max(it_learned,1):.2f}x speedup)", flush=True)

    # Correlation: learned tau_K vs element size h.
    tau_arr = np.asarray(tau_K)
    rho = float(np.corrcoef(tau_arr, h)[0, 1])
    print(f"  Pearson corr(tau_K, h_K): {rho:+.3f}", flush=True)

    # Plots.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(losses, "o-", ms=3)
    axes[0, 0].set_xlabel("Adam iter"); axes[0, 0].set_ylabel("log V-cycle rate")
    axes[0, 0].axhline(L0, color="r", ls="--",
                        label=f"uniform rate {float(jnp.exp(L0)):.4f}")
    axes[0, 0].set_title(f"per-element tau on AMR mesh, BDM_{args.order}")
    axes[0, 0].grid(alpha=0.3); axes[0, 0].legend(fontsize=8)
    axes[0, 1].scatter(h, tau_arr, s=30, alpha=0.7)
    axes[0, 1].axhline(args.tau_init, color="r", ls="--",
                        label=f"uniform {args.tau_init}")
    axes[0, 1].set_xlabel(r"element size $h_K$"); axes[0, 1].set_ylabel(r"learned $\tau_K$")
    axes[0, 1].set_title(f"tau vs h (corr = {rho:+.3f})")
    axes[0, 1].grid(alpha=0.3); axes[0, 1].legend(fontsize=8)
    # Mesh + tau scatter.
    VX = np.asarray(mesh.VX); VY = np.asarray(mesh.VY)
    EToV = np.asarray(mesh.EToV)
    # Plot mesh edges.
    for tri in EToV:
        xs = np.append(VX[tri], VX[tri[0]])
        ys = np.append(VY[tri], VY[tri[0]])
        axes[1, 0].plot(xs, ys, "k-", lw=0.5)
    axes[1, 0].set_aspect("equal"); axes[1, 0].set_title("active mesh")
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    cx = VX[EToV].mean(axis=1); cy = VY[EToV].mean(axis=1)
    sc = axes[1, 1].scatter(cx, cy, c=tau_arr, s=40, cmap="coolwarm",
                              edgecolor="k", linewidth=0.3)
    for tri in EToV:
        xs = np.append(VX[tri], VX[tri[0]])
        ys = np.append(VY[tri], VY[tri[0]])
        axes[1, 1].plot(xs, ys, "k-", lw=0.3, alpha=0.5)
    axes[1, 1].set_aspect("equal"); axes[1, 1].set_title("spatial tau_K")
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    plt.colorbar(sc, ax=axes[1, 1], shrink=0.8)
    fig.tight_layout()
    out = args.out_dir / "learn_per_element_tau_amr.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}", flush=True)
    summary = dict(
        order=args.order, kx0=args.kx0, levels=args.levels,
        region=list(args.region), K=K, n_fine=int(n_fine),
        h_min=float(h.min()), h_max=float(h.max()),
        tau_init=float(args.tau_init), cycle=args.cycle,
        omegas_fixed=[float(x) for x in om0],
        init_rate=float(jnp.exp(L0)),
        learned_rate=float(jnp.exp(losses[-1])),
        learned_tau_K=[float(x) for x in tau_K],
        element_h=[float(x) for x in h],
        corr_tau_h=rho,
        learned_iters_to_1e_9=int(it_learned),
        uniform_iters_to_1e_9=int(it_uniform),
        losses=losses,
    )
    with open(args.out_dir / "learn_per_element_tau_amr.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'learn_per_element_tau_amr.json'}", flush=True)


if __name__ == "__main__":
    main()
