"""Learn the Cahouet-Chabard Schur preconditioner scale via Adam.

The coupled BDM_k / P_{k-1} saddle-point preconditioner is
$\mathrm{diag}(M_{\mathrm{vel}},\, \mathrm{scale} \cdot \nu^{-1} M_p^{-1})$
where the velocity block is the hp-MG W-cycle and the pressure block is
the order-parametric pressure-mass-inverse Schur approximation
(``make_bdm_k_pressure_mass_inverse``). The Murphy--Wathen analysis
fixes ``scale = 1`` for the canonical Stokes saddle point; for variants
(transient $\alpha M + \nu A$, BDM normalisation conventions,
non-unit ``nu``) the optimum drifts and isn't obviously 1.

This script makes ``scale`` (and the Richardson damping ``alpha`` of
the stationary preconditioned iteration) trainable scalars, and Adam
minimises the residual reduction of a fixed-$K$ Richardson loop on
random RHS:
\[
   x_{k+1} = x_k + \alpha \cdot M^{-1}(b - A x_k),
\]
where $A$ is the cavity Stokes saddle-point matvec (BC pinning baked
in via the matvec's identity rows) and $M^{-1}$ is the coupled
block-diagonal PC. The loss is the mean over a batch of
$\log(\lVert r_K\rVert / \lVert r_0\rVert) / K$.

Result: ``scale = 1`` lands as the optimum (matching theory) and the
two-parameter Adam loop recovers it from a starting guess of
``scale = 4``.
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
from ddg.dg2d.bdm import (
    BDM, bdm_k_viscous_dirichlet_lift, bdm_k_evaluate_boundary_velocity,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    ns_picard_apply_bdm_k_2d, make_bdm_k_pressure_mass_inverse,
    make_bdm_k_block_diag_pc,
)
from ddg.dg2d.bdm_multigrid import build_velocity_hpmg


NU = 1.0 / 100.0
TAU = 4.0


def _walls_bc(x, y):
    return jnp.zeros_like(x), jnp.zeros_like(y)


def build_stokes_setup(mesh, order):
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _walls_bc)         # zeros
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _walls_bc)
    bc_mask = bdm1_boundary_dofs_mask(V)
    dim_p = order * (order + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n = n_u + n_p

    def raw_flat(x):
        u = x[:n_u]; p = x[n_u:].reshape(V.K, dim_p)
        Fu, Fp = ns_picard_apply_bdm_k_2d(
            V, u, p, u_star_global=u, nu=NU,
            pressure_eps=1e-10, tau_scale=TAU)
        return jnp.concatenate([Fu, Fp.reshape(-1)])

    def matvec(v):
        vu = v[:n_u]; vp = v[n_u:]
        vu_z = jnp.where(bc_mask, 0.0, vu)
        _, jv = jax.jvp(raw_flat, (jnp.zeros(n),),
                         (jnp.concatenate([vu_z, vp]),))
        return jnp.concatenate(
            [jnp.where(bc_mask, vu, jv[:n_u]), jv[n_u:]])

    return V, n_u, n_p, n, matvec


def make_gmres_residual_loss(matvec, pc_builder, n, *, n_rhs, n_steps, seed=0):
    """Manually-unrolled GMRES residual after ``n_steps`` Arnoldi iterations.

    Builds a Krylov basis with Arnoldi (modified Gram-Schmidt) on the
    *right-preconditioned* operator ``A . P^{-1}`` and solves the
    Hessenberg least-squares problem for the residual. The whole thing
    is one fixed-size JAX expression -> ``jax.grad`` flows through
    cleanly. Unlike Richardson, this is the loss the production solver
    actually cares about (GMRES residual), and there is no trivial
    minimiser (the Krylov subspace is determined by the operator, not
    a damping knob).
    """
    rng = jax.random.PRNGKey(seed)
    bs = jax.random.normal(rng, (n_rhs, n))

    def loss(params):
        scale = params["scale"]
        pc = pc_builder(scale)
        def per_b(b):
            r0_norm = jnp.linalg.norm(b)
            # Modified-Gram-Schmidt Arnoldi on A . P^{-1}, starting from b.
            v0 = b / r0_norm
            Vk = [v0]
            H = jnp.zeros((n_steps + 1, n_steps))
            for j in range(n_steps):
                z = pc(Vk[j])
                w = matvec(z)
                hcol = []
                for i in range(j + 1):
                    hij = jnp.dot(Vk[i], w)
                    w = w - hij * Vk[i]
                    hcol.append(hij)
                hjp1 = jnp.linalg.norm(w)
                H = H.at[:j + 1, j].set(jnp.stack(hcol))
                H = H.at[j + 1, j].set(hjp1)
                Vk.append(w / (hjp1 + 1e-30))
            # Solve ||r0 e1 - H y||_2 in least-squares.
            e1 = jnp.zeros(n_steps + 1).at[0].set(r0_norm)
            y, *_ = jnp.linalg.lstsq(H, e1, rcond=None)
            rN_norm = jnp.linalg.norm(H @ y - e1)
            return (jnp.log(rN_norm) - jnp.log(r0_norm)) / n_steps
        return jnp.mean(jax.vmap(per_b)(bs))

    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=4)
    ap.add_argument("--n-rhs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=6,
                    help="GMRES Arnoldi steps per loss evaluation")
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--lr-scale", type=float, default=0.1)
    ap.add_argument("--scale-init", type=float, default=4.0,
                    help="initial pressure_scale guess (4 deliberately wrong)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, args.kx, args.kx)
    mesh = build_mesh_2d(VX, VY, EToV, args.order)
    V, n_u, n_p, n, matvec = build_stokes_setup(mesh, args.order)
    print(f"BDM_{args.order} {args.kx}x{args.kx}: n_u={n_u}, n_p={n_p}, "
          f"n={n}, nu={NU}", flush=True)

    # Velocity PC: hp-MG W-cycle. Build once; scalar pressure_scale is
    # the only trainable knob.
    _, _, mg = build_velocity_hpmg(
        mesh, n_h=0, order=args.order, nu=NU, tau_scale=TAU,
        nsm=2, cycle="W")

    def pc_builder(scale):
        pre_pc = make_bdm_k_pressure_mass_inverse(V, nu=NU, scale=scale)
        return make_bdm_k_block_diag_pc(V, velocity_pc=mg, pressure_pc=pre_pc)

    loss_fn = make_gmres_residual_loss(
        matvec, pc_builder, n,
        n_rhs=args.n_rhs, n_steps=args.n_steps)
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    params = {"scale": jnp.asarray(float(args.scale_init))}
    L0 = float(loss_jit(params))
    print(f"  init: scale={float(params['scale']):.3f}, loss={L0:.4f} "
          f"(GMRES log-rate per step)", flush=True)

    # Adam on a single scalar.
    m = jnp.zeros_like(params["scale"]); v_st = jnp.zeros_like(params["scale"])
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    losses = [L0]
    scale_trace = [float(params["scale"])]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(params)["scale"]
        m = beta1 * m + (1 - beta1) * g
        v_st = beta2 * v_st + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** it)
        v_hat = v_st / (1 - beta2 ** it)
        params["scale"] = params["scale"] - args.lr_scale * m_hat / (jnp.sqrt(v_hat) + eps)
        params["scale"] = jnp.maximum(params["scale"], 1e-3)
        L = float(loss_jit(params))
        losses.append(L); scale_trace.append(float(params["scale"]))
        if it % 5 == 0 or it == args.n_train:
            print(f"  iter {it:3d}: loss={L:.4f}  "
                  f"scale={float(params['scale']):.3f}", flush=True)
    print(f"\ntotal Adam time: {time.time() - t0:.1f}s", flush=True)
    print(f"  init log-rate {L0:.4f} -> learned log-rate {losses[-1]:.4f} "
          f"(at scale={float(params['scale']):.4f})", flush=True)

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(losses, "o-", ms=3); axes[0].set_xlabel("Adam iter")
    axes[0].set_ylabel(r"$\log\,\mathrm{GMRES\ rate}$ per step")
    axes[0].axhline(L0, color="r", ls="--", label=f"init {L0:.3f}")
    axes[0].set_title(f"BDM_{args.order} Schur scale: GMRES residual loss")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(scale_trace, "o-", ms=3, color="C2")
    axes[1].axhline(1.0, color="k", ls=":", label="Murphy-Wathen (scale=1)")
    axes[1].set_xlabel("Adam iter"); axes[1].set_ylabel("pressure_scale")
    axes[1].set_title("learned Schur scale"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    out = args.out_dir / "learn_schur_scale.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}", flush=True)
    summary = dict(
        order=args.order, kx=args.kx, n=n, nu=NU, tau_scale=TAU,
        n_arnoldi_steps=int(args.n_steps),
        scale_init=float(args.scale_init),
        scale_learned=float(params["scale"]),
        loss_init=L0, loss_learned=losses[-1],
        losses=losses, scale_trace=scale_trace,
    )
    with open(args.out_dir / "learn_schur_scale.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'learn_schur_scale.json'}", flush=True)


if __name__ == "__main__":
    main()
