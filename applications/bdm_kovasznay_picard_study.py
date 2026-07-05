"""Picard convergence study for BDM_k Kovasznay.

Now that the SIPG J-factor fix is in place, characterise the
behaviour of fixed-point Picard iteration on Kovasznay as a function
of Re. Tests three strategies for stabilising Picard:

  1. Plain (alpha=1.0): no relaxation. Diverges above some critical
     Re because the corrected discrete operator's Picard map has
     spectral radius > 1 near the analytic IC.
  2. Damped: u_new = alpha * u_picard + (1 - alpha) * u_old, with
     alpha < 1 to enforce contraction.
  3. Low-Re continuation: solve a sequence Re_0 < Re_1 < ... < Re_max
     using the previous solution as the next IC. Each step is a
     small Re step so Picard contracts even from a slightly-off
     starting point.

Output:
    output/bdm_kovasznay_picard_study.png
    output/bdm_kovasznay_picard_study.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM, bdm_k_basis_at_points, _element_jacobian_matrices,
    bdm_k_face_dof_coordinates, _bdm_k_face_quadrature_data,
    _bdm_k_interior_test_functions, _eval_interior_test,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    bdm1_gather, ns_picard_apply_bdm_k_2d,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


DOMAIN_X = (-0.5, 1.5)
DOMAIN_Y = (-0.5, 1.5)
DOMAIN_LENGTH = DOMAIN_X[1] - DOMAIN_X[0]


def kovasznay_velocity_at_Re(Re):
    nu = 1.0 / Re
    lam = Re / 2.0 - np.sqrt(Re ** 2 / 4.0 + 4.0 * np.pi ** 2)
    def fn(x, y):
        ux = 1.0 - jnp.exp(lam * x) * jnp.cos(2.0 * jnp.pi * y)
        uy = (lam / (2.0 * jnp.pi)) * jnp.exp(lam * x) * jnp.sin(2.0 * jnp.pi * y)
        return ux, uy
    return fn


def bdm_k_canonical_interpolant(V, u_fn):
    """BDM_k canonical interpolant. See bdm_kovasznay_spatial_convergence."""
    k = V.order
    K = V.K
    Q = k + 1
    n_face_total = 3 * Q
    n_interior = (k + 1) * (k - 1)
    fdata = _bdm_k_face_quadrature_data(V)
    n_face = fdata['n_face']
    coords = bdm_k_face_dof_coordinates(k)
    r = np.asarray(coords[:, 0]); s = np.asarray(coords[:, 1])
    EToV = np.asarray(V.mesh.EToV); VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    x_at = Vx0[:, None] + 0.5 * (r + 1.0)[None, :] * (Vx1 - Vx0)[:, None] \
           + 0.5 * (s + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
    y_at = Vy0[:, None] + 0.5 * (r + 1.0)[None, :] * (Vy1 - Vy0)[:, None] \
           + 0.5 * (s + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
    x_face = jnp.asarray(x_at).reshape(K, 3, Q)
    y_face = jnp.asarray(y_at).reshape(K, 3, Q)
    ux, uy = u_fn(x_face, y_face)
    un = ux * n_face[:, :, 0:1] + uy * n_face[:, :, 1:2]
    u_local_face = un.reshape(K, n_face_total)
    u_local_interior = jnp.zeros((K, n_interior))
    if n_interior > 0:
        interior_tests = _bdm_k_interior_test_functions(k)
        r_q, s_q, w_q = triangle_cubature(2 * k + 2)
        DF, _ = _element_jacobian_matrices(V.mesh)
        DF_inv = jnp.linalg.inv(DF)
        J = V.mesh.J[0]
        rqn = np.asarray(r_q); sqn = np.asarray(s_q)
        x_q = Vx0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None] \
              + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
        y_q = Vy0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None] \
              + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
        ux_q, uy_q = u_fn(jnp.asarray(x_q), jnp.asarray(y_q))
        u_phys_q = jnp.stack([ux_q, uy_q], axis=-1)
        u_ref_q = jnp.einsum('k,kab,kqb->kqa', J, DF_inv, u_phys_q)
        for i, w_fn in enumerate(interior_tests):
            w_val = _eval_interior_test(w_fn, r_q, s_q)
            mom = jnp.einsum('q,kqc,qc->k', w_q, u_ref_q, w_val)
            u_local_interior = u_local_interior.at[:, i].set(mom)
    u_local = jnp.concatenate([u_local_face, u_local_interior], axis=1)
    sign = V.topology.local_sign
    owner_mask = sign > 0
    lg = V.topology.local_to_global
    u_global = jnp.zeros(V.total_dofs)
    u_global = u_global.at[lg.reshape(-1)].add(
        jnp.where(owner_mask, u_local, 0).reshape(-1)
    )
    return u_global


def picard_kovasznay(
    V, *, Re, alpha=1.0, u_init=None, u_bc_global=None,
    max_iters=80, tol=1e-10,
):
    """Damped Picard fixed-point iteration on steady NS.

    Iteration: solve (A_h + C(u_k)) u_picard + G p = -lift_terms,
    then u_{k+1} = alpha * u_picard + (1 - alpha) * u_k.

    Uses non-homog ghost in both viscous and convective (no separate
    lift on the residual; bilinear form is self-consistent).
    BC pinning: overwrite boundary DOFs with u_bc each step.

    Returns ``(u_sol, p_sol, history)`` -- history is the per-step
    Picard residual ||u_{k+1} - u_k||.
    """
    nu = 1.0 / Re
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    u_bc = bdm1_evaluate_dirichlet_bc(V, kovasznay_velocity_at_Re(Re))
    bc_mask = bdm1_boundary_dofs_mask(V)
    if u_init is None:
        u_init = bdm_k_canonical_interpolant(V, kovasznay_velocity_at_Re(Re))
    if u_bc_global is None:
        u_bc_global = u_init   # use IC's full BDM rep for non-homog ghost data

    n = n_u + n_p
    bc_mask_full = jnp.concatenate(
        [bc_mask, jnp.zeros(n_p, dtype=jnp.bool_)]
    )

    def apply_for_star(u_star, zf):
        """F(z; u_star) = (A_h + C(u_star)) u + G p + lift_const(u_bc)."""
        uz = zf[:n_u]
        pz = zf[n_u:].reshape(V.K, dim_p)
        Fuz, Fpz = ns_picard_apply_bdm_k_2d(
            V, uz, pz, u_star_global=u_star, nu=nu,
            pressure_eps=1e-10, tau_scale=4.0,
            u_bc_global=u_bc_global,
        )
        return jnp.concatenate([Fuz, Fpz.reshape(-1)])

    def assemble_picard_system(u_star):
        """Returns (A_linear, F0) where F(z; u_star) = A_linear z + F0."""
        F0 = apply_for_star(u_star, jnp.zeros(n))
        A_linear = jax.jacfwd(lambda zf: apply_for_star(u_star, zf))(jnp.zeros(n))
        return A_linear, F0

    u_k = u_init
    p_k = jnp.zeros((V.K, dim_p))
    history = []
    for it in range(max_iters):
        A_lin, F0 = assemble_picard_system(u_k)
        # Solve A_lin z + F0 = 0 with strong BC on velocity bdry: z_bdry = u_bc.
        A_mod = jnp.where(bc_mask_full[:, None], jnp.eye(n), A_lin)
        rhs = -F0
        rhs = rhs.at[:n_u].set(jnp.where(bc_mask, u_bc, rhs[:n_u]))
        sol = jnp.linalg.solve(A_mod, rhs)
        u_picard = sol[:n_u]
        p_picard = sol[n_u:].reshape(V.K, dim_p)
        # Damping.
        u_new = alpha * u_picard + (1 - alpha) * u_k
        p_new = alpha * p_picard + (1 - alpha) * p_k
        u_new = jnp.where(bc_mask, u_bc, u_new)   # strong BC pinning
        delta = float(jnp.linalg.norm(u_new - u_k))
        history.append(delta)
        if delta < tol:
            u_k, p_k = u_new, p_new
            break
        u_k, p_k = u_new, p_new
    return u_k, p_k, history


def l2_velocity_error(V, u_global, Re):
    u_fn = kovasznay_velocity_at_Re(Re)
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 4)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum('kca,qla->kqlc', DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    uh = jnp.einsum('kl,kqlc->kqc', u_local, phi_phys)
    EToV = np.asarray(V.mesh.EToV); VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    rqn = np.asarray(r_q); sqn = np.asarray(s_q)
    x_q = Vx0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None] \
          + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
    y_q = Vy0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None] \
          + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
    ux, uy = u_fn(jnp.asarray(x_q), jnp.asarray(y_q))
    dx = uh[..., 0] - ux; dy = uh[..., 1] - uy
    return float(jnp.sqrt(jnp.einsum('q,k,kq->', w_q, J, dx ** 2 + dy ** 2)))


def build_mesh(n):
    VX, VY, EToV = uniform_rectangle_mesh_2d(
        DOMAIN_X[0], DOMAIN_X[1], DOMAIN_Y[0], DOMAIN_Y[1], n, n,
    )
    return build_mesh_2d(VX, VY, EToV, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re-sweep", type=str, default="5,10,20,40")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--alphas", type=str, default="1.0,0.5,0.2,0.1")
    ap.add_argument("--rate-Re", type=float, default=10.0,
                     help="Re at which to do the Picard rate plot")
    ap.add_argument("--rate-alpha", type=float, default=0.5,
                     help="Damping alpha for the rate plot")
    ap.add_argument("--rate-ns", type=str, default="4,6,8,12,16")
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    re_values = [float(s) for s in args.re_sweep.split(",")]
    alphas = [float(s) for s in args.alphas.split(",")]

    mesh = build_mesh(args.n)
    V = BDM.from_mesh(mesh, order=args.k)
    print(f"BDM_{args.k}, n={args.n}, total_dofs={V.total_dofs}", flush=True)

    summary = {
        "k": args.k, "n": args.n, "Re_sweep": re_values, "alphas": alphas,
        "results": {},
    }

    # Part 1: plain Picard (alpha=1) across Re ----------------------------------
    print("\n=== Plain Picard (alpha=1.0), IC = analytic interpolant ===", flush=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for Re in re_values:
        t0 = time.time()
        _, _, hist = picard_kovasznay(V, Re=Re, alpha=1.0, max_iters=60, tol=1e-10)
        wall = time.time() - t0
        converged = hist[-1] < 1e-9
        print(f"  Re={Re:.0f}: iters={len(hist)}, final={hist[-1]:.2e}, "
              f"converged={converged}, {wall:.1f}s", flush=True)
        summary["results"][f"plain_Re{int(Re)}"] = {
            "history": hist, "converged": bool(converged), "wall_s": wall,
        }
        ax.semilogy(range(1, len(hist) + 1), hist, "o-",
                    label=f"Re={Re:.0f} ({'conv' if converged else 'div'})")
    ax.set_xlabel("Picard iteration")
    ax.set_ylabel(r"$\|u_{k+1} - u_k\|$")
    ax.set_title(f"Plain Picard (alpha=1.0) on BDM_{args.k} Kovasznay, n={args.n}")
    ax.grid(which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "bdm_kovasznay_picard_plain.png", dpi=140)
    plt.close(fig)
    print(f"  wrote bdm_kovasznay_picard_plain.png", flush=True)

    # Part 2: damped Picard at Re=40 across alpha -------------------------------
    Re = re_values[-1]
    print(f"\n=== Damped Picard at Re={Re:.0f}, alpha sweep ===", flush=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for alpha in alphas:
        t0 = time.time()
        u_sol, _, hist = picard_kovasznay(V, Re=Re, alpha=alpha, max_iters=200, tol=1e-10)
        wall = time.time() - t0
        converged = hist[-1] < 1e-9
        err = l2_velocity_error(V, u_sol, Re) if converged else float("nan")
        print(f"  alpha={alpha}: iters={len(hist)}, final={hist[-1]:.2e}, "
              f"L2 err = {err:.3e}, conv={converged}, {wall:.1f}s", flush=True)
        summary["results"][f"damped_alpha{alpha}"] = {
            "alpha": alpha, "Re": Re, "history": hist,
            "converged": bool(converged), "L2_err": float(err) if converged else None,
            "wall_s": wall,
        }
        ax.semilogy(range(1, len(hist) + 1), hist, "o-",
                    label=f"alpha={alpha} ({'conv' if converged else 'div'})")
    ax.set_xlabel("Picard iteration")
    ax.set_ylabel(r"$\|u_{k+1} - u_k\|$")
    ax.set_title(f"Damped Picard on BDM_{args.k} Kovasznay Re={Re:.0f}, n={args.n}")
    ax.grid(which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "bdm_kovasznay_picard_damped.png", dpi=140)
    plt.close(fig)
    print(f"  wrote bdm_kovasznay_picard_damped.png", flush=True)

    # Part 3: continuation -- smaller Re steps + damping at higher Re ---------
    print(f"\n=== Continuation: smaller Re steps + alpha schedule ===", flush=True)
    cont_re = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    cont_alphas = [1.0, 1.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.25]
    u_curr = None
    fig, ax = plt.subplots(figsize=(8, 5))
    total_iters = 0
    for Re, alpha_step in zip(cont_re, cont_alphas):
        u_bc_global_full = bdm_k_canonical_interpolant(V, kovasznay_velocity_at_Re(Re))
        u_init = u_bc_global_full if u_curr is None else u_curr
        u_sol, _, hist = picard_kovasznay(
            V, Re=Re, alpha=alpha_step, u_init=u_init,
            u_bc_global=u_bc_global_full,
            max_iters=300, tol=1e-9,
        )
        converged = hist[-1] < 1e-8
        err = l2_velocity_error(V, u_sol, Re) if converged else float("nan")
        print(f"  Re={Re:.0f} alpha={alpha_step}: iters={len(hist)}, "
              f"final={hist[-1]:.2e}, L2 err = {err:.3e}, "
              f"conv={converged}", flush=True)
        ax.semilogy(range(total_iters + 1, total_iters + 1 + len(hist)),
                    hist, "-", label=f"Re={Re:.0f} alpha={alpha_step}")
        total_iters += len(hist)
        u_curr = u_sol
        summary["results"][f"continuation_Re{int(Re)}"] = {
            "Re": Re, "alpha": alpha_step, "history": hist,
            "converged": bool(converged),
            "L2_err": float(err) if converged else None,
        }
    ax.set_xlabel("Cumulative Picard iteration")
    ax.set_ylabel(r"$\|u_{k+1} - u_k\|$")
    ax.set_title(f"Picard continuation 1->40, BDM_{args.k}, n={args.n}")
    ax.grid(which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "bdm_kovasznay_picard_continuation.png", dpi=140)
    plt.close(fig)
    print(f"  wrote bdm_kovasznay_picard_continuation.png", flush=True)

    # Part 4: Picard rate plot at a fixed Re where it converges ---------------
    print(f"\n=== Picard rate study at Re={args.rate_Re:.0f}, "
          f"alpha={args.rate_alpha} ===", flush=True)
    rate_ns = [int(s) for s in args.rate_ns.split(",")]
    fig, ax = plt.subplots(figsize=(7, 5))
    rate_hs, rate_errs = [], []
    for n in rate_ns:
        m = build_mesh(n)
        Vn = BDM.from_mesh(m, order=args.k)
        u_sol, _, hist = picard_kovasznay(
            Vn, Re=args.rate_Re, alpha=args.rate_alpha,
            max_iters=300, tol=1e-10,
        )
        converged = hist[-1] < 1e-9
        if converged:
            err = l2_velocity_error(Vn, u_sol, args.rate_Re)
            h = DOMAIN_LENGTH / n
            rate_hs.append(h); rate_errs.append(err)
            print(f"  n={n}: iters={len(hist)}, final={hist[-1]:.2e}, "
                  f"L2 err = {err:.3e}", flush=True)
        else:
            print(f"  n={n}: DID NOT CONVERGE in {len(hist)} iters "
                  f"(final={hist[-1]:.2e})", flush=True)
    if len(rate_hs) >= 2:
        rate_hs = np.array(rate_hs); rate_errs = np.array(rate_errs)
        slopes = (np.log(rate_errs[:-1]) - np.log(rate_errs[1:])) / \
                 (np.log(rate_hs[:-1]) - np.log(rate_hs[1:]))
        for i, sl in enumerate(slopes):
            print(f"  slope({rate_hs[i]:.4f} -> {rate_hs[i+1]:.4f}) = {sl:.3f}",
                  flush=True)
        ax.loglog(rate_hs, rate_errs, "o-",
                  label=f"BDM_{args.k} (asymp slope {slopes[-1]:.2f}, theory {args.k+1})")
        ref = rate_errs[0] * (rate_hs / rate_hs[0]) ** (args.k + 1)
        ax.loglog(rate_hs, ref, "--", alpha=0.4,
                   label=f"slope {args.k+1} reference")
        ax.set_xlabel("h")
        ax.set_ylabel(r"$\|u_h - u_{\rm Kov}\|_{L^2}$")
        ax.set_title(f"Picard rate, BDM_{args.k} Kovasznay Re={args.rate_Re:.0f}, "
                      f"damping alpha={args.rate_alpha}")
        ax.grid(which="both", alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / "bdm_kovasznay_picard_rate.png", dpi=140)
        plt.close(fig)
        print(f"  wrote bdm_kovasznay_picard_rate.png", flush=True)
        summary["results"]["rate_study"] = {
            "Re": args.rate_Re, "alpha": args.rate_alpha,
            "ns": rate_ns, "hs": rate_hs.tolist(),
            "errs": rate_errs.tolist(),
            "slopes": slopes.tolist(),
            "asymptotic_slope": float(slopes[-1]),
            "expected_slope": args.k + 1,
        }

    json_path = args.out_dir / "bdm_kovasznay_picard_study.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
