"""Steady-state drag-minimising shape optimisation through the
differentiable curved mesh + monolithic BDM Navier-Stokes solver.

Compared with the earlier transient demo, this version is built to be
physically correct and to flatline:

  * Reynolds number Re = U_mean * D / nu = 20 (stable, well-defined
    steady wake; no vortex shedding), so the time-marched solution
    converges to a genuine steady state with a positive, O(1) drag
    coefficient (Schaefer-Turek 2D-1 reference C_D ~ 5.58).
  * The inflow is imposed by pinning the boundary velocity DOFs to the
    parabolic profile (u_n initialised to u_bc); the Newton increment
    leaves those DOFs unchanged, so the channel is actually driven.
  * The drag is read from the developed (final) field after marching to
    steady state, not time-averaged over an undeveloped transient.

The cylinder boundary is r(theta) = R0 (1 + a cos 2 theta); a > 0
streamlines the body along the flow (lower drag), a < 0 widens it
across the flow (more bluff, higher drag). One scalar design variable
exercises the full reverse-mode path: shape parameter -> Fourier curve
-> curved mesh metrics -> BDM forward solve -> surface-integral C_D.
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

from inverse_nu_cylinder_2d import make_step_nu_param
from inverse_obstacle_cy_diff_2d import build_diff_setup, CX, RADIUS, D_CYL, U_MEAN, _bc_fn
from shape_opt_drag_cylinder_2d import _make_drag_eval
from ddg.dg2d.bdm import (
    BDM, bdm_k_evaluate_boundary_velocity, refresh_piola_scale,
    bdm_k_viscous_dirichlet_lift,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_evaluate_dirichlet_bc, BDF_COEFFS, bdm_k_mass_apply,
    ns_picard_apply_bdm_k_2d, ns_picard_matrix_bdm_k_2d,
)
from ddg.dg2d.differentiable_mesh import (
    differentiable_mesh_with_shape, differentiable_mesh_with_circle,
)


def make_picard_step(V, u_bc, u_bc_vec, bc_mask, is_natural_face, *,
                     nu, tau_scale):
    """One BDF Picard (Oseen) step: a single dense saddle solve per
    step, NO dense Newton-Jacobian column build. The Picard
    linearisation converges to the same steady fixed point as Newton at
    this Reynolds number, and its reverse-mode graph is just the
    linear-solve adjoint, so the unrolled time march is cheap to
    differentiate (the dense vmap(eye) Newton term is what makes the
    monolithic-Newton tape explode)."""
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n_full = n_u + n_p
    lift_u = bdm_k_viscous_dirichlet_lift(
        V, u_bc_vec, nu=nu, tau_scale=tau_scale, is_natural_face=is_natural_face)
    mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])

    def step(u_n, u_nm1, dt, order_eff: int):
        coeffs = BDF_COEFFS[order_eff]
        alpha = coeffs["gamma0"] / dt
        f_velocity = (coeffs["beta"][0] / dt) * bdm_k_mass_apply(V, u_n)
        if order_eff == 2:
            f_velocity = f_velocity + (coeffs["beta"][1] / dt) * bdm_k_mass_apply(V, u_nm1)
        F_u, F_p = ns_picard_apply_bdm_k_2d(
            V, u_n, jnp.zeros((V.K, dim_p)), u_star_global=u_n, nu=nu,
            pressure_eps=1e-10, tau_scale=tau_scale, velocity_alpha=alpha,
            f_velocity=f_velocity, u_bc_global=u_bc, is_natural_face=is_natural_face)
        F_u = F_u - lift_u
        F_u_inner = jnp.where(bc_mask, 0.0, F_u)
        rhs = jnp.concatenate([-F_u_inner, -F_p.reshape(-1)])
        A = ns_picard_matrix_bdm_k_2d(
            V, u_star_global=u_n, nu=nu, pressure_eps=1e-10, tau_scale=tau_scale,
            velocity_alpha=alpha, is_natural_face=is_natural_face)
        I_n = jnp.eye(n_full)
        A_c = jnp.where(mask_full[:, None], I_n,
                        jnp.where(mask_full[None, :], 0.0, A))
        delta = jnp.linalg.solve(A_c, rhs)
        du = jnp.where(bc_mask, 0.0, delta[:n_u])
        u_new = u_n + du
        p_new = delta[n_u:].reshape(V.K, dim_p)
        return u_new, p_new
    return step


def _cyl_face_mask(mesh, cy):
    Nf = 3
    Fmask = np.asarray(mesh.Fmask)
    x = np.asarray(mesh.x); y = np.asarray(mesh.y)
    xf = np.zeros((Nf, mesh.K)); yf = np.zeros((Nf, mesh.K))
    for f in range(Nf):
        xf[f] = x[Fmask[:, f], :].mean(0); yf[f] = y[Fmask[:, f], :].mean(0)
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(mesh.K)[:, None]).T
    is_in = is_bdy & (xf < 1e-3); is_out = is_bdy & (xf > 1 - 1e-3)
    is_oy = is_bdy & ((yf < 1e-3) | (yf > 0.5 - 1e-3))
    rc = np.sqrt((xf - CX) ** 2 + (yf - cy) ** 2)
    is_cyl = (is_bdy & (rc < RADIUS * 1.05) & (~is_in) & (~is_out) & (~is_oy)).T
    return is_cyl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--cy", type=float, default=0.25)
    ap.add_argument("--re", type=float, default=20.0)
    ap.add_argument("--a-init", type=float, default=0.0)
    ap.add_argument("--a-max", type=float, default=0.2,
                    help="bound on the shape amplitude: a = a_max*tanh(theta), "
                         "keeps the optimiser inside the valid-mesh region")
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=20.0,
                    help="clip |dC_D/da| to this; guards against the "
                         "occasional spurious huge gradient from a "
                         "marginally-converged steady solve at large a")
    ap.add_argument("--tol", type=float, default=1.0e-4,
                    help="relative-C_D-change flatline threshold (early stop)")
    ap.add_argument("--patience", type=int, default=6,
                    help="consecutive sub-tol iters before declaring flatline")
    ap.add_argument("--forward-only", action="store_true",
                    help="just print the C_D trajectory at a-init and exit")
    ap.add_argument("--grad-check", action="store_true",
                    help="compare AD dC_D/da to a central finite difference and exit")
    ap.add_argument("--fd-h", type=float, default=0.02)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    NU = U_MEAN * D_CYL / args.re
    print(f"BDM_{args.order} steady shape-opt: Re={args.re:.0f} (nu={NU:.3e}), "
          f"n_steps={args.n_steps}, dt={args.dt}, n_newton={args.n_newton}",
          flush=True)

    template, curving, cached_topology, bc_mask, is_natural_face = \
        build_diff_setup(args.kx, args.ky, args.order, args.cy)
    nominal = differentiable_mesh_with_circle(
        template.VX_nominal, template.VY_nominal, template, curving,
        jnp.asarray(CX), jnp.asarray(args.cy), jnp.asarray(RADIUS))
    is_cyl = _cyl_face_mask(nominal, args.cy)
    n_steps = int(args.n_steps)

    def steady_cd(a):
        coeffs = jnp.array([0.0, 0.0, a, 0.0])      # cos(2 theta) mode
        mesh = differentiable_mesh_with_shape(
            template.VX_nominal, template.VY_nominal, template, curving,
            jnp.asarray(CX), jnp.asarray(args.cy), jnp.asarray(RADIUS), coeffs)
        topo = refresh_piola_scale(cached_topology, mesh)
        V = BDM(mesh=mesh, topology=topo, order=args.order)
        u_bc = bdm1_evaluate_dirichlet_bc(V, _bc_fn)
        u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _bc_fn)
        step = make_step_nu_param(
            V, u_bc, u_bc_vec, bc_mask, is_natural_face,
            tau_scale=args.tau_scale, n_newton=args.n_newton)
        drag = _make_drag_eval(V, is_cyl)
        # Strong inlet: start from u_bc so the masked boundary DOFs hold
        # the parabolic profile (the Newton increment leaves them fixed).
        # March to steady state. The step graph (with its dense Newton
        # column build) is compiled ONCE via lax.scan rather than unrolled
        # n_steps times -- a Python loop would explode the reverse-mode
        # compile; jax.checkpoint bounds the backward memory.
        u1, _ = step(u_bc, u_bc, args.dt, NU, 1)               # BDF1 first step

        def body(carry, _):
            u_n, u_nm1 = carry
            u_new, p_new = step(u_n, u_nm1, args.dt, NU, 2)    # BDF2 thereafter
            return (u_new, u_n), p_new

        (u_final, _), ps = jax.lax.scan(
            jax.checkpoint(body), (u1, u_bc), None, length=n_steps - 1)
        return drag(u_final, ps[-1], NU)

    if args.forward_only:
        # Forward trajectory at a-init: confirm C_D flatlines to a
        # positive O(1) value before any optimisation.
        a = jnp.asarray(float(args.a_init))
        coeffs = jnp.array([0.0, 0.0, a, 0.0])
        mesh = differentiable_mesh_with_shape(
            template.VX_nominal, template.VY_nominal, template, curving,
            jnp.asarray(CX), jnp.asarray(args.cy), jnp.asarray(RADIUS), coeffs)
        topo = refresh_piola_scale(cached_topology, mesh)
        V = BDM(mesh=mesh, topology=topo, order=args.order)
        u_bc = bdm1_evaluate_dirichlet_bc(V, _bc_fn)
        u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _bc_fn)
        step = jax.jit(make_step_nu_param(
            V, u_bc, u_bc_vec, bc_mask, is_natural_face,
            tau_scale=args.tau_scale, n_newton=args.n_newton),
            static_argnums=(4,))
        drag = _make_drag_eval(V, is_cyl)
        u_n = u_bc; u_nm1 = u_bc
        prev = None
        for s in range(n_steps):
            order_eff = 1 if s == 0 else 2
            u_new, p_new = step(u_n, u_nm1, args.dt, NU, order_eff)
            u_nm1, u_n = u_n, u_new
            if (s + 1) % 5 == 0 or s == 0:
                cd = float(drag(u_n, p_new, NU))
                d = "" if prev is None else f"  dC_D={cd-prev:+.2e}"
                print(f"  step {s+1:3d}  t={args.dt*(s+1):5.2f}  C_D={cd:+.5f}{d}",
                      flush=True)
                prev = cd
        print(f"FINAL a={float(a):+.3f}  C_D={float(drag(u_n,p_new,NU)):+.5f}",
              flush=True)
        return

    if args.grad_check:
        a0 = float(args.a_init); h = args.fd_h
        print(f"grad check at a={a0:+.3f}, h={h}", flush=True)
        g_ad = float(jax.jit(jax.grad(steady_cd))(jnp.asarray(a0)))
        cp = float(jax.jit(steady_cd)(jnp.asarray(a0 + h)))
        cm = float(jax.jit(steady_cd)(jnp.asarray(a0 - h)))
        g_fd = (cp - cm) / (2 * h)
        print(f"  C_D(a+h)={cp:+.6f}  C_D(a-h)={cm:+.6f}", flush=True)
        print(f"  dC_D/da: AD={g_ad:+.5e}  FD={g_fd:+.5e}  "
              f"rel.err={abs(g_ad-g_fd)/(abs(g_fd)+1e-30):.2e}", flush=True)
        return

    # Projected Adam on the shape amplitude a, box-constrained to
    # [-a_max, a_max]. The bound is a DESIGN-SPACE constraint that keeps
    # the body-fitted curved mesh valid: beyond a ~ 0.16 the Gordon-Hall
    # curving tangles elements near the wall (J -> 0), making the geometric
    # factors singular and the gradient explode. Inside the bound the
    # objective is smooth, so projected gradient descends cleanly to a
    # flatline. Hard projection (vs a tanh squash) reaches the bound
    # directly without slow saturation.
    grad_fn = jax.jit(jax.grad(steady_cd))
    loss_jit = jax.jit(steady_cd)

    stem = f"shape_opt_drag_steady_k{args.order}_re{int(args.re)}"
    out_json = args.out_dir / f"{stem}.json"

    def save(history, complete):
        """Atomic, every-iteration checkpoint: write .tmp then os.replace
        so an interrupted-but-successful run is always recoverable."""
        payload = dict(order=args.order, kx=args.kx, ky=args.ky, re=args.re,
                       nu=NU, n_steps=args.n_steps, dt=args.dt,
                       n_newton=args.n_newton, tau_scale=args.tau_scale,
                       a_init=args.a_init, a_max=args.a_max, lr=args.lr,
                       n_train=args.n_train, tol=args.tol, patience=args.patience,
                       complete=complete, history=history)
        tmp = args.out_dir / f"{stem}.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(out_json)                                  # atomic

    a = jnp.clip(jnp.asarray(float(args.a_init)), -args.a_max, args.a_max)
    print("compiling forward + reverse ...", flush=True)
    t0 = time.time()
    L0 = float(loss_jit(a))
    print(f"  initial C_D = {L0:.5f}  (a={float(a):+.3f}, a_max={args.a_max}, "
          f"compile {time.time()-t0:.1f}s)", flush=True)
    if not np.isfinite(L0):
        print("  -> non-finite initial C_D (CFL/blowup?); aborting", flush=True)
        return

    m_t = jnp.zeros_like(a); v_t = jnp.zeros_like(a)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history = [dict(iter=0, a=float(a), cd=L0, grad=0.0)]
    save(history, complete=False)                              # save AS IT GOES
    t0 = time.time(); L_prev = L0; n_flat = 0
    for it in range(1, args.n_train + 1):
        g_raw = grad_fn(a)
        g = jnp.clip(g_raw, -args.grad_clip, args.grad_clip)
        clipped = "" if g == g_raw else f"  (clipped from {float(g_raw):+.3e})"
        m_t = beta1 * m_t + (1 - beta1) * g
        v_t = beta2 * v_t + (1 - beta2) * g * g
        m_hat = m_t / (1 - beta1 ** it)
        v_hat = v_t / (1 - beta2 ** it)
        a = a - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        a = jnp.clip(a, -args.a_max, args.a_max)               # project to box
        L = float(loss_jit(a))
        rel = abs(L_prev - L) / max(abs(L_prev), 1e-30)
        print(f"  iter {it:3d}: a={float(a):+.4f}  C_D={L:.5f}  "
              f"g={float(g):+.4e}  rel_dCd={rel:.2e}{clipped}", flush=True)
        history.append(dict(iter=int(it), a=float(a), cd=L,
                            grad=float(g), grad_raw=float(g_raw), rel=rel))
        save(history, complete=False)                          # atomic, each iter
        if not np.isfinite(L):
            print("  -> non-finite C_D; aborting (state saved)", flush=True)
            return
        n_flat = n_flat + 1 if rel < args.tol else 0
        if n_flat >= args.patience:
            print(f"  flatlined: rel_dCd < {args.tol:.1e} for {args.patience} "
                  f"iters -> stop at iter {it}", flush=True)
            break
        L_prev = L
    print(f"\ntotal Adam wall: {time.time()-t0:.1f}s", flush=True)
    save(history, complete=True)
    print(f"  wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
