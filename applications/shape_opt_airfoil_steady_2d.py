"""Steady-state drag-minimising AIRFOIL shape optimisation through a
differentiable Joukowski conformal map + monolithic BDM Navier-Stokes.

This is the richer successor to ``shape_opt_drag_steady_2d.py`` (single
mode-2 amplitude -> ellipse). The cylinder hole is morphed into a
Joukowski airfoil by a conformal map of the whole body-fit mesh
(``differentiable_mesh_with_airfoil``); because a Joukowski map is
injective on the circle exterior, the mesh stays valid under large
deformation rather than tangling the way radial Gordon-Hall curving does.

Two design variables exercise a genuine 2-D shape space:

    mu_x = -thickness * R0   (thickness; mu_x<0 fattens the body)
    mu_y =  camber    * R0   (camber; up/down asymmetry)

We minimise the steady drag at Re = 20 subject to a soft area constraint
that holds the enclosed area at the cylinder's (pi R0^2), so the body is
streamlined at fixed cross-section rather than collapsing to a sliver:

    L(thick, camber) = C_D + lambda_area * ((A - A0)/A0)^2 .

The gradient flows end-to-end: design params -> Joukowski map -> curved
mesh metrics / Piola scale -> BDM steady solve -> surface-integral C_D
(and -> closed-form airfoil area), with no hand-coded adjoint.
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

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inverse_nu_cylinder_2d import make_step_nu_param
from inverse_obstacle_cy_diff_2d import build_diff_setup, CX, RADIUS, D_CYL, U_MEAN, _bc_fn
from shape_opt_drag_cylinder_2d import _make_drag_eval
from shape_opt_drag_steady_2d import _cyl_face_mask
from ddg.dg2d.bdm import BDM, bdm_k_evaluate_boundary_velocity, refresh_piola_scale
from ddg.dg2d.coupled_ns_bdm import bdm1_evaluate_dirichlet_bc
from ddg.dg2d.differentiable_mesh import (
    differentiable_mesh_with_airfoil, differentiable_mesh_with_circle,
    joukowski_airfoil_area,
)

CY = 0.25
R1, R2 = 0.10, 0.18        # conformal-map window (airfoil ~0.09, inlet at 0.20)
TE_ROUND = 0.80            # trailing-edge rounding kappa (keeps J>0 over the box)
A0 = float(np.pi * RADIUS ** 2)   # cylinder cross-sectional area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--re", type=float, default=20.0)
    ap.add_argument("--symmetric", action="store_true",
                    help="pin camber=0 (no lift wanted); optimise thickness only")
    ap.add_argument("--thick-init", type=float, default=0.35)
    ap.add_argument("--camber-init", type=float, default=0.12)
    ap.add_argument("--thick-box", type=float, nargs=2, default=(0.18, 0.80))
    ap.add_argument("--camber-box", type=float, nargs=2, default=(-0.25, 0.25))
    ap.add_argument("--area-frac", type=float, default=1.0,
                    help="target area = area_frac * (cylinder area)")
    ap.add_argument("--lam-area", type=float, default=2.0,
                    help="area-penalty weight in L = C_D + lam*((A-At)/At)^2")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--grad-clip", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=1.0e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--grad-check", action="store_true")
    ap.add_argument("--fd-h", type=float, default=0.02)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    NU = U_MEAN * D_CYL / args.re
    A_target = args.area_frac * A0
    tlo, thi = args.thick_box
    clo, chi = args.camber_box
    print(f"BDM_{args.order} airfoil shape-opt: Re={args.re:.0f} (nu={NU:.3e}), "
          f"n_steps={args.n_steps}, dt={args.dt}, kappa={TE_ROUND}, "
          f"A_target/A0={args.area_frac:.2f}", flush=True)

    template, curving, cached_topology, bc_mask, is_natural_face = \
        build_diff_setup(args.kx, args.ky, args.order, CY)
    nominal = differentiable_mesh_with_circle(
        template.VX_nominal, template.VY_nominal, template, curving,
        jnp.asarray(CX), jnp.asarray(CY), jnp.asarray(RADIUS))
    is_cyl = _cyl_face_mask(nominal, CY)
    n_steps = int(args.n_steps)

    def steady_cd_area(p):
        thick = p[0]
        camber = 0.0 if args.symmetric else p[1]
        mu_x = -thick * RADIUS
        mu_y = camber * RADIUS
        mesh = differentiable_mesh_with_airfoil(
            template.VX_nominal, template.VY_nominal, template, curving,
            jnp.asarray(CX), jnp.asarray(CY), jnp.asarray(RADIUS),
            mu_x, mu_y, R1, R2, TE_ROUND)
        topo = refresh_piola_scale(cached_topology, mesh)
        V = BDM(mesh=mesh, topology=topo, order=args.order)
        u_bc = bdm1_evaluate_dirichlet_bc(V, _bc_fn)
        u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _bc_fn)
        step = make_step_nu_param(
            V, u_bc, u_bc_vec, bc_mask, is_natural_face,
            tau_scale=args.tau_scale, n_newton=args.n_newton)
        drag = _make_drag_eval(V, is_cyl)
        u1, _ = step(u_bc, u_bc, args.dt, NU, 1)

        def body(carry, _):
            u_n, u_nm1 = carry
            u_new, p_new = step(u_n, u_nm1, args.dt, NU, 2)
            return (u_new, u_n), p_new

        (u_final, _), ps = jax.lax.scan(
            jax.checkpoint(body), (u1, u_bc), None, length=n_steps - 1)
        cd = drag(u_final, ps[-1], NU)
        area = joukowski_airfoil_area(
            jnp.asarray(CX), jnp.asarray(CY), jnp.asarray(RADIUS),
            mu_x, mu_y, 512, TE_ROUND)
        return cd, area

    def loss(p):
        cd, area = steady_cd_area(p)
        return cd + args.lam_area * ((area - A_target) / A_target) ** 2

    p0 = jnp.asarray([float(args.thick_init), float(args.camber_init)])

    if args.forward_only:
        cd, area = jax.jit(steady_cd_area)(p0)
        print(f"  thick={float(p0[0]):.3f} camber={float(p0[1]):.3f}  "
              f"C_D={float(cd):+.5f}  area/A0={float(area)/A0:.4f}  "
              f"L={float(loss(p0)):+.5f}", flush=True)
        return

    if args.grad_check:
        h = args.fd_h
        g_ad = np.asarray(jax.jit(jax.grad(loss))(p0))
        g_fd = np.zeros(2)
        for i in range(2):
            ep = p0.at[i].add(h); em = p0.at[i].add(-h)
            g_fd[i] = (float(jax.jit(loss)(ep)) - float(jax.jit(loss)(em))) / (2 * h)
        print(f"  grad L at {np.asarray(p0)}: AD={g_ad}  FD={g_fd}", flush=True)
        for i, nm in enumerate(("thick", "camber")):
            print(f"    d/d{nm}: AD={g_ad[i]:+.5e} FD={g_fd[i]:+.5e} "
                  f"rel={abs(g_ad[i]-g_fd[i])/(abs(g_fd[i])+1e-30):.2e}", flush=True)
        return

    grad_fn = jax.jit(jax.grad(loss))
    cd_area_jit = jax.jit(steady_cd_area)

    def project(p):
        c = 0.0 if args.symmetric else jnp.clip(p[1], clo, chi)
        return jnp.array([jnp.clip(p[0], tlo, thi), c])

    stem = f"shape_opt_airfoil_steady_k{args.order}_re{int(args.re)}"
    out_json = args.out_dir / f"{stem}.json"

    def rec(it, p, cd, area, g):
        return dict(iter=int(it), thick=float(p[0]), camber=float(p[1]),
                    cd=float(cd), area=float(area), area_frac=float(area) / A0,
                    grad=[float(g[0]), float(g[1])])

    def save(history, complete):
        payload = dict(order=args.order, kx=args.kx, ky=args.ky, re=args.re,
                       nu=NU, n_steps=args.n_steps, dt=args.dt,
                       tau_scale=args.tau_scale, te_round=TE_ROUND,
                       r1=R1, r2=R2, cx=CX, cy=CY, radius=RADIUS,
                       symmetric=args.symmetric,
                       A0=A0, area_frac=args.area_frac, lam_area=args.lam_area,
                       thick_box=list(args.thick_box), camber_box=list(args.camber_box),
                       lr=args.lr, n_train=args.n_train, tol=args.tol,
                       patience=args.patience, complete=complete, history=history)
        tmp = args.out_dir / f"{stem}.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(out_json)

    p = project(p0)
    print("compiling forward + reverse ...", flush=True)
    t0 = time.time()
    cd0, area0 = (float(v) for v in cd_area_jit(p))
    L0 = cd0 + args.lam_area * ((area0 - A_target) / A_target) ** 2
    print(f"  initial: thick={float(p[0]):.3f} camber={float(p[1]):.3f}  "
          f"C_D={cd0:.5f}  area/A0={area0/A0:.4f}  L={L0:.5f}  "
          f"(compile {time.time()-t0:.1f}s)", flush=True)
    if not np.isfinite(L0):
        print("  -> non-finite initial loss; aborting", flush=True)
        return

    m_t = jnp.zeros_like(p); v_t = jnp.zeros_like(p)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history = [rec(0, p, cd0, area0, jnp.zeros(2))]
    save(history, complete=False)
    t0 = time.time(); L_prev = L0; n_flat = 0
    for it in range(1, args.n_train + 1):
        g_raw = grad_fn(p)
        g = jnp.clip(g_raw, -args.grad_clip, args.grad_clip)
        m_t = beta1 * m_t + (1 - beta1) * g
        v_t = beta2 * v_t + (1 - beta2) * g * g
        m_hat = m_t / (1 - beta1 ** it)
        v_hat = v_t / (1 - beta2 ** it)
        p = project(p - args.lr * m_hat / (jnp.sqrt(v_hat) + eps))
        cd, area = (float(v) for v in cd_area_jit(p))
        L = cd + args.lam_area * ((area - A_target) / A_target) ** 2
        rel = abs(L_prev - L) / max(abs(L_prev), 1e-30)
        print(f"  iter {it:3d}: thick={float(p[0]):.4f} camber={float(p[1]):+.4f}  "
              f"C_D={cd:.5f}  area/A0={area/A0:.4f}  L={L:.5f}  rel={rel:.2e}",
              flush=True)
        history.append(rec(it, p, cd, area, g))
        save(history, complete=False)
        if not np.isfinite(L):
            print("  -> non-finite loss; aborting (state saved)", flush=True)
            return
        n_flat = n_flat + 1 if rel < args.tol else 0
        if n_flat >= args.patience:
            print(f"  flatlined for {args.patience} iters -> stop at iter {it}",
                  flush=True)
            break
        L_prev = L
    print(f"\ntotal Adam wall: {time.time()-t0:.1f}s", flush=True)
    save(history, complete=True)
    print(f"  wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
