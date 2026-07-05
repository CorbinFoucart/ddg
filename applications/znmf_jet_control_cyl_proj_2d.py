"""Zero-net-mass-flux (synthetic-jet) boundary control of the cylinder wake.

Actuator: a prescribed NORMAL velocity on the cylinder surface,

    v_n(theta, t) = sum_m a_m(t) * U_MAX * g_m(theta) ,   g_m in {cos m*theta, sin m*theta}, m>=1,

applied as the cylinder Dirichlet trace  u = v_n(theta,t) * n_hat. Every mode
g_m integrates to zero around the cylinder, so the net mass flux through the
body is ZERO by construction (a synthetic jet: blow on one arc, suck on
another, no net injection). This is the body-located analogue of the
mass-conserving inlet control, but with authority right at the separation
points -- and, crucially, it cannot be cheated by "just blowing harder",
both because of the zero-net-flux constraint and an explicit effort penalty:

    L(a) = < downstream wake enstrophy >_t  +  lambda_eff * < |a|^2 >_t .

The optimum is therefore a structured, efficient (ideally phase-locked)
actuation rather than a saturated knob.

Same projection forward model + restart-from-shedding + short-window adjoint
as the inlet control; only the actuator (cylinder jet vs inlet profile)
differs, so make_proj_step_profile is reused verbatim with jet modes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diff_cylinder_dt_tau import (
    build_static, _bc_fn, X_MIN, X_MAX, Y_MIN, Y_MAX, CX, CY, RADIUS, D_CYL,
    U_MAX, U_MEAN, NU, H_CHANNEL,
)
from inflow_control_cylinder_2d import make_vorticity_at_nodes_fn
from inflow_profile_control_cyl_proj_2d import make_proj_step_profile
from ddg.dg2d.coupled_ns_bdm import bdm1_evaluate_dirichlet_bc
from ddg.dg2d.bdm import bdm_k_evaluate_boundary_velocity


def make_cyl_jet_mode(V, m, kind):
    """BC DOF vectors for one zero-net-flux cylinder synthetic-jet mode:
    normal velocity g_m(theta) * n_hat on the cylinder surface, 0 elsewhere.
    kind in {'cos','sin'}; g integrates to 0 around the body for m>=1."""
    def bc_fn(x, y):
        dx = x - CX
        dy = y - CY
        r = jnp.sqrt(dx * dx + dy * dy)
        on_cyl = r < RADIUS * 1.05
        th = jnp.arctan2(dy, dx)
        g = jnp.cos(m * th) if kind == "cos" else jnp.sin(m * th)
        rr = jnp.where(r > 1e-12, r, 1.0)
        amp = U_MAX * jnp.where(on_cyl, g, 0.0) / rr      # g * n_hat = g*(dx,dy)/r
        return amp * dx, amp * dy
    return (bdm1_evaluate_dirichlet_bc(V, bc_fn),
            bdm_k_evaluate_boundary_velocity(V, bc_fn))


def make_cyl_flux_fn(V):
    """Q_cyl = integral over the cylinder of u . n ds (n outward), by Gauss
    quadrature on the cylinder faces -- used to VERIFY zero net mass flux."""
    from ddg.dg2d.bdm import (bdm_k_basis_at_points, bdm1_gather,
                              _element_jacobian_matrices)
    k = V.order; mesh = V.mesh
    n_gl = max(k + 1, 3)
    xi, w = np.polynomial.legendre.leggauss(n_gl)
    xi = jnp.asarray(xi); w = jnp.asarray(w)

    def face_ref(f, t):
        o = jnp.ones_like(t)
        return (t, -o) if f == 0 else ((-t, t) if f == 1 else (-o, -t))
    phif = [bdm_k_basis_at_points(k, *face_ref(f, xi)) for f in range(3)]
    DF, _ = _element_jacobian_matrices(mesh); Jall = mesh.J[0]
    nxKF = jnp.asarray(mesh.nx).reshape(3, mesh.Nfp, mesh.K).transpose(2, 0, 1)[:, :, 0]
    nyKF = jnp.asarray(mesh.ny).reshape(3, mesh.Nfp, mesh.K).transpose(2, 0, 1)[:, :, 0]
    sJKF = jnp.asarray(mesh.sJ).reshape(3, mesh.Nfp, mesh.K).transpose(2, 0, 1)[:, :, 0]
    Fmask = np.asarray(mesh.Fmask); x = np.asarray(mesh.x); y = np.asarray(mesh.y)
    EToE = np.asarray(mesh.EToE); is_bdy = (EToE == np.arange(mesh.K)[:, None]).T
    xf = np.zeros((3, mesh.K)); yf = np.zeros((3, mesh.K))
    for f in range(3):
        xf[f] = x[Fmask[:, f]].mean(0); yf[f] = y[Fmask[:, f]].mean(0)
    rfc = np.sqrt((xf - CX) ** 2 + (yf - CY) ** 2)
    pairs = [(K_, f_) for K_ in range(mesh.K) for f_ in range(3)
             if is_bdy[f_, K_] and rfc[f_, K_] < RADIUS * 1.05]

    def flux(u_global):
        ul = bdm1_gather(V.topology, u_global); Q = 0.0
        for (K_, f_) in pairs:
            phys = jnp.einsum("cb,qlb->qlc", DF[K_], phif[f_]) / Jall[K_]
            u_at = jnp.einsum("l,qlc->qc", ul[K_], phys)
            nv = jnp.stack([nxKF[K_, f_], nyKF[K_, f_]])
            Q = Q + jnp.sum(w * sJKF[K_, f_] * (u_at @ nv))
        return Q
    return flux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-radial", type=int, default=2)
    ap.add_argument("--r-outer", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=5.0e-4)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--n-modes", type=int, default=4,
                    help="azimuthal modes m=1..M; jet basis is cos & sin of each (2M dofs)")
    ap.add_argument("--c-max", type=float, default=0.6,
                    help="bound on per-mode amplitude: a = c_max*tanh(theta)")
    ap.add_argument("--n-window", type=int, default=3000)
    ap.add_argument("--n-control", type=int, default=24)
    ap.add_argument("--lam-eff", type=float, default=0.0,
                    help="actuation-effort penalty weight in L = <enstrophy> + lam*<|a|^2>")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1.0e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--restart-file", type=Path,
                    default=ROOT / "output" / "cyl_shed_restart_state.npz")
    ap.add_argument("--from-rest", action="store_true",
                    help="start the window from rest (u=u_bc) -- machinery smoke only")
    ap.add_argument("--verify-only", action="store_true",
                    help="build a test jet state and report cylinder net flux (ZNMF check)")
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--grad-check", action="store_true")
    ap.add_argument("--fd-h", type=float, default=1.0e-3)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    nu = U_MEAN * D_CYL / args.re
    V, u_bc, u_bc_vec, bc_mask, is_natural_face, _ = build_static(
        args.kx, args.ky, args.order, n_radial=args.n_radial, r_outer=args.r_outer)
    print(f"[znmf-jet] BDM_{args.order} Re={args.re:.0f} K={V.K} n_u={u_bc.size} "
          f"dt={args.dt} M={args.n_modes} (2M={2*args.n_modes} dofs) "
          f"window={args.n_window}", flush=True)

    # zero-net-flux cylinder jet modes: cos & sin of m=1..M
    jm_u, jm_v = [], []
    for m in range(1, args.n_modes + 1):
        for kind in ("cos", "sin"):
            a_, b_ = make_cyl_jet_mode(V, m, kind)
            jm_u.append(a_); jm_v.append(b_)
    mode_u = jnp.stack(jm_u); mode_uvec = jnp.stack(jm_v)     # (2M, ...)
    n_jet = mode_u.shape[0]

    step_fn = make_proj_step_profile(
        V, u_bc, u_bc_vec, mode_u, mode_uvec, bc_mask, is_natural_face,
        nu=nu, tau_scale=args.tau_scale, dt=args.dt)
    vort = make_vorticity_at_nodes_fn(V)
    cyl_flux = make_cyl_flux_fn(V)

    # initial state
    if args.from_rest:
        u0 = jnp.zeros_like(u_bc); u0m = u0      # true rest; BC drives the flow
        print("  starting window from REST (machinery smoke)", flush=True)
    else:
        rs = np.load(args.restart_file)
        u0 = jnp.asarray(rs["u_n"]); u0m = jnp.asarray(rs["u_nm1"])
        print(f"  restart from {args.restart_file} (n1={int(rs['n1'])})", flush=True)

    # downstream enstrophy weight (same box as the inlet control)
    x_np = np.asarray(V.mesh.x); y_np = np.asarray(V.mesh.y)
    in_box = ((x_np > CX + 1.5 * D_CYL) & (x_np < CX + 6.0 * D_CYL)
              & (y_np > Y_MIN + 0.02) & (y_np < Y_MAX - 0.02))
    J_node = jnp.asarray(np.tile(V.mesh.J[0][None, :], (V.mesh.Np, 1)))
    weight = jnp.asarray(in_box.astype(np.float64)) * J_node

    knot_of_step = jnp.asarray(np.minimum(
        (np.arange(args.n_window) * args.n_control) // args.n_window,
        args.n_control - 1).astype(np.int32))

    def amps(theta):
        return args.c_max * jnp.tanh(theta)              # (n_control, n_jet)

    def run_window(theta):
        a_step = amps(theta)[knot_of_step]               # (n_window, n_jet)
        u1, _ = step_fn(u0, u0m, a_step[0], 2)

        def body(carry, a):
            u_n, u_nm1 = carry
            u_new, _ = step_fn(u_n, u_nm1, a, 2)
            e = jnp.sum(weight * vort(u_new) ** 2)
            return (u_new, u_n), e

        (u_last, _), E = jax.lax.scan(jax.checkpoint(body), (u1, u0), a_step[1:])
        return jnp.mean(E), u_last

    def effort(theta):
        return jnp.mean(jnp.sum(amps(theta) ** 2, axis=1))

    def objective(theta):
        return run_window(theta)[0] + args.lam_eff * effort(theta)

    # ---- ZNMF verification: impose a random-ish jet, one step, measure flux
    if args.verify_only:
        th = jnp.zeros((args.n_control, n_jet)).at[0].set(0.5)   # nonzero jet
        a0 = amps(th)[0]
        u1, _ = jax.jit(step_fn, static_argnums=(3,))(u0, u0m, a0, 2)
        Q0 = float(jax.jit(cyl_flux)(u0)); Q1 = float(jax.jit(cyl_flux)(u1))
        # reference inflow scale for relative size
        print(f"  cylinder net flux: uncontrolled Q={Q0:.3e}, "
              f"controlled Q={Q1:.3e}  (ZNMF target ~ 0)", flush=True)
        umax_in = U_MEAN * H_CHANNEL
        print(f"  |Q_ctrl| / (U_mean*H) = {abs(Q1)/umax_in:.2e}  "
              f"({'ZNMF OK' if abs(Q1)/umax_in < 1e-3 else 'CHECK'})", flush=True)
        return

    theta0 = jnp.zeros((args.n_control, n_jet))

    if args.forward_only:
        th = theta0.at[:, 0].set(0.4)                    # a steady cos1 jet
        L0, _ = jax.jit(run_window)(theta0)
        Lj, _ = jax.jit(run_window)(th)
        print(f"  <enstrophy> baseline(theta=0)={float(L0):.6e}  "
              f"with test jet={float(Lj):.6e}  effort={float(effort(th)):.4e}", flush=True)
        return

    if args.grad_check:
        th = theta0.at[0, 0].set(0.3).at[1, 1].set(-0.2)
        g_ad = jax.jit(jax.grad(objective))(th)
        h = args.fd_h
        for (i, j) in ((0, 0), (1, 1)):
            ep = th.at[i, j].add(h); em = th.at[i, j].add(-h)
            gfd = (float(jax.jit(objective)(ep)) - float(jax.jit(objective)(em))) / (2 * h)
            gad = float(g_ad[i, j])
            print(f"  d/dtheta[{i},{j}]: AD={gad:+.5e} FD={gfd:+.5e} "
                  f"rel={abs(gad-gfd)/(abs(gfd)+1e-30):.2e}", flush=True)
        return

    # ---- optimise
    grad_fn = jax.jit(jax.value_and_grad(objective))
    stem = f"znmf_jet_ctrl_cyl_proj_k{args.order}"
    out = args.out_dir / f"{stem}.json"
    print("compiling forward+reverse ...", flush=True)
    t0 = time.time()
    L0, g0 = grad_fn(theta0)
    L0 = float(L0); Lw0, _ = run_window(theta0)
    print(f"  baseline L={L0:.6e} (<enstrophy>={float(Lw0):.6e})  "
          f"(compile {time.time()-t0:.1f}s)", flush=True)
    theta = theta0
    mt = jnp.zeros_like(theta); vt = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = [dict(iter=0, loss=L0)]; Lprev = L0; nflat = 0
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        ti = time.time()
        L, g = grad_fn(theta)
        L = float(L); gn = float(jnp.linalg.norm(g))
        mt = b1 * mt + (1 - b1) * g; vt = b2 * vt + (1 - b2) * g * g
        theta = theta - args.lr * (mt / (1 - b1 ** it)) / (jnp.sqrt(vt / (1 - b2 ** it)) + eps)
        rel = abs(Lprev - L) / max(abs(Lprev), 1e-30)
        Lw, _ = run_window(theta); eff = float(effort(theta))
        print(f"  iter {it:3d}: L={L:.6e} <ens>={float(Lw):.6e} eff={eff:.4e} "
              f"|g|={gn:.3e} frac={float(Lw)/float(Lw0):.3f} rel={rel:.2e} "
              f"({time.time()-ti:.0f}s)", flush=True)
        hist.append(dict(iter=it, loss=L, enstrophy=float(Lw), effort=eff,
                         gnorm=gn, rel=rel))
        tmp = args.out_dir / f"{stem}.json.tmp"
        tmp.write_text(json.dumps(dict(
            order=args.order, re=args.re, dt=args.dt, n_window=args.n_window,
            n_control=args.n_control, n_modes=args.n_modes, n_jet=n_jet,
            c_max=args.c_max, lam_eff=args.lam_eff, lr=args.lr,
            enstrophy_baseline=float(Lw0), theta=np.asarray(theta).tolist(),
            history=hist, complete=False)))
        tmp.replace(out)
        if not np.isfinite(L):
            print("  -> non-finite; abort", flush=True); return
        nflat = nflat + 1 if rel < args.tol else 0
        if nflat >= args.patience:
            print(f"  flatlined -> stop at iter {it}", flush=True); break
        Lprev = L
    print(f"\ntotal Adam wall {time.time()-t0:.0f}s; wrote {out}", flush=True)
    payload = json.loads(out.read_text()); payload["complete"] = True
    out.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
