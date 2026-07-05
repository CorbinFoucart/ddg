"""Mass-conserving spatial-pulsing inlet control of the cylinder wake.

The boundary control is a time-varying inlet *profile*

    vp(y, t) = vp0(y) + sum_{m=1..M} c_m(t) * U_MAX * sin(2 pi m y / H)

where vp0 is the baseline parabolic profile and the perturbation modes
sin(2 pi m y / H) each VANISH at the walls (y=0, H) and have ZERO
y-mean, so

    integral_0^H vp(y, t) dy = integral_0^H vp0(y) dy = Q0   for all t.

The volumetric flow rate through the inlet is therefore held constant by
construction -- a hard constraint, not a penalty. The optimiser cannot
suppress the wake by throttling the throughput; it can only REDISTRIBUTE
momentum across the channel height and pulse it in time. Finding a
nontrivial controller that lowers the wake enstrophy then certifies that
the natural shedding state is not a constrained energy minimiser.

Architecture (control of a limit cycle):
  * Initial condition = a developed-shedding restart state (u_n, u_nm1)
    produced ON THIS MACHINE by ``cyl_shed_restart_2d.py --mode precursor``
    (no lossy data shipped; setup is self-consistent).
  * AD only over a SHORT window of a few shedding cycles (Re=100 shedding
    is a periodic limit cycle, so short-horizon gradients are well posed).
  * Objective = TIME-AVERAGED downstream enstrophy over the window.

Forward model: the block-Schur projection step (control enters only the
inflow Dirichlet trace, so the LU factors are control-independent). The
window is a ``lax.scan`` with per-step ``jax.checkpoint``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
from ddg.dg2d.projection_ns_bdm import (
    make_projection_solvers, projection_step_bdm_k_2d,
)
from ddg.dg2d.coupled_ns_bdm import bdm1_evaluate_dirichlet_bc
from ddg.dg2d.bdm import bdm_k_evaluate_boundary_velocity


def make_mode_bc(V, m):
    """BC DOF vectors for the zero-mean inlet mode sin(2 pi m y / H)."""
    def bc_fn(x, y):
        on_in = x < X_MIN + 1e-6
        ux = jnp.where(on_in, U_MAX * jnp.sin(2.0 * np.pi * m * y / H_CHANNEL), 0.0)
        return ux, jnp.zeros_like(y)
    return bdm1_evaluate_dirichlet_bc(V, bc_fn), bdm_k_evaluate_boundary_velocity(V, bc_fn)


def make_proj_step_profile(V, u_bc0, u_bc_vec0, mode_u, mode_uvec,
                           bc_mask, is_natural_face, *, nu, tau_scale, dt):
    """Projection step with a spatial-profile inlet control.

    step(u_n, u_nm1, c, order_eff) -> (u_new, p_new), where ``c`` is the
    (M,) vector of mode amplitudes; the inflow trace is
    ``u_bc0 + sum_m c_m * mode_u[m]`` (and likewise for u_bc_vec).
    """
    ps1, hinv1 = make_projection_solvers(
        V, dt=dt, nu=nu, order=1, tau_scale=tau_scale,
        bc_mask=bc_mask, is_natural_face=is_natural_face)
    ps2, hinv2 = make_projection_solvers(
        V, dt=dt, nu=nu, order=2, tau_scale=tau_scale,
        bc_mask=bc_mask, is_natural_face=is_natural_face)

    def step(u_n, u_nm1, c, order_eff):
        u_bc_t = u_bc0 + jnp.tensordot(c, mode_u, axes=(0, 0))
        u_bc_vec_t = u_bc_vec0 + jnp.tensordot(c, mode_uvec, axes=(0, 0))
        if order_eff == 1:
            ps, hinv, u_hist = ps1, hinv1, [u_n]
        else:
            ps, hinv, u_hist = ps2, hinv2, [u_n, u_nm1]
        return projection_step_bdm_k_2d(
            V, u_hist, dt=dt, nu=nu, order=order_eff,
            pressure_solver=ps, h_inv=hinv,
            u_bc_global=u_bc_t, u_bc_vec=u_bc_vec_t,
            bc_mask=bc_mask, is_natural_face=is_natural_face,
            tau_scale=tau_scale)
    return step


def make_inlet_flux_fn(V):
    """Q = integral_inlet u . n ds, computed from the solution by Gauss
    quadrature over the inflow faces (n = -x_hat). Used to VERIFY the
    volumetric flow rate stays constant over the controlled window."""
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
    # inflow (K,f) pairs
    Fmask = np.asarray(mesh.Fmask); x = np.asarray(mesh.x)
    EToE = np.asarray(mesh.EToE); is_bdy = (EToE == np.arange(mesh.K)[:, None]).T
    xf = np.zeros((3, mesh.K))
    for f in range(3):
        xf[f] = x[Fmask[:, f]].mean(0)
    pairs = [(K_, f_) for K_ in range(mesh.K) for f_ in range(3)
             if is_bdy[f_, K_] and xf[f_, K_] < X_MIN + 1e-3]

    def flux(u_global):
        ul = bdm1_gather(V.topology, u_global); Q = 0.0
        for (K_, f_) in pairs:
            phys = jnp.einsum("cb,qlb->qlc", DF[K_], phif[f_]) / Jall[K_]
            u_at = jnp.einsum("l,qlc->qc", ul[K_], phys)
            nv = jnp.stack([nxKF[K_, f_], nyKF[K_, f_]])
            un = u_at @ nv
            Q = Q + jnp.sum(w * sJKF[K_, f_] * un)
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
    ap.add_argument("--n-modes", type=int, default=4)
    ap.add_argument("--c-max", type=float, default=0.5,
                    help="bound on |c_m| (fraction of U_MAX)")
    ap.add_argument("--n-window", type=int, default=4000,
                    help="control-window steps (~ a few shedding cycles)")
    ap.add_argument("--n-control", type=int, default=20,
                    help="piecewise-constant control knots over the window")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--tol", type=float, default=2.0e-3,
                    help="relative enstrophy-change flatline threshold (early stop)")
    ap.add_argument("--patience", type=int, default=8,
                    help="consecutive sub-tol iters before declaring flatline")
    ap.add_argument("--lambda-ctrl", type=float, default=0.0,
                    help="optional control-effort regulariser weight")
    ap.add_argument("--restart-file", type=Path,
                    default=ROOT / "output" / "cyl_shed_restart_state.npz")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    ap.add_argument("--verify-only", action="store_true",
                    help="forward window at c=0, print enstrophy + inlet flux Q(t)")
    ap.add_argument("--dump-fields", action="store_true",
                    help="reconstruct baseline + optimised vorticity series for the figure")
    ap.add_argument("--field-every", type=int, default=30,
                    help="save a vorticity snapshot every N steps in --dump-fields")
    ap.add_argument("--check-control", type=Path, default=None,
                    help="forward-only horizon-exploitation check: tile this "
                         "control JSON's c_knots over --n-window steps and record "
                         "baseline (c=0) vs controlled downstream-enstrophy traces")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    nu = U_MEAN * D_CYL / args.re
    V, u_bc, u_bc_vec, bc_mask, is_natural_face, _ = build_static(
        args.kx, args.ky, args.order, n_radial=args.n_radial, r_outer=args.r_outer)
    n_u = V.total_dofs
    Re = U_MEAN * D_CYL / nu
    print(f"[profile-ctrl] BDM_{args.order} Re={Re:.0f} K={V.K} n_u={n_u} "
          f"dt={args.dt} M={args.n_modes} window={args.n_window}", flush=True)

    # restart state from the on-machine precursor
    rs = np.load(args.restart_file)
    u0 = jnp.asarray(rs["u_n"]); u0m = jnp.asarray(rs["u_nm1"])
    print(f"  loaded restart state from {args.restart_file} "
          f"(precursor n1={int(rs['n1'])}, dt={float(rs['dt'])})", flush=True)

    # zero-mean spatial modes
    mode_u = []; mode_uvec = []
    for m in range(1, args.n_modes + 1):
        mu, mv = make_mode_bc(V, m)
        mode_u.append(mu); mode_uvec.append(mv)
    mode_u = jnp.stack(mode_u); mode_uvec = jnp.stack(mode_uvec)   # (M, ...)

    step_fn = make_proj_step_profile(
        V, u_bc, u_bc_vec, mode_u, mode_uvec, bc_mask, is_natural_face,
        nu=nu, tau_scale=args.tau_scale, dt=args.dt)
    vort = make_vorticity_at_nodes_fn(V)
    flux_fn = jax.jit(make_inlet_flux_fn(V))

    x_np = np.asarray(V.mesh.x); y_np = np.asarray(V.mesh.y)
    in_window = ((x_np > CX + 1.5 * D_CYL) & (x_np < CX + 6.0 * D_CYL)
                 & (y_np > Y_MIN + 0.02) & (y_np < Y_MAX - 0.02))
    J_node = jnp.asarray(np.tile(V.mesh.J[0][None, :], (V.mesh.Np, 1)))
    weight = jnp.asarray(in_window.astype(np.float64)) * J_node

    knot_of_step = np.minimum(
        (np.arange(args.n_window) * args.n_control) // args.n_window,
        args.n_control - 1).astype(np.int32)
    knot_of_step = jnp.asarray(knot_of_step)

    def c_of_theta(theta):                       # theta: (n_control, M)
        return args.c_max * jnp.tanh(theta)

    def run_window(theta, accumulate=True):
        """March the window from the restart state. Returns
        (mean_enstrophy, last_u). c is piecewise-constant per knot."""
        cs = c_of_theta(theta)                   # (n_control, M)
        c_step = cs[knot_of_step]                # (n_window, M)
        u1, _ = step_fn(u0, u0m, c_step[0], 2)   # already BDF2-ready (restart)

        def body(carry, c):
            u_n, u_nm1 = carry
            u_new, _ = step_fn(u_n, u_nm1, c, 2)
            w = vort(u_new)
            e = jnp.sum(weight * w**2)
            return (u_new, u_n), e

        (u_last, _), E = jax.lax.scan(
            jax.checkpoint(body), (u1, u0), c_step[1:])
        return jnp.mean(E), u_last

    if args.verify_only:
        # forward at c=0: enstrophy + inlet flux constancy
        theta0 = jnp.zeros((args.n_control, args.n_modes))
        cs = c_of_theta(theta0)[knot_of_step]
        u_n, u_nm1 = u0, u0m
        Q0 = float(flux_fn(u0))
        print(f"  baseline inlet flux Q0 = {Q0:.6e}", flush=True)
        # fine-sample ~3 shedding periods to detect the limit cycle (the
        # T=1.0 precursor logging aliased the ~T=0.5 shedding period)
        nfs = min(args.n_window, 3000)
        Es = []; Qs = []
        for s in range(nfs):
            u_new, _ = step_fn(u_n, u_nm1, cs[s], 2)   # cs is all-zero (c=0)
            u_nm1, u_n = u_n, u_new
            if (s + 1) % 50 == 0:
                e = float(jnp.sum(weight * vort(u_n)**2)); q = float(flux_fn(u_n))
                Es.append(e); Qs.append(q)
                print(f"  step {s+1:5d} t={args.dt*(s+1):.3f} E={e:.4e} "
                      f"Q={q:.6e} dQ/Q0={(q-Q0)/abs(Q0):.2e}", flush=True)
        Es = np.array(Es); Qs = np.array(Qs)
        amp = (Es.max() - Es.min()) / (Es.mean() + 1e-30)
        print(f"  enstrophy fine-sample: rel oscillation amplitude = {amp:.3f}  "
              f"({'SHEDDING' if amp > 0.05 else 'STEADY (no shedding at this Re)'})",
              flush=True)
        print(f"  inlet flux drift over window: max|dQ/Q0| = "
              f"{np.max(np.abs((Qs-Q0)/abs(Q0))):.2e}", flush=True)
        return

    if args.dump_fields:
        # Reconstruct baseline (c=0) and optimised (c=c_knots) vorticity
        # time series over the window, for the wake-suppression figure.
        import json as _json
        d = _json.loads((args.out_dir / f"profile_ctrl_cyl_proj_k{args.order}.json").read_text())
        ck = jnp.asarray(d["c_knots"])                 # (n_control, M)
        ck_step = ck[knot_of_step]                      # (n_window, M)
        step_j = jax.jit(step_fn, static_argnums=(3,))
        vort_j = jax.jit(vort)

        def collect(cstep):
            u_n, u_nm1 = u0, u0m; W = []
            for s in range(args.n_window):
                u_new, _ = step_j(u_n, u_nm1, cstep[s], 2)
                u_nm1, u_n = u_n, u_new
                if (s + 1) % args.field_every == 0:
                    W.append(np.asarray(vort_j(u_n)))
            return np.stack(W)

        print("  collecting baseline (c=0) field series ...", flush=True)
        w_base = collect(jnp.zeros_like(ck_step))
        print("  collecting optimised (c=c_knots) field series ...", flush=True)
        w_ctrl = collect(ck_step)
        px = np.asarray(V.mesh.x); py = np.asarray(V.mesh.y)
        nsv = w_base.shape[0]
        np.savez_compressed(
            args.out_dir / f"profile_ctrl_cyl_proj_k{args.order}_fields.npz",
            mesh_x=px, mesh_y=py, omega_baseline=w_base, omega_controlled=w_ctrl,
            c_knots=np.asarray(ck), n_control=args.n_control, n_modes=args.n_modes,
            times=(np.arange(nsv) + 1) * args.field_every * args.dt,
            CX=CX, CY=CY, RADIUS=RADIUS, X_MIN=X_MIN, X_MAX=X_MAX,
            Y_MIN=Y_MIN, Y_MAX=Y_MAX, D_CYL=D_CYL)
        print(f"  wrote {args.out_dir}/profile_ctrl_cyl_proj_k{args.order}_fields.npz "
              f"({nsv} snapshots each)", flush=True)
        return

    if args.check_control is not None:
        # Horizon-exploitation check: take an already-learned control (its
        # c_knots span the ORIGINAL short window), TILE it periodically over a
        # longer window --n-window (same actuation frequency), and march
        # forward-only recording the downstream enstrophy for baseline (c=0)
        # vs controlled. If the controlled trace stays suppressed past the
        # original window, the suppression is sustained, not a finite-horizon
        # trick. Run with --n-modes matching the loaded control.
        import json as _json
        d = _json.loads(Path(args.check_control).read_text())
        ck = np.asarray(d["c_knots"])                  # (n_control0, M0)
        n_ctrl0, M0 = ck.shape
        n_win0 = int(d["n_window"])
        assert M0 == args.n_modes, f"--n-modes {args.n_modes} != control modes {M0}"
        kos0 = np.minimum((np.arange(n_win0) * n_ctrl0) // n_win0, n_ctrl0 - 1)
        cstep0 = ck[kos0]                              # (n_win0, M0) per-step
        N = args.n_window
        cstep_long = jnp.asarray(cstep0[np.arange(N) % n_win0])   # tiled
        step_j = jax.jit(step_fn, static_argnums=(3,))
        vort_j = jax.jit(vort)

        def etrace(cstep):
            u_n, u_nm1 = u0, u0m; es = []
            for s in range(N):
                u_new, _ = step_j(u_n, u_nm1, cstep[s], 2)
                u_nm1, u_n = u_n, u_new
                if (s + 1) % args.field_every == 0:
                    es.append(float(jnp.sum(weight * vort_j(u_n) ** 2)))
            return np.array(es)

        print(f"  horizon check: tiling {n_ctrl0}-knot control (orig window "
              f"{n_win0} steps) over N={N} steps ({N/n_win0:.1f}x)", flush=True)
        e_base = etrace(jnp.zeros_like(cstep_long))
        print(f"  baseline trace done; <E>_base={e_base.mean():.4e}", flush=True)
        e_ctrl = etrace(cstep_long)
        print(f"  controlled trace done; <E>_ctrl={e_ctrl.mean():.4e}", flush=True)
        ts = (np.arange(e_base.size) + 1) * args.field_every * args.dt
        half = ts > (n_win0 * args.dt)               # samples beyond orig window
        red_all = 100 * (1 - e_ctrl.mean() / e_base.mean())
        red_tail = 100 * (1 - e_ctrl[half].mean() / e_base[half].mean()) if half.any() else float("nan")
        out = args.out_dir / f"profile_ctrl_horizon_check_k{args.order}.json"
        out.write_text(_json.dumps(dict(
            n_window=N, n_window_orig=n_win0, dt=args.dt, field_every=args.field_every,
            t=ts.tolist(), e_baseline=e_base.tolist(), e_controlled=e_ctrl.tolist(),
            reduction_pct_all=red_all, reduction_pct_beyond_orig_window=red_tail)))
        print(f"  SUPPRESSION over full {N/n_win0:.1f}x window: {red_all:.1f}%  | "
              f"beyond original window: {red_tail:.1f}%", flush=True)
        print(f"  wrote {out}", flush=True)
        return

    grad_fn = jax.jit(jax.value_and_grad(lambda th: run_window(th)[0]))
    theta = jnp.zeros((args.n_control, args.n_modes))
    print("compiling forward+reverse over the control window ...", flush=True)
    t0 = time.time()
    L0, _ = grad_fn(theta)
    L0 = float(L0)
    print(f"  baseline <enstrophy> = {L0:.6e}  (compile {time.time()-t0:.1f}s)",
          flush=True)
    if not np.isfinite(L0):
        print("  -> non-finite baseline; aborting", flush=True); return

    stem = f"profile_ctrl_cyl_proj_k{args.order}"
    def save(history, theta_now, complete):
        ak = np.asarray(c_of_theta(theta_now))
        payload = dict(solver="projection_profile_ctrl", order=args.order,
                       re=Re, dt=args.dt, n_window=args.n_window,
                       n_control=args.n_control, n_modes=args.n_modes,
                       c_max=args.c_max, lr=args.lr, n_train=args.n_train,
                       loss_baseline=L0, complete=complete,
                       c_knots=ak.tolist(), history=history)
        tmp = args.out_dir / f"{stem}.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(args.out_dir / f"{stem}.json")

    m_t = jnp.zeros_like(theta); v_t = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = [dict(iter=0, loss=L0)]
    save(history, theta, False)
    t0 = time.time(); L_prev = L0; n_flat = 0
    for it in range(1, args.n_train + 1):
        ti = time.time()
        L, g = grad_fn(theta)
        L = float(L)
        m_t = b1 * m_t + (1 - b1) * g
        v_t = b2 * v_t + (1 - b2) * g * g
        theta = theta - args.lr * (m_t / (1 - b1**it)) / (jnp.sqrt(v_t / (1 - b2**it)) + eps)
        gn = float(jnp.linalg.norm(g))
        rel = abs(L_prev - L) / max(abs(L_prev), 1e-30)
        print(f"  iter {it:3d}: <enstrophy>={L:.6e}  |g|={gn:.3e}  "
              f"frac={L/L0:.3f}  rel_dL={rel:.2e}  ({time.time()-ti:.0f}s)", flush=True)
        history.append(dict(iter=it, loss=L, gnorm=gn, rel=rel))
        save(history, theta, False)
        if not np.isfinite(L):
            print("  -> non-finite; aborting (state saved)", flush=True); return
        # flatline early-stop: run until the enstrophy genuinely bottoms out
        n_flat = n_flat + 1 if rel < args.tol else 0
        if n_flat >= args.patience:
            print(f"  flatlined: rel_dL < {args.tol:.1e} for {args.patience} "
                  f"iters -> stop at iter {it}", flush=True)
            break
        L_prev = L
    print(f"\nAdam wall {time.time()-t0:.1f}s  ({L0:.4e} -> {history[-1]['loss']:.4e})",
          flush=True)
    save(history, theta, True)


if __name__ == "__main__":
    main()
