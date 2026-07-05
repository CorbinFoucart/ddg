"""Projection-solver cylinder shedding: precursor-to-shedding + restart.

Phase-A driver for the boundary-control demo. Two jobs, NO autodiff:

  --mode verify     Run a CONTINUOUS trajectory from rest for
                    (n1 + n2) steps, snapshot the state (u_n, u_nm1) at
                    step n1, then RESTART from that snapshot and march n2
                    steps. The restart trajectory must match the
                    continuous one over [n1, n1+n2] to round-off -- this
                    certifies the restart mechanism is exact (no blow-up,
                    no drift). Small sizes; mechanism check only.

  --mode precursor  March from rest for n1 steps (long enough to reach
                    developed vortex shedding), logging the downstream
                    enstrophy E(t) every --log-every steps so the
                    shedding onset is visible, and saving the restart
                    state (u_n, u_nm1) atomically to --restart-file. This
                    state becomes the initial condition for the AD
                    boundary-control window.

The projection step's only state is (u_n, u_nm1) (the Helmholtz /
pressure-Schur LU factors are control-independent and rebuilt here), so
the restart is exact given those two velocity DOF vectors.
"""
from __future__ import annotations

import argparse
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
    build_static, X_MIN, X_MAX, Y_MIN, Y_MAX, CX, CY, RADIUS, D_CYL,
    U_MEAN, NU,
)
from inflow_control_cylinder_2d import make_vorticity_at_nodes_fn
from inflow_control_cylinder_projection_2d import make_proj_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "precursor"], default="verify")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-radial", type=int, default=2)
    ap.add_argument("--r-outer", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--n1", type=int, default=200, help="precursor / pre-snapshot steps")
    ap.add_argument("--n2", type=int, default=100, help="post-restart steps (verify)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--restart-file", type=Path,
                    default=ROOT / "output" / "cyl_shed_restart_state.npz")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    nu = U_MEAN * D_CYL / args.re
    V, u_bc, u_bc_vec, bc_mask, is_natural_face, _is_cyl = build_static(
        args.kx, args.ky, args.order, n_radial=args.n_radial, r_outer=args.r_outer)
    n_u = V.total_dofs
    Re = U_MEAN * D_CYL / nu
    print(f"[restart] BDM_{args.order} proj cylinder: Re={Re:.0f} (nu={nu:.3e}), "
          f"K={V.K}, n_u={n_u}, dt={args.dt}", flush=True)

    step_fn = make_proj_step(V, u_bc, u_bc_vec, bc_mask, is_natural_face,
                             nu=nu, tau_scale=4.0, dt=args.dt)
    step = jax.jit(step_fn, static_argnums=(3,))
    vort = jax.jit(make_vorticity_at_nodes_fn(V))

    # downstream enstrophy window
    x_np = np.asarray(V.mesh.x); y_np = np.asarray(V.mesh.y)
    in_window = ((x_np > CX + 1.5 * D_CYL) & (x_np < CX + 6.0 * D_CYL)
                 & (y_np > Y_MIN + 0.02) & (y_np < Y_MAX - 0.02))
    J_node = jnp.asarray(np.tile(V.mesh.J[0][None, :], (V.mesh.Np, 1)))
    weight = jnp.asarray(in_window.astype(np.float64)) * J_node

    def enstrophy(u):
        w = vort(u)
        return float(jnp.sum(weight * w**2))

    ALPHA1 = jnp.asarray(1.0)   # no control: baseline inflow

    def march(u_n, u_nm1, n, first_order_eff, record=True, label=""):
        E = []
        for s in range(n):
            oe = first_order_eff if s == 0 else 2
            u_new, _ = step(u_n, u_nm1, ALPHA1, oe)
            u_nm1, u_n = u_n, u_new
            if record and ((s + 1) % args.log_every == 0 or s == 0):
                e = enstrophy(u_n)
                E.append((s + 1, e))
                if not np.isfinite(e):
                    print(f"  {label} step {s+1}: NON-FINITE enstrophy -> blow-up",
                          flush=True)
                    return u_n, u_nm1, E, False
                print(f"  {label} step {s+1:5d}  t={args.dt*(s+1):7.3f}  E={e:.6e}",
                      flush=True)
        return u_n, u_nm1, E, True

    if args.mode == "verify":
        print("=== CONTINUOUS run (rest -> n1+n2) ===", flush=True)
        u_n = jnp.zeros(n_u); u_nm1 = jnp.zeros(n_u)
        # march to snapshot point n1
        u_n, u_nm1, _, ok = march(u_n, u_nm1, args.n1, 1, label="cont")
        if not ok:
            print("BLEW UP before snapshot", flush=True); return
        snap_u_n, snap_u_nm1 = u_n, u_nm1
        e_snap = enstrophy(snap_u_n)
        print(f"  snapshot at step {args.n1}: E={e_snap:.6e}", flush=True)
        # continue continuous for n2, record E
        _, _, E_cont, ok = march(u_n, u_nm1, args.n2, 2, label="cont+")
        print("=== RESTART run (from snapshot, n2 steps) ===", flush=True)
        u_n2, u_nm12, E_rst, ok2 = march(snap_u_n, snap_u_nm1, args.n2, 2,
                                         label="restart")
        # compare
        ec = np.array([e for _, e in E_cont]); er = np.array([e for _, e in E_rst])
        if ec.shape == er.shape and ec.size:
            rel = np.max(np.abs(ec - er) / (np.abs(ec) + 1e-30))
            print(f"=== restart vs continuous enstrophy: max rel diff = {rel:.3e} ===",
                  flush=True)
            print("  -> RESTART EXACT" if rel < 1e-10 else
                  ("  -> restart close (rel<1e-6)" if rel < 1e-6 else
                   "  -> RESTART MISMATCH (investigate)"), flush=True)
        return

    # precursor: long march to shedding, save restart state
    print(f"=== PRECURSOR: rest -> {args.n1} steps (T={args.dt*args.n1:.2f}) ===",
          flush=True)
    t0 = time.time()
    u_n = jnp.zeros(n_u); u_nm1 = jnp.zeros(n_u)
    u_n, u_nm1, E, ok = march(u_n, u_nm1, args.n1, 1, label="prec")
    if not ok:
        print("PRECURSOR BLEW UP", flush=True); return
    # np.savez auto-appends ".npz"; write to a sibling .npz then rename
    # (atomic) so the final path is exactly --restart-file.
    tmp = args.restart_file.parent / (args.restart_file.stem + "_tmp.npz")
    np.savez(tmp, u_n=np.asarray(u_n), u_nm1=np.asarray(u_nm1),
             order=args.order, kx=args.kx, ky=args.ky, n_radial=args.n_radial,
             r_outer=args.r_outer, dt=args.dt, re=Re, nu=nu, n1=args.n1,
             E_series=np.array(E))
    tmp.replace(args.restart_file)
    print(f"  saved restart state -> {args.restart_file}  "
          f"(wall {time.time()-t0:.1f}s)", flush=True)
    # report shedding signature: last few enstrophy values (oscillating => shedding)
    Es = np.array([e for _, e in E])
    if Es.size >= 6:
        tail = Es[-6:]
        print(f"  enstrophy tail: {np.array2string(tail, precision=4)}", flush=True)
        amp = (tail.max() - tail.min()) / (tail.mean() + 1e-30)
        print(f"  tail rel oscillation amplitude = {amp:.3f}  "
              f"({'SHEDDING' if amp > 0.02 else 'steady/decaying — raise Re or T'})",
              flush=True)


if __name__ == "__main__":
    main()
