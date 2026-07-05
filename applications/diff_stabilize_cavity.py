"""Differentiable stabilization probe for the BDM_3 cavity blowup.

The Re=10000, K_x=8, dt=2.0, BDM_3 transient cavity blows up on
step 1 (KE = 2.5e+95): the convective CFL is ~16, BDF2 + Newton
finds a high-amplitude root rather than the physical one. This
script tests whether bumping the SIPG penalty tau_scale (more
discrete numerical damping) closes the failure boundary.

Two phases:

1. **Finite-difference scan** over tau_scale in {4, 16, 64, 256, 1024}
   at the failure config. Run a short window (1-2 BDF2 steps), record
   final KE / enstrophy / div violation, classify as "blow" vs
   "bounded". The bounded threshold for KE is 1.0 -- physically the
   cavity peaks at KE ~ 0.01, so anything bigger than 1 is
   numerically broken.

2. **Smaller dt probe** for comparison: re-run baseline tau=4 with
   dt=0.1, 0.2, 0.5, 1.0 to map the bare-knob failure boundary.

The two scans together establish whether tau_scale can save the
large-dt run, or whether we need a different stabilisation knob.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "applications"))
from ns_bdm_cavity_transient_2d import run_transient_cavity


def classify(result):
    """Classify a run as 'bounded' / 'blown' / 'nan'."""
    ke = np.asarray(result["kinetic_energy"])
    enstr = np.asarray(result["enstrophy"])
    if not np.all(np.isfinite(ke)):
        return "nan", float("nan"), float("nan")
    ke_max = float(ke.max())
    enstr_max = float(enstr.max())
    if ke_max > 1.0 or enstr_max > 1.0e3:
        return "blown", ke_max, enstr_max
    return "bounded", ke_max, enstr_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=10000.0)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=8)
    ap.add_argument("--tfinal", type=float, default=4.0,
                    help="two BDF2 steps at dt=2.0")
    ap.add_argument("--dt-bad", type=float, default=2.0,
                    help="the unstable dt at which tau is varied")
    ap.add_argument("--max-outer", type=int, default=30)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    base = dict(
        Re=args.re, Kx=args.kx, Ky=args.kx,
        order=args.order,
        bdf_order=2,
        snap_every=1000,
        tol_outer=1e-8, tol_inner=1e-8,
        max_outer=args.max_outer,
        linearisation="newton",
        use_iterative=False,
        verbose=False,
    )

    # Phase 1: tau sweep at the failure dt
    print(f"=== Phase 1: tau sweep at dt={args.dt_bad}, "
          f"T={args.tfinal} (Re={args.re}, BDM_{args.order}, "
          f"K_x={args.kx}) ===", flush=True)
    tau_values = [4.0, 16.0, 64.0, 256.0, 1024.0]
    phase1 = []
    for tau in tau_values:
        t0 = time.time()
        try:
            r = run_transient_cavity(
                dt=args.dt_bad, t_final=args.tfinal,
                tau_scale=tau,
                **base,
            )
            status, ke_max, enstr_max = classify(r)
            row = dict(
                tau_scale=float(tau), dt=float(args.dt_bad),
                status=status, ke_max=ke_max, enstr_max=enstr_max,
                wall=time.time() - t0,
            )
        except Exception as exc:
            row = dict(tau_scale=float(tau), dt=float(args.dt_bad),
                       status=f"error: {exc!r}",
                       wall=time.time() - t0)
        phase1.append(row)
        print(f"  tau={tau:7.1f}  status={row['status']:8s}  "
              f"ke_max={row.get('ke_max', float('nan')):.3e}  "
              f"wall={row['wall']:5.1f}s",
              flush=True)

    # Phase 2: dt sweep at baseline tau
    print(f"\n=== Phase 2: dt sweep at tau=4 (Re={args.re}) ===",
          flush=True)
    dt_values = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    phase2 = []
    for dt in dt_values:
        n_steps = max(2, int(round(args.tfinal / dt)))
        t_final = n_steps * dt
        t0 = time.time()
        try:
            r = run_transient_cavity(
                dt=dt, t_final=t_final, tau_scale=4.0,
                **base,
            )
            status, ke_max, enstr_max = classify(r)
            row = dict(
                tau_scale=4.0, dt=float(dt),
                t_final=float(t_final), n_steps=int(n_steps),
                status=status, ke_max=ke_max, enstr_max=enstr_max,
                wall=time.time() - t0,
            )
        except Exception as exc:
            row = dict(tau_scale=4.0, dt=float(dt),
                       status=f"error: {exc!r}",
                       wall=time.time() - t0)
        phase2.append(row)
        print(f"  dt={dt:5.2f}  status={row['status']:8s}  "
              f"ke_max={row.get('ke_max', float('nan')):.3e}  "
              f"n_steps={row.get('n_steps', '--')}  "
              f"wall={row['wall']:5.1f}s",
              flush=True)

    # Combined plot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # Phase 1: tau scan at fixed bad dt.
    taus = [r["tau_scale"] for r in phase1]
    ke_maxes_p1 = [r.get("ke_max", float("nan")) for r in phase1]
    statuses_p1 = [r["status"] for r in phase1]
    colors_p1 = ["C2" if s == "bounded" else "C3" for s in statuses_p1]
    axes[0].bar(range(len(taus)), [max(k, 1e-3) for k in ke_maxes_p1],
                  color=colors_p1, log=True)
    axes[0].set_xticks(range(len(taus)))
    axes[0].set_xticklabels([f"{t:g}" for t in taus])
    axes[0].axhline(1.0, color="k", ls="--", alpha=0.5,
                     label="bounded threshold")
    axes[0].set_xlabel(r"$\tau_{\rm scale}$")
    axes[0].set_ylabel("max KE over run")
    axes[0].set_title(
        f"Phase 1: tau sweep at dt={args.dt_bad}, Re={args.re}\n"
        "green=bounded, red=blown"
    )
    axes[0].legend(); axes[0].grid(alpha=0.3, axis="y", which="both")
    # Phase 2: dt scan at baseline tau.
    dts = [r["dt"] for r in phase2]
    ke_maxes_p2 = [r.get("ke_max", float("nan")) for r in phase2]
    statuses_p2 = [r["status"] for r in phase2]
    colors_p2 = ["C2" if s == "bounded" else "C3" for s in statuses_p2]
    axes[1].bar(range(len(dts)), [max(k, 1e-3) for k in ke_maxes_p2],
                  color=colors_p2, log=True)
    axes[1].set_xticks(range(len(dts)))
    axes[1].set_xticklabels([f"{d:g}" for d in dts])
    axes[1].axhline(1.0, color="k", ls="--", alpha=0.5)
    axes[1].set_xlabel(r"$\Delta t$")
    axes[1].set_ylabel("max KE over run")
    axes[1].set_title(
        f"Phase 2: dt sweep at tau=4, Re={args.re}\n"
        "green=bounded, red=blown"
    )
    axes[1].grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    out = args.out_dir / "diff_stabilize_cavity.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}", flush=True)

    summary = dict(
        re=float(args.re), order=int(args.order),
        kx=int(args.kx), tfinal=float(args.tfinal),
        dt_bad=float(args.dt_bad),
        max_outer=int(args.max_outer),
        phase1_tau_sweep=phase1,
        phase2_dt_sweep=phase2,
    )
    with open(args.out_dir / "diff_stabilize_cavity.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'diff_stabilize_cavity.json'}",
          flush=True)


if __name__ == "__main__":
    main()
