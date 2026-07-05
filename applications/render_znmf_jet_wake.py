"""Render baseline-vs-controlled wake for the ZNMF synthetic-jet cylinder control.

Re-runs the forward window twice -- uncontrolled (theta=0) and with the
optimised jet schedule from output/znmf_jet_ctrl_cyl_proj_k<order>.json -- and
plots matched-time vorticity snapshots with the downstream enstrophy box (the
objective region) drawn as a dashed black rectangle.

Local preliminary: order 2, low-res restart.  Mirrors the solver setup in
scripts/znmf_jet_control_cyl_proj_2d.py exactly.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from diff_cylinder_dt_tau import (
    build_static, X_MIN, X_MAX, Y_MIN, Y_MAX, CX, CY, RADIUS, D_CYL,
    U_MAX, U_MEAN,
)
from inflow_control_cylinder_2d import make_vorticity_at_nodes_fn
from inflow_profile_control_cyl_proj_2d import make_proj_step_profile
from ddg.dg2d.coupled_ns_bdm import bdm1_evaluate_dirichlet_bc
from ddg.dg2d.bdm import bdm_k_evaluate_boundary_velocity

# ---- config: must match the optimisation run ------------------------------
ORDER, KX, KY = 2, 20, 10
N_RADIAL, R_OUTER = 2, 0.1
DT, RE, TAU_SCALE = 5.0e-4, 100.0, 4.0
N_MODES, C_MAX = 3, 0.6
N_WINDOW, N_CONTROL = 1200, 8
RESTART = ROOT / "output" / "cyl_shed_restart_lowres_k2.npz"
JSON_IN = ROOT / "output" / f"znmf_jet_ctrl_cyl_proj_k{ORDER}.json"
OUT_PNG = ROOT / "report" / "img" / "candidates" / "znmf_jet_wake_local.png"
OUT_NPZ = ROOT / "output" / "znmf_jet_wake_fields_k2.npz"

# snapshot fractions through the window (matched times, baseline vs controlled)
SNAP_FRACS = (0.5, 0.75, 1.0)

# --- CLI overrides so this runs for the k3 production config too ----------
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--order", type=int, default=ORDER)
_ap.add_argument("--n-window", type=int, default=N_WINDOW)
_ap.add_argument("--n-control", type=int, default=N_CONTROL)
_ap.add_argument("--restart", type=Path, default=RESTART)
_ap.add_argument("--json-in", type=Path, default=JSON_IN)
_ap.add_argument("--out-npz", type=Path, default=OUT_NPZ)
_ap.add_argument("--out-png", type=Path, default=OUT_PNG)
_A, _ = _ap.parse_known_args()
ORDER = _A.order
N_WINDOW = _A.n_window
N_CONTROL = _A.n_control
RESTART = _A.restart
JSON_IN = _A.json_in
OUT_NPZ = _A.out_npz
OUT_PNG = _A.out_png


def make_cyl_jet_mode(V, m, kind):
    def bc_fn(x, y):
        dx = x - CX
        dy = y - CY
        r = jnp.sqrt(dx * dx + dy * dy)
        on_cyl = r < RADIUS * 1.05
        th = jnp.arctan2(dy, dx)
        g = jnp.cos(m * th) if kind == "cos" else jnp.sin(m * th)
        rr = jnp.where(r > 1e-12, r, 1.0)
        amp = U_MAX * jnp.where(on_cyl, g, 0.0) / rr
        return amp * dx, amp * dy
    return (bdm1_evaluate_dirichlet_bc(V, bc_fn),
            bdm_k_evaluate_boundary_velocity(V, bc_fn))


def main():
    nu = U_MEAN * D_CYL / RE
    V, u_bc, u_bc_vec, bc_mask, is_natural_face, _ = build_static(
        KX, KY, ORDER, n_radial=N_RADIAL, r_outer=R_OUTER)

    jm_u, jm_v = [], []
    for m in range(1, N_MODES + 1):
        for kind in ("cos", "sin"):
            a_, b_ = make_cyl_jet_mode(V, m, kind)
            jm_u.append(a_); jm_v.append(b_)
    mode_u = jnp.stack(jm_u); mode_uvec = jnp.stack(jm_v)
    n_jet = mode_u.shape[0]

    step_fn = make_proj_step_profile(
        V, u_bc, u_bc_vec, mode_u, mode_uvec, bc_mask, is_natural_face,
        nu=nu, tau_scale=TAU_SCALE, dt=DT)
    step_j = jax.jit(step_fn, static_argnums=(3,))
    vort = jax.jit(make_vorticity_at_nodes_fn(V))

    rs = np.load(RESTART)
    u0 = jnp.asarray(rs["u_n"]); u0m = jnp.asarray(rs["u_nm1"])

    d = json.load(open(JSON_IN))
    theta = jnp.asarray(d["theta"])
    print(f"[render] loaded theta {tuple(theta.shape)} from {JSON_IN.name}; "
          f"baseline <ens>={d['enstrophy_baseline']:.2f}", flush=True)

    knot_of_step = np.minimum(
        (np.arange(N_WINDOW) * N_CONTROL) // N_WINDOW, N_CONTROL - 1).astype(int)

    # enstrophy box (objective region) -- identical to the driver
    x_np = np.asarray(V.mesh.x); y_np = np.asarray(V.mesh.y)
    in_box = ((x_np > CX + 1.5 * D_CYL) & (x_np < CX + 6.0 * D_CYL)
              & (y_np > Y_MIN + 0.02) & (y_np < Y_MAX - 0.02))
    J_node = np.tile(np.asarray(V.mesh.J[0])[None, :], (V.mesh.Np, 1))
    weight = in_box.astype(float) * J_node

    snap_steps = sorted({int(f * (N_WINDOW - 1)) for f in SNAP_FRACS})

    def run(amps_step, tag):
        a0 = amps_step[knot_of_step[0]]
        u_n, _ = step_j(u0, u0m, a0, 2)
        u_nm1 = u0
        snaps = {}
        ebox = np.empty(N_WINDOW)
        ebox[0] = float(jnp.sum(weight * vort(u_n) ** 2))
        for k in range(1, N_WINDOW):
            a = amps_step[knot_of_step[k]]
            u_new, _ = step_j(u_n, u_nm1, a, 2)
            u_nm1, u_n = u_n, u_new
            w = vort(u_n)
            ebox[k] = float(jnp.sum(weight * w ** 2))
            if k in snap_steps:
                snaps[k] = np.asarray(w)
            if k % 300 == 0:
                print(f"   {tag} step {k}/{N_WINDOW}  Ebox={ebox[k]:.1f}",
                      flush=True)
        print(f"  {tag}: <Ebox>={ebox.mean():.2f}", flush=True)
        return snaps, ebox

    amps_full = float(C_MAX) * np.tanh(np.asarray(theta))            # (nc, n_jet)
    amps_zero = np.zeros_like(amps_full)
    base_snaps, base_e = run(jnp.asarray(amps_zero), "baseline")
    ctrl_snaps, ctrl_e = run(jnp.asarray(amps_full), "controlled")

    px = x_np.ravel(); py = y_np.ravel()
    tri = mtri.Triangulation(px, py)
    vmax = float(np.percentile(np.abs(base_snaps[snap_steps[-1]]), 99.0))
    levels = np.linspace(-vmax, vmax, 41)

    np.savez(OUT_NPZ, px=px, py=py, snap_steps=np.array(snap_steps),
             base=np.stack([base_snaps[k] for k in snap_steps]),
             ctrl=np.stack([ctrl_snaps[k] for k in snap_steps]),
             base_e=base_e, ctrl_e=ctrl_e, vmax=vmax)

    bx0, bx1 = CX + 1.5 * D_CYL, CX + 6.0 * D_CYL
    by0, by1 = Y_MIN + 0.02, Y_MAX - 0.02
    xlim = (X_MIN, min(X_MAX, CX + 9 * D_CYL))

    nC = len(snap_steps)
    fig, axes = plt.subplots(2, nC, figsize=(4.3 * nC, 4.6), squeeze=False)
    rows = (("uncontrolled", base_snaps, base_e),
            ("ZNMF jet control", ctrl_snaps, ctrl_e))
    for r, (label, snaps, ev) in enumerate(rows):
        for c, k in enumerate(snap_steps):
            ax = axes[r][c]
            ax.tricontourf(tri, snaps[k].ravel(), levels=levels,
                           cmap="RdBu_r", extend="both")
            ax.add_patch(Circle((CX, CY), RADIUS, fc="0.6", ec="k", lw=0.6,
                                zorder=5))
            ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                   fill=False, ec="k", ls="--", lw=1.3,
                                   zorder=6))
            ax.set_xlim(*xlim); ax.set_ylim(Y_MIN, Y_MAX)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {k * DT:.2f}", fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
    red = 100.0 * (1.0 - ctrl_e.mean() / base_e.mean())
    fig.suptitle(f"Downstream wake enstrophy (box):  "
                 f"{base_e.mean():.1f} -> {ctrl_e.mean():.1f}   "
                 f"({red:.1f}% reduction)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"  wrote {OUT_PNG}", flush=True)
    print(f"  box enstrophy reduction: {red:.1f}%  "
          f"({base_e.mean():.2f} -> {ctrl_e.mean():.2f})", flush=True)


if __name__ == "__main__":
    main()
