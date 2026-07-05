"""Polished von-Karman wake movie from a cylinder snapshot dump.

Improves on the raw tricontourf renders: the (coarse, order-2) nodal
vorticity is interpolated onto a fine regular grid and shown with
bilinear imshow (no element faceting), the cylinder is masked, and
frames are temporally interpolated for a smooth loop. Writes an mp4 and
a downsampled GIF.

  python scripts/render_cylinder_wake_movie.py \
      output/cyl_v6/cyl_proj_v6_nr3_fixed --order 2 --tsub 3 \
      --out assets/cylinder_wake.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ddg.dg2d.mesh import build_mesh_2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="path stem (expects _mesh.npz/_snapshots.npz)")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--tsub", type=int, default=3, help="temporal upsample factor")
    ap.add_argument("--nx", type=int, default=900, help="render grid width")
    ap.add_argument("--pctl", type=float, default=96.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--cmap", default="RdBu_r")
    ap.add_argument("--t0", type=float, default=1.0, help="start time (skip startup)")
    ap.add_argument("--out", default="assets/cylinder_wake.gif")
    args = ap.parse_args()

    m = np.load(args.stem + "_mesh.npz")
    s = np.load(args.stem + "_snapshots.npz")
    Dr, Ds = np.asarray(m["Dr"]), np.asarray(m["Ds"])
    rx, ry, sx, sy = (np.asarray(m[k]) for k in ("rx", "ry", "sx", "sy"))
    x, y = np.asarray(m["x"]), np.asarray(m["y"])
    CX, CY, R = float(m["CX"]), float(m["CY"]), float(m["RADIUS"])
    X0, X1 = float(m["X_MIN"]), float(m["X_MAX"])
    Y0, Y1 = float(m["Y_MIN"]), float(m["Y_MAX"])
    t = np.asarray(s["t"]); ux = np.asarray(s["ux"]); uy = np.asarray(s["uy"])

    # vorticity per snapshot (nodal)
    W = np.empty_like(ux)
    for k in range(len(ux)):
        uxk, uyk = ux[k], uy[k]
        W[k] = ((rx * (Dr @ uyk) + sx * (Ds @ uyk))
                - (ry * (Dr @ uxk) + sy * (Ds @ uxk)))             # (nf,Np,K)

    keep = t >= args.t0
    W = W[keep]; tt = t[keep]
    tri = mtri.Triangulation(x.T.reshape(-1), y.T.reshape(-1))

    # fine render grid, aspect-preserving
    ny = max(2, int(args.nx * (Y1 - Y0) / (X1 - X0)))
    gx = np.linspace(X0, X1, args.nx); gy = np.linspace(Y0, Y1, ny)
    GX, GY = np.meshgrid(gx, gy)
    inside_cyl = (GX - CX) ** 2 + (GY - CY) ** 2 <= (R * 1.02) ** 2

    print(f"interpolating {len(W)} frames -> grid {args.nx}x{ny} ...", flush=True)
    grids = []
    for k in range(len(W)):
        interp = mtri.LinearTriInterpolator(tri, W[k].T.reshape(-1))
        g = np.asarray(interp(GX, GY))
        g = np.where(inside_cyl, np.nan, g)
        grids.append(g)
    grids = np.array(grids)                                    # (nf, ny, nx)
    vmax = float(np.nanpercentile(np.abs(grids), args.pctl))

    # temporal upsample by linear blending
    nf = len(grids)
    fine_idx = np.linspace(0, nf - 1, (nf - 1) * args.tsub + 1)
    cmap = plt.get_cmap(args.cmap).copy(); cmap.set_bad("0.82")

    fig, ax = plt.subplots(figsize=(11, 11 * (Y1 - Y0) / (X1 - X0)))
    fig.subplots_adjust(0, 0, 1, 1)
    im = ax.imshow(grids[0], origin="lower", extent=[X0, X1, Y0, Y1],
                   cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="bilinear",
                   aspect="equal")
    th = np.linspace(0, 2 * np.pi, 80)
    ax.fill(CX + R * np.cos(th), CY + R * np.sin(th), color="0.15", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.012, 0.93, "flow past cylinder  $Re=100$   (vorticity)",
            transform=ax.transAxes, fontsize=12, color="0.1", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))
    tlab = ax.text(0.988, 0.93, "", transform=ax.transAxes, fontsize=12,
                   color="0.1", va="top", ha="right",
                   bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))

    def draw(j):
        f = fine_idx[j]; i0 = int(np.floor(f)); a = f - i0
        i1 = min(i0 + 1, nf - 1)
        im.set_data((1 - a) * grids[i0] + a * grids[i1])
        ti = (1 - a) * tt[i0] + a * tt[i1]
        tlab.set_text(f"$t={ti:.1f}$")
        return [im, tlab]

    print(f"rendering {len(fine_idx)} frames ...", flush=True)
    anim = FuncAnimation(fig, draw, frames=len(fine_idx), blit=False)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gif":
        anim.save(str(out), writer="pillow", fps=args.fps, dpi=70)
    else:
        anim.save(str(out), writer="ffmpeg", fps=args.fps, dpi=120)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
