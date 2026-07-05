"""Standalone replot for cylinder runs from saved snapshots.

Loads ``{stem}_mesh.npz`` and ``{stem}_snapshots.npz`` (produced by
``ns_bdm_cylinder_projection_2d.py``) and renders an mp4 + a set of
still frames at user-chosen times. Decoupled from the solver so plot
style can be iterated without re-running.

Usage
-----
::

    # mp4 from the full snapshot list at default settings:
    python applications/cylinder_replot.py output/cyl_proj_shedding

    # add still frames at given times (best for the report figure):
    python applications/cylinder_replot.py output/cyl_proj_shedding \\
        --still-times 2.0 4.0 6.0 8.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _nodal_vorticity_np(Dr, Ds, rx, ry, sx, sy, ux, uy):
    ux_r = Dr @ ux; ux_s = Ds @ ux
    uy_r = Dr @ uy; uy_s = Ds @ uy
    duydx = rx * uy_r + sx * uy_s
    duxdy = ry * ux_r + sy * ux_s
    return duydx - duxdy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", type=str,
                    help="Path stem (without _mesh.npz / _snapshots.npz)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory; defaults to stem's directory")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--xlim", type=float, nargs=2, default=None,
                    help="x-axis crop; defaults to mesh bounding box")
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    ap.add_argument("--w-percentile", type=float, default=98.0,
                    help="vorticity colour-scale percentile")
    ap.add_argument("--s-percentile", type=float, default=99.0)
    ap.add_argument("--still-times", type=float, nargs="*", default=None,
                    help="Render PNG frames at these times")
    ap.add_argument("--skip", type=int, default=1,
                    help="Render every N-th snapshot for the mp4")
    args = ap.parse_args()

    stem = Path(args.stem)
    out_dir = args.out_dir if args.out_dir is not None else stem.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem.name

    mesh_path = stem.with_name(base + "_mesh.npz")
    snap_path = stem.with_name(base + "_snapshots.npz")
    m = np.load(mesh_path)
    s = np.load(snap_path)
    x = m["x"]; y = m["y"]
    Dr = m["Dr"]; Ds = m["Ds"]
    rx = m["rx"]; ry = m["ry"]; sx = m["sx"]; sy = m["sy"]
    CX = float(m["CX"]); CY = float(m["CY"]); RADIUS = float(m["RADIUS"])

    t = s["t"]
    uxs = s["ux"].astype(np.float64)
    uys = s["uy"].astype(np.float64)
    n_snap = len(t)
    print(f"  {n_snap} snapshots, K={x.shape[1]}, Np={x.shape[0]}, "
          f"t in [{t.min():.3f}, {t.max():.3f}]", flush=True)

    px = x.T.reshape(-1); py = y.T.reshape(-1)
    theta = np.linspace(0, 2 * np.pi, 80)
    cyl_x = CX + RADIUS * np.cos(theta)
    cyl_y = CY + RADIUS * np.sin(theta)

    print("  precomputing vorticity + speed ...", flush=True)
    omegas = np.empty((n_snap, x.shape[0], x.shape[1]), dtype=np.float32)
    speeds = np.empty_like(omegas)
    for i in range(n_snap):
        omegas[i] = _nodal_vorticity_np(
            Dr, Ds, rx, ry, sx, sy, uxs[i], uys[i],
        ).astype(np.float32)
        speeds[i] = np.sqrt(uxs[i] ** 2 + uys[i] ** 2).astype(np.float32)

    w_max = float(np.percentile(np.abs(omegas), args.w_percentile))
    s_max = float(np.percentile(speeds, args.s_percentile))
    levels_w = np.linspace(-w_max, w_max, 31)
    levels_s = np.linspace(0, s_max, 21)
    print(f"  w_max={w_max:.2f}  s_max={s_max:.2f}", flush=True)

    xlim = args.xlim if args.xlim is not None else (
        float(px.min()), float(px.max())
    )
    ylim = args.ylim if args.ylim is not None else (
        float(py.min()), float(py.max())
    )

    def _draw(ax_w, ax_s, i, t_i):
        ax_w.clear(); ax_s.clear()
        w_i = omegas[i].T.reshape(-1)
        s_i = speeds[i].T.reshape(-1)
        ax_w.tricontourf(px, py, w_i, levels=levels_w,
                          cmap="RdBu_r", extend="both")
        ax_w.fill(cyl_x, cyl_y, color="black")
        ax_w.set_aspect("equal")
        ax_w.set_xlim(xlim); ax_w.set_ylim(ylim)
        ax_w.set_title(f"vorticity $\\omega$    t = {t_i:.2f}")
        ax_s.tricontourf(px, py, s_i, levels=levels_s,
                          cmap="viridis", extend="max")
        ax_s.fill(cyl_x, cyl_y, color="black")
        ax_s.set_aspect("equal")
        ax_s.set_xlim(xlim); ax_s.set_ylim(ylim)
        ax_s.set_title("speed $|U|$")

    # --- still frames ---
    if args.still_times:
        print(f"  rendering {len(args.still_times)} still frames ...",
              flush=True)
        for t_target in args.still_times:
            i = int(np.argmin(np.abs(t - t_target)))
            fig, (ax_w, ax_s) = plt.subplots(
                2, 1, figsize=(13, 5), sharex=True,
                constrained_layout=True,
            )
            _draw(ax_w, ax_s, i, float(t[i]))
            out_png = out_dir / f"{base}_t{float(t[i]):.2f}.png"
            fig.savefig(out_png, dpi=args.dpi)
            plt.close(fig)
            print(f"    {out_png}", flush=True)

    # --- mp4 ---
    print("  rendering mp4 ...", flush=True)
    from matplotlib.animation import FuncAnimation
    idxs = list(range(0, n_snap, args.skip))
    fig2, (ax_w, ax_s) = plt.subplots(
        2, 1, figsize=(13, 5), sharex=True, constrained_layout=True,
    )

    def update(j):
        i = idxs[j]
        _draw(ax_w, ax_s, i, float(t[i]))
        return []

    anim = FuncAnimation(
        fig2, update, frames=len(idxs),
        interval=int(1000 / args.fps), blit=False,
    )
    out_mp4 = out_dir / f"{base}.mp4"
    anim.save(str(out_mp4), writer="ffmpeg", dpi=args.dpi, fps=args.fps)
    plt.close(fig2)
    print(f"  wrote {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
