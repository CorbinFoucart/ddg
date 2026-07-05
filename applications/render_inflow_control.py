"""Re-render the inflow-control wake plots + mp4 from the saved .npz.

The Adam loop in ``scripts/inflow_control_cylinder_2d.py`` dumps a
1.1 MB compressed .npz that holds the full per-step velocity / vorticity
fields for both the baseline and the optimised trajectories. This
script loads that .npz and re-renders the summary png + side-by-side
mp4 with adjustable vmin/vmax clipping so the wake is visible against
the cylinder boundary-layer spike.

Default clipping uses the 90th percentile of |omega| inside the
downstream window (the loss region), not the global field, so the
colour scale tracks what the optimiser actually cares about.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path,
                    default=Path("output/inflow_control_cylinder_k2.npz"))
    ap.add_argument("--json", type=Path,
                    default=Path("output/inflow_control_cylinder_k2.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--stem", type=str, default=None,
                    help="output stem; defaults to the .npz stem")
    ap.add_argument("--w-clip", type=float, default=None,
                    help="absolute |omega| clip for the colour scale; "
                         "if None, use --w-pct percentile of the "
                         "downstream-window magnitudes")
    ap.add_argument("--w-pct", type=float, default=90.0,
                    help="percentile inside the loss window used when "
                         "--w-clip is not given")
    ap.add_argument("--xlim", type=float, nargs=2, default=None,
                    help="x-axis bounds for the wake panels; "
                         "default = full channel")
    ap.add_argument("--ylim", type=float, nargs=2, default=None,
                    help="y-axis bounds for the wake panels; "
                         "default = full channel")
    args = ap.parse_args()
    stem = args.stem if args.stem else args.npz.stem

    d = np.load(args.npz)
    px = d["mesh_x"]; py = d["mesh_y"]
    alphas_opt = d["alphas_optimised"]
    alphas_base = d["alphas_baseline"]
    times = d["times"]
    loss_history = d["loss_history"]
    w_base = d["omega_baseline"]    # (n_steps, Np, K)
    w_opt = d["omega_optimised"]
    CX = float(d["CX"]); CY = float(d["CY"]); RADIUS = float(d["RADIUS"])
    X_MIN = float(d["X_MIN"]); X_MAX = float(d["X_MAX"])
    Y_MIN = float(d["Y_MIN"]); Y_MAX = float(d["Y_MAX"])
    D_CYL = float(d["D_CYL"])
    n_steps = w_base.shape[0]

    meta = json.loads(args.json.read_text()) if args.json.exists() else {}
    L_baseline = float(meta.get("loss_baseline", float("nan")))
    Re = float(meta.get("re", float("nan")))
    order = int(meta.get("order", 2))

    # Downstream-window mask (matches the script's loss region).
    x0, x1 = CX + 1.5 * D_CYL, CX + 6.0 * D_CYL
    y0, y1 = Y_MIN + 0.02, Y_MAX - 0.02
    in_win = (px > x0) & (px < x1) & (py > y0) & (py < y1)

    # Wider post-cylinder region (excludes inflow + side-wall BL) for
    # the colour-scale percentile. The downstream window alone is too
    # narrow when the channel wall BLs already enter it.
    post_cyl = (px > CX + 0.5 * D_CYL) & (px < X_MAX - 0.05) & (
        py > Y_MIN + 0.04) & (py < Y_MAX - 0.04)
    if args.w_clip is not None:
        w_clip = args.w_clip
    else:
        w_in_base = np.abs(w_base[-1][post_cyl])
        w_in_opt = np.abs(w_opt[-1][post_cyl])
        w_clip = max(np.percentile(w_in_base, args.w_pct),
                     np.percentile(w_in_opt, args.w_pct))
    print(f"clipping |omega| at {w_clip:.3f} "
          f"(p{args.w_pct:.0f} of downstream-window magnitudes)")

    # Colour-scale levels with extension at both ends so the BL spike
    # saturates but is still distinguishable from zero.
    levels = np.linspace(-w_clip, w_clip, 25)

    # ---------- summary PNG ----------
    px_flat = px.T.reshape(-1)
    py_flat = py.T.reshape(-1)
    theta_arc = np.linspace(0, 2 * np.pi, 80)
    cyl_x = CX + RADIUS * np.cos(theta_arc)
    cyl_y = CY + RADIUS * np.sin(theta_arc)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].step(times, alphas_opt, "b-", where="post", lw=2,
                    label="optimised")
    axes[0, 0].step(times, alphas_base, "k--", where="post", lw=1,
                    label="baseline (alpha=1)")
    axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel(r"$\alpha(t)$")
    axes[0, 0].set_title("Inflow multiplier")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    iters = np.arange(len(loss_history))
    axes[0, 1].semilogy(iters, loss_history, "k-")
    if np.isfinite(L_baseline):
        axes[0, 1].axhline(L_baseline, color="r", ls="--", lw=1,
                           label="baseline")
        axes[0, 1].legend()
    axes[0, 1].set_xlabel("Adam iter"); axes[0, 1].set_ylabel("loss")
    axes[0, 1].set_title("Adam loss"); axes[0, 1].grid(alpha=0.3, which="both")

    cf_last = None
    for ax, w_field, title in (
        (axes[1, 0], w_base[-1], "Baseline wake (alpha=1)"),
        (axes[1, 1], w_opt[-1], "Optimised wake"),
    ):
        pz = w_field.T.reshape(-1)
        cf_last = ax.tricontourf(px_flat, py_flat, pz, levels=levels,
                                 cmap="RdBu_r", extend="both")
        ax.fill(cyl_x, cyl_y, color="black",
                edgecolor="white", linewidth=1.2)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                "k-", lw=1, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(*(args.xlim if args.xlim else (X_MIN, X_MAX)))
        if args.ylim:
            ax.set_ylim(*args.ylim)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    cbar = fig.colorbar(cf_last, ax=axes[1, :], orientation="vertical",
                        fraction=0.025, pad=0.02)
    cbar.set_label(r"vorticity $\omega$  (clipped at $\pm$"
                   f"{w_clip:.3f})")

    fig.suptitle(
        f"Inflow control: BDM_{order} cylinder, Re={Re:.0f}, "
        f"T_final={float(times[-1]):.2f}, |omega| clip={w_clip:.2f}"
    )
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}")

    # ---------- side-by-side mp4 ----------
    from matplotlib.animation import FuncAnimation
    # Figure size chosen so the rendered video dimensions are even (h264 req).
    fig2, axes2 = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, constrained_layout=True,
    )
    # Static colour bar bound to the level range (no live update needed).
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    sm = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=-w_clip, vmax=w_clip), cmap="RdBu_r")
    sm.set_array([])
    cbar2 = fig2.colorbar(sm, ax=axes2, orientation="vertical",
                          fraction=0.025, pad=0.02)
    cbar2.set_label(r"vorticity $\omega$  (clipped)")

    def draw(i):
        for ax in axes2:
            ax.clear()
        wb = w_base[i].T.reshape(-1)
        wo = w_opt[i].T.reshape(-1)
        axes2[0].tricontourf(px_flat, py_flat, wb, levels=levels,
                             cmap="RdBu_r", extend="both")
        axes2[0].fill(cyl_x, cyl_y, color="black",
                      edgecolor="white", linewidth=1.2)
        axes2[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                      "k-", lw=1, alpha=0.5)
        axes2[0].set_aspect("equal")
        axes2[0].set_xlim(*(args.xlim if args.xlim else (X_MIN, X_MAX)))
        if args.ylim:
            axes2[0].set_ylim(*args.ylim)
        axes2[0].set_title(
            f"Baseline (alpha=1)  t={times[i]:.2f}  "
            f"alpha_t={float(alphas_base[i]):.2f}")
        axes2[1].tricontourf(px_flat, py_flat, wo, levels=levels,
                             cmap="RdBu_r", extend="both")
        axes2[1].fill(cyl_x, cyl_y, color="black",
                      edgecolor="white", linewidth=1.2)
        axes2[1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                      "k-", lw=1, alpha=0.5)
        axes2[1].set_aspect("equal")
        axes2[1].set_xlim(*(args.xlim if args.xlim else (X_MIN, X_MAX)))
        if args.ylim:
            axes2[1].set_ylim(*args.ylim)
        axes2[1].set_title(
            f"Optimised  t={times[i]:.2f}  "
            f"alpha_t={float(alphas_opt[i]):.2f}")
        return []

    anim = FuncAnimation(fig2, draw, frames=n_steps,
                         interval=500, blit=False)
    out_mp4 = args.out_dir / f"{stem}.mp4"
    # dpi=100 with figsize=(12, 6) -> 1200x600, both even, h264-compatible.
    anim.save(str(out_mp4), writer="ffmpeg", dpi=100)
    plt.close(fig2)
    print(f"  wrote {out_mp4}")


if __name__ == "__main__":
    main()
