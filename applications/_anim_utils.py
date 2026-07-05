"""Animation helpers for DG fields.

Each element carries its own polynomial on its own nodes, so plotting a
DG solution honestly means drawing per-element polylines with breaks
between elements. We do that by interleaving the ``(Np, K)`` nodal
arrays with rows of ``NaN`` and flattening column-major; matplotlib
drops segments connecting a NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def dg_plot_arrays(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``(Np, K)`` nodal data into plot-ready arrays with NaN breaks."""
    x = np.asarray(x)
    u = np.asarray(u)
    Np, K = x.shape
    pad_x = np.vstack([x, np.full((1, K), np.nan)])
    pad_u = np.vstack([u, np.full((1, K), np.nan)])
    return pad_x.flatten(order="F")[:-1], pad_u.flatten(order="F")[:-1]


def save_field_animation(
    x: np.ndarray,
    fields: Iterable[tuple[str, np.ndarray, str]],
    times: np.ndarray,
    output_path: Path | str,
    *,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "u",
    fps: int = 30,
    figsize: tuple[float, float] = (9.0, 4.0),
) -> Path:
    """Render an animation of one or more fields and save as MP4.

    ``fields`` is an iterable of ``(label, traj, color)`` where ``traj``
    has shape ``(n_frames, Np, K)``. All trajectories must share the
    same time axis ``times`` (length ``n_frames``).
    """
    fields = list(fields)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(times)
    x_flat = dg_plot_arrays(x, np.asarray(fields[0][1])[0])[0]

    fig, ax = plt.subplots(figsize=figsize)
    lines = []
    y_min, y_max = np.inf, -np.inf
    for label, traj, color in fields:
        traj = np.asarray(traj)
        y_min = min(y_min, float(np.nanmin(traj)))
        y_max = max(y_max, float(np.nanmax(traj)))
        line, = ax.plot([], [], color=color, lw=1.5, label=label)
        lines.append((line, traj))
    pad = 0.1 * max(abs(y_min), abs(y_max), 1e-12)
    ax.set_xlim(float(np.min(x_flat[~np.isnan(x_flat)])),
                float(np.max(x_flat[~np.isnan(x_flat)])))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(fields) > 1:
        ax.legend(loc="upper right")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        family="monospace", fontsize=10)

    def update(frame_idx: int):
        for line, traj in lines:
            _, y_flat = dg_plot_arrays(x, traj[frame_idx])
            line.set_data(x_flat, y_flat)
        time_text.set_text(f"t = {float(times[frame_idx]):6.3f}")
        artists = [time_text] + [ln for ln, _ in lines]
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=True, interval=1000 / fps
    )
    # Force QuickTime-compatible H.264: 4:2:0 chroma subsampling, faststart
    # moov atom for progressive playback. Without these matplotlib's default
    # picks `yuv444p` which QuickTime refuses to decode.
    writer = animation.FFMpegWriter(
        fps=fps,
        bitrate=2400,
        codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-movflags", "+faststart",
        ],
    )
    anim.save(str(output_path), writer=writer, dpi=120)
    plt.close(fig)
    return output_path
