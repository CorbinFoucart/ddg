"""Animation helpers for 2D DG fields on triangle meshes.

Each macro-element carries an (N+1)(N+2)/2-node polynomial. For
visualisation we sub-triangulate the reference element once (Delaunay
on (r, s)) and replicate that node-level connectivity across every
macro-element. The resulting ``matplotlib.tri.Triangulation`` is dense
enough to ``tripcolor`` at high N without the per-frame cost of
re-sampling onto a structured grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay


def _reference_subtriangulation(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Delaunay sub-triangulation of the reference element's Np nodes."""
    return Delaunay(np.stack([np.asarray(r), np.asarray(s)], axis=1)).simplices


def _global_tri_indices(
    Np: int, K: int, ref_simplices: np.ndarray
) -> np.ndarray:
    """Stack per-element triangulations into a global ``(K * Nt_ref, 3)`` array
    in column-major node order (matches ``flatten(order='F')`` on ``(Np, K)``)."""
    Nt_ref = ref_simplices.shape[0]
    out = np.empty((K * Nt_ref, 3), dtype=np.int64)
    for k in range(K):
        out[k * Nt_ref:(k + 1) * Nt_ref] = ref_simplices + k * Np
    return out


def save_scalar_field_animation_2d(
    mesh,
    field_traj: np.ndarray,
    times: np.ndarray,
    output_path: Path | str,
    *,
    title: str = "",
    cmap: str = "RdBu_r",
    fps: int = 30,
    figsize: tuple[float, float] = (6.5, 6.0),
    symmetric: bool = True,
    show_mesh: bool = False,
) -> Path:
    """Render and save an MP4 of a scalar field over a 2D triangle mesh.

    ``field_traj`` has shape ``(n_frames+1, Np, K)``. We flatten to
    ``(n_frames+1, Np*K)`` in Fortran order to match the global node
    indices used in ``_global_tri_indices``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r = np.asarray(mesh.r)
    s = np.asarray(mesh.s)
    ref_simplices = _reference_subtriangulation(r, s)
    tris = _global_tri_indices(mesh.Np, mesh.K, ref_simplices)

    x_flat = np.asarray(mesh.x).flatten(order="F")
    y_flat = np.asarray(mesh.y).flatten(order="F")
    triang = Triangulation(x_flat, y_flat, triangles=tris)

    field_flat = field_traj.transpose(0, 2, 1).reshape(field_traj.shape[0], -1)

    if symmetric:
        vmax = float(np.max(np.abs(field_flat)))
        vmin = -vmax
    else:
        vmin, vmax = float(np.min(field_flat)), float(np.max(field_flat))

    fig, ax = plt.subplots(figsize=figsize)
    tpc = ax.tripcolor(triang, field_flat[0], shading="gouraud",
                       cmap=cmap, vmin=vmin, vmax=vmax)
    if show_mesh:
        ax.triplot(triang, color="k", lw=0.15, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(y_flat.min(), y_flat.max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    time_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, family="monospace",
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def update(idx: int):
        tpc.set_array(field_flat[idx])
        time_text.set_text(f"t = {float(times[idx]):6.3f}")
        return tpc, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(times), blit=True, interval=1000 / fps
    )
    writer = animation.FFMpegWriter(
        fps=fps, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    anim.save(str(output_path), writer=writer, dpi=120)
    plt.close(fig)
    return output_path
