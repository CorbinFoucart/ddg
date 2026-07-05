"""Render an mp4 from a transient-cavity .npz snapshot file.

Reads the output of ``ns_bdm_cavity_transient_2d.py`` and produces an
animation of the velocity speed + streamlines on a regular plot grid.
Separate from the simulation script so renders can be re-tweaked
without re-running the (expensive) simulation.

Usage:
    JAX_PLATFORMS=cpu python applications/ns_bdm_cavity_render.py \\
        output/bdm_cavity_transient_re100_kx16_*.npz

The mp4 lands next to the input npz with the same stem.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM, bdm1_basis_at_points, bdm1_gather,
    _element_jacobian_matrices,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


U_LID = 1.0


def _build_cavity_mesh(Kx: int, Ky: int):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, 1)
    V = BDM.from_mesh(mesh)
    return V


def _build_fast_evaluator(V: BDM, x_grid: np.ndarray, y_grid: np.ndarray):
    """Precompute (element index, reference (r, s), Piola-mapped basis)
    for each grid point. Returns a callable that maps a global DOF
    vector to the (Nx, Ny, 2) velocity field on the grid.

    The expensive step (containment search + basis eval) runs once;
    each snapshot only needs einsums.
    """
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX)
    VY = np.asarray(V.mesh.VY)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]

    Nx, Ny = x_grid.shape
    n_pts = Nx * Ny
    flat_x = x_grid.ravel()
    flat_y = y_grid.ravel()

    elem_of = np.full(n_pts, -1, dtype=np.int64)
    r_of = np.zeros(n_pts)
    s_of = np.zeros(n_pts)
    for qi, (xq, yq) in enumerate(zip(flat_x, flat_y)):
        for k in range(V.K):
            v0 = (VX[EToV[k, 0]], VY[EToV[k, 0]])
            v1 = (VX[EToV[k, 1]], VY[EToV[k, 1]])
            v2 = (VX[EToV[k, 2]], VY[EToV[k, 2]])
            det = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
            a = ((v1[0] - xq) * (v2[1] - yq) - (v2[0] - xq) * (v1[1] - yq)) / det
            b = ((v2[0] - xq) * (v0[1] - yq) - (v0[0] - xq) * (v2[1] - yq)) / det
            c = 1.0 - a - b
            if a >= -1e-9 and b >= -1e-9 and c >= -1e-9:
                elem_of[qi] = k
                r_of[qi] = 2.0 * b - 1.0
                s_of[qi] = 2.0 * c - 1.0
                break
    # Evaluate basis at the (r, s) points; we'll dispatch per element.
    phi_ref_at = np.asarray(bdm1_basis_at_points(
        jnp.asarray(r_of), jnp.asarray(s_of),
    ))                                          # (n_pts, 6, 2)
    # Piola map: phi_phys[q, l, c] = (1/J_K) DF_K[c, b] * phi_ref[q, l, b]
    # for K = elem_of[q].
    DF_per_pt = np.asarray(DF)[elem_of]         # (n_pts, 2, 2)
    J_per_pt = np.asarray(J)[elem_of]           # (n_pts,)
    phi_phys = np.einsum("qcb,qlb->qlc", DF_per_pt, phi_ref_at) / J_per_pt[:, None, None]
    # (n_pts, 6, 2)
    elem_of_jnp = jnp.asarray(elem_of)
    phi_phys_jnp = jnp.asarray(phi_phys)

    def evaluate(u_global: jax.Array) -> jax.Array:
        u_local = bdm1_gather(V.topology, u_global)              # (K, 6)
        u_local_at_pt = u_local[elem_of_jnp]                     # (n_pts, 6)
        u_xy = jnp.einsum("ql,qlc->qc", u_local_at_pt, phi_phys_jnp)
        return u_xy.reshape(Nx, Ny, 2)

    return evaluate


def render_mp4(npz_path: Path, out_path: Path | None = None,
                grid_n: int = 60, fps: int = 12):
    """Render a speed + streamlines mp4 from a transient cavity .npz.

    grid_n      number of plot grid points per side
    fps         animation framerate
    """
    data = np.load(npz_path)
    Kx, Ky = int(data["Kx"]), int(data["Ky"])
    V = _build_cavity_mesh(Kx, Ky)

    # Plot grid: avoid the corners where the BC is discontinuous.
    pad = 0.5 / grid_n
    xs = np.linspace(pad, 1.0 - pad, grid_n)
    ys = np.linspace(pad, 1.0 - pad, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    print(f"Building fast evaluator on {grid_n}x{grid_n} grid (one-time cost)...")
    t0 = time.time()
    evaluate = _build_fast_evaluator(V, X, Y)
    print(f"  ready in {time.time() - t0:.1f}s")

    snapshots = data["snapshots_u"]
    n_frames = snapshots.shape[0]
    times = data["times"][data["snap_indices"] - 1]
    print(f"Rendering {n_frames} frames...")

    # Precompute all velocity fields (small memory at 60x60).
    fields = []
    t0 = time.time()
    for u_global in snapshots:
        u_xy = np.asarray(evaluate(jnp.asarray(u_global)))
        fields.append(u_xy)
    print(f"  velocity evaluations done in {time.time() - t0:.1f}s")

    speed_max = max(float(np.linalg.norm(f, axis=-1).max()) for f in fields)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    Re = float(data["Re"])
    dt = float(data["dt"])
    img = ax.pcolormesh(X, Y, np.linalg.norm(fields[0], axis=-1),
                          shading="auto", cmap="viridis",
                          vmin=0.0, vmax=speed_max)
    fig.colorbar(img, ax=ax, label="|u|")
    # Streamlines (recomputed each frame).
    sl = ax.streamplot(X.T, Y.T, fields[0][..., 0].T, fields[0][..., 1].T,
                        color="w", linewidth=0.5, density=1.2)
    title = ax.set_title(f"BDM transient cavity Re={Re:.0f}, t={times[0]:.3f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def update(i):
        nonlocal sl
        f = fields[i]
        speed = np.linalg.norm(f, axis=-1)
        img.set_array(speed.ravel())
        # streamplot doesn't support in-place update; replot.
        for coll in list(ax.collections):
            if coll is img:
                continue
            coll.remove()
        for patch in list(ax.patches):
            patch.remove()
        sl = ax.streamplot(X.T, Y.T, f[..., 0].T, f[..., 1].T,
                            color="w", linewidth=0.5, density=1.2)
        title.set_text(f"BDM transient cavity Re={Re:.0f}, t={times[i]:.3f}")
        return [img, title]

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False,
    )
    if out_path is None:
        out_path = npz_path.with_suffix("").with_suffix(".mp4")
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    print(f"Writing {out_path}...")
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"  done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--grid-n", type=int, default=60)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    render_mp4(args.npz, out_path=args.out,
                grid_n=args.grid_n, fps=args.fps)


if __name__ == "__main__":
    main()
