"""2D analog of the shape-optimisation demo.

Mesh is a structured rectangular grid of ``Kx x Ky`` cells split into
right-triangles. The free parameters are ``Kx + Ky`` element widths
(one per row/column of cells), softmax-reparameterised so the mesh
stays non-degenerate. The cost is the L2 interpolation error of a
sharp 2D Gaussian; gradients flow through ``update_mesh_2d_vertices``
via JAX.

Outputs (``output/``)
---------------------
* ``shape_opt_2d.png``: static summary — target heatmap, uniform and
  optimised meshes overlaid, and the loss + gain curves.
* ``shape_opt_2d.mp4``: animation of the optimisation in progress.
* ``shape_opt_2d.json``: numerical record (initial / final widths,
  loss history, settings).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from ddg.dg1d.reference import jacobi_gauss_quadrature  # noqa: E402
from ddg.dg2d import (  # noqa: E402
    build_mesh_2d,
    uniform_rectangle_mesh_2d,
    update_mesh_2d_vertices,
    vandermonde_2d,
)


XMIN, XMAX = -1.0, 1.0
YMIN, YMAX = -1.0, 1.0
X0, Y0 = 0.3, 0.2          # peak location
SIGMA = 0.10               # peak width


def target(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.exp(-((x - X0) ** 2 + (y - Y0) ** 2) / (2.0 * SIGMA**2))


# ---------------------------------------------------------------------------
# Mesh parameterisation: softmax over per-row/column widths
# ---------------------------------------------------------------------------

def widths_from_logits(logits: jax.Array, length: float) -> jax.Array:
    return jax.nn.softmax(logits) * length


def lines_from_widths(widths: jax.Array, x0: float) -> jax.Array:
    return jnp.concatenate([jnp.array([x0]), x0 + jnp.cumsum(widths)])


def tensor_product_vertices(
    x_lines: jax.Array, y_lines: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """``VX, VY`` for a tensor-product rectangle mesh from 1D line positions."""
    Nx = x_lines.shape[0]
    Ny = y_lines.shape[0]
    VX = jnp.tile(x_lines, Ny)
    VY = jnp.repeat(y_lines, Nx)
    return VX, VY


# ---------------------------------------------------------------------------
# Triangle sub-quadrature (Duffy map of a Gauss-Gauss product on the square)
# ---------------------------------------------------------------------------

def triangle_sub_quadrature(
    n_per_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(r_sub, s_sub, w_sub)`` for the standard reference triangle.

    Tensor product of two ``(n_per_dim + 1)``-point Gauss-Legendre rules
    on ``[-1, 1]``, mapped through the Duffy transform ``(a, b) ->
    (r, s)``. The Jacobian ``0.5 (1 - b)`` is baked into the weights so
    ``sum(w) = 2`` (the reference-triangle area).
    """
    a_pts, a_w = jacobi_gauss_quadrature(0.0, 0.0, n_per_dim)
    b_pts, b_w = jacobi_gauss_quadrature(0.0, 0.0, n_per_dim)
    aa, bb = jnp.meshgrid(a_pts, b_pts, indexing="ij")
    r_sub = 0.5 * (1.0 + aa) * (1.0 - bb) - 1.0
    s_sub = bb
    w_2d = (a_w[:, None] * b_w[None, :]) * 0.5 * (1.0 - bb)
    return r_sub.reshape(-1), s_sub.reshape(-1), w_2d.reshape(-1)


def interpolation_l2_error_sq_2d(
    mesh,
    target_fn,
    interp_op: jax.Array,
    r_sub: jax.Array,
    s_sub: jax.Array,
    w_sub: jax.Array,
) -> jax.Array:
    """``∫ (I_h f - f)^2 dx`` over the whole mesh."""
    EToV_np = np.asarray(mesh.EToV)
    va = jnp.asarray(EToV_np[:, 0])
    vb = jnp.asarray(EToV_np[:, 1])
    vc = jnp.asarray(EToV_np[:, 2])

    # Nodal interpolant evaluated at the (Np, K) mesh nodes.
    u_h = target_fn(mesh.x, mesh.y)
    # Upsample to (n_sub, K).
    u_h_sub = interp_op @ u_h

    # Physical positions at the sub-quad points, per element.
    x_sub = (
        jnp.outer(0.5 * (-(r_sub + s_sub)), mesh.VX[va])
        + jnp.outer(0.5 * (1.0 + r_sub), mesh.VX[vb])
        + jnp.outer(0.5 * (1.0 + s_sub), mesh.VX[vc])
    )
    y_sub = (
        jnp.outer(0.5 * (-(r_sub + s_sub)), mesh.VY[va])
        + jnp.outer(0.5 * (1.0 + r_sub), mesh.VY[vb])
        + jnp.outer(0.5 * (1.0 + s_sub), mesh.VY[vc])
    )
    f_sub = target_fn(x_sub, y_sub)

    err_sq = (u_h_sub - f_sub) ** 2  # (n_sub, K)
    # For affine triangles the physical Jacobian is constant per element.
    J_per_elem = mesh.J[0, :]
    per_elem = jnp.sum(w_sub[:, None] * err_sq, axis=0)
    return jnp.sum(J_per_elem * per_elem)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _mesh_edges(x_lines: np.ndarray, y_lines: np.ndarray) -> list:
    """Build a list of LineCollection segments for a tensor-product rectangle
    mesh: all vertical lines, all horizontal lines, and the bottom-left ->
    top-right diagonals."""
    segments = []
    for x in x_lines:
        segments.append([(x, y_lines[0]), (x, y_lines[-1])])
    for y in y_lines:
        segments.append([(x_lines[0], y), (x_lines[-1], y)])
    for j in range(len(y_lines) - 1):
        for i in range(len(x_lines) - 1):
            segments.append([
                (x_lines[i], y_lines[j]),
                (x_lines[i + 1], y_lines[j + 1]),
            ])
    return segments


def _make_target_image(n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(XMIN, XMAX, n)
    ys = np.linspace(YMIN, YMAX, n)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.exp(-((XX - X0) ** 2 + (YY - Y0) ** 2) / (2.0 * SIGMA**2))
    return xs, ys, Z


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> Path:
    N = 3
    Kx = Ky = 8
    n_steps = 4000
    learning_rate = 5.0
    momentum = 0.9
    # LR schedule: drop to half at 60% of training, quarter at 85%.
    def lr_at(step: int) -> float:
        if step >= int(0.85 * n_steps):
            return 0.25 * learning_rate
        if step >= int(0.60 * n_steps):
            return 0.5 * learning_rate
        return learning_rate

    VX_init, VY_init, EToV = uniform_rectangle_mesh_2d(XMIN, XMAX, YMIN, YMAX, Kx, Ky)
    mesh_init = build_mesh_2d(VX_init, VY_init, EToV, N=N)

    n_sub_per_dim = max(N + 4, 7)
    r_sub, s_sub, w_sub = triangle_sub_quadrature(n_sub_per_dim)
    V_sub = vandermonde_2d(N, r_sub, s_sub)
    interp_op = V_sub @ jnp.linalg.inv(mesh_init.V)

    domain_x = XMAX - XMIN
    domain_y = YMAX - YMIN

    def loss_fn(logits: jax.Array) -> jax.Array:
        x_logits = logits[:Kx]
        y_logits = logits[Kx:Kx + Ky]
        x_widths = widths_from_logits(x_logits, domain_x)
        y_widths = widths_from_logits(y_logits, domain_y)
        x_lines = lines_from_widths(x_widths, XMIN)
        y_lines = lines_from_widths(y_widths, YMIN)
        VX, VY = tensor_product_vertices(x_lines, y_lines)
        mesh = update_mesh_2d_vertices(mesh_init, VX, VY)
        return interpolation_l2_error_sq_2d(
            mesh, target, interp_op, r_sub, s_sub, w_sub
        )

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    logits = jnp.zeros(Kx + Ky)
    velocity = jnp.zeros_like(logits)

    record_idx_target = np.unique(
        np.round(np.logspace(0, np.log10(n_steps), 100)).astype(int)
    ) - 1
    record_idx_target = np.clip(record_idx_target, 0, n_steps - 1)
    record_steps = set(record_idx_target.tolist()) | {0, n_steps - 1}

    snapshots: list[tuple[int, np.ndarray]] = []
    loss_history: list[float] = []
    for step in range(n_steps):
        val, grad = value_and_grad(logits)
        loss_history.append(float(val))
        if step in record_steps:
            snapshots.append((step, np.asarray(logits)))
        velocity = momentum * velocity - lr_at(step) * grad
        logits = logits + velocity

    val_final, _ = value_and_grad(logits)
    loss_history.append(float(val_final))
    snapshots.append((n_steps, np.asarray(logits)))

    print(f"loss: uniform = {loss_history[0]:.3e}  -> optimised = {loss_history[-1]:.3e}")
    print(f"reduction factor: {loss_history[0] / loss_history[-1]:.1f}x")

    def lines_from_logits(logits_arr):
        x_logits = logits_arr[:Kx]
        y_logits = logits_arr[Kx:Kx + Ky]
        x_widths = np.asarray(widths_from_logits(jnp.asarray(x_logits), domain_x))
        y_widths = np.asarray(widths_from_logits(jnp.asarray(y_logits), domain_y))
        x_lines = np.concatenate([[XMIN], XMIN + np.cumsum(x_widths)])
        y_lines = np.concatenate([[YMIN], YMIN + np.cumsum(y_widths)])
        return x_lines, y_lines

    x_lines_uniform = np.linspace(XMIN, XMAX, Kx + 1)
    y_lines_uniform = np.linspace(YMIN, YMAX, Ky + 1)
    x_lines_final, y_lines_final = lines_from_logits(np.asarray(snapshots[-1][1]))

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- static summary PNG ----------
    target_xs, target_ys, target_Z = _make_target_image(n=400)
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 9),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    for ax, x_lines, y_lines, title in [
        (axes[0, 0], x_lines_uniform, y_lines_uniform, "uniform mesh"),
        (axes[0, 1], x_lines_final, y_lines_final, "optimised mesh"),
    ]:
        ax.pcolormesh(target_xs, target_ys, target_Z, cmap="viridis", shading="auto")
        lc = LineCollection(_mesh_edges(x_lines, y_lines),
                            colors="white", linewidths=0.6, alpha=0.8)
        ax.add_collection(lc)
        ax.set_aspect("equal")
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    baseline_loss = loss_history[0]
    gain_history = [baseline_loss / max(l, 1e-300) for l in loss_history]

    # Combine the two bottom cells into a single wide axis for loss / gain.
    for ax in axes[1, :]:
        ax.remove()
    ax_loss = fig.add_subplot(2, 1, 2)
    ax_gain = ax_loss.twinx()
    ax_loss.semilogy(loss_history, color="#2ca02c", lw=1.5, label="loss L(VX, VY)")
    ax_loss.axhline(
        baseline_loss, color="#1f77b4", lw=1.0, linestyle="--",
        label=f"uniform baseline = {baseline_loss:.2e}",
    )
    ax_gain.semilogy(gain_history, color="#ff7f0e", lw=1.5,
                     label="gain = L_uniform / L")
    ax_loss.set_xlabel("optimisation step")
    ax_loss.set_ylabel("L2 interpolation error (squared)", color="#2ca02c")
    ax_gain.set_ylabel("gain factor over uniform mesh", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_gain.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_g, labels_g = ax_gain.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_g, labels_l + labels_g, loc="center right")

    fig.suptitle(
        f"2D shape optimisation: target = exp(-((x - {X0:.1f})² + (y - {Y0:.1f})²) / (2·{SIGMA:.2f}²))",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = out_dir / "shape_opt_2d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    # ---------- JSON ----------
    json_path = out_dir / "shape_opt_2d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "Kx": Kx, "Ky": Ky,
                "n_steps": n_steps,
                "learning_rate": learning_rate,
                "momentum": momentum,
                "x_lines_uniform": x_lines_uniform.tolist(),
                "y_lines_uniform": y_lines_uniform.tolist(),
                "x_lines_optimised": x_lines_final.tolist(),
                "y_lines_optimised": y_lines_final.tolist(),
                "loss_initial": loss_history[0],
                "loss_final": loss_history[-1],
                "loss_history": loss_history,
            },
            fh,
            indent=2,
        )
    print(f"wrote {json_path}")

    # ---------- animation ----------
    mp4_path = _render_optimisation_animation(
        snapshots=snapshots,
        loss_history=loss_history,
        gain_history=gain_history,
        n_steps=n_steps,
        Kx=Kx,
        Ky=Ky,
        lines_from_logits=lines_from_logits,
        x_lines_uniform=x_lines_uniform,
        y_lines_uniform=y_lines_uniform,
        target_image=(target_xs, target_ys, target_Z),
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_optimisation_animation(
    *,
    snapshots,
    loss_history,
    gain_history,
    n_steps,
    Kx,
    Ky,
    lines_from_logits,
    x_lines_uniform,
    y_lines_uniform,
    target_image,
    out_dir,
) -> Path:
    """Render an MP4 of the 2D optimisation in progress."""
    target_xs, target_ys, target_Z = target_image

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
    ax_uniform = fig.add_subplot(gs[0, 0])
    ax_current = fig.add_subplot(gs[0, 1])
    ax_loss = fig.add_subplot(gs[1, :])
    ax_gain = ax_loss.twinx()

    for ax in (ax_uniform, ax_current):
        ax.pcolormesh(target_xs, target_ys, target_Z, cmap="viridis",
                      shading="auto")
        ax.set_aspect("equal")
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax_uniform.set_title(f"uniform mesh ({Kx}x{Ky} cells)")
    ax_current.set_title("optimised mesh")

    lc_uniform = LineCollection(
        _mesh_edges(x_lines_uniform, y_lines_uniform),
        colors="white", linewidths=0.6, alpha=0.85,
    )
    ax_uniform.add_collection(lc_uniform)

    lc_current = LineCollection(
        _mesh_edges(x_lines_uniform, y_lines_uniform),
        colors="#d62728", linewidths=0.8, alpha=0.95,
    )
    ax_current.add_collection(lc_current)

    label_artist = ax_current.text(
        0.02, 0.97, "", transform=ax_current.transAxes,
        ha="left", va="top", family="monospace", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="none"),
    )

    # Loss + gain curves
    baseline_loss = loss_history[0]
    loss_line, = ax_loss.semilogy([], [], color="#2ca02c", lw=1.5,
                                  label="loss L(VX, VY)")
    gain_line, = ax_gain.semilogy([], [], color="#ff7f0e", lw=1.5,
                                  label="gain = L_uniform / L")
    cur_loss, = ax_loss.semilogy([], [], "o", color="#2ca02c", markersize=7)
    cur_gain, = ax_gain.semilogy([], [], "o", color="#ff7f0e", markersize=7)
    ax_loss.axhline(
        baseline_loss, color="#1f77b4", lw=1.0, linestyle="--",
        label=f"uniform baseline = {baseline_loss:.2e}",
    )
    ax_loss.set_xlim(0, n_steps)
    ax_loss.set_ylim(min(loss_history) / 2, max(loss_history) * 2)
    ax_gain.set_ylim(0.5, max(gain_history) * 1.5)
    ax_loss.set_xlabel("optimisation step")
    ax_loss.set_ylabel("L2 error (squared)", color="#2ca02c")
    ax_gain.set_ylabel("gain factor", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_gain.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_g, labels_g = ax_gain.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_g, labels_l + labels_g,
                   loc="center right", fontsize=9)

    fig.suptitle("2D shape optimisation in progress", fontsize=13)
    fig.tight_layout()

    snap_steps = [s for s, _ in snapshots]
    snap_lines = [lines_from_logits(logits) for _, logits in snapshots]

    def update(frame_idx: int):
        step = snap_steps[frame_idx]
        x_lines, y_lines = snap_lines[frame_idx]
        lc_current.set_segments(_mesh_edges(x_lines, y_lines))
        upto = step + 1
        steps_arr = np.arange(upto)
        loss_line.set_data(steps_arr, loss_history[:upto])
        gain_line.set_data(steps_arr, gain_history[:upto])
        cur_loss.set_data([step], [loss_history[step]])
        cur_gain.set_data([step], [gain_history[step]])
        label_artist.set_text(
            f"step {step:4d} / {n_steps}\n"
            f"loss = {loss_history[step]:.2e}\n"
            f"gain = {gain_history[step]:.1f}x"
        )
        return [lc_current, loss_line, gain_line, cur_loss, cur_gain, label_artist]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "shape_opt_2d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
