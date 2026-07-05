"""Shape-optimisation demo: optimise mesh vertex positions to minimise the
L2 interpolation error of a sharply-peaked target on a fixed-K mesh.

Setup
-----
Target ``f(x) = exp(-100*(x-0.3)^2)`` on ``[-1, 1]``, with ``K`` elements
of degree ``N``. We minimise the L2 error of the nodal interpolant
``I_h f`` over the positions of the ``K-1`` interior vertices,
parameterised through a softmax over element widths so the mesh stays
non-degenerate (positive widths, sums to the domain length).

We don't solve a PDE here — the point is to show that the mesh builder
is end-to-end differentiable, so ``jax.grad`` flows from ``L`` through
``Mesh1D.x, J, rx, Fscale`` back to ``VX``.

Outputs (``output/``)
---------------------
* ``shape_opt_1d.png``: static summary — target + uniform-vs-optimised
  interpolants, mesh vertex layout for each, and the loss curve.
* ``shape_opt_1d.mp4``: animation of the optimisation in progress. Each
  frame shows the current interpolant tracking the target, the current
  mesh vertex layout migrating, and the loss curve filling in.
* ``shape_opt_1d.json``: numerical record (initial/final VX, loss
  history, settings).
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

from ddg.dg1d import (  # noqa: E402
    build_mesh_1d,
    uniform_mesh_1d,
    update_mesh_vertices,
    vandermonde_1d,
)
from ddg.dg1d.reference import jacobi_gauss_quadrature  # noqa: E402


XMIN, XMAX = -1.0, 1.0


def target(x: jax.Array) -> jax.Array:
    return jnp.exp(-100.0 * (x - 0.3) ** 2)


def widths_from_logits(logits: jax.Array, domain_length: float) -> jax.Array:
    """Soft-positive element widths that sum to ``domain_length``."""
    w = jax.nn.softmax(logits)
    return w * domain_length


def vx_from_widths(widths: jax.Array, x_left: float) -> jax.Array:
    """Build vertex positions from element widths."""
    return jnp.concatenate([jnp.array([x_left]), x_left + jnp.cumsum(widths)])


def interpolation_l2_error_sq(
    mesh,
    target_fn,
    interp_op: jax.Array,
    r_sub: jax.Array,
    w_sub: jax.Array,
) -> jax.Array:
    """L2 error squared of the nodal interpolant of ``target_fn`` on ``mesh``.

    Uses a high-order sub-quadrature on the reference element to integrate
    ``(I_h f - f)^2`` exactly for polynomials up to degree ``2*N``.
    """
    EToV_np = np.asarray(mesh.EToV)
    va = jnp.asarray(EToV_np[:, 0])
    vb = jnp.asarray(EToV_np[:, 1])

    # Nodal values at mesh nodes are exact (interpolation), then upsample to
    # sub-quadrature points via the precomputed interp op.
    u_h = target_fn(mesh.x)            # (Np, K)
    u_h_sub = interp_op @ u_h          # (M, K)

    # Physical sub-quadrature positions, per element.
    x_sub = (
        jnp.outer(jnp.ones_like(r_sub), mesh.VX[va])
        + 0.5 * jnp.outer(r_sub + 1.0, mesh.VX[vb] - mesh.VX[va])
    )                                   # (M, K)
    f_sub = target_fn(x_sub)            # (M, K)

    # Per-element affine Jacobian is constant: J = (VX[vb] - VX[va]) / 2.
    J_per_elem = 0.5 * (mesh.VX[vb] - mesh.VX[va])   # (K,)

    weighted = jnp.sum(w_sub[:, None] * (u_h_sub - f_sub) ** 2, axis=0)  # (K,)
    return jnp.sum(J_per_elem * weighted)


def main() -> Path:
    N, K = 4, 8
    n_steps = 1500
    learning_rate = 5.0  # logits live in log-prob space; gradients are small
    momentum = 0.9

    # Initial uniform mesh; this is also our topology reference.
    VX_init, EToV = uniform_mesh_1d(XMIN, XMAX, K)
    mesh_init = build_mesh_1d(VX_init, EToV, N=N)
    domain_length = XMAX - XMIN

    # Sub-quadrature on the reference element for error integration.
    M_sub = 3 * N + 5
    r_sub, w_sub = jacobi_gauss_quadrature(0.0, 0.0, M_sub)
    V_sub = vandermonde_1d(N, r_sub)
    interp_op = V_sub @ jnp.linalg.inv(mesh_init.V)

    def loss_fn(logits: jax.Array) -> jax.Array:
        widths = widths_from_logits(logits, domain_length)
        VX = vx_from_widths(widths, XMIN)
        mesh = update_mesh_vertices(mesh_init, VX)
        return interpolation_l2_error_sq(mesh, target, interp_op, r_sub, w_sub)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Uniform widths correspond to all-equal logits.
    logits = jnp.zeros(K)
    velocity = jnp.zeros_like(logits)

    # Log-spaced snapshots: early steps capture rapid migration, later
    # steps capture the long refinement tail.
    record_idx_target = np.unique(
        np.round(np.logspace(0, np.log10(n_steps), 100)).astype(int)
    ) - 1
    record_idx_target = np.clip(record_idx_target, 0, n_steps - 1)
    record_steps = set(record_idx_target.tolist()) | {0, n_steps - 1}
    snapshots: list[tuple[int, np.ndarray]] = []

    loss_history = []
    for step in range(n_steps):
        val, grad = value_and_grad(logits)
        loss_history.append(float(val))
        if step in record_steps:
            snapshots.append((step, np.asarray(logits)))
        velocity = momentum * velocity - learning_rate * grad
        logits = logits + velocity

    val_final, _ = value_and_grad(logits)
    loss_history.append(float(val_final))
    snapshots.append((n_steps, np.asarray(logits)))

    VX_final = np.asarray(vx_from_widths(widths_from_logits(logits, domain_length), XMIN))
    VX_uniform = np.asarray(VX_init)

    print(f"loss: uniform = {loss_history[0]:.3e}  -> optimised = {loss_history[-1]:.3e}")
    print(f"reduction factor: {loss_history[0] / loss_history[-1]:.1f}x")

    # ----- render the comparison -----
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 0.8, 2]})

    # Top: target + interpolants
    x_fine = np.linspace(XMIN, XMAX, 4001)
    f_fine = np.exp(-100.0 * (x_fine - 0.3) ** 2)
    axes[0].plot(x_fine, f_fine, "k-", lw=1.5, label="target f(x)", alpha=0.7)

    for VX, color, label in [(VX_uniform, "#1f77b4", "uniform mesh"),
                              (VX_final, "#d62728", "optimised mesh")]:
        mesh = build_mesh_1d(jnp.asarray(VX), EToV, N=N)
        u_h = target(mesh.x)
        # Plot per-element polynomials by upsampling each one.
        for k in range(K):
            r_plot = np.linspace(-1, 1, 50)
            V_plot = np.asarray(vandermonde_1d(N, jnp.asarray(r_plot)))
            interp_plot = V_plot @ np.asarray(jnp.linalg.inv(mesh.V))
            x_plot = VX[k] + 0.5 * (r_plot + 1.0) * (VX[k + 1] - VX[k])
            u_plot = interp_plot @ np.asarray(u_h[:, k])
            axes[0].plot(x_plot, u_plot, color=color, lw=1.2,
                         label=label if k == 0 else None)
    axes[0].set_ylabel("u(x)")
    axes[0].set_title(f"Interpolation of a sharp Gaussian (N={N}, K={K})")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # Middle: mesh vertex layout
    axes[1].plot(VX_uniform, np.full_like(VX_uniform, 1), "|", color="#1f77b4",
                 markersize=20, markeredgewidth=2, label="uniform")
    axes[1].plot(VX_final, np.full_like(VX_final, 0), "|", color="#d62728",
                 markersize=20, markeredgewidth=2, label="optimised")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["optimised", "uniform"])
    axes[1].set_xlim(XMIN, XMAX)
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].set_xlabel("x")
    axes[1].grid(alpha=0.3, axis="x")

    # Bottom: loss curve (left axis) + gain factor over uniform baseline (right axis).
    baseline_loss = loss_history[0]
    gain_history = [baseline_loss / max(l, 1e-300) for l in loss_history]
    ax_loss = axes[2]
    ax_gain = ax_loss.twinx()
    ax_loss.semilogy(loss_history, color="#2ca02c", lw=1.5, label="loss L(VX)")
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

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "shape_opt_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    json_path = out_dir / "shape_opt_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N,
                "K": K,
                "n_steps": n_steps,
                "learning_rate": learning_rate,
                "VX_uniform": VX_uniform.tolist(),
                "VX_optimised": VX_final.tolist(),
                "loss_initial": loss_history[0],
                "loss_final": loss_history[-1],
                "loss_history": loss_history,
            },
            fh,
            indent=2,
        )
    print(f"wrote {png_path}")
    print(f"wrote {json_path}")

    # ----- animation of the optimisation in progress -----
    mp4_path = _render_optimisation_animation(
        snapshots=snapshots,
        loss_history=loss_history,
        n_steps=n_steps,
        domain_length=domain_length,
        EToV=EToV,
        N=N,
        K=K,
        mesh_init=mesh_init,
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_optimisation_animation(
    *,
    snapshots: list[tuple[int, np.ndarray]],
    loss_history: list[float],
    n_steps: int,
    domain_length: float,
    EToV: jax.Array,
    N: int,
    K: int,
    mesh_init,
    out_dir: Path,
) -> Path:
    """Build an MP4 of the optimisation: interpolant, mesh vertices, loss."""
    # Pre-compute per-snapshot data: VX, per-element interpolant on a fine grid.
    n_plot = 40
    r_plot = jnp.linspace(-1.0, 1.0, n_plot)
    from ddg.dg1d.reference import vandermonde_1d
    V_plot = np.asarray(vandermonde_1d(N, r_plot))
    interp_op = V_plot @ np.asarray(jnp.linalg.inv(mesh_init.V))

    target_grid_x = np.linspace(XMIN, XMAX, 4001)
    target_grid_y = np.exp(-100.0 * (target_grid_x - 0.3) ** 2)

    snap_steps = [s for s, _ in snapshots]
    snap_VX = []
    snap_x_pieces = []  # list of (K, n_plot) arrays
    snap_u_pieces = []
    for step, logits in snapshots:
        widths = widths_from_logits(jnp.asarray(logits), domain_length)
        VX = vx_from_widths(widths, XMIN)
        mesh = update_mesh_vertices(mesh_init, VX)
        u_nodes = target(mesh.x)                       # (Np, K)
        u_plot = interp_op @ np.asarray(u_nodes)       # (n_plot, K)
        VX_np = np.asarray(VX)
        x_plot = np.zeros((n_plot, K))
        for k in range(K):
            x_plot[:, k] = VX_np[k] + 0.5 * (np.asarray(r_plot) + 1.0) * (VX_np[k + 1] - VX_np[k])
        snap_VX.append(VX_np)
        snap_x_pieces.append(x_plot)
        snap_u_pieces.append(u_plot)
    n_frames = len(snapshots)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 0.8, 2]},
    )

    # --- top: target + initial uniform interpolant (static reference) + current interpolant (animated) ---
    ax_top = axes[0]
    ax_top.plot(target_grid_x, target_grid_y, "k-", lw=1.5, alpha=0.55,
                label="target f(x) = exp(-100 (x - 0.3)^2)")
    # Uniform-mesh interpolant (frame 0 of snapshots = step 0 = uniform).
    for k in range(K):
        ax_top.plot(
            snap_x_pieces[0][:, k], snap_u_pieces[0][:, k],
            color="#1f77b4", lw=1.0, alpha=0.4,
            label="uniform mesh interpolant" if k == 0 else None,
        )
    current_lines = []
    for k in range(K):
        line, = ax_top.plot([], [], color="#d62728", lw=1.5)
        current_lines.append(line)
    ax_top.set_xlim(XMIN, XMAX)
    ax_top.set_ylim(-0.15, 1.1)
    ax_top.set_ylabel("u(x)")
    ax_top.set_title(f"1D shape optimisation in progress  (N={N}, K={K})")
    ax_top.legend(loc="upper left")
    ax_top.grid(alpha=0.3)
    label_artist = ax_top.text(
        0.98, 0.95, "", transform=ax_top.transAxes, ha="right", va="top",
        family="monospace", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # --- middle: mesh vertex positions (uniform static + current animated) ---
    ax_mid = axes[1]
    ax_mid.plot(snap_VX[0], np.full_like(snap_VX[0], 1), "|", color="#1f77b4",
                markersize=20, markeredgewidth=2)
    current_dots, = ax_mid.plot([], [], "|", color="#d62728",
                                markersize=20, markeredgewidth=2)
    ax_mid.set_yticks([0, 1])
    ax_mid.set_yticklabels(["current", "uniform"])
    ax_mid.set_xlim(XMIN, XMAX)
    ax_mid.set_ylim(-0.5, 1.5)
    ax_mid.set_xlabel("x")
    ax_mid.grid(alpha=0.3, axis="x")

    # --- bottom: loss curve (left axis) + gain over uniform baseline (right axis) ---
    baseline_loss = loss_history[0]
    gain_history = [baseline_loss / max(l, 1e-300) for l in loss_history]
    ax_loss = axes[2]
    ax_gain = ax_loss.twinx()
    full_loss_line, = ax_loss.semilogy([], [], color="#2ca02c", lw=1.5,
                                       label="loss L(VX)")
    full_gain_line, = ax_gain.semilogy([], [], color="#ff7f0e", lw=1.5,
                                       label="gain = L_uniform / L")
    current_pt_loss, = ax_loss.semilogy([], [], "o", color="#2ca02c",
                                        markersize=7)
    current_pt_gain, = ax_gain.semilogy([], [], "o", color="#ff7f0e",
                                        markersize=7)
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
    ax_loss.legend(lines_l + lines_g, labels_l + labels_g, loc="center right",
                   fontsize=9)

    fig.tight_layout()

    def update(frame_idx: int):
        step = snap_steps[frame_idx]
        for k in range(K):
            current_lines[k].set_data(
                snap_x_pieces[frame_idx][:, k],
                snap_u_pieces[frame_idx][:, k],
            )
        current_dots.set_data(snap_VX[frame_idx], np.zeros_like(snap_VX[frame_idx]))
        upto = step + 1
        steps_arr = np.arange(upto)
        full_loss_line.set_data(steps_arr, loss_history[:upto])
        full_gain_line.set_data(steps_arr, gain_history[:upto])
        current_pt_loss.set_data([step], [loss_history[step]])
        current_pt_gain.set_data([step], [gain_history[step]])
        label_artist.set_text(
            f"step {step:4d} / {n_steps}\n"
            f"loss = {loss_history[step]:.2e}\n"
            f"gain = {gain_history[step]:.1f}x"
        )
        return current_lines + [
            current_dots, full_loss_line, full_gain_line,
            current_pt_loss, current_pt_gain, label_artist,
        ]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "shape_opt_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
