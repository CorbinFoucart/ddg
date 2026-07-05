"""Shape-optimisation through the 1D viscous Burgers solver.

The previous shape-opt demos optimise mesh layout for a *static* loss
(an interpolation error). This one cranks the difficulty up: the loss
is the L2 error of the viscous-Burgers solution at a final time T,
i.e. the optimiser sees the gradient of an error that flows through

    update_mesh_vertices(mesh, VX)
        -> Burgers RHS (volume + face flux + LDG auxiliary q)
            -> ~120 RK4 stages via lax.scan
                -> L2 inner product on a high-order sub-quadrature

end-to-end. Everything is pure JAX so ``jax.value_and_grad`` produces
finite gradients without a single hand-written sensitivity.

Setup
-----
Hesthaven's textbook traveling wave on ``[-1, 1]``:

    u_exact(x, t) = 1 - tanh((x + 0.5 - t) / (2 nu))         (nu = 0.1)

so the steep front moves rightward at speed 1. ``N = 4``, ``K = 8``,
``T = 0.5`` puts the front at ``x = 0`` at termination. The uniform
mesh is already pretty good there; the optimiser tightens cells near
the trajectory of the front (which crosses roughly the band
``x ∈ [-0.5, 0]``) and gives up resolution in the flat regions at the
two ends.

Outputs (``output/``)
---------------------
* ``shape_opt_burgers_1d.png``: static summary — final-time interpolant
  (uniform vs optimised vs exact), the two mesh vertex layouts, and
  the loss + gain twin curves.
* ``shape_opt_burgers_1d.mp4``: animation of the optimisation in
  progress.
* ``shape_opt_burgers_1d.json``: numerical record.
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ddg.dg1d import build_mesh_1d, uniform_mesh_1d, update_mesh_vertices  # noqa: E402
from ddg.dg1d.burgers import burgers_rhs_1d  # noqa: E402
from ddg.dg1d.integrator import low_storage_rk4_step  # noqa: E402
from ddg.dg1d.reference import jacobi_gauss_quadrature, vandermonde_1d  # noqa: E402


XMIN, XMAX = -1.0, 1.0
NU = 0.02
T_FINAL = 0.5
N_TIME_STEPS = 400  # fixed for jit / autodiff; CFL-safe for any mesh in this family


def u_exact(x: jax.Array, t: float) -> jax.Array:
    return 1.0 - jnp.tanh((x + 0.5 - t) / (2.0 * NU))


def widths_from_logits(logits: jax.Array, length: float) -> jax.Array:
    return jax.nn.softmax(logits) * length


def vx_from_widths(widths: jax.Array, x0: float) -> jax.Array:
    return jnp.concatenate([jnp.array([x0]), x0 + jnp.cumsum(widths)])


def burgers_final_state(mesh, u0: jax.Array, dt: float, n_steps: int) -> jax.Array:
    """Burgers traveling-wave Dirichlet BCs, fixed dt + n_steps so the function
    is autodiff-clean (no python-side concretisation)."""
    rhs = partial(
        burgers_rhs_1d, mesh, nu=NU,
        inflow_fn=lambda t: u_exact(jnp.array(XMIN), t),
        outflow_fn=lambda t: u_exact(jnp.array(XMAX), t),
    )

    def body(carry, _):
        u, t = carry
        u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
        return (u_new, t + dt), None

    (u_final, _), _ = jax.lax.scan(body, (u0, 0.0), jnp.arange(n_steps))
    return u_final


def main() -> Path:
    N, K = 4, 8
    n_opt_steps = 1500
    learning_rate = 5.0
    momentum = 0.9

    def lr_at(step: int) -> float:
        if step >= int(0.85 * n_opt_steps):
            return 0.25 * learning_rate
        if step >= int(0.60 * n_opt_steps):
            return 0.5 * learning_rate
        return learning_rate

    VX_init, EToV = uniform_mesh_1d(XMIN, XMAX, K)
    mesh_init = build_mesh_1d(VX_init, EToV, N=N)
    domain_length = XMAX - XMIN
    dt = T_FINAL / N_TIME_STEPS

    # High-order sub-quadrature on the reference element for the L2 norm.
    M_sub = 3 * N + 5
    r_sub, w_sub = jacobi_gauss_quadrature(0.0, 0.0, M_sub)
    V_sub = vandermonde_1d(N, r_sub)
    interp_op = V_sub @ jnp.linalg.inv(mesh_init.V)

    EToV_np = np.asarray(EToV)
    va = jnp.asarray(EToV_np[:, 0])
    vb = jnp.asarray(EToV_np[:, 1])

    def loss_fn(logits: jax.Array) -> jax.Array:
        widths = widths_from_logits(logits, domain_length)
        VX = vx_from_widths(widths, XMIN)
        mesh = update_mesh_vertices(mesh_init, VX)
        u0 = u_exact(mesh.x, 0.0)
        u_h = burgers_final_state(mesh, u0, dt, N_TIME_STEPS)
        u_h_sub = interp_op @ u_h
        x_sub = (
            jnp.outer(jnp.ones_like(r_sub), mesh.VX[va])
            + 0.5 * jnp.outer(r_sub + 1.0, mesh.VX[vb] - mesh.VX[va])
        )
        f_sub = u_exact(x_sub, T_FINAL)
        J_per_elem = 0.5 * (mesh.VX[vb] - mesh.VX[va])
        return jnp.sum(J_per_elem * jnp.sum(w_sub[:, None] * (u_h_sub - f_sub) ** 2, axis=0))

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    logits = jnp.zeros(K)
    velocity = jnp.zeros_like(logits)
    record_idx_target = np.unique(
        np.round(np.logspace(0, np.log10(n_opt_steps), 100)).astype(int)
    ) - 1
    record_idx_target = np.clip(record_idx_target, 0, n_opt_steps - 1)
    record_steps = set(record_idx_target.tolist()) | {0, n_opt_steps - 1}

    snapshots: list[tuple[int, np.ndarray]] = []
    loss_history: list[float] = []
    for step in range(n_opt_steps):
        val, grad = value_and_grad(logits)
        loss_history.append(float(val))
        if step in record_steps:
            snapshots.append((step, np.asarray(logits)))
        velocity = momentum * velocity - lr_at(step) * grad
        logits = logits + velocity

    val_final, _ = value_and_grad(logits)
    loss_history.append(float(val_final))
    snapshots.append((n_opt_steps, np.asarray(logits)))

    print(f"loss: uniform = {loss_history[0]:.3e}  -> optimised = {loss_history[-1]:.3e}")
    print(f"reduction factor: {loss_history[0] / loss_history[-1]:.1f}x")

    VX_uniform = np.asarray(VX_init)
    VX_final = np.asarray(vx_from_widths(widths_from_logits(logits, domain_length), XMIN))

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------- static PNG ----------
    n_plot = 50
    r_plot = jnp.linspace(-1.0, 1.0, n_plot)
    V_plot = np.asarray(vandermonde_1d(N, r_plot))
    interp_plot = V_plot @ np.asarray(jnp.linalg.inv(mesh_init.V))

    def evaluate_at_final_time(VX_np):
        VX_j = jnp.asarray(VX_np)
        mesh = update_mesh_vertices(mesh_init, VX_j)
        u0 = u_exact(mesh.x, 0.0)
        u_h = burgers_final_state(mesh, u0, dt, N_TIME_STEPS)
        return mesh, u_h

    mesh_u, u_h_u = evaluate_at_final_time(VX_uniform)
    mesh_f, u_h_f = evaluate_at_final_time(VX_final)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9),
        gridspec_kw={"height_ratios": [3, 0.8, 2]},
    )
    x_fine = np.linspace(XMIN, XMAX, 2001)
    axes[0].plot(x_fine, np.asarray(u_exact(jnp.asarray(x_fine), T_FINAL)),
                 "k--", lw=1.2, label="u_exact(x, T)", alpha=0.7)
    for color, label, mesh, u_h, VX in [
        ("#1f77b4", "uniform mesh", mesh_u, u_h_u, VX_uniform),
        ("#d62728", "optimised mesh", mesh_f, u_h_f, VX_final),
    ]:
        for k in range(K):
            x_plot = VX[k] + 0.5 * (np.asarray(r_plot) + 1.0) * (VX[k + 1] - VX[k])
            u_plot = interp_plot @ np.asarray(u_h[:, k])
            axes[0].plot(x_plot, u_plot, color=color, lw=1.3,
                         label=label if k == 0 else None)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_ylabel("u(x, T)")
    axes[0].set_title(
        f"Shape opt through viscous Burgers (nu={NU}, T={T_FINAL}, N={N}, K={K})"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(VX_uniform, np.full_like(VX_uniform, 1), "|", color="#1f77b4",
                 markersize=20, markeredgewidth=2)
    axes[1].plot(VX_final, np.full_like(VX_final, 0), "|", color="#d62728",
                 markersize=20, markeredgewidth=2)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["optimised", "uniform"])
    axes[1].set_xlim(XMIN, XMAX)
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].set_xlabel("x")
    axes[1].grid(alpha=0.3, axis="x")

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
    ax_loss.set_ylabel("L2 error at T (squared)", color="#2ca02c")
    ax_gain.set_ylabel("gain factor over uniform mesh", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_gain.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_g, labels_g = ax_gain.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_g, labels_l + labels_g, loc="center right")
    fig.tight_layout()
    png_path = out_dir / "shape_opt_burgers_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "shape_opt_burgers_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": K, "nu": NU, "T_final": T_FINAL,
                "n_time_steps": N_TIME_STEPS,
                "n_opt_steps": n_opt_steps,
                "VX_uniform": VX_uniform.tolist(),
                "VX_optimised": VX_final.tolist(),
                "loss_initial": loss_history[0],
                "loss_final": loss_history[-1],
                "loss_history": loss_history,
            },
            fh, indent=2,
        )
    print(f"wrote {json_path}")

    mp4_path = _render_optimisation_animation(
        snapshots=snapshots,
        loss_history=loss_history,
        gain_history=gain_history,
        n_steps=n_opt_steps,
        K=K, N=N,
        domain_length=domain_length,
        mesh_init=mesh_init,
        dt=dt,
        n_time_steps=N_TIME_STEPS,
        interp_plot=interp_plot,
        r_plot=np.asarray(r_plot),
        x_fine=x_fine,
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
    K,
    N,
    domain_length,
    mesh_init,
    dt,
    n_time_steps,
    interp_plot,
    r_plot,
    x_fine,
    out_dir,
) -> Path:
    @jax.jit
    def eval_final(logits):
        widths = widths_from_logits(logits, domain_length)
        VX = vx_from_widths(widths, XMIN)
        mesh = update_mesh_vertices(mesh_init, VX)
        u0 = u_exact(mesh.x, 0.0)
        u_h = burgers_final_state(mesh, u0, dt, n_time_steps)
        return VX, u_h

    snap_steps = []
    snap_VX = []
    snap_x_pieces = []
    snap_u_pieces = []
    for step, logits in snapshots:
        VX, u_h = eval_final(jnp.asarray(logits))
        VX_np = np.asarray(VX)
        u_h_np = np.asarray(u_h)
        x_pieces = np.zeros((r_plot.size, K))
        u_pieces = interp_plot @ u_h_np
        for k in range(K):
            x_pieces[:, k] = VX_np[k] + 0.5 * (r_plot + 1.0) * (VX_np[k + 1] - VX_np[k])
        snap_steps.append(step)
        snap_VX.append(VX_np)
        snap_x_pieces.append(x_pieces)
        snap_u_pieces.append(u_pieces)
    n_frames = len(snapshots)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9),
        gridspec_kw={"height_ratios": [3, 0.8, 2]},
    )
    u_target_curve = np.asarray(u_exact(jnp.asarray(x_fine), T_FINAL))
    axes[0].plot(x_fine, u_target_curve, "k--", lw=1.2,
                 label="u_exact(x, T)", alpha=0.6)
    for k in range(K):
        axes[0].plot(
            snap_x_pieces[0][:, k], snap_u_pieces[0][:, k],
            color="#1f77b4", lw=1.0, alpha=0.4,
            label="uniform-mesh u_h(x, T)" if k == 0 else None,
        )
    current_lines = []
    for k in range(K):
        line, = axes[0].plot([], [], color="#d62728", lw=1.5)
        current_lines.append(line)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_ylim(-0.1, 2.1)
    axes[0].set_ylabel("u(x, T)")
    axes[0].set_title(
        f"Shape-opt through Burgers (nu={NU}, T={T_FINAL}, N={N}, K={K})"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    label_artist = axes[0].text(
        0.02, 0.97, "", transform=axes[0].transAxes, ha="left", va="top",
        family="monospace", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="none"),
    )

    axes[1].plot(snap_VX[0], np.full_like(snap_VX[0], 1), "|",
                 color="#1f77b4", markersize=20, markeredgewidth=2)
    current_dots, = axes[1].plot([], [], "|", color="#d62728",
                                 markersize=20, markeredgewidth=2)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["current", "uniform"])
    axes[1].set_xlim(XMIN, XMAX)
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].set_xlabel("x")
    axes[1].grid(alpha=0.3, axis="x")

    baseline_loss = loss_history[0]
    ax_loss = axes[2]
    ax_gain = ax_loss.twinx()
    loss_line, = ax_loss.semilogy([], [], color="#2ca02c", lw=1.5,
                                  label="loss L(VX)")
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
    ax_loss.set_ylabel("L2 error at T (squared)", color="#2ca02c")
    ax_gain.set_ylabel("gain factor", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_gain.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_g, labels_g = ax_gain.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_g, labels_l + labels_g,
                   loc="center right", fontsize=9)

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
        loss_line.set_data(steps_arr, loss_history[:upto])
        gain_line.set_data(steps_arr, gain_history[:upto])
        cur_loss.set_data([step], [loss_history[step]])
        cur_gain.set_data([step], [gain_history[step]])
        label_artist.set_text(
            f"step {step:4d} / {n_steps}\n"
            f"loss = {loss_history[step]:.2e}\n"
            f"gain = {gain_history[step]:.1f}x"
        )
        return current_lines + [
            current_dots, loss_line, gain_line, cur_loss, cur_gain, label_artist,
        ]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "shape_opt_burgers_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
