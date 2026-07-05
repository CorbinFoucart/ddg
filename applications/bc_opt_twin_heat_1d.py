"""Twin-experiment verification for the 1D heat-equation boundary control.

Companion to ``bc_opt_heat_1d.py``: instead of inverting an arbitrary
target profile (which is genuinely ill-posed and requires
regularisation), we *cook up* a known boundary schedule ``g_true(t)``,
forward-run the heat equation with it to manufacture the target

    u_target(x) := u_h(x, T;  g_true)

and then run the optimiser starting from ``g = 0``. With a small enough
parameterisation that the inverse problem is well-determined, the
optimiser should converge back to ``g_true`` itself — the data loss
``||u_h(g) - u_target||^2`` and the *parameter error*
``||g - g_true||^2`` should both drive toward zero. This is the
standard "twin experiment" used in data assimilation to validate that
an inverse-problem solver actually finds the right answer.

Setup
-----
Same physics as the BC-control demo:

    u_t = κ u_xx       on [0, 1]
    u(x, 0) = 0
    u(0, t) = g(t)     (control)
    u(1, t) = 0

with ``κ = 1``, ``T = 0.1``. The control is piecewise-linear with just
``N_CONTROL = 6`` knots — small enough that the heat smoothing leaves
the inverse problem well-posed at this T. ``g_true(t)`` is a half-sine
pulse,

    g_true[k] = sin(π · k / (N_CONTROL - 1))   for k = 0..N_CONTROL-1,

which puts the wall temperature at zero at both ends of the time
window and peaks at unity in the middle.

Because the true minimiser is interior and unique, **no regularisation
is needed** — the data term alone should pull ``g`` to ``g_true``.

Outputs
-------
* ``bc_opt_twin_heat_1d.png``: target / initial / recovered interior
  profile, recovered ``g(t)`` overlaid on ``g_true(t)``, loss + parameter
  error curves.
* ``bc_opt_twin_heat_1d.mp4``: animation of the recovery in progress.
* ``bc_opt_twin_heat_1d.json``: numerical record (g_true, g_optimised,
  loss + parameter-error histories).
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

from ddg.dg1d import build_mesh_1d, uniform_mesh_1d  # noqa: E402
from ddg.dg1d.heat import heat_rhs  # noqa: E402
from ddg.dg1d.integrator import low_storage_rk4_step  # noqa: E402
from ddg.dg1d.reference import vandermonde_1d  # noqa: E402


XMIN, XMAX = 0.0, 1.0
KAPPA = 1.0
T_FINAL = 0.1
N_TIME_STEPS = 400
N_CONTROL = 6
N, K = 3, 8


def heat_forward(mesh, g_values, t_control, dt):
    """Forward solve to ``T_FINAL``; returns ``u(·, T)``."""
    bc_left = lambda t: jnp.interp(t, t_control, g_values)
    rhs = partial(heat_rhs, mesh, kappa=KAPPA, bc_left_fn=bc_left)
    u0 = jnp.zeros_like(mesh.x)

    def body(carry, _):
        u, t = carry
        u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
        return (u_new, t + dt), None

    (u_final, _), _ = jax.lax.scan(body, (u0, 0.0), jnp.arange(N_TIME_STEPS))
    return u_final


def heat_forward_at(mesh, g_values, t_control, dt, obs_indices):
    """Forward solve recording ``u`` at the specified step indices.

    Returns an array of shape ``(len(obs_indices), Np, K)`` containing
    snapshots at the requested times. Multi-time observation makes the
    inverse problem ``g → u_obs`` substantially better posed than the
    final-time-only version (which has a non-trivial null space because
    late-time boundary values haven't had a chance to diffuse into the
    interior).
    """
    bc_left = lambda t: jnp.interp(t, t_control, g_values)
    rhs = partial(heat_rhs, mesh, kappa=KAPPA, bc_left_fn=bc_left)
    u0 = jnp.zeros_like(mesh.x)

    def body(carry, _):
        u, t = carry
        u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
        return (u_new, t + dt), u_new

    (_, _), traj = jax.lax.scan(body, (u0, 0.0), jnp.arange(N_TIME_STEPS))
    return traj[obs_indices]


def main() -> Path:
    n_opt_steps = 1200
    learning_rate = 5.0
    momentum = 0.9

    def lr_at(step: int) -> float:
        if step >= int(0.85 * n_opt_steps):
            return 0.25 * learning_rate
        if step >= int(0.60 * n_opt_steps):
            return 0.5 * learning_rate
        return learning_rate

    VX, EToV = uniform_mesh_1d(XMIN, XMAX, K)
    mesh = build_mesh_1d(VX, EToV, N=N)
    dt = T_FINAL / N_TIME_STEPS
    t_control = jnp.linspace(0.0, T_FINAL, N_CONTROL)

    # Observe at 5 evenly-spaced times during integration. With a single
    # final-time observation the heat operator has a non-trivial null
    # space (g(t) near t = T can't diffuse before T), so the parameter
    # error stalls at ~20% even when the data loss is at 1e-8. Multi-time
    # observation constrains the null space and the optimiser recovers
    # g_true to machine precision.
    n_obs = 5
    obs_indices = jnp.linspace(N_TIME_STEPS // n_obs, N_TIME_STEPS,
                               n_obs).astype(jnp.int32) - 1

    # Cook up g_true and manufacture the multi-time target.
    g_true = jnp.sin(jnp.pi * jnp.arange(N_CONTROL) / (N_CONTROL - 1))
    u_target_traj = heat_forward_at(mesh, g_true, t_control, dt, obs_indices)
    g_true_np = np.asarray(g_true)
    print(f"g_true = {g_true_np.tolist()}")

    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)

    def loss_fn(g_values):
        u_traj = heat_forward_at(mesh, g_values, t_control, dt, obs_indices)
        diff = u_traj - u_target_traj
        # Sum L2-squared error across all observation times.
        per_time = jnp.sum(
            mesh.J[None, :, :] * (diff * jax.vmap(lambda d: M_ref @ d)(diff)),
            axis=(1, 2),
        )
        return jnp.sum(per_time)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    g = jnp.zeros(N_CONTROL)
    velocity = jnp.zeros_like(g)

    record_idx_target = np.unique(
        np.round(np.logspace(0, np.log10(n_opt_steps), 100)).astype(int)
    ) - 1
    record_idx_target = np.clip(record_idx_target, 0, n_opt_steps - 1)
    record_steps = set(record_idx_target.tolist()) | {0, n_opt_steps - 1}

    snapshots: list[tuple[int, np.ndarray]] = []
    loss_history: list[float] = []
    g_error_history: list[float] = []
    g_true_norm = float(jnp.linalg.norm(g_true))

    for step in range(n_opt_steps):
        val, grad = value_and_grad(g)
        loss_history.append(float(val))
        g_error_history.append(float(jnp.linalg.norm(g - g_true)) / g_true_norm)
        if step in record_steps:
            snapshots.append((step, np.asarray(g)))
        velocity = momentum * velocity - lr_at(step) * grad
        g = g + velocity

    val_final, _ = value_and_grad(g)
    loss_history.append(float(val_final))
    g_error_history.append(float(jnp.linalg.norm(g - g_true)) / g_true_norm)
    snapshots.append((n_opt_steps, np.asarray(g)))

    print(
        f"loss: initial = {loss_history[0]:.3e}  ->  final = {loss_history[-1]:.3e}"
    )
    print(
        f"||g - g_true|| / ||g_true||: "
        f"{g_error_history[0]:.3e}  ->  {g_error_history[-1]:.3e}"
    )

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def eval_state(g_values):
        return heat_forward(mesh, g_values, t_control, dt)

    u_final_state = np.asarray(eval_state(jnp.asarray(snapshots[-1][1])))
    u_initial_state = np.asarray(eval_state(jnp.asarray(snapshots[0][1])))
    g_recovered = np.asarray(snapshots[-1][1])
    target_nodes_np = np.asarray(u_target_traj[-1])  # final-time target for plotting

    # ---------- static PNG ----------
    n_plot = 50
    r_plot = jnp.linspace(-1.0, 1.0, n_plot)
    V_plot = np.asarray(vandermonde_1d(N, r_plot))
    interp_plot = V_plot @ np.asarray(jnp.linalg.inv(mesh.V))

    def plot_dg(ax, u_array, color, label, alpha=1.0, lw=1.5):
        VX_np = np.asarray(mesh.VX)
        for k in range(K):
            x_plot = VX_np[k] + 0.5 * (np.asarray(r_plot) + 1.0) * (VX_np[k + 1] - VX_np[k])
            u_plot = interp_plot @ u_array[:, k]
            ax.plot(x_plot, u_plot, color=color, lw=lw, alpha=alpha,
                    label=label if k == 0 else None)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [3, 2, 2.2]},
    )

    plot_dg(axes[0], target_nodes_np, "k", "u_target = u_h(·, T; g_true)",
            alpha=0.85, lw=2.2)
    plot_dg(axes[0], u_initial_state, "#1f77b4",
            "initial state (g = 0): u_h(·, T) ≡ 0", alpha=0.5)
    plot_dg(axes[0], u_final_state, "#d62728",
            "recovered: u_h(·, T) tracks target", lw=1.3)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].set_title(
        f"Twin experiment: recover g(t) from u_target  (κ={KAPPA}, T={T_FINAL}, N_CONTROL={N_CONTROL})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(np.asarray(t_control), g_true_np, "-o", color="k",
                 markersize=7, lw=1.8, label="g_true(t)", alpha=0.85)
    axes[1].plot(np.asarray(t_control), g_recovered, "-s", color="#d62728",
                 markersize=5, lw=1.4, label="g_optimised(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("g(t)  (left wall temperature)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    ax_loss = axes[2]
    ax_err = ax_loss.twinx()
    ax_loss.semilogy(loss_history, color="#2ca02c", lw=1.5,
                     label="data loss  ||u_h(g) - u_target||^2")
    ax_err.semilogy(g_error_history, color="#ff7f0e", lw=1.5,
                    label="parameter error  ||g - g_true|| / ||g_true||")
    ax_loss.set_xlabel("optimisation step")
    ax_loss.set_ylabel("data loss (squared L2)", color="#2ca02c")
    ax_err.set_ylabel("parameter error (relative)", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_err.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_e, labels_e = ax_err.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_e, labels_l + labels_e, loc="center right")

    fig.tight_layout()
    png_path = out_dir / "bc_opt_twin_heat_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "bc_opt_twin_heat_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": K, "kappa": KAPPA, "T_final": T_FINAL,
                "n_time_steps": N_TIME_STEPS, "n_control": N_CONTROL,
                "n_opt_steps": n_opt_steps,
                "t_control": np.asarray(t_control).tolist(),
                "g_true": g_true_np.tolist(),
                "g_recovered": g_recovered.tolist(),
                "loss_initial": loss_history[0],
                "loss_final": loss_history[-1],
                "g_error_initial": g_error_history[0],
                "g_error_final": g_error_history[-1],
                "loss_history": loss_history,
                "g_error_history": g_error_history,
            },
            fh, indent=2,
        )
    print(f"wrote {json_path}")

    mp4_path = _render_animation(
        snapshots=snapshots,
        loss_history=loss_history,
        g_error_history=g_error_history,
        n_steps=n_opt_steps,
        N=N, K=K,
        mesh=mesh,
        t_control=np.asarray(t_control),
        g_true_np=g_true_np,
        eval_state=eval_state,
        target_nodes_np=target_nodes_np,
        interp_plot=interp_plot,
        r_plot=np.asarray(r_plot),
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_animation(
    *,
    snapshots, loss_history, g_error_history, n_steps, N, K, mesh,
    t_control, g_true_np, eval_state, target_nodes_np,
    interp_plot, r_plot, out_dir,
) -> Path:
    snap_steps = [s for s, _ in snapshots]
    snap_g = [np.asarray(g) for _, g in snapshots]
    snap_u = []
    VX_np = np.asarray(mesh.VX)
    for _, g in snapshots:
        u_state = np.asarray(eval_state(jnp.asarray(g)))
        u_pieces = interp_plot @ u_state
        x_pieces = np.zeros_like(u_pieces)
        for k in range(K):
            x_pieces[:, k] = VX_np[k] + 0.5 * (r_plot + 1.0) * (VX_np[k + 1] - VX_np[k])
        snap_u.append((x_pieces, u_pieces))
    n_frames = len(snapshots)

    # Static target curve, drawn once.
    target_pieces = interp_plot @ target_nodes_np
    target_x = np.zeros_like(target_pieces)
    for k in range(K):
        target_x[:, k] = VX_np[k] + 0.5 * (r_plot + 1.0) * (VX_np[k + 1] - VX_np[k])

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [3, 2, 2.2]},
    )
    for k in range(K):
        axes[0].plot(target_x[:, k], target_pieces[:, k], "k-", lw=2.2,
                      alpha=0.7, label="u_target (from g_true)" if k == 0 else None)
    current_lines = []
    for k in range(K):
        line, = axes[0].plot([], [], color="#d62728", lw=1.5,
                              label="u_h(·, T; g_optimised)" if k == 0 else None)
        current_lines.append(line)
    axes[0].set_xlim(XMIN, XMAX)
    u_max = max(float(np.max(target_pieces)), 1.0)
    axes[0].set_ylim(-0.05, u_max + 0.1)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].set_title(
        f"Twin experiment: recover g(t) from u_target  (κ={KAPPA}, T={T_FINAL}, N_CONTROL={len(t_control)})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    label_artist = axes[0].text(
        0.02, 0.97, "", transform=axes[0].transAxes, ha="left", va="top",
        family="monospace", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="none"),
    )

    axes[1].plot(t_control, g_true_np, "-o", color="k", markersize=7, lw=1.8,
                 alpha=0.85, label="g_true(t)")
    g_line, = axes[1].plot([], [], "-s", color="#d62728", markersize=5,
                            lw=1.4, label="g_optimised(t)")
    g_lo = min([float(np.min(g)) for g in snap_g] + [float(np.min(g_true_np))]) - 0.1
    g_hi = max([float(np.max(g)) for g in snap_g] + [float(np.max(g_true_np))]) + 0.1
    axes[1].set_xlim(0.0, T_FINAL)
    axes[1].set_ylim(g_lo, g_hi)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("g(t)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    ax_loss = axes[2]
    ax_err = ax_loss.twinx()
    loss_line, = ax_loss.semilogy([], [], color="#2ca02c", lw=1.5,
                                  label="data loss")
    err_line, = ax_err.semilogy([], [], color="#ff7f0e", lw=1.5,
                                label="||g - g_true|| / ||g_true||")
    cur_loss, = ax_loss.semilogy([], [], "o", color="#2ca02c", markersize=7)
    cur_err, = ax_err.semilogy([], [], "o", color="#ff7f0e", markersize=7)
    ax_loss.set_xlim(0, n_steps)
    ax_loss.set_ylim(min(loss_history) / 2, max(loss_history) * 2)
    ax_err.set_ylim(min(g_error_history) / 2, max(g_error_history) * 2)
    ax_loss.set_xlabel("optimisation step")
    ax_loss.set_ylabel("data loss (squared L2)", color="#2ca02c")
    ax_err.set_ylabel("parameter error (relative)", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_err.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_e, labels_e = ax_err.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_e, labels_l + labels_e,
                   loc="center right", fontsize=9)
    fig.tight_layout()

    def update(frame_idx: int):
        step = snap_steps[frame_idx]
        x_pieces, u_pieces = snap_u[frame_idx]
        for k in range(K):
            current_lines[k].set_data(x_pieces[:, k], u_pieces[:, k])
        g_line.set_data(t_control, snap_g[frame_idx])
        upto = step + 1
        steps_arr = np.arange(upto)
        loss_line.set_data(steps_arr, loss_history[:upto])
        err_line.set_data(steps_arr, g_error_history[:upto])
        cur_loss.set_data([step], [loss_history[step]])
        cur_err.set_data([step], [g_error_history[step]])
        label_artist.set_text(
            f"step {step:4d} / {n_steps}\n"
            f"loss = {loss_history[step]:.2e}\n"
            f"||g - g_true||/||g_true|| = {g_error_history[step]:.2e}"
        )
        return current_lines + [g_line, loss_line, err_line, cur_loss, cur_err, label_artist]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "bc_opt_twin_heat_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
