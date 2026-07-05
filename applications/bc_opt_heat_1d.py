"""Boundary-control optimisation for the 1D heat equation.

Inverse heat-conduction setup: find the time-varying *left wall*
temperature ``g(t)`` that drives the interior of a 1D rod toward a
specified target profile at a chosen final time.

Setup
-----
``u_t = κ u_xx`` on ``[0, 1]``, ``κ = 1``. ``u(1, t) = 0`` is fixed
(right wall held cold); ``u(0, t) = g(t)`` is the *control*. Initial
condition ``u(x, 0) = 0``. We parameterise ``g(t)`` as a piecewise-
linear function with ``N_CONTROL = 20`` knots evenly spaced over
``[0, T_FINAL]`` and learn the knot values directly.

Target: a Gaussian "warm spot" inside the rod,

    u_target(x) = 0.8 · exp(-50 · (x - 0.2)²),

slightly off-centre toward the controlled wall so the limited
penetration depth of the heat ``√(2 κ T) ≈ 0.45`` actually reaches the
peak.

Why this is interesting
-----------------------
The loss is the L2 error of the *interior* state at the *final time*,
so ``jax.value_and_grad`` has to backprop:

    g (control knots) → jnp.interp at every RK4 sub-stage
        → heat RHS (BC enters through the half-jump at face 0)
            → ``N_TIME_STEPS`` forward integration via lax.scan
                → L2 inner product against the target

The chain is pure JAX, so no manual adjoint is required — the same
``value_and_grad`` that handles parameter optimisation in any other
JAX program just works on the full PDE trajectory.

Outputs (``output/``)
---------------------
* ``bc_opt_heat_1d.png``: static summary — target / initial / final
  interior profile, recovered ``g(t)`` schedule, and loss + gain curves.
* ``bc_opt_heat_1d.mp4``: animation of the optimisation in progress.
* ``bc_opt_heat_1d.json``: numerical record.
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


XMIN, XMAX = 0.0, 1.0
KAPPA = 1.0
T_FINAL = 0.1
N_TIME_STEPS = 400
N_CONTROL = 20
N, K = 3, 8

# Tikhonov regulariser on the second difference of g, kept small so the data
# term dominates but the late-time-spike artifact (boundary control near T
# has no time to diffuse, so the optimiser can use it to noisily patch up
# the boundary node of u_h) is suppressed.
LAMBDA_SMOOTH = 5e-5


def target(x: jax.Array) -> jax.Array:
    return 0.8 * jnp.exp(-50.0 * (x - 0.2) ** 2)


def heat_forward(mesh, g_values: jax.Array, t_control: jax.Array, dt: float) -> jax.Array:
    """Run the heat solver to T_FINAL with the parameterised left BC.

    Fixed-dt / fixed-n_steps so the entire integration is
    autodiff-friendly (no Python-side concretisation of u).
    """
    def bc_left(t):
        return jnp.interp(t, t_control, g_values)

    rhs = partial(heat_rhs, mesh, kappa=KAPPA, bc_left_fn=bc_left)
    u0 = jnp.zeros_like(mesh.x)

    def body(carry, _):
        u, t = carry
        u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
        return (u_new, t + dt), None

    (u_final, _), _ = jax.lax.scan(body, (u0, 0.0), jnp.arange(N_TIME_STEPS))
    return u_final


def main() -> Path:
    n_opt_steps = 1500
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

    target_nodes = target(mesh.x)
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)

    def loss_fn(g_values: jax.Array) -> jax.Array:
        u_h = heat_forward(mesh, g_values, t_control, dt)
        diff = u_h - target_nodes
        data_loss = jnp.sum(mesh.J * (diff * (M_ref @ diff)))
        d2g = g_values[2:] - 2.0 * g_values[1:-1] + g_values[:-2]
        smoothness = jnp.sum(d2g ** 2)
        return data_loss + LAMBDA_SMOOTH * smoothness

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
    for step in range(n_opt_steps):
        val, grad = value_and_grad(g)
        loss_history.append(float(val))
        if step in record_steps:
            snapshots.append((step, np.asarray(g)))
        velocity = momentum * velocity - lr_at(step) * grad
        g = g + velocity

    val_final, _ = value_and_grad(g)
    loss_history.append(float(val_final))
    snapshots.append((n_opt_steps, np.asarray(g)))

    print(f"loss: initial (g=0) = {loss_history[0]:.3e}  -> optimised = {loss_history[-1]:.3e}")
    print(f"reduction factor: {loss_history[0] / loss_history[-1]:.1f}x")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def eval_state(g_values):
        return heat_forward(mesh, g_values, t_control, dt)

    u_final_state = np.asarray(eval_state(jnp.asarray(snapshots[-1][1])))
    u_initial_state = np.asarray(eval_state(jnp.asarray(snapshots[0][1])))
    g_final = np.asarray(snapshots[-1][1])

    # --------- static PNG ----------
    from ddg.dg1d.reference import vandermonde_1d
    n_plot = 50
    r_plot = jnp.linspace(-1.0, 1.0, n_plot)
    V_plot = np.asarray(vandermonde_1d(N, r_plot))
    interp_plot = V_plot @ np.asarray(jnp.linalg.inv(mesh.V))

    def plot_dg(ax, u_array, color, label, alpha=1.0):
        VX_np = np.asarray(mesh.VX)
        for k in range(K):
            x_plot = VX_np[k] + 0.5 * (np.asarray(r_plot) + 1.0) * (VX_np[k + 1] - VX_np[k])
            u_plot = interp_plot @ u_array[:, k]
            ax.plot(x_plot, u_plot, color=color, lw=1.5, alpha=alpha,
                    label=label if k == 0 else None)

    x_fine = np.linspace(XMIN, XMAX, 2001)
    target_fine = np.asarray(target(jnp.asarray(x_fine)))

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9),
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )
    axes[0].plot(x_fine, target_fine, "k--", lw=1.5, label="target u_target(x)", alpha=0.75)
    plot_dg(axes[0], u_initial_state, "#1f77b4", "initial state (g = 0): u_h(x, T) ≡ 0", alpha=0.6)
    plot_dg(axes[0], u_final_state, "#d62728", "optimised: u_h(x, T) tracks target")
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].set_title(
        f"Boundary control of 1D heat equation (κ={KAPPA}, T={T_FINAL}, N={N}, K={K})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(np.asarray(t_control), g_final, "-o", color="#d62728",
                 markersize=5, lw=1.4, label="optimised g(t)")
    axes[1].axhline(0.0, color="#1f77b4", lw=1.0, linestyle="--", label="initial g(t) ≡ 0")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("g(t)  (left wall temperature)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    baseline_loss = loss_history[0]
    gain_history = [baseline_loss / max(l, 1e-300) for l in loss_history]
    ax_loss = axes[2]
    ax_gain = ax_loss.twinx()
    ax_loss.semilogy(loss_history, color="#2ca02c", lw=1.5, label="loss L(g)")
    ax_loss.axhline(baseline_loss, color="#1f77b4", lw=1.0, linestyle="--",
                    label=f"baseline (g=0) = {baseline_loss:.2e}")
    ax_gain.semilogy(gain_history, color="#ff7f0e", lw=1.5,
                     label="gain = L(g=0) / L(g)")
    ax_loss.set_xlabel("optimisation step")
    ax_loss.set_ylabel("L2 error at T (squared)", color="#2ca02c")
    ax_gain.set_ylabel("gain factor", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_gain.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_g, labels_g = ax_gain.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_g, labels_l + labels_g, loc="center right")
    fig.tight_layout()
    png_path = out_dir / "bc_opt_heat_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "bc_opt_heat_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": K, "kappa": KAPPA, "T_final": T_FINAL,
                "n_time_steps": N_TIME_STEPS, "n_control": N_CONTROL,
                "n_opt_steps": n_opt_steps,
                "t_control": np.asarray(t_control).tolist(),
                "g_optimised": g_final.tolist(),
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
        N=N, K=K,
        mesh=mesh,
        t_control=np.asarray(t_control),
        eval_state=eval_state,
        interp_plot=interp_plot,
        r_plot=np.asarray(r_plot),
        x_fine=x_fine,
        target_fine=target_fine,
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_optimisation_animation(
    *,
    snapshots, loss_history, gain_history, n_steps, N, K, mesh,
    t_control, eval_state, interp_plot, r_plot, x_fine, target_fine,
    out_dir,
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

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9),
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )
    axes[0].plot(x_fine, target_fine, "k--", lw=1.5, label="target u_target(x)", alpha=0.7)
    current_lines = []
    for k in range(K):
        line, = axes[0].plot([], [], color="#d62728", lw=1.5,
                              label="u_h(x, T)" if k == 0 else None)
        current_lines.append(line)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_ylim(-0.05, 1.0)
    axes[0].set_ylabel("u")
    axes[0].set_xlabel("x")
    axes[0].set_title(
        f"Boundary control of 1D heat (κ={KAPPA}, T={T_FINAL}, N={N}, K={K})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    label_artist = axes[0].text(
        0.02, 0.97, "", transform=axes[0].transAxes, ha="left", va="top",
        family="monospace", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="none"),
    )

    g_line, = axes[1].plot([], [], "-o", color="#d62728", markersize=5, lw=1.4,
                            label="g(t)")
    axes[1].axhline(0.0, color="#1f77b4", lw=1.0, linestyle="--",
                    label="initial g(t) ≡ 0")
    g_lo = min([float(np.min(g)) for g in snap_g]) - 0.05
    g_hi = max([float(np.max(g)) for g in snap_g]) + 0.05
    axes[1].set_xlim(0.0, T_FINAL)
    axes[1].set_ylim(g_lo, g_hi)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("g(t)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    baseline_loss = loss_history[0]
    ax_loss = axes[2]
    ax_gain = ax_loss.twinx()
    loss_line, = ax_loss.semilogy([], [], color="#2ca02c", lw=1.5,
                                  label="loss L(g)")
    gain_line, = ax_gain.semilogy([], [], color="#ff7f0e", lw=1.5,
                                  label="gain = L(g=0) / L(g)")
    cur_loss, = ax_loss.semilogy([], [], "o", color="#2ca02c", markersize=7)
    cur_gain, = ax_gain.semilogy([], [], "o", color="#ff7f0e", markersize=7)
    ax_loss.axhline(baseline_loss, color="#1f77b4", lw=1.0, linestyle="--",
                    label=f"baseline (g=0) = {baseline_loss:.2e}")
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
    ax_loss.legend(lines_l + lines_g, labels_l + labels_g, loc="center right",
                   fontsize=9)
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
        gain_line.set_data(steps_arr, gain_history[:upto])
        cur_loss.set_data([step], [loss_history[step]])
        cur_gain.set_data([step], [gain_history[step]])
        label_artist.set_text(
            f"step {step:4d} / {n_steps}\n"
            f"loss = {loss_history[step]:.2e}\n"
            f"gain = {gain_history[step]:.1f}x"
        )
        return current_lines + [g_line, loss_line, gain_line, cur_loss, cur_gain, label_artist]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "bc_opt_heat_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
