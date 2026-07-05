"""Optimal inflow control for the 1D viscous Burgers equation.

Boundary-control analog of the heat-equation BC twin demo, but on a
nonlinear PDE: instead of optimising a time-varying wall temperature
for ``u_t = κ u_xx``, we optimise the *inflow profile* ``u(xL, t)`` of
the viscous Burgers equation so the shock at final time lands at a
target location.

Setup
-----
``u_t + (u²/2)_x = ν u_xx`` on ``[-1, 1]`` with ``ν = 0.02``. The
right boundary is held at ``u = 0``. The left boundary value
``u(-1, t) = g(t)`` is the control, parameterised as a 16-knot
piecewise-linear function on ``[0, T]``. The initial condition is
``u_0(x) = 0``.

For these parameters the steady-state traveling-wave solution exists
for *any* constant left inflow value, with a shock front that drifts
rightward. By tuning the *time profile* of the inflow we can place the
front at a specified target location at the chosen final time.

Target
------
We pick a target ``u_target(x)`` corresponding to the analytical
traveling-wave shock parked at ``x = -0.2`` with thickness ``2ν``,
i.e. roughly the answer you'd get from a steady inflow ``g ≡ 1.0`` and
a particular final time. The optimiser has to figure out the
*right time history* of ``g`` to land the front exactly there.

Outputs (``output/``)
---------------------
* ``inflow_control_burgers_1d.png``: target vs final-state, recovered
  ``g(t)``, loss + gain curves.
* ``inflow_control_burgers_1d.mp4`` / ``.gif``: optimisation in
  progress.
* ``inflow_control_burgers_1d.json``: numerical record.
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
from ddg.dg1d.burgers import burgers_rhs_1d  # noqa: E402
from ddg.dg1d.integrator import low_storage_rk4_step  # noqa: E402
from ddg.dg1d.reference import vandermonde_1d  # noqa: E402


XMIN, XMAX = -1.0, 1.0
NU = 0.02
T_FINAL = 0.5
N_TIME_STEPS = 400
N_CONTROL = 16   # piecewise-linear inflow control knots


def burgers_forward(mesh, g_values, t_control, dt):
    """Run Burgers to T_FINAL with the parameterised left inflow."""
    inflow = lambda t: jnp.interp(t, t_control, g_values)
    outflow = lambda t: 0.0
    rhs = partial(
        burgers_rhs_1d, mesh, nu=NU,
        inflow_fn=inflow, outflow_fn=outflow,
    )
    u0 = jnp.zeros_like(mesh.x)

    def body(carry, _):
        u, t = carry
        u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
        return (u_new, t + dt), None

    (u_final, _), _ = jax.lax.scan(body, (u0, 0.0), jnp.arange(N_TIME_STEPS))
    return u_final


def main() -> Path:
    N, K = 4, 16
    n_opt_steps = 600
    # Adam optimiser — Burgers + inflow control has a poorly-scaled Hessian
    # similar to the source-ID demo.
    learning_rate = 0.05
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    VX, EToV = uniform_mesh_1d(XMIN, XMAX, K)
    mesh = build_mesh_1d(VX, EToV, N=N)
    dt = T_FINAL / N_TIME_STEPS
    t_control = jnp.linspace(0.0, T_FINAL, N_CONTROL)

    # Target = traveling-wave shock parked at x_front = -0.2 at time T.
    x_front = -0.2
    u_target = 1.0 - jnp.tanh((mesh.x - x_front) / (2.0 * NU))
    target_nodes_np = np.asarray(u_target)
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)

    def loss_fn(g_values):
        u_h = burgers_forward(mesh, g_values, t_control, dt)
        diff = u_h - u_target
        return jnp.sum(mesh.J * (diff * (M_ref @ diff)))

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    g = jnp.zeros(N_CONTROL)
    m_t = jnp.zeros_like(g)
    v_t = jnp.zeros_like(g)

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
        m_t = beta1 * m_t + (1.0 - beta1) * grad
        v_t = beta2 * v_t + (1.0 - beta2) * grad ** 2
        m_hat = m_t / (1.0 - beta1 ** (step + 1))
        v_hat = v_t / (1.0 - beta2 ** (step + 1))
        g = g - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps_adam)
    val_final, _ = value_and_grad(g)
    loss_history.append(float(val_final))
    snapshots.append((n_opt_steps, np.asarray(g)))

    print(f"loss: initial = {loss_history[0]:.3e}  ->  final = {loss_history[-1]:.3e}")
    print(f"reduction factor: {loss_history[0] / loss_history[-1]:.1f}x")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    g_final = np.asarray(snapshots[-1][1])
    u_final = np.asarray(burgers_forward(mesh, jnp.asarray(g_final), t_control, dt))

    # ---------- static PNG ----------
    n_plot = 50
    r_plot = jnp.linspace(-1.0, 1.0, n_plot)
    V_plot = np.asarray(vandermonde_1d(N, r_plot))
    interp_plot = V_plot @ np.asarray(jnp.linalg.inv(mesh.V))
    VX_np = np.asarray(mesh.VX)

    def plot_dg(ax, u_array, color, label, lw=1.5, alpha=1.0):
        for k in range(K):
            x_plot = VX_np[k] + 0.5 * (np.asarray(r_plot) + 1.0) * (VX_np[k + 1] - VX_np[k])
            u_plot = interp_plot @ u_array[:, k]
            ax.plot(x_plot, u_plot, color=color, lw=lw, alpha=alpha,
                    label=label if k == 0 else None)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [3, 2, 2.2]},
    )
    plot_dg(axes[0], target_nodes_np, "k", f"u_target  (front at x = {x_front})",
            lw=2.2, alpha=0.7)
    plot_dg(axes[0], u_final, "#d62728",
            "u_h(·, T) under optimised inflow", lw=1.4)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x, T)")
    axes[0].set_title(
        f"Optimal inflow control: Burgers shock placement  (ν={NU}, T={T_FINAL}, N={N}, K={K})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower left")

    axes[1].plot(np.asarray(t_control), g_final, "-o", color="#d62728",
                 markersize=5, lw=1.4, label="optimised g(t) = u(xL, t)")
    axes[1].axhline(0.0, color="#1f77b4", lw=1.0, linestyle="--",
                    label="initial g(t) ≡ 0")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("g(t)  (left inflow)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")

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
    png_path = out_dir / "inflow_control_burgers_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "inflow_control_burgers_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": K, "nu": NU, "T_final": T_FINAL,
                "n_time_steps": N_TIME_STEPS, "n_control": N_CONTROL,
                "n_opt_steps": n_opt_steps, "x_front_target": x_front,
                "t_control": np.asarray(t_control).tolist(),
                "g_optimised": g_final.tolist(),
                "loss_initial": loss_history[0],
                "loss_final": loss_history[-1],
                "loss_history": loss_history,
            },
            fh, indent=2,
        )
    print(f"wrote {json_path}")

    mp4_path = _render_animation(
        snapshots=snapshots,
        loss_history=loss_history,
        gain_history=gain_history,
        n_steps=n_opt_steps,
        K=K, N=N, mesh=mesh,
        target_nodes_np=target_nodes_np,
        t_control=np.asarray(t_control),
        forward_jit=jax.jit(lambda g: burgers_forward(mesh, g, t_control, dt)),
        interp_plot=interp_plot,
        r_plot=np.asarray(r_plot),
        x_front=x_front,
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_animation(
    *, snapshots, loss_history, gain_history, n_steps, K, N, mesh,
    target_nodes_np, t_control, forward_jit, interp_plot, r_plot, x_front,
    out_dir,
) -> Path:
    snap_steps = [s for s, _ in snapshots]
    snap_g = [np.asarray(g) for _, g in snapshots]
    snap_u = [np.asarray(forward_jit(jnp.asarray(g))) for g in snap_g]
    VX_np = np.asarray(mesh.VX)

    def dg_data(arr):
        u_pieces = interp_plot @ arr
        x_pieces = np.zeros_like(u_pieces)
        for k in range(K):
            x_pieces[:, k] = VX_np[k] + 0.5 * (r_plot + 1.0) * (VX_np[k + 1] - VX_np[k])
        return x_pieces, u_pieces

    true_x, true_y = dg_data(target_nodes_np)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [3, 2, 2.2]},
    )
    for k in range(K):
        axes[0].plot(true_x[:, k], true_y[:, k], "k-", lw=2.2, alpha=0.7,
                     label=f"u_target  (front at x = {x_front})" if k == 0 else None)
    cur_u_lines = []
    for k in range(K):
        line, = axes[0].plot([], [], color="#d62728", lw=1.4,
                              label="u_h(·, T)" if k == 0 else None)
        cur_u_lines.append(line)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_ylim(-0.1, 2.2)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x, T)")
    axes[0].set_title(
        f"Burgers shock placement via inflow control  (ν={NU}, T={T_FINAL}, N={N}, K={K})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower left")
    label_artist = axes[0].text(
        0.02, 0.97, "", transform=axes[0].transAxes, ha="left", va="top",
        family="monospace", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="none"),
    )

    g_line, = axes[1].plot([], [], "-o", color="#d62728", markersize=5,
                            lw=1.4, label="g(t) = u(xL, t)")
    axes[1].axhline(0.0, color="#1f77b4", lw=1.0, linestyle="--",
                    label="initial g(t) ≡ 0")
    g_lo = min([float(np.min(g)) for g in snap_g]) - 0.1
    g_hi = max([float(np.max(g)) for g in snap_g]) + 0.1
    axes[1].set_xlim(0.0, T_FINAL)
    axes[1].set_ylim(g_lo, g_hi)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("g(t)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")

    baseline_loss = loss_history[0]
    ax_loss = axes[2]
    ax_gain = ax_loss.twinx()
    loss_line, = ax_loss.semilogy([], [], color="#2ca02c", lw=1.5, label="loss L(g)")
    gain_line, = ax_gain.semilogy([], [], color="#ff7f0e", lw=1.5,
                                  label="gain = L(g=0) / L(g)")
    cur_loss, = ax_loss.semilogy([], [], "o", color="#2ca02c", markersize=7)
    cur_gain, = ax_gain.semilogy([], [], "o", color="#ff7f0e", markersize=7)
    ax_loss.axhline(baseline_loss, color="#1f77b4", lw=1.0, linestyle="--",
                    label=f"baseline = {baseline_loss:.2e}")
    ax_loss.set_xlim(0, n_steps)
    ax_loss.set_ylim(min(loss_history) / 2, max(loss_history) * 2)
    ax_gain.set_ylim(0.5, max(gain_history) * 1.5)
    ax_loss.set_xlabel("optimisation step")
    ax_loss.set_ylabel("L² error", color="#2ca02c")
    ax_gain.set_ylabel("gain factor", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_gain.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_g, labels_g = ax_gain.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_g, labels_l + labels_g, loc="center right",
                   fontsize=9)
    fig.tight_layout()

    def update(frame_idx):
        step = snap_steps[frame_idx]
        x_pieces, u_pieces = dg_data(snap_u[frame_idx])
        for k in range(K):
            cur_u_lines[k].set_data(x_pieces[:, k], u_pieces[:, k])
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
        return cur_u_lines + [
            g_line, loss_line, gain_line, cur_loss, cur_gain, label_artist,
        ]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "inflow_control_burgers_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
