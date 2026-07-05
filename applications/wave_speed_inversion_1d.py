"""Wave-speed inversion from receiver traces (1D Maxwell-style wave).

Seismic-style full-waveform inversion (FWI) toy problem. We have a 1D
domain ``[-1, 1]`` with PEC walls and a few interior *receivers*. A
Gaussian source-pulse is launched at ``t = 0``; the wave propagates
through a heterogeneous medium ``ε(x)`` (varying permittivity ⇒
spatially-varying wave speed ``c(x) = 1/√ε``). Each receiver records
``E_z(x_r, t)`` over time.

Twin experiment: cook up ``ε_true(x)`` (a smoothly varying refractive
profile), forward-run the wave solver to record the *true* traces,
then start from ``ε ≡ 1`` and let ``jax.grad`` recover ``ε_true``.

Setup
-----
* Domain ``[-1, 1]`` with PEC walls.
* Source: ``E_z(x, 0) = exp(-200 (x + 0.5)²)``, ``H = 0``.
* Receivers at ``x ∈ {-0.1, 0.2, 0.5, 0.8}``, recording ``E_z(t)`` at
  every time step.
* Material: ``ε(x) = exp(Σ_i θ_i RBF_i(x))``, 10-bump RBF expansion
  exactly as in the material-ID demo.
* Loss: ``Σ_r Σ_t (E_z^obs(x_r, t) - E_z(x_r, t))²``.

This is the standard FWI objective. The data — multiple receivers,
many time samples — gives substantial information about ``c(x)``;
without it the inverse is rank-deficient.

Outputs (``output/``)
---------------------
* ``wave_speed_inversion_1d.png``: ε_true vs ε_recovered, receiver
  trace comparison, loss + param-error curves.
* ``wave_speed_inversion_1d.mp4`` / ``.gif``: optimisation animation.
* ``wave_speed_inversion_1d.json``: numerical record.
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
from ddg.dg1d.integrator import low_storage_rk4_step  # noqa: E402
from ddg.dg1d.reference import vandermonde_1d  # noqa: E402
from ddg.dg1d.wave import wave_rhs  # noqa: E402


XMIN, XMAX = -1.0, 1.0
T_FINAL = 1.5
N_TIME_STEPS = 800
N_BASIS = 10
BASIS_WIDTH = 0.15
RECEIVER_X = jnp.array([-0.1, 0.2, 0.5, 0.8])


def wave_forward(mesh, eps_field, dt, receiver_idx):
    """Run 1D wave solver and record E at receiver indices every step.

    1D wave system from ``dg1d/wave.py``:
        E_t = -(1/ε) H_x
        H_t = -(1/μ) E_x
    with PEC walls (E = 0 at the domain endpoints).
    """
    mu_field = jnp.ones_like(eps_field)

    def rhs_full(state, t):
        E, H = state
        return wave_rhs(mesh, E, H, t, eps=eps_field, mu=mu_field)

    E0 = jnp.exp(-200.0 * (mesh.x + 0.5) ** 2)
    H0 = jnp.zeros_like(mesh.x)

    def body(carry, _):
        state, t = carry
        new_state = low_storage_rk4_step(rhs_full, state, t, dt)
        return (new_state, t + dt), new_state[0]  # output E at every step

    (state_final, _), E_traj = jax.lax.scan(
        body, ((E0, H0), 0.0), jnp.arange(N_TIME_STEPS),
    )
    # Receiver readings: ``E`` at the chosen volume indices, all times.
    E_flat = E_traj.reshape(N_TIME_STEPS, -1)
    traces = E_flat[:, receiver_idx]  # (N_TIME_STEPS, n_receivers)
    return state_final, traces


def main() -> Path:
    N, K = 4, 30
    n_opt_steps = 600
    learning_rate = 0.05
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    VX, EToV = uniform_mesh_1d(XMIN, XMAX, K)
    mesh = build_mesh_1d(VX, EToV, N=N)
    dt = T_FINAL / N_TIME_STEPS

    # Find the volume index of the node closest to each receiver location.
    x_flat = np.asarray(mesh.x).flatten(order="F")
    receiver_idx_list = []
    for xr in np.asarray(RECEIVER_X):
        receiver_idx_list.append(int(np.argmin(np.abs(x_flat - xr))))
    receiver_idx = jnp.array(receiver_idx_list, dtype=jnp.int32)

    # RBF expansion of log ε(x).
    basis_centers = jnp.linspace(XMIN, XMAX, N_BASIS)
    rbf_matrix = jnp.exp(
        -((mesh.x[None, :, :] - basis_centers[:, None, None]) / BASIS_WIDTH) ** 2
    )

    def eps_from_theta(theta):
        log_eps = jnp.tensordot(theta, rbf_matrix, axes=1)
        return jnp.exp(log_eps)

    # Cook up θ_true such that ε_true has a "fast region" near x = -0.3
    # and a "slow region" near x = 0.5 (slow wave speed = high ε).
    centers_np = np.asarray(basis_centers)
    theta_true_np = np.zeros(N_BASIS)
    for i, c in enumerate(centers_np):
        theta_true_np[i] = (
            -0.5 * np.exp(-((c - (-0.3)) / 0.20) ** 2)  # fast (eps < 1)
            + 0.7 * np.exp(-((c - 0.5) / 0.18) ** 2)    # slow (eps > 1)
        )
    theta_true = jnp.asarray(theta_true_np)
    eps_true = eps_from_theta(theta_true)
    print(f"ε_true range: [{float(jnp.min(eps_true)):.3f}, {float(jnp.max(eps_true)):.3f}]")

    _, traces_target = wave_forward(mesh, eps_true, dt, receiver_idx)

    theta_true_norm = float(jnp.linalg.norm(theta_true))

    def loss_fn(theta):
        eps_field = eps_from_theta(theta)
        _, traces = wave_forward(mesh, eps_field, dt, receiver_idx)
        diff = traces - traces_target
        return jnp.mean(diff ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    theta = jnp.zeros(N_BASIS)
    m_t = jnp.zeros_like(theta)
    v_t = jnp.zeros_like(theta)

    record_idx_target = np.unique(
        np.round(np.logspace(0, np.log10(n_opt_steps), 100)).astype(int)
    ) - 1
    record_idx_target = np.clip(record_idx_target, 0, n_opt_steps - 1)
    record_steps = set(record_idx_target.tolist()) | {0, n_opt_steps - 1}

    snapshots: list[tuple[int, np.ndarray]] = []
    loss_history: list[float] = []
    param_err_history: list[float] = []
    for step in range(n_opt_steps):
        val, grad = value_and_grad(theta)
        loss_history.append(float(val))
        param_err_history.append(
            float(jnp.linalg.norm(theta - theta_true)) / max(theta_true_norm, 1e-12)
        )
        if step in record_steps:
            snapshots.append((step, np.asarray(theta)))
        m_t = beta1 * m_t + (1.0 - beta1) * grad
        v_t = beta2 * v_t + (1.0 - beta2) * grad ** 2
        m_hat = m_t / (1.0 - beta1 ** (step + 1))
        v_hat = v_t / (1.0 - beta2 ** (step + 1))
        theta = theta - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps_adam)

    val_final, _ = value_and_grad(theta)
    loss_history.append(float(val_final))
    param_err_history.append(
        float(jnp.linalg.norm(theta - theta_true)) / max(theta_true_norm, 1e-12)
    )
    snapshots.append((n_opt_steps, np.asarray(theta)))

    print(f"loss: {loss_history[0]:.3e} -> {loss_history[-1]:.3e}")
    print(f"||θ - θ_true|| / ||θ_true||: {param_err_history[0]:.3e} -> {param_err_history[-1]:.3e}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    theta_final = jnp.asarray(snapshots[-1][1])
    eps_final = np.asarray(eps_from_theta(theta_final))
    _, traces_final = wave_forward(mesh, eps_from_theta(theta_final), dt, receiver_idx)

    # ---------- static PNG ----------
    n_plot = 50
    r_plot = jnp.linspace(-1.0, 1.0, n_plot)
    V_plot = np.asarray(vandermonde_1d(N, r_plot))
    interp_plot = V_plot @ np.asarray(jnp.linalg.inv(mesh.V))
    VX_np = np.asarray(mesh.VX)
    eps_true_np = np.asarray(eps_true)

    def plot_dg(ax, u_array, color, label, lw=1.5, alpha=1.0, linestyle="-"):
        for k in range(K):
            x_plot = VX_np[k] + 0.5 * (np.asarray(r_plot) + 1.0) * (VX_np[k + 1] - VX_np[k])
            u_plot = interp_plot @ u_array[:, k]
            ax.plot(x_plot, u_plot, color=color, lw=lw, alpha=alpha,
                    linestyle=linestyle,
                    label=label if k == 0 else None)

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9.6),
        gridspec_kw={"height_ratios": [3, 3, 2.2]},
    )
    plot_dg(axes[0], eps_true_np, "k", "ε_true(x)", lw=2.2, alpha=0.7)
    plot_dg(axes[0], eps_final, "#d62728", "ε_recovered(x)", lw=1.4)
    plot_dg(axes[0], np.ones_like(eps_true_np), "#1f77b4",
            "initial guess ε ≡ 1", lw=1.0, alpha=0.4, linestyle="--")
    for xr in np.asarray(RECEIVER_X):
        axes[0].axvline(xr, color="#666666", lw=0.6, alpha=0.4)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("ε(x)")
    axes[0].set_title(
        f"Wave-speed inversion (FWI-style)  (N={N}, K={K}, T={T_FINAL})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    traces_target_np = np.asarray(traces_target)
    traces_final_np = np.asarray(traces_final)
    t_axis = np.linspace(0.0, T_FINAL, N_TIME_STEPS)
    for i, xr in enumerate(np.asarray(RECEIVER_X)):
        axes[1].plot(t_axis, traces_target_np[:, i], "k-", lw=1.6, alpha=0.6,
                     label=f"true x={xr:.2f}" if i == 0 else None)
        axes[1].plot(t_axis, traces_final_np[:, i], "--", color="#d62728", lw=1.0,
                     alpha=0.85,
                     label=f"recovered" if i == 0 else None)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("E_z at receivers")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

    ax_loss = axes[2]
    ax_err = ax_loss.twinx()
    ax_loss.semilogy(loss_history, color="#2ca02c", lw=1.5,
                     label="data loss (mean-squared trace error)")
    ax_err.semilogy(param_err_history, color="#ff7f0e", lw=1.5,
                    label="||θ - θ_true|| / ||θ_true||")
    ax_loss.set_xlabel("optimisation step")
    ax_loss.set_ylabel("data loss", color="#2ca02c")
    ax_err.set_ylabel("parameter error (relative)", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_err.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_e, labels_e = ax_err.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_e, labels_l + labels_e, loc="center right")

    fig.tight_layout()
    png_path = out_dir / "wave_speed_inversion_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "wave_speed_inversion_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": K, "T_final": T_FINAL,
                "n_time_steps": N_TIME_STEPS, "n_opt_steps": n_opt_steps,
                "n_basis": N_BASIS, "basis_width": BASIS_WIDTH,
                "receivers": np.asarray(RECEIVER_X).tolist(),
                "loss_initial": loss_history[0],
                "loss_final": loss_history[-1],
                "param_error_initial": param_err_history[0],
                "param_error_final": param_err_history[-1],
                "loss_history": loss_history,
                "param_error_history": param_err_history,
            },
            fh, indent=2,
        )
    print(f"wrote {json_path}")

    mp4_path = _render_animation(
        snapshots=snapshots,
        loss_history=loss_history,
        param_err_history=param_err_history,
        n_steps=n_opt_steps,
        K=K, N=N, mesh=mesh,
        eps_true_np=eps_true_np,
        traces_target_np=traces_target_np,
        t_axis=t_axis,
        eps_from_theta=jax.jit(eps_from_theta),
        traces_forward=jax.jit(lambda th: wave_forward(mesh, eps_from_theta(th), dt, receiver_idx)[1]),
        receivers=np.asarray(RECEIVER_X),
        interp_plot=interp_plot,
        r_plot=np.asarray(r_plot),
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_animation(
    *, snapshots, loss_history, param_err_history, n_steps, K, N, mesh,
    eps_true_np, traces_target_np, t_axis, eps_from_theta, traces_forward,
    receivers, interp_plot, r_plot, out_dir,
) -> Path:
    snap_steps = [s for s, _ in snapshots]
    snap_eps = [np.asarray(eps_from_theta(jnp.asarray(th))) for _, th in snapshots]
    snap_traces = [np.asarray(traces_forward(jnp.asarray(th))) for _, th in snapshots]
    VX_np = np.asarray(mesh.VX)

    def dg_data(arr):
        u_pieces = interp_plot @ arr
        x_pieces = np.zeros_like(u_pieces)
        for k in range(K):
            x_pieces[:, k] = VX_np[k] + 0.5 * (r_plot + 1.0) * (VX_np[k + 1] - VX_np[k])
        return x_pieces, u_pieces

    true_x, true_y = dg_data(eps_true_np)

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9.6),
        gridspec_kw={"height_ratios": [3, 3, 2.2]},
    )
    for k in range(K):
        axes[0].plot(true_x[:, k], true_y[:, k], "k-", lw=2.2, alpha=0.7,
                     label="ε_true(x)" if k == 0 else None)
    cur_eps_lines = []
    for k in range(K):
        line, = axes[0].plot([], [], color="#d62728", lw=1.4,
                              label="ε_recovered(x)" if k == 0 else None)
        cur_eps_lines.append(line)
    for xr in receivers:
        axes[0].axvline(xr, color="#666666", lw=0.6, alpha=0.4)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_ylim(0.0, float(np.max(eps_true_np)) * 1.2)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("ε(x)")
    axes[0].set_title(
        f"FWI-style wave-speed recovery (N={N}, K={K}, T={T_FINAL})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    label_artist = axes[0].text(
        0.02, 0.97, "", transform=axes[0].transAxes, ha="left", va="top",
        family="monospace", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="none"),
    )

    n_rec = traces_target_np.shape[1]
    cur_trace_lines = []
    for i in range(n_rec):
        axes[1].plot(t_axis, traces_target_np[:, i], "k-", lw=1.4, alpha=0.55,
                     label=f"true (x = {receivers[i]:.2f})" if i == 0 else None)
        line, = axes[1].plot([], [], "--", color="#d62728", lw=1.0, alpha=0.85,
                              label="recovered" if i == 0 else None)
        cur_trace_lines.append(line)
    axes[1].set_xlim(0.0, t_axis[-1])
    axes[1].set_ylim(
        float(np.min(traces_target_np)) * 1.4,
        float(np.max(traces_target_np)) * 1.4,
    )
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("E_z at receivers")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

    ax_loss = axes[2]
    ax_err = ax_loss.twinx()
    loss_line, = ax_loss.semilogy([], [], color="#2ca02c", lw=1.5,
                                  label="data loss")
    err_line, = ax_err.semilogy([], [], color="#ff7f0e", lw=1.5,
                                label="||θ - θ_true|| / ||θ_true||")
    cur_loss, = ax_loss.semilogy([], [], "o", color="#2ca02c", markersize=7)
    cur_err, = ax_err.semilogy([], [], "o", color="#ff7f0e", markersize=7)
    ax_loss.set_xlim(0, n_steps)
    ax_loss.set_ylim(min(loss_history) / 2, max(loss_history) * 2)
    ax_err.set_ylim(min(param_err_history) / 2, max(param_err_history) * 2)
    ax_loss.set_xlabel("optimisation step")
    ax_loss.set_ylabel("data loss", color="#2ca02c")
    ax_err.set_ylabel("parameter error (relative)", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_err.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_e, labels_e = ax_err.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_e, labels_l + labels_e, loc="center right",
                   fontsize=9)
    fig.tight_layout()

    def update(frame_idx):
        step = snap_steps[frame_idx]
        xeps, yeps = dg_data(snap_eps[frame_idx])
        for k in range(K):
            cur_eps_lines[k].set_data(xeps[:, k], yeps[:, k])
        for i in range(n_rec):
            cur_trace_lines[i].set_data(t_axis, snap_traces[frame_idx][:, i])
        upto = step + 1
        steps_arr = np.arange(upto)
        loss_line.set_data(steps_arr, loss_history[:upto])
        err_line.set_data(steps_arr, param_err_history[:upto])
        cur_loss.set_data([step], [loss_history[step]])
        cur_err.set_data([step], [param_err_history[step]])
        label_artist.set_text(
            f"step {step:4d} / {n_steps}\n"
            f"loss = {loss_history[step]:.2e}\n"
            f"||θ - θ_true||/||θ_true|| = {param_err_history[step]:.2e}"
        )
        return cur_eps_lines + cur_trace_lines + [
            loss_line, err_line, cur_loss, cur_err, label_artist,
        ]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "wave_speed_inversion_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
