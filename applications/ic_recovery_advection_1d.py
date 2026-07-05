"""Initial-condition recovery for 1D advection: autodiff = adjoint.

Twin-experiment punchline of the differentiable solver: pick an unknown
initial condition ``u_0(x)``, forward-advect it to time ``T`` to
manufacture a synthetic observation ``u_target(x) = u(x, T; u_0_true)``,
then *invert*: start from ``u_0 ≡ 0`` and let ``jax.grad`` find the
``u_0`` that produces ``u_target``.

For pure linear advection on a periodic domain the forward map
``u_0 → u(·, T)`` is unitary (just a shift). Its adjoint is the
backward shift, so backprop through the forward solver
*literally reverses the wave in time*. No regularisation is needed and
recovery is exact to machine precision.

This is the same trick that powers data assimilation in weather
forecasting (4D-VAR), seismic full-waveform inversion, and a host of
other inverse problems — but here we get the adjoint for free,
because the entire solver is just a JAX program.

Setup
-----
``u_t + a u_x = 0`` on periodic ``[0, 2π]``, ``a = 2π``, ``T = 0.5``
(half period — the wave shifts by ``π``). The unknown ``u_0`` is
parameterised directly as the ``Np × K`` mesh nodal values
(no basis truncation), so we're optimising in the full DG space.

Outputs (``output/``)
---------------------
* ``ic_recovery_advection_1d.png``: static summary — recovered vs
  true ``u_0`` and ``u_T``, loss + parameter-error curves.
* ``ic_recovery_advection_1d.mp4``: animation of the recovery in
  progress.
* ``ic_recovery_advection_1d.json``: numerical record.
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
from ddg.dg1d.advection import advec_rhs  # noqa: E402
from ddg.dg1d.integrator import low_storage_rk4_step  # noqa: E402


XMIN, XMAX = 0.0, 2.0 * np.pi
A_SPEED = 2.0 * np.pi
T_FINAL = 0.5
N_TIME_STEPS = 400


def advec_forward(mesh, u0, dt):
    """Run periodic 1D advection forward in time."""
    rhs = partial(advec_rhs, mesh, a=A_SPEED)

    def body(carry, _):
        u, t = carry
        u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
        return (u_new, t + dt), None

    (u_final, _), _ = jax.lax.scan(body, (u0, 0.0), jnp.arange(N_TIME_STEPS))
    return u_final


def main() -> Path:
    N, K = 6, 16
    n_opt_steps = 800
    learning_rate = 1.0
    momentum = 0.9

    def lr_at(step: int) -> float:
        if step >= int(0.85 * n_opt_steps):
            return 0.25 * learning_rate
        if step >= int(0.60 * n_opt_steps):
            return 0.5 * learning_rate
        return learning_rate

    VX, EToV = uniform_mesh_1d(XMIN, XMAX, K)
    mesh = build_mesh_1d(VX, EToV, N=N, periodic=True)
    dt = T_FINAL / N_TIME_STEPS

    # Cook up a non-trivial smooth IC: a Gaussian bump + half-cosine.
    x = np.asarray(mesh.x)
    u0_true_np = (
        np.exp(-((np.asarray(mesh.x) - np.pi) ** 2) / 0.3)
        + 0.4 * np.exp(-((np.asarray(mesh.x) - np.pi / 2) ** 2) / 0.15)
    )
    u0_true = jnp.asarray(u0_true_np)

    # Manufacture the target by forward integration.
    u_target = advec_forward(mesh, u0_true, dt)

    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)

    def loss_fn(u0):
        u_final = advec_forward(mesh, u0, dt)
        diff = u_final - u_target
        return jnp.sum(mesh.J * (diff * (M_ref @ diff)))

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    u0_opt = jnp.zeros_like(mesh.x)
    velocity = jnp.zeros_like(u0_opt)

    record_idx_target = np.unique(
        np.round(np.logspace(0, np.log10(n_opt_steps), 100)).astype(int)
    ) - 1
    record_idx_target = np.clip(record_idx_target, 0, n_opt_steps - 1)
    record_steps = set(record_idx_target.tolist()) | {0, n_opt_steps - 1}

    snapshots: list[tuple[int, np.ndarray]] = []
    loss_history: list[float] = []
    param_error_history: list[float] = []

    u0_true_norm = float(jnp.linalg.norm(u0_true))

    for step in range(n_opt_steps):
        val, grad = value_and_grad(u0_opt)
        loss_history.append(float(val))
        param_error_history.append(
            float(jnp.linalg.norm(u0_opt - u0_true)) / u0_true_norm
        )
        if step in record_steps:
            snapshots.append((step, np.asarray(u0_opt)))
        velocity = momentum * velocity - lr_at(step) * grad
        u0_opt = u0_opt + velocity

    val_final, _ = value_and_grad(u0_opt)
    loss_history.append(float(val_final))
    param_error_history.append(
        float(jnp.linalg.norm(u0_opt - u0_true)) / u0_true_norm
    )
    snapshots.append((n_opt_steps, np.asarray(u0_opt)))

    print(
        f"loss: initial = {loss_history[0]:.3e}  ->  final = {loss_history[-1]:.3e}"
    )
    print(
        f"||u0_opt - u0_true|| / ||u0_true||: "
        f"{param_error_history[0]:.3e}  ->  {param_error_history[-1]:.3e}"
    )

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def eval_final(u0_arr):
        return advec_forward(mesh, u0_arr, dt)

    u0_opt_np = np.asarray(snapshots[-1][1])
    u_final_opt = np.asarray(eval_final(jnp.asarray(u0_opt_np)))
    target_np = np.asarray(u_target)

    # --------- static PNG ----------
    from ddg.dg1d.reference import vandermonde_1d
    n_plot = 50
    r_plot = jnp.linspace(-1.0, 1.0, n_plot)
    V_plot = np.asarray(vandermonde_1d(N, r_plot))
    interp_plot = V_plot @ np.asarray(jnp.linalg.inv(mesh.V))

    def plot_dg(ax, u_array, color, label, alpha=1.0, lw=1.5, linestyle="-"):
        VX_np = np.asarray(mesh.VX)
        for k in range(K):
            x_plot = VX_np[k] + 0.5 * (np.asarray(r_plot) + 1.0) * (VX_np[k + 1] - VX_np[k])
            u_plot = interp_plot @ u_array[:, k]
            ax.plot(x_plot, u_plot, color=color, lw=lw, alpha=alpha,
                    linestyle=linestyle,
                    label=label if k == 0 else None)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [2.6, 2.6, 2.2]},
    )

    plot_dg(axes[0], u0_true_np, "k", "u0_true (unknown IC)", lw=2.2, alpha=0.7)
    plot_dg(axes[0], u0_opt_np, "#d62728", "u0_optimised (recovered)", lw=1.4)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x, 0)")
    axes[0].set_title(
        f"Initial condition recovery (a={A_SPEED:.2f}, T={T_FINAL}, N={N}, K={K})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    plot_dg(axes[1], target_np, "k", "u_target = u(·, T; u0_true)", lw=2.2, alpha=0.7)
    plot_dg(axes[1], u_final_opt, "#d62728", "u(·, T; u0_optimised)", lw=1.4)
    axes[1].set_xlim(XMIN, XMAX)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x, T)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    ax_loss = axes[2]
    ax_err = ax_loss.twinx()
    ax_loss.semilogy(loss_history, color="#2ca02c", lw=1.5,
                     label="data loss  ||u(T; u0) - u_target||²")
    ax_err.semilogy(param_error_history, color="#ff7f0e", lw=1.5,
                    label="parameter error  ||u0 - u0_true|| / ||u0_true||")
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
    png_path = out_dir / "ic_recovery_advection_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "ic_recovery_advection_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": K, "T_final": T_FINAL, "a": A_SPEED,
                "n_time_steps": N_TIME_STEPS, "n_opt_steps": n_opt_steps,
                "loss_initial": loss_history[0],
                "loss_final": loss_history[-1],
                "param_error_initial": param_error_history[0],
                "param_error_final": param_error_history[-1],
                "loss_history": loss_history,
                "param_error_history": param_error_history,
            },
            fh, indent=2,
        )
    print(f"wrote {json_path}")

    mp4_path = _render_animation(
        snapshots=snapshots,
        loss_history=loss_history,
        param_error_history=param_error_history,
        n_steps=n_opt_steps,
        K=K, N=N,
        mesh=mesh,
        u0_true_np=u0_true_np,
        target_np=target_np,
        eval_final=eval_final,
        interp_plot=interp_plot,
        r_plot=np.asarray(r_plot),
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_animation(
    *, snapshots, loss_history, param_error_history, n_steps, K, N, mesh,
    u0_true_np, target_np, eval_final, interp_plot, r_plot, out_dir,
) -> Path:
    snap_steps = [s for s, _ in snapshots]
    snap_u0 = [np.asarray(u0) for _, u0 in snapshots]
    snap_uT = [np.asarray(eval_final(jnp.asarray(u0))) for u0 in snap_u0]
    VX_np = np.asarray(mesh.VX)

    def dg_data(u_array):
        u_pieces = interp_plot @ u_array
        x_pieces = np.zeros_like(u_pieces)
        for k in range(K):
            x_pieces[:, k] = VX_np[k] + 0.5 * (r_plot + 1.0) * (VX_np[k + 1] - VX_np[k])
        return x_pieces, u_pieces

    true_x0, true_y0 = dg_data(u0_true_np)
    true_xT, true_yT = dg_data(target_np)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [2.6, 2.6, 2.2]},
    )

    for k in range(K):
        axes[0].plot(true_x0[:, k], true_y0[:, k], "k-", lw=2.2, alpha=0.7,
                     label="u0_true (unknown)" if k == 0 else None)
    cur_u0_lines = []
    for k in range(K):
        line, = axes[0].plot([], [], color="#d62728", lw=1.4,
                              label="u0_optimised" if k == 0 else None)
        cur_u0_lines.append(line)
    axes[0].set_xlim(np.asarray(mesh.VX)[0], np.asarray(mesh.VX)[-1])
    ylo = float(min(u0_true_np.min(), -0.1)) - 0.1
    yhi = float(max(u0_true_np.max(), 1.0)) + 0.1
    axes[0].set_ylim(ylo, yhi)
    axes[0].set_ylabel("u(x, 0)")
    axes[0].set_xlabel("x")
    axes[0].set_title(
        f"Initial condition recovery via autodiff (a={A_SPEED:.2f}, T={T_FINAL}, N={N}, K={K})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    label_artist = axes[0].text(
        0.02, 0.97, "", transform=axes[0].transAxes, ha="left", va="top",
        family="monospace", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="none"),
    )

    for k in range(K):
        axes[1].plot(true_xT[:, k], true_yT[:, k], "k-", lw=2.2, alpha=0.7,
                     label="u_target" if k == 0 else None)
    cur_uT_lines = []
    for k in range(K):
        line, = axes[1].plot([], [], color="#d62728", lw=1.4,
                              label="u(·, T; u0_optimised)" if k == 0 else None)
        cur_uT_lines.append(line)
    axes[1].set_xlim(np.asarray(mesh.VX)[0], np.asarray(mesh.VX)[-1])
    axes[1].set_ylim(ylo, yhi)
    axes[1].set_ylabel("u(x, T)")
    axes[1].set_xlabel("x")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    ax_loss = axes[2]
    ax_err = ax_loss.twinx()
    loss_line, = ax_loss.semilogy([], [], color="#2ca02c", lw=1.5, label="data loss")
    err_line, = ax_err.semilogy([], [], color="#ff7f0e", lw=1.5,
                                label="||u0 - u0_true|| / ||u0_true||")
    cur_loss, = ax_loss.semilogy([], [], "o", color="#2ca02c", markersize=7)
    cur_err, = ax_err.semilogy([], [], "o", color="#ff7f0e", markersize=7)
    ax_loss.set_xlim(0, n_steps)
    ax_loss.set_ylim(min(loss_history) / 2, max(loss_history) * 2)
    ax_err.set_ylim(min(param_error_history) / 2, max(param_error_history) * 2)
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
        x0, y0 = dg_data(snap_u0[frame_idx])
        xT, yT = dg_data(snap_uT[frame_idx])
        for k in range(K):
            cur_u0_lines[k].set_data(x0[:, k], y0[:, k])
            cur_uT_lines[k].set_data(xT[:, k], yT[:, k])
        upto = step + 1
        steps_arr = np.arange(upto)
        loss_line.set_data(steps_arr, loss_history[:upto])
        err_line.set_data(steps_arr, param_error_history[:upto])
        cur_loss.set_data([step], [loss_history[step]])
        cur_err.set_data([step], [param_error_history[step]])
        label_artist.set_text(
            f"step {step:4d} / {n_steps}\n"
            f"loss = {loss_history[step]:.2e}\n"
            f"||u0 - u0_true||/||u0_true|| = {param_error_history[step]:.2e}"
        )
        return cur_u0_lines + cur_uT_lines + [
            loss_line, err_line, cur_loss, cur_err, label_artist,
        ]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "ic_recovery_advection_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
