"""Source-term recovery for the 1D Poisson equation.

Inverse Poisson: given the steady-state solution ``u(x)`` to

    -u''(x) = f(x)     on (0, L), with u(0) = u(L) = 0,

recover the source ``f(x)``. Twin experiment: cook up ``f_true(x)``,
solve forward with ``solve_poisson`` to manufacture ``u_target``, then
optimise ``f`` from ``f ≡ 0`` to match the target.

Why this differs from the other twin demos
------------------------------------------
* The forward map ``f → u`` here is **linear**, *steady-state*, and
  **invertible** (Poisson with Dirichlet BCs has a unique solution for
  every right-hand side). Recovery should be exact, with no null space
  to worry about, and the loss landscape is strictly convex.
* No multi-time observation needed — there's only one state ``u(x)``.
* No PDE-trajectory backprop — the autodiff just goes through one
  linear solve (``jax.numpy.linalg.solve`` is fully differentiable).

So this is the cleanest "I really do recover the truth" demo of the
three: BC control, IC recovery, and material identification each
fight a low-pass smoothing operator with a non-trivial null space.
Source recovery doesn't — the kernel of ``-u''`` on the Dirichlet space
is trivial, and the optimiser drives the parameter error to machine
precision.

Outputs (``output/``)
---------------------
* ``source_id_poisson_1d.png``: f_true vs f_recovered, u_target vs
  u_h, loss + parameter-error curves.
* ``source_id_poisson_1d.mp4``: animation of the recovery in progress.
* ``source_id_poisson_1d.json``: numerical record.
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
from ddg.dg1d.poisson import poisson_ip_matrix  # noqa: E402
from ddg.dg1d.reference import vandermonde_1d  # noqa: E402


XMIN, XMAX = 0.0, 1.0


def main() -> Path:
    N, K = 4, 16
    n_opt_steps = 2000
    # Inverse Poisson has a poorly-scaled Hessian (condition number ~ Np*K),
    # so vanilla momentum SGD converges glacially. Adam adapts to the
    # per-parameter curvature automatically.
    learning_rate = 0.5
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    VX, EToV = uniform_mesh_1d(XMIN, XMAX, K)
    mesh = build_mesh_1d(VX, EToV, N=N)

    # Assemble the IPDG stiffness matrix once. The forward solve becomes a
    # single linear solve "A u = M f", which jax handles natively. Doing
    # things this way exposes the linearity of the inverse Poisson problem.
    A = poisson_ip_matrix(mesh)
    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)

    def _flatten(a):
        return a.T.reshape(-1)

    def _unflatten(a_flat):
        return a_flat.reshape(K, mesh.Np).T

    def forward(f_field: jax.Array) -> jax.Array:
        """``f`` on the nodes -> ``u`` on the nodes via the IPDG solve."""
        Mf = mesh.J * (M_ref @ f_field)
        u_flat = jnp.linalg.solve(A, _flatten(Mf))
        return _unflatten(u_flat)

    # Cook up a fairly localised source: two opposite-sign bumps.
    f_true = (
        3.0 * jnp.exp(-200.0 * (mesh.x - 0.3) ** 2)
        - 2.0 * jnp.exp(-200.0 * (mesh.x - 0.7) ** 2)
    )
    u_target = forward(f_true)

    f_true_np = np.asarray(f_true)
    f_true_norm = float(jnp.linalg.norm(f_true))

    def loss_fn(f_field: jax.Array) -> jax.Array:
        u_h = forward(f_field)
        diff = u_h - u_target
        return jnp.sum(mesh.J * (diff * (M_ref @ diff)))

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    f = jnp.zeros_like(mesh.x)
    m_t = jnp.zeros_like(f)   # Adam: first-moment estimate
    v_t = jnp.zeros_like(f)   # Adam: second-moment estimate

    record_idx_target = np.unique(
        np.round(np.logspace(0, np.log10(n_opt_steps), 80)).astype(int)
    ) - 1
    record_idx_target = np.clip(record_idx_target, 0, n_opt_steps - 1)
    record_steps = set(record_idx_target.tolist()) | {0, n_opt_steps - 1}

    snapshots: list[tuple[int, np.ndarray]] = []
    loss_history: list[float] = []
    param_err_history: list[float] = []
    for step in range(n_opt_steps):
        val, grad = value_and_grad(f)
        loss_history.append(float(val))
        param_err_history.append(
            float(jnp.linalg.norm(f - f_true)) / max(f_true_norm, 1e-12)
        )
        if step in record_steps:
            snapshots.append((step, np.asarray(f)))
        m_t = beta1 * m_t + (1.0 - beta1) * grad
        v_t = beta2 * v_t + (1.0 - beta2) * grad ** 2
        m_hat = m_t / (1.0 - beta1 ** (step + 1))
        v_hat = v_t / (1.0 - beta2 ** (step + 1))
        f = f - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps_adam)
    val_final, _ = value_and_grad(f)
    loss_history.append(float(val_final))
    param_err_history.append(
        float(jnp.linalg.norm(f - f_true)) / max(f_true_norm, 1e-12)
    )
    snapshots.append((n_opt_steps, np.asarray(f)))

    print(f"loss: initial = {loss_history[0]:.3e}  ->  final = {loss_history[-1]:.3e}")
    print(f"||f - f_true|| / ||f_true||: {param_err_history[0]:.3e}  ->  {param_err_history[-1]:.3e}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    f_final_np = np.asarray(snapshots[-1][1])
    u_final_np = np.asarray(forward(jnp.asarray(f_final_np)))
    target_np = np.asarray(u_target)

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
        gridspec_kw={"height_ratios": [3, 2.5, 2.2]},
    )
    plot_dg(axes[0], f_true_np, "k", "f_true(x)", lw=2.2, alpha=0.7)
    plot_dg(axes[0], f_final_np, "#d62728", "f_recovered(x)", lw=1.4)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title(
        f"Source-term recovery in inverse Poisson: -u'' = f  (N={N}, K={K})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    plot_dg(axes[1], target_np, "k", "u_target = solve(-u'' = f_true)", lw=2.2, alpha=0.7)
    plot_dg(axes[1], u_final_np, "#d62728", "u_h under f_recovered", lw=1.4)
    axes[1].set_xlim(XMIN, XMAX)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    ax_loss = axes[2]
    ax_err = ax_loss.twinx()
    ax_loss.semilogy(loss_history, color="#2ca02c", lw=1.5,
                     label="data loss")
    ax_err.semilogy(param_err_history, color="#ff7f0e", lw=1.5,
                    label="||f - f_true|| / ||f_true||")
    ax_loss.set_xlabel("optimisation step")
    ax_loss.set_ylabel("data loss (squared L²)", color="#2ca02c")
    ax_err.set_ylabel("parameter error (relative)", color="#ff7f0e")
    ax_loss.tick_params(axis="y", labelcolor="#2ca02c")
    ax_err.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_loss.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_loss.get_legend_handles_labels()
    lines_e, labels_e = ax_err.get_legend_handles_labels()
    ax_loss.legend(lines_l + lines_e, labels_l + labels_e, loc="center right")

    fig.tight_layout()
    png_path = out_dir / "source_id_poisson_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "source_id_poisson_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": K, "n_opt_steps": n_opt_steps,
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
        f_true_np=f_true_np, target_np=target_np,
        forward=jax.jit(forward),
        interp_plot=interp_plot,
        r_plot=np.asarray(r_plot),
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_animation(
    *, snapshots, loss_history, param_err_history, n_steps, K, N, mesh,
    f_true_np, target_np, forward, interp_plot, r_plot, out_dir,
) -> Path:
    snap_steps = [s for s, _ in snapshots]
    snap_f = [np.asarray(f) for _, f in snapshots]
    snap_u = [np.asarray(forward(jnp.asarray(f))) for f in snap_f]
    VX_np = np.asarray(mesh.VX)

    def dg_data(arr):
        u_pieces = interp_plot @ arr
        x_pieces = np.zeros_like(u_pieces)
        for k in range(K):
            x_pieces[:, k] = VX_np[k] + 0.5 * (r_plot + 1.0) * (VX_np[k + 1] - VX_np[k])
        return x_pieces, u_pieces

    true_xf, true_yf = dg_data(f_true_np)
    true_xu, true_yu = dg_data(target_np)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [3, 2.5, 2.2]},
    )
    for k in range(K):
        axes[0].plot(true_xf[:, k], true_yf[:, k], "k-", lw=2.2, alpha=0.7,
                     label="f_true(x)" if k == 0 else None)
    cur_f_lines = []
    for k in range(K):
        line, = axes[0].plot([], [], color="#d62728", lw=1.4,
                              label="f_recovered(x)" if k == 0 else None)
        cur_f_lines.append(line)
    axes[0].set_xlim(np.asarray(mesh.VX)[0], np.asarray(mesh.VX)[-1])
    yf_lo = min(float(f_true_np.min()), -0.1) - 0.5
    yf_hi = max(float(f_true_np.max()), 0.1) + 0.5
    axes[0].set_ylim(yf_lo, yf_hi)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title(
        f"Source recovery for -u'' = f  (N={N}, K={K}, linear and convex)"
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
        axes[1].plot(true_xu[:, k], true_yu[:, k], "k-", lw=2.2, alpha=0.7,
                     label="u_target" if k == 0 else None)
    cur_u_lines = []
    for k in range(K):
        line, = axes[1].plot([], [], color="#d62728", lw=1.4,
                              label="u_h under f_recovered" if k == 0 else None)
        cur_u_lines.append(line)
    axes[1].set_xlim(np.asarray(mesh.VX)[0], np.asarray(mesh.VX)[-1])
    yu_lo = min(float(target_np.min()), -0.01) - 0.01
    yu_hi = max(float(target_np.max()), 0.01) + 0.01
    axes[1].set_ylim(yu_lo, yu_hi)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    ax_loss = axes[2]
    ax_err = ax_loss.twinx()
    loss_line, = ax_loss.semilogy([], [], color="#2ca02c", lw=1.5, label="data loss")
    err_line, = ax_err.semilogy([], [], color="#ff7f0e", lw=1.5,
                                label="||f - f_true|| / ||f_true||")
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
        xf, yf = dg_data(snap_f[frame_idx])
        xu, yu = dg_data(snap_u[frame_idx])
        for k in range(K):
            cur_f_lines[k].set_data(xf[:, k], yf[:, k])
            cur_u_lines[k].set_data(xu[:, k], yu[:, k])
        upto = step + 1
        steps_arr = np.arange(upto)
        loss_line.set_data(steps_arr, loss_history[:upto])
        err_line.set_data(steps_arr, param_err_history[:upto])
        cur_loss.set_data([step], [loss_history[step]])
        cur_err.set_data([step], [param_err_history[step]])
        label_artist.set_text(
            f"step {step:4d} / {n_steps}\n"
            f"loss = {loss_history[step]:.2e}\n"
            f"||f - f_true||/||f_true|| = {param_err_history[step]:.2e}"
        )
        return cur_f_lines + cur_u_lines + [
            loss_line, err_line, cur_loss, cur_err, label_artist,
        ]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "source_id_poisson_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
