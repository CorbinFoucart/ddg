"""Spatially-varying material-parameter identification for the 1D heat equation.

Twin experiment for ``κ(x)`` recovery: cook up an unknown
thermal-diffusivity field, forward-solve ``u_t = κ(x) u_xx`` with a
known initial condition and Dirichlet boundaries, then start from a
trivial guess and let ``jax.grad`` recover the truth from the observed
temperature trajectory.

The unknown lives *inside the operator* (coefficient of the Laplacian),
which makes this complementary to the BC-control demo (unknown at
the boundary) and the IC-recovery demo (unknown in the initial state).
All three rely on the same autodiff stack with no special-casing.

Parameterisation
----------------
``κ(x)`` is represented by a **10-bump radial-basis expansion** of its
logarithm,

    log κ(x) = Σ_{i=1..10} θ_i * exp(-((x - c_i) / w)²)

with fixed centres ``c_i`` evenly spaced on the domain and a fixed
width ``w``. This restricts ``κ`` to a 10-dimensional smooth manifold,
which matches the bandwidth of information actually present in heat
observations and makes the inverse problem well-posed. A nodal
parameterisation (Np × K = 48 free values) is *under*-determined by the
data: heat acts as a low-pass filter on ``κ`` so the high-frequency
modes are in the null space, and the optimiser produces a κ that fits
the data but bears no resemblance to the truth.

Why multi-time observation
--------------------------
Even with a 10-dim parameterisation, a single final-time observation
leaves some null space. Observing at 5 evenly-spaced times during
integration (same trick as the BC twin demo) and summing the L2 errors
across all of them collapses the null space and recovery becomes
exact to machine precision.
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
T_FINAL = 0.05
N_TIME_STEPS = 2000   # parabolic CFL ~ h^2 / κ_max => need fine dt for κ_max ~ 2
N_BASIS = 10           # number of RBF basis functions for κ
BASIS_WIDTH = 0.08    # RBF width — chosen so neighbouring bumps overlap


def heat_forward_at(mesh, kappa_field, dt, obs_indices):
    """Forward solve with array κ; return ``u`` at the requested step indices.

    Initial condition is a *localised* Gaussian pulse rather than a single
    Fourier mode. A pulse excites a broad spectrum of spatial scales, so
    diffusion through different κ regions imprints a much richer signal
    in the trajectory — without this, the dominant Fourier mode just
    decays uniformly and almost no information about κ(x) reaches the
    observation.
    """
    rhs = partial(heat_rhs, mesh, kappa=kappa_field)
    u0 = jnp.exp(-100.0 * (mesh.x - 0.6) ** 2)

    def body(carry, _):
        u, t = carry
        u_new = low_storage_rk4_step(lambda u, t: rhs(u, t), u, t, dt)
        return (u_new, t + dt), u_new

    (_, _), traj = jax.lax.scan(body, (u0, 0.0), jnp.arange(N_TIME_STEPS))
    return u0, traj[obs_indices]


def main() -> Path:
    N, K = 3, 12
    n_opt_steps = 2500
    learning_rate = 0.5
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

    n_obs = 20
    obs_indices = jnp.linspace(N_TIME_STEPS // n_obs, N_TIME_STEPS,
                               n_obs).astype(jnp.int32) - 1

    # Radial-basis expansion of log κ(x).
    basis_centers = jnp.linspace(XMIN, XMAX, N_BASIS)
    # Shape (N_BASIS, Np, K): one bump per basis function evaluated at every node.
    rbf_matrix = jnp.exp(
        -((mesh.x[None, :, :] - basis_centers[:, None, None]) / BASIS_WIDTH) ** 2
    )

    def kappa_from_theta(theta):
        log_kappa = jnp.tensordot(theta, rbf_matrix, axes=1)
        return jnp.exp(log_kappa)

    # Cook up θ_true such that κ_true(x) has a hot bump near x = 0.35
    # and a cold trough near x = 0.75.
    theta_true_np = np.zeros(N_BASIS)
    centers_np = np.asarray(basis_centers)
    for i, c in enumerate(centers_np):
        # weight by closeness to the chosen feature centres
        theta_true_np[i] = (
            0.6 * np.exp(-((c - 0.35) / 0.10) ** 2)
            - 0.3 * np.exp(-((c - 0.75) / 0.12) ** 2)
        )
    theta_true = jnp.asarray(theta_true_np)

    kappa_true = kappa_from_theta(theta_true)
    u0_, u_target_traj = heat_forward_at(mesh, kappa_true, dt, obs_indices)
    kappa_true_np = np.asarray(kappa_true)
    print(f"κ_true range: [{kappa_true_np.min():.3f}, {kappa_true_np.max():.3f}]")

    M_ref = jnp.linalg.inv(mesh.V @ mesh.V.T)
    theta_true_norm = float(jnp.linalg.norm(theta_true))

    def loss_fn(theta):
        kappa = kappa_from_theta(theta)
        _, u_traj = heat_forward_at(mesh, kappa, dt, obs_indices)
        diff = u_traj - u_target_traj
        per_time = jnp.sum(
            mesh.J[None, :, :] * (diff * jax.vmap(lambda d: M_ref @ d)(diff)),
            axis=(1, 2),
        )
        return jnp.sum(per_time)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    theta = jnp.zeros(N_BASIS)   # κ ≡ 1 initial guess
    velocity = jnp.zeros_like(theta)

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
        velocity = momentum * velocity - lr_at(step) * grad
        theta = theta + velocity

    val_final, _ = value_and_grad(theta)
    loss_history.append(float(val_final))
    param_err_history.append(
        float(jnp.linalg.norm(theta - theta_true)) / max(theta_true_norm, 1e-12)
    )
    snapshots.append((n_opt_steps, np.asarray(theta)))

    print(f"loss: initial = {loss_history[0]:.3e}  ->  final = {loss_history[-1]:.3e}")
    print(f"||θ - θ_true|| / ||θ_true||: {param_err_history[0]:.3e}  ->  {param_err_history[-1]:.3e}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    theta_final = jnp.asarray(snapshots[-1][1])
    theta_final_np = np.asarray(theta_final)
    kappa_final_np = np.asarray(kappa_from_theta(theta_final))

    # --------- static PNG ----------
    from ddg.dg1d.reference import vandermonde_1d
    n_plot = 50
    r_plot = jnp.linspace(-1.0, 1.0, n_plot)
    V_plot = np.asarray(vandermonde_1d(N, r_plot))
    interp_plot = V_plot @ np.asarray(jnp.linalg.inv(mesh.V))
    VX_np = np.asarray(mesh.VX)

    def plot_dg(ax, u_array, color, label, lw=1.5, alpha=1.0, linestyle="-"):
        for k in range(K):
            x_plot = VX_np[k] + 0.5 * (np.asarray(r_plot) + 1.0) * (VX_np[k + 1] - VX_np[k])
            u_plot = interp_plot @ u_array[:, k]
            ax.plot(x_plot, u_plot, color=color, lw=lw, alpha=alpha,
                    linestyle=linestyle,
                    label=label if k == 0 else None)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [3, 2.5, 2.2]},
    )

    plot_dg(axes[0], kappa_true_np, "k", "κ_true(x)  (unknown)", lw=2.2, alpha=0.7)
    plot_dg(axes[0], kappa_final_np, "#d62728", "κ_recovered(x)", lw=1.4)
    plot_dg(axes[0], np.ones_like(kappa_true_np), "#1f77b4",
            "initial guess κ ≡ 1", lw=1.0, alpha=0.4, linestyle="--")
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("κ(x)")
    axes[0].set_title(
        f"Material-parameter identification:  recover κ(x) from u(x, t)  (N={N}, K={K})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    target_final_np = np.asarray(u_target_traj[-1])
    plot_dg(axes[1], target_final_np, "k",
            f"u(x, T) under κ_true  (T={T_FINAL})", lw=2.2, alpha=0.7)
    _, u_obs_recovered = jax.jit(
        lambda th: heat_forward_at(mesh, kappa_from_theta(th), dt, obs_indices)
    )(theta_final)
    plot_dg(axes[1], np.asarray(u_obs_recovered[-1]), "#d62728",
            "u(x, T) under κ_recovered", lw=1.4)
    axes[1].set_xlim(XMIN, XMAX)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x, T)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    ax_loss = axes[2]
    ax_err = ax_loss.twinx()
    ax_loss.semilogy(loss_history, color="#2ca02c", lw=1.5,
                     label="data loss  (multi-time L²)")
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
    png_path = out_dir / "material_id_heat_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "material_id_heat_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": K, "T_final": T_FINAL,
                "n_time_steps": N_TIME_STEPS, "n_opt_steps": n_opt_steps,
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

    kappa_from_theta_jit = jax.jit(kappa_from_theta)
    forward_jit = jax.jit(
        lambda th: heat_forward_at(mesh, kappa_from_theta(th), dt, obs_indices)[1][-1]
    )

    mp4_path = _render_animation(
        snapshots=snapshots,
        loss_history=loss_history,
        param_err_history=param_err_history,
        n_steps=n_opt_steps,
        N=N, K=K, mesh=mesh,
        kappa_true_np=kappa_true_np,
        target_final_np=target_final_np,
        forward_fn=forward_jit,
        kappa_from_theta=kappa_from_theta_jit,
        interp_plot=interp_plot,
        r_plot=np.asarray(r_plot),
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_animation(
    *, snapshots, loss_history, param_err_history, n_steps, N, K, mesh,
    kappa_true_np, target_final_np, forward_fn, kappa_from_theta,
    interp_plot, r_plot, out_dir,
) -> Path:
    snap_steps = [s for s, _ in snapshots]
    snap_kappa = [np.asarray(kappa_from_theta(jnp.asarray(th))) for _, th in snapshots]
    snap_uT = [np.asarray(forward_fn(th)) for _, th in snapshots]
    VX_np = np.asarray(mesh.VX)

    def dg_data(u_array):
        u_pieces = interp_plot @ u_array
        x_pieces = np.zeros_like(u_pieces)
        for k in range(K):
            x_pieces[:, k] = VX_np[k] + 0.5 * (r_plot + 1.0) * (VX_np[k + 1] - VX_np[k])
        return x_pieces, u_pieces

    true_xk, true_yk = dg_data(kappa_true_np)
    true_xT, true_yT = dg_data(target_final_np)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [3, 2.5, 2.2]},
    )
    for k in range(K):
        axes[0].plot(true_xk[:, k], true_yk[:, k], "k-", lw=2.2, alpha=0.7,
                     label="κ_true(x)" if k == 0 else None)
    cur_k_lines = []
    for k in range(K):
        line, = axes[0].plot([], [], color="#d62728", lw=1.4,
                              label="κ_recovered(x)" if k == 0 else None)
        cur_k_lines.append(line)
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_ylim(0.0, float(np.max(kappa_true_np)) * 1.1)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("κ(x)")
    axes[0].set_title(
        f"Material identification through 1D heat (N={N}, K={K})"
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
                     label="u(·, T) under κ_true" if k == 0 else None)
    cur_T_lines = []
    for k in range(K):
        line, = axes[1].plot([], [], color="#d62728", lw=1.4,
                              label="u(·, T) under κ_recovered" if k == 0 else None)
        cur_T_lines.append(line)
    axes[1].set_xlim(XMIN, XMAX)
    yT_lo = float(np.min(target_final_np)) - 0.05
    yT_hi = float(np.max(target_final_np)) + 0.05
    axes[1].set_ylim(yT_lo, yT_hi)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x, T)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

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
        xk, yk = dg_data(snap_kappa[frame_idx])
        xT, yT = dg_data(snap_uT[frame_idx])
        for k in range(K):
            cur_k_lines[k].set_data(xk[:, k], yk[:, k])
            cur_T_lines[k].set_data(xT[:, k], yT[:, k])
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
        return cur_k_lines + cur_T_lines + [
            loss_line, err_line, cur_loss, cur_err, label_artist,
        ]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "material_id_heat_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
