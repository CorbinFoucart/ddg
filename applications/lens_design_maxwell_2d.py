"""Inverse permittivity design through 2D TM Maxwell.

Twin experiment in two dimensions: place a Gaussian Ez pulse inside a
square PEC cavity, let the wave bounce through a heterogeneous
material ``ε(x, y)`` (an unknown "lens"), and observe ``Ez(x, y)`` at
a single final time T. Recover ``ε(x, y)`` from that one snapshot.

This is the 2D analog of ``wave_speed_inversion_1d.py`` but with full-
field rather than point-receiver observation. The map
``ε → Ez(·, T)`` runs through:

    ε parameter (radial-basis amplitudes)
        → ε(x, y) at every triangle node
            → 2D Maxwell TM RHS (now ε-aware) at every RK4 sub-stage
                → N_TIME_STEPS forward integration via lax.scan
                    → L2 inner product against the observed final state

and ``jax.grad`` flows through the whole thing.

Outputs (``output/``)
---------------------
* ``lens_design_maxwell_2d.png``: ε_true vs ε_recovered (side-by-side
  heatmaps), the observed vs simulated final-time Ez, loss + param
  error curves.
* ``lens_design_maxwell_2d.json``: numerical record.
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ddg.dg1d.integrator import low_storage_rk4_step  # noqa: E402
from ddg.dg2d import (  # noqa: E402
    build_mesh_2d,
    maxwell_rhs_2d,
    uniform_rectangle_mesh_2d,
)


XMIN, XMAX = -1.0, 1.0
YMIN, YMAX = -1.0, 1.0
T_FINAL = 0.6
N_TIME_STEPS = 300
N_BASIS = 16   # 4x4 grid of RBFs
BASIS_WIDTH = 0.30


def wave_forward(mesh, eps_field, dt):
    """Forward solve 2D Maxwell TM and return the final ``(Hx, Hy, Ez)``."""
    mu_field = jnp.ones_like(eps_field)

    def rhs_state(state, t):
        Hx, Hy, Ez = state
        return maxwell_rhs_2d(mesh, Hx, Hy, Ez, t, eps=eps_field, mu=mu_field)

    Ez0 = jnp.exp(-50.0 * ((mesh.x + 0.5) ** 2 + (mesh.y + 0.5) ** 2))
    Hx0 = jnp.zeros_like(mesh.x)
    Hy0 = jnp.zeros_like(mesh.x)

    def body(carry, _):
        state, t = carry
        new_state = low_storage_rk4_step(rhs_state, state, t, dt)
        return (new_state, t + dt), None

    (state_final, _), _ = jax.lax.scan(
        body, ((Hx0, Hy0, Ez0), 0.0), jnp.arange(N_TIME_STEPS),
    )
    return state_final


def main() -> Path:
    N = 3
    K_side = 10
    n_opt_steps = 300
    learning_rate = 0.05
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    VX, VY, EToV = uniform_rectangle_mesh_2d(XMIN, XMAX, YMIN, YMAX, K_side, K_side)
    mesh = build_mesh_2d(VX, VY, EToV, N=N)
    dt = T_FINAL / N_TIME_STEPS
    print(f"K = {mesh.K} triangles, Np = {mesh.Np}, dofs/field = {mesh.K * mesh.Np}")

    # 4x4 grid of RBF basis bumps over the domain.
    centers_x = jnp.linspace(XMIN, XMAX, 4)
    centers_y = jnp.linspace(YMIN, YMAX, 4)
    CX, CY = jnp.meshgrid(centers_x, centers_y, indexing="ij")
    centers_flat_x = CX.reshape(-1)
    centers_flat_y = CY.reshape(-1)

    rbf_matrix = jnp.exp(
        -(
            (mesh.x[None, :, :] - centers_flat_x[:, None, None]) ** 2
            + (mesh.y[None, :, :] - centers_flat_y[:, None, None]) ** 2
        ) / BASIS_WIDTH ** 2
    )

    def eps_from_theta(theta):
        log_eps = jnp.tensordot(theta, rbf_matrix, axes=1)
        return jnp.exp(log_eps)

    # Cook up θ_true: a positive bump at (0.2, 0.2) (high ε = slow), a
    # negative bump near (-0.4, 0.3) (low ε = fast).
    theta_true_np = np.zeros(N_BASIS)
    for i in range(N_BASIS):
        cx = float(centers_flat_x[i])
        cy = float(centers_flat_y[i])
        theta_true_np[i] = (
            0.8 * np.exp(-((cx - 0.2) ** 2 + (cy - 0.2) ** 2) / 0.05)
            - 0.4 * np.exp(-((cx + 0.4) ** 2 + (cy - 0.3) ** 2) / 0.05)
        )
    theta_true = jnp.asarray(theta_true_np)
    eps_true = eps_from_theta(theta_true)
    print(f"ε_true range: [{float(jnp.min(eps_true)):.3f}, {float(jnp.max(eps_true)):.3f}]")

    Hx_T, Hy_T, Ez_T_target = wave_forward(mesh, eps_true, dt)
    Ez_T_target_np = np.asarray(Ez_T_target)
    theta_true_norm = float(jnp.linalg.norm(theta_true))

    def loss_fn(theta):
        eps_field = eps_from_theta(theta)
        _, _, Ez_T = wave_forward(mesh, eps_field, dt)
        diff = Ez_T - Ez_T_target
        return jnp.mean(diff ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    theta = jnp.zeros(N_BASIS)
    m_t = jnp.zeros_like(theta)
    v_t = jnp.zeros_like(theta)

    loss_history: list[float] = []
    param_err_history: list[float] = []
    for step in range(n_opt_steps):
        val, grad = value_and_grad(theta)
        loss_history.append(float(val))
        param_err_history.append(
            float(jnp.linalg.norm(theta - theta_true)) / max(theta_true_norm, 1e-12)
        )
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

    print(f"loss: {loss_history[0]:.3e} -> {loss_history[-1]:.3e}")
    print(f"||θ - θ_true|| / ||θ_true||: {param_err_history[0]:.3e} -> {param_err_history[-1]:.3e}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- static PNG ----------
    from scipy.spatial import Delaunay
    from matplotlib.tri import Triangulation

    eps_final = np.asarray(eps_from_theta(theta))
    _, _, Ez_T_final = wave_forward(mesh, eps_from_theta(theta), dt)
    Ez_T_final_np = np.asarray(Ez_T_final)

    # Build sub-triangulation per macro-element for nice rendering.
    r_np = np.asarray(mesh.r)
    s_np = np.asarray(mesh.s)
    ref_simplices = Delaunay(np.stack([r_np, s_np], axis=1)).simplices
    n_ref_tri = ref_simplices.shape[0]
    tris = np.empty((mesh.K * n_ref_tri, 3), dtype=np.int64)
    for k in range(mesh.K):
        tris[k * n_ref_tri:(k + 1) * n_ref_tri] = ref_simplices + k * mesh.Np
    x_flat = np.asarray(mesh.x).flatten(order="F")
    y_flat = np.asarray(mesh.y).flatten(order="F")
    triang = Triangulation(x_flat, y_flat, triangles=tris)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    eps_true_flat = np.asarray(eps_true).T.flatten()
    eps_final_flat = eps_final.T.flatten()
    Ez_target_flat = Ez_T_target_np.T.flatten()
    Ez_final_flat = Ez_T_final_np.T.flatten()

    vmax_eps = max(float(np.max(eps_true_flat)), float(np.max(eps_final_flat)))
    vmin_eps = min(float(np.min(eps_true_flat)), float(np.min(eps_final_flat)))
    vmax_Ez = max(float(np.max(np.abs(Ez_target_flat))), float(np.max(np.abs(Ez_final_flat))))

    for ax, data, title, cmap, vmin, vmax in [
        (axes[0, 0], eps_true_flat, "ε_true(x, y)", "viridis", vmin_eps, vmax_eps),
        (axes[0, 1], eps_final_flat, "ε_recovered(x, y)", "viridis", vmin_eps, vmax_eps),
        (axes[0, 2], eps_final_flat - eps_true_flat, "ε_recovered - ε_true", "RdBu_r",
         -0.3 * (vmax_eps - vmin_eps), 0.3 * (vmax_eps - vmin_eps)),
        (axes[1, 0], Ez_target_flat, "Ez(·, T) target", "RdBu_r", -vmax_Ez, vmax_Ez),
        (axes[1, 1], Ez_final_flat, "Ez(·, T) recovered", "RdBu_r", -vmax_Ez, vmax_Ez),
    ]:
        tpc = ax.tripcolor(triang, data, shading="gouraud", cmap=cmap,
                           vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    ax_curves = axes[1, 2]
    ax_err = ax_curves.twinx()
    ax_curves.semilogy(loss_history, color="#2ca02c", lw=1.5, label="data loss (MSE on Ez(T))")
    ax_err.semilogy(param_err_history, color="#ff7f0e", lw=1.5,
                    label="||θ - θ_true|| / ||θ_true||")
    ax_curves.set_xlabel("optimisation step")
    ax_curves.set_ylabel("data loss", color="#2ca02c")
    ax_err.set_ylabel("parameter error (relative)", color="#ff7f0e")
    ax_curves.tick_params(axis="y", labelcolor="#2ca02c")
    ax_err.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_curves.grid(alpha=0.3, which="both")
    lines_l, labels_l = ax_curves.get_legend_handles_labels()
    lines_e, labels_e = ax_err.get_legend_handles_labels()
    ax_curves.legend(lines_l + lines_e, labels_l + labels_e, loc="center right",
                     fontsize=9)

    fig.suptitle(
        f"Inverse permittivity design through 2D Maxwell  (N={N}, {mesh.K} triangles, T={T_FINAL})",
        fontsize=13,
    )
    fig.tight_layout()
    png_path = out_dir / "lens_design_maxwell_2d.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "lens_design_maxwell_2d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": int(mesh.K), "K_side": K_side, "T_final": T_FINAL,
                "n_time_steps": N_TIME_STEPS, "n_basis": N_BASIS,
                "basis_width": BASIS_WIDTH, "n_opt_steps": n_opt_steps,
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
    return png_path


if __name__ == "__main__":
    main()
