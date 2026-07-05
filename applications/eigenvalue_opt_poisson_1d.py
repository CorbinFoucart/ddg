"""Eigenvalue optimisation through the differentiable Poisson operator.

Punchline: the assembled IPDG stiffness matrix ``A(VX)`` depends
smoothly on the mesh vertex positions ``VX``, and its eigenvalues
``λ_i(VX) = eigh(A(VX))[i]`` are smooth functions of ``VX`` away from
spectral crossings. ``jax.grad`` composes with the eigendecomposition,
so we can optimise mesh layout to drive a chosen eigenvalue to a
target — no perturbation-theory derivation required.

Setup
-----
``-u'' = λ u`` on ``(0, 1)`` with homogeneous Dirichlet BCs. On a
uniform mesh of order ``N``, ``K`` elements, the smallest eigenvalue
of the IPDG operator approximates ``π²`` (the continuous answer).
We pick a target ``λ_target = 12.0`` (which is ``> π² ≈ 9.87``, so
the mesh has to *compress* somewhere — concentrating cells creates a
narrower effective domain in the eigenvalue sense).

Free parameters: the same softmax-element-widths reparameterisation
used by the shape-opt demos, optimised so that
``λ_1(VX) ≈ λ_target``.

Outputs (``output/``)
---------------------
* ``eigenvalue_opt_poisson_1d.png``: uniform vs optimised mesh,
  eigenvalue convergence curve.
* ``eigenvalue_opt_poisson_1d.mp4`` / ``.gif``: optimisation animation.
* ``eigenvalue_opt_poisson_1d.json``: numerical record.
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

from ddg.dg1d import build_mesh_1d, uniform_mesh_1d, update_mesh_vertices  # noqa: E402
from ddg.dg1d.poisson import _local_mass, poisson_ip_apply  # noqa: E402


XMIN, XMAX = 0.0, 1.0
LAMBDA_TARGET = 0.5    # smallest eigenvalue of assembled IPDG operator;
                       # uniform K=8 N=4 gives ~ 0.246, target ~ 2x above


def widths_from_logits(logits, length):
    return jax.nn.softmax(logits) * length


def vx_from_widths(widths, x0):
    return jnp.concatenate([jnp.array([x0]), x0 + jnp.cumsum(widths)])


def assemble_stiffness(mesh):
    """Dense IPDG stiffness matrix on the current mesh.

    Built via ``jax.jacobian`` on the linear apply function (same trick
    as ``poisson_ip_matrix``, but re-implemented here so it traces
    through ``mesh.x``, ``mesh.J``, ``mesh.rx``, ... -- the geometric
    fields that depend on ``VX``).
    """
    n_dof = mesh.Np * mesh.K

    def op_flat(u_flat):
        u = u_flat.reshape(mesh.K, mesh.Np).T
        return poisson_ip_apply(mesh, u).T.reshape(-1)

    return jax.jacobian(op_flat)(jnp.zeros(n_dof))


def smallest_eigenvalue(mesh):
    """Smallest eigenvalue of the assembled IPDG stiffness ``A = M(-Δ_h)``.

    Note: we use the *un-normalised* operator (no ``M^{-1}`` pull-out)
    on purpose — its eigenvalues depend non-trivially on the mesh
    through the block-diagonal mass factor, so the optimisation
    landscape is non-flat. For the *physical* eigenvalues of ``-u''``
    (which on Dirichlet ``[0, 1]`` are ``(kπ)²`` independent of mesh
    distribution) you'd solve the generalised problem ``Av = λ M v``;
    but that landscape is essentially flat for any reasonable mesh,
    which makes for a poor demo.

    The forward map ``VX → eigvals[0](A(VX))`` is everything we need
    to showcase: ``jax.grad`` composes with ``jnp.linalg.eigvalsh``,
    and the mesh that minimises ``|eigvals[0] - target|`` exists and
    is reachable by gradient descent.
    """
    A = assemble_stiffness(mesh)
    A_sym = (A + A.T) / 2.0
    return jnp.linalg.eigvalsh(A_sym)[0]


def main() -> Path:
    N, K = 4, 8
    n_opt_steps = 600
    learning_rate = 0.05
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    VX_init, EToV = uniform_mesh_1d(XMIN, XMAX, K)
    mesh_init = build_mesh_1d(VX_init, EToV, N=N)
    domain_length = XMAX - XMIN

    # Baseline (uniform) eigenvalue.
    lambda1_uniform = float(smallest_eigenvalue(mesh_init))
    print(f"λ₁(uniform) = {lambda1_uniform:.6f}  (π² = {np.pi**2:.6f})")
    print(f"target λ = {LAMBDA_TARGET}")

    def loss_fn(logits):
        widths = widths_from_logits(logits, domain_length)
        VX = vx_from_widths(widths, XMIN)
        mesh = update_mesh_vertices(mesh_init, VX)
        lam = smallest_eigenvalue(mesh)
        return (lam - LAMBDA_TARGET) ** 2

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    logits = jnp.zeros(K)
    m_t = jnp.zeros_like(logits)
    v_t = jnp.zeros_like(logits)

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
        m_t = beta1 * m_t + (1.0 - beta1) * grad
        v_t = beta2 * v_t + (1.0 - beta2) * grad ** 2
        m_hat = m_t / (1.0 - beta1 ** (step + 1))
        v_hat = v_t / (1.0 - beta2 ** (step + 1))
        logits = logits - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps_adam)
    val_final, _ = value_and_grad(logits)
    loss_history.append(float(val_final))
    snapshots.append((n_opt_steps, np.asarray(logits)))

    # Recompute final λ
    widths_final = widths_from_logits(logits, domain_length)
    VX_final = vx_from_widths(widths_final, XMIN)
    mesh_final = update_mesh_vertices(mesh_init, VX_final)
    lambda1_final = float(smallest_eigenvalue(mesh_final))
    print(f"λ₁(optimised) = {lambda1_final:.6f}")
    print(f"|λ - target| = {abs(lambda1_final - LAMBDA_TARGET):.3e}")

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    VX_uniform_np = np.asarray(VX_init)
    VX_final_np = np.asarray(VX_final)

    # Compute λ at every recorded snapshot.
    @jax.jit
    def lam_of(logits_):
        widths = widths_from_logits(logits_, domain_length)
        VX = vx_from_widths(widths, XMIN)
        mesh = update_mesh_vertices(mesh_init, VX)
        return smallest_eigenvalue(mesh)

    snap_lam = [float(lam_of(jnp.asarray(th))) for _, th in snapshots]

    # ---------- static PNG ----------
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [1.5, 2, 2]},
    )

    axes[0].plot(VX_uniform_np, np.full_like(VX_uniform_np, 1), "|",
                 color="#1f77b4", markersize=20, markeredgewidth=2,
                 label=f"uniform  (λ₁ = {lambda1_uniform:.4f})")
    axes[0].plot(VX_final_np, np.full_like(VX_final_np, 0), "|",
                 color="#d62728", markersize=20, markeredgewidth=2,
                 label=f"optimised  (λ₁ = {lambda1_final:.4f})")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["optimised", "uniform"])
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_ylim(-0.5, 1.5)
    axes[0].set_xlabel("x")
    axes[0].set_title(
        f"Eigenvalue optimisation: place λ₁ at target  (N={N}, K={K}, target = {LAMBDA_TARGET})"
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].plot(np.arange(len(loss_history)),
                 [np.pi**2] * len(loss_history),
                 color="#888888", lw=1.0, linestyle=":", label=f"π² = {np.pi**2:.4f}")
    axes[1].axhline(LAMBDA_TARGET, color="#1f77b4", lw=1.2, linestyle="--",
                    label=f"target = {LAMBDA_TARGET}")
    # Plot the eigenvalue at recorded snapshots, linearly interpolated.
    snap_steps = [s for s, _ in snapshots]
    lam_curve = np.interp(np.arange(len(loss_history)), snap_steps, snap_lam)
    axes[1].plot(lam_curve, color="#d62728", lw=1.4, label="λ₁(VX) during opt")
    axes[1].set_xlabel("optimisation step")
    axes[1].set_ylabel("λ₁")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].semilogy(loss_history, color="#2ca02c", lw=1.5,
                     label="(λ₁ - target)²")
    axes[2].set_xlabel("optimisation step")
    axes[2].set_ylabel("eigenvalue-mismatch loss")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    png_path = out_dir / "eigenvalue_opt_poisson_1d.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")

    json_path = out_dir / "eigenvalue_opt_poisson_1d.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "N": N, "K": K, "n_opt_steps": n_opt_steps,
                "lambda_target": LAMBDA_TARGET,
                "lambda_uniform": lambda1_uniform,
                "lambda_final": lambda1_final,
                "VX_uniform": VX_uniform_np.tolist(),
                "VX_optimised": VX_final_np.tolist(),
                "loss_history": loss_history,
                "lambda_history": lam_curve.tolist(),
            },
            fh, indent=2,
        )
    print(f"wrote {json_path}")

    mp4_path = _render_animation(
        snapshots=snapshots,
        snap_lam=snap_lam,
        loss_history=loss_history,
        n_steps=n_opt_steps,
        domain_length=domain_length,
        widths_from_logits=widths_from_logits,
        vx_from_widths=vx_from_widths,
        VX_uniform_np=VX_uniform_np,
        lambda1_uniform=lambda1_uniform,
        out_dir=out_dir,
    )
    print(f"wrote {mp4_path}")
    return png_path


def _render_animation(
    *, snapshots, snap_lam, loss_history, n_steps, domain_length,
    widths_from_logits, vx_from_widths, VX_uniform_np, lambda1_uniform,
    out_dir,
) -> Path:
    snap_steps = [s for s, _ in snapshots]
    snap_VX = []
    for _, th in snapshots:
        widths = np.asarray(widths_from_logits(jnp.asarray(th), domain_length))
        VX = np.concatenate([[XMIN], XMIN + np.cumsum(widths)])
        snap_VX.append(VX)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9.6),
        gridspec_kw={"height_ratios": [1.5, 2, 2]},
    )
    axes[0].plot(VX_uniform_np, np.full_like(VX_uniform_np, 1), "|",
                 color="#1f77b4", markersize=20, markeredgewidth=2,
                 label=f"uniform  (λ₁ = {lambda1_uniform:.4f})")
    current_dots, = axes[0].plot([], [], "|", color="#d62728",
                                  markersize=20, markeredgewidth=2,
                                  label="current mesh")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["current", "uniform"])
    axes[0].set_xlim(XMIN, XMAX)
    axes[0].set_ylim(-0.5, 1.5)
    axes[0].set_xlabel("x")
    axes[0].set_title(
        f"Eigenvalue optimisation through Poisson  (target λ₁ = {LAMBDA_TARGET})"
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3, axis="x")
    label_artist = axes[0].text(
        0.02, 0.97, "", transform=axes[0].transAxes, ha="left", va="top",
        family="monospace", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="none"),
    )

    axes[1].axhline(np.pi**2, color="#888888", lw=1.0, linestyle=":",
                    label=f"π² = {np.pi**2:.4f}")
    axes[1].axhline(LAMBDA_TARGET, color="#1f77b4", lw=1.2, linestyle="--",
                    label=f"target = {LAMBDA_TARGET}")
    lam_line, = axes[1].plot([], [], "-o", color="#d62728", lw=1.4, markersize=4,
                              label="λ₁(VX)")
    axes[1].set_xlim(0, n_steps)
    lo = min(min(snap_lam), float(np.pi**2)) - 0.5
    hi = max(max(snap_lam), LAMBDA_TARGET) + 1.0
    axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel("optimisation step")
    axes[1].set_ylabel("λ₁")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")

    loss_line, = axes[2].semilogy([], [], color="#2ca02c", lw=1.5,
                                  label="(λ₁ - target)²")
    cur_loss, = axes[2].semilogy([], [], "o", color="#2ca02c", markersize=7)
    axes[2].set_xlim(0, n_steps)
    axes[2].set_ylim(min(loss_history) / 2, max(loss_history) * 2)
    axes[2].set_xlabel("optimisation step")
    axes[2].set_ylabel("eigenvalue-mismatch loss")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(loc="upper right")
    fig.tight_layout()

    def update(frame_idx):
        step = snap_steps[frame_idx]
        current_dots.set_data(snap_VX[frame_idx], np.zeros_like(snap_VX[frame_idx]))
        # eigenvalue plotted at snapshots so far
        lam_line.set_data(snap_steps[:frame_idx + 1], snap_lam[:frame_idx + 1])
        upto = step + 1
        loss_line.set_data(np.arange(upto), loss_history[:upto])
        cur_loss.set_data([step], [loss_history[step]])
        label_artist.set_text(
            f"step {step:4d} / {n_steps}\n"
            f"λ₁ = {snap_lam[frame_idx]:.4f}\n"
            f"|λ - target| = {abs(snap_lam[frame_idx] - LAMBDA_TARGET):.2e}"
        )
        return [current_dots, lam_line, loss_line, cur_loss, label_artist]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), blit=False, interval=70,
    )
    writer = animation.FFMpegWriter(
        fps=15, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    mp4_path = out_dir / "eigenvalue_opt_poisson_1d.mp4"
    anim.save(str(mp4_path), writer=writer, dpi=110)
    plt.close(fig)
    return mp4_path


if __name__ == "__main__":
    main()
