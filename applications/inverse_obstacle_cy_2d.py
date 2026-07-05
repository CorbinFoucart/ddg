"""Parameter-inverse demo: recover the cylinder Y-position from sparse
velocity probes on the BDM cylinder transient.

Companion to ``scripts/inverse_nu_cylinder_2d.py`` (which recovers the
viscosity). Here the unknown is the cylinder centre's Y-coordinate
``CY`` --- an "object position" inverse problem.

Why finite differences
----------------------
The cylinder position enters the forward map through the mesh-build
chain (``channel_with_circle_mesh_2d`` + ``apply_circular_hole_2d``),
which is implemented host-side with NumPy --- gradients do not flow
back through it via ``jax.grad`` without a significant refactor of the
mesh module. The mesh-build cost is modest (~5 s) compared to the
forward time-step trajectory (~4 min for 10 BDF2 steps with two
Newton iters each at BDM_2 K = 432), so a single-parameter central
finite-difference gradient is cheap relative to the forward solve.
The Adam loop:

  for it in range(n_train):
      CY_plus  = CY + h ;  L_plus  = forward(CY_plus)
      CY_minus = CY - h ;  L_minus = forward(CY_minus)
      grad = (L_plus - L_minus) / (2 h)
      CY = CY - lr * grad

burns 2 forward calls per gradient + 1 to track loss at the current
iterate; with the JIT cache hot the marginal cost is roughly
3 * 4 min = 12 min per iter.

Probes
------
Six probes along ``y = CY_truth`` at ``x = CX + i D`` for
``i in {1, 2, 3, 4, 5, 6}``. These observe ``u_x`` and ``u_y`` at
each BDF step. The loss is the standard L2 mismatch on the probe
traces. Because the probes sit on the truth cylinder's wake
centreline, the trace pattern is exquisitely sensitive to the
cylinder's vertical position.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# JAX setup BEFORE we import any JAX module.
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inverse_nu_cylinder_2d import (
    make_step_nu_param, make_probe_fn,
)
from ddg.dg2d.bdm import BDM
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
)
from ddg.dg2d.bdm import bdm_k_evaluate_boundary_velocity
from ddg.dg2d.curved import apply_circular_hole_2d
from ddg.dg2d.mesh import build_mesh_2d, channel_with_circle_mesh_2d


# Geometry constants — must match diff_cylinder_dt_tau / inverse_nu_cylinder.
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 0.5
CX = 0.2
RADIUS = 0.05
D_CYL = 2.0 * RADIUS
U_MAX = 1.5
U_MEAN = (2.0 / 3.0) * U_MAX
NU = 1.0e-3
H_CHANNEL = Y_MAX - Y_MIN


def _parabolic_inflow(y):
    return 4.0 * U_MAX * y * (H_CHANNEL - y) / H_CHANNEL**2


def _bc_fn(x, y):
    on_inflow = x < X_MIN + 1e-6
    u_x = jnp.where(on_inflow, _parabolic_inflow(y), 0.0)
    u_y = jnp.zeros_like(y)
    return u_x, u_y


def build_static_at_cy(kx, ky, order, cy, n_radial=2, r_outer=0.1):
    """Same as diff_cylinder_dt_tau.build_static but cy is a parameter.

    Returns the V, BC arrays, and is_natural_face mask.
    """
    VX, VY, EToV = channel_with_circle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, kx, ky,
        cx=CX, cy=cy, radius=RADIUS,
        R_outer=r_outer, n_radial=n_radial,
        radial_growth=1.3,
    )
    mesh = build_mesh_2d(VX, VY, EToV, order)
    mesh = apply_circular_hole_2d(
        mesh, cx=CX, cy=cy, radius=RADIUS,
        blend_interior=True, detection_tol=0.05,
    )
    V = BDM.from_mesh(mesh, order=order)

    Nfaces = 3
    Fmask = np.asarray(mesh.Fmask)
    x_np = np.asarray(mesh.x); y_np = np.asarray(mesh.y)
    x_face = np.zeros((Nfaces, mesh.K))
    y_face = np.zeros((Nfaces, mesh.K))
    for f in range(Nfaces):
        x_face[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE_np = np.asarray(mesh.EToE)
    is_bdy = (EToE_np == np.arange(mesh.K)[:, None]).T
    is_inflow_face = is_bdy & (x_face < X_MIN + 1e-3)
    is_outflow_face = is_bdy & (x_face > X_MAX - 1e-3)
    is_outer_y = is_bdy & (
        (y_face < Y_MIN + 1e-3) | (y_face > Y_MAX - 1e-3)
    )
    r_from_cyl = np.sqrt((x_face - CX) ** 2 + (y_face - cy) ** 2)
    is_cyl_face = (
        is_bdy & (r_from_cyl < RADIUS * 1.05)
        & (~is_inflow_face) & (~is_outflow_face) & (~is_outer_y)
    )
    is_wall_face = is_outer_y
    is_dirichlet = jnp.asarray(
        (is_inflow_face | is_wall_face | is_cyl_face).T
    )
    is_natural_face = jnp.asarray(is_outflow_face.T)

    u_bc = bdm1_evaluate_dirichlet_bc(V, _bc_fn)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _bc_fn)
    bc_mask = bdm1_boundary_dofs_mask(V, dirichlet_face=is_dirichlet)
    return V, u_bc, u_bc_vec, bc_mask, is_natural_face


def make_loss_for_cy(*, kx, ky, order, n_newton, n_steps, dt, tau_scale,
                     nu, probe_xy, y_obs):
    """Build a host-callable loss(cy) that rebuilds the mesh, jit-compiles
    a single forward step bound to the new mesh, and integrates the
    trajectory. The jit cache is keyed on the structure of the operands
    so calling at a different cy fully recompiles.
    """
    n_steps_const = int(n_steps)
    def loss_for_cy(cy):
        cy_val = float(cy)
        V, u_bc, u_bc_vec, bc_mask, is_natural_face = build_static_at_cy(
            kx, ky, order, cy_val,
        )
        step_fn = make_step_nu_param(
            V, u_bc, u_bc_vec, bc_mask, is_natural_face,
            tau_scale=tau_scale, n_newton=n_newton,
        )
        probe_fn = make_probe_fn(V, probe_xy)
        step_jit = jax.jit(step_fn, static_argnames=("order_eff",))

        u_n = jnp.zeros(V.total_dofs)
        u_nm1 = jnp.zeros(V.total_dofs)
        traces = []
        for s in range(n_steps_const):
            order_eff = 1 if s == 0 else 2
            u_new, _ = step_jit(u_n, u_nm1, dt, nu, order_eff)
            traces.append(probe_fn(u_new))
            u_nm1, u_n = u_n, u_new
        y_pred = jnp.stack(traces)
        L = float(jnp.sum((y_pred - y_obs) ** 2))
        return L, np.asarray(y_pred)
    return loss_for_cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--cy-true", type=float, default=0.25)
    ap.add_argument("--cy-init", type=float, default=0.20)
    ap.add_argument("--fd-h", type=float, default=0.01,
                    help="finite-difference half-step in CY")
    ap.add_argument("--lr", type=float, default=0.003,
                    help="Adam lr on CY (linear, not log)")
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    print(f"BDM_{args.order} cylinder smoke: n_steps={args.n_steps}, "
          f"dt={args.dt}, T={args.n_steps * args.dt}",
          flush=True)
    print(f"  truth CY = {args.cy_true}, init CY = {args.cy_init}",
          flush=True)
    print(f"  FD half-step h = {args.fd_h}, lr = {args.lr}",
          flush=True)

    # Probe rake on the truth wake centreline.
    probe_xy = np.array(
        [[CX + i * D_CYL, args.cy_true] for i in (1, 2, 3, 4, 5, 6)],
        dtype=float,
    )
    n_probe = probe_xy.shape[0]

    # ---- Ground truth -------------------------------------------------
    print("generating ground truth at truth CY ...", flush=True)
    t0 = time.time()
    V_t, u_bc_t, u_bc_vec_t, bc_mask_t, is_nat_t = build_static_at_cy(
        args.kx, args.ky, args.order, args.cy_true,
    )
    step_t = make_step_nu_param(
        V_t, u_bc_t, u_bc_vec_t, bc_mask_t, is_nat_t,
        tau_scale=args.tau_scale, n_newton=args.n_newton,
    )
    step_t_jit = jax.jit(step_t, static_argnames=("order_eff",))
    probe_t = make_probe_fn(V_t, probe_xy)
    u_n = jnp.zeros(V_t.total_dofs); u_nm1 = jnp.zeros(V_t.total_dofs)
    traces = []
    for s in range(args.n_steps):
        order_eff = 1 if s == 0 else 2
        u_new, _ = step_t_jit(u_n, u_nm1, args.dt, NU, order_eff)
        traces.append(probe_t(u_new))
        u_nm1, u_n = u_n, u_new
    y_obs = jnp.stack(traces)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    # ---- Adam on CY via FD -------------------------------------------
    loss_fn = make_loss_for_cy(
        kx=args.kx, ky=args.ky, order=args.order,
        n_newton=args.n_newton, n_steps=args.n_steps,
        dt=args.dt, tau_scale=args.tau_scale, nu=NU,
        probe_xy=probe_xy, y_obs=y_obs,
    )

    cy = float(args.cy_init)
    L0, y_init = loss_fn(cy)
    print(f"  initial loss = {L0:.4e}", flush=True)

    m_t = 0.0; v_t = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_loss = L0; best_cy = cy; best_iter = 0
    history = [dict(iter=0, cy=cy, loss=L0, grad=0.0)]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        L_plus, _ = loss_fn(cy + args.fd_h)
        L_minus, _ = loss_fn(cy - args.fd_h)
        g = (L_plus - L_minus) / (2.0 * args.fd_h)
        m_t = beta1 * m_t + (1 - beta1) * g
        v_t = beta2 * v_t + (1 - beta2) * g * g
        m_hat = m_t / (1 - beta1 ** it)
        v_hat = v_t / (1 - beta2 ** it)
        cy = cy - args.lr * m_hat / (float(np.sqrt(v_hat)) + eps)
        L, _ = loss_fn(cy)
        if L < best_loss:
            best_loss = L; best_cy = cy; best_iter = it
        rel = abs(cy - args.cy_true) / args.cy_true
        print(f"  iter {it:3d}: CY = {cy:.4f}  (rel err {rel*100:5.2f}%)  "
              f"loss={L:.3e}  g={g:+.3e}",
              flush=True)
        history.append(dict(iter=int(it), cy=cy, loss=L, grad=float(g)))
    print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)
    print(f"\nlast-iter CY = {cy:.4f}  (truth {args.cy_true}, "
          f"rel err {100*abs(cy - args.cy_true)/args.cy_true:.2f}%)",
          flush=True)
    print(f"best-loss CY = {best_cy:.4f} at iter {best_iter}",
          flush=True)
    cy_final = best_cy
    L_final, y_final = loss_fn(cy_final)

    stem = f"inverse_obstacle_cy_cylinder_k{args.order}"
    out_npz = args.out_dir / f"{stem}.npz"
    np.savez_compressed(
        out_npz,
        cy_true=args.cy_true, cy_init=args.cy_init,
        cy_final=cy_final, n_steps=args.n_steps, dt=args.dt,
        times=(np.arange(args.n_steps) + 1) * args.dt,
        probe_xy=probe_xy,
        y_obs=np.asarray(y_obs),
        y_final=np.asarray(y_final),
        y_init=np.asarray(y_init),
        cy_hist=np.array([r["cy"] for r in history]),
        loss_hist=np.array([r["loss"] for r in history]),
    )
    print(f"  wrote {out_npz}", flush=True)

    out_json = args.out_dir / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky,
            cy_true=args.cy_true, cy_init=args.cy_init,
            cy_final=cy_final,
            n_steps=args.n_steps, dt=args.dt,
            n_newton=args.n_newton, tau_scale=args.tau_scale,
            fd_h=args.fd_h, lr=args.lr, n_train=args.n_train,
            history=history,
        ), f, indent=2)
    print(f"  wrote {out_json}", flush=True)

    t_axis = (np.arange(args.n_steps) + 1) * args.dt
    iters = [r["iter"] for r in history]
    cys = [r["cy"] for r in history]
    losses = [r["loss"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    axes[0, 0].plot(iters, cys, "b-o", ms=4, label="recovered")
    axes[0, 0].axhline(args.cy_true, color="r", ls="--", lw=1,
                       label="truth")
    axes[0, 0].axhline(args.cy_init, color="gray", ls=":", lw=1,
                       label="init")
    axes[0, 0].set_xlabel("Adam iter"); axes[0, 0].set_ylabel(r"$C_y$")
    axes[0, 0].set_title(r"Recovered cylinder Y position")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].semilogy(iters, losses, "k-")
    axes[0, 1].set_xlabel("Adam iter"); axes[0, 1].set_ylabel("loss")
    axes[0, 1].set_title("L2 mismatch on probe traces")
    axes[0, 1].grid(alpha=0.3, which="both")

    y_obs_np = np.asarray(y_obs)
    p_show = 0
    axes[1, 0].plot(t_axis, y_obs_np[:, p_show, 0], "k-", lw=2,
                    label="truth")
    axes[1, 0].plot(t_axis, np.asarray(y_init)[:, p_show, 0], "r--",
                    lw=1.2, label="init")
    axes[1, 0].plot(t_axis, np.asarray(y_final)[:, p_show, 0], "b:",
                    lw=1.5, label="recovered")
    axes[1, 0].set_xlabel("t"); axes[1, 0].set_ylabel(r"$u_x$")
    axes[1, 0].set_title(
        f"Probe 0 trace ($x = {probe_xy[p_show, 0]:.2f}$)"
    )
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    final_obs = y_obs_np[-1, :, 0]
    final_init = np.asarray(y_init)[-1, :, 0]
    final_recovered = np.asarray(y_final)[-1, :, 0]
    pts = np.arange(n_probe) + 1
    axes[1, 1].plot(pts, final_obs, "k-o", ms=6, label="truth")
    axes[1, 1].plot(pts, final_init, "rx", ms=8, label="init")
    axes[1, 1].plot(pts, final_recovered, "b+", ms=10, label="recovered")
    axes[1, 1].set_xlabel("probe index")
    axes[1, 1].set_ylabel(r"$u_x(t = T_{\rm final})$")
    axes[1, 1].set_title("Final-time probe profile")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Inverse cylinder Y: BDM_{args.order}, "
        f"CY_init = {args.cy_init}, CY_truth = {args.cy_true}, "
        f"final = {cy_final:.4f}"
    )
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
