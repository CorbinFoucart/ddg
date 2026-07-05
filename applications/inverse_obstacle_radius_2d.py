"""Parameter-inverse demo: recover the cylinder radius from sparse
velocity probes on the BDM cylinder transient.

Companion to ``scripts/inverse_nu_cylinder_2d.py`` (recovers viscosity)
and the planned cylinder-position demo. We use the *radius* as the
geometric parameter because cylinder position is constrained by the
grid-alignment of the mesh constructor's bounding square, while the
radius can vary continuously inside the bounding square without any
mesh-topology change.

Setup
-----
* Mesh topology built ONCE at the nominal bounding-square geometry
  (the Cartesian grid + the polygonal "square hole" + the body-fit
  ring connecting them). The geometry of the ring depends only on
  the outer R_outer and the radial-growth parameters --- not on the
  cylinder radius itself.
* For each loss evaluation, we *re-curve* the same base mesh with
  ``apply_circular_hole_2d`` using the current radius. The body-fit
  ring's inner-edge vertices are then projected onto a circle of the
  given radius. Boundary classification is redone with the new radius
  (used to identify the cylinder faces).
* Forward map: 10 BDF2 steps from rest, BDM_2, ν = 10⁻³.
* Loss: L2 mismatch on six probe traces along the wake centreline.
* Gradient: central finite differences in R (host-side mesh rebuild +
  JIT recompile each FD call, same cost structure as the failed CY
  demo --- ~5 min per call, ~15 min per Adam iter).

The truth radius is 0.05 (matches the cylinder-smoke geometry); the
initial guess is 0.04. The optimiser must recover the truth from
probe traces.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

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
from ddg.dg2d.bdm import BDM, bdm_k_evaluate_boundary_velocity
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
)
from ddg.dg2d.curved import apply_circular_hole_2d
from ddg.dg2d.mesh import build_mesh_2d, channel_with_circle_mesh_2d


X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 0.5
CX, CY = 0.2, 0.25
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


def build_static_at_radius(kx, ky, order, radius,
                            *, n_radial=2, r_outer=0.1):
    """Build mesh + curve at the requested radius; classify boundaries
    using the same radius."""
    VX, VY, EToV = channel_with_circle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, kx, ky,
        cx=CX, cy=CY, radius=radius,
        R_outer=r_outer, n_radial=n_radial,
        radial_growth=1.3,
    )
    mesh = build_mesh_2d(VX, VY, EToV, order)
    mesh = apply_circular_hole_2d(
        mesh, cx=CX, cy=CY, radius=radius,
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
    r_from_cyl = np.sqrt((x_face - CX) ** 2 + (y_face - CY) ** 2)
    is_cyl_face = (
        is_bdy & (r_from_cyl < radius * 1.05)
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


def make_loss_for_radius(*, kx, ky, order, n_newton, n_steps, dt,
                          tau_scale, nu, probe_xy, y_obs):
    n_steps_const = int(n_steps)
    def loss_for_radius(radius):
        radius_val = float(radius)
        V, u_bc, u_bc_vec, bc_mask, is_nat = build_static_at_radius(
            kx, ky, order, radius_val,
        )
        step_fn = make_step_nu_param(
            V, u_bc, u_bc_vec, bc_mask, is_nat,
            tau_scale=tau_scale, n_newton=n_newton,
        )
        probe_fn = make_probe_fn(V, probe_xy)
        step_jit = jax.jit(step_fn, static_argnames=("order_eff",))

        u_n = jnp.zeros(V.total_dofs); u_nm1 = jnp.zeros(V.total_dofs)
        traces = []
        for s in range(n_steps_const):
            order_eff = 1 if s == 0 else 2
            u_new, _ = step_jit(u_n, u_nm1, dt, nu, order_eff)
            traces.append(probe_fn(u_new))
            u_nm1, u_n = u_n, u_new
        y_pred = jnp.stack(traces)
        return float(jnp.sum((y_pred - y_obs) ** 2)), np.asarray(y_pred)
    return loss_for_radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--r-true", type=float, default=0.05)
    ap.add_argument("--r-init", type=float, default=0.04)
    ap.add_argument("--fd-h", type=float, default=0.001,
                    help="central-FD half-step in radius")
    ap.add_argument("--lr", type=float, default=0.001,
                    help="Adam lr on R (linear, not log)")
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    print(f"BDM_{args.order} cylinder smoke: n_steps={args.n_steps}, "
          f"dt={args.dt}, T={args.n_steps * args.dt}", flush=True)
    print(f"  truth R = {args.r_true}, init R = {args.r_init}",
          flush=True)
    print(f"  FD h = {args.fd_h}, lr = {args.lr}", flush=True)

    # Probes downstream of the cylinder, along the wake centreline at
    # the cylinder's CY. We use the TRUTH cylinder diameter for probe
    # spacing.
    D_true = 2 * args.r_true
    probe_xy = np.array(
        [[CX + i * D_true, CY] for i in (1, 2, 3, 4, 5, 6)],
        dtype=float,
    )

    # ---- ground truth -----------------------------------------------
    print("generating ground truth at truth R ...", flush=True)
    t0 = time.time()
    V_t, u_bc_t, u_bc_vec_t, bc_mask_t, is_nat_t = build_static_at_radius(
        args.kx, args.ky, args.order, args.r_true,
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

    # ---- Adam on R via central FD -----------------------------------
    loss_fn = make_loss_for_radius(
        kx=args.kx, ky=args.ky, order=args.order,
        n_newton=args.n_newton, n_steps=args.n_steps,
        dt=args.dt, tau_scale=args.tau_scale, nu=NU,
        probe_xy=probe_xy, y_obs=y_obs,
    )

    R = float(args.r_init)
    L0, y_init = loss_fn(R)
    print(f"  initial loss = {L0:.4e}", flush=True)

    m_t = 0.0; v_t = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_loss = L0; best_R = R; best_iter = 0
    history = [dict(iter=0, R=R, loss=L0, grad=0.0)]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        L_plus, _ = loss_fn(R + args.fd_h)
        L_minus, _ = loss_fn(R - args.fd_h)
        g = (L_plus - L_minus) / (2.0 * args.fd_h)
        m_t = beta1 * m_t + (1 - beta1) * g
        v_t = beta2 * v_t + (1 - beta2) * g * g
        m_hat = m_t / (1 - beta1 ** it)
        v_hat = v_t / (1 - beta2 ** it)
        R = R - args.lr * m_hat / (float(np.sqrt(v_hat)) + eps)
        L, _ = loss_fn(R)
        if L < best_loss:
            best_loss = L; best_R = R; best_iter = it
        rel = abs(R - args.r_true) / args.r_true
        print(f"  iter {it:3d}: R = {R:.4f}  (rel err {rel*100:5.2f}%)  "
              f"loss={L:.3e}  g={g:+.3e}",
              flush=True)
        history.append(dict(iter=int(it), R=R, loss=L, grad=float(g)))
    print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)
    print(f"\nlast-iter R = {R:.4f}  (truth {args.r_true}, "
          f"rel err {100*abs(R - args.r_true)/args.r_true:.2f}%)",
          flush=True)
    print(f"best-loss R = {best_R:.4f} at iter {best_iter}",
          flush=True)
    R_final = best_R
    L_final, y_final = loss_fn(R_final)

    stem = f"inverse_obstacle_radius_cylinder_k{args.order}"
    out_npz = args.out_dir / f"{stem}.npz"
    np.savez_compressed(
        out_npz,
        r_true=args.r_true, r_init=args.r_init,
        r_final=R_final, n_steps=args.n_steps, dt=args.dt,
        times=(np.arange(args.n_steps) + 1) * args.dt,
        probe_xy=probe_xy,
        y_obs=np.asarray(y_obs),
        y_final=np.asarray(y_final),
        y_init=np.asarray(y_init),
        r_hist=np.array([r["R"] for r in history]),
        loss_hist=np.array([r["loss"] for r in history]),
    )
    print(f"  wrote {out_npz}", flush=True)
    out_json = args.out_dir / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky,
            r_true=args.r_true, r_init=args.r_init,
            r_final=R_final,
            n_steps=args.n_steps, dt=args.dt,
            n_newton=args.n_newton, tau_scale=args.tau_scale,
            fd_h=args.fd_h, lr=args.lr, n_train=args.n_train,
            history=history,
        ), f, indent=2)
    print(f"  wrote {out_json}", flush=True)

    n_probe = probe_xy.shape[0]
    t_axis = (np.arange(args.n_steps) + 1) * args.dt
    iters = [r["iter"] for r in history]
    Rs = [r["R"] for r in history]
    losses = [r["loss"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    axes[0, 0].plot(iters, Rs, "b-o", ms=4, label="recovered")
    axes[0, 0].axhline(args.r_true, color="r", ls="--", lw=1,
                       label="truth")
    axes[0, 0].axhline(args.r_init, color="gray", ls=":", lw=1,
                       label="init")
    axes[0, 0].set_xlabel("Adam iter"); axes[0, 0].set_ylabel(r"$R$")
    axes[0, 0].set_title("Recovered cylinder radius")
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
        f"Inverse cylinder radius: BDM_{args.order}, "
        f"R_init = {args.r_init}, R_truth = {args.r_true}, "
        f"final = {R_final:.4f}"
    )
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
