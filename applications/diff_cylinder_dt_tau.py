"""Joint differentiable optimization of (Delta t, tau_scale) on the
BDM cylinder smoke (Re=100, Schaefer-Turek-style flow-past-cylinder).

Adapted from ``diff_dt_tau_joint.py`` (cavity stability) but for a
TRANSIENT FLOW-PAST-CYLINDER target. The loss has two pieces:

    L = stability_barrier  -  gamma_dt * log Delta t  +  w_cd * (CD_mean - CD_ref)^2

* The stability barrier penalises peak ||div(u)|| exceeding a
  threshold; gradient through it pushes Delta t toward the cliff.
* The dt-efficiency term rewards larger Delta t.
* The drag-error term ties the optimisation to a physically
  meaningful target -- it's not just "biggest dt that doesn't
  explode" but "biggest dt that still resolves the Schaefer-Turek
  drag plateau". The Schaefer-Turek 2D-2 mean C_D at Re=100 is
  ~3.22.

End-to-end JAX-traceable: fixed-Newton-iter BDF2 loop, no Python
``if`` on tracer state. Forward is jit-compiled once; reverse
mode AD gives the gradient of the loss w.r.t. (log Delta t,
log tau_scale).

Run-time at the smoke geometry (Kx=20, Ky=10, BDM_2, K=432,
n_steps=5, n_newton=2): roughly 30-60 s per (loss, gradient)
pair after the first JIT compile.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "applications"))

from ddg.dg2d.bdm import (
    BDM, bdm_k_convective_apply, bdm_k_evaluate_boundary_velocity,
    bdm_k_viscous_dirichlet_lift, bdm_k_pkm1_basis_at_points,
    bdm_k_basis_at_points, bdm_k_basis_grad_at_points, bdm1_gather,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdf_history_k,
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    bdm_k_mass_apply,
    ns_picard_apply_bdm_k_2d, ns_picard_matrix_bdm_k_2d,
)
from ddg.dg2d.curved import apply_circular_hole_2d
from ddg.dg2d.mesh import build_mesh_2d, channel_with_circle_mesh_2d


# Smoke geometry (matches applications/ns_bdm_flow_past_cylinder_2d.py).
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 0.5
CX, CY = 0.2, 0.25
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


def build_static(kx, ky, order, n_radial=2, r_outer=0.1):
    """Mesh + BC arrays that don't depend on (dt, tau)."""
    VX, VY, EToV = channel_with_circle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, kx, ky,
        cx=CX, cy=CY, radius=RADIUS,
        R_outer=r_outer, n_radial=n_radial,
        radial_growth=1.3,
    )
    mesh = build_mesh_2d(VX, VY, EToV, order)
    mesh = apply_circular_hole_2d(
        mesh, cx=CX, cy=CY, radius=RADIUS,
        blend_interior=True, detection_tol=0.05,
    )
    V = BDM.from_mesh(mesh, order=order)

    # Boundary classification.
    Nfaces = 3
    Fmask = np.asarray(mesh.Fmask)
    x_np = np.asarray(mesh.x); y_np = np.asarray(mesh.y)
    x_face = np.zeros((Nfaces, mesh.K))
    y_face = np.zeros((Nfaces, mesh.K))
    for f in range(Nfaces):
        x_face[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE_np = np.asarray(mesh.EToE)
    is_bdy = (EToE_np == np.arange(mesh.K)[:, None]).T   # (Nfaces, K)
    is_inflow_face = is_bdy & (x_face < X_MIN + 1e-3)
    is_outflow_face = is_bdy & (x_face > X_MAX - 1e-3)
    is_outer_y = is_bdy & (
        (y_face < Y_MIN + 1e-3) | (y_face > Y_MAX - 1e-3)
    )
    r_from_cyl = np.sqrt((x_face - CX) ** 2 + (y_face - CY) ** 2)
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
    return V, u_bc, u_bc_vec, bc_mask, is_natural_face, jnp.asarray(is_cyl_face.T)


def make_step(V, u_bc, u_bc_vec, bc_mask, is_natural_face,
              *, nu, tau_scale_for_lift_only, n_newton):
    """One fully-traceable BDF step.

    Returns ``step(u_n, u_nm1, dt, tau_scale, order_eff)`` that runs
    ``n_newton`` Newton iters with a fresh dense LU per iteration.
    """
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n_full = n_u + n_p

    def step(u_n, u_nm1, dt, tau_scale, order_eff: int):
        coeffs = BDF_COEFFS[order_eff]
        alpha = coeffs["gamma0"] / dt
        f_velocity = (coeffs["beta"][0] / dt) * bdm_k_mass_apply(V, u_n)
        if order_eff == 2:
            f_velocity = f_velocity + (
                coeffs["beta"][1] / dt
            ) * bdm_k_mass_apply(V, u_nm1)

        lift_u = bdm_k_viscous_dirichlet_lift(
            V, u_bc_vec, nu=nu, tau_scale=tau_scale,
            is_natural_face=is_natural_face,
        )

        u_k = u_n
        p_k = jnp.zeros((V.K, dim_p))
        mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])

        for _it in range(n_newton):
            F_u, F_p = ns_picard_apply_bdm_k_2d(
                V, u_k, p_k,
                u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=alpha,
                f_velocity=f_velocity,
                u_bc_global=u_bc,
                is_natural_face=is_natural_face,
            )
            F_u = F_u - lift_u
            F_u_inner = jnp.where(bc_mask, 0.0, F_u)
            F_p_flat = F_p.reshape(-1)
            rhs_full = jnp.concatenate([-F_u_inner, -F_p_flat])

            A_full = ns_picard_matrix_bdm_k_2d(
                V, u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=alpha,
                is_natural_face=is_natural_face,
            )
            # Newton extra column (one full pass; small enough at smoke K).
            def newton_col(i):
                e = jnp.zeros(n_u).at[i].set(1.0)
                return bdm_k_convective_apply(
                    V, u_k, u_star_global=e,
                    is_natural_face=is_natural_face,
                )
            N_ext = jax.vmap(newton_col)(jnp.arange(n_u)).T
            A_full = A_full.at[:n_u, :n_u].add(N_ext)

            I_n = jnp.eye(n_full)
            A_c = jnp.where(
                mask_full[:, None], I_n,
                jnp.where(mask_full[None, :], 0.0, A_full),
            )
            delta = jnp.linalg.solve(A_c, rhs_full)
            delta_u = delta[:n_u]
            delta_p_flat = delta[n_u:]
            delta_u = jnp.where(bc_mask, 0.0, delta_u)
            u_k = u_k + delta_u
            p_k = p_k + delta_p_flat.reshape(V.K, dim_p)
        return u_k, p_k

    return step


def make_drag_fn(V, is_cyl_face):
    """Differentiable surface-integral C_D from (u, p)."""
    k = V.order
    n_gl = max(k + 1, 3)
    xi_gl, w_gl = np.polynomial.legendre.leggauss(n_gl)
    xi_gl_j = jnp.asarray(xi_gl)
    w_gl_j = jnp.asarray(w_gl)

    def face_ref(f, xi):
        ones = jnp.ones_like(xi)
        if f == 0: return xi, -ones
        if f == 1: return -xi, xi
        return -ones, -xi

    # Precompute reference basis on each face.
    phi_face = []
    grad_face = []
    q_face = []
    for f in range(3):
        r_f, s_f = face_ref(f, xi_gl_j)
        phi_face.append(bdm_k_basis_at_points(k, r_f, s_f))
        grad_face.append(bdm_k_basis_grad_at_points(k, r_f, s_f))
        q_face.append(bdm_k_pkm1_basis_at_points(k, r_f, s_f))

    mesh = V.mesh
    DF_all, _ = _element_jacobian_matrices(mesh)
    J_all = mesh.J[0]
    invJ_all = jnp.linalg.inv(DF_all)
    nx_KF = jnp.asarray(mesh.nx).reshape(3, mesh.Nfp, mesh.K).transpose(2, 0, 1)[:, :, 0]
    ny_KF = jnp.asarray(mesh.ny).reshape(3, mesh.Nfp, mesh.K).transpose(2, 0, 1)[:, :, 0]
    sJ_KF = jnp.asarray(mesh.sJ).reshape(3, mesh.Nfp, mesh.K).transpose(2, 0, 1)[:, :, 0]

    # Collect (K, f) cylinder face indices once.
    cyl_pairs = []
    is_cyl_np = np.asarray(is_cyl_face)
    for K_idx in range(mesh.K):
        for f_idx in range(3):
            if is_cyl_np[K_idx, f_idx]:
                cyl_pairs.append((K_idx, f_idx))
    Ks = jnp.asarray([k for k, _ in cyl_pairs])
    Fs = jnp.asarray([f for _, f in cyl_pairs])

    denom = 0.5 * U_MEAN**2 * D_CYL

    def compute_cd(u_global, p_per_elem, nu):
        u_local = bdm1_gather(V.topology, u_global)
        F_x = 0.0
        # Loop over cyl faces. Static unroll (cyl_pairs known at trace).
        for (K_idx, f_idx) in cyl_pairs:
            DF_K = DF_all[K_idx]; invJ_K = invJ_all[K_idx]
            J_K = J_all[K_idx]
            phi_phys = jnp.einsum(
                "cb,qlb->qlc", DF_K, phi_face[f_idx]) / J_K
            grad_phys = (jnp.einsum(
                "cb,qlbd,da->qlca",
                DF_K, grad_face[f_idx], invJ_K) / J_K)
            u_at = jnp.einsum("l,qlc->qc", u_local[K_idx], phi_phys)
            grad_u_at = jnp.einsum(
                "l,qlca->qca", u_local[K_idx], grad_phys)
            p_at = jnp.einsum(
                "m,qm->q", p_per_elem[K_idx], q_face[f_idx])
            sym_grad = 0.5 * (
                grad_u_at + jnp.transpose(grad_u_at, (0, 2, 1)))
            nvec = jnp.stack([nx_KF[K_idx, f_idx], ny_KF[K_idx, f_idx]])
            sig_n = (-p_at[:, None] * nvec[None, :]
                     + 2.0 * nu * jnp.einsum("qca,a->qc", sym_grad, nvec))
            w_face_arr = w_gl_j * sJ_KF[K_idx, f_idx]
            F_x = F_x + jnp.sum(w_face_arr * sig_n[:, 0])
        return -F_x / denom    # negate: face normal points away from fluid
    return compute_cd


def make_loss(
    V, u_bc, u_bc_vec, bc_mask, is_natural_face, is_cyl_face,
    *, nu, n_newton, n_steps, cd_ref, w_cd, w_div_barrier,
):
    step_fn = make_step(
        V, u_bc, u_bc_vec, bc_mask, is_natural_face,
        nu=nu, tau_scale_for_lift_only=None, n_newton=n_newton,
    )
    drag_fn = make_drag_fn(V, is_cyl_face)

    def trajectory(dt, tau_scale):
        u_n = u_bc; u_nm1 = u_bc
        p_last = jnp.zeros((V.K, V.order * (V.order + 1) // 2))
        peak_div = jnp.asarray(0.0)
        cd_sum = jnp.asarray(0.0)
        cd_count = 0
        for s in range(n_steps):
            order_eff = 1 if s == 0 else 2
            u_new, p_new = step_fn(u_n, u_nm1, dt, tau_scale, order_eff)
            from ddg.dg2d.bdm import bdm_k_divergence_apply
            div = bdm_k_divergence_apply(V, u_new)
            peak_div = jnp.maximum(peak_div, jnp.max(jnp.abs(div)))
            # Drag only on the last 3 steps (skip BDF1 + transient).
            if s >= max(0, n_steps - 3):
                cd_sum = cd_sum + drag_fn(u_new, p_new, nu)
                cd_count += 1
            u_nm1 = u_n; u_n = u_new; p_last = p_new
        cd_mean = cd_sum / cd_count
        return peak_div, cd_mean

    div_target = 1.0e-3   # tolerable peak divergence
    beta = 4.0
    gamma_dt = 0.1

    def loss(params):
        dt = jnp.exp(params["log_dt"])
        tau = jnp.exp(params["log_tau"])
        peak_div, cd_mean = trajectory(dt, tau)
        z = beta * (jnp.log(peak_div + 1e-30) - jnp.log(div_target))
        barrier = jax.nn.softplus(z) / beta
        return (w_div_barrier * barrier
                - gamma_dt * params["log_dt"]
                + w_cd * (cd_mean - cd_ref) ** 2)

    return loss, trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=5)
    ap.add_argument("--log-dt-init", type=float, default=np.log(0.02))
    ap.add_argument("--log-tau-init", type=float, default=np.log(4.0))
    ap.add_argument("--cd-ref", type=float, default=3.22,
                    help="Schaefer-Turek 2D-2 mean C_D reference.")
    ap.add_argument("--w-cd", type=float, default=0.02)
    ap.add_argument("--w-div", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-train", type=int, default=15)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    nu = NU
    V, u_bc, u_bc_vec, bc_mask, is_natural_face, is_cyl_face = build_static(
        args.kx, args.ky, args.order,
    )
    n_u = V.total_dofs
    dim_p = args.order * (args.order + 1) // 2
    n_p = V.K * dim_p
    Re = U_MEAN * D_CYL / nu
    print(f"BDM_{args.order} cylinder smoke: Re={Re:.0f}, K={V.K}, "
          f"n_u={n_u}, n_p={n_p}, n_steps={args.n_steps}, "
          f"n_newton={args.n_newton}", flush=True)

    loss_fn, traj_fn = make_loss(
        V, u_bc, u_bc_vec, bc_mask, is_natural_face, is_cyl_face,
        nu=nu, n_newton=args.n_newton, n_steps=args.n_steps,
        cd_ref=args.cd_ref, w_cd=args.w_cd, w_div_barrier=args.w_div,
    )
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    params = {
        "log_dt": jnp.asarray(float(args.log_dt_init)),
        "log_tau": jnp.asarray(float(args.log_tau_init)),
    }
    print(f"  init: dt={float(jnp.exp(params['log_dt'])):.4f}, "
          f"tau={float(jnp.exp(params['log_tau'])):.3f}", flush=True)
    t0 = time.time()
    L0 = float(loss_jit(params))
    print(f"  loss={L0:.4f}  (jit compile: {time.time()-t0:.1f}s)",
          flush=True)

    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v_st = {k: jnp.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    g_clip = 1.0

    history = [dict(iter=0,
                    dt=float(jnp.exp(params["log_dt"])),
                    tau=float(jnp.exp(params["log_tau"])),
                    loss=L0, g_dt=0.0, g_tau=0.0)]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(params)
        g_clipped = {k: jnp.clip(v, -g_clip, g_clip) for k, v in g.items()}
        for k in params:
            m[k] = beta1 * m[k] + (1 - beta1) * g_clipped[k]
            v_st[k] = beta2 * v_st[k] + (1 - beta2) * g_clipped[k] ** 2
            m_hat = m[k] / (1 - beta1 ** it)
            v_hat = v_st[k] / (1 - beta2 ** it)
            params[k] = params[k] - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        L = float(loss_jit(params))
        history.append(dict(
            iter=int(it),
            dt=float(jnp.exp(params["log_dt"])),
            tau=float(jnp.exp(params["log_tau"])),
            loss=L,
            g_dt=float(g["log_dt"]),
            g_tau=float(g["log_tau"]),
        ))
        print(f"  iter {it:3d}: dt={float(jnp.exp(params['log_dt'])):5.4f}  "
              f"tau={float(jnp.exp(params['log_tau'])):6.3f}  "
              f"loss={L:+.4f}  g_dt={float(g['log_dt']):+.2e}  "
              f"g_tau={float(g['log_tau']):+.2e}",
              flush=True)
    print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)

    # Plot
    iters = [r["iter"] for r in history]
    dts = [r["dt"] for r in history]
    taus = [r["tau"] for r in history]
    losses = [r["loss"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    axes[0].plot(dts, taus, "o-", ms=3, alpha=0.7)
    axes[0].plot([dts[0]], [taus[0]], "go", ms=10, label="init")
    axes[0].plot([dts[-1]], [taus[-1]], "rs", ms=10, label="final")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\Delta t$"); axes[0].set_ylabel(r"$\tau_{\rm scale}$")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(
        f"Joint Adam trajectory (BDM_{args.order} cylinder, "
        f"Re={Re:.0f})"
    )
    axes[1].plot(iters, dts, "b-", label=r"$\Delta t$")
    axes[1].plot(iters, taus, "r-", label=r"$\tau$")
    axes[1].set_xlabel("Adam iter"); axes[1].set_yscale("log")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(iters, losses, "k-")
    axes[2].set_xlabel("Adam iter"); axes[2].set_ylabel("loss")
    axes[2].grid(alpha=0.3)

    stem = f"diff_cylinder_dt_tau_k{args.order}"
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}", flush=True)

    out_json = args.out_dir / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky, K=int(V.K),
            re=Re, nu=nu, cd_ref=args.cd_ref,
            w_cd=args.w_cd, w_div=args.w_div, lr=args.lr,
            n_steps=args.n_steps, n_newton=args.n_newton,
            history=history,
        ), f, indent=2)
    print(f"  wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
