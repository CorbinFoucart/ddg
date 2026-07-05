"""Drag-minimising shape optimisation via jax.grad through the
differentiable mesh.

This is the payoff of the Phase 1-4 differentiable-mesh work
(``ddg.dg2d.differentiable_mesh``). The cylinder boundary is
parameterised as

    r(theta) = R0 * (1 + a cos(2 theta))

where ``a`` is a scalar "eccentricity" parameter. ``a > 0`` stretches
the cylinder along the flow direction (a streamlined teardrop-ish
shape that should produce less drag); ``a < 0`` widens it across the
flow (more bluff, higher drag).

We parameterise only this single mode for the demo --- enough to
exercise the gradient path through differentiable_mesh_with_shape +
BDM forward solve + drag surface integral, all in one reverse-mode
pass. The Adam loop minimises a time-averaged ``<C_D>`` over the
last two BDF2 steps of a short transient from rest.

Cost note
---------
Like the diff-mesh inverse-CY demo, the laptop is memory-bound on
the reverse-mode trace. n_steps = 3, n_newton = 1 is the
laptop-comfortable size; that's also short enough that the wake is
still developing, so the gradient signal on the shape parameter is
small but well-defined. A meaningful drag-minimisation result wants
the remote.
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

from inverse_nu_cylinder_2d import make_step_nu_param
from inverse_obstacle_cy_diff_2d import (
    build_diff_setup, NU, CX, RADIUS, D_CYL, U_MEAN, _bc_fn,
)
from ddg.dg2d.bdm import (
    BDM, bdm_k_evaluate_boundary_velocity, refresh_piola_scale,
    bdm_k_basis_at_points, bdm_k_basis_grad_at_points,
    bdm_k_pkm1_basis_at_points, bdm1_gather, _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import bdm1_evaluate_dirichlet_bc
from ddg.dg2d.differentiable_mesh import (
    make_template, make_curving_info,
    differentiable_mesh_with_shape,
)


def _make_drag_eval(V, is_cyl_face_np):
    """Build a JAX-traceable C_D from (u_global, p_per_elem, nu).

    Re-uses the drag-integration recipe from
    ``scripts/diff_cylinder_dt_tau.py``: integrate the symmetric-
    gradient + pressure surface stress sigma . n over every cylinder
    face with Gauss-Legendre quadrature, then divide by
    ``0.5 * U_MEAN^2 * D``.

    The (k, f) pairs for cylinder faces are precomputed host-side
    using ``is_cyl_face_np`` (the host-side BC classification on the
    nominal mesh).
    """
    k = V.order
    n_gl = max(k + 1, 3)
    xi_gl, w_gl = np.polynomial.legendre.leggauss(n_gl)
    xi_gl_j = jnp.asarray(xi_gl); w_gl_j = jnp.asarray(w_gl)

    def face_ref(f, xi):
        ones = jnp.ones_like(xi)
        if f == 0: return xi, -ones
        if f == 1: return -xi, xi
        return -ones, -xi

    phi_face = []; grad_face = []; q_face = []
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

    cyl_pairs = []
    is_cyl_np = np.asarray(is_cyl_face_np)
    for K_idx in range(mesh.K):
        for f_idx in range(3):
            if is_cyl_np[K_idx, f_idx]:
                cyl_pairs.append((K_idx, f_idx))

    denom = 0.5 * U_MEAN**2 * D_CYL

    def compute_cd(u_global, p_per_elem, nu):
        u_local = bdm1_gather(V.topology, u_global)
        F_x = 0.0
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
        return -F_x / denom
    return compute_cd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--cy-nominal", type=float, default=0.25)
    ap.add_argument("--a-init", type=float, default=0.05,
                    help="initial eccentricity (a > 0: streamlined; "
                         "a < 0: bluff). 0 = circle.")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-train", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    print(f"BDM_{args.order} cylinder smoke (shape opt): "
          f"n_steps={args.n_steps}, dt={args.dt}", flush=True)
    print(f"  template (circle) at cy = {args.cy_nominal}", flush=True)
    print(f"  init eccentricity a = {args.a_init}", flush=True)
    print(f"  lr = {args.lr}", flush=True)

    template, curving, cached_topology, bc_mask, is_natural_face = (
        build_diff_setup(
            args.kx, args.ky, args.order, args.cy_nominal,
        )
    )
    # Also need the cylinder-face mask for drag integration. Derived
    # from the nominal mesh classification done inside build_diff_setup;
    # extract by re-running the classification.
    from ddg.dg2d.differentiable_mesh import differentiable_mesh_with_circle
    nominal_mesh = differentiable_mesh_with_circle(
        template.VX_nominal, template.VY_nominal, template, curving,
        jnp.asarray(CX), jnp.asarray(args.cy_nominal),
        jnp.asarray(RADIUS),
    )
    Nfaces = 3
    Fmask = np.asarray(nominal_mesh.Fmask)
    x_np = np.asarray(nominal_mesh.x); y_np = np.asarray(nominal_mesh.y)
    x_face = np.zeros((Nfaces, nominal_mesh.K))
    y_face = np.zeros((Nfaces, nominal_mesh.K))
    for f in range(Nfaces):
        x_face[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE_np = np.asarray(nominal_mesh.EToE)
    is_bdy = (EToE_np == np.arange(nominal_mesh.K)[:, None]).T
    is_inflow_face = is_bdy & (x_face < 1e-3)
    is_outflow_face = is_bdy & (x_face > 1.0 - 1e-3)
    is_outer_y = is_bdy & (
        (y_face < 1e-3) | (y_face > 0.5 - 1e-3)
    )
    r_from_cyl = np.sqrt(
        (x_face - CX) ** 2 + (y_face - args.cy_nominal) ** 2)
    is_cyl_face = (
        is_bdy & (r_from_cyl < RADIUS * 1.05)
        & (~is_inflow_face) & (~is_outflow_face) & (~is_outer_y)
    ).T   # (K, Nfaces)

    def trajectory_with_drag(a):
        coeffs = jnp.array([0.0, 0.0, a, 0.0])    # mode-2 cosine
        mesh_diff = differentiable_mesh_with_shape(
            template.VX_nominal, template.VY_nominal, template, curving,
            jnp.asarray(CX), jnp.asarray(args.cy_nominal),
            jnp.asarray(RADIUS), coeffs,
        )
        topo = refresh_piola_scale(cached_topology, mesh_diff)
        V = BDM(mesh=mesh_diff, topology=topo, order=args.order)
        u_bc = bdm1_evaluate_dirichlet_bc(V, _bc_fn)
        u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _bc_fn)
        step_fn = make_step_nu_param(
            V, u_bc, u_bc_vec, bc_mask, is_natural_face,
            tau_scale=args.tau_scale, n_newton=args.n_newton,
        )
        # Wrap the monolithic-Newton step in jax.checkpoint so the
        # reverse-mode AD tape does NOT retain the per-step
        # vmap(eye)-built dense saddle intermediates -- they get
        # rematerialised on backward instead. Knocks peak memory
        # from ~135 GiB at n_steps=30 down to a few GiB; costs ~2x
        # forward FLOPs on the backward pass. order_eff is a Python
        # int (drives a Python-level branch inside `step`) so it
        # must be static_argnum.
        step_ckpt = jax.checkpoint(step_fn, static_argnums=(4,))
        drag_fn = _make_drag_eval(V, is_cyl_face)

        u_n = jnp.zeros(V.total_dofs); u_nm1 = jnp.zeros(V.total_dofs)
        cd_sum = 0.0; cd_count = 0
        last_p = jnp.zeros(
            (V.K, args.order * (args.order + 1) // 2))
        # Average <C_D> over the developed second half of the trajectory.
        # The flow starts from rest with a full (unramped) inflow, so the
        # early steps are an unsteady added-mass transient whose drag is
        # not the physical developed value (and can even be negative);
        # sampling only the second half t in [T/2, T] gives the developed
        # mean drag the optimiser should minimise. Requires n_steps large
        # enough that T = n_steps*dt reaches the developed regime (T~3).
        cd_start = args.n_steps // 2
        for s in range(args.n_steps):
            order_eff = 1 if s == 0 else 2
            u_new, p_new = step_ckpt(u_n, u_nm1, args.dt, NU, order_eff)
            if s >= cd_start:
                cd_sum = cd_sum + drag_fn(u_new, p_new, NU)
                cd_count = cd_count + 1
            u_nm1, u_n = u_n, u_new
            last_p = p_new
        return cd_sum / cd_count

    grad_fn = jax.jit(jax.grad(trajectory_with_drag))
    loss_jit = jax.jit(trajectory_with_drag)

    a = jnp.asarray(float(args.a_init))
    print("compiling forward + reverse ...", flush=True)
    t0 = time.time()
    L0 = float(loss_jit(a))
    print(f"  initial <C_D> = {L0:.4e}  (compile {time.time() - t0:.1f}s)",
          flush=True)

    m_t = jnp.zeros_like(a); v_t = jnp.zeros_like(a)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history = [dict(iter=0, a=float(a), cd=L0, grad=0.0)]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(a)
        m_t = beta1 * m_t + (1 - beta1) * g
        v_t = beta2 * v_t + (1 - beta2) * g * g
        m_hat = m_t / (1 - beta1 ** it)
        v_hat = v_t / (1 - beta2 ** it)
        a = a - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        L = float(loss_jit(a))
        print(f"  iter {it:3d}: a = {float(a):+.4f}  "
              f"<C_D> = {L:.4e}  g = {float(g):+.4e}",
              flush=True)
        history.append(dict(iter=int(it), a=float(a),
                            cd=L, grad=float(g)))
    print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)

    stem = f"shape_opt_drag_cylinder_k{args.order}"
    out_json = args.out_dir / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky,
            n_steps=args.n_steps, dt=args.dt,
            n_newton=args.n_newton, tau_scale=args.tau_scale,
            a_init=args.a_init, lr=args.lr, n_train=args.n_train,
            history=history,
        ), f, indent=2)
    print(f"  wrote {out_json}", flush=True)

    iters = [r["iter"] for r in history]
    as_ = [r["a"] for r in history]
    cds = [r["cd"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(iters, as_, "b-o", ms=4)
    axes[0].axhline(0, color="k", ls="--", lw=0.7, alpha=0.6,
                    label="circle")
    axes[0].set_xlabel("Adam iter"); axes[0].set_ylabel("shape param $a$")
    axes[0].set_title(
        r"Drag-min shape opt:  $r(\theta) = R_0 (1 + a \cos 2\theta)$"
        + "\n($a > 0$: streamlined; $a < 0$: bluff)"
    )
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(iters, cds, "k-o", ms=4)
    axes[1].set_xlabel("Adam iter")
    axes[1].set_ylabel(r"$\langle C_D \rangle$")
    axes[1].set_title("Mean drag")
    axes[1].grid(alpha=0.3)
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
