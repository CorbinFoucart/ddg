"""Sensitivity analysis on the BDM cylinder smoke via jax.grad.

This is the cheapest of the differentiable-INS demos: one forward and
one reverse pass through the same fixed-shape 10-step BDF2 trajectory
used by the other diff scripts, returning all first-order
sensitivities of physical observables to numerical / physical
parameters in a single call.

What we compute
---------------
For a trajectory parameterised by

  params = (log dt, log tau_scale, log nu),

with the trajectory built exactly as in
``scripts/diff_cylinder_dt_tau.py``, we compute the Jacobian of three
observables:

  * mean drag      <C_D> over the last few BDF steps
  * peak divergence ||div u||_inf over the trajectory
  * final kinetic energy KE(T) = 0.5 sum_i M_ii u_i^2

with respect to all three parameters.  That's a 3x3 sensitivity
matrix, obtained from three reverse-mode passes (one per scalar
output) sharing a single forward.

Validation
----------
For each entry we compute the central finite-difference approximation
at h = 0.01 in log-space and report the relative discrepancy.  AD vs
FD agreement at the 1-3% level on smooth observables is the
"sensitivities are believable" result.

Cost
----
Cheap: ~10 minutes total on a single CPU thread (one JIT compile +
three reverse passes + six FD forward calls).
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
sys.path.insert(0, str(ROOT / "scripts"))

from diff_cylinder_dt_tau import (
    build_static, _bc_fn,
    X_MIN, X_MAX, Y_MIN, Y_MAX, CX, CY, RADIUS, D_CYL,
    U_MAX, U_MEAN, NU,
    make_step, make_drag_fn,
)
from ddg.dg2d.bdm import bdm_k_divergence_apply
from ddg.dg2d.coupled_ns_bdm import bdm_k_mass_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--nu", type=float, default=NU)
    ap.add_argument("--fd-h", type=float, default=0.01,
                    help="log-space finite-difference half-step")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    V, u_bc, u_bc_vec, bc_mask, is_natural_face, is_cyl_face = build_static(
        args.kx, args.ky, args.order,
    )
    Re = U_MEAN * D_CYL / args.nu
    print(f"BDM_{args.order} cylinder smoke: K={V.K}, "
          f"n_steps={args.n_steps}, dt={args.dt}, "
          f"tau_scale={args.tau_scale}, nu={args.nu:.3e} "
          f"(Re={Re:.0f})", flush=True)

    # diff_cylinder_dt_tau.make_step closes over nu, so we need a
    # nu-parameterised variant. We reuse the same primitives but
    # rebuild the closure here so all three parameters thread through.
    from ddg.dg2d.bdm import (
        bdm_k_convective_apply,
        bdm_k_viscous_dirichlet_lift,
    )
    from ddg.dg2d.coupled_ns_bdm import (
        BDF_COEFFS,
        ns_picard_apply_bdm_k_2d, ns_picard_matrix_bdm_k_2d,
    )

    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n_full = n_u + n_p
    n_newton = args.n_newton

    def step(u_n, u_nm1, dt, tau_scale, nu, order_eff: int):
        coeffs = BDF_COEFFS[order_eff]
        alpha = coeffs["gamma0"] / dt
        f_vel = (coeffs["beta"][0] / dt) * bdm_k_mass_apply(V, u_n)
        if order_eff == 2:
            f_vel = f_vel + (coeffs["beta"][1] / dt) * bdm_k_mass_apply(V, u_nm1)
        lift_u = bdm_k_viscous_dirichlet_lift(
            V, u_bc_vec, nu=nu, tau_scale=tau_scale,
            is_natural_face=is_natural_face,
        )
        u_k = u_n
        p_k = jnp.zeros((V.K, dim_p))
        mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])
        for _ in range(n_newton):
            F_u, F_p = ns_picard_apply_bdm_k_2d(
                V, u_k, p_k, u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=alpha, f_velocity=f_vel,
                u_bc_global=u_bc, is_natural_face=is_natural_face,
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

    drag_fn = make_drag_fn(V, is_cyl_face)

    def trajectory_outputs(log_dt, log_tau, log_nu):
        """Returns (mean_drag, peak_div, final_KE) for the trajectory."""
        dt = jnp.exp(log_dt)
        tau = jnp.exp(log_tau)
        nu = jnp.exp(log_nu)
        u_n = jnp.zeros(n_u)
        u_nm1 = jnp.zeros(n_u)
        peak_div = jnp.asarray(0.0)
        cd_sum = jnp.asarray(0.0)
        cd_count = 0
        for s in range(args.n_steps):
            order_eff = 1 if s == 0 else 2
            u_new, p_new = step(u_n, u_nm1, dt, tau, nu, order_eff)
            div = bdm_k_divergence_apply(V, u_new)
            peak_div = jnp.maximum(peak_div, jnp.max(jnp.abs(div)))
            if s >= max(0, args.n_steps - 3):
                cd_sum = cd_sum + drag_fn(u_new, p_new, nu)
                cd_count += 1
            u_nm1, u_n = u_n, u_new
        mean_cd = cd_sum / cd_count
        # Final KE = 0.5 * u^T M u
        m_u = bdm_k_mass_apply(V, u_n)
        ke = 0.5 * jnp.dot(u_n, m_u)
        return mean_cd, peak_div, ke

    print("compiling forward + reverse passes ...", flush=True)
    t0 = time.time()
    f_jit = jax.jit(trajectory_outputs)
    # Each output -> three grads. Use jax.grad with argnums for each output.
    log_dt = float(np.log(args.dt))
    log_tau = float(np.log(args.tau_scale))
    log_nu = float(np.log(args.nu))
    p0 = (jnp.asarray(log_dt), jnp.asarray(log_tau), jnp.asarray(log_nu))

    cd0, div0, ke0 = f_jit(*p0)
    print(f"  baseline outputs:", flush=True)
    print(f"    <C_D>     = {float(cd0):+.4e}", flush=True)
    print(f"    ||div||   = {float(div0):+.4e}", flush=True)
    print(f"    KE_final  = {float(ke0):+.4e}", flush=True)
    print(f"  forward compile + run: {time.time() - t0:.1f}s",
          flush=True)

    grad_cd = jax.jit(jax.grad(lambda *p: trajectory_outputs(*p)[0],
                               argnums=(0, 1, 2)))
    grad_div = jax.jit(jax.grad(lambda *p: trajectory_outputs(*p)[1],
                                argnums=(0, 1, 2)))
    grad_ke = jax.jit(jax.grad(lambda *p: trajectory_outputs(*p)[2],
                               argnums=(0, 1, 2)))

    t0 = time.time()
    g_cd = grad_cd(*p0)
    print(f"  reverse <C_D>:      {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    g_div = grad_div(*p0)
    print(f"  reverse ||div||:    {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    g_ke = grad_ke(*p0)
    print(f"  reverse KE_final:   {time.time() - t0:.1f}s", flush=True)

    # FD validation in log-space.
    print("\nFD validation (h = {:.0e}, log-space) ...".format(args.fd_h),
          flush=True)
    h = args.fd_h
    fd = np.zeros((3, 3))     # rows: outputs, cols: params
    for j, name in enumerate(("log_dt", "log_tau", "log_nu")):
        plus = list(p0); plus[j] = p0[j] + h
        minus = list(p0); minus[j] = p0[j] - h
        cd_p, div_p, ke_p = f_jit(*plus)
        cd_m, div_m, ke_m = f_jit(*minus)
        fd[0, j] = float((cd_p - cd_m) / (2 * h))
        fd[1, j] = float((div_p - div_m) / (2 * h))
        fd[2, j] = float((ke_p - ke_m) / (2 * h))

    ad = np.zeros((3, 3))
    ad[0, :] = [float(g_cd[0]), float(g_cd[1]), float(g_cd[2])]
    ad[1, :] = [float(g_div[0]), float(g_div[1]), float(g_div[2])]
    ad[2, :] = [float(g_ke[0]), float(g_ke[1]), float(g_ke[2])]

    print("\nSensitivity matrix (rows: outputs; cols: parameters):",
          flush=True)
    headers = ["d/dlog(dt)", "d/dlog(tau)", "d/dlog(nu)"]
    output_names = ["<C_D>", "||div||", "KE_final"]
    print(f"{'output':>12}  {headers[0]:>14}  {headers[1]:>14}  "
          f"{headers[2]:>14}", flush=True)
    for i, name in enumerate(output_names):
        print(f"{name:>12}", end="", flush=True)
        for j in range(3):
            a = ad[i, j]; f = fd[i, j]
            if abs(a) > 0 or abs(f) > 0:
                rel = abs(a - f) / max(abs(a), abs(f), 1e-30) * 100
            else:
                rel = 0.0
            print(f"  AD {a:+.3e} (FD {f:+.3e}, {rel:4.1f}%)",
                  end="", flush=True)
        print("", flush=True)

    out_json = args.out_dir / "sensitivity_cylinder_k{}.json".format(args.order)
    with open(out_json, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky, K=int(V.K),
            n_steps=args.n_steps, dt=args.dt,
            tau_scale=args.tau_scale, nu=args.nu, re=Re,
            outputs=dict(
                mean_cd=float(cd0),
                peak_div=float(div0),
                ke_final=float(ke0),
            ),
            ad_matrix=ad.tolist(),
            fd_matrix=fd.tolist(),
            param_labels=["log_dt", "log_tau", "log_nu"],
            output_labels=output_names,
        ), f, indent=2)
    print(f"\nwrote {out_json}", flush=True)

    # Plot: side-by-side bar chart of AD vs FD for each (output, param)
    # combo, with relative-error percentage annotated.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    x = np.arange(3); width = 0.35
    for i, ax in enumerate(axes):
        ax.bar(x - width / 2, ad[i], width, label="AD", color="C0",
               edgecolor="k")
        ax.bar(x + width / 2, fd[i], width, label="FD", color="C1",
               edgecolor="k")
        ax.set_xticks(x)
        ax.set_xticklabels(["log(dt)", "log(tau)", "log(nu)"])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"$\\partial${output_names[i]}/$\\partial$param")
        ax.set_ylabel("derivative")
        ax.grid(alpha=0.3, axis="y")
        for j in range(3):
            a = ad[i, j]; f = fd[i, j]
            if abs(a) > 0 or abs(f) > 0:
                rel = abs(a - f) / max(abs(a), abs(f), 1e-30) * 100
                ax.annotate(
                    f"{rel:.1f}%", xy=(j, max(a, f, 0) * 1.05),
                    ha="center", va="bottom", fontsize=8, color="gray",
                )
        ax.legend()
    fig.suptitle(
        f"Cylinder sensitivity (BDM_{args.order}, Re={Re:.0f}, "
        f"AD vs central FD)"
    )
    out_png = args.out_dir / f"sensitivity_cylinder_k{args.order}.png"
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
