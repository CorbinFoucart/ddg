"""BDM_1 / P_0 temporal convergence on time-dependent Taylor-Green.

Manufactured analytical NS solution (no body force needed):
    u(x, y, t) = (sin(pi x) cos(pi y), -cos(pi x) sin(pi y)) * exp(-2 pi^2 nu t)
    p(x, y, t) = (1/4) exp(-4 pi^2 nu t) (cos(2 pi x) + cos(2 pi y))

Verification: at fixed spatial mesh, sweep dt with BDF1 and BDF2, both
using Newton inner solve. The L2 velocity error at the final time
should reduce at slope = bdf_order (asymptotically, once dt is small
enough that temporal error dominates spatial discretisation error).

Outputs:
    output/bdm_taylor_green_temporal_convergence.png
    output/bdm_taylor_green_temporal_convergence.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM, bdm1_basis_at_points, bdm1_evaluate_boundary_velocity,
    bdm1_gather, _element_jacobian_matrices,
    bdm_k_face_dof_coordinates, _bdm_k_face_quadrature_data,
)
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdf_history,
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    ns_newton_solve_bdm_2d,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


U_DOMAIN = (0.0, 1.0)
V_DOMAIN = (0.0, 1.0)
NU_DEFAULT = 0.01


def make_tg_velocity(nu):
    """Return ``u_tg(x, y, t) -> (u_x, u_y)`` for the given nu."""
    def fn(x, y, t):
        decay = jnp.exp(-2.0 * jnp.pi ** 2 * nu * t)
        ux = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * decay
        uy = -jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) * decay
        return ux, uy
    return fn


def make_tg_bc_at_time(nu, t):
    """Boundary-velocity callable u_BC(x, y) at fixed time t."""
    u_tg = make_tg_velocity(nu)
    def fn(x, y):
        return u_tg(x, y, t)
    return fn


def build_mesh(n):
    VX, VY, EToV = uniform_rectangle_mesh_2d(
        U_DOMAIN[0], U_DOMAIN[1], V_DOMAIN[0], V_DOMAIN[1], n, n,
    )
    return build_mesh_2d(VX, VY, EToV, 1)


def bdm1_full_interpolant(V, u_fn):
    """BDM_1 canonical interpolant on ALL face DOFs (interior +
    boundary). Each global DOF = u_fn . n_canonical at the GL point.

    For TG which has u . n = 0 on the domain boundary,
    bdm1_evaluate_dirichlet_bc returns all zeros (which would
    correspond to u = 0, not u_TG). The full interpolant has
    non-zero normal traces on INTERIOR faces and captures u_TG
    properly.
    """
    K = V.K
    Q = 2
    fdata = _bdm_k_face_quadrature_data(V)
    n_face = fdata['n_face']
    coords = bdm_k_face_dof_coordinates(1)
    r = np.asarray(coords[:, 0]); s = np.asarray(coords[:, 1])
    EToV = np.asarray(V.mesh.EToV); VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    x_at = Vx0[:, None] + 0.5 * (r + 1.0)[None, :] * (Vx1 - Vx0)[:, None] \
           + 0.5 * (s + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
    y_at = Vy0[:, None] + 0.5 * (r + 1.0)[None, :] * (Vy1 - Vy0)[:, None] \
           + 0.5 * (s + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
    x_face = jnp.asarray(x_at).reshape(K, 3, Q)
    y_face = jnp.asarray(y_at).reshape(K, 3, Q)
    ux, uy = u_fn(x_face, y_face)
    un = ux * n_face[:, :, 0:1] + uy * n_face[:, :, 1:2]
    u_local = un.reshape(K, 6)
    # Owner-only scatter via .add() (avoids the .set() duplicate-index
    # undefined behaviour on shared interior face DOFs).
    sign = V.topology.local_sign
    owner_mask = sign > 0
    lg = V.topology.local_to_global
    u_global = jnp.zeros(V.total_dofs)
    u_global = u_global.at[lg.reshape(-1)].add(
        jnp.where(owner_mask, u_local, 0).reshape(-1)
    )
    return u_global


def l2_error_at_time(V, u_global, nu, t):
    """``||u_h - u_tg(., t)||_{L^2(Omega)}``."""
    r_q, s_q, w_q = triangle_cubature(6)
    phi_ref = bdm1_basis_at_points(r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum('kca,qla->kqlc', DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    uh = jnp.einsum('kl,kqlc->kqc', u_local, phi_phys)
    EToV = np.asarray(V.mesh.EToV); VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    rqn = np.asarray(r_q); sqn = np.asarray(s_q)
    x_q = Vx0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None] \
          + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
    y_q = Vy0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None] \
          + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
    u_tg = make_tg_velocity(nu)
    ux, uy = u_tg(jnp.asarray(x_q), jnp.asarray(y_q), t)
    dx = uh[..., 0] - ux; dy = uh[..., 1] - uy
    return float(jnp.sqrt(jnp.einsum('q,k,kq->', w_q, J, dx ** 2 + dy ** 2)))


def time_march(V, nu, dt, t_final, bdf_order, seed_analytic_history=True):
    """March from t=0 using BDFk. When bdf_order >= 2 and
    seed_analytic_history is set, ``u_{n-1}`` is seeded from the
    analytic u_TG(-dt) so BDF2 starts at step 0 without a BDF1
    bootstrap. The bootstrap contributes O(dt^2) local error that
    confounds the BDF2 rate measurement at large dt; seeding from
    the closed-form solution eliminates that contamination."""
    n_steps = int(round(t_final / dt))
    actual_T = n_steps * dt

    # IC: FULL BDM_1 canonical interpolant of u_TG at t=0.
    u_bc_fn0 = make_tg_bc_at_time(nu, 0.0)
    bc_mask = bdm1_boundary_dofs_mask(V)
    u_n = bdm1_full_interpolant(V, u_bc_fn0)
    if bdf_order >= 2 and seed_analytic_history:
        u_nm1 = bdm1_full_interpolant(V, make_tg_bc_at_time(nu, -dt))
    else:
        u_nm1 = None
    p_n = jnp.zeros(V.K)

    for step in range(n_steps):
        t_new = (step + 1) * dt
        # Use BDF1 only when u_{n-1} not available (no seeding + step 0).
        if bdf_order > 1 and step == 0 and u_nm1 is None:
            order_eff = 1
        else:
            order_eff = bdf_order
        gamma0 = BDF_COEFFS[order_eff]["gamma0"]
        alpha = gamma0 / dt
        u_hist = [u_n] if order_eff == 1 else [u_n, u_nm1]
        f_history = bdf_history(V, u_hist, dt=dt, order=order_eff)

        # Time-dependent Dirichlet BC at new time.
        u_bc_fn = make_tg_bc_at_time(nu, t_new)
        u_bc = bdm1_evaluate_dirichlet_bc(V, u_bc_fn)
        u_bc_vec = bdm1_evaluate_boundary_velocity(V, u_bc_fn)

        # Newton solve at this step.
        u_new, p_new, _ = ns_newton_solve_bdm_2d(
            V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
            u_init=u_n, p_init=p_n,
            velocity_alpha=alpha,
            f_velocity=f_history,
            pressure_eps=1e-10, tau_scale=4.0,
            max_iters=20, tol=1e-10, atol=1e-12,
            linearisation="newton",
        )
        u_nm1, u_n = u_n, u_new
        p_n = p_new
    return u_n, actual_T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16,
                     help="Mesh resolution Kx = Ky")
    ap.add_argument("--nu", type=float, default=NU_DEFAULT)
    ap.add_argument("--tfinal", type=float, default=0.1)
    ap.add_argument("--dts", type=str, default="0.05,0.025,0.0125,0.00625")
    ap.add_argument("--bdf-orders", type=str, default="1,2")
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    dts = [float(s) for s in args.dts.split(",")]
    bdf_orders = [int(s) for s in args.bdf_orders.split(",")]

    mesh = build_mesh(args.n)
    V = BDM.from_mesh(mesh, order=1)
    print(f"BDM_1, n={args.n}, total_dofs={V.total_dofs}", flush=True)
    print(f"nu={args.nu}, T_final={args.tfinal}", flush=True)

    summary = {
        "n": args.n, "nu": args.nu, "T_final": args.tfinal,
        "results": {},
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {1: "C0", 2: "C1"}

    for bdf in bdf_orders:
        print(f"\n=== BDF{bdf} ===", flush=True)
        errs, ts = [], []
        for dt in dts:
            t0 = time.time()
            u_sol, actual_T = time_march(V, args.nu, dt, args.tfinal, bdf)
            wall = time.time() - t0
            err = l2_error_at_time(V, u_sol, args.nu, actual_T)
            n_steps = int(round(args.tfinal / dt))
            print(f"  dt={dt:.5f}, n_steps={n_steps}, T={actual_T:.4f}, "
                  f"L2 err = {err:.3e}, {wall:.1f}s", flush=True)
            errs.append(err); ts.append(wall)
        errs = np.array(errs); dt_arr = np.array(dts)
        slopes = (np.log(errs[:-1]) - np.log(errs[1:])) / \
                 (np.log(dt_arr[:-1]) - np.log(dt_arr[1:]))
        for i, sl in enumerate(slopes):
            print(f"  slope({dt_arr[i]:.4f} -> {dt_arr[i+1]:.4f}) = {sl:.3f}",
                  flush=True)
        ax.loglog(dt_arr, errs, "o-", color=colors.get(bdf, "k"),
                   label=f"BDF{bdf} (asymp slope {slopes[-1]:.2f}, theory {bdf})")
        ref = errs[0] * (dt_arr / dt_arr[0]) ** bdf
        ax.loglog(dt_arr, ref, "--", color=colors.get(bdf, "k"), alpha=0.4,
                   label=f"slope {bdf} reference")
        summary["results"][f"BDF{bdf}"] = {
            "dts": dts, "errs": errs.tolist(),
            "slopes": slopes.tolist(),
            "asymptotic_slope": float(slopes[-1]),
            "expected_slope": bdf,
            "wall_seconds": ts,
        }

    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\|u_h(T) - u_{\rm TG}(T)\|_{L^2}$")
    ax.set_title(f"Taylor-Green temporal convergence "
                  f"(BDM_1, n={args.n}, nu={args.nu}, T={args.tfinal})")
    ax.grid(which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    png_path = args.out_dir / "bdm_taylor_green_temporal_convergence.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png_path}", flush=True)
    json_path = args.out_dir / "bdm_taylor_green_temporal_convergence.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
