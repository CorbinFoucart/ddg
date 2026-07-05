"""BDM_k / P_{k-1} Stokes spatial convergence (linear problem).

Manufactured Stokes solution on [0, 1]^2:
    u(x, y) = (sin(pi x) cos(pi y), -cos(pi x) sin(pi y))   (div-free)
    p(x, y) = 0
    nu      = 1
    Delta u = -2 pi^2 u  =>  body force f = -nu Delta u = 2 pi^2 u.

Solves the linear Stokes saddle point directly (dense LU, no Newton),
sweeps the mesh, and plots the L2 velocity error. Expected rate is
h^(k+1): slope 2 / 3 / 4 at k = 1 / 2 / 3. This is the LINEAR-operator
counterpart of the nonlinear Kovasznay spatial study.

Output:
    output/bdm_stokes_spatial_convergence.png
    output/bdm_stokes_spatial_convergence.json
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
    BDM, bdm_k_basis_at_points, bdm_k_evaluate_boundary_velocity,
    bdm_k_viscous_dirichlet_lift, bdm1_scatter_add,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc, bdm1_gather,
    stokes_matrix_bdm_k_2d,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


PI = jnp.pi
NU = 1.0


def u_anal(x, y):
    return (jnp.sin(PI * x) * jnp.cos(PI * y),
            -jnp.cos(PI * x) * jnp.sin(PI * y))


def build_load(V):
    """``f_u[ell] = integral_K f . phi_phys_ell dx`` with f = 2 pi^2 nu u."""
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 2)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum("kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]
    EToV = np.asarray(V.mesh.EToV); VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    rqn = np.asarray(r_q); sqn = np.asarray(s_q)
    x_q = Vx0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None] \
          + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
    y_q = Vy0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None] \
          + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
    ux, uy = u_anal(jnp.asarray(x_q), jnp.asarray(y_q))
    fx = 2 * PI ** 2 * NU * ux
    fy = 2 * PI ** 2 * NU * uy
    f_local = (
        jnp.einsum("q,k,kq,kql->kl", w_q, J, fx, phi_phys[..., 0])
        + jnp.einsum("q,k,kq,kql->kl", w_q, J, fy, phi_phys[..., 1])
    )
    return bdm1_scatter_add(V.topology, f_local)


def l2_velocity_error(V, u_global):
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 4)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum("kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    uh = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)
    EToV = np.asarray(V.mesh.EToV); VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    rqn = np.asarray(r_q); sqn = np.asarray(s_q)
    x_q = Vx0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None] \
          + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
    y_q = Vy0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None] \
          + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
    ux, uy = u_anal(jnp.asarray(x_q), jnp.asarray(y_q))
    dx = uh[..., 0] - ux; dy = uh[..., 1] - uy
    return float(jnp.sqrt(jnp.einsum("q,k,kq->", w_q, J, dx ** 2 + dy ** 2)))


def solve_stokes(V):
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    u_bc = bdm1_evaluate_dirichlet_bc(V, u_anal)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, u_anal)
    bc_mask = bdm1_boundary_dofs_mask(V)
    lift = bdm_k_viscous_dirichlet_lift(V, u_bc_vec, nu=NU, tau_scale=4.0)
    f_load = build_load(V)
    A = stokes_matrix_bdm_k_2d(V, nu=NU, pressure_eps=1e-10, tau_scale=4.0)
    rhs = jnp.concatenate([f_load + lift, jnp.zeros(n_p)])
    sol = jnp.linalg.solve(A, rhs)
    u_sol = sol[:n_u]
    u_sol = jnp.where(bc_mask, u_bc, u_sol)
    return u_sol


def build_mesh(n):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, n, n)
    return build_mesh_2d(VX, VY, EToV, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=str, default="1,2,3")
    ap.add_argument("--meshes-k1", type=str, default="4,8,12,16,24")
    ap.add_argument("--meshes-k2", type=str, default="3,4,6,8,12")
    ap.add_argument("--meshes-k3", type=str, default="3,4,6,8")
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    orders = [int(s) for s in args.orders.split(",")]
    meshes = {
        1: [int(s) for s in args.meshes_k1.split(",")],
        2: [int(s) for s in args.meshes_k2.split(",")],
        3: [int(s) for s in args.meshes_k3.split(",")],
    }

    summary = {"problem": "manufactured Stokes (Taylor-Green spatial)",
               "nu": NU, "results": {}}
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {1: "C0", 2: "C1", 3: "C2"}
    for k in orders:
        ns = meshes.get(k, [3, 4, 6])
        print(f"\n=== Stokes spatial, k = {k} ===", flush=True)
        hs, errs = [], []
        for n in ns:
            mesh = build_mesh(n)
            V = BDM.from_mesh(mesh, order=k)
            t0 = time.time()
            u_sol = solve_stokes(V)
            err = l2_velocity_error(V, u_sol)
            h = 1.0 / n
            hs.append(h); errs.append(err)
            print(f"  n={n}: h={h:.4f}, L2 err = {err:.3e}, {time.time()-t0:.1f}s",
                  flush=True)
        hs = np.array(hs); errs = np.array(errs)
        slopes = (np.log(errs[:-1]) - np.log(errs[1:])) / \
                 (np.log(hs[:-1]) - np.log(hs[1:]))
        for i, sl in enumerate(slopes):
            print(f"  slope({hs[i]:.4f} -> {hs[i+1]:.4f}) = {sl:.3f}", flush=True)
        summary["results"][f"k={k}"] = {
            "hs": hs.tolist(), "errs": errs.tolist(),
            "slopes": slopes.tolist(),
            "asymptotic_slope": float(slopes[-1]),
            "expected_slope": k + 1,
        }
        ax.loglog(hs, errs, "o-", color=colors.get(k, "k"),
                   label=f"BDM_{k} (asymp slope {slopes[-1]:.2f}, theory {k+1})")
        ref = errs[0] * (hs / hs[0]) ** (k + 1)
        ax.loglog(hs, ref, "--", color=colors.get(k, "k"), alpha=0.4,
                   label=f"slope {k+1} reference")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\|u_h - u_{\rm exact}\|_{L^2}$")
    ax.set_title("BDM_k / P_{k-1} Stokes spatial convergence (linear, dense LU)")
    ax.grid(which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    png_path = args.out_dir / "bdm_stokes_spatial_convergence.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png_path}", flush=True)
    json_path = args.out_dir / "bdm_stokes_spatial_convergence.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
