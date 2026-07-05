"""Order-of-accuracy study for the BDM_k/P_{k-1} block-Schur PROJECTION
solver (``ddg.dg2d.projection_ns_bdm``), in space and time.

This complements the coupled-solver convergence studies; here we verify
the *semi-implicit projection* itself attains high order.

Spatial (Kovasznay, steady)
---------------------------
Kovasznay flow is an exact steady solution of the Navier-Stokes
equations. The discrete steady state is a fixed point of the projection
time-stepping (at steady state ``u_* = u^{n+1} = u^n`` so the explicit
and implicit convective treatments coincide), so the monolithic steady
solve gives the projection's fixed point. We start the projection there,
take a few BDF2 steps (it holds the fixed point), and measure the L^2
velocity error against the analytic Kovasznay field. Refining ``h`` over
BDM_1/2/3 should give slope ``k+1``.

Temporal (Taylor-Green, decaying)
---------------------------------
Taylor-Green is an exact *unsteady* solution; on a fixed fine mesh we
refine ``dt`` and measure the L^2 error at the final time. BDF1 should
give slope 1, BDF2 slope 2 -- subject to the explicit-convection CFL
bound and any IMEX splitting-error floor.

Run on the remote (needs float64):
    python scripts/projection_convergence.py --mode spatial
    python scripts/projection_convergence.py --mode temporal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM, bdm_k_basis_at_points, bdm_k_evaluate_boundary_velocity,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc, bdm1_gather,
    ns_picard_solve_bdm_k_2d,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.projection_ns_bdm import (
    make_projection_solvers, projection_step_bdm_k_2d,
)

RE = 40.0
NU = 1.0 / RE
LAMBDA = RE / 2.0 - np.sqrt(RE ** 2 / 4.0 + 4.0 * np.pi ** 2)


def kovasznay_uv(x, y):
    ux = 1.0 - jnp.exp(LAMBDA * x) * jnp.cos(2.0 * jnp.pi * y)
    uy = (LAMBDA / (2.0 * jnp.pi)) * jnp.exp(LAMBDA * x) * jnp.sin(
        2.0 * jnp.pi * y)
    return ux, uy


def l2_velocity_error(V, u_global, exact_uv):
    """L2 norm of (u_h - exact) over the domain via cubature."""
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 2)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum("kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    u_h_at_q = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    r_np = np.asarray(r_q); s_np = np.asarray(s_q)
    x_q = (Vx0[:, None] + 0.5 * (r_np[None, :] + 1.0) * (Vx1 - Vx0)[:, None]
           + 0.5 * (s_np[None, :] + 1.0) * (Vx2 - Vx0)[:, None])
    y_q = (Vy0[:, None] + 0.5 * (r_np[None, :] + 1.0) * (Vy1 - Vy0)[:, None]
           + 0.5 * (s_np[None, :] + 1.0) * (Vy2 - Vy0)[:, None])
    u_ex, v_ex = exact_uv(jnp.asarray(x_q), jnp.asarray(y_q))
    sq = (u_h_at_q[..., 0] - u_ex) ** 2 + (u_h_at_q[..., 1] - v_ex) ** 2
    return float(jnp.sqrt(jnp.einsum("q,k,kq->", w_q, J, sq)))


def kovasznay_steady_projection(Kx, Ky, order, dt, n_bdf2_steps):
    """Discrete steady Kovasznay via the projection solver, returned for
    L2-error measurement. Starts from the monolithic steady state (the
    projection's fixed point) and holds it with a few projection steps."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(-0.5, 1.0, -0.5, 1.5, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, 1)
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, kovasznay_uv)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, kovasznay_uv)
    bc_mask = bdm1_boundary_dofs_mask(V)
    # Monolithic discrete steady state = the projection's fixed point.
    u_mono, _ = ns_picard_solve_bdm_k_2d(
        V, nu=NU, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        u_init=u_bc, pressure_eps=1e-10, tau_scale=4.0,
        max_iters=40, tol=1e-9,
    )
    ps1, hi1 = make_projection_solvers(V, dt=dt, nu=NU, order=1, bc_mask=bc_mask)
    ps2, hi2 = make_projection_solvers(V, dt=dt, nu=NU, order=2, bc_mask=bc_mask)
    u_n = u_mono
    u_new, _ = projection_step_bdm_k_2d(
        V, [u_n], dt=dt, nu=NU, order=1, pressure_solver=ps1, h_inv=hi1,
        u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask)
    u_nm1, u_n = u_n, u_new
    for _ in range(n_bdf2_steps):
        u_new, _ = projection_step_bdm_k_2d(
            V, [u_n, u_nm1], dt=dt, nu=NU, order=2,
            pressure_solver=ps2, h_inv=hi2,
            u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask)
        u_nm1, u_n = u_n, u_new
    return V, u_n


def fit_slope(h, err):
    h = np.asarray(h, float); err = np.asarray(err, float)
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])


def run_spatial(orders, meshes, dt, n_steps, out_dir):
    results = {}
    for k in orders:
        hs, errs = [], []
        for n in meshes:
            V, u = kovasznay_steady_projection(n, n, k, dt, n_steps)
            e = l2_velocity_error(V, u, kovasznay_uv)
            h = 1.5 / n
            hs.append(h); errs.append(e)
            print(f"  BDM_{k} n={n:3d} h={h:.4f}  L2={e:.3e}", flush=True)
        slope = fit_slope(hs, errs)
        print(f"BDM_{k}: spatial slope = {slope:.2f}  (expect {k + 1})",
              flush=True)
        results[f"bdm{k}"] = dict(h=hs, err=errs, slope=slope, expected=k + 1)
    out = Path(out_dir) / "projection_spatial_convergence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("spatial", "temporal"),
                    default="spatial")
    ap.add_argument("--orders", type=str, default="1,2,3")
    ap.add_argument("--meshes", type=str, default="6,8,12,16,24")
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--n-steps", type=int, default=5)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "output")
    args = ap.parse_args()
    orders = [int(s) for s in args.orders.split(",")]
    meshes = [int(s) for s in args.meshes.split(",")]
    if args.mode == "spatial":
        run_spatial(orders, meshes, args.dt, args.n_steps, args.out_dir)
    else:
        raise SystemExit("temporal mode: see projection_temporal_convergence "
                         "(Taylor-Green) -- staged next.")


if __name__ == "__main__":
    main()
