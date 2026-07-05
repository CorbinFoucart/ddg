"""Temporal order-of-accuracy study for the BDM_k/P_{k-1} block-Schur
PROJECTION solver on the unsteady Taylor-Green vortex.

Taylor-Green is an exact, source-free solution of the incompressible
Navier-Stokes equations on [0, 2*pi]^2:

    u(x,y,t) = -cos(x) sin(y) exp(-2 nu t)
    v(x,y,t) =  sin(x) cos(y) exp(-2 nu t)
    p(x,y,t) = -1/4 (cos 2x + cos 2y) exp(-4 nu t)

(the nonlinear term is an exact gradient, absorbed by the pressure). It
is the natural vehicle for the projection's *temporal* order: the
scheme is designed for unsteady flow, and convection is genuinely
active. We fix a fine mesh, seed the projection from the exact field
at t=0 (canonical BDM interpolant), impose the exact time-dependent
Dirichlet trace at each step, march to a fixed final time T, and
measure the L^2 velocity error vs the exact field at T as dt is
refined. BDF1 should give slope 1, BDF2 slope 2 (subject to the
explicit-convection CFL bound and the IMEX splitting-error floor).

Run on the remote (needs float64):
    python scripts/projection_temporal_convergence.py --bdf 2
    python scripts/projection_temporal_convergence.py --bdf 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (  # noqa: E402
    BDM, bdm_k_basis_at_points, bdm_k_canonical_interpolant,
    bdm_k_evaluate_boundary_velocity, _element_jacobian_matrices,
    bdm_k_mass_apply,
)
from ddg.dg2d.coupled_ns_bdm import (  # noqa: E402
    bdm1_evaluate_dirichlet_bc, bdm1_boundary_dofs_mask, bdm1_gather,
)
from ddg.dg2d.cubature import triangle_cubature  # noqa: E402
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d  # noqa: E402
from ddg.dg2d.projection_ns_bdm import (  # noqa: E402
    make_projection_solvers, projection_step_bdm_k_2d,
)

TWO_PI = 2.0 * np.pi


def make_tg(nu):
    """Return ``uv(x,y,t)`` for the Taylor-Green vortex at viscosity nu."""
    def uv(x, y, t):
        decay = jnp.exp(-2.0 * nu * t)
        ux = -jnp.cos(x) * jnp.sin(y) * decay
        uy = jnp.sin(x) * jnp.cos(y) * decay
        return ux, uy
    return uv


def l2_velocity_error(V, u_global, exact_uv):
    """L2 velocity error of ``u_global`` against ``exact_uv(x,y)``."""
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 4)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum("kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    uh = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    rqn = np.asarray(r_q); sqn = np.asarray(s_q)
    x_q = (Vx0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None]
           + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None])
    y_q = (Vy0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None]
           + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None])
    ux, uy = exact_uv(jnp.asarray(x_q), jnp.asarray(y_q))
    dx = uh[..., 0] - ux; dy = uh[..., 1] - uy
    return float(jnp.sqrt(jnp.einsum("q,k,kq->", w_q, J, dx ** 2 + dy ** 2)))


def march_tg(V, nu, order, dt, n_steps, tau_scale=4.0):
    """March the projection through ``n_steps`` of Taylor-Green from the
    exact interpolant, with the exact time-dependent Dirichlet trace.
    Returns the final velocity coefficient vector."""
    uv = make_tg(nu)
    bc_mask = bdm1_boundary_dofs_mask(V)

    def bc_at(t):
        fn = lambda x, y: uv(x, y, t)  # noqa: E731
        return bdm1_evaluate_dirichlet_bc(V, fn), bdm_k_evaluate_boundary_velocity(V, fn)

    u0 = bdm_k_canonical_interpolant(V, lambda x, y: uv(x, y, 0.0))
    ps1, hi1 = make_projection_solvers(V, dt=dt, nu=nu, order=1, bc_mask=bc_mask, tau_scale=tau_scale)
    ps2, hi2 = make_projection_solvers(V, dt=dt, nu=nu, order=2, bc_mask=bc_mask, tau_scale=tau_scale)
    # BDF1 bootstrap step to t = dt.
    u_bc, u_bc_vec = bc_at(dt)
    u1, _ = projection_step_bdm_k_2d(
        V, [u0], dt=dt, nu=nu, order=1, pressure_solver=ps1, h_inv=hi1,
        u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask, tau_scale=tau_scale)
    if order == 1:
        u_n = u1
        for s in range(2, n_steps + 1):
            u_bc, u_bc_vec = bc_at(s * dt)
            u_n, _ = projection_step_bdm_k_2d(
                V, [u_n], dt=dt, nu=nu, order=1, pressure_solver=ps1, h_inv=hi1,
                u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask, tau_scale=tau_scale)
        return u_n
    u_nm1, u_n = u0, u1
    for s in range(2, n_steps + 1):
        u_bc, u_bc_vec = bc_at(s * dt)
        u_new, _ = projection_step_bdm_k_2d(
            V, [u_n, u_nm1], dt=dt, nu=nu, order=2, pressure_solver=ps2, h_inv=hi2,
            u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask, tau_scale=tau_scale)
        u_nm1, u_n = u_n, u_new
    return u_n


def fit_slopes(dts, errs):
    dts = np.asarray(dts, float); errs = np.asarray(errs, float)
    return [float((np.log(errs[i]) - np.log(errs[i + 1]))
                  / (np.log(dts[i]) - np.log(dts[i + 1])))
            for i in range(len(errs) - 1)]


def l2_diff(V, u1, u2):
    """L2 norm of the difference field (u1 - u2) over the domain, via the
    BDM mass matrix: ||v||_M = sqrt(v . M v). Both fields live in the same
    BDM space, so this is the exact L2 function-space difference."""
    d = u1 - u2
    return float(jnp.sqrt(jnp.abs(jnp.dot(d, bdm_k_mass_apply(V, d)))))


def run(order, k, n_mesh, nu, T, n_steps_list, out_dir):
    """Fixed BDM_k mesh, refine dt; isolate the temporal error by
    SELF-CONVERGENCE -- measure each dt solution against the finest-dt
    solution on the SAME mesh, so the (constant) spatial error cancels
    and only the O(dt^q) temporal/splitting error remains. (Measuring vs
    the exact solution instead floors at the spatial error and hides the
    temporal slope.)"""
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, TWO_PI, 0.0, TWO_PI, n_mesh, n_mesh)
    V = BDM.from_mesh(build_mesh_2d(VX, VY, EToV, 1), order=k)
    exact_T = lambda x, y: make_tg(nu)(x, y, T)  # noqa: E731
    sols = {}
    for ns in sorted(n_steps_list):
        dt = T / ns
        sols[ns] = march_tg(V, nu, order, dt, ns)
        e_exact = l2_velocity_error(V, sols[ns], exact_T)
        print(f"  BDF{order} dt={dt:.4e} (n_steps={ns})  L2_vs_exact={e_exact:.3e}",
              flush=True)
    ns_ref = max(n_steps_list)          # finest dt = self-conv reference
    u_ref = sols[ns_ref]
    dts, errs = [], []
    for ns in sorted(n_steps_list):
        if ns == ns_ref:
            continue
        dts.append(T / ns); errs.append(l2_diff(V, sols[ns], u_ref))
        print(f"  BDF{order} dt={T/ns:.4e}: self-conv L2 vs finest dt = "
              f"{errs[-1]:.3e}", flush=True)
    slopes = fit_slopes(dts, errs)
    print(f"BDF{order}: temporal (self-conv) asymptotic slope = {slopes[-1]:.2f}  "
          f"(expect {order})", flush=True)
    res = dict(order=order, n_mesh=n_mesh, k=k, nu=nu, T=T,
               dts=dts, errs=errs, slopes=slopes, ref_n_steps=ns_ref,
               asymptotic_slope=slopes[-1], expected_slope=order)
    out = Path(out_dir) / f"projection_taylor_green_temporal_bdf{order}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdf", type=int, choices=(1, 2), default=2)
    ap.add_argument("--k", type=int, default=3,
                    help="BDM order of the fixed spatial mesh (high enough "
                         "that spatial error sits below the temporal error)")
    ap.add_argument("--n-mesh", type=int, default=12)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--n-steps", type=str, default="10,20,40,80")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "output")
    args = ap.parse_args()
    nu = 1.0 / args.re
    n_steps_list = [int(s) for s in args.n_steps.split(",")]
    run(args.bdf, args.k, args.n_mesh, nu, args.T, n_steps_list, args.out_dir)


if __name__ == "__main__":
    main()
