"""2D Taylor-Green vortex on the BDM-P_0 transient solver -- time-accuracy
verification.

Exact analytic solution to incompressible NS on the unit square with
no body force:

    u(x, y, t) =  sin(pi x) cos(pi y) exp(-2 pi^2 nu t)
    v(x, y, t) = -cos(pi x) sin(pi y) exp(-2 pi^2 nu t)
    p(x, y, t) = -1/4 [cos(2 pi x) + cos(2 pi y)] exp(-4 pi^2 nu t)

div u = 0 strongly; the convective and pressure-gradient terms balance
exactly. The flow is a pair of counter-rotating vortices that decay
exponentially under viscosity.

We use this as the time-accuracy benchmark for the transient BDM
solver. Time-dependent Dirichlet velocity BCs are sampled from
the analytic solution at every BDF step; the initial condition is the
analytic field at t=0 (interpolated via point evaluation onto BDM_1).

Two diagnostics:

  1. **Final-time L2 error**: ||u_h(T) - u_TG(T)||_L2 at a fixed final
     time, for a single (Kx, dt) configuration. Sanity check that the
     time-stepper actually drives toward the analytic decay.

  2. **Self-convergence rate in dt** (the load-bearing measurement):
     run with dt and dt/2 on the same mesh, compare final solutions.
     For BDF1 the error halves per dt halving (first-order); polluted
     when dt is small enough that spatial discretisation error
     dominates (~h^2).
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
)
from ddg.dg2d.coupled_ns import IterativeSolverConfig
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdf_history,
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    bdm1_mass_apply,
    make_bdm_block_diag_pc, make_bdm_cahouet_chabard_pc,
    make_bdm_velocity_block_jacobi_pc,
    ns_newton_solve_bdm_2d,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


HAMMER_R = jnp.array([0.0, -1.0, 0.0])
HAMMER_S = jnp.array([-1.0, 0.0, 0.0])
HAMMER_W = jnp.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])


def taylor_green_velocity(x, y, t, *, nu):
    """Analytic Taylor-Green velocity at time t."""
    decay = jnp.exp(-2.0 * jnp.pi ** 2 * nu * t)
    u_x = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * decay
    u_y = -jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) * decay
    return u_x, u_y


def _bc_fn_at_time(t, *, nu):
    """Closure capturing t + nu, returning a callable suitable for
    bdm1_evaluate_dirichlet_bc / bdm1_evaluate_boundary_velocity."""
    def fn(x, y):
        return taylor_green_velocity(x, y, t, nu=nu)
    return fn


def _build_setup(Kx: int, Ky: int):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, 1)
    V = BDM.from_mesh(mesh)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, bc_mask


def _ic_at_t0(V: BDM, nu: float) -> jax.Array:
    """Initial condition: BDM representation of u_TG(x, y, 0) at all
    DOFs (boundary + interior). Constructed as the normal-trace
    point-evaluation, matching how the time-dependent Dirichlet lift
    builds u_bc_vec.

    NB: this is the BDM nodal *interpolant* of the analytic field, not
    the L2-projection. For the time-stepper to begin from a
    realistically discretised IC, this is fine -- the time-marching
    drives the field toward whatever the discrete dynamics produce.
    """
    from ddg.dg2d.bdm import (
        bdm1_reference_dof_coordinates,
        _bdm1_face_quadrature_data,
    )
    fdata = _bdm1_face_quadrature_data(V)
    n_face = fdata["n_face"]
    ref_dofs = bdm1_reference_dof_coordinates()
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX)
    VY = np.asarray(V.mesh.VY)
    K = V.K
    r = np.asarray(ref_dofs[:, 0])
    s = np.asarray(ref_dofs[:, 1])
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    x_at = (
        Vx0[:, None]
        + 0.5 * (r[None, :] + 1.0) * (Vx1 - Vx0)[:, None]
        + 0.5 * (s[None, :] + 1.0) * (Vx2 - Vx0)[:, None]
    ).reshape(K, 3, 2)
    y_at = (
        Vy0[:, None]
        + 0.5 * (r[None, :] + 1.0) * (Vy1 - Vy0)[:, None]
        + 0.5 * (s[None, :] + 1.0) * (Vy2 - Vy0)[:, None]
    ).reshape(K, 3, 2)
    ux_f, uy_f = taylor_green_velocity(jnp.asarray(x_at),
                                          jnp.asarray(y_at), 0.0, nu=nu)
    un_phys = ux_f * n_face[:, :, 0:1] + uy_f * n_face[:, :, 1:2]
    u_local = un_phys.reshape(K, 6)
    sign = V.topology.local_sign
    lg = V.topology.local_to_global
    u_global = jnp.zeros(V.total_dofs)
    return u_global.at[lg.reshape(-1)].set(
        (sign * u_local).reshape(-1)
    )


def _l2_error_vs_analytic(V: BDM, u_global: jax.Array, t: float, nu: float) -> float:
    """||u_h(T) - u_TG(T)||_L2 via per-element Hammer-3 cubature on the
    physical mesh.

    For BDM_1 the discrete u_h is piecewise-linear in (r, s); evaluating
    at the Hammer points and comparing to the analytic field's value at
    those physical points gives a degree-2-accurate quadrature, which
    matches BDM_1's polynomial order.
    """
    u_local = bdm1_gather(V.topology, u_global)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_ref = bdm1_basis_at_points(HAMMER_R, HAMMER_S)        # (Q, 6, 2)
    phi_phys = jnp.einsum(
        "kab,qlb->kqla", DF, phi_ref
    ) / J[:, None, None, None]                                 # (K, Q, 6, 2)
    u_at_q = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)    # (K, Q, 2)

    # Physical positions of the Hammer pts per element.
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX)
    VY = np.asarray(V.mesh.VY)
    K = V.K
    r_q = np.asarray(HAMMER_R)
    s_q = np.asarray(HAMMER_S)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    x_q_phys = (
        Vx0[:, None]
        + 0.5 * (r_q[None, :] + 1.0) * (Vx1 - Vx0)[:, None]
        + 0.5 * (s_q[None, :] + 1.0) * (Vx2 - Vx0)[:, None]
    )                                                          # (K, Q)
    y_q_phys = (
        Vy0[:, None]
        + 0.5 * (r_q[None, :] + 1.0) * (Vy1 - Vy0)[:, None]
        + 0.5 * (s_q[None, :] + 1.0) * (Vy2 - Vy0)[:, None]
    )
    ux_a, uy_a = taylor_green_velocity(
        jnp.asarray(x_q_phys), jnp.asarray(y_q_phys), t, nu=nu,
    )
    err_x = u_at_q[..., 0] - ux_a
    err_y = u_at_q[..., 1] - uy_a
    sq = err_x ** 2 + err_y ** 2                              # (K, Q)
    return float(jnp.sqrt(
        jnp.einsum("q,k,kq->", HAMMER_W, J, sq)
    ))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_taylor_green(
    *, Kx: int, dt: float, t_final: float, nu: float,
    bdf_order: int = 2,
    linearisation: str = "picard",
    use_iterative: bool = True,
    tol_outer: float = 1e-8,
    tol_inner: float = 1e-8,
    max_outer: int = 10,
    verbose: bool = True,
):
    """Time-march Taylor-Green from t=0 to t_final, sampling the
    analytic BC at every step (Dirichlet, time-dependent). Step 0
    bootstraps with BDF1; thereafter ``bdf_order``."""
    if bdf_order not in (1, 2):
        raise ValueError(f"bdf_order must be 1 or 2, got {bdf_order}")
    V, bc_mask = _build_setup(Kx, Kx)
    n_u, n_p = V.total_dofs, V.K

    pcs_per_order = {}
    iter_cfg = None
    if use_iterative:
        iter_cfg = IterativeSolverConfig(
            method="gmres", tol=tol_inner, atol=1e-12,
            maxiter=500, restart=50,
        )
        for order in {1, bdf_order}:
            gamma0 = BDF_COEFFS[order]["gamma0"]
            alpha = gamma0 / dt
            vel_pc = make_bdm_velocity_block_jacobi_pc(V, alpha=alpha, nu=nu)
            schur_pc = make_bdm_cahouet_chabard_pc(
                V, nu=nu, dt=dt, gamma0=gamma0,
            )
            pcs_per_order[order] = make_bdm_block_diag_pc(
                V, velocity_pc=vel_pc, pressure_pc=schur_pc,
            )

    u_n = _ic_at_t0(V, nu)
    u_nm1 = None
    p_prev = jnp.zeros(n_p)

    n_steps = int(round(t_final / dt))
    err_history = [_l2_error_vs_analytic(V, u_n, 0.0, nu)]
    if verbose:
        print(f"Taylor-Green: Kx={Kx}, dt={dt:.4e}, nu={nu:.3e}, T={t_final}, "
              f"n_steps={n_steps}, DOFs={n_u+n_p}, BDF{bdf_order}, "
              f"IC L2 error={err_history[0]:.3e}")

    t0 = time.time()
    for step in range(n_steps):
        t_next = (step + 1) * dt
        order_eff = 1 if (bdf_order > 1 and step == 0) else bdf_order
        alpha = BDF_COEFFS[order_eff]["gamma0"] / dt
        u_hist = [u_n] if order_eff == 1 else [u_n, u_nm1]
        f_history = bdf_history(V, u_hist, dt=dt, order=order_eff)
        pc = pcs_per_order.get(order_eff)

        # Time-dependent Dirichlet BC: rebuild u_bc + u_bc_vec at t_next.
        u_bc = bdm1_evaluate_dirichlet_bc(V, _bc_fn_at_time(t_next, nu=nu))
        u_bc_vec = bdm1_evaluate_boundary_velocity(V, _bc_fn_at_time(t_next, nu=nu))

        u_new, p_new, _ = ns_newton_solve_bdm_2d(
            V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
            u_init=u_n, p_init=p_prev,
            velocity_alpha=alpha, f_velocity=f_history,
            pressure_eps=1e-10, tau_scale=4.0,
            max_iters=max_outer, tol=tol_outer, atol=1e-12,
            linearisation=linearisation,
            iterative=iter_cfg, coupled_preconditioner=pc,
        )
        err_history.append(_l2_error_vs_analytic(V, u_new, t_next, nu))
        if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
            print(f"  step {step+1:4d}/{n_steps} t={t_next:.4f} BDF{order_eff}: "
                  f"L2 err = {err_history[-1]:.3e}")
        u_nm1 = u_n
        u_n = u_new
        p_prev = p_new
    wall = time.time() - t0
    if verbose:
        print(f"Done in {wall:.1f}s ({wall/n_steps*1000:.0f} ms/step). "
              f"Final L2 err = {err_history[-1]:.3e}")
    return dict(
        V=V, Kx=Kx, dt=dt, t_final=t_final, nu=nu, bdf_order=int(bdf_order),
        n_steps=n_steps, err_history=np.asarray(err_history),
        wall_seconds=wall,
    )


def convergence_sweep(
    *, Kx: int, t_final: float, nu: float, dts,
    bdf_order: int = 2,
    out_dir: Path | None = None,
):
    """Run the Taylor-Green time-marcher at several Δt values, record
    the final-time L2 error for each. Returns the (Δt, error, slope)
    summary. Expected slope is ``bdf_order`` once Δt is small enough
    for temporal error to dominate but not so small that spatial error
    takes over."""
    rows = []
    for dt in dts:
        result = run_taylor_green(
            Kx=Kx, dt=dt, t_final=t_final, nu=nu,
            bdf_order=bdf_order, verbose=False,
        )
        final_err = float(result["err_history"][-1])
        rows.append(dict(dt=dt, n_steps=result["n_steps"],
                          final_l2_err=final_err,
                          wall_seconds=result["wall_seconds"]))
        print(f"  dt={dt:.4e}: n_steps={result['n_steps']:4d}, "
              f"final L2 err = {final_err:.3e} "
              f"({result['wall_seconds']:.1f}s)")
    # Pairwise convergence slopes.
    arr_dt = np.array([r["dt"] for r in rows])
    arr_err = np.array([r["final_l2_err"] for r in rows])
    slopes = np.diff(np.log(arr_err)) / np.diff(np.log(arr_dt))
    print(f"Pairwise convergence slope (BDF{bdf_order} expects ~{bdf_order}):")
    for i, s in enumerate(slopes):
        print(f"  {arr_dt[i]:.4e} -> {arr_dt[i+1]:.4e}:  slope = {s:.3f}")

    summary = dict(
        Kx=Kx, t_final=t_final, nu=nu, bdf_order=int(bdf_order),
        rows=rows,
        slopes=slopes.tolist(),
    )
    if out_dir is not None:
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"bdm_taylor_green_convergence_kx{Kx}_bdf{bdf_order}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {out_path}")
        # Convergence plot.
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(arr_dt, arr_err, "o-", label=f"BDM-P_0 BDF{bdf_order}")
        ref = arr_err[0] * (arr_dt / arr_dt[0]) ** bdf_order
        ax.loglog(arr_dt, ref, "k--", alpha=0.4,
                   label=f"slope {bdf_order} reference")
        ax.set_xlabel("$\\Delta t$"); ax.set_ylabel("$\\|u_h - u_{TG}\\|_{L^2}$ at $T$")
        ax.set_title(f"BDM transient Taylor-Green: time accuracy "
                      f"(Kx={Kx}, nu={nu}, BDF{bdf_order})")
        ax.grid(which="both", alpha=0.3); ax.legend()
        fig.tight_layout()
        png_path = out_dir / f"bdm_taylor_green_convergence_kx{Kx}_bdf{bdf_order}.png"
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        print(f"  wrote {png_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kx", type=int, default=8)
    ap.add_argument("--tfinal", type=float, default=0.2)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--bdf-order", type=int, choices=(1, 2), default=2)
    ap.add_argument("--sweep", action="store_true",
                     help="Run a Δt convergence sweep (default: single run)")
    ap.add_argument("--dt", type=float, default=0.025,
                     help="(non-sweep mode) single Δt")
    ap.add_argument("--dts", type=str, default="0.04,0.02,0.01,0.005",
                     help="(sweep mode) comma-separated Δt values")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    if args.sweep:
        dts = [float(s) for s in args.dts.split(",")]
        convergence_sweep(
            Kx=args.kx, t_final=args.tfinal, nu=args.nu,
            bdf_order=args.bdf_order, dts=dts, out_dir=out_dir,
        )
    else:
        run_taylor_green(
            Kx=args.kx, dt=args.dt, t_final=args.tfinal, nu=args.nu,
            bdf_order=args.bdf_order,
        )


if __name__ == "__main__":
    main()
