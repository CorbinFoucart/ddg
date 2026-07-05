"""Spatial order-of-accuracy study for the BDM_k/P_{k-1} block-Schur
PROJECTION solver on the steady Stokes problem.

Why Stokes (not Kovasznay): the projection treats convection
explicitly, so on a steady *convective* problem (Kovasznay Re=40) it
drifts off the monolithic steady state -- the explicit-convection
limitation characterised in the CFL section. In the Stokes limit the
projection's Helmholtz block is H = nu A (no convective term), and the
block-Schur factorisation is an exact factorisation of the resulting
saddle (``test_projection_block_schur_factorisation``). So Stokes is
where the projection's *spatial* discretisation order is cleanly
testable; its temporal order is tested separately on the unsteady
Taylor-Green flow (``projection_temporal_convergence.py``).

The projection has no convection-off flag, so we drive its block-Schur
solver directly: build (pressure_solver, h_inv) with a huge dt so the
mass term gamma0/dt M is negligible and H ~= nu A, then run the three
projection substeps on the manufactured-Stokes residual
``F_u = nu A u_extrap - f_load - lift`` (no convective term). The
manufactured solution, load and L2-error machinery are reused verbatim
from ``test/test_bdm_stokes_rate.py``.

Run on the remote (needs float64):
    python scripts/projection_stokes_convergence.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Reuse the verified manufactured-Stokes harness.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
import test_bdm_stokes_rate as S  # noqa: E402

from ddg.dg2d.bdm import (  # noqa: E402
    BDM, bdm_k_viscous_apply, bdm_k_viscous_dirichlet_lift,
    bdm_k_divergence_apply, bdm_k_gradient_apply,
    bdm_k_evaluate_boundary_velocity,
)
from ddg.dg2d.coupled_ns_bdm import (  # noqa: E402
    bdm1_evaluate_dirichlet_bc, bdm1_boundary_dofs_mask,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d  # noqa: E402
from ddg.dg2d.projection_ns_bdm import make_projection_solvers  # noqa: E402

DT_HUGE = 1.0e12  # H = gamma0/dt M + nu A  ->  ~ nu A (mass term ~1e-12)


def solve_stokes_projection(V):
    """Steady Stokes solve via the projection's block-Schur factorisation
    (H ~= nu A, convection excluded by construction). Returns the BDM
    velocity coefficient vector."""
    u_bc = bdm1_evaluate_dirichlet_bc(V, S._u_anal)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, S._u_anal)
    bc_mask = bdm1_boundary_dofs_mask(V)
    lift = bdm_k_viscous_dirichlet_lift(V, u_bc_vec, nu=S.NU, tau_scale=4.0)
    f_load = S._build_load(V)
    pressure_solver, h_inv = make_projection_solvers(
        V, dt=DT_HUGE, nu=S.NU, order=1, bc_mask=bc_mask, tau_scale=4.0,
    )
    # Stokes residual at u_extrap = (BC trace, interior 0): no mass term
    # (alpha ~ 1e-12), no convective term.
    u_extrap = jnp.where(bc_mask, u_bc, jnp.zeros(V.total_dofs))
    F_u = bdm_k_viscous_apply(V, u_extrap, nu=S.NU, tau_scale=4.0) - f_load - lift
    F_p = -bdm_k_divergence_apply(V, u_extrap)
    rhs_aux = -F_u
    # Block-Schur substeps (identical to projection_step_bdm_k_2d).
    u_hat = h_inv(rhs_aux)
    p = pressure_solver(bdm_k_divergence_apply(V, u_hat) - F_p)
    delta_u = h_inv(rhs_aux - bdm_k_gradient_apply(V, p))
    return jnp.where(bc_mask, u_bc, u_extrap + delta_u)


def fit_slopes(hs, errs):
    hs = np.asarray(hs, float); errs = np.asarray(errs, float)
    return [
        float((np.log(errs[i]) - np.log(errs[i + 1]))
              / (np.log(hs[i]) - np.log(hs[i + 1])))
        for i in range(len(errs) - 1)
    ]


def run(orders, meshes_by_order, out_dir):
    results = {}
    for k in orders:
        meshes = meshes_by_order[k]
        hs, errs_proj, errs_mono = [], [], []
        for n in meshes:
            VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, n, n)
            V = BDM.from_mesh(build_mesh_2d(VX, VY, EToV, 1), order=k)
            up = solve_stokes_projection(V)
            um = S._solve_stokes(V)
            ep = S._l2_velocity_error(V, up)
            em = S._l2_velocity_error(V, um)
            hs.append(1.0 / n); errs_proj.append(ep); errs_mono.append(em)
            print(f"  BDM_{k} n={n:3d} h={1.0/n:.4f}  proj_L2={ep:.3e}  "
                  f"mono_L2={em:.3e}  |proj-mono|={float(jnp.linalg.norm(up-um)):.2e}",
                  flush=True)
        sp = fit_slopes(hs, errs_proj)
        sm = fit_slopes(hs, errs_mono)
        print(f"BDM_{k}: projection asymptotic slope = {sp[-1]:.2f}  "
              f"(monolithic {sm[-1]:.2f}; expect {k + 1})", flush=True)
        results[f"bdm{k}"] = dict(
            hs=hs, errs_projection=errs_proj, errs_monolithic=errs_mono,
            slopes_projection=sp, slopes_monolithic=sm,
            asymptotic_slope=sp[-1], expected_slope=k + 1,
        )
    out = Path(out_dir) / "projection_stokes_spatial_convergence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=str, default="1,2,3")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "output")
    args = ap.parse_args()
    orders = [int(s) for s in args.orders.split(",")]
    # Mesh ladders per order (match the monolithic Stokes-rate test, with
    # an extra finer level for a cleaner asymptotic slope).
    meshes_by_order = {1: [4, 8, 16, 24], 2: [3, 4, 6, 8], 3: [3, 4, 6]}
    run(orders, meshes_by_order, args.out_dir)


if __name__ == "__main__":
    main()
