"""Head-to-head: BDM dual-splitting projection vs monolithic Newton on
the cylinder smoke (Re=100). Same mesh, same dt, same nu; compare per-step
wall-clock and the BDM divergence norm and a probe velocity trace.

The monolithic path uses ``ns_picard_apply_bdm_k_2d`` + dense LU at each
Newton iteration (the production transient stack from the existing
cylinder app). The projection path uses
``ddg.dg2d.projection_ns_bdm.projection_step_bdm_k_2d``: one
pressure-Schur + one Helmholtz dense-LU solve per BDF step, with the
convective term frozen at the BDF-extrapolated history.

This is the integration smoke test for the projection solver on a
non-trivial (outflow + curved obstacle) geometry. If the projection
remains bounded across N steps and the probe trace stays within a few
percent of the monolithic, the splitting is wired correctly on the
production setup.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ddg.dg2d.bdm import (
    bdm_k_basis_at_points, bdm_k_divergence_apply, bdm1_gather,
    _element_jacobian_matrices,
)
from ddg.dg2d.projection_ns_bdm import (
    make_projection_solvers, projection_step_bdm_k_2d,
)
from diff_cylinder_dt_tau import (
    build_static, X_MIN, Y_MIN, Y_MAX, CX, CY, RADIUS, D_CYL,
    U_MAX, U_MEAN, NU,
)
from inverse_nu_cylinder_2d import make_step_nu_param, make_probe_fn


def make_step_projection(V, u_bc, u_bc_vec, bc_mask, is_natural_face,
                          *, dt, nu, tau_scale, order):
    """Build a projection-step closure with cached LU factors."""
    pressure_solver, h_inv = make_projection_solvers(
        V, dt=dt, nu=nu, order=order,
        tau_scale=tau_scale,
        bc_mask=bc_mask,
        is_natural_face=is_natural_face,
    )

    def step(u_n, u_nm1, order_eff):
        u_hist = [u_n] if order_eff == 1 else [u_n, u_nm1]
        u_new, p_new = projection_step_bdm_k_2d(
            V, u_hist, dt=dt, nu=nu, order=order_eff,
            pressure_solver=pressure_solver, h_inv=h_inv,
            u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
            is_natural_face=is_natural_face,
            tau_scale=tau_scale,
        )
        return u_new, p_new

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--n-newton", type=int, default=2,
                    help="Newton iters per BDF step (monolithic)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    V, u_bc, u_bc_vec, bc_mask, is_natural_face, _ = build_static(
        args.kx, args.ky, args.order,
    )
    print(f"BDM_{args.order} cylinder smoke: K={V.K}, n_u={V.total_dofs}",
          flush=True)

    # --- Monolithic step ---
    step_mono_raw = make_step_nu_param(
        V, u_bc, u_bc_vec, bc_mask, is_natural_face,
        tau_scale=args.tau_scale, n_newton=args.n_newton,
    )
    step_mono = jax.jit(step_mono_raw, static_argnames=("order_eff",))

    # --- Projection step (build BDF1 and BDF2 variants) ---
    step_proj_bdf1 = jax.jit(make_step_projection(
        V, u_bc, u_bc_vec, bc_mask, is_natural_face,
        dt=args.dt, nu=NU, tau_scale=args.tau_scale, order=1,
    ), static_argnames=("order_eff",))
    step_proj_bdf2 = jax.jit(make_step_projection(
        V, u_bc, u_bc_vec, bc_mask, is_natural_face,
        dt=args.dt, nu=NU, tau_scale=args.tau_scale, order=2,
    ), static_argnames=("order_eff",))

    # --- Probe rake (downstream) for trace comparison ---
    probe_xy = np.array(
        [[CX + i * D_CYL, CY] for i in (1, 2, 3, 4, 5, 6)],
        dtype=float,
    )
    probe_fn = make_probe_fn(V, probe_xy)

    n_u = V.total_dofs

    def divergence_norm(u):
        Du = bdm_k_divergence_apply(V, u)
        return float(jnp.linalg.norm(Du))

    # =================== MONOLITHIC ROLLOUT ===================
    print("\n--- Monolithic Newton rollout ---", flush=True)
    u_n_m = jnp.zeros(n_u); u_nm1_m = jnp.zeros(n_u)
    t0 = time.time()
    mono_record = []
    for s in range(args.n_steps):
        order_eff = 1 if s == 0 else 2
        u_new, _ = step_mono(u_n_m, u_nm1_m, args.dt, NU, order_eff)
        u_new.block_until_ready()
        divn = divergence_norm(u_new)
        probe = np.asarray(probe_fn(u_new))
        mono_record.append(dict(
            step=s, t=(s + 1) * args.dt,
            div_norm=divn,
            probe_ux=float(probe[0, 0]),  # first probe x-velocity
        ))
        u_nm1_m, u_n_m = u_n_m, u_new
        if s % 5 == 0:
            print(f"  step {s:3d}: ||u||={float(jnp.linalg.norm(u_new)):.3e}  "
                  f"||Du||={divn:.3e}  "
                  f"probe[0,ux]={probe[0,0]:.4f}", flush=True)
    mono_wall = time.time() - t0
    u_mono_final = u_n_m
    print(f"  monolithic wall: {mono_wall:.1f}s "
          f"({mono_wall / args.n_steps:.2f}s/step)", flush=True)

    # =================== PROJECTION ROLLOUT ===================
    print("\n--- Projection (block-Schur) rollout ---", flush=True)
    u_n_p = jnp.zeros(n_u); u_nm1_p = jnp.zeros(n_u)
    t0 = time.time()
    proj_record = []
    for s in range(args.n_steps):
        order_eff = 1 if s == 0 else 2
        step_fn = step_proj_bdf1 if order_eff == 1 else step_proj_bdf2
        u_new, _ = step_fn(u_n_p, u_nm1_p, order_eff)
        u_new.block_until_ready()
        if not bool(jnp.all(jnp.isfinite(u_new))):
            print(f"  *** NaN at step {s} ***", flush=True)
            break
        divn = divergence_norm(u_new)
        probe = np.asarray(probe_fn(u_new))
        proj_record.append(dict(
            step=s, t=(s + 1) * args.dt,
            div_norm=divn,
            probe_ux=float(probe[0, 0]),
        ))
        u_nm1_p, u_n_p = u_n_p, u_new
        if s % 5 == 0:
            print(f"  step {s:3d}: ||u||={float(jnp.linalg.norm(u_new)):.3e}  "
                  f"||Du||={divn:.3e}  "
                  f"probe[0,ux]={probe[0,0]:.4f}", flush=True)
    proj_wall = time.time() - t0
    u_proj_final = u_n_p
    print(f"  projection wall: {proj_wall:.1f}s "
          f"({proj_wall / args.n_steps:.2f}s/step)", flush=True)

    # --- Comparison ---
    diff = float(jnp.linalg.norm(u_proj_final - u_mono_final))
    norm_m = float(jnp.linalg.norm(u_mono_final))
    print(f"\n||u_proj - u_mono|| / ||u_mono|| = {diff/norm_m:.3e}",
          flush=True)
    print(f"speedup (mono / proj wall) = {mono_wall / proj_wall:.2f}x",
          flush=True)

    out = dict(
        order=args.order, kx=args.kx, ky=args.ky, K=int(V.K),
        n_u=int(n_u), dt=args.dt, n_steps=args.n_steps,
        tau_scale=args.tau_scale, n_newton_mono=args.n_newton,
        mono_wall=mono_wall, proj_wall=proj_wall,
        speedup=mono_wall / proj_wall,
        final_rel_diff=diff / norm_m,
        mono_record=mono_record, proj_record=proj_record,
    )
    out_json = args.out_dir / "projection_vs_monolithic_cylinder.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
