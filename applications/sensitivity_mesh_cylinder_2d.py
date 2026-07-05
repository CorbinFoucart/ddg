"""Sensitivity-through-differentiable-mesh smoke test.

End-to-end demonstration that ``jax.grad`` flows from a scalar
functional through:

  cy (parameter)
    -> mesh.x, mesh.y                         (differentiable mesh affine map)
    -> mesh.J, mesh.rx/sx/ry/sy               (geometric factors)
    -> mesh.nx/ny/sJ                          (normals)
    -> BDMTopology.piola_scale                (refreshed from sJ)
    -> BDM mass apply / divergence apply      (Piola-mapped basis on diff mesh)
    -> scalar loss

The diff-INS forward solve (BDF2 transient + nonlinear iterations)
through the diff mesh additionally needs ~6 ``np.asarray(mesh.…)``
calls in ``ddg.dg2d.coupled_ns_bdm`` to be rewritten in JAX --- this
is the obvious next refactor, blocked by a substantial amount of
machinery touching BC sampling, Newton extra columns, and the lift
operator. The Phase 1-3 differentiable-mesh infrastructure
(``ddg.dg2d.differentiable_mesh``) and the BDM piola-scale refresh
(``ddg.dg2d.bdm.refresh_piola_scale``) suffice to drive the mesh-AD
chain through the *operators* themselves, which is what this script
exercises.

What it validates
-----------------
* AD vs central FD on ``L(cy) = sum((M u_const)^2)`` --- mass apply on
  a fixed velocity ``u_const``, sensitive to cy via the Piola weights
  + geometric factors. Should agree to <1e-6 in relative magnitude.
* AD vs FD on ``L(cy) = sum((D u_const)^2)`` --- divergence apply on
  the same ``u_const``, exercises the cross-space operator on
  ``mesh.rx/sx/ry/sy``.
* AD vs FD jointly: a small 3-parameter sensitivity matrix
  ``∂L / ∂(cx, cy, R)`` matches FD on all 6 entries.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))

from ddg.dg2d.bdm import (
    BDM, bdm_k_mass_apply, refresh_piola_scale,
)
from ddg.dg2d.coupled_ns_bdm import bdm_k_mass_apply as _mass_alias
from ddg.dg2d.differentiable_mesh import (
    make_template, make_curving_info, differentiable_mesh_with_circle,
)
from ddg.dg2d.bdm import bdm_k_divergence_apply
from ddg.dg2d.mesh import build_mesh_2d, channel_with_circle_mesh_2d


def _setup(kx=20, ky=10, order=2,
           cx=0.2, cy=0.25, radius=0.05):
    VX, VY, EToV = channel_with_circle_mesh_2d(
        0.0, 1.0, 0.0, 0.5, kx, ky,
        cx=cx, cy=cy, radius=radius,
        R_outer=0.1, n_radial=2, radial_growth=1.3,
    )
    mesh = build_mesh_2d(VX, VY, EToV, order)
    template = make_template(mesh)
    curving = make_curving_info(
        mesh, cx=cx, cy=cy, radius=radius, detection_tol=0.05,
    )
    mesh_curved = differentiable_mesh_with_circle(
        template.VX_nominal, template.VY_nominal, template, curving,
        jnp.asarray(cx), jnp.asarray(cy), jnp.asarray(radius),
    )
    V_nominal = BDM.from_mesh(mesh_curved, order=order)
    topology = V_nominal.topology
    return template, curving, topology


def _fixed_velocity(V):
    """A deterministic, BC-mask-free velocity vector for sensitivity
    checks. Just an arange normalised; doesn't need to be div-free."""
    n = V.total_dofs
    rng = np.random.default_rng(seed=0xDEADBEEF)
    return jnp.asarray(rng.standard_normal(n))


def _build_diff_BDM(template, curving, topology, cx, cy, radius, order):
    mesh_diff = differentiable_mesh_with_circle(
        template.VX_nominal, template.VY_nominal, template, curving,
        cx, cy, radius,
    )
    topo = refresh_piola_scale(topology, mesh_diff)
    return BDM(mesh=mesh_diff, topology=topo, order=order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--cx", type=float, default=0.2)
    ap.add_argument("--cy", type=float, default=0.25)
    ap.add_argument("--r", type=float, default=0.05)
    ap.add_argument("--h", type=float, default=1e-6)
    args = ap.parse_args()

    template, curving, topology = _setup(
        kx=args.kx, ky=args.ky, order=args.order,
        cx=args.cx, cy=args.cy, radius=args.r,
    )

    # Fix the velocity vector at the nominal mesh; reuse across all
    # perturbed mesh evaluations.
    V_nominal = _build_diff_BDM(
        template, curving, topology,
        jnp.asarray(args.cx), jnp.asarray(args.cy), jnp.asarray(args.r),
        args.order,
    )
    u_const = _fixed_velocity(V_nominal)
    n_u = V_nominal.total_dofs
    print(f"BDM_{args.order} smoke: K={V_nominal.K}, n_u={n_u}",
          flush=True)
    print(f"  nominal (cx, cy, r) = "
          f"({args.cx}, {args.cy}, {args.r})", flush=True)

    def loss_M(cx, cy, R):
        V = _build_diff_BDM(template, curving, topology,
                            cx, cy, R, args.order)
        Mu = bdm_k_mass_apply(V, u_const)
        return jnp.sum(Mu ** 2)

    def loss_D(cx, cy, R):
        V = _build_diff_BDM(template, curving, topology,
                            cx, cy, R, args.order)
        Du = bdm_k_divergence_apply(V, u_const)
        return jnp.sum(Du ** 2)

    grad_M = jax.jit(jax.grad(loss_M, argnums=(0, 1, 2)))
    grad_D = jax.jit(jax.grad(loss_D, argnums=(0, 1, 2)))
    loss_M_jit = jax.jit(loss_M)
    loss_D_jit = jax.jit(loss_D)

    cx0 = jnp.asarray(args.cx)
    cy0 = jnp.asarray(args.cy)
    R0 = jnp.asarray(args.r)

    print("compiling + computing AD ...", flush=True)
    t0 = time.time()
    g_M = grad_M(cx0, cy0, R0)
    g_D = grad_D(cx0, cy0, R0)
    print(f"  AD pass: {time.time() - t0:.1f}s", flush=True)

    LM0 = float(loss_M_jit(cx0, cy0, R0))
    LD0 = float(loss_D_jit(cx0, cy0, R0))
    print(f"  baseline: L_M = {LM0:.4e}, L_D = {LD0:.4e}", flush=True)

    print("FD validation ...", flush=True)
    h = args.h
    fd_M = []
    fd_D = []
    for j, base in enumerate((cx0, cy0, R0)):
        plus = [cx0, cy0, R0]; plus[j] = base + h
        minus = [cx0, cy0, R0]; minus[j] = base - h
        fM = (float(loss_M_jit(*plus)) - float(loss_M_jit(*minus))) / (2 * h)
        fD = (float(loss_D_jit(*plus)) - float(loss_D_jit(*minus))) / (2 * h)
        fd_M.append(fM); fd_D.append(fD)
    fd_M = np.array(fd_M); fd_D = np.array(fd_D)

    print()
    print("Sensitivity matrix (BDM mass apply, sum((M u)^2)):",
          flush=True)
    names = ("cx", "cy", "R")
    for j in range(3):
        ad = float(g_M[j]); f = fd_M[j]
        rel = abs(ad - f) / max(abs(ad), abs(f), 1e-30) * 100
        print(f"  d/d{names[j]:>2}: AD {ad:+.4e}  FD {f:+.4e}  "
              f"rel {rel:.2e}%", flush=True)

    print()
    print("Sensitivity matrix (BDM divergence apply, sum((D u)^2)):",
          flush=True)
    for j in range(3):
        ad = float(g_D[j]); f = fd_D[j]
        rel = abs(ad - f) / max(abs(ad), abs(f), 1e-30) * 100
        print(f"  d/d{names[j]:>2}: AD {ad:+.4e}  FD {f:+.4e}  "
              f"rel {rel:.2e}%", flush=True)


if __name__ == "__main__":
    main()
