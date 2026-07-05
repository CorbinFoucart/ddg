"""Cylinder Y-position inverse via TRUE jax.grad through the
differentiable mesh (no finite differences).

Sibling of ``scripts/inverse_obstacle_radius_2d.py`` --- same probe
rake, same loss, same Adam loop, but the gradient on ``cy`` flows
through ``ddg.dg2d.differentiable_mesh.differentiable_mesh_with_circle``
into the BDM forward solve, end-to-end traceable.

This is the demo unblocked by the JAX-traceable mesh + curving work in
``src/ddg/dg2d/differentiable_mesh.py``: in the FD version, the
mesh-build/recompile cost is 3 forward calls per Adam iter; here it
is one forward + one reverse pass, the same as the inverse-nu and
inflow-control demos.

Setup
-----
* BDM_2 cylinder smoke (Re=100, K=432). The mesh is built ONCE at the
  nominal cy; perturbations to cy are applied via the JAX curving.
* 10-step BDF2 trajectory from rest with two Newton iters per step,
  ν = 10^-3, parabolic inflow.
* Six probes at (CX + iD, CY_truth), i = 1..6.
* Loss: ||y_pred - y_obs||^2; ``jax.grad(loss)(cy)`` flows through
  mesh + BDM_k forward.

Cost
----
Single-parameter gradient via reverse-mode AD: ~5 min/iter steady
(forward compile ~ 80 s; subsequent iters reuse the JIT cache because
the mesh-topology static data is invariant). Compared to the FD
version's 15 min/iter, this is a ~3x speedup at the same Adam budget.
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
from ddg.dg2d.bdm import (
    bdm_k_basis_at_points, bdm1_gather, _element_jacobian_matrices,
)


def _locate_probes(mesh, probe_xy):
    """Host-side: find element + reference coords for each probe. Returns
    ``(elem_idx_np, r_ref_np, s_ref_np)``."""
    VX = np.asarray(mesh.VX); VY = np.asarray(mesh.VY)
    EToV = np.asarray(mesh.EToV)
    n_probe = probe_xy.shape[0]
    elem_idx = np.zeros(n_probe, dtype=int)
    rs_ref = np.zeros((n_probe, 2), dtype=float)
    for p in range(n_probe):
        xp, yp = probe_xy[p]
        for K_idx in range(mesh.K):
            v0, v1, v2 = EToV[K_idx]
            x0, y0 = VX[v0], VY[v0]
            x1, y1 = VX[v1], VY[v1]
            x2, y2 = VX[v2], VY[v2]
            denom = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(denom) < 1e-14:
                continue
            r_aff = ((xp - x0) * (y2 - y0) - (yp - y0) * (x2 - x0)) / denom
            s_aff = (-(xp - x0) * (y1 - y0) + (yp - y0) * (x1 - x0)) / denom
            if r_aff > -1e-9 and s_aff > -1e-9 and r_aff + s_aff < 1 + 1e-9:
                elem_idx[p] = K_idx
                rs_ref[p, 0] = 2.0 * r_aff - 1.0
                rs_ref[p, 1] = 2.0 * s_aff - 1.0
                break
    return elem_idx, rs_ref[:, 0], rs_ref[:, 1]


def _make_probe_eval(V, elem_idx_np, r_ref_np, s_ref_np):
    """JAX-traceable probe evaluation given pre-located element +
    reference coords for each probe."""
    k = V.order
    elem_idx_j = jnp.asarray(elem_idx_np)
    r_j = jnp.asarray(r_ref_np)
    s_j = jnp.asarray(s_ref_np)
    phi_ref = bdm_k_basis_at_points(k, r_j, s_j)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    DF_probe = DF[elem_idx_j]
    J_probe = J[elem_idx_j]

    def probes(u_global):
        u_local = bdm1_gather(V.topology, u_global)
        u_at_probes = u_local[elem_idx_j]
        phi_phys = jnp.einsum(
            "pcb,plb->plc", DF_probe, phi_ref) / J_probe[:, None, None]
        return jnp.einsum("pl,plc->pc", u_at_probes, phi_phys)
    return probes
from ddg.dg2d.bdm import (
    BDM, bdm_k_evaluate_boundary_velocity, refresh_piola_scale,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
)
from ddg.dg2d.differentiable_mesh import (
    make_template, make_curving_info, differentiable_mesh_with_circle,
)
from ddg.dg2d.mesh import build_mesh_2d, channel_with_circle_mesh_2d


# Geometry constants.
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 0.5
CX = 0.2
RADIUS = 0.05
D_CYL = 2.0 * RADIUS
U_MAX = 1.5
U_MEAN = (2.0 / 3.0) * U_MAX
NU = 1.0e-3
H_CHANNEL = Y_MAX - Y_MIN


def _parabolic_inflow(y):
    return 4.0 * U_MAX * y * (H_CHANNEL - y) / H_CHANNEL**2


def _bc_fn(x, y):
    on_inflow = x < X_MIN + 1e-6
    u_x = jnp.where(on_inflow, _parabolic_inflow(y), 0.0)
    u_y = jnp.zeros_like(y)
    return u_x, u_y


def build_diff_setup(kx, ky, order, cy_nominal,
                      *, n_radial=2, r_outer=0.1, radius=RADIUS):
    """Build the template + curving info + BC classification ONCE.

    The mesh topology and BC mask depend on ``cy_nominal``; the JAX
    forward then re-curves the body-fit ring at any ``cy_eval`` close
    to ``cy_nominal``.
    """
    VX, VY, EToV = channel_with_circle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, kx, ky,
        cx=CX, cy=cy_nominal, radius=radius,
        R_outer=r_outer, n_radial=n_radial,
        radial_growth=1.3,
    )
    mesh = build_mesh_2d(VX, VY, EToV, order)
    template = make_template(mesh)
    curving = make_curving_info(
        mesh, cx=CX, cy=cy_nominal, radius=radius, detection_tol=0.05,
    )

    # Build the curved mesh ONCE at nominal cy to do boundary
    # classification + BC sampling AND build the BDM topology, which
    # we then cache for reuse inside the JIT'd forward solve.
    mesh_curved_nominal = differentiable_mesh_with_circle(
        template.VX_nominal, template.VY_nominal, template, curving,
        jnp.asarray(CX), jnp.asarray(cy_nominal), jnp.asarray(radius),
    )
    V_nominal = BDM.from_mesh(mesh_curved_nominal, order=order)
    cached_topology = V_nominal.topology

    Nfaces = 3
    Fmask = np.asarray(mesh_curved_nominal.Fmask)
    x_np = np.asarray(mesh_curved_nominal.x)
    y_np = np.asarray(mesh_curved_nominal.y)
    x_face = np.zeros((Nfaces, mesh.K)); y_face = np.zeros((Nfaces, mesh.K))
    for f in range(Nfaces):
        x_face[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE_np = np.asarray(mesh.EToE)
    is_bdy = (EToE_np == np.arange(mesh.K)[:, None]).T
    is_inflow_face = is_bdy & (x_face < X_MIN + 1e-3)
    is_outflow_face = is_bdy & (x_face > X_MAX - 1e-3)
    is_outer_y = is_bdy & (
        (y_face < Y_MIN + 1e-3) | (y_face > Y_MAX - 1e-3)
    )
    r_from_cyl = np.sqrt((x_face - CX) ** 2
                         + (y_face - cy_nominal) ** 2)
    is_cyl_face = (
        is_bdy & (r_from_cyl < radius * 1.05)
        & (~is_inflow_face) & (~is_outflow_face) & (~is_outer_y)
    )
    is_wall_face = is_outer_y
    is_dirichlet = jnp.asarray(
        (is_inflow_face | is_wall_face | is_cyl_face).T
    )
    is_natural_face = jnp.asarray(is_outflow_face.T)
    bc_mask = bdm1_boundary_dofs_mask(V_nominal, dirichlet_face=is_dirichlet)
    return template, curving, cached_topology, bc_mask, is_natural_face


def make_loss_fn(template, curving, cached_topology,
                  bc_mask, is_natural_face, probe_locator, *,
                 order, n_newton, n_steps, dt, tau_scale, nu,
                 probe_xy, y_obs):
    """Return a JAX-traceable loss(cy) that flows through mesh +
    forward solve. Closes over the host-static template + curving
    info + BC mask."""
    n_steps_const = int(n_steps)

    def loss_for_cy(cy):
        mesh_diff = differentiable_mesh_with_circle(
            template.VX_nominal, template.VY_nominal,
            template, curving,
            jnp.asarray(CX), cy, jnp.asarray(RADIUS),
        )
        # Use cached integer topology, refresh only piola_scale --
        # avoids the np.asarray(mesh.EToE) calls in build_bdm_topology
        # that fail under jax.grad tracing.
        topo = refresh_piola_scale(cached_topology, mesh_diff)
        V = BDM(mesh=mesh_diff, topology=topo, order=order)
        u_bc = bdm1_evaluate_dirichlet_bc(V, _bc_fn)
        u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _bc_fn)
        step_fn = make_step_nu_param(
            V, u_bc, u_bc_vec, bc_mask, is_natural_face,
            tau_scale=tau_scale, n_newton=n_newton,
        )
        probe_fn = _make_probe_eval(
            V, probe_locator[0], probe_locator[1], probe_locator[2])

        u_n = jnp.zeros(V.total_dofs)
        u_nm1 = jnp.zeros(V.total_dofs)
        traces = []
        for s in range(n_steps_const):
            order_eff = 1 if s == 0 else 2
            u_new, _ = step_fn(u_n, u_nm1, dt, nu, order_eff)
            traces.append(probe_fn(u_new))
            u_nm1, u_n = u_n, u_new
        y_pred = jnp.stack(traces)
        return jnp.sum((y_pred - y_obs) ** 2)
    return loss_for_cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--cy-true", type=float, default=0.25)
    ap.add_argument("--cy-init", type=float, default=0.225)
    ap.add_argument("--cy-nominal", type=float, default=0.25,
                    help="cy at which the template mesh is built; "
                         "all FD/AD probes are perturbations from "
                         "this value via the JAX curving.")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--n-train", type=int, default=15)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    print(f"BDM_{args.order} cylinder smoke (differentiable mesh): "
          f"n_steps={args.n_steps}, dt={args.dt}", flush=True)
    print(f"  template at cy = {args.cy_nominal}", flush=True)
    print(f"  truth CY = {args.cy_true}, init CY = {args.cy_init}",
          flush=True)
    print(f"  lr = {args.lr}", flush=True)

    template, curving, cached_topology, bc_mask, is_natural_face = (
        build_diff_setup(
            args.kx, args.ky, args.order, args.cy_nominal,
        )
    )

    probe_xy = np.array(
        [[CX + i * D_CYL, args.cy_true] for i in (1, 2, 3, 4, 5, 6)],
        dtype=float,
    )

    # Locate probes ONCE on the nominal mesh (host-side). The probe ->
    # element + reference coordinate mapping is static; only the
    # geometric quantities (DF, J) at those probe reference coords
    # change with cy.
    nominal_mesh = differentiable_mesh_with_circle(
        template.VX_nominal, template.VY_nominal, template, curving,
        jnp.asarray(CX), jnp.asarray(args.cy_nominal), jnp.asarray(RADIUS),
    )
    probe_locator = _locate_probes(nominal_mesh, probe_xy)

    # --- ground truth: forward at cy_true ---
    print("generating ground truth ...", flush=True)
    t0 = time.time()
    # Build a fresh loss_fn pinned to a dummy y_obs to JIT-compile the
    # forward only.
    dummy_y = jnp.zeros((args.n_steps, len(probe_xy), 2))
    # Trajectory function that returns the probe traces (not the loss).
    def trajectory(cy):
        mesh_diff = differentiable_mesh_with_circle(
            template.VX_nominal, template.VY_nominal,
            template, curving,
            jnp.asarray(CX), cy, jnp.asarray(RADIUS),
        )
        topo = refresh_piola_scale(cached_topology, mesh_diff)
        V = BDM(mesh=mesh_diff, topology=topo, order=args.order)
        u_bc = bdm1_evaluate_dirichlet_bc(V, _bc_fn)
        u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _bc_fn)
        step_fn = make_step_nu_param(
            V, u_bc, u_bc_vec, bc_mask, is_natural_face,
            tau_scale=args.tau_scale, n_newton=args.n_newton,
        )
        probe_fn = _make_probe_eval(
            V, probe_locator[0], probe_locator[1], probe_locator[2])
        u_n = jnp.zeros(V.total_dofs); u_nm1 = jnp.zeros(V.total_dofs)
        traces = []
        for s in range(args.n_steps):
            order_eff = 1 if s == 0 else 2
            u_new, _ = step_fn(u_n, u_nm1, args.dt, NU, order_eff)
            traces.append(probe_fn(u_new))
            u_nm1, u_n = u_n, u_new
        return jnp.stack(traces)

    traj_jit = jax.jit(trajectory)
    y_obs = traj_jit(jnp.asarray(args.cy_true))
    y_obs.block_until_ready()
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    # --- Adam on cy via jax.grad ---
    loss_fn = make_loss_fn(
        template, curving, cached_topology,
        bc_mask, is_natural_face, probe_locator,
        order=args.order, n_newton=args.n_newton, n_steps=args.n_steps,
        dt=args.dt, tau_scale=args.tau_scale, nu=NU,
        probe_xy=probe_xy, y_obs=y_obs,
    )
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    cy = jnp.asarray(float(args.cy_init))
    L0 = float(loss_jit(cy))
    print(f"  initial loss = {L0:.4e}", flush=True)

    m_t = jnp.zeros_like(cy); v_t = jnp.zeros_like(cy)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_loss = L0; best_cy = float(cy); best_iter = 0
    history = [dict(iter=0, cy=float(cy), loss=L0, grad=0.0)]
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(cy)
        m_t = beta1 * m_t + (1 - beta1) * g
        v_t = beta2 * v_t + (1 - beta2) * g * g
        m_hat = m_t / (1 - beta1 ** it)
        v_hat = v_t / (1 - beta2 ** it)
        cy = cy - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        L = float(loss_jit(cy))
        cy_val = float(cy)
        if L < best_loss:
            best_loss = L; best_cy = cy_val; best_iter = it
        rel = abs(cy_val - args.cy_true) / args.cy_true
        print(f"  iter {it:3d}: CY = {cy_val:.4f}  "
              f"(rel err {rel*100:5.2f}%)  loss={L:.3e}  g={float(g):+.3e}",
              flush=True)
        history.append(dict(iter=int(it), cy=cy_val,
                            loss=L, grad=float(g)))
    print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)
    print(f"\nlast-iter CY = {float(cy):.4f}  (truth {args.cy_true})",
          flush=True)
    print(f"best-loss CY = {best_cy:.4f} at iter {best_iter}",
          flush=True)
    cy_final = best_cy

    stem = f"inverse_obstacle_cy_diff_cylinder_k{args.order}"
    out_npz = args.out_dir / f"{stem}.npz"
    np.savez_compressed(
        out_npz,
        cy_true=args.cy_true, cy_init=args.cy_init,
        cy_nominal=args.cy_nominal,
        cy_final=cy_final, n_steps=args.n_steps, dt=args.dt,
        times=(np.arange(args.n_steps) + 1) * args.dt,
        probe_xy=probe_xy,
        y_obs=np.asarray(y_obs),
        cy_hist=np.array([r["cy"] for r in history]),
        loss_hist=np.array([r["loss"] for r in history]),
    )
    print(f"  wrote {out_npz}", flush=True)
    out_json = args.out_dir / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky,
            cy_true=args.cy_true, cy_init=args.cy_init,
            cy_nominal=args.cy_nominal, cy_final=cy_final,
            n_steps=args.n_steps, dt=args.dt,
            n_newton=args.n_newton, tau_scale=args.tau_scale,
            lr=args.lr, n_train=args.n_train,
            history=history,
        ), f, indent=2)
    print(f"  wrote {out_json}", flush=True)

    iters = [r["iter"] for r in history]
    cys = [r["cy"] for r in history]
    losses = [r["loss"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(iters, cys, "b-o", ms=4, label="recovered")
    axes[0].axhline(args.cy_true, color="r", ls="--", lw=1, label="truth")
    axes[0].axhline(args.cy_init, color="gray", ls=":", lw=1, label="init")
    axes[0].set_xlabel("Adam iter"); axes[0].set_ylabel(r"$C_y$")
    axes[0].set_title(
        "Cylinder Y-position via differentiable mesh\n"
        "(jax.grad through mesh build, no FD)"
    )
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].semilogy(iters, losses, "k-")
    axes[1].set_xlabel("Adam iter"); axes[1].set_ylabel("loss")
    axes[1].set_title("L2 probe-trace mismatch")
    axes[1].grid(alpha=0.3, which="both")
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
