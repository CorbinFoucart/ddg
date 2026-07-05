"""Kovasznay Re=40 spatial convergence study on BDM_k / P_{k-1}.

This is the headline validation that the order-parametric BDM stack
delivers the theoretical ``h^(k+1)`` convergence rate on a smooth
nonlinear NS solution. Newton finds the discrete solution; the L2
velocity error against the analytic Kovasznay is computed at several
mesh sizes and plotted.

Outputs:
    output/bdm_kovasznay_spatial_convergence.png
    output/bdm_kovasznay_spatial_convergence.json
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
    bdm_k_face_dof_coordinates, _bdm_k_face_quadrature_data,
    _bdm_k_interior_test_functions, _eval_interior_test,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    bdm1_gather, ns_picard_apply_bdm_k_2d,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


RE = 40.0
NU = 1.0 / RE
LAMBDA = RE / 2.0 - np.sqrt(RE ** 2 / 4.0 + 4.0 * np.pi ** 2)
# Square domain [-0.5, 1.5]^2 to capture the full Kovasznay flow.
DOMAIN_X = (-0.5, 1.5)
DOMAIN_Y = (-0.5, 1.5)
DOMAIN_LENGTH = DOMAIN_X[1] - DOMAIN_X[0]


def kovasznay_velocity(x, y):
    ux = 1.0 - jnp.exp(LAMBDA * x) * jnp.cos(2.0 * jnp.pi * y)
    uy = (LAMBDA / (2.0 * jnp.pi)) * jnp.exp(LAMBDA * x) * jnp.sin(2.0 * jnp.pi * y)
    return ux, uy


def bdm_k_canonical_interpolant(V, u_fn):
    """BDM_k canonical interpolant of ``u_fn``: face DOFs from u_fn.n
    at GL points, interior DOFs from reference-coord moments
    int_T u_ref . w_ref dr ds with u_ref = J * DF^{-1} * u_phys (the
    inverse Piola transformation that makes the interior moments
    correspond to the BDM basis coefficient duality)."""
    k = V.order
    K = V.K
    Q = k + 1
    n_face_total = 3 * Q
    n_interior = (k + 1) * (k - 1)

    fdata = _bdm_k_face_quadrature_data(V)
    n_face = fdata['n_face']
    coords = bdm_k_face_dof_coordinates(k)
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
    u_local_face = un.reshape(K, n_face_total)

    u_local_interior = jnp.zeros((K, n_interior))
    if n_interior > 0:
        interior_tests = _bdm_k_interior_test_functions(k)
        r_q, s_q, w_q = triangle_cubature(2 * k + 2)
        DF, _ = _element_jacobian_matrices(V.mesh)
        DF_inv = jnp.linalg.inv(DF)
        J = V.mesh.J[0]
        rqn = np.asarray(r_q); sqn = np.asarray(s_q)
        x_q = Vx0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None] \
              + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
        y_q = Vy0[:, None] + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None] \
              + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
        ux_q, uy_q = u_fn(jnp.asarray(x_q), jnp.asarray(y_q))
        u_phys_q = jnp.stack([ux_q, uy_q], axis=-1)
        u_ref_q = jnp.einsum('k,kab,kqb->kqa', J, DF_inv, u_phys_q)
        for i, w_fn in enumerate(interior_tests):
            w_val = _eval_interior_test(w_fn, r_q, s_q)
            mom = jnp.einsum('q,kqc,qc->k', w_q, u_ref_q, w_val)
            u_local_interior = u_local_interior.at[:, i].set(mom)

    u_local = jnp.concatenate([u_local_face, u_local_interior], axis=1)
    sign = V.topology.local_sign
    owner_mask = sign > 0
    lg = V.topology.local_to_global
    u_global = jnp.zeros(V.total_dofs)
    u_global = u_global.at[lg.reshape(-1)].add(
        jnp.where(owner_mask, u_local, 0).reshape(-1)
    )
    return u_global


def newton_kovasznay(V, max_iters=20, tol=1e-10):
    """Solve steady NS at Re=40 on BDM_k via direct Newton (dense LU
    on the autodiff Jacobian).  Initial guess is the BDM_k canonical
    interpolant of u_kov; this lands close to the discrete solution
    so Newton converges quadratically.

    Returns ``(u_sol, p_sol, history)``.
    """
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    u_bc = bdm1_evaluate_dirichlet_bc(V, kovasznay_velocity)
    bc_mask = bdm1_boundary_dofs_mask(V)
    u = bdm_k_canonical_interpolant(V, kovasznay_velocity)
    # Freeze the non-homog ghost boundary data at the IC (which has u_bc
    # values on boundary DOFs). Critical: closure capture must not let
    # Newton's Jacobian differentiate through u_bc_global as u updates.
    u_bc_frozen = u
    p = jnp.zeros((V.K, dim_p))
    history = []
    n = n_u + n_p
    bc_mask_full = jnp.concatenate(
        [bc_mask, jnp.zeros(n_p, dtype=jnp.bool_)]
    )

    def F_no_mask(zf):
        uz = zf[:n_u]
        pz = zf[n_u:].reshape(V.K, dim_p)
        Fuz, Fpz = ns_picard_apply_bdm_k_2d(
            V, uz, pz, u_star_global=uz, nu=NU,
            pressure_eps=1e-10, tau_scale=4.0,
            u_bc_global=u_bc_frozen,
        )
        return jnp.concatenate([Fuz, Fpz.reshape(-1)])

    for _ in range(max_iters):
        z = jnp.concatenate([u, p.reshape(-1)])
        F = F_no_mask(z)
        F = jnp.where(bc_mask_full, 0.0, F)
        res = float(jnp.linalg.norm(F))
        history.append(res)
        if res < tol:
            break
        J_mat = jax.jacfwd(F_no_mask)(z)
        J_mat = jnp.where(bc_mask_full[:, None], jnp.eye(n), J_mat)
        delta = jnp.linalg.solve(J_mat, -F)
        z = z + delta
        u = jnp.where(bc_mask, u_bc, z[:n_u])
        p = z[n_u:].reshape(V.K, dim_p)
    return u, p, history


def l2_velocity_error(V, u_global):
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 4)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
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
    ux, uy = kovasznay_velocity(jnp.asarray(x_q), jnp.asarray(y_q))
    dx = uh[..., 0] - ux; dy = uh[..., 1] - uy
    return float(jnp.sqrt(jnp.einsum('q,k,kq->', w_q, J, dx ** 2 + dy ** 2)))


def build_mesh(n: int):
    VX, VY, EToV = uniform_rectangle_mesh_2d(
        DOMAIN_X[0], DOMAIN_X[1], DOMAIN_Y[0], DOMAIN_Y[1], n, n,
    )
    return build_mesh_2d(VX, VY, EToV, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=str, default="1,2",
                     help="Comma-separated polynomial orders (k).")
    ap.add_argument("--meshes-k1", type=str, default="4,6,8,12,16")
    ap.add_argument("--meshes-k2", type=str, default="3,4,6,8")
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    orders = [int(s) for s in args.orders.split(",")]
    meshes = {
        1: [int(s) for s in args.meshes_k1.split(",")],
        2: [int(s) for s in args.meshes_k2.split(",")],
    }

    summary = {"Re": RE, "domain": [DOMAIN_X, DOMAIN_Y], "results": {}}
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {1: "C0", 2: "C1", 3: "C2"}
    for k in orders:
        ns = meshes.get(k, [3, 4, 6])
        print(f"\n=== k = {k} ===", flush=True)
        hs, errs, iters = [], [], []
        for n in ns:
            mesh = build_mesh(n)
            V = BDM.from_mesh(mesh, order=k)
            t0 = time.time()
            u_sol, p_sol, history = newton_kovasznay(V, max_iters=20, tol=1e-10)
            wall = time.time() - t0
            err = l2_velocity_error(V, u_sol)
            h = DOMAIN_LENGTH / n
            hs.append(h); errs.append(err); iters.append(len(history))
            print(f"  n={n}: h={h:.4f}, iters={len(history)}, "
                  f"final_res={history[-1]:.2e}, L2 err = {err:.3e}, "
                  f"{wall:.1f}s", flush=True)
        hs = np.array(hs); errs = np.array(errs)
        # Slopes.
        slopes = (np.log(errs[:-1]) - np.log(errs[1:])) / \
                 (np.log(hs[:-1]) - np.log(hs[1:]))
        for i, sl in enumerate(slopes):
            print(f"  slope({hs[i]:.4f} -> {hs[i+1]:.4f}) = {sl:.3f}", flush=True)
        summary["results"][f"k={k}"] = {
            "hs": hs.tolist(),
            "errs": errs.tolist(),
            "iters": iters,
            "slopes": slopes.tolist(),
            "asymptotic_slope": float(slopes[-1]),
            "expected_slope": k + 1,
        }
        # Plot data.
        ax.loglog(hs, errs, "o-", color=colors.get(k, "k"),
                   label=f"BDM_{k} (asymp slope {slopes[-1]:.2f}, theory {k+1})")
        # Reference slope.
        ref = errs[0] * (hs / hs[0]) ** (k + 1)
        ax.loglog(hs, ref, "--", color=colors.get(k, "k"), alpha=0.4,
                   label=f"slope {k+1} reference")
    ax.set_xlabel("h (= domain_length / n)")
    ax.set_ylabel(r"$\|u_h - u_{\rm Kov}\|_{L^2}$")
    ax.set_title(f"BDM_k / P_{{k-1}} Kovasznay Re={RE:.0f} spatial convergence "
                  f"(Newton + dense LU)")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = args.out_dir / "bdm_kovasznay_spatial_convergence.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png_path}", flush=True)
    json_path = args.out_dir / "bdm_kovasznay_spatial_convergence.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
