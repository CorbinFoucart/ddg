"""BDM_k rising bubble (Boussinesq, transient) -- smoke test of the
high-order machinery on bubble physics.

A light blob in a closed [0, 1]^2 box rises under gravity, with the
density carried as a Boussinesq scalar (Lagrangian DG, advected by the
BDM velocity sampled at scalar nodes). No-slip walls (homogeneous
Dirichlet) are enforced weakly through the SIPG ghost. BDF1 bootstrap +
BDF2 thereafter.

This is a uniform-mesh smoke of the BDM transient machinery (mass term
+ body force + Boussinesq coupling). The modular linear-solver pattern
from ``ns_bdm_cavity_amr_2d`` carries over (``--solver dense`` is the
default; ``--solver iterative`` uses the coupled hp-MG PC). AMR and
deeper validation are follow-ups.
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

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.bdm import (
    BDM, bdm_k_basis_at_points, bdm_k_viscous_dirichlet_lift,
    _element_jacobian_matrices, bdm1_gather, bdm1_scatter_add,
    bdm_k_evaluate_boundary_velocity,
    bdm_k_canonical_interpolant, bdm_k_evaluate_at_points,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    ns_picard_apply_bdm_k_2d, BDF_COEFFS, bdf_history_k,
)
from ddg.dg2d.refinement import RefinementForest, kelly_error_indicator
from ddg.dg2d.bdm_multigrid import make_coupled_velocity_mg_pc
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.scalar_transport import (
    build_scalar_operators, scalar_advection_2d, scalar_transport_step_2d,
)
from ddg.dg2d.incompressible_ns import BDF1, BDF2
from ddg.dg2d.sipg import BC_NEUMANN
from ddg.dg2d.reference import vandermonde_2d


X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
GRAVITY_Y = 1.0   # buoyancy magnitude; c=1 inside the blob -> upward force


def _walls_bc(x, y):
    """No-slip closed box: ``u = 0`` on every wall."""
    return jnp.zeros_like(x), jnp.zeros_like(y)


def _build_space(mesh, order):
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _walls_bc)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _walls_bc)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


def _bdm_velocity_at_nodes(V, u_global):
    """Sample the BDM velocity at the mesh's Lagrangian nodal points.

    Returns ``(ux, uy)`` with shape ``(Np, K)``. Used by the Lagrangian
    scalar transport, which expects a nodal velocity field. ``Np`` and
    the nodal pattern are those of ``V.mesh``."""
    k = V.order
    r, s = V.mesh.r, V.mesh.s
    phi_ref = bdm_k_basis_at_points(k, r, s)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum(
        "kca,nla->knlc", DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    u_nodes = jnp.einsum("kl,knlc->knc", u_local, phi_phys)   # (K, Np, 2)
    return u_nodes[..., 0].T, u_nodes[..., 1].T               # (Np, K) each


def _scalar_at_bdm_cubature(mesh, c_nodal, k_order):
    """Interpolate Lagrangian-nodal scalar ``c`` to the BDM cubature
    points of the same mesh (degree ``2*k_order+2``). Returns ``(Q, K)``."""
    r_q, s_q, _w = triangle_cubature(2 * k_order + 2)
    V_q = vandermonde_2d(mesh.N, r_q, s_q)
    return V_q @ mesh.invV @ c_nodal                           # (Q, K)


def _buoyancy_body_force(V, c_nodal):
    """Boussinesq buoyancy: ``f = (0, GRAVITY_Y * c)`` (lighter blob, c=1,
    feels an upward force). Assemble as a BDM velocity-DOF RHS by
    integrating ``f . phi_bdm`` per element and scattering with the
    sign*piola weights -- same pattern as :func:`build_load` in the
    BDM Stokes spatial app."""
    k = V.order
    mesh = V.mesh
    r_q, s_q, w_q = triangle_cubature(2 * k + 2)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)               # (Q, Nbdm, 2)
    DF, _ = _element_jacobian_matrices(mesh)
    J = mesh.J[0]
    phi_phys = jnp.einsum(
        "kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]  # (K, Q, Nbdm, 2)
    c_q = _scalar_at_bdm_cubature(mesh, c_nodal, k)            # (Q, K)
    fy_q = GRAVITY_Y * c_q                                     # (Q, K)
    # ``f_local[k, ell] = sum_q w_q J_k f_y(q, k) phi_phys[k, q, ell, 1]``.
    f_local = jnp.einsum(
        "q,k,qk,kql->kl", w_q, J, fy_q, phi_phys[..., 1])
    return bdm1_scatter_add(V.topology, f_local)


# ---------------------------------------------------------------------------
# Momentum step (BDM_k Newton with mass term + body force, dense LU)
# ---------------------------------------------------------------------------

def _newton_momentum_step(V, *, nu, tau_scale, bc_mask,
                          velocity_alpha, f_velocity,
                          u_prev, p_prev, max_iters=12, tol=1e-7,
                          coupled_pc=None,
                          gmres_tol=1e-9, gmres_restart=200,
                          gmres_maxiter=40):
    """One Newton solve of the transient BDM saddle point. Boundary
    normal traces are pinned to zero (walls); the convective term is
    full nonlinear (``u_star = u``) so the JVP gives full Newton.

    ``coupled_pc=None`` -> dense LU (jacfwd + constrained solve);
    ``coupled_pc != None`` -> matrix-free JVP + right-preconditioned GMRES
    with the supplied block-diagonal PC (velocity hp-MG + pressure Schur,
    built once outside the time loop)."""
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n = n_u + n_p

    def raw_flat(x):
        u = x[:n_u]; p = x[n_u:].reshape(V.K, dim_p)
        Fu, Fp = ns_picard_apply_bdm_k_2d(
            V, u, p, u_star_global=u, nu=nu,
            pressure_eps=1e-10, tau_scale=tau_scale,
            velocity_alpha=velocity_alpha, f_velocity=f_velocity,
        )
        return jnp.concatenate([Fu, Fp.reshape(-1)])

    mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])

    if coupled_pc is None:
        jac = jax.jacfwd(raw_flat)
        I_n = jnp.eye(n)
        def linear_solve(x_k, rhs):
            J = jac(x_k)
            J_con = jnp.where(mask_full[:, None], I_n,
                              jnp.where(mask_full[None, :], 0.0, J))
            return jnp.linalg.solve(J_con, rhs)
    else:
        def linear_solve(x_k, rhs):
            def matvec(v):
                vu = v[:n_u]; vp = v[n_u:]
                vu_z = jnp.where(bc_mask, 0.0, vu)
                _, jv = jax.jvp(
                    raw_flat, (x_k,),
                    (jnp.concatenate([vu_z, vp]),))
                out_u = jnp.where(bc_mask, vu, jv[:n_u])
                return jnp.concatenate([out_u, jv[n_u:]])
            d, _ = jax.scipy.sparse.linalg.gmres(
                matvec, rhs, x0=jnp.zeros_like(rhs), M=coupled_pc,
                tol=gmres_tol, atol=1e-12,
                restart=gmres_restart, maxiter=gmres_maxiter,
            )
            return d

    u_k = jnp.where(bc_mask, 0.0, u_prev)
    p_k = p_prev
    x_k = jnp.concatenate([u_k, p_k.reshape(-1)])

    norm_F0 = None
    converged = False
    for it in range(max_iters):
        raw = raw_flat(x_k)
        Fu_inner = jnp.where(bc_mask, 0.0, raw[:n_u])
        norm_F = float(jnp.linalg.norm(Fu_inner)
                       + jnp.linalg.norm(raw[n_u:]))
        if norm_F0 is None:
            norm_F0 = norm_F
        if norm_F <= tol * norm_F0 + 1e-12:
            converged = True
            break
        rhs = -jnp.concatenate([Fu_inner, raw[n_u:]])
        delta = linear_solve(x_k, rhs)
        x_k = x_k + delta
        x_k = x_k.at[:n_u].set(jnp.where(bc_mask, 0.0, x_k[:n_u]))

    return x_k[:n_u], x_k[n_u:].reshape(V.K, dim_p), converged, it + 1


# ---------------------------------------------------------------------------
# Initial condition: Gaussian-blob density
# ---------------------------------------------------------------------------

def _initial_bubble(mesh, *, centre=(0.5, 0.25), radius=0.15):
    """Smooth bubble: ``c = exp(-(r/sigma)^2)`` evaluated at mesh nodes
    -- a light Gaussian blob with sigma = radius / sqrt(2)."""
    sigma = radius / np.sqrt(2.0)
    x = np.asarray(mesh.x); y = np.asarray(mesh.y)           # (Np, K)
    r2 = (x - centre[0]) ** 2 + (y - centre[1]) ** 2
    return jnp.asarray(np.exp(-r2 / sigma ** 2))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=8)
    ap.add_argument("--nu", type=float, default=1.0e-2)
    ap.add_argument("--kappa", type=float, default=1.0e-3,
                    help="scalar diffusivity")
    ap.add_argument("--dt", type=float, default=2.0e-2)
    ap.add_argument("--tfinal", type=float, default=0.5)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--snap-every", type=int, default=5,
                    help="save a frame every N steps")
    ap.add_argument("--adapt-every", type=int, default=0,
                    help="AMR adaptation cadence in steps (0 disables AMR)")
    ap.add_argument("--theta", type=float, default=0.5,
                    help="Dorfler bulk-marking fraction (AMR)")
    ap.add_argument("--max-level", type=int, default=4,
                    help="AMR refinement-level cap")
    ap.add_argument("--solver", choices=("dense", "iterative"),
                    default="dense",
                    help="dense LU (small) or matrix-free GMRES + hp-MG PC")
    ap.add_argument("--mg-base-kx", type=int, default=2,
                    help="base mesh size for the hp-MG hierarchy (iterative)")
    ap.add_argument("--mg-n-h", type=int, default=0,
                    help="number of h-levels in the hp-MG hierarchy")
    ap.add_argument("--mg-sparse-fine", action="store_true",
                    help="sparse matrix-free finest-level assembly (large meshes)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    print(f"BDM_{args.order} rising bubble: {args.kx}x{args.kx}, "
          f"nu={args.nu}, kappa={args.kappa}, dt={args.dt}, "
          f"T={args.tfinal}"
          + (f", AMR every {args.adapt_every} steps"
             if args.adapt_every > 0 else ", uniform mesh"), flush=True)
    VX, VY, EToV = uniform_rectangle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, args.kx, args.kx)
    base_mesh = build_mesh_2d(VX, VY, EToV, args.order)
    forest = (RefinementForest(base_mesh, max_level=args.max_level)
              if args.adapt_every > 0 else None)
    if forest is not None:
        mesh, a2l = forest.active_mesh()
    else:
        mesh, a2l = base_mesh, None
    V, u_bc, u_bc_vec, bc_mask = _build_space(mesh, args.order)
    dim_p = args.order * (args.order + 1) // 2
    n_u, n_p = V.total_dofs, V.K * dim_p
    print(f"  DOFs: {n_u + n_p} ({n_u} u + {n_p} p) on K={V.K} elements",
          flush=True)

    # Scalar operators: Helmholtz factor for the implicit diffusion solve.
    # All-Neumann no-flux walls (BC_NEUMANN on every face). Rebuilt after
    # each adaptation since the mesh changes.
    def _build_scalar_ops(mesh):
        bc = jnp.full((mesh.K, 3), BC_NEUMANN, dtype=jnp.int32)
        return (build_scalar_operators(
                    mesh, dt=args.dt, kappa=args.kappa, g0=BDF1.g0,
                    diffusion_bc_types=bc, tau_scale=200.0),
                build_scalar_operators(
                    mesh, dt=args.dt, kappa=args.kappa, g0=BDF2.g0,
                    diffusion_bc_types=bc, tau_scale=200.0))
    ops_c_bdf1, ops_c_bdf2 = _build_scalar_ops(mesh)

    # Coupled hp-MG preconditioner for the iterative solver. Built once
    # per BDF order (velocity_alpha = gamma0/dt differs BDF1 vs BDF2). The
    # PC matches the actual transient velocity block ``alpha*M + nu*A`` --
    # a viscous-only PC is useless at small dt where alpha*M dominates.
    # ``--mg-n-h 0`` uses the current solve mesh as the MG finest level
    # (no extra h-coarsening below it); the BDM_3 vertex-patch smoother
    # + W-cycle on the BDM_2/BDM_1 p-coarser levels handles the rest.
    def _build_coupled_pcs(V_cur, mesh_cur):
        if args.mg_n_h != 0:
            raise NotImplementedError(
                "--mg-n-h>0 (forest-refined hp-MG hierarchy) is not wired "
                "through the bubble app yet -- the bubble solve mesh is "
                "either uniform or adaptively forest-refined, neither of "
                "which matches a simple base+n_h tower. Use mg-n-h=0.")
        a1 = BDF_COEFFS[1]["gamma0"] / args.dt
        a2 = BDF_COEFFS[2]["gamma0"] / args.dt
        return (
            make_coupled_velocity_mg_pc(
                V_cur, mesh_cur, n_h=0, nu=args.nu,
                tau_scale=args.tau_scale, nsm=2, cycle="W",
                sparse_fine=args.mg_sparse_fine, velocity_alpha=a1),
            make_coupled_velocity_mg_pc(
                V_cur, mesh_cur, n_h=0, nu=args.nu,
                tau_scale=args.tau_scale, nsm=2, cycle="W",
                sparse_fine=args.mg_sparse_fine, velocity_alpha=a2),
        )
    pc_bdf1 = pc_bdf2 = None
    if args.solver == "iterative":
        pc_bdf1, pc_bdf2 = _build_coupled_pcs(V, mesh)
        print(f"  built coupled hp-MG PCs "
              f"(BDF1 alpha={BDF_COEFFS[1]['gamma0']/args.dt:.1f}, "
              f"BDF2 alpha={BDF_COEFFS[2]['gamma0']/args.dt:.1f})", flush=True)

    # Initial state.
    u_state = jnp.zeros(n_u)
    p_state = jnp.zeros((V.K, dim_p))
    c_state = _initial_bubble(mesh)                            # (Np, K)
    c_prev = c_state                                            # for BDF2 scalar
    Nc_state = jnp.zeros_like(c_state)
    u_hist = [u_state, u_state]                                 # BDF velocity history (n, n-1)
    snapshots = [{"t": 0.0,
                  "x": np.asarray(mesh.x), "y": np.asarray(mesh.y),
                  "c": np.asarray(c_state),
                  "u_mag": np.zeros_like(np.asarray(c_state))}]

    n_steps = int(np.ceil(args.tfinal / args.dt))
    t = 0.0
    t0 = time.time()
    for step in range(n_steps):
        # BDF1 bootstrap on step 0, BDF2 thereafter.
        order_eff = 1 if step == 0 else 2
        ops_c = ops_c_bdf1 if order_eff == 1 else ops_c_bdf2
        bdf_s = BDF1 if order_eff == 1 else BDF2
        coeffs_v = BDF_COEFFS[order_eff]
        velocity_alpha = coeffs_v["gamma0"] / args.dt

        # --- scalar transport: advect c by the current BDM velocity ---
        ux_nodes, uy_nodes = _bdm_velocity_at_nodes(V, u_state)
        c_new, Nc_state = scalar_transport_step_2d(
            mesh, c_state, c_prev, ux_nodes, uy_nodes, Nc_state,
            dt=args.dt, kappa=args.kappa, bdf=bdf_s, ops=ops_c)
        c_prev = c_state                                        # roll
        c_state = c_new

        # --- momentum step ---
        f_buoy = _buoyancy_body_force(V, c_state)              # (n_u,)
        f_hist_u = bdf_history_k(V, u_hist, dt=args.dt, order=order_eff)
        f_velocity = f_hist_u + f_buoy
        pc_this = (pc_bdf1 if order_eff == 1 else pc_bdf2)
        u_new, p_new, conv, n_iters = _newton_momentum_step(
            V, nu=args.nu, tau_scale=args.tau_scale, bc_mask=bc_mask,
            velocity_alpha=velocity_alpha, f_velocity=f_velocity,
            u_prev=u_state, p_prev=p_state, coupled_pc=pc_this)
        u_hist = [u_new, u_hist[0]]                            # roll: (n+1, n)
        u_state, p_state = u_new, p_new
        t += args.dt

        # --- AMR adaptation pass (every adapt-every steps) ---------
        if (forest is not None and args.adapt_every > 0
                and (step + 1) % args.adapt_every == 0
                and step < n_steps - 1):
            eta = kelly_error_indicator(mesh, c_state)
            new_mesh, fields_new, a2l, n_ref = forest.adapt(
                mesh, a2l,
                {"c": c_state, "c_prev": c_prev, "Nc": Nc_state},
                eta, theta=args.theta)
            if n_ref > 0:
                # Transfer BDM velocity (and BDF history) to the new
                # space via canonical interpolation of the *old* field.
                V_old = V
                u_old = u_state; u_hist_old = u_hist
                V_new, u_bc, u_bc_vec, bc_mask = _build_space(
                    new_mesh, args.order)
                def _interp(V_o, u_o):
                    return bdm_k_canonical_interpolant(
                        V_new,
                        lambda x, y: bdm_k_evaluate_at_points(V_o, u_o, x, y),
                    )
                u_state = _interp(V_old, u_old)
                u_hist = [_interp(V_old, uh) for uh in u_hist_old]
                # Pin boundary normal-trace DOFs on the new space.
                u_state = jnp.where(bc_mask, 0.0, u_state)
                u_hist = [jnp.where(bc_mask, 0.0, uh) for uh in u_hist]
                # Pressure: simplest -- restart at zero on new mesh.
                p_state = jnp.zeros((V_new.K, dim_p))
                V = V_new
                mesh = new_mesh
                c_state = fields_new["c"]
                c_prev = fields_new["c_prev"]
                Nc_state = fields_new["Nc"]
                ops_c_bdf1, ops_c_bdf2 = _build_scalar_ops(mesh)
                if args.solver == "iterative":
                    pc_bdf1, pc_bdf2 = _build_coupled_pcs(V, mesh)
                n_u = V.total_dofs
                print(f"  [adapt] step {step+1}: refined {n_ref} leaves, "
                      f"K={V.K}, DOFs={V.total_dofs + V.K * dim_p}",
                      flush=True)

        if (step + 1) % args.snap_every == 0 or step == n_steps - 1:
            ux_n, uy_n = _bdm_velocity_at_nodes(V, u_state)
            umag = np.sqrt(np.asarray(ux_n) ** 2 + np.asarray(uy_n) ** 2)
            snapshots.append({"t": float(t),
                              "x": np.asarray(mesh.x),
                              "y": np.asarray(mesh.y),
                              "c": np.asarray(c_state),
                              "u_mag": umag})
            print(f"  step {step+1}/{n_steps}: t={t:.3f}, "
                  f"newton_iters={n_iters}, conv={conv}, "
                  f"||u||={float(jnp.linalg.norm(u_state)):.3e}, "
                  f"c_max={float(jnp.max(c_state)):.3f}",
                  flush=True)
    print(f"total {time.time()-t0:.1f}s", flush=True)

    # ----- output: a final-frame field + a sequence montage ----------
    stem = f"bdm_bubble_k{args.order}_kx{args.kx}_dt{args.dt}_T{args.tfinal}"
    _plot_snapshots(mesh, snapshots, args.out_dir / f"{stem}.png")
    diag = dict(order=args.order, kx=args.kx, nu=args.nu, kappa=args.kappa,
                dt=args.dt, tfinal=args.tfinal, n_steps=n_steps,
                snapshot_count=len(snapshots))
    with open(args.out_dir / f"{stem}.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"  wrote {args.out_dir / (stem + '.png')}", flush=True)


def _plot_snapshots(mesh, snapshots, out_path):
    """Montage: scalar density + speed |u| at a few times. Each snapshot
    carries its own mesh coordinates so AMR-refined frames render
    correctly on their own (post-adaptation) meshes."""
    pick = [0, len(snapshots) // 3, 2 * len(snapshots) // 3, len(snapshots) - 1]
    fig, axes = plt.subplots(2, len(pick), figsize=(3.0 * len(pick), 6))
    for col, i in enumerate(pick):
        snap = snapshots[i]
        x = snap["x"].flatten(); y = snap["y"].flatten()
        ax_c = axes[0, col]; ax_u = axes[1, col]
        ax_c.scatter(x, y, c=snap["c"].flatten(), s=5,
                     cmap="viridis", vmin=0, vmax=1)
        ax_c.set_aspect("equal"); ax_c.set_xticks([]); ax_c.set_yticks([])
        ax_c.set_title(f"c, t={snap['t']:.3f}", fontsize=9)
        ax_u.scatter(x, y, c=snap["u_mag"].flatten(), s=5, cmap="plasma")
        ax_u.set_aspect("equal"); ax_u.set_xticks([]); ax_u.set_yticks([])
        ax_u.set_title(f"|u|, t={snap['t']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
