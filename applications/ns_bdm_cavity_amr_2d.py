"""Lid-driven cavity on BDM_k with adaptive mesh refinement.

Steady Ghia-Ghia-Shin 1982 benchmark on the H(div)-conforming
saddle-point solver, now driven by ``h``-adaptive refinement
(``src/ddg/dg2d/refinement.py``). The refinement is conforming
red--green, so each adapted mesh is an ordinary ``Mesh2D`` and
``BDM.from_mesh`` rebuilds its topology on it unchanged -- there is no
hanging-node H(div) continuity problem.

Loop (steady, solve-refine-resolve):

    1. extract the conforming active mesh from the forest
    2. build the BDM_k / P_{k-1} space on it
    3. solve the steady NS saddle point with Newton (Re-continuation
       for the higher Reynolds numbers)
    4. form a per-element Kelly indicator from the velocity speed
       sampled at the mesh nodes
    5. Dörfler-mark + red--green refine; repeat

A steady Newton solve needs no field transfer between meshes -- each
level re-solves from scratch (cheap continuation), so the
nodal-Lagrangian ``transfer_field`` (which does not apply to the BDM
Piola/moment DOFs) is not needed here.

The flow is driven entirely by the SIPG Dirichlet *lift*: the lid is
tangential (``u . n = 0`` on every wall, so the normal-trace BDM DOFs
are all zero), and ``bdm_k_viscous_dirichlet_lift`` injects the
tangential lid velocity through the viscous penalty/symmetry terms.

Run:
    JAX_PLATFORMS=cpu python applications/ns_bdm_cavity_amr_2d.py \
        --re 100 --order 3 --kx 6 --levels 3 --theta 0.5

Writes to ``output/``:
    bdm_cavity_amr_re{Re}_k{order}_mesh.png        refinement sequence
    bdm_cavity_amr_re{Re}_k{order}_centrelines.png u/v vs Ghia
    bdm_cavity_amr_re{Re}_k{order}_speed.png       final speed field
    bdm_cavity_amr_re{Re}_k{order}.json            per-level diagnostics
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
    BDM, bdm1_gather, bdm_k_basis_at_points, _element_jacobian_matrices,
    bdm_k_evaluate_boundary_velocity, bdm_k_viscous_dirichlet_lift,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    ns_picard_apply_bdm_k_2d,
)
from ddg.dg2d.bdm_multigrid import make_coupled_velocity_mg_pc
from ddg.dg2d.refinement import RefinementForest
from ddg.dg2d.refinement import (
    RefinementForest, kelly_error_indicator,
)

# Reuse the published Ghia-Ghia-Shin 1982 reference tables.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ns_lid_driven_cavity_2d import (  # noqa: E402
    GHIA_Y, GHIA_U, GHIA_X, GHIA_V,
)


X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
U_LID = 1.0


def _lid_bc_fn(x, y, *, tol=1e-9):
    """U_LID in +x on the lid (y = Y_MAX), zero on the other three
    walls. BDM face GL points sit strictly interior to each edge, so
    this is single-valued at every DOF location (the corner
    discontinuity lives at a vertex, which is not a BDM DOF)."""
    on_lid = jnp.abs(y - Y_MAX) < tol
    u_x = jnp.where(on_lid, U_LID, 0.0)
    u_y = jnp.zeros_like(u_x)
    return u_x, u_y


def _build_space(mesh, order):
    """BDM_k space + the BC quantities for the tangential-lid lift."""
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _lid_bc_fn)        # normal trace = 0
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _lid_bc_fn)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


# ---------------------------------------------------------------------------
# Order-parametric steady Newton solve with the SIPG Dirichlet lift
# ---------------------------------------------------------------------------

def _make_dense_lu_solver(raw_flat, n_u, n_p, bc_mask):
    """Linear-solver backend: build the constrained Jacobian (``jax.jacfwd``
    with boundary rows/cols masked to identity/zero) and solve via dense
    LU. The default; only viable at small sizes but exact and robust."""
    n = n_u + n_p
    jac = jax.jacfwd(raw_flat)
    mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])
    I_n = jnp.eye(n)

    def solve(x_k, rhs):
        J = jac(x_k)
        J_con = jnp.where(
            mask_full[:, None], I_n,
            jnp.where(mask_full[None, :], 0.0, J),
        )
        return jnp.linalg.solve(J_con, rhs)

    return solve


def _make_matfree_gmres_solver(raw_flat, n_u, bc_mask, pc, *,
                                tol=1e-8, restart=200, maxiter=20):
    """Linear-solver backend: matrix-free JVP Jacobian inside right-
    preconditioned GMRES. The Jacobian is applied as ``jax.jvp`` of
    ``raw_flat`` at the current ``x_k``; boundary rows are pinned to
    identity (constrained delta_boundary = 0). ``pc`` is the coupled
    block-diagonal preconditioner (velocity hp-MG + pressure-mass Schur,
    built once per ``nu`` by the caller)."""
    def solve(x_k, rhs):
        def matvec(v):
            vu = v[:n_u]
            vp = v[n_u:]
            vu_z = jnp.where(bc_mask, 0.0, vu)
            _, jv = jax.jvp(raw_flat, (x_k,),
                            (jnp.concatenate([vu_z, vp]),))
            out_u = jnp.where(bc_mask, vu, jv[:n_u])
            return jnp.concatenate([out_u, jv[n_u:]])

        delta, _ = jax.scipy.sparse.linalg.gmres(
            matvec, rhs, x0=jnp.zeros_like(rhs), M=pc,
            tol=tol, atol=1e-12, restart=restart, maxiter=maxiter,
        )
        return delta

    return solve


def _newton_solve(V, *, nu, u_bc_global, u_bc_vec, bc_mask,
                  u_init, p_init, pressure_eps=1e-10, tau_scale=4.0,
                  max_iters=30, tol=1e-9, atol=1e-12,
                  residual_history=None, verbose=False,
                  linear_solver=None):
    """Full-Newton steady NS solve at arbitrary order ``k``.

    Residual ``F = (A + C(u)) u + G p - lift, -D u - eps p`` with the
    homogeneous viscous/convective ghost and the tangential lid carried
    by ``lift = bdm_k_viscous_dirichlet_lift``. Boundary normal-trace
    DOFs are pinned via ``bc_mask`` (the constrained-row/col trick), so
    only interior DOFs are updated.

    The linear solve at each Newton step is pluggable via
    ``linear_solver(x_k, rhs) -> delta``; if ``None``, the default
    :func:`_make_dense_lu_solver` is used (jacfwd + constrained LU).
    Pass :func:`_make_matfree_gmres_solver` (with a coupled PC) for the
    scalable matrix-free path. The Newton outer loop is identical for
    both -- the only difference is how each linearised system is solved."""
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p

    lift = bdm_k_viscous_dirichlet_lift(V, u_bc_vec, nu=nu, tau_scale=tau_scale)

    def raw_flat(x):
        u = x[:n_u]
        p = x[n_u:].reshape(V.K, dim_p)
        Fu, Fp = ns_picard_apply_bdm_k_2d(
            V, u, p, u_star_global=u, nu=nu,
            pressure_eps=pressure_eps, tau_scale=tau_scale,
        )
        return jnp.concatenate([Fu, Fp.reshape(-1)])

    if linear_solver is None:
        linear_solver = _make_dense_lu_solver(raw_flat, n_u, n_p, bc_mask)

    u_k = jnp.where(bc_mask, u_bc_global, u_init)
    p_k = p_init
    x_k = jnp.concatenate([u_k, p_k.reshape(-1)])

    norm_F0 = None
    converged = False
    for it in range(max_iters):
        raw = raw_flat(x_k)
        Fu_inner = jnp.where(bc_mask, 0.0, raw[:n_u] - lift)
        Fp = raw[n_u:]
        norm_F = float(jnp.linalg.norm(Fu_inner) + jnp.linalg.norm(Fp))
        if residual_history is not None:
            residual_history.append(norm_F)
        if norm_F0 is None:
            norm_F0 = norm_F
        if verbose:
            print(f"    newton it={it}  ||F||={norm_F:.3e}", flush=True)
        if norm_F <= tol * norm_F0 + atol:
            converged = True
            break
        rhs = -jnp.concatenate([Fu_inner, Fp])
        delta = linear_solver(x_k, rhs)
        x_k = x_k + delta
        # Re-pin the boundary normal-trace DOFs exactly.
        x_k = x_k.at[:n_u].set(jnp.where(bc_mask, u_bc_global, x_k[:n_u]))

    return x_k[:n_u], x_k[n_u:].reshape(V.K, dim_p), converged


def _reynolds_ramp(Re):
    """Continuation ladder ending at ``Re``, always starting low.

    Low rungs (Re=25, 50, ...) are included even for a Re=100 target:
    full Newton from a cold start has a small basin of attraction, and
    each rung warm-starts the next, so the convection is switched on
    gradually."""
    base = [25.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0, 1500.0]
    rungs = [r for r in base if r < Re - 1e-9]
    return rungs + [float(Re)]


def _solve_cavity(V, Re, u_bc, u_bc_vec, bc_mask, *,
                  tau_scale=4.0, verbose=False,
                  mg_base_mesh=None, mg_n_h=0,
                  gmres_tol=1e-8, gmres_restart=100, gmres_maxiter=20):
    """Steady solve at ``Re`` with automatic Reynolds continuation.

    The cavity lid is tangential, so ``u_bc = 0`` (all normal traces
    vanish) and the cold start ``u = 0`` is a clean Newton seed: the
    residual is just ``-lift``, so the first Newton step solves the
    Stokes-with-lift problem and convection is added thereafter. Each
    continuation rung warm-starts from the previous converged field."""
    ramp = _reynolds_ramp(Re)
    dim_p = V.order * (V.order + 1) // 2
    u_init = u_bc                          # zeros for the tangential lid
    p_init = jnp.zeros((V.K, dim_p))

    hist = []
    converged = False
    for re_step in ramp:
        nu = U_LID * (X_MAX - X_MIN) / re_step
        # Build the per-rung linear solver. ``mg_base_mesh=None`` selects
        # the dense-LU default; passing a base mesh triggers the modular
        # matrix-free GMRES path with the coupled hp-MG + Schur PC.
        linear_solver = None
        if mg_base_mesh is not None:
            pc = make_coupled_velocity_mg_pc(
                V, mg_base_mesh, mg_n_h, nu=nu, tau_scale=tau_scale,
                nsm=2, cycle="W",
            )
            from functools import partial
            n_u = V.total_dofs
            dim_p = V.order * (V.order + 1) // 2

            def raw_flat(x, _nu=nu):
                u = x[:n_u]; p = x[n_u:].reshape(V.K, dim_p)
                Fu, Fp = ns_picard_apply_bdm_k_2d(
                    V, u, p, u_star_global=u, nu=_nu,
                    pressure_eps=1e-10, tau_scale=tau_scale)
                return jnp.concatenate([Fu, Fp.reshape(-1)])

            linear_solver = _make_matfree_gmres_solver(
                raw_flat, n_u, bc_mask, pc,
                tol=gmres_tol, restart=gmres_restart, maxiter=gmres_maxiter)
        u_init, p_init, converged = _newton_solve(
            V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
            u_init=u_init, p_init=p_init, tau_scale=tau_scale,
            residual_history=hist, verbose=verbose,
            linear_solver=linear_solver,
        )
        if verbose:
            print(f"  Re={re_step:.0f}: converged={converged}, "
                  f"||F||={hist[-1]:.3e}", flush=True)
    return u_init, p_init, converged, hist


# ---------------------------------------------------------------------------
# BDM velocity sampling: speed at mesh nodes (for the Kelly indicator) and
# at arbitrary physical points (for the Ghia centreline comparison)
# ---------------------------------------------------------------------------

def _speed_at_nodes(V, u_global):
    """Velocity speed ``|u|`` at the mesh nodal points -> ``(Np, K)``.

    Fed to :func:`kelly_error_indicator`; the lid-corner velocity
    discontinuity makes ``|u|`` non-smooth exactly at the corners, so
    the normal-derivative jump drives refinement there (and along the
    boundary layers)."""
    k = V.order
    r, s = V.mesh.r, V.mesh.s
    phi_ref = bdm_k_basis_at_points(k, r, s)                # (Np, Nbdm, 2)
    DF, _ = _element_jacobian_matrices(V.mesh)              # (K, 2, 2)
    J = V.mesh.J[0]                                         # (K,)
    phi_phys = jnp.einsum("kca,nla->knlc", DF, phi_ref) \
        / J[:, None, None, None]                            # (K, Np, Nbdm, 2)
    u_local = bdm1_gather(V.topology, u_global)             # (K, Nbdm)
    u_nodes = jnp.einsum("kl,knlc->knc", u_local, phi_phys)  # (K, Np, 2)
    speed = jnp.sqrt(u_nodes[..., 0] ** 2 + u_nodes[..., 1] ** 2)  # (K, Np)
    return speed.T                                          # (Np, K)


def _evaluate_velocity_at_points(V, u_global, x_query, y_query):
    """BDM velocity at physical query points (brute-force element search).

    Slow but adequate for ~tens of centreline points."""
    k = V.order
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    K = V.K
    u_local = bdm1_gather(V.topology, u_global)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    x_q = np.asarray(x_query).reshape(-1)
    y_q = np.asarray(y_query).reshape(-1)
    out_x = np.zeros_like(x_q)
    out_y = np.zeros_like(y_q)
    for qi, (xq, yq) in enumerate(zip(x_q, y_q)):
        for kk in range(K):
            v0 = (VX[EToV[kk, 0]], VY[EToV[kk, 0]])
            v1 = (VX[EToV[kk, 1]], VY[EToV[kk, 1]])
            v2 = (VX[EToV[kk, 2]], VY[EToV[kk, 2]])
            det = (v1[0] - v0[0]) * (v2[1] - v0[1]) \
                - (v2[0] - v0[0]) * (v1[1] - v0[1])
            a = ((v1[0] - xq) * (v2[1] - yq)
                 - (v2[0] - xq) * (v1[1] - yq)) / det
            b = ((v2[0] - xq) * (v0[1] - yq)
                 - (v0[0] - xq) * (v2[1] - yq)) / det
            c = 1.0 - a - b
            if a >= -1e-10 and b >= -1e-10 and c >= -1e-10:
                r = 2.0 * b - 1.0
                s = 2.0 * c - 1.0
                phi_ref = bdm_k_basis_at_points(
                    k, jnp.array([r]), jnp.array([s]))      # (1, Nbdm, 2)
                phi_phys = (DF[kk] @ phi_ref[0].T).T / J[kk]  # (Nbdm, 2)
                u_xy = np.asarray(jnp.einsum("l,lc->c", u_local[kk], phi_phys))
                out_x[qi] = u_xy[0]
                out_y[qi] = u_xy[1]
                break
    return out_x.reshape(np.shape(x_query)), out_y.reshape(np.shape(y_query))


def _ghia_centreline_error(V, u_sol, Re):
    """Max/RMS error of u_x(0.5, y) vs the Ghia table for this Re."""
    Re_key = int(round(Re))
    if Re_key not in GHIA_U:
        return None
    ys = GHIA_Y
    ux, _ = _evaluate_velocity_at_points(
        V, u_sol, np.full_like(ys, 0.5), ys)
    ref = GHIA_U[Re_key]
    err = np.abs(ux - ref)
    return dict(max=float(err.max()), rms=float(np.sqrt(np.mean(err ** 2))))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_mesh_sequence(meshes, dofs, out_path):
    n = len(meshes)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, m, ndof in zip(axes, meshes, dofs):
        VX = np.asarray(m.VX); VY = np.asarray(m.VY)
        EToV = np.asarray(m.EToV)
        ax.triplot(VX, VY, EToV, color="0.3", lw=0.4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"K={EToV.shape[0]}, DOFs={ndof}", fontsize=9)
    fig.suptitle("Cavity AMR: refinement sequence (lid at top)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def _plot_centrelines(V, u_sol, Re, out_path):
    Re_key = int(round(Re))
    y_line = np.linspace(0.0, 1.0, 80)
    x_line = np.linspace(0.0, 1.0, 80)
    ux_vert, _ = _evaluate_velocity_at_points(
        V, u_sol, np.full(80, 0.5), y_line)
    _, uy_horz = _evaluate_velocity_at_points(
        V, u_sol, x_line, np.full(80, 0.5))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ux_vert, y_line, "b-", label="BDM AMR")
    if Re_key in GHIA_U:
        axes[0].plot(GHIA_U[Re_key], GHIA_Y, "ks", ms=4,
                     label="Ghia 1982")
    axes[0].set_xlabel("$u_x$"); axes[0].set_ylabel("$y$")
    axes[0].set_title(f"$u_x$ along x=0.5, Re={Re_key}")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(x_line, uy_horz, "r-", label="BDM AMR")
    if Re_key in GHIA_V:
        axes[1].plot(GHIA_X, GHIA_V[Re_key], "ks", ms=4,
                     label="Ghia 1982")
    axes[1].set_xlabel("$x$"); axes[1].set_ylabel("$u_y$")
    axes[1].set_title(f"$u_y$ along y=0.5, Re={Re_key}")
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def _plot_speed_field(V, u_sol, out_path):
    """Filled speed field on the final adapted mesh (per-node scatter)."""
    speed = np.asarray(_speed_at_nodes(V, u_sol))           # (Np, K)
    r, s = np.asarray(V.mesh.r), np.asarray(V.mesh.s)
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    xs, ys, cs = [], [], []
    for kk in range(V.K):
        v0 = (VX[EToV[kk, 0]], VY[EToV[kk, 0]])
        v1 = (VX[EToV[kk, 1]], VY[EToV[kk, 1]])
        v2 = (VX[EToV[kk, 2]], VY[EToV[kk, 2]])
        a = 0.5 * (r + 1.0); b = 0.5 * (s + 1.0)
        x = v0[0] + a * (v1[0] - v0[0]) + b * (v2[0] - v0[0])
        y = v0[1] + a * (v1[1] - v0[1]) + b * (v2[1] - v0[1])
        xs.append(x); ys.append(y); cs.append(speed[:, kk])
    xs = np.concatenate(xs); ys = np.concatenate(ys); cs = np.concatenate(cs)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(xs, ys, c=cs, s=6, cmap="viridis")
    ax.set_aspect("equal"); ax.set_title("speed |u| (final mesh)")
    fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=6, help="initial mesh Kx=Ky")
    ap.add_argument("--levels", type=int, default=3,
                    help="number of refinement passes")
    ap.add_argument("--theta", type=float, default=0.5,
                    help="Dorfler bulk-marking fraction")
    ap.add_argument("--max-level", type=int, default=4)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--solver", choices=("dense", "iterative"),
                    default="dense",
                    help="dense LU (small) or matrix-free hp-MG GMRES (scales)")
    ap.add_argument("--mg-base-kx", type=int, default=2,
                    help="base mesh size for the hp-MG hierarchy (iterative)")
    ap.add_argument("--mg-n-h", type=int, default=0,
                    help="number of h-levels in the hp-MG hierarchy. The "
                         "initial solve mesh is the base mesh forest-refined "
                         "n_h times -- if >0, overrides --kx.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    # In iterative mode, the solve mesh is the base mesh forest-refined
    # ``mg_n_h`` times, so the same forest gives the MG hierarchy. ``--kx``
    # is then the *base* size (default 2) refined --mg-n-h times.
    iterative = (args.solver == "iterative")
    use_mg_hierarchy = iterative and args.mg_n_h > 0
    if use_mg_hierarchy:
        bx = args.mg_base_kx
        VX, VY, EToV = uniform_rectangle_mesh_2d(
            X_MIN, X_MAX, Y_MIN, Y_MAX, bx, bx)
        mg_base_mesh = build_mesh_2d(VX, VY, EToV, args.order)
        eff_kx = bx * (2 ** args.mg_n_h)
        # Pre-refine the forest mg_n_h times to get the initial solve mesh.
        forest = RefinementForest(mg_base_mesh, max_level=args.max_level)
        for _ in range(args.mg_n_h):
            forest.refine(forest.leaves())
    else:
        mg_base_mesh = None
        eff_kx = args.kx
        VX, VY, EToV = uniform_rectangle_mesh_2d(
            X_MIN, X_MAX, Y_MIN, Y_MAX, args.kx, args.kx)
        mesh0 = build_mesh_2d(VX, VY, EToV, args.order)
        forest = RefinementForest(mesh0, max_level=args.max_level)

    print(f"Cavity AMR: Re={args.re:.0f}, BDM_{args.order}, "
          f"init {eff_kx}x{eff_kx}, {args.levels} levels, "
          f"theta={args.theta}, solver={args.solver}"
          + (f" (mg base={args.mg_base_kx}, n_h={args.mg_n_h})"
             if use_mg_hierarchy else ""), flush=True)

    meshes, level_diag = [], []
    V_final = u_final = None
    for level in range(args.levels + 1):
        mesh_active, active_to_leaf = forest.active_mesh()
        V, u_bc, u_bc_vec, bc_mask = _build_space(mesh_active, args.order)
        ndof = V.total_dofs + V.K * (args.order * (args.order + 1) // 2)
        t0 = time.time()
        u_sol, p_sol, converged, hist = _solve_cavity(
            V, args.re, u_bc, u_bc_vec, bc_mask,
            tau_scale=args.tau_scale, verbose=False,
            mg_base_mesh=mg_base_mesh, mg_n_h=args.mg_n_h)
        gh = _ghia_centreline_error(V, u_sol, args.re)
        dt = time.time() - t0
        gh_str = (f", Ghia max={gh['max']:.4f} rms={gh['rms']:.4f}"
                  if gh else "")
        print(f"  level {level}: K={V.K}, DOFs={ndof}, "
              f"converged={converged}, ||F||={hist[-1]:.2e}, "
              f"{dt:.1f}s{gh_str}", flush=True)
        meshes.append(mesh_active)
        level_diag.append(dict(
            level=level, K=int(V.K), dofs=int(ndof),
            converged=bool(converged), final_res=float(hist[-1]),
            ghia=gh, seconds=float(dt)))
        V_final, u_final = V, u_sol

        if level < args.levels:
            eta = kelly_error_indicator(mesh_active, _speed_at_nodes(V, u_sol))
            n_ref = forest.refine_by_indicator(
                np.asarray(eta), active_to_leaf, theta=args.theta)
            print(f"    refined {n_ref} leaves", flush=True)

    stem = f"bdm_cavity_amr_re{int(args.re):d}_k{args.order:d}"
    _plot_mesh_sequence(meshes, [d["dofs"] for d in level_diag],
                        args.out_dir / f"{stem}_mesh.png")
    _plot_centrelines(V_final, u_final, args.re,
                      args.out_dir / f"{stem}_centrelines.png")
    _plot_speed_field(V_final, u_final, args.out_dir / f"{stem}_speed.png")
    with open(args.out_dir / f"{stem}.json", "w") as f:
        json.dump(dict(Re=args.re, order=args.order, kx=args.kx,
                       levels=args.levels, theta=args.theta,
                       per_level=level_diag), f, indent=2)
    print(f"  wrote {args.out_dir / (stem + '.json')}", flush=True)


if __name__ == "__main__":
    main()
