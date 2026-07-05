"""Transient lid-driven cavity on the BDM-P_0 coupled NS solver.

Time-marches the cavity from rest (u=0 interior, u_BC on boundary) using
BDF1 implicit time-stepping. Each time step solves a fully implicit
NS Newton/Picard problem on the BDM monolithic saddle-point with the
iterative GMRES + block-PC stack added in commit e8b5a1a.

Diagnostics saved at every step (cheap):
    kinetic_energy       0.5 * integral |u|^2 dx
    enstrophy            0.5 * integral |curl u|^2 dx
    divergence_violation L2 norm of the discrete divergence
    probe values         u at canonical Ghia (x, y) points

Velocity-field snapshots saved every N_snap steps (configurable).
These are the dataset the eventual projection / IMEX scheme runs
will compare against -- the monolithic BDM run is the ground truth.

Run locally (smoke):
    JAX_PLATFORMS=cpu python applications/ns_bdm_cavity_transient_2d.py \\
        --re 100 --kx 8 --dt 0.05 --tfinal 1.0 --snap-every 5

Run on the remote (production):
    ssh corbinfoucart@34.136.137.187
    cd ddg && conda activate ddg
    JAX_PLATFORMS=cpu python applications/ns_bdm_cavity_transient_2d.py \\
        --re 100 --kx 16 --dt 0.05 --tfinal 30 --snap-every 20

Output:
    output/bdm_cavity_transient_re{Re}_kx{Kx}.npz       -- snapshots + time-series
    output/bdm_cavity_transient_re{Re}_kx{Kx}.json      -- run metadata
    output/bdm_cavity_transient_re{Re}_kx{Kx}_diag.png  -- diagnostics plot
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
    bdm_k_evaluate_boundary_velocity,
    bdm1_gather, _bdm1_physical_gradient_matrices,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns import IterativeSolverConfig
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdf_history, bdf_history_k,
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    bdm1_mass_apply,
    make_bdm_block_diag_pc, make_bdm_cahouet_chabard_pc,
    make_bdm_k_iterative_inner_solvers,
    make_bdm_velocity_block_jacobi_pc,
    ns_newton_solve_bdm_2d, ns_newton_solve_bdm_k_2d,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


U_LID = 1.0
HAMMER_R = jnp.array([0.0, -1.0, 0.0])
HAMMER_S = jnp.array([-1.0, 0.0, 0.0])
HAMMER_W = jnp.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])


def _lid_bc_fn(x, y):
    on_lid = jnp.abs(y - 1.0) < 1e-9
    return jnp.where(on_lid, U_LID, 0.0), jnp.zeros_like(x)


def _build_cavity_setup(Kx: int, Ky: int, order: int = 1):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, order)
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _lid_bc_fn)        # normal trace
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _lid_bc_fn)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _kinetic_energy(V: BDM, u_global: jax.Array) -> float:
    """0.5 * integral_Omega |u|^2 dx.

    BDM_1 closed form (piecewise-linear in (r, s), Hammer 3-point exact);
    for BDM_k>1 we fall back to a coarse element-centred sample (good
    enough for blowup detection / qualitative growth, not a quantitative
    integral)."""
    if V.order == 1:
        u_local = bdm1_gather(V.topology, u_global)
        DF, _ = _element_jacobian_matrices(V.mesh)
        J = V.mesh.J[0]
        phi_ref = bdm1_basis_at_points(HAMMER_R, HAMMER_S)        # (Q, 6, 2)
        phi_phys = jnp.einsum("kab,qlb->kqla", DF, phi_ref) / J[:, None, None, None]
        u_at_q = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)    # (K, Q, 2)
        integrand = jnp.sum(u_at_q ** 2, axis=-1)                  # (K, Q)
        return 0.5 * float(jnp.einsum("q,k,kq->", HAMMER_W, J, integrand))
    # Order k>1: integrate |u|^2 with the same Hammer rule applied to
    # the order-parametric basis evaluation. Approximate (Hammer is
    # exact only up to degree 2) but adequate as a qualitative trace.
    from ddg.dg2d.bdm import bdm_k_basis_at_points
    u_local = bdm1_gather(V.topology, u_global)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_ref = bdm_k_basis_at_points(V.order, HAMMER_R, HAMMER_S)   # (Q, Np, 2)
    phi_phys = jnp.einsum("kab,qlb->kqla", DF, phi_ref) / J[:, None, None, None]
    u_at_q = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)
    integrand = jnp.sum(u_at_q ** 2, axis=-1)
    return 0.5 * float(jnp.einsum("q,k,kq->", HAMMER_W, J, integrand))


def _enstrophy(V: BDM, u_global: jax.Array) -> float:
    """0.5 * integral |curl u|^2 dx.

    BDM_1: closed form (curl is element-constant). BDM_k>1: order-aware
    via ``bdm_k_basis_grad_at_points`` at the Hammer points; same
    qualitative-trace caveat as :func:`_kinetic_energy`."""
    if V.order == 1:
        u_local = bdm1_gather(V.topology, u_global)
        M_grad = _bdm1_physical_gradient_matrices(V)         # (K, 6, 2, 2)
        grad_u = jnp.einsum("kl,klij->kij", u_local, M_grad) # (K, 2, 2)
        omega = grad_u[:, 1, 0] - grad_u[:, 0, 1]            # (K,)
        J = V.mesh.J[0]
        area = 2.0 * J
        return 0.5 * float(jnp.sum(omega ** 2 * area))
    # Order k>1: numerical curl via grad of the physical basis.
    from ddg.dg2d.bdm import bdm_k_basis_grad_at_points
    u_local = bdm1_gather(V.topology, u_global)
    DF, _ = _element_jacobian_matrices(V.mesh)
    DF_inv = jnp.linalg.inv(DF)                                  # (K, 2, 2)
    J = V.mesh.J[0]
    inv_J = 1.0 / J                                              # (K,)
    grad_ref = bdm_k_basis_grad_at_points(
        V.order, HAMMER_R, HAMMER_S,
    )                                                            # (Q, Np, 2, 2)
    # Piola contravariant grad in physical: (1/J) DF · grad_ref · DF^{-1}
    grad_phys = jnp.einsum(
        "k,kca,qlae,ked->kqlcd", inv_J, DF, grad_ref, DF_inv,
    )                                                            # (K, Q, Np, 2, 2)
    grad_u = jnp.einsum("kl,kqlcd->kqcd", u_local, grad_phys)    # (K, Q, 2, 2)
    omega = grad_u[..., 1, 0] - grad_u[..., 0, 1]                # (K, Q)
    return 0.5 * float(jnp.einsum("q,k,kq->", HAMMER_W, J, omega ** 2))


def _divergence_violation(V: BDM, u_global: jax.Array) -> float:
    """L2 norm of the discrete divergence. For BDM-P_{k-1} this is
    machine zero by construction (the H(div) pair gives pointwise
    discrete div-free) -- monitoring it confirms the solver stays in
    that regime."""
    from ddg.dg2d.bdm import bdm_k_divergence_apply
    div_p = bdm_k_divergence_apply(V, u_global)
    return float(jnp.linalg.norm(div_p))


def _eval_at_probe(V: BDM, u_global: jax.Array, x: float, y: float):
    """Sample u at a single physical (x, y) point. Brute-force per-element
    containment search; OK because we sample only a few probes per step."""
    from ddg.dg2d.bdm import bdm_k_basis_at_points as _bdmk_basis
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX)
    VY = np.asarray(V.mesh.VY)
    u_local = bdm1_gather(V.topology, u_global)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    for k in range(V.K):
        v0 = (VX[EToV[k, 0]], VY[EToV[k, 0]])
        v1 = (VX[EToV[k, 1]], VY[EToV[k, 1]])
        v2 = (VX[EToV[k, 2]], VY[EToV[k, 2]])
        det = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
        a = ((v1[0] - x) * (v2[1] - y) - (v2[0] - x) * (v1[1] - y)) / det
        b = ((v2[0] - x) * (v0[1] - y) - (v0[0] - x) * (v2[1] - y)) / det
        c = 1.0 - a - b
        if a >= -1e-10 and b >= -1e-10 and c >= -1e-10:
            r = 2.0 * b - 1.0
            s = 2.0 * c - 1.0
            if V.order == 1:
                phi_ref = bdm1_basis_at_points(jnp.array([r]), jnp.array([s]))
            else:
                phi_ref = _bdmk_basis(V.order, jnp.array([r]), jnp.array([s]))
            phi_phys = (DF[k] @ phi_ref[0].T).T / J[k]
            u_xy = np.asarray(jnp.einsum("l,lc->c", u_local[k], phi_phys))
            return float(u_xy[0]), float(u_xy[1])
    return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Time-stepping driver
# ---------------------------------------------------------------------------

def run_transient_cavity(
    *, Re: float, Kx: int, Ky: int, dt: float, t_final: float,
    bdf_order: int = 2,
    order: int = 1,
    snap_every: int = 10,
    tol_outer: float = 1e-8,
    tol_inner: float = 1e-8,
    max_outer: int = 10,
    linearisation: str = "picard",
    use_iterative: bool = True,
    verbose: bool = True,
    tau_scale: float = 4.0,
):
    """BDF1/BDF2 time-stepping of the cavity. Step 0 is always BDF1
    (the bootstrap, since u^{n-1} doesn't yet exist); steps 1+ run at
    ``bdf_order``. Returns dict of arrays for saving / plotting."""
    if bdf_order not in (1, 2):
        raise ValueError(f"bdf_order must be 1 or 2, got {bdf_order}")
    nu = U_LID * 1.0 / Re
    V, u_bc, u_bc_vec, bc_mask = _build_cavity_setup(Kx, Ky, order=order)
    n_u, n_p = V.total_dofs, V.K * (order * (order + 1) // 2)

    # PCs: order-parametric block-Jacobi velocity + Cahouet-Chabard
    # Schur, one per BDF order we'll actually hit. Inner solvers
    # are JIT'd once per (order, alpha, PC) and re-used across BDF
    # steps; the trace cache survives every time-step call so first
    # step is compile-heavy and subsequent steps run at compiled
    # speed.
    pcs_per_order = {}
    inner_solvers_per_order = {}
    iter_cfg = None
    if use_iterative:
        iter_cfg = IterativeSolverConfig(
            method="gmres", tol=tol_inner, atol=1e-12,
            maxiter=500, restart=50,
        )
        orders_used = {1, bdf_order}
        if order == 1:
            from ddg.dg2d.coupled_ns_bdm import (
                make_bdm_velocity_block_jacobi_pc as _make_vel_pc,
                make_bdm_cahouet_chabard_pc as _make_schur_pc,
                make_bdm_block_diag_pc as _make_block_pc,
            )
            _vel_kwargs = lambda alpha: dict(alpha=alpha, nu=nu)
        else:
            from ddg.dg2d.coupled_ns_bdm import (
                make_bdm_k_velocity_block_jacobi_pc as _make_vel_pc,
                make_bdm_k_cahouet_chabard_pc as _make_schur_pc,
                make_bdm_k_block_diag_pc as _make_block_pc,
            )
            _vel_kwargs = lambda alpha: dict(
                alpha=alpha, nu=nu, tau_scale=tau_scale,
            )
        for bdf_o in orders_used:
            gamma0 = BDF_COEFFS[bdf_o]["gamma0"]
            alpha = gamma0 / dt
            vel_pc = _make_vel_pc(V, **_vel_kwargs(alpha))
            schur_pc = _make_schur_pc(V, nu=nu, dt=dt, gamma0=gamma0)
            pcs_per_order[bdf_o] = _make_block_pc(
                V, velocity_pc=vel_pc, pressure_pc=schur_pc,
            )
            # Order > 1 uses the hoist-able inner_solvers factory.
            # Order 1 still uses the BDM-P_0 path which has its own
            # JIT structure inside ns_newton_solve_bdm_2d.
            if order > 1:
                inner_solvers_per_order[bdf_o] = make_bdm_k_iterative_inner_solvers(
                    V, iterative=iter_cfg,
                    coupled_preconditioner=pcs_per_order[bdf_o],
                    nu=nu, pressure_eps=1e-10,
                    tau_scale=tau_scale, velocity_alpha=alpha,
                    bc_mask=bc_mask, u_bc_global=None,
                    build_newton=(linearisation in ("newton", "hybrid")),
                )

    # Initial condition: u_bc on boundary, zero interior. Sometimes
    # called the "no-pre-load" IC -- the lid pulls the fluid into motion.
    u_n = u_bc            # u^n (most recent state)
    u_nm1 = None          # u^{n-1} (older, populated after step 0)
    p_prev = jnp.zeros(n_p)

    n_steps = int(round(t_final / dt))
    snap_indices = []
    snapshots_u = []
    snapshots_p = []
    times = []
    ke = []
    ens = []
    div_viol = []
    probe_centre = []
    outer_iters_per_step = []
    inner_residual_final = []

    if verbose:
        print(f"Transient cavity: Re={Re:.1f}, nu={nu:.3e}, "
              f"Kx={Kx}, Ky={Ky}, dt={dt:.3e}, T_final={t_final:.2f}, "
              f"n_steps={n_steps}, DOFs={n_u + n_p}, BDF{bdf_order}")
        print(f"  linearisation={linearisation}, "
              f"{'iterative GMRES+PC' if use_iterative else 'direct LU'}, "
              f"snap_every={snap_every}")

    t0 = time.time()
    for step in range(n_steps):
        t = (step + 1) * dt
        # Bootstrap step 0 at order 1; thereafter at bdf_order.
        order_eff = 1 if (bdf_order > 1 and step == 0) else bdf_order
        alpha = BDF_COEFFS[order_eff]["gamma0"] / dt
        u_hist = [u_n] if order_eff == 1 else [u_n, u_nm1]
        f_history = bdf_history_k(V, u_hist, dt=dt, order=order_eff)
        pc = pcs_per_order.get(order_eff)
        outer_hist = []
        inner_hist = []
        # Choose the order-parametric solver for k > 1; the BDM_1
        # transient path stays on ns_newton_solve_bdm_2d to preserve
        # legacy artifact reproducibility at order=1.
        if order == 1:
            u_new, p_new_flat, _ = ns_newton_solve_bdm_2d(
                V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
                u_init=u_n, p_init=p_prev,
                velocity_alpha=alpha,
                f_velocity=f_history,
                pressure_eps=1e-10, tau_scale=tau_scale,
                max_iters=max_outer, tol=tol_outer, atol=1e-12,
                residual_history=outer_hist,
                linearisation=linearisation,
                iterative=iter_cfg, coupled_preconditioner=pc,
                inner_residual_history=inner_hist,
            )
            p_new = p_new_flat
        else:
            u_new, p_new, _ = ns_newton_solve_bdm_k_2d(
                V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
                u_init=u_n, p_init=p_prev,
                velocity_alpha=alpha,
                f_velocity=f_history,
                pressure_eps=1e-10, tau_scale=tau_scale,
                max_iters=max_outer, tol=tol_outer, atol=1e-12,
                residual_history=outer_hist,
                linearisation=linearisation,
                iterative=iter_cfg, coupled_preconditioner=pc,
                inner_residual_history=inner_hist,
                inner_solvers=inner_solvers_per_order.get(order_eff),
            )
        # Diagnostics every step.
        ke.append(_kinetic_energy(V, u_new))
        ens.append(_enstrophy(V, u_new))
        div_viol.append(_divergence_violation(V, u_new))
        probe_centre.append(_eval_at_probe(V, u_new, 0.5, 0.5))
        outer_iters_per_step.append(len(outer_hist))
        inner_residual_final.append(inner_hist[-1] if inner_hist else 0.0)
        times.append(float(t))
        # Snapshot velocity every snap_every steps (always include step 0
        # and the last step).
        if step % snap_every == 0 or step == n_steps - 1:
            snap_indices.append(step + 1)
            snapshots_u.append(np.asarray(u_new))
            snapshots_p.append(np.asarray(p_new))
        if verbose and (step % max(1, n_steps // 20) == 0 or step == n_steps - 1):
            print(f"  step {step+1:5d}/{n_steps} t={t:.3f} BDF{order_eff}: "
                  f"KE={ke[-1]:.4e}, enstr={ens[-1]:.4e}, "
                  f"||div||={div_viol[-1]:.2e}, "
                  f"outer={outer_iters_per_step[-1]}, "
                  f"inner_res={inner_residual_final[-1]:.2e}")
        u_nm1 = u_n
        u_n = u_new
        p_prev = p_new
    wall = time.time() - t0
    if verbose:
        print(f"Done in {wall:.1f}s ({wall/n_steps*1000:.0f} ms/step)")

    return dict(
        V=V,
        nu=nu,
        Re=float(Re), Kx=int(Kx), Ky=int(Ky), dt=float(dt),
        bdf_order=int(bdf_order),
        t_final=float(t_final), n_steps=int(n_steps),
        snap_indices=np.asarray(snap_indices, dtype=np.int64),
        snapshots_u=np.stack(snapshots_u),
        snapshots_p=np.stack(snapshots_p),
        times=np.asarray(times),
        kinetic_energy=np.asarray(ke),
        enstrophy=np.asarray(ens),
        divergence_violation=np.asarray(div_viol),
        probe_centre=np.asarray(probe_centre),
        outer_iters_per_step=np.asarray(outer_iters_per_step),
        inner_residual_final=np.asarray(inner_residual_final),
        wall_seconds=float(wall),
        linearisation=linearisation,
        use_iterative=bool(use_iterative),
    )


def _save_outputs(result: dict, stem: str, out_dir: Path):
    """Save .npz (full reference dataset) + .json (metadata) +
    diagnostic .png. .npz contains all snapshots; .json has the cheap
    scalars."""
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        Re=result["Re"], nu=result["nu"], Kx=result["Kx"], Ky=result["Ky"],
        dt=result["dt"], t_final=result["t_final"], n_steps=result["n_steps"],
        snap_indices=result["snap_indices"],
        snapshots_u=result["snapshots_u"],
        snapshots_p=result["snapshots_p"],
        times=result["times"],
        kinetic_energy=result["kinetic_energy"],
        enstrophy=result["enstrophy"],
        divergence_violation=result["divergence_violation"],
        probe_centre=result["probe_centre"],
        outer_iters_per_step=result["outer_iters_per_step"],
        inner_residual_final=result["inner_residual_final"],
    )
    json_path = out_dir / f"{stem}.json"
    meta = {k: result[k] for k in (
        "Re", "nu", "Kx", "Ky", "dt", "bdf_order", "t_final", "n_steps",
        "wall_seconds", "linearisation", "use_iterative",
    )}
    meta["final_ke"] = float(result["kinetic_energy"][-1])
    meta["final_enstrophy"] = float(result["enstrophy"][-1])
    meta["final_div_viol"] = float(result["divergence_violation"][-1])
    meta["n_snapshots"] = int(result["snap_indices"].shape[0])
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {npz_path}")
    print(f"  wrote {json_path}")

    # Diagnostics plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    t = result["times"]
    axes[0, 0].semilogy(t, result["kinetic_energy"])
    axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel("KE = $\\frac{1}{2}\\int |u|^2$")
    axes[0, 0].set_title("Kinetic energy")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].semilogy(t, result["enstrophy"])
    axes[0, 1].set_xlabel("t"); axes[0, 1].set_ylabel("$\\frac{1}{2}\\int |\\omega|^2$")
    axes[0, 1].set_title("Enstrophy")
    axes[0, 1].grid(alpha=0.3)
    axes[1, 0].semilogy(t, result["divergence_violation"] + 1e-300)
    axes[1, 0].set_xlabel("t"); axes[1, 0].set_ylabel("$\\|\\nabla\\cdot u_h\\|$")
    axes[1, 0].set_title("Divergence violation (should be ~machine zero for BDM)")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(t, result["probe_centre"][:, 0], label="$u_x(0.5, 0.5)$")
    axes[1, 1].plot(t, result["probe_centre"][:, 1], label="$u_y(0.5, 0.5)$")
    axes[1, 1].axhline(-0.2109, color="C0", ls="--", alpha=0.5,
                        label="Ghia 1982 Re=100")
    axes[1, 1].set_xlabel("t"); axes[1, 1].set_ylabel("u at (0.5, 0.5)")
    axes[1, 1].set_title("Centre-probe velocity")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)
    fig.suptitle(f"BDM transient cavity Re={result['Re']:.0f} "
                  f"({result['Kx']}x{result['Ky']}, dt={result['dt']})")
    fig.tight_layout()
    png_path = out_dir / f"{stem}_diag.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {png_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--kx", type=int, default=8)
    ap.add_argument("--order", type=int, default=1,
                    help="BDM_k velocity order (default 1; use 3 for high-order)")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tfinal", type=float, default=1.0)
    ap.add_argument("--bdf-order", type=int, choices=(1, 2), default=2)
    ap.add_argument("--snap-every", type=int, default=5)
    ap.add_argument("--linearisation",
                     choices=("picard", "newton", "hybrid"),
                     default="newton",
                     help="Default: pure Newton (quadratic local rate). "
                          "Use 'hybrid' (first iter Picard, then Newton) "
                          "as a safety net for hard cases where the Newton "
                          "basin shrinks (large dt jump, severe Re, etc.).")
    ap.add_argument("--no-iterative", action="store_true",
                     help="Use direct LU instead of GMRES+PC")
    ap.add_argument("--tol-outer", type=float, default=1e-8)
    ap.add_argument("--tol-inner", type=float, default=1e-8)
    ap.add_argument("--max-outer", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    result = run_transient_cavity(
        Re=args.re, Kx=args.kx, Ky=args.kx,
        order=args.order,
        dt=args.dt, t_final=args.tfinal,
        bdf_order=args.bdf_order,
        snap_every=args.snap_every,
        tol_outer=args.tol_outer, tol_inner=args.tol_inner,
        max_outer=args.max_outer,
        linearisation=args.linearisation,
        use_iterative=not args.no_iterative,
    )
    stem = (
        f"bdm_cavity_transient_re{int(args.re):d}_k{args.order:d}_kx{args.kx:d}"
        f"_dt{args.dt:.3f}".replace(".", "p")
        + f"_bdf{args.bdf_order}"
        + f"_{args.linearisation}"
    )
    _save_outputs(result, stem, out_dir)


if __name__ == "__main__":
    main()
