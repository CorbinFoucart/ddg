"""Parameter-inverse demo: recover viscosity nu from sparse velocity
probes on the BDM cylinder transient.

This is the structural twin of
``applications/material_id_heat_1d.py`` --- a 1-D material-id demo on
the heat equation --- but on the 2-D BDM transient INS forward map.

Setup
-----
* BDM_2 cylinder smoke (Re_truth = 100, K = 432), the same geometry
  used by ``scripts/diff_cylinder_dt_tau.py`` and
  ``scripts/inflow_control_cylinder_2d.py``.
* Forward map: ``n_steps`` BDF2 steps from rest with ``n_newton``
  Newton iters per step; the inflow is held at its canonical
  parabolic value (no inflow control here).
* The trajectory is parameterised by ``nu`` (the dynamic viscosity).
  We generate a "ground-truth" trajectory at ``nu_true``, sample the
  velocity at a handful of downstream probe points at each time step,
  and call those samples ``y_obs``. The inverse problem: given
  ``y_obs``, recover ``nu``.
* Loss: standard L2 mismatch on the probe traces,

      L(log nu) = || y_obs - y(log nu) ||^2_2.

  Adam on ``log nu`` lets us span orders of magnitude with stable
  Newton convergence near the truth.

Cost note
---------
Forward + backward through the same fixed-shape 10-step trajectory is
~4 min/iter steady (matches the inflow-control demo), so a 15-iter
recovery is ~1 h wall on a single Apple-silicon thread.

Outputs (``output/``)
---------------------
``inverse_nu_cylinder_k2.{png,json,npz}`` --- recovered nu trajectory,
loss curve, probe-trace comparison at the final iter, .npz dump of the
ground-truth and final-iter probe traces.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diff_cylinder_dt_tau import (
    build_static,
    X_MIN, X_MAX, Y_MIN, Y_MAX, CX, CY, RADIUS, D_CYL,
    U_MAX, U_MEAN,
)
from ddg.dg2d.bdm import (
    bdm_k_basis_at_points, bdm_k_convective_apply,
    bdm_k_viscous_dirichlet_lift, bdm1_gather,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdm_k_mass_apply,
    ns_picard_apply_bdm_k_2d, ns_picard_matrix_bdm_k_2d,
)


def make_step_nu_param(V, u_bc, u_bc_vec, bc_mask, is_natural_face,
                       *, tau_scale, n_newton):
    """A fully traceable BDF step that takes ``nu`` as a runtime
    parameter (not a closure constant). Inflow is held at canonical
    strength (no time-varying control)."""
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    n_full = n_u + n_p

    def step(u_n, u_nm1, dt, nu, order_eff: int):
        coeffs = BDF_COEFFS[order_eff]
        velocity_alpha = coeffs["gamma0"] / dt
        f_velocity = (coeffs["beta"][0] / dt) * bdm_k_mass_apply(V, u_n)
        if order_eff == 2:
            f_velocity = f_velocity + (
                coeffs["beta"][1] / dt
            ) * bdm_k_mass_apply(V, u_nm1)

        lift_u = bdm_k_viscous_dirichlet_lift(
            V, u_bc_vec, nu=nu, tau_scale=tau_scale,
            is_natural_face=is_natural_face,
        )
        u_k = u_n
        p_k = jnp.zeros((V.K, dim_p))
        mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])

        for _it in range(n_newton):
            F_u, F_p = ns_picard_apply_bdm_k_2d(
                V, u_k, p_k,
                u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=velocity_alpha,
                f_velocity=f_velocity,
                u_bc_global=u_bc,
                is_natural_face=is_natural_face,
            )
            F_u = F_u - lift_u
            F_u_inner = jnp.where(bc_mask, 0.0, F_u)
            F_p_flat = F_p.reshape(-1)
            rhs_full = jnp.concatenate([-F_u_inner, -F_p_flat])
            A_full = ns_picard_matrix_bdm_k_2d(
                V, u_star_global=u_k, nu=nu,
                pressure_eps=1e-10, tau_scale=tau_scale,
                velocity_alpha=velocity_alpha,
                is_natural_face=is_natural_face,
            )
            def newton_col(i):
                e = jnp.zeros(n_u).at[i].set(1.0)
                return bdm_k_convective_apply(
                    V, u_k, u_star_global=e,
                    is_natural_face=is_natural_face,
                )
            N_ext = jax.vmap(newton_col)(jnp.arange(n_u)).T
            A_full = A_full.at[:n_u, :n_u].add(N_ext)
            I_n = jnp.eye(n_full)
            A_c = jnp.where(
                mask_full[:, None], I_n,
                jnp.where(mask_full[None, :], 0.0, A_full),
            )
            delta = jnp.linalg.solve(A_c, rhs_full)
            delta_u = delta[:n_u]
            delta_p_flat = delta[n_u:]
            delta_u = jnp.where(bc_mask, 0.0, delta_u)
            u_k = u_k + delta_u
            p_k = p_k + delta_p_flat.reshape(V.K, dim_p)
        return u_k, p_k
    return step


def make_probe_fn(V, probe_xy):
    """Build a closure that maps ``u_global -> probe velocity (n_probe, 2)``
    via point evaluation of the Piola-mapped BDM basis.

    ``probe_xy`` is an array of shape (n_probe, 2) in physical coords.
    The host code finds the enclosing element and the reference (r, s)
    once at build time; the closure is fully traceable.
    """
    k = V.order
    mesh = V.mesh
    probe_xy = np.asarray(probe_xy, dtype=float)
    n_probe = probe_xy.shape[0]

    # Locate each probe in an element via barycentric coords on the
    # affine map (xi, eta) -> physical (x, y). Mesh is straight-faced
    # outside the body-fit ring; probes are placed downstream where
    # elements are affine, so this is fine.
    VX = np.asarray(mesh.VX); VY = np.asarray(mesh.VY)
    EToV = np.asarray(mesh.EToV)
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
                # Reference triangle r in [-1, 1] x [-1, 1] convention:
                # r_aff, s_aff are barycentric with sum <= 1 in (0,1).
                # The HW convention maps (r, s) = (-1, -1) at v0,
                # (1, -1) at v1, (-1, 1) at v2.
                rs_ref[p, 0] = 2.0 * r_aff - 1.0
                rs_ref[p, 1] = 2.0 * s_aff - 1.0
                break

    elem_idx_j = jnp.asarray(elem_idx)
    r_j = jnp.asarray(rs_ref[:, 0])
    s_j = jnp.asarray(rs_ref[:, 1])

    # Pre-evaluate the reference BDM basis at every probe.
    phi_ref = bdm_k_basis_at_points(k, r_j, s_j)         # (n_probe, dim, 2)
    DF, _ = _element_jacobian_matrices(mesh)              # (K, 2, 2)
    J = mesh.J[0]                                         # (K,)
    # Gather DF and J per probe.
    DF_probe = DF[elem_idx_j]                             # (n_probe, 2, 2)
    J_probe = J[elem_idx_j]                               # (n_probe,)

    def probes(u_global):
        u_local = bdm1_gather(V.topology, u_global)
        u_at_probes = u_local[elem_idx_j]                # (n_probe, dim)
        # Piola map: phi_phys = (DF @ phi_ref) / J
        phi_phys = jnp.einsum(
            "pcb,plb->plc", DF_probe, phi_ref) / J_probe[:, None, None]
        u_eval = jnp.einsum("pl,plc->pc", u_at_probes, phi_phys)
        return u_eval                                     # (n_probe, 2)
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--kx", type=int, default=20)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--n-newton", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--nu-true", type=float, default=1.0e-3,
                    help="ground-truth viscosity")
    ap.add_argument("--nu-init", type=float, default=5.0e-3,
                    help="starting guess for nu (5x truth by default)")
    ap.add_argument("--lr", type=float, default=0.5,
                    help="Adam lr on log(nu); 0.5 makes O(1) jumps in nu")
    ap.add_argument("--n-train", type=int, default=15)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    V, u_bc, u_bc_vec, bc_mask, is_natural_face, _is_cyl = build_static(
        args.kx, args.ky, args.order,
    )
    n_u = V.total_dofs
    Re_true = U_MEAN * D_CYL / args.nu_true
    print(f"BDM_{args.order} cylinder smoke: K={V.K}, n_u={n_u}, "
          f"n_steps={args.n_steps}, dt={args.dt}, T={args.n_steps * args.dt}",
          flush=True)
    print(f"  truth: nu = {args.nu_true:.3e} (Re = {Re_true:.0f})",
          flush=True)
    print(f"  init : nu = {args.nu_init:.3e} (Re = "
          f"{U_MEAN * D_CYL / args.nu_init:.0f})", flush=True)

    step_fn = make_step_nu_param(
        V, u_bc, u_bc_vec, bc_mask, is_natural_face,
        tau_scale=args.tau_scale, n_newton=args.n_newton,
    )

    # Probe rake: 6 downstream points along the wake centreline at
    # x = CX + {1, 2, 3, 4, 5, 6} * D, all at y = CY.
    probe_xy = np.array(
        [[CX + i * D_CYL, CY] for i in (1, 2, 3, 4, 5, 6)],
        dtype=float,
    )
    n_probe = probe_xy.shape[0]
    print(f"  probes: {n_probe} points along y=CY", flush=True)
    probe_fn = make_probe_fn(V, probe_xy)

    def trajectory(nu):
        u_n = jnp.zeros(n_u)
        u_nm1 = jnp.zeros(n_u)
        probes_steps = []
        for s in range(args.n_steps):
            order_eff = 1 if s == 0 else 2
            u_new, _ = step_fn(u_n, u_nm1, args.dt, nu, order_eff)
            probes_steps.append(probe_fn(u_new))
            u_nm1, u_n = u_n, u_new
        return jnp.stack(probes_steps)        # (n_steps, n_probe, 2)

    # Ground truth: one forward pass at nu_true (no AD).
    print("computing ground truth ...", flush=True)
    t0 = time.time()
    y_obs = jax.jit(trajectory)(args.nu_true)
    y_obs.block_until_ready()
    print(f"  done in {time.time() - t0:.1f}s "
          f"(includes JIT compile)", flush=True)

    def loss_fn(log_nu):
        nu = jnp.exp(log_nu)
        y_pred = trajectory(nu)
        return jnp.sum((y_pred - y_obs)**2)

    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)
    traj_jit = jax.jit(trajectory)

    log_nu = jnp.asarray(float(np.log(args.nu_init)))
    L0 = float(loss_jit(log_nu))
    print(f"  initial loss = {L0:.4e}", flush=True)

    m_t = jnp.zeros_like(log_nu); v_t = jnp.zeros_like(log_nu)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    history = [dict(iter=0, nu=float(jnp.exp(log_nu)),
                    log_nu=float(log_nu),
                    loss=L0, grad=0.0)]
    best_loss = L0; best_log_nu = float(log_nu); best_iter = 0
    t0 = time.time()
    for it in range(1, args.n_train + 1):
        g = grad_fn(log_nu)
        m_t = beta1 * m_t + (1 - beta1) * g
        v_t = beta2 * v_t + (1 - beta2) * g**2
        m_hat = m_t / (1 - beta1**it)
        v_hat = v_t / (1 - beta2**it)
        log_nu = log_nu - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)
        L = float(loss_jit(log_nu))
        nu = float(jnp.exp(log_nu))
        history.append(dict(iter=int(it), nu=nu, log_nu=float(log_nu),
                            loss=L, grad=float(g)))
        if L < best_loss:
            best_loss = L; best_log_nu = float(log_nu); best_iter = it
        rel = abs(nu - args.nu_true) / args.nu_true
        print(f"  iter {it:3d}: nu = {nu:.4e}  (rel err {rel*100:5.1f}%)  "
              f"loss={L:.3e}  g={float(g):+.2e}",
              flush=True)
    print(f"\ntotal Adam wall: {time.time() - t0:.1f}s", flush=True)

    nu_final = float(jnp.exp(log_nu))
    nu_best = float(np.exp(best_log_nu))
    print(f"\nlast-iter nu = {nu_final:.4e}  "
          f"(truth {args.nu_true:.4e}, rel err "
          f"{100*abs(nu_final - args.nu_true)/args.nu_true:.2f}%)",
          flush=True)
    print(f"best-loss nu = {nu_best:.4e} at iter {best_iter} "
          f"(rel err {100*abs(nu_best - args.nu_true)/args.nu_true:.2f}%)",
          flush=True)
    # Use the best-loss iter as the reported recovery (early-stopping
    # convention; Adam can overshoot at large lr and the L2 mismatch
    # is the only signal we have on which iter solved the problem).
    log_nu = jnp.asarray(best_log_nu)
    nu_final = nu_best

    # Final probe trace comparison for the figure.
    y_final = traj_jit(jnp.exp(log_nu))
    y_init = traj_jit(jnp.asarray(np.log(args.nu_init)))

    stem = f"inverse_nu_cylinder_k{args.order}"
    out_npz = args.out_dir / f"{stem}.npz"
    np.savez_compressed(
        out_npz,
        nu_true=args.nu_true, nu_init=args.nu_init,
        nu_final=nu_final, n_steps=args.n_steps, dt=args.dt,
        times=(np.arange(args.n_steps) + 1) * args.dt,
        probe_xy=probe_xy,
        y_obs=np.asarray(y_obs),
        y_final=np.asarray(y_final),
        y_init=np.asarray(y_init),
        nu_hist=np.array([r["nu"] for r in history]),
        loss_hist=np.array([r["loss"] for r in history]),
    )
    print(f"  wrote {out_npz}", flush=True)

    out_json = args.out_dir / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky, K=int(V.K),
            nu_true=args.nu_true, nu_init=args.nu_init,
            nu_final=nu_final,
            n_steps=args.n_steps, dt=args.dt,
            n_newton=args.n_newton, tau_scale=args.tau_scale,
            lr=args.lr, n_train=args.n_train,
            history=history,
        ), f, indent=2)
    print(f"  wrote {out_json}", flush=True)

    # Plot.
    t_axis = (np.arange(args.n_steps) + 1) * args.dt
    iters = [r["iter"] for r in history]
    nus = [r["nu"] for r in history]
    losses = [r["loss"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    axes[0, 0].semilogy(iters, nus, "b-o", ms=4, label="recovered")
    axes[0, 0].axhline(args.nu_true, color="r", ls="--", lw=1,
                       label="truth")
    axes[0, 0].axhline(args.nu_init, color="gray", ls=":", lw=1,
                       label="init")
    axes[0, 0].set_xlabel("Adam iter"); axes[0, 0].set_ylabel(r"$\nu$")
    axes[0, 0].set_title("Recovered viscosity")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3, which="both")

    axes[0, 1].semilogy(iters, losses, "k-")
    axes[0, 1].set_xlabel("Adam iter"); axes[0, 1].set_ylabel("loss")
    axes[0, 1].set_title(r"L2 mismatch on probe traces")
    axes[0, 1].grid(alpha=0.3, which="both")

    # Probe-trace comparison: u_x at probe 0 (closest to cylinder)
    # vs time, for truth / init / final.
    p_show = 0
    y_obs_np = np.asarray(y_obs)
    axes[1, 0].plot(t_axis, y_obs_np[:, p_show, 0], "k-", lw=2,
                    label="truth")
    axes[1, 0].plot(t_axis, np.asarray(y_init)[:, p_show, 0], "r--",
                    lw=1.2, label="init")
    axes[1, 0].plot(t_axis, np.asarray(y_final)[:, p_show, 0], "b:",
                    lw=1.5, label="recovered")
    axes[1, 0].set_xlabel("t"); axes[1, 0].set_ylabel(r"$u_x$")
    axes[1, 0].set_title(
        f"Probe 0 trace ($x = {probe_xy[p_show, 0]:.2f}$, $y = "
        f"{probe_xy[p_show, 1]:.2f}$)"
    )
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # All probes, final-time u_x value.
    final_obs = y_obs_np[-1, :, 0]
    final_init = np.asarray(y_init)[-1, :, 0]
    final_recovered = np.asarray(y_final)[-1, :, 0]
    pts = np.arange(n_probe) + 1
    axes[1, 1].plot(pts, final_obs, "k-o", ms=6, label="truth")
    axes[1, 1].plot(pts, final_init, "rx", ms=8, label="init")
    axes[1, 1].plot(pts, final_recovered, "b+", ms=10, label="recovered")
    axes[1, 1].set_xlabel("probe index (downstream of cylinder)")
    axes[1, 1].set_ylabel(r"$u_x(t = T_{\rm final})$")
    axes[1, 1].set_title("Final-time probe profile")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Inverse nu: BDM_{args.order} cylinder, "
        f"nu_init = {args.nu_init:.1e}, nu_truth = {args.nu_true:.1e}, "
        f"final = {nu_final:.3e}"
    )
    out_png = args.out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
