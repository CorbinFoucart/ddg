"""Spatial + temporal convergence studies for the INS schemes.

Two benchmarks with analytical solutions:

  * **Taylor-Green vortex** (time-dependent, decaying):
    ``u = (-cos x sin y, sin x cos y) e^{-2νt}``, used to measure
    *temporal* convergence — vary ``dt`` on a fixed mesh, fit the log
    slope of L2 velocity error vs ``dt``. Expected order: 2 (BDF2 +
    explicit-extrapolated advection at consistent splitting / dual
    splitting, after the BDF1 first step).

  * **Kovasznay flow** (steady): used to measure *spatial* convergence
    — vary mesh resolution at fixed ``dt`` and final time, fit log
    slope of error vs ``h``. Expected order: ``N+1`` for the SIPG
    discretisation.

Reports a markdown table + JSON for each (scheme × benchmark × axis).
Run on CPU first to validate; the grids are small enough that CPU
finishes in minutes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OUTFLOW, BC_INS_WALL,
    BDF1, BDF2, build_ins_operators, ns_advection_2d,
    ns_step_2d, ns_step_consistent_splitting_2d,
)
from ddg.dg2d.coupled_ns import coupled_ns_step_2d
from ddg.dg2d.ins_metrics import rel_l2_velocity_error


def _build_rect_mesh(xmin, xmax, ymin, ymax, Kx, Ky, N):
    VX, VY, EToV = uniform_rectangle_mesh_2d(xmin, xmax, ymin, ymax, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def _bc_three_inflow_one_outflow(mesh, outflow_normal: str):
    Nfaces, K = mesh.Nfaces, mesh.K
    nx = np.asarray(mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    ny = np.asarray(mesh.ny.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T
    is_out = {
        "+x": nx > 0.5, "-x": nx < -0.5,
        "+y": ny > 0.5, "-y": ny < -0.5,
    }[outflow_normal]
    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    bc = np.where(is_bdy & ~is_out, BC_INS_INFLOW, bc)
    bc = np.where(is_bdy & is_out, BC_INS_OUTFLOW, bc)
    return jnp.asarray(bc.T)


# ---------------------------------------------------------------------------
# Taylor-Green analytics (same as in test/test_incompressible_ns_2d.py)
# ---------------------------------------------------------------------------

def _taylor_green(mesh, t, nu):
    x, y = mesh.x, mesh.y
    e_v = jnp.exp(-2.0 * nu * t)
    e_p = jnp.exp(-4.0 * nu * t)
    ux = -jnp.cos(x) * jnp.sin(y) * e_v
    uy = jnp.sin(x) * jnp.cos(y) * e_v
    p = -0.25 * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y)) * e_p
    return ux, uy, p


def _taylor_green_face_data(mesh, t, nu):
    """Velocity Neumann BC + pressure Dirichlet at outflow."""
    x_M = mesh.x.T.reshape(-1)[mesh.vmapM]
    y_M = mesh.y.T.reshape(-1)[mesh.vmapM]
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    e_v = jnp.exp(-2.0 * nu * t)
    dux_dx = jnp.sin(x_M) * jnp.sin(y_M) * e_v
    dux_dy = -jnp.cos(x_M) * jnp.cos(y_M) * e_v
    duy_dx = jnp.cos(x_M) * jnp.cos(y_M) * e_v
    duy_dy = -jnp.sin(x_M) * jnp.sin(y_M) * e_v
    q_ux = (dux_dx * nx_flat + dux_dy * ny_flat).reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F"
    )
    q_uy = (duy_dx * nx_flat + duy_dy * ny_flat).reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F"
    )
    _, _, p_dir = _taylor_green(mesh, t, nu)
    return q_ux, q_uy, p_dir


# ---------------------------------------------------------------------------
# Kovasznay analytics
# ---------------------------------------------------------------------------

def _kovasznay(mesh, nu):
    Re = 1.0 / nu
    lam = 0.5 * Re - np.sqrt(0.25 * Re ** 2 + 4 * np.pi ** 2)
    x, y = mesh.x, mesh.y
    e = jnp.exp(lam * x)
    ux = 1.0 - e * jnp.cos(2 * jnp.pi * y)
    uy = (lam / (2 * jnp.pi)) * e * jnp.sin(2 * jnp.pi * y)
    p = 0.5 * (1.0 - jnp.exp(2 * lam * x))
    return ux, uy, p, lam


def _kovasznay_face_data(mesh, nu, lam):
    x_M = mesh.x.T.reshape(-1)[mesh.vmapM]
    y_M = mesh.y.T.reshape(-1)[mesh.vmapM]
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    e = jnp.exp(lam * x_M)
    twopi = 2 * jnp.pi
    dux_dx = -lam * e * jnp.cos(twopi * y_M)
    dux_dy = twopi * e * jnp.sin(twopi * y_M)
    duy_dx = (lam ** 2 / twopi) * e * jnp.sin(twopi * y_M)
    duy_dy = lam * e * jnp.cos(twopi * y_M)
    q_ux = (dux_dx * nx_flat + dux_dy * ny_flat).reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F"
    )
    q_uy = (duy_dx * nx_flat + duy_dy * ny_flat).reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F"
    )
    _, _, p_dir, _ = _kovasznay(mesh, nu)
    return q_ux, q_uy, p_dir


# ---------------------------------------------------------------------------
# Scheme runners (uniform interface for the sweep)
# ---------------------------------------------------------------------------

def _step_dual(mesh, ins_bc, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
                bc_ux, bc_uy, q_ux_neu, q_uy_neu, p_dir, ops, dt, nu, bdf):
    return ns_step_2d(
        mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
        dt=dt, nu=nu, bdf=bdf, ops=ops, ins_bc_types=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
        p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
    )


def _step_consistent(mesh, ins_bc, ux, uy, ux_old, uy_old,
                      bc_ux, bc_uy, q_ux_neu, q_uy_neu, p_dir, ops, dt, nu, bdf):
    ux_new, uy_new, p = ns_step_consistent_splitting_2d(
        mesh, ux, uy, ux_old, uy_old,
        dt=dt, nu=nu, bdf=bdf, ops=ops, ins_bc_types=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
        p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
    )
    Np_face = mesh.Nfp * mesh.Nfaces
    return ux_new, uy_new, p, jnp.zeros_like(ux), jnp.zeros_like(uy), jnp.zeros((Np_face, mesh.K))


def _step_coupled(mesh, ins_bc, ux, uy, ux_old, uy_old,
                   bc_ux, bc_uy, q_ux_neu, q_uy_neu, p_dir, dt, nu, bdf,
                   tau_div=1.0, tau_cont=1.0, pressure_eps=1e-3):
    ux_new, uy_new, p, _ = coupled_ns_step_2d(
        mesh, ux, uy, ux_old, uy_old,
        dt=dt, nu=nu, bdf=bdf, ins_bc_types=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
        p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
        tau_div=tau_div, tau_cont=tau_cont, pressure_eps=pressure_eps,
    )
    Np_face = mesh.Nfp * mesh.Nfaces
    return ux_new, uy_new, p, jnp.zeros_like(ux), jnp.zeros_like(uy), jnp.zeros((Np_face, mesh.K))


# ---------------------------------------------------------------------------
# Sweep drivers
# ---------------------------------------------------------------------------

def run_tg_at(scheme: str, Kx: int, N: int, dt: float, T: float, nu: float) -> float:
    """Run Taylor-Green to T with given (Kx, N, dt); return rel L2 err."""
    mesh = _build_rect_mesh(0.0, 2 * np.pi, 0.0, 2 * np.pi, Kx, Kx, N)
    ins_bc = _bc_three_inflow_one_outflow(mesh, "+x")
    ux, uy, _ = _taylor_green(mesh, 0.0, nu)
    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=ux, bc_uy=uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    ops_bdf1 = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc,
    )
    ops_bdf2 = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF2.g0, ins_bc_types=ins_bc,
    )
    n_steps = int(round(T / dt))
    for step in range(1, n_steps + 1):
        t_np1 = step * dt
        bc_ux, bc_uy, _ = _taylor_green(mesh, t_np1, nu)
        q_ux_neu, q_uy_neu, p_dir = _taylor_green_face_data(mesh, t_np1, nu)
        bdf = BDF1 if step == 1 else BDF2
        ops = ops_bdf1 if step == 1 else ops_bdf2
        if scheme == "dual":
            ux, uy, p, Nux_old, Nuy_old, dpdn_old = _step_dual(
                mesh, ins_bc, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
                bc_ux, bc_uy, q_ux_neu, q_uy_neu, p_dir, ops, dt, nu, bdf,
            )
        elif scheme == "consistent":
            ux, uy, p, _, _, _ = _step_consistent(
                mesh, ins_bc, ux, uy, ux_old, uy_old,
                bc_ux, bc_uy, q_ux_neu, q_uy_neu, p_dir, ops, dt, nu, bdf,
            )
        elif scheme == "coupled":
            ux, uy, p, _, _, _ = _step_coupled(
                mesh, ins_bc, ux, uy, ux_old, uy_old,
                bc_ux, bc_uy, q_ux_neu, q_uy_neu, p_dir, dt, nu, bdf,
            )
        ux_old, uy_old = ux, uy
    ux_ex, uy_ex, _ = _taylor_green(mesh, T, nu)
    return rel_l2_velocity_error(mesh, ux, uy, ux_ex, uy_ex)


def run_kov_at(scheme: str, Kx: int, N: int, dt: float, T: float, nu: float) -> float:
    """Run Kovasznay to T; return rel L2 err."""
    mesh = _build_rect_mesh(-0.5, 1.5, -0.5, 1.5, Kx, Kx, N)
    ins_bc = _bc_three_inflow_one_outflow(mesh, "+x")
    ux, uy, _, lam = _kovasznay(mesh, nu)
    bc_ux, bc_uy = ux, uy
    q_ux_neu, q_uy_neu, p_dir = _kovasznay_face_data(mesh, nu, lam)
    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    ops_bdf1 = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc,
    )
    ops_bdf2 = build_ins_operators(
        mesh, dt=dt, nu=nu, g0=BDF2.g0, ins_bc_types=ins_bc,
    )
    n_steps = int(round(T / dt))
    for step in range(1, n_steps + 1):
        bdf = BDF1 if step == 1 else BDF2
        ops = ops_bdf1 if step == 1 else ops_bdf2
        if scheme == "dual":
            ux, uy, _, Nux_old, Nuy_old, dpdn_old = _step_dual(
                mesh, ins_bc, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
                bc_ux, bc_uy, q_ux_neu, q_uy_neu, p_dir, ops, dt, nu, bdf,
            )
        elif scheme == "consistent":
            ux, uy, _, _, _, _ = _step_consistent(
                mesh, ins_bc, ux, uy, ux_old, uy_old,
                bc_ux, bc_uy, q_ux_neu, q_uy_neu, p_dir, ops, dt, nu, bdf,
            )
        elif scheme == "coupled":
            ux, uy, _, _, _, _ = _step_coupled(
                mesh, ins_bc, ux, uy, ux_old, uy_old,
                bc_ux, bc_uy, q_ux_neu, q_uy_neu, p_dir, dt, nu, bdf,
            )
        ux_old, uy_old = ux, uy
    ux_ex, uy_ex, _, _ = _kovasznay(mesh, nu)
    return rel_l2_velocity_error(mesh, ux, uy, ux_ex, uy_ex)


def fit_log_slope(xs: list[float], ys: list[float]) -> float:
    """Slope of log-log fit. Returns ``np.nan`` for invalid inputs."""
    if len(xs) < 2 or any(y <= 0 for y in ys):
        return float("nan")
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main(args):
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out = {"temporal": {}, "spatial": {}}

    # Temporal convergence on Taylor-Green
    if args.temporal:
        print("=" * 60)
        print("Temporal convergence (Taylor-Green, fixed mesh)")
        print("=" * 60)
        T = 0.1
        nu = 0.1
        Kx, N = 8, 4
        dts = [0.04, 0.02, 0.01, 0.005]
        for scheme in args.schemes:
            errs = []
            for dt in dts:
                t0 = time.time()
                err = run_tg_at(scheme, Kx=Kx, N=N, dt=dt, T=T, nu=nu)
                wall = time.time() - t0
                errs.append(err)
                print(f"  {scheme:11s} dt={dt:>6.3f}: err = {err:.3e}  "
                      f"({wall:.1f}s)")
            slope = fit_log_slope(dts, errs)
            print(f"  {scheme:11s} → temporal order ≈ {slope:.2f}")
            out["temporal"][scheme] = {
                "dt": dts, "err": errs, "order": slope,
                "Kx": Kx, "N": N, "T": T, "nu": nu,
            }
        print()

    # Spatial convergence on Kovasznay
    if args.spatial:
        print("=" * 60)
        print("Spatial convergence (Kovasznay, fixed dt)")
        print("=" * 60)
        nu = 0.1
        T = 0.1
        dt = 0.005
        configs = [(4, 4), (6, 4), (8, 4)]  # (Kx, N)
        for scheme in args.schemes:
            errs, hs = [], []
            for Kx, N in configs:
                h = 2.0 / Kx
                t0 = time.time()
                err = run_kov_at(scheme, Kx=Kx, N=N, dt=dt, T=T, nu=nu)
                wall = time.time() - t0
                errs.append(err)
                hs.append(h)
                print(f"  {scheme:11s} Kx={Kx:2d} N={N}: h={h:.3f}, "
                      f"err = {err:.3e}  ({wall:.1f}s)")
            slope = fit_log_slope(hs, errs)
            print(f"  {scheme:11s} → spatial order ≈ {slope:.2f} "
                  f"(expected ≈ N+1 = {N+1})")
            out["spatial"][scheme] = {
                "h": hs, "err": errs, "order": slope,
                "configs": [list(c) for c in configs], "dt": dt, "T": T, "nu": nu,
            }
        print()

    json_path = out_dir / "ns_convergence.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {json_path}")

    md_path = out_dir / "ns_convergence.md"
    with open(md_path, "w") as f:
        f.write(_render_md(out))
    print(f"Wrote {md_path}")


def _render_md(out: dict) -> str:
    lines = ["# INS convergence study\n"]
    if out.get("temporal"):
        lines.append("## Temporal convergence (Taylor-Green decay)\n")
        any_run = next(iter(out["temporal"].values()))
        lines.append(f"Mesh fixed at Kx={any_run['Kx']}, N={any_run['N']}; "
                     f"T = {any_run['T']}, ν = {any_run['nu']}.\n")
        lines.append("| scheme | " + " | ".join(
            f"dt={dt}" for dt in any_run["dt"]
        ) + " | observed order |")
        lines.append("|" + "---|" * (len(any_run["dt"]) + 2))
        for scheme, data in out["temporal"].items():
            row = [scheme]
            for e in data["err"]:
                row.append(f"{e:.2e}")
            row.append(f"{data['order']:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    if out.get("spatial"):
        lines.append("## Spatial convergence (Kovasznay)\n")
        any_run = next(iter(out["spatial"].values()))
        lines.append(f"dt = {any_run['dt']}, T = {any_run['T']}, "
                     f"ν = {any_run['nu']}.\n")
        cols = [f"Kx={c[0]} N={c[1]}" for c in any_run["configs"]]
        lines.append("| scheme | " + " | ".join(cols) + " | observed order |")
        lines.append("|" + "---|" * (len(cols) + 2))
        for scheme, data in out["spatial"].items():
            row = [scheme]
            for e in data["err"]:
                row.append(f"{e:.2e}")
            row.append(f"{data['order']:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemes", nargs="+",
        default=["dual", "consistent"],
        choices=["dual", "consistent", "coupled"],
    )
    parser.add_argument("--temporal", action="store_true",
                        help="Run temporal convergence (Taylor-Green).")
    parser.add_argument("--spatial", action="store_true",
                        help="Run spatial convergence (Kovasznay).")
    args = parser.parse_args()
    if not args.temporal and not args.spatial:
        args.temporal = True
        args.spatial = True
    main(args)
