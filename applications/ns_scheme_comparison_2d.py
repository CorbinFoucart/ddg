"""Compare INS time-integration schemes on the flow-past-square demo.

Runs the same channel-with-square-hole setup through any subset of:

  * **dual_lu / dual_cg** — Hesthaven dual splitting, direct LU or
    matrix-free CG on the pressure Poisson + viscous Helmholtz.
  * **consistent_lu / consistent_cg** — Fehn 2018 consistent splitting,
    LU or CG.
  * **coupled_lu / coupled_gmres** — monolithic saddle-point
    (linearly-implicit convection + divergence/continuity penalty),
    direct LU or matrix-free GMRES.

Records per-step physics (KE, enstrophy, divergence L2, mass-flux
balance, max-norms) and performance (wall time, throughput) via
:mod:`ddg.dg2d.run_metrics`. Writes a tidy bundle into ``output/``:

  * ``ns_scheme_comparison_summary.json``  — run-level totals
  * ``ns_scheme_comparison_timeseries.csv`` — one row per step
  * ``ns_scheme_comparison_report.md``     — auto-generated summary card
  * ``ns_scheme_comparison.png``           — side-by-side vorticity

Run on CPU::

    JAX_PLATFORMS=cpu python applications/ns_scheme_comparison_2d.py

Run on GPU::

    JAX_PLATFORMS=cuda python applications/ns_scheme_comparison_2d.py --large
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, rectangle_with_square_hole_mesh_2d
from ddg.dg2d.maxwell import curl_z_2d
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OBSTACLE, BC_INS_OUTFLOW,
    BC_INS_WALL,
    BDF1, BDF2, build_ins_operators, ns_advection_2d,
    ns_step_2d, ns_step_consistent_splitting_2d,
)
from ddg.dg2d.coupled_ns import (
    coupled_ns_step_2d, IterativeSolverConfig,
)
from ddg.dg2d.sipg import IterativeConfig as SIPGIterConfig
from ddg.dg2d.run_metrics import RunReporter, SchemeRun


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

@dataclass
class ProblemConfig:
    x_range: tuple = (0.0, 6.0)
    y_range: tuple = (0.0, 2.0)
    hole_x: tuple = (1.5, 2.0)
    hole_y: tuple = (0.75, 1.25)
    Kx: int = 12
    Ky: int = 4
    N: int = 2
    U_inf: float = 1.0
    nu: float = 0.02
    T_final: float = 2.0
    dt: float = 0.04

    @property
    def H(self) -> float:
        return self.y_range[1] - self.y_range[0]


def build_setup(cfg: ProblemConfig):
    VX, VY, EToV = rectangle_with_square_hole_mesh_2d(
        *cfg.x_range, *cfg.y_range, cfg.Kx, cfg.Ky,
        *cfg.hole_x, *cfg.hole_y,
    )
    mesh = build_mesh_2d(VX, VY, EToV, cfg.N)
    ins_bc = _tag_boundary_faces(mesh, cfg)
    bc_ux, bc_uy = _build_bc_fields(mesh, ins_bc, cfg)
    inflow_mask, outflow_mask = _bc_masks_mapB(mesh, ins_bc)
    return mesh, ins_bc, bc_ux, bc_uy, inflow_mask, outflow_mask


def _tag_boundary_faces(mesh, cfg: ProblemConfig):
    Nfaces, K = mesh.Nfaces, mesh.K
    nx_face = np.asarray(mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    ny_face = np.asarray(mesh.ny.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    Fmask = np.asarray(mesh.Fmask)
    x_np, y_np = np.asarray(mesh.x), np.asarray(mesh.y)
    x_face = np.zeros((Nfaces, K))
    y_face = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        x_face[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T

    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    inflow = is_bdy & (np.abs(nx_face) > 0.5) & (x_face < cfg.x_range[0] + 0.1)
    outflow = is_bdy & (np.abs(nx_face) > 0.5) & (x_face > cfg.x_range[1] - 0.1)
    walls = is_bdy & (np.abs(ny_face) > 0.5) & (
        (y_face < cfg.y_range[0] + 0.1) | (y_face > cfg.y_range[1] - 0.1)
    )
    obstacle = is_bdy & (~inflow) & (~outflow) & (~walls)
    bc = np.where(inflow, BC_INS_INFLOW, bc)
    bc = np.where(outflow, BC_INS_OUTFLOW, bc)
    bc = np.where(walls, BC_INS_WALL, bc)
    bc = np.where(obstacle, BC_INS_OBSTACLE, bc)
    return jnp.asarray(bc.T)


def _build_bc_fields(mesh, ins_bc, cfg: ProblemConfig):
    yfrac = (mesh.y - cfg.y_range[0]) / cfg.H
    bc_ux = 4.0 * cfg.U_inf * yfrac * (1.0 - yfrac)
    bc_uy = jnp.zeros_like(mesh.y)
    Fmask = np.asarray(mesh.Fmask)
    K, Np, Nfp, Nfaces = mesh.K, mesh.Np, mesh.Nfp, mesh.Nfaces
    bc_per_face = np.asarray(ins_bc).T
    wall_or_obs = (bc_per_face == BC_INS_WALL) | (bc_per_face == BC_INS_OBSTACLE)
    mask = np.zeros((Np, K), dtype=bool)
    for f in range(Nfaces):
        for k in np.where(wall_or_obs[f, :])[0]:
            mask[Fmask[:, f], k] = True
    mask = jnp.asarray(mask)
    return jnp.where(mask, 0.0, bc_ux), jnp.where(mask, 0.0, bc_uy)


def _bc_masks_mapB(mesh, ins_bc) -> tuple[jax.Array, jax.Array]:
    """Build boolean masks over ``mapB`` selecting inflow and outflow nodes.

    Used by the mass-flux balance metric.
    """
    bc_per_face = np.asarray(ins_bc).T            # (Nfaces, K)
    Nfp, Nfaces, K = mesh.Nfp, mesh.Nfaces, mesh.K
    bc_per_node = np.repeat(bc_per_face[None, :, :], Nfp, axis=0)
    bc_flat = bc_per_node.reshape(-1, order="F")  # (Nfp*Nfaces*K,)
    bc_at_mapB = bc_flat[np.asarray(mesh.mapB)]
    return (
        jnp.asarray(bc_at_mapB == BC_INS_INFLOW),
        jnp.asarray(bc_at_mapB == BC_INS_OUTFLOW),
    )


# ---------------------------------------------------------------------------
# Per-scheme runners (instrumented with the reporter)
# ---------------------------------------------------------------------------

def run_dual_splitting(cfg, mesh, ins_bc, bc_ux, bc_uy,
                       reporter: RunReporter, *, iterative: bool):
    name = "dual_cg" if iterative else "dual_lu"
    iter_cfg = SIPGIterConfig(method="cg", tol=1e-9, maxiter=2000) if iterative else None

    setup_start = time.time()
    ops_bdf1 = build_ins_operators(
        mesh, dt=cfg.dt, nu=cfg.nu, g0=BDF1.g0, ins_bc_types=ins_bc,
        skip_factor=iterative,
    )
    ops_bdf2 = build_ins_operators(
        mesh, dt=cfg.dt, nu=cfg.nu, g0=BDF2.g0, ins_bc_types=ins_bc,
        skip_factor=iterative,
    )
    setup_t = time.time() - setup_start

    ux, uy = bc_ux, bc_uy
    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))

    n_steps = int(round(cfg.T_final / cfg.dt))
    run = reporter.new_run(name=name, n_steps=n_steps)
    run.setup_time = setup_t

    run_start = time.time()
    for step in range(1, n_steps + 1):
        step_start = time.time()
        bdf = BDF1 if step == 1 else BDF2
        ops = ops_bdf1 if step == 1 else ops_bdf2
        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=cfg.dt, nu=cfg.nu, bdf=bdf, ops=ops, ins_bc_types=ins_bc,
            bc_ux=bc_ux, bc_uy=bc_uy, iterative=iter_cfg,
        )
        wall_dt = time.time() - step_start
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n
        reporter.record_step(run, step, step * cfg.dt, wall_dt, ux, uy)

    run.wall_time_total = time.time() - run_start
    run.final_ux = ux
    run.final_uy = uy
    run.final_p = p
    return run


def run_consistent_splitting(cfg, mesh, ins_bc, bc_ux, bc_uy,
                              reporter: RunReporter, *, iterative: bool):
    name = "consistent_cg" if iterative else "consistent_lu"
    iter_cfg = SIPGIterConfig(method="cg", tol=1e-9, maxiter=2000) if iterative else None

    setup_start = time.time()
    ops_bdf1 = build_ins_operators(
        mesh, dt=cfg.dt, nu=cfg.nu, g0=BDF1.g0, ins_bc_types=ins_bc,
        skip_factor=iterative,
    )
    ops_bdf2 = build_ins_operators(
        mesh, dt=cfg.dt, nu=cfg.nu, g0=BDF2.g0, ins_bc_types=ins_bc,
        skip_factor=iterative,
    )
    setup_t = time.time() - setup_start

    ux, uy = bc_ux, bc_uy
    ux_old, uy_old = ux, uy
    n_steps = int(round(cfg.T_final / cfg.dt))
    run = reporter.new_run(name=name, n_steps=n_steps)
    run.setup_time = setup_t

    run_start = time.time()
    for step in range(1, n_steps + 1):
        step_start = time.time()
        bdf = BDF1 if step == 1 else BDF2
        ops = ops_bdf1 if step == 1 else ops_bdf2
        ux_new, uy_new, p = ns_step_consistent_splitting_2d(
            mesh, ux, uy, ux_old, uy_old,
            dt=cfg.dt, nu=cfg.nu, bdf=bdf, ops=ops, ins_bc_types=ins_bc,
            bc_ux=bc_ux, bc_uy=bc_uy, iterative=iter_cfg,
        )
        wall_dt = time.time() - step_start
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        reporter.record_step(run, step, step * cfg.dt, wall_dt, ux, uy)
    run.wall_time_total = time.time() - run_start
    run.final_ux = ux
    run.final_uy = uy
    run.final_p = p
    return run


def run_coupled(cfg, mesh, ins_bc, bc_ux, bc_uy,
                reporter: RunReporter, *,
                iterative: bool,
                tau_div: float = 1.0,
                tau_cont: float = 1.0,
                pressure_eps: float = 1e-3):
    name = "coupled_gmres" if iterative else "coupled_lu"
    n_steps = int(round(cfg.T_final / cfg.dt))
    run = reporter.new_run(name=name, n_steps=n_steps)
    run.setup_time = 0.0  # coupled factors per step

    gmres_cfg = IterativeSolverConfig(
        method="gmres", tol=1e-7, maxiter=500, restart=30,
    ) if iterative else None

    ux, uy = bc_ux, bc_uy
    ux_old, uy_old = ux, uy
    prev_sol = (ux, uy, jnp.zeros_like(ux))

    run_start = time.time()
    for step in range(1, n_steps + 1):
        step_start = time.time()
        bdf = BDF1 if step == 1 else BDF2
        ux_new, uy_new, p, _ = coupled_ns_step_2d(
            mesh, ux, uy, ux_old, uy_old,
            dt=cfg.dt, nu=cfg.nu, bdf=bdf, ins_bc_types=ins_bc,
            bc_ux=bc_ux, bc_uy=bc_uy,
            tau_div=tau_div, tau_cont=tau_cont, pressure_eps=pressure_eps,
            iterative=gmres_cfg, x0_iterative=prev_sol if iterative else None,
        )
        wall_dt = time.time() - step_start
        prev_sol = (ux_new, uy_new, p)
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        reporter.record_step(run, step, step * cfg.dt, wall_dt, ux, uy)
    run.wall_time_total = time.time() - run_start
    run.final_ux = ux
    run.final_uy = uy
    run.final_p = p
    return run


# ---------------------------------------------------------------------------
# Vorticity figure
# ---------------------------------------------------------------------------

def render_vorticity_figure(cfg, mesh, runs, out_path: Path):
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    levels = np.linspace(-5, 5, 31)
    for ax, r in zip(axes, runs):
        omega = curl_z_2d(mesh, r.final_ux, r.final_uy)
        pz = np.asarray(omega).T.reshape(-1)
        ax.tricontourf(px, py, pz, levels=levels, cmap="RdBu_r", extend="both")
        ax.add_patch(plt.Rectangle(
            (cfg.hole_x[0], cfg.hole_y[0]),
            cfg.hole_x[1] - cfg.hole_x[0],
            cfg.hole_y[1] - cfg.hole_y[0],
            facecolor="black", edgecolor="black",
        ))
        ax.set_aspect("equal")
        ax.set_title(
            f"{r.name}    wall={r.wall_time_total:.1f}s  "
            f"({r.per_step_s*1000:.0f} ms/step)"
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main(args):
    cfg = ProblemConfig()
    if args.large:
        cfg = dataclasses.replace(cfg, Kx=24, Ky=8, N=3, T_final=4.0, dt=0.02)
    # Explicit overrides (used by the scaling sweep). Any non-None
    # kwarg replaces the preset / --large value.
    overrides = {
        k: v for k, v in {
            "Kx": args.Kx, "Ky": args.Ky, "N": args.N,
            "T_final": args.T, "dt": args.dt,
        }.items() if v is not None
    }
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    out_dir = Path(__file__).resolve().parent.parent / "output"
    if args.output_suffix:
        out_dir = out_dir
    out_dir.mkdir(exist_ok=True)

    print(f"Mesh: Kx={cfg.Kx} Ky={cfg.Ky} N={cfg.N}")
    print(f"Time: T={cfg.T_final}s dt={cfg.dt} "
          f"(n_steps={int(round(cfg.T_final / cfg.dt))})")
    print(f"Re ≈ {cfg.U_inf * (cfg.hole_y[1] - cfg.hole_y[0]) / cfg.nu:.0f}")
    print(f"JAX platform: {jax.devices()[0]}\n")

    mesh, ins_bc, bc_ux, bc_uy, inflow_mask, outflow_mask = build_setup(cfg)
    n_dofs_per_field = mesh.K * mesh.Np
    print(f"Mesh: K={mesh.K}  Np={mesh.Np}  velocity DOFs = {n_dofs_per_field}\n")

    reporter = RunReporter(
        out_dir=out_dir, name="ns_scheme_comparison", mesh=mesh,
        n_dofs_per_field=n_dofs_per_field,
        metadata={
            "Kx": cfg.Kx, "Ky": cfg.Ky, "N": cfg.N,
            "T_final": cfg.T_final, "dt": cfg.dt,
            "U_inf": cfg.U_inf, "nu": cfg.nu,
            "Re_approx": cfg.U_inf * (cfg.hole_y[1] - cfg.hole_y[0]) / cfg.nu,
            "platform": str(jax.devices()[0]),
            "K": mesh.K, "Np": mesh.Np,
            "n_dofs_per_field": n_dofs_per_field,
        },
        record_every=args.record_every,
        inflow_mapB_mask=inflow_mask,
        outflow_mapB_mask=outflow_mask,
    )

    schemes = []
    sched = {
        "dual_lu": lambda r: run_dual_splitting(
            cfg, mesh, ins_bc, bc_ux, bc_uy, r, iterative=False),
        "dual_cg": lambda r: run_dual_splitting(
            cfg, mesh, ins_bc, bc_ux, bc_uy, r, iterative=True),
        "consistent_lu": lambda r: run_consistent_splitting(
            cfg, mesh, ins_bc, bc_ux, bc_uy, r, iterative=False),
        "consistent_cg": lambda r: run_consistent_splitting(
            cfg, mesh, ins_bc, bc_ux, bc_uy, r, iterative=True),
        "coupled_lu": lambda r: run_coupled(
            cfg, mesh, ins_bc, bc_ux, bc_uy, r, iterative=False),
        "coupled_gmres": lambda r: run_coupled(
            cfg, mesh, ins_bc, bc_ux, bc_uy, r, iterative=True),
    }
    for s in args.schemes:
        print(f"--- {s} ---")
        run = sched[s](reporter)
        last = run.step_records[-1] if run.step_records else None
        if last:
            print(f"  wall = {run.wall_time_total:.1f}s "
                  f"({run.per_step_s * 1000:.0f} ms/step) "
                  f"setup = {run.setup_time:.1f}s")
            print(f"  final: KE={last.ke:.4f}  enstrophy={last.enstrophy:.3f}  "
                  f"‖∇·U‖/‖U‖={last.divu_l2_rel:.2e}  "
                  f"max|U|={last.max_speed:.3f}  max|ω|={last.max_vort:.2f}  "
                  f"mass-bal={last.mass_balance:.2e}")
        print()

    suffix = args.output_suffix or ""
    name_base = "ns_scheme_comparison" + (f"_{suffix}" if suffix else "")
    # The reporter wrote files using the default name; rename if a
    # suffix was requested.
    if suffix:
        for ext in ("_summary.json", "_timeseries.csv", "_report.md"):
            src = out_dir / f"ns_scheme_comparison{ext}"
            dst = out_dir / f"{name_base}{ext}"
            if src.exists():
                src.rename(dst)
    render_vorticity_figure(cfg, mesh, reporter._runs,
                            out_dir / f"{name_base}.png")
    if not suffix:
        reporter.write()
    else:
        # Rewrite under suffixed name.
        reporter.name = name_base
        reporter.write()
    print(f"Wrote {out_dir / f'{name_base}_summary.json'}")
    print(f"Wrote {out_dir / f'{name_base}_timeseries.csv'}")
    print(f"Wrote {out_dir / f'{name_base}_report.md'}")
    print(f"Wrote {out_dir / f'{name_base}.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemes", nargs="+",
        default=["dual_lu", "consistent_lu", "coupled_lu"],
        choices=[
            "dual_lu", "dual_cg",
            "consistent_lu", "consistent_cg",
            "coupled_lu", "coupled_gmres",
        ],
    )
    parser.add_argument("--large", action="store_true",
                        help="Bigger mesh + longer T (GPU-friendly).")
    parser.add_argument("--record-every", type=int, default=1,
                        help="Record physics metrics every N steps.")
    # Explicit mesh / time overrides for the scaling sweep.
    parser.add_argument("--Kx", type=int, default=None)
    parser.add_argument("--Ky", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--T", type=float, default=None,
                        help="Final integration time (s)")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to output filenames so a "
                             "sweep doesn't overwrite previous results.")
    args = parser.parse_args()
    main(args)
