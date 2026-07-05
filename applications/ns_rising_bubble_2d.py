"""Rising bubble: a buoyant blob driven by the Boussinesq body force.

A single circular bubble of light (warm) fluid, initially at rest in
a closed box of heavier fluid, rises under buoyancy, deforms into the
classic mushroom / cap shape and sheds a trailing wake.

This mirrors the ExaDG ``incompressible_flow_with_transport/
rising_bubble`` case — a *Boussinesq* rising bubble (scalar transport
+ buoyancy body force, no surface tension, no large density ratio),
distinct from the surface-tension two-phase Hysing-Turek 2009
benchmark. The same coupled machinery as :mod:`ns_rayleigh_taylor_2d`.

The fluids are tracked by one normalised density scalar
``c ∈ [-1, 0]``: ``c = 0`` is the ambient heavy fluid, ``c → -1`` the
lightest fluid at the bubble core. The initial bubble is a smooth
cosine bump (matching ExaDG):

    c(r) = -½ (1 + cos(π r / r_b))   for r < r_b,   else 0

with ``r`` the distance from the bubble centre. Under the Boussinesq
approximation the only coupling into momentum is the buoyancy force
``f = (0, -A g c)`` — light fluid (``c < 0``) feels an upward force,
consistent with the sign convention in :mod:`ns_rayleigh_taylor_2d`.

Setup:
  * domain ``[0, W] × [0, H]``, no-slip on all four walls (closed box)
  * bubble radius ``r_b``, centre ``(W/2, 0.35 H)``
  * scalar: Dirichlet ``c = 0`` on every wall (ambient reference)
  * diagnostics: bubble centroid height, mass-weighted rise velocity
    and bubble area versus time, plus a density-field animation
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.incompressible_ns import (
    BC_INS_INTERIOR, BC_INS_WALL,
    BDF1, BDF2, build_ins_operators, make_penalty_config_dimensional,
    ns_advection_2d, ns_step_2d,
    pressure_bc_types_from_ins, viscous_bc_types_from_ins,
)
from ddg.dg2d.sipg import (
    BC_DIRICHLET, BC_INTERIOR, IterativeConfig, _local_mass,
    make_helmholtz_block_jacobi_preconditioner,
)
from ddg.dg2d.multigrid import make_sipg_hpmg_preconditioner
from ddg.dg2d.scalar_transport import (
    build_scalar_operators, scalar_advection_2d, scalar_transport_step_2d,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
W, H = 1.0, 2.0          # domain width, height
# Finest mesh = KX_BASE * 2^N_H_LEVELS  by  KY_BASE * 2^N_H_LEVELS triangles
# per rectangle; the forest hierarchy is what feeds the hp-multigrid
# pressure preconditioner (the benchmark winner for the singular Poisson).
KX_BASE, KY_BASE = 8, 16    # base mesh: 8x16 rectangles -> 128 quads -> K_base=256
N_H_LEVELS = 2              # 2 h-refinements -> finest = 32x64 rectangles
N = 3

R_BUBBLE = 0.25          # bubble radius
CX_BUBBLE = 0.5 * W      # bubble centre x
CY_BUBBLE = 0.35 * H     # bubble centre y (low — leaves room to rise)

ATWOOD = 0.5             # buoyancy contrast
GRAVITY = 1.0            # gravitational acceleration magnitude

NU = 1.0e-4              # kinematic viscosity (near-inviscid, like ExaDG)
KAPPA = 1.0e-4           # scalar diffusivity

T_FINAL = 20.0
DT = 3.0e-3
SAVE_EVERY = 40

# Checkpoint / restart support.
# ``CHECKPOINT_EVERY`` -- write a full restart state every N steps.
# ``RESTART_FROM_CHECKPOINT`` -- if True and a checkpoint exists at
# startup, resume from it; otherwise start from the cosine-bump IC.
# A separate ``..._data.npz`` is always written at the end of a run
# carrying the snapshot stack (re-render-ready raw data).
CHECKPOINT_EVERY = 200
RESTART_FROM_CHECKPOINT = True

# Matrix-free iterative path with the winning preconditioners from the
# wallclock benchmark: hp-multigrid V-cycle for the (singular) pressure
# Poisson; matrix-free block-Jacobi for the viscous + scalar Helmholtz.
# The pressure singularity (closed box, all-Neumann) is handled inside
# ``_pressure_solve`` by deflating the RHS onto the mean-zero subspace.
ITER_TOL = 1.0e-9
ITER_MAXITER = 5000

# Divergence + continuity penalty post-projection (Fehn 2018), on by
# default — the robustness audit's best-performing stabilisation.
# Characteristic velocity is the buoyant rise scale sqrt(A g r_b).
USE_PENALTY = True
PENALTY_SCALE = 1.0
U_CHAR = (ATWOOD * GRAVITY * R_BUBBLE) ** 0.5

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)
CKPT_PATH = OUT_DIR / "ns_rising_bubble_2d_checkpoint.npz"
DATA_PATH = OUT_DIR / "ns_rising_bubble_2d_data.npz"


# ---------------------------------------------------------------------------
# Checkpoint / restart
# ---------------------------------------------------------------------------

def _save_checkpoint(path, *, step, ux, uy, ux_old, uy_old,
                     Nux_old, Nuy_old, dpdn_old,
                     c, c_old, Nc_old,
                     times, yc_series, vrise_series, area_series,
                     snapshots_c, snapshot_steps, m0):
    """Write a self-contained restart bundle. Atomic via tmp + rename.

    np.savez auto-appends ``.npz`` to any path that doesn't already end
    in it -- so the tmp path itself must end in ``.npz`` for the rename
    target to match.
    """
    import numpy as _np
    tmp = path.with_name(path.stem + "__tmp__" + path.suffix)
    _np.savez(
        tmp,
        step=_np.int64(step),
        ux=_np.asarray(ux), uy=_np.asarray(uy),
        ux_old=_np.asarray(ux_old), uy_old=_np.asarray(uy_old),
        Nux_old=_np.asarray(Nux_old), Nuy_old=_np.asarray(Nuy_old),
        dpdn_old=_np.asarray(dpdn_old),
        c=_np.asarray(c), c_old=_np.asarray(c_old),
        Nc_old=_np.asarray(Nc_old),
        times=_np.asarray(times),
        yc_series=_np.asarray(yc_series),
        vrise_series=_np.asarray(vrise_series),
        area_series=_np.asarray(area_series),
        snapshots_c=_np.stack(snapshots_c) if snapshots_c else _np.empty((0,)),
        snapshot_steps=_np.asarray(snapshot_steps, dtype=_np.int64),
        m0=_np.float64(m0),
    )
    tmp.replace(path)


def _load_checkpoint(path):
    import numpy as _np
    ck = _np.load(path)
    return {k: ck[k] for k in ck.files}


# ---------------------------------------------------------------------------
# Boundary tagging + initial condition
# ---------------------------------------------------------------------------

def tag_velocity_bc(mesh):
    """Closed box — every boundary face is a no-slip wall."""
    K, Nfaces = mesh.K, mesh.Nfaces
    EToE = np.asarray(mesh.EToE)
    is_bdy = EToE == np.arange(K)[:, None]
    bc = np.where(is_bdy, BC_INS_WALL, BC_INS_INTERIOR).astype(np.int32)
    return jnp.asarray(bc)


def scalar_dirichlet_bc(mesh):
    """Scalar Dirichlet (c = 0, ambient) on every wall."""
    K, Nfaces = mesh.K, mesh.Nfaces
    EToE = np.asarray(mesh.EToE)
    is_bdy = EToE == np.arange(K)[:, None]
    bc = np.where(is_bdy, BC_DIRICHLET, BC_INTERIOR).astype(np.int32)
    return jnp.asarray(bc)


def initial_bubble(mesh):
    """Smooth cosine-bump bubble: c = -½(1+cos(π r/r_b)) inside, 0 out."""
    x = np.asarray(mesh.x)
    y = np.asarray(mesh.y)
    r = np.sqrt((x - CX_BUBBLE) ** 2 + (y - CY_BUBBLE) ** 2)
    bump = -0.5 * (1.0 + np.cos(np.pi * np.clip(r, 0.0, R_BUBBLE) / R_BUBBLE))
    c = np.where(r < R_BUBBLE, bump, 0.0)
    return jnp.asarray(c)


# ---------------------------------------------------------------------------
# Bubble diagnostics
# ---------------------------------------------------------------------------

def bubble_metrics(mesh, c, uy, M_ref):
    """Mass-weighted bubble centroid height, rise velocity and area.

    The bubble "mass" weight is ``-c`` (positive inside the light
    blob). Integrals use the DG mass matrix.
    """
    def integ(f):
        return float(jnp.sum(mesh.J * (M_ref @ f)))
    weight = jnp.maximum(-c, 0.0)
    m = integ(weight)
    if m < 1e-12:
        return 0.0, 0.0, 0.0
    y_c = integ(mesh.y * weight) / m
    v_rise = integ(uy * weight) / m
    return y_c, v_rise, m


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    KX_FINE = KX_BASE * (2 ** N_H_LEVELS)
    KY_FINE = KY_BASE * (2 ** N_H_LEVELS)
    print(
        f"Rising bubble (matrix-free iterative + hp-MG / block-Jacobi):\n"
        f"  base {KX_BASE}x{KY_BASE} -> finest {KX_FINE}x{KY_FINE}, "
        f"N={N}, n_h={N_H_LEVELS}\n"
        f"  r_b={R_BUBBLE}, centre=({CX_BUBBLE}, {CY_BUBBLE}), "
        f"A={ATWOOD}, g={GRAVITY}"
    )
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, W, 0.0, H, KX_BASE, KY_BASE)
    base_mesh = build_mesh_2d(VX, VY, EToV, N)

    # hp-multigrid pressure preconditioner: the V-cycle owns the forest
    # hierarchy and returns the finest mesh that the solve runs on.
    # alpha = 0 (Poisson) -- the pressure PC does NOT depend on g0 or dt
    # so it is built once for the whole run. regularize=True closes the
    # singular pure-Neumann coarse-grid solve.
    pressure_bc_fn = lambda m: pressure_bc_types_from_ins(tag_velocity_bc(m))
    print("building hp-multigrid pressure preconditioner ...")
    t0 = time.time()
    pressure_pc, mesh = make_sipg_hpmg_preconditioner(
        base_mesh, N_H_LEVELS,
        bc_fn=pressure_bc_fn, alpha=0.0, regularize=True,
    )
    print(f"  hp-MG build took {time.time() - t0:.1f}s")
    print(f"  finest mesh: K={mesh.K} triangles, Np={mesh.Np}, "
          f"dofs/field={mesh.K * mesh.Np}")

    ins_bc_types = tag_velocity_bc(mesh)
    scalar_diff_bc = scalar_dirichlet_bc(mesh)
    v_bc_types = viscous_bc_types_from_ins(ins_bc_types)
    M_ref = _local_mass(mesh)

    ux = jnp.zeros((mesh.Np, mesh.K))
    uy = jnp.zeros((mesh.Np, mesh.K))
    bc_ux = jnp.zeros((mesh.Np, mesh.K))
    bc_uy = jnp.zeros((mesh.Np, mesh.K))
    c = initial_bubble(mesh)
    c_dir = jnp.zeros((mesh.Np, mesh.K))   # ambient c = 0 on walls

    iter_cfg = IterativeConfig(tol=ITER_TOL, maxiter=ITER_MAXITER)

    def build_step_pcs(g0):
        """Block-Jacobi PCs for viscous + scalar; both alphas depend on g0."""
        alpha_v = g0 / (NU * DT)
        alpha_c = g0 / (KAPPA * DT)
        pc_v = make_helmholtz_block_jacobi_preconditioner(
            mesh, alpha=alpha_v, nu=NU, bc_types=v_bc_types, matfree=True,
        )
        pc_c = make_helmholtz_block_jacobi_preconditioner(
            mesh, alpha=alpha_c, nu=KAPPA, bc_types=scalar_diff_bc,
            matfree=True,
        )
        return pc_v, pc_c

    print("building skip-factor operators + block-Jacobi PCs for BDF1 ...")
    t0 = time.time()
    ops_v1 = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=BDF1.g0, ins_bc_types=ins_bc_types,
        skip_factor=True,
    )
    ops_c1 = build_scalar_operators(
        mesh, dt=DT, kappa=KAPPA, g0=BDF1.g0,
        diffusion_bc_types=scalar_diff_bc,
        skip_factor=True,
    )
    viscous_pc, scalar_pc = build_step_pcs(BDF1.g0)
    print(f"  build took {time.time() - t0:.1f}s")

    n_steps = int(np.ceil(T_FINAL / DT))
    print(f"running {n_steps} steps with dt = {DT}")

    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    c_old = c
    Nc_old = scalar_advection_2d(mesh, c, ux, uy)

    y0, _, m0 = bubble_metrics(mesh, c, uy, M_ref)
    pen_cfg = None
    if USE_PENALTY:
        pen_cfg = make_penalty_config_dimensional(
            mesh, u_char=U_CHAR,
            scale_div=PENALTY_SCALE, scale_cont=PENALTY_SCALE,
        )
        print(f"  divergence/continuity penalty ON (scale={PENALTY_SCALE})")

    times, yc_series, vrise_series, area_series = [], [], [], []
    snapshots_c = []
    snapshot_steps = []

    # Restart from checkpoint if present + requested.
    start_step = 0
    if RESTART_FROM_CHECKPOINT and CKPT_PATH.exists():
        ck = _load_checkpoint(CKPT_PATH)
        start_step = int(ck["step"])
        ux = jnp.asarray(ck["ux"]); uy = jnp.asarray(ck["uy"])
        ux_old = jnp.asarray(ck["ux_old"]); uy_old = jnp.asarray(ck["uy_old"])
        Nux_old = jnp.asarray(ck["Nux_old"]); Nuy_old = jnp.asarray(ck["Nuy_old"])
        dpdn_old = jnp.asarray(ck["dpdn_old"])
        c = jnp.asarray(ck["c"]); c_old = jnp.asarray(ck["c_old"])
        Nc_old = jnp.asarray(ck["Nc_old"])
        times = list(map(float, ck["times"]))
        yc_series = list(map(float, ck["yc_series"]))
        vrise_series = list(map(float, ck["vrise_series"]))
        area_series = list(map(float, ck["area_series"]))
        if ck["snapshots_c"].ndim == 3:
            snapshots_c = [np.asarray(s) for s in ck["snapshots_c"]]
        snapshot_steps = list(map(int, ck["snapshot_steps"]))
        m0 = float(ck["m0"])
        print(f"restarted from checkpoint at step {start_step}, "
              f"t={start_step * DT:.4f}")

    ops_v, ops_c, bdf = ops_v1, ops_c1, BDF1
    if start_step >= 1:
        # We've already taken the BDF1 -> BDF2 transition; use BDF2 ops.
        ops_v = build_ins_operators(
            mesh, dt=DT, nu=NU, g0=BDF2.g0, ins_bc_types=ins_bc_types,
            skip_factor=True,
        )
        ops_c = build_scalar_operators(
            mesh, dt=DT, kappa=KAPPA, g0=BDF2.g0,
            diffusion_bc_types=scalar_diff_bc,
            skip_factor=True,
        )
        viscous_pc, scalar_pc = build_step_pcs(BDF2.g0)
        bdf = BDF2
    step_start = time.time()
    for step in range(start_step + 1, n_steps + 1):
        if step == 2:
            ops_v = build_ins_operators(
                mesh, dt=DT, nu=NU, g0=BDF2.g0, ins_bc_types=ins_bc_types,
                skip_factor=True,
            )
            ops_c = build_scalar_operators(
                mesh, dt=DT, kappa=KAPPA, g0=BDF2.g0,
                diffusion_bc_types=scalar_diff_bc,
                skip_factor=True,
            )
            # alpha changed (g0: 1 -> 1.5) -- rebuild the Helmholtz PCs.
            # Pressure hp-MG (alpha=0) is unaffected.
            viscous_pc, scalar_pc = build_step_pcs(BDF2.g0)
            bdf = BDF2

        c_new, Nc_n = scalar_transport_step_2d(
            mesh, c, c_old, ux, uy, Nc_old,
            dt=DT, kappa=KAPPA, bdf=bdf, ops=ops_c, c_dir=c_dir,
            iterative=iter_cfg, preconditioner=scalar_pc,
        )
        fy = -ATWOOD * GRAVITY * c_new       # light (c<0) -> upward
        fx = jnp.zeros_like(fy)
        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=DT, nu=NU, bdf=bdf, ops=ops_v,
            ins_bc_types=ins_bc_types, bc_ux=bc_ux, bc_uy=bc_uy,
            body_force=(fx, fy),
            penalty_iterative=pen_cfg,
            iterative=iter_cfg,
            pressure_preconditioner=pressure_pc,
            viscous_preconditioner=viscous_pc,
        )
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n
        c_old, c, Nc_old = c, c_new, Nc_n

        y_c, v_rise, m = bubble_metrics(mesh, c, uy, M_ref)
        times.append(step * DT)
        yc_series.append(y_c)
        vrise_series.append(v_rise)
        area_series.append(m)

        if step % 50 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            print(f"  step {step:5d}/{n_steps}  t={step*DT:6.3f}  "
                  f"max|U|={max_u:.4f}  y_centroid={y_c:.4f}  "
                  f"v_rise={v_rise:+.4f}  mass/mass0={m/m0:.4f}  "
                  f"elapsed={elapsed:.1f}s")
        if step % SAVE_EVERY == 0 or step == n_steps:
            snapshots_c.append(np.asarray(c))
            snapshot_steps.append(step)
        if (CHECKPOINT_EVERY > 0
                and (step % CHECKPOINT_EVERY == 0 or step == n_steps)):
            _save_checkpoint(
                CKPT_PATH,
                step=step,
                ux=ux, uy=uy, ux_old=ux_old, uy_old=uy_old,
                Nux_old=Nux_old, Nuy_old=Nuy_old, dpdn_old=dpdn_old,
                c=c, c_old=c_old, Nc_old=Nc_old,
                times=times, yc_series=yc_series,
                vrise_series=vrise_series, area_series=area_series,
                snapshots_c=snapshots_c, snapshot_steps=snapshot_steps,
                m0=m0,
            )

    # ----- Diagnostics plot -----
    t_arr = np.array(times)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(t_arr, yc_series)
    axes[0].axhline(CY_BUBBLE, color="k", ls=":", label="initial centre")
    axes[0].set_xlabel("$t$"); axes[0].set_ylabel("centroid height $y_c$")
    axes[0].set_title("bubble centroid"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_arr, vrise_series, color="C1")
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel("rise velocity")
    axes[1].set_title("mass-weighted rise velocity")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t_arr, np.array(area_series) / m0, color="C2")
    axes[2].set_xlabel("$t$"); axes[2].set_ylabel("mass / mass(0)")
    axes[2].set_title("bubble mass conservation")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"Rising bubble   A={ATWOOD}, g={GRAVITY}, "
                 f"N={N}, {KX_FINE}×{KY_FINE} cells (hp-MG + block-Jacobi)")
    fig.tight_layout()
    diag_path = OUT_DIR / "ns_rising_bubble_2d_diagnostics.png"
    fig.savefig(diag_path, dpi=110)
    plt.close(fig)
    print(f"wrote {diag_path}")

    # ----- Density-field animation -----
    print("rendering density animation ...")
    from matplotlib.animation import FuncAnimation
    fig2, ax2 = plt.subplots(figsize=(4.5, 8))
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    levels = np.linspace(-1.0, 0.0, 21)
    # Times that match the snapshots_c list: regular saves at step %
    # SAVE_EVERY == 0, plus the final step (whether or not it's a multiple
    # of SAVE_EVERY). The conditional final-step append in the main loop
    # added one extra snapshot when n_steps % SAVE_EVERY != 0, so include
    # the final time in that case too.
    save_times = list(times[SAVE_EVERY - 1::SAVE_EVERY])
    if n_steps % SAVE_EVERY != 0:
        save_times.append(times[-1])

    def draw(i):
        ax2.clear()
        ax2.tricontourf(px, py, snapshots_c[i].T.reshape(-1),
                        levels=levels, cmap="viridis", extend="both")
        ax2.set_aspect("equal")
        ax2.set_title(f"density c   t = {save_times[i]:.2f}")
        return []

    anim = FuncAnimation(fig2, draw, frames=len(snapshots_c),
                         interval=60, blit=False)
    out_mp4 = OUT_DIR / "ns_rising_bubble_2d.mp4"
    anim.save(str(out_mp4), writer="ffmpeg", dpi=100)
    plt.close(fig2)
    print(f"wrote {out_mp4}")

    meta = {
        "benchmark": "Boussinesq rising bubble (ExaDG-style)",
        "solver": "matrix-free CG: hp-MG pressure + block-Jacobi viscous/scalar",
        "W": W, "H": H,
        "KX_base": KX_BASE, "KY_base": KY_BASE, "n_h_levels": N_H_LEVELS,
        "KX": KX_FINE, "KY": KY_FINE, "N": N,
        "K": int(mesh.K), "Np": int(mesh.Np),
        "iter_tol": ITER_TOL, "iter_maxiter": ITER_MAXITER,
        "R_BUBBLE": R_BUBBLE,
        "bubble_center": [CX_BUBBLE, CY_BUBBLE],
        "ATWOOD": ATWOOD, "GRAVITY": GRAVITY, "NU": NU, "KAPPA": KAPPA,
        "T_FINAL": T_FINAL, "DT": DT, "n_steps": n_steps,
        "initial_centroid": y0,
        "final_centroid": yc_series[-1],
        "max_rise_velocity": max(vrise_series),
        "final_mass_fraction": area_series[-1] / m0,
    }
    meta_path = OUT_DIR / "ns_rising_bubble_2d_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")

    # Re-render-ready data dump: every snapshot + mesh nodes + config.
    np.savez(
        DATA_PATH,
        snapshots_c=np.stack(snapshots_c),
        snapshot_steps=np.asarray(snapshot_steps, dtype=np.int64),
        save_times=np.asarray(save_times),
        times=np.asarray(times),
        yc_series=np.asarray(yc_series),
        vrise_series=np.asarray(vrise_series),
        area_series=np.asarray(area_series),
        mesh_x=np.asarray(mesh.x), mesh_y=np.asarray(mesh.y),
        ux_final=np.asarray(ux), uy_final=np.asarray(uy),
        c_final=np.asarray(c),
        m0=np.float64(m0),
        # Config:
        W=np.float64(W), H=np.float64(H),
        KX=np.int64(KX_FINE), KY=np.int64(KY_FINE),
        N=np.int64(N), K=np.int64(mesh.K), Np=np.int64(mesh.Np),
        T_FINAL=np.float64(T_FINAL), DT=np.float64(DT),
        ATWOOD=np.float64(ATWOOD), GRAVITY=np.float64(GRAVITY),
        NU=np.float64(NU), KAPPA=np.float64(KAPPA),
        R_BUBBLE=np.float64(R_BUBBLE),
        CX_BUBBLE=np.float64(CX_BUBBLE), CY_BUBBLE=np.float64(CY_BUBBLE),
    )
    print(f"wrote {DATA_PATH}")


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Boussinesq rising bubble (matrix-free hp-MG + block-Jacobi)."
    )
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="Write a restart checkpoint every N steps "
                        "(default %d; 0 disables)." % CHECKPOINT_EVERY)
    p.add_argument("--restart-from", type=str, default=None,
                   help="Path to a previous checkpoint .npz to restart from "
                        "(default: %s if it exists)." % CKPT_PATH.name)
    p.add_argument("--no-restart", action="store_true",
                   help="Ignore any existing checkpoint and start fresh.")
    p.add_argument("--t-final", type=float, default=None,
                   help="Override T_FINAL (default %g)." % T_FINAL)
    p.add_argument("--n-h-levels", type=int, default=None,
                   help="Override N_H_LEVELS: finest mesh = KX_BASE*2^L x "
                        "KY_BASE*2^L (default %d -> %dx%d)."
                        % (N_H_LEVELS, KX_BASE * 2 ** N_H_LEVELS,
                           KY_BASE * 2 ** N_H_LEVELS))
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    if _args.checkpoint_every is not None:
        CHECKPOINT_EVERY = _args.checkpoint_every
    if _args.no_restart:
        RESTART_FROM_CHECKPOINT = False
    if _args.restart_from is not None:
        CKPT_PATH = Path(_args.restart_from).expanduser().resolve()
        RESTART_FROM_CHECKPOINT = True
    if _args.t_final is not None:
        T_FINAL = _args.t_final
    if _args.n_h_levels is not None:
        N_H_LEVELS = _args.n_h_levels
    main()
