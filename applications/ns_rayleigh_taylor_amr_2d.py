"""Rayleigh-Taylor instability with in-the-loop adaptive mesh refinement.

The AMR analogue of :mod:`ns_rayleigh_taylor_2d`. Same problem --- a
heavy fluid over a light one in an x-periodic channel, the density
carried as a Boussinesq scalar --- but started on a coarse mesh and
adaptively refined while marching: every ``ADAPT_EVERY`` steps a Kelly
indicator on the density ``c`` flags the elements at the mixing
interface, the forest red-refines them (keeping the x-periodic
boundaries in lockstep), the six volume fields are transferred onto
the new mesh, and the velocity / scalar operators are rebuilt.

The mixing layer is a localised, growing feature: refinement tracks
the interface and the unmixed fluid above and below stays coarse, so
adaptivity is expected to save real work.

Periodic AMR: the refinement forest is built with
``periodic={"x": (0, W)}`` so opposite x-boundaries refine
identically, and ``apply_rectangle_periodic_2d`` is re-applied to the
adapted mesh after every adaptation. As in the rising-bubble AMR run,
the face-shaped ``dpdn`` trace is reset and the BDF advection history
recomputed at each remesh (see report §6).
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

from ddg.dg2d.mesh import (
    apply_rectangle_periodic_2d, build_mesh_2d, uniform_rectangle_mesh_2d,
)
from ddg.dg2d.incompressible_ns import (
    BDF1, BDF2, build_ins_operators, make_penalty_config_dimensional,
    ns_advection_2d, ns_step_2d,
)
from ddg.dg2d.scalar_transport import (
    all_neumann_bc_types, build_scalar_operators, scalar_advection_2d,
    scalar_transport_step_2d,
)
from ddg.dg2d.refinement import RefinementForest, kelly_error_indicator

from applications.ns_rayleigh_taylor_2d import (
    W, H, ATWOOD, GRAVITY, NU, KAPPA, DT, T_FINAL, TAU, U_CHAR,
    USE_PENALTY, PENALTY_SCALE,
    tag_velocity_bc, initial_density, mixing_fronts,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 3
BASE_KX, BASE_KY = 12, 24      # coarse base mesh; AMR refines from here

# MAX_LEVEL = 1 keeps the finest element at the 24x48-equivalent
# resolution of the validated uniform run, so the fixed DT stays safe.
MAX_LEVEL = 1
ADAPT_EVERY = 100              # steps between adaptation passes
THETA = 0.5                    # Dörfler bulk fraction

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def _zeros(mesh):
    return jnp.zeros((mesh.Np, mesh.K))


def _periodize(raw_mesh):
    """Re-apply x-periodicity to a freshly extracted active mesh."""
    return apply_rectangle_periodic_2d(
        raw_mesh, xmin=0.0, xmax=W, ymin=0.0, ymax=H,
        periodic_x=True, periodic_y=False)


def _rebuild_operators(mesh, ins_bc, sc_bc, g0):
    ops_v = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=g0, ins_bc_types=ins_bc)
    ops_c = build_scalar_operators(
        mesh, dt=DT, kappa=KAPPA, g0=g0, diffusion_bc_types=sc_bc)
    return ops_v, ops_c


def main():
    print(f"Rayleigh-Taylor (AMR): base Kx={BASE_KX}, Ky={BASE_KY}, N={N}, "
          f"A={ATWOOD}, g={GRAVITY}, tau={TAU:.3f}")

    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, W, 0.0, H, BASE_KX, BASE_KY)
    base_mesh = build_mesh_2d(VX, VY, EToV, N)
    forest = RefinementForest(
        base_mesh, max_level=MAX_LEVEL, periodic={"x": (0.0, W)})
    raw, a2l = forest.active_mesh()
    mesh = _periodize(raw)
    print(f"base mesh: K={mesh.K} triangles, Np={mesh.Np}")

    ins_bc = tag_velocity_bc(mesh)
    sc_bc = all_neumann_bc_types(mesh)

    ux, uy = _zeros(mesh), _zeros(mesh)
    bc_ux, bc_uy = _zeros(mesh), _zeros(mesh)
    c = initial_density(mesh)

    print("LU-factoring velocity + scalar operators ...")
    t0 = time.time()
    ops_v, ops_c = _rebuild_operators(mesh, ins_bc, sc_bc, BDF1.g0)
    print(f"  factor took {time.time() - t0:.1f}s")

    n_steps = int(np.ceil(T_FINAL / DT))
    print(f"running {n_steps} steps, dt={DT}; adapt every {ADAPT_EVERY}")

    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy)
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    c_old = c
    Nc_old = scalar_advection_2d(mesh, c, ux, uy)

    pen_cfg = None
    if USE_PENALTY:
        pen_cfg = make_penalty_config_dimensional(
            mesh, u_char=U_CHAR,
            scale_div=PENALTY_SCALE, scale_cont=PENALTY_SCALE)

    times, spike_series, bubble_series, k_series = [], [], [], []
    bdf = BDF1
    n_adapts = 0
    step_start = time.time()

    for step in range(1, n_steps + 1):
        if step == 2:
            ops_v, ops_c = _rebuild_operators(mesh, ins_bc, sc_bc, BDF2.g0)
            bdf = BDF2

        c_new, Nc_n = scalar_transport_step_2d(
            mesh, c, c_old, ux, uy, Nc_old,
            dt=DT, kappa=KAPPA, bdf=bdf, ops=ops_c)
        fy = -ATWOOD * GRAVITY * c_new
        fx = jnp.zeros_like(fy)
        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=DT, nu=NU, bdf=bdf, ops=ops_v,
            ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
            body_force=(fx, fy), penalty_iterative=pen_cfg)

        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n
        c_old, c, Nc_old = c, c_new, Nc_n

        spike_y, bubble_y = mixing_fronts(mesh, c)
        times.append(step * DT)
        spike_series.append(spike_y)
        bubble_series.append(bubble_y)
        k_series.append(int(mesh.K))

        if step % 50 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            print(f"  step {step:5d}/{n_steps}  t={step*DT:6.3f}  K={mesh.K:5d}"
                  f"  max|U|={max_u:.4f}  mixing-width={bubble_y-spike_y:.4f}"
                  f"  elapsed={elapsed:.1f}s")

        # --- adaptation pass ---------------------------------------
        if step % ADAPT_EVERY == 0 and step < n_steps:
            eta = kelly_error_indicator(mesh, c)
            raw, fields, a2l, n = forest.adapt(
                mesh, a2l,
                {"ux": ux, "uy": uy, "ux_old": ux_old, "uy_old": uy_old,
                 "c": c, "c_old": c_old},
                eta, theta=THETA)
            if n == 0:
                continue
            n_adapts += 1
            mesh = _periodize(raw)
            ux, uy = fields["ux"], fields["uy"]
            ux_old, uy_old = fields["ux_old"], fields["uy_old"]
            c, c_old = fields["c"], fields["c_old"]
            ins_bc = tag_velocity_bc(mesh)
            sc_bc = all_neumann_bc_types(mesh)
            bc_ux, bc_uy = _zeros(mesh), _zeros(mesh)
            Nux_old, Nuy_old = ns_advection_2d(
                mesh, ux_old, uy_old, ins_bc_types=ins_bc,
                bc_ux=bc_ux, bc_uy=bc_uy)
            Nc_old = scalar_advection_2d(mesh, c_old, ux_old, uy_old)
            dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
            ops_v, ops_c = _rebuild_operators(mesh, ins_bc, sc_bc, BDF2.g0)
            if USE_PENALTY:
                pen_cfg = make_penalty_config_dimensional(
                    mesh, u_char=U_CHAR,
                    scale_div=PENALTY_SCALE, scale_cont=PENALTY_SCALE)
            top = int(forest.leaf_levels().max())
            print(f"  [adapt {n_adapts}] step {step}: refined {n} leaves, "
                  f"K={mesh.K}, max level={top}")

    # ----- Diagnostics -----
    t_arr = np.array(times)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_arr, np.array(bubble_series) - np.array(spike_series))
    axes[0].set_xlabel("$t$"); axes[0].set_ylabel("mixing-layer width")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t_arr, k_series, color="C2")
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel("active elements $K$")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Rayleigh-Taylor (AMR)  base {BASE_KX}×{BASE_KY}, "
                 f"N={N}, {n_adapts} adaptations, K_final={mesh.K}")
    fig.tight_layout()
    growth_path = OUT_DIR / "ns_rayleigh_taylor_amr_2d_growth.png"
    fig.savefig(growth_path, dpi=120)
    plt.close(fig)
    print(f"wrote {growth_path}")

    # Final density field + adapted mesh.
    fig2, ax2 = plt.subplots(figsize=(4.5, 7))
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    cc = np.asarray(c).T.reshape(-1)
    tpc = ax2.tricontourf(px, py, cc, levels=24, cmap="RdBu_r")
    ax2.triplot(np.asarray(mesh.VX), np.asarray(mesh.VY),
                np.asarray(mesh.EToV), color="k", lw=0.25, alpha=0.4)
    ax2.set_aspect("equal")
    ax2.set_title(f"density at T={T_FINAL:.2f} + adapted mesh")
    fig2.colorbar(tpc, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    field_path = OUT_DIR / "ns_rayleigh_taylor_amr_2d_field.png"
    fig2.savefig(field_path, dpi=120)
    plt.close(fig2)
    print(f"wrote {field_path}")

    uniform_K = BASE_KX * BASE_KY * 2 * (4 ** MAX_LEVEL)
    amr_elem_steps = int(np.sum(k_series))
    uniform_elem_steps = uniform_K * n_steps
    print(f"element-steps: AMR {amr_elem_steps}, "
          f"uniform-equivalent {uniform_elem_steps}, "
          f"ratio {amr_elem_steps / uniform_elem_steps:.3f}")

    meta = {
        "benchmark": "Rayleigh-Taylor instability (AMR)",
        "ATWOOD": ATWOOD, "GRAVITY": GRAVITY, "NU": NU, "KAPPA": KAPPA,
        "N": N, "base_KX": BASE_KX, "base_KY": BASE_KY,
        "max_level": MAX_LEVEL, "adapt_every": ADAPT_EVERY, "theta": THETA,
        "n_adapts": n_adapts, "K_base": int(base_mesh.K),
        "K_final": int(mesh.K), "T_FINAL": T_FINAL, "DT": DT,
        "amr_element_steps": amr_elem_steps,
        "uniform_equiv_element_steps": uniform_elem_steps,
        "element_step_ratio": amr_elem_steps / uniform_elem_steps,
        "final_mixing_width": bubble_series[-1] - spike_series[-1],
    }
    meta_path = OUT_DIR / "ns_rayleigh_taylor_amr_2d_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
