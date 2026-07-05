"""Rising bubble with in-the-loop adaptive mesh refinement.

The AMR analogue of :mod:`ns_rising_bubble_2d`. Same buoyant-bubble
problem --- a light blob in a closed box rising under gravity, the
density carried as a Boussinesq scalar --- but started on a coarse
mesh and adaptively refined while marching: every ``ADAPT_EVERY``
steps a Kelly indicator on the scalar ``c`` flags the elements at the
bubble interface, the refinement forest red-refines them, the six
volume fields (velocity, scalar, and one step of each one's history)
are transferred onto the new mesh, and the velocity / scalar
operators are rebuilt.

This is the first AMR case with a genuinely *localised* feature: the
bubble interface occupies a small fraction of the domain and the
quiescent fluid around it stays coarse, so adaptivity is expected to
save real work --- unlike the Taylor-Green verification of
:mod:`ns_taylor_green_amr_2d`, whose vortex fills the domain.

Velocity-pressure AMR carries two caveats (see report §6): the
face-shaped dual-splitting pressure trace ``dpdn`` cannot ride a
remesh and is reset to zero, and the BDF advection history is
recomputed on the new mesh from the transferred fields. Both inject a
small transient at each adaptation that decays over a few steps.
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
    BDF1, BDF2, build_ins_operators, make_penalty_config_dimensional,
    ns_advection_2d, ns_step_2d,
)
from ddg.dg2d.sipg import _local_mass
from ddg.dg2d.scalar_transport import (
    build_scalar_operators, scalar_advection_2d, scalar_transport_step_2d,
)
from ddg.dg2d.refinement import RefinementForest, kelly_error_indicator

from applications.ns_rising_bubble_2d import (
    W, H, ATWOOD, GRAVITY, NU, KAPPA, DT, T_FINAL, U_CHAR,
    CY_BUBBLE, USE_PENALTY, PENALTY_SCALE,
    tag_velocity_bc, scalar_dirichlet_bc, initial_bubble, bubble_metrics,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 3
BASE_KX, BASE_KY = 12, 24     # coarse base mesh; AMR refines from here

# MAX_LEVEL = 1 keeps the finest element at the 24x48-equivalent
# resolution of the validated uniform run, so the fixed DT stays safe.
MAX_LEVEL = 1
ADAPT_EVERY = 100             # steps between adaptation passes
THETA = 0.5                   # Dörfler bulk fraction

SAVE_EVERY = 25

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def _zeros(mesh):
    return jnp.zeros((mesh.Np, mesh.K))


def _rebuild_operators(mesh, ins_bc, sc_bc, g0):
    ops_v = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=g0, ins_bc_types=ins_bc)
    ops_c = build_scalar_operators(
        mesh, dt=DT, kappa=KAPPA, g0=g0, diffusion_bc_types=sc_bc)
    return ops_v, ops_c


def main():
    print(f"Rising bubble (AMR): base Kx={BASE_KX}, Ky={BASE_KY}, N={N}, "
          f"A={ATWOOD}, g={GRAVITY}")

    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, W, 0.0, H, BASE_KX, BASE_KY)
    base_mesh = build_mesh_2d(VX, VY, EToV, N)
    forest = RefinementForest(base_mesh, max_level=MAX_LEVEL)
    mesh, a2l = forest.active_mesh()
    print(f"base mesh: K={mesh.K} triangles, Np={mesh.Np}")

    ins_bc = tag_velocity_bc(mesh)
    sc_bc = scalar_dirichlet_bc(mesh)
    M_ref = _local_mass(mesh)

    ux, uy = _zeros(mesh), _zeros(mesh)
    bc_ux, bc_uy = _zeros(mesh), _zeros(mesh)
    c = initial_bubble(mesh)
    c_dir = _zeros(mesh)

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

    y0, _, m0 = bubble_metrics(mesh, c, uy, M_ref)
    times, yc_series, vrise_series, area_series, k_series = [], [], [], [], []
    bdf = BDF1
    n_adapts = 0
    step_start = time.time()

    for step in range(1, n_steps + 1):
        if step == 2:
            ops_v, ops_c = _rebuild_operators(mesh, ins_bc, sc_bc, BDF2.g0)
            bdf = BDF2

        c_new, Nc_n = scalar_transport_step_2d(
            mesh, c, c_old, ux, uy, Nc_old,
            dt=DT, kappa=KAPPA, bdf=bdf, ops=ops_c, c_dir=c_dir)
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

        y_c, v_rise, m = bubble_metrics(mesh, c, uy, M_ref)
        times.append(step * DT)
        yc_series.append(y_c)
        vrise_series.append(v_rise)
        area_series.append(m)
        k_series.append(int(mesh.K))

        if step % 50 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            print(f"  step {step:5d}/{n_steps}  t={step*DT:6.3f}  K={mesh.K:5d}"
                  f"  max|U|={max_u:.4f}  y_c={y_c:.4f}  v_rise={v_rise:+.4f}"
                  f"  mass/mass0={m/m0:.4f}  elapsed={elapsed:.1f}s")

        # --- adaptation pass ---------------------------------------
        if step % ADAPT_EVERY == 0 and step < n_steps:
            eta = kelly_error_indicator(mesh, c)
            new_mesh, fields, a2l, n = forest.adapt(
                mesh, a2l,
                {"ux": ux, "uy": uy, "ux_old": ux_old, "uy_old": uy_old,
                 "c": c, "c_old": c_old},
                eta, theta=THETA)
            if n == 0:
                continue
            n_adapts += 1
            mesh = new_mesh
            ux, uy = fields["ux"], fields["uy"]
            ux_old, uy_old = fields["ux_old"], fields["uy_old"]
            c, c_old = fields["c"], fields["c_old"]
            ins_bc = tag_velocity_bc(mesh)
            sc_bc = scalar_dirichlet_bc(mesh)
            M_ref = _local_mass(mesh)
            bc_ux, bc_uy = _zeros(mesh), _zeros(mesh)
            c_dir = _zeros(mesh)
            # history consistent with the transferred fields
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(t_arr, yc_series)
    axes[0].axhline(CY_BUBBLE, color="k", ls=":", label="initial centre")
    axes[0].set_xlabel("$t$"); axes[0].set_ylabel("centroid height $y_c$")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(t_arr, vrise_series, color="C1")
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel("rise velocity")
    axes[1].grid(alpha=0.3)
    axes[2].plot(t_arr, k_series, color="C2")
    axes[2].set_xlabel("$t$"); axes[2].set_ylabel("active elements $K$")
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"Rising bubble (AMR)  base {BASE_KX}×{BASE_KY}, "
                 f"N={N}, {n_adapts} adaptations, K_final={mesh.K}")
    fig.tight_layout()
    diag_path = OUT_DIR / "ns_rising_bubble_amr_2d_diagnostics.png"
    fig.savefig(diag_path, dpi=120)
    plt.close(fig)
    print(f"wrote {diag_path}")

    # Final scalar field + adapted mesh.
    fig2, ax2 = plt.subplots(figsize=(4.5, 7))
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    cc = np.asarray(c).T.reshape(-1)
    tpc = ax2.tricontourf(px, py, cc, levels=24, cmap="viridis")
    ax2.triplot(np.asarray(mesh.VX), np.asarray(mesh.VY),
                np.asarray(mesh.EToV), color="w", lw=0.25, alpha=0.5)
    ax2.set_aspect("equal")
    ax2.set_title(f"c at T={T_FINAL} + adapted mesh")
    fig2.colorbar(tpc, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    field_path = OUT_DIR / "ns_rising_bubble_amr_2d_field.png"
    fig2.savefig(field_path, dpi=120)
    plt.close(fig2)
    print(f"wrote {field_path}")

    # Element-step accounting vs the equivalent uniform run.
    uniform_K = BASE_KX * BASE_KY * 2 * (4 ** MAX_LEVEL)
    amr_elem_steps = int(np.sum(k_series))
    uniform_elem_steps = uniform_K * n_steps
    print(f"element-steps: AMR {amr_elem_steps}, "
          f"uniform-equivalent {uniform_elem_steps}, "
          f"ratio {amr_elem_steps / uniform_elem_steps:.3f}")

    meta = {
        "benchmark": "rising bubble (AMR)",
        "ATWOOD": ATWOOD, "GRAVITY": GRAVITY, "NU": NU, "KAPPA": KAPPA,
        "N": N, "base_KX": BASE_KX, "base_KY": BASE_KY,
        "max_level": MAX_LEVEL, "adapt_every": ADAPT_EVERY, "theta": THETA,
        "n_adapts": n_adapts, "K_base": int(base_mesh.K),
        "K_final": int(mesh.K), "T_FINAL": T_FINAL, "DT": DT,
        "amr_element_steps": amr_elem_steps,
        "uniform_equiv_element_steps": uniform_elem_steps,
        "element_step_ratio": amr_elem_steps / uniform_elem_steps,
        "final_centroid_height": yc_series[-1],
    }
    meta_path = OUT_DIR / "ns_rising_bubble_amr_2d_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
