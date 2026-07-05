"""Kelvin-Helmholtz instability with in-the-loop adaptive mesh refinement.

The AMR analogue of :mod:`ns_kelvin_helmholtz_2d`. Same doubly-periodic
double-shear-layer problem, but started on a coarse mesh and
adaptively refined while marching: a Kelly indicator on the vorticity
flags the shear layers and the rolling-up vortices, the forest
red-refines them (keeping both periodic axes in lockstep), the four
velocity volume fields are transferred onto the new mesh, and the INS
operators are rebuilt.

The shear layers and their roll-up vortices are a localised feature
in an otherwise smooth flow, so adaptivity is expected to save real
work. The initial condition is pre-refined a few rounds before time
stepping so the thin tanh shear layers are resolved from step 1 ---
this matters at Re = 1000, where an under-resolved layer is the
Mechanism-1 divergence route of report §5.

Periodic AMR: the forest is built doubly periodic
(``periodic={"x":..., "y":...}``) and ``apply_rectangle_periodic_2d``
is re-applied after every adaptation. As elsewhere, the face-shaped
``dpdn`` trace is reset and the BDF advection history recomputed at
each remesh (report §6).
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
    BC_INS_INTERIOR, BDF1, BDF2, build_ins_operators,
    make_penalty_config_dimensional, ns_advection_2d, ns_step_2d,
)
from ddg.dg2d.ins_metrics import kinetic_energy, enstrophy
from ddg.dg2d.maxwell import curl_z_2d
from ddg.dg2d.refinement import RefinementForest, kelly_error_indicator

from applications.ns_kelvin_helmholtz_2d import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, NU, RE, U_CHAR, DT, T_FINAL,
    USE_PENALTY, PENALTY_SCALE, double_shear_layer_ic,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 4
BASE_KX, BASE_KY = 16, 16      # coarse base mesh; AMR refines from here

# MAX_LEVEL = 1 keeps the finest element at the 32x32-equivalent
# resolution of the validated uniform run, so the fixed DT stays safe.
MAX_LEVEL = 1
ADAPT_EVERY = 100              # steps between adaptation passes
THETA = 0.5                    # Dörfler bulk fraction
IC_PREREFINE_ROUNDS = 2        # resolve the shear layers before stepping

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def _periodize(raw_mesh):
    """Re-apply double periodicity to a freshly extracted active mesh."""
    return apply_rectangle_periodic_2d(
        raw_mesh, xmin=X_MIN, xmax=X_MAX, ymin=Y_MIN, ymax=Y_MAX,
        periodic_x=True, periodic_y=True)


def _all_interior_bc(mesh):
    return jnp.full((mesh.K, mesh.Nfaces), BC_INS_INTERIOR, dtype=jnp.int32)


def main():
    print(f"Kelvin-Helmholtz (AMR): base Kx={BASE_KX}, Ky={BASE_KY}, N={N}, "
          f"Re={RE:.0f} (nu={NU:.4g})")

    VX, VY, EToV = uniform_rectangle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, BASE_KX, BASE_KY)
    base_mesh = build_mesh_2d(VX, VY, EToV, N)
    forest = RefinementForest(
        base_mesh, max_level=MAX_LEVEL,
        periodic={"x": (X_MIN, X_MAX), "y": (Y_MIN, Y_MAX)})
    raw, a2l = forest.active_mesh()
    mesh = _periodize(raw)

    ux, uy = double_shear_layer_ic(mesh)

    # Pre-refine the initial condition so the thin shear layers are
    # resolved before time stepping starts.
    for r in range(IC_PREREFINE_ROUNDS):
        eta = kelly_error_indicator(mesh, curl_z_2d(mesh, ux, uy))
        raw, fields, a2l, n = forest.adapt(
            mesh, a2l, {"ux": ux, "uy": uy}, eta, theta=THETA)
        if n == 0:
            break
        mesh = _periodize(raw)
        ux, uy = fields["ux"], fields["uy"]
        print(f"  IC pre-refine {r + 1}: refined {n}, K={mesh.K}")
    print(f"mesh after IC pre-refine: K={mesh.K} triangles, Np={mesh.Np}")

    ins_bc = _all_interior_bc(mesh)
    bc_ux = jnp.zeros((mesh.Np, mesh.K))
    bc_uy = jnp.zeros((mesh.Np, mesh.K))

    print("LU-factoring pressure + viscous matrices ...")
    t0 = time.time()
    ops = build_ins_operators(
        mesh, dt=DT, nu=NU, g0=BDF1.g0, ins_bc_types=ins_bc)
    print(f"  factor took {time.time() - t0:.1f}s")

    n_steps = int(np.ceil(T_FINAL / DT))
    print(f"running {n_steps} steps, dt={DT}; adapt every {ADAPT_EVERY}")

    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy)
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))

    pen_cfg = None
    if USE_PENALTY:
        pen_cfg = make_penalty_config_dimensional(
            mesh, u_char=U_CHAR,
            scale_div=PENALTY_SCALE, scale_cont=PENALTY_SCALE)

    times, ke_series, ens_series, k_series = [], [], [], []
    bdf = BDF1
    n_adapts = 0
    step_start = time.time()

    for step in range(1, n_steps + 1):
        if step == 2:
            ops = build_ins_operators(
                mesh, dt=DT, nu=NU, g0=BDF2.g0, ins_bc_types=ins_bc)
            bdf = BDF2

        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=DT, nu=NU, bdf=bdf, ops=ops,
            ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
            penalty_iterative=pen_cfg)

        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n

        times.append(step * DT)
        ke_series.append(float(kinetic_energy(mesh, ux, uy)))
        ens_series.append(float(enstrophy(mesh, ux, uy)))
        k_series.append(int(mesh.K))

        if step % 50 == 0 or step == n_steps:
            elapsed = time.time() - step_start
            max_u = float(jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)))
            print(f"  step {step:5d}/{n_steps}  t={step*DT:6.3f}  K={mesh.K:5d}"
                  f"  max|U|={max_u:.4f}  enstrophy={ens_series[-1]:.2f}"
                  f"  elapsed={elapsed:.1f}s")

        # --- adaptation pass ---------------------------------------
        if step % ADAPT_EVERY == 0 and step < n_steps:
            eta = kelly_error_indicator(mesh, curl_z_2d(mesh, ux, uy))
            raw, fields, a2l, n = forest.adapt(
                mesh, a2l,
                {"ux": ux, "uy": uy, "ux_old": ux_old, "uy_old": uy_old},
                eta, theta=THETA)
            if n == 0:
                continue
            n_adapts += 1
            mesh = _periodize(raw)
            ux, uy = fields["ux"], fields["uy"]
            ux_old, uy_old = fields["ux_old"], fields["uy_old"]
            ins_bc = _all_interior_bc(mesh)
            bc_ux = jnp.zeros((mesh.Np, mesh.K))
            bc_uy = jnp.zeros((mesh.Np, mesh.K))
            Nux_old, Nuy_old = ns_advection_2d(
                mesh, ux_old, uy_old, ins_bc_types=ins_bc,
                bc_ux=bc_ux, bc_uy=bc_uy)
            dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
            ops = build_ins_operators(
                mesh, dt=DT, nu=NU, g0=BDF2.g0, ins_bc_types=ins_bc)
            if USE_PENALTY:
                pen_cfg = make_penalty_config_dimensional(
                    mesh, u_char=U_CHAR,
                    scale_div=PENALTY_SCALE, scale_cont=PENALTY_SCALE)
            top = int(forest.leaf_levels().max())
            print(f"  [adapt {n_adapts}] step {step}: refined {n} leaves, "
                  f"K={mesh.K}, max level={top}")

    # ----- Diagnostics -----
    t_arr = np.array(times)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t_arr, ke_series)
    axes[0].set_xlabel("$t$"); axes[0].set_ylabel("kinetic energy")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t_arr, ens_series, color="C1")
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel("enstrophy")
    axes[1].set_title("enstrophy (spikes as layers roll up)")
    axes[1].grid(alpha=0.3)
    axes[2].plot(t_arr, k_series, color="C2")
    axes[2].set_xlabel("$t$"); axes[2].set_ylabel("active elements $K$")
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"Kelvin-Helmholtz (AMR)  base {BASE_KX}×{BASE_KY}, "
                 f"N={N}, {n_adapts} adaptations, K_final={mesh.K}")
    fig.tight_layout()
    diag_path = OUT_DIR / "ns_kelvin_helmholtz_amr_2d_diagnostics.png"
    fig.savefig(diag_path, dpi=120)
    plt.close(fig)
    print(f"wrote {diag_path}")

    # Final vorticity field + adapted mesh.
    w = curl_z_2d(mesh, ux, uy)
    fig2, ax2 = plt.subplots(figsize=(6, 5.5))
    px = np.asarray(mesh.x).T.reshape(-1)
    py = np.asarray(mesh.y).T.reshape(-1)
    wf = np.asarray(w).T.reshape(-1)
    lim = float(np.max(np.abs(wf)))
    tpc = ax2.tricontourf(px, py, wf, levels=24, cmap="RdBu_r",
                          vmin=-lim, vmax=lim)
    ax2.triplot(np.asarray(mesh.VX), np.asarray(mesh.VY),
                np.asarray(mesh.EToV), color="k", lw=0.2, alpha=0.35)
    ax2.set_aspect("equal")
    ax2.set_title(f"vorticity at T={T_FINAL} + adapted mesh")
    fig2.colorbar(tpc, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    field_path = OUT_DIR / "ns_kelvin_helmholtz_amr_2d_field.png"
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
        "benchmark": "Kelvin-Helmholtz instability (AMR)",
        "RE": RE, "NU": NU, "N": N,
        "base_KX": BASE_KX, "base_KY": BASE_KY,
        "max_level": MAX_LEVEL, "adapt_every": ADAPT_EVERY, "theta": THETA,
        "ic_prerefine_rounds": IC_PREREFINE_ROUNDS,
        "n_adapts": n_adapts, "K_base": int(base_mesh.K),
        "K_final": int(mesh.K), "T_FINAL": T_FINAL, "DT": DT,
        "amr_element_steps": amr_elem_steps,
        "uniform_equiv_element_steps": uniform_elem_steps,
        "element_step_ratio": amr_elem_steps / uniform_elem_steps,
    }
    meta_path = OUT_DIR / "ns_kelvin_helmholtz_amr_2d_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
