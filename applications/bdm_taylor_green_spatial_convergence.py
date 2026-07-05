"""BDM_1 Taylor-Green SPATIAL convergence (transient operator).

Companion to the temporal study. Here dt is held small (so the BDF
temporal error is negligible) and the mesh is refined; the L2
velocity error at the final time then measures the SPATIAL
discretisation rate of the transient solver -- expected h^(k+1).

This confirms the spatial operator delivers its rate WITH the
BDF mass term active (the temporal study held the mesh fixed and
varied dt; this is the orthogonal sweep).

Manufactured time-dependent TG solution (self-consistent pressure):
    u(x, y, t) = (sin pi x cos pi y, -cos pi x sin pi y) exp(-2 pi^2 nu t)

Output:
    output/bdm_taylor_green_spatial_convergence.png
    output/bdm_taylor_green_spatial_convergence.json
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

# Reuse the validated machinery from the temporal study.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bdm_taylor_green_temporal_convergence import (  # noqa: E402
    make_tg_velocity, build_mesh, time_march, l2_error_at_time,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=0.1)
    ap.add_argument("--tfinal", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=0.0025,
                     help="Small fixed dt so BDF temporal error << spatial")
    ap.add_argument("--bdf-order", type=int, default=2)
    ap.add_argument("--meshes", type=str, default="6,8,12,16,24")
    ap.add_argument("--out-dir", type=Path,
                     default=Path(__file__).resolve().parent.parent / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    ns = [int(s) for s in args.meshes.split(",")]
    print(f"Taylor-Green spatial convergence (BDM_1)", flush=True)
    print(f"nu={args.nu}, T={args.tfinal}, dt={args.dt} (fixed), "
          f"BDF{args.bdf_order}", flush=True)

    hs, errs = [], []
    for n in ns:
        mesh = build_mesh(n)
        from ddg.dg2d.bdm import BDM
        V = BDM.from_mesh(mesh, order=1)
        t0 = time.time()
        u_sol, actual_T = time_march(
            V, args.nu, args.dt, args.tfinal, args.bdf_order,
            seed_analytic_history=True,
        )
        err = l2_error_at_time(V, u_sol, args.nu, actual_T)
        h = 1.0 / n
        hs.append(h); errs.append(err)
        print(f"  n={n}: h={h:.4f}, L2 err = {err:.3e}, {time.time()-t0:.1f}s",
              flush=True)
    hs = np.array(hs); errs = np.array(errs)
    slopes = (np.log(errs[:-1]) - np.log(errs[1:])) / \
             (np.log(hs[:-1]) - np.log(hs[1:]))
    for i, sl in enumerate(slopes):
        print(f"  slope({hs[i]:.4f} -> {hs[i+1]:.4f}) = {sl:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(hs, errs, "o-", color="C0",
               label=f"BDM_1 (asymp slope {slopes[-1]:.2f}, theory 2)")
    ref = errs[0] * (hs / hs[0]) ** 2
    ax.loglog(hs, ref, "--", color="C0", alpha=0.4, label="slope 2 reference")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\|u_h(T) - u_{\rm TG}(T)\|_{L^2}$")
    ax.set_title(f"Taylor-Green spatial convergence "
                  f"(BDM_1, BDF{args.bdf_order}, dt={args.dt}, nu={args.nu}, T={args.tfinal})")
    ax.grid(which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    png_path = args.out_dir / "bdm_taylor_green_spatial_convergence.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png_path}", flush=True)
    summary = {
        "nu": args.nu, "T_final": args.tfinal, "dt": args.dt,
        "bdf_order": args.bdf_order,
        "hs": hs.tolist(), "errs": errs.tolist(),
        "slopes": slopes.tolist(),
        "asymptotic_slope": float(slopes[-1]), "expected_slope": 2,
    }
    json_path = args.out_dir / "bdm_taylor_green_spatial_convergence.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
