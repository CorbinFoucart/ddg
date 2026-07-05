"""Smoke-test the Joukowski conformal airfoil mesh: build it for a few
(thickness, camber) design params, check min(J) > 0 (no tangling), report
the enclosed area, and render the deformed mesh + airfoil outline."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_obstacle_cy_diff_2d import build_diff_setup, CX, RADIUS

CY = 0.25
from ddg.dg2d.differentiable_mesh import (
    differentiable_mesh_with_airfoil, joukowski_airfoil_area,
    windowed_joukowski_map,
)

ORDER, KX, KY = 2, 20, 10
R1, R2 = 0.10, 0.18
KAPPA = 0.80                 # TE-rounding used by the shape-opt driver
A0 = np.pi * RADIUS ** 2     # original cylinder area

template, curving, _, _, _ = build_diff_setup(KX, KY, ORDER, CY)


def airfoil_outline(mu_x, mu_y, n=400):
    th = np.linspace(0, 2 * np.pi, n)
    zx = RADIUS * np.cos(th); zy = RADIUS * np.sin(th)
    x = CX + zx; y = CY + zy
    X, Y = windowed_joukowski_map(jnp.asarray(x), jnp.asarray(y),
                                  CX, CY, RADIUS, mu_x, mu_y, R1, R2, KAPPA)
    return np.asarray(X), np.asarray(Y)


cases = [(0.22, 0.0, "thin  (thick=0.22)"), (0.32, 0.0, "medium (thick=0.32)"),
         (0.45, 0.0, "thick (thick=0.45)"), (0.32, 0.15, "cambered (for contrast)")]

fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
for ax, (thick, camber, label) in zip(axes.ravel(), cases):
    mu_x = jnp.asarray(-thick * RADIUS)
    mu_y = jnp.asarray(camber * RADIUS)
    mesh = differentiable_mesh_with_airfoil(
        template.VX_nominal, template.VY_nominal, template, curving,
        jnp.asarray(CX), jnp.asarray(CY), jnp.asarray(RADIUS),
        mu_x, mu_y, R1, R2, KAPPA)
    J = np.asarray(mesh.J)
    area = float(joukowski_airfoil_area(jnp.asarray(CX), jnp.asarray(CY),
                                        jnp.asarray(RADIUS), mu_x, mu_y, 512, KAPPA))
    print(f"{label:30s}  min(J)={J.min():+.4e}  max(J)={J.max():.4e}  "
          f"area={area:.5f}  area/A0={area/A0:.3f}", flush=True)
    x = np.asarray(mesh.x); y = np.asarray(mesh.y)
    # element edges (faces) for a wireframe
    Fmask = np.asarray(mesh.Fmask)
    for f in range(3):
        fx = x[Fmask[:, f], :]; fy = y[Fmask[:, f], :]
        ax.plot(fx, fy, "-", color="0.6", lw=0.3)
    ox, oy = airfoil_outline(mu_x, mu_y)
    ax.fill(ox, oy, color="black", zorder=3)
    ax.set_xlim(CX - 0.22, CX + 0.30); ax.set_ylim(CY - 0.22, CY + 0.22)
    ax.set_aspect("equal")
    bad = "  *** J<=0 ***" if J.min() <= 0 else ""
    ax.set_title(f"{label}  (min J={J.min():.2e}){bad}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

out = ROOT / "report" / "img" / "candidates" / "airfoil_mesh_smoke.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out, flush=True)
