"""High-fidelity curved-mesh rendering + proper curved-element validity.

Each element is mapped from the reference triangle to physical space through
the isoparametric basis (the same map the solver uses): the curved boundary
is sampled densely and the element is filled translucently, so any element
overlap shows as a darkened region. ``min J`` is evaluated over a dense
INTERIOR lattice per element (not just the nodes / corner triangle), the
correct injectivity test for curved geometry.

  python report/scripts/render_curved_mesh_hifi.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_obstacle_cy_diff_2d import build_diff_setup, CX, RADIUS
from ddg.dg2d.differentiable_mesh import (
    differentiable_mesh_with_shape, differentiable_mesh_with_airfoil)
from ddg.dg2d.reference import (
    equilateral_nodes_2d, xy_to_rs, vandermonde_2d, grad_vandermonde_2d)

CY = 0.25
N = 2
KX, KY = 20, 10
R1, R2, KAPPA = 0.10, 0.18, 0.80
template, curving, *_ = build_diff_setup(KX, KY, N, CY)

# reference-element data (canonical node order matching mesh.x columns)
rx_, ry_ = equilateral_nodes_2d(N)
r_ref, s_ref = xy_to_rs(rx_, ry_)
invV = jnp.linalg.inv(vandermonde_2d(N, r_ref, s_ref))


def _ops(r_d, s_d):
    L = vandermonde_2d(N, r_d, s_d) @ invV
    Vr, Vs = grad_vandermonde_2d(N, r_d, s_d)
    return L, Vr @ invV, Vs @ invV


# dense interior lattice over the reference triangle (r,s>=-1, r+s<=0)
M = 26
pts = [(-1 + 2 * i / M, -1 + 2 * j / M)
       for i in range(M + 1) for j in range(M + 1 - i)]
r_lat = jnp.asarray([p[0] for p in pts]); s_lat = jnp.asarray([p[1] for p in pts])
L_lat, Dr_lat, Ds_lat = _ops(r_lat, s_lat)

# dense boundary loop (3 reference edges, 80 pts each)
nb = 80
t = np.linspace(-1, 1, nb)
e0 = (t, -np.ones(nb)); e1 = (1 - (t + 1), -1 + (t + 1)); e2 = (-np.ones(nb), 1 - (t + 1))
r_b = jnp.asarray(np.concatenate([e0[0], e1[0], e2[0]]))
s_b = jnp.asarray(np.concatenate([e0[1], e1[1], e2[1]]))
L_b, _, _ = _ops(r_b, s_b)


def min_J_dense(mesh):
    x = mesh.x; y = mesh.y
    xr = Dr_lat @ x; xs = Ds_lat @ x; yr = Dr_lat @ y; ys = Ds_lat @ y
    J = xr * ys - xs * yr                          # (Nlat, K)
    return float(jnp.min(J)), int(jnp.sum(J <= 0))


def render(ax, mesh, title):
    x = np.asarray(L_b @ mesh.x); y = np.asarray(L_b @ mesh.y)   # (Nbdy, K)
    for k in range(x.shape[1]):
        ax.fill(x[:, k], y[:, k], facecolor=(0.15, 0.35, 0.75, 0.10),
                edgecolor="k", lw=0.35, zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(CX - 0.14, CX + 0.14); ax.set_ylim(CY - 0.14, CY + 0.14)
    ax.set_title(title, fontsize=11)


m0 = differentiable_mesh_with_shape(
    template.VX_nominal, template.VY_nominal, template, curving,
    jnp.asarray(CX), jnp.asarray(CY), jnp.asarray(RADIUS),
    jnp.array([0.0, 0.0, 0.0, 0.0]))
m2 = differentiable_mesh_with_shape(
    template.VX_nominal, template.VY_nominal, template, curving,
    jnp.asarray(CX), jnp.asarray(CY), jnp.asarray(RADIUS),
    jnp.array([0.0, 0.0, 0.15, 0.0]))
mc = differentiable_mesh_with_airfoil(
    template.VX_nominal, template.VY_nominal, template, curving,
    jnp.asarray(CX), jnp.asarray(CY), jnp.asarray(RADIUS),
    jnp.asarray(-0.20 * RADIUS), jnp.asarray(0.0), R1, R2, KAPPA)

for tag, m in [("nominal cylinder (a=0)", m0), ("mode-2 a=0.15", m2),
               ("conformal thick=0.20", mc)]:
    mj, nneg = min_J_dense(m)
    print(f"{tag:26s}: dense-interior min J = {mj:+.4e}   #(J<=0) = {nneg}",
          flush=True)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
render(axes[0], m0, "nominal cylinder (a=0)")
render(axes[1], m2, "mode-2  a=0.15")
render(axes[2], mc, "conformal  thick=0.20")
fig.tight_layout()
out = ROOT / "output" / "_mesh_hifi.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print("wrote", out, flush=True)
