"""Lid-driven cavity on the BDM-P_0 coupled NS solver.

Steady-state solve of the Ghia-Ghia-Shin 1982 benchmark using the
H(div)-conforming saddle-point solver built on
``feat/nonlinear-exp``. Unit square ``[0, 1]^2``, lid sliding at
``U_LID = 1`` in ``+x``, no-slip on the other three walls. Corner
treatment follows the standard non-leaky convention: the lid value
``(U_LID, 0)`` applies on the entire top face GL quadrature
(including the GL points closest to the corners), with the side
walls held at zero -- the discontinuity sits at the corner vertex,
which is not a BDM DOF.

Reynolds number ``Re = U_LID * L / nu`` with ``L = 1``. At Re=100
the steady-state flow is dominated by a single recirculating
vortex plus small secondary corner vortices. At Re=400 / 1000 the
primary vortex shifts toward the geometric centre and secondary
vortices grow.

Run directly to produce diagnostics:
    JAX_PLATFORMS=cpu python applications/ns_bdm_cavity_2d.py

Use the ``--re`` / ``--kx`` flags to override the defaults. The
script writes residual histories and a centreline-profile PNG to
``output/``.
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

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.bdm import (
    BDM, bdm1_gather, bdm1_basis_at_points, _element_jacobian_matrices,
    bdm1_evaluate_boundary_velocity,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask,
    bdm1_evaluate_dirichlet_bc,
    ns_newton_solve_bdm_2d,
)


X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
U_LID = 1.0


def _is_top_face(x, y, *, tol=1e-9):
    """Sample whether a (x, y) face quadrature point lies on the
    lid (top edge y = Y_MAX). Used to set the discontinuous lid BC:
    top -> (U_LID, 0); everywhere else -> (0, 0)."""
    return jnp.abs(y - Y_MAX) < tol


def _lid_bc_fn(x, y):
    """Velocity BC: U_LID in +x on the top edge, zero on the other
    three walls. The top corner vertices land on the boundary between
    top and side walls -- BDM face GL points are at xi = +/- 1/sqrt(3)
    along the edge (strictly interior to the edge), so this function
    is single-valued at every BDM DOF location."""
    on_lid = _is_top_face(x, y)
    u_x = jnp.where(on_lid, U_LID, 0.0)
    u_y = jnp.zeros_like(u_x)
    return u_x, u_y


def _build_setup(Kx: int, Ky: int):
    """Build mesh, BDM space, Dirichlet BC normal trace (for strong pin),
    and the full u_BC vector at boundary GL points (for the SIPG lift --
    the tangential lid velocity has u_BC . n = 0 so the normal trace
    alone wouldn't drive the flow)."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(X_MIN, X_MAX, Y_MIN, Y_MAX, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, 1)
    V = BDM.from_mesh(mesh)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _lid_bc_fn)
    u_bc_vec = bdm1_evaluate_boundary_velocity(V, _lid_bc_fn)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


def _evaluate_velocity_field(V: BDM, u_global, x_query, y_query):
    """Evaluate the BDM velocity field at the given physical points.

    Brute-force per-query-point search for the containing element, then
    apply the Piola map. Returns ``(u_x, u_y)`` arrays matching
    ``x_query.shape``. Slow but adequate for the centreline-profile
    plotting (200 query points).
    """
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX)
    VY = np.asarray(V.mesh.VY)
    K = V.K
    u_local = bdm1_gather(V.topology, u_global)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]

    x_q = np.asarray(x_query).reshape(-1)
    y_q = np.asarray(y_query).reshape(-1)
    out_x = np.zeros_like(x_q)
    out_y = np.zeros_like(y_q)
    for qi, (xq, yq) in enumerate(zip(x_q, y_q)):
        # Find any containing triangle by barycentric coords. Loop in
        # Python -- small Q, fine.
        for k in range(K):
            v0 = (VX[EToV[k, 0]], VY[EToV[k, 0]])
            v1 = (VX[EToV[k, 1]], VY[EToV[k, 1]])
            v2 = (VX[EToV[k, 2]], VY[EToV[k, 2]])
            # Barycentric:
            det = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
            a = ((v1[0] - xq) * (v2[1] - yq) - (v2[0] - xq) * (v1[1] - yq)) / det
            b = ((v2[0] - xq) * (v0[1] - yq) - (v0[0] - xq) * (v2[1] - yq)) / det
            c = 1.0 - a - b
            if a >= -1e-10 and b >= -1e-10 and c >= -1e-10:
                # Reference coords on the standard reference triangle
                # vertices (-1,-1), (1,-1), (-1,1):
                # F(r,s) = V0 + 0.5(r+1)(V1-V0) + 0.5(s+1)(V2-V0)
                # Inverting: r = 2*b - 1, s = 2*c - 1.
                r = 2.0 * b - 1.0
                s = 2.0 * c - 1.0
                phi_ref = bdm1_basis_at_points(
                    jnp.array([r]), jnp.array([s]),
                )                                         # (1, 6, 2)
                phi_phys = (DF[k] @ phi_ref[0].T).T / J[k]   # (6, 2)
                u_xy = np.asarray(jnp.einsum("l,lc->c", u_local[k], phi_phys))
                out_x[qi] = u_xy[0]
                out_y[qi] = u_xy[1]
                break
    return out_x.reshape(x_query.shape), out_y.reshape(y_query.shape)


def run_cavity(Re: float, Kx: int, Ky: int,
               *, max_iters: int = 100, tol: float = 1e-8,
               linearisation: str = "picard",
               ic_seed: int = 0,
               verbose: bool = True):
    """Solve the cavity at the given Reynolds number on a Kx x Ky
    uniform mesh. Returns ``(V, u_sol, p_sol, history)``.

    The IC is ``u_bc + small interior noise`` -- the boundary stays at
    the prescribed Dirichlet values and the noise sets the iteration
    going (without it, the IC equals the lift function, which the
    SIPG-lift formulation makes a false root)."""
    nu = U_LID * (X_MAX - X_MIN) / Re
    V, u_bc, u_bc_vec, bc_mask = _build_setup(Kx, Ky)
    if verbose:
        print(f"Cavity: Re={Re:.1f}, nu={nu:.3e}, Kx={Kx}, Ky={Ky}, "
              f"DOFs={V.total_dofs + V.K} ({V.total_dofs} u + {V.K} p)")
        print(f"  linearisation={linearisation}, max_iters={max_iters}, tol={tol:.0e}")

    rng = np.random.default_rng(ic_seed)
    interior_noise = jnp.asarray(rng.standard_normal(V.total_dofs)) * 0.1
    u_init = u_bc + jnp.where(bc_mask, 0.0, interior_noise)

    history = []
    t0 = time.time()
    u_sol, p_sol, converged = ns_newton_solve_bdm_2d(
        V, nu=nu, u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
        u_init=u_init, p_init=jnp.zeros(V.K),
        pressure_eps=1e-10, tau_scale=4.0,
        max_iters=max_iters, tol=tol, atol=1e-12,
        residual_history=history,
        linearisation=linearisation,
    )
    elapsed = time.time() - t0
    if verbose:
        print(f"  {len(history)} iters in {elapsed:.1f}s, "
              f"||F||_init={history[0]:.3e}, ||F||_final={history[-1]:.3e}, "
              f"reduction={history[-1]/history[0]:.3e}, converged={converged}")
    return V, u_sol, p_sol, history


def _plot_centrelines(V, u_sol, *, Re, out_path):
    """Plot u_x along the vertical centreline x=0.5 and u_y along
    the horizontal centreline y=0.5."""
    y_line = np.linspace(0.01, 0.99, 60)
    x_line = np.linspace(0.01, 0.99, 60)
    u_x_vert, _ = _evaluate_velocity_field(
        V, u_sol, jnp.full(60, 0.5), jnp.asarray(y_line),
    )
    _, u_y_horz = _evaluate_velocity_field(
        V, u_sol, jnp.asarray(x_line), jnp.full(60, 0.5),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(u_x_vert, y_line, "b-")
    axes[0].set_xlabel("$u_x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title(f"u_x along x=0.5, Re={Re}")
    axes[0].grid(alpha=0.3)
    axes[0].axvline(0, color="k", lw=0.5)
    axes[1].plot(x_line, u_y_horz, "r-")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$u_y$")
    axes[1].set_title(f"u_y along y=0.5, Re={Re}")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--kx", type=int, default=12)
    ap.add_argument("--linearisation", choices=("picard", "newton"),
                     default="picard")
    ap.add_argument("--max-iters", type=int, default=80)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    V, u_sol, p_sol, history = run_cavity(
        Re=args.re, Kx=args.kx, Ky=args.kx,
        max_iters=args.max_iters, tol=args.tol,
        linearisation=args.linearisation,
    )

    stem = f"bdm_cavity_re{int(args.re):d}_kx{args.kx:d}_{args.linearisation}"
    diag = dict(
        Re=args.re, Kx=args.kx, Ky=args.kx, U_LID=U_LID,
        nu=float(U_LID * (X_MAX - X_MIN) / args.re),
        linearisation=args.linearisation,
        residual_history=[float(h) for h in history],
    )
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"  wrote {out_dir / (stem + '.json')}")

    _plot_centrelines(V, u_sol, Re=args.re,
                       out_path=out_dir / f"{stem}_centrelines.png")


if __name__ == "__main__":
    main()
