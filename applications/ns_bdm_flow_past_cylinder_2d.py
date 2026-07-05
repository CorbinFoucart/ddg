"""Flow past a circular cylinder on the BDM_k stack: Schäfer-Turek 2D-2.

Port of ``ns_flow_past_cylinder_2d.py`` from the older splitting-DG
path onto the BDM_k transient nonlinear solver
(:func:`ns_newton_solve_bdm_k_2d`). Uses the order-parametric
machinery developed for the cavity sentinel: BDF2 + Newton, full
implicit viscous + convective coupling, H(div)-conforming velocity
that gives ``\\|\\nabla\\cdot u\\| \\sim 10^{-10}`` per step.

Geometry: channel ``[0, 2.2] x [0, 0.41]`` with a cylinder at
``(0.2, 0.2)``, radius 0.05. Body-fitted O-grid via
:func:`channel_with_circle_mesh_2d`; inner-ring faces bent onto the
true cylinder via :func:`apply_circular_hole_2d`. SIPG operator uses
the high-order cubature (:func:`attach_cubature`) on the curved
inner-ring elements; that fix is what made the splitting-cylinder
work (\\cref{sec:closed-curved} in the report).

Boundary conditions (four types):
* Inflow (x = xmin): parabolic ``u_x(y) = 4 U_max y (H - y) / H^2``,
  ``u_y = 0``. Smoothly ramped over ``T_ramp`` to avoid the
  impulsive-start pressure transient.
* Outflow (x = xmax): natural free-stress / zero-traction (the new
  ``is_natural_face`` plumbing through ``bdm_k_viscous_*_apply``).
* Top + bottom walls (y = ymin, y = ymax): no-slip Dirichlet.
* Cylinder surface: no-slip Dirichlet.

This is the first BDM app with a mixed Dirichlet/Natural BC.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ddg.dg2d.bdm import (
    BDM, bdm_k_basis_at_points, bdm_k_evaluate_boundary_velocity,
    bdm1_gather, _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns import IterativeSolverConfig
from ddg.dg2d.coupled_ns_bdm import (
    BDF_COEFFS, bdf_history_k,
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    make_bdm_k_block_diag_pc,
    make_bdm_k_block_triangular_pc,
    make_bdm_k_cahouet_chabard_pc,
    make_bdm_k_iterative_inner_solvers,
    make_bdm_k_pressure_mass_inverse,
    make_bdm_k_sipg_pressure_laplacian_inverse,
    make_bdm_k_velocity_block_jacobi_pc,
    ns_newton_solve_bdm_k_2d,
)
from ddg.dg2d.bdm_multigrid import build_velocity_hpmg
from ddg.dg2d.curved import apply_circular_hole_2d
from ddg.dg2d.mesh import build_mesh_2d, channel_with_circle_mesh_2d


# ---------------------------------------------------------------------------
# Geometry configuration. Defaults are Schäfer-Turek 2D-2; the
# ``--smoke-geometry`` flag swaps in a smaller mesh-friendly setup
# that lets us validate the multi-BC BDM_k machinery at low DOF count
# before scaling to the canonical benchmark.
# ---------------------------------------------------------------------------
# Schäfer-Turek 2D-2 (default). The canonical height is 0.41 but is
# not grid-alignable at modest Ky (h_y=0.41/K_y never divides
# C_y=0.2 or R_outer=0.05 exactly for K_y < 41). We use 0.40 instead
# -- a 2.4% departure -- which gives clean alignment at K_y=4, hx=hy,
# and preserves the asymmetric-wake geometry (cylinder at y=0.2,
# off-centre).
SCHAEFER = dict(
    X_MIN=0.0, X_MAX=2.2, Y_MIN=0.0, Y_MAX=0.40,
    CX=0.2, CY=0.2, RADIUS=0.05,
    U_MAX=1.5, NU=1.0e-3,
)
# Smoke geometry: smaller channel, larger cylinder, allows the grid to
# align with the bounding square at much lower Kx/Ky.
SMOKE = dict(
    X_MIN=0.0, X_MAX=1.0, Y_MIN=0.0, Y_MAX=0.5,
    CX=0.2, CY=0.25, RADIUS=0.05,
    U_MAX=1.5, NU=1.0e-3,
)

# Patched at runtime by main() based on --smoke-geometry.
X_MIN = SCHAEFER["X_MIN"]; X_MAX = SCHAEFER["X_MAX"]
Y_MIN = SCHAEFER["Y_MIN"]; Y_MAX = SCHAEFER["Y_MAX"]
H_CHANNEL = Y_MAX - Y_MIN
CX, CY = SCHAEFER["CX"], SCHAEFER["CY"]
RADIUS = SCHAEFER["RADIUS"]
D_CYL = 2.0 * RADIUS

U_MAX = SCHAEFER["U_MAX"]
U_MEAN = (2.0 / 3.0) * U_MAX                # = 1.0
NU = SCHAEFER["NU"]                          # Re = U_mean * D / nu = 100


def _select_geometry(smoke: bool):
    """Patch the module-level geometry constants. Hacky but lets the
    BC-classification + BC-function closures pick the right values
    without threading geom config through every call."""
    global X_MIN, X_MAX, Y_MIN, Y_MAX, H_CHANNEL, CX, CY, RADIUS, D_CYL
    global U_MAX, U_MEAN, NU
    g = SMOKE if smoke else SCHAEFER
    X_MIN = g["X_MIN"]; X_MAX = g["X_MAX"]
    Y_MIN = g["Y_MIN"]; Y_MAX = g["Y_MAX"]
    H_CHANNEL = Y_MAX - Y_MIN
    CX = g["CX"]; CY = g["CY"]
    RADIUS = g["RADIUS"]
    D_CYL = 2.0 * RADIUS
    U_MAX = g["U_MAX"]; U_MEAN = (2.0 / 3.0) * U_MAX
    NU = g["NU"]


# ---------------------------------------------------------------------------
# Boundary classification + BC sampling
# ---------------------------------------------------------------------------

def classify_boundary_faces(mesh, *, edge_tol=1e-3):
    """Return three (K, Nfaces) bool masks: ``is_inflow``,
    ``is_outflow``, and ``is_wall_or_cyl``. Wall and cylinder are
    grouped because both impose homogeneous Dirichlet no-slip; the
    distinction matters only for force extraction (the cylinder mask
    is computed separately below).
    """
    Nfaces = 3
    K = mesh.K
    Fmask = np.asarray(mesh.Fmask)
    x_np = np.asarray(mesh.x); y_np = np.asarray(mesh.y)
    x_face = np.zeros((Nfaces, K)); y_face = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        x_face[f, :] = x_np[Fmask[:, f], :].mean(axis=0)
        y_face[f, :] = y_np[Fmask[:, f], :].mean(axis=0)
    EToE_np = np.asarray(mesh.EToE)
    is_bdy = (EToE_np == np.arange(K)[:, None]).T   # (Nfaces, K)

    is_inflow_face = is_bdy & (x_face < X_MIN + edge_tol)
    is_outflow_face = is_bdy & (x_face > X_MAX - edge_tol)
    is_outer_y = is_bdy & (
        (y_face < Y_MIN + edge_tol) | (y_face > Y_MAX - edge_tol)
    )
    r_from_cyl = np.sqrt((x_face - CX) ** 2 + (y_face - CY) ** 2)
    is_cyl_face = (
        is_bdy & (r_from_cyl < RADIUS * 1.05)
        & (~is_inflow_face) & (~is_outflow_face) & (~is_outer_y)
    )
    is_wall_face = is_outer_y                        # top + bottom walls

    # Convert to (K, Nfaces) layout to match BDM_k face-data ordering.
    return (
        jnp.asarray(is_inflow_face.T),
        jnp.asarray(is_outflow_face.T),
        jnp.asarray((is_wall_face | is_cyl_face).T),
        jnp.asarray(is_cyl_face.T),
    )


def _ramp_factor(t, T_ramp):
    return jnp.where(
        t >= T_ramp,
        1.0,
        0.5 * (1.0 - jnp.cos(
            jnp.pi * jnp.clip(t, 0.0, T_ramp) / T_ramp)),
    )


def make_bc_fn(*, ramp_factor: float, inflow_pert: float = 0.0):
    """Closure: BC velocity ``u_BC(x, y)`` at boundary GL points.

    Walls + cylinder: zero. Inflow: parabolic, scaled by the current
    ramp factor. Outflow: irrelevant (those faces will be flagged
    ``is_natural_face`` so their BC contribution is dropped).

    ``inflow_pert`` > 0 adds a small one-sided transverse velocity
    ``uy = inflow_pert * sin(pi*yfrac)`` (peaks at mid-channel, same
    sign for all y) to the inflow, breaking the up-down symmetry so
    the von-Karman shedding can grow at a Re where the symmetric wake
    is otherwise a metastable attractor (the Schaefer-Turek geometry
    here has the cylinder exactly centred at Y_MAX/2, so without this
    the wake never sheds within a practical T). Mirrors the projection
    driver's ``_bc_with_pert``. The perturbation is ramped with the
    parabolic profile for a smooth start-up.
    """
    def _bc(x, y):
        yfrac = (y - Y_MIN) / H_CHANNEL
        parabolic_ux = 4.0 * U_MAX * yfrac * (1.0 - yfrac) * ramp_factor
        on_inflow = jnp.abs(x - X_MIN) < 1e-3
        ux = jnp.where(on_inflow, parabolic_ux, 0.0)
        uy_pert = inflow_pert * jnp.sin(jnp.pi * yfrac) * ramp_factor
        uy = jnp.where(on_inflow, uy_pert, 0.0)
        return ux, uy
    return _bc


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _kinetic_energy_bdmk(V: BDM, u_global) -> float:
    """0.5 ∫ |u|² dx using Hammer cubature + order-k basis."""
    from ddg.dg2d.bdm import (
        bdm_k_basis_at_points, bdm1_gather, _element_jacobian_matrices,
    )
    HAMMER_R = jnp.array([1.0 / 3.0, 0.6, 0.2, 0.2]) * 2.0 - 1.0
    HAMMER_S = jnp.array([1.0 / 3.0, 0.2, 0.6, 0.2]) * 2.0 - 1.0
    HAMMER_W = jnp.array([-0.5625, 0.5208333333333333,
                           0.5208333333333333, 0.5208333333333333]) * 0.5
    u_local = bdm1_gather(V.topology, u_global)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_ref = bdm_k_basis_at_points(V.order, HAMMER_R, HAMMER_S)
    phi_phys = jnp.einsum("kab,qlb->kqla", DF, phi_ref) / J[:, None, None, None]
    u_at_q = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)
    integrand = jnp.sum(u_at_q ** 2, axis=-1)
    return 0.5 * float(jnp.einsum("q,k,kq->", HAMMER_W, J, integrand))


def _face_ref_points(f_idx: int, xi: np.ndarray):
    """Reference-triangle (r, s) coords for the f-th face at line GL points
    ``xi`` in [-1, 1]. Matches the Hesthaven convention used elsewhere."""
    ones = np.ones_like(xi)
    if f_idx == 0:
        return xi, -ones
    if f_idx == 1:
        return -xi, xi
    return -ones, -xi  # f_idx == 2


def _compute_cylinder_forces(
    V: BDM, u_global, p_per_elem, is_cyl_face, nu: float,
):
    """Surface integral of sigma . n over the cylinder boundary.

    sigma = -p I + 2 nu * eps(u) where eps(u) = 0.5 (grad u + grad u^T)
    is the symmetric velocity gradient. Returns ``(F_x, F_y)``: the total
    horizontal/vertical force the fluid exerts on the cylinder.

    Direct surface-quadrature implementation: for each cylinder face,
    evaluate u, grad u, p at Gauss-Legendre points using the BDM_k
    Piola-mapped basis and the P_{k-1} monomial basis, then integrate.
    Pure Python loop over the ~16 cylinder faces of the smoke geometry;
    fine at this scale.
    """
    from ddg.dg2d.bdm import (
        bdm_k_basis_at_points, bdm_k_basis_grad_at_points,
        bdm_k_pkm1_basis_at_points, bdm1_gather,
        _element_jacobian_matrices,
    )
    k = V.order
    mesh = V.mesh
    n_gl = max(k + 2, 3)
    xi_gl, w_gl = np.polynomial.legendre.leggauss(n_gl)

    u_local = bdm1_gather(V.topology, u_global)        # (K, Np)
    DF_all, _ = _element_jacobian_matrices(mesh)       # (K, 2, 2)
    J_all = mesh.J[0]                                  # (K,)

    nx_arr = np.asarray(mesh.nx); ny_arr = np.asarray(mesh.ny)
    sJ_arr = np.asarray(mesh.sJ)
    Nfp = mesh.Nfp
    K_ = mesh.K
    nx_KF = nx_arr.reshape(3, Nfp, K_).transpose(2, 0, 1)[:, :, 0]   # (K, 3)
    ny_KF = ny_arr.reshape(3, Nfp, K_).transpose(2, 0, 1)[:, :, 0]
    sJ_KF = sJ_arr.reshape(3, Nfp, K_).transpose(2, 0, 1)[:, :, 0]

    is_cyl = np.asarray(is_cyl_face)

    F_x = 0.0
    F_y = 0.0
    for K_idx in range(K_):
        for f_idx in range(3):
            if not bool(is_cyl[K_idx, f_idx]):
                continue
            r_face, s_face = _face_ref_points(f_idx, xi_gl)
            phi_ref = bdm_k_basis_at_points(
                k, jnp.asarray(r_face), jnp.asarray(s_face))   # (Q, Np, 2)
            grad_ref = bdm_k_basis_grad_at_points(
                k, jnp.asarray(r_face), jnp.asarray(s_face))   # (Q, Np, 2, 2)
            q_at = bdm_k_pkm1_basis_at_points(
                k, jnp.asarray(r_face), jnp.asarray(s_face))   # (Q, dim_p)

            DF_K = DF_all[K_idx]                       # (2, 2)
            invJ_K = jnp.linalg.inv(DF_K)              # (2, 2)
            J_K = J_all[K_idx]

            # Piola-mapped physical basis: phi_phys = (1/J) DF @ phi_ref
            phi_phys = jnp.einsum("cb,qlb->qlc", DF_K, phi_ref) / J_K
            # Physical gradient (Piola + chain rule):
            # grad_phys[q, l, c, a] = (1/J) DF[c, b] grad_ref[q, l, b, d] invJ[d, a]
            grad_phys = (
                jnp.einsum("cb,qlbd,da->qlca",
                           DF_K, grad_ref, invJ_K) / J_K
            )

            u_at = jnp.einsum("l,qlc->qc", u_local[K_idx], phi_phys)
            grad_u_at = jnp.einsum("l,qlca->qca", u_local[K_idx], grad_phys)
            p_at = jnp.einsum("m,qm->q", p_per_elem[K_idx], q_at)

            sym_grad = 0.5 * (grad_u_at + jnp.transpose(grad_u_at, (0, 2, 1)))
            nvec = np.array([nx_KF[K_idx, f_idx], ny_KF[K_idx, f_idx]])

            # sigma . n = -p n + 2 nu * sym_grad . n
            sig_n = (-p_at[:, None] * nvec[None, :]
                     + 2.0 * nu * jnp.einsum("qca,a->qc", sym_grad, nvec))

            w_face = w_gl * float(sJ_KF[K_idx, f_idx])
            F_x += float(jnp.sum(jnp.asarray(w_face) * sig_n[:, 0]))
            F_y += float(jnp.sum(jnp.asarray(w_face) * sig_n[:, 1]))
    # The triangle outward normal on a cylinder face points away from
    # the fluid (into the cylinder body). The force the fluid exerts on
    # the cylinder uses the opposite normal -- flip sign.
    return -F_x, -F_y


def _evaluate_pressure_at(V: BDM, p_per_elem, xq: float, yq: float):
    """Evaluate the per-element P_{k-1} pressure at a single physical
    point ``(xq, yq)`` by finding the containing triangle and using its
    monomial basis. Returns ``nan`` if the point is outside the mesh."""
    from ddg.dg2d.bdm import bdm_k_pkm1_basis_at_points
    k = V.order
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    p_np = np.asarray(p_per_elem)
    for kk in range(V.K):
        v0 = (VX[EToV[kk, 0]], VY[EToV[kk, 0]])
        v1 = (VX[EToV[kk, 1]], VY[EToV[kk, 1]])
        v2 = (VX[EToV[kk, 2]], VY[EToV[kk, 2]])
        det = ((v1[0] - v0[0]) * (v2[1] - v0[1])
               - (v2[0] - v0[0]) * (v1[1] - v0[1]))
        a = ((v1[0] - xq) * (v2[1] - yq)
             - (v2[0] - xq) * (v1[1] - yq)) / det
        b = ((v2[0] - xq) * (v0[1] - yq)
             - (v0[0] - xq) * (v2[1] - yq)) / det
        c = 1.0 - a - b
        if a >= -1e-10 and b >= -1e-10 and c >= -1e-10:
            r = 2.0 * b - 1.0
            s = 2.0 * c - 1.0
            q_at = bdm_k_pkm1_basis_at_points(
                k, jnp.array([r]), jnp.array([s]))[0]
            return float(jnp.dot(jnp.asarray(p_np[kk]), q_at))
    return float("nan")


def _divergence_violation(V, u_global) -> float:
    from ddg.dg2d.bdm import bdm_k_divergence_apply
    return float(jnp.linalg.norm(bdm_k_divergence_apply(V, u_global)))


# ---------------------------------------------------------------------------
# Snapshot sampling for wake-video rendering
# ---------------------------------------------------------------------------

def _bdm_velocity_at_nodes(V, u_global):
    """Sample the BDM velocity at the mesh's Lagrangian nodal points.

    Returns ``(ux, uy)`` with shape ``(Np, K)``. Mirrors the helper in
    ``ns_bdm_rising_bubble_2d.py``.
    """
    k = V.order
    r, s = V.mesh.r, V.mesh.s
    phi_ref = bdm_k_basis_at_points(k, r, s)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum(
        "kca,nla->knlc", DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    u_nodes = jnp.einsum("kl,knlc->knc", u_local, phi_phys)
    return u_nodes[..., 0].T, u_nodes[..., 1].T


def _nodal_vorticity(mesh, ux, uy):
    """Compute omega = du_y/dx - du_x/dy at the Lagrangian nodes."""
    ux_r = mesh.Dr @ ux; ux_s = mesh.Ds @ ux
    uy_r = mesh.Dr @ uy; uy_s = mesh.Ds @ uy
    duydx = mesh.rx * uy_r + mesh.sx * uy_s
    duxdy = mesh.ry * ux_r + mesh.sy * ux_s
    return duydx - duxdy


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--kx", type=int, default=10)
    ap.add_argument("--ky", type=int, default=10)
    ap.add_argument("--smoke-geometry", action="store_true",
                    help="Use a smaller mesh-friendly geometry "
                         "(channel 1.0x0.5) instead of Schaefer-Turek "
                         "for first-cut multi-BC validation.")
    ap.add_argument("--n-radial", type=int, default=3,
                    help="annular layers between cylinder and bounding sq")
    ap.add_argument("--r-outer", type=float, default=0.1,
                    help="bounding-square half side around cylinder")
    ap.add_argument("--radial-growth", type=float, default=1.3)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--t-final", type=float, default=0.05,
                    help="smoke value -- bump up for production")
    ap.add_argument("--t-ramp", type=float, default=0.02)
    ap.add_argument("--bdf-order", type=int, choices=(1, 2), default=2)
    ap.add_argument("--linearisation",
                     choices=("picard", "newton", "hybrid"),
                     default="newton")
    ap.add_argument("--tau-scale", type=float, default=4.0)
    ap.add_argument("--inflow-pert", type=float, default=0.0,
                    help="One-sided transverse inflow perturbation "
                         "magnitude uy = pert*sin(pi*y/H), breaks up-down "
                         "symmetry to seed von-Karman shedding on the "
                         "centred-cylinder geometry. 0.02 is a good "
                         "default for Re=100; smaller for higher Re.")
    ap.add_argument("--max-outer", type=int, default=15)
    ap.add_argument("--tol-outer", type=float, default=1e-8)
    ap.add_argument("--iterative", action="store_true",
                    help="Use matrix-free GMRES with block-diag CC PC "
                         "instead of dense LU. Required for production K.")
    ap.add_argument("--max-inner", type=int, default=500)
    ap.add_argument("--tol-inner", type=float, default=1e-6)
    ap.add_argument("--gmres-restart", type=int, default=0,
                    help="GMRES restart. 0 = no restart "
                         "(restart=maxiter, full Arnoldi). Empirically "
                         "restart=50 is fatal at BDM_3 cylinder.")
    ap.add_argument("--schur-pc",
                     choices=("cc", "mass", "lapl", "al"),
                     default="mass",
                     help="Schur PC choice. 'mass' = (dt/gamma0) M_p^-1 "
                          "(default; best in mass-dominated regime, ie "
                          "small dt and/or small nu). 'lapl' = (1/nu) "
                          "K_p_SIPG^-1 (steady Stokes-dominated). "
                          "'cc' = additive Cahouet-Chabard sum. "
                          "'al' = augmented-Lagrangian Schur "
                          "(gamma * M_p^-1); requires --gamma-al > 0.")
    ap.add_argument("--gamma-al", type=float, default=0.0,
                    help="Augmented-Lagrangian parameter. Adds gamma * "
                         "D^T M_p^-1 D to the velocity block of A "
                         "(grad-div stabilisation) and uses gamma * M_p^-1 "
                         "as the Schur PC. Typical: gamma ~ alpha = "
                         "gamma0/dt or somewhat larger. Disabled when 0.")
    ap.add_argument("--pc-form",
                     choices=("block-diag", "block-tri"),
                     default="block-diag",
                     help="Saddle PC topology. 'block-diag' is the "
                          "Murphy-Wathen form; 'block-tri' is the "
                          "upper-triangular form (recommended with "
                          "--schur-pc al).")
    ap.add_argument("--velocity-pc",
                     choices=("block-jacobi", "hpmg"),
                     default="block-jacobi",
                     help="Velocity-block PC. 'block-jacobi' is per-"
                          "element inversion; 'hpmg' is one hp-multigrid "
                          "V/W-cycle on the BDM_k viscous-Helmholtz block "
                          "(Fehn/ExaDG-style; production recipe).")
    ap.add_argument("--mg-cycle", choices=("V", "W"), default="V",
                    help="hp-MG cycle type when --velocity-pc=hpmg. "
                         "V-cycle is ~4x cheaper per apply than W and "
                         "usually delivers comparable preconditioning "
                         "for this saddle point on body-fitted meshes.")
    ap.add_argument("--mg-nsm", type=int, default=2,
                    help="Pre/post smoothing steps per hp-MG level.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "output")
    ap.add_argument("--snapshot-stride", type=int, default=0,
                    help="Save velocity at mesh nodes every N steps; "
                         "render an mp4 of vorticity + speed at end. "
                         "0 = disabled (default).")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    _select_geometry(args.smoke_geometry)

    Re = U_MEAN * D_CYL / NU
    print(f"BDM_{args.order} flow past cylinder (Schäfer-Turek 2D-2): "
          f"Re={Re:.0f}, Kx={args.kx}, Ky={args.ky}, "
          f"dt={args.dt}, T_final={args.t_final}", flush=True)

    # --- Mesh + curving ------------------------------------------------
    VX, VY, EToV = channel_with_circle_mesh_2d(
        X_MIN, X_MAX, Y_MIN, Y_MAX, args.kx, args.ky,
        cx=CX, cy=CY, radius=RADIUS,
        R_outer=args.r_outer, n_radial=args.n_radial,
        radial_growth=args.radial_growth,
    )
    mesh = build_mesh_2d(VX, VY, EToV, args.order)
    mesh = apply_circular_hole_2d(
        mesh, cx=CX, cy=CY, radius=RADIUS,
        blend_interior=True, detection_tol=0.05,
    )
    J = np.asarray(mesh.J)
    n_inv = int((J <= 0).sum())
    if n_inv > 0:
        raise RuntimeError(f"Mesh has {n_inv} inverted nodes after curving")
    n_curved = int((J.max(axis=0) - J.min(axis=0) > 1e-12).sum())
    print(f"  mesh: K={mesh.K} triangles, curved-elements={n_curved}",
          flush=True)

    # --- BDM space + BC classification ---------------------------------
    V = BDM.from_mesh(mesh, order=args.order)
    n_u = V.total_dofs
    dim_p = args.order * (args.order + 1) // 2
    n_p = V.K * dim_p
    print(f"  BDM_{args.order}: n_u={n_u}, n_p={n_p}, total={n_u + n_p}",
          flush=True)

    is_inflow, is_outflow, is_wall_or_cyl, is_cyl = classify_boundary_faces(mesh)
    n_inf = int(is_inflow.sum()); n_out = int(is_outflow.sum())
    n_wc = int(is_wall_or_cyl.sum()); n_cyl = int(is_cyl.sum())
    print(f"  BC face counts: inflow={n_inf}, outflow={n_out}, "
          f"wall+cyl={n_wc} (cyl={n_cyl})", flush=True)

    # ``is_natural_face`` for the BDM solver: only outflow faces.
    is_natural_face = is_outflow

    # ``bc_mask`` (Dirichlet boundary DOFs): all boundary except outflow.
    # We exploit the fact that ``bdm1_boundary_dofs_mask`` now takes an
    # optional per-face Dirichlet mask.
    dirichlet_face = is_inflow | is_wall_or_cyl
    bc_mask = bdm1_boundary_dofs_mask(V, dirichlet_face=dirichlet_face)
    print(f"  Dirichlet DOFs: {int(bc_mask.sum())}/{n_u}", flush=True)

    # --- Time-stepping --------------------------------------------------
    n_steps = int(round(args.t_final / args.dt))
    u_n = jnp.zeros(n_u)
    u_nm1 = jnp.zeros(n_u)
    p_prev = jnp.zeros((V.K, dim_p))

    ke_series, div_series, times = [], [], []
    outer_iters_per_step = []
    inner_residual_final = []
    cd_series, cl_series, dp_series = [], [], []
    snapshots = []   # list of (t, ux_Np_K, uy_Np_K) when --snapshot-stride > 0

    # Snapshot persistence: write the mesh once and re-write the snapshot
    # trajectory incrementally so a crash mid-run (the box has OOM'd
    # before) keeps everything collected so far. Same NPZ format as the
    # projection driver so the paper replotter reads either unchanged.
    _snap_stem = (f"bdm_cylinder_k{args.order}_kx{args.kx}_ky{args.ky}"
                  f"_dt{args.dt:.3f}").replace(".", "p")
    _mesh_written = [False]

    def _dump_snapshots():
        if not (args.snapshot_stride > 0 and snapshots):
            return
        if not _mesh_written[0]:
            np.savez_compressed(
                args.out_dir / f"{_snap_stem}_mesh.npz",
                x=np.asarray(mesh.x), y=np.asarray(mesh.y),
                VX=np.asarray(mesh.VX), VY=np.asarray(mesh.VY),
                EToV=np.asarray(mesh.EToV),
                rx=np.asarray(mesh.rx), ry=np.asarray(mesh.ry),
                sx=np.asarray(mesh.sx), sy=np.asarray(mesh.sy),
                J=np.asarray(mesh.J),
                Dr=np.asarray(mesh.Dr), Ds=np.asarray(mesh.Ds),
                CX=CX, CY=CY, RADIUS=RADIUS,
                X_MIN=X_MIN, X_MAX=X_MAX, Y_MIN=Y_MIN, Y_MAX=Y_MAX,
                order=args.order, kx=args.kx, ky=args.ky,
                dt=args.dt, t_final=args.t_final, Re=Re,
            )
            _mesh_written[0] = True
        ts = np.asarray([s[0] for s in snapshots], dtype=np.float64)
        uxs = np.stack([s[1] for s in snapshots], axis=0).astype(np.float32)
        uys = np.stack([s[2] for s in snapshots], axis=0).astype(np.float32)
        np.savez_compressed(
            args.out_dir / f"{_snap_stem}_snapshots.npz",
            t=ts, ux=uxs, uy=uys,
        )

    # Iterative path: per-BDF-order PC (alpha depends on gamma0 / dt).
    # Inner solvers are also per-BDF-order and built ONCE so the
    # JIT cache survives across time steps.
    iter_cfg = None
    pcs_per_order = {}
    inner_solvers_per_order = {}
    if args.iterative:
        restart = (
            args.gmres_restart if args.gmres_restart > 0
            else args.max_inner
        )
        iter_cfg = IterativeSolverConfig(
            method="gmres", tol=args.tol_inner, atol=1e-12,
            maxiter=args.max_inner, restart=restart,
        )
        orders_used = {1, args.bdf_order}
        for bdf_o in orders_used:
            gamma0 = BDF_COEFFS[bdf_o]["gamma0"]
            alpha = gamma0 / args.dt
            if args.velocity_pc == "hpmg":
                if args.gamma_al != 0.0:
                    print("  warning: hp-MG velocity PC does not include "
                          "the AL term yet; gamma_al ignored for the PC "
                          "(matvec still uses it).", flush=True)
                # Pure p-MG (n_h=0) on the body-fitted cylinder mesh.
                # BDM_3 -> BDM_2 -> BDM_1 with vertex-patch smoother and
                # dense LU on the coarsest BDM_1 level. Production
                # Fehn/ExaDG recipe.
                _, _, mg_apply = build_velocity_hpmg(
                    mesh, n_h=0, order=args.order, nu=NU,
                    tau_scale=args.tau_scale,
                    nsm=args.mg_nsm, cycle=args.mg_cycle,
                    velocity_alpha=alpha,
                )
                vel_pc = mg_apply
            else:
                vel_pc = make_bdm_k_velocity_block_jacobi_pc(
                    V, nu=NU, alpha=alpha, tau_scale=args.tau_scale,
                    gamma_al=args.gamma_al,
                )
            if args.schur_pc == "cc":
                schur_pc = make_bdm_k_cahouet_chabard_pc(
                    V, nu=NU, dt=args.dt, gamma0=gamma0,
                )
            elif args.schur_pc == "mass":
                schur_pc = make_bdm_k_pressure_mass_inverse(
                    V, nu=gamma0 / args.dt, scale=1.0,
                )
            elif args.schur_pc == "lapl":
                schur_pc = make_bdm_k_sipg_pressure_laplacian_inverse(
                    V, scale=1.0 / NU,
                )
            else:                                        # "al"
                # Augmented Lagrangian Schur PC: S_gamma^{-1} = gamma * M_p^-1.
                # Encoded via make_bdm_k_pressure_mass_inverse: returns
                # (scale/nu) * M_p^{-1}; pick nu=1/gamma -> gamma * M_p^-1.
                if args.gamma_al <= 0.0:
                    raise SystemExit(
                        "--schur-pc al requires --gamma-al > 0"
                    )
                schur_pc = make_bdm_k_pressure_mass_inverse(
                    V, nu=1.0 / args.gamma_al, scale=1.0,
                )
            if args.pc_form == "block-tri":
                pcs_per_order[bdf_o] = make_bdm_k_block_triangular_pc(
                    V, velocity_pc=vel_pc, pressure_pc=schur_pc,
                    bc_mask=bc_mask,
                )
            else:
                pcs_per_order[bdf_o] = make_bdm_k_block_diag_pc(
                    V, velocity_pc=vel_pc, pressure_pc=schur_pc,
                )
            # JIT'd inner solvers, built once per BDF order and
            # reused across all time steps -- this lets the
            # gmres+matvec+PC trace cache hit on every step after
            # the first.
            inner_solvers_per_order[bdf_o] = make_bdm_k_iterative_inner_solvers(
                V, iterative=iter_cfg,
                coupled_preconditioner=pcs_per_order[bdf_o],
                nu=NU, pressure_eps=1e-10,
                tau_scale=args.tau_scale, velocity_alpha=alpha,
                is_natural_face=is_natural_face,
                bc_mask=bc_mask,
                u_bc_global=None,         # not used inside matvec
                project_pressure_mean=True,
                gamma_al=args.gamma_al,
                build_newton=(args.linearisation in ("newton", "hybrid")),
            )

    solver_tag = (
        f"iterative GMRES+{args.schur_pc} Schur (restart="
        f"{iter_cfg.restart})" if args.iterative else (
            "direct LU at order > 1" if args.order > 1 else "direct LU"
        )
    )
    print(f"  running {n_steps} steps with {args.linearisation} "
          f"linearisation, {solver_tag}", flush=True)
    t0 = time.time()
    last_step_t = t0
    for step in range(n_steps):
        t = (step + 1) * args.dt
        r = _ramp_factor(t, args.t_ramp)

        # Refresh BCs with current ramp factor.
        bc_fn = make_bc_fn(ramp_factor=float(r), inflow_pert=args.inflow_pert)
        u_bc = bdm1_evaluate_dirichlet_bc(V, bc_fn)
        u_bc_vec = bdm_k_evaluate_boundary_velocity(V, bc_fn)

        # BDF history -- step 0 is BDF1 bootstrap, rest are bdf_order.
        order_eff = 1 if (args.bdf_order > 1 and step == 0) else args.bdf_order
        alpha = BDF_COEFFS[order_eff]["gamma0"] / args.dt
        u_hist = [u_n] if order_eff == 1 else [u_n, u_nm1]
        f_history = bdf_history_k(V, u_hist, dt=args.dt, order=order_eff)

        outer_hist = []
        inner_hist = []
        u_new, p_new, _ = ns_newton_solve_bdm_k_2d(
            V, nu=NU,
            u_bc_global=u_bc, u_bc_vec=u_bc_vec, bc_mask=bc_mask,
            u_init=u_n, p_init=p_prev,
            velocity_alpha=alpha,
            f_velocity=f_history,
            pressure_eps=1e-10, tau_scale=args.tau_scale,
            max_iters=args.max_outer, tol=args.tol_outer, atol=1e-12,
            residual_history=outer_hist,
            linearisation=args.linearisation,
            iterative=iter_cfg,
            coupled_preconditioner=pcs_per_order.get(order_eff),
            inner_residual_history=inner_hist,
            is_natural_face=is_natural_face,
            project_pressure_mean=args.iterative,
            gamma_al=args.gamma_al,
            inner_solvers=inner_solvers_per_order.get(order_eff),
        )
        inner_residual_final.append(
            inner_hist[-1] if inner_hist else 0.0
        )

        ke = _kinetic_energy_bdmk(V, u_new)
        dv = _divergence_violation(V, u_new)
        # Schaefer-Turek 2D-2 force diagnostics.
        F_x, F_y = _compute_cylinder_forces(V, u_new, p_new, is_cyl, nu=NU)
        # C_D, C_L with rho=1 and the standard reference quantities.
        denom = 0.5 * U_MEAN**2 * D_CYL
        cd = F_x / denom
        cl = F_y / denom
        # delta p across the cylinder along the centreline.
        p_front = _evaluate_pressure_at(V, p_new, CX - RADIUS, CY)
        p_back = _evaluate_pressure_at(V, p_new, CX + RADIUS, CY)
        dp = float(p_front - p_back)
        ke_series.append(ke); div_series.append(dv); times.append(float(t))
        cd_series.append(cd); cl_series.append(cl); dp_series.append(dp)
        outer_iters_per_step.append(len(outer_hist))
        inner_res = inner_hist[-1] if inner_hist else 0.0
        step_t = time.time() - last_step_t; last_step_t = time.time()
        print(f"  step {step+1:3d}/{n_steps} t={t:.3f} BDF{order_eff}: "
              f"KE={ke:.4e}, ||div||={dv:.2e}, "
              f"C_D={cd:.3f}, C_L={cl:+.3f}, dp={dp:+.3f}, "
              f"outer_iters={len(outer_hist)}, "
              f"inner_res={inner_res:.2e}, "
              f"wall={step_t:.0f}s, "
              f"ramp={float(r):.3f}",
              flush=True)
        if args.snapshot_stride > 0 and (step + 1) % args.snapshot_stride == 0:
            ux_n, uy_n = _bdm_velocity_at_nodes(V, u_new)
            snapshots.append((float(t), np.asarray(ux_n), np.asarray(uy_n)))
            # Incremental checkpoint every 10 collected snapshots.
            if len(snapshots) % 10 == 0:
                _dump_snapshots()
                print(f"  checkpointed {len(snapshots)} snapshots", flush=True)
        u_nm1 = u_n
        u_n = u_new
        p_prev = p_new
    wall = time.time() - t0
    print(f"\nDone in {wall:.1f}s ({wall/max(n_steps,1)*1000:.0f} ms/step)",
          flush=True)

    # --- Save -----------------------------------------------------------
    stem = f"bdm_cylinder_k{args.order}_kx{args.kx}_ky{args.ky}_dt{args.dt:.3f}".replace(".", "p")
    out_npz = args.out_dir / f"{stem}.npz"
    np.savez_compressed(
        out_npz,
        u_final=np.asarray(u_n), p_final=np.asarray(p_new),
        ke=np.asarray(ke_series), div=np.asarray(div_series),
        times=np.asarray(times),
        outer_iters=np.asarray(outer_iters_per_step),
        cd=np.asarray(cd_series), cl=np.asarray(cl_series),
        dp=np.asarray(dp_series),
        K=mesh.K, order=args.order,
    )

    # Final snapshot + mesh persistence (same format as the projection
    # driver, so the paper replotter reads either unchanged). Incremental
    # checkpoints during the loop already wrote partial data; this is the
    # complete final dump.
    if args.snapshot_stride > 0 and snapshots:
        _dump_snapshots()
        print(f"wrote {_snap_stem}_mesh.npz + {_snap_stem}_snapshots.npz "
              f"({len(snapshots)} snapshots)", flush=True)
    out_json = args.out_dir / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump(dict(
            order=args.order, kx=args.kx, ky=args.ky, K=int(mesh.K),
            n_u=int(n_u), n_p=int(n_p),
            dt=float(args.dt), t_final=float(args.t_final),
            bdf_order=int(args.bdf_order),
            tau_scale=float(args.tau_scale),
            linearisation=args.linearisation,
            re=float(Re),
            ke=ke_series, div=div_series, times=times,
            cd=cd_series, cl=cl_series, dp=dp_series,
            outer_iters_per_step=outer_iters_per_step,
            wall_seconds=float(wall),
        ), f, indent=2)
    print(f"  wrote {out_npz}\n  wrote {out_json}", flush=True)

    # --- Plot diagnostics -----------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    t_arr = np.asarray(times)
    axs[0, 0].plot(t_arr, cd_series, "b-")
    axs[0, 0].set_xlabel("t"); axs[0, 0].set_ylabel(r"$C_D$")
    axs[0, 0].grid(alpha=0.3)
    axs[0, 1].plot(t_arr, cl_series, "r-")
    axs[0, 1].set_xlabel("t"); axs[0, 1].set_ylabel(r"$C_L$")
    axs[0, 1].grid(alpha=0.3)
    axs[1, 0].plot(t_arr, dp_series, "g-")
    axs[1, 0].set_xlabel("t"); axs[1, 0].set_ylabel(r"$\Delta p$")
    axs[1, 0].grid(alpha=0.3)
    axs[1, 1].semilogy(t_arr, div_series, "k-")
    axs[1, 1].set_xlabel("t"); axs[1, 1].set_ylabel(r"$\|\nabla\cdot u\|$")
    axs[1, 1].grid(alpha=0.3, which="both")
    fig.suptitle(
        f"BDM_{args.order} cylinder smoke "
        f"(Re={Re:.0f}, K={mesh.K}, dt={args.dt}, T={args.t_final})"
    )
    out_png = args.out_dir / f"{stem}_diag.png"
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}", flush=True)

    # --- Wake mp4 (only if snapshots were collected) --------------------
    if snapshots:
        print(f"rendering wake mp4 from {len(snapshots)} snapshots ...",
              flush=True)
        from matplotlib.animation import FuncAnimation
        px = np.asarray(mesh.x).T.reshape(-1)
        py = np.asarray(mesh.y).T.reshape(-1)
        theta = np.linspace(0, 2 * np.pi, 80)
        cyl_x = CX + RADIUS * np.cos(theta)
        cyl_y = CY + RADIUS * np.sin(theta)
        omegas = []
        speeds = []
        for (_t, ux, uy) in snapshots:
            omegas.append(np.asarray(_nodal_vorticity(mesh, ux, uy)))
            speeds.append(np.sqrt(ux ** 2 + uy ** 2))
        w_max = max(np.percentile(np.abs(w), 98) for w in omegas)
        s_max = max(np.percentile(s, 99) for s in speeds)
        levels_w = np.linspace(-w_max, w_max, 31)
        levels_s = np.linspace(0, s_max, 21)
        fig2, axes = plt.subplots(
            2, 1, figsize=(13, 5),
            sharex=True, constrained_layout=True,
        )
        def draw(i):
            for ax in axes:
                ax.clear()
            t_i, ux_i, uy_i = snapshots[i]
            w_i = omegas[i].T.reshape(-1)
            s_i = speeds[i].T.reshape(-1)
            axes[0].tricontourf(px, py, w_i, levels=levels_w,
                                cmap="RdBu_r", extend="both")
            axes[0].fill(cyl_x, cyl_y, color="black")
            axes[0].set_aspect("equal")
            axes[0].set_title(
                f"vorticity $\\omega$    t = {t_i:.2f}")
            axes[1].tricontourf(px, py, s_i, levels=levels_s,
                                cmap="viridis", extend="max")
            axes[1].fill(cyl_x, cyl_y, color="black")
            axes[1].set_aspect("equal")
            axes[1].set_title("speed $|U|$")
            return []
        anim = FuncAnimation(
            fig2, draw, frames=len(snapshots), interval=60, blit=False)
        out_mp4 = args.out_dir / f"{stem}.mp4"
        anim.save(str(out_mp4), writer="ffmpeg", dpi=100)
        plt.close(fig2)
        print(f"  wrote {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
