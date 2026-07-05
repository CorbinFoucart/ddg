"""BDM_k / P_{k-1} Stokes convergence-rate sentinel against a smooth
manufactured solution.

This is the canonical sentinel for SIPG consistency: with a smooth
non-polynomial analytical solution, the L^2 velocity error must
reduce at the theoretical rate ``h^(k+1)`` as the mesh refines. A
J-factor inconsistency between the volume and face/lift terms (as
existed in the pre-fix bdm1 viscous bilinear form) shows up as
slope-1 instead of slope-(k+1).

Manufactured solution:
    u(x, y) = (sin(pi x) cos(pi y), -cos(pi x) sin(pi y))     (Taylor-Green)
    p(x, y) = 0
    nu      = 1
    Delta u = -2 pi^2 u  =>  f = -nu Delta u = 2 pi^2 u    (body force)
Domain: [0, 1]^2.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM, bdm_k_basis_at_points, bdm_k_evaluate_boundary_velocity,
    bdm_k_viscous_dirichlet_lift, bdm1_scatter_add,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc, bdm1_gather,
    stokes_matrix_bdm_k_2d,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


PI = jnp.pi
NU = 1.0


def _u_anal(x, y):
    return (jnp.sin(PI * x) * jnp.cos(PI * y),
            -jnp.cos(PI * x) * jnp.sin(PI * y))


def _build_load(V):
    """``f_u[ell] = integral_K f . phi_phys_ell dx`` with
    ``f = 2 pi^2 nu u_anal``."""
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 2)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum(
        "kca,qla->kqlc", DF, phi_ref,
    ) / J[:, None, None, None]
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    rqn = np.asarray(r_q); sqn = np.asarray(s_q)
    x_q = (
        Vx0[:, None]
        + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None]
        + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
    )
    y_q = (
        Vy0[:, None]
        + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None]
        + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
    )
    ux, uy = _u_anal(jnp.asarray(x_q), jnp.asarray(y_q))
    fx = 2 * PI ** 2 * NU * ux
    fy = 2 * PI ** 2 * NU * uy
    f_local = (
        jnp.einsum("q,k,kq,kql->kl", w_q, J, fx, phi_phys[..., 0])
        + jnp.einsum("q,k,kq,kql->kl", w_q, J, fy, phi_phys[..., 1])
    )
    return bdm1_scatter_add(V.topology, f_local)


def _l2_velocity_error(V, u_global):
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 4)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum(
        "kca,qla->kqlc", DF, phi_ref,
    ) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    uh = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    rqn = np.asarray(r_q); sqn = np.asarray(s_q)
    x_q = (
        Vx0[:, None]
        + 0.5 * (rqn + 1.0)[None, :] * (Vx1 - Vx0)[:, None]
        + 0.5 * (sqn + 1.0)[None, :] * (Vx2 - Vx0)[:, None]
    )
    y_q = (
        Vy0[:, None]
        + 0.5 * (rqn + 1.0)[None, :] * (Vy1 - Vy0)[:, None]
        + 0.5 * (sqn + 1.0)[None, :] * (Vy2 - Vy0)[:, None]
    )
    ux, uy = _u_anal(jnp.asarray(x_q), jnp.asarray(y_q))
    dx = uh[..., 0] - ux
    dy = uh[..., 1] - uy
    return float(jnp.sqrt(jnp.einsum("q,k,kq->", w_q, J, dx ** 2 + dy ** 2)))


def _solve_stokes(V):
    k = V.order
    dim_p = k * (k + 1) // 2
    n_u = V.total_dofs
    n_p = V.K * dim_p
    u_bc = bdm1_evaluate_dirichlet_bc(V, _u_anal)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _u_anal)
    bc_mask = bdm1_boundary_dofs_mask(V)
    lift = bdm_k_viscous_dirichlet_lift(V, u_bc_vec, nu=NU, tau_scale=4.0)
    f_load = _build_load(V)
    A = stokes_matrix_bdm_k_2d(V, nu=NU, pressure_eps=1e-10, tau_scale=4.0)
    rhs = jnp.concatenate([f_load + lift, jnp.zeros(n_p)])
    sol = jnp.linalg.solve(A, rhs)
    u_sol = sol[:n_u]
    u_sol = jnp.where(bc_mask, u_bc, u_sol)
    return u_sol


@pytest.mark.parametrize(
    "k,ns,min_asymptotic_slope",
    [
        (1, [4, 8, 12, 16], 1.7),
        (2, [3, 4, 6, 8], 2.8),
        (3, [3, 4, 6], 3.8),
    ],
    ids=["k=1", "k=2", "k=3"],
)
def test_stokes_manufactured_solution_converges_at_h_kplus1(
    k, ns, min_asymptotic_slope,
):
    """Stokes manufactured solution must converge at ``h^(k+1)``.
    Bilinear-form consistency is the underlying property tested.
    Asymptotic slope (finest two meshes) must hit at least the
    threshold below, which is a small margin under the theoretical
    rate to allow for preasymptotic effects."""
    hs, errs = [], []
    for n in ns:
        VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, n, n)
        mesh = build_mesh_2d(VX, VY, EToV, 1)
        V = BDM.from_mesh(mesh, order=k)
        u_sol = _solve_stokes(V)
        err = _l2_velocity_error(V, u_sol)
        h = 1.0 / n
        hs.append(h); errs.append(err)
    slopes = []
    for i in range(len(errs) - 1):
        slope = (np.log(errs[i]) - np.log(errs[i + 1])) / (
            np.log(hs[i]) - np.log(hs[i + 1])
        )
        slopes.append(slope)
        print(f"  k={k}: slope({hs[i]:.4f} -> {hs[i+1]:.4f}) = {slope:.3f}")
    asymp = slopes[-1]
    assert asymp > min_asymptotic_slope, (
        f"k={k}: asymptotic L^2 slope {asymp:.3f} below threshold "
        f"{min_asymptotic_slope}. All slopes: {slopes}"
    )
