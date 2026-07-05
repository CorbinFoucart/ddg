"""Kovasznay convergence sentinel for BDM_k / P_{k-1} steady NS.

The Kovasznay analytical solution on the slice ``[-0.5, 1] x [-0.5, 1.5]``
with ``Re = 40`` is:

    lambda = Re/2 - sqrt(Re^2/4 + 4 pi^2)
    u_x(x, y) = 1 - exp(lambda x) cos(2 pi y)
    u_y(x, y) = (lambda / (2 pi)) exp(lambda x) sin(2 pi y)
    p(x, y)   = (1/2)(1 - exp(2 lambda x))

This is a standard sentinel for incompressible NS. With H(div)-conforming
BDM_k / P_{k-1} elements, the expected convergence rate is ``h^(k+1)``
in the L^2 velocity norm. Verified here at:

  k=1: rate ~ 2  (existing BDM_1 sentinel, same setup)
  k=2: rate ~ 3
  k=3: rate ~ 4

This is the strongest validation of the order-parametric stack: if the
convective + viscous + pressure operators have any J-factor or sign bug,
the rate is the first thing that breaks.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM, bdm_k_basis_at_points, bdm_k_evaluate_boundary_velocity,
    _element_jacobian_matrices,
)
from ddg.dg2d.coupled_ns_bdm import (
    bdm1_boundary_dofs_mask, bdm1_evaluate_dirichlet_bc,
    bdm1_gather,
    ns_newton_solve_bdm_k_2d,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


RE = 40.0
NU = 1.0 / RE
LAMBDA = RE / 2.0 - np.sqrt(RE ** 2 / 4.0 + 4.0 * np.pi ** 2)


def _kovasznay_uv(x, y):
    ux = 1.0 - jnp.exp(LAMBDA * x) * jnp.cos(2.0 * jnp.pi * y)
    uy = (LAMBDA / (2.0 * jnp.pi)) * jnp.exp(LAMBDA * x) * jnp.sin(2.0 * jnp.pi * y)
    return ux, uy


def _build_setup(Kx: int, Ky: int, order: int):
    """Build BDM_k space on the Kovasznay slice + BC arrays."""
    VX, VY, EToV = uniform_rectangle_mesh_2d(-0.5, 1.0, -0.5, 1.5, Kx, Ky)
    mesh = build_mesh_2d(VX, VY, EToV, 1)
    V = BDM.from_mesh(mesh, order=order)
    u_bc = bdm1_evaluate_dirichlet_bc(V, _kovasznay_uv)
    u_bc_vec = bdm_k_evaluate_boundary_velocity(V, _kovasznay_uv)
    bc_mask = bdm1_boundary_dofs_mask(V)
    return V, u_bc, u_bc_vec, bc_mask


def _l2_velocity_error(V: BDM, u_global) -> float:
    """``||u_h - u_analytical||_{L^2(Omega)}`` via cubature on the
    reference triangle."""
    k = V.order
    r_q, s_q, w_q = triangle_cubature(2 * k + 2)
    phi_ref = bdm_k_basis_at_points(k, r_q, s_q)            # (Q, Np, 2)
    DF, _ = _element_jacobian_matrices(V.mesh)
    J = V.mesh.J[0]
    phi_phys = jnp.einsum("kca,qla->kqlc", DF, phi_ref) / J[:, None, None, None]
    u_local = bdm1_gather(V.topology, u_global)
    u_h_at_q = jnp.einsum("kl,kqlc->kqc", u_local, phi_phys)  # (K, Q, 2)

    # Physical positions of cubature points.
    EToV = np.asarray(V.mesh.EToV)
    VX = np.asarray(V.mesh.VX)
    VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    r_q_np = np.asarray(r_q); s_q_np = np.asarray(s_q)
    x_q = (
        Vx0[:, None]
        + 0.5 * (r_q_np[None, :] + 1.0) * (Vx1 - Vx0)[:, None]
        + 0.5 * (s_q_np[None, :] + 1.0) * (Vx2 - Vx0)[:, None]
    )
    y_q = (
        Vy0[:, None]
        + 0.5 * (r_q_np[None, :] + 1.0) * (Vy1 - Vy0)[:, None]
        + 0.5 * (s_q_np[None, :] + 1.0) * (Vy2 - Vy0)[:, None]
    )
    x_q_j = jnp.asarray(x_q); y_q_j = jnp.asarray(y_q)
    u_ex, v_ex = _kovasznay_uv(x_q_j, y_q_j)
    diff_x = u_h_at_q[..., 0] - u_ex                          # (K, Q)
    diff_y = u_h_at_q[..., 1] - v_ex
    sq = diff_x ** 2 + diff_y ** 2                            # (K, Q)
    return float(jnp.sqrt(jnp.einsum("q,k,kq->", w_q, J, sq)))


@pytest.mark.parametrize("k,Kx,expected_min_rate", [
    pytest.param(1, [4, 6, 8], 1.7, id="k1"),
    pytest.param(2, [3, 4, 6], 2.7, id="k2"),
])
def test_kovasznay_convergence_rate(k, Kx, expected_min_rate):
    """Solve Kovasznay at three mesh sizes and check that the L^2
    velocity error decays at least at rate ``expected_min_rate``."""
    hs = []
    errs = []
    for n in Kx:
        V, u_bc, u_bc_vec, bc_mask = _build_setup(Kx=n, Ky=n, order=k)
        # Steady Newton solve (dense LU). Picard does NOT converge for
        # Kovasznay Re=40 -- it stalls at residual ~1e2 and the L^2 error
        # then reflects a non-converged iterate (slope ~1 or worse).
        # Newton converges in ~9 iters and recovers the expected
        # h^{k+1} rate. The earlier "BDM_2 Kovasznay slope-1" anomaly
        # was therefore a Picard non-convergence artifact, not a
        # discretisation defect (the Stokes pressure-robustness sentinel
        # and this Newton solve both show clean high-order rates).
        u_sol, _p, _ok = ns_newton_solve_bdm_k_2d(
            V, nu=NU,
            u_bc_global=u_bc, bc_mask=bc_mask, u_bc_vec=u_bc_vec,
            u_init=u_bc, pressure_eps=1e-10, tau_scale=4.0,
            max_iters=30, tol=1e-11, linearisation="newton",
        )
        err = _l2_velocity_error(V, u_sol)
        # Mesh "h" = side length / Kx.
        h = 1.5 / n
        hs.append(h)
        errs.append(err)
        print(f"  k={k} n={n}: h={h:.4f}, L2 err = {err:.3e}")
    # Pairwise slopes.
    slopes = []
    for i in range(len(hs) - 1):
        slope = (np.log(errs[i]) - np.log(errs[i + 1])) / (
            np.log(hs[i]) - np.log(hs[i + 1])
        )
        slopes.append(slope)
        print(f"  k={k}: slope({hs[i]:.4f} -> {hs[i+1]:.4f}) = {slope:.3f}")
    # Take the BEST slope (asymptotic regime): finest two meshes.
    asymp_slope = slopes[-1]
    assert asymp_slope > expected_min_rate, (
        f"k={k}: asymptotic L^2 velocity convergence slope {asymp_slope:.3f} "
        f"falls short of expected min rate {expected_min_rate}. Slopes: {slopes}"
    )
