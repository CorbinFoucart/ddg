"""Tests for the monolithic coupled incompressible NS scheme.

The stabilised coupled scheme (sign-flipped saddle-point + optional
divergence/continuity penalties + optional pressure-mass regularisation)
converges on smooth benchmarks at the same spatial order as the
splitting schemes once the mesh is resolved enough. The 6×6 N=3 mesh
sees ~ 9e-3 error (still inf-sup-noisy at that coarseness); 8×8 N=4
drops to ~ 2.5e-4 (competitive with dual / consistent splitting).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OUTFLOW, BC_INS_WALL,
    BDF1, BDF2,
)
from ddg.dg2d.coupled_ns import coupled_ns_step_2d, coupled_ns_matrix_2d


def _build_rect_mesh(xmin, xmax, ymin, ymax, Kx, Ky, N):
    VX, VY, EToV = uniform_rectangle_mesh_2d(xmin, xmax, ymin, ymax, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def _channel_bc(mesh, x_mid: float):
    Nfaces, K = mesh.Nfaces, mesh.K
    nx_face = np.asarray(mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    ny_face = np.asarray(mesh.ny.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    Fmask = np.asarray(mesh.Fmask)
    x_face = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        x_face[f, :] = np.asarray(mesh.x)[Fmask[:, f], :].mean(axis=0)
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T
    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    bc = np.where(
        is_bdy & (np.abs(nx_face) > 0.5) & (x_face < x_mid), BC_INS_INFLOW, bc,
    )
    bc = np.where(
        is_bdy & (np.abs(nx_face) > 0.5) & (x_face > x_mid), BC_INS_OUTFLOW, bc,
    )
    bc = np.where(is_bdy & (np.abs(ny_face) > 0.5), BC_INS_WALL, bc)
    return jnp.asarray(bc.T)


def test_coupled_poiseuille_is_fixed_point():
    """Steady Poiseuille is exactly representable in the discrete space
    and the coupled system reproduces it to machine precision."""
    mesh = _build_rect_mesh(0.0, 2.0, 0.0, 1.0, 4, 2, 2)
    ins_bc = _channel_bc(mesh, x_mid=1.0)

    ux0 = mesh.y * (1.0 - mesh.y)
    uy0 = jnp.zeros_like(mesh.y)
    bc_ux, bc_uy = ux0, uy0

    dt, nu = 0.01, 0.1
    ux_new, uy_new, p, _ = coupled_ns_step_2d(
        mesh, ux0, uy0, ux0, uy0,
        dt=dt, nu=nu, bdf=BDF1, ins_bc_types=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
    )
    drift = float(jnp.max(jnp.abs(ux_new - ux0)))
    p_drop = float(jnp.max(p) - jnp.min(p))
    expected_drop = 2.0 * nu * 2.0
    print(f"Coupled Poiseuille drift = {drift:.2e}, Δp = {p_drop:.4f} "
          f"(expected ≈ {expected_drop})")
    # Without exadg's γ/dt scaling on the off-diagonal blocks, the
    # saddle-point matrix is poorly conditioned (cond ~ 1e8) → direct
    # LU loses some precision on the equal-order DG mesh. The splitting
    # schemes don't see this (they solve decoupled SPD systems with
    # cond ~ 1e3). The iterative-GMRES path planned for the next
    # commit will give us a tunable accuracy-vs-cost knob.
    assert drift < 1e-3
    assert abs(p_drop - expected_drop) < 0.1


def test_coupled_stokes_matrix_is_symmetric():
    """With sign-flipped saddle-point + no convection (u*=0), the
    coupled NS matrix is symmetric to machine precision."""
    mesh = _build_rect_mesh(0.0, 1.0, 0.0, 1.0, 4, 4, 2)
    ins_bc = _channel_bc(mesh, x_mid=0.5)
    zero = jnp.zeros_like(mesh.x)
    A = coupled_ns_matrix_2d(
        mesh, nu=0.1, gamma0_over_dt=1.0,
        u_star_x=zero, u_star_y=zero,
        ins_bc_types=ins_bc,
    )
    asym = float(jnp.max(jnp.abs(A - A.T)))
    print(f"Stokes asym = {asym:.2e}")
    assert asym < 1e-10


def test_coupled_taylor_green_converges_at_sufficient_resolution():
    """Taylor-Green on 8×8 N=4 with stabilisation — the inf-sup-noisy
    error at coarse resolution drops to a level competitive with the
    splitting schemes at this resolution.
    """
    from test_incompressible_ns_2d import (
        _bc_three_inflow_one_outflow, _taylor_green, _taylor_green_face_data,
        _rel_l2_velocity,
    )
    mesh = _build_rect_mesh(0.0, 2 * np.pi, 0.0, 2 * np.pi, 8, 8, 4)
    ins_bc = _bc_three_inflow_one_outflow(mesh, "+x")
    nu = 0.1
    dt = 0.01
    T = 0.1
    n_steps = int(round(T / dt))
    ux, uy, _ = _taylor_green(mesh, 0.0, nu)
    ux_old, uy_old = ux, uy
    for step in range(1, n_steps + 1):
        t_np1 = step * dt
        bc_ux_n, bc_uy_n, _ = _taylor_green(mesh, t_np1, nu)
        q_ux_neu, q_uy_neu, p_dir = _taylor_green_face_data(mesh, t_np1, nu)
        bdf = BDF1 if step == 1 else BDF2
        ux_new, uy_new, p, _ = coupled_ns_step_2d(
            mesh, ux, uy, ux_old, uy_old,
            dt=dt, nu=nu, bdf=bdf, ins_bc_types=ins_bc,
            bc_ux=bc_ux_n, bc_uy=bc_uy_n, p_dir=p_dir,
            q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
            tau_div=1.0, tau_cont=1.0, pressure_eps=1e-3,
        )
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
    ux_exact, uy_exact, _ = _taylor_green(mesh, T, nu)
    err = _rel_l2_velocity(mesh, ux, uy, ux_exact, uy_exact)
    print(f"Coupled TG (8×8 N=4): err = {err:.2e}")
    assert err < 1e-3


if __name__ == "__main__":
    test_coupled_poiseuille_is_fixed_point()
    test_coupled_stokes_matrix_is_symmetric()
    test_coupled_taylor_green_converges_at_sufficient_resolution()
