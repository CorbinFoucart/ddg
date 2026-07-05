"""Validation tests for the velocity-pressure incompressible NS solver.

Three benchmarks against analytical solutions:

1. **Plane Poiseuille flow** (steady): trivial fixed-point check —
   the solver should preserve the parabolic inflow profile to
   machine precision.

2. **Taylor-Green vortex** (decaying, time-dependent): the canonical
   2D NS analytical solution with exponential decay. We tag three
   sides as inflow (Dirichlet velocity = exact, Neumann pressure)
   and one as outflow (Neumann velocity = exact ∂u/∂n, Dirichlet
   pressure = exact p) so the system is well-posed without periodic
   BCs. Exercises both spatial and temporal accuracy.

3. **Kovasznay flow** (steady, smooth): wake-like analytical solution
   parameterised by Reynolds number. Steady so only spatial accuracy
   is tested; uses the same one-face-outflow trick.
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
    BDF1, BDF2, build_ins_operators, ns_advection_2d, ns_step_2d,
    ns_step_consistent_splitting_2d,
)


def _run_one_step(
    scheme, mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
    *, dt, nu, bdf, ops, ins_bc, bc_ux, bc_uy,
    p_dir=None, q_ux_neu=None, q_uy_neu=None,
):
    """Adapter so tests can plug in either splitting scheme uniformly.

    Returns ``(ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n)``. For consistent
    splitting the last three are zeros (it doesn't use that history).
    """
    if scheme == "dual":
        return ns_step_2d(
            mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=dt, nu=nu, bdf=bdf, ops=ops, ins_bc_types=ins_bc,
            bc_ux=bc_ux, bc_uy=bc_uy,
            p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
        )
    elif scheme == "consistent":
        ux_new, uy_new, p = ns_step_consistent_splitting_2d(
            mesh, ux, uy, ux_old, uy_old,
            dt=dt, nu=nu, bdf=bdf, ops=ops, ins_bc_types=ins_bc,
            bc_ux=bc_ux, bc_uy=bc_uy,
            p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
        )
        zero_face = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
        return ux_new, uy_new, p, jnp.zeros_like(ux), jnp.zeros_like(uy), zero_face
    raise ValueError(f"unknown scheme {scheme!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel_l2_velocity(mesh, ux, uy, ux_exact, uy_exact):
    num = jnp.sum(mesh.J * ((ux - ux_exact) ** 2 + (uy - uy_exact) ** 2))
    den = jnp.sum(mesh.J * (ux_exact ** 2 + uy_exact ** 2))
    return float(jnp.sqrt(num / den))


def _build_rect_mesh(xmin, xmax, ymin, ymax, Kx, Ky, N):
    VX, VY, EToV = uniform_rectangle_mesh_2d(xmin, xmax, ymin, ymax, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def _bc_three_inflow_one_outflow(mesh, outflow_normal):
    """Tag three sides as inflow + one as outflow.

    ``outflow_normal`` is ``"+x" | "-x" | "+y" | "-y"`` selecting which
    side gets the outflow tag. All other boundary faces become inflow.
    Interior stays interior.
    """
    Nfaces, K = mesh.Nfaces, mesh.K
    nx_face = np.asarray(mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    ny_face = np.asarray(mesh.ny.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T

    is_outflow = np.zeros_like(is_bdy)
    if outflow_normal == "+x":
        is_outflow = is_bdy & (nx_face > 0.5)
    elif outflow_normal == "-x":
        is_outflow = is_bdy & (nx_face < -0.5)
    elif outflow_normal == "+y":
        is_outflow = is_bdy & (ny_face > 0.5)
    elif outflow_normal == "-y":
        is_outflow = is_bdy & (ny_face < -0.5)
    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    bc = np.where(is_bdy & ~is_outflow, BC_INS_INFLOW, bc)
    bc = np.where(is_outflow, BC_INS_OUTFLOW, bc)
    return jnp.asarray(bc.T)


# ---------------------------------------------------------------------------
# Test 1: Steady plane Poiseuille flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["dual", "consistent"])
def test_steady_poiseuille_is_fixed_point(scheme):
    """Plane Poiseuille between no-slip walls is an exact NS steady
    state; one BDF1 step from it must reproduce it to machine
    precision (and the pressure drop matches the Stokes prediction
    Δp = −2νL). Both splitting schemes should preserve it exactly.
    """
    mesh = _build_rect_mesh(0.0, 2.0, 0.0, 1.0, 4, 2, 2)
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
    bc = np.where(is_bdy & (np.abs(nx_face) > 0.5) & (x_face < 1.0), BC_INS_INFLOW, bc)
    bc = np.where(is_bdy & (np.abs(nx_face) > 0.5) & (x_face > 1.0), BC_INS_OUTFLOW, bc)
    bc = np.where(is_bdy & (np.abs(ny_face) > 0.5), BC_INS_WALL, bc)
    ins_bc = jnp.asarray(bc.T)

    ux0 = mesh.y * (1.0 - mesh.y)
    uy0 = jnp.zeros_like(mesh.y)
    bc_ux, bc_uy = ux0, uy0

    dt, nu = 0.01, 0.1
    ops = build_ins_operators(mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc)
    dpdn_zero = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    ux_new, uy_new, p, _, _, _ = _run_one_step(
        scheme, mesh, ux0, uy0, ux0, uy0,
        jnp.zeros_like(ux0), jnp.zeros_like(ux0), dpdn_zero,
        dt=dt, nu=nu, bdf=BDF1, ops=ops, ins_bc=ins_bc,
        bc_ux=bc_ux, bc_uy=bc_uy,
    )
    drift_u = float(jnp.max(jnp.abs(ux_new - ux0)))
    p_drop = float(jnp.max(p) - jnp.min(p))
    expected_drop = 2.0 * nu * 2.0  # = -2ν · L (L=2)
    print(f"{scheme}: Poiseuille drift = {drift_u:.2e}, Δp = {p_drop:.4f} "
          f"(expected ≈ {expected_drop:.4f})")
    assert drift_u < 1e-10
    assert abs(p_drop - expected_drop) < 1e-6


# ---------------------------------------------------------------------------
# Test 2: Taylor-Green vortex (decaying)
# ---------------------------------------------------------------------------

def _taylor_green(mesh, t, nu):
    """Analytical Taylor-Green velocity and pressure."""
    x, y = mesh.x, mesh.y
    decay_v = jnp.exp(-2.0 * nu * t)
    decay_p = jnp.exp(-4.0 * nu * t)
    ux = -jnp.cos(x) * jnp.sin(y) * decay_v
    uy = jnp.sin(x) * jnp.cos(y) * decay_v
    p = -0.25 * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y)) * decay_p
    return ux, uy, p


def _taylor_green_face_data(mesh, t, nu):
    """Build face-trace ``(Nfp*Nfaces, K)`` arrays for Neumann + pressure BC.

    Returns ``(q_ux_neu, q_uy_neu, p_dir_volume)`` where the velocity
    Neumann arrays carry ``∂u/∂n`` evaluated at face nodes (using the
    K-side outward normal), and ``p_dir_volume`` is the analytical
    pressure as a volume field (only boundary face traces matter).

    ``∂u/∂n = ∇u · n``, so we compute the velocity-gradient components
    analytically and dot with the face normals.
    """
    x_M_flat = mesh.x.T.reshape(-1)[mesh.vmapM]
    y_M_flat = mesh.y.T.reshape(-1)[mesh.vmapM]
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    decay_v = jnp.exp(-2.0 * nu * t)

    # Analytical gradients:
    #   ux = -cos(x) sin(y) → ∂_x ux =  sin(x) sin(y), ∂_y ux = -cos(x) cos(y)
    #   uy =  sin(x) cos(y) → ∂_x uy =  cos(x) cos(y), ∂_y uy = -sin(x) sin(y)
    dux_dx = jnp.sin(x_M_flat) * jnp.sin(y_M_flat) * decay_v
    dux_dy = -jnp.cos(x_M_flat) * jnp.cos(y_M_flat) * decay_v
    duy_dx = jnp.cos(x_M_flat) * jnp.cos(y_M_flat) * decay_v
    duy_dy = -jnp.sin(x_M_flat) * jnp.sin(y_M_flat) * decay_v

    q_ux_neu_flat = dux_dx * nx_flat + dux_dy * ny_flat
    q_uy_neu_flat = duy_dx * nx_flat + duy_dy * ny_flat

    q_ux = q_ux_neu_flat.reshape((mesh.Nfp * mesh.Nfaces, mesh.K), order="F")
    q_uy = q_uy_neu_flat.reshape((mesh.Nfp * mesh.Nfaces, mesh.K), order="F")
    _, _, p_dir = _taylor_green(mesh, t, nu)
    return q_ux, q_uy, p_dir


@pytest.mark.parametrize("scheme", ["dual", "consistent"])
@pytest.mark.parametrize("Kx,Ky,N,dt,T", [
    (8, 8, 4, 0.02, 0.1),
    (8, 8, 4, 0.01, 0.1),
])
def test_taylor_green_decays_correctly(scheme, Kx, Ky, N, dt, T):
    """Run Taylor-Green for time T and check L2 error vs exact decay."""
    mesh = _build_rect_mesh(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, Kx, Ky, N)
    ins_bc = _bc_three_inflow_one_outflow(mesh, "+x")
    nu = 0.1

    ux, uy, _ = _taylor_green(mesh, 0.0, nu)

    n_steps = int(round(T / dt))
    ops_bdf1 = build_ins_operators(mesh, dt=dt, nu=nu, g0=BDF1.g0,
                                    ins_bc_types=ins_bc)
    ops_bdf2 = build_ins_operators(mesh, dt=dt, nu=nu, g0=BDF2.g0,
                                    ins_bc_types=ins_bc)

    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=ux, bc_uy=uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))

    for step in range(1, n_steps + 1):
        t_np1 = step * dt
        bc_ux_n, bc_uy_n, _ = _taylor_green(mesh, t_np1, nu)
        q_ux_neu, q_uy_neu, p_dir = _taylor_green_face_data(mesh, t_np1, nu)
        bdf = BDF1 if step == 1 else BDF2
        ops = ops_bdf1 if step == 1 else ops_bdf2

        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = _run_one_step(
            scheme, mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=dt, nu=nu, bdf=bdf, ops=ops, ins_bc=ins_bc,
            bc_ux=bc_ux_n, bc_uy=bc_uy_n,
            p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
        )
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n

    ux_exact, uy_exact, _ = _taylor_green(mesh, T, nu)
    err = _rel_l2_velocity(mesh, ux, uy, ux_exact, uy_exact)
    print(f"{scheme}: TG Kx={Kx} Ky={Ky} N={N} dt={dt} T={T}: "
          f"rel L2 err = {err:.2e}")
    assert err < 5e-3


# ---------------------------------------------------------------------------
# Test 3: Kovasznay flow (steady, smooth)
# ---------------------------------------------------------------------------

def _kovasznay(mesh, nu):
    """Kovasznay analytical solution.

    ``λ = Re/2 − √(Re²/4 + 4π²)``, ``Re = 1/ν``.
    ``u = 1 − exp(λx) cos(2πy)``
    ``v = (λ/(2π)) exp(λx) sin(2πy)``
    ``p = 0.5 (1 − exp(2λx))``     (so ``p = 0`` at ``x = 0``)
    """
    Re = 1.0 / nu
    lam = 0.5 * Re - np.sqrt(0.25 * Re ** 2 + 4 * np.pi ** 2)
    x, y = mesh.x, mesh.y
    e = jnp.exp(lam * x)
    ux = 1.0 - e * jnp.cos(2 * jnp.pi * y)
    uy = (lam / (2 * jnp.pi)) * e * jnp.sin(2 * jnp.pi * y)
    p = 0.5 * (1.0 - jnp.exp(2 * lam * x))
    return ux, uy, p, lam


def _kovasznay_face_data(mesh, nu, lam):
    """Build ``(q_ux_neu, q_uy_neu, p_dir)`` for Kovasznay outflow BC."""
    x_M = mesh.x.T.reshape(-1)[mesh.vmapM]
    y_M = mesh.y.T.reshape(-1)[mesh.vmapM]
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    e = jnp.exp(lam * x_M)
    twopi = 2 * jnp.pi
    # ∂u_x/∂x = -λ exp(λx) cos(2π y),  ∂u_x/∂y = 2π exp(λx) sin(2π y)
    dux_dx = -lam * e * jnp.cos(twopi * y_M)
    dux_dy = twopi * e * jnp.sin(twopi * y_M)
    # ∂u_y/∂x = (λ²/(2π)) exp(λx) sin(2π y),  ∂u_y/∂y = λ exp(λx) cos(2π y)
    duy_dx = (lam ** 2 / twopi) * e * jnp.sin(twopi * y_M)
    duy_dy = lam * e * jnp.cos(twopi * y_M)
    q_ux_flat = dux_dx * nx_flat + dux_dy * ny_flat
    q_uy_flat = duy_dx * nx_flat + duy_dy * ny_flat
    q_ux = q_ux_flat.reshape((mesh.Nfp * mesh.Nfaces, mesh.K), order="F")
    q_uy = q_uy_flat.reshape((mesh.Nfp * mesh.Nfaces, mesh.K), order="F")
    _, _, p_dir, _ = _kovasznay(mesh, nu)
    return q_ux, q_uy, p_dir


@pytest.mark.parametrize("scheme", ["dual", "consistent"])
@pytest.mark.parametrize("Kx,Ky,N", [(8, 8, 4)])
def test_kovasznay_steady_solution(scheme, Kx, Ky, N):
    """Time-march from the exact Kovasznay IC; should stay close to it."""
    mesh = _build_rect_mesh(-0.5, 1.5, -0.5, 1.5, Kx, Ky, N)
    ins_bc = _bc_three_inflow_one_outflow(mesh, "+x")
    nu = 0.1
    ux, uy, p_exact, lam = _kovasznay(mesh, nu)
    bc_ux, bc_uy = ux, uy
    q_ux_neu, q_uy_neu, p_dir = _kovasznay_face_data(mesh, nu, lam)

    dt = 0.01
    T = 0.1
    n_steps = int(round(T / dt))
    ops_bdf1 = build_ins_operators(mesh, dt=dt, nu=nu, g0=BDF1.g0,
                                    ins_bc_types=ins_bc)
    ops_bdf2 = build_ins_operators(mesh, dt=dt, nu=nu, g0=BDF2.g0,
                                    ins_bc_types=ins_bc)
    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=bc_ux, bc_uy=bc_uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    for step in range(1, n_steps + 1):
        bdf = BDF1 if step == 1 else BDF2
        ops = ops_bdf1 if step == 1 else ops_bdf2
        ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = _run_one_step(
            scheme, mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
            dt=dt, nu=nu, bdf=bdf, ops=ops, ins_bc=ins_bc,
            bc_ux=bc_ux, bc_uy=bc_uy,
            p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
        )
        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n

    ux_exact, uy_exact, _, _ = _kovasznay(mesh, nu)
    err = _rel_l2_velocity(mesh, ux, uy, ux_exact, uy_exact)
    print(f"{scheme}: Kovasznay Kx={Kx} Ky={Ky} N={N}: rel L2 err = {err:.2e}")
    assert err < 5e-3


@pytest.mark.parametrize("scheme", ["dual", "consistent"])
def test_penalty_wire_in_doesnt_break_taylor_green(scheme):
    """Pass a moderate-tau penalty projection in via the step driver.
    The result should still match Taylor-Green at the expected accuracy
    (small penalty acts as a mild regulariser; doesn't blow up the
    smooth, well-resolved benchmark)."""
    from ddg.dg2d.incompressible_ns import (
        factor_penalty_projection_2d,
    )
    mesh = _build_rect_mesh(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 8, 8, 4)
    ins_bc = _bc_three_inflow_one_outflow(mesh, "+x")
    nu = 0.1
    dt = 0.01
    T = 0.1
    n_steps = int(round(T / dt))

    # Tiny penalty: should be ~ noop for a smooth benchmark
    penalty = factor_penalty_projection_2d(mesh, tau_div=0.01, tau_cont=0.01)

    ops_bdf1 = build_ins_operators(mesh, dt=dt, nu=nu, g0=BDF1.g0, ins_bc_types=ins_bc)
    ops_bdf2 = build_ins_operators(mesh, dt=dt, nu=nu, g0=BDF2.g0, ins_bc_types=ins_bc)
    ux, uy, _ = _taylor_green(mesh, 0.0, nu)
    ux_old, uy_old = ux, uy
    Nux_old, Nuy_old = ns_advection_2d(
        mesh, ux, uy, ins_bc_types=ins_bc, bc_ux=ux, bc_uy=uy,
    )
    dpdn_old = jnp.zeros((mesh.Nfp * mesh.Nfaces, mesh.K))
    for step in range(1, n_steps + 1):
        t_np1 = step * dt
        bc_ux_n, bc_uy_n, _ = _taylor_green(mesh, t_np1, nu)
        q_ux_neu, q_uy_neu, p_dir = _taylor_green_face_data(mesh, t_np1, nu)
        bdf = BDF1 if step == 1 else BDF2
        ops = ops_bdf1 if step == 1 else ops_bdf2

        if scheme == "dual":
            from ddg.dg2d.incompressible_ns import ns_step_2d
            ux_new, uy_new, p, Nux_n, Nuy_n, dpdn_n = ns_step_2d(
                mesh, ux, uy, ux_old, uy_old, Nux_old, Nuy_old, dpdn_old,
                dt=dt, nu=nu, bdf=bdf, ops=ops, ins_bc_types=ins_bc,
                bc_ux=bc_ux_n, bc_uy=bc_uy_n,
                p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
                penalty_factored=penalty,
            )
        else:
            from ddg.dg2d.incompressible_ns import (
                ns_step_consistent_splitting_2d,
            )
            ux_new, uy_new, p = ns_step_consistent_splitting_2d(
                mesh, ux, uy, ux_old, uy_old,
                dt=dt, nu=nu, bdf=bdf, ops=ops, ins_bc_types=ins_bc,
                bc_ux=bc_ux_n, bc_uy=bc_uy_n,
                p_dir=p_dir, q_ux_neu=q_ux_neu, q_uy_neu=q_uy_neu,
                penalty_factored=penalty,
            )
            Nux_n, Nuy_n = jnp.zeros_like(ux), jnp.zeros_like(uy)
            dpdn_n = jnp.zeros_like(dpdn_old)

        ux_old, uy_old = ux, uy
        ux, uy = ux_new, uy_new
        Nux_old, Nuy_old = Nux_n, Nuy_n
        dpdn_old = dpdn_n

    ux_exact, uy_exact, _ = _taylor_green(mesh, T, nu)
    err = _rel_l2_velocity(mesh, ux, uy, ux_exact, uy_exact)
    print(f"{scheme} + penalty: TG err = {err:.2e}")
    assert err < 5e-3


if __name__ == "__main__":
    test_steady_poiseuille_is_fixed_point()
    test_taylor_green_decays_correctly(8, 8, 4, 0.02, 0.1)
    test_taylor_green_decays_correctly(8, 8, 4, 0.01, 0.1)
    test_kovasznay_steady_solution(8, 8, 4)
