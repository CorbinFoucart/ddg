"""Tests for the SIPG-form 2D Poisson solver with mixed BCs.

Manufactured solutions on the unit square ``[0, 1] × [0, 1]``:

  - **Full Dirichlet**: ``u = sin(πx) sin(πy)`` with homogeneous
    Dirichlet on all four sides. Source ``2π² sin(πx) sin(πy)``.

  - **Mixed homog Dir/Neu**: ``u = sin(πx) cos(πy)`` with hom-Dirichlet
    on x ∈ {0, 1} and hom-Neumann on y ∈ {0, 1}. Source
    ``2π² sin(πx) cos(πy)``.

  - **Mixed inhomog**: ``u = x(1-x)y`` with hom-Dirichlet on x ∈ {0, 1}
    and y = 0, non-hom Neumann ``∂u/∂y = x(1-x)`` on y = 1.
    Source ``2y``.

We also check that the matrix is symmetric and positive definite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.sipg import (
    BC_DIRICHLET, BC_INTERIOR, BC_NEUMANN, IterativeConfig,
    factor_sipg_2d, sipg_matrix_2d, solve_sipg_helmholtz_2d,
    solve_sipg_poisson_2d,
)

jax.config.update("jax_enable_x64", True)


def _build_unit_square(Kx: int, Ky: int, N: int):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def _bc_types_by_normal(mesh, dir_predicate, neu_predicate):
    """Tag each face by its outward-normal direction.

    Both predicates take ``(nx_face_avg, ny_face_avg)`` and return a
    boolean per (K, f). Interior faces (where ``EToE == self``) are left
    as :data:`BC_INTERIOR`.
    """
    Nfaces = mesh.Nfaces
    K = mesh.K
    # Average outward normal across the Nfp face nodes (constant for affine)
    nx_face = mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F")[0]  # (Nfaces, K)
    ny_face = mesh.ny.reshape((mesh.Nfp, Nfaces, K), order="F")[0]
    # Boundary detection: self-pointing in EToE
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T.astype(bool)  # (Nfaces, K)
    nx_face_np = np.asarray(nx_face)
    ny_face_np = np.asarray(ny_face)
    bc = np.full((Nfaces, K), BC_INTERIOR, dtype=np.int32)
    dir_mask = dir_predicate(nx_face_np, ny_face_np) & is_bdy
    neu_mask = neu_predicate(nx_face_np, ny_face_np) & is_bdy
    bc = np.where(dir_mask, BC_DIRICHLET, bc)
    bc = np.where(neu_mask, BC_NEUMANN, bc)
    # bc is (Nfaces, K); we want (K, Nfaces) per the SIPG API
    return jnp.asarray(bc.T)


def _full_dirichlet_bc(mesh):
    return _bc_types_by_normal(
        mesh,
        dir_predicate=lambda nx, ny: np.ones_like(nx, dtype=bool),
        neu_predicate=lambda nx, ny: np.zeros_like(nx, dtype=bool),
    )


def _mixed_dir_x_neu_y(mesh):
    return _bc_types_by_normal(
        mesh,
        dir_predicate=lambda nx, ny: np.abs(nx) > 0.5,
        neu_predicate=lambda nx, ny: np.abs(ny) > 0.5,
    )


def _mixed_neu_top_only(mesh):
    """Dirichlet on x sides + y=0, Neumann on y=1 only."""
    Nfaces = mesh.Nfaces
    K = mesh.K
    ny_face_avg = mesh.ny.reshape((mesh.Nfp, Nfaces, K), order="F")[0]
    y_face_avg = mesh.y.reshape((mesh.Np, K))
    Fmask = np.asarray(mesh.Fmask)
    # average y at face: mean of face-node y values
    y_at_face = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        y_at_face[f, :] = np.asarray(mesh.y)[Fmask[:, f], :].mean(axis=0)
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T.astype(bool)
    ny_np = np.asarray(ny_face_avg)
    bc = np.full((Nfaces, K), BC_INTERIOR, dtype=np.int32)
    neu_mask = is_bdy & (np.abs(ny_np) > 0.5) & (y_at_face > 0.5)
    dir_mask = is_bdy & (~neu_mask)
    bc = np.where(dir_mask, BC_DIRICHLET, bc)
    bc = np.where(neu_mask, BC_NEUMANN, bc)
    return jnp.asarray(bc.T)


def _rel_l2(mesh, u, u_exact):
    return float(jnp.sqrt(
        jnp.sum(mesh.J * (u - u_exact) ** 2)
        / jnp.sum(mesh.J * u_exact ** 2)
    ))


def test_sipg_matrix_is_symmetric_for_full_dirichlet():
    mesh = _build_unit_square(6, 6, 3)
    bc = _full_dirichlet_bc(mesh)
    A = sipg_matrix_2d(mesh, bc_types=bc)
    asym = float(jnp.max(jnp.abs(A - A.T)))
    print(f"Full-Dirichlet SIPG: shape={A.shape}, asym={asym:.2e}")
    assert asym < 1e-10


def test_sipg_matrix_is_symmetric_for_mixed_bc():
    mesh = _build_unit_square(6, 6, 3)
    bc = _mixed_dir_x_neu_y(mesh)
    A = sipg_matrix_2d(mesh, bc_types=bc)
    asym = float(jnp.max(jnp.abs(A - A.T)))
    print(f"Mixed SIPG: shape={A.shape}, asym={asym:.2e}")
    assert asym < 1e-10


@pytest.mark.parametrize("N,Kx,Ky", [(3, 6, 6), (3, 10, 10), (4, 8, 8)])
def test_sipg_full_dirichlet(N, Kx, Ky):
    mesh = _build_unit_square(Kx, Ky, N)
    pi = jnp.pi
    u_exact = jnp.sin(pi * mesh.x) * jnp.sin(pi * mesh.y)
    source = 2 * pi ** 2 * u_exact
    bc = _full_dirichlet_bc(mesh)
    u = solve_sipg_poisson_2d(mesh, source, bc_types=bc)
    err = _rel_l2(mesh, u, u_exact)
    print(f"Full Dirichlet N={N} Kx={Kx} Ky={Ky}: rel L2 err = {err:.2e}")
    assert err < 1e-3


@pytest.mark.parametrize("N,Kx,Ky", [(3, 6, 6), (3, 10, 10), (4, 8, 8)])
def test_sipg_mixed_homog_neumann(N, Kx, Ky):
    """u = sin(πx) cos(πy): hom-Dir on x, hom-Neu on y."""
    mesh = _build_unit_square(Kx, Ky, N)
    pi = jnp.pi
    u_exact = jnp.sin(pi * mesh.x) * jnp.cos(pi * mesh.y)
    source = 2 * pi ** 2 * u_exact
    bc = _mixed_dir_x_neu_y(mesh)
    u = solve_sipg_poisson_2d(mesh, source, bc_types=bc)
    err = _rel_l2(mesh, u, u_exact)
    print(f"Mixed homog N={N} Kx={Kx} Ky={Ky}: rel L2 err = {err:.2e}")
    assert err < 1e-3


@pytest.mark.parametrize("N,Kx,Ky", [(3, 8, 8), (4, 8, 8)])
def test_sipg_mixed_nonhomog_neumann(N, Kx, Ky):
    """u = x(1-x)y: hom-Dir on x and y=0, non-hom Neu on y=1."""
    mesh = _build_unit_square(Kx, Ky, N)
    x, y = mesh.x, mesh.y
    u_exact = x * (1.0 - x) * y
    source = 2.0 * y
    bc = _mixed_neu_top_only(mesh)
    # Build Neumann data: q = ∂u/∂y at y=1 = x(1-x)
    # Evaluate at face-trace nodes: shape (Nfp*Nfaces, K)
    x_face = mesh.x.reshape(-1, order="F")[mesh.vmapM]
    q_face_flat = x_face * (1.0 - x_face)
    q_neu = q_face_flat.reshape(
        (mesh.Nfp * mesh.Nfaces, mesh.K), order="F"
    )
    u = solve_sipg_poisson_2d(
        mesh, source, bc_types=bc, q_neu=q_neu,
    )
    err = _rel_l2(mesh, u, u_exact)
    print(f"Mixed non-homog N={N} Kx={Kx} Ky={Ky}: rel L2 err = {err:.2e}")
    assert err < 1e-3


@pytest.mark.parametrize("alpha", [1.0, 100.0])
def test_sipg_helmholtz_full_dirichlet(alpha):
    """Helmholtz (αM − Δ) u = f with u=sin(πx)sin(πy), Dirichlet BCs."""
    mesh = _build_unit_square(8, 8, 3)
    pi = jnp.pi
    u_exact = jnp.sin(pi * mesh.x) * jnp.sin(pi * mesh.y)
    source = (2 * pi ** 2 + alpha) * u_exact
    bc = _full_dirichlet_bc(mesh)
    u = solve_sipg_helmholtz_2d(
        mesh, source, alpha=alpha, nu=1.0, bc_types=bc,
    )
    err = _rel_l2(mesh, u, u_exact)
    print(f"Helmholtz α={alpha}: rel L2 err = {err:.2e}")
    assert err < 1e-3


def test_sipg_nonhomog_dirichlet():
    """u = x²-y² (harmonic): non-hom Dir all around."""
    mesh = _build_unit_square(8, 8, 3)
    x, y = mesh.x, mesh.y
    u_exact = x ** 2 - y ** 2
    source = jnp.zeros_like(x)  # harmonic
    bc = _full_dirichlet_bc(mesh)
    u_dir = u_exact  # boundary values match the exact solution
    u = solve_sipg_poisson_2d(mesh, source, bc_types=bc, u_dir=u_dir)
    err = _rel_l2(mesh, u, u_exact)
    print(f"Non-homog Dirichlet (harmonic): rel L2 err = {err:.2e}")
    assert err < 1e-6


def test_iterative_cg_matches_direct_lu():
    """Matrix-free CG should match direct LU on SIPG Poisson — both
    invert the same SPD operator, just by different paths. CG path
    matters for GPU scaling (no dense matrix in memory)."""
    mesh = _build_unit_square(8, 8, 3)
    pi = jnp.pi
    u_exact = jnp.sin(pi * mesh.x) * jnp.sin(pi * mesh.y)
    source = 2 * pi ** 2 * u_exact
    bc = _full_dirichlet_bc(mesh)

    u_lu = solve_sipg_poisson_2d(mesh, source, bc_types=bc)
    u_cg = solve_sipg_poisson_2d(
        mesh, source, bc_types=bc,
        iterative=IterativeConfig(method="cg", tol=1e-12, maxiter=2000),
    )
    diff = float(jnp.max(jnp.abs(u_cg - u_lu)))
    print(f"  CG vs LU max-diff: {diff:.2e}")
    assert diff < 1e-8


def test_iterative_cg_on_helmholtz():
    """Helmholtz operator is also SPD; CG should converge."""
    mesh = _build_unit_square(8, 8, 3)
    pi = jnp.pi
    alpha = 100.0
    u_exact = jnp.sin(pi * mesh.x) * jnp.sin(pi * mesh.y)
    source = (2 * pi ** 2 + alpha) * u_exact
    bc = _full_dirichlet_bc(mesh)
    u_lu = solve_sipg_helmholtz_2d(
        mesh, source, alpha=alpha, nu=1.0, bc_types=bc,
    )
    u_cg = solve_sipg_helmholtz_2d(
        mesh, source, alpha=alpha, nu=1.0, bc_types=bc,
        iterative=IterativeConfig(method="cg", tol=1e-12, maxiter=2000),
    )
    diff = float(jnp.max(jnp.abs(u_cg - u_lu)))
    print(f"  Helmholtz CG vs LU max-diff: {diff:.2e}")
    assert diff < 1e-8


if __name__ == "__main__":
    test_sipg_matrix_is_symmetric_for_full_dirichlet()
    test_sipg_matrix_is_symmetric_for_mixed_bc()
    test_sipg_full_dirichlet(3, 6, 6)
    test_sipg_mixed_homog_neumann(3, 6, 6)
    test_sipg_mixed_nonhomog_neumann(3, 8, 8)
    test_sipg_nonhomog_dirichlet()
    test_iterative_cg_matches_direct_lu()
    test_iterative_cg_on_helmholtz()
