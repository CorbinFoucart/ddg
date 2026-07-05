"""2D Poisson equation via interior-penalty discontinuous Galerkin (IPDG).

Direct extension of the 1D port: solves ``-Δu = f`` on a 2D
triangular mesh with homogeneous Dirichlet boundaries. Same recipe —
LDG-style auxiliary gradient ``q``, jump of u against the boundary
average of the strong-form gradient, symmetric interior-penalty
stabilisation. Assembled into a dense ``(Np*K, Np*K)`` matrix via
``jax.jacobian`` on the linear apply function.

This is the workhorse for the stream-function step of the 2D
incompressible Navier-Stokes solver: each time step solves
``-Δψ = ω`` for the stream function given the current vorticity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ddg.dg2d.mesh import Mesh2D


def _flatten_f(a: jax.Array) -> jax.Array:
    """Column-major flatten of ``(Np, K) -> (Np*K,)``."""
    return a.T.reshape(-1)


def _unflatten_f(a_flat: jax.Array, Np: int, K: int) -> jax.Array:
    """Inverse of _flatten_f: ``(Np*K,) -> (Np, K)``."""
    return a_flat.reshape(K, Np).T


def _local_mass(mesh: Mesh2D) -> jax.Array:
    """Reference-element mass matrix ``M_ref = inv(V V^T)`` (shape Np x Np)."""
    return jnp.linalg.inv(mesh.V @ mesh.V.T)


def _default_tau_2d(mesh: Mesh2D) -> jax.Array:
    """IPDG penalty scaled by ``Np² * (sJ / J)`` per face — Hesthaven's
    recipe for the curved IPDG operator, restricted to scalar here."""
    return mesh.Np * mesh.Np * jnp.max(mesh.Fscale)


def poisson_ip_apply_2d(
    mesh: Mesh2D,
    u: jax.Array,
    *,
    tau: float | None = None,
    u_boundary: jax.Array | None = None,
) -> jax.Array:
    """Apply ``M · (-Δ_h) u`` (positive-definite IPDG stiffness operator).

    Equation: ``-Δu = f`` with Dirichlet BCs ``u = g`` on ``∂Ω``.
    Setting the result equal to ``M f`` yields the discrete equation
    ``(M·(-Δ_h)) u = M f``.

    If ``u_boundary`` is supplied, the boundary jump uses
    ``2·(u_M - g_M)`` so the operator enforces ``u = g`` weakly (mirror
    ghost ``2 g - u_M``). With the default ``u_boundary=None`` we get
    homogeneous Dirichlet ``u = 0``. The Jacobian of this operator
    with respect to ``u`` is the same in both cases (g enters only as
    an affine offset), so the assembled matrix from
    ``poisson_ip_matrix_2d`` is reusable; only the boundary load
    differs.

    Note on mixed Dirichlet/Neumann BCs: this LDG-form operator does
    *not* support mixed BCs cleanly — the per-element LIFT correction
    from Dirichlet faces spills into the q value at Neumann face nodes
    of the same element, breaking the symmetry of the bilinear form.
    For mixed BCs use the SIPG-form operator in :mod:`ddg.dg2d.sipg`
    instead.
    """
    if tau is None:
        tau = _default_tau_2d(mesh)

    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K
    u_flat = _flatten_f(u)

    # 1) Jump of u. Dirichlet BC mirror ghost = 2 g - u_M, so jump = 2 (u_M - g).
    du = u_flat[mesh.vmapM] - u_flat[mesh.vmapP]
    if mesh.mapB.size > 0:
        if u_boundary is None:
            du = du.at[mesh.mapB].set(2.0 * u_flat[mesh.vmapB])
        else:
            g_flat = _flatten_f(u_boundary)
            du = du.at[mesh.mapB].set(
                2.0 * (u_flat[mesh.vmapB] - g_flat[mesh.vmapB])
            )

    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    flux_ux_face = (nx_flat * du / 2.0).reshape((Nfp_Nf, K), order="F")
    flux_uy_face = (ny_flat * du / 2.0).reshape((Nfp_Nf, K), order="F")

    # 2) Strong-form gradient of u.
    ur = mesh.Dr @ u
    us = mesh.Ds @ u
    ux = mesh.rx * ur + mesh.sx * us
    uy = mesh.ry * ur + mesh.sy * us

    # 3) Corrected gradient: q = ∇u - lift(flux_u·n).
    qx = ux - mesh.LIFT @ (mesh.Fscale * flux_ux_face)
    qy = uy - mesh.LIFT @ (mesh.Fscale * flux_uy_face)

    # 4) Jump of q·n against the average of strong-form gradient·n.
    qx_flat = _flatten_f(qx)
    qy_flat = _flatten_f(qy)
    ux_flat = _flatten_f(ux)
    uy_flat = _flatten_f(uy)

    qn_M = qx_flat[mesh.vmapM] * nx_flat + qy_flat[mesh.vmapM] * ny_flat
    grad_avg_n = (
        0.5 * (ux_flat[mesh.vmapM] + ux_flat[mesh.vmapP]) * nx_flat
        + 0.5 * (uy_flat[mesh.vmapM] + uy_flat[mesh.vmapP]) * ny_flat
    )
    dq = qn_M - grad_avg_n
    if mesh.mapB.size > 0:
        nx_bdy = nx_flat[mesh.mapB]
        ny_bdy = ny_flat[mesh.mapB]
        qn_M_bdy = (
            qx_flat[mesh.vmapB] * nx_bdy + qy_flat[mesh.vmapB] * ny_bdy
        )
        grad_M_n_bdy = (
            ux_flat[mesh.vmapB] * nx_bdy + uy_flat[mesh.vmapB] * ny_bdy
        )
        dq = dq.at[mesh.mapB].set(qn_M_bdy - grad_M_n_bdy)

    flux_q_face = (dq + tau * du).reshape((Nfp_Nf, K), order="F")

    # 5) Divergence of q minus the surface lift contribution.
    qx_r = mesh.Dr @ qx
    qx_s = mesh.Ds @ qx
    qy_r = mesh.Dr @ qy
    qy_s = mesh.Ds @ qy
    div_q = (mesh.rx * qx_r + mesh.sx * qx_s
             + mesh.ry * qy_r + mesh.sy * qy_s)

    lap = div_q - mesh.LIFT @ (mesh.Fscale * flux_q_face)
    M_ref = _local_mass(mesh)
    return -mesh.J * (M_ref @ lap)


def poisson_ip_matrix_2d(
    mesh: Mesh2D, *, tau: float | None = None
) -> jax.Array:
    """Dense IPDG stiffness matrix of shape ``(Np*K, Np*K)``."""
    n_dof = mesh.Np * mesh.K

    def op_flat(u_flat: jax.Array) -> jax.Array:
        u = _unflatten_f(u_flat, mesh.Np, mesh.K)
        return _flatten_f(poisson_ip_apply_2d(mesh, u, tau=tau))

    return jax.jacobian(op_flat)(jnp.zeros(n_dof))


def factor_poisson_2d(
    mesh: Mesh2D, *, tau: float | None = None
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Assemble + LU-factor the IPDG Poisson matrix.

    Returns a tuple ``(A, lu, piv)`` suitable for passing to
    ``solve_poisson_2d`` via ``A_factored``. With this, every subsequent
    Poisson solve is a single triangular solve (``jax.scipy.linalg.lu_solve``)
    rather than a fresh dense LU each time — orders of magnitude faster
    when ψ is solved repeatedly inside a time-stepping loop.
    """
    A = poisson_ip_matrix_2d(mesh, tau=tau)
    lu, piv = jax.scipy.linalg.lu_factor(A)
    return A, lu, piv


def solve_poisson_2d(
    mesh: Mesh2D,
    source: jax.Array,
    *,
    tau: float | None = None,
    A_factored: tuple | None = None,
    u_boundary: jax.Array | None = None,
) -> jax.Array:
    """Solve ``-Δu = source`` with Dirichlet BCs.

    Defaults to homogeneous Dirichlet ``u = 0``. To impose
    non-homogeneous BCs ``u = g`` on the boundary, pass an
    ``(Np, K)``-shaped array ``u_boundary`` evaluated at the mesh
    nodes (treated as the nodal interpolant of ``g`` and used as a
    lifting: ``u = u_boundary + δ`` where ``δ`` solves the
    homogeneous problem with corrected source). Only the boundary
    values of ``u_boundary`` matter for satisfying the BC; interior
    values of the lifting are absorbed into ``δ``.

    Pass ``A_factored`` to skip assembly and LU-factor cost on
    repeated solves:
      * ``A_factored = (A,)`` or ``(A, ...)`` triggers a per-call
        ``jnp.linalg.solve(A, ...)``.
      * ``A_factored = (A, lu, piv)`` from ``factor_poisson_2d`` uses
        cached ``lu_solve`` — the right shape for time-stepping loops
        where ψ is solved many times against the same operator.

    For mixed Dirichlet / Neumann BCs use
    :func:`ddg.dg2d.sipg.solve_sipg_poisson_2d` instead.
    """
    if A_factored is None:
        A = poisson_ip_matrix_2d(mesh, tau=tau)
        lu_piv = None
    else:
        A = A_factored[0]
        lu_piv = (A_factored[1], A_factored[2]) if len(A_factored) >= 3 else None
    M_ref = _local_mass(mesh)
    Mf = mesh.J * (M_ref @ source)

    if u_boundary is None:
        rhs_flat = _flatten_f(Mf)
    else:
        # Non-homogeneous Dirichlet. apply(u, g) = A·u + L(g), so we
        # solve A·u = M f - L(g) using only the assembled / factored A.
        b_g = poisson_ip_apply_2d(
            mesh, jnp.zeros_like(u_boundary), tau=tau, u_boundary=u_boundary,
        )
        rhs_flat = _flatten_f(Mf - b_g)

    if lu_piv is not None:
        u_flat = jax.scipy.linalg.lu_solve(lu_piv, rhs_flat)
    else:
        u_flat = jnp.linalg.solve(A, rhs_flat)
    return _unflatten_f(u_flat, mesh.Np, mesh.K)
