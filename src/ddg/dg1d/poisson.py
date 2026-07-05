"""1D Poisson equation via interior-penalty discontinuous Galerkin (IPDG).

Solves ``-u'' = f`` on a 1D mesh with homogeneous Dirichlet boundaries
(``u = 0`` at the two endpoints). The discrete operator is the
symmetric IP-stabilised one from Hesthaven & Warburton:

    a(u, v) = sum_K (u_x, v_x)_K
            - sum_F ({u_x}, [v])_F - sum_F ([u], {v_x})_F
            + sum_F tau [u] [v]_F

with penalty ``tau = Np^2 / h_min``. Direct port of
``Codes1.1/Codes1D/PoissonIPstabRHS1D.m``, sign-flipped so the
assembled matrix is positive-definite and a textbook ``-u'' = f``
equation maps cleanly to ``A u = M f``.

The operator is assembled into a dense ``(Np*K, Np*K)`` matrix via
``jax.jacobian`` on the linear apply function. For 1D this is fine;
larger problems should swap in a Krylov solve on the matrix-free
``poisson_ip_apply``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ddg.dg1d.mesh import Mesh1D


def _flatten_f(a: jax.Array) -> jax.Array:
    """Column-major flatten of (Np, K) -> (Np*K,)."""
    return a.T.reshape(-1)


def _unflatten_f(a_flat: jax.Array, Np: int, K: int) -> jax.Array:
    """Inverse of _flatten_f: (Np*K,) -> (Np, K)."""
    return a_flat.reshape(K, Np).T


def _local_mass(mesh: Mesh1D) -> jax.Array:
    """Reference-element mass matrix ``M_ref = inv(V V^T)``, shape (Np, Np)."""
    return jnp.linalg.inv(mesh.V @ mesh.V.T)


def _default_tau(mesh: Mesh1D):
    """IPDG penalty ``Np^2 / h_min`` (matches Hesthaven's choice).

    Returns a JAX scalar (not a Python float) so this composes with
    autodiff through ``mesh.rx``.
    """
    return mesh.Np**2 / (2.0 / jnp.max(mesh.rx))


def poisson_ip_apply(
    mesh: Mesh1D,
    u: jax.Array,
    *,
    tau: float | None = None,
) -> jax.Array:
    """Apply ``M * (-Δ_h) u`` (positive-definite IPDG stiffness operator).

    The result, set equal to ``M f``, yields the textbook discrete
    equation for ``-u'' = f`` with homogeneous Dirichlet BCs.
    """
    if tau is None:
        tau = _default_tau(mesh)

    Nfp_Nf, K = mesh.Nfp * mesh.Nfaces, mesh.K
    u_flat = _flatten_f(u)

    # 1) Jump of u at faces. Dirichlet BC: mirror gives du = 2 u_- at boundary.
    du = u_flat[mesh.vmapM] - u_flat[mesh.vmapP]
    if mesh.mapB.size > 0:
        du = du.at[mesh.mapB].set(2.0 * u_flat[mesh.vmapB])

    nx_flat = mesh.nx.reshape(-1, order="F")
    flux_u = (nx_flat * du / 2.0).reshape((Nfp_Nf, K), order="F")

    # 2) Corrected gradient q = u_x - lift(flux_u).
    ux = mesh.rx * (mesh.Dr @ u)
    q = ux - mesh.LIFT @ (mesh.Fscale * flux_u)

    # 3) Jump of q against the *average* of strong-form gradients.
    q_flat = _flatten_f(q)
    ux_flat = _flatten_f(ux)
    ux_avg = 0.5 * (ux_flat[mesh.vmapM] + ux_flat[mesh.vmapP])
    dq = q_flat[mesh.vmapM] - ux_avg
    # Neumann-style BC: qin = qout = ux at boundary => dq = q - ux there.
    if mesh.mapB.size > 0:
        dq = dq.at[mesh.mapB].set(q_flat[mesh.vmapB] - ux_flat[mesh.vmapB])

    flux_q = (nx_flat * (dq + tau * nx_flat * du)).reshape((Nfp_Nf, K), order="F")

    # 4) Mass-weighted divergence of q with penalty contribution.
    M_ref = _local_mass(mesh)
    lap = mesh.rx * (mesh.Dr @ q) - mesh.LIFT @ (mesh.Fscale * flux_q)
    return -mesh.J * (M_ref @ lap)  # negate so operator is positive-definite


def poisson_ip_matrix(mesh: Mesh1D, *, tau: float | None = None) -> jax.Array:
    """Dense IPDG stiffness matrix of shape ``(Np*K, Np*K)``.

    Built by autodiff (``jax.jacobian``) on the linear apply function:
    cheap to maintain, easy to reuse for many right-hand sides.
    """
    n_dof = mesh.Np * mesh.K

    def op_flat(u_flat: jax.Array) -> jax.Array:
        u = _unflatten_f(u_flat, mesh.Np, mesh.K)
        return _flatten_f(poisson_ip_apply(mesh, u, tau=tau))

    return jax.jacobian(op_flat)(jnp.zeros(n_dof))


def solve_poisson(
    mesh: Mesh1D,
    source: jax.Array,
    *,
    tau: float | None = None,
) -> jax.Array:
    """Solve ``-u'' = source`` with homogeneous Dirichlet boundaries.

    ``source`` has shape ``(Np, K)``; the returned solution has the
    same shape. The mass-matrix-weighted load vector ``M f`` is
    assembled internally.
    """
    A = poisson_ip_matrix(mesh, tau=tau)
    M_ref = _local_mass(mesh)
    Mf = mesh.J * (M_ref @ source)
    u_flat = jnp.linalg.solve(A, _flatten_f(Mf))
    return _unflatten_f(u_flat, mesh.Np, mesh.K)
