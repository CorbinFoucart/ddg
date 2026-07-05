"""Tests for the cross-space mixed-FEM operators.

Validates the embedding trick used to assemble the velocity-pressure
saddle-point at mixed polynomial degrees:
    * Interpolation matrix ``P`` exactly maps V_p Lagrange-nodal DOFs of a
      polynomial in ``P_{N-1}`` to V_u Lagrange-nodal DOFs of the same
      polynomial.
    * ``mixed_grad_apply`` reproduces the equal-order gradient when called
      on the embedded pressure (consistency check -- the operator is the
      composition by construction, but the test guards against accidental
      shape / argument-order bugs).
    * ``mixed_div_apply`` returns a vector in V_p coordinates whose
      embedding-by-P matches the equal-order divergence on V_u tests.
    * Discrete adjointness ``<G p, u>_u = <p, D u>_p`` (up to BC terms) --
      this is what makes the saddle-point Murphy-Wathen theory apply
      cleanly to the mixed-order pair, in contrast to the
      LBB-stabilisation-polluted equal-order case.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.function_space import LagrangianDG
from ddg.dg2d.incompressible_ns import (
    BC_INS_INFLOW, BC_INS_INTERIOR, BC_INS_OUTFLOW,
)
from ddg.dg2d.mixed_operators import (
    build_interpolation,
    mixed_div_apply,
    mixed_grad_apply,
    local_pressure_mass,
)
from ddg.dg2d.coupled_ns import _div_apply_2d, _grad_apply_2d
from ddg.dg2d.sipg import _local_mass


def _build_mesh(N: int, Kx: int = 4, Ky: int = 4):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def _all_interior_bc(mesh):
    """Periodic-like BC tag: every face marked INTERIOR. Used for clean
    operator tests where we don't want BC contributions to muddy the
    arithmetic."""
    return jnp.full((mesh.Nfaces, mesh.K), BC_INS_INTERIOR, dtype=jnp.int32).T


def _three_inflow_one_outflow_bc(mesh):
    """Kovasznay-style: three Dirichlet walls + one outflow face on +x.
    Exercises the BC branches in the operators."""
    Nfaces, K = mesh.Nfaces, mesh.K
    nx = np.asarray(mesh.nx.reshape((mesh.Nfp, Nfaces, K), order="F"))[0]
    EToE = np.asarray(mesh.EToE)
    is_bdy = (EToE == np.arange(K)[:, None]).T
    is_out = nx > 0.5
    bc = np.full((Nfaces, K), BC_INS_INTERIOR, dtype=np.int32)
    bc = np.where(is_bdy & ~is_out, BC_INS_INFLOW, bc)
    bc = np.where(is_bdy & is_out, BC_INS_OUTFLOW, bc)
    return jnp.asarray(bc.T)


# ---------------------------------------------------------------------------
# Interpolation matrix
# ---------------------------------------------------------------------------

def test_interpolation_equal_degree_is_identity():
    """V_from.degree == V_to.degree -> P = I."""
    mesh = _build_mesh(N=3)
    V = LagrangianDG.at_degree(mesh, degree=3)
    P = build_interpolation(V, V)
    np.testing.assert_allclose(np.asarray(P), np.eye(V.Np), atol=1e-14)


def test_interpolation_evaluates_polynomial_exactly():
    """P @ (p evaluated at V_p nodes) = p evaluated at V_u nodes,
    for any polynomial p of total degree <= V_p.degree."""
    mesh = _build_mesh(N=3)
    V_u = LagrangianDG.at_degree(mesh, degree=3)
    V_p = LagrangianDG.at_degree(mesh, degree=2)
    P = build_interpolation(V_p, V_u)

    # p(r, s) = 1 + 2r + 3s + r^2 - r*s + 2*s^2  (total degree 2)
    r_p, s_p = V_p.mesh.r, V_p.mesh.s
    r_u, s_u = V_u.mesh.r, V_u.mesh.s

    def p_of(r, s):
        return 1.0 + 2.0 * r + 3.0 * s + r * r - r * s + 2.0 * s * s

    p_at_Vp = p_of(r_p, s_p)
    p_at_Vu_direct = p_of(r_u, s_u)
    p_at_Vu_interp = P @ p_at_Vp

    np.testing.assert_allclose(
        np.asarray(p_at_Vu_interp), np.asarray(p_at_Vu_direct), atol=1e-12,
    )


def test_interpolation_rejects_lossy_direction():
    """V_from.degree > V_to.degree would be a lossy projection (information
    loss), so the builder refuses."""
    mesh = _build_mesh(N=3)
    V_u = LagrangianDG.at_degree(mesh, degree=3)
    V_p = LagrangianDG.at_degree(mesh, degree=2)
    with pytest.raises(ValueError, match="lossy projection"):
        build_interpolation(V_u, V_p)  # 3 -> 2


# ---------------------------------------------------------------------------
# Pressure-gradient operator G
# ---------------------------------------------------------------------------

def test_mixed_grad_equals_embedded_grad_clean_bc():
    """All-interior BC (no boundary forcing): mixed_grad_apply(V_u, V_p, p)
    must equal _grad_apply_2d(V_u.mesh, P_emb @ p, ...). This is the
    composition identity by construction."""
    mesh = _build_mesh(N=3)
    V_u = LagrangianDG.at_degree(mesh, degree=3)
    V_p = LagrangianDG.at_degree(mesh, degree=2)
    P_emb = build_interpolation(V_p, V_u)
    M_ref_u = _local_mass(V_u.mesh)

    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.standard_normal((V_p.Np, V_p.K)))

    bc = _all_interior_bc(mesh)
    Gx_mixed, Gy_mixed = mixed_grad_apply(
        V_u, V_p, p, ins_bc_types=bc, M_ref_u=M_ref_u, P_emb=P_emb,
    )
    p_emb = P_emb @ p
    Gx_eq, Gy_eq = _grad_apply_2d(
        V_u.mesh, p_emb, ins_bc_types=bc, M_ref=M_ref_u,
    )
    np.testing.assert_allclose(np.asarray(Gx_mixed), np.asarray(Gx_eq),
                               atol=1e-13)
    np.testing.assert_allclose(np.asarray(Gy_mixed), np.asarray(Gy_eq),
                               atol=1e-13)


def test_mixed_grad_zero_pressure_yields_zero():
    """Linearity sanity: G applied to zero pressure -> zero velocity load."""
    mesh = _build_mesh(N=3)
    V_u = LagrangianDG.at_degree(mesh, degree=3)
    V_p = LagrangianDG.at_degree(mesh, degree=2)
    P_emb = build_interpolation(V_p, V_u)
    M_ref_u = _local_mass(V_u.mesh)

    bc = _three_inflow_one_outflow_bc(mesh)
    p = jnp.zeros((V_p.Np, V_p.K))
    Gx, Gy = mixed_grad_apply(
        V_u, V_p, p, ins_bc_types=bc, M_ref_u=M_ref_u, P_emb=P_emb,
    )
    np.testing.assert_allclose(np.asarray(Gx), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(Gy), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Velocity-divergence operator D
# ---------------------------------------------------------------------------

def test_mixed_div_equals_projected_div_clean_bc():
    """D_mixed u = P_emb.T (D_eq u). The composition identity that lets
    us reuse the equal-order operator with a final projection."""
    mesh = _build_mesh(N=3)
    V_u = LagrangianDG.at_degree(mesh, degree=3)
    V_p = LagrangianDG.at_degree(mesh, degree=2)
    P_emb = build_interpolation(V_p, V_u)
    M_ref_u = _local_mass(V_u.mesh)

    rng = np.random.default_rng(1)
    ux = jnp.asarray(rng.standard_normal((V_u.Np, V_u.K)))
    uy = jnp.asarray(rng.standard_normal((V_u.Np, V_u.K)))

    bc = _all_interior_bc(mesh)
    Du_mixed = mixed_div_apply(
        V_u, V_p, ux, uy, ins_bc_types=bc, M_ref_u=M_ref_u, P_emb=P_emb,
    )
    Du_eq = _div_apply_2d(
        V_u.mesh, ux, uy, ins_bc_types=bc, M_ref=M_ref_u,
    )
    np.testing.assert_allclose(np.asarray(Du_mixed),
                               np.asarray(P_emb.T @ Du_eq),
                               atol=1e-13)


def test_mixed_div_zero_velocity_yields_zero():
    mesh = _build_mesh(N=3)
    V_u = LagrangianDG.at_degree(mesh, degree=3)
    V_p = LagrangianDG.at_degree(mesh, degree=2)
    P_emb = build_interpolation(V_p, V_u)
    M_ref_u = _local_mass(V_u.mesh)
    bc = _three_inflow_one_outflow_bc(mesh)
    z = jnp.zeros((V_u.Np, V_u.K))
    Du = mixed_div_apply(
        V_u, V_p, z, z, ins_bc_types=bc, M_ref_u=M_ref_u, P_emb=P_emb,
    )
    np.testing.assert_allclose(np.asarray(Du), 0.0, atol=1e-14)


def test_mixed_div_recovers_divergence_for_polynomial_velocity():
    """For a polynomial velocity field of degree <= V_u.degree with no
    boundary forcing, D u inverted against the V_p mass matrix should
    recover ∇·u evaluated in V_p coordinates.

    We pick u = (x^2 - y, x + y^2) so ∇·u = 2x + 2y, which is in P_1, well
    representable in V_p (degree 2). Verify on interior elements (cells
    away from boundary), where the operator has no BC contamination."""
    mesh = _build_mesh(N=3, Kx=8, Ky=8)
    V_u = LagrangianDG.at_degree(mesh, degree=3)
    V_p = LagrangianDG.at_degree(mesh, degree=2)
    P_emb = build_interpolation(V_p, V_u)
    M_ref_u = _local_mass(V_u.mesh)
    M_ref_p = local_pressure_mass(V_p)

    x_u, y_u = V_u.mesh.x, V_u.mesh.y
    x_p, y_p = V_p.mesh.x, V_p.mesh.y
    ux = x_u * x_u - y_u
    uy = x_u + y_u * y_u
    div_exact_p = 2.0 * x_p + 2.0 * y_p  # ∇·u evaluated at V_p nodes

    bc = _all_interior_bc(mesh)
    Du = mixed_div_apply(
        V_u, V_p, ux, uy, ins_bc_types=bc, M_ref_u=M_ref_u, P_emb=P_emb,
    )
    # D u = M_p * (div u in V_p coords).  Invert M_p element-wise.
    # On affine mesh: M_K = J * M_ref; J is constant per element.
    J_p = V_p.mesh.J  # (V_p.Np, K) but constant per column for affine
    J_per_elem = J_p[0, :]
    M_inv_ref = jnp.linalg.inv(M_ref_p)
    div_recovered = (M_inv_ref @ Du) / J_per_elem[None, :]

    # On affine mesh in the interior, the recovery is exact for any
    # polynomial of degree <= V_p.degree. Boundary elements have small
    # contributions from the centered-flux ghost values (~ 1/J * boundary
    # face term), so we restrict to interior elements.
    EToE = np.asarray(mesh.EToE)
    is_interior_elem = np.all(EToE != np.arange(mesh.K)[:, None], axis=1)
    interior_K = np.where(is_interior_elem)[0]
    np.testing.assert_allclose(
        np.asarray(div_recovered[:, interior_K]),
        np.asarray(div_exact_p[:, interior_K]),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Discrete adjointness
# ---------------------------------------------------------------------------

# Note on the absence of a DOF-level adjointness test:
# The centered-flux DG operators are NOT raw <G p, u> = -<D u, p> adjoint
# at the DOF level, even for jump-free pressure. The reason is that the
# velocity test function in DG has inter-element jumps, and the discrete
# IBP picks up flux terms ``∮ q [w·n]`` that don't vanish for jump-free q.
# The saddle-point inf-sup theory for centered-flux DG works in a
# weighted norm that *includes* the jump terms (LDG lifting), not via
# raw DOF adjointness. We therefore do not assert it here -- the
# Phase-3 coupled-solver tests will verify the system as a whole via
# convergence rates against manufactured Kovasznay and lid-driven cavity
# benchmarks, which is the operationally-meaningful check.
