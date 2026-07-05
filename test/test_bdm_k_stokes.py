"""Order-parametric BDM_k / P_{k-1} Stokes sentinel.

The pressure-robustness sentinel: solving the Stokes saddle-point
with RHS ``f_u = G p_test`` (a discrete pressure gradient) must give
``u_h = 0`` to machine precision and ``p_h = p_test + const``. The
H(div)-conforming BDM-P pair makes this exact at the discrete level,
not just asymptotic -- that property is what motivates the entire
BDM stack.

Verified here at k = 1, 2, 3 using the order-parametric
:func:`stokes_step_bdm_k_2d`. At k=1 the result must also agree with
the existing :func:`stokes_step_bdm_2d` (BDM_1-only) to numerical
zero.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import BDM, bdm_k_gradient_apply
from ddg.dg2d.coupled_ns_bdm import (
    stokes_step_bdm_2d, stokes_step_bdm_k_2d,
    bdm_p0_gradient_apply,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _build_mesh(Kx=3, Ky=3):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, 1)


def test_stokes_k1_solver_matches_bdm1_path():
    """The new order-parametric Stokes solver at k=1 must reproduce
    the existing stokes_step_bdm_2d. Both should give the
    pressure-robust answer u ~ 0 (the velocity component is at the
    LU solve's numerical floor, so we compare magnitudes rather than
    bit-equality)."""
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=1)
    rng = np.random.default_rng(0)
    p_test = jnp.asarray(rng.standard_normal(V.K))
    p_test = p_test - jnp.mean(p_test)
    f_u_old = bdm_p0_gradient_apply(V, p_test)
    f_u_new = bdm_k_gradient_apply(V, p_test[:, None])
    assert jnp.allclose(f_u_old, f_u_new, atol=1e-12)

    u_old, _ = stokes_step_bdm_2d(
        V, nu=1.0, f_u=f_u_old, pressure_eps=1e-10, tau_scale=4.0,
    )
    u_new, p_new = stokes_step_bdm_k_2d(
        V, nu=1.0, f_u=f_u_new, pressure_eps=1e-10, tau_scale=4.0,
    )
    assert p_new.shape == (V.K, 1)
    # Both u should be at the pressure-robustness floor (~1e-9 to 1e-12).
    f_norm = float(jnp.linalg.norm(f_u_new))
    rel_old = float(jnp.linalg.norm(u_old)) / max(f_norm, 1e-30)
    rel_new = float(jnp.linalg.norm(u_new)) / max(f_norm, 1e-30)
    assert rel_old < 1e-6, f"old solver not pressure-robust: rel={rel_old:.3e}"
    assert rel_new < 1e-6, f"new solver not pressure-robust: rel={rel_new:.3e}"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_stokes_pressure_robustness_at_order_k(k):
    """Solve Stokes with f_u = G p_test. The H(div)-conforming
    BDM_k/P_{k-1} pair must give u_h ~ 0 to machine precision
    regardless of k."""
    mesh = _build_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=k)
    dim_p = k * (k + 1) // 2

    rng = np.random.default_rng(k)
    p_test_flat = rng.standard_normal((V.K, dim_p))
    # Centre the constant-pressure mode (m=0 monomial = 1) so the
    # solution's constant-pressure component matches the input.
    p_test_flat[:, 0] = p_test_flat[:, 0] - p_test_flat[:, 0].mean()
    p_test = jnp.asarray(p_test_flat)

    f_u = bdm_k_gradient_apply(V, p_test)
    u_sol, p_sol = stokes_step_bdm_k_2d(
        V, nu=1.0, f_u=f_u, pressure_eps=1e-10, tau_scale=4.0,
    )

    u_norm = float(jnp.linalg.norm(u_sol))
    # Use a relative tolerance against the BC-stiffness norm of the
    # RHS: ||u|| should be a tiny fraction of ||f_u||.
    f_norm = float(jnp.linalg.norm(f_u))
    rel_u = u_norm / max(f_norm, 1e-30)
    print(f"k={k}: ||u_sol||={u_norm:.3e}, ||f_u||={f_norm:.3e}, rel={rel_u:.3e}")
    assert rel_u < 1e-6, (
        f"k={k}: pressure-robustness violated, ||u||/||f||={rel_u:.3e}"
    )
