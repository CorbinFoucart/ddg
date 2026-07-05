"""Matrix-free CG penalty projection: equivalence + grad checks.

The dense LU path (``factor_penalty_projection_2d`` +
``penalty_projection_step_2d``) and the matrix-free CG path
(``penalty_projection_step_cg_2d``) solve the same SPD system

    (M + τ_d D + τ_c C) U_new = M U

and must produce identical results up to CG tolerance. We also check
that ``jax.grad`` of a scalar loss through the CG step agrees with
the gradient through the dense LU step (the differentiability
invariant: swapping the solver should not change the gradient).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.incompressible_ns import (
    PenaltyConfig,
    make_penalty_config_dimensional,
    factor_penalty_projection_2d,
    penalty_projection_step_2d,
    penalty_projection_step_cg_2d,
)


def _mesh(N: int = 3, Kx: int = 4, Ky: int = 4):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def test_cg_matches_lu_low_penalty():
    """Small τ: both paths give the same answer."""
    mesh = _mesh(N=3, Kx=4, Ky=4)
    ux = mesh.x * mesh.y
    uy = mesh.x ** 2 - mesh.y ** 2

    factored = factor_penalty_projection_2d(mesh, tau_div=0.1, tau_cont=0.05)
    ux_lu, uy_lu = penalty_projection_step_2d(mesh, ux, uy, factored=factored)
    ux_cg, uy_cg = penalty_projection_step_cg_2d(
        mesh, ux, uy, tau_div=0.1, tau_cont=0.05, tol=1e-12,
    )

    err_x = float(jnp.max(jnp.abs(ux_lu - ux_cg)))
    err_y = float(jnp.max(jnp.abs(uy_lu - uy_cg)))
    print(f"  ||ux_lu - ux_cg||_∞ = {err_x:.2e}")
    print(f"  ||uy_lu - uy_cg||_∞ = {err_y:.2e}")
    assert err_x < 1e-9
    assert err_y < 1e-9


def test_cg_matches_lu_strong_penalty():
    """Large τ: tighter system, still matches."""
    mesh = _mesh(N=2, Kx=6, Ky=6)
    rng = np.random.default_rng(0)
    ux = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))

    factored = factor_penalty_projection_2d(mesh, tau_div=10.0, tau_cont=10.0)
    ux_lu, uy_lu = penalty_projection_step_2d(mesh, ux, uy, factored=factored)
    ux_cg, uy_cg = penalty_projection_step_cg_2d(
        mesh, ux, uy, tau_div=10.0, tau_cont=10.0, tol=1e-12,
    )

    err_x = float(jnp.max(jnp.abs(ux_lu - ux_cg)))
    err_y = float(jnp.max(jnp.abs(uy_lu - uy_cg)))
    print(f"  τ=10: ||ux_lu - ux_cg||_∞ = {err_x:.2e}")
    # Larger τ means larger condition number; CG hits the tol=1e-12
    # residual floor but the solution agreement is limited by κ. 1e-7
    # is still 5 orders better than the audit-relevant tolerance.
    assert err_x < 1e-7
    assert err_y < 1e-7


def test_zero_penalty_is_identity():
    """``τ_d = τ_c = 0`` ⇒ ``M U = M U`` ⇒ projection is identity."""
    mesh = _mesh()
    ux = mesh.x * mesh.y
    uy = mesh.x ** 2 - mesh.y ** 2
    ux_new, uy_new = penalty_projection_step_cg_2d(
        mesh, ux, uy, tau_div=0.0, tau_cont=0.0, tol=1e-12,
    )
    assert float(jnp.max(jnp.abs(ux_new - ux))) < 1e-9
    assert float(jnp.max(jnp.abs(uy_new - uy))) < 1e-9


def test_grad_through_cg_matches_grad_through_lu():
    """The differentiability invariant: swapping the linear solver
    should not change the gradient of a loss with respect to the input.
    """
    mesh = _mesh(N=2, Kx=4, Ky=4)

    def loss_lu(ux_in, uy_in):
        factored = factor_penalty_projection_2d(
            mesh, tau_div=1.0, tau_cont=1.0,
        )
        ux_new, uy_new = penalty_projection_step_2d(
            mesh, ux_in, uy_in, factored=factored,
        )
        return jnp.sum(ux_new ** 2 + uy_new ** 2)

    def loss_cg(ux_in, uy_in):
        ux_new, uy_new = penalty_projection_step_cg_2d(
            mesh, ux_in, uy_in, tau_div=1.0, tau_cont=1.0, tol=1e-12,
        )
        return jnp.sum(ux_new ** 2 + uy_new ** 2)

    rng = np.random.default_rng(1)
    ux0 = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy0 = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))

    g_lu = jax.grad(loss_lu, argnums=(0, 1))(ux0, uy0)
    g_cg = jax.grad(loss_cg, argnums=(0, 1))(ux0, uy0)

    err_x = float(jnp.max(jnp.abs(g_lu[0] - g_cg[0])))
    err_y = float(jnp.max(jnp.abs(g_lu[1] - g_cg[1])))
    print(f"  ||grad_x_lu - grad_x_cg||_∞ = {err_x:.2e}")
    print(f"  ||grad_y_lu - grad_y_cg||_∞ = {err_y:.2e}")
    # CG tolerance is tighter than gradient comparison needs.
    assert err_x < 1e-6
    assert err_y < 1e-6


def test_penalty_config_dataclass():
    """``PenaltyConfig`` is a frozen dataclass with the expected fields."""
    cfg = PenaltyConfig(tau_div=1.0, tau_cont=2.0)
    assert cfg.tau_div == 1.0
    assert cfg.tau_cont == 2.0
    assert cfg.tol == 1e-9          # default
    assert cfg.maxiter == 1000      # default


def test_array_tau_cg_matches_scalar_tau_cg():
    """An array tau filled with a single value must equal scalar tau."""
    mesh = _mesh(N=2, Kx=4, Ky=4)
    rng = np.random.default_rng(7)
    ux = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))

    # Scalar tau
    ux_s, uy_s = penalty_projection_step_cg_2d(
        mesh, ux, uy, tau_div=0.5, tau_cont=0.3, tol=1e-12,
    )
    # Array tau, uniform value
    tau_div_arr = jnp.full((mesh.K,), 0.5)
    tau_cont_arr = jnp.full((mesh.K,), 0.3)
    ux_a, uy_a = penalty_projection_step_cg_2d(
        mesh, ux, uy, tau_div=tau_div_arr, tau_cont=tau_cont_arr, tol=1e-12,
    )
    err = float(jnp.max(jnp.abs(ux_s - ux_a)))
    print(f"  scalar vs uniform-array tau: ||Δ||_∞ = {err:.2e}")
    assert err < 1e-10


def test_array_tau_lu_matches_cg():
    """Element-local array tau through LU factor matches CG."""
    mesh = _mesh(N=2, Kx=4, Ky=4)
    rng = np.random.default_rng(8)
    ux = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))
    uy = jnp.asarray(rng.standard_normal((mesh.Np, mesh.K)))

    # Build a non-uniform per-element tau so we genuinely test array broadcast
    tau_div = jnp.asarray(0.2 + 0.5 * rng.random(mesh.K))
    tau_cont = jnp.asarray(0.1 + 0.4 * rng.random(mesh.K))

    factored = factor_penalty_projection_2d(
        mesh, tau_div=tau_div, tau_cont=tau_cont,
    )
    ux_lu, uy_lu = penalty_projection_step_2d(
        mesh, ux, uy, factored=factored,
    )
    ux_cg, uy_cg = penalty_projection_step_cg_2d(
        mesh, ux, uy, tau_div=tau_div, tau_cont=tau_cont, tol=1e-12,
    )
    err_x = float(jnp.max(jnp.abs(ux_lu - ux_cg)))
    err_y = float(jnp.max(jnp.abs(uy_lu - uy_cg)))
    print(f"  per-elem tau: ||u_lu - u_cg||_∞ = ({err_x:.2e}, {err_y:.2e})")
    assert err_x < 1e-8
    assert err_y < 1e-8


def test_make_penalty_config_dimensional():
    """The dimensional-scaling helper returns sensible array taus."""
    mesh = _mesh(N=2, Kx=4, Ky=4)
    cfg = make_penalty_config_dimensional(mesh, u_char=1.0, scale_div=1.0)
    assert cfg.tau_div.shape == (mesh.K,)
    assert cfg.tau_cont.shape == (mesh.K,)
    # All positive
    assert float(jnp.min(cfg.tau_div)) > 0.0
    # On a uniform mesh the per-element tau is constant
    assert float(jnp.std(cfg.tau_div)) < 1e-12
    # Scale knob works
    cfg2 = make_penalty_config_dimensional(mesh, u_char=1.0, scale_div=2.0)
    assert float(jnp.max(jnp.abs(cfg2.tau_div - 2.0 * cfg.tau_div))) < 1e-12


if __name__ == "__main__":
    print("test_cg_matches_lu_low_penalty"); test_cg_matches_lu_low_penalty()
    print("test_cg_matches_lu_strong_penalty"); test_cg_matches_lu_strong_penalty()
    print("test_zero_penalty_is_identity"); test_zero_penalty_is_identity()
    print("test_grad_through_cg_matches_grad_through_lu"); test_grad_through_cg_matches_grad_through_lu()
    print("test_penalty_config_dataclass"); test_penalty_config_dataclass()
    print("test_array_tau_cg_matches_scalar_tau_cg"); test_array_tau_cg_matches_scalar_tau_cg()
    print("test_array_tau_lu_matches_cg"); test_array_tau_lu_matches_cg()
    print("test_make_penalty_config_dimensional"); test_make_penalty_config_dimensional()
    print("all ok")
