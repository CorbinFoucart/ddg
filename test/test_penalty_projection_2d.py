"""Tests for the divergence + continuity penalty post-projection.

Tests follow Fehn et al. JCP 2018 (arXiv:1801.08103). The penalty
operators add consistent stabilisation to the velocity field via a
separate solve

    (M + τ_d D + τ_c C) U_new = M U_int

where D is the divergence-penalty bilinear form
``∫_K (∇·U)(∇·V) dx`` and C is the (normal-only) continuity-penalty
``∫_∂_int [U·n][V·n] dS``.

We verify the limiting properties (zero penalty preserves input;
div-/jump-free fields stay unchanged; penalty smoothing reduces L2
norms of div and jumps) without re-doing the under-resolved
turbulence validation from the paper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.incompressible_ns import (
    divergence_penalty_apply_2d,
    continuity_penalty_apply_2d,
    factor_penalty_projection_2d,
    penalty_projection_step_2d,
    _div_velocity_2d,
)


def _mesh(N: int = 3, Kx: int = 4, Ky: int = 4):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, N)


def test_zero_penalty_is_identity():
    """``τ_d = τ_c = 0`` ⇒ ``(M + 0)U = MU`` ⇒ projection is identity."""
    mesh = _mesh()
    ux = mesh.x * mesh.y
    uy = mesh.x ** 2 - mesh.y ** 2
    factored = factor_penalty_projection_2d(mesh, tau_div=0.0, tau_cont=0.0)
    ux_new, uy_new = penalty_projection_step_2d(mesh, ux, uy, factored=factored)
    assert float(jnp.max(jnp.abs(ux_new - ux))) < 1e-10
    assert float(jnp.max(jnp.abs(uy_new - uy))) < 1e-10


def test_constant_field_invariant_under_continuity_penalty():
    """Globally constant velocity has zero inter-element jumps, so any
    pure-continuity penalty must leave it unchanged."""
    mesh = _mesh()
    ux_const = jnp.ones_like(mesh.x) * 0.7
    uy_const = jnp.ones_like(mesh.x) * -0.3
    factored = factor_penalty_projection_2d(mesh, tau_div=0.0, tau_cont=10.0)
    ux_new, uy_new = penalty_projection_step_2d(
        mesh, ux_const, uy_const, factored=factored,
    )
    assert float(jnp.max(jnp.abs(ux_new - ux_const))) < 1e-10
    assert float(jnp.max(jnp.abs(uy_new - uy_const))) < 1e-10


def test_continuity_penalty_reduces_jumps():
    """Discontinuous initial field (per-element-constant, alternating
    sign) should get smoothed by the continuity penalty."""
    mesh = _mesh(N=2, Kx=8, Ky=8)
    # Assign a different constant per element by element index — this
    # creates real inter-element jumps the penalty can act on.
    rng = np.random.default_rng(0)
    per_elem = jnp.asarray(rng.standard_normal(mesh.K))
    ux_disc = jnp.broadcast_to(per_elem[None, :], (mesh.Np, mesh.K))
    uy_disc = jnp.zeros_like(ux_disc)

    ux_flat = ux_disc.T.reshape(-1)
    jump_initial = float(
        jnp.sqrt(jnp.sum((ux_flat[mesh.vmapM] - ux_flat[mesh.vmapP]) ** 2))
    )

    factored = factor_penalty_projection_2d(mesh, tau_div=0.0, tau_cont=10.0)
    ux_new, _ = penalty_projection_step_2d(mesh, ux_disc, uy_disc, factored=factored)
    ux_new_flat = ux_new.T.reshape(-1)
    jump_after = float(
        jnp.sqrt(jnp.sum((ux_new_flat[mesh.vmapM] - ux_new_flat[mesh.vmapP]) ** 2))
    )
    print(f"  continuity smoothing: ‖jump‖₂ {jump_initial:.3f} → "
          f"{jump_after:.3f}")
    assert jump_after < jump_initial * 0.7


def test_divergence_penalty_reduces_divergence():
    """Field with non-trivial divergence gets ∇·U pushed toward 0."""
    mesh = _mesh(N=3, Kx=6, Ky=6)
    # An obviously not-divergence-free field
    ux = mesh.x ** 2
    uy = mesh.y ** 2  # ∇·U = 2x + 2y
    div0 = _div_velocity_2d(mesh, ux, uy)
    l2_div_0 = float(jnp.sqrt(jnp.sum(mesh.J * div0 ** 2)))

    factored = factor_penalty_projection_2d(mesh, tau_div=10.0, tau_cont=0.0)
    ux_new, uy_new = penalty_projection_step_2d(mesh, ux, uy, factored=factored)
    div_new = _div_velocity_2d(mesh, ux_new, uy_new)
    l2_div_new = float(jnp.sqrt(jnp.sum(mesh.J * div_new ** 2)))
    print(f"  divergence reduction: ‖∇·U‖₂  {l2_div_0:.3f} → {l2_div_new:.3f}")
    assert l2_div_new < 0.2 * l2_div_0  # at least 80% reduction


def test_penalty_operator_is_symmetric():
    """Both penalty bilinear forms are SPD; the assembled matrix
    should be symmetric (and the projection matrix M + τ_d D + τ_c C
    therefore SPD)."""
    mesh = _mesh(N=2, Kx=4, Ky=4)
    from ddg.dg2d.incompressible_ns import penalty_projection_matrix_2d
    A = penalty_projection_matrix_2d(mesh, tau_div=1.0, tau_cont=1.0)
    asym = float(jnp.max(jnp.abs(A - A.T)))
    print(f"  asymmetry = {asym:.2e}")
    assert asym < 1e-10
    # Positive-definite check via min eigenvalue
    eigs = jnp.linalg.eigvalsh((A + A.T) / 2)
    min_eig = float(jnp.min(eigs))
    print(f"  min eigenvalue = {min_eig:.3e}")
    assert min_eig > 0.0


if __name__ == "__main__":
    test_zero_penalty_is_identity()
    test_constant_field_invariant_under_continuity_penalty()
    test_continuity_penalty_reduces_jumps()
    test_divergence_penalty_reduces_divergence()
    test_penalty_operator_is_symmetric()
