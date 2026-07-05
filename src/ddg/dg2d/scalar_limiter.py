"""Zhang--Shu bound-preserving (maximum-principle) limiter for nodal DG
scalar fields.

QUARANTINED MODULE: this is developed and unit-tested in isolation. It
is NOT wired into the scalar-transport step until its unit tests pass
(conservation, bound enforcement, identity-on-smooth). Only then is it
plugged into ``scalar_transport_step_2d`` behind a flag.

Motivation
----------
The explicit high-order DG advection of a transported scalar carries no
maximum principle: on a sharp interface the nodal values overshoot the
physical range, and when the scalar drives a buoyancy body force the
overshoot feeds back and the solution diverges (see the rising-bubble
failure of the H(div) solvers). The standard remedy is the Zhang--Shu
limiter: rescale each element's deviation from its cell mean so all
nodal values lie within ``[c_lo, c_hi]``, while preserving the cell
mean exactly (so the scheme stays conservative). On data already within
bounds the scaling factor ``theta = 1`` and the limiter is the
identity, so it does not degrade the high-order convergence rate.

Reference: X. Zhang and C.-W. Shu, "On maximum-principle-satisfying high
order schemes for scalar conservation laws", JCP 229 (2010) 3091--3120.

A nodal field is stored as ``c`` of shape ``(Np, K)`` -- ``Np`` nodes
per element, ``K`` elements. The reference mass matrix ``mass_ref`` is
``(Np, Np)`` (``= inv(V @ V.T)`` for the nodal Vandermonde ``V``); on an
affine element the constant Jacobian cancels in the cell-mean ratio, so
the reference mass alone suffices.
"""
from __future__ import annotations

import jax.numpy as jnp


def cell_means(c: jnp.ndarray, mass_ref: jnp.ndarray) -> jnp.ndarray:
    """Mass-weighted cell mean of nodal field ``c`` (Np, K).

    ``cbar_K = (1^T M c_K) / (1^T M 1)`` where ``M`` is the reference
    element mass matrix. Returns shape ``(K,)``.
    """
    w = mass_ref.sum(axis=0)            # (Np,)  = 1^T M  (M symmetric)
    return (w @ c) / w.sum()            # (K,)


def zhang_shu_limit(c: jnp.ndarray, mass_ref: jnp.ndarray,
                    c_lo: float, c_hi: float) -> jnp.ndarray:
    """Bound-preserving limiter on a nodal field ``c`` (Np, K).

    Returns the limited field of the same shape, with every nodal value
    in ``[c_lo, c_hi]`` and each element's mass-weighted cell mean
    unchanged. Assumes the cell means already lie within the bounds
    (true for a conservative scheme started from in-bounds data); if a
    mean is marginally outside, the element is left at its mean.
    """
    w = mass_ref.sum(axis=0)            # (Np,)
    cbar = (w @ c) / w.sum()            # (K,)
    cmax = jnp.max(c, axis=0)           # (K,)
    cmin = jnp.min(c, axis=0)           # (K,)

    # theta in [0,1] is the largest scaling of (c - cbar) that keeps the
    # element extrema within [c_lo, c_hi]. Guard the divisions: only
    # shrink toward the mean where an extremum actually exceeds a bound.
    dmax = cmax - cbar                  # >= 0
    dmin = cbar - cmin                  # >= 0
    t_hi = jnp.where(dmax > 0.0, (c_hi - cbar) / jnp.where(dmax > 0.0, dmax, 1.0), 1.0)
    t_lo = jnp.where(dmin > 0.0, (cbar - c_lo) / jnp.where(dmin > 0.0, dmin, 1.0), 1.0)
    theta = jnp.clip(jnp.minimum(jnp.minimum(t_hi, t_lo), 1.0), 0.0, 1.0)  # (K,)

    return cbar[None, :] + theta[None, :] * (c - cbar[None, :])
