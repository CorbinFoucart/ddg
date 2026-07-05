"""Pressure-robust velocity reconstructions (H(div)-projections).

Concept: the standard L²-DG velocity space is not pointwise
divergence-free, so the discrete velocity has a non-zero normal-jump
across faces. In the discrete momentum balance this jump couples to
the pressure gradient, contaminating the velocity error with a term
proportional to ``1/ν``. Linke (2014, CMAME) showed that replacing
the DG velocity ``u_h`` with its projection onto an H(div)-conforming
space ``Π(u_h)`` — wherever it feeds the right-hand side or a
divergence operator — restores pressure-robustness without changing
the underlying basis or the rest of the time-stepping machinery.

This module provides two reconstruction operators:

  ``reconstruct_velocity_rt0_2d``       (the proper Linke / Brezzi-Fortin
                                          RT_0 projection — closed form per
                                          element, integrated face-flux
                                          continuity matches at the moment
                                          level, divergence-rate is constant
                                          per element)

  ``reconstruct_velocity_face_normal_avg_2d``  (a cheap face-node-level
                                                normal-averaging — *not* a
                                                proper Linke reconstruction;
                                                kept as a baseline for
                                                comparison only)

For lowest-order RT_0:
  * 3 DOFs per triangle (one normal-flux moment per face)
  * Basis: ``φ_i(x) = (x − v_i^opp) / (2|K|)`` where ``v_i^opp`` is the
    vertex opposite face ``i``; satisfies
    ``∫_{F_j} φ_i · n_j dS = δ_{ij}``.
  * Reconstructed velocity in physical-node form:
    ``Π u(x) = Σ_i q_i (x − v_i^opp) / (2|K|)``
    where ``q_i = ∫_{F_i} {{u}} · n_i dS`` is the average face flux.

Both reconstruction operators are differentiable JAX primitives, so
``jax.grad`` through them continues to work end-to-end.

References
----------
* Linke, A. (2014). "On the role of the Helmholtz decomposition in
  mixed methods for incompressible flows and a new variational crime."
  *Computer Methods in Applied Mechanics and Engineering* 268,
  782--800.
* John, V., Linke, A., Merdon, C., Neilan, M., Rebholz, L. (2017).
  "On the divergence constraint in mixed finite element methods for
  incompressible flows." *SIAM Review* 59(3), 492--544.
* Fehn, N., Kronbichler, M., Lube, G. (2025). *Flow, Turbulence and
  Combustion* 115:347--388.  §4.7 contrasts the L² penalty path with
  the H(div) reconstruction path.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ddg.dg2d.mesh import Mesh2D


def reconstruct_velocity_face_normal_avg_2d(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    *,
    bc_normal_overrides: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Replace face-node values with their (M+P)/2 normal average.

    Returns ``(ux_rec, uy_rec)`` of shape ``(Np, K)`` — same as the
    input but with face-node entries modified so the normal component
    at each face node is the M-P average. Interior volume nodes are
    unchanged.

    ``bc_normal_overrides`` (optional): tuple of two ``(Np, K)`` arrays
    holding the prescribed ``(ux_BC, uy_BC)`` at boundary face nodes.
    When given, the boundary-face normal trace is replaced by
    ``u_BC · n`` instead of the M average. This is what we'd want at
    inflow / wall / obstacle faces in the flow-past-square setup.

    Notes on the math:
      *  At interior faces the M and P traces in our face indexing
         already point at the two sides of the same face. Averaging
         them gives the harmonic / arithmetic mean of the two traces.
      *  At boundary faces ``vmapM == vmapP``, so the bare average is
         just the M trace. We accept ``bc_normal_overrides`` so that
         the caller can supply the Dirichlet velocity field directly,
         matching what the rest of the splitting pipeline does.
      *  The tangential component is left untouched. This is what
         distinguishes "normal-average" from a full BDM projection.
    """
    Np, K = mesh.Np, mesh.K
    Nfp, Nfaces = mesh.Nfp, mesh.Nfaces
    n_face = Nfp * Nfaces

    nx_flat = mesh.nx.reshape(-1, order="F")        # (Nfp*Nfaces*K,)
    ny_flat = mesh.ny.reshape(-1, order="F")

    ux_flat = ux.T.reshape(-1)                      # (Np*K,)
    uy_flat = uy.T.reshape(-1)

    # Face traces, M and P sides.
    ux_M = ux_flat[mesh.vmapM]                      # (Nfp*Nfaces*K,)
    ux_P = ux_flat[mesh.vmapP]
    uy_M = uy_flat[mesh.vmapM]
    uy_P = uy_flat[mesh.vmapP]

    # Optional BC override on boundary faces.
    if bc_normal_overrides is not None:
        bc_ux, bc_uy = bc_normal_overrides
        bc_ux_flat = bc_ux.T.reshape(-1)
        bc_uy_flat = bc_uy.T.reshape(-1)
        # boundary <=> vmapP == vmapM
        is_bdy = mesh.vmapP == mesh.vmapM
        ux_P = jnp.where(is_bdy, bc_ux_flat[mesh.vmapM], ux_P)
        uy_P = jnp.where(is_bdy, bc_uy_flat[mesh.vmapM], uy_P)

    # Average the normal component of the two traces.
    un_M = ux_M * nx_flat + uy_M * ny_flat
    un_P = ux_P * nx_flat + uy_P * ny_flat
    un_avg = 0.5 * (un_M + un_P)

    # Reconstructed M-side trace: replace the normal component, keep
    # the tangential. Decompose u_M = (u_M · n) n + (u_M · t) t.
    # u_rec = un_avg * n + (u_M - un_M * n) = u_M + (un_avg - un_M) * n
    delta_n = un_avg - un_M
    ux_rec_M = ux_M + delta_n * nx_flat
    uy_rec_M = uy_M + delta_n * ny_flat

    # Scatter the reconstructed face-node values back into the volume
    # arrays. We need to write to mesh.vmapM. Build (Np*K,) arrays.
    n_vol = Np * K
    # Start from the original volume values.
    ux_out_flat = ux_flat
    uy_out_flat = uy_flat
    # Scatter face-node corrections via an index update. Each volume
    # node may appear multiple times in vmapM (if it lies on multiple
    # faces — at triangle vertices); the last write wins. For the
    # normal-average reconstruction this is consistent because the
    # corner contributions from neighbouring faces should produce the
    # same averaged value when the mesh is well-conditioned.
    ux_out_flat = ux_out_flat.at[mesh.vmapM].set(ux_rec_M)
    uy_out_flat = uy_out_flat.at[mesh.vmapM].set(uy_rec_M)

    ux_out = ux_out_flat.reshape(K, Np).T
    uy_out = uy_out_flat.reshape(K, Np).T
    return ux_out, uy_out


def reconstruct_velocity_rt0_2d(
    mesh: Mesh2D,
    ux: jax.Array,
    uy: jax.Array,
    *,
    bc_normal_overrides: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Proper lowest-order Raviart-Thomas (RT_0) Linke projection.

    Element-local closed form:
        Π u(x) = Σ_i q_i · (x − v_i^opp) / (2|K|)
    with
        q_i = ∫_{F_i} {{u}} · n_i dS   (average face flux from both sides)

    Properties:
      * ``∫_{F_i} Π u · n_i dS = q_i`` on every face — the *moment* of
        the normal flux is continuous across interior faces by
        construction.
      * ``∇·(Π u)`` is constant per element, equal to
        ``Σ_i q_i / |K|`` (the integrated outward flux divided by area).
      * Tangential information past the face-flux moment is lost (RT_0
        is the *lowest* order); a higher-order BDM_k projection would
        preserve more.
      * Differentiable through ``jax.grad`` end-to-end — pure JAX
        gather, multiply, sum.

    ``bc_normal_overrides``: optional ``(ux_bc, uy_bc)`` boundary
    velocity fields. On boundary faces (``vmapP == vmapM``) the
    averaging uses the prescribed BC trace instead of the bare M-side
    value; this matches the upstream handling in the convective term.
    """
    Np, K = mesh.Np, mesh.K
    Nfp, Nfaces = mesh.Nfp, mesh.Nfaces

    # ----- face traces, average ----- ----- ----- ----- ----- -----
    nx_flat = mesh.nx.reshape(-1, order="F")        # (Nfp*Nfaces*K,)
    ny_flat = mesh.ny.reshape(-1, order="F")
    sJ_flat = mesh.sJ.reshape(-1, order="F")

    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)
    ux_M = ux_flat[mesh.vmapM]
    ux_P = ux_flat[mesh.vmapP]
    uy_M = uy_flat[mesh.vmapM]
    uy_P = uy_flat[mesh.vmapP]
    if bc_normal_overrides is not None:
        bc_ux, bc_uy = bc_normal_overrides
        bc_ux_flat = bc_ux.T.reshape(-1)
        bc_uy_flat = bc_uy.T.reshape(-1)
        is_bdy = mesh.vmapP == mesh.vmapM
        ux_P = jnp.where(is_bdy, bc_ux_flat[mesh.vmapM], ux_P)
        uy_P = jnp.where(is_bdy, bc_uy_flat[mesh.vmapM], uy_P)
    un_avg = 0.5 * ((ux_M + ux_P) * nx_flat + (uy_M + uy_P) * ny_flat)

    # ----- integrated face flux per face ----- ----- ----- -----
    # Use the 1D face mass matrix from sipg. Per-face weights along
    # the face are the row sum of m1d (in reference parameter space).
    # Face length in physical coords is sJ_F · (face length in ref) =
    # sJ_F · 2.  The integration is
    #   q_F^K = ∫_F un_avg dS = sJ_F · w_1d^T · un_avg|_face_nodes
    # where w_1d sums to 2 (reference face length).
    from ddg.dg2d.sipg import _face_1d_mass_matrices
    m1d = _face_1d_mass_matrices(mesh)              # (Nfaces, Nfp, Nfp)
    w_1d = jnp.sum(m1d, axis=2)                     # (Nfaces, Nfp)

    un_avg_face = un_avg.reshape((Nfp, Nfaces, K), order="F")  # (Nfp, Nfaces, K)
    w_per_face = w_1d.T                              # (Nfp, Nfaces)
    sJ_face = sJ_flat.reshape((Nfp, Nfaces, K), order="F")[0]  # (Nfaces, K)
    face_flux = sJ_face * jnp.einsum("pf,pfk->fk", w_per_face, un_avg_face)
    # face_flux has shape (Nfaces, K) — q_i^K for face i of element K.

    # ----- element vertex coords and opposite vertices ---------------
    v_x = mesh.VX[mesh.EToV]                         # (K, 3)
    v_y = mesh.VY[mesh.EToV]                         # (K, 3)

    # Hesthaven convention: face 0 between v0,v1 (opp v2); face 1
    # between v1,v2 (opp v0); face 2 between v0,v2 (opp v1). So:
    opp = jnp.array([2, 0, 1])                       # face_idx -> vertex_idx
    v_opp_x = v_x[:, opp]                            # (K, 3) — vertex opposite each face
    v_opp_y = v_y[:, opp]                            # (K, 3)

    # Element area / RT_0 normalisation.
    # The reference triangle in (r, s) has area 2 (vertices at
    # (-1,-1), (1,-1), (-1,1)), so the physical area is |K| = 2·J.
    # The RT_0 basis normalisation in the closed form
    # ``Π u(x) = Σ q_i (x - v_i^opp) / (2|K|)`` thus uses 2|K| = 4·J.
    twoK = 4.0 * mesh.J[0]                           # (K,)

    # ----- evaluate Π u at every volume node ---------------------------
    # face_flux:    (Nfaces, K)         -> (1, Nfaces, K)
    # v_opp_x/y:   (K, Nfaces)          -> (1, Nfaces, K)  (after .T)
    # mesh.x / .y: (Np, K)              -> (Np, 1, K)
    ff = face_flux[None, :, :]                       # (1, 3, K)
    vx_opp = v_opp_x.T[None, :, :]                   # (1, 3, K)
    vy_opp = v_opp_y.T[None, :, :]                   # (1, 3, K)
    x_nodes = mesh.x[:, None, :]                     # (Np, 1, K)
    y_nodes = mesh.y[:, None, :]                     # (Np, 1, K)
    inv_twoK = (1.0 / twoK)[None, :]                 # (1, K)

    Pi_ux = jnp.sum(ff * (x_nodes - vx_opp), axis=1) * inv_twoK  # (Np, K)
    Pi_uy = jnp.sum(ff * (y_nodes - vy_opp), axis=1) * inv_twoK  # (Np, K)
    return Pi_ux, Pi_uy


def face_normal_jump_l2_2d(
    mesh: Mesh2D, ux: jax.Array, uy: jax.Array,
) -> float:
    """Diagnostic: total ``Σ_F sJ_F · ([[u·n]])²`` summed over interior
    faces (pointwise jump). RT_0 does *not* drive this to zero — only
    the integrated moment of the jump is annihilated. Use
    :func:`face_normal_flux_moment_jump_2d` for the moment-level
    diagnostic.
    """
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    sJ_flat = mesh.sJ.reshape(-1, order="F")
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)

    un_M = ux_flat[mesh.vmapM] * nx_flat + uy_flat[mesh.vmapM] * ny_flat
    un_P = ux_flat[mesh.vmapP] * nx_flat + uy_flat[mesh.vmapP] * ny_flat
    jump = un_M - un_P
    # interior faces have vmapP != vmapM; boundary faces have jump=0
    return float(jnp.sum(sJ_flat * jump * jump))


def face_normal_flux_moment_jump_2d(
    mesh: Mesh2D, ux: jax.Array, uy: jax.Array,
) -> float:
    """Diagnostic: ``Σ_F (∫_F [[u·n]] dS)²`` over interior faces.

    The integrated jump of the normal velocity, per face, squared and
    summed. RT_0 reconstruction (:func:`reconstruct_velocity_rt0_2d`)
    drives this to machine zero by construction. The pointwise jump
    (above) is not annihilated.
    """
    Nfp, Nfaces, K = mesh.Nfp, mesh.Nfaces, mesh.K
    nx_flat = mesh.nx.reshape(-1, order="F")
    ny_flat = mesh.ny.reshape(-1, order="F")
    sJ_flat = mesh.sJ.reshape(-1, order="F")
    ux_flat = ux.T.reshape(-1)
    uy_flat = uy.T.reshape(-1)

    un_M = ux_flat[mesh.vmapM] * nx_flat + uy_flat[mesh.vmapM] * ny_flat
    un_P = ux_flat[mesh.vmapP] * nx_flat + uy_flat[mesh.vmapP] * ny_flat
    jump = un_M - un_P                              # (Nfp*Nfaces*K,)

    from ddg.dg2d.sipg import _face_1d_mass_matrices
    m1d = _face_1d_mass_matrices(mesh)              # (Nfaces, Nfp, Nfp)
    w_1d = jnp.sum(m1d, axis=2)                     # (Nfaces, Nfp)
    w_per_face = w_1d.T                              # (Nfp, Nfaces)

    jump_face = jump.reshape((Nfp, Nfaces, K), order="F")
    sJ_face = sJ_flat.reshape((Nfp, Nfaces, K), order="F")[0]  # (Nfaces, K)
    flux_jump = sJ_face * jnp.einsum("pf,pfk->fk", w_per_face, jump_face)
    return float(jnp.sum(flux_jump * flux_jump))
