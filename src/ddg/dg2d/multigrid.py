"""hp-multigrid preconditioning for the SIPG Poisson / Helmholtz operators.

The matrix-free CG solves in :mod:`ddg.dg2d.sipg` take a preconditioner
callable ``M^{-1}(v_flat) -> v_flat``; this module builds a multigrid
V-cycle to fill that role. Block-Jacobi (the existing preconditioner)
damps only high-frequency error, so its CG iteration count grows as
the mesh refines or the degree rises; a multigrid cycle adds the
coarse-level corrections that kill the low-frequency error too, giving
a near mesh- and degree-independent iteration count.

Two builders, both geometric (rediscretised, not Galerkin) multigrid:

* :func:`make_sipg_mg_preconditioner` — **p-multigrid**: a V-cycle over
  polynomial degrees ``p, p-1, ..., p_coarse`` on a fixed mesh. The
  inter-level transfer is the exact reference-element nodal
  interpolation between the nested polynomial spaces.

* :func:`make_sipg_hpmg_preconditioner` — **hp-multigrid**: the
  p-levels above, then ``h``-levels below ``p_coarse`` that coarsen the
  *mesh* through a forest-built uniform hierarchy. Uniform refinement
  makes the h-transfer structured — every coarse element splits into
  four children the same way — so it is four fixed reference
  child-interpolation matrices, applied as a batched einsum.

The smoother is weighted block-Jacobi (matrix-free, colour-probe block
extraction); the coarsest level is solved by a direct LU factor. One
``alpha`` parameter serves both INS solves: ``alpha=0`` for the
pressure Poisson operator ``M(-Delta)`` and ``alpha>0`` for the
viscous Helmholtz operator ``alpha M + M(-Delta)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.mesh import Mesh2D, build_mesh_2d
from ddg.dg2d.reference import vandermonde_2d
from ddg.dg2d.sipg import (
    _face_1d_mass_matrices, _local_mass, extract_block_diag_matfree,
    sipg_apply_2d,
)


# ---------------------------------------------------------------------------
# Operator and level construction
# ---------------------------------------------------------------------------

def _operator_apply(mesh: Mesh2D, bc_types, tau_scale: float,
                    alpha: float) -> Callable:
    """Matrix-free apply of ``alpha M + M(-Delta_h)`` on the flat vector.

    ``alpha = 0`` gives the pure SIPG Poisson operator; ``alpha > 0``
    the viscous Helmholtz operator. The flat layout is the
    element-blocked ``u.T.reshape(-1)`` of an ``(Np, K)`` field.
    """
    M_ref = _local_mass(mesh)
    m1d = _face_1d_mass_matrices(mesh)
    Np, K = mesh.Np, mesh.K

    def apply(u_flat):
        u = u_flat.reshape(K, Np).T
        out = sipg_apply_2d(mesh, u, bc_types=bc_types, u_dir=None,
                            tau_scale=tau_scale, M_ref=M_ref,
                            m1d_per_face=m1d)
        if alpha != 0.0:
            out = out + alpha * (mesh.J * (M_ref @ u))
        return out.T.reshape(-1)

    return apply


@dataclass
class _Level:
    """One level of the multigrid hierarchy."""
    mesh: Mesh2D
    apply: Callable                 # operator A_l on the flat vector
    smooth_inv: Callable            # block-Jacobi M^{-1} for the smoother
    prolong: Callable | None        # from the next-coarser level
    restrict: Callable | None       # to the next-coarser level


def _make_level(mesh: Mesh2D, bc_types, tau_scale: float,
                alpha: float) -> _Level:
    """Build a level: matrix-free operator + block-Jacobi smoother."""
    apply = _operator_apply(mesh, bc_types, tau_scale, alpha)
    blocks = extract_block_diag_matfree(apply, mesh)
    binv = jax.vmap(jnp.linalg.inv)(blocks)
    Np = mesh.Np

    def smooth_inv(v_flat):
        v = v_flat.reshape(-1, Np)
        return jnp.einsum("kij,kj->ki", binv, v).reshape(-1)

    return _Level(mesh, apply, smooth_inv, None, None)


# ---------------------------------------------------------------------------
# Inter-level transfer — p (degree) and h (mesh)
# ---------------------------------------------------------------------------

def _p_interp_matrix(mesh_lo: Mesh2D, mesh_hi: Mesh2D) -> jax.Array:
    """Reference-element nodal interpolation, degree ``N_lo -> N_hi``.

    Shape ``(Np_hi, Np_lo)``: maps a degree-``N_lo`` nodal field to its
    values at the degree-``N_hi`` nodes. Exact, since the low-degree
    polynomial space is contained in the high-degree one.
    """
    V_lo_at_hi = vandermonde_2d(mesh_lo.N, mesh_hi.r, mesh_hi.s)
    return V_lo_at_hi @ mesh_lo.invV


def _make_p_prolong(I: jax.Array) -> Callable:
    """Coarse(lo) -> fine(hi): apply ``I`` (Np_hi, Np_lo) per element."""
    Np_lo = I.shape[1]

    def prolong(flat_lo):
        f = flat_lo.reshape(-1, Np_lo)
        return jnp.einsum("hl,kl->kh", I, f).reshape(-1)
    return prolong


def _make_p_restrict(I: jax.Array) -> Callable:
    """Fine(hi) -> coarse(lo): apply ``I^T`` per element (symmetric)."""
    Np_hi = I.shape[0]

    def restrict(flat_hi):
        f = flat_hi.reshape(-1, Np_hi)
        return jnp.einsum("hl,kh->kl", I, f).reshape(-1)
    return restrict


def _h_child_interp_matrices(mesh: Mesh2D) -> jax.Array:
    """Four reference child-interpolation matrices for uniform refinement.

    Red-refining a reference triangle gives four sub-triangles in a
    fixed arrangement. ``child[c]`` (shape ``(Np, Np)``) maps a coarse
    element's degree-``N`` nodal values to the nodal values of its
    ``c``-th red child. Returned stacked as ``(4, Np, Np)``. Child
    ordering matches ``RefinementForest._red_refine_node``.
    """
    A = np.array([-1.0, -1.0])
    B = np.array([1.0, -1.0])
    C = np.array([-1.0, 1.0])
    mAB, mBC, mCA = 0.5 * (A + B), 0.5 * (B + C), 0.5 * (C + A)
    children = [(A, mAB, mCA), (mAB, B, mBC), (mCA, mBC, C),
                (mAB, mBC, mCA)]
    r = np.asarray(mesh.r)
    s = np.asarray(mesh.s)
    mats = []
    for P0, P1, P2 in children:
        # child-local reference node (r, s) -> parent reference coord
        pr = (-(r + s) / 2) * P0[0] + ((1 + r) / 2) * P1[0] \
            + ((1 + s) / 2) * P2[0]
        ps = (-(r + s) / 2) * P0[1] + ((1 + r) / 2) * P1[1] \
            + ((1 + s) / 2) * P2[1]
        V = vandermonde_2d(mesh.N, jnp.asarray(pr), jnp.asarray(ps))
        mats.append(np.asarray(V @ mesh.invV))
    return jnp.asarray(np.stack(mats))                  # (4, Np, Np)


def _make_h_prolong(I_child: jax.Array) -> Callable:
    """Coarse -> fine (4x): fine element ``4k+c`` is child ``c`` of ``k``."""
    Np = I_child.shape[1]

    def prolong(flat_c):
        c = flat_c.reshape(-1, Np)                      # (Kc, Np)
        f = jnp.einsum("cij,kj->kci", I_child, c)       # (Kc, 4, Np)
        return f.reshape(-1)
    return prolong


def _make_h_restrict(I_child: jax.Array) -> Callable:
    """Fine (4x) -> coarse: sum the four child transposes (symmetric)."""
    Np = I_child.shape[1]

    def restrict(flat_f):
        f = flat_f.reshape(-1, 4, Np)                   # (Kc, 4, Np)
        return jnp.einsum("cij,kci->kj", I_child, f).reshape(-1)
    return restrict


def _build_h_meshes(base_mesh: Mesh2D, n_levels: int) -> list[Mesh2D]:
    """Uniform-refinement mesh hierarchy ``[coarsest, ..., finest]``.

    Built with a :class:`RefinementForest` refining every leaf at each
    step; the leaf-ordered active-mesh extraction makes element ``4k+c``
    of a finer level the ``c``-th red child of element ``k`` of the
    next-coarser one, which is what the structured h-transfer assumes.
    """
    from ddg.dg2d.refinement import RefinementForest

    forest = RefinementForest(base_mesh)
    m0, _ = forest.active_mesh()
    meshes = [m0]
    for _ in range(n_levels):
        forest.refine(forest.leaves())
        mk, _ = forest.active_mesh()
        meshes.append(mk)
    return meshes


# ---------------------------------------------------------------------------
# V-cycle assembly
# ---------------------------------------------------------------------------

def _assemble_vcycle(levels: list[_Level], *, nu_smooth: int,
                     omega: float, regularize: bool, eps: float) -> Callable:
    """Assemble the V-cycle preconditioner from a wired level stack.

    The coarsest level is solved directly (assemble via ``jax.jacobian``
    of its apply, then LU-factor). Every finer level does ``nu_smooth``
    weighted block-Jacobi pre/post-smoothing sweeps around one recursive
    coarse-grid correction.
    """
    coarse = levels[-1]
    n_c = int(coarse.mesh.Np * coarse.mesh.K)
    A_c = jax.jacobian(coarse.apply)(jnp.zeros(n_c))
    if regularize:
        A_c = A_c + eps * jnp.max(jnp.abs(jnp.diag(A_c))) * jnp.eye(n_c)
    lu_c, piv_c = jax.scipy.linalg.lu_factor(A_c)

    def coarse_solve(b):
        return jax.scipy.linalg.lu_solve((lu_c, piv_c), b)

    n_levels = len(levels)

    def vcycle(li: int, b: jax.Array) -> jax.Array:
        if li == n_levels - 1:
            return coarse_solve(b)
        lvl = levels[li]
        A, Minv = lvl.apply, lvl.smooth_inv
        x = jnp.zeros_like(b)
        for _ in range(nu_smooth):                      # pre-smooth
            x = x + omega * Minv(b - A(x))
        e = vcycle(li + 1, lvl.restrict(b - A(x)))      # coarse correction
        x = x + lvl.prolong(e)
        for _ in range(nu_smooth):                      # post-smooth
            x = x + omega * Minv(b - A(x))
        return x

    return lambda v_flat: vcycle(0, v_flat)


# ---------------------------------------------------------------------------
# p-multigrid
# ---------------------------------------------------------------------------

def make_sipg_mg_preconditioner(
    mesh: Mesh2D,
    *,
    bc_types: jax.Array,
    tau_scale: float = 200.0,
    alpha: float = 0.0,
    p_coarse: int = 1,
    nu_smooth: int = 2,
    omega: float = 0.7,
    regularize: bool = False,
    eps: float = 1.0e-8,
) -> Callable:
    """Build a p-multigrid V-cycle preconditioner for a SIPG operator.

    Returns a callable ``M^{-1}(v_flat) -> v_flat`` suitable as the
    ``preconditioner`` argument of :func:`solve_sipg_poisson_2d` /
    :func:`solve_sipg_helmholtz_2d`.

    ``alpha`` is ``0`` for the Poisson operator and the Helmholtz
    coefficient ``g0/(nu dt)`` for the viscous operator. ``p_coarse``
    is the degree of the directly-solved coarsest level. ``regularize``
    adds ``eps*max|diag|`` to the coarse matrix — needed for a singular
    operator (pure-Neumann / periodic Poisson).
    """
    p_fine = int(mesh.N)
    if p_coarse < 1 or p_coarse > p_fine:
        raise ValueError(f"p_coarse must be in [1, {p_fine}], got {p_coarse}")
    degrees = list(range(p_fine, p_coarse - 1, -1))     # fine -> coarse

    meshes = {d: (mesh if d == p_fine
                  else build_mesh_2d(mesh.VX, mesh.VY, mesh.EToV, d))
              for d in degrees}
    levels = [_make_level(meshes[d], bc_types, tau_scale, alpha)
              for d in degrees]

    for li in range(len(degrees) - 1):
        I = _p_interp_matrix(meshes[degrees[li + 1]], meshes[degrees[li]])
        levels[li].prolong = _make_p_prolong(I)
        levels[li].restrict = _make_p_restrict(I)

    return _assemble_vcycle(levels, nu_smooth=nu_smooth, omega=omega,
                            regularize=regularize, eps=eps)


# ---------------------------------------------------------------------------
# hp-multigrid
# ---------------------------------------------------------------------------

def make_sipg_hpmg_preconditioner(
    base_mesh: Mesh2D,
    n_h_levels: int,
    *,
    bc_fn: Callable[[Mesh2D], jax.Array],
    tau_scale: float = 200.0,
    alpha: float = 0.0,
    p_coarse: int = 1,
    nu_smooth: int = 2,
    omega: float = 0.7,
    regularize: bool = False,
    eps: float = 1.0e-8,
) -> tuple[Callable, Mesh2D]:
    """Build an hp-multigrid V-cycle preconditioner.

    The hierarchy is ``base_mesh`` uniformly refined ``n_h_levels``
    times; the *finest* of those meshes is the one the solve runs on
    and is returned alongside the preconditioner. The V-cycle descends
    the polynomial degrees ``p_fine, ..., p_coarse`` on that finest
    mesh, then the ``h``-levels at degree ``p_coarse`` down to
    ``base_mesh``, which is solved directly.

    ``bc_fn`` maps a mesh to its SIPG ``bc_types`` ``(K, Nfaces)`` array
    — it is called per level since the h-levels have different meshes.
    Other parameters are as for :func:`make_sipg_mg_preconditioner`.

    Returns ``(preconditioner, finest_mesh)``.
    """
    h_meshes = _build_h_meshes(base_mesh, n_h_levels)   # [coarse..fine]
    finest = h_meshes[-1]
    p_fine = int(finest.N)
    if p_coarse < 1 or p_coarse > p_fine:
        raise ValueError(f"p_coarse must be in [1, {p_fine}], got {p_coarse}")
    degrees = list(range(p_fine, p_coarse - 1, -1))

    # p-levels on the finest mesh (bc_types is degree-independent).
    bc_fine = bc_fn(finest)
    p_meshes = {d: (finest if d == p_fine
                    else build_mesh_2d(finest.VX, finest.VY, finest.EToV, d))
                for d in degrees}
    levels = [_make_level(p_meshes[d], bc_fine, tau_scale, alpha)
              for d in degrees]

    # h-levels at degree p_coarse: finest-1 down to the base mesh.
    h_level_meshes = [build_mesh_2d(hm.VX, hm.VY, hm.EToV, p_coarse)
                      for hm in h_meshes[-2::-1]]
    levels += [_make_level(hm, bc_fn(hm), tau_scale, alpha)
               for hm in h_level_meshes]

    # p-transfers between consecutive p-levels.
    n_p = len(degrees)
    for li in range(n_p - 1):
        I = _p_interp_matrix(p_meshes[degrees[li + 1]], p_meshes[degrees[li]])
        levels[li].prolong = _make_p_prolong(I)
        levels[li].restrict = _make_p_restrict(I)

    # h-transfers: the p_coarse p-level down through the h-levels.
    I_child = _h_child_interp_matrices(p_meshes[p_coarse])
    h_prolong = _make_h_prolong(I_child)
    h_restrict = _make_h_restrict(I_child)
    for li in range(n_p - 1, len(levels) - 1):
        levels[li].prolong = h_prolong
        levels[li].restrict = h_restrict

    precond = _assemble_vcycle(levels, nu_smooth=nu_smooth, omega=omega,
                               regularize=regularize, eps=eps)
    return precond, finest
