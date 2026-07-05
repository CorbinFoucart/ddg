"""hp-multigrid for the BDM_k velocity (viscous) block.

The monolithic BDM_k / P_{k-1} Navier-Stokes saddle point is solved by a
block-preconditioned Krylov method whose dominant cost is inverting the
velocity viscous block ``A_vv`` (a vector SIPG Helmholtz operator). This
module builds an hp-multigrid V/W-cycle that does that inversion with a
near mesh- and order-independent iteration count.

Why hp and not just p or just block-Jacobi (the hard-won findings):

* **block-Jacobi / element-block Schwarz fails** as a standalone Krylov
  PC and as an MG smoother — even the *exact* element-diagonal block
  gives a two-grid contraction of ~0.87. High-order DG-SIPG couples
  neighbours as strongly as the diagonal block, so an element-local
  smoother cannot damp the error the coarse grid leaves behind.

* **p-multigrid alone fails** (two-grid rho = 1.0 with exact Galerkin
  consistency): p-coarsening removes only high-*polynomial-order* error;
  the globally-smooth, low-spatial-frequency modes live identically at
  every degree and survive both the local smoother and p-coarsening.

* **hp + vertex-patch smoother + W-cycle works** and is mesh
  independent: the *h*-levels kill the global smooth modes, and the
  **vertex-patch additive Schwarz** smoother (overlapping star of
  elements around each vertex) damps the strongly-coupled high-frequency
  error that element-block smoothers miss. Measured W-cycle contraction
  on the cavity viscous block (nu=0.01, BDM_3): ~0.25 (nsm=2) / ~0.13
  (nsm=3), flat across eff. 4x4 -> 16x16.

Hierarchy (fine -> coarse): p-levels ``BDM_3 -> BDM_2 -> BDM_1`` on the
finest mesh, then h-levels coarsening the ``BDM_1`` mesh through the
uniform-refinement forest to a base mesh solved directly.

The inter-level transfers are *exact* (each coarse space is a true
subspace of the fine one): BDM_k = [P_k]^2 is nested in degree
(p-transfer) and a coarse element's field is exactly representable on its
red-refined children (h-transfer). Both are validated function-preserving
to ~1e-14 (see ``test/test_bdm_multigrid.py``).

NOTE: this reference implementation assembles each level's operator
densely (``jax.jacobian``) to extract the vertex-patch blocks and the
Galerkin coarse operators. That is fine to ~1e4 velocity DOFs per level;
the scalability follow-up is matrix-free / coloring-based sparse assembly
of the patch blocks (the V/W-cycle and transfers are unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg2d.bdm import (
    BDM, bdm1_gather, bdm_k_basis_at_points, bdm_k_viscous_apply,
)
from ddg.dg2d.cubature import triangle_cubature
from ddg.dg2d.mesh import Mesh2D, build_mesh_2d
from ddg.dg2d.refinement import RefinementForest


# ---------------------------------------------------------------------------
# p-transfer (degree): BDM_{k-1} -> BDM_k, exact polynomial inclusion
# ---------------------------------------------------------------------------

def p_transfer_reference(k_fine: int, k_coarse: int) -> jax.Array:
    """``(Np_fine, Np_coarse)`` reference inclusion: each coarse unsigned
    BDM basis expressed in the fine unsigned basis (exact, via LSQ since
    ``[P_{k-1}]^2 subset [P_k]^2``)."""
    r_q, s_q, _ = triangle_cubature(2 * k_fine + 2)
    phi_f = bdm_k_basis_at_points(k_fine, r_q, s_q)
    phi_c = bdm_k_basis_at_points(k_coarse, r_q, s_q)
    Q = phi_f.shape[0]
    A_f = phi_f.transpose(0, 2, 1).reshape(Q * 2, -1)
    A_c = phi_c.transpose(0, 2, 1).reshape(Q * 2, -1)
    P, *_ = jnp.linalg.lstsq(A_f, A_c, rcond=None)
    return P


def make_p_prolong(V_coarse: BDM, V_fine: BDM) -> Callable:
    """Prolongation BDM_{k-1} (global) -> BDM_k (global) on the same mesh."""
    P_ref = p_transfer_reference(V_fine.order, V_coarse.order)
    piola_f = V_fine.topology.piola_scale
    owner_f = V_fine.topology.local_sign > 0
    lg_f = V_fine.topology.local_to_global
    n_f = V_fine.total_dofs

    def prolong(coarse_global):
        cl = bdm1_gather(V_coarse.topology, coarse_global)
        dl = jnp.einsum("fc,kc->kf", P_ref, cl)
        contrib = jnp.where(owner_f, dl / piola_f, 0.0)
        return jnp.zeros(n_f).at[lg_f.reshape(-1)].add(contrib.reshape(-1))

    return prolong


# ---------------------------------------------------------------------------
# h-transfer (mesh): coarse element -> 4 red children, contravariant Piola
# ---------------------------------------------------------------------------

# Child placements in parent reference coords; ordering matches
# RefinementForest red refinement (and multigrid._h_child_interp_matrices).
_A = np.array([-1.0, -1.0]); _B = np.array([1.0, -1.0]); _C = np.array([-1.0, 1.0])
_mAB, _mBC, _mCA = 0.5 * (_A + _B), 0.5 * (_B + _C), 0.5 * (_C + _A)
_CHILDREN = [(_A, _mAB, _mCA), (_mAB, _B, _mBC), (_mCA, _mBC, _C),
             (_mAB, _mBC, _mCA)]


def h_child_transfer_matrices(k: int) -> jax.Array:
    """``(4, Np, Np)`` transfer of coarse unsigned coeffs to each red
    child's unsigned coeffs. For red refinement the child reference->parent
    map ``G_c(xi) = P0 + 0.5(xi_r+1)(P1-P0) + 0.5(xi_s+1)(P2-P0)`` is fixed,
    so under the contravariant Piola map
    ``sum_m T_c[m,l] phi_m(xi) = det(B_c) B_c^{-1} phi_l(G_c(xi))``."""
    r_q, s_q, _ = triangle_cubature(2 * k + 2)
    phi_child = bdm_k_basis_at_points(k, r_q, s_q)            # (Q, Np, 2)
    Q = phi_child.shape[0]
    A_child = phi_child.transpose(0, 2, 1).reshape(Q * 2, -1)  # (Q*2, Np)
    mats = []
    for (P0, P1, P2) in _CHILDREN:
        Bc = np.stack([0.5 * (P1 - P0), 0.5 * (P2 - P0)], axis=1)
        detB = float(np.linalg.det(Bc))
        Binv = jnp.asarray(np.linalg.inv(Bc))
        eta = (P0[None, :] + 0.5 * (r_q[:, None] + 1.0) * (P1 - P0)[None, :]
               + 0.5 * (s_q[:, None] + 1.0) * (P2 - P0)[None, :])
        phi_parent = bdm_k_basis_at_points(k, eta[:, 0], eta[:, 1])
        rhs = detB * jnp.einsum("ab,qlb->qla", Binv, phi_parent)
        rhs_flat = rhs.transpose(0, 2, 1).reshape(Q * 2, -1)
        T, *_ = jnp.linalg.lstsq(A_child, rhs_flat, rcond=None)
        mats.append(T)
    return jnp.stack(mats)


def make_h_prolong(V_coarse: BDM, V_fine: BDM) -> Callable:
    """Prolongation: coarse mesh -> uniformly-refined fine mesh (4x).
    Assumes fine element ``4k+c`` is child ``c`` of coarse element ``k``
    (the forest's uniform-refinement ordering)."""
    Tc = h_child_transfer_matrices(V_coarse.order)
    piola_f = V_fine.topology.piola_scale
    owner_f = V_fine.topology.local_sign > 0
    lg_f = V_fine.topology.local_to_global
    n_f = V_fine.total_dofs
    Kc = V_coarse.K

    def prolong(coarse_global):
        cl = bdm1_gather(V_coarse.topology, coarse_global)
        d = jnp.einsum("cml,kl->kcm", Tc, cl)                # (Kc, 4, Np)
        d_fine = d.reshape(Kc * 4, -1)
        contrib = jnp.where(owner_f, d_fine / piola_f, 0.0)
        return jnp.zeros(n_f).at[lg_f.reshape(-1)].add(contrib.reshape(-1))

    return prolong


# ---------------------------------------------------------------------------
# Vertex-patch (overlapping Schwarz) smoother
# ---------------------------------------------------------------------------

def vertex_patches(V: BDM) -> list[np.ndarray]:
    """Per-vertex DOF index arrays: the union of the global DOFs of all
    elements in the star of each mesh vertex (overlapping patches)."""
    EToV = np.asarray(V.mesh.EToV)
    lg = np.asarray(V.topology.local_to_global)
    nverts = int(EToV.max()) + 1
    elems_of_v: list[list[int]] = [[] for _ in range(nverts)]
    for e in range(EToV.shape[0]):
        for vtx in EToV[e]:
            elems_of_v[int(vtx)].append(e)
    patches = []
    for v in range(nverts):
        if elems_of_v[v]:
            patches.append(
                np.unique(np.concatenate([lg[e] for e in elems_of_v[v]])))
    return patches


def make_vertex_patch_smoother(A_dense: np.ndarray,
                               patches: list[np.ndarray]) -> Callable:
    """Additive-Schwarz smoother ``M^{-1} = sum_p R_p^T A_pp^{-1} R_p`` from
    the dense operator. Returns a matvec callable on the flat vector."""
    n = A_dense.shape[0]
    blocks = [(jnp.asarray(p), jnp.asarray(np.linalg.inv(A_dense[np.ix_(p, p)])))
              for p in patches]

    def M_inv(r):
        out = jnp.zeros(n)
        for idx, binv in blocks:
            out = out.at[idx].add(binv @ r[idx])
        return out

    return M_inv


# ---------------------------------------------------------------------------
# Coloring-based sparse assembly of the viscous block
# ---------------------------------------------------------------------------
#
# Dense ``jax.jacobian`` of the viscous apply uses O(n^2) memory, which
# becomes infeasible past ~1e4 velocity DOFs (a 14k-DOF block is ~1.6 GB).
# Element-coloring assembly is the standard fix for DG-block-sparse
# operators: colour elements so face-adjacent ones disagree (the only
# coupling channel for BDM-SIPG), then probe each (colour, local-slot)
# with a single matrix-free apply -- the colouring guarantees the columns
# don't alias, so a constant number of applies (~n_colours x Np ~ 100)
# yields the full sparse matrix regardless of mesh size.

def _color_elements_distance_3(mesh: Mesh2D) -> np.ndarray:
    """Greedy distance-3 colouring: ``colors[k] = c`` such that no two
    elements within three face-hops share a colour. Returns ``(K,)``.

    Distance-3 is required for SIPG sparse-assembly: when probing column
    ``g'`` owned by element ``k'``, the row contamination reaches row
    ``r in n`` through the SIPG face-coupling chain
    ``k -> n -> m -> k'`` (face-adjacencies). To avoid contamination
    when ``k`` is also colour ``c``, ``k`` and ``k'`` must have different
    colours, i.e. distance-3 separation. Distance-2 (which prevents
    "shared third common neighbour" aliasing for diffusion-only DG) is
    not enough because SIPG couples DOFs across interior faces. For
    triangle meshes ~13-20 colours suffice -- still a bounded constant."""
    EToE = np.asarray(mesh.EToE)
    K = mesh.K
    colors = -np.ones(K, dtype=np.int64)
    for k in range(K):
        ball = set()
        frontier = {k}
        for _hop in range(3):
            next_frontier = set()
            for kk in frontier:
                for f in range(3):
                    kn = int(EToE[kk, f])
                    if kn != kk:
                        next_frontier.add(kn)
            ball.update(next_frontier)
            frontier = next_frontier
        ball.discard(k)
        forbidden = {int(colors[n]) for n in ball if colors[n] >= 0}
        c = 0
        while c in forbidden:
            c += 1
        colors[k] = c
    return colors


def _assemble_sparse_via_probing(V: BDM, apply_fn: Callable):
    """Matrix-free single-column probing assembly of any linear operator
    ``apply_fn(u_global) -> y_global`` defined on the BDM-DOF space.
    Returns ``scipy.sparse.csr_matrix`` of shape ``(n_u, n_u)``.

    For each global DOF ``g`` we probe via its OWNER side
    (``sign[k, j] = +1``, ``lg[k, j] = g``): set ``probe[g] = 1/piola``,
    apply, scale and read all nonzero rows. Cost: ``n_u`` jitted applies.
    Memory ``O(nnz)`` instead of dense ``O(n_u^2)``. Works for viscous,
    transient-Helmholtz (``alpha*M + nu*A_visc``), Stokes velocity
    block, etc. -- whatever linear operator the caller supplies."""
    from scipy.sparse import coo_matrix
    K = V.K
    lg = np.asarray(V.topology.local_to_global)
    sign = np.asarray(V.topology.local_sign).astype(np.float64)
    piola = np.asarray(V.topology.piola_scale).astype(np.float64)
    Np = lg.shape[1]
    n = V.total_dofs
    sp = sign * piola                                       # (K, Np)

    owner_kj = -np.ones((n, 2), dtype=np.int64)
    for k in range(K):
        for j in range(Np):
            g = int(lg[k, j])
            if owner_kj[g, 0] < 0 and sign[k, j] > 0:
                owner_kj[g, 0] = k; owner_kj[g, 1] = j

    rows, cols, data = [], [], []
    for g in range(n):
        k = int(owner_kj[g, 0]); j = int(owner_kj[g, 1])
        if k < 0:
            continue                                        # unused DOF index
        sc = sp[k, j]                                       # = +piola (owner)
        probe = np.zeros(n)
        probe[g] = 1.0 / sc
        y = np.asarray(apply_fn(jnp.asarray(probe)))
        scaled = sc * y                                     # = A[:, g]
        nonzero = np.where(np.abs(scaled) > 1.0e-14)[0]
        rows.extend(int(r) for r in nonzero)
        cols.extend([g] * len(nonzero))
        data.extend(float(scaled[r]) for r in nonzero)
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A.eliminate_zeros()
    return A


def assemble_sparse_viscous(V: BDM, *, nu: float, tau_scale: float = 4.0):
    """Sparse assembly of the BDM viscous block ``nu * A_visc``. Thin
    wrapper around :func:`_assemble_sparse_via_probing` for the
    steady-Stokes/cavity case (no mass term)."""
    apply_fn = jax.jit(lambda u: bdm_k_viscous_apply(
        V, u, nu=nu, tau_scale=tau_scale))
    return _assemble_sparse_via_probing(V, apply_fn)


def assemble_sparse_velocity_helmholtz(
    V: BDM, *, alpha: float, nu: float, tau_scale: float = 4.0,
):
    """Sparse assembly of the BDM transient-Helmholtz velocity block
    ``alpha * M + nu * A_visc``. ``alpha = gamma0 / dt`` for BDF. Used as
    the level operator of the velocity hp-MG when preconditioning the
    transient (Boussinesq, bubble, transient cavity) saddle point --
    *the mass term is essential* there: at small ``dt`` ``alpha * M``
    completely dominates the viscous part, so a viscous-only PC fails."""
    from ddg.dg2d.coupled_ns_bdm import bdm_k_mass_apply
    apply_fn = jax.jit(lambda u: (
        alpha * bdm_k_mass_apply(V, u)
        + bdm_k_viscous_apply(V, u, nu=nu, tau_scale=tau_scale)))
    return _assemble_sparse_via_probing(V, apply_fn)


def _make_vertex_patch_smoother_sparse(A_sparse,
                                       patches: list[np.ndarray]) -> Callable:
    """Additive-Schwarz smoother from a sparse operator: extract each
    patch's dense submatrix, factor once, apply matrix-free in the cycle."""
    from scipy.linalg import lu_factor, lu_solve
    factored = []
    A_csc = A_sparse.tocsc()
    for p in patches:
        sub = np.asarray(A_csc[p[:, None], p[None, :]].todense())
        factored.append((jnp.asarray(p), lu_factor(sub)))

    def M_inv(r):
        r_np = np.asarray(r)
        out = np.zeros_like(r_np)
        for idx, lu in factored:
            idx_np = np.asarray(idx)
            out[idx_np] += lu_solve(lu, r_np[idx_np])
        return jnp.asarray(out)

    return M_inv


def _lam_max(M_inv: Callable, A: Callable, n: int, *, seed=0, iters=40) -> float:
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(n)); v = v / jnp.linalg.norm(v)
    lam = 1.0
    for _ in range(iters):
        w = M_inv(A(v)); lam = float(jnp.linalg.norm(w)); v = w / lam
    return lam


# ---------------------------------------------------------------------------
# Hierarchy + V/W-cycle
# ---------------------------------------------------------------------------

@dataclass
class _Level:
    V: BDM
    apply: Callable
    smooth_inv: Callable | None
    omega: float
    prolong: Callable | None      # from next-coarser level
    A_dense: np.ndarray | None    # for Galerkin coarse build + coarse solve


def _h_mesh_hierarchy(base_mesh: Mesh2D, n_h: int) -> list[Mesh2D]:
    forest = RefinementForest(base_mesh)
    meshes = [forest.active_mesh()[0]]
    for _ in range(n_h):
        forest.refine(forest.leaves())
        meshes.append(forest.active_mesh()[0])
    return meshes                                   # [coarse .. fine]


def build_velocity_hpmg(
    base_mesh: Mesh2D, n_h: int, *, order: int = 3, nu: float,
    tau_scale: float = 4.0, nsm: int = 2, cycle: str = "W",
    sparse_fine: bool = False, velocity_alpha: float = 0.0,
) -> tuple[BDM, Callable, Callable]:
    """Build the hp-MG cycle for the BDM_``order`` velocity viscous block
    on ``base_mesh`` uniformly refined ``n_h`` times.

    Returns ``(V_fine, A_apply, mg_apply)`` where ``V_fine`` is the finest
    BDM space, ``A_apply(u)`` the matrix-free viscous block on it, and
    ``mg_apply(r)`` one V/W-cycle (a preconditioner ``M^{-1} r``).
    """
    if cycle not in ("V", "W"):
        raise ValueError("cycle must be 'V' or 'W'")
    gamma = 1 if cycle == "V" else 2
    hm = _h_mesh_hierarchy(base_mesh, n_h)
    finest = hm[-1]

    Vp = [BDM.from_mesh(finest, order=k) for k in range(order, 0, -1)]
    Vh = [BDM.from_mesh(m, order=1) for m in hm[-2::-1]]   # finest-1 .. base
    Vs = Vp + Vh                                            # fine -> coarse

    def visc(V):
        """Per-level velocity-block apply: ``alpha * M + nu * A_visc``.
        For ``alpha = 0`` (steady) this is the pure viscous block."""
        from ddg.dg2d.coupled_ns_bdm import bdm_k_mass_apply
        if velocity_alpha == 0.0:
            return lambda u: bdm_k_viscous_apply(
                V, u, nu=nu, tau_scale=tau_scale)
        return lambda u: (
            velocity_alpha * bdm_k_mass_apply(V, u)
            + bdm_k_viscous_apply(V, u, nu=nu, tau_scale=tau_scale))

    # Transfers (prolong from next-coarser into each level).
    prolongs: list[Callable | None] = [None] * len(Vs)
    for li in range(order - 1):                             # p-levels
        prolongs[li] = make_p_prolong(Vp[li + 1], Vp[li])
    V1_levels = [Vp[-1]] + Vh                               # BDM_1 chain
    for j in range(len(V1_levels) - 1):                     # h-levels
        prolongs[order - 1 + j] = make_h_prolong(V1_levels[j + 1], V1_levels[j])

    # Operators: finest level is dense (default) or sparse (via
    # coloring-based matrix-free assembly, ``sparse_fine=True``); coarser
    # levels are dense via Galerkin ``A_l = P^T A_{l-1} P``. With
    # ``sparse_fine``, only the bottleneck (finest) matrix avoids the
    # O(n^2) dense ``jacobian`` memory cost.
    A_sparse_fine = None
    if sparse_fine:
        if velocity_alpha == 0.0:
            A_sparse_fine = assemble_sparse_viscous(
                Vs[0], nu=nu, tau_scale=tau_scale)
        else:
            A_sparse_fine = assemble_sparse_velocity_helmholtz(
                Vs[0], alpha=velocity_alpha, nu=nu, tau_scale=tau_scale)
        Pl0 = np.asarray(
            jax.jacobian(prolongs[0])(jnp.zeros(Vs[1].total_dofs)))
        A1 = Pl0.T @ np.asarray(A_sparse_fine @ Pl0)
        A_dense: list = [None, A1]
        for li in range(2, len(Vs)):
            Pl = np.asarray(
                jax.jacobian(prolongs[li - 1])(jnp.zeros(Vs[li].total_dofs)))
            A_dense.append(Pl.T @ A_dense[-1] @ Pl)
    else:
        A0 = np.asarray(
            jax.jacobian(visc(Vs[0]))(jnp.zeros(Vs[0].total_dofs)))
        A_dense = [A0]
        for li in range(1, len(Vs)):
            Pl = np.asarray(
                jax.jacobian(prolongs[li - 1])(jnp.zeros(Vs[li].total_dofs)))
            A_dense.append(Pl.T @ A_dense[-1] @ Pl)

    applies = [visc(V) for V in Vs]
    levels: list[_Level] = []
    for li in range(len(Vs)):
        if li < len(Vs) - 1:
            patches = vertex_patches(Vs[li])
            if li == 0 and sparse_fine:
                Minv = _make_vertex_patch_smoother_sparse(
                    A_sparse_fine, patches)
            else:
                Minv = make_vertex_patch_smoother(A_dense[li], patches)
            om = 1.0 / _lam_max(Minv, applies[li], Vs[li].total_dofs)
        else:
            Minv, om = None, 0.0
        levels.append(_Level(Vs[li], applies[li], Minv, om,
                             prolongs[li], A_dense[li]))

    coarse_lu = jax.scipy.linalg.lu_factor(jnp.asarray(A_dense[-1]))
    nlev = len(levels)

    def cyc(li, b):
        if li == nlev - 1:
            return jax.scipy.linalg.lu_solve(coarse_lu, b)
        lvl = levels[li]
        # ``lvl.prolong`` maps the next-coarser level (li+1) up to li.
        A, M, om, P = lvl.apply, lvl.smooth_inv, lvl.omega, lvl.prolong
        x = jnp.zeros_like(b)
        for _ in range(nsm):
            x = x + om * M(b - A(x))
        # restrict residual with P^T (exact adjoint of the prolongation)
        restrict = jax.linear_transpose(P, jnp.zeros(levels[li + 1].V.total_dofs))
        rc = restrict(b - A(x))[0]
        e = cyc(li + 1, rc)
        for _ in range(gamma - 1):
            e = e + cyc(li + 1, rc - levels[li + 1].apply(e))
        x = x + P(e)
        for _ in range(nsm):
            x = x + om * M(b - A(x))
        return x

    return Vs[0], applies[0], (lambda r: cyc(0, r))


# ---------------------------------------------------------------------------
# Coupled block-diagonal PC for the BDM_k / P_{k-1} saddle point
# ---------------------------------------------------------------------------

def make_coupled_velocity_mg_pc(
    V_fine: BDM, base_mesh: Mesh2D, n_h: int, *, nu: float,
    tau_scale: float = 4.0, nsm: int = 2, cycle: str = "W",
    pressure_scale: float = 1.0, sparse_fine: bool = False,
    velocity_alpha: float = 0.0,
) -> Callable:
    """Block-diagonal coupled PC for the BDM_k / P_{k-1} steady saddle
    point: the velocity block is one hp-MG V/W-cycle for the viscous
    block, the pressure block is the order-parametric pressure-mass
    Schur (Cahouet-Chabard viscous limit). Operates on the Krylov-flat
    ``(u, p)`` vector of size ``n_u + K * dim_p``.

    Built once per ``(mesh, nu)`` -- the hierarchy and pressure mass do
    not change inside a Newton solve, so this is one-time setup that is
    reused across Newton iterations and GMRES iterations. The convective
    Jacobian (which does change per Newton step) is treated as a
    perturbation handled by the outer Krylov method, not by the PC.
    """
    from ddg.dg2d.coupled_ns_bdm import (
        make_bdm_k_pressure_mass_inverse, make_bdm_k_block_diag_pc,
    )
    if V_fine.order < 1:
        raise ValueError(f"V_fine.order must be >= 1; got {V_fine.order}")
    _, _, mg_apply = build_velocity_hpmg(
        base_mesh, n_h, order=V_fine.order, nu=nu, tau_scale=tau_scale,
        nsm=nsm, cycle=cycle, sparse_fine=sparse_fine,
        velocity_alpha=velocity_alpha,
    )
    p_pc = make_bdm_k_pressure_mass_inverse(
        V_fine, nu=nu, scale=pressure_scale)
    return make_bdm_k_block_diag_pc(
        V_fine, velocity_pc=mg_apply, pressure_pc=p_pc)
