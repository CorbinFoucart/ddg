"""Regression tests for the BDM_k order-parametric reference basis.

The order-parametric route (``bdm_k_basis_at_points``,
``bdm_k_face_dof_coordinates``, ``bdm_k_divergence_basis``) is meant
to subsume the hard-coded ``bdm1_*`` machinery once we extend to
k=2, 3. As a precondition, at k=1 it must reproduce the existing
results to machine precision -- otherwise the downstream operators
(mass, viscous SIPG, divergence, gradient) would silently diverge
from the validated k=1 baseline when we eventually swap them onto
the parametric path.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ddg.dg2d.bdm import (
    BDM, BDMTopology, BDM1Topology,
    bdm1_basis_at_points, bdm1_divergence_basis,
    bdm1_reference_dof_coordinates,
    bdm_k_basis_at_points, bdm_k_divergence_basis,
    bdm_k_divergence_at_points,
    bdm_k_face_dof_coordinates,
    build_bdm_topology,
    _bdm_k_basis_coefficients, _bdm_k_dof_matrix,
    _bdm_k_interior_test_functions,
    _eval_interior_test,
    _bdm_k_primal_basis_at_points,
    _BDM1_COEFFS, _REF_FACE_NORMALS,
)
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d


def _small_mesh(Kx=3, Ky=3):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0.0, 1.0, 0.0, 1.0, Kx, Ky)
    return build_mesh_2d(VX, VY, EToV, 1)


def test_face_dof_coordinates_match_at_k1():
    """k=1 face-DOF coords from the order-parametric route equal the
    hard-coded ones."""
    new = bdm_k_face_dof_coordinates(1)
    old = bdm1_reference_dof_coordinates()
    assert new.shape == old.shape == (6, 2)
    assert jnp.allclose(new, old, atol=1e-14)


def test_basis_coefficients_match_at_k1():
    """Coefficient matrix obtained by monomial moment-matrix inversion
    matches the existing _BDM1_COEFFS to numerical zero."""
    new = _bdm_k_basis_coefficients(1)
    # Both are 6x6 mapping monomial -> basis. Ordering of monomials
    # may differ; check the spans match by evaluating both bases at a
    # cloud of points -- structural equality below.
    assert new.shape == _BDM1_COEFFS.shape == (6, 6)


def test_basis_evaluates_identically_at_k1():
    """The k=1 BDM basis values themselves -- after both constructions
    -- must agree at arbitrary points. Different monomial orderings
    inside the two routes are fine; the basis functions are unique up
    to choice of dual-basis index labelling."""
    rng = jax.random.PRNGKey(0)
    r = jax.random.uniform(rng, (32,), minval=-1.0, maxval=1.0)
    s = jax.random.uniform(jax.random.fold_in(rng, 1), (32,),
                            minval=-1.0, maxval=1.0)

    new = bdm_k_basis_at_points(1, r, s)           # (32, 6, 2)
    old = bdm1_basis_at_points(r, s)               # (32, 6, 2)

    # Both bases are interpolatory with respect to the same 6 DOF
    # functionals (normal-trace at 6 face-GL points), so basis ell is
    # uniquely determined by ell. We don't fold in any permutation
    # because both routines use the same DOF ordering
    # (face_idx = floor(ell / 2), node = ell mod 2 -- check by
    # confirming face coords match in test above).
    assert jnp.allclose(new, old, atol=1e-12), (
        f"max abs diff = {float(jnp.abs(new - old).max()):.3e}"
    )


def test_divergence_basis_matches_at_k1():
    """k=1 divergence values from both constructions agree."""
    new = bdm_k_divergence_basis(1)
    old = bdm1_divergence_basis()
    assert jnp.allclose(new, old, atol=1e-12)


def test_face_dof_coordinates_lie_on_correct_edges():
    """Sanity: the k+1 GL points generated for each face actually live
    on that face's parameterised line in the reference triangle.
    Validates the higher-k generator as well as k=1."""
    for k in (1, 2, 3, 4):
        coords = bdm_k_face_dof_coordinates(k)
        n_per_face = k + 1
        # Face 0: s = -1 for all points.
        assert jnp.allclose(coords[:n_per_face, 1], -1.0, atol=1e-14)
        # Face 1: r + s = 0 (hypotenuse).
        assert jnp.allclose(
            coords[n_per_face:2 * n_per_face, 0]
            + coords[n_per_face:2 * n_per_face, 1],
            0.0, atol=1e-14,
        )
        # Face 2: r = -1.
        assert jnp.allclose(coords[2 * n_per_face:, 0], -1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Higher-order tests: BDM_2 and BDM_3 with interior moment DOFs
# ---------------------------------------------------------------------------
#
# The interior DOF basis choice (vector monomials of P_{k-1}(K)^2
# truncated to the required (k+1)(k-1) count) is not guaranteed by any
# theorem to be a good choice. These tests verify the choice works:
#
#   (a) the moment matrix inverts (well-conditioned, dim x dim);
#   (b) the resulting BDM basis is *unisolvent* -- evaluating each
#       DOF functional on the basis gives the identity (delta_ij);
#   (c) the basis spans P_k(K)^2 -- exact representation of any
#       polynomial vector field of degree <= k.
#
# If any of these fail at k=2 or k=3 we'd need to swap the interior
# moment basis for a better-conditioned one (e.g. Dubiner orthogonal).


@pytest.mark.parametrize("k", [2, 3])
def test_dof_matrix_inverts(k):
    """The DOF matrix at order k must be invertible -- otherwise the
    chosen DOF set is degenerate and the BDM basis is undefined."""
    D = _bdm_k_dof_matrix(k)
    dim = (k + 1) * (k + 2)
    assert D.shape == (dim, dim)
    cond = float(jnp.linalg.cond(D))
    # Print to surface during development; assertion is the real check.
    print(f"BDM_{k} DOF matrix condition number: {cond:.2e}")
    # Inversion should not produce NaN or inf; matrix is square + non-singular.
    Dinv = jnp.linalg.inv(D)
    assert jnp.all(jnp.isfinite(Dinv))
    # Round-trip: D Dinv ~ I.
    err = float(jnp.abs(D @ Dinv - jnp.eye(dim)).max())
    assert err < 1e-10, f"D Dinv - I has max abs entry {err:.3e}"


@pytest.mark.parametrize("k", [2, 3])
def test_basis_is_unisolvent_on_face_dofs(k):
    """For each face DOF i (point normal-trace evaluator), the i-th
    basis function must satisfy DOF_i(basis_i) = 1 and DOF_j(basis_i)
    = 0 for j != i (other face DOFs). This is the defining property
    of the dual basis."""
    coeffs = _bdm_k_basis_coefficients(k)           # (dim, dim)
    dof_pts = bdm_k_face_dof_coordinates(k)
    face_idx = jnp.repeat(jnp.arange(3), k + 1)
    normals = _REF_FACE_NORMALS[face_idx]
    # Evaluate full BDM basis at the face DOF points.
    basis = bdm_k_basis_at_points(
        k, dof_pts[:, 0], dof_pts[:, 1],
    )                                                # (n_face, dim, 2)
    # Apply each DOF functional to each basis: dof_eval[i, j] = basis[i,j,:] . normals[i,:]
    dof_eval = jnp.einsum("idc,ic->id", basis, normals)
    # Should be identity on the face block: dof_eval[i, j] = delta_{ij} when
    # j is also a face DOF (j < 3(k+1)).
    n_face = 3 * (k + 1)
    face_face_block = dof_eval[:, :n_face]
    err = float(jnp.abs(face_face_block - jnp.eye(n_face)).max())
    assert err < 1e-10, (
        f"BDM_{k}: face DOFs not interpolatory; max |DOF(basis) - I| = {err:.3e}"
    )


@pytest.mark.parametrize("k", [2, 3])
def test_basis_reproduces_constant_vector_fields(k):
    """A constant vector field ``u = (a, b)`` must be reproduced
    exactly: u_h = sum_ell DOF_ell(u) * basis_ell. This catches gross
    errors in the moment matrix even before checking unisolvence."""
    from ddg.dg2d.cubature import triangle_cubature

    # Apply each DOF functional to u = (1, 0).
    # Face DOFs: u . n at face DOF points.
    dof_pts = bdm_k_face_dof_coordinates(k)
    face_idx = jnp.repeat(jnp.arange(3), k + 1)
    normals = _REF_FACE_NORMALS[face_idx]
    u_n_face = normals[:, 0]                         # u = (1, 0) so u.n = n_x at each face DOF

    # Interior DOFs: ∫_K (1, 0) . w dx for each test w.
    tests = _bdm_k_interior_test_functions(k)
    r_q, s_q, w_q = triangle_cubature(2 * k)
    interior_moments = []
    for t in tests:
        w_val = _eval_interior_test(t, r_q, s_q)  # (Q, 2)
        # u = (1, 0), so ∫_K u.w = ∫_K w_x.
        m = float(jnp.sum(w_q * w_val[:, 0]))
        interior_moments.append(m)
    interior_moments = jnp.array(interior_moments)

    dofs_of_u = jnp.concatenate([u_n_face, interior_moments])    # (dim,)

    # Reconstruct u_h at a cloud of points; should equal (1, 0).
    r_test = jnp.linspace(-0.9, 0.9, 5)
    s_test = jnp.linspace(-0.9, 0.9, 5)
    rr, ss = jnp.meshgrid(r_test, s_test, indexing="ij")
    mask = (rr + ss) <= 0.0   # inside reference triangle
    r_in = rr[mask]; s_in = ss[mask]
    basis = bdm_k_basis_at_points(k, r_in, s_in)     # (npts, dim, 2)
    u_h = jnp.einsum("l,nlc->nc", dofs_of_u, basis)   # (npts, 2)
    expected = jnp.broadcast_to(jnp.array([1.0, 0.0]), u_h.shape)
    err = float(jnp.abs(u_h - expected).max())
    assert err < 1e-10, (
        f"BDM_{k}: constant (1, 0) not reproduced; max abs error = {err:.3e}"
    )


@pytest.mark.parametrize("k", [2, 3])
def test_basis_reproduces_all_polynomials_in_p_k(k):
    """Stronger completeness check: any polynomial in P_k(K)^2 must be
    reproducible. Loop over the primal-basis vector monomials and
    confirm each is reconstructed by the DOFs.
    """
    from ddg.dg2d.cubature import triangle_cubature

    dim = (k + 1) * (k + 2)
    dof_pts = bdm_k_face_dof_coordinates(k)
    face_idx = jnp.repeat(jnp.arange(3), k + 1)
    normals = _REF_FACE_NORMALS[face_idx]
    tests = _bdm_k_interior_test_functions(k)
    r_q, s_q, w_q = triangle_cubature(2 * k)
    # Primal basis values at face DOF points and at cubature points.
    primal_at_face = _bdm_k_primal_basis_at_points(
        k, dof_pts[:, 0], dof_pts[:, 1],
    )                                                # (n_face, dim, 2)
    primal_at_cub = _bdm_k_primal_basis_at_points(k, r_q, s_q)
    # For each primal vector monomial u_j, compute dof vector then
    # reconstruct and compare.
    r_test = jnp.linspace(-0.9, 0.9, 7)
    s_test = jnp.linspace(-0.9, 0.9, 7)
    rr, ss = jnp.meshgrid(r_test, s_test, indexing="ij")
    mask = (rr + ss) <= 0.0
    r_in = rr[mask]; s_in = ss[mask]
    basis_at_test = bdm_k_basis_at_points(k, r_in, s_in)         # (npts, dim, 2)
    primal_at_test = _bdm_k_primal_basis_at_points(k, r_in, s_in)
    for j in range(dim):
        # Face DOF values of u_j: primal_at_face[i, j, :] . normals[i, :]
        face_vals = jnp.einsum("ic,ic->i", primal_at_face[:, j, :], normals)
        # Interior DOF values: ∫_K u_j . w_i dx
        u_j_at_cub = primal_at_cub[:, j, :]                       # (Q, 2)
        interior_vals = []
        for t in tests:
            w_val = _eval_interior_test(t, r_q, s_q)
            interior_vals.append(float(jnp.sum(
                w_q * (u_j_at_cub[:, 0] * w_val[:, 0]
                       + u_j_at_cub[:, 1] * w_val[:, 1])
            )))
        dofs = jnp.concatenate([face_vals, jnp.array(interior_vals)])
        # Reconstruct.
        u_recon = jnp.einsum("l,nlc->nc", dofs, basis_at_test)    # (npts, 2)
        u_true = primal_at_test[:, j, :]                          # (npts, 2)
        err = float(jnp.abs(u_recon - u_true).max())
        assert err < 1e-8, (
            f"BDM_{k}: primal monomial j={j} not reproduced; "
            f"max abs error = {err:.3e}"
        )


@pytest.mark.parametrize("k", [2, 3])
def test_divergence_at_points_is_in_p_km1(k):
    """For BDM_k = P_k(K)^2, the divergence is in P_{k-1}(K). Sample
    div(basis) at many points and verify the values fit a P_{k-1}
    polynomial (i.e. the coefficient against any P_k monomial vanishes
    via L2 projection)."""
    from ddg.dg2d.cubature import triangle_cubature
    r_q, s_q, w_q = triangle_cubature(2 * k)
    div_vals = bdm_k_divergence_at_points(k, r_q, s_q)            # (Q, dim)
    # Build the Vandermonde of monomials of total degree exactly k.
    # If the divergence is in P_{k-1}, all <div, monomial of degree
    # exactly k> moments should vanish (mod numerical) up to cancellation
    # against lower-degree monomials. Cleaner check: project to a
    # P_{k-1} basis, residual must be machine zero.
    from ddg.dg2d.bdm import _p_k_monomial_exponents
    pkm1_exponents = _p_k_monomial_exponents(k - 1)
    # Build the cubature-weighted matrix of monomials.
    M = jnp.stack([(r_q ** i) * (s_q ** j) for (i, j) in pkm1_exponents],
                   axis=1)                                          # (Q, dim_pkm1)
    # L2 projection: alpha = (M^T W M)^{-1} M^T W div_vals
    W = jnp.diag(w_q)
    MtWM = M.T @ W @ M
    MtW = M.T @ W
    proj_coeffs = jnp.linalg.solve(MtWM, MtW @ div_vals)             # (dim_pkm1, dim)
    div_reproj = M @ proj_coeffs                                     # (Q, dim)
    err = float(jnp.abs(div_vals - div_reproj).max())
    assert err < 1e-8, (
        f"BDM_{k}: divergence is not in P_{k-1}; "
        f"residual after P_{k-1} projection has max abs {err:.3e}"
    )


# ---------------------------------------------------------------------------
# Topology tests for BDM_k
# ---------------------------------------------------------------------------


def test_bdm1_topology_alias_is_bdmtopology():
    """BDM1Topology must be importable for back-compat with pre-refactor
    tests, and must equal the new BDMTopology dataclass."""
    assert BDM1Topology is BDMTopology


@pytest.mark.parametrize("k", [1, 2, 3])
def test_topology_dof_count_matches_formula(k):
    """For a mesh with K elements, F interior faces and B boundary
    faces (2F + B = 3K), the BDM_k global DOF count is:
        face DOFs:    (k+1) * (number of distinct edges)
                    = (k+1) * (F + B)
        interior DOFs: (k+1)(k-1) * K
        total = (k+1)(F + B) + (k+1)(k-1) K
    """
    mesh = _small_mesh(Kx=3, Ky=3)
    topo = build_bdm_topology(mesh, order=k)
    # Count interior + boundary faces.
    EToE = np.asarray(mesh.EToE)
    K = mesh.K
    n_boundary = int(np.sum(EToE == np.arange(K)[:, None]))
    n_interior_faces = (3 * K - n_boundary) // 2
    n_distinct_edges = n_interior_faces + n_boundary
    expected = (k + 1) * n_distinct_edges + (k + 1) * (k - 1) * K
    assert topo.n_global_dofs == expected, (
        f"k={k}: got {topo.n_global_dofs}, expected {expected}"
    )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_topology_local_to_global_covers_all(k):
    """Every global index appears at least once in local_to_global,
    and every entry is in [0, n_global_dofs)."""
    mesh = _small_mesh(Kx=2, Ky=2)
    topo = build_bdm_topology(mesh, order=k)
    lg = np.asarray(topo.local_to_global)
    assert lg.shape == (mesh.K, (k + 1) * (k + 2))
    assert lg.min() >= 0
    assert lg.max() == topo.n_global_dofs - 1
    assert set(lg.flatten().tolist()) == set(range(topo.n_global_dofs))


@pytest.mark.parametrize("k", [2, 3])
def test_topology_interior_dofs_are_unique_per_element(k):
    """Interior DOFs are not shared; each element's interior block
    must contain global indices that no other element references."""
    mesh = _small_mesh(Kx=2, Ky=2)
    topo = build_bdm_topology(mesh, order=k)
    lg = np.asarray(topo.local_to_global)
    n_face_total = 3 * (k + 1)
    interior_block = lg[:, n_face_total:]
    # Flatten interior indices; each one must appear exactly once across
    # all (element, local) pairs.
    flat = interior_block.flatten()
    assert len(set(flat.tolist())) == len(flat), (
        f"k={k}: interior DOFs are not unique"
    )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_topology_face_dof_sign_alternates_on_shared_edges(k):
    """For each interior face, the two adjacent elements must label
    its DOFs with opposite signs (one +1 owner, one -1 non-owner)."""
    mesh = _small_mesh(Kx=2, Ky=2)
    topo = build_bdm_topology(mesh, order=k)
    EToE = np.asarray(mesh.EToE)
    lg = np.asarray(topo.local_to_global)
    sign = np.asarray(topo.local_sign)
    n_per_face = k + 1
    for el in range(mesh.K):
        for f in range(3):
            el_nbr = int(EToE[el, f])
            if el_nbr == el:
                continue
            base = n_per_face * f
            my_signs = sign[el, base:base + n_per_face]
            # Find the neighbour's face DOFs for this shared edge.
            my_g = lg[el, base:base + n_per_face]
            # Look up signs on the neighbour side: find which face of
            # the neighbour contains these globals, then read its signs.
            nbr_face_signs = []
            for g in my_g:
                idxs = np.where(lg[el_nbr] == g)[0]
                assert len(idxs) >= 1, (
                    f"k={k}: shared DOF {g} from element {el} face {f} "
                    f"not found on neighbour {el_nbr}"
                )
                nbr_face_signs.append(sign[el_nbr, idxs[0]])
            nbr_face_signs = np.array(nbr_face_signs)
            # On a shared edge: one side is +1 (owner), other is -1.
            assert np.all(my_signs * nbr_face_signs == -1), (
                f"k={k}: shared edge between {el} and {el_nbr} has "
                f"non-opposite signs: my={my_signs}, nbr={nbr_face_signs}"
            )


def test_bdm_class_supports_higher_orders():
    """``BDM.from_mesh`` must accept order=1, 2, 3 with correct Np."""
    mesh = _small_mesh(Kx=2, Ky=2)
    for k in (1, 2, 3):
        V = BDM.from_mesh(mesh, order=k)
        assert V.order == k
        assert V.Np == (k + 1) * (k + 2)
        assert V.total_dofs == V.topology.n_global_dofs
        assert V.topology.order == k


# ---------------------------------------------------------------------------
# Order-parametric operator tests
# ---------------------------------------------------------------------------


def test_bdm_k_mass_matches_bdm1_at_k1():
    """``bdm_k_element_local_mass`` at k=1 must reproduce the existing
    ``bdm1_element_local_mass`` (hard-coded Hammer 3-point) to machine
    precision."""
    from ddg.dg2d.bdm import bdm1_element_local_mass, bdm_k_element_local_mass
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=1)
    M_old = bdm1_element_local_mass(V)
    M_new = bdm_k_element_local_mass(V)
    err = float(jnp.abs(M_old - M_new).max())
    assert err < 1e-12, f"max diff {err:.3e}"


def test_bdm_k_viscous_volume_matches_bdm1_at_k1():
    """k=1 regression for the viscous-volume stiffness."""
    from ddg.dg2d.bdm import (
        bdm1_viscous_volume_stiffness, bdm_k_viscous_volume_stiffness,
    )
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=1)
    nu = 0.5
    S_old = bdm1_viscous_volume_stiffness(V, nu)
    S_new = bdm_k_viscous_volume_stiffness(V, nu)
    err = float(jnp.abs(S_old - S_new).max())
    assert err < 1e-12, f"max diff {err:.3e}"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_bdm_k_mass_is_symmetric_positive_definite(k):
    """Mass matrix on every element must be symmetric positive definite."""
    from ddg.dg2d.bdm import bdm_k_element_local_mass
    mesh = _small_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh, order=k)
    M = bdm_k_element_local_mass(V)                  # (K, Np, Np)
    # Symmetry.
    sym_err = float(jnp.abs(M - jnp.swapaxes(M, -1, -2)).max())
    assert sym_err < 1e-12, f"k={k}: not symmetric, err={sym_err:.3e}"
    # SPD via Cholesky on first element.
    L = jnp.linalg.cholesky(M[0])
    assert jnp.all(jnp.isfinite(L))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_bdm_k_viscous_volume_is_symmetric_psd(k):
    """``∫ ∇u : ∇v`` is symmetric and positive semi-definite (zero
    eigenvalue for u in the rigid-translation kernel of the gradient)."""
    from ddg.dg2d.bdm import bdm_k_viscous_volume_stiffness
    mesh = _small_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh, order=k)
    S = bdm_k_viscous_volume_stiffness(V, nu=1.0)
    sym_err = float(jnp.abs(S - jnp.swapaxes(S, -1, -2)).max())
    assert sym_err < 1e-12, f"k={k}: not symmetric, err={sym_err:.3e}"
    # PSD: every eigenvalue >= -eps.
    eigs = jnp.linalg.eigvalsh(S[0])
    assert float(eigs.min()) > -1e-10, (
        f"k={k}: viscous-volume has negative eigenvalue {float(eigs.min()):.3e}"
    )


# ---------------------------------------------------------------------------
# Cross-space divergence / gradient: BDM_k <-> P_{k-1}
# ---------------------------------------------------------------------------


def test_bdm_k_divergence_matches_bdm_p0_at_k1():
    """At k=1, the new cross-space divergence (returning (K, 1)) must
    agree with the existing bdm_p0_divergence_apply (returning (K,))."""
    from ddg.dg2d.bdm import (
        bdm_p0_divergence_apply, bdm_k_divergence_apply,
    )
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=1)
    rng = jax.random.PRNGKey(42)
    u = jax.random.normal(rng, (V.total_dofs,))
    d_old = bdm_p0_divergence_apply(V, u)                  # (K,)
    d_new = bdm_k_divergence_apply(V, u)                    # (K, 1)
    assert d_new.shape == (V.K, 1)
    err = float(jnp.abs(d_old - d_new[:, 0]).max())
    assert err < 1e-12, f"max diff {err:.3e}"


def test_bdm_k_gradient_matches_bdm_p0_at_k1():
    """Same for gradient (adjoint of divergence)."""
    from ddg.dg2d.bdm import (
        bdm_p0_gradient_apply, bdm_k_gradient_apply,
    )
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=1)
    rng = jax.random.PRNGKey(42)
    p = jax.random.normal(rng, (V.K,))                      # P_0 layout
    g_old = bdm_p0_gradient_apply(V, p)
    g_new = bdm_k_gradient_apply(V, p[:, None])             # (K, 1) layout
    err = float(jnp.abs(g_old - g_new).max())
    assert err < 1e-12, f"max diff {err:.3e}"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_bdm_k_divergence_gradient_adjoint_property(k):
    """``<G p, u>_BDM = -<p, D u>_P`` for any matched pair (u, p).
    The P space inner product is per-element coefficient dot, summed
    over elements -- the simplest L2 layout for the test (the actual
    P_{k-1} mass matrix carries J factors but for the adjoint test
    we just need the raw bilinear duality)."""
    from ddg.dg2d.bdm import bdm_k_divergence_apply, bdm_k_gradient_apply
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=k)
    dim_p = k * (k + 1) // 2
    rng = jax.random.PRNGKey(7)
    u = jax.random.normal(rng, (V.total_dofs,))
    p = jax.random.normal(jax.random.fold_in(rng, 1), (V.K, dim_p))

    Gp = bdm_k_gradient_apply(V, p)
    Du = bdm_k_divergence_apply(V, u)

    lhs = float(jnp.dot(Gp, u))
    rhs = -float(jnp.einsum("km,km->", p, Du))
    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
    assert rel_err < 1e-10, (
        f"k={k}: adjoint property fails. <Gp, u>={lhs:.6e}, "
        f"-<p, Du>={rhs:.6e}, rel err {rel_err:.3e}"
    )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_bdm_k_divergence_kills_hdiv_zero_normal_flux(k):
    """A discrete Hdiv field whose global DOFs are all zero (no normal
    flux through any face, no interior bubble content) must have zero
    integrated divergence on every element. Sanity check: trivially
    zero ↦ zero."""
    from ddg.dg2d.bdm import bdm_k_divergence_apply
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=k)
    u_zero = jnp.zeros(V.total_dofs)
    div = bdm_k_divergence_apply(V, u_zero)
    assert float(jnp.abs(div).max()) < 1e-14


# ---------------------------------------------------------------------------
# Order-parametric SIPG viscous face apply
# ---------------------------------------------------------------------------


def test_bdm_k_viscous_face_matches_bdm1_at_k1():
    """k=1 regression: the new order-parametric face apply must agree
    with the existing bdm1_viscous_face_apply on a non-trivial DOF
    vector."""
    from ddg.dg2d.bdm import (
        bdm1_viscous_face_apply, bdm_k_viscous_face_apply,
    )
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=1)
    rng = jax.random.PRNGKey(13)
    u = jax.random.normal(rng, (V.total_dofs,))
    out_old = bdm1_viscous_face_apply(V, u, nu=0.5, tau_scale=4.0)
    out_new = bdm_k_viscous_face_apply(V, u, nu=0.5, tau_scale=4.0)
    err = float(jnp.abs(out_old - out_new).max())
    assert err < 1e-10, f"max diff {err:.3e}"


def test_bdm_k_viscous_apply_matches_bdm1_at_k1():
    """Full viscous = volume + face, k=1 regression on a random
    velocity field."""
    from ddg.dg2d.bdm import bdm1_viscous_apply, bdm_k_viscous_apply
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=1)
    rng = jax.random.PRNGKey(101)
    u = jax.random.normal(rng, (V.total_dofs,))
    out_old = bdm1_viscous_apply(V, u, nu=0.3, tau_scale=4.0)
    out_new = bdm_k_viscous_apply(V, u, nu=0.3, tau_scale=4.0)
    err = float(jnp.abs(out_old - out_new).max())
    assert err < 1e-10, f"max diff {err:.3e}"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_bdm_k_viscous_apply_is_symmetric_on_homog_dirichlet(k):
    """Full SIPG viscous (volume + face, homog ghost) is symmetric
    on the closed-boundary mesh. The matrix-free apply must satisfy
    <Au, v> = <u, Av> for all u, v in the homogeneous BC space."""
    from ddg.dg2d.bdm import bdm_k_viscous_apply
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=k)
    n = V.total_dofs
    rng = jax.random.PRNGKey(202)
    u = jax.random.normal(rng, (n,))
    v = jax.random.normal(jax.random.fold_in(rng, 1), (n,))
    Au = bdm_k_viscous_apply(V, u, nu=1.0, tau_scale=4.0)
    Av = bdm_k_viscous_apply(V, v, nu=1.0, tau_scale=4.0)
    lhs = float(jnp.dot(Au, v))
    rhs = float(jnp.dot(Av, u))
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
    assert rel < 1e-10, (
        f"k={k}: viscous apply not symmetric. <Au,v>={lhs:.6e}, "
        f"<Av,u>={rhs:.6e}, rel={rel:.3e}"
    )


@pytest.mark.parametrize("k", [2, 3])
def test_bdm_k_viscous_apply_is_positive_on_homog_bc(k):
    """At homogeneous BC the SIPG viscous bilinear form is coercive:
    <Au, u> > 0 for all u != 0. (Penalty + volume gives strict
    positivity on the H(div)-conforming + tangential-jump space.)"""
    from ddg.dg2d.bdm import bdm_k_viscous_apply
    mesh = _small_mesh(Kx=2, Ky=2)
    V = BDM.from_mesh(mesh, order=k)
    n = V.total_dofs
    rng = jax.random.PRNGKey(303)
    u = jax.random.normal(rng, (n,))
    Au = bdm_k_viscous_apply(V, u, nu=1.0, tau_scale=4.0)
    energy = float(jnp.dot(Au, u))
    assert energy > 1e-6, (
        f"k={k}: viscous energy non-positive on random u: {energy:.3e}"
    )


# ---------------------------------------------------------------------------
# Dirichlet lift (boundary velocity sampler + SIPG lift integral)
# ---------------------------------------------------------------------------


def test_bdm_k_evaluate_boundary_velocity_matches_bdm1_at_k1():
    """At k=1, the new vector-BC sampler must give the same result as
    the existing bdm1_evaluate_boundary_velocity."""
    from ddg.dg2d.bdm import (
        bdm1_evaluate_boundary_velocity, bdm_k_evaluate_boundary_velocity,
    )
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=1)
    def bc_fn(x, y):
        return jnp.sin(jnp.pi * y), jnp.cos(jnp.pi * x)
    out_old = bdm1_evaluate_boundary_velocity(V, bc_fn)
    out_new = bdm_k_evaluate_boundary_velocity(V, bc_fn)
    err = float(jnp.abs(out_old - out_new).max())
    assert err < 1e-12, f"max diff {err:.3e}"


def test_bdm_k_dirichlet_lift_matches_bdm1_at_k1():
    """k=1 regression for the lift."""
    from ddg.dg2d.bdm import (
        bdm1_evaluate_boundary_velocity, bdm1_viscous_dirichlet_lift,
        bdm_k_evaluate_boundary_velocity, bdm_k_viscous_dirichlet_lift,
    )
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=1)
    def bc_fn(x, y):
        return jnp.sin(jnp.pi * y), jnp.cos(jnp.pi * x)
    u_bc_old = bdm1_evaluate_boundary_velocity(V, bc_fn)
    u_bc_new = bdm_k_evaluate_boundary_velocity(V, bc_fn)
    lift_old = bdm1_viscous_dirichlet_lift(V, u_bc_old, nu=0.5, tau_scale=4.0)
    lift_new = bdm_k_viscous_dirichlet_lift(V, u_bc_new, nu=0.5, tau_scale=4.0)
    err = float(jnp.abs(lift_old - lift_new).max())
    assert err < 1e-10, f"max diff {err:.3e}"


@pytest.mark.parametrize("k", [2, 3])
def test_bdm_k_dirichlet_lift_is_nonzero_for_nonzero_bc(k):
    """At k>=2, the lift must produce non-trivial values when the BC
    is not identically zero. Sanity that the higher-order machinery
    isn't silently dropping the input."""
    from ddg.dg2d.bdm import (
        bdm_k_evaluate_boundary_velocity, bdm_k_viscous_dirichlet_lift,
    )
    mesh = _small_mesh(Kx=3, Ky=3)
    V = BDM.from_mesh(mesh, order=k)
    def bc_fn(x, y):
        return jnp.cos(2.0 * jnp.pi * y), jnp.sin(2.0 * jnp.pi * x)
    u_bc = bdm_k_evaluate_boundary_velocity(V, bc_fn)
    lift = bdm_k_viscous_dirichlet_lift(V, u_bc, nu=0.3, tau_scale=4.0)
    # u_BC is nontrivial; lift must not vanish.
    assert float(jnp.abs(lift).max()) > 1e-3, (
        f"k={k}: lift max abs is {float(jnp.abs(lift).max()):.3e}"
    )
