"""Tests for the BDM velocity hp-multigrid (transfers + W-cycle).

* p- and h-transfers are exact polynomial inclusions -> function-preserving
  to ~1e-12.
* the hp W-cycle is a mesh-independent solver for the viscous block: the
  contraction factor stays roughly constant as the mesh refines.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from ddg.dg2d.bdm import BDM, bdm_k_viscous_apply
from ddg.dg2d.mesh import build_mesh_2d, uniform_rectangle_mesh_2d
from ddg.dg2d.refinement import RefinementForest
from ddg.dg2d.bdm_multigrid import (
    make_p_prolong, make_h_prolong, build_velocity_hpmg,
)


def _interp(V, u_fn):
    """Canonical interpolant (face normal moments + interior moments)."""
    from ddg.dg2d.bdm import (
        bdm_k_face_dof_coordinates, _bdm_k_face_quadrature_data,
        _bdm_k_interior_test_functions, _eval_interior_test,
        bdm_k_basis_at_points, _element_jacobian_matrices,
    )
    from ddg.dg2d.cubature import triangle_cubature
    k = V.order; K = V.K; Q = k + 1
    n_face_total = 3 * Q; n_interior = (k + 1) * (k - 1)
    fdata = _bdm_k_face_quadrature_data(V); n_face = fdata['n_face']
    coords = bdm_k_face_dof_coordinates(k)
    r = np.asarray(coords[:, 0]); s = np.asarray(coords[:, 1])
    EToV = np.asarray(V.mesh.EToV); VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    Vx0, Vy0 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    Vx1, Vy1 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    Vx2, Vy2 = VX[EToV[:, 2]], VY[EToV[:, 2]]
    x_at = Vx0[:, None] + 0.5*(r+1)[None]*(Vx1-Vx0)[:, None] + 0.5*(s+1)[None]*(Vx2-Vx0)[:, None]
    y_at = Vy0[:, None] + 0.5*(r+1)[None]*(Vy1-Vy0)[:, None] + 0.5*(s+1)[None]*(Vy2-Vy0)[:, None]
    xf = jnp.asarray(x_at).reshape(K, 3, Q); yf = jnp.asarray(y_at).reshape(K, 3, Q)
    ux, uy = u_fn(xf, yf)
    un = ux*n_face[:, :, 0:1] + uy*n_face[:, :, 1:2]
    u_face = un.reshape(K, n_face_total)
    u_int = jnp.zeros((K, n_interior))
    if n_interior > 0:
        tests = _bdm_k_interior_test_functions(k)
        r_q, s_q, w_q = triangle_cubature(2*k+2)
        DF, _ = _element_jacobian_matrices(V.mesh); DF_inv = jnp.linalg.inv(DF); J = V.mesh.J[0]
        rqn = np.asarray(r_q); sqn = np.asarray(s_q)
        xq = Vx0[:, None] + 0.5*(rqn+1)[None]*(Vx1-Vx0)[:, None] + 0.5*(sqn+1)[None]*(Vx2-Vx0)[:, None]
        yq = Vy0[:, None] + 0.5*(rqn+1)[None]*(Vy1-Vy0)[:, None] + 0.5*(sqn+1)[None]*(Vy2-Vy0)[:, None]
        uxq, uyq = u_fn(jnp.asarray(xq), jnp.asarray(yq))
        u_ref = jnp.einsum('k,kab,kqb->kqa', J, DF_inv, jnp.stack([uxq, uyq], -1))
        for i, wfn in enumerate(tests):
            wv = _eval_interior_test(wfn, r_q, s_q)
            u_int = u_int.at[:, i].set(jnp.einsum('q,kqc,qc->k', w_q, u_ref, wv))
    u_local = jnp.concatenate([u_face, u_int], axis=1)
    owner = V.topology.local_sign > 0; lg = V.topology.local_to_global
    return jnp.zeros(V.total_dofs).at[lg.reshape(-1)].add(
        jnp.where(owner, u_local, 0).reshape(-1))


def _eval_pts(V, u_global, pts):
    from ddg.dg2d.bdm import bdm_k_basis_at_points, bdm1_gather, _element_jacobian_matrices
    k = V.order
    EToV = np.asarray(V.mesh.EToV); VX = np.asarray(V.mesh.VX); VY = np.asarray(V.mesh.VY)
    ul = bdm1_gather(V.topology, u_global); DF, _ = _element_jacobian_matrices(V.mesh); J = V.mesh.J[0]
    out = np.zeros((len(pts), 2))
    for qi, (xq, yq) in enumerate(pts):
        for e in range(V.K):
            v0=(VX[EToV[e,0]],VY[EToV[e,0]]); v1=(VX[EToV[e,1]],VY[EToV[e,1]]); v2=(VX[EToV[e,2]],VY[EToV[e,2]])
            det=(v1[0]-v0[0])*(v2[1]-v0[1])-(v2[0]-v0[0])*(v1[1]-v0[1])
            a=((v1[0]-xq)*(v2[1]-yq)-(v2[0]-xq)*(v1[1]-yq))/det
            b=((v2[0]-xq)*(v0[1]-yq)-(v0[0]-xq)*(v2[1]-yq))/det
            c=1-a-b
            if a>=-1e-10 and b>=-1e-10 and c>=-1e-10:
                rr=2*b-1; ss=2*c-1
                phi=bdm_k_basis_at_points(k, jnp.array([rr]), jnp.array([ss]))
                pp=(DF[e]@phi[0].T).T/J[e]
                out[qi]=np.asarray(jnp.einsum("l,lc->c", ul[e], pp)); break
    return out


def _u_fn(x, y):
    return (jnp.sin(1.3*x)*jnp.cos(0.7*y), -jnp.cos(1.3*x)*jnp.sin(0.7*y))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_p_transfer_exact(k):
    if k == 1:
        pytest.skip("p-transfer needs a coarser space")
    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, 3, 3)
    mesh = build_mesh_2d(VX, VY, EToV, 3)
    Vc = BDM.from_mesh(mesh, order=k - 1); Vf = BDM.from_mesh(mesh, order=k)
    c = _interp(Vc, _u_fn)
    f = make_p_prolong(Vc, Vf)(c)
    rng = np.random.default_rng(0); pts = rng.uniform(0.05, 0.95, (30, 2))
    err = np.max(np.abs(_eval_pts(Vc, c, pts) - _eval_pts(Vf, f, pts)))
    assert err < 1e-10, f"p-transfer not function-preserving: {err:.2e}"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_h_transfer_exact(k):
    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, 2, 2)
    base = build_mesh_2d(VX, VY, EToV, 3)
    forest = RefinementForest(base); m0, _ = forest.active_mesh()
    forest.refine(forest.leaves()); m1, _ = forest.active_mesh()
    Vc = BDM.from_mesh(m0, order=k); Vf = BDM.from_mesh(m1, order=k)
    c = _interp(Vc, _u_fn)
    f = make_h_prolong(Vc, Vf)(c)
    rng = np.random.default_rng(1); pts = rng.uniform(0.05, 0.95, (30, 2))
    err = np.max(np.abs(_eval_pts(Vc, c, pts) - _eval_pts(Vf, f, pts)))
    assert err < 1e-10, f"h-transfer not function-preserving: {err:.2e}"


def test_coupled_hpmg_cavity_stokes_step():
    """The coupled block-diagonal PC (velocity hp-MG W-cycle + pressure-
    mass Schur) lets matrix-free GMRES solve the cavity Stokes step
    (first Newton iterate from u=0) and match dense LU. This is the
    payoff of the hp-MG groundwork: a matrix-free coupled solve."""
    import sys, os
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "applications"))
    import ns_bdm_cavity_amr_2d as app
    from ddg.dg2d.bdm import bdm_k_viscous_dirichlet_lift
    from ddg.dg2d.coupled_ns_bdm import ns_picard_apply_bdm_k_2d
    from ddg.dg2d.bdm_multigrid import make_coupled_velocity_mg_pc
    NU, TAU = 1.0 / 100.0, 4.0

    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, 2, 2)
    base = build_mesh_2d(VX, VY, EToV, 3)
    forest = RefinementForest(base)
    forest.refine(forest.leaves())                       # eff 4x4
    mesh, _ = forest.active_mesh()
    V, u_bc, u_bc_vec, bc_mask = app._build_space(mesh, 3)
    dim_p = 6; n_u, n_p = V.total_dofs, V.K * dim_p; n = n_u + n_p

    lift = bdm_k_viscous_dirichlet_lift(V, u_bc_vec, nu=NU, tau_scale=TAU)
    def raw_flat(x):
        u = x[:n_u]; p = x[n_u:].reshape(V.K, dim_p)
        Fu, Fp = ns_picard_apply_bdm_k_2d(V, u, p, u_star_global=u, nu=NU,
                                          pressure_eps=1e-10, tau_scale=TAU)
        return jnp.concatenate([Fu, Fp.reshape(-1)])
    x0 = jnp.concatenate([u_bc, jnp.zeros(n_p)])         # cavity: u_bc = 0
    raw = raw_flat(x0)
    Fu_inner = jnp.where(bc_mask, 0.0, raw[:n_u] - lift)
    rhs = -jnp.concatenate([Fu_inner, raw[n_u:]])
    def matvec(v):
        vu = v[:n_u]; vp = v[n_u:]
        vu_z = jnp.where(bc_mask, 0.0, vu)
        _, jv = jax.jvp(raw_flat, (x0,), (jnp.concatenate([vu_z, vp]),))
        return jnp.concatenate([jnp.where(bc_mask, vu, jv[:n_u]), jv[n_u:]])

    # Dense reference.
    mask_full = jnp.concatenate([bc_mask, jnp.zeros(n_p, dtype=bool)])
    J = jax.jacfwd(raw_flat)(x0)
    J_con = jnp.where(mask_full[:, None], jnp.eye(n),
                      jnp.where(mask_full[None, :], 0.0, J))
    d_dense = jnp.linalg.solve(J_con, rhs)

    # Coupled MG-GMRES (n_h=1 so the hp hierarchy matches the cavity mesh).
    pc = make_coupled_velocity_mg_pc(V, base, n_h=1, nu=NU, nsm=2, cycle="W")
    d_iter, _ = jax.scipy.sparse.linalg.gmres(
        matvec, rhs, x0=jnp.zeros_like(rhs), M=pc,
        tol=1e-8, atol=1e-12, restart=80, maxiter=10)

    # Compare velocity (pressure has a constant-pressure near-nullspace,
    # so its absolute components are ambiguous; the velocity is unique).
    rel_err_u = float(jnp.linalg.norm(d_iter[:n_u] - d_dense[:n_u])
                      / jnp.linalg.norm(d_dense[:n_u]))
    rel_resid = float(jnp.linalg.norm(matvec(d_iter) - rhs)
                      / jnp.linalg.norm(rhs))
    assert rel_resid < 1e-6, f"GMRES failed to converge: rel-resid={rel_resid:.2e}"
    assert rel_err_u < 1e-3, f"velocity mismatch vs dense: {rel_err_u:.2e}"


def test_sparse_assembly_matches_dense_jacobian():
    """The coloring-based sparse viscous assembly matches dense jacobian.
    This is the building block for the scalable patch smoother."""
    from ddg.dg2d.bdm_multigrid import assemble_sparse_viscous
    from ddg.dg2d.bdm import bdm_k_viscous_apply
    NU = 1.0 / 100.0
    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, 4, 4)
    mesh = build_mesh_2d(VX, VY, EToV, 3)
    V = BDM.from_mesh(mesh, order=3)
    A_sparse = assemble_sparse_viscous(V, nu=NU, tau_scale=4.0)
    A_dense = np.asarray(jax.jacobian(
        lambda u: bdm_k_viscous_apply(V, u, nu=NU, tau_scale=4.0)
    )(jnp.zeros(V.total_dofs)))
    err = float(np.linalg.norm(A_sparse.toarray() - A_dense)
                / np.linalg.norm(A_dense))
    assert err < 1e-10, f"sparse vs dense: {err:.2e}"


def test_sparse_wcycle_matches_dense_wcycle():
    """The hp-MG W-cycle gives matching contraction with sparse_fine=True
    vs sparse_fine=False -- the algorithm is unchanged, only the
    finest-level assembly differs."""
    rng = np.random.default_rng(0)
    VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, 2, 2)
    base = build_mesh_2d(VX, VY, EToV, 3)
    V_d, A_d, mg_d = build_velocity_hpmg(
        base, n_h=1, order=3, nu=0.01, nsm=2, cycle="W", sparse_fine=False)
    V_s, A_s, mg_s = build_velocity_hpmg(
        base, n_h=1, order=3, nu=0.01, nsm=2, cycle="W", sparse_fine=True)
    n = V_d.total_dofs
    b = jnp.asarray(rng.standard_normal(n))
    nb = float(jnp.linalg.norm(b))
    # Same b: apply one V-cycle, compare results.
    x_d = mg_d(b); x_s = mg_s(b)
    rel = float(jnp.linalg.norm(x_d - x_s) / jnp.linalg.norm(x_d))
    assert rel < 1e-6, f"sparse W-cycle deviates from dense: {rel:.2e}"


def test_hpmg_mesh_independent():
    """W-cycle contraction stays bounded (< 0.4) and roughly flat as the
    mesh refines -- the defining property of a working multigrid."""
    rng = np.random.default_rng(0)
    rates = []
    for n_h in (1, 2):
        VX, VY, EToV = uniform_rectangle_mesh_2d(0, 1, 0, 1, 2, 2)
        base = build_mesh_2d(VX, VY, EToV, 3)
        V3, A, mg = build_velocity_hpmg(base, n_h, order=3, nu=0.01,
                                        nsm=3, cycle="W")
        n = V3.total_dofs; b = jnp.asarray(rng.standard_normal(n))
        nb = float(jnp.linalg.norm(b)); x = jnp.zeros(n); prev = nb; rr_list = []
        for _ in range(20):
            x = x + mg(b - A(x)); rr = float(jnp.linalg.norm(A(x) - b))
            rr_list.append(rr / prev); prev = rr
            if rr / nb < 1e-9:
                break
        rate = float(np.mean(rr_list[-3:]))
        rates.append(rate)
        assert rate < 0.4, f"n_h={n_h}: W-cycle rate {rate:.3f} too large"
    # mesh independence: the rate should not blow up with refinement.
    assert rates[-1] < 2.0 * rates[0] + 0.1, f"rates not flat: {rates}"
