"""Reference-element operators for 2D nodal DG on the triangle.

Direct ports of ``Codes1.1/Codes2D/{Nodes2D, Warpfactor, xytors, rstoab,
Simplex2DP, GradSimplex2DP, Vandermonde2D, GradVandermonde2D,
Dmatrices2D, Lift2D}.m``. The basis on the simplex is the orthonormal
Koornwinder/Dubiner family ``P_i^{(0,0)}(a) P_j^{(2i+1, 0)}(b)
(1-b)^i * sqrt(2)``, evaluated via the singular coordinate map
``(r, s) -> (a, b)``.

Standard reference triangle (matches Hesthaven):
    vertices V1 = (-1, -1), V2 = (1, -1), V3 = (-1, 1)
    face 0: s = -1   (V1 -> V2)
    face 1: r + s = 0 (V2 -> V3)
    face 2: r = -1   (V3 -> V1)
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from ddg.dg1d.reference import jacobi_p, vandermonde_1d


_ALPHA_OPT = np.array([
    0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
    1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258,
])


def _alpha_for(N: int) -> float:
    if 1 <= N <= 15:
        return float(_ALPHA_OPT[N - 1])
    return 5.0 / 3.0


def _warpfactor(N: int, r_out: jnp.ndarray) -> jnp.ndarray:
    """Scaled warp function used by Nodes2D to push equispaced points to LGL."""
    from ddg.dg1d.reference import jacobi_gauss_lobatto_nodes

    r_out = jnp.asarray(r_out)
    LGL_r = jacobi_gauss_lobatto_nodes(0.0, 0.0, N)
    req = jnp.linspace(-1.0, 1.0, N + 1)
    Veq = vandermonde_1d(N, req)

    Nr = r_out.shape[0]
    Pmat = jnp.stack(
        [jacobi_p(r_out, 0.0, 0.0, i) for i in range(N + 1)],
        axis=0,
    )  # (N+1, Nr)
    Lmat = jnp.linalg.solve(Veq.T, Pmat)
    warp = Lmat.T @ (LGL_r - req)

    # Scale: avoid the singularity at r_out = ±1.
    zerof = jnp.abs(r_out) < 1.0 - 1e-10
    sf = 1.0 - jnp.where(zerof, r_out, 0.0) ** 2
    return warp / sf + warp * (zerof.astype(warp.dtype) - 1.0)


def equilateral_nodes_2d(N: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Warp-and-blend nodes on the equilateral triangle (Hesthaven Nodes2D)."""
    Np = (N + 1) * (N + 2) // 2
    alpha = _alpha_for(N)

    # Barycentric equidistributed nodes.
    L1 = np.zeros(Np)
    L3 = np.zeros(Np)
    sk = 0
    for n in range(1, N + 2):
        for m in range(1, N + 3 - n):
            L1[sk] = (n - 1) / N if N > 0 else 0.0
            L3[sk] = (m - 1) / N if N > 0 else 0.0
            sk += 1
    L2 = 1.0 - L1 - L3
    L1 = jnp.asarray(L1)
    L2 = jnp.asarray(L2)
    L3 = jnp.asarray(L3)

    x = -L2 + L3
    y = (-L2 - L3 + 2.0 * L1) / math.sqrt(3.0)

    if N == 0:
        return x, y

    blend1 = 4.0 * L2 * L3
    blend2 = 4.0 * L1 * L3
    blend3 = 4.0 * L1 * L2

    warpf1 = _warpfactor(N, L3 - L2)
    warpf2 = _warpfactor(N, L1 - L3)
    warpf3 = _warpfactor(N, L2 - L1)

    warp1 = blend1 * warpf1 * (1.0 + (alpha * L1) ** 2)
    warp2 = blend2 * warpf2 * (1.0 + (alpha * L2) ** 2)
    warp3 = blend3 * warpf3 * (1.0 + (alpha * L3) ** 2)

    x = x + warp1 + math.cos(2 * math.pi / 3) * warp2 + math.cos(4 * math.pi / 3) * warp3
    y = y + math.sin(2 * math.pi / 3) * warp2 + math.sin(4 * math.pi / 3) * warp3
    return x, y


def xy_to_rs(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Equilateral -> standard reference triangle."""
    L1 = (math.sqrt(3.0) * y + 1.0) / 3.0
    L2 = (-3.0 * x - math.sqrt(3.0) * y + 2.0) / 6.0
    L3 = (3.0 * x - math.sqrt(3.0) * y + 2.0) / 6.0
    r = -L2 + L3 - L1
    s = -L2 - L3 + L1
    return r, s


def rs_to_ab(r: jnp.ndarray, s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Singular map from the simplex to a square ``(a, b) in [-1, 1]^2``.

    Used to factor the Koornwinder basis into a product of Jacobi
    polynomials. At the apex ``s = 1`` we collapse ``a = -1``.
    """
    a = jnp.where(jnp.abs(s - 1.0) > 1e-12, 2.0 * (1.0 + r) / (1.0 - s) - 1.0, -1.0)
    return a, s


def simplex_2d_p(
    a: jnp.ndarray, b: jnp.ndarray, i: int, j: int
) -> jnp.ndarray:
    """Orthonormal Koornwinder polynomial of order ``(i, j)`` at ``(a, b)``."""
    h1 = jacobi_p(a, 0.0, 0.0, i)
    h2 = jacobi_p(b, 2 * i + 1.0, 0.0, j)
    factor = (1.0 - b) ** i if i > 0 else 1.0
    return math.sqrt(2.0) * h1 * h2 * factor


def grad_simplex_2d_p(
    a: jnp.ndarray, b: jnp.ndarray, id: int, jd: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Derivatives ``(dP/dr, dP/ds)`` of the orthonormal basis at ``(a, b)``."""
    from ddg.dg1d.reference import grad_jacobi_p

    fa = jacobi_p(a, 0.0, 0.0, id)
    dfa = grad_jacobi_p(a, 0.0, 0.0, id)
    gb = jacobi_p(b, 2 * id + 1.0, 0.0, jd)
    dgb = grad_jacobi_p(b, 2 * id + 1.0, 0.0, jd)

    # r-derivative: ((2 / (1-b)) d/da) applied to fa(a) * gb(b) * (1-b)^id.
    dmode_dr = dfa * gb
    if id > 0:
        dmode_dr = dmode_dr * ((0.5 * (1.0 - b)) ** (id - 1))

    # s-derivative comes from chain rule: ((1+a)/2)/((1-b)/2) d/da + d/db.
    dmode_ds = dfa * (gb * (0.5 * (1.0 + a)))
    if id > 0:
        dmode_ds = dmode_ds * ((0.5 * (1.0 - b)) ** (id - 1))

    tmp = dgb * ((0.5 * (1.0 - b)) ** id)
    if id > 0:
        tmp = tmp - 0.5 * id * gb * ((0.5 * (1.0 - b)) ** (id - 1))
    dmode_ds = dmode_ds + fa * tmp

    norm = 2.0 ** (id + 0.5)
    return norm * dmode_dr, norm * dmode_ds


def vandermonde_2d(N: int, r: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """2D Vandermonde ``V[k, ij] = phi_{ij}(r_k, s_k)`` for total degree <= N."""
    a, b = rs_to_ab(r, s)
    cols = []
    for i in range(N + 1):
        for j in range(N - i + 1):
            cols.append(simplex_2d_p(a, b, i, j))
    return jnp.stack(cols, axis=1)


def grad_vandermonde_2d(
    N: int, r: jnp.ndarray, s: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(V_r, V_s)`` Vandermondes of the basis derivatives."""
    a, b = rs_to_ab(r, s)
    cols_r, cols_s = [], []
    for i in range(N + 1):
        for j in range(N - i + 1):
            dr, ds = grad_simplex_2d_p(a, b, i, j)
            cols_r.append(dr)
            cols_s.append(ds)
    return jnp.stack(cols_r, axis=1), jnp.stack(cols_s, axis=1)


def d_matrices_2d(
    N: int, r: jnp.ndarray, s: jnp.ndarray, V: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``Dr, Ds = V_r V^{-1}, V_s V^{-1}`` on the reference triangle."""
    Vr, Vs = grad_vandermonde_2d(N, r, s)
    Dr = jnp.linalg.solve(V.T, Vr.T).T
    Ds = jnp.linalg.solve(V.T, Vs.T).T
    return Dr, Ds


def face_masks_2d(N: int, r: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Indices of the ``N+1`` nodes lying on each of the three edges.

    Returns an integer array of shape ``(Nfp, 3)`` with columns
    ``[face_0_nodes, face_1_nodes, face_2_nodes]``.
    """
    NODE_TOL = 1e-12
    r_np = np.asarray(r)
    s_np = np.asarray(s)
    fmask0 = np.where(np.abs(s_np + 1.0) < NODE_TOL)[0]
    fmask1 = np.where(np.abs(r_np + s_np) < NODE_TOL)[0]
    fmask2 = np.where(np.abs(r_np + 1.0) < NODE_TOL)[0]
    return np.stack([fmask0, fmask1, fmask2], axis=1)


def lift_2d(
    N: int,
    r: jnp.ndarray,
    s: jnp.ndarray,
    V: jnp.ndarray,
    Fmask: np.ndarray,
) -> jnp.ndarray:
    """Surface lift operator ``M^{-1} E`` of shape ``(Np, 3*(N+1))``.

    ``E`` is built block by block: each face contributes the 1D mass
    matrix (in the appropriate edge parameter) at the rows indexed by
    that face's nodes. The reference-element ``M^{-1}`` is ``V V^T``,
    so ``LIFT = V (V^T E)``.
    """
    Np = (N + 1) * (N + 2) // 2
    Nfp = N + 1
    Emat = jnp.zeros((Np, 3 * Nfp))

    # Face 0 (s = -1): parametrise by r.
    face_r = r[Fmask[:, 0]]
    V1D = vandermonde_1d(N, face_r)
    mass_edge_0 = jnp.linalg.inv(V1D @ V1D.T)
    rows = jnp.asarray(Fmask[:, 0])
    cols = jnp.arange(0, Nfp)
    Emat = Emat.at[rows[:, None], cols[None, :]].set(mass_edge_0)

    # Face 1 (r + s = 0): parametrise by r (s = -r at edge).
    face_r = r[Fmask[:, 1]]
    V1D = vandermonde_1d(N, face_r)
    mass_edge_1 = jnp.linalg.inv(V1D @ V1D.T)
    rows = jnp.asarray(Fmask[:, 1])
    cols = jnp.arange(Nfp, 2 * Nfp)
    Emat = Emat.at[rows[:, None], cols[None, :]].set(mass_edge_1)

    # Face 2 (r = -1): parametrise by s.
    face_s = s[Fmask[:, 2]]
    V1D = vandermonde_1d(N, face_s)
    mass_edge_2 = jnp.linalg.inv(V1D @ V1D.T)
    rows = jnp.asarray(Fmask[:, 2])
    cols = jnp.arange(2 * Nfp, 3 * Nfp)
    Emat = Emat.at[rows[:, None], cols[None, :]].set(mass_edge_2)

    return V @ (V.T @ Emat)
