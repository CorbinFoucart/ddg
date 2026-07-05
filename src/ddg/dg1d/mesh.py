"""1D mesh + StartUp1D state, packaged as a pytree.

Bundles the data that the MATLAB reference stashes in ``Globals1D``
into a single immutable container so RHS functions can stay pure and
JAX-traceable.

Layout convention (matches the reference):
    * Solution arrays have shape ``(Np, K)`` (nodes per element, elements).
    * Face arrays have shape ``(Nfp * Nfaces, K)`` flattened in
      ``(face, fp)`` order, with face 0 = left, face 1 = right.
    * ``vmapM``, ``vmapP`` are 1D index arrays into the flattened
      ``u.T.reshape(-1)`` (column-major / Fortran order, same as MATLAB)
      so the gather ``u.flatten(order='F')[vmapM]`` reproduces the
      MATLAB ``u(vmapM)`` exactly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ddg.dg1d.reference import (
    d_matrix_1d,
    jacobi_gauss_lobatto_nodes,
    lift_1d,
    vandermonde_1d,
)


# Field categories: arrays are pytree leaves; ints are static metadata.
_ARRAY_FIELDS = (
    "VX", "EToV",
    "r", "V", "invV", "Dr", "LIFT",
    "x",
    "rx", "J",
    "Fmask", "nx", "Fscale",
    "EToE", "EToF",
    "vmapM", "vmapP", "vmapB", "mapB",
)
_META_FIELDS = ("N", "Np", "K", "Nfp", "Nfaces", "periodic")


@dataclass(frozen=True)
class Mesh1D:
    """All operators + connectivity for a 1D nodal DG mesh."""

    # --- meta (static) ---
    N: int
    Np: int
    K: int
    Nfp: int
    Nfaces: int
    periodic: bool

    # --- mesh ---
    VX: jax.Array   # (Nv,) vertex coordinates
    EToV: jax.Array  # (K, 2) element-to-vertex (0-indexed)

    # --- reference element ---
    r: jax.Array     # (Np,) LGL nodes on [-1, 1]
    V: jax.Array     # (Np, Np) Legendre Vandermonde
    invV: jax.Array  # (Np, Np)
    Dr: jax.Array    # (Np, Np) reference-element differentiation
    LIFT: jax.Array  # (Np, Nfaces*Nfp)

    # --- physical grid ---
    x: jax.Array     # (Np, K) physical node coordinates

    # --- geometric factors ---
    rx: jax.Array    # (Np, K) dr/dx
    J: jax.Array     # (Np, K) det of mapping

    # --- face data ---
    Fmask: jax.Array   # (Nfp, Nfaces) within-element node indices on each face
    nx: jax.Array      # (Nfp*Nfaces, K) outward normals
    Fscale: jax.Array  # (Nfp*Nfaces, K) = 1/J at face nodes

    # --- connectivity ---
    EToE: jax.Array  # (K, Nfaces) neighbour element (0-indexed)
    EToF: jax.Array  # (K, Nfaces) neighbour face (0-indexed)

    # --- node-level maps into flattened u (column-major / Fortran order) ---
    vmapM: jax.Array  # (Nfp*Nfaces*K,) "minus" indices
    vmapP: jax.Array  # (Nfp*Nfaces*K,) "plus" indices
    vmapB: jax.Array  # (#boundary nodes,) indices of boundary nodes
    mapB: jax.Array   # (#boundary nodes,) positions in flattened du array

    # ------------------------------------------------------------------
    # pytree registration: arrays = leaves, ints/bools = static metadata
    # ------------------------------------------------------------------
    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in _ARRAY_FIELDS)
        aux = tuple(getattr(self, name) for name in _META_FIELDS)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        kwargs = dict(zip(_META_FIELDS, aux))
        kwargs.update(dict(zip(_ARRAY_FIELDS, children)))
        return cls(**kwargs)


jax.tree_util.register_pytree_node(
    Mesh1D, Mesh1D.tree_flatten, Mesh1D.tree_unflatten
)


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------

def uniform_mesh_1d(xmin: float, xmax: float, K: int) -> tuple[jax.Array, jax.Array]:
    """Equispaced K-element mesh of [xmin, xmax]. Returns (VX, EToV)."""
    VX = jnp.linspace(xmin, xmax, K + 1)
    EToV = jnp.stack([jnp.arange(K), jnp.arange(1, K + 1)], axis=1)
    return VX, EToV


def _connect_1d(EToV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (EToE, EToF) from element-to-vertex map.

    Pure-numpy because the construction is graph-flavoured (sparse face
    incidence, find shared vertices); it runs once at mesh build time.
    """
    K = EToV.shape[0]
    Nfaces = 2
    Nv = int(EToV.max()) + 1
    total_faces = Nfaces * K

    # Face -> vertex incidence: face f of element k touches vertex EToV[k, f].
    sp_FtoV = np.zeros((total_faces, Nv), dtype=np.int64)
    sk = 0
    for k in range(K):
        for f in range(Nfaces):
            sp_FtoV[sk, EToV[k, f]] = 1
            sk += 1

    sp_FtoF = sp_FtoV @ sp_FtoV.T - np.eye(total_faces, dtype=np.int64)
    faces1, faces2 = np.where(sp_FtoF == 1)

    EToE = np.tile(np.arange(K).reshape(-1, 1), (1, Nfaces))
    EToF = np.tile(np.arange(Nfaces).reshape(1, -1), (K, 1))

    elem1 = faces1 // Nfaces
    face1 = faces1 % Nfaces
    elem2 = faces2 // Nfaces
    face2 = faces2 % Nfaces

    EToE[elem1, face1] = elem2
    EToF[elem1, face1] = face2
    return EToE, EToF


def _build_maps_1d(
    Np: int, Nfp: int, Nfaces: int, K: int,
    Fmask: np.ndarray, x: np.ndarray,
    EToE: np.ndarray, EToF: np.ndarray,
    node_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (vmapM, vmapP, vmapB, mapB).

    ``vmapM`` and ``vmapP`` index into ``x.flatten(order='F')`` (Fortran),
    matching MATLAB's column-major linearisation of ``x[Np, K]``.
    """
    # MATLAB's nodeids = reshape(1:K*Np, Np, K) is column-major; in 0-indexed
    # numpy that is just np.arange(K*Np).reshape(K, Np).T, i.e. the same shape
    # (Np, K) but with the natural flattening order being column-major.
    node_ids = np.arange(K * Np).reshape(K, Np).T  # shape (Np, K)

    vmapM = np.zeros((Nfp, Nfaces, K), dtype=np.int64)
    vmapP = np.zeros((Nfp, Nfaces, K), dtype=np.int64)

    for k in range(K):
        for f in range(Nfaces):
            vmapM[:, f, k] = node_ids[Fmask[:, f], k]

    x_flat = x.flatten(order="F")
    for k1 in range(K):
        for f1 in range(Nfaces):
            k2 = EToE[k1, f1]
            f2 = EToF[k1, f1]
            vidM = vmapM[:, f1, k1]
            vidP = vmapM[:, f2, k2]
            d2 = (x_flat[vidM] - x_flat[vidP]) ** 2
            if np.all(d2 < node_tol):
                vmapP[:, f1, k1] = vidP

    # Flatten in (Nfp, Nfaces, K) order to match MATLAB's vmapP(:) / vmapM(:).
    vmapM = vmapM.flatten(order="F")
    vmapP = vmapP.flatten(order="F")

    # Boundary nodes: positions where vmapP fell back to vmapM (no neighbour).
    mapB = np.where(vmapP == vmapM)[0]
    vmapB = vmapM[mapB]
    return vmapM, vmapP, vmapB, mapB


def _apply_periodic_maps_1d(
    vmapM: np.ndarray, vmapP: np.ndarray, K: int, Np: int, Nfaces: int, Nfp: int
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap the two domain endpoints: left face of element 0 sees right
    face of element K-1, and vice versa."""
    # In the flattened (Nfp, Nfaces, K) layout with Nfp=1, Nfaces=2:
    #   index of (face=0, elem=0)        = 0
    #   index of (face=1, elem=K-1)      = 1 + Nfaces*(K-1) = 2K-1
    left = 0
    right = Nfaces * K - 1
    vmapP = vmapP.copy()
    vmapP[left] = vmapM[right]
    vmapP[right] = vmapM[left]
    return vmapM, vmapP


def build_mesh_1d(
    VX: jax.Array,
    EToV: jax.Array,
    N: int,
    *,
    periodic: bool = False,
) -> Mesh1D:
    """Assemble a Mesh1D from vertex coords + element-to-vertex map.

    This is the equivalent of MATLAB's ``StartUp1D``.
    """
    VX_np = np.asarray(VX, dtype=np.float64)
    EToV_np = np.asarray(EToV, dtype=np.int64)

    K = EToV_np.shape[0]
    Np = N + 1
    Nfp = 1
    Nfaces = 2

    # Reference element
    r = jacobi_gauss_lobatto_nodes(0.0, 0.0, N)
    V = vandermonde_1d(N, r)
    invV = jnp.linalg.inv(V)
    Dr = d_matrix_1d(N, r, V)
    LIFT = lift_1d(Np, Nfaces, Nfp, V)

    # Physical grid: x = 0.5*((1-r)*VX[va] + (1+r)*VX[vb])
    r_np = np.asarray(r, dtype=np.float64)
    va = EToV_np[:, 0]
    vb = EToV_np[:, 1]
    # x has shape (Np, K)
    x_np = np.outer(np.ones(Np), VX_np[va]) + 0.5 * (
        np.outer(r_np + 1.0, VX_np[vb] - VX_np[va])
    )

    # Geometric factors: J = Dr @ x (Np, K); rx = 1/J
    Dr_np = np.asarray(Dr, dtype=np.float64)
    J_np = Dr_np @ x_np
    rx_np = 1.0 / J_np

    # Face masks: which reference nodes lie on each face.
    NODE_TOL = 1e-10
    fmask_left = np.where(np.abs(r_np + 1.0) < NODE_TOL)[0]
    fmask_right = np.where(np.abs(r_np - 1.0) < NODE_TOL)[0]
    Fmask_np = np.stack([fmask_left, fmask_right], axis=1)  # (Nfp, Nfaces)

    # Outward normals: -1 on left face, +1 on right.
    nx_np = np.zeros((Nfp * Nfaces, K))
    nx_np[0, :] = -1.0
    nx_np[1, :] = 1.0

    # Face metric: 1/J at the face nodes.
    Fscale_np = 1.0 / J_np[Fmask_np.flatten(order="F"), :]

    # Connectivity + node maps
    EToE_np, EToF_np = _connect_1d(EToV_np)
    vmapM_np, vmapP_np, vmapB_np, mapB_np = _build_maps_1d(
        Np, Nfp, Nfaces, K, Fmask_np, x_np, EToE_np, EToF_np
    )
    if periodic:
        vmapM_np, vmapP_np = _apply_periodic_maps_1d(
            vmapM_np, vmapP_np, K, Np, Nfaces, Nfp
        )
        # With periodic wrap there are no Dirichlet boundary nodes.
        mapB_np = np.array([], dtype=np.int64)
        vmapB_np = np.array([], dtype=np.int64)

    return Mesh1D(
        N=N, Np=Np, K=K, Nfp=Nfp, Nfaces=Nfaces, periodic=periodic,
        VX=jnp.asarray(VX_np),
        EToV=jnp.asarray(EToV_np),
        r=r, V=V, invV=invV, Dr=Dr, LIFT=LIFT,
        x=jnp.asarray(x_np),
        rx=jnp.asarray(rx_np),
        J=jnp.asarray(J_np),
        Fmask=jnp.asarray(Fmask_np),
        nx=jnp.asarray(nx_np),
        Fscale=jnp.asarray(Fscale_np),
        EToE=jnp.asarray(EToE_np),
        EToF=jnp.asarray(EToF_np),
        vmapM=jnp.asarray(vmapM_np),
        vmapP=jnp.asarray(vmapP_np),
        vmapB=jnp.asarray(vmapB_np),
        mapB=jnp.asarray(mapB_np),
    )


def update_mesh_vertices(mesh: Mesh1D, VX: jax.Array) -> Mesh1D:
    """Return a new ``Mesh1D`` with updated vertex positions, topology unchanged.

    Use this when you want gradients to flow through vertex coordinates
    (e.g. for shape optimisation): the integer connectivity arrays
    (``vmapM``, ``vmapP``, ``EToE``, ...) are reused, while ``x``,
    ``J``, ``rx``, and ``Fscale`` are recomputed from ``VX`` in pure
    JAX so a ``jax.grad`` through ``VX`` is well-defined.

    ``VX`` may be a JAX tracer; ``EToV`` and ``Fmask`` are read as
    concrete integer arrays.
    """
    VX = jnp.asarray(VX, dtype=jnp.float64)
    EToV_np = np.asarray(mesh.EToV)
    va = EToV_np[:, 0]
    vb = EToV_np[:, 1]

    x = jnp.outer(jnp.ones(mesh.Np), VX[va]) + 0.5 * jnp.outer(
        mesh.r + 1.0, VX[vb] - VX[va]
    )
    J = mesh.Dr @ x
    rx = 1.0 / J
    Fmask_flat = np.asarray(mesh.Fmask).flatten(order="F")
    Fscale = 1.0 / J[Fmask_flat, :]

    return dataclasses.replace(mesh, VX=VX, x=x, J=J, rx=rx, Fscale=Fscale)
