"""JAX-traceable mesh quantities.

``ddg.dg2d.mesh.build_mesh_2d`` converts ``VX``/``VY`` to NumPy
internally so that connectivity inferred from ``EToV`` can be built
host-side. The downstream geometric quantities ``(x, y, J, rx, sx,
ry, sy, nx, ny, sJ)`` are produced by JAX helpers
(``_geometric_factors_2d``, ``_normals_2d``) that already accept JAX
inputs, so the only thing in the way of an end-to-end differentiable
forward solve is the ``np.outer`` step that computes ``x_np`` and
``y_np`` from ``(VX_np[va], VX_np[vb], VX_np[vc])``.

This module exposes a small public API that re-runs the geometric
chain in JAX:

* :func:`differentiable_mesh_quantities` --- maps ``(VX_jax, VY_jax)``
  to all geometric quantities the BDM/Lagrangian forward solvers
  consume. Connectivity (Fmask, EToE, EToF, vmap*, mapB) is supplied
  by a host-built template and is reused unchanged.
* :func:`make_template` --- builds the template from an existing
  ``Mesh2D``; freezes connectivity + reference-element data and
  exposes ``EToV``, ``VX_nominal``, ``VY_nominal`` to be perturbed.

The returned ``DiffMesh2D`` dataclass has the same field layout as
``Mesh2D``, so it drops into the same downstream code without
modification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .mesh import (
    Mesh2D, build_mesh_2d, _geometric_factors_2d, _normals_2d,
)
from .curved import _blending_weight_for_face


@dataclass(frozen=True)
class MeshTemplate:
    """Static, host-built scaffolding around a parameterised mesh.

    Holds the integer connectivity, reference-element data, and
    nominal vertex positions. Building this once via
    :func:`make_template` lets the downstream forward solve be a pure
    function of ``(VX, VY)``.
    """
    # Topology + reference element (static during AD).
    N: int
    Np: int
    Nfp: int
    K: int
    Nfaces: int
    EToV: np.ndarray         # (K, 3) int64
    va: np.ndarray           # vertex 0 per element (K,)
    vb: np.ndarray           # vertex 1 per element (K,)
    vc: np.ndarray           # vertex 2 per element (K,)
    r: jax.Array             # reference nodes (Np,)
    s: jax.Array
    V: jax.Array             # reference Vandermonde (Np, Np)
    invV: jax.Array
    Dr: jax.Array
    Ds: jax.Array
    LIFT: jax.Array
    Fmask: jax.Array         # (Nfp, 3) int
    # Nominal vertex positions (for "no-perturbation" smoke tests).
    VX_nominal: jax.Array
    VY_nominal: jax.Array
    # Connectivity arrays carried unchanged from the template build.
    EToE: jax.Array
    EToF: jax.Array
    vmapM: jax.Array
    vmapP: jax.Array
    vmapB: jax.Array
    mapB: jax.Array


@dataclass(frozen=True)
class DiffMesh2D:
    """A :class:`Mesh2D` lookalike whose geometric arrays are tracers.

    Same shape and naming as ``Mesh2D`` so the existing BDM /
    Lagrangian operators don't need modification.
    """
    N: int
    Np: int
    K: int
    Nfp: int
    Nfaces: int
    VX: jax.Array
    VY: jax.Array
    EToV: jax.Array          # int; static
    r: jax.Array
    s: jax.Array
    V: jax.Array
    invV: jax.Array
    Dr: jax.Array
    Ds: jax.Array
    LIFT: jax.Array
    Fmask: jax.Array
    x: jax.Array             # tracer
    y: jax.Array             # tracer
    rx: jax.Array
    sx: jax.Array
    ry: jax.Array
    sy: jax.Array
    J: jax.Array
    nx: jax.Array
    ny: jax.Array
    sJ: jax.Array
    Fscale: jax.Array
    EToE: jax.Array
    EToF: jax.Array
    vmapM: jax.Array
    vmapP: jax.Array
    vmapB: jax.Array
    mapB: jax.Array


@dataclass(frozen=True)
class CurvingInfo:
    """Per-face data needed to apply a JAX-traceable curving.

    Built once host-side at template time from the nominal cylinder
    parameters. Holds the (f, ks_f) selection of body-fit faces to be
    curved and the precomputed Gordon-Hall interpolation matrices that
    propagate face-trace projections into the interior.
    """
    # Per face f in {0, 1, 2}: list of element indices whose face f is
    # on the curve, the Lagrangian-node indices on the face (Fmask col),
    # the (Np, Nfp) face-to-volume interp matrix, and (Np,) interior
    # blending weights.
    ks_per_face: tuple[np.ndarray, np.ndarray, np.ndarray]
    Fmask_per_face: tuple[np.ndarray, np.ndarray, np.ndarray]
    I_face_to_vol: tuple[jax.Array, jax.Array, jax.Array]
    weight: tuple[jax.Array, jax.Array, jax.Array]


def make_curving_info(
    mesh: Mesh2D,
    *,
    cx: float, cy: float, radius: float,
    detection_tol: float = 0.05,
) -> CurvingInfo:
    """Identify which (f, k) faces lie on the cylinder, host-side.

    This decision is made ONCE at the nominal (cx, cy, radius); the
    resulting (f, ks) selection is then reused for arbitrary nearby
    parameter values. The differentiable curving in
    :func:`differentiable_curve` projects exactly these face nodes
    onto the perturbed circle.
    """
    Nfaces = 3
    K = mesh.K
    Fmask_np = np.asarray(mesh.Fmask)
    x_np = np.asarray(mesh.x)
    y_np = np.asarray(mesh.y)
    x_face_avg = np.zeros((Nfaces, K))
    y_face_avg = np.zeros((Nfaces, K))
    for f in range(Nfaces):
        x_face_avg[f, :] = x_np[Fmask_np[:, f], :].mean(axis=0)
        y_face_avg[f, :] = y_np[Fmask_np[:, f], :].mean(axis=0)
    EToE_np = np.asarray(mesh.EToE)
    is_bdy = (EToE_np == np.arange(K)[:, None]).T   # (Nfaces, K)
    d2 = (x_face_avg - cx) ** 2 + (y_face_avg - cy) ** 2
    in_circle = d2 < (radius * (1 + detection_tol)) ** 2
    is_curved = in_circle & is_bdy

    ks_per_face = []
    Fmask_per_face = []
    I_per_face = []
    weight_per_face = []
    r_np = np.asarray(mesh.r)
    s_np = np.asarray(mesh.s)
    from ddg.dg1d.reference import vandermonde_1d
    for f in range(Nfaces):
        ks = np.where(is_curved[f, :])[0].astype(np.int64)
        Fm = Fmask_np[:, f].astype(np.int64)
        if f == 0:
            t = r_np; t_face = r_np[Fm]
        elif f == 1:
            t = r_np; t_face = r_np[Fm]
        else:
            t = s_np; t_face = s_np[Fm]
        V_face = vandermonde_1d(mesh.N, jnp.asarray(t_face))
        V_vol = vandermonde_1d(mesh.N, jnp.asarray(t))
        I = V_vol @ jnp.linalg.inv(V_face)
        weight = _blending_weight_for_face(mesh.r, mesh.s, f)
        ks_per_face.append(ks)
        Fmask_per_face.append(Fm)
        I_per_face.append(I)
        weight_per_face.append(weight)
    return CurvingInfo(
        ks_per_face=tuple(ks_per_face),
        Fmask_per_face=tuple(Fmask_per_face),
        I_face_to_vol=tuple(I_per_face),
        weight=tuple(weight_per_face),
    )


def differentiable_curve(
    x: jax.Array, y: jax.Array, info: CurvingInfo,
    cx: jax.Array, cy: jax.Array, radius: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Project marked face nodes onto a circle of radius ``radius``
    centred at ``(cx, cy)`` and blend the offset into interior nodes.

    JAX-traceable in ``(cx, cy, radius)``; the host-side selection of
    "which faces are on the curve" comes from ``info``.
    """
    x_new = x
    y_new = y
    for f in range(3):
        ks = info.ks_per_face[f]
        if ks.size == 0:
            continue
        Fm = info.Fmask_per_face[f]
        x_old_face = x_new[Fm[:, None], ks[None, :]]    # (Nfp, |ks|)
        y_old_face = y_new[Fm[:, None], ks[None, :]]
        dxf = x_old_face - cx
        dyf = y_old_face - cy
        rf = jnp.sqrt(dxf * dxf + dyf * dyf)
        x_proj = cx + radius * dxf / rf
        y_proj = cy + radius * dyf / rf
        dx_face = x_proj - x_old_face
        dy_face = y_proj - y_old_face

        # Set the face nodes to the projected positions.
        x_new = x_new.at[Fm[:, None], ks[None, :]].set(x_proj)
        y_new = y_new.at[Fm[:, None], ks[None, :]].set(y_proj)

        # Blend into the interior of the curved elements.
        I = info.I_face_to_vol[f]
        weight = info.weight[f]
        dx_interp = I @ dx_face                          # (Np, |ks|)
        dy_interp = I @ dy_face
        x_new = x_new.at[:, ks].add(weight[:, None] * dx_interp)
        y_new = y_new.at[:, ks].add(weight[:, None] * dy_interp)
        # Reset face nodes (weight==1 at face but Vandermonde inversion
        # has tiny noise; this matches the host curving exactly).
        x_new = x_new.at[Fm[:, None], ks[None, :]].set(x_proj)
        y_new = y_new.at[Fm[:, None], ks[None, :]].set(y_proj)
    return x_new, y_new


def differentiable_curve_shape(
    x: jax.Array, y: jax.Array, info: CurvingInfo,
    cx: jax.Array, cy: jax.Array, R0: jax.Array,
    fourier_coeffs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Project marked face nodes onto a shape-parameterised curve.

    ``r(theta) = R0 * (1 + sum_k a_k cos(k*theta) + b_k sin(k*theta))``
    where ``fourier_coeffs[2i] = a_{i+1}, fourier_coeffs[2i+1] = b_{i+1}``
    (mode 1 is cos/sin theta, mode 2 is cos/sin 2theta, etc.). With all
    coefficients zero, the curve is a circle of radius ``R0`` and the
    output matches :func:`differentiable_curve`.

    JAX-traceable in ``(cx, cy, R0, fourier_coeffs)``.
    """
    n_modes = fourier_coeffs.shape[0] // 2
    x_new = x
    y_new = y
    for f in range(3):
        ks = info.ks_per_face[f]
        if ks.size == 0:
            continue
        Fm = info.Fmask_per_face[f]
        x_old_face = x_new[Fm[:, None], ks[None, :]]
        y_old_face = y_new[Fm[:, None], ks[None, :]]
        dxf = x_old_face - cx
        dyf = y_old_face - cy
        rf = jnp.sqrt(dxf * dxf + dyf * dyf)
        theta = jnp.arctan2(dyf, dxf)
        # Local radius from Fourier expansion.
        shape_factor = jnp.ones_like(theta)
        for i in range(n_modes):
            k_mode = i + 1
            shape_factor = (
                shape_factor
                + fourier_coeffs[2 * i] * jnp.cos(k_mode * theta)
                + fourier_coeffs[2 * i + 1] * jnp.sin(k_mode * theta)
            )
        r_local = R0 * shape_factor
        x_proj = cx + r_local * dxf / rf
        y_proj = cy + r_local * dyf / rf
        dx_face = x_proj - x_old_face
        dy_face = y_proj - y_old_face

        x_new = x_new.at[Fm[:, None], ks[None, :]].set(x_proj)
        y_new = y_new.at[Fm[:, None], ks[None, :]].set(y_proj)
        I = info.I_face_to_vol[f]
        weight = info.weight[f]
        dx_interp = I @ dx_face
        dy_interp = I @ dy_face
        x_new = x_new.at[:, ks].add(weight[:, None] * dx_interp)
        y_new = y_new.at[:, ks].add(weight[:, None] * dy_interp)
        x_new = x_new.at[Fm[:, None], ks[None, :]].set(x_proj)
        y_new = y_new.at[Fm[:, None], ks[None, :]].set(y_proj)
    return x_new, y_new


def differentiable_mesh_with_shape(
    VX: jax.Array, VY: jax.Array,
    template: MeshTemplate, curving: CurvingInfo,
    cx: jax.Array, cy: jax.Array, R0: jax.Array,
    fourier_coeffs: jax.Array,
):
    """Same as :func:`differentiable_mesh_with_circle` but the curve
    is a Fourier-mode-parameterised radial shape rather than a circle."""
    x_aff, y_aff = _xy_from_vx_vy(
        VX, VY, template.va, template.vb, template.vc,
        template.r, template.s,
    )
    x, y = differentiable_curve_shape(
        x_aff, y_aff, curving, cx, cy, R0, fourier_coeffs,
    )
    rx, sx, ry, sy, J = _geometric_factors_2d(
        x, y, template.Dr, template.Ds,
    )
    Fmask_np = np.asarray(template.Fmask)
    nx, ny, sJ = _normals_2d(
        x, y, template.Dr, template.Ds,
        Fmask_np, template.Nfp, template.K,
    )
    Fmask_flat = jnp.asarray(Fmask_np.flatten(order="F"))
    Fscale = sJ / J[Fmask_flat, :]
    return Mesh2D(
        N=template.N, Np=template.Np, K=template.K,
        Nfp=template.Nfp, Nfaces=template.Nfaces,
        VX=VX, VY=VY, EToV=jnp.asarray(template.EToV),
        r=template.r, s=template.s,
        V=template.V, invV=template.invV,
        Dr=template.Dr, Ds=template.Ds, LIFT=template.LIFT,
        Fmask=template.Fmask,
        x=x, y=y, rx=rx, sx=sx, ry=ry, sy=sy, J=J,
        nx=nx, ny=ny, sJ=sJ, Fscale=Fscale,
        EToE=template.EToE, EToF=template.EToF,
        vmapM=template.vmapM, vmapP=template.vmapP,
        vmapB=template.vmapB, mapB=template.mapB,
    )


def differentiable_mesh_with_circle(
    VX: jax.Array, VY: jax.Array,
    template: MeshTemplate, curving: CurvingInfo,
    cx: jax.Array, cy: jax.Array, radius: jax.Array,
) -> Mesh2D:
    """One-stop shop: take ``(VX, VY)`` + ``(cx, cy, radius)``, return
    a fully assembled differentiable mesh whose body-fit ring is bent
    onto the requested circle.
    """
    # Affine x, y from VX, VY (no curving).
    x_aff, y_aff = _xy_from_vx_vy(
        VX, VY, template.va, template.vb, template.vc,
        template.r, template.s,
    )
    # Apply the JAX-traceable curving.
    x, y = differentiable_curve(x_aff, y_aff, curving, cx, cy, radius)
    rx, sx, ry, sy, J = _geometric_factors_2d(
        x, y, template.Dr, template.Ds,
    )
    Fmask_np = np.asarray(template.Fmask)
    nx, ny, sJ = _normals_2d(
        x, y, template.Dr, template.Ds,
        Fmask_np, template.Nfp, template.K,
    )
    Fmask_flat = jnp.asarray(Fmask_np.flatten(order="F"))
    Fscale = sJ / J[Fmask_flat, :]
    return Mesh2D(
        N=template.N, Np=template.Np, K=template.K,
        Nfp=template.Nfp, Nfaces=template.Nfaces,
        VX=VX, VY=VY, EToV=jnp.asarray(template.EToV),
        r=template.r, s=template.s,
        V=template.V, invV=template.invV,
        Dr=template.Dr, Ds=template.Ds, LIFT=template.LIFT,
        Fmask=template.Fmask,
        x=x, y=y, rx=rx, sx=sx, ry=ry, sy=sy, J=J,
        nx=nx, ny=ny, sJ=sJ, Fscale=Fscale,
        EToE=template.EToE, EToF=template.EToF,
        vmapM=template.vmapM, vmapP=template.vmapP,
        vmapB=template.vmapB, mapB=template.mapB,
    )


def _smoothstep_window(r: jax.Array, r1: float, r2: float) -> jax.Array:
    """Smooth 1 -> 0 window: 1 for r<=r1, 0 for r>=r2, C1 in between."""
    t = jnp.clip((r - r1) / (r2 - r1), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def windowed_joukowski_map(
    x: jax.Array, y: jax.Array,
    cx: jax.Array, cy: jax.Array, R0: jax.Array,
    mu_x: jax.Array, mu_y: jax.Array,
    r1: float, r2: float, te_round: float = 0.92,
) -> tuple[jax.Array, jax.Array]:
    r"""Conformal Joukowski map that turns the circle :math:`|\zeta|=R_0`
    (with :math:`\zeta=(x-c_x)+i(y-c_y)`) into an airfoil, windowed back
    to the identity for :math:`|\zeta|\ge r_2` so the outer mesh and the
    channel walls stay fixed:

    .. math::
        M(\zeta) = \zeta + w(|\zeta|)\,\frac{b^2}{\zeta + Z_c},\qquad
        Z_c = \mu_x + i\mu_y,\quad
        b = \kappa\,(\mu_x + \sqrt{R_0^2 - \mu_y^2}).

    ``mu_x < 0`` thickens the body and ``mu_y`` adds camber. The
    trailing-edge rounding ``kappa = te_round`` (slightly below 1) keeps
    the Joukowski critical points :math:`Z=\pm b` strictly *inside* the
    circle, so the map is regular (derivative bounded away from zero) on
    the whole mesh exterior -- a finite-thickness trailing edge with no
    cusp singularity, hence no degenerate elements. ``kappa = 1`` would
    place the critical point on the boundary (a sharp cusp, where the
    element Jacobian collapses). The map is injective on the exterior, so
    it cannot invert elements; the smooth window only acts where the
    displacement is already tiny. JAX-traceable in
    ``(cx, cy, R0, mu_x, mu_y)``.
    """
    zx = x - cx
    zy = y - cy
    r = jnp.sqrt(zx * zx + zy * zy)
    b = te_round * (mu_x + jnp.sqrt(jnp.clip(R0 * R0 - mu_y * mu_y, 1e-12)))
    # complex displacement b^2 / (zeta + Zc), Zc = mu_x + i mu_y
    ax = zx + mu_x
    ay = zy + mu_y
    den = ax * ax + ay * ay
    b2 = b * b
    disp_x = b2 * ax / den
    disp_y = -b2 * ay / den
    w = _smoothstep_window(r, r1, r2)
    return x + w * disp_x, y + w * disp_y


def joukowski_airfoil_area(
    cx: jax.Array, cy: jax.Array, R0: jax.Array,
    mu_x: jax.Array, mu_y: jax.Array, n_sample: int = 512,
    te_round: float = 0.92,
) -> jax.Array:
    """Enclosed area of the Joukowski airfoil (shoelace on a fine sample
    of the mapped boundary circle). Differentiable in ``(mu_x, mu_y)``;
    used as the area constraint in shape optimisation."""
    th = jnp.linspace(0.0, 2.0 * jnp.pi, n_sample, endpoint=False)
    zx = R0 * jnp.cos(th)
    zy = R0 * jnp.sin(th)
    b = te_round * (mu_x + jnp.sqrt(jnp.clip(R0 * R0 - mu_y * mu_y, 1e-12)))
    ax = zx + mu_x
    ay = zy + mu_y
    den = ax * ax + ay * ay
    b2 = b * b
    X = zx + b2 * ax / den
    Y = zy - b2 * ay / den
    return 0.5 * jnp.abs(jnp.sum(X * jnp.roll(Y, -1) - jnp.roll(X, -1) * Y))


def differentiable_mesh_with_airfoil(
    VX: jax.Array, VY: jax.Array,
    template: MeshTemplate, curving: CurvingInfo,
    cx: jax.Array, cy: jax.Array, R0: jax.Array,
    mu_x: jax.Array, mu_y: jax.Array,
    r1: float = 0.10, r2: float = 0.18, te_round: float = 0.92,
) -> Mesh2D:
    """Assemble a differentiable mesh whose cylinder hole has been
    conformally morphed into a Joukowski airfoil. The body-fit ring is
    first bent onto the exact circle of radius ``R0`` (reusing the
    validated :func:`differentiable_curve`), then every node is carried
    through :func:`windowed_joukowski_map`."""
    x_aff, y_aff = _xy_from_vx_vy(
        VX, VY, template.va, template.vb, template.vc,
        template.r, template.s,
    )
    x_circ, y_circ = differentiable_curve(x_aff, y_aff, curving, cx, cy, R0)
    x, y = windowed_joukowski_map(
        x_circ, y_circ, cx, cy, R0, mu_x, mu_y, r1, r2, te_round)
    rx, sx, ry, sy, J = _geometric_factors_2d(x, y, template.Dr, template.Ds)
    Fmask_np = np.asarray(template.Fmask)
    nx, ny, sJ = _normals_2d(
        x, y, template.Dr, template.Ds, Fmask_np, template.Nfp, template.K)
    Fmask_flat = jnp.asarray(Fmask_np.flatten(order="F"))
    Fscale = sJ / J[Fmask_flat, :]
    return Mesh2D(
        N=template.N, Np=template.Np, K=template.K,
        Nfp=template.Nfp, Nfaces=template.Nfaces,
        VX=VX, VY=VY, EToV=jnp.asarray(template.EToV),
        r=template.r, s=template.s,
        V=template.V, invV=template.invV,
        Dr=template.Dr, Ds=template.Ds, LIFT=template.LIFT,
        Fmask=template.Fmask,
        x=x, y=y, rx=rx, sx=sx, ry=ry, sy=sy, J=J,
        nx=nx, ny=ny, sJ=sJ, Fscale=Fscale,
        EToE=template.EToE, EToF=template.EToF,
        vmapM=template.vmapM, vmapP=template.vmapP,
        vmapB=template.vmapB, mapB=template.mapB,
    )


def make_template(mesh: Mesh2D) -> MeshTemplate:
    """Freeze the connectivity + reference element of an existing
    ``Mesh2D`` so we can perturb only ``(VX, VY)`` afterwards."""
    EToV_np = np.asarray(mesh.EToV, dtype=np.int64)
    return MeshTemplate(
        N=int(mesh.N), Np=int(mesh.Np), Nfp=int(mesh.Nfp),
        K=int(mesh.K), Nfaces=int(mesh.Nfaces),
        EToV=EToV_np,
        va=EToV_np[:, 0], vb=EToV_np[:, 1], vc=EToV_np[:, 2],
        r=mesh.r, s=mesh.s, V=mesh.V, invV=mesh.invV,
        Dr=mesh.Dr, Ds=mesh.Ds, LIFT=mesh.LIFT,
        Fmask=mesh.Fmask,
        VX_nominal=mesh.VX,
        VY_nominal=mesh.VY,
        EToE=mesh.EToE, EToF=mesh.EToF,
        vmapM=mesh.vmapM, vmapP=mesh.vmapP,
        vmapB=mesh.vmapB, mapB=mesh.mapB,
    )


def _xy_from_vx_vy(VX: jax.Array, VY: jax.Array,
                   va: np.ndarray, vb: np.ndarray, vc: np.ndarray,
                   r: jax.Array, s: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Affine reference-to-physical map: same expression as the
    NumPy version in ``build_mesh_2d`` but entirely in JAX.

    ``r``, ``s`` have shape ``(Np,)``; the vertex-index arrays have
    shape ``(K,)``. Returns ``x, y`` of shape ``(Np, K)``.
    """
    va_j = jnp.asarray(va); vb_j = jnp.asarray(vb); vc_j = jnp.asarray(vc)
    VX_a = VX[va_j]; VX_b = VX[vb_j]; VX_c = VX[vc_j]
    VY_a = VY[va_j]; VY_b = VY[vb_j]; VY_c = VY[vc_j]
    one = jnp.ones_like(r)
    x = 0.5 * (
        jnp.outer(-(r + s), VX_a)
        + jnp.outer(one + r, VX_b)
        + jnp.outer(one + s, VX_c)
    )
    y = 0.5 * (
        jnp.outer(-(r + s), VY_a)
        + jnp.outer(one + r, VY_b)
        + jnp.outer(one + s, VY_c)
    )
    return x, y


def differentiable_mesh_quantities(
    VX: jax.Array, VY: jax.Array, template: MeshTemplate,
) -> Mesh2D:
    """Pure-JAX mesh assembly from ``(VX, VY)`` and a frozen template.

    Recomputes the geometric chain (x, y, J, rx, sx, ry, sy, nx, ny,
    sJ, Fscale) entirely in JAX so ``jax.grad`` flows through to
    ``VX``, ``VY``. Connectivity, reference-element data, and node
    maps are taken from the template unchanged.
    """
    x, y = _xy_from_vx_vy(
        VX, VY, template.va, template.vb, template.vc,
        template.r, template.s,
    )
    rx, sx, ry, sy, J = _geometric_factors_2d(
        x, y, template.Dr, template.Ds,
    )
    Fmask_np = np.asarray(template.Fmask)
    nx, ny, sJ = _normals_2d(
        x, y, template.Dr, template.Ds,
        Fmask_np, template.Nfp, template.K,
    )
    Fmask_flat = jnp.asarray(Fmask_np.flatten(order="F"))
    Fscale = sJ / J[Fmask_flat, :]

    return Mesh2D(
        N=template.N, Np=template.Np, K=template.K,
        Nfp=template.Nfp, Nfaces=template.Nfaces,
        VX=VX, VY=VY, EToV=jnp.asarray(template.EToV),
        r=template.r, s=template.s,
        V=template.V, invV=template.invV,
        Dr=template.Dr, Ds=template.Ds, LIFT=template.LIFT,
        Fmask=template.Fmask,
        x=x, y=y, rx=rx, sx=sx, ry=ry, sy=sy, J=J,
        nx=nx, ny=ny, sJ=sJ, Fscale=Fscale,
        EToE=template.EToE, EToF=template.EToF,
        vmapM=template.vmapM, vmapP=template.vmapP,
        vmapB=template.vmapB, mapB=template.mapB,
    )
