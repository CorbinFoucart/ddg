"""2D nodal discontinuous Galerkin building blocks (triangular meshes)."""

from ddg.dg2d.advection import (
    advec_rhs_2d,
    solve_advection_2d,
    solve_advection_2d_trajectory,
)
from ddg.dg2d.maxwell import (
    curl_z_2d,
    grad_2d,
    maxwell_rhs_2d,
    solve_maxwell_2d,
    solve_maxwell_2d_trajectory,
)
from ddg.dg2d.mesh import (
    Mesh2D,
    build_mesh_2d,
    uniform_rectangle_mesh_2d,
    update_mesh_2d_vertices,
)
from ddg.dg2d.reference import (
    d_matrices_2d,
    equilateral_nodes_2d,
    face_masks_2d,
    grad_simplex_2d_p,
    grad_vandermonde_2d,
    lift_2d,
    rs_to_ab,
    simplex_2d_p,
    vandermonde_2d,
    xy_to_rs,
)

__all__ = [
    "Mesh2D",
    "advec_rhs_2d",
    "build_mesh_2d",
    "curl_z_2d",
    "d_matrices_2d",
    "equilateral_nodes_2d",
    "face_masks_2d",
    "grad_2d",
    "grad_simplex_2d_p",
    "grad_vandermonde_2d",
    "lift_2d",
    "maxwell_rhs_2d",
    "rs_to_ab",
    "simplex_2d_p",
    "solve_advection_2d",
    "solve_advection_2d_trajectory",
    "solve_maxwell_2d",
    "solve_maxwell_2d_trajectory",
    "uniform_rectangle_mesh_2d",
    "update_mesh_2d_vertices",
    "vandermonde_2d",
    "xy_to_rs",
]
