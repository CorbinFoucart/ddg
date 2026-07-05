"""1D nodal discontinuous Galerkin building blocks."""

from ddg.dg1d.advection import advec_rhs, solve_advection, solve_advection_trajectory
from ddg.dg1d.burgers import (
    burgers_rhs_1d,
    slope_limit_n,
    solve_burgers_1d,
    solve_burgers_1d_trajectory,
)
from ddg.dg1d.integrator import (
    low_storage_rk4,
    low_storage_rk4_step,
    low_storage_rk4_trajectory,
)
from ddg.dg1d.heat import heat_rhs, solve_heat, solve_heat_trajectory
from ddg.dg1d.mesh import Mesh1D, build_mesh_1d, uniform_mesh_1d, update_mesh_vertices
from ddg.dg1d.poisson import (
    poisson_ip_apply,
    poisson_ip_matrix,
    solve_poisson,
)
from ddg.dg1d.wave import solve_wave, solve_wave_trajectory, wave_rhs
from ddg.dg1d.reference import (
    d_matrix_1d,
    grad_jacobi_p,
    grad_vandermonde_1d,
    jacobi_gauss_lobatto_nodes,
    jacobi_gauss_quadrature,
    jacobi_p,
    lift_1d,
    vandermonde_1d,
)

__all__ = [
    "Mesh1D",
    "advec_rhs",
    "build_mesh_1d",
    "burgers_rhs_1d",
    "d_matrix_1d",
    "grad_jacobi_p",
    "grad_vandermonde_1d",
    "heat_rhs",
    "jacobi_gauss_lobatto_nodes",
    "jacobi_gauss_quadrature",
    "jacobi_p",
    "lift_1d",
    "low_storage_rk4",
    "low_storage_rk4_step",
    "low_storage_rk4_trajectory",
    "poisson_ip_apply",
    "poisson_ip_matrix",
    "slope_limit_n",
    "solve_advection",
    "solve_advection_trajectory",
    "solve_burgers_1d",
    "solve_burgers_1d_trajectory",
    "solve_heat",
    "solve_heat_trajectory",
    "solve_poisson",
    "solve_wave",
    "solve_wave_trajectory",
    "uniform_mesh_1d",
    "update_mesh_vertices",
    "vandermonde_1d",
    "wave_rhs",
]
