"""Differentiable discontinuous Galerkin solvers in JAX.

We enable JAX's float64 mode on import: spectral DG operators are
ill-conditioned in float32, and the Apple Metal backend is float32-only,
so the default workflow is CPU + float64.

To experiment on Metal (float32 only), set ``JAX_PLATFORMS=metal`` and
``JAX_ENABLE_X64=false`` *before* importing ``ddg``.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"
