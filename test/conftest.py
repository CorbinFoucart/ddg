"""Pytest configuration: pin JAX to CPU + float64 for deterministic tests.

``jax-metal`` is float32-only, so any spectral-DG correctness test must
run on the CPU backend.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
