# Running on JAX GPU backend (L4 / A100 / H100)

Quick guide for taking the INS solvers from CPU laptop runs to a cloud GPU.

## State of GPU readiness (current branch)

| Path | GPU-compatible | Notes |
|---|---|---|
| Splitting schemes (dual / consistent) + LU | ✅ yes, slow | LU on the assembled dense matrices works but the `O(n²)` memory wins are wasted. Use only for small problems / sanity checks. |
| Splitting schemes + matrix-free CG | ✅ yes, fast | `IterativeConfig(method='cg', ...)` plumbed through `ns_step_2d` / `ns_step_consistent_splitting_2d`. `O(n)` memory, CG converges in `O(√cond)` iters on SPD systems. **Recommended path for large GPU runs.** Use `build_ins_operators(..., skip_factor=True)` to also skip the (unused) LU factorisation. CG vs LU verified to match to ~ 5e-8. |
| Coupled scheme + LU | ✅ yes, slow | Re-factorises every step (`u*` changes). Only viable for small problems. |
| Coupled scheme + matrix-free GMRES | ✅ yes, slow without preconditioning | `IterativeSolverConfig` to `coupled_ns_step_2d`. Works but converges slowly because the saddle-point matrix has condition number ~ 1e8. Block-triangular MG preconditioning (cf. exadg's `PreconditionerCoupled`) is the natural fix and a deferred TODO. |
| Penalty post-projection | ✅ yes, fast | The `(M + τ_d D + τ_c C)` operator is SPD; same CG story as the splitting inner solves. |
| Auto-diff through any of the above | ✅ yes | Everything is pure JAX; `jax.grad`, `jax.jacobian`, `jax.vjp`, `jax.jvp` all work. |

## Precision

We set `jax.config.update("jax_enable_x64", True)` globally. On L4 (Ampere) FP64
throughput is ~1 TFLOP vs ~30 TFLOP FP32 — so FP64 runs are ~30× slower per
flop. For prototyping / validating against analytical solutions, keep FP64 on.
For production turbulence runs, the path forward is:

1. Toggle `jax_enable_x64 = False` and re-validate the splitting schemes
   (the SIPG operator + LF flux should be fine in FP32; iterative tolerances
   need to be loosened to ~ 1e-5).
2. Mixed-precision iterative refinement: FP32 Krylov + FP64 residual.

This isn't done yet — it's a follow-up.

## JIT compilation

None of the step functions are `jax.jit`-decorated yet. Without JIT, each step
runs through the Python dispatcher every iteration and most of the matrix /
operator setup happens eagerly. JIT-ing the inner step is the single biggest
optimisation lever:

```python
import jax
ns_step_jit = jax.jit(ns_step_2d, static_argnames=("bdf", "ops", "ins_bc_types"))
```

Caveat: `bdf` and `ops` are dataclasses / tuples — need to either register them
as pytrees (already done for `BDFCoeffs` via `@dataclass(frozen=True)`) or pass
their leaves explicitly. This needs a careful refactor before turning on; it's
listed as a TODO.

## How to spin up an L4 run

1. Provision an L4 instance with CUDA 12+. NVIDIA's `nvidia/cuda` Docker image
   works; just `pip install jax[cuda12]` and the conda environment from
   the README otherwise.

2. Set `JAX_PLATFORMS=cuda` and confirm `jax.devices()` shows the GPU.

3. Sanity-test the existing suite:

   ```sh
   JAX_PLATFORMS=cuda pytest test/ -q
   ```

   This should pass exactly as on CPU. If anything fails, that's a
   GPU-compatibility bug — file an issue.

4. Run the comparison demo on a larger mesh:

   ```sh
   JAX_PLATFORMS=cuda python applications/ns_scheme_comparison_2d.py --large
   ```

   Default `--large` is `Kx=24 Ky=8 N=3 T_final=4 dt=0.02` (~7000 velocity DOFs,
   200 time steps). Expect:

   * dual_lu / consistent_lu: ~ minute per scheme (LU dominates).
   * coupled_lu: refactors every step, will dominate runtime; consider running
     it alone on a coarser mesh or just running the splitting schemes.

5. For *much* larger meshes (where LU is infeasible), use the iterative path.
   Once `ns_step_2d` is wired to pass through `IterativeConfig`, the splitting
   schemes will run in `O(n)` memory + reasonable iterations per step.

## What to send back

After a GPU run, the script writes:

* `output/ns_scheme_comparison.png` — side-by-side vorticity snapshots.
* `output/ns_scheme_comparison.json` — wall-times, max U/ω, per-scheme DOF
  counts, JAX platform tag.

Drop those back into the repo (or paste the JSON contents) and we can iterate
on what to chase next — preconditioner work, JIT compilation, mixed precision,
or new schemes.
