# exadg INS survey

Survey of [exadg](https://github.com/exadg/exadg)'s incompressible Navier-Stokes
machinery, in the context of our differentiable JAX DG codebase. exadg is a
mature high-order DG framework built on deal.II by the Kronbichler / Wall group
at TUM and U. Augsburg, and ships a much wider menu of INS schemes than the
Hesthaven dual-splitting we currently have.

A clone lives at `reference/exadg/` (gitignored). Key paths:

- `include/exadg/incompressible_navier_stokes/time_integration/` — 4 schemes
- `include/exadg/incompressible_navier_stokes/spatial_discretization/` — operator decompositions per scheme
- `include/exadg/incompressible_navier_stokes/spatial_discretization/operators/` — re-usable building blocks
- `include/exadg/incompressible_navier_stokes/preconditioners/` — multigrid preconditioners
- `applications/incompressible_navier_stokes/` — 21 benchmarks

## Time-integration schemes

```
enum class TemporalDiscretization {
  Undefined,
  BDFDualSplitting,          // we have this
  BDFConsistentSplitting,    // Karniadakis-style with stronger consistency
  BDFPressureCorrection,     // Goda-style projection correction
  BDFCoupledSolution,        // monolithic saddle-point
  InterpolateAnalyticalSolution,
};
```

| Scheme | Pressure solve | Viscous solve | Convective term |
|---|---|---|---|
| **Dual splitting** | Poisson, Neumann BC from convective + curl-curl trace | Helmholtz, Dirichlet velocity | Explicit (BDF + LF) |
| **Consistent splitting** | Poisson, *consistent* Neumann derived from momentum eq. | Helmholtz | Explicit or implicit |
| **Pressure correction** | Goda-type p* prediction then projection | Helmholtz with predicted p | Explicit, implicit, or linearly implicit |
| **Coupled (monolithic)** | Solved together with velocity as a saddle-point system | Same monolithic solve | Implicit / linearly implicit |

The *consistent* in "consistent splitting" refers to imposing the momentum
equation's normal trace as the Neumann BC for pressure — vs. the dual
splitting's pressure-Neumann derived from `−n·(N + ν ∇×∇×U)`. The two differ in
which terms get pushed into the BC; both are formally high-order accurate but
the consistent variant is reportedly more robust at small `dt`.

### Treatment of the convective term

```
enum class TreatmentOfConvectiveTerm {
  Explicit,           // we have this for dual splitting
  Implicit,
  LinearlyImplicit,   // Picard-style linearisation (cheaper than full Newton)
};
```

Implicit / linearly-implicit convection unlocks much larger CFL but needs a
non-trivial inner solver (we'd need Newton-Krylov, like the implicit vorticity-
streamfunction work we already have in `navier_stokes.py`).

## Spatial discretisation knobs

These are independent of the time integrator and can all be combined:

```
enum class FormulationViscousTerm        { DivergenceFormulation, LaplaceFormulation };
enum class FormulationConvectiveTerm     { DivergenceFormulation, ConvectiveFormulation };
enum class FormulationVelocityDivergenceTerm { Weak, Strong };
enum class FormulationPressureGradientTerm   { Weak, Strong };
enum class InteriorPenaltyFormulation    { SIPG, NIPG };        // we have SIPG
enum class PenaltyTermDivergenceFormulation { Symmetrized, NotSymmetrized };
enum class ContinuityPenaltyComponents   { all, normal };
```

Of note:

- **NIPG** (non-symmetric IPDG) — alternative to our SIPG. Less common but
  doesn't require the penalty parameter to exceed a coercivity threshold.
- **Continuity-penalty** and **divergence-penalty** terms — added to the
  viscous Helmholtz to weakly enforce inter-element continuity of velocity and
  ``∇·U = 0`` respectively. Important for inf-sup stability of equal-order DG
  (which we use — same polynomial order for velocity and pressure).
- **Weak vs. strong** formulation of `∇·U` and `∇p` — affects which terms
  get integrated by parts and what the resulting boundary terms look like.
  Choosing wrong can cause the small-dt instability discussed below.

## Operator decomposition

exadg implements each NS subsystem as a re-usable operator with its own
boundary handling:

```
spatial_discretization/operators/
├── convective_operator       N(U) = ∇·(U⊗U) — both divergence + convective forms
├── viscous_operator          ν Δu — SIPG or NIPG
├── momentum_operator         (γ I − ν Δ) U + N(U) — for implicit advection
├── divergence_operator       ∇·U  — weak or strong
├── gradient_operator         ∇p   — weak or strong
├── projection_operator       Helmholtz augmented with continuity + divergence penalties
├── continuity_penalty_operator   τ_c ⟨[U], [V]⟩
├── divergence_penalty_operator   τ_d ⟨[∇·U], [∇·V]⟩
└── rhs_operator              forcing
```

Our `incompressible_ns.py` currently bakes all of this into a single
`ns_step_2d`. A modular operator layout would let us swap the time integrator
without rewriting the spatial discretisation.

## Preconditioners

```
preconditioners/
├── multigrid_preconditioner_momentum    p-multigrid for the viscous Helmholtz
└── multigrid_preconditioner_projection  multigrid for the pressure Poisson
```

We currently use direct (LU) solves throughout, which is fine for the
``≤ 10⁴`` DOF / field problems we run. For ``> 10⁵`` DOFs we'd need iterative
solvers with multigrid; this is the natural next step if/when we push to 3D
or higher polynomial order.

## Turbulence + non-Newtonian

```
generalized_newtonian_model.{cpp,h}   // viscosity depends on shear rate
turbulence_model.{cpp,h}              // LES / RANS
```

Both extensions are out of scope for our current focus but signal that the
spatial-discretisation core in exadg cleanly supports variable viscosity.

## Key papers behind these schemes

1. **Fehn, Wall, Kronbichler (JCP 2018)** —
   [arXiv:1706.09252](https://arxiv.org/abs/1706.09252) /
   [ScienceDirect S0021999118304157](https://www.sciencedirect.com/science/article/abs/pii/S0021999118304157).
   Identifies that DG-based dual splitting is **unstable at small `dt`** — the
   problem traces to how the velocity-divergence and pressure-gradient terms
   couple at element interfaces. Cure: integrate those terms by parts and
   re-derive the BC; the resulting *consistent* dual splitting is stable
   uniformly in `dt` for both equal-order and mixed-order polynomial spaces.
   This is the paper behind the `BDFConsistentSplitting` option in exadg.

2. **Fehn, Wall, Kronbichler (JCP 2018)** —
   [arXiv:1801.08103](https://arxiv.org/abs/1801.08103). Adds *consistent
   penalty terms* for robustness on under-resolved turbulent incompressible
   flows. Two penalties, both applied in a post-processing projection step
   (separate solve after the momentum step):

       a_D(U, V) = τ_d ∫_K (∇·U)(∇·V) dx        (divergence penalty)
       a_C(U, V) = τ_c ∫_∂_int [U·n][V·n] dS    (continuity penalty, normal-only)

   The post-projection solves ``(M + τ_d D + τ_c C) U_new = M U_int``. Penalty
   parameters come from dimensional analysis (``τ ~ O(h · |U|)`` with ``ζ ~
   O(1)``). Keeps the solver stable on LES-style under-resolved problems
   without changing the underlying time-stepper. Hooks into exadg via the
   ``divergence_penalty_operator`` / ``continuity_penalty_operator``
   / ``projection_operator`` building blocks.

3. **Fehn, Wall, Kronbichler (IJNMF 2019)** —
   [10.1002/fld.4683](https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.4683).
   Matrix-free high-order DG solver for incompressible turbulence. Compares an
   incompressible projection-based formulation against a compressible solver
   run at low Mach number, on canonical turbulence benchmarks (Taylor-Green
   vortex, channel flow). Background reading for if/when we tackle
   turbulent flow + need to think about discretisation efficiency.

## What we have vs. what would be worth reproducing

What we already cover:

- Dual-splitting BDF1 / BDF2 with explicit advection (matches Hesthaven and
  exadg's `BDFDualSplitting`).
- SIPG-form pressure + viscous solves with mixed Dirichlet / Neumann BCs.
- Curved-boundary toggle for smooth obstacles.

Concrete reproduction ideas, in roughly the order they'd matter for our use
case:

1. **Consistent splitting (Fehn 2018)**. Highest payoff. Our current dual
   splitting may already hit the small-`dt` instability on stiff problems;
   switching the BC derivation in the pressure step to the *consistent*
   form would close that gap. The change is local — same Poisson operator,
   different right-hand-side construction.

2. **Continuity / divergence penalty operators**. Even staying with dual
   splitting, adding these stabilisation terms to the viscous step makes the
   solver robust against inf-sup-related instabilities at high Re. Small code
   change, big reliability win.

3. **Pressure correction (Goda)**. Different projection scheme — predict
   pressure with the old solution, correct after the viscous step. Often
   converges to steady state faster than dual splitting. Useful if we
   target steady benchmarks.

4. **Linearly-implicit convection**. Lets us run at much larger CFL on
   convection-dominated problems. Hooks into our existing matrix-free
   Newton-Krylov infrastructure (currently used by the vortex-streamfunction
   implicit solver).

5. **More benchmarks**. exadg ships 21 application directories; the ones with
   well-known reference data we don't have yet:
   `cavity` (lid-driven), `couette`, `backward_facing_step`,
   `flow_past_cylinder` (which is the Volker cylinder we're already aiming at),
   `taylor_green_vortex` (we have the 2D variant — exadg also has the 3D
   reference results), `kelvin_helmholtz`, `orr_sommerfeld` (linear stability).
   These add value mostly as validation rather than new methods.

6. **Multigrid preconditioners**. Only worth pursuing once direct-LU costs
   start dominating — i.e., 3D or much higher `Np`. Distant follow-up.

Pure-stretch items (would meaningfully extend the codebase but require
substantial work):

- **Coupled monolithic solver** — needs the full saddle-point system with
  block-triangular preconditioners. Conceptually clean but a big lift.
- **Turbulence modelling** (LES / RANS) — requires variable viscosity and
  filtering machinery on top of the base discretisation.
- **3D extension** of the whole stack.

For our next session: option (1) — *consistent splitting* — is the most
self-contained improvement and the one most likely to pay off on the kind of
problems we're already running.
