# `aragog.jax`

The `aragog.jax` package contains JAX-traceable replicas of the EOS, phase evaluator, and dSdt right-hand side. They are used to build an analytic Jacobian via `jax.jacrev` and feed it to SUNDIALS CVODE through the registered factory in `aragog.solver.cvode_jax`.

This module is loaded only when `solver.use_jax_jacobian = true` (the production CHILI default). The numpy path in `aragog.solver.entropy_state` remains the reference implementation and is exercised by the standalone tests; both must agree to numerical precision (see [first-principles verification](../../Explanations/verification.md)).

For when and why to opt into the JAX path, see [CVODE and JAX-derived Jacobians](../../Explanations/cvode_jax.md).

| Submodule | Role |
|-----------|------|
| [`aragog.jax.eos`](#aragogjaxeos) | `EntropyEOS_JAX`, the JAX-traceable P-S table loader, and `PhaseState`, the per-cell phase cache. Mirrors the public surface of `aragog.eos.EntropyEOS`. |
| [`aragog.jax.phase`](#aragogjaxphase) | `PhaseParams`, `MeshArrays`, `PhaseProperties`, `FluxOutput`, `compute_fluxes`, `compute_mlt`, `evaluate_phase`. SPIDER-parity two-stage blend, mixing-length convection, and pure-functional flux assembly. |
| [`aragog.jax.solver`](#aragogjaxsolver) | `BoundaryParams`, `SolveResult`. The standalone JAX solve path used by the verification suite; in production the JAX path supplies only the Jacobian and CVODE drives the integration. |
| [`aragog.jax.nondim`](#aragogjaxnondim) | `NonDimScales`. Reference scales used to non-dimensionalise state and time before passing them to the integrator. |

## `aragog.jax.eos`

JAX-traceable P-S equation of state. Loads SPIDER-format two-phase tables and provides `temperature`, `density`, `melt_fraction`, and the latent-heat / phase-boundary derivatives via `jax.jit`-compatible bilinear interpolation.

::: aragog.jax.eos
    options:
      members:
        - EntropyEOS_JAX
        - PhaseState

## `aragog.jax.phase`

Phase-aware property evaluation and flux assembly. `evaluate_phase` performs the SPIDER two-stage tanh blend at one cell; `compute_fluxes` is the pure-functional RHS that returns heat flux at basic nodes, internal heating at staggered nodes, eddy diffusivity, and capacitance.

::: aragog.jax.phase
    options:
      members:
        - PhaseParams
        - MeshArrays
        - PhaseProperties
        - FluxOutput
        - compute_fluxes
        - compute_mlt
        - evaluate_phase

## `aragog.jax.solver`

Standalone JAX solve path used by the verification suite. The production CHILI path uses CVODE driven by `aragog.solver.entropy_solver` and only borrows the Jacobian from `jax.jacrev`; this module's `solve_entropy` is kept for parity tests and gradient-based sensitivity studies.

::: aragog.jax.solver
    options:
      members:
        - BoundaryParams
        - SolveResult

## `aragog.jax.nondim`

Reference scales used to non-dimensionalise the state vector and time before they enter the integrator. Mirrors the historical SPIDER scaling conventions; values are read from `Parameters.scalings`, which is forced to unity in production runs.

::: aragog.jax.nondim
    options:
      members:
        - NonDimScales
