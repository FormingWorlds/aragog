# `aragog.jax`

The `aragog.jax` package contains JAX-traceable replicas of the EOS, phase evaluator, and dSdt RHS. They are used to build an analytic Jacobian via `jax.jacrev` and feed it to SUNDIALS CVODE through the registered factory in `aragog.solver.cvode_jax`.

This module is loaded only when `solver.use_jax_jacobian = true` (the production CHILI default). The numpy path in `aragog.solver.entropy_state` remains the reference implementation and is exercised by the standalone tests; both must agree to numerical precision (see [first-principles verification](../../Explanations/verification.md)).

For when and why to opt into the JAX path, see [CVODE and JAX-derived Jacobians](../../Explanations/cvode_jax.md).

| Name | Role |
|------|------|
| `EntropyEOS_JAX` | JAX-traceable P-S table loader and bilinear interpolator. Mirrors the public surface of `aragog.eos.EntropyEOS`. |
| `PhaseState` | NamedTuple of cached phase quantities at one cell (entropy, T, density, $c_p$, $\alpha$, $k$). |
| `PhaseProperties` | NamedTuple of mushy-zone two-phase blends. |
| `FluxOutput` | NamedTuple returned by `compute_fluxes`: heat flux at basic nodes, heating at staggered nodes, eddy diffusivity, capacitance. |
| `PhaseParams` | `equinox.Module` of static phase parameters (gravitational separation, mixing toggles, grain size). |
| `MeshArrays` | `equinox.Module` of mesh arrays needed by the JAX RHS (basic and staggered radii, volumes, mixing length, pressure, gravity profile). |
| `compute_fluxes` | Pure-functional flux assembly: takes $S$ (and time and heating), returns the full `FluxOutput`. |
| `compute_mlt` | Mixing-length convective velocity from the local entropy gradient. |
| `evaluate_phase` | SPIDER-parity two-stage blend at one cell. |
| `BoundaryParams` | Surface-BC parameters threaded through the JAX RHS (UTBL toggle, emissivity, equilibrium temperature). |
| `SolveResult` | NamedTuple returned by the JAX standalone solve path. |
| `NonDimScales` | Reference scales used to non-dimensionalise state and time before passing them to the integrator. |

::: aragog.jax
