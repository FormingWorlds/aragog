# `aragog.solver`

The `aragog.solver` package contains the time-integration driver, the per-RHS state container, the boundary-condition handler, and the output dataclass.

The public surface re-exported from `aragog.solver` is:

| Name | Role |
|------|------|
| `EntropySolver` | The ODE driver. Owns the integrator dispatch (Radau / BDF / CVODE), the nondimensionalisation layer, the retry-ladder hooks, and the `SolverOutput` post-processing. |
| `EntropyState` | Per-RHS state container. Computes phase, density, $T$, $c_p$, $\alpha$, $k$, the four flux contributions, and the internal heating at each call. |
| `BoundaryConditions` | Surface (grey-body, UTBL, prescribed flux/T) and inner (core cooling, prescribed flux/T) BC dispatch. |
| `SolverOutput` | Dataclass returned by `EntropySolver.get_state()`. Carries the staggered-node profiles, basic-node fluxes, scalar diagnostics, and the integration status flag. |

::: aragog.solver
