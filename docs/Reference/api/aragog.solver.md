# `aragog.solver`

The `aragog.solver` package contains the time-integration driver, the per-RHS state container, the boundary-condition handler, and the output dataclass.

The public surface re-exported from `aragog.solver` is:

| Name | Role |
|------|------|
| `EntropySolver` | The ODE driver. Owns the integrator dispatch (Radau / BDF / CVODE), the nondimensionalisation layer, the retry-ladder hooks, and the `SolverOutput` post-processing. |
| `EntropyState` | Per-RHS state container. Computes phase, density, $T$, $c_p$, $\alpha$, $k$, the four flux contributions, and the internal heating at each call. |
| `BoundaryConditions` | Holds the BC settings that `EntropySolver` reads, plus a standalone flux dispatch: surface (grey-body, UTBL, prescribed flux/T) and inner (prescribed flux/T). Core cooling (`inner_boundary_condition = 1`) is applied inside `EntropySolver`. |
| `SolverOutput` | Dataclass returned by `EntropySolver.get_state()`. Carries the staggered-node profiles, basic-node fluxes, scalar diagnostics, the per-call energy integrals (`step_dE_F_int_J`, `step_dE_F_cmb_J`, `step_dE_Q_radio_J`, `step_dE_Q_tidal_J`, plus frozen-mass `step_dE_Q_radio_cons_J`/`step_dE_Q_tidal_cons_J` and the entropy-ODE solver residual `step_solver_residual_J`), the conservation-grade integrated mantle enthalpy `E_state_cons` (frozen-mass), and the integration status flag. See [Energy diagnostics](../../Explanations/energy_diagnostics.md) for the conservation-residual interpretation. |
| `SECS_PER_YEAR` | Module-level constant `scipy.constants.Julian_year` ($31{,}557{,}600$ s). The ODE is integrated in years; converting flux divergence (J/kg/K/s) to per-year requires this factor. |

The solver checks the CMB closure of each `energy_balance`, `gradient` and `core_module` call ([ledger check of the CMB closure](../../Explanations/core_bc.md#ledger-check-of-the-cmb-closure)):

- `aragog.solver.entropy_solver.CMBClosureError`, a `RuntimeError`, is raised when a ledger miss is above 10 times its bound or the node slope $(\partial T/\partial S)_P$ is not finite and positive. The solver then puts back the solution it held at the start of the call.
- `aragog.solver.entropy_state.check_cmb_convective_flux` checks the convective flux of the CMB basic node in every `EntropyState.update` against the inputs of its eddy diffusivity rule ([Eddy diffusivity at the CMB node](../../Explanations/heat_transport.md#eddy-diffusivity-at-the-cmb-node)); it raises `aragog.solver.entropy_state.CMBEddyFluxError`, a `RuntimeError`, and names the part that fails. `EntropyState.cmb_eddy_diffusivity_inputs` holds the 2 inputs of the last update: the mixing-length eddy diffusivity of node 1 before the floor and the floor of node 0, both in m$^2$ s$^{-1}$. The check also requires the flux to equal the rule within $10^{-12}$ relative.
- `EntropySolver.cmb_ledger` is a dict with the terms of the last checked call: `r` and `bound` (J in `energy_balance` and `gradient`, K in `core_module`), `rel` (dimensionless) and `excluded` (true when the check was skipped). It is `None` after a call that the check does not cover.

::: aragog.solver
