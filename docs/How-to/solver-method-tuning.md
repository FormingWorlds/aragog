# Solver method tuning

Aragog dispatches stiff time integration through `_EnergyParameters.solver_method` (see `src/aragog/parser.py`). Three values are accepted: `"cvode"` (default), `"radau"`, and `"bdf"`. This page is the practical recipe for picking one and tuning the associated tolerances.

!!! note "Default is CVODE"
    Production PROTEUS runs rely on `solver_method = "cvode"` with `use_jax_jacobian = true`. The other two values are fallback paths, not equivalents.

## Available integrators

| `solver_method` | Backend | When to use |
|-----------------|---------|-------------|
| `"cvode"` | SUNDIALS CVODE via `scikits_odes` | Default for all production runs. Modified Newton with cached Jacobian; matches SPIDER's solver class. Required for the JAX-traced analytic Jacobian path. |
| `"radau"` | scipy `solve_ivp(method="Radau")` | Pure-Python fallback when `scikits_odes` is not installed. Fully implicit; reliable on smooth profiles but stalls at the rheological transition on long cooling runs. |
| `"bdf"` | scipy `solve_ivp(method="BDF")` | Pure-Python fallback. Cheaper per step than Radau, weaker at sharp Jacobians. Use only for quick smoke tests. |

If `solver_method = "cvode"` is selected and `scikits_odes` is missing, the solver logs a warning and falls back to `Radau`. See [Installation: production solver path](installation.md#production-solver-path-sundials-cvode-jax-analytic-jacobian).

## Picking tolerances

The relevant keys are `solver.atol` (absolute) and `solver.rtol` (relative); `atol` is floored at $10^{-8}$ inside the solver, and `rtol` is used as given. Suggested starting points:

| Regime | `atol` | `rtol` | Notes |
|--------|--------|--------|-------|
| Production coupled run, CVODE + JAX | `1e-8` (from `atol_temperature_equivalent`) | `1e-8` | Matches the PROTEUS defaults for Aragog in `proteus.config._interior`. Tight enough that energy-conservation checks close. |
| Standalone smoke test | `1e-7` | `1e-7` | Lets the integrator march through the rheological transition in seconds. Acceptable for first-run sanity checks; not for paper plots. With CVODE, logs the tolerance warning. |
| Tight verification or parity test | `1e-10` (applied as `1e-8`) | `1e-10` | Useful for SPIDER bit-parity diagnostics. Wall time roughly doubles. |

When a CVODE call stalls or runs far slower than expected, check whether CVODE is locked at the switch $dS/dr = 0$ (the convective mask and the $\kappa_h$ floor). In a lock the step stays at one small value and the entropy difference across one basic node oscillates with period 2 around zero; in the measured locks CVODE used about 1.4 RHS evaluations per step and evaluated its Jacobian about every 50 to 60 steps (with INFO logging, the `CVODE stats` line shows `rhs_wrap/nst` and a small `last_step`). A lock can run for a long time without an error, or end in `CV_TOO_MUCH_WORK` once `max_steps` is used up in one output interval. Locks were seen at `rtol` `1e-6` and not at `1e-8`; the remedy is `rtol` and `atol` at `1e-8`, and a CVODE run logs one warning when `solver.rtol` is above `1e-8`. The counts below come from the energy_balance isentropic start of `tests/test_phase_boundary_cap.py` (24 nodes, 200 yr, fixed cap unless stated), with the initial entropy drawn uniformly between 8000 and 8400 J/kg/K for each run, on macOS arm64. A run counts as locked when it has not reached 200 yr after 600 s of wall time: every run was screened at 60 s, and each run that missed that was run again alone to 600 s, by which time a locked run had used about $1.7 \times 10^5$ RHS evaluations. With `rtol` = `atol` = `1e-6`, CVODE locked in 12 of 1133 runs, in 4 of 474 on the quasi_steady core and in 2 of 286 with the rate cap, where 2 more runs finished only after 233 s and 346 s. With both at `1e-5`, none of 80 runs locked (too few runs to show it safe), but the median run needed about $1.6 \times 10^4$ RHS evaluations, against about $1.3 \times 10^3$ at `1e-8`. With both at `1e-8`, CVODE locked in none of 1026, 627 and 1224 runs for three setups (fixed and rate cap on the energy_balance core, fixed cap on quasi_steady). With `atol` at `1e-8` and `rtol` at `2e-8`, `5e-8` or `1e-7`, the values of the PROTEUS retry ladder, it locked in none of the runs: 1308 to 1836 for each value and core with the fixed cap, and 620 and 636 with the rate cap at `5e-8` and `1e-7`. These counts hold for the 24-node standalone case, not for a coupled solve. On CI the AVX-512 numpy path makes the energy_balance case lock at `1e-6`. For a stall without this pattern, raise `max_steps`; the PROTEUS retry ladder raises `max_steps` and relaxes `rtol`, and the relaxed `rtol` can itself be above `1e-8` (see [PROTEUS coupling](proteus_coupling.md)). If the stall persists with an effective `rtol` of `1e-8`, the issue is usually a sharp solidus/liquidus crossing, not the integrator class; consider enabling [`phi_step_cap`](phi-step-cap.md).

## CVODE specifics

CVODE is selected by default and is the only path that supports the JAX-traced analytic Jacobian (`use_jax_jacobian = true`). With JAX absent, CVODE falls back to a finite-difference Jacobian: correct, but $O(N)$ RHS evaluations per Jacobian build and noisier on stiff profiles. See the [CVODE and JAX explainer](../Explanations/cvode_jax.md).

CVODE also accepts a SUNDIALS root function for melt-fraction step capping; see [`phi_step_cap` how-to](phi-step-cap.md). The scipy fallbacks register the equivalent capping logic as a `solve_ivp` event with `terminal=True`, so the same TOML config behaves consistently.

## Radau and BDF specifics

Both scipy paths use `solve_ivp` with `dense_output=False` and a 1 yr `max_step` near phase boundaries. They do not support the JAX-traced Jacobian; a Jacobian is computed by scipy via finite differences when needed.

With `phase_boundary_cap = "fixed"` the 1 yr `max_step` activates at the same trigger on all three integrators. The default `"rate"` runs only on CVODE: it arms within the larger of `phase_boundary_entropy_margin` and the stiff-zone half-width and integrates the call in segments; the scipy paths fall back to the 1 yr step.

## Cross-references

- [CVODE and JAX explainer](../Explanations/cvode_jax.md): why the analytic Jacobian helps and how `set_jax_cvode_factory` is wired.
- [`phi_step_cap` how-to](phi-step-cap.md): SUNDIALS-rootfn capping of $|\Delta\phi|$ across a step.
- [Installation: production solver path](installation.md#production-solver-path-sundials-cvode-jax-analytic-jacobian): runtime requirement for `use_jax_jacobian = true`.
