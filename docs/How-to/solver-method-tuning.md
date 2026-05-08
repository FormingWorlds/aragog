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

The relevant keys are `solver.atol` (absolute) and `solver.rtol` (relative); both are floored at $10^{-8}$ inside the solver. Suggested starting points:

| Regime | `atol` | `rtol` | Notes |
|--------|--------|--------|-------|
| Production coupled run, CVODE + JAX | `1e-10` | `1e-10` | Matches the PROTEUS schema default in `proteus.config._interior`. Tight enough that energy-conservation checks close. |
| Standalone smoke test | `1e-7` | `1e-7` | Lets the integrator march through the rheological transition in seconds. Acceptable for first-run sanity checks; not for paper plots. |
| Tight verification or parity test | `1e-10` | `1e-10` | The floor; useful for SPIDER bit-parity diagnostics. Wall time roughly doubles. |

When the integrator stalls (status code -1, `dt_actual` short of `end_time`), loosen `atol`/`rtol` first. If the stall persists, the issue is usually a sharp solidus/liquidus crossing, not the integrator class; consider enabling [`phi_step_cap`](phi-step-cap.md).

## CVODE specifics

CVODE is selected by default and is the only path that supports the JAX-traced analytic Jacobian (`use_jax_jacobian = true`). With JAX absent, CVODE falls back to a finite-difference Jacobian: correct, but $O(N)$ RHS evaluations per Jacobian build and noisier on stiff profiles. See the [CVODE and JAX explainer](../Explanations/cvode_jax.md).

CVODE also accepts a SUNDIALS root function for melt-fraction step capping; see [`phi_step_cap` how-to](phi-step-cap.md). The scipy fallbacks register the equivalent capping logic as a `solve_ivp` event with `terminal=True`, so the same TOML config behaves consistently.

## Radau and BDF specifics

Both scipy paths use `solve_ivp` with `dense_output=False` and a `max_step` cap that the solver shrinks adaptively near phase boundaries. They do not support the JAX-traced Jacobian; a Jacobian is computed by scipy via finite differences when needed.

The `phase-aware max_step` reduction activates at the same trigger conditions on all three integrators; it is independent of the integrator class.

## Cross-references

- [CVODE and JAX explainer](../Explanations/cvode_jax.md): why the analytic Jacobian helps and how `set_jax_cvode_factory` is wired.
- [`phi_step_cap` how-to](phi-step-cap.md): SUNDIALS-rootfn capping of $|\Delta\phi|$ across a step.
- [Installation: production solver path](installation.md#production-solver-path-sundials-cvode-jax-analytic-jacobian): runtime requirement for `use_jax_jacobian = true`.
