# Capping melt-fraction steps with `phi_step_cap`

The `energy.phi_step_cap` knob limits the per-call change in volume-weighted mean melt fraction $\Phi_\mathrm{global}$. It is implemented as a SUNDIALS root function that intercepts the integrator at the exact step where $|\Delta\Phi_\mathrm{global}|$ first reaches the cap. Use it when the rheological transition is being traversed quickly enough that the integrator's adaptive step would otherwise overshoot a physically meaningful melt-fraction excursion in a single call.

## When to enable it

| Situation | Recommended `phi_step_cap` |
|-----------|----------------------------|
| Standard CHILI run (1 to 10 $M_\oplus$) | `0.05` |
| Mushy-zone oscillations in early snapshots | `0.001` to `0.01` |
| Static-Zalmoxis parity diagnostic | `0.0` (disabled) |
| Pure cooling with smooth solidus crossing | `0.0` (disabled) |

The default in `src/aragog/parser.py` is `0.0` (cap disabled). Production CHILI configs (`input/chili/*.toml`) pin it to `0.05` to keep the rheological transition trackable when atmospheric flux or interior structure changes quickly.

## How it works

When `phi_step_cap > 0` and the mantle straddles the rheological transition, `EntropySolver._solve_cvode` registers a `_PhiCapRootFunction` (see `src/aragog/solver/entropy_solver.py`) on the CVODE call. The function evaluates

$$
g(t, y) = \mathrm{cap} - |\Phi_\mathrm{global}(t, y) - \Phi_\mathrm{global}(t_\mathrm{start})|
$$

at every internal CVODE step. CVODE returns at the exact zero-crossing (flag `CV_ROOT_RETURN`), so the integrator never overshoots the cap.

The scipy fallbacks (`Radau`, `BDF`) register the equivalent logic as a `solve_ivp` event with `terminal=True` and `direction=-1`. Both paths see the same TOML config and behave consistently up to integrator class.

The mushy-zone gating (`near_liq`, `near_sol`, `in_mushy` checks at `entropy_solver.py:1869`) prevents the rootfn from arming on profiles that are far from the rheological transition: in those regimes, the cap would never fire and the rootfn evaluation is wasted work.

## Picking a value

`0.05` is a useful upper bound for 1 $M_\oplus$ runs: large enough that the cap fires only on genuinely fast transitions, small enough that one CVODE call corresponds to a manageable physical state change. Tighten to `0.01` or `0.001` only if the first few snapshots show melt-fraction oscillations that suggest mushy-zone overshoot.

`0.0` (disabled) is appropriate for static-Zalmoxis diagnostics where the structure is held fixed and the integrator's native step control is sufficient, or for runs where you specifically want to characterise the unconstrained step distribution.

## Diagnostic logging

When the cap fires, the solver logs an INFO line with the firing time and the resulting $\Delta\Phi$. With several rapid fires per call, the logs make the rheological-transition traversal visually obvious in `aragog.log`.

## Common pitfalls

- **`phi_step_cap = 0.05` plus very loose tolerances**: with `atol = rtol = 1e-5`, CVODE may already step coarsely enough that the cap never arms. Tighten tolerances to `1e-8` or below for cap-driven runs.
- **`phi_step_cap` set on a fully-molten or fully-solid profile**: the gating skips registration, so the value is silently inert. This is intentional, not a bug.
- **PROTEUS retry ladder + cap**: if `solve()` returns a partial trajectory because the cap fired, the PROTEUS wrapper accepts that as a successful step and continues. The cap is not a failure mode.

## Cross-references

- [Solver method tuning](solver-method-tuning.md): the integrator selection that determines which rootfn / event backend is used.
- [Heat-transport explainer](../Explanations/heat_transport.md): the mushy-zone smoothing that motivates the cap.
- [Configuration: `[energy]`](configuration.md#energy): the full `phi_step_cap` schema entry.
