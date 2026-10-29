# Capping melt-fraction steps with `phi_step_cap`

The `energy.phi_step_cap` knob limits the per-call change in volume-weighted mean melt fraction $\Phi_\mathrm{global}$. It is implemented as a SUNDIALS root function that intercepts the integrator at the exact step where $|\Delta\Phi_\mathrm{global}|$ first reaches the cap. Use it when the rheological transition is being traversed quickly enough that the integrator's adaptive step would otherwise overshoot a physically meaningful melt-fraction excursion in a single call.

## When to enable it

| Situation | Recommended `phi_step_cap` |
|-----------|----------------------------|
| Standard PROTEUS run (1 to 10 $M_\oplus$) | `0.05` |
| Mushy-zone oscillations in early snapshots | `0.001` to `0.01` |
| Static-Zalmoxis parity diagnostic | `0.0` (disabled) |
| Pure cooling with smooth solidus crossing | `0.0` (disabled) |

The default in `src/aragog/parser.py` is `0.0` (cap disabled). Production PROTEUS configs typically pin it to `0.05` to keep the rheological transition trackable when atmospheric flux or interior structure changes quickly.

## How it works

When `phi_step_cap > 0` and the mantle straddles the rheological transition, `EntropySolver._solve_cvode` registers a `_PhiCapRootFunction` (see `src/aragog/solver/entropy_solver.py`) on the CVODE call. The function evaluates

$$
g(t, y) = \mathrm{cap} - |\Phi_\mathrm{global}(t, y) - \Phi_\mathrm{global}(t_\mathrm{start})|
$$

at every internal CVODE step. CVODE returns at the exact zero-crossing (flag `CV_ROOT_RETURN`), so the integrator never overshoots the cap.

The scipy fallbacks (`Radau`, `BDF`) register the equivalent logic as a `solve_ivp` event with `terminal=True` and `direction=-1`. Both paths see the same TOML config and behave consistently up to integrator class.

The mushy-zone gating (the `near_liq` / `near_sol` / `in_mushy` checks inside `EntropySolver.solve()` before rootfn registration) prevents the rootfn from arming on profiles that are far from the rheological transition: in those regimes, the cap would never fire and the rootfn evaluation is wasted work.

## Picking a value

`0.05` is a useful upper bound for 1 $M_\oplus$ runs: large enough that the cap fires only on fast transitions, small enough that one CVODE call corresponds to a manageable physical state change. Tighten to `0.01` or `0.001` only if the first few snapshots show melt-fraction oscillations that suggest mushy-zone overshoot.

`0.0` (disabled) is appropriate for static-Zalmoxis diagnostics where the structure is held fixed and the integrator's native step control is sufficient, or for runs where you specifically want to characterise the unconstrained step distribution.

## Diagnostic logging

When the cap fires, the solver logs an INFO line with the firing time and the resulting $\Delta\Phi$. With several rapid fires per call, the logs make the rheological-transition traversal visually obvious in `aragog.log`.

## Common pitfalls

- With `phi_step_cap = 0.05` and loose tolerances (`atol = rtol = 1e-5`), CVODE may step coarsely enough that the cap never arms. Tighten tolerances to `1e-8` or below for cap-driven runs.
- When `phi_step_cap` is set on a fully molten or fully solid profile, the gating skips registration so the value is silently inert. This is intentional, not a bug.
- Under the PROTEUS retry ladder, if `solve()` returns a partial trajectory because the cap fired, the PROTEUS wrapper accepts that as a successful step and continues. The cap is not a failure mode.

## Runnable example: cooling through the solidus with `phase_boundary_cap = 'rate'`

In addition to `phi_step_cap`, `energy.phase_boundary_cap = 'rate'` (the default) segments CVODE steps as cells cross the liquidus and solidus and relaxes the $100\ \mathrm{yr}$ step ceiling once every cell cools to or below the solidus (see [Why `phi_step_cap` is a SUNDIALS root function](../Explanations/phi_step_cap_rationale.md#how-phase_boundary_cap-rate-segments-steps-through-the-solidus)).

The script below initializes a 12-node Earth-sized mantle from `src/aragog/cfg/abe_solid.toml` on the SPIDER phase-boundary lookup tables (`ARAGOG_TEST_EOS_DIR`), sets the initial entropy uniformly to $S_{\mathrm{sol}}[-1] + 5.0\ \mathrm{J\,kg^{-1}\,K^{-1}}$ ($880.54\ \mathrm{J\,kg^{-1}\,K^{-1}}$, so the outermost cell starts $5.0\ \mathrm{J\,kg^{-1}\,K^{-1}}$ above its solidus $875.54\ \mathrm{J\,kg^{-1}\,K^{-1}}$ while deeper cells are already below their solidus up to $1364.69\ \mathrm{J\,kg^{-1}\,K^{-1}}$), and integrates for $200\ \mathrm{kyr}$ under a prescribed surface heat flux of $200.0\ \mathrm{W\,m^{-2}}$:

```python
import dataclasses
import os
import numpy as np
from aragog.eos.entropy import EntropyEOS
from aragog.parser import Parameters
from aragog.solver import EntropySolver

eos_dir = os.environ['ARAGOG_TEST_EOS_DIR']
eos = EntropyEOS(eos_dir)
params = Parameters.from_file('src/aragog/cfg/abe_solid.toml')
params.solver = dataclasses.replace(
    params.solver, start_time=0.0, end_time=2.0e5, atol=1.0e-8, rtol=1.0e-8, max_steps=1500
)
params.mesh = dataclasses.replace(
    params.mesh, inner_radius=3480000.0, number_of_nodes=12, eos_method=1
)
params.boundary_conditions = dataclasses.replace(
    params.boundary_conditions,
    outer_boundary_condition=4,
    outer_boundary_value=200.0,
    inner_boundary_condition=2,
    inner_boundary_value=0.0,
    core_bc='quasi_steady',
)
params.energy = dataclasses.replace(
    params.energy,
    solver_method='cvode',
    phase_boundary_cap='rate',
    phi_step_cap=None,
    convective_threshold=0.0,
    radionuclides=False,
)

solver = EntropySolver(params, entropy_eos=eos)
solver.initialize()
p_stag = np.asarray(solver.evaluator.mesh.staggered_pressure).ravel()
s_sol = np.asarray(eos.solidus_entropy(p_stag)).ravel()
solver.set_initial_entropy(np.full_like(s_sol, float(s_sol[-1] + 5.0)))
solver.solve()
state = solver.get_state()
sol = solver.solution
print(f'status={state.status}, steps={sol.get("cvode_nst")}, segments={len(sol.segments)}')
print(f'first_cap={sol.segments[0][2]:.1f} yr, second_cap={sol.segments[1][2]:.2f} yr, final_cap={sol.segments[-1][2]}')
print(f'S_end in [{state.S_final.min():.2f}, {state.S_final.max():.2f}] J/(kg K), T_end in [{state.T_stag.min():.2f}, {state.T_stag.max():.2f}] K')
```

The public `SolverOutput` returned by `solver.get_state()` exposes the final physical state (`status`, `S_final`, `T_stag`, `phi_stag`, `P_stag`), while no dedicated accessor exists on `SolverOutput` for the per-segment root records (`segments`) or cumulative CVODE step count (`cvode_nst`), which are read from the public `solver.solution` `OptimizeResult` property.
Running this script prints `status=0, steps=1034, segments=12`, with `first_cap=100.0 yr` in the initial segment (while the top cell is still above its solidus, $\phi_{\max} = 0.000896$), `second_cap=163.77 yr` at $t = 1611.87\ \mathrm{yr}$ once the top cell crosses the solidus ($\phi_{\max} = 0.0$) and the $100\ \mathrm{yr}$ ceiling relaxes inside the solidus stiff zone, and `final_cap=inf` over the final 9 sub-solidus segments (ending with $S_{\mathrm{end}} \in [587.31,\; 880.39]\ \mathrm{J\,kg^{-1}\,K^{-1}}$ and $T_{\mathrm{end}} \in [1735.27,\; 3316.24]\ \mathrm{K}$).

## Cross-references

- [Solver method tuning](solver-method-tuning.md): the integrator selection that determines which rootfn / event backend is used.
- [Heat-transport explainer](../Explanations/heat_transport.md): the mushy-zone smoothing that motivates the cap.
- [Configuration: `[energy]`](configuration.md#energy): the full `phi_step_cap` schema entry.
