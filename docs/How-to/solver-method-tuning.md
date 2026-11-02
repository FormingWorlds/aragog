# Solver method tuning

Use this page to choose an integrator, set its tolerances, get a stalled run moving, and decide what to do when a run leaves the EOS tables. `_EnergyParameters.solver_method` (see `src/aragog/parser.py`) selects the integrator for the stiff time integration. It accepts 3 values: `"cvode"` (the default), `"radau"` and `"bdf"`.

!!! note "Default is CVODE"
    Normal PROTEUS runs use `solver_method = "cvode"` with `use_jax_jacobian = true`. The other two values are fallback paths, not equivalents.

## Choose an integrator

| `solver_method` | Solver | When to use |
|-----------------|---------|-------------|
| `"cvode"` | SUNDIALS CVODE via `scikits_odes` | The default for all normal runs. Modified Newton with a cached Jacobian, the solver class of SPIDER. Required for the JAX-traced analytic Jacobian path. |
| `"radau"` | scipy `solve_ivp(method="Radau")` | Pure-Python fallback when `scikits_odes` is not installed. Fully implicit. Reliable on smooth profiles, but stalls at the rheological transition on long cooling runs. |
| `"bdf"` | scipy `solve_ivp(method="BDF")` | Pure-Python fallback. Cheaper per step than Radau, weaker at sharp Jacobians. Use it only for quick smoke tests. |

If you select `"cvode"` and `scikits_odes` is missing, the solver logs a warning and falls back to `Radau`. See [Installation: production solver path](installation.md#production-solver-path-sundials-cvode-jax-analytic-jacobian).

## Pick tolerances

Start with `solver.atol` (absolute) and `solver.rtol` (relative) both at `1e-8`. Loosen them only for a quick check, because a loose tolerance can lock CVODE (see below). The solver floors `atol` at $10^{-8}$ and uses `rtol` as given.

| Use | `atol` | `rtol` | What you get |
|--------|--------|--------|-------|
| Coupled run, CVODE + JAX | `1e-8` (from `atol_temperature_equivalent`) | `1e-8` | The PROTEUS defaults for Aragog in `proteus.config._interior`. Tight enough that the energy-conservation checks close. |
| Standalone smoke test | `1e-7` | `1e-7` | Lets the integrator march through the rheological transition in seconds. Acceptable for a first-run check, not for paper plots. With CVODE it logs the tolerance warning. Not measured for locks. |
| Tight verification or parity test | `1e-10` (applied as `1e-8`) | `1e-10` | For SPIDER bit-parity diagnostics. Wall time roughly doubles. |

### When a CVODE call stalls

A call that stalls or runs far slower than expected can be locked at the convective switch, which is the convective mask together with the $\kappa_h$ floor. Check for a lock first, because its remedy is different.

Signs of a lock:

- The step stays at one small value.
- The entropy difference across one basic node oscillates with period 2 (around zero in the measured locks, all at $D = 0$).
- In the measured locks CVODE used about 1.4 RHS evaluations per step (1.14 in a recorded run on the quasi_steady core) and evaluated its Jacobian about every 50 to 60 steps. With INFO logging, the `CVODE stats` line shows `rhs_wrap/nst` and a small `last_step`.

A lock can run for a long time without an error, or end in `CV_TOO_MUCH_WORK` once `max_steps` is used up in one output interval.

What to do:

1. For a lock at the default `convective_threshold = 0`, set `rtol` and `atol` to `1e-8`. Locks were seen at `rtol` `1e-6` and not at `1e-8`. A CVODE run logs one warning when `solver.rtol` is above `1e-8`.
2. For a stall without this pattern, raise `max_steps`. The PROTEUS retry ladder raises `max_steps` and relaxes `rtol`, and the relaxed `rtol` can itself be above `1e-8` (see [PROTEUS coupling](proteus_coupling.md)).
3. If the stall persists with an effective `rtol` of `1e-8`, the cause is usually a sharp solidus or liquidus crossing, not the integrator class. Consider enabling [`phi_step_cap`](phi-step-cap.md).
4. With `convective_threshold` above 0, the cause can also be a node whose gradient stays near $dS/dr = 0$. The default 0 does not have that jump. CVODE also stalled at `1e-8` at $D > 0$, and no remedy was measured there (see [Heat transport](../Explanations/heat_transport.md#convection-mixing-length-theory)).

The measured counts behind this advice are in [Lock measurements](#lock-measurements).

## CVODE

- CVODE is the default and the only path that supports the JAX-traced analytic Jacobian (`use_jax_jacobian = true`).
- With no JAX CVODE factory registered, CVODE uses a finite-difference Jacobian. In a PROTEUS run that also happens when JAX is not importable. It is correct, but takes $O(N)$ RHS evaluations per Jacobian build and is noisier on stiff profiles. See the [CVODE and JAX explainer](../Explanations/cvode_jax.md).
- The tolerance warning also names a configured `atol` above `1e-8`, but only on the `energy_balance` core, the one setup whose lock data show a loose `atol` adding to the lock. No lock data exist for the other core boundary conditions, `core_module` included, so the warning makes no `atol` claim for them.
- CVODE accepts a SUNDIALS root function for melt-fraction step capping; see the [`phi_step_cap` how-to](phi-step-cap.md).

## Radau and BDF

- Both use `solve_ivp` with `dense_output=False` and a 1 yr `max_step` near phase boundaries.
- They do not support the JAX-traced Jacobian. scipy computes a Jacobian by finite differences when needed.
- They register the melt-fraction capping logic as a `solve_ivp` event with `terminal=True`, so the same TOML config behaves consistently.
- With `phase_boundary_cap = "fixed"` the 1 yr `max_step` activates at the same trigger on all 3 integrators.
- The default `"rate"` runs only on CVODE: it arms within the larger of `phase_boundary_entropy_margin` and the stiff-zone half-width and integrates the call in segments. The scipy paths fall back to the 1 yr step.

## Leaving the EOS tables

The entropy state of CVODE has no lower bound, but the EOS tables stop at a minimum entropy $S_\mathrm{min}$. Below it every lookup is clamped to the table edge, so a cell keeps losing entropy at a fixed temperature. This happens when a prescribed surface flux, fixed over a coupling step, drains the top cell after it crystallises.

Two controls deal with this. The guard needs no setting and fails a call that has left the tables. The floor is off by default and stops a call before it leaves them.

Trial steps and Newton iterates of the integrator can evaluate the right-hand side below $S_\mathrm{min}$, with clamped lookups, with the floor on or off and at any margin.

### EOS table guard

The guard is a safety net. It does not set the step and does not remove the cause of a drain.

**What it does.** With the rheology off, a call fails when a cell entropy lies outside the global entropy range of the loaded tables by more than $10^{-3}$ J kg$^{-1}$ K$^{-1}$, in an accepted or root-search state. With the rheology on, an invariant raises instead.

**What you see.**

- The call ends with status -1 after it has run to its end, with an ERROR message that names the guard, the cell, its entropy and the table range.
- The result holds the states from before the first one outside, or the start state for a start outside. `dt_actual` is the time from the call start to the last state kept. The energy integrals are zero, and `F_cmb_step_avg` is `None`.
- `cvode_flag_name` is `EOS_TABLE_GUARD`, and `cvode_flag` keeps its value, which is 0 on the scipy paths.
- Before that, the EOS lookup logs an out-of-range WARNING on the 1st, 10th, 100th, ... occurrence of each lookup. For an EOS that a caller builds with `strict_range=True`, the lookup raises instead: on a scipy path the error reaches the caller, and on CVODE the solver call reports it.

**What to do.**

- Retry the call with a shorter step. This does not help for a start outside, and the message says so.
- In PROTEUS the retry ladder halves the step and keeps the previous interior state when the ladder fails. It stops the run at the third such failure in a row.
- Look at the coupling step: a flux fixed over a step is what drains the top cell.

**What it does not judge.**

- The extrapolated basic nodes: with the floor off they can lie below $S_\mathrm{min}$ in a returned state.
- The padded rows of a pressure column.
- A run with constant properties or without tables, and a call that does not advance.

**A cell at a table edge.** It passes until it has moved past the edge by the tolerance, so it fails after about $10^{-3}$ J kg$^{-1}$ K$^{-1}$ over its outward rate.

**Where the guard runs.** Only `EntropySolver.solve` has the guard. The direct JAX integration in `aragog.jax.solver` does not.

### EOS floor

Turn the floor on in a standalone run when a drain must stop at a known entropy, with a state that stays inside the table. PROTEUS has no key for it, so PROTEUS runs have the floor off.

**How.** Set `energy.eos_floor_margin` $m$ to a positive value below the entropy span of the tables. The tested value is 100 J kg$^{-1}$ K$^{-1}$. The default is $m = 0$, which is off, as is $m \le 0$.

**What it does.** The floor is one more component of the step-cap root function. It ends the call when the entropy it watches reaches $S_\mathrm{min} + m$, or $S_\mathrm{min} + m/2$ when the call starts below $S_\mathrm{min} + m$.

- It watches the smaller of the top basic-node entropy, which sets the surface temperature, and the smallest cell entropy. The CMB basic node is not watched.
- With `top_node = 'limited'` and with a prescribed surface temperature (`outer_boundary_condition = 5`), the top basic node is not the linear extrapolation of the cells, so the floor watches the top cell in its place.

**What you see.**

- At a stop, CVODE returns status 0 with a shorter `dt_actual`, and scipy returns status 1 at the event. A WARNING logs the stop and its level.
- A refused call returns status 1, `dt_actual = 0` and its start state, with a WARNING. A start close to the table can be refused, and a call that ends below the table is discarded in the same way; see [Floor refusals](#floor-refusals).

**What it costs.** With $m > 0$ the floor is armed in every solve with the entropy EOS, except on the `gradient` core. A call in which it does not fire takes the same steps and gives the same result as without it.

**What it does not do.** The floor guards the table range, not its resolution. Where a table is flat in temperature above $S_\mathrm{min}$, a stop at the level still gives the clamped edge temperature.

- In the solid table of the floor tests, $T$ is the same over the first 4 entropy rows ($S_\mathrm{min}$ to $S_\mathrm{min} + 322$ J kg$^{-1}$ K$^{-1}$) at every pressure up to 6 GPa. With $m = 100$ the floor keeps the state inside the table but the surface temperature at the edge value.
- In one PALEOS solid table at the lowest pressure, $T$ is the same over the rows from about $S_\mathrm{min} + 70$ to $S_\mathrm{min} + 620$ J kg$^{-1}$ K$^{-1}$.

With the floor on, every state a call returns is inside the table when the call starts inside it, and the returned states and the energy trace cannot hold a state below $S_\mathrm{min}$.

## Reference

### Floor refusals

??? note "When the floor refuses a call"
    **Levels.** A start counts as above the level $S_\mathrm{min} + m$ or $S_\mathrm{min} + m/2$ only when it lies more than $10^{-6} m$ above it. The band of a start at the table is $S_\mathrm{min}$ to $S_\mathrm{min} + 2 \times 10^{-6} m$.

    **Start at or below $S_\mathrm{min} + m/2$.** The call arms a level at half its distance to $S_\mathrm{min}$. It does not advance (status 1, `dt_actual = 0`) when the rate of that entropy at the call start, taken over every component no more than $10^{-6} m$ above that entropy, reaches the level before the call ends. Between $S_\mathrm{min} + 2 \times 10^{-6} m$ and $S_\mathrm{min} + m/2$ the refusal depends on the call span, so a caller that retries with a shorter span can then advance.

    **Rate of a linear top node.** Above $S_\mathrm{min} + 2 \times 10^{-6} m$, the rate of a linear top basic node (not with `top_node = 'limited'` or outer boundary condition 5) is the larger of its own rate and the rate of the top cell. The node is a linear extrapolation of the top 2 cells and can fall when only the gradient below it steepens. Inside the band its own rate counts.

    **Start at or below $S_\mathrm{min} + 2 \times 10^{-6} m$.** The call also arms half its distance to $S_\mathrm{min}$, but does not advance on any fall of its lowest component, however slow, at every span, nor when no level fits strictly between $S_\mathrm{min}$ and the start.

    **Start at or below $S_\mathrm{min}$.** Every such start is refused. This includes a linear top basic node extrapolated below $S_\mathrm{min}$ from a steep top gradient while every cell is inside the table and the node or the top cell rises, so no call advances from a top node outside the table.

    **A top node that falls only by extrapolation.** While the top cell warms, it advances above the band in calls that each stop at half its distance, and is refused once inside the band. That takes at most $\lceil \log_2 (d_0 / 2 \times 10^{-6} m) \rceil$ calls that stop at the floor from a distance $d_0 \le m/2$. A call too short to reach its level adds one that does not stop.

    **What continues.** Above $S_\mathrm{min} + 2 \times 10^{-6} m$, a drain too slow to reach half the distance in the call, a pause or a warming surface continues. A call with no span (start equal to end) skips the floor.

    **A call that ends below the table.** A call whose returned states or energy trace hold a top basic-node or cell entropy below $S_\mathrm{min}$ is discarded. It is solved again as a refusal, so its state, time and energy trace stay at the call start. The WARNING names the time after the call start, and that entropy and the level as offsets from $S_\mathrm{min}$.

    **Warnings.** The refusal WARNING says that the entropy is at or below the table minimum, leaves no level between it and the table minimum, falls at the table minimum, or falls too fast between the band and $S_\mathrm{min} + m/2$, also when the start rate is NaN. A floor stop at the call start, where the level lies within the root location error of the start, is a refusal with its own WARNING.

    **In PROTEUS, if the floor were on.** PROTEUS would treat a refused call as a failed attempt. Its retry ladder retries with half the span and an `atol` up to 5 times larger. That can then advance between $S_\mathrm{min} + 2 \times 10^{-6} m$ and $S_\mathrm{min} + m/2$, so under a drain that does not stop the surface entropy can come closer to $S_\mathrm{min}$ than $m/2$. At the table minimum a shorter span does not advance a falling start. When the ladder runs out, PROTEUS would keep the interior state of the previous step while the other modules advance, and stop the run at the third failure in a row.

### Root location limits

??? note "How exactly a stop lands on its level"
    CVODE locates a root to a time tolerance of about $100\,u\,(|t| + |h|)$, with $u$ the unit roundoff, $t$ the absolute call time and $h$ the step. A stop can lie below its level by up to about $|dS/dt| \times 2.2 \times 10^{-14} \times (|t| + |h|)$, with $t$ and $h$ in yr. For steps much shorter than $t$ that is about $|dS/dt| \times 2.2 \times 10^{-14} \times t$.

    scipy event location also loses precision with $t$, more slowly, and a scipy event can stop on either side of its level.

    For the 11-cell state of the floor tests under $10^6$ W m$^{-2}$, a stop at $10^9$ yr lay:

    | Integrator | Distance from the level (J kg$^{-1}$ K$^{-1}$) |
    |---|---|
    | CVODE | $1.6 \times 10^{-3}$ below |
    | Radau | $6 \times 10^{-6}$ below |
    | BDF | $9 \times 10^{-6}$ above |

### Lock measurements

??? note "What the lock counts rest on"
    **Setup.** A probe variant of the energy_balance isentropic start of `tests/test_phase_boundary_cap.py`: 24 nodes, 200 yr, fixed cap unless stated. It draws the initial entropy uniformly between 8000 and 8400 J/kg/K for each run and adds a relative noise of $10^{-15}$. The probe script is not in the repository, and the runs were on macOS arm64. The counts hold for this 24-node standalone case at the default `convective_threshold = 0`, where the switch is at $dS/dr = 0$, not for a coupled solve.

    **What counts as locked.** A run that has not reached 200 yr after 600 s of wall time. Every run was screened at 60 s, and each run that missed that was run again alone to 600 s, by which time a locked run had used about $1.7 \times 10^5$ RHS evaluations.

    | `rtol` and `atol` | Setup | Locked |
    |---|---|---|
    | both `1e-6` | the setup above | 12 of 1133 |
    | both `1e-6` | quasi_steady core | 4 of 474 |
    | both `1e-6` | rate cap | 2 of 286; 2 more finished only after 233 s and 346 s |
    | both `1e-5` | the setup above, each run to 600 s | 0 of 80, too few to show it safe; the slowest finished after 100.6 s and $6.5 \times 10^4$ RHS evaluations |
    | both `1e-8` | fixed cap, energy_balance core | 0 of 1026 |
    | both `1e-8` | rate cap, energy_balance core | 0 of 627 |
    | both `1e-8` | fixed cap, quasi_steady core | 0 of 1224 |
    | `atol` `1e-8`, `rtol` `2e-8`, `5e-8` or `1e-7` (the PROTEUS retry ladder) | fixed cap, each value and core | 0 in each set of 1308 to 1836 runs |
    | `atol` `1e-8`, `rtol` `5e-8` | rate cap | 0 of 620 |
    | `atol` `1e-8`, `rtol` `1e-7` | rate cap | 0 of 636 |

    The median run needed about $1.6 \times 10^4$ RHS evaluations at `1e-5`, against about $1.3 \times 10^3$ at `1e-8`.

    **Recorded steps.** 2 of the counted runs were run again with every accepted step recorded. Both kept a constant step with a period-2 oscillation of the entropy difference at mid-mantle nodes, at 1.41 RHS evaluations per step on the energy_balance core, where the difference stayed just below zero, and at 1.14 on the quasi_steady core, around zero. The other counted runs were not checked for the pattern.

    **Why $D > 0$ differs.** At $D = 0$ the interior flux is continuous at the switch and only its slope jumps, but with `core_bc` `energy_balance` or `gradient` the convective CMB flux jumps by $\rho T \kappa_h^\mathrm{floor}\,|\partial S/\partial r|_\mathrm{cmb}$ when the node above the CMB crosses 0. At $D > 0$ the convective flux stays continuous at 0, but the mixing flux of a mush and the CMB flux jump at 0 and the floor flux jumps at $D$ (see [Heat transport](../Explanations/heat_transport.md#convection-mixing-length-theory)). So the $D = 0$ counts and the `1e-8` remedy do not carry over to $D > 0$, and the probe did not cover it.

    **On CI.** The AVX-512 numpy path makes the test's own case, at its single fixed initial entropy, stall at `1e-6`, judged by the CI test timeout rather than by the 600 s criterion.

## See also

- [CVODE and JAX explainer](../Explanations/cvode_jax.md): why the analytic Jacobian helps and how `set_jax_cvode_factory` is wired.
- [`phi_step_cap` how-to](phi-step-cap.md): SUNDIALS-rootfn capping of $|\Delta\phi|$ across a step.
- [Installation: production solver path](installation.md#production-solver-path-sundials-cvode-jax-analytic-jacobian): runtime requirement for `use_jax_jacobian = true`.
- [Solid-state convection explainer](../Explanations/solid_state_convection.md): mantle rheology, stagnant-lid closures, and step control during solid-state thermal evolution.
