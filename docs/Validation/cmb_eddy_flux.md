# CMB eddy flux

## What this page checks

This page checks the eddy diffusivity at the core-mantle boundary (CMB) node, and with it the convective heat flux from the core into the mantle (see [Eddy diffusivity at the CMB node](../Explanations/heat_transport.md#eddy-diffusivity-at-the-cmb-node)).
The CMB node takes the mixing-length eddy diffusivity of the node above it (node 1), or its own floor, whichever is larger.
Validation here means that we run the code on states where we know the right answer from the rule written out by hand, from a second implementation (the NumPy and the JAX right-hand sides), or from runs that stop without the rule, and check that the code gives that answer.
If this part were wrong, the heat flow out of the core could jump between 2 values when the entropy of the 2 lowest cells crosses, or fall to 0 while the CMB gradient is unstable; the core would then cool at the wrong rate, and the energy check of a run would stop it.
Each case says why the check exists, what it tests, how we know the right answer and why it matters, then shows a plot or a table and the test details.

In the formulas below, $S_0$ and $S_1$ are the entropies of the 2 lowest cells, $\partial S/\partial r$ is the entropy gradient at a node, $\kappa_h^\mathrm{mlt}[1]$ is the mixing-length eddy diffusivity of node 1 before the floor, $\kappa_h^\mathrm{floor}[0]$ is the floor of the CMB node, and $D$ is `energy.convective_threshold` (0 unless set).

## The CMB flux has no step where node 1 changes sign

**Why this check exists.** If the CMB node copied the floored eddy diffusivity of node 1, a stable CMB gradient would carry a downward floor flux while $S_1 < S_0$ and none while $S_1 > S_0$. The flux out of the core would then jump each time $S_1 - S_0$ changes sign, by about $2.9 \times 10^3$ W m$^{-2}$ in the test state.

**What it tests.** A mush of 30 cells at a melt fraction of 0.4, built with the SPIDER test tables, with a stable CMB gradient of $2.36 \times 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ and node 1 on its floor. The test sets $S_1 - S_0$ to $\pm 10^{-9}$, $\pm 10^{-12}$ and $\pm 10^{-17}$ J kg$^{-1}$ K$^{-1}$ and compares the CMB flux on the 2 sides. The difference must shrink with the step, by at least a factor of 100 from $10^{-9}$ to $10^{-12}$, and stay below $10^{-5}$ of the jump of the copied value. The test runs with the `nearest_boundary` and the `constant` mixing length, on the NumPy and the JAX path.

**How we know the right answer.** Over a stable CMB gradient the convective CMB flux is $-\rho T \kappa_h^\mathrm{mlt}[1]\,\partial S/\partial r$, and the conductive part follows Bower et al. (2018)[^cite-bower2018]. The test evaluates both from the material properties at node 0 and compares them with the code. The mixing-length part of node 1 goes to 0 as $S_1 - S_0$ goes to 0 from below, so the flux has no step.

**Why it matters.** A CMB flux that jumps with the sign of $S_1 - S_0$ makes the core heat of a call depend on round-off. The energy check of the core then refuses the call, and the run stops.

![CMB heat flux against S1 minus S0 on a symmetric log axis, for Aragog and for the copied node 1 value](../assets/validation/cmb_eddy_flux_continuity.svg#only-light){ width="70%" }
![CMB heat flux against S1 minus S0 on a symmetric log axis, for Aragog and for the copied node 1 value](../assets/validation/cmb_eddy_flux_continuity_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The horizontal axis is $S_1 - S_0$ in J kg$^{-1}$ K$^{-1}$, the vertical axis the CMB heat flux in W m$^{-2}$; both are on symmetric log scales. The dots are Aragog. The squares are the flux with the floored node 1 value copied to the CMB node, shown for context. The dots pass through $S_1 = S_0$ without a step. The squares fall by about $2.9 \times 10^3$ W m$^{-2}$ there.

??? info "Test details"
    - Reference: the convective flux $-\rho T \kappa_h^\mathrm{mlt}[1]\,\partial S/\partial r$ and the conductive flux of Bower et al. (2018)[^cite-bower2018], from the properties at node 0; the JAX right-hand side.
    - Tolerance: formulas 1e-12 relative; NumPy against JAX 1e-10 relative; the side difference shrinks by at least 100 from a step of 1e-9 to 1e-12 J/kg/K, stays below 1e-5 of the copied-value jump at every step, and below 1e-6 of the CMB flux at 1e-17 J/kg/K. Measured side differences, `nearest_boundary`: 2.2e-4, 2.2e-7 and 2.2e-12 W m$^{-2}$; `constant`: 3.6e-3, 3.6e-6 and 3.6e-11 W m$^{-2}$; the copied value jumps by 2890.8 W m$^{-2}$ at every step.
    - Test id: `tests/test_cmb_eddy_diffusivity.py::test_cmb_flux_is_continuous_where_node_1_changes_sign`, `tests/test_cmb_eddy_diffusivity.py::test_cmb_flux_across_s1_equal_s0`
    - Command: `pytest tests/test_cmb_eddy_diffusivity.py` (add `--validation-plots` to write the plot pair)

## The rule in each kind of CMB state

**Why this check exists.** The rule must give the right value in each kind of state the CMB can be in. A wrong floor (the floor of node 1, or no floor) changes the CMB flux in some kinds only, so each kind is checked on its own.

**What it tests.** 5 built states on the JAX right-hand side, with the gradient and the melt fraction $\phi$ of nodes 0 and 1 set by hand and a mixing length of 100 km. With the floor parameter `kappah_floor` = 10 m$^2$ s$^{-1}$, the floor is $10 f(\phi)$ m$^2$ s$^{-1}$, with $f$ the tanh weight around the melt fraction 0.4. The test asserts $\kappa_h[0] = \max(\kappa_h^\mathrm{mlt}[1], \kappa_h^\mathrm{floor}[0])$ in each kind, for a scaled and a constant `eddy_diffusivity_thermal`. With `kappah_floor = 0` the CMB node must equal node 1 bit for bit, as in SPIDER. The NumPy update is checked in the same way across node 1 at 0 and at $D$.

**How we know the right answer.** The rule written out by hand: the test takes $\kappa_h^\mathrm{mlt}[1]$ from the same state without the floor and evaluates the floor of node 0 from its melt fraction.

**Why it matters.** Kind b is the state that stopped runs: a stable CMB with node 1 on its floor. Kinds c and d occur in long runs: in a 1 Earth-mass run, node 1 turned stable over an unstable CMB gradient at about 30 kyr.

| Kind | $\partial S/\partial r$ at nodes 0, 1 | $\phi_0$, $\phi_1$ | $\kappa_h^\mathrm{mlt}[1]$ | floor of node 0 | floor of node 1 | $\kappa_h[0]$ | copy of node 1 |
|---|---|---|---|---|---|---|---|
| a: node 1 above both floors | $-10^{-4}$, $-10^{-4}$ | 0.6, 0.4 | 857759 | 9.35031 | 5 | 857759 | 857759 |
| b: stable CMB, node 1 on its floor | $10^{-4}$, $-10^{-31}$ | 0.6, 0.4 | $3.6 \times 10^{-15}$ | 0 | 5 | $3.6 \times 10^{-15}$ | 5 |
| c: unstable CMB, node 1 on its floor | $-10^{-4}$, $-10^{-31}$ | 0.6, 0.4 | $3.6 \times 10^{-15}$ | 9.35031 | 5 | 9.35031 | 5 |
| d: unstable CMB, node 1 stable | $-10^{-4}$, $10^{-4}$ | 0.6, 0.4 | 0 | 9.35031 | 0 | 9.35031 | 0 |
| e: equal floors | $-10^{-31}$, $-10^{-31}$ | 0.4, 0.4 | $3.6 \times 10^{-15}$ | 5 | 5 | 5 | 5 |

The table gives eddy diffusivities in m$^2$ s$^{-1}$ and gradients in J kg$^{-1}$ K$^{-1}$ m$^{-1}$. This case has no plot: it checks the rule on 5 single states, and the table holds every compared value.

??? info "Test details"
    - Reference: $\max(\kappa_h^\mathrm{mlt}[1], \kappa_h^\mathrm{floor}[0])$ evaluated in the test; the SPIDER rule (`energy.c`) for `kappah_floor = 0`.
    - Tolerance: 1e-12 relative; bit-equal for `kappah_floor = 0`.
    - Test id: `tests/test_jax_kappah_floor_mask.py::test_cmb_node_known_answer_in_each_kind`, `tests/test_jax_kappah_floor_mask.py::test_cmb_node_without_floor_is_the_copy_of_node_1`, `tests/test_jax_kappah_floor_mask.py::test_cmb_node_takes_the_mixing_length_of_node_1_and_its_own_floor`, `tests/test_entropy_state_kappah_floor_mask.py::test_numpy_cmb_node_takes_the_mixing_length_of_node_1_and_its_own_floor`
    - Command: `pytest tests/test_jax_kappah_floor_mask.py tests/test_entropy_state_kappah_floor_mask.py`

## The rule on random states

**Why this check exists.** The hand-built states cover 5 points. A fault that acts only for some melt fractions, thresholds or diffusivity modes, or a NumPy path that differs from the JAX path, would pass them.

**What it tests.** 300 random states drawn by the Hypothesis package: the gradients at nodes 0 and 1 (signed, from $10^{-12}$ to $10^{-2}$), the melt fractions of the 2 lowest cells, $D$ (0 or from $10^{-9}$ to $10^{-5}$), a scaled or constant `eddy_diffusivity_thermal` and `kappah_floor` (0 or from 0.1 to 100 m$^2$ s$^{-1}$). Each state runs through the NumPy update and the JAX right-hand side.

**How we know the right answer.** The rule written out in the test, with the floor of node 0 from the melt fraction that the NumPy update gives at node 0.

**Why it matters.** With `backend = "jax"` the solver uses the JAX right-hand side, while the output arrays come from the NumPy update after each call, so both must give the same CMB value.

This case has no plot: each of the 300 states is a single comparison of 2 numbers.

??? info "Test details"
    - Reference: the rule evaluated in the test; the JAX right-hand side for the NumPy path.
    - Tolerance: the rule 1e-12 relative (JAX); NumPy against JAX 1e-10 relative.
    - Test id: `tests/test_cmb_eddy_diffusivity.py::test_cmb_rule_holds_on_random_states`
    - Command: `pytest tests/test_cmb_eddy_diffusivity.py -k random`

## The CMB flux check in the code

**Why this check exists.** The rule is a few lines in 2 functions. A later change to the floor, the mask or the copy can break it in a state that no test builds. `EntropyState.update` therefore checks the convective CMB flux $J$ against the inputs of the rule in every call.

**What it tests.** With $g$ the CMB gradient, the check `check_cmb_convective_flux` raises `CMBEddyFluxError` when:

- $g \ge D$ and $|J| > \rho T \kappa_h^\mathrm{mlt}[1]\,|g|$: a floor of node 1 crosses a stable CMB;
- $g < D$ and $|J| < \rho T \kappa_h^\mathrm{floor}[0]\,|g|$: the CMB node lost its own floor;
- $J g > 0$: the flux runs against its gradient.

The tests pass each kind of state through the check, give it a flux that breaks each part, and confirm that the update calls it with the values it used.

**How we know the right answer.** The 3 parts follow from the rule: $\kappa_h[0]$ is at least the floor of node 0, equals $\kappa_h^\mathrm{mlt}[1]$ over a stable gradient, and is never negative.

**Why it matters.** A run that breaks the rule stops at the first call with a message that names the part, instead of carrying a wrong core heat flow.

This case has no plot: each test is a single pass or refusal.

??? info "Test details"
    - Reference: the 3 parts of the rule above.
    - Tolerance: 1e-12 relative on parts 1 and 2.
    - Test id: `tests/test_cmb_eddy_diffusivity.py::test_invariant_passes_the_rule`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_refuses_each_broken_part`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_uses_the_threshold`, `tests/test_cmb_eddy_diffusivity.py::test_update_checks_the_invariant`
    - Command: `pytest tests/test_cmb_eddy_diffusivity.py -k invariant`

## A large planet runs through the state that stopped it

**Why this check exists.** With the main rule, a run of 8.63 Earth masses (case 38) stops at 6.77 Myr: in that state the CMB gradient is stable, node 1 sits on its floor and $S_1 - S_0$ is about $10^{-9}$ J kg$^{-1}$ K$^{-1}$. The copied floor makes the CMB power jump with the sign of $S_1 - S_0$, and the energy check of the core refuses the calls.

**What it tests.** The run is resumed from its stored state at 6.769704 Myr and runs to 6.82 Myr on Habrok, once with the main rule and once with the rule of this page, in the same build of PROTEUS. In the first calls the runner also evaluates the CMB power at $S_1 = S_0 \pm 10^{-9}$ and $\pm 10^{-12}$ J kg$^{-1}$ K$^{-1}$ at every output point.

**How we know the right answer.** The main rule stops here, so it gives the failure to compare with. The energy check of each call gives the pass mark: the core heat change must match the CMB heat of the call within the solver's own error bound $B$. A call with a miss above $10 B$ is refused.

**Why it matters.** This is the failure that the rule removes. Without it, the run cannot pass 6.77 Myr.

![CMB heat flux and ledger miss of each call against time, main and the CMB node rule, case 38 resume](../assets/validation/cmb_eddy_flux_case38_resume.svg#only-light){ width="70%" }
![CMB heat flux and ledger miss of each call against time, main and the CMB node rule, case 38 resume](../assets/validation/cmb_eddy_flux_case38_resume_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Time runs from the resume. Panel (a) shows the CMB heat flux $F_\mathrm{cmb}$ of each helpfile row. Panel (b) shows the ledger miss of each call divided by its bound, $|r|/B$, on a log scale; the dashed line is the refusal limit of 10. Main (squares) has 3 calls above the limit and stops. The rule (circles) stays near $10^{-9}$ in every call and runs to 6.82 Myr.

![Difference of the CMB power between the 2 sides of S1 = S0 at every output point of the first calls, main and the CMB node rule](../assets/validation/cmb_eddy_flux_case38_sweep.svg#only-light){ width="70%" }
![Difference of the CMB power between the 2 sides of S1 = S0 at every output point of the first calls, main and the CMB node rule](../assets/validation/cmb_eddy_flux_case38_sweep_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Each mark is one output point. The vertical axis is the difference of the CMB power in W between $S_1 = S_0 + 10^{-9}$ and $S_1 = S_0 - 10^{-9}$ J kg$^{-1}$ K$^{-1}$. Main jumps by $9.1 \times 10^{16}$ W at every point. The rule differs by at most $10^{-3}$ W on a CMB power of $8 \times 10^{11}$ W, which is the round-off of that power.

??? info "Test details"
    - Reference: the main run in the same build (aragog 548ea357, PROTEUS ce0b4283); the energy check of each call.
    - Tolerance: every call below the bound (measured: at most $|r|/B = 5.2 \times 10^{-9}$ in 14 calls); no helpfile row with a CMB heat below $-10^{25}$ J. Main: 3 calls above $10 B$ (at most 21.1), status 20.
    - Step test: main $|\Delta P_\mathrm{cmb}| = 9.105 \times 10^{16}$ W at both steps, at all 416 points of 3 calls. Rule: at most $9.8 \times 10^{-4}$ W at $10^{-9}$ (relative $1.2 \times 10^{-15}$) and 0 at $10^{-12}$, at all 544 points of 5 calls.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_case_38_runs_through_the_old_stop`, `tests/test_cmb_eddy_flux_runs.py::test_case_38_cmb_power_has_no_step_at_s1_equal_s0`
    - Data: `tests/validation/data/cmb_eddy_flux_runs.npz`, built by `tests/reference/build_cmb_eddy_flux_runs.py` (provenance in `cmb_eddy_flux_runs_provenance.json`).
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py` (add `--validation-plots` to write the plot pairs)

## An Earth-mass run where node 1 turns stable

**Why this check exists.** In a run of 1 Earth mass with the `nearest_boundary` mixing length, node 1 turns stable at about 30 kyr while the CMB gradient stays unstable. With the main rule the CMB node then copies 0, the flux out of the core falls to about $10^{-3}$ W m$^{-2}$, and the energy check stops the run in the call after its last row at 30952 yr. This case needs the floor of node 0.

**What it tests.** The run is resumed at 30169.3 yr and runs to 32500 yr, with the main rule and with the rule of this page. The test checks the status of both runs, $|r|/B$ of every call, and the CMB flux of the rule.

**How we know the right answer.** The main run stops here. The pass mark is the same energy check as above: no call above $10 B$.

**Why it matters.** Kind d states occur in long Earth-mass runs. If the CMB node loses its floor there, the core is cut off from the mantle.

![CMB heat flux and ledger miss of each call against time, main and the CMB node rule, Earth-mass resume with the nearest-boundary mixing length](../assets/validation/cmb_eddy_flux_nearest_resume.svg#only-light){ width="70%" }
![CMB heat flux and ledger miss of each call against time, main and the CMB node rule, Earth-mass resume with the nearest-boundary mixing length](../assets/validation/cmb_eddy_flux_nearest_resume_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Panel (a) shows the CMB heat flux on a log scale. For main (squares) and the rule (filled circles) it is the average over each solver call, the value of the helpfile. For the rule it also shows the peak inside each call (up triangles) and the node flux at the end of each call (down triangles). Panel (b) shows $|r|/B$ of each call; the dashed line is the refusal limit of 10. Main falls to about $10^{-3}$ W m$^{-2}$ and stops; the rule keeps 128 to 6092 W m$^{-2}$ and stays below the line.

**Known limit.** 3 of the 12 calls of the rule miss by more than the bound: $|r|/B$ = 1.160, 1.248 and 4.323. These are calls with a short peak of the CMB flux inside the call: $2.8 \times 10^4$ to $6.9 \times 10^5$ W m$^{-2}$, while 6 of the 8 calls with a peak end between 130 and 350 W m$^{-2}$. All 8 calls with $|r|/B$ above 0.3 have such a peak, and the 4 calls without a peak stay below 0.004. The quadrature error of the CMB heat on the stored points is at most 0.01 $B$, so the miss is not the trapezoid sum of the stored points.

**Known limit.** The helpfile flux is the average over a call, and the snapshot of a call holds the node flux at its end. At 31773 yr the call average is 1776 W m$^{-2}$, while the end state sits in a peak, at $7.738 \times 10^4$ W m$^{-2}$ with a CMB gradient of $-4.0 \times 10^{-4}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$.

??? info "Test details"
    - Reference: the main run (aragog 548ea357, PROTEUS afc419fe); the energy check of each call.
    - Tolerance: every call below $10 B$. Main: status 20 at 30952.1 yr, $|r|/B$ up to 787. Rule: status 13 at 32500 yr, 12 calls, 3 above 1.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_nearest_mixing_length_resume_runs_where_main_stops`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py`

## An Earth-mass run where the rule keeps the main flux

**Why this check exists.** A rule for one state must not break a run that the main rule handles. A run of 1 Earth mass with the `constant` mixing length passes through states where a rule without the downward eddy flux over a stable CMB gradient fails.

**What it tests.** The run is resumed at 37854.6 yr and runs to 38600 yr, with the main rule and with the rule of this page. The test compares the CMB flux and the CMB temperature, and checks the energy check of each call.

**How we know the right answer.** Main closes the energy check in every call here, so the rule must close it too and stay close to main.

**Why it matters.** This is a convective CMB state of a normal run, where the rule should change little.

![CMB heat flux against time, main and the CMB node rule, Earth-mass resume with the constant mixing length](../assets/validation/cmb_eddy_flux_constant_resume.svg#only-light){ width="70%" }
![CMB heat flux against time, main and the CMB node rule, Earth-mass resume with the constant mixing length](../assets/validation/cmb_eddy_flux_constant_resume_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Each mark is the CMB heat flux of one helpfile row after the resume. The squares (main) and the circles (rule) lie on each other.

??? info "Test details"
    - Reference: the main run (aragog 548ea357, PROTEUS afc419fe).
    - Tolerance: every call below the bound (measured $|r|/B$ at most 0.28 for main, 0.30 for the rule); measured differences against main, with the rule interpolated to the times of main: $F_\mathrm{cmb}$ $1.7 \times 10^{-3}$ relative, $T_\mathrm{cmb}$ $1.7 \times 10^{-9}$ relative.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_constant_mixing_length_resume_keeps_the_ledger`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py`

## A run from the start where the rule does not act

**Why this check exists.** In most states the CMB node takes the mixing-length value of node 1, and the rule equals the main rule. A run in such states must not change.

**What it tests.** The same Earth-mass run with the `constant` mixing length, from $t = 0$ to 5 kyr. An instrumented copy of the code counts the CMB evaluations where the rule differs from main: in the JAX right-hand side (a debug callback) and in the NumPy update. A second run of main changes the same code line to a form with the same values, $\kappa_h[0] = \max(\kappa_h[1], 0 \cdot \kappa_h^\mathrm{floor}[0])$, which compiles to another graph.

**How we know the right answer.** If the rule never acts, the run of the rule can differ from main only as much as main differs from main with the same-value edit.

**Why it matters.** It shows that the change of a run in states where the rule does not act is the round-off of the code, not the rule.

| Run | CMB evaluations where the rule differs from main | Helpfile against main |
|---|---|---|
| main, second run | | equal in every column except the run time |
| main with the same-value edit | | 135 of 1040 columns differ |
| rule | 0 of 88499 (JAX), 0 of 454 (NumPy) | equal to main with the same-value edit, bit for bit |
| rule with the debug callback | 0 of 88499 | equal to main, bit for bit |

This case has no plot: the compared runs are bit-equal, which the table states.

**Known limit.** This run changes between 2 bit patterns with the compiled graph alone. The start transient grows the round-off: the first difference is at 2 yr, $9.9 \times 10^{-9}$ relative in $F_\mathrm{cmb}$, and by 5 kyr $F_\mathrm{cmb}$ differs by up to 25.6 W m$^{-2}$. A comparison with main in this run is valid only above this floor.

??? info "Test details"
    - Reference: main, a second run of main, and main with the same-value edit (aragog 548ea357, PROTEUS afc419fe).
    - Tolerance: 0 differing evaluations; the rule run bit-equal to main with the same-value edit.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_run_from_t0_differs_only_at_the_graph_round_off`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py`

[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
