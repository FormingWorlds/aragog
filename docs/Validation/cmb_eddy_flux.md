# CMB eddy flux

## What this page checks

This page checks the eddy diffusivity at the core-mantle boundary (CMB) node, and with it the convective heat flux from the core into the mantle.
The CMB node (node 0) takes the mixing-length eddy diffusivity of node 1, the node above it, or its own floor, whichever is larger ([Eddy diffusivity at the CMB node](../Explanations/heat_transport.md#eddy-diffusivity-at-the-cmb-node)).
We call this the CMB node rule. The cases compare it with the copy rule, where the CMB node copies the floored value of node 1.

Validation here means that we run the code on a case where we know the right answer, and check that the code gives that answer.
The right answer comes from 3 sources:

- the rule written out by hand in the test;
- a second implementation: the NumPy update and the JAX function `compute_mlt`;
- coupled PROTEUS runs that stop under the copy rule.

If this part were wrong, the heat flow out of the core could jump between 2 values when the entropies of the 2 lowest cells cross, or fall to 0 while the CMB gradient is unstable.
The core would then cool at the wrong rate, and the energy check of the run would stop it.
Each case says why the check exists, what it tests, how we know the right answer and why it matters; then it shows a plot or a table, and the test details.

In the formulas below, $S_0$ and $S_1$ are the entropies of the 2 lowest cells, $\partial S/\partial r$ is the entropy gradient at a node, $\kappa_h^\mathrm{mlt}[1]$ is the mixing-length eddy diffusivity of node 1 before the floor, $\kappa_h^\mathrm{floor}[0]$ is the floor of the CMB node, and $D$ is `energy.convective_threshold` (0 unless set).
The energy check is the [ledger check of the CMB closure](../Explanations/core_bc.md#ledger-check-of-the-cmb-closure): in each solver call the change of the core heat must match the CMB heat of the call within the bound $B$; $r$ is the miss, and a call with $|r| > 10 B$ is refused.

## The CMB flux has no step where node 1 changes sign

**Why this check exists.** Under the copy rule a stable CMB gradient carries a downward floor flux while $S_1 < S_0$ and none while $S_1 > S_0$. The flux out of the core then jumps each time $S_1 - S_0$ changes sign, by about $2.9 \times 10^3$ W m$^{-2}$ in the test state.

**What it tests.** A mush of 30 cells at a melt fraction of 0.4, built with the SPIDER test tables, with a stable CMB gradient of $2.36 \times 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$, $D = 0$ and node 1 on its floor. The test sets $S_1 - S_0$ to $\pm 10^{-9}$, $\pm 10^{-12}$ and $\pm 10^{-17}$ J kg$^{-1}$ K$^{-1}$ and compares the CMB flux on the 2 sides. The difference must shrink with the step, by at least a factor of 100 from $10^{-9}$ to $10^{-12}$, and stay below $10^{-5}$ of the step of the copy rule. The test runs with the `nearest_boundary` and the `constant` mixing length, on the NumPy update and the JAX `compute_mlt`.

**How we know the right answer.** Over a stable CMB gradient the convective CMB flux is $-\rho T \kappa_h^\mathrm{mlt}[1]\,\partial S/\partial r$, and the conductive part follows Bower et al. (2018)[^cite-bower2018]. The test evaluates both from the material properties at node 0 and compares them with the code. The mixing-length part of node 1 goes to 0 as $S_1 - S_0$ goes to 0 from below, so the flux has no step.

**Why it matters.** A CMB flux that jumps with the sign of $S_1 - S_0$ makes the core heat of a call depend on round-off. The energy check of the core then refuses the call, and the run stops.

![CMB heat flux against S1 minus S0 on a symmetric log axis, for the CMB node rule and the copy rule](../assets/validation/cmb_eddy_flux_continuity.svg#only-light){ width="70%" }
![CMB heat flux against S1 minus S0 on a symmetric log axis, for the CMB node rule and the copy rule](../assets/validation/cmb_eddy_flux_continuity_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The horizontal axis is $S_1 - S_0$ in J kg$^{-1}$ K$^{-1}$, the vertical axis the CMB heat flux in W m$^{-2}$; both are on symmetric log scales. The dots are the CMB node rule. The squares are the copy rule, shown for context. The dots pass through $S_1 = S_0$ without a step. The squares differ by about $2.9 \times 10^3$ W m$^{-2}$ between the 2 sides.

**Known limit.** The test runs at $D = 0$, the value of every PROTEUS run: PROTEUS has no setting for `energy.convective_threshold`, and its default in Aragog is 0. For $D > 0$ and a positive `eddy_diffusivity_thermal` both rules step, with $\kappa_h(D)$ the mixing-length value of a gradient $D$ and $g_0 = \partial S/\partial r[0]$:

- CMB node rule: where node 1 crosses $\partial S/\partial r = 0$, by $\rho T \max(\kappa_h(D) - \kappa_h^\mathrm{floor}[0], 0)\,|g_0|$, with $\kappa_h^\mathrm{floor}[0]$ taken as 0 when $g_0 \ge D$; and where $g_0$ crosses $D$, by $\rho T \max(\kappa_h^\mathrm{floor}[0] - \kappa_h^\mathrm{mlt}[1], 0)\,D$.
- Copy rule: where node 1 crosses 0, by $\rho T (\max(\kappa_h(D), \kappa_h^\mathrm{floor}[1]) - \kappa_h^\mathrm{floor}[1])\,|g_0|$; and where node 1 crosses $D$, by $\rho T \kappa_h^\mathrm{floor}[1]\,|g_0|$.

??? info "Test details"
    - Reference: the convective flux $-\rho T \kappa_h^\mathrm{mlt}[1]\,\partial S/\partial r$ and the conductive flux of Bower et al. (2018)[^cite-bower2018], from the properties at node 0; the JAX `compute_mlt`.
    - Tolerance: formulas 1e-12 relative; NumPy against JAX 1e-10 relative (absolute 1e-25 m$^2$ s$^{-1}$, the residual drive of the NumPy path on the stable side); the side difference shrinks by at least 100 from a step of 1e-9 to 1e-12 J/kg/K, stays below 1e-5 of the copy-rule step at every step, and below 1e-6 of the CMB flux at 1e-17 J/kg/K. Measured side differences, `nearest_boundary`: 2.2e-4, 2.2e-7 and 2.2e-12 W m$^{-2}$; `constant`: 3.6e-3, 3.6e-6 and 3.6e-11 W m$^{-2}$; the copy rule steps by 2890.8 W m$^{-2}$ at every step.
    - Test id: `tests/test_cmb_eddy_diffusivity.py::test_cmb_flux_is_continuous_where_node_1_changes_sign`, `tests/test_cmb_eddy_diffusivity.py::test_cmb_flux_across_s1_equal_s0`
    - Command: `pytest tests/test_cmb_eddy_diffusivity.py` (add `--validation-plots` to write the plot pair)

## The rule in each kind of CMB state

**Why this check exists.** The rule must give the right value in each kind of state the CMB can be in. A wrong floor (the floor of node 1, or no floor) changes the CMB flux in some kinds only, so each kind is checked on its own.

**What it tests.** 5 built states on the JAX `compute_mlt`, with the gradient and the melt fraction $\phi$ of nodes 0 and 1 set by hand, $D = 0$ and a mixing length of 100 km. With the floor parameter `kappah_floor` = 10 m$^2$ s$^{-1}$, the floor is $10 f(\phi)$ m$^2$ s$^{-1}$, with $f$ the tanh weight around the melt fraction 0.4. The test asserts $\kappa_h[0] = \max(\kappa_h^\mathrm{mlt}[1], \kappa_h^\mathrm{floor}[0])$ in each kind, for a scaled and a constant `eddy_diffusivity_thermal`. With `kappah_floor = 0` the CMB node must equal node 1 bit for bit, as in SPIDER, on both paths. With the solid rheology on, the floor of node 0 uses the zero-anchored weight of its own melt fraction (0.298 m$^2$ s$^{-1}$ at 0.15, against 0.344 with the plain tanh weight). For $D > 0$, the CMB node holds its floor just below $\partial S/\partial r[0] = D$ and loses it at $D$, on both paths. The NumPy update is checked in the same way across node 1 at 0 and at $D$.

**How we know the right answer.** The rule written out by hand: the test takes $\kappa_h^\mathrm{mlt}[1]$ from the same state without the floor and evaluates the floor of node 0 from its melt fraction.

**Why it matters.** Kind b is the state where the copy rule stops runs: a stable CMB with node 1 on its floor. Kinds c and d occur in long runs: in a 1 Earth-mass run, node 1 turns stable over an unstable CMB gradient at about 30 kyr.

| Kind | $\partial S/\partial r$ at nodes 0, 1 | $\phi_0$, $\phi_1$ | $\kappa_h^\mathrm{mlt}[1]$ | floor of node 0 | floor of node 1 | $\kappa_h[0]$ | copy rule |
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
    - Test id: `tests/test_jax_kappah_floor_mask.py::test_cmb_node_known_answer_in_each_kind`, `tests/test_jax_kappah_floor_mask.py::test_cmb_node_without_floor_is_the_copy_of_node_1`, `tests/test_jax_kappah_floor_mask.py::test_cmb_node_takes_its_own_zero_anchored_floor_with_rheology`, `tests/test_jax_kappah_floor_mask.py::test_cmb_flux_steps_where_the_cmb_gradient_crosses_d`, `tests/test_jax_kappah_floor_mask.py::test_cmb_node_takes_the_mixing_length_of_node_1_and_its_own_floor`, `tests/test_entropy_state_kappah_floor_mask.py::test_numpy_cmb_node_takes_the_mixing_length_of_node_1_and_its_own_floor`, `tests/test_entropy_state_kappah_floor_mask.py::test_numpy_cmb_node_without_floor_is_the_copy_of_node_1`, `tests/test_entropy_state_kappah_floor_mask.py::test_numpy_cmb_node_floor_steps_where_the_cmb_gradient_crosses_d`
    - Command: `pytest tests/test_jax_kappah_floor_mask.py tests/test_entropy_state_kappah_floor_mask.py`

## The rule on random states

**Why this check exists.** The hand-built states cover 5 points. A fault that acts only for some melt fractions, thresholds or diffusivity modes, or a NumPy path that differs from the JAX path, would pass them.

**What it tests.** 300 random states drawn by the Hypothesis package: the gradients at nodes 0 and 1 (signed, from $10^{-12}$ to $10^{-2}$), the melt fractions of the 2 lowest cells, $D$ (0 or from $10^{-9}$ to $10^{-5}$), a scaled or constant `eddy_diffusivity_thermal` and `kappah_floor` (0 or from 0.1 to 100 m$^2$ s$^{-1}$). Each state runs through the NumPy update and the JAX `compute_mlt`.

**How we know the right answer.** The rule written out in the test, with the floor of node 0 from the melt fraction that the NumPy update gives at node 0.

**Why it matters.** With `use_jax_jacobian = true` (the default) and a JAX CVODE factory, as PROTEUS sets with `backend = "jax"`, the solver integrates with the JAX right-hand side and evaluates the energy replay of each call with the JAX batch, both with `compute_mlt`. The NumPy update gives the end state of each call and the output arrays. Without the factory every part uses the NumPy update. Both paths must give the same CMB value.

This case has no plot: each of the 300 states is a single comparison of 2 numbers.

??? info "Test details"
    - Reference: the rule evaluated in the test; the JAX `compute_mlt` for the NumPy path.
    - Tolerance: the rule 1e-12 relative (JAX); NumPy against JAX 1e-10 relative.
    - Test id: `tests/test_cmb_eddy_diffusivity.py::test_cmb_rule_holds_on_random_states`
    - Command: `pytest tests/test_cmb_eddy_diffusivity.py -k random`

## The CMB flux check in the code

**Why this check exists.** The NumPy `EntropyState.update` sets the eddy diffusivity and the convective flux $J$ of the CMB node from 2 recorded inputs, $\kappa_h^\mathrm{mlt}[1]$ and $\kappa_h^\mathrm{floor}[0]$. The check `check_cmb_convective_flux` tests $J$ against those inputs in every update, so an edit that sets $J$ or $\kappa_h[0]$ against the rule fails at once. It does not test how the 2 inputs are computed, and it does not run on the JAX path; the known-answer tests above do both.

**What it tests.** The check raises `CMBEddyFluxError` when:

- $\partial S/\partial r[0] \ge D$ and $|J| > \rho T \kappa_h^\mathrm{mlt}[1]\,|\partial S/\partial r[0]|$: a floor of node 1 crosses a stable CMB;
- $\partial S/\partial r[0] < D$ and $|J| < \rho T \kappa_h^\mathrm{floor}[0]\,|\partial S/\partial r[0]|$: the CMB node lost its own floor;
- $J\,\partial S/\partial r[0] > 0$: the flux runs against its gradient;
- $J$ differs from $-\rho T \max(\kappa_h^\mathrm{mlt}[1], \kappa_h^\mathrm{floor}[0])\,\partial S/\partial r[0]$ by more than $10^{-12}$ relative: any other flux, for example one with the mixing-length part of node 1 dropped.

The tests pass each kind of state through the check, give it a flux that breaks each part, and confirm that the update calls it with the values it used. 2 solver tests make the check fail inside a solve.

**How we know the right answer.** The 4 parts follow from the rule: $\kappa_h[0]$ is at least the floor of node 0, equals $\kappa_h^\mathrm{mlt}[1]$ over a stable gradient, is never negative, and is the larger of the 2 inputs.

**Why it matters.** On the JAX path the check sees the end state of each call; on the NumPy path (scipy, or CVODE without the JAX factory) it also sees every state of the energy replay and of the right-hand side. A call that breaks the rule fails like a refused ledger call: in the NumPy CVODE right-hand side it ends the call, and in the energy replay the solver goes back to the solution held at the call start. The error names the part.

This case has no plot: each test is a single pass or refusal.

??? info "Test details"
    - Reference: the 3 parts of the rule above.
    - Tolerance: 1e-12 relative on parts 1, 2 and 4.
    - Test id: `tests/test_cmb_eddy_diffusivity.py::test_invariant_passes_the_rule`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_refuses_each_broken_part`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_refuses_a_flux_between_the_bounds`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_passes_round_off_of_the_rule`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_uses_the_threshold`, `tests/test_cmb_eddy_diffusivity.py::test_update_checks_the_invariant`, `tests/test_cmb_eddy_diffusivity.py::test_a_flux_fault_in_the_cvode_rhs_reaches_the_caller`, `tests/test_cmb_eddy_diffusivity.py::test_a_flux_fault_keeps_the_last_accepted_state`
    - Command: `pytest tests/test_cmb_eddy_diffusivity.py -k "invariant or fault"`

## A large planet runs through the state where the copy rule stops

**Why this check exists.** Under the copy rule a run of 8.63 Earth masses stops at 6.77 Myr. In that state the CMB gradient is stable, node 1 sits on its floor and $S_1 - S_0$ is about $10^{-9}$ J kg$^{-1}$ K$^{-1}$. The copied floor makes the CMB power jump with the sign of $S_1 - S_0$, and the energy check of the core refuses the calls.

**What it tests.** The run is resumed from its stored state at 6.769704 Myr and runs to 6.82 Myr on the Habrok cluster, once under the copy rule and once under the CMB node rule, in the same build of PROTEUS. In the first calls the runner also evaluates the CMB power at $S_1 = S_0 \pm 10^{-9}$ and $\pm 10^{-12}$ J kg$^{-1}$ K$^{-1}$ at every output point. The helpfile is the PROTEUS output table, 1 row per coupling step.

**How we know the right answer.** The copy rule stops here, so it gives the failure to compare with. The energy check of each call gives the pass mark: no call above $10 B$.

**Why it matters.** This is the failure that the CMB node rule removes. Under the copy rule the run takes no interior step past 6.7697 Myr.

![CMB heat flux and ledger miss of each call against time, copy rule and CMB node rule, resume of the 8.63 Earth-mass run](../assets/validation/cmb_eddy_flux_case38_resume.svg#only-light){ width="70%" }
![CMB heat flux and ledger miss of each call against time, copy rule and CMB node rule, resume of the 8.63 Earth-mass run](../assets/validation/cmb_eddy_flux_case38_resume_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Time runs from the resume. Panel (a) shows the CMB heat flux $F_\mathrm{cmb}$ of each helpfile row. Panel (b) shows $|r|/B$ of each call on a log scale; the dashed line is the refusal limit of 10. The copy rule (squares) has 3 calls above the limit and stops. The CMB node rule (circles) stays at or below $5.2 \times 10^{-9}$ in every call and runs to 6.82 Myr.

![Difference of the CMB power between the 2 sides of S1 = S0 at every output point of the first calls, copy rule and CMB node rule](../assets/validation/cmb_eddy_flux_case38_sweep.svg#only-light){ width="70%" }
![Difference of the CMB power between the 2 sides of S1 = S0 at every output point of the first calls, copy rule and CMB node rule](../assets/validation/cmb_eddy_flux_case38_sweep_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Each mark is one output point. The vertical axis is the difference of the CMB power in W between $S_1 = S_0 + 10^{-9}$ and $S_1 = S_0 - 10^{-9}$ J kg$^{-1}$ K$^{-1}$. The copy rule steps by $9.1$ to $9.2 \times 10^{16}$ W at every point. The CMB node rule differs by at most $10^{-3}$ W on a CMB power of $8.0 \times 10^{11}$ W, which is the round-off of that power.

??? info "Test details"
    - Reference: the copy rule in the same build (versions in `cmb_eddy_flux_runs_provenance.json`); the energy check of each call.
    - Tolerance: every call below the bound (measured: at most $|r|/B = 5.2 \times 10^{-9}$ in 14 calls); no helpfile row with a CMB heat below $-10^{25}$ J (the copy rule has none either: it stops before an accepted step). Copy rule: 3 calls above $10 B$ (at most 21.1), PROTEUS status 20 (error); CMB node rule: status 13 (completed at the target time).
    - Step test: copy rule $|\Delta P_\mathrm{cmb}| = 9.105$ to $9.212 \times 10^{16}$ W at a step of $10^{-9}$, and at least $9.105 \times 10^{16}$ W at $10^{-12}$, at all 416 points of 3 calls. CMB node rule: at most $9.8 \times 10^{-4}$ W at $10^{-9}$ (relative $1.2 \times 10^{-15}$) and 0 at $10^{-12}$, at all 544 points of 5 calls.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_case_38_runs_where_the_copy_rule_stops`, `tests/test_cmb_eddy_flux_runs.py::test_case_38_cmb_power_has_no_step_at_s1_equal_s0`
    - Data: `tests/validation/data/cmb_eddy_flux_runs.npz`, built by `tests/reference/build_cmb_eddy_flux_runs.py` (versions and checksums in `cmb_eddy_flux_runs_provenance.json`). The tests of the 4 run cases read these stored runs and do not run the code again, so a change of the code does not fail them.
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py` (add `--validation-plots` to write the plot pairs)

## An Earth-mass run where node 1 turns stable

**Why this check exists.** In a run of 1 Earth mass with the `nearest_boundary` mixing length, node 1 turns stable at about 30 kyr while the CMB gradient stays unstable. Under the copy rule the CMB node then copies 0, the flux out of the core falls to about $10^{-3}$ W m$^{-2}$, and the energy check stops the run in the call after its last row at 30952 yr. This case needs the floor of node 0.

**What it tests.** The run is resumed at 30169.3 yr and runs to 32500 yr, under both rules. The test checks the status of both runs, $|r|/B$ of every call, and the CMB flux of the CMB node rule.

**How we know the right answer.** The copy rule stops here. The pass mark is the same energy check as above: no call above $10 B$.

**Why it matters.** Kind d states occur in long Earth-mass runs. If the CMB node loses its floor there, the core is cut off from the mantle.

![CMB heat flux and ledger miss of each call against time, copy rule and CMB node rule, Earth-mass resume with the nearest-boundary mixing length](../assets/validation/cmb_eddy_flux_nearest_resume.svg#only-light){ width="70%" }
![CMB heat flux and ledger miss of each call against time, copy rule and CMB node rule, Earth-mass resume with the nearest-boundary mixing length](../assets/validation/cmb_eddy_flux_nearest_resume_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Panel (a) shows the CMB heat flux on a log scale. For the copy rule (squares) and the CMB node rule (filled circles) it is the average over each solver call, the value of the helpfile. For the CMB node rule it also shows the peak inside each call (up triangles) and the node flux at the end of each call (down triangles). Panel (b) shows $|r|/B$ of each call; the dashed line is the refusal limit of 10. The copy rule falls to about $10^{-3}$ W m$^{-2}$ and stops; the CMB node rule keeps 128 to 6092 W m$^{-2}$ and stays below the line.

**Known limit.** 3 of the 12 calls of the CMB node rule miss by more than the bound: $|r|/B$ = 1.160, 1.248 and 4.323. The cause is not known.

- These are calls with a short peak of the CMB flux inside the call, $2.8 \times 10^4$ to $6.9 \times 10^5$ W m$^{-2}$; 6 of the 8 calls with a peak end between 130 and 350 W m$^{-2}$.
- All 8 calls with $|r|/B$ above 0.3 have such a peak; the 4 calls without a peak stay below 0.004.
- The quadrature error of the CMB heat on the stored points is at most 0.01 $B$, so the trapezoid sum of the stored points does not cause the miss.
- In this state (kind d) the CMB node takes its own floor, so the CMB flux is the floor flux $\rho T \kappa_h^\mathrm{floor}[0]\,|\partial S/\partial r[0]|$: at a given CMB gradient it scales with `kappah_floor`, a numerical setting (10 m$^2$ s$^{-1}$ here; no run with another value was made).

**Known limit.** The helpfile flux is the average over a call, and the snapshot of a call holds the node flux at its end. At 31773 yr the call average is 1776 W m$^{-2}$, while the end state sits in a peak, at $7.738 \times 10^4$ W m$^{-2}$ with a CMB gradient of $-4.0 \times 10^{-4}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$.

??? info "Test details"
    - Reference: the copy rule in the same build (versions in `cmb_eddy_flux_runs_provenance.json`); the energy check of each call.
    - Tolerance: every call below $10 B$. Copy rule: status 20 after its row at 30952.1 yr, $|r|/B$ up to 787. CMB node rule: status 13 at 32500 yr, 12 calls, 3 above 1.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_nearest_mixing_length_resume_runs_where_the_copy_rule_stops`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py`

## An Earth-mass run where the CMB node rule keeps the flux of the copy rule

**Why this check exists.** A rule for one state must not break a run that the copy rule handles. A run of 1 Earth mass with the `constant` mixing length passes through states where a rule without the downward eddy flux over a stable CMB gradient failed in a probe outside this repository.

**What it tests.** The run is resumed at 37854.6 yr and runs to 38600 yr, under both rules. The test compares the CMB flux and the CMB temperature, and checks the energy check of each call.

**How we know the right answer.** The copy rule closes the energy check in every call here, so the CMB node rule must close it too and stay close to the copy rule.

**Why it matters.** This is a state of a normal run where the CMB node rule should change little. The first row after the resume carries heat into the core ($-592$ W m$^{-2}$) under both rules.

![CMB heat flux against time, copy rule and CMB node rule, Earth-mass resume with the constant mixing length](../assets/validation/cmb_eddy_flux_constant_resume.svg#only-light){ width="70%" }
![CMB heat flux against time, copy rule and CMB node rule, Earth-mass resume with the constant mixing length](../assets/validation/cmb_eddy_flux_constant_resume_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Each mark is the CMB heat flux of one helpfile row after the resume. The squares (copy rule) and the circles (CMB node rule) lie on each other.

??? info "Test details"
    - Reference: the copy rule in the same build (versions in `cmb_eddy_flux_runs_provenance.json`).
    - Tolerance: every call below the bound (measured $|r|/B$ at most 0.28 for the copy rule, 0.30 for the CMB node rule); measured differences, with the CMB node rule interpolated to the times of the copy rule: $F_\mathrm{cmb}$ $1.7 \times 10^{-3}$ relative, $T_\mathrm{cmb}$ $1.7 \times 10^{-9}$ relative.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_constant_mixing_length_resume_keeps_the_ledger`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py`

## A run from the start where the rule does not act

**Why this check exists.** In most states the CMB node takes the mixing-length value of node 1, and the 2 rules give the same value. A run in such states must not change.

**What it tests.** The same Earth-mass run with the `constant` mixing length, from $t = 0$ to 5 kyr. An instrumented copy of the code counts the CMB evaluations where the 2 rules differ: in `compute_mlt` (a debug callback) and in the NumPy update. A second run under the copy rule changes the same code line to a form with the same values, $\kappa_h[0] = \max(\kappa_h[1], 0 \cdot \kappa_h^\mathrm{floor}[0])$, which compiles to another graph.

**How we know the right answer.** If the rule never acts, the run of the CMB node rule can differ from the copy rule only as much as the copy rule differs from itself with the same-value edit.

**Why it matters.** It shows that the change of a run in states where the rule does not act is the round-off of the code, not the rule.

| Run | CMB evaluations where the 2 rules differ | Helpfile against the copy rule |
|---|---|---|
| copy rule, second run | not counted (no instrumented copy) | equal in every column except the run time |
| copy rule with the same-value edit | not counted (no instrumented copy) | 136 of 1040 columns differ |
| CMB node rule | 0 of 88499 (`compute_mlt`), 0 of 454 (NumPy) | equal to the copy rule with the same-value edit, bit for bit |
| CMB node rule with the debug callback | 0 of 88499 | equal to the copy rule, bit for bit |

This case has no plot: the compared runs are bit-equal, which the table states.

**Known limit.** This run changes between 2 bit patterns with the compiled graph alone. The start transient grows the round-off: the first difference is at 2 yr, $9.9 \times 10^{-9}$ relative in $F_\mathrm{cmb}$, and by 5 kyr $F_\mathrm{cmb}$ differs by up to 25.6 W m$^{-2}$. A comparison with the copy rule in this run is valid only above this floor.

??? info "Test details"
    - Reference: the copy rule, a second run of it, and the copy rule with the same-value edit (versions in `cmb_eddy_flux_runs_provenance.json`).
    - Tolerance: 0 differing evaluations; the CMB node rule run bit-equal to the copy rule with the same-value edit; each row of the table from the whole stored helpfiles, compared value by value.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_run_from_t0_differs_only_at_the_graph_round_off`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py`

[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
