# CMB eddy flux

## What this page checks

This page checks the eddy diffusivity at the core-mantle boundary (CMB) node, and with it the convective heat flux from the core into the mantle.
The CMB node (node 0) takes the mixing-length eddy diffusivity of node 1, the node above it, or its own floor, whichever is larger ([Eddy diffusivity at the CMB node](../Explanations/heat_transport.md#eddy-diffusivity-at-the-cmb-node)).
We call this the CMB node rule. The cases compare it with the copy rule, where the CMB node copies the floored value of node 1.
If this part were wrong, the heat flow out of the core could jump between 2 values when the entropies of the 2 lowest cells cross, or fall to 0 while the CMB gradient is unstable.
The core would then cool at the wrong rate: the energy check can stop the run, or the run can go on with the core almost cut off from the mantle.
A low CMB flux is not always a fault: when the CMB node and node 1 are both stable, so that a stable layer sits on the CMB, the CMB node has no eddy diffusivity (with $D = 0$, the default); heat crosses the CMB by conduction only ([A stable mush layer at the base of the mantle](#a-stable-mush-layer-at-the-base-of-the-mantle)).

## How the cases are checked

Validation here means that we run the code on a case where we know the right answer, and check that the code gives that answer.
The right answer comes from 3 sources:

- the rule written out by hand in the test;
- a second implementation: the NumPy update and the JAX function `compute_mlt`;
- coupled PROTEUS runs: the energy check of each solver call (below) gives the pass mark, and runs under the copy rule give the failure to compare with; 2 of these copy-rule runs stop, and in 3 the core stays almost cut off from the mantle for 19 to 29 kyr.

Each case says why the check exists, what it tests, how we know the right answer and why it matters; then it shows a plot or a table, and the test details.

## Symbols and terms

- $S_0$ and $S_1$ are the entropies of the 2 lowest cells, and $\partial S/\partial r$ is the entropy gradient at a node.
- The floor is the minimum eddy diffusivity that `kappah_floor` and the melt fraction set on nodes with $\partial S/\partial r < D$ ([Phase-modulated floor](../Explanations/heat_transport.md#phase-modulated-floor)). A node is on its floor when its mixing-length value is below the floor.
- $\kappa_h^\mathrm{mlt}[1]$ is the mixing-length eddy diffusivity of node 1 before the floor, $\kappa_h^\mathrm{floor}[0]$ is the floor of the CMB node, and $D$ is `energy.convective_threshold` (0 unless set).
- The CMB heat flux is in W m$^{-2}$. The CMB power, in W, is that flux times the area of the CMB. The CMB heat of a call, in J, is the energy that crosses the CMB during the call.
- $F_\mathrm{cmb}$ (column `F_cmb` of the PROTEUS helpfile) is the CMB heat flux averaged over one solver call. A snapshot (the file `<time>_int.nc` of a call) holds the state of every node at the end of the call; the flux of the CMB node in a snapshot is the flux at that moment.
- The low-flux interval of a run is the span of helpfile rows with $F_\mathrm{cmb}$ below 1 W m$^{-2}$, from the first such row to the first row back above 1 W m$^{-2}$.
- $T_\mathrm{cmb}$ (column `T_cmb`) is the temperature of the lowest mantle cell, called $T_0$ in a snapshot; $T_\mathrm{cmb,node}$ (column `T_cmb_node`) is the temperature of the CMB node.
- A stable layer is the set of nodes, counted up from the CMB node, where the entropy rises with height ($\partial S/\partial r > 0$), so that the fluid does not convect. A stable node above an unstable CMB node is not part of a stable layer.
- The energy check is the [ledger check of the CMB closure](../Explanations/core_bc.md#ledger-check-of-the-cmb-closure): in each solver call the change of the core heat must match the CMB heat of the call. $r$ is the miss and $B$ the bound that the solver tolerances allow. A call with $|r| \le B$ is within the bound, a call with $B < |r| \le 10 B$ is kept with a warning, and a call with $|r| > 10 B$ is refused.
- rtol and atol are the relative and the absolute tolerance that the CVODE time integrator holds in each step. Aragog applies atol in the units of each state variable and raises an atol below $10^{-8}$ to $10^{-8}$, so a tighter atol is not available.
- With the `energy_balance` core the gradient at the CMB node is a state variable, so its atol is $10^{-8}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ at any rtol.
- The runs of the 2 cases on the stable layer use atol $10^{-8}$. In the calls that the solver retried with half the step, atol is 3 or 5 times larger. A call retried after the solver reached its step limit runs at twice the rtol and keeps atol $10^{-8}$.
- A result that changes when a tolerance is made tighter is set partly by the integration error.

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

**Why this check exists.** The NumPy `EntropyState.update` sets the eddy diffusivity and the convective flux $J$ of the CMB node from 2 recorded inputs, $\kappa_h^\mathrm{mlt}[1]$ and $\kappa_h^\mathrm{floor}[0]$. The check `check_cmb_convective_flux` tests $J$ against those inputs in every update, so an edit that sets $J$ or $\kappa_h[0]$ against the rule fails at once. The check takes the 2 inputs from the same mixing-length call that sets $\kappa_h[0]$, so it tests the value that this call gives the CMB node, not how the call computes the inputs: an edit that takes the floor of node 0 from the melt fraction of node 1 passes the check and the other tests of `tests/test_cmb_eddy_diffusivity.py`, and fails the test on random states above. The check runs on the NumPy path (the energy replay, the NumPy right-hand side and the update of the end state of each call), not inside the JAX right-hand side; the known-answer tests above check the JAX function `compute_mlt`.

**What it tests.** The check raises `CMBEddyFluxError` when:

- (i) $\partial S/\partial r[0] \ge D$ and $|J| > \rho T \kappa_h^\mathrm{mlt}[1]\,|\partial S/\partial r[0]|$: a floor of node 1 crosses a stable CMB;
- (ii) $\partial S/\partial r[0] < D$ and $|J| < \rho T \kappa_h^\mathrm{floor}[0]\,|\partial S/\partial r[0]|$: the CMB node lost its own floor;
- (iii) $J\,\partial S/\partial r[0] > 0$: the flux runs against its gradient;
- (iv) $J$ differs from $-\rho T \max(\kappa_h^\mathrm{mlt}[1], \kappa_h^\mathrm{floor}[0])\,\partial S/\partial r[0]$ by more than $10^{-12}$ relative: any other flux, for example one with the mixing-length part of node 1 dropped.

The check skips a state where $J$, $\rho T$, the gradient or one of the 2 inputs is not finite (NaN or infinite).
The tests pass each kind of state through the check, give it a flux that breaks each part, and confirm that the update calls it with the values it used. 2 solver tests make the check fail inside a solve.

**How we know the right answer.** The 4 parts follow from the rule: $\kappa_h[0]$ is at least the floor of node 0, equals $\kappa_h^\mathrm{mlt}[1]$ over a stable gradient, is never negative, and is the larger of the 2 inputs.

**Why it matters.** On the JAX path the check sees the end state of each call; on the NumPy path (scipy, or CVODE without the JAX factory) it also sees every state of the energy replay and of the right-hand side. A call that breaks the rule fails like a refused ledger call: in the NumPy CVODE right-hand side it ends the call, and in the energy replay the solver goes back to the solution held at the call start. The error names the part.

This case has no plot: each test is a single pass or refusal.

??? info "Test details"
    - Reference: the 4 parts of the rule above.
    - Tolerance: 1e-12 relative on parts (i), (ii) and (iv).
    - Test id: `tests/test_cmb_eddy_diffusivity.py::test_invariant_passes_the_rule`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_refuses_each_broken_part`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_refuses_a_flux_between_the_bounds`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_passes_round_off_of_the_rule`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_uses_the_threshold`, `tests/test_cmb_eddy_diffusivity.py::test_invariant_skips_a_state_that_is_not_finite`, `tests/test_cmb_eddy_diffusivity.py::test_update_checks_the_invariant`, `tests/test_cmb_eddy_diffusivity.py::test_a_flux_fault_in_the_cvode_rhs_reaches_the_caller`, `tests/test_cmb_eddy_diffusivity.py::test_a_flux_fault_keeps_the_last_accepted_state`
    - Command: `pytest tests/test_cmb_eddy_diffusivity.py -k "invariant or fault"`

## A large planet runs through the state where the copy rule stops

**Why this check exists.** Under the copy rule a run of 8.63 Earth masses stops at 6.77 Myr. In that state the CMB gradient is stable, node 1 sits on its floor and $S_1 - S_0$ is about $10^{-9}$ J kg$^{-1}$ K$^{-1}$. The copied floor makes the CMB power jump with the sign of $S_1 - S_0$, and the energy check of the core refuses the calls.

**What it tests.** The run is resumed from its stored state at 6.769704 Myr and runs to 6.82 Myr on the Habrok cluster, once under the copy rule and once under the CMB node rule, in the same build of PROTEUS. In the first calls the runner also evaluates the CMB power at $S_1 = S_0 \pm 10^{-9}$ and $\pm 10^{-12}$ J kg$^{-1}$ K$^{-1}$ at every output point. The helpfile is the PROTEUS output table, 1 row per coupling step.

**How we know the right answer.** The copy rule stops here, so it gives the failure to compare with. The energy check of each call gives the pass mark: no call above $10 B$.

**Why it matters.** This is the failure that the CMB node rule removes. Under the copy rule the run takes no interior step past 6.7697 Myr.

![CMB heat flux and miss of the energy check of each call against time, copy rule and CMB node rule, resume of the 8.63 Earth-mass run](../assets/validation/cmb_eddy_flux_case38_resume.svg#only-light){ width="70%" }
![CMB heat flux and miss of the energy check of each call against time, copy rule and CMB node rule, resume of the 8.63 Earth-mass run](../assets/validation/cmb_eddy_flux_case38_resume_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Time runs from the resume. Panel (a) shows the CMB heat flux $F_\mathrm{cmb}$ of each helpfile row. Panel (b) shows $|r|/B$ of each call on a log scale; the dashed line is the refusal limit of 10. The copy rule (squares) has 3 calls above the limit and stops. The CMB node rule (circles) stays at or below $5.2 \times 10^{-9}$ in every call and runs to 6.82 Myr.

![Difference of the CMB power between the 2 sides of S1 = S0 at every output point of the first calls, copy rule and CMB node rule](../assets/validation/cmb_eddy_flux_case38_sweep.svg#only-light){ width="70%" }
![Difference of the CMB power between the 2 sides of S1 = S0 at every output point of the first calls, copy rule and CMB node rule](../assets/validation/cmb_eddy_flux_case38_sweep_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Each mark is one output point. The vertical axis is the difference of the CMB power in W between $S_1 = S_0 + 10^{-9}$ and $S_1 = S_0 - 10^{-9}$ J kg$^{-1}$ K$^{-1}$. The copy rule steps by $9.1$ to $9.2 \times 10^{16}$ W at every point. The CMB node rule differs by at most $10^{-3}$ W on a CMB power of $8.0 \times 10^{11}$ W, which is the round-off of that power.

??? info "Test details"
    - Reference: the copy rule in the same build (versions in `cmb_eddy_flux_runs_provenance.json`); the energy check of each call.
    - Tolerance: every call below the bound (measured: at most $|r|/B = 5.2 \times 10^{-9}$ in 14 calls); no helpfile row with a CMB heat below $-10^{25}$ J (the copy rule has none either: it stops before an accepted step). Copy rule: 3 calls above $10 B$ (at most 21.1), PROTEUS status 20 (error); CMB node rule: status 13 (completed at the target time).
    - Step test: copy rule $|\Delta P_\mathrm{cmb}| = 9.105$ to $9.212 \times 10^{16}$ W at a step of $10^{-9}$, and at least $9.105 \times 10^{16}$ W at $10^{-12}$, at all 416 points of 3 calls. CMB node rule: at most $9.8 \times 10^{-4}$ W at $10^{-9}$ (relative $1.2 \times 10^{-15}$) and 0 at $10^{-12}$, at all 544 points of 5 calls.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_case_38_runs_where_the_copy_rule_stops`, `tests/test_cmb_eddy_flux_runs.py::test_case_38_cmb_power_has_no_step_at_s1_equal_s0`
    - Data: `tests/validation/data/cmb_eddy_flux_runs.npz`, built by `tests/reference/build_cmb_eddy_flux_runs.py` (versions and checksums in `cmb_eddy_flux_runs_provenance.json`). The tests of the 8 run cases read these stored runs and do not run the code again, so a change of the code does not fail them.
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py` (add `--validation-plots` to write the plot pairs)

## An Earth-mass run where node 1 turns stable

**Why this check exists.** In a run of 1 Earth mass with the `nearest_boundary` mixing length, node 1 turns stable at about 30 kyr while the CMB gradient stays unstable. Under the copy rule the CMB node then copies 0, the flux out of the core falls to about $10^{-3}$ W m$^{-2}$, and the energy check stops the run in the call after its last row at 30952 yr. This case needs the floor of node 0.

**What it tests.** The run is resumed at 30169.3 yr and runs to 32500 yr, under both rules. The test checks the status of both runs, $|r|/B$ of every call, and the CMB flux of the CMB node rule.

**How we know the right answer.** The copy rule stops here. The pass mark is the same energy check as above: no call above $10 B$.

**Why it matters.** Kind d states occur in long Earth-mass runs. If the CMB node loses its floor there, the core is cut off from the mantle.

![CMB heat flux and miss of the energy check of each call against time, copy rule and CMB node rule, Earth-mass resume with the nearest-boundary mixing length](../assets/validation/cmb_eddy_flux_nearest_resume.svg#only-light){ width="70%" }
![CMB heat flux and miss of the energy check of each call against time, copy rule and CMB node rule, Earth-mass resume with the nearest-boundary mixing length](../assets/validation/cmb_eddy_flux_nearest_resume_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Panel (a) shows the CMB heat flux on a log scale. For the copy rule (squares) and the CMB node rule (filled circles) it is the average over each solver call, the value of the helpfile. For the CMB node rule it also shows the peak inside each call (up triangles) and the node flux at the end of each call (down triangles). Panel (b) shows $|r|/B$ of each call; the dashed line is the refusal limit of 10. The copy rule falls to about $10^{-3}$ W m$^{-2}$ and stops; the CMB node rule keeps 128 to 6092 W m$^{-2}$ and stays below the line.

**Known limit.** 3 of the 12 calls of the CMB node rule miss by more than the bound: $|r|/B$ = 1.160, 1.248 and 4.323. The misses are consistent with integration error of the solver at the Aragog rtol of this run, $10^{-8}$: at rtol $10^{-10}$ no call of the 60 kyr run of this case is above the bound, while at $10^{-9}$ the call that ends at 31979.6 yr misses by 5.51 $B$, against 0.71 $B$ at $10^{-8}$ in this run ([The misses of the energy check and the solver tolerance](#the-misses-of-the-energy-check-and-the-solver-tolerance)).

- These are calls with a short peak of the CMB flux inside the call, $2.8 \times 10^4$ to $6.9 \times 10^5$ W m$^{-2}$; 6 of the 8 calls with a peak end between 130 and 350 W m$^{-2}$.
- All 8 calls with $|r|/B$ above 0.3 have such a peak; the 4 calls without a peak stay below 0.004.
- The quadrature error of the CMB heat on the stored points is at most 0.01 $B$, so the trapezoid sum of the stored points does not cause the miss.
- The peaks are not settled values: the call that ends at 32394.5 yr, rerun at rtol $10^{-9}$ from its start state in the 60 kyr run below, misses by 0.24 $B$, and its CMB heat is 3.2 times smaller.
- In this state (kind d) the CMB node takes its own floor, so the CMB flux is the floor flux $\rho T \kappa_h^\mathrm{floor}[0]\,|\partial S/\partial r[0]|$: at a given CMB gradient it scales with `kappah_floor`, a numerical setting (10 m$^2$ s$^{-1}$ here; no run with another value was made).

**Known limit.** The helpfile flux is the average over a call, and the snapshot of a call holds the node flux at its end. At 31773 yr the call average is 1776 W m$^{-2}$, while the end state sits in a peak, at $7.738 \times 10^4$ W m$^{-2}$ with a CMB gradient of $-4.0 \times 10^{-4}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$.

??? info "Test details"
    - Reference: the copy rule in the same build (versions in `cmb_eddy_flux_runs_provenance.json`); the energy check of each call.
    - Tolerance: every call below $10 B$. Copy rule: status 20 after its row at 30952.1 yr, $|r|/B$ up to 787. CMB node rule: status 13 at 32500 yr, 12 calls, 3 above 1.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_nearest_mixing_length_resume_runs_where_the_copy_rule_stops`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py`

## The case of issue #113 to 60 kyr

**Why this check exists.** [Aragog issue #113](https://github.com/FormingWorlds/aragog/issues/113) reports that in this Earth-mass case under the copy rule the CMB heat flux stays near 0.1 W m$^{-2}$ from 30.5 to 57.3 kyr, while the lowest cell falls 428 K below the CMB node. The resume above covers only the start of this interval.

**What it tests.** The run above, resumed at 30169.3 yr, runs to 60 kyr under the CMB node rule. The runs use the `energy_balance` core. In the PROTEUS helpfile, $T_\mathrm{cmb}$ (column `T_cmb`) is the temperature of the lowest mantle cell and $T_\mathrm{cmb,node}$ (column `T_cmb_node`) the temperature of the CMB node, which is the core temperature of the [energy balance](../Explanations/core_bc.md#energy_balance-default). The run is compared with the run of the #113 case under the copy rule, from $t = 0$ to 140 kyr in an older build whose configuration sets the same rtol of $10^{-8}$ (versions in `cmb_eddy_flux_runs_provenance.json`); that run wrote the state of the resume, and it gives the rows of the table in the issue. In the build of the copy-rule run of the case above, the copy rule stops in this case at 30952 yr. The test checks the CMB heat flux $F_\mathrm{cmb}$, the difference $T_\mathrm{cmb} - T_\mathrm{cmb,node}$ and the energy check of each call.

**How we know the right answer.** This run has no exact answer. The check is that the core stays coupled to the mantle, as in the runs of this case with the constant mixing length or 160 levels, which issue #113 reports without a low-flux interval, and that no call is refused.

**Why it matters.** The heat flow out of the core sets how fast the core cools and how much heat the base of the mantle gets. Under the copy rule of the older build this case runs for 27 kyr with almost no heat out of the core, and the run does not stop.

![CMB heat flux and the temperature difference between the lowest cell and the CMB node against time, copy rule and CMB node rule, the case of issue 113 to 60 kyr](../assets/validation/cmb_eddy_flux_seal_60kyr.svg#only-light){ width="70%" }
![CMB heat flux and the temperature difference between the lowest cell and the CMB node against time, copy rule and CMB node rule, the case of issue 113 to 60 kyr](../assets/validation/cmb_eddy_flux_seal_60kyr_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Panel (a) shows $F_\mathrm{cmb}$ of each helpfile row on a log scale, panel (b) the difference $T_\mathrm{cmb} - T_\mathrm{cmb,node}$. The dashed line is the copy rule, the solid line the CMB node rule. The copy rule falls to between $10^{-3}$ and 0.1 W m$^{-2}$ at 30.5 kyr and returns after a spike at 57.5 kyr. Between 30.5 and 57.3 kyr the CMB node rule keeps a median of 1953 W m$^{-2}$, and the lowest cell stays 8.2 to 9.6 K below the node.

**Known limit.** In these states (kind d) the CMB flux of the CMB node rule is the floor flux of node 0, which scales with `kappah_floor`, a numerical setting (10 m$^2$ s$^{-1}$ here). The check shows that the core stays coupled; it does not test the size of the flux.

**Known limit.** 1 of the 131 rows of the CMB node rule is below 1 W m$^{-2}$, at 43665 yr, the call between the calls with the largest and the third-largest misses of the energy check (7.20 $B$ before it, 5.69 $B$ after it). At rtol $10^{-9}$ and $10^{-10}$ no row is below 1 W m$^{-2}$ ([The misses of the energy check and the solver tolerance](#the-misses-of-the-energy-check-and-the-solver-tolerance)).

??? info "Test details"
    - Reference: the copy rule in the run of issue #113 (an older build, versions in `cmb_eddy_flux_runs_provenance.json`); the energy check of each call.
    - Tolerance: every call below $10 B$. Copy rule: the rows of the issue table, $F_\mathrm{cmb} = 1.4 \times 10^{-3}$ W m$^{-2}$ at 30546 yr (1 %), $T_\mathrm{cmb} = 4811.5$ K and $T_\mathrm{cmb,node} = 5239.9$ K at 50 kyr (0.05 K); median $F_\mathrm{cmb}$ 0.10 W m$^{-2}$ from 30.5 to 57.3 kyr, 116 of 130 rows below 1 W m$^{-2}$, 1 row above $10^4$ W m$^{-2}$ at 57.5 kyr, $T_\mathrm{cmb} - T_\mathrm{cmb,node}$ down to $-428.5$ K. CMB node rule: status 13 at 60 kyr at each rtol, median 1953 W m$^{-2}$, 1 of 131 rows below 1 W m$^{-2}$ (at 43665 yr, between the calls with misses 7.20 and 5.69 $B$), $-9.55$ to $-8.17$ K; no row below 1 W m$^{-2}$ at rtol $10^{-9}$ and $10^{-10}$.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_seal_case_to_60_kyr_keeps_the_core_coupled`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py` (add `--validation-plots` to write the plot pair)

## The misses of the energy check and the solver tolerance

**Why this check exists.** In the Earth-mass runs with the `nearest_boundary` mixing length, some calls of the CMB node rule miss the energy check by more than the bound $B$, though below the refusal limit. This case shows how the misses depend on the solver tolerance.

**What it tests.** rtol is the relative tolerance that the CVODE time integrator holds in each step. The case runs the 60 kyr run of the case above at the Aragog rtol $10^{-8}$, $10^{-9}$ and $10^{-10}$, and single calls rerun from a stored start state at rtol $10^{-8}$ to $10^{-11}$. PROTEUS sets rtol $10^{-8}$ for Aragog when a configuration does not set it. The run at $10^{-10}$ takes 34917 solver steps, against 61781 at $10^{-8}$ and 66110 at $10^{-9}$; this page does not test why.

**How we know the right answer.** In each call the change of the core heat must equal the CMB heat of the call: energy is conserved. If the misses fall as rtol gets tighter, the equations with the CMB node rule conserve energy, and the misses come from the time integration. Here the median miss of the 131 calls falls from $4.7 \times 10^{-3}$ to $3.4 \times 10^{-4}$ and $1.0 \times 10^{-5}$ $B$ at rtol $10^{-8}$, $10^{-9}$ and $10^{-10}$; the largest miss falls from 7.20 to 5.51 $B$ at $10^{-9}$ and to 0.061 $B$ at $10^{-10}$, and single calls do not fall in step: the call that ends at 31979.6 yr misses by 0.71 $B$ in the run at $10^{-8}$ and by 5.51 $B$ in the run at $10^{-9}$. Whether the rule makes the equations harder to integrate is not tested here. The energy check tests this balance only: a call can miss by less than $B$ while its CMB heat still changes with rtol (see the known limit below).

**Why it matters.** At the default rtol a call can be kept by the energy check with a CMB heat far from its value at tight rtol: the call that ends at 43407.5 yr misses by 7.20 $B$, below the refusal limit, with a CMB heat 3.59 times its value at rtol $10^{-9}$ to $10^{-11}$.

![Miss of the energy check of each call, sorted from the largest, at 3 solver tolerances, the case of issue 113 to 60 kyr](../assets/validation/cmb_eddy_flux_seal_rtol.svg#only-light){ width="70%" }
![Miss of the energy check of each call, sorted from the largest, at 3 solver tolerances, the case of issue 113 to 60 kyr](../assets/validation/cmb_eddy_flux_seal_rtol_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The horizontal axis numbers the 131 calls of one run, the call with the largest miss first; each run is sorted on its own, so one position on the axis is not the same call in the 3 runs. The vertical axis is $|r|/B$ on a log scale. The coloured solid, dashed and dotted lines are the runs at rtol $10^{-8}$, $10^{-9}$ and $10^{-10}$. The thin dotted line in the text colour is the bound ($|r|/B = 1$), the thin dash-dot line the refusal limit of 10. At rtol $10^{-8}$ 15 calls are above the bound (largest 7.20), at $10^{-9}$ 13 calls (largest 5.51), at $10^{-10}$ none (largest 0.061).

The single calls give $|r|/B$ at each rtol:

| Call ends at | Start state from | $10^{-8}$ | $10^{-9}$ | $10^{-10}$ | $10^{-11}$ |
|---|---|---|---|---|---|
| 43407.5 yr | run at $10^{-8}$ | 7.20 | $6.1 \times 10^{-5}$ | $6.8 \times 10^{-5}$ | $9.2 \times 10^{-6}$ |
| 32394.5 yr | run at $10^{-8}$ | 4.32 | 0.24 | not run | not run |
| 31979.6 yr | run at $10^{-9}$ | not run | 5.51 | 0.066 | $2.7 \times 10^{-4}$ |

At rtol $10^{-9}$, $10^{-10}$ and $10^{-11}$ the CMB heat of the call that ends at 43407.5 yr is the same within $10^{-5}$ relative; at $10^{-8}$ it is 3.59 times larger.

**Known limit.** The call that ends at 31979.6 yr misses by less than $B$ at rtol $10^{-10}$ and $10^{-11}$, but its CMB heat still changes: $2.84$, $2.63$ and $2.16 \times 10^{27}$ J at $10^{-9}$, $10^{-10}$ and $10^{-11}$. Its CMB heat is not settled at $10^{-11}$.

??? info "Test details"
    - Reference: the energy check of each call (the core heat change against the CMB heat of the call).
    - Tolerance: each value quoted here equals the stored value rounded to the quoted digits; the number of calls above the bound and the solver steps exactly; the 3 heats of the call that ends at 43407.5 yr within $10^{-5}$ of each other; every call below $10 B$ at each rtol.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_seal_case_ledger_misses_fall_with_rtol`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py` (add `--validation-plots` to write the plot pair)

## A stable mush layer at the base of the mantle

**Why this check exists.** With a CMB cell of 750 m, a run under the CMB node rule holds $F_\mathrm{cmb}$ near $10^{-3}$ W m$^{-2}$ from 14.8 to 63.3 kyr. A low flux is also the sign of the fault of issue #113, where the CMB node is unstable but has no eddy diffusivity. The test tells the 2 apart: it checks the state of the nodes in every snapshot of the low-flux interval.

**What it tests.** The case of issue #113 from $t = 0$ with a CMB cell of 750 m (`cmb_cell_thickness`) and the `nearest_boundary` mixing-length profile, in 5 stored runs. Each run continued once from its stored state; runs with the same build letter use the same build of Aragog, and `cmb_eddy_flux_runs_provenance.json` lists the build of each run.

| Run (npz key) | Rule | rtol | Time span | Continued at | Build | What it shows |
|---|---|---|---|---|---|---|
| `basal750_tip` | CMB node rule | $10^{-8}$ | 0 to 140 kyr | 60 kyr | a | the stable layer |
| `basal750_main` | copy rule | $10^{-8}$ | 0 to 140 kyr | 60 kyr | b | the failure to compare with |
| `basal750_tip_r10` | CMB node rule | $10^{-10}$ | 0 to 70 kyr | 45.1 kyr | c | which results the integration error sets |
| `basal750_tip_rerun` | CMB node rule | $10^{-8}$ | 0 to 70 kyr | 42.8 kyr | c | that build c gives the rows of `basal750_tip` |
| `basal750_main_r10` | copy rule | $10^{-10}$ | 0 to 46.9 kyr | 14682.0 yr | d | the copy rule past the end of its low-flux interval |

$S_0$ and $T_0$ are the entropy and the temperature of the lowest cell. $S_\mathrm{above}$ and $T_\mathrm{above}$ are those of the cell just above the last stable node, at the top of the stable layer. The test reads the snapshots from 10 to 70 kyr. Under the CMB node rule it checks the low-flux interval, the state of the nodes in each snapshot inside it, the entropy and temperature steps across the stable layer, and the snapshots just before and after the layer. Under the copy rule it checks the low-flux interval and the state of the CMB node in it. For both rules it checks $T_\mathrm{cmb}$ after the layer. The test details list each checked value.

**How we know the right answer.** This run has no exact answer. When nodes 0 and 1 are both stable, the mixing-length theory gives node 1 no eddy diffusivity. With $D = 0$, the default, the floor acts only on unstable nodes. So the CMB node rule gives the CMB node no eddy diffusivity either, and the heat flux at the CMB is conduction only. The low flux holds only with a stable CMB node: if the CMB node were unstable, it would take its floor and carry an eddy flux $\rho_0 T_0 \kappa_h^\mathrm{floor}[0]\,|\partial S/\partial r[0]|$ of 29.5 to 12700 W m$^{-2}$ in the snapshots of the low-flux interval (medians 2160 W m$^{-2}$ at rtol $10^{-8}$ and 1670 W m$^{-2}$ at $10^{-10}$), against the stored $10^{-3}$ W m$^{-2}$ and of the order of the flux after the layer.

The stable layer lasts while the entropy rises across it. In the run, $S_0$ rises by 0.05 J kg$^{-1}$ K$^{-1}$, while $S_\mathrm{above}$ falls from 7018 to 5508.5 J kg$^{-1}$ K$^{-1}$, slowly from 22 to 42.4 kyr and faster after. In the last snapshot of the layer the step $S_\mathrm{above} - S_0$ is 4.6 J kg$^{-1}$ K$^{-1}$; in the next snapshot, 601 yr later, nodes 0 and 1 are unstable. This agrees with a layer that ends when the entropy at its top has fallen to that of the lowest cell; the stored output does not show the end itself.

The copy rule gives the failure to compare with. In its low-flux interval the gradient at the CMB node is negative in all 61 snapshots; in the 54 where node 1 is stable, the CMB node has no eddy diffusivity, the fault of issue #113. In these 54 the gradient is at least $10^{-6}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ in size, 100 times its tolerance; in the other 7, from 14.7 to 16.5 kyr, it is at most $2.9 \times 10^{-9}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ in size, so the run does not resolve its sign. Inside that interval the lowest cell is 0.17 to 0.38 K colder than the CMB node. Before it, from 3.9 to 10.7 kyr, the lowest cell of the copy rule is more than 100 K below the CMB node, by up to 923.5 K at 5405 yr. Under the CMB node rule the difference stays between $-1.78$ and 2.52 K over the whole run. The energy check of each call gives the pass mark: no call above the bound $B$.

**Why it matters.** The core loses almost no heat while the stable layer lasts, so the time when the layer ends sets when the core starts to cool again. In the 750 m run under the CMB node rule the low flux comes from a stable layer of the model; under the copy rule it comes mostly from the fault of issue #113.

![CMB heat flux and temperature of the lowest mantle cell against time, copy rule and CMB node rule, the case of issue 113 with a 750 m CMB cell to 140 kyr](../assets/validation/cmb_eddy_flux_basal_layer_750m.svg#only-light){ width="70%" }
![CMB heat flux and temperature of the lowest mantle cell against time, copy rule and CMB node rule, the case of issue 113 with a 750 m CMB cell to 140 kyr](../assets/validation/cmb_eddy_flux_basal_layer_750m_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Panel (a) shows $F_\mathrm{cmb}$ of each helpfile row on a log scale; the dotted line is 1 W m$^{-2}$. Panel (b) shows $T_\mathrm{cmb}$. The dashed line is the copy rule, the solid line the CMB node rule. The CMB node rule holds $F_\mathrm{cmb}$ near $10^{-3}$ W m$^{-2}$ and $T_\mathrm{cmb}$ flat from 14.8 to 63.3 kyr. From the row at 63338.1 yr, the first with $F_\mathrm{cmb}$ back above 1 W m$^{-2}$, the 2 lines differ by at most 3.01 K, at that row; they differ by less than 1 K after 93.9 kyr.

![Height of the top of the stable layer above the CMB and entropy step above the layer against time, CMB node rule, 750 m CMB cell](../assets/validation/cmb_eddy_flux_basal_layer_750m_step.svg#only-light){ width="70%" }
![Height of the top of the stable layer above the CMB and entropy step above the layer against time, CMB node rule, 750 m CMB cell](../assets/validation/cmb_eddy_flux_basal_layer_750m_step_dark.svg#only-dark){ width="70%" }

**How to read the plot.** Both panels show the snapshots of the CMB node rule inside the low-flux interval. Panel (a) shows the height of the top stable node above the CMB in km; it changes in steps of whole nodes. Panel (b) shows $S_\mathrm{above} - S_0$ on a log scale. The step falls between every 2 snapshots and is 4.6 J kg$^{-1}$ K$^{-1}$ in the last snapshot before the layer ends.

**Known limit.** A stable layer that holds the flux low forms with the 750 m cell and not with the 3000 m cell ([With a 3000 m CMB cell no layer holds the flux low](#with-a-3000-m-cmb-cell-no-layer-holds-the-flux-low)), so the result depends on the mesh. No run with a thinner CMB cell or more nodes was made, and no run with the `constant` profile. This page does not test whether the layer is resolved.

**Known limit.** Some results depend on the solver tolerance. In `basal750_tip_r10` nodes 0 and 1 are stable in every snapshot of the low-flux interval, and $T_\mathrm{cmb}$ stays at 4789.2 K, as in `basal750_tip`. In both runs the layer is already there in the snapshot at 14495.2 yr; in `basal750_tip_r10` the call that ends there ran at rtol $2 \times 10^{-10}$, after the solver reached its step limit at $10^{-10}$. The low-flux interval starts at 14788.2 yr, against 14788.5 yr, and ends at 63632.1 yr, against 63338.1 yr. The layer is thinner at $10^{-10}$: it has 4 stable nodes, up to node 3, in 91 of the 100 snapshots, and 3 to 5 nodes and at most 3.41 km in all, against 3 to 8 nodes and 6.80 km at $10^{-8}$. $S_\mathrm{above}$ belongs to the cell above the top node, so the step is larger at $10^{-10}$. From 22.3 to 40.0 kyr, with 4 stable nodes, it falls from 971.9 to 964.1 J kg$^{-1}$ K$^{-1}$; at $10^{-8}$, with 5 stable nodes, it falls from 876.3 to 869.3 J kg$^{-1}$ K$^{-1}$ from 22.3 to 42.4 kyr. Both runs fall slowly in that time.

**Known limit.** The margin of the sign of the CMB gradient is small at the start of the layer. The 30 snapshots with $|\partial S/\partial r[0]| < 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ are the first 30 of the low-flux interval in both runs, to 26133.7 yr at rtol $10^{-8}$ and 26118.6 yr at $10^{-10}$; the snapshot at 14495.2 yr before them has the smallest gradient, and there the flux bound above is 33.2 and 13.6 W m$^{-2}$. After them the gradient is at least $10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$, 1000 times atol, over 75.9 % ($10^{-8}$) and 76.1 % ($10^{-10}$) of the time of the layer. The gradient at node 0 is the state `dSdr_cmb` of the solver, with the tolerance $10^{-8}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ at both rtol; in the interval its smallest value is 59 times that tolerance at rtol $10^{-8}$ and 35.4 times at $10^{-10}$ (39.9 and 16.3 times at 14495.2 yr). The gradient at node 1 is $(S_1 - S_0)/\Delta r$ over the 782 m between the centres of cells 0 and 1, so its tolerance is that of the 2 cell entropies over $\Delta r$; its smallest value is 3.5 times that tolerance at rtol $10^{-8}$ (0.96 times at 14495.2 yr) and 310 times at $10^{-10}$ (70 times). The run at $10^{-10}$ therefore carries the sign of node 1. These are tolerances of one step, not bounds on the error of the run: CVODE accepts a step when the root mean square of the local errors over the $N = 80$ state variables (79 cells and the CMB gradient), each divided by its tolerance, is at most 1, so one variable can take $\sqrt{N} \approx 8.9$ times its tolerance in a step, and the errors of many steps add up. Against $\sqrt{N}$ times the tolerance the smallest margins are 6.6 and 4.0 at node 0 and 0.55 and 50 at node 1. In the 5 calls of `basal750_tip_r10` retried with half the step and atol 3 or 5 times larger, from 29.8 to 32.1 kyr, the gradient at node 0 is at least $1.5 \times 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$. A check with another solver or another scaling of the CMB gradient is open.

**Known limit.** The layer ends between 2 snapshots 601 yr apart (62736.9 and 63338.1 yr), so the stored output does not show how it ends.

**Known limit.** The runs come from 4 builds of Aragog. `basal750_tip_rerun` repeats `basal750_tip` from $t = 0$ with build c and gives the same helpfile rows to 59939.2 yr, with $F_\mathrm{cmb}$ within $10^{-15}$ relative and the same $T_\mathrm{cmb}$; it continues from its stored state at 42.8 kyr, and its rows after that time are the same too. `basal750_tip` stops at 60 kyr and continues from its state there, so its call that starts at 59939.2 yr ends at 60000 yr, and the call times differ after that. In the repeat the low-flux interval ends at 62950.0 yr instead of 63338.1 yr, 388 yr or about 1 call earlier: the end of the interval depends on the call times. The 2 copy-rule runs differ in build and in rtol, and `basal750_main_r10` continued from its stored state at the first row of its low-flux interval, so only its end and its node states compare with `basal750_main`: the interval ends at 44283.1 yr against 44087.6 yr, and in both runs the CMB node has no eddy diffusivity exactly where node 1 is stable.

??? info "Test details"
    - Reference: the copy rule; the energy check of each call; the node states of the stored snapshots from 10 to 70 kyr; `cmb_eddy_flux_runs_provenance.json` lists the build of each run.
    - Checks of `basal750_tip` (CMB node rule, rtol $10^{-8}$):
        - 1 interval with $F_\mathrm{cmb}$ below 1 W m$^{-2}$, from the row at 14788.5 yr to the first row back above 1 W m$^{-2}$, at 63338.1 yr, with $F_\mathrm{cmb}$ from $9.94 \times 10^{-4}$ to $1.09 \times 10^{-3}$ W m$^{-2}$;
        - in all 95 snapshots inside it: the flux of the CMB node within 0.13 % of $F_\mathrm{cmb}$ of its row; nodes 0 and 1 stable, no eddy diffusivity at any node of the stable layer, a layer of 3 to 8 nodes, 1.56 to 6.80 km thick (both depend on rtol, see `basal750_tip_r10`), and a melt fraction of 0.350 at the CMB node;
        - the smallest gradient inside is $5.9 \times 10^{-7}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ at node 0 and $4.9 \times 10^{-7}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ at node 1; the 30 snapshots with $|\partial S/\partial r[0]| < 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ are the first 30, to 26133.7 yr, and 7 have $|\partial S/\partial r[1]| < 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$; after them, 75.9 % of the time of the layer;
        - the flux bound $\rho_0 T_0 \kappa_h^\mathrm{floor}[0]\,|\partial S/\partial r[0]|$ from 49.1 to 11600 W m$^{-2}$, median 2160 W m$^{-2}$, and 33.2 W m$^{-2}$ at 14495.2 yr; the margins 59 and 6.6 (node 0), 3.5 and 0.55 (node 1), and 39.9, 4.5 and 0.96 at 14495.2 yr;
        - the layer is already there 1 snapshot earlier, at 14495.2 yr, at the end of a call whose $F_\mathrm{cmb}$ is 5.18 W m$^{-2}$, with a flux of the CMB node of $1.09 \times 10^{-3}$ W m$^{-2}$; in the snapshot before it, at 14225.1 yr, nodes 0 and 1 are unstable;
        - in every snapshot inside, the flux of the CMB node equals its conductive flux; inside the layer, gravitational separation moves 57.7 and 20.7 W m$^{-2}$ at one node in the first 2 snapshots; after them at most 0.055 W m$^{-2}$;
        - $T_\mathrm{cmb}$ stays at 4789.2 K, within 0.01 K;
        - $S_0$ rises between every 2 snapshots, by $6.4 \times 10^{-5}$ to $1.5 \times 10^{-3}$ J kg$^{-1}$ K$^{-1}$, from 5503.78 to 5503.84 J kg$^{-1}$ K$^{-1}$; $S_\mathrm{above}$ falls from 7018 to 5508.5 J kg$^{-1}$ K$^{-1}$; the step $S_\mathrm{above} - S_0$ falls between every 2 snapshots, from 1514 to 4.6 J kg$^{-1}$ K$^{-1}$, by less than $10^{-3}$ J kg$^{-1}$ K$^{-1}$ in 20 of the 94 pairs; from 22 to 42.4 kyr the layer has 5 stable nodes and the step stays between 869 and 876 J kg$^{-1}$ K$^{-1}$ (the count of 20 pairs and this range depend on rtol, see `basal750_tip_r10`);
        - the temperature step $T_\mathrm{above} - T_0$ goes from 260 K to 0.06 K; it falls in 93 of the 94 pairs of snapshots, by less than $10^{-3}$ K in 23 of them, and rises by 0.25 K in 1 (the counts and the rise depend on rtol, see `basal750_tip_r10`);
        - in the next snapshot, 601 yr later, and in every later snapshot to 70 kyr, nodes 0 and 1 are unstable and the CMB node has an eddy diffusivity.
    - Checks of `basal750_tip_r10` (CMB node rule, rtol $10^{-10}$, $t = 0$ to 70 kyr):
        - status 13, 217 calls, $|r|/B$ at most 0.9995;
        - 7 calls retried: the 2 that start at 14.23 and 44.47 kyr at rtol $2 \times 10^{-10}$ with atol unchanged, after the solver reached its step limit, and 5 from 29.79 to 32.07 kyr with half the step and atol 3 or 5 times larger (9 retries);
        - the layer is there in the snapshot at 14495.2 yr, 5 stable nodes, at the end of the call retried at 14.23 kyr;
        - $F_\mathrm{cmb}$ below 1 W m$^{-2}$ in 1 row at 9754.3 yr (back above at 9940.0 yr; the same row has 2.77 W m$^{-2}$ at rtol $10^{-8}$, and it comes before the first stored snapshot, so its node states are not stored) and from 14788.2 to 63632.1 yr, with $T_\mathrm{cmb}$ at 4789.2 K within 0.01 K;
        - in all 100 snapshots of that interval: nodes 0 and 1 stable, no eddy diffusivity at the CMB node, a layer of 3 to 5 nodes, 1.56 to 3.41 km thick, with 4 stable nodes in 91;
        - the smallest gradient is $3.5 \times 10^{-7}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ at node 0 and $4.5 \times 10^{-7}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ at node 1; the 30 snapshots with $|\partial S/\partial r[0]| < 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ are the first 30, to 26118.6 yr, and 7 have $|\partial S/\partial r[1]| < 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$; after them, 76.1 % of the time of the layer;
        - the flux bound from 29.5 to 12700 W m$^{-2}$, median 1670 W m$^{-2}$, and 13.6 W m$^{-2}$ at 14495.2 yr; the margins 35.4 and 4.0 (node 0), 310 and 50 (node 1), and 16.3, 1.8 and 70 at 14495.2 yr;
        - in the 5 snapshots that end the calls with the larger atol (29923.3 to 32299.9 yr), the gradient at node 0 is at least $1.5 \times 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$;
        - $S_0$ rises between every 2 snapshots, by 0.056 J kg$^{-1}$ K$^{-1}$ in all;
        - both steps fall between every 2 snapshots, each by less than $10^{-3}$ (J kg$^{-1}$ K$^{-1}$ or K) in 2 pairs, and the temperature step does not rise;
        - from 22.3 to 40.0 kyr the layer has 4 stable nodes and the entropy step goes from 971.9 to 964.1 J kg$^{-1}$ K$^{-1}$, and from 22 to 42.4 kyr from 972 to 890 J kg$^{-1}$ K$^{-1}$;
        - after the layer, nodes 0 and 1 are unstable and the CMB node has an eddy diffusivity in every snapshot to 70 kyr (132 snapshots from 10 kyr, the last at 70000.0 yr).
    - Checks of `basal750_tip_rerun` (CMB node rule, rtol $10^{-8}$, build c): status 13 at 70 kyr, 209 calls, $|r|/B$ at most 0.9942, no retry; the first 196 helpfile rows, to 59939.2 yr, at the same times as in `basal750_tip`, with the same $T_\mathrm{cmb}$ and $T_\mathrm{cmb,node}$, $F_\mathrm{cmb}$ within $10^{-15}$ relative, and the same end times of the 192 calls; the next row at 60542.1 yr, against 60000 yr in `basal750_tip`; low flux from 14788.5 to 62950.0 yr.
    - Checks of `basal750_main` (copy rule, rtol $10^{-8}$): low flux from 14684.1 to 44087.6 yr; in its 61 snapshots the gradient at the CMB node is negative, and node 1 is stable in exactly the 54 where the CMB node has no eddy diffusivity; in these 54 the gradient is at least $10^{-6}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ in size, in the other 7 (14684.1 to 16527.9 yr) at most $2.9 \times 10^{-9}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$.
    - Checks of `basal750_main_r10` (copy rule, rtol $10^{-10}$, $t = 0$ to 46939.4 yr; the later rows are not stored, see the provenance file):
        - 194 calls, $|r|/B$ at most 0.9995; 29 retries in 19 calls: 2 calls at rtol $2 \times 10^{-10}$, and 17 calls from 29.57 to 35.10 kyr with half the step and atol 3 or 5 times larger;
        - low flux from the row at 14682.0 yr, where the run continued from its stored state, to 44283.1 yr, with $F_\mathrm{cmb}$ from $1.19 \times 10^{-3}$ to 0.69 W m$^{-2}$ and $T_\mathrm{cmb}$ from 4930.27 to 4930.44 K;
        - in its 85 snapshots the gradient at the CMB node is negative, and node 1 is stable in exactly the 79 where the CMB node has no eddy diffusivity; in these 79 the gradient is at least $6.8 \times 10^{-6}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$ in size, in the other 6 (14682.0 to 16207.8 yr) at most $2.3 \times 10^{-9}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$;
        - no stored snapshot has a stable layer.
    - Every run: atol $10^{-8}$ and the `energy_balance` core, read from its log.
    - Tolerance:
        - every call below the bound $B$; both runs status 13 at 140 kyr;
        - CMB node rule: 424 calls, $|r|/B$ at most 0.9942; $T_\mathrm{cmb} - T_\mathrm{cmb,node}$ from $-1.78$ to 2.52 K;
        - copy rule: 423 calls, $|r|/B$ at most 0.9941; $T_\mathrm{cmb} - T_\mathrm{cmb,node}$ from $-0.38$ to $-0.17$ K inside the low-flux interval and down to $-923.5$ K at 5405 yr, before it;
        - $T_\mathrm{cmb}$ at 140 kyr: 4586.46 K (CMB node rule) and 4586.86 K (copy rule);
        - interval ends to 0.05 yr; quoted values rounded to the quoted digits; the agreement time, after which the 2 lines of $T_\mathrm{cmb}$ differ by less than 1 K, is the last time with a difference of 1 K or more, from the 2 helpfiles interpolated linearly onto a 10 yr grid.
    - Test ids: `tests/test_cmb_eddy_flux_runs.py::test_basal_mush_layer_insulates_the_core_until_the_mantle_cools_to_it`, `tests/test_cmb_eddy_flux_runs.py::test_basal_mush_layer_forms_at_rtol_1e_10`, `tests/test_cmb_eddy_flux_runs.py::test_basal_mush_layer_run_repeats_with_a_newer_build`, `tests/test_cmb_eddy_flux_runs.py::test_copy_rule_at_rtol_1e_10_leaves_the_cmb_node_without_eddy_diffusivity`, `tests/test_cmb_eddy_flux_runs.py::test_spans_keeps_an_interval_open_at_the_last_row`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py` (add `--validation-plots` to write the plot pairs)

## With a 3000 m CMB cell no layer holds the flux low

**Why this check exists.** With the 750 m cell a stable layer holds $F_\mathrm{cmb}$ low from 14.8 to 63.3 kyr. The same case with the default CMB cell shows whether such a layer also lasts on the default mesh.

**What it tests.** The case of issue #113 from $t = 0$ to 140 kyr with a CMB cell of 3000 m, the default cell of Aragog ([Mesh refinement](../Explanations/mesh_refinement.md#the-default-mesh)), and the `nearest_boundary` mixing-length profile, under both rules at the Aragog rtol $10^{-8}$. The test reads the snapshots from 10 to 70 kyr. It checks:

- under the CMB node rule, 1 interval with $F_\mathrm{cmb}$ below 1 W m$^{-2}$, from the row at 39097.7 yr to the first row back above, at 42859.0 yr (0.10 to 0.20 W m$^{-2}$, for 3.8 kyr); in its 7 snapshots the gradient at the CMB node is $-1.3 \times 10^{-9}$ to $-6.5 \times 10^{-10}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$, below its absolute tolerance of $10^{-8}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$, so the run does not resolve its sign; with this negative gradient the CMB node keeps its own floor, 7.07 m$^2$ s$^{-1}$; node 1 is unstable in the first 3 of these snapshots and stable in the last 4;
- under the copy rule, 1 interval from the row at 18148.2 yr to the first row back above, at 37620.1 yr; in all 49 snapshots the CMB node is unstable without eddy diffusivity, over a stable node 1;
- the stable layers: 1 node at 38574.7 yr and 5 nodes, 13.2 km thick, at 48540.5 yr under the CMB node rule, and 5 nodes at 48628.5 yr under the copy rule. Each is the state at the end of 1 call. The next snapshot has no stable layer. Under the 5-node layers the flux of the CMB node is $1.15 \times 10^{-3}$ W m$^{-2}$, while $F_\mathrm{cmb}$ of those calls is above 1000 W m$^{-2}$;
- node 1 stable over a CMB node with a negative gradient: in 6 snapshots under the CMB node rule (the 4 above, from 40688.1 to 42310.8 yr, with the sign not resolved, and 2 from 17870.8 to 18148.2 yr) and in 50 under the copy rule, from 17870.8 to 37105.1 yr.

**How we know the right answer.** As in the case above, the state of the nodes tells a stable layer from the fault of issue #113. The low flux of the CMB node rule from 39.1 to 42.9 kyr is neither: the gradient at the CMB node is below the tolerance of the solver, so the run does not resolve whether the CMB node is stable; with the negative gradient of the run the CMB node keeps its floor, and the eddy flux $-\rho T \kappa_h\,\partial S/\partial r$ falls with the gradient. The flux of the CMB node is then 0.10 to 0.20 W m$^{-2}$, of which conduction is $1.19 \times 10^{-3}$ W m$^{-2}$. The energy check of each call gives the pass mark.

**Why it matters.** With the default mesh, in the snapshots from 10 to 70 kyr, a stable layer shows in single snapshots only, and the call of each has $F_\mathrm{cmb}$ above 1000 W m$^{-2}$. The long low-flux interval with a stable layer occurs with the 750 m cell and not with the 3000 m cell.

![CMB heat flux and temperature of the lowest mantle cell against time, copy rule and CMB node rule, the case of issue 113 with a 3000 m CMB cell to 140 kyr](../assets/validation/cmb_eddy_flux_basal_layer_3000m.svg#only-light){ width="70%" }
![CMB heat flux and temperature of the lowest mantle cell against time, copy rule and CMB node rule, the case of issue 113 with a 3000 m CMB cell to 140 kyr](../assets/validation/cmb_eddy_flux_basal_layer_3000m_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The panels and lines are as in the first plot of the case above, for the 3000 m cell. The copy rule falls below 1 W m$^{-2}$ from 18.1 to 37.6 kyr, the CMB node rule from 39.1 to 42.9 kyr. The 2 lines of $T_\mathrm{cmb}$ differ by less than 1 K after 42.8 kyr.

??? info "Test details"
    - Reference: the copy rule; the energy check of each call; the node states of the stored snapshots from 10 to 70 kyr. The 2 runs use the builds a and b of the 750 m case; `cmb_eddy_flux_runs_provenance.json` lists them.
    - The runs of this case and of the 750 m case start at $t = 0$, and the copy rule runs to 140 kyr. The copy rule stops in the resumed runs of [An Earth-mass run where node 1 turns stable](#an-earth-mass-run-where-node-1-turns-stable), which use a uniform mesh and start from the state of an older build.
    - Tolerance:
        - every call below the bound $B$; both runs status 13 at 140 kyr;
        - CMB node rule: 538 calls, $|r|/B$ at most 0.9918; $T_\mathrm{cmb} - T_\mathrm{cmb,node}$ from $-2.35$ to 1.94 K;
        - copy rule: 538 calls, $|r|/B$ at most 0.9919; $T_\mathrm{cmb} - T_\mathrm{cmb,node}$ down to $-246.25$ K;
        - $T_\mathrm{cmb}$ at 140 kyr: 4586.03 K (CMB node rule) and 4586.08 K (copy rule);
        - interval ends to 0.05 yr; quoted values rounded to the quoted digits; the agreement time as in the case above.
    - Test id: `tests/test_cmb_eddy_flux_runs.py::test_no_basal_layer_with_a_3000_m_cmb_cell`
    - Command: `pytest tests/test_cmb_eddy_flux_runs.py` (add `--validation-plots` to write the plot pair)

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
