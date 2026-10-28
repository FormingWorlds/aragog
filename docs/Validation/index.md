# Validation

This tab lists the tests that compare Aragog with an independent result: a published benchmark, an analytical limit or a second implementation. Each case in the table carries the `reference_pinned` marker. Where a page section exists, the table links it.

- [Core](core.md): the core module against Nimmo (2015), the `leeds` and `leeds_thermal` models of thermal_history (Greenwood et al. 2021) and analytical limits.
- [Surface boundary](surface.md): the prescribed surface temperature (`outer_boundary_condition = 5`) against closed-form conduction solutions and a grid-convergence study.
- [Phase transitions](phase_transitions.md): the blend of the material properties at the solidus and the liquidus against the blend formula of SPIDER, and the NumPy and JAX right-hand sides inside the blend.
- [Numerics](numerics.md): the first-principles checks of the solver, the equation of state and the NumPy and JAX right-hand sides, including the `reference_pinned` solid-rheology solver-wiring benchmarks.

## Cases by process

| Process | Case | Test | Reference | Tolerance | Page |
|---|---|---|---|---|---|
| Core stable layer | Young layer against the ierfc solution | `tests/test_core_layer.py::test_a_young_layer_follows_the_erfc_solution_of_a_half_space` | constant-flux half space (analytical) | 1 % of the expected rise of the top cell | [Core](core.md#14-young-layer-against-the-ierfc-solution) |
| Core stable layer | Quasi-static depth of a steady layer | `tests/test_core_layer.py::test_a_steady_layer_reaches_the_quasi_static_depth` | conducted adiabatic flow equal to the CMB flow | 5 % | [Core](core.md#13-stable-layer-against-thermal_history) |
| Core stable layer | Layers under fixed and eroding CMB flows | `tests/test_core_verification_page.py::test_the_stable_layer_meets_the_thermal_history_bounds` | `leeds_thermal` (Greenwood et al. 2021) | T_cmb 5 K, T_cen 10 K, layer depth within a factor 1.5, inner core 2 % at the end, onset 5 % | [Core](core.md#13-stable-layer-against-thermal_history) |
| Core energy budget | Earth budget | `tests/test_core_verification_page.py::test_the_earth_budget_meets_nimmo_tables_4_and_5` | Nimmo (2015), Tables 4 and 5 | 2 %, four documented exceptions | [Core](core.md#5-nimmo-2015-earth-core-budget) |
| Core energy budget | Printed values of the two Nimmo core models | `tests/test_core_nimmo_benchmarks.py` (the `reference_pinned` tests except the cross-check) | Nimmo (2015), ch. 9.08 Table 2, Figure 3 caption (p. 211) and p. 212; ch. 8.02 Table 2 | 5e-4 to 3 % per printed value, Cr 5 % and 15 %, the baseline headline 5 % to 45 % | [Core](core.md#5-nimmo-2015-earth-core-budget) |
| Core energy budget | Model-2 budget terms at 4180 K | `tests/test_core_nimmo_benchmarks.py::test_budget_terms_match_thermal_history_cross_check` | `leeds` (Greenwood et al. 2021); Nimmo (2015), ch. 8.02 Tables 2 and 4 | 1e-5 to 2 % per term | [Core](core.md#6-budget-terms-against-thermal_history) |
| Core energy budget | Budget terms | `tests/test_core_verification_page.py::test_budget_terms_match_the_thermal_history_table` | `leeds` (Greenwood et al. 2021) | 1e-3 relative | [Core](core.md#6-budget-terms-against-thermal_history) |
| Core energy budget | Entropy terms | `tests/test_core_entropy.py::test_entropy_capacities_match_thermal_history` | `leeds` (Greenwood et al. 2021) | 1e-6 to 1e-4 relative | [Core](core.md#6-budget-terms-against-thermal_history) |
| Core dynamo | Efficiency factors and dynamo threshold | `tests/test_core_entropy.py::test_chr09_efficiency_factors_reproduce_printed_values`, `::test_dynamo_threshold_and_margin` | Christensen et al. (2009); Nimmo (2015) | 1e-3 relative; threshold below 6.5 TW | [Core](core.md#7-dynamo-criterion-and-field-strength) |
| Core thermal history | Core history | `tests/test_core_verification_page.py::test_core_history_matches_the_thermal_history_table` | `leeds` (Greenwood et al. 2021) | T_cmb 0.2 K, T_cen 1 K, ICB radius 1 %, the same onset | [Core](core.md#10-a-core-thermal-history-against-thermal_history) |
| Core numerics | NumPy and JAX right-hand sides | `tests/test_jax_dsdt_core_module.py::test_rhs_parity_with_numpy_on_driven_state` | NumPy implementation | 1e-8 relative | [Core](core.md#12-numpy-and-jax-right-hand-sides-and-the-analytic-jacobian) |
| Surface boundary | Steady heated shell | `tests/test_entropy_bc5_validation.py::test_the_steady_heated_shell_matches_the_closed_form` | Turcotte and Schubert (2014), eq. 4.40 | surface flux 1e-8 relative; temperature order 2.0 ± 0.2; N² times the top-cell error 0.159 of the rise ± 10 % (a pin of the scheme, measured) | [Surface boundary](surface.md#a-heated-shell-in-steady-state) |
| Surface boundary | Cooling below a fixed surface temperature | `tests/test_entropy_bc5_validation.py::test_cooling_below_a_fixed_surface_temperature_follows_the_half_space_flux` | Turcotte and Schubert (2014), eq. 4.116 | 1 % at every output time; signed error within 5e-4 of the grid and curvature estimates | [Surface boundary](surface.md#cooling-below-a-fixed-surface-temperature) |
| Surface boundary | Grid convergence with convection | `tests/test_entropy_bc5_validation.py::test_the_convecting_shell_converges_with_second_order` | convergence order and Richardson extrapolation | order 2.0 ± 0.4; 1e-3 of the rise from the extrapolation | [Surface boundary](surface.md#grid-convergence-with-convection) |
| Phase transitions | Properties on the solidus and the liquidus | `tests/test_matprop_smooth_width_validation.py::test_on_each_boundary_every_property_is_the_mean_of_the_two_branches` | SPIDER `get_smoothing` (`util.c`): weight 1/2 on each boundary | 1e-12 relative | [Phase transitions](phase_transitions.md#on-each-boundary-a-property-is-the-mean-of-its-two-values) |
| Mantle heat transport | Theta-dependent top slope of the mixing length | `tests/test_mlt_calibration.py::test_theta_dependent_top_slope_hand_values_and_limits` | hand values and limits | 1e-15 to 1e-13 relative | no page section yet |
| Mantle rheology | Arrhenius viscosity, yield stress, NumPy and JAX kernels | `tests/test_rheology_verification.py` (its `reference_pinned` tests) | reference limits, NumPy implementation | 1e-12 relative (one bound 1e-9) | no page section yet ([Solid-state convection](../Explanations/solid_state_convection.md) explains the rheology) |
| Mantle numerics | NumPy and JAX right-hand sides with solid rheology | `tests/test_rheology_solver_wiring_smoke.py::test_rhs_numpy_jax_parity_rheology_on` | NumPy implementation on active-solid `abe_solid.toml` variant | 1e-6 relative (on vs off >= 1e-4) | [Numerics](numerics.md#numpy-jax-rhs-parity) |
| Mantle numerics | Round-off perturbation spread with solid rheology | `tests/test_rheology_solver_wiring.py::test_perturbed_starts_spread` | 5 perturbed starts $S_0(1 + k \cdot 10^{-12})$ over 100 yr | $\Delta T_\text{on}(r) \le \max(2\Delta T_\text{off}(r), 0.05\text{ K})$ | [Numerics](numerics.md#round-off-perturbation-spread-with-solid-rheology) |
| Mantle numerics | Two-pass solid rheology RHS call counts and cost | `tests/test_rheology_solver_wiring.py::test_rhs_call_counts_and_cost` | per-RHS call-count budget and off-path wall time | $(0, +2, +2, +1, +1, +2)$ calls; JAX ratio $\le 1.25$ | [Numerics](numerics.md#two-pass-solid-rheology-rhs-cost) |

The command of a case is `pytest <test>`; the young-layer case, the surface-boundary cases and the phase-transition case have sections with their own reference, tolerance, test id and command, and the other cases are described by the prose of their linked section.

`pytest -m reference_pinned` runs every case. The core-page tests need the SPIDER P-S tables (`ARAGOG_TEST_EOS_DIR`, [Build tests](../How-to/build_tests.md)); the stable-layer and Earth-budget cases take minutes.

## Plots written by the tests

A case with a plot pair shows the tested values against their reference in a light and a dark version. The test writes the pair with the same numbers it asserts, through the `validation_plot` fixture of proteus-mpl. This command writes all pairs to `docs/assets/validation/`:

```console
pytest -m reference_pinned --validation-plots
```

Without `--validation-plots` the tests assert and write nothing. The figures in `docs/figures/vv/` are made by the scripts in `tools/verification/`, not by the tests.
