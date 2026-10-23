# Validation

This tab lists the tests that compare Aragog with an independent result: a published benchmark, an analytical limit or a second implementation. Each case carries the `reference_pinned` marker. Where a page section exists, the table links it; a case with a plot pair has a subsection with its reference, tolerance, test id and command.

- [Core](core.md): the core module against Nimmo (2015), the `leeds` and `leeds_thermal` models of thermal_history (Greenwood et al. 2021) and analytical limits.
- [Numerics](numerics.md): the first-principles checks of the solver, the equation of state and the NumPy and JAX right-hand sides.

## Cases by process

| Process | Case | Test | Reference | Page |
|---|---|---|---|---|
| Core stable layer | Young layer against the ierfc solution | `tests/test_core_layer.py::test_a_young_layer_follows_the_erfc_solution_of_a_half_space` | constant-flux half space (analytical) | [Core](core.md#young-layer-against-the-ierfc-solution) |
| Core stable layer | Quasi-static depth of a steady layer | `tests/test_core_layer.py::test_a_steady_layer_reaches_the_quasi_static_depth` | conducted adiabatic flow equal to the CMB flow | [Core](core.md#13-stable-layer-against-thermal_history) |
| Core stable layer | Layers under fixed and eroding CMB flows | `tests/test_core_verification_page.py::test_the_stable_layer_meets_the_thermal_history_bounds` | `leeds_thermal` (Greenwood et al. 2021) | [Core](core.md#13-stable-layer-against-thermal_history) |
| Core energy budget | Earth budget | `tests/test_core_verification_page.py::test_the_earth_budget_meets_nimmo_tables_4_and_5` | Nimmo (2015), Tables 4 and 5 | [Core](core.md#5-nimmo-2015-earth-core-budget) |
| Core energy budget | Printed values of the two Nimmo core models | `tests/test_core_nimmo_benchmarks.py` (the `reference_pinned` tests except the cross-check) | Nimmo (2015), ch. 9.08 Table 2, Figure 3 caption (p. 211) and p. 212; ch. 8.02 Table 2 | [Core](core.md#5-nimmo-2015-earth-core-budget) |
| Core energy budget | Model-2 budget terms at 4180 K | `tests/test_core_nimmo_benchmarks.py::test_budget_terms_match_thermal_history_cross_check` | `leeds` (Greenwood et al. 2021); Nimmo (2015), ch. 8.02 Tables 2 and 4 | [Core](core.md#6-budget-terms-against-thermal_history) |
| Core energy budget | Budget terms | `tests/test_core_verification_page.py::test_budget_terms_match_the_thermal_history_table` | `leeds` (Greenwood et al. 2021) | [Core](core.md#6-budget-terms-against-thermal_history) |
| Core energy budget | Entropy terms | `tests/test_core_entropy.py::test_entropy_capacities_match_thermal_history` | `leeds` (Greenwood et al. 2021) | [Core](core.md#6-budget-terms-against-thermal_history) |
| Core dynamo | Efficiency factors and dynamo threshold | `tests/test_core_entropy.py::test_chr09_efficiency_factors_reproduce_printed_values`, `::test_dynamo_threshold_and_margin` | Christensen et al. (2009); Nimmo (2015) | [Core](core.md#7-dynamo-criterion-and-field-strength) |
| Core thermal history | Core history | `tests/test_core_verification_page.py::test_core_history_matches_the_thermal_history_table` | `leeds` (Greenwood et al. 2021) | [Core](core.md#10-a-core-thermal-history-against-thermal_history) |
| Core numerics | NumPy and JAX right-hand sides | `tests/test_jax_dsdt_core_module.py::test_rhs_parity_with_numpy_on_driven_state` | NumPy implementation | [Core](core.md#12-numpy-and-jax-right-hand-sides-and-the-analytic-jacobian) |
| Mantle heat transport | Theta-dependent top slope of the mixing length | `tests/test_mlt_calibration.py::test_theta_dependent_top_slope_hand_values_and_limits` | hand values and limits | no page section yet |
| Mantle rheology | Arrhenius viscosity, yield stress, NumPy and JAX kernels | `tests/test_rheology_verification.py` (its `reference_pinned` tests) | reference limits, NumPy implementation | no page section yet ([Solid-state convection](../Explanations/solid_state_convection.md) explains the rheology) |

`pytest -m reference_pinned` runs every case. The core-page tests need the SPIDER P-S tables (`ARAGOG_TEST_EOS_DIR`, [Build tests](../How-to/build_tests.md)); the stable-layer and Earth-budget cases take minutes.

## Plots written by the tests

A case with a plot pair shows the tested values against their reference in a light and a dark version. The test writes the pair with the same numbers it asserts, through the `validation_plot` fixture of proteus-mpl. This command writes all pairs to `docs/assets/validation/`:

```console
pytest -m reference_pinned --validation-plots
```

Without `--validation-plots` the tests assert and write nothing. The figures in `docs/figures/vv/` are made by the scripts in `tools/verification/`, not by the tests.
