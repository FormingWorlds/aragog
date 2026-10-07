# Core module verification

This page checks the `core_module` core boundary condition ([core_bc.md](core_bc.md)) and the `aragog.core` model behind it against analytic results, published values, an independent core evolution code and coupled runs. Each section states the physics a check tests, the reference and where it comes from, the value aragog gives, a figure, and the tests that pin it. A check shows that the code solves the stated equations to the stated accuracy; whether the equations describe a real core is a separate question, which the literature comparisons in sections 5 to 8 and the limits stated with each check address.

Every number on this page and every figure is produced by `tools/verification/run_core_verification.py`, which writes the figures to `docs/figures/vv/` and the numbers to `docs/figures/vv/core_verification_values.json`; `tests/test_core_verification_page.py` checks each number on the page against that file. The script needs the SPIDER-format EOS tables (`ARAGOG_TEST_EOS_DIR`) for sections 9 and 12 and the optional `verification` dependencies for the figure style, and runs in about seven minutes on one core.

## 1. Gaussian core structure

The core density is a Gaussian in radius, $\rho(r) = \rho_\mathrm{cen} \exp(-r^2/L^2)$, and the adiabat is $T_a(r) = T_\mathrm{cmb} \exp((r_\mathrm{cmb}^2 - r^2)/D^2)$ with $D^2 = 3 c_p / (2\pi \alpha \rho_\mathrm{cen} G)$ (Labrosse et al. 2001, eqs. 7, 9 and 11). Mass and gravity are closed-form integrals of the density; the pressure is the hydrostatic integral of $\rho g$ from the CMB inward, by 32-point Gauss-Legendre quadrature (`pressure_mode = "quadrature"`), or the third-order expansion in $r/L$ that Labrosse et al. (2001, eq. 8) use for gravity (`pressure_mode = "labrosse"`).

On the Earth-like test core ($\rho_\mathrm{cen} = 12500$ kg m$^{-3}$, $L = 7200$ km, $r_\mathrm{cmb} = 3480$ km), the closed-form enclosed mass and gravity agree with an independent adaptive quadrature of the density to 3.6e-13<!--k:1.mass_rel_err-->, and gravity near the centre follows the linear limit $4\pi G \rho_\mathrm{cen} r/3$ to 2.2e-16<!--k:1.gravity_centre_slope_rel_err-->. The pressure satisfies $dP/dr = -\rho g$ to 2.5e-08<!--k:1.hydrostatic_max_rel_err-->, the error of the 50 m central difference used to test it. The core has a mass of 1.92e+24<!--k:1.m_core--> kg, a CMB density of 9896<!--k:1.rho_cmb--> kg m$^{-3}$ and a central pressure of 356.8<!--k:1.p_cen_GPa--> GPa. The third-order pressure differs from the quadrature by up to 3.0e-03<!--k:1.pressure_lab_vs_quad_max_rel-->, the truncation of the expansion; both meet the CMB anchor exactly.

![Gaussian core structure](../figures/vv/fig_08_core_structure.png)

**Figure 8.** Gaussian core structure of the Earth-like test core. (a) Density, gravity and pressure against radius. (b) Relative residual of hydrostatic balance $|dP/dr + \rho g|/\rho g$ for the quadrature pressure, and the relative difference between the third-order and the quadrature pressure; both panels share the radius axis.

Pinned by `tests/test_core_profiles.py::test_density_positive_monotone_and_mass_matches_quadrature`, `::test_gravity_centre_limit_and_shell_integral` and `::test_pressure_hydrostatic_balance_and_anchor`.

## 2. Energy identities of the core budget

The core cools at $dT_\mathrm{cmb}/dt = -(Q_\mathrm{cmb} - Q_\mathrm{radio}) / \tilde C(T_\mathrm{cmb})$, where the effective capacity $\tilde C = C_s + C_L + C_g$ adds the secular term, the latent heat of inner-core growth and the gravitational energy of the light elements it rejects. The heat content $H(T)$ is defined so that $H(T_2) - H(T_1) = \int_{T_1}^{T_2} \tilde C\, dT$; the solver books the core heat change from it, so this identity is what ties the core ledger to the CMB heat. Inner-core growth starts at the onset temperature $T_\mathrm{on}$, where the adiabat first touches the melting curve at the centre, and ends at freeze-out $T_\mathrm{fr}$, where the whole core has crystallised and the latent and gravitational terms vanish.

For the iron-alloy curve on the test core, onset is at 4054.2<!--k:2.t_onset--> K and freeze-out at 3688.9<!--k:2.t_freeze--> K. The heat-content difference matches the integral of $\tilde C$ to 4.0e-07<!--k:2.content_vs_integral_max_rel--> of the total over 150 K on either side of the growth band; the residual above the onset is the trapezoid error of the 40001-point reference at the square-root cusp of the latent term, not an error of $H$. At freeze-out the capacity drops by a factor of 4.460<!--k:2.quadratic_freeze_out_ratio--> for the Nimmo (2015) quadratic curve and by 55.2 %<!--k:2.iron_freeze_out_drop:-100--> for the iron-alloy curve, without the gravitational term.

![Energy identities](../figures/vv/fig_09_core_energy_identities.png)

**Figure 9.** Effective heat capacity of the core across inner-core growth. (a) Secular, latent and gravitational capacities and their sum against CMB temperature, with the onset and freeze-out marked. (b) Relative difference between the heat-content difference and the integral of the total capacity at 120 temperatures, normalised by the total.

Pinned by `tests/test_core_budget.py::test_heat_content_difference_is_the_capacity_integral`, `::test_latent_and_gravitational_energy_conservation` (latent and gravitational capacities integrate to the geometric latent heat and gravitational energy to below 1e-6) and `::test_freeze_out_capacity_jump`.

## 3. CMB boundary-layer flux

The CMB heat flux is $q = k\, \Delta T / \min(\Delta r_{1/2}, \delta)$ for a core hotter than the mantle side of the CMB, where $\Delta T = T_\mathrm{core} - T_m$, $\Delta r_{1/2}$ is the distance to the bottom mantle cell and $\delta = (\kappa \eta\, Ra_\mathrm{crit} / (\rho g \alpha \Delta T))^{1/3}$ is the boundary-layer thickness at the critical Rayleigh number (Foley & Driscoll 2016, eq. 20 and the text below it; Thiriet et al. 2019, eqs. 13 and 14). On the convective branch $q \propto \Delta T^{4/3}$; a core colder than the mantle gains heat by conduction across the half cell only, so the flux always has the sign of $\Delta T$. The default $Ra_\mathrm{crit} = 450$ is the critical value Thiriet et al. (2019, Table 2) use for the upper boundary layer of their mantle; for the lower boundary layer they scale it with the internal Rayleigh number (their eq. 16), so here it is a parameter, `ra_crit_cmb`.

For bottom-cell properties of a liquid base ($\eta = 0.1$ Pa s), the convective branch has the slope 1.333<!--k:3.convective_slope--> in log-log and $Ra_\mathrm{crit} = 1800$ lowers the flux by the factor 0.630<!--k:3.ra_crit_ratio_1800_450--> $= (450/1800)^{1/3}$. At a 100 K contrast the flux is 5.55e+04<!--k:3.q_liquid_100K--> W m$^{-2}$ over the liquid base and 0.04<!--k:3.q_solid_100K--> W m$^{-2}$ over a solid base ($\eta = 10^{21}$ Pa s), where conduction across the half cell sets it; a colder core conducts at the same rate over either base, to 4.2e-15<!--k:3.cold_core_flux_equals_solid_base-->.

![CMB boundary-layer flux](../figures/vv/fig_10_cmb_boundary_layer_flux.png)

**Figure 10.** Magnitude of the CMB heat flux against the core-mantle temperature contrast, for a liquid and a solid bottom cell, at $Ra_\mathrm{crit} = 450$ (solid) and 1800 (dashed); the dotted line is a core colder than the mantle, which conducts across the half cell over either base.

Pinned by `tests/test_core_boundary_layer.py` (sign, hand-computed convective value, conduction branch, $Ra_\mathrm{crit}$ scaling, continuity and slopes through zero contrast).

## 4. Inner-core nucleation

The inner-core radius $r_\mathrm{icb}(T_\mathrm{cmb})$ is the radius where the adiabat crosses the melting curve. The adiabat and the melting curve both have zero radial slope at the centre, so just below the onset their difference is quadratic in $r$ and $r_\mathrm{icb} \propto (T_\mathrm{on} - T_\mathrm{cmb})^{1/2}$: the inner core appears at a finite rate of radius per kelvin squared. The budget's latent and gravitational capacities need $dr_\mathrm{icb}/dT_\mathrm{cmb}$, which the code supplies by the implicit-function theorem as a custom derivative rule.

Fitted over the first kelvin of undercooling, the log-log slope is 0.4993<!--k:4.sqrt_slope-->. The implicit-function derivative agrees with a central difference with a step of $10^{-3}$ of the undercooling to 1.4e-06<!--k:4.jvp_vs_fd_max_rel--> from $10^{-3}$ to 30 K below the onset.

![Inner-core nucleation](../figures/vv/fig_11_inner_core_nucleation.png)

**Figure 11.** Inner-core growth below the onset. (a) Inner-core radius against undercooling, with the square-root law through the first point. (b) Relative difference between the implicit-function derivative and a central difference.

Pinned by `tests/test_core_verification_page.py::test_inner_core_radius_grows_as_the_square_root_of_undercooling` and `tests/test_jax_dsdt_core_module.py::test_jacobian_core_column_matches_central_differences`.

## 5. Nimmo (2015) Earth core budget

This section is pending the comparison against Nimmo (2015).

## 6. Budget terms against thermal_history

The `leeds` core model of thermal_history (Greenwood et al. 2021) computes the same budget with polynomial profiles, a trapezoid quadrature on a radial grid and its own inner-core growth. The comparison runs it on the Nimmo (2015, Table 2) core, with the Gaussian density and adiabat given to it as eighth-order Taylor polynomials, the same quadratic melting curve without a light-element depression, a fixed latent heat of 750 kJ kg$^{-1}$, complete rejection of the light element at the inner-core boundary, and the CODATA gravitational constant aragog uses. A core-only history starts at a CMB temperature of 4400 K under a fixed CMB heat flow of 10 TW; aragog's capacities are evaluated on each thermal_history state. Inputs, versions and licence are in `tools/verification/data/`.

The secular term agrees to 1.4e-08<!--k:6.secular_max_rel--> and the conduction entropy sink to 1.5e-07<!--k:6.conduction_sink_rel-->; the latent term agrees to 1.1e-04<!--k:6.latent_max_rel-->. The gravitational term differs by up to 2.8 %<!--k:6.gravitational_max_rel:100-->, because thermal_history enriches the outer core in the light element as the inner core grows (by the factor 1.029<!--k:6.enrichment_end--> at the end), while aragog holds the composition fixed; with that factor applied the two agree to 1.0e-04<!--k:6.gravitational_enrichment_corrected_max_rel-->.

The stratified layer is where the two models differ in kind. aragog places the layer base where the conducted adiabatic heat flow equals the CMB heat flow and treats the layer as quasi-static; thermal_history grows the layer by thermal diffusion from a moving interface (`leeds_thermal`). Under fixed CMB heat flows below the adiabatic one (15.8<!--k:6.q_k_TW--> TW at the start), the thermal_history layer is 443<!--k:6.layer_leeds_km_8TW_50myr-->, 995<!--k:6.layer_leeds_km_8TW_200myr--> and 1601<!--k:6.layer_leeds_km_8TW_500myr--> km thick at 50, 200 and 500 Myr for 8 TW, while the quasi-static depth on the same states is 836<!--k:6.layer_aragog_km_8TW_50myr-->, 844<!--k:6.layer_aragog_km_8TW_200myr--> and 847<!--k:6.layer_aragog_km_8TW_500myr--> km; for 12 TW the values are 384<!--k:6.layer_leeds_km_12TW_50myr-->, 902<!--k:6.layer_leeds_km_12TW_200myr--> and 1448<!--k:6.layer_leeds_km_12TW_500myr--> km against 375<!--k:6.layer_aragog_km_12TW_50myr-->, 373<!--k:6.layer_aragog_km_12TW_200myr--> and 362<!--k:6.layer_aragog_km_12TW_500myr--> km. The quasi-static depth appears at once, while a diffusive layer grows over the diffusion time of its thickness, about 2 Gyr for 1000 km, and holds heat that the quasi-static layer does not.

![Budget terms against thermal_history](../figures/vv/fig_13_leeds_budget_terms.png)

**Figure 13.** Comparison with the `leeds` model of thermal_history on the same inputs. (a) Relative difference of the secular, latent and gravitational capacities along the 10 TW history; the dotted line applies thermal_history's outer-core enrichment to aragog's gravitational term. (b) Stratified-layer thickness under fixed CMB heat flows of 8 and 12 TW: thermal_history's diffusive layer (dashed) and aragog's quasi-static depth on the same states (solid).

Pinned by `tests/test_core_verification_page.py::test_budget_terms_match_the_thermal_history_table`, and on the Nimmo (2015) state by `tests/test_core_nimmo_benchmarks.py::test_budget_terms_match_thermal_history_cross_check` and `tests/test_core_entropy.py::test_entropy_capacities_match_thermal_history`.

## 7. Dynamo criterion and field strength

The entropy budget gives the entropy production available to the dynamo, $\Delta E = E_s + E_L + E_g + E_R - E_k$, positive when the core can sustain a dynamo. The field strength follows the energy-flux scaling of Christensen et al. (2009, eq. 2), $\langle B \rangle^2/(2\mu_0) = c f_\mathrm{ohm} \langle \rho \rangle^{1/3} (F q_o)^{2/3}$ with $c = 0.63$ and the efficiency factor $F = 0.88\, \alpha g_\mathrm{cmb} r_\mathrm{cmb} / c_p$ for a constant convected flux or $0.45\, \alpha g_\mathrm{cmb} r_\mathrm{cmb} / c_p$ for a flux that vanishes at the outer boundary (both p. 168); $q_o$ here is the superadiabatic part of the CMB heat flow over the CMB area.

With Christensen et al.'s Earth inputs ($\alpha = 1.35\times10^{-5}$ K$^{-1}$, $g = 10.7$ m s$^{-2}$, $R = 3480$ km, $c_p = 840$ J kg$^{-1}$ K$^{-1}$) the two factors are 0.527<!--k:7.F_const_flux_printed_inputs--> and 0.269<!--k:7.F_zero_outer_printed_inputs-->, their printed 0.52 and 0.27; on the Nimmo (2015) core, whose CMB gravity is lower, the constant-flux factor is 0.484<!--k:7.F_const_flux_profile-->. On that core at $T_\mathrm{cmb} = 4180$ K the adiabatic heat flow is 14.97<!--k:7.Q_k_TW--> TW and the entropy margin vanishes at 5.30<!--k:7.dynamo_threshold_TW--> TW, against Nimmo's (2015, p. 46) minimum of about 5 TW for a present-day dynamo without radiogenic heat. At 17 TW the core field is 1.10<!--k:7.b_rms_17TW_mT--> mT, against 1.82<!--k:7.earth_internal_field_mT--> mT for Earth from Christensen et al.'s dipole field of 0.26 mT at the dynamo surface and their ratio of about 7 between the total and the dipole field (p. 168). The scaling counts the thermal buoyancy flux only, so the field estimate is a lower bound where compositional convection drives the dynamo.

![Dynamo scaling](../figures/vv/fig_14_dynamo_scaling.png)

**Figure 14.** Dynamo criterion and field strength on the Nimmo (2015) core at $T_\mathrm{cmb} = 4180$ K. (a) Entropy margin against CMB heat flow, with the threshold where it vanishes. (b) Volume-averaged core field from Christensen et al. (2009, eq. 2), zero up to the adiabatic heat flow $Q_k$, with Earth's estimated internal field.

Pinned by `tests/test_core_entropy.py::test_chr09_efficiency_factors_reproduce_printed_values`, `::test_field_scaling_bounds_and_earth_magnitude` and `::test_dynamo_threshold_and_margin`.

## 8. Iron melting curve

The pure-iron melting curve is the two-branch Simon-Glatzel fit of Anzellini et al. (2013, eqs. 2 and 3, p. 466), $(P - P_0)/27.39 = (T_m/T_0)^{2.38} - 1$ on the $\gamma$-Fe branch with $T_0 = 1991$ K and $P_0 = 5.2$ GPa, and $(P - P_\mathrm{TP})/161.2 = (T_m/T_\mathrm{TP})^{1.72} - 1$ on the $\epsilon$-Fe branch from the $\gamma$-$\epsilon$-liquid triple point at 98.5 GPa and 3712 K; this is the PALEOS prescription. The two branches meet with a jump of 0.725<!--k:8.branch_jump_K--> K at the triple point, which a $C^2$ smootherstep over a 3 GPa half-width removes; the blended curve departs from the branches by at most 0.361<!--k:8.blend_max_dev_K--> K and equals them exactly outside the blend. Light elements depress the curve by the factor $1 - \mathrm{depression} \times x$, 0.88<!--k:8.depression_factor--> for $x = 0.1$ and a depression of 1.2; this depression is a model choice, not part of the Anzellini fit.

The pure-iron curve gives 4192.0<!--k:8.t_melt_136GPa--> K at 136 GPa and 6229.2<!--k:8.t_melt_330GPa--> K at 330 GPa.

![Iron melting curve](../figures/vv/fig_15_iron_melting_curve.png)

**Figure 15.** Iron melting curve. (a) Pure-iron and alloy melting temperature against pressure, with the triple point marked. (b) Blended curve minus the unblended Anzellini et al. (2013) branches around the triple point; the jump is that of the unblended fit.

Pinned by `tests/test_core_melting.py::test_pure_iron_pins_against_the_paleos_source`, `::test_branch_switch_matches_unblended_outside_transition` and `::test_branch_switch_deviation_and_c1_continuity`.

## 9. A CVODE solve across the inner-core onset

The coupled right-hand side carries the core temperature as a state, so CVODE integrates the core through the onset, where the latent capacity rises as a square root. A core 2 K above the onset over a liquid mantle base cools from 6271.3<!--k:9.t_core_start--> K to 6268.3<!--k:9.t_core_end--> K in 4 yr, through the onset at 6269.3<!--k:9.t_onset--> K, and grows an inner core of 148<!--k:9.r_icb_end_km--> km. The solver's core heat change matches the heat-content difference to 1.7e-13<!--k:9.core_vs_content_rel--> and the CMB heat it books to 2.6e-05<!--k:9.core_vs_cmb_rel-->, the solve's tolerance; the CMB heat reconstructed from the 71<!--k:9.n_outputs--> output times by the trapezoid rule matches to 7.5e-04<!--k:9.reconstructed_cmb_max_rel-->, the error of that sampling.

![CVODE solve across the onset](../figures/vv/fig_16_cvode_onset_ledger.png)

**Figure 16.** A CVODE solve through the inner-core onset. (a) Core temperature against time, with the onset temperature. (b) Core heat lost from the heat-content difference and from the time integral of the CMB heat flow. (c) Relative difference between the two curves in (b).

Pinned by `tests/test_entropy_solver_core_module_smoke.py::test_core_module_cvode_solve_crosses_the_inner_core_onset`.

## 10. A core thermal history against thermal_history

The core-only history of section 6 tests the integrated evolution rather than the terms: aragog integrates its cooling rate under the same fixed 10 TW CMB heat flow by an implicit solver with a relative tolerance of $10^{-10}$, and thermal_history steps it with its own fixed 1 Myr steps. Over 1499<!--k:10.t_end_myr--> Myr the CMB temperatures differ by at most 0.057<!--k:10.t_cmb_max_abs_diff--> K. The inner core nucleates at 1034<!--k:10.onset_myr_aragog--> Myr in aragog and at 1034<!--k:10.onset_myr_leeds--> Myr in thermal_history, on its 1 Myr grid, and reaches 1017.8<!--k:10.r_icb_end_km_aragog--> km against 1017.0<!--k:10.r_icb_end_km_leeds--> km, an inner-core age of 465<!--k:10.inner_core_age_myr--> Myr in both. The largest difference sits at the onset, where thermal_history's explicit step crosses the start of the latent release in one step; afterwards the outer-core enrichment of section 6 shifts the gravitational term.

![Core thermal history against thermal_history](../figures/vv/fig_17_leeds_thermal_history.png)

**Figure 17.** Core-only thermal history under a fixed 10 TW CMB heat flow. (a) CMB temperature from aragog and thermal_history. (b) Inner-core radius. (c) Absolute difference of the CMB temperatures.

Pinned by `tests/test_core_verification_page.py::test_core_history_matches_the_thermal_history_table`.

## 11. Coupled PROTEUS case

This section is pending the coupled runs.

## 12. NumPy and JAX right-hand sides and the analytic Jacobian

The solver evaluates the coupled right-hand side in NumPy and, for CVODE's analytic Jacobian, in JAX; the two must agree, and the JAX Jacobian must equal the derivative of that right-hand side. On a 10-node mesh with the core in three states (inside the nucleation band, 100 K above the onset, and stratified with the core 50 K above the mantle), the two right-hand sides agree to 1.4e-12<!--k:12.rhs_max_rel_nucleating--> in every component. The analytic $\partial \dot T_\mathrm{core}/\partial T_\mathrm{core}$ matches a central difference to 3.0e-11<!--k:12.jac_tcore_min_rel_nucleating--> in the nucleation band and 3.2e-11<!--k:12.jac_tcore_min_rel_stratified--> with the layer, at the step that balances truncation and rounding; above the onset the core is colder than the mantle base, the flux is on its linear conduction branch, and the difference stays at rounding for every step.

![NumPy and JAX parity](../figures/vv/fig_19_numpy_jax_parity.png)

**Figure 19.** Agreement of the NumPy and JAX core right-hand sides. (a) Relative difference per state component, the entropy cells first and the CMB entropy gradient and core temperature last, in three core states. (b) Relative error of a central difference of $\dot T_\mathrm{core}$ against the analytic Jacobian entry, against the difference step.

Pinned by `tests/test_jax_dsdt_core_module.py::test_boundary_slots_match_numpy_on_a_five_node_mesh`, `::test_jacobian_core_column_matches_central_differences` and `::test_jacobian_carries_boundary_couplings`.

## References

- Anzellini, S., Dewaele, A., Mezouar, M., Loubeyre, P., & Morard, G. (2013). Melting of iron at Earth's inner core boundary based on fast X-ray diffraction. *Science*, 340(6131), 464-466. https://doi.org/10.1126/science.1233514
- Christensen, U. R., Holzwarth, V., & Reiners, A. (2009). Energy flux determines magnetic field strength of planets and stars. *Nature*, 457(7226), 167-169. https://doi.org/10.1038/nature07626
- Foley, B. J., & Driscoll, P. E. (2016). Whole planet coupling between climate, mantle, and core: Implications for rocky planet evolution. *Geochemistry, Geophysics, Geosystems*, 17(5), 1885-1914. https://doi.org/10.1002/2015GC006210
- Greenwood, S., Davies, C. J., & Mound, J. E. (2021). On the evolution of thermally stratified layers at the top of Earth's core. *Physics of the Earth and Planetary Interiors*, 318, 106763. https://doi.org/10.1016/j.pepi.2021.106763
- Labrosse, S., Poirier, J.-P., & Le Mouël, J.-L. (2001). The age of the inner core. *Earth and Planetary Science Letters*, 190(3-4), 111-123. https://doi.org/10.1016/S0012-821X(01)00387-9
- Nimmo, F. (2015). Energetics of the Core. In *Treatise on Geophysics* (2nd ed., Vol. 8, pp. 27-55). Elsevier. https://doi.org/10.1016/B978-0-444-53802-4.00139-1
- Thiriet, M., Breuer, D., Michaut, C., & Plesa, A.-C. (2019). Scaling laws of convection for cooling planets in a stagnant lid regime. *Physics of the Earth and Planetary Interiors*, 286, 138-153. https://doi.org/10.1016/j.pepi.2018.11.003
