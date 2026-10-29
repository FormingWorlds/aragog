# Surface lid

## What this page checks

This page checks the sub-grid surface lid, `subgrid_lid = true`: a conductive crust thinner than the top cell of the mesh, which Aragog carries as two extra states, the crust thickness and its energy per area. Validation here means that we run the lid on a case where the right answer is known from a textbook solution, and check that the lid gives that answer. If the lid were wrong, a run with the lid would grow its crust at the wrong rate, and the heat that leaves the planet through the crust would be wrong. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, then shows a plot and the test details.

The three cases drive the lid functions of `aragog.surface_lid` through time. These are the functions the solver calls in every right-hand side. The material properties are constant: thermal conductivity $k$ = 4 W/m/K, density $\rho$ = 3000 kg/m³ and heat capacity $c_p$ = 1200 J/kg/K, so the thermal diffusivity is $\kappa = k / (\rho c_p)$ = 1.11 × 10⁻⁶ m²/s. The base of the lid is at $T_b$ = 1400 K. These cases check the lid on its own; the coupling of the lid to the mesh is checked by the solver tests listed at the end.

## A crust growing over melt

**Why this check exists.** The temperature profile across the lid sets how much heat the lid conducts away from its base, and so how fast new crust forms. The simplest choice, a straight-line profile, grows the crust 1.6, 14 and 37 % too fast at the three Stefan numbers below, and the test sees it.

**What it tests.** A lid lies over melt at its melting temperature $T_b$. The melt delivers no heat to the base of the lid, and the surface is held at $T_s$ = 300 K. The Stefan number $\mathrm{St} = c_p (T_b - T_s) / L$ compares the heat to cool the crust with $L$, the latent heat released when 1 kg of melt freezes; the test uses St = 0.1, 1 and 3.381. The lid starts at 10⁶ s on the textbook profile, and the test checks its thickness $\delta$ at 10⁸, 10¹⁰ and 10¹² s. A second lid starts at 1 m with a straight-line profile and must approach the same law.

**How we know the right answer.** This is the Stefan problem of a solidifying lava layer: the crust thickness is $\delta = 2 \lambda \sqrt{\kappa t}$, with $\lambda$ the root of $\lambda\, e^{\lambda^2} \operatorname{erf}\lambda = \mathrm{St} / \sqrt{\pi}$ (Turcotte and Schubert 2014, section 4.18, eqs. 4.136 and 4.141, pp. 197-198). The roots are $\lambda$ = 0.220, 0.620 and 0.948. The lid uses the same family of profiles as this solution. It should hold the growth law to round-off.

**Why it matters.** When the surface of a magma ocean starts to freeze, the crust thickness sets how much heat the planet loses, and so how fast the interior cools.

![Crust thickness divided by the square root of kappa t against time for three Stefan numbers, with the Neumann values as horizontal lines](../assets/validation/surface_lid_v1_neumann_growth.svg#only-light){ width="80%" }
![Crust thickness divided by the square root of kappa t against time for three Stefan numbers, with the Neumann values as horizontal lines](../assets/validation/surface_lid_v1_neumann_growth_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The vertical axis is the crust thickness divided by $\sqrt{\kappa t}$, which is constant, $2\lambda$, on the textbook solution. The horizontal lines are $2\lambda$ for each Stefan number, and the dots are the lid at the three test times. The dots lie on the lines. The dashed curves are the lids that start at 1 m: they start too thick for their age and reach the lines within a few years.

??? info "Test details"
    - Reference: Turcotte and Schubert (2014), section 4.18, eqs. 4.136 and 4.141 (pp. 197-198).
    - Tolerance: relative 10⁻² at each test time (measured: 2.5 × 10⁻¹²); the test also checks 10⁻⁸. The lid that starts at 1 m is within 10⁻² after 7.6, 1.4 and 0.47 years at St = 0.1, 1 and 3.381.
    - Discrimination: the straight-line growth law $\delta = \sqrt{2\, \mathrm{St}\, \kappa t}$ misses the reference by 1.6, 14 and 37 %.
    - Test id: `tests/test_surface_lid_validation.py::test_v1_crust_growth_follows_neumann`
    - Command: `pytest tests/test_surface_lid_validation.py -k v1` (add `--validation-plots` to write the plot pair)

## A cooling solid without latent heat

**Why this check exists.** Without latent heat, the lid must behave as a solid that cools from its surface. A wrong surface slope of the profile gives the wrong heat flux. A flux twice the right one misses by 100 %. The test sees it.

**What it tests.** The same lid, with the latent heat at the small floor the solver keeps (10³ J/kg), so the Stefan number is 1320. The surface is held at 300 K, the base at 1400 K. The test compares the heat flux through the surface with the textbook flux from 0.095 to 31,700 years.

**How we know the right answer.** A half-space that starts at $T_b$, with its surface held at $T_s$, loses heat at the rate $F = k (T_b - T_s) / \sqrt{\pi \kappa t}$ (Turcotte and Schubert 2014, section 4.15, eq. 4.116, p. 189). At this Stefan number the lid profile is the half-space profile cut at the depth $2 \lambda \sqrt{\kappa t}$ with $\lambda$ = 2.40, so the lid flux is higher by the factor $1 / \operatorname{erf}\lambda$, a relative 7.0 × 10⁻⁴.

**Why it matters.** Once a crust has frozen through, it cools as a solid; this sets the late heat loss of a planet with a stagnant lid.

![Surface heat flux of the lid divided by the half-space flux against time, inside a band of plus or minus 1 percent](../assets/validation/surface_lid_v2_half_space.svg#only-light){ width="80%" }
![Surface heat flux of the lid divided by the half-space flux against time, inside a band of plus or minus 1 percent](../assets/validation/surface_lid_v2_half_space_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The vertical axis is the surface flux of the lid divided by the half-space flux; the grey band is the tolerance of ±1 %. The dots stay at 1.0007, the small offset that the depth of the lid explains, inside the band at all times.

??? info "Test details"
    - Reference: Turcotte and Schubert (2014), section 4.15, eq. 4.116 (p. 189).
    - Tolerance: relative 10⁻² at 25 times from three lid-diffusion times after the start (measured: 7.0 × 10⁻⁴).
    - Discrimination: twice the half-space flux misses by more than 40 % at every time.
    - Test id: `tests/test_surface_lid_validation.py::test_v2_half_space_cooling`
    - Command: `pytest tests/test_surface_lid_validation.py -k v2` (add `--validation-plots` to write the plot pair)

## A crust in steady state

**Why this check exists.** Under a constant heat flux from below, a crust must settle at the thickness where it conducts exactly that flux. A lid whose growth law or surface balance is wrong settles at the wrong thickness, or its profile stays curved; the test sees both.

**What it tests.** The melt below delivers a constant flux $F_b$ = 100 W/m² to the base of the lid. The surface radiates as a black body to an equilibrium temperature of 255 K, and the latent heat is 4 × 10⁵ J/kg. The lid starts 10 m thick, and the test checks it after ten relaxation times $\rho L \delta_\mathrm{eq}^2 / (k \Delta T)$, here 57 years each.

**How we know the right answer.** In steady state the crust conducts the base flux with a straight-line profile, so its thickness is $\delta_\mathrm{eq} = k (T_b - T_s) / F_b$, with the surface temperature $T_s$ set by the radiation balance $\sigma (T_s^4 - T_\mathrm{eq}^4) = F_b$: $T_s$ = 278.2 K and $\delta_\mathrm{eq}$ = 44.87 m. This follows from the conservation of energy alone.

**Why it matters.** A lid that floats on a convecting magma ocean sits close to this balance, so the crust thickness and the surface temperature of a run with the lid depend on it.

![Lid thickness and base heat flux divided by their steady values against time in units of the relaxation time](../assets/validation/surface_lid_v3_steady_crust.svg#only-light){ width="80%" }
![Lid thickness and base heat flux divided by their steady values against time in units of the relaxation time](../assets/validation/surface_lid_v3_steady_crust_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The horizontal axis is time divided by the relaxation time. The solid line is the lid thickness and the dashed line the heat conducted into the base of the lid, both divided by their steady values. The thin lid first conducts too much heat and grows; both lines stay inside 1 % of 1 from 2.5 relaxation times on.

??? info "Test details"
    - Reference: energy balance of a steady conductive crust under a grey-body surface.
    - Tolerance: thickness and base flux to a relative 10⁻⁴ after ten relaxation times, shape parameter $|s| < 10^{-3}$ (measured: 2.1 × 10⁻⁷, 8.8 × 10⁻⁸ and 1.7 × 10⁻⁷).
    - Test id: `tests/test_surface_lid_validation.py::test_v3_quasi_steady_crust`
    - Command: `pytest tests/test_surface_lid_validation.py -k v3` (add `--validation-plots` to write the plot pair)

## Checks of the lid inside the solver

These tests run the lid inside the full solver; they check exact properties of the code and have no plot.

- The energy of the mantle and the lid together changes only by the heat that crosses the surface and the core boundary, on random lid thicknesses up to the growth stop and random entropy profiles: `tests/test_subgrid_lid_solver.py::test_ledger_per_call_closes`.
- Over a solve, the energy budget closes to round-off, and a lid over a cooling mantle column grows as the Stefan law with the run's own Stefan number, to 2 %: `tests/test_subgrid_lid_solver.py::test_ledger_over_a_solve`.
- The hand-over of the lid to the mesh keeps the energy, and a run resumed after it evolves as a run without the lid: `tests/test_subgrid_lid_solver.py::test_merge_is_energy_exact`, `::test_resume_after_handover_matches_the_lid_off_run`.

**Known limit.** The lid hands over to the mesh only with the solid rheology on and `stress_closure_mode = 'lid'`, whose lid-base isotherm sets the hand-over depth; with another closure, or without the rheology, the lid never hands over (`tests/test_subgrid_lid_solver.py::test_no_handover_without_the_lid_closure`).
