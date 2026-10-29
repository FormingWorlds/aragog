# Surface lid

## What this page checks

This page checks the sub-grid surface lid, `subgrid_lid = true`: a conductive crust thinner than the top cell of the mesh, which Aragog carries as two extra states, the crust thickness and its energy per area. Validation here means that we run the lid on a case where the right answer is known from a textbook solution or from a run on a much finer mesh, and check that the lid gives that answer. If the lid were wrong, a run with the lid would grow its crust at the wrong rate, and the heat that leaves the planet through the crust would be wrong. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, then shows a plot and the test details.

Three of the cases drive the lid functions of `aragog.surface_lid` through time. These are the functions the solver calls in every right-hand side. The material properties are constant: thermal conductivity $k$ = 4 W/m/K, density $\rho$ = 3000 kg/m³ and heat capacity $c_p$ = 1200 J/kg/K, so the thermal diffusivity is $\kappa = k / (\rho c_p)$ = 1.11 × 10⁻⁶ m²/s. The base of the lid is at $T_b$ = 1400 K. These three check the lid on its own; the coupling of the lid to the mesh is checked by the fine-mesh case and by the solver tests listed at the end.

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

## A cooling solid shell on a fine mesh

**Why this check exists.** The two cases above drive the lid functions alone, with a constant heat capacity. This case runs the lid inside the solver, where the heat content of the crust comes from the equation of state, and compares it with the solver's own conduction on a mesh fine enough to resolve the crust. A lid whose heat content took the heat capacity and density of its base would miss by +2.3 to +2.5 %; with the values at 780 K, by -1.2 to -1.3 %. The test sees both.

**What it tests.** A solid shell 25 km thick starts at 1300 K, below the solidus (1421 K at the surface), and cools by conduction only through a black-body surface that radiates to 255 K. One run uses 10 nodes (cells of 2.8 km) with the lid; the other uses 4000 nodes (cells of 6.25 m) without the lid. The test compares their surface fluxes at 8 times from 300 to 5000 years. The surface of the fine mesh cools below 300 K, below the floor of the SPIDER test tables (about 320 K), so both runs use the PALEOS table extended to 67 K (md5 of `temperature_solid.dat`: `18d6f9eb5ba654d58fe3fad30ce118f0`).

**How we know the right answer.** At 300 years the cooled layer is about 97 m thick, 15 cells of the fine mesh, so the fine mesh resolves the crust that the lid replaces. Runs on 500, 1000 and 4000 nodes bound the error of the fine mesh itself: the change from 1000 to 4000 nodes is 1.79 % at 300 years and falls to 0.17 % at 5000 years.

| time (yr) | lid / 4000 nodes | bound of the 4000-node mesh |
|---|---|---|
| 300 | 1.0046 | 1.79 % |
| 448 | 1.0037 | 1.30 % |
| 670 | 1.0036 | 0.92 % |
| 1002 | 1.0035 | 0.71 % |
| 1497 | 1.0034 | 0.46 % |
| 2238 | 1.0034 | 0.36 % |
| 3345 | 1.0033 | 0.24 % |
| 5000 | 1.0033 | 0.17 % |

**Why it matters.** The lid exists so that a coarse mesh loses heat as a fine one would. Without the lid, the 10-node shell loses only 8 to 30 % of the heat the fine mesh loses over these times.

![Surface flux of the 10-node shell with the lid divided by that of the 4000-node shell, against time, with the error bound of the fine mesh](../assets/validation/surface_lid_v2_resolved_mesh.svg#only-light){ width="80%" }
![Surface flux of the 10-node shell with the lid divided by that of the 4000-node shell, against time, with the error bound of the fine mesh](../assets/validation/surface_lid_v2_resolved_mesh_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The dots are the surface flux of the shell with the lid divided by that of the fine mesh; the shaded band is the tolerance of ±1 %. The bars show how much the fine mesh itself can be off: long at early times, when the cooled layer is thin, and short later. The dots sit 0.3 to 0.5 % above 1, inside the band. Until about 2000 years they also lie inside the bars; later the bars are shorter than the offset of 0.3 %, which is then the difference of the lid itself.

??? info "Test details"
    - Reference: the 4000-node solver run without the lid, stored in `tests/reference/surface_lid_v2_grid.npz` by `tests/reference/build_surface_lid_v2_grid.py`.
    - Tolerance: relative 10⁻² at each time (measured: 3.3 × 10⁻³ to 4.6 × 10⁻³).
    - Discrimination: the lid with the heat content of its base misses by 2.3 % or more.
    - Test id: `tests/test_surface_lid_v2_grid.py::test_v2_lid_matches_the_resolved_mesh`; on the SPIDER test tables `::test_v2_lid_run_on_the_test_tables` reruns the lid shell against a stored pin.
    - Command: `pytest tests/test_surface_lid_v2_grid.py` (set `ARAGOG_LOWT_EOS_DIR` to the extended table to repeat the lid run on it; add `--validation-plots` to write the plot pair)

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
- Over a solve, the energy budget closes to round-off: `tests/test_subgrid_lid_solver.py::test_ledger_over_a_solve`.
- Sanity check: with no heat from the melt, a lid over a molten column grows as the Stefan law taken step by step along the run, with the Stefan number and diffusivity of each state, to 1 % at 10, 20 and 40 nodes (measured within 0.45 %): `tests/test_subgrid_lid_solver.py::test_lid_growth_sanity_against_quasi_static_neumann`. The grey surface cools during the run, so a law with one surface temperature does not apply. The check takes its heat capacity from the run, so it does not tell one heat content from another; the fine-mesh case above does.
- The hand-over of the lid to the mesh keeps the energy, and a run resumed after it evolves as a run without the lid: `tests/test_subgrid_lid_solver.py::test_merge_is_energy_exact`, `::test_resume_after_handover_matches_the_lid_off_run`.

**Known limit.** The lid hands over to the mesh only with the solid rheology on and `stress_closure_mode = 'lid'`, whose lid-base isotherm sets the hand-over depth; with another closure, or without the rheology, the lid never hands over (`tests/test_subgrid_lid_solver.py::test_no_handover_without_the_lid_closure`). Outside the solid regime (no active stagnant lid, or a top cell past the rheological transition) it does not hand over either (`::test_no_handover_outside_the_solid_regime`), and a hand-over whose isotherm has passed the face fires where the solid regime begins (`::test_handover_fires_where_the_gate_opens`).
