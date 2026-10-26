# Surface boundary

## What this page checks

This page checks the prescribed surface temperature, `outer_boundary_condition = 5`: the surface of the mantle is held at a fixed temperature, and the solver works out how much heat leaves through it (see [Surface boundary conditions](../Explanations/surface_bc.md)). Validation here means that we run Aragog on a case where the right answer is known from a textbook solution or from a convergence law, and check that Aragog gives that answer. If this boundary condition were wrong, every run with a fixed surface temperature would lose heat at the wrong rate, and its interior temperature and lid thickness would be wrong. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, then shows a plot and the test details.

All cases use the constant-properties mode: thermal conductivity $k$ = 4 W/m/K, density times heat capacity $\rho c_p$ = 4 × 10⁶ J/m³/K, so the thermal diffusivity is $\kappa = k / (\rho c_p)$ = 10⁻⁶ m²/s. The outer radius is 6371 km, the mesh is uniform in radius, no heat crosses the bottom of the shell, and the floor of the eddy diffusivity (`kappah_floor`) is 0. The tests drive the solver's own right-hand side with SciPy, so they test the same code a run uses.

## A heated shell in steady state

**Why this check exists.** The surface face could carry the wrong conduction: a full cell in place of a half cell, the wrong sign, or a gradient copied from the cell below. A heated shell then settles at the wrong temperatures, and the test sees it.

**What it tests.** A mantle shell from 3480 km to 6371 km radius is heated uniformly, $H$ = 3.4 × 10⁻¹³ W/kg, with the surface held at $T_s$ = 1000 K and convection off. The test finds the steady state, where heating and heat loss balance, on 50, 100 and 200 cells. It checks the surface heat flux and the temperature in every cell.

**How we know the right answer.** Conduction with uniform heating in a spherical shell has a closed-form steady state (Turcotte and Schubert 2014, eq. 4.40). With no flux at the inner radius $r_\mathrm{in}$ and $T_s$ at the outer radius $r_\mathrm{out}$ it is

$$T(r) = T_s + \frac{\rho H}{6k}\left(r_\mathrm{out}^2 - r^2\right) + \frac{\rho H r_\mathrm{in}^3}{3k}\left(\frac{1}{r_\mathrm{out}} - \frac{1}{r}\right),$$

and all heat made in the shell leaves through the surface: $F_\mathrm{top} = \rho H (r_\mathrm{out}^3 - r_\mathrm{in}^3) / (3 r_\mathrm{out}^2)$. The solver uses exact shell volumes, so this flux holds on every grid. The temperature error of a second-order scheme falls as $N^{-2}$, with $N$ the number of cells.

**Why it matters.** The interior temperature of a run with a fixed surface temperature depends on how the surface face converts the temperature difference into a flux.

![Steady temperature of the heated shell against the closed form, and N squared times the temperature error against N](../assets/validation/bc5_steady_heated_shell.svg#only-light){ width="80%" }
![Steady temperature of the heated shell against the closed form, and N squared times the temperature error against N](../assets/validation/bc5_steady_heated_shell_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Top: the temperature against radius from the closed form (line) and from Aragog on 50 cells (dots); the dots lie on the line. Bottom: $N^2$ times the largest temperature error, divided by the temperature rise at the inner radius; a horizontal line means the error falls as $N^{-2}$.

??? info "Test details"
    - Reference: Turcotte and Schubert (2014), section 4.9, eq. 4.40, with zero flux at the inner radius.
    - Tolerance: surface flux to a relative 10⁻⁸ on each grid; temperature error with order 2.0 ± 0.2 between 50, 100 and 200 cells.
    - Discrimination: the test asserts that half the expected flux, the flux of a full-cell face, misses by more than 1 %.
    - Test id: `tests/test_entropy_bc5_validation.py::test_the_steady_heated_shell_matches_the_closed_form`
    - Command: `pytest tests/test_entropy_bc5_validation.py -k steady_heated` (add `--validation-plots` to write the plot pair)

## Cooling below a fixed surface temperature

**Why this check exists.** A wrong surface face also gives a wrong heat flux while the mantle cools, not only in steady state. This case follows the flux through time.

**What it tests.** A shell 25 km thick, divided into 400 cells (62.5 m each), starts at $T_i$ = 1100 K everywhere. Its surface is held at $T_s$ = 1000 K, and convection is off. The test records the surface flux at 16 times from 5.3 × 10¹⁰ s to 7.5 × 10¹² s (about 1700 to 240,000 years).

**How we know the right answer.** A half space that starts at $T_i$, with its surface held at $T_s$, loses heat at the rate $F(t) = k (T_i - T_s) / \sqrt{\pi \kappa t}$ (Turcotte and Schubert 2014, section 4.15, eq. 4.116). The shell differs from a half space in 3 small ways, each estimated before the run:

- The grid error of the flux is about $0.063\,(h/\delta)^2$, with $h$ the cell size and $\delta = \sqrt{\kappa t}$ the depth of the cooled layer: 4.7 × 10⁻³ at the first time and 3.2 × 10⁻⁵ at the last.
- The curvature of the shell lowers the flux by $\sqrt{\pi \kappa t} / r_\mathrm{out}$, at most 7.6 × 10⁻⁴.
- The bottom of the shell lies more than 9 cooled-layer depths down, so the shell never feels it.

**Why it matters.** The cooling of the lid below a fixed surface temperature sets how fast a stagnant-lid planet loses heat.

![Surface heat flux against time compared with the half-space solution, and the relative difference](../assets/validation/bc5_half_space_cooling.svg#only-light){ width="80%" }
![Surface heat flux against time compared with the half-space solution, and the relative difference](../assets/validation/bc5_half_space_cooling_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Top: the surface flux against time on log axes, from the half-space solution (line) and from Aragog (dots). Bottom: the relative difference, with the tolerance as dashed lines; the points stay between them.

??? info "Test details"
    - Reference: Turcotte and Schubert (2014), section 4.15, eq. 4.116.
    - Tolerance: relative 10⁻² at every output time (sum of the estimates above: below 5.5 × 10⁻³).
    - Discrimination: the test asserts that half the flux, the flux of a full-cell face, misses by more than 40 %.
    - Test id: `tests/test_entropy_bc5_validation.py::test_cooling_below_a_fixed_surface_temperature_follows_the_half_space_flux`
    - Command: `pytest tests/test_entropy_bc5_validation.py -k half_space` (add `--validation-plots` to write the plot pair)

## Grid convergence with convection

**Why this check exists.** With convection on, the entropy gradient at the surface also drives the convective flux. A gradient that is inconsistent with the conduction would make the result depend on the grid in an irregular way. This check shows that the result converges at the rate of the scheme.

**What it tests.** The heated shell of the first case, now with mixing-length convection at a viscosity of 10²¹ Pa s, once with the mixing length 0 at both boundaries (`nearest_boundary`) and once with a constant mixing length (`constant`). The test runs each to steady state on 100, 200 and 400 cells. It compares the mean temperature and the temperature at 5 fixed radii between the grids.

**How we know the right answer.** For a second-order scheme the difference between two grids falls by a factor of 4 each time the cells are halved. From 3 grids, Richardson extrapolation estimates the value on an infinitely fine grid. The surface flux needs no extrapolation: in steady state it equals the total heating divided by the surface area on every grid. Before the check the test asserts that the surface boundary layer (the depth where the temperature reaches 90 % of its rise at mid-mantle) spans at least 8 cells, so that the finest grid resolves it.

**Why it matters.** Most runs use convection; this check covers the part of the surface flux that conduction alone does not test.

![Steady temperature of the convecting shell for two mixing-length profiles, and N squared times the error of the mean temperature](../assets/validation/bc5_convecting_shell_convergence.svg#only-light){ width="80%" }
![Steady temperature of the convecting shell for two mixing-length profiles, and N squared times the error of the mean temperature](../assets/validation/bc5_convecting_shell_convergence_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Top: the steady temperature against radius for the two mixing-length profiles on 400 cells. Bottom: $N^2$ times the error of the mean temperature against its Richardson value, divided by the temperature rise; horizontal lines mean second-order convergence.

??? info "Test details"
    - Reference: the convergence order of the scheme and Richardson extrapolation; the steady surface flux from the energy balance.
    - Tolerance: order 2.0 ± 0.4 for the mean temperature and the 5 fixed-radius temperatures; the 400-cell values within 10⁻³ of the temperature rise from their extrapolation; surface flux to a relative 10⁻⁸; boundary layer at least 8 cells wide on 400 cells.
    - Test id: `tests/test_entropy_bc5_validation.py::test_the_convecting_shell_converges_with_second_order`
    - Command: `pytest tests/test_entropy_bc5_validation.py -k convecting` (add `--validation-plots` to write the plot pair)

## Stagnant-lid benchmarks without the surface patch

**Why this check exists.** The stagnant-lid scaling benchmarks A6-1 and A6-2 hold the surface at 273 K. Their tools replace the surface flux with half-cell conduction from outside the solver. This check runs the stored benchmark states with the boundary condition of the solver instead.

**What it tests.** The 5 stored A6-2 steady states and 3 stored A6-1 states, evaluated once with the tool replacement and once with `outer_boundary_condition = 5` alone. The surface flux must be the same.

**How we know the right answer.** The tool replacement sets the same half-cell conduction as the boundary condition, and both tools switch the eddy-diffusivity floor off, so the surface flux must agree to round-off.

**Why it matters.** With the boundary condition in the solver, the benchmarks run without code that changes the solver from outside.

**Known limit.** The surface flux agrees on all 8 states. The tools and their stored states belong to the stagnant-lid verification work and are not on this branch yet, so the committed test of this case comes with the change that removes the tool replacement after both are merged. The tool replacement left the surface node at the temperature extrapolated from the cells below, which the stagnant-lid closure reads; with the boundary condition the node is at 273 K, the stored states are not exactly steady, and that change regenerates them.

??? info "Test details"
    - Reference: the stored A6-1 and A6-2 states and their surface fluxes.
    - Tolerance: surface flux to a relative 10⁻¹⁰.
    - Test id: none on this branch; the committed test comes with the removal of the tool replacement.
    - Command: none on this branch.

## References

- Turcotte, D. L., & Schubert, G. (2014). *Geodynamics* (3rd ed.). Cambridge University Press. ISBN 978-1-107-00653-9.
