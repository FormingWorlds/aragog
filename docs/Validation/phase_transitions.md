# Phase transitions

## What this page checks

This page checks how Aragog changes the material properties of the mantle (heat capacity, density, expansivity, adiabatic gradient and viscosity) when the entropy of a node crosses the solidus or the liquidus (see [Phase transitions](../Explanations/phase_transitions.md#width-of-the-blend-at-the-solidus-and-the-liquidus)). Validation here means that we run the phase evaluator on the SPIDER test tables, where the right answer follows from the blend formula, and check that Aragog gives that answer. If this part were wrong, the heat flux through a cell face would jump when a node starts or stops to melt, and the solver would take very small time steps there. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, then shows a plot or says why it has none, and the test details.

All cases use the SPIDER P-S test tables (`ARAGOG_TEST_EOS_DIR`, [Build tests](../How-to/build_tests.md)) on a 12-node mesh from the core-mantle boundary (146 GPa) to the surface (0 GPa). The symbol $g = (S - S_\mathrm{sol}) / (S_\mathrm{liq} - S_\mathrm{sol})$ is the melt coordinate: 0 on the solidus, 1 on the liquidus, below 0 in the solid and above 1 in the liquid. The width $w$ is the setting `matprop_smooth_width`; the default is 0.01.

## On each boundary, a property is the mean of its two values

**Why this check exists.** Near a boundary every property has two values: the two-phase value of the mushy zone and the single-phase value from the solid or the melt table. If the code gave the wrong weight on the boundary, for example 1 as with $w = 0$, the property would take the two-phase value alone. On the test tables the two values of the heat capacity differ by 3.5 % to a factor of 23, so the test sees that error at every pressure.

**What it tests.** The test puts every node exactly on the solidus ($g = 0$), then exactly on the liquidus ($g = 1$), at the default width. It compares the heat capacity, the density, the adiabatic gradient $(\partial T/\partial P)_S$ and the logarithm of the viscosity with the mean of the two values. It takes the two-phase value from the evaluator at $w = 0$, which uses weight 1 on the boundary, and the single-phase value from the tables directly. The two values of the density are the same on the boundary, so the density is compared but does not test the weight; the test asserts the gap between the two values for the heat capacity.

**How we know the right answer.** SPIDER's `get_smoothing` (`util.c`) gives the two-phase value the weight $\tfrac{1}{2}[1 + \tanh(g/w)]$ below $g = 0.5$ and $\tfrac{1}{2}[1 - \tanh((g - 1)/w)]$ above it. Both are exactly $\tfrac{1}{2}$ on the boundary for any $w > 0$, so each property is the mean of its two values. The expansivity is not compared, because the evaluator changes negative values after the blend with a function that is not linear.

**Why it matters.** The properties on the boundary set the heat flux of a node that is just starting to melt or to freeze.

![Heat capacity and viscosity near the solidus and the liquidus at 65.7 GPa, divided by their value on the boundary, at width 0.01 and width 0](../assets/validation/phase_blend_at_solidus_and_liquidus.svg#only-light){ width="80%" }
![Heat capacity and viscosity near the solidus and the liquidus at 65.7 GPa, divided by their value on the boundary, at width 0.01 and width 0](../assets/validation/phase_blend_at_solidus_and_liquidus_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Top: the solidus; bottom: the liquidus, both at 65.7 GPa. The horizontal axis is the entropy minus the entropy of the boundary, in J/kg/K. The vertical axis is a property $X$ divided by its value on the boundary. Solid lines use the default width 0.01, dotted lines width 0. Solid lines pass through 1 on the boundary and change smoothly over about 81 J/kg/K (0.01 of the melting interval of 8148 J/kg/K). Dotted lines jump there; the jump looks like a steep line only because the points are 20 J/kg/K apart.

??? info "Test details"
    - Reference: SPIDER `get_smoothing` (`util.c`): the weight is $\tfrac{1}{2}$ at $g = 0$ and $g = 1$ for any $w > 0$.
    - Tolerance: 10⁻¹² relative at all 12 node pressures, for the heat capacity, the density, the adiabatic gradient and the logarithm of the viscosity (measured: 0).
    - Discrimination: the two values of the heat capacity differ by at least 155 % of the mean at the solidus and 3.5 % at the liquidus; the test asserts more than 2 %, so a weight of 0 or 1 on the boundary fails.
    - Test id: `tests/test_matprop_smooth_width_validation.py::test_on_each_boundary_every_property_is_the_mean_of_the_two_branches`
    - Command: `pytest tests/test_matprop_smooth_width_validation.py -k mean` (add `--validation-plots` to write the plot pair)

## No jump at the solidus and the liquidus

**Why this check exists.** With $w = 0$, or with a weight that jumps, a property changes by a finite step between two entropies that are almost equal. On the test tables the heat capacity then jumps by a factor of 7.9 to 23 at the solidus.

**What it tests.** The test puts every node 10⁻⁹ J/kg/K below a boundary, then 10⁻⁹ J/kg/K above it, at the default width. It compares the heat capacity, the density, the expansivity, the adiabatic gradient and the logarithm of the viscosity between the two states.

**How we know the right answer.** The weight changes by about $10^{-9} / (w \Delta S)$ over that step, with $\Delta S = S_\mathrm{liq} - S_\mathrm{sol}$ between 2563 and 8394 J/kg/K on this mesh. A continuous property therefore changes by far less than 10⁻⁶ of its value.

**Why it matters.** A step in a property gives a step in the heat flux through a face, and the time stepper must then take very small steps each time a node crosses a boundary.

This case has no plot of its own: the plot of the case above shows the same curves near both boundaries.

??? info "Test details"
    - Reference: continuity of the blend for $w > 0$.
    - Tolerance: 10⁻⁶ relative at all 12 node pressures (measured: at most 1.2 × 10⁻⁷, at the surface node, where the blended expansivity is negative, −2.1 × 10⁻⁴ 1/K, and the clamp of negative values maps it to 1.2 × 10⁻¹³ 1/K, so a round-off change of the blended value shows as a larger relative change; below 3 × 10⁻¹¹ at the other nodes).
    - Discrimination: at $w = 0$ the heat capacity jumps by more than a factor of 7 at every node at the solidus (measured 7.9 to 23).
    - Test id: `tests/test_matprop_smooth_width_validation.py::test_the_default_width_keeps_every_property_continuous_across_the_boundary`
    - Command: `pytest tests/test_matprop_smooth_width_validation.py -k continuous`

## Far from both boundaries the width changes nothing

**Why this check exists.** The blend must only act near the boundaries. If the width leaked into the solid, the deep mushy zone or the liquid, a change of the width would change runs that never come near a boundary.

**What it tests.** The test puts every cell at $g = -0.25$, $0.35$ and $1.3$, and checks first that every node, the surface node included, is at least 0.2 in $g$ from both boundaries. It then compares the right-hand side of the entropy equation at the default width and at $w = 0$. As a control, it repeats this with every cell at $g = 0.02$, inside the blend.

**How we know the right answer.** More than 0.2 from a boundary, $|g/w|$ or $|(g - 1)/w|$ is at least 20, and $\tanh(20)$ rounds to exactly 1 in double precision. The weight is then exactly 0 or 1, as at $w = 0$.

**Why it matters.** It shows that the default only changes states near a phase boundary.

This case has no plot: it checks that two arrays are equal bit for bit.

??? info "Test details"
    - Reference: $\tanh(20) = 1$ in double precision.
    - Tolerance: bit-identical right-hand side in all 3 states; at $g = 0.02$ the right-hand sides differ.
    - Test id: `tests/test_matprop_smooth_width_validation.py::test_far_from_both_boundaries_the_rhs_does_not_change_with_the_width`
    - Command: `pytest tests/test_matprop_smooth_width_validation.py -k far_from`

## The JAX right-hand side follows the width

**Why this check exists.** The production solver uses the JAX right-hand side and its Jacobian. If the JAX kernel blended the properties with a different width from the NumPy evaluator, the two would disagree near every boundary.

**What it tests.** The test puts 9 cells at $g$ = −0.02, −0.005, 0, 0.004, 0.012, 0.5, 0.99, 1 and 1.015, on both sides of both boundaries. A test helper builds the JAX parameters with the width of the configuration, as PROTEUS does in a coupled run; Aragog itself does not build JAX parameters from a configuration. The test compares the JAX right-hand side with the NumPy one at the default width. As a control, it sets the JAX width to 0. This case does not test the weight exactly on a boundary: with that weight set to 1 in the NumPy evaluator it still passes, and the first case fails.

**How we know the right answer.** The NumPy evaluator is the reference; both paths use the same blend formula.

**Why it matters.** A difference between the two paths would make the Jacobian of the solver wrong near a boundary.

This case has no plot: it checks the agreement of two arrays to round-off.

??? info "Test details"
    - Reference: the NumPy right-hand side.
    - Tolerance: 10⁻¹⁰ relative (measured: 5.9 × 10⁻¹⁵); at JAX width 0 the difference is more than 10⁻⁶ (measured: a relative difference of 6.1).
    - Test id: `tests/test_matprop_smooth_width_validation.py::test_the_jax_rhs_matches_numpy_inside_the_blend_at_the_default_width`
    - Command: `pytest tests/test_matprop_smooth_width_validation.py -k jax`
