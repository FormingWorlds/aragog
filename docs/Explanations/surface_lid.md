# Surface lid

How does Aragog follow a crust that is much thinner than the top cell of the mesh?

**Short answer.** With `subgrid_lid = true`, Aragog carries the crust as two extra states: its thickness $\delta$ \[m\] and its energy per area $E_\mathrm{lid}$ \[J/m²\]. The temperature across the crust follows a profile from the textbook solution for a freezing lava layer, so the crust conducts the right heat from its first metres on. The top cell keeps only the melt below the crust, and new crust forms at the base by the balance of the heat conducted up through the crust and the heat the melt delivers.

## Why the mesh alone cannot do it

A magma ocean need not freeze from its surface. The adiabat is steeper than the solidus, so the two meet first at depth and solidification is expected to begin at the bottom, the more so the larger the pressure range of the planet. An insulating atmosphere can keep the surface above its solidus for much of the solidification, and a quench crust on a planet-sized body is then unlikely (Elkins-Tanton (2012)[^cite-elkinstanton2012] section 4.1).

A crust can still form at the surface when the surface loses heat faster than the melt below delivers it, for example under a thin atmosphere. The lid treats that crust; it does not decide where in the mantle freezing starts.

That crust starts metres thick, while the top cell of a 100-node mantle is 29 to 35 km thick. A cell that thick has one temperature: it is either all melt or all solid. While the crust is thin, neither choice is right. As melt, the cell radiates from a hot surface; as solid, it blocks the heat of the whole cell. The heat the planet loses then jumps between two wrong values as the top cell freezes and thaws. A mesh fine enough to resolve a metre-thick crust needs metre-thick cells, which makes the solver steps tiny.

## The two states and the profile

The crust runs from the surface, at temperature $T_s$, down to its base at depth $\delta$, at temperature $T_b$. Material joins the crust at its base, so $T_b$ is the temperature at which the melt becomes rigid: the solidus above the rheological transition of the melt, and the lid-base temperature of the solid rheology below it. The base is never warmer than the solidus and never warmer than the melt below the crust: $T_b \le T_f$, with $T_f$ the temperature of the top cell.

Between the surface and the base, the temperature follows one member of a family of profiles. With $z$ the depth below the surface,

$$T(z) = T_s + (T_b - T_s)\, f_s(z / \delta), \qquad f_s(x) = \frac{\operatorname{erf}(\sqrt{s}\, x)}{\operatorname{erf}\sqrt{s}},$$

and $f_s(x) = x$ for $s = 0$ (with the imaginary error function for $s < 0$). The energy $E_\mathrm{lid}$ is the heat content of the crust: the heat content per volume $H(T)$ of the solid, taken from the equation of state at the pressure of the top cell, integrated over the profile. For each shape the surface balance below fixes $T_s$, so one shape $s$ gives the stored energy.

A growing crust has $s > 0$, a steady crust has the straight line $s = 0$, and a thinning crust has $s < 0$. This family contains the exact solution for a crust that freezes over melt, $\delta = 2\lambda\sqrt{\kappa t}$ with $s = \lambda^2$ (Turcotte and Schubert (2014)[^cite-turcotte2014] section 4.18, Eqs. 4.136 and 4.141, pp. 197-198), so the lid grows exactly as that solution.

## The surface and the base

The surface temperature follows from the balance of the conducted heat and the grey-body radiation,

$$\varepsilon \sigma \left(T_s^4 - T_\mathrm{eq}^4\right) = k\, (T_b - T_s)\, \frac{f_s'(0)}{\delta},$$

with $\varepsilon$ the emissivity, $\sigma$ the Stefan-Boltzmann constant, $T_\mathrm{eq}$ the equilibrium temperature and $k$ the conductivity of the crust. The left side is $F_s$, the heat the planet loses \[W/m²\].

At the base the crust conducts $q_b = k (T_b - T_s) f_s'(1) / \delta$ upward, and the melt delivers $F_b$. The difference freezes new crust:

$$\left(W(S_f) - W(S_b)\right) \frac{d\delta}{dt} = q_b - F_b, \qquad W(S) = \int_{S_\mathrm{sol}}^{S} \rho\, T\, dS'.$$

Here $W$ \[J/m³\] is the heat content per volume at the top-cell pressure, counted from the solidus entropy $S_\mathrm{sol}$. It is the measure of the heat content the solver reports.

The entropy equation of the mesh has the capacitance $\rho T$ with the two phases blended near the solidus and the liquidus; there it differs from the table $\rho T$ in $W$ by less than 0.1 %. The crust holds heat in the same measure, as a function of its temperature, wherever the temperature rises with entropy. Some tables keep the temperature at the solidus over a band of entropy below $S_\mathrm{sol}$; a crust profile at the solidus temperature then holds the heat of the lower end of that band, not $W(S_\mathrm{sol}) = 0$.

$S_f$ is the entropy of the melt and $S_b$ that of the crust at its base. The left side is the heat released when melt joins the crust, latent heat of the melt fraction included; a melt at the base entropy releases none. The density sits inside the integral because it changes along the way from $S_b$ to $S_f$. The melt delivers $F_b$, the larger of the heat flux into the top cell from below and the flux of a boundary layer at the critical Rayleigh number `lid_ra_crit`, and at least 0. With `energy.convection` off the melt does not convect, so $F_b$ is the flux into the top cell from below, or 0.

### The base at the melt temperature

**Why.** The lid-base temperature of the solid rheology can lie above the temperature of the melt below the crust, for example a fixed `lid_base_temperature` of 1300 K over a top cell at 1200 K. Material at the melt temperature cannot join a crust whose base is warmer: the jump $W(S_f) - W(S_b)$ would turn negative and the crust would grow from heat it does not get.

**What.** Below the rheological transition the base entropy is capped at the melt, $S_b = \min(S_\mathrm{rh}, S_\mathrm{sol}, S_f)$, with $S_\mathrm{rh}$ the entropy of the rheological lid-base temperature. At the cap the jump is 0, and the balance above has no finite growth rate. The heat jump per mass, $(W(S_f) - W(S_b)) / \rho_b$ with $\rho_b$ the density at the base, therefore has a floor `LID_DH_MIN` of 10³ J/kg, a numerical help of the solver. At the cap the crust grows at $d\delta/dt = (q_b - F_b) / (\rho_b \times 10^3\ \mathrm{J/kg})$.

**Why it matters.** At the cap the growth rate depends on the floor and not on a latent heat, so the crust thickness and the surface flux there depend on the floor and on the mesh. The [Validation page](../Validation/surface_lid.md) gives the size of that dependence.

## Energy

The heat content of the crust follows the equation of state across the crust, so its heat capacity and density change from the base to the surface as they do in the mesh. A crust that took them at its base would hold too much heat under a cold surface: on the PALEOS table at 10 MPa the heat capacity of the solid at 255 K is 55 % of its value at 1300 K.

The top cell keeps its melt part: its volume shrinks by the crust volume, area times $\delta$, and the heat $W(S_f)$ of each unit of that volume passes to the crust. The flux out of the melt follows from the energy of the melt and the crust together, so the energy of the mantle changes only by the heat that crosses the surface and the core boundary. The energy diagnostics count the crust, and their solver residual reports this balance for every call. Radiogenic and tidal heat act on the melt part of the top cell and on the crust mass, and the reported source powers state that heating.

## When the crust exists, and when it goes

A hot, fast-convecting melt carries more heat than a crust at $T_b$ can radiate. No crust can then form. The lid weight $w$ \[0 to 1\] compares the thickness of a crust in steady state, $\delta_\mathrm{eq} = k (T_b - T_s) / F_b$, with `lid_delta_on`: below it the lid is hidden ($w = 0$) and the surface is the mesh surface, above it the lid acts ($w = 1$). A hidden lid relaxes to a thin phantom lid of thickness `lid_delta_lo`.

When the crust is resolved by the mesh, the lid hands over to the mesh: the solver merges the crust into the top cell with its energy, sets both lid states to zero and continues. A zero lid stays zero. Which condition starts the hand-over depends on the solid rheology.

**With the solid rheology and `stress_closure_mode = 'lid'`.** When the mantle near the surface is solid, a crust that has grown thicker than a few top cells is resolved by the mesh. The lid hands over when three conditions hold together: the base of the cold region (the deepest point where the column rises through the lid-base isotherm of the solid rheology) lies below the face of `lid_handover_cells` top cells, the stagnant-lid state of the solid rheology is active (its weight `w_active` above 0.5), and the top cell is below the rheological transition (the melt ramp $w_\phi$ of the lid base below 0.5).

Why all three: in a molten or mushy mantle the solid rheology has no stagnant lid, so its isotherm marks no crust and can lie hundreds of kilometres deep.

**Without the solid rheology.** The lid base is the solidus. When the top cell cools to its solidus, $S_f \le S_\mathrm{sol}$, the jump $W(S_f) - W(S_b)$ is 0: no melt is left below the crust to freeze, and the freezing front has passed through the top cell. From there the mesh carries the cold top cell, and the lid hands over. A top cell at or below the solidus at the start of a call hands over before the call runs; a top cell that cools through the solidus during a call hands over at that time.

**With the solid rheology and another closure.** The lid does not hand over. The base then lies at the rheological temperature, at or below the solidus, and a top cell can freeze through while melt stays under a thin resolved crust.

## Where the lid is valid

**Why.** The lid replaces a crust the mesh cannot resolve, so it is right only where its answer has been compared with a mesh that does resolve that crust.

**What.** The [Validation page](../Validation/surface_lid.md) compares the lid with fine meshes for two starts: a solid shell below the solidus and a shell in the mush that freezes from its surface. For a molten shell without the solid rheology it compares only the hand-over at the solidus and the coarse mesh after it; the lid itself has no measured error before the hand-over. A molten start with the solid rheology on, with either closure, is not validated: no run resolves its crust yet, so no converged reference exists.

Where the lid base sits at the melt temperature, a hot shell with the solid rheology and the `'lid'` closure gives a crust that changes by up to 13 % from 200 to 400 nodes from 3 × 10³ to 3 × 10⁵ years; later its thickness depends on the mesh and on the floor (see [mesh and floor](../Validation/surface_lid.md#the-lid-base-at-the-melt-temperature-mesh-and-floor)).

**Why it matters.** Use the lid for a molten start with the solid rheology only with this limit in mind; its heat loss there has no measured error.

## A worked example

A melt layer at $T_b$ = 1400 K freezes under a surface held at 300 K, with $c_p$ = 1200 J/kg/K and a latent heat of 3.9 × 10⁵ J/kg, so the Stefan number $c_p (T_b - T_s) / L$ is 3.385 and $\lambda$ = 0.948. With $\kappa$ = 1.1 × 10⁻⁶ m²/s, the crust is $2 \lambda \sqrt{\kappa t}$ = 11 m thick after one year and 353 m after a thousand years. Both are far below a 30 km top cell.

## Settings and limits

- `subgrid_lid` (bool, default false): the switch. Only with the grey-body surface, `outer_boundary_condition = 1`. It is refused at setup with `core_bc = 'gradient'` or `'bower2018'`, and with `param_utbl = true`, because the lid replaces the surface flux that the correction acts on.
- `lid_delta_on` \[m\] (default 1): must be finite, positive and below 0.8 of the top-cell width; a larger value is refused at solver setup with the width named.
- `lid_delta_lo` \[m\] (default `lid_delta_on / 2`): must lie in (0, `lid_delta_on`).
- `lid_handover_cells` (integer ≥ 1, default 3): must leave at least one cell below the hand-over face.
- `lid_ra_crit` (default 450): positive and finite.
- The crust stops growing at 0.8 of the top-cell width (one logged warning per run) and stops thinning at `lid_delta_lo`. The integrator can leave a crust at the growth stop slightly above it; the step check accepts up to 10⁻³ above the stop and records the largest excess of each call as `lid_cap_excess` on the solution.
- The heat jump per mass, $(W(S_f) - W(S_b)) / \rho_b$ with $\rho_b$ the density at the lid base, has a floor of 10³ J/kg (one logged warning when it acts).
- The floors cost solver steps. While the top cell freezes and thaws, the lid forms and vanishes at its phantom thickness and passes the weight floor and the lower stop many times.
- The lid hands over with the solid rheology and `stress_closure_mode = 'lid'`, or without the solid rheology. With the solid rheology and another closure it never hands over. A molten start with the solid rheology is not validated (see [Where the lid is valid](#where-the-lid-is-valid)).
- The lid takes its base temperature, its heat content and its density at the pressure of the top-cell centre, not at the pressure of its base. On a coarse mesh the two differ: on the 10-node mantle of the [tutorial](../Tutorials/surface_lid.md) the top-cell centre lies at 7.8 GPa, where the solidus is 2149 K, against 1439 K at the surface. Use a mesh whose top cell is thin enough that the solidus changes little across it.
- A coupled run with a prescribed surface flux (`outer_boundary_condition = 4`) cannot use the lid: the lid sets the surface flux itself through the grey-body balance. PROTEUS does not pass the lid settings to Aragog, so a PROTEUS run cannot turn the lid on without a change in PROTEUS.
- The lid has no buoyancy test: it stays at the surface until the hand-over, whatever the density contrast with the fluid below. A crust denser than the melt below it is gravitationally unstable if the two have a similar composition (Miyazaki and Stevenson (2024)[^cite-miyazaki2024] p. 1). That study finds that on small bodies the foundering of the whole crust is often improbable, but this is not modelled here, and a crust that founders in a run is not followed.

    On the tables of the [Validation page](../Validation/surface_lid.md), at its top-cell pressure of 55.7 MPa, the sign is the opposite: the melt at the liquidus is denser than the solid at the solidus (3603 against 3041 kg/m³ on the extended PALEOS table, a ratio of 1.185). This sign is a property of those tables and the opposite of what that study takes for silicates of one composition, so do not read it as a measured result. The density sign in a run comes from the table, not from a check in the lid.

- At the growth stop the lid base lies deeper than the node of the top cell, by $\delta - h/2$ with $h$ the top-cell width. The base has the temperature of that node, so the column below is warmer than on a mesh that resolves the crust, by about $(\delta - h/2)(T_b - T_s)/\delta$, with the mean temperature gradient of the lid. This size is an estimate from the geometry: no run exists with the lid base moved.
- The melt flux $F_b$ at the lid base is the flux per area at the lower face of the top cell, not at the lid base. The bias in the growth rate and the weight is about $2h/R$, with $h$ the top-cell width and $R$ the planet radius: 1.1 % for a 35 km top cell of the Earth, 0.09 % on the shell of the Validation page. The energy budget is not affected, because the flux out of the melt follows from conservation.
- With the solid rheology off, the lid checks and the EOS table guard accept an entropy up to 10⁻³ J/kg/K past a table end; in that band the heat content continues linearly. With the solid rheology on, its own check refuses a state outside the table at the first step.
- Below the first knot of the heat table the lid continues the heat content with the slope at that knot; one logged warning says so. On the JAX path the check reads only the last state of a call, as the growth-stop warning does. The SPIDER test tables start near 320 K, so a lid surface can lie below them under a 255 K sky.
- The lid runs on the NumPy and on the JAX right-hand side. A JAX factory for a run with the lid must be built with `lid_params=solver.jax_lid_params()`; a factory that disagrees with the run on the lid is refused before the solve.

**Where this is checked.** [Surface lid](../Validation/surface_lid.md) compares the lid with the freezing lava layer, a cooling half-space and a steady crust, and with a fine mesh for a solid shell and for a [shell that freezes from the mush](../Validation/surface_lid.md#a-shell-freezing-from-the-mush-on-a-fine-mesh), which tests the Stefan jump. It also lists the solver tests of the energy budget, the Stefan jump and the hand-over.

## Restarting a run

A new solver takes the lid from `set_initial_lid(delta, E_lid)`, in either order with `set_initial_entropy`; the lid states are the last two entries of the solver state. Without it, a new solver starts the phantom lid, so a run that resumes must pass its lid on. A lid that has handed over resumes with `set_initial_lid(0.0, 0.0, handover_time=t)`, where `t` is the hand-over time in years; without the time, `lid_handover_time` in the output is NaN.

A solver that is used again continues from its previous solution, unless `reset` has changed the number of nodes: the phantom lid then starts, and a logged warning says so. When `reset` has made the top cell thinner than that lid can fit, the next start stops with an error that names both thicknesses, and leaves the solver as it was: call `set_initial_lid` with a start, then `set_initial_entropy`. A start with a live lid clears the hand-over time.

[^cite-elkinstanton2012]: L. T. Elkins-Tanton, *[Magma oceans in the inner solar system](https://doi.org/10.1146/annurev-earth-042711-105503)*, Annual Review of Earth and Planetary Sciences, 40, 113–139, 2012. [SciX](https://scixplorer.org/abs/2012AREPS..40..113E/abstract).
[^cite-miyazaki2024]: Y. Miyazaki, D. J. Stevenson, *[The stability of a dense crust situated on small bodies](https://doi.org/10.3847/PSJ/ad65d2)*, The Planetary Science Journal, 5, 192, 2024. [SciX](https://scixplorer.org/abs/2024PSJ.....5..192M/abstract).
[^cite-turcotte2014]: D. L. Turcotte, G. Schubert, *[Geodynamics](https://doi.org/10.1017/CBO9780511843877)*, 3rd edition, Cambridge University Press, 2014. [SciX](https://scixplorer.org/abs/2014geod.book.....T/abstract).
