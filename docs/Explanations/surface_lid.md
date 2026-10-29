# Surface lid

How does Aragog follow a crust that is much thinner than the top cell of the mesh?

**Short answer.** With `subgrid_lid = true`, Aragog carries the crust as two extra states: its thickness $\delta$ \[m\] and its energy per area $E_\mathrm{lid}$ \[J/m²\]. The temperature across the crust follows a profile from the textbook solution for a freezing lava layer, so the crust conducts the right heat from its first metres on. The top cell keeps only the melt below the crust, and new crust forms at the base by the balance of the heat conducted up through the crust and the heat the melt delivers.

## Why the mesh alone cannot do it

A magma ocean freezes from its surface down. The first crust is metres thick, while the top cell of a 100-node mantle is about 11 km thick. A cell that thick has one temperature: it is either all melt or all solid. While the crust is thin, neither choice is right. As melt, the cell radiates from a hot surface; as solid, it blocks the heat of the whole cell. The heat the planet loses then jumps between two wrong values as the top cell freezes and thaws. A mesh fine enough to resolve a metre-thick crust needs metre-thick cells, which makes the solver steps tiny.

## The two states and the profile

The crust runs from the surface, at temperature $T_s$, down to its base at depth $\delta$, at temperature $T_b$. Material joins the crust at its base, so $T_b$ is the temperature at which the melt becomes rigid: the solidus above the rheological transition of the melt, and the lid-base temperature of the solid rheology below it.

Between the surface and the base, the temperature follows one member of a family of profiles. With $z$ the depth below the surface,

$$T(z) = T_s + (T_b - T_s)\, f_s(z / \delta), \qquad f_s(x) = \frac{\operatorname{erf}(\sqrt{s}\, x)}{\operatorname{erf}\sqrt{s}},$$

and $f_s(x) = x$ for $s = 0$ (with the imaginary error function for $s < 0$). The energy $E_\mathrm{lid}$ is the heat content of the crust: the heat content per volume $H(T)$ of the solid, taken from the equation of state at the pressure of the top cell, integrated over the profile. For each shape the surface balance below fixes $T_s$, so one shape $s$ gives the stored energy. A growing crust has $s > 0$, a steady crust has the straight line $s = 0$, and a thinning crust has $s < 0$. This family contains the exact solution for a crust that freezes over melt, $\delta = 2\lambda\sqrt{\kappa t}$ with $s = \lambda^2$ (Turcotte and Schubert 2014, section 4.18, eqs. 4.136 and 4.141, pp. 197-198), so the lid grows exactly as that solution.

## The surface and the base

The surface temperature follows from the balance of the conducted heat and the grey-body radiation,

$$\varepsilon \sigma \left(T_s^4 - T_\mathrm{eq}^4\right) = k\, (T_b - T_s)\, \frac{f_s'(0)}{\delta},$$

with $\varepsilon$ the emissivity, $\sigma$ the Stefan-Boltzmann constant, $T_\mathrm{eq}$ the equilibrium temperature and $k$ the conductivity of the crust. The left side is $F_s$, the heat the planet loses \[W/m²\].

At the base the crust conducts $q_b = k (T_b - T_s) f_s'(1) / \delta$ upward, and the melt delivers $F_b$. The difference freezes new crust:

$$\left(W(S_f) - W(S_b)\right) \frac{d\delta}{dt} = q_b - F_b, \qquad W(S) = \int_{S_\mathrm{sol}}^{S} \rho\, T\, dS'.$$

Here $W$ \[J/m³\] is the heat content per volume at the top-cell pressure, counted from the solidus entropy $S_\mathrm{sol}$. It is the measure of the heat content the solver reports. The entropy equation of the mesh has the capacitance $\rho T$ with the two phases blended near the solidus and the liquidus; there it differs from the table $\rho T$ in $W$ by less than 0.1 %. The crust holds heat in the same measure, as a function of its temperature, wherever the temperature rises with entropy. Some tables keep the temperature at the solidus over a band of entropy below $S_\mathrm{sol}$; a crust profile at the solidus temperature then holds the heat of the lower end of that band, not $W(S_\mathrm{sol}) = 0$. $S_f$ is the entropy of the melt and $S_b$ that of the crust at its base. The left side is the heat released when melt joins the crust, latent heat of the melt fraction included; a melt at the base entropy releases none. The density sits inside the integral because it changes along the way from $S_b$ to $S_f$. The melt delivers $F_b$, the larger of the heat flux into the top cell from below and the flux of a boundary layer at the critical Rayleigh number `lid_ra_crit`.

## Energy

The heat content of the crust follows the equation of state across the crust, so its heat capacity and density change from the base to the surface as they do in the mesh. A crust that took them at its base would hold too much heat under a cold surface: on the PALEOS table at 10 MPa the heat capacity of the solid at 255 K is 55 % of its value at 1300 K.

The top cell keeps its melt part: its volume shrinks by the crust volume, area times $\delta$, and the heat $W(S_f)$ of each unit of that volume passes to the crust. The flux out of the melt follows from the energy of the melt and the crust together, so the energy of the mantle changes only by the heat that crosses the surface and the core boundary. The energy diagnostics count the crust, and their solver residual reports this balance for every call.

## When the crust exists, and when it goes

A hot, fast-convecting melt carries more heat than a crust at $T_b$ can radiate. No crust can then form. The lid weight $w$ \[0 to 1\] compares the thickness of a crust in steady state, $\delta_\mathrm{eq} = k (T_b - T_s) / F_b$, with `lid_delta_on`: below it the lid is hidden ($w = 0$) and the surface is the mesh surface, above it the lid acts ($w = 1$). A hidden lid relaxes to a thin phantom lid of thickness `lid_delta_lo`.

When the mantle near the surface is solid, a crust that has grown thicker than a few top cells is resolved by the mesh. The lid then hands over to the mesh. The solver stops when three conditions hold together: the base of the cold region (the deepest point where the column rises through the lid-base isotherm of the solid rheology) lies below the face of `lid_handover_cells` top cells, the stagnant-lid state of the solid rheology is active (its weight `w_active` above 0.5), and the top cell is below the rheological transition (the melt ramp $w_\phi$ of the lid base below 0.5). Why all three: in a molten or mushy mantle the solid rheology has no stagnant lid, so its isotherm marks no crust and can lie hundreds of kilometres deep. At the hand-over the solver merges the crust into the top cell with its energy, sets both lid states to zero and continues. A zero lid stays zero.

## A worked example

A melt layer at $T_b$ = 1400 K freezes under a surface held at 300 K, with $c_p$ = 1200 J/kg/K and a latent heat of 3.9 × 10⁵ J/kg, so the Stefan number $c_p (T_b - T_s) / L$ is 3.381 and $\lambda$ = 0.948. With $\kappa$ = 1.1 × 10⁻⁶ m²/s, the crust is $2 \lambda \sqrt{\kappa t}$ = 11 m thick after one year and 355 m after a thousand years. Both are far below an 11 km top cell.

## Settings and limits

- `subgrid_lid` (bool, default false): the switch. Only with the grey-body surface, `outer_boundary_condition = 1`, and not with `core_bc = 'gradient'`; both are refused at setup.
- `lid_delta_on` \[m\] (default 1): must be finite, positive and below 0.8 of the top-cell width; a larger value is refused at solver setup with the width named.
- `lid_delta_lo` \[m\] (default `lid_delta_on / 2`): must lie in (0, `lid_delta_on`).
- `lid_handover_cells` (integer ≥ 1, default 3): must leave at least one cell below the hand-over face.
- `lid_ra_crit` (default 450): positive and finite.
- The crust stops growing at 0.8 of the top-cell width (one logged warning per run) and stops thinning at `lid_delta_lo`.
- The heat jump per mass, $(W(S_f) - W(S_b)) / \rho_b$ with $\rho_b$ the density at the lid base, has a floor of 10³ J/kg (one logged warning when it acts).
- The floors cost solver steps. While the top cell freezes and thaws, the lid forms and vanishes at its phantom thickness and passes the weight floor and the lower stop many times. In a 100-node molten-start column from 3000 to 4000 yr the solver takes about 5 times the steps of the same run without the lid (46,109 against 9,366).
- The hand-over needs the solid rheology with `stress_closure_mode = 'lid'`; with another closure, or without the rheology, the lid never hands over.
- The lid runs on the NumPy and on the JAX right-hand side. A JAX factory for a run with the lid must be built with `lid_params=solver.jax_lid_params()`; a factory that disagrees with the run on the lid is refused before the solve.

**Where this is checked.** [Surface lid](../Validation/surface_lid.md) compares the lid with the freezing lava layer, a cooling half-space and a steady crust, and lists the solver tests of the energy budget and the hand-over.
