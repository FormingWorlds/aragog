# A core on the PALEOS liquid iron isentrope

How much heat does a liquid iron core give up as it cools, when its structure comes from Zalmoxis and its temperature from the PALEOS equation of state (EOS)?

The short answer: Aragog keeps the core structure that Zalmoxis computed and puts the convecting core on one isentrope of the PALEOS liquid iron table. An isentrope is a curve of constant specific entropy; for a convecting liquid it is also called the adiabat, the word the Gaussian profile uses. The heat the core gives up per kelvin of cooling at the core-mantle boundary (CMB) is then an integral of the temperature along that isentrope. Aragog tabulates it once per structure, so a run costs no more than with the Gaussian profile of [the core module](core_bc.md).

## When to use it

The core module has 2 profile sources, set with `profile_source` in `core_module_params` ([parameters](core_bc.md#parameters)):

- `"gaussian"`, the default: a closed-form density profile with a constant thermal expansivity $\alpha$ and heat capacity $c_p$.
- `"paleos"`: the density, pressure and gravity of a Zalmoxis core, and the temperature from the PALEOS liquid iron EOS of Attia et al. (2026)[^cite-attia2026].

Use `"paleos"` when Zalmoxis solves the structure with PALEOS iron. Then the core structure and the core temperature come from one EOS. The Gaussian profile takes $\alpha$ and $c_p$ of the Earth's core, which do not hold for the cores of larger planets.

## Where the structure comes from

Zalmoxis integrates the structure of the planet from the centre outward. For the core it returns, at each of its nodes, the radius $r$ (m), the enclosed mass $m$ (kg), the pressure $P$ (Pa), the density $\rho$ (kg m$^{-3}$) and the gravity $g$ (m s$^{-2}$). These arrays go into an `.npz` file, which `core_profile_file` names.

Aragog reads the arrays once and holds them fixed until the next structure solve. Between the nodes it uses cubic Hermite curves. Their slopes come from the physics where possible: $dm/dr = 4 \pi r^2 \rho$ and $dP/dr = -\rho g$. The gravitational potential $\psi$ (J kg$^{-1}$) is the integral of $g$, zero at the CMB.

When the profile is built, Aragog checks the arrays against the mantle mesh and the structure:

- The last radius must equal the CMB radius of the mesh to 1 m or a relative $10^{-9}$, whichever is larger.
- When `m_core` is given, the mass of the last node must equal it to a relative $10^{-10}$. PROTEUS passes the core mass of its structure solve.
- A CMB pressure of the arrays more than 1 % from that of the mesh is logged as a warning.

## The isentrope of the core

The liquid core convects, so it is well mixed. A well-mixed liquid has the same specific entropy $s$ (J kg$^{-1}$ K$^{-1}$) everywhere. At the CMB temperature $T_\text{cmb}$ this entropy is

$$
s_c = s(P_\text{cmb}, T_\text{cmb}).
$$

At each node $k$, the temperature $T_{a,k}$ of the isentrope solves $s(P_k, T_{a,k}) = s_c$. Aragog finds it with Newton steps on the table entropy.

The table is the liquid side of the PALEOS iron EOS, which Zalmoxis writes and `core_liquid_table` names. It holds the liquid also below the melting curve, where the liquid is metastable. So the isentrope stays liquid at all temperatures of the table, and no latent heat enters it. The inner core is treated separately (below).

Aragog solves the isentrope on a grid of CMB temperatures, 10 K apart by default (`core_t_cmb_step`). The grid covers every CMB temperature whose isentrope stays inside the table. The slopes in radius and in $T_\text{cmb}$ come from the table derivatives, and the mixed slope from a spline in $T_\text{cmb}$, so the solver reads a surface $T_a(r, T_\text{cmb})$ with continuous slopes. Aragog checks the entropy of this surface at the nodes on the grid temperatures, at the nodes between the grid temperatures, at the radial midpoints and at the centres of the grid cells. At each point it must equal $s_c$ to a relative $10^{-6}$; [What the path refuses](#what-the-path-refuses) says what happens when it does not.

A worked example: for the Zalmoxis core of a 1 Earth-mass planet in the [tutorial](../Tutorials/core_tabulated.md), a CMB at 4000 K gives a centre at 6111 K.

## The secular heat capacity

The secular term is the heat a core gives up as its whole convecting mass cools. At fixed pressure, the change of the specific enthalpy $h$ is $dh = T\,ds$. Every parcel shares the change of $s_c$. At the CMB the pressure does not change, so $ds_c = c_p\,dT_\text{cmb}/T_\text{cmb}$, with $c_p$ (J kg$^{-1}$ K$^{-1}$) the heat capacity at the CMB. The heat content $H$ of the core then changes with $T_\text{cmb}$ as

$$
C_s = \frac{dH}{dT_\text{cmb}} = \frac{ds_c}{dT_\text{cmb}} \int T_a \, dm = \frac{c_p}{T_\text{cmb}} \int T_a \, dm.
$$

For a constant $c_p$ and the Gaussian adiabat this is the secular capacity of the Gaussian profile.

By default $c_p$ is 840 J kg$^{-1}$ K$^{-1}$ (`core_cp_override`), not the table value. Attia et al. (2026), p. 5, set the heat capacity at constant volume of the PALEOS liquid to $3nR$, with $n$ atoms per formula unit and $R$ the gas constant, so it holds no electronic heat capacity. In the worked example above, the table gives $c_p$ = 506 J kg$^{-1}$ K$^{-1}$ at the CMB at 4000 K. `core_cp_override = "table"` takes the table value.

In the example, $C_s$ is $1.989\times10^{27}$ J K$^{-1}$ at 4000 K. That is 1.22 times $c_p$ times the core mass. The factor 1.22 is the mass-weighted mean of $T_a / T_\text{cmb}$.

## Why the core energy balance closes in each solver call

The core energy balance of a solver call compares 2 numbers: the heat the core loses, and the heat that crosses the CMB. Aragog checks this balance after every call.

Aragog tabulates $C_s$ at the grid temperatures, with its derivative, and reads it as a cubic in $T_\text{cmb}$. The heat change of a call is the exact integral of that same cubic over the temperature change of the call. So the solver integrates and books one function, and the balance closes to the solver tolerance.

With an inner core, the latent and gravitational heat of a call is a quadrature of their capacities over the temperature change of the call. These capacities have a kink where the inner-core boundary crosses a core node, so Aragog cuts the quadrature into pieces at freeze-out and at those temperatures.

## The inner core

The inner core starts to grow when the isentrope meets the melting curve at the centre. Aragog finds this onset temperature as the $T_\text{cmb}$ whose isentrope has its centre on the melting curve. When no isentrope of the grid meets it, the onset is the end of the grid: the lowest grid temperature when the centre stays liquid on the whole grid, the highest when the centre is frozen on the whole grid. In both cases the value is a grid end, not a nucleation temperature. The latent heat per unit mass is $T_\text{icb}\,\Delta s$, with $T_\text{icb}$ the temperature at the inner-core boundary and $\Delta s = 172.8$ J kg$^{-1}$ K$^{-1}$ (`ds_fusion`), or the constant `latent_heat` when it is set. The latent and gravitational terms of [the budget](core_bc.md) are otherwise unchanged.

The constant $\Delta s$ lies 10.5 % above the ab initio 1.05 $k_B$ per atom at 300 GPa of González-Cataldo & Militzer (2023)[^cite-gonzalezcataldo2023], with $k_B$ the Boltzmann constant.

## What the path refuses

- The core pressures must lie inside the pressure range of the table that `core_liquid_table` names. The range comes from the table itself; for the test cut in `tests/validation/data/` it is 45 GPa to 3 TPa. A core outside it stops the build with an error that names the pressures of the core and of the table; give a table that covers the core.
- The CMB temperature must lie on the grid of isentropes. The grid runs from the lowest table temperature to the CMB temperature whose isentrope reaches the highest table temperature at the centre. Outside it, or at a NaN temperature, the solver and `CoreModule` stop with an error that names the temperature.
- The grid holds at most 2,000,000 cells, nodes times temperatures. A larger grid stops with an error; make `core_t_cmb_step` larger.
- An isentrope that misses $s_c$ by more than a relative $10^{-6}$ stops the build with an error that names where: between the grid temperatures (make `core_t_cmb_step` smaller), between the radial nodes (give the core arrays more nodes), or inside the grid cells (either). The cap asks for a larger step, the check between grid temperatures for a smaller one.
- `stratification = true` is refused: the heat content of a stratified core depends on the CMB heat flow, not on $T_\text{cmb}$ alone.
- `capacity_mode = "legacy"` is refused, for the same reason.
- The dynamo diagnostics, the entropy margin and the field strength, are not computed on this path. Their entropy budget (`CoreEntropyBudget`) takes a constant $\alpha$ and $c_p$ and refuses a tabulated profile.

## Limits of the model

- The Gaussian profile keys are accepted and not used ([parameters](core_bc.md#parameters) lists them). The heat capacity of the secular term comes from `core_cp_override`.
- The liquid of the PALEOS table is calibrated between 8000 and 14,000 K and between 50 and 1300 GPa (Attia et al. (2026), p. 5). `TabulatedCoreProfiles.outside_calibration_fraction` gives the core mass fraction outside this window, and the diagnostics of `CoreModule` report it as `core_eos_outside_calibration`; the run continues.
- With the override, the isentrope shape comes from the PALEOS $c_p$ and the heat capacity from 840 J kg$^{-1}$ K$^{-1}$, so the two are not from one EOS.
- The isentrope is that of pure liquid iron, also when the density of the arrays holds light elements.
- The entropy of fusion is one constant at all pressures. The ab initio value falls from 1.05 $k_B$ per atom at 300 GPa to 0.8 $k_B$ per atom at 1000 GPa (González-Cataldo & Militzer (2023), p. 9), so the constant is 1.45 times the value at 1000 GPa.
- The interpolation of a table loses accuracy near its edges, with no warning, so a core whose pressures reach close to the top of its table is less accurate than one well inside the range. On the 3 TPa test table the Maxwell relation $(\partial s/\partial P)_T = -\alpha/\rho$ holds to $10^{-5}$ on the table nodes from 50 GPa to 2.1 TPa. It misses by up to $1.3\times10^{-5}$ between nodes below 2 TPa, by $3.6\times10^{-5}$ at the lowest pressure, 45 GPa, and by more than $10^{-4}$ above 2.38 TPa.

**Where this is checked.** [Tabulated core](../Validation/core_tabulated.md) on the Validation tab.

[^cite-attia2026]: M. Attia, T. Lichtenberg, E. Jungová, M. Sastre, *[PALEOS: Multiphase equations of state and mass-radius relations for exoplanet interiors](https://doi.org/10.1051/0004-6361/202660790)*, Astronomy & Astrophysics, 713, A225, 2026.
[^cite-gonzalezcataldo2023]: F. González-Cataldo, B. Militzer, *[Ab initio determination of iron melting at terapascal pressures and Super-Earths core crystallization](https://doi.org/10.1103/PhysRevResearch.5.033194)*, Physical Review Research, 5, 033194, 2023. [SciX](https://scixplorer.org/abs/2023PhRvR...5c3194G/abstract).
