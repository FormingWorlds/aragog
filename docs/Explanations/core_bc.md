# Core boundary condition modes

The `boundary_conditions.core_bc` setting selects the formulation used at the core-mantle boundary (CMB) when `inner_boundary_condition = 1` (core cooling). Five modes are available; they differ in what is treated as the primary state variable, what is reconstructed, and how strongly the bottom mantle cell is coupled to the core.

## `quasi_steady`

State vector length: $N$ (entropy at staggered nodes only).

The CMB heat flux is computed from a quasi-steady balance using an $\alpha$-factor partition between the bottom mantle cell and the core, weighted by the heat-capacity ratio. The core temperature is reconstructed from the bottom-cell entropy and pressure via the EOS at every output time.

This mode is fast, stable, and conservative in the sense that it does not require an extra ODE state. It underestimates the true CMB heat loss relative to the SPIDER reference by 5-10 % over the first solidification cycle of a 1 M$_\oplus$ run, which is the right trade-off for quick standalone exploration where SPIDER parity is not the goal.

## `energy_balance` (default)

State vector length: $N + 1$ (entropy at staggered nodes plus the entropy gradient at the CMB basic node, $dS/dr |_\text{cmb}$).

The entropy gradient at the CMB is added as an extra state variable and is integrated via SPIDER's [`bc.c:76-131`](https://github.com/FormingWorlds/SPIDER) formula:

$$
\frac{d}{dt}\left(\frac{dS}{dr}\bigg|_\text{cmb}\right) = \frac{2}{\Delta r}\left( -F_\text{cmb} A_\text{cmb} \cdot \text{fac}_\text{cmb} - \frac{dS}{dt}\bigg|_0\right)
$$

with

$$
\text{fac}_\text{cmb} = \frac{c_p^\text{cmb}}{c_p^\text{core} \, T_\text{cmb} \, \text{tfac} \, M_\text{core}}
$$

This is the SPIDER-parity formulation. It produces bit-for-bit agreement with SPIDER on the Earth reference. Use this mode for any run that needs to reproduce SPIDER results, including the published verification suite. State vector size and Jacobian sparsity are slightly larger than `quasi_steady`, but the integrator overhead is small.

## `gradient`

State vector length: $N + 2$ (the entropy gradient is the primary field; entropy is reconstructed by cumulative integration from the surface).

Treats $dS/dr$ as the integrated quantity rather than $S$ itself, with two extra boundary states (one for each surface). Produces the same physics as `energy_balance` but with a different conditioning that is more stable in regimes where $S$ has a sharp kink at the rheological transition. Has not been validated against SPIDER as thoroughly as `energy_balance`; treat as experimental.

## `bower2018`

State vector length: $N + 1$ (with $T_\text{core}$ as the extra state).

Treats the core temperature as an ODE state variable, with the CMB heat flux computed from conduction across the bottom half-cell. The conduction-only flux underestimates the true CMB heat loss by orders of magnitude for any planet with active mantle convection; this mode is retained for parity testing only and is **not recommended for production**.

## `core_module`

State vector length: $N + 2$ (with the CMB entropy gradient $dS/dr|_\text{cmb}$ and $T_\text{cmb}$ as the extra states; the temperature is driven by the core evolution budget of [`aragog.core`](../Reference/api/aragog.core.md)).

The core carries its own physics instead of an isothermal reservoir: the CMB temperature evolves through an effective heat capacity

$$
Q_\text{cmb} = -\tilde{C}(T_\text{cmb})\, \frac{dT_\text{cmb}}{dt} + Q_\text{sources}, \qquad \tilde{C} = \tilde{C}_\text{secular} + \tilde{C}_\text{latent} + \tilde{C}_\text{grav},
$$

where the secular term integrates the mass-weighted adiabat over the closed-form Gaussian core profiles, and the latent and gravitational terms follow inner-core growth through the implicit-function sensitivity of the adiabat-liquidus crossing. At nucleation onset, the latent capacity rises from zero as $(T_\text{onset} - T)^{1/2}$: the inner-core radius grows as the square root of the undercooling at the centre. At freeze-out completion, latent heat release ceases abruptly once the entire core has solidified, causing a discontinuous drop in $\tilde{C}$, by a factor of $\approx 4.46$ for the quadratic melting curve and by $\approx 55\%$ for the iron alloy curve on an Earth-like core profile (length scale 7200 km, CMB radius 3480 km, $\Delta s_\text{fusion} = 170$ J/kg/K); the size of the drop depends on the core geometry and $\Delta s_\text{fusion}$. The melting curve is the PALEOS iron prescription (Anzellini et al. 2013) with a light-element depression, or the Nimmo (2015) quadratic. The piecewise Anzellini fit carries a ~0.73 K jump at the 98.5 GPa triple point, which is blended smoothly with a C^2 smootherstep over a 3.0 GPa half-width, bounding the temperature adjustment to 0.363 K and recovering exact branch values outside [95.5, 101.5] GPa. This adjustment is far below the experimental uncertainty of the measured curve. All terms are cross-validated against the Leeds `thermal_history` implementation and pinned against Nimmo (2015) Table 2; the entropy side ([`CoreEntropyBudget`](../Reference/api/aragog.core.md)) adds the dynamo criterion and field-strength scaling.

Two properties matter for coupled stability. The reported core temperature is the integrated boundary state, not the lowermost mantle node's EOS read-off, so it does not inherit the node's phase-branch snaps at crystallisation onset. The boundary solve carries a custom JVP, so reverse-mode autodiff sees the implicit-function sensitivities through inner-core growth; the CVODE analytic-Jacobian factory covers the mode through `dSdt_core_module` (the budget rides inside the JAX trace), so the production path runs the analytic Jacobian, and a solver without the factory registered falls back to a finite-difference Jacobian.

The CMB heat flux is a boundary-layer law in the core-mantle temperature contrast $\Delta T = T_\text{cmb} - T_m$, where $T_m$ is the bottom mantle cell's entropy evaluated at the CMB pressure (the cell carried adiabatically to the boundary). Following the parameterised-convection closure of Foley & Driscoll (2016, eq. 20) with the boundary-layer thickness set by a critical Rayleigh number (Thiriet et al. 2019, eqs. 13 and 14 with exponent 1/3),

$$
q_\text{conv} = \frac{k\,\Delta T}{\delta}, \qquad \delta = \left(\frac{\mathrm{Ra}_\text{crit}\, \kappa\, \eta}{\alpha \rho g\, \Delta T}\right)^{1/3},
$$

with the bottom cell's conductivity $k$, density $\rho$, heat capacity (giving $\kappa = k / \rho c_p$), expansivity $\alpha$ and viscosity $\eta$, and the CMB gravity $g$. The default $\mathrm{Ra}_\text{crit} = 450$ is the theoretical value Thiriet et al. (2019, Table 2) take for the upper boundary layer; their lower-layer law (Deschamps & Sotin 2000) is not adopted. Two closure choices are Aragog's own, not the papers': the flux never falls below conduction $k\,\Delta T / \Delta r_{1/2}$ across the half cell between the CMB and the bottom staggered node, and a core colder than the mantle above it (a stably stratified layer) gains heat by that conduction only. The flux therefore has the sign of $\Delta T$ and is continuous through zero. The same function sets the flux in the numpy and JAX right-hand sides and in every diagnostic that reads it, and the bottom cell and the core see the same flux, so the CMB exchange conserves energy.

The core temperature feeds the flux back, and the core cools at the rate $A_\text{cmb}\, q / \tilde{C}$ its heat capacity allows; the contrast itself also follows the mantle base, which can cool much faster. In the 10-node test of a 300 K contrast over a liquid base (melt fraction 0.67 to 0.65 over 4 yr), the flux is about $10^5$ W m$^{-2}$ and the core cools by about 0.3 K per year, while the base cools faster and the contrast grows to about 330 K. Over a partly crystalline base the flux is small even on the boundary-layer branch: a 1000 K contrast over a base at melt fraction 0.19 carries about 0.3 W m$^{-2}$, so the core is close to insulated. Where $\delta$ exceeds the half cell, the flux is the conduction across it, $k\,\Delta T / \Delta r_{1/2}$, which scales with the mesh spacing. The CMB entropy gradient stays in the state so the layout matches `energy_balance`, and it evolves by the same balance with $\tilde{C}(T_\text{cmb})$ in place of the reservoir factor, but no other rate, no flux and no output reads it: it has no effect on the run, and `--initial-dsdr-cmb` sets nothing for this mode. The output's CMB node is the bottom cell carried to the CMB pressure: `T_basic[0]` is $T_m$, so $T_\text{cmb} - T_m$ is the contrast the flux acts on, its entropy gradient (`dSdr_b[0]`) is zero, and its flux components carry the applied flux as conduction. The default initial core temperature is $T_m$, so a run started without one has no CMB flux at first. The solver warns once when a supplied core temperature differs from $T_m$ by more than 20 percent; a hot start from the previous solution does not warn.

With `stratification = true` in the module parameters, a stably stratified layer forms under the CMB whenever the heat flow drops below what conduction carries along the adiabat, and the convecting volume in the energy and entropy budgets shrinks to the layer base: the capacity integrals, the entropy sources, and the conduction sink all run over the convecting region, so a subadiabatic core cools faster and its dynamo margin reflects the smaller convecting shell. The layer base sits at the radius $r_s$ where the adiabatic conducted flow $Q_\text{ad}(r) = 4\pi r^2 k |dT_a/dr|$ equals the CMB heat flow. This is the steady state of a conducting layer whose base takes the adiabatic gradient, the lower boundary condition of the Leeds `thermal_history` stable-layer model (`leeds_thermal` in [sam-greenwood/thermal_history](https://github.com/sam-greenwood/thermal_history), version ea9aa99), which instead resolves the layer's diffusion in time and starts a layer when the ratio of the CMB gradient to the adiabatic one falls below one. The closure is quasi-static: the layer conducts the CMB heat flow without storing it, and its heat content is excluded from the budget, an error of order the layer's volume fraction of the secular term, so thin layers are the regime the closure serves. The depth solve carries an implicit-function sensitivity so the analytic-Jacobian path sees the layer move with the state. A layer reaching below the inner-core boundary closes the gravitational term to zero (no convecting shell remains to mix light elements).

Radiogenic heat $Q_\text{radio}$ is spread uniformly over the core mass, $h = Q_\text{radio} / M_\text{core}$. The temperature equation gives all of it to the convecting region, which is exact under the quasi-static layer: the layer's own heat $h M_L$ leaves through the CMB in the same step, so the convecting region's balance is $\tilde{C}\, dT/dt = h M_\text{conv} - (Q_\text{cmb} - h M_L) = Q_\text{radio} - Q_\text{cmb}$. The entropy budget counts $h$ over the convecting volume only, since the layer's heat is conducted out and drives no convection. The depth, however, is matched to $Q_\text{cmb}$, while the flow through the layer base is $Q_\text{cmb} - h M_L(r_s)$, so with $Q_\text{radio} > 0$ the layer comes out thinner than the quasi-static model implies; the error in the matched flow is $(Q_\text{radio} / Q_\text{cmb})(M_L / M_\text{core})$.

The convecting radius has a floor at 10 percent of the CMB radius, which the layer base reaches when the CMB heat flow is at most $Q_\text{ad}(0.1\, r_\text{cmb})$, every zero or negative flow included. On the floor the capacity is that of the inner tenth of the radius, about 700 times below the full core's on the test profile ($k = 130$ W m$^{-1}$ K$^{-1}$). The quasi-static model does not hold there: a core that loses no heat through the CMB does not convect, and heat entering from the mantle goes into a conducting top layer that thickens with time, which `leeds_thermal` resolves and this closure does not. An order-of-magnitude estimate from the diffusion length $\sqrt{\kappa t}$, not a published result, with $\kappa = k / \rho c_p \approx 1.4\times 10^{-5}$ m$^2$ s$^{-1}$: such a layer holds the floor's capacity after about $10^4$ yr (about 2 km thick) and 10 times more after 1 Myr, so on the floor the CMB temperature of a heated core responds too slowly at first and too fast later, while the full-core capacity would hold it almost fixed. The solver warns once when the core is on the floor with $|Q_\text{cmb}|$ above $10^{-3} Q_k$, $Q_k = Q_\text{ad}(r_\text{cmb})$. The default start does no harm: the core starts at the mantle side of the CMB, so the flow starts at zero, and in the 5 yr test over a mushy base it stays far below $10^{-3} Q_k$ and the core temperature does not change.

## How to choose

| Need | Recommended `core_bc` |
|------|----------------------|
| Production PROTEUS runs and SPIDER-parity validation | `energy_balance` (default) |
| Quick standalone exploration where SPIDER parity is not required | `quasi_steady` |
| Very steep mushy-band gradient that destabilises `energy_balance` | `gradient` (experimental) |
| Reproducing pre-2026 results | `bower2018` (legacy) |
| Core evolution with inner-core growth, or a core temperature decoupled from basal-node phase snaps | `core_module` |

The state-vector layout for each mode is documented in [`solver/entropy_solver.py`](https://github.com/FormingWorlds/aragog/blob/main/src/aragog/solver/entropy_solver.py) at the `_build_jac_sparsity` and `set_initial_entropy` methods; the test class `TestEnergyBalanceCoreBC` in `tests/test_entropy_pytest.py` exercises the `energy_balance` mode directly.

## Configuration parameters and diagnostics for `core_module`

### Parameters

When `core_bc = "core_module"`, the module options are specified under `[boundary_conditions.core_module_params]`; any other key is refused when the solver is built:

- `fit_profile`, `m_core`, `p_cen`: the core mass [kg] and central pressure [Pa] the Gaussian profile is fitted to. With both given the profile is fitted (unless `fit_profile = false`); `fit_profile = true` with one of them missing is refused. PROTEUS fills both from its structure solve.
- `rho_cen`: Core central density [kg m$^{-3}$], used when no fit is made. Default $12500.0$. The fit does not start from it.
- `length_scale`: Core Gaussian density length scale [m], used when no fit is made. Default $7.272\times 10^6$. The fit brackets it on a fixed range.
- `p_cmb`: CMB pressure [Pa] for the profile. Default: the solver's CMB pressure.
- `pressure_mode`: How the profile's pressure is computed, `"quadrature"` (default) or `"labrosse"`.
- `alpha`: Thermal expansivity [K$^{-1}$]. Default $1.35\times 10^{-5}$.
- `c_p`: Specific heat capacity [J kg$^{-1}$ K$^{-1}$]. Default $840.0$.
- `melting_curve`: Iron melting curve parameterisation, `"iron"` (PALEOS / Anzellini et al. 2013) or `"quadratic"` (Nimmo 2015 polynomial). Default `"iron"`; the triple-point blend of the iron curve is described above.
- `light_element_fraction`: Mole fraction of light elements depressing the iron melting curve. Default $0.0$.
- `depression`: Melting-point depression per unit mole fraction [dimensionless]. Default $0.0$.
- `t_m0`, `t_m1`, `t_m2`: Coefficients of the quadratic melting curve [K, Pa$^{-1}$, Pa$^{-2}$], all three required with `melting_curve = "quadratic"`. (PROTEUS defaults: $2677.0$, $2.95\times 10^{-12}$, $8.37\times 10^{-25}$.)
- `ds_fusion`: Entropy of fusion at the inner-core boundary [J kg$^{-1}$ K$^{-1}$]. Default $172.8$.
- `latent_heat`: Constant latent heat of fusion [J kg$^{-1}$], in place of $T_\text{icb}\,\Delta s$ from `ds_fusion` (Nimmo 2015 Table 2 uses 750 kJ kg$^{-1}$).
- `icn_width`: Temperature width of the inner-core nucleation diagnostic sigmoid [K]. Diagnostic only; does not affect the effective heat capacity. Default $10.0$.
- `alpha_c`: Compositional expansivity of the outer-core alloy [dimensionless]. Default $0.0$.
- `c_light`: Light-element mass fraction of the outer core [dimensionless]. Default $0.0$.
- `capacity_mode`, `legacy_rho_core`, `legacy_tfac`: `"profile"` (default) integrates the Gaussian profile; `"legacy"` uses the isothermal reservoir $\rho_\text{core} V c_p \,\mathrm{tfac}$ with the two legacy values.
- `stratification`: Boolean flag enabling stable layer tracking under subadiabatic conditions. Default `false`.
- `k_core`: Core thermal conductivity [W m$^{-1}$ K$^{-1}$], required with `stratification = true` (no default here; PROTEUS uses $130.0$).
- `q_radio`: Core radiogenic power [W]. Default $0.0$.
- `ra_crit_cmb`: Critical Rayleigh number of the mantle-side CMB boundary layer [dimensionless]. Default $450.0$; positive and finite.

The dynamo field-strength options `f_ohm` and `flux_geometry` belong to PROTEUS, which builds the entropy budget itself; they are not accepted here.

### Diagnostic outputs

PROTEUS writes six diagnostics of this budget to its output helpfile on every core_module row: the stable layer thickness below the CMB, the entropy margin available to a dynamo, the rms field strength, the crystallisation regime code (0 fully liquid, 1 bottom-up, 2 top-down, 3 snow, 4 fully frozen), the effective heat capacity $\tilde{C}(T_\text{cmb})$ and the inner-core radius (see the PROTEUS output reference).

In aragog `SolverOutput`, the per-call core energy change is recorded in `step_dE_core_J` [J], evaluating $\int \tilde{C} dT_\text{cmb}$ over the call.

## Energy conservation and core closure

From `SolverOutput`, the core energy closure residual of a call compares the core's heat change with the CMB heat transport:

$$
r_\text{core} = \frac{|\Delta E_\text{core} + \Delta E(F_\text{cmb}) - Q_\text{radio} \Delta t|}{\max(|\Delta E_\text{core}|, |\Delta E(F_\text{cmb})|)}
$$

where $\Delta E(F_\text{cmb}) = \int F_\text{cmb} A_\text{cmb} dt$ and $\Delta E_\text{core} = \int \tilde{C} dT_\text{core}$.

A relative tolerance rtol on the core temperature allows an integration error of about $\text{rtol} \times T_\text{core}$ per step (0.058 mK at $\text{rtol} = 10^{-8}$ and $T_\text{core} \approx 5800$ K), so the residual is meaningful only where the core's temperature change over the call is much larger. Core energy closure is therefore a meaningful check only where

$$
|\Delta T_\text{core}| \ge 10^3 \times \text{rtol} \times T_\text{core},
$$

58 mK per call for the values above; a strongly stratified core with a CMB flux near $10^{-3}$ W m$^{-2}$ changes temperature far more slowly and falls outside it. The smoke test `test_core_module_core_cools_through_the_boundary_layer_and_closes_its_energy` (a core cooling by about 1 K over a liquid base) holds the residual below $10^{-5}$ and the core heat change equal to the `heat_content` difference to $10^{-6}$.

The core heat change of a call, $\int \tilde{C}\, dT_\text{core}$, uses 32-point Gauss-Legendre quadrature in $T_\text{core}$ on each segment between the inner-core onset and freeze-out temperatures, with a square-root substitution across nucleation; with `stratification = true` the capacity also depends on the CMB heat flow, so the integral is a trapezoid rule over the call's output points. The mantle heat change (`step_dE_state_heat_J`) integrates $\rho T\, dS$ along each cell's entropy path with a 512-point trapezoid rule, in one EOS call for all points and cells; the rule must be fine because $\rho T$ has a kink at the solidus, and a coarse uniform rule aliases over it on fine meshes.

## References

- Anzellini, S., Dewaele, A., Mezouar, M., Loubeyre, P., & Morard, G. (2013). Melting of iron at Earth's inner core boundary based on fast X-ray diffraction. *Science*, 340(6131), 464-466. https://doi.org/10.1126/science.1233514
- Labrosse, S., Poirier, J.-P., & Le Mouël, J.-L. (2001). The age of the inner core. *Earth and Planetary Science Letters*, 190(3-4), 111-123. https://doi.org/10.1016/S0012-821X(01)00387-9
- Nimmo, F. (2015). Energetics of the Core. In G. Schubert (Ed.), *Treatise on Geophysics* (2nd ed., Vol. 9, pp. 27-55). Elsevier. https://doi.org/10.1016/B978-0-444-53802-4.00139-1
