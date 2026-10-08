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

where the secular term integrates the mass-weighted adiabat over the closed-form Gaussian core profiles, and the latent and gravitational terms follow inner-core growth through the implicit-function sensitivity of the adiabat-liquidus crossing. At nucleation onset, the latent capacity rises from zero as $(T_\text{onset} - T)^{1/2}$: the inner-core radius grows as the square root of the undercooling at the centre. At freeze-out completion, latent heat release ceases abruptly once the entire core has solidified, causing a discontinuous drop in $\tilde{C}$, by a factor of $\approx 4.46$ for the quadratic melting curve and by $\approx 55\%$ for the iron alloy curve on an Earth-like core profile (length scale 7200 km, CMB radius 3480 km, $\Delta s_\text{fusion} = 170$ J/kg/K); the size of the drop depends on the core geometry and $\Delta s_\text{fusion}$. The melting curve is the PALEOS iron prescription (Anzellini et al. 2013) with a light-element depression, or the Nimmo (2015, ch. 9.08, Eq. 6) quadratic. The piecewise Anzellini fit carries a ~0.73 K jump at the 98.5 GPa triple point, which is blended smoothly with a C^2 smootherstep over a 3.0 GPa half-width, bounding the temperature adjustment to 0.363 K and recovering exact branch values outside [95.5, 101.5] GPa. This adjustment is far below the experimental uncertainty of the measured curve. All terms are cross-validated against the Leeds `thermal_history` implementation and pinned against Nimmo (2015, ch. 9.08) Table 2; the entropy side ([`CoreEntropyBudget`](../Reference/api/aragog.core.md)) adds the dynamo criterion and field-strength scaling.

Two properties matter for coupled stability. The reported core temperature is the integrated boundary state, not the lowermost mantle node's EOS read-off, so it does not inherit the node's phase-branch snaps at crystallisation onset. The boundary solve carries a custom JVP, so reverse-mode autodiff sees the implicit-function sensitivities through inner-core growth; the CVODE analytic-Jacobian factory covers the mode through `dSdt_core_module` (the budget rides inside the JAX trace), so the production path runs the analytic Jacobian, and a solver without the factory registered falls back to a finite-difference Jacobian.

The CMB heat flux is a boundary-layer law in the core-mantle temperature contrast $\Delta T = T_\text{cmb} - T_m$, where $T_m$ is the bottom mantle cell's entropy evaluated at the CMB pressure (the cell carried adiabatically to the boundary). Following the parameterised-convection closure of Foley & Driscoll (2016, eq. 20) with the boundary-layer thickness set by a critical Rayleigh number (Thiriet et al. 2019, eqs. 13 and 14 with exponent 1/3),

$$
q_\text{conv} = \frac{k\,\Delta T}{\delta}, \qquad \delta = \left(\frac{\mathrm{Ra}_\text{crit}\, \kappa\, \eta}{\alpha \rho g\, \Delta T}\right)^{1/3},
$$

with the bottom cell's conductivity $k$, density $\rho$, heat capacity (giving $\kappa = k / \rho c_p$), expansivity $\alpha$ and viscosity $\eta$, and the CMB gravity $g$. The default $\mathrm{Ra}_\text{crit} = 450$ is the theoretical value Thiriet et al. (2019, Table 2) take for the upper boundary layer; their lower-layer law (Deschamps & Sotin 2000) is not adopted. Two closure choices are Aragog's own, not the papers': the flux never falls below conduction $k\,\Delta T / \Delta r_{1/2}$ across the half cell between the CMB and the bottom staggered node, and a core colder than the mantle above it (a stably stratified layer) gains heat by that conduction only. The flux therefore has the sign of $\Delta T$ and is continuous through zero. The same function sets the flux in the numpy and JAX right-hand sides and in every diagnostic that reads it, and the bottom cell and the core see the same flux, so the CMB exchange conserves energy.

The core temperature feeds the flux back, and the core cools at the rate $A_\text{cmb}\, q / \tilde{C}$ its heat capacity allows; the contrast itself also follows the mantle base, which can cool much faster. In the 10-node test of a 300 K contrast over a liquid base (melt fraction 0.67 to 0.65 over 4 yr), the flux is about $10^5$ W m$^{-2}$ and the core cools by about 0.3 K per year, while the base cools faster and the contrast grows to about 330 K. Over a partly crystalline base the flux is small even on the boundary-layer branch: a 1000 K contrast over a base at melt fraction 0.19 carries about 0.3 W m$^{-2}$, so the core is close to insulated. Where $\delta$ exceeds the half cell, the flux is the conduction across it, $k\,\Delta T / \Delta r_{1/2}$, which scales with the mesh spacing. The CMB entropy gradient stays in the state so the layout matches `energy_balance`, and it evolves by the same balance with $\tilde{C}(T_\text{cmb})$ in place of the reservoir factor, but no other rate, no flux and no output reads it; it enters only the integrator's error control, and `--initial-dsdr-cmb` sets no flux for this mode. The output's CMB node is the bottom cell carried to the CMB pressure: `T_basic[0]` is $T_m$, so $T_\text{cmb} - T_m$ is the contrast the flux acts on, its entropy gradient (`dSdr_b[0]`) is zero, and its flux components carry the applied flux as conduction. The default initial core temperature is $T_m$, so a run started without one has no CMB flux at first. The solver warns once when a core temperature set with `set_initial_core_temperature` differs from $T_m$ by more than 20 percent at a start. The set value applies to every later start until it is cleared with `None`; a start without a set value (the default, or a hot start from the previous solution) does not warn.

With `stratification = true` in the module parameters, the outer core above the radius `layer_base_fraction` $\times\, r_\text{cmb}$ is a resolved shell of finite volumes, finest at the CMB, whose temperatures are part of the solver state after `T_core`; the state indices up to `T_core` are those of the unstratified mode. In the shell heat moves by conduction, $\rho c_p\, \partial T / \partial t = \nabla \cdot (k \nabla T)$ (Greenwood et al. 2021, eq. 20), and, where the temperature falls outward faster than the adiabat, also by convective mixing. The mixing closure is a mixing-length flux $\rho c_p K_\text{mix} (-a)^{3/2} / \sqrt{g_\text{mix}}$, with $a = \partial (T - T_a) / \partial r$ the gradient of the departure from the convecting adiabat; it is zero for $a \ge 0$ and continuous with its derivative at $a = 0$. Its two constants are `layer_k_mix` ($K_\text{mix}$), the eddy diffusivity at the reference superadiabatic gradient, and `layer_g_mix`, that reference gradient as a fraction of the adiabatic gradient at the CMB; with the defaults a shell under a CMB flux of $10^5$ W m$^{-2}$ mixes within about 40 days. The subadiabatic part conducts only and is the stable layer. Its base, reported as `core_layer_base`, is the shell base plus the width of every cell whose lower face is not stably stratified (anomaly gradient below the reference), weighted smoothly. Onset, growth, erosion and re-formation follow from the profile without events; the scheme of Greenwood et al. (2021, eq. 25, p. 6), the `leeds_thermal` model of [thermal_history](https://github.com/sam-greenwood/thermal_history), instead moves the layer base by a stability check after each step and mixes an unstable part instantly.

The CMB flux law sees the top shell cell, so a core heated from above warms its layer first and its convecting part only when the heat reaches the layer base. The shell base sits on the convecting adiabat, and the heat crossing it is what the convecting core loses: its capacity integrals run to the shell base, the gravitational term to the layer base, and the shell carries its own sensible heat. Radiogenic heat is spread uniformly over the core mass, $h = Q_\text{radio} / M_\text{core}$: the convecting core keeps $h M(r_\text{sh})$ and each shell cell $h\, m_j$, so a layer stores the heat of its own mass.

The entropy margin is the sum over the convecting core and the shell of the balance of Greenwood et al. (2021, eq. 4): every source is delivered at the temperature of the top cell, the CMB temperature, and each part has the conduction sink of its own gradient. The CMB heat flow reaches the margin only through the state, since the top cell is the reference temperature.

The shell must hold the layer and lie above the inner core. The solver refuses an inner core that reaches the shell base and warns once when the layer base comes within three cells of it; a lower `layer_base_fraction` gives a deeper shell. A stratified core runs on CVODE or SciPy BDF, which the solver also falls back to without CVODE; Radau is refused, since the stiff mixing of the shell takes it minutes per call.

The resolved layer is experimental, and the solver warns once when `stratification = true`. Against the `leeds_thermal` model, integrated alone with SciPy BDF, it reproduces a layer under a CMB flow below the conducted adiabatic flow, with heat removed, added or neither at the CMB; when the flow rises above that value the two models erode the layer differently (aragog mixes it from the CMB down, `leeds_thermal` removes it in one step of its stability check), and when the flow falls back both form a new layer of similar depth ([verification, section 13](core_verification.md#13-stable-layer-against-thermal_history)); it meets its analytic limits and resumes where it stopped, and its NumPy and JAX right-hand sides match ([section 12](core_verification.md#12-numpy-and-jax-right-hand-sides-and-the-analytic-jacobian)). In the solver, an eroding layer stalls CVODE at the default tolerances: the step size collapses where the mixing switches on in the eroding cells. In the SciPy BDF comparison the centre temperature and the inner-core onset have not converged at a relative tolerance of 1e-8, and the comparison runs at 1e-11; the `solver.rtol` that CVODE needs for them is not measured. With a layer, the core heat of a 5 yr CVODE call that heats the core from above closes against the CMB heat to $1.9 \times 10^{-6}$; it stays at that value at an rtol of $10^{-10}$ (atol at its floor of $10^{-8}$), rises to $3.1 \times 10^{-6}$ with 1025 output points, and its source is not established. `test_a_core_heated_from_above_warms_its_top_and_closes_its_heat` holds that call to $5 \times 10^{-6}$. Without a layer the smoke tests hold $10^{-6}$ with Radau at an rtol of $10^{-6}$, with CVODE at the default rtol for a hot core above the onset, and with CVODE at an rtol of $10^{-10}$ across the inner-core onset, which closes to $1.4 \times 10^{-6}$ at the default rtol ([verification, section 9](core_verification.md#9-a-cvode-solve-across-the-inner-core-onset)).

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

The mode needs JAX (`pip install 'fwl-aragog[jax]'`); without it the solver refuses the mode when it is built. When `core_bc = "core_module"`, the module options are specified under `[boundary_conditions.core_module_params]`; any other key is refused when the solver is built:

- `fit_profile`, `m_core`, `p_cen`: the core mass [kg] and central pressure [Pa] the Gaussian profile is fitted to. With both given the profile is fitted (unless `fit_profile = false`); one of them alone, or `fit_profile = true` without them, is refused, and `fit_profile` must be true or false (1 or 0 accepted). PROTEUS fills both from its structure solve.
- `rho_cen`: Core central density [kg m$^{-3}$], used when no fit is made. Default $12500.0$. The fit does not start from it.
- `length_scale`: Core Gaussian density length scale [m], used when no fit is made. Default $7.272\times 10^6$. The fit searches it in $[r_\text{cmb}/3, 100\, r_\text{cmb}]$ and does not start from this value.
- `p_cmb`: CMB pressure [Pa] for the profile. Default: the solver's CMB pressure.
- `pressure_mode`: How the profile's pressure is computed, `"quadrature"` (default) or `"labrosse"`.
- `alpha`: Thermal expansivity [K$^{-1}$]. Default $1.35\times 10^{-5}$.
- `c_p`: Specific heat capacity [J kg$^{-1}$ K$^{-1}$]. Default $840.0$.
- `melting_curve`: Iron melting curve parameterisation, `"iron"` (PALEOS / Anzellini et al. 2013) or `"quadratic"` (Nimmo 2015, ch. 9.08, Eq. 6). Default `"iron"`; the triple-point blend of the iron curve is described above.
- `light_element_fraction`: Mole fraction of light elements depressing the iron melting curve. Default $0.0$.
- `depression`: Melting-point depression per unit mole fraction [dimensionless]. Default $0.0$.
- `t_m0`, `t_m1`, `t_m2`: Coefficients of the quadratic melting curve [K, Pa$^{-1}$, Pa$^{-2}$], all three required with `melting_curve = "quadratic"`. (PROTEUS defaults: $2677.0$, $2.95\times 10^{-12}$, $8.37\times 10^{-25}$.)
- `ds_fusion`: Entropy of fusion at the inner-core boundary [J kg$^{-1}$ K$^{-1}$]. Default $172.8$.
- `latent_heat`: Constant latent heat of fusion [J kg$^{-1}$], in place of $T_\text{icb}\,\Delta s$ from `ds_fusion` (Nimmo 2015, ch. 8.02 and 9.08, Table 2, uses 750 kJ kg$^{-1}$).
- `icn_width`: Temperature width of the inner-core nucleation diagnostic sigmoid [K]. Diagnostic only; does not affect the effective heat capacity. Default $10.0$.
- `alpha_c`: Compositional expansivity of the outer-core alloy [dimensionless]. Default $0.0$.
- `c_light`: Light-element mass fraction of the outer core [dimensionless]. Default $0.0$.
- `capacity_mode`, `legacy_rho_core`, `legacy_tfac`: `"profile"` (default) integrates the Gaussian profile; `"legacy"` uses the isothermal reservoir $\rho_\text{core} V c_p \,\mathrm{tfac}$ with the two legacy values.
- `stratification`: Boolean flag enabling the resolved stable layer (experimental, see above). Default `false`.
- `k_core`: Core thermal conductivity [W m$^{-1}$ K$^{-1}$], required with `stratification = true` (no default here; PROTEUS uses $130.0$).
- `layer_base_fraction`: Base of the resolved shell as a fraction of the CMB radius. Default $0.4$.
- `layer_cells`: Number of shell cells. Default $64$.
- `layer_top_cell`: Width of the shell cell at the CMB [m]; the cells widen geometrically downward. Default $2000$.
- `layer_k_mix`: Eddy diffusivity of convective mixing at the reference superadiabatic gradient [m$^2$ s$^{-1}$]. Default $10^7$.
- `layer_g_mix`: Reference superadiabatic gradient, as a fraction of the adiabatic gradient at the CMB. Default $10^{-3}$.
- `q_radio`: Core radiogenic power [W]. Default $0.0$.
- `ra_crit_cmb`: Critical Rayleigh number of the mantle-side CMB boundary layer [dimensionless]. Default $450.0$; positive and finite.

The dynamo field-strength options `f_ohm` and `flux_geometry` belong to PROTEUS, which builds the entropy budget itself; they are not accepted here.

### Diagnostic outputs

PROTEUS writes six diagnostics of this budget to its output helpfile on every core_module row: the stable layer thickness below the CMB, the entropy margin available to a dynamo, the rms field strength, the crystallisation regime code (0 fully liquid, 1 bottom-up, 4 fully frozen; the codes 2 top-down and 3 snow mark states the solver refuses, below), the effective heat capacity $\tilde{C}(T_\text{cmb})$ and the inner-core radius.

The budget books latent and gravitational heat only for an inner core that grows from the centre and for its freeze-out. A call whose core enters the top-down or snow regime, or freezes completely without growing from the centre, raises an error at the end of the call: the energetics of those regimes are not modelled.

In aragog `SolverOutput`, the per-call core energy change is recorded in `step_dE_core_J` [J], evaluating $\int \tilde{C} dT_\text{cmb}$ over the call, plus the shell's heat change with `stratification = true`; the shell temperatures (`core_T_shell`), the layer base and the CMB temperature (`core_T_top`) are in the output, and a resumed run passes the shell back with `set_initial_shell_temperature`.

## Energy conservation and core closure

From `SolverOutput`, the core energy closure residual of a call compares the core's heat change with the CMB heat transport:

$$
r_\text{core} = \frac{|\Delta E_\text{core} + \Delta E(F_\text{cmb}) - Q_\text{radio} \Delta t|}{\max(|\Delta E_\text{core}|, |\Delta E(F_\text{cmb})|)}
$$

where $\Delta E(F_\text{cmb}) = \int F_\text{cmb} A_\text{cmb} dt$ and $\Delta E_\text{core} = \int \tilde{C} dT_\text{core}$.

The solver integrates the core temperature, and the shell temperatures of a stratified core, as their change within the call: the state is offset by its value at the start of the call, so the solver tolerance acts on that change and not on the absolute temperature. The smoke test `test_a_hot_core_call_closes_its_heat_to_1e_minus_6` holds a 10-yr call in which the core cools by about $2 \times 10^{-6}$ K to a residual below $10^{-6}$, and `test_core_module_core_cools_through_the_boundary_layer_and_closes_its_energy` (a core cooling by about 1 K over a liquid base) holds the residual below $10^{-6}$ and the core heat change equal to the `heat_content` difference to $10^{-6}$.

The core heat change of a call, $\int \tilde{C}\, dT_\text{core}$, uses 32-point Gauss-Legendre quadrature in $T_\text{core}$ on each segment between the inner-core onset and freeze-out temperatures, with a square-root substitution across nucleation; with `stratification = true` the capacity also depends on the layer base, so the integral is a trapezoid rule over the call's output points, and the shell adds the change of its heat content. The mantle heat change (`step_dE_state_heat_J`) integrates $\rho T\, dS$ along each cell's entropy path with a trapezoid rule, in one EOS call for all points and cells. In `core_module` the rule has 512 points, since $\rho T$ has a kink at the solidus and a coarse uniform rule aliases over it on fine meshes; the other modes use 16.

## References

- Anzellini, S., Dewaele, A., Mezouar, M., Loubeyre, P., & Morard, G. (2013). Melting of iron at Earth's inner core boundary based on fast X-ray diffraction. *Science*, 340(6131), 464-466. https://doi.org/10.1126/science.1233514
- Greenwood, S., Davies, C. J., & Mound, J. E. (2021). On the evolution of thermally stratified layers at the top of Earth's core. *Physics of the Earth and Planetary Interiors*, 318, 106763. https://doi.org/10.1016/j.pepi.2021.106763
- Labrosse, S., Poirier, J.-P., & Le Mouël, J.-L. (2001). The age of the inner core. *Earth and Planetary Science Letters*, 190(3-4), 111-123. https://doi.org/10.1016/S0012-821X(01)00387-9
- Nimmo, F. (2015). Energetics of the Core. In G. Schubert (Ed.), *Treatise on Geophysics* (2nd ed., Vol. 8, pp. 27-55). Elsevier. https://doi.org/10.1016/B978-0-444-53802-4.00139-1
- Nimmo, F. (2015). Thermal and Compositional Evolution of the Core. In *Treatise on Geophysics* (2nd ed., Vol. 9, ch. 9.08, pp. 201-219). Elsevier. https://doi.org/10.1016/B978-0-444-53802-4.00160-3
