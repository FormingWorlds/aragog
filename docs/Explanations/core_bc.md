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

where the secular term integrates the mass-weighted adiabat over the closed-form Gaussian core profiles, and the latent and gravitational terms follow inner-core growth through the implicit-function sensitivity of the adiabat-liquidus crossing. At nucleation onset, the latent capacity develops an inverse-square-root singularity $(T_\text{onset} - T)^{-1/2}$ from the central geometry. At freeze-out completion, latent heat release ceases abruptly once the entire core has solidified, causing a discontinuous drop in $\tilde{C}$, by a factor of $\approx 4.46$ for the quadratic melting curve and by $\approx 55\%$ for the iron alloy curve on an Earth-like core profile (length scale 7200 km, CMB radius 3480 km, $\Delta s_\text{fusion} = 170$ J/kg/K); the size of the drop depends on the core geometry and $\Delta s_\text{fusion}$. The melting curve is the PALEOS iron prescription (Anzellini et al. 2013) with a light-element depression, or the Nimmo (2015) quadratic. The piecewise Anzellini fit carries a ~0.73 K jump at the 98.5 GPa triple point, which is blended smoothly with a C^2 smootherstep over a 3.0 GPa half-width, bounding the temperature adjustment to 0.363 K and recovering exact branch values outside [95.5, 101.5] GPa. This adjustment is negligible compared to the experimental uncertainty of Anzellini et al. (2013) (+/- 100 K from Fig. 2 scatter, and 200 K melt-detection uncertainty). All terms are cross-validated against the Leeds `thermal_history` implementation and pinned against Nimmo (2015) Table 2; the entropy side ([`CoreEntropyBudget`](../Reference/api/aragog.core.md)) adds the dynamo criterion and field-strength scaling.

Two properties matter for coupled stability. The reported core temperature is the integrated boundary state, not the lowermost mantle node's EOS read-off, so it does not inherit the node's phase-branch snaps at crystallisation onset. The boundary solve carries a custom JVP, so reverse-mode autodiff sees the implicit-function sensitivities through inner-core growth; the CVODE analytic-Jacobian factory covers the mode through `dSdt_core_module` (the budget rides inside the JAX trace), so the production path runs the analytic Jacobian, and a solver without the factory registered falls back to a finite-difference Jacobian.

The CMB heat flux is a boundary-layer law in the core-mantle temperature contrast $\Delta T = T_\text{cmb} - T_m$, where $T_m$ is the bottom mantle cell's entropy evaluated at the CMB pressure (the cell carried adiabatically to the boundary). Following the parameterised-convection closure of Foley & Driscoll (2016, eq. 20) with the boundary-layer thickness set by a critical Rayleigh number (Thiriet et al. 2019, eqs. 13 and 14 with exponent 1/3),

$$
q_\text{conv} = \frac{k\,\Delta T}{\delta}, \qquad \delta = \left(\frac{\mathrm{Ra}_\text{crit}\, \kappa\, \eta}{\alpha \rho g\, \Delta T}\right)^{1/3},
$$

with the bottom cell's conductivity $k$, density $\rho$, heat capacity (giving $\kappa = k / \rho c_p$), expansivity $\alpha$ and viscosity $\eta$, and the CMB gravity $g$. The default $\mathrm{Ra}_\text{crit} = 450$ is the theoretical value Thiriet et al. (2019, Table 2) take for the upper boundary layer; their lower-layer law (Deschamps & Sotin 2000) is not adopted. Two closure choices are Aragog's own, not the papers': the flux never falls below conduction $k\,\Delta T / \Delta r_{1/2}$ across the half cell between the CMB and the bottom staggered node, and a core colder than the mantle above it (a stably stratified layer) gains heat by that conduction only. The flux therefore has the sign of $\Delta T$ and is continuous through zero. The same function sets the flux in the numpy and JAX right-hand sides and in every diagnostic that reads it, and the bottom cell and the core see the same flux, so the CMB exchange conserves energy.

The core temperature feeds the flux back, so an offset between the core and the basal mantle relaxes at the rate the boundary layer allows: within years under a molten base (viscosity of order 0.1 Pa s), but extremely slowly once the base passes the rheological transition, where $\delta$ exceeds tens of kilometres and the core is close to insulated. The boundary entropy gradient remains a state and rides on the core cooling rate through the same balance as `energy_balance` with $\tilde{C}(T_\text{cmb})$ in place of the reservoir factor; it no longer sets the flux and only defines the reported mantle-side node temperature. The solver warns at initialisation when the supplied core temperature differs from the basal-node EOS value by more than 20 percent.

With `stratification = true` in the module parameters, a stably stratified layer forms under the CMB whenever the heat flow drops below what conduction carries along the adiabat. The layer sits at its equilibrium conductive-matching depth (the radius where the adiabatic conducted flow equals the CMB heat flow, the criterion of the Leeds `thermal_history` stable-layer model), and the convecting volume in the energy and entropy budgets shrinks to the layer base: the capacity integrals, the entropy sources, and the conduction sink all run over the convecting region, so a subadiabatic core cools faster and its dynamo margin reflects the smaller convecting shell. The closure is quasi-static: the layer conducts the CMB heat flow without storing it, its own thermal evolution is not resolved, and its heat content is excluded from the budget, an error of order the layer's volume fraction of the secular term, so thin layers are the regime the closure serves. The depth solve carries an implicit-function sensitivity so the analytic-Jacobian path sees the layer move with the state. The convecting radius is floored at 10 percent of the CMB radius so a fully stratified transient cannot collapse the budget volume to zero; at that floor the effective thermal inertia sits orders of magnitude below the full core's, which keeps the ODE finite but is not a physical regime, and results there should not be interpreted. A layer reaching below the inner-core boundary closes the gravitational term gracefully to zero (no convecting shell remains to mix light elements).

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

When `core_bc = "core_module"`, the module options are specified under `[boundary_conditions.core_module_params]`:

- `fit_profile`: Boolean flag indicating whether core density profiles are fitted to interior structure constraints ($M_\text{core}$ and $P_\text{center}$). Default `true` in PROTEUS coupled runs.
- `m_core`: Core mass [kg] used when `fit_profile = true`. Automatically extracted from interior structure (`M_core`).
- `p_cen`: Core central pressure [Pa] used when `fit_profile = true`. Automatically extracted from interior structure (`P_center`).
- `rho_cen`: Core central density [kg m$^{-3}$]. Default $12500.0$. Used as initial guess or when `fit_profile = false`. When structure constraints ($M_\text{core}$ and $P_\text{center}$) are supplied, central density and length scale are fitted to the interior structure constraints; at equal Zalmoxis Earth geometry, the fit increases $C_\text{eff}$ by $+5.3\%$ and shifts onset temperature by $+53\text{ K}$ ($3901.1\text{ K} \to 3954.4\text{ K}$), whereas the change from old default Earth geometry ($3480\text{ km}, 136\text{ GPa}, T_\text{onset}=4119.0\text{ K}$) causes a $-76\%$ shift.
- `length_scale`: Core Gaussian density length scale [m]. Default $7.272\times 10^6$. Used as initial guess or when `fit_profile = false`. Fitted jointly with `rho_cen` when structure constraints are supplied.
- `alpha`: Thermal expansivity [K$^{-1}$]. Default $1.35\times 10^{-5}$.
- `c_p`: Specific heat capacity [J kg$^{-1}$ K$^{-1}$]. Default $840.0$.
- `melting_curve`: Iron melting curve parameterisation (`"iron"` for PALEOS / Anzellini et al. 2013 or `"quadratic"` for Nimmo 2015 polynomial). Choices: `"iron"`, `"quadratic"`. Default `"iron"`. The iron curve incorporates a C^2 smootherstep transition around the 98.5 GPa triple point over a 3.0 GPa half-width, removing the published 0.73 K step discontinuity while bounding deviation to 0.363 K, well within the experimental uncertainty (+/- 100 K scatter, 200 K melt-detection threshold; Anzellini et al. 2013).
- `light_element_fraction`: Initial mole fraction of light elements depressing the iron melting curve. Default $0.0$.
- `depression`: Melting-point depression per unit mole fraction [dimensionless]. Default $0.0$.
- `t_m0`: Polynomial prefactor for the quadratic melting curve [K]. Default $2677.0$ in PROTEUS configuration; required parameter in aragog constructor.
- `t_m1`: Linear coefficient for the quadratic melting curve [Pa$^{-1}$]. Default $2.95\times 10^{-12}$ in PROTEUS configuration; required parameter in aragog constructor.
- `t_m2`: Quadratic coefficient for the quadratic melting curve [Pa$^{-2}$]. Default $8.37\times 10^{-25}$ in PROTEUS configuration; required parameter in aragog constructor.
- `ds_fusion`: Entropy of fusion at the inner-core boundary [J kg$^{-1}$ K$^{-1}$]. Default $172.8$.
- `icn_width`: Temperature width of the inner-core nucleation diagnostic sigmoid [K]. Diagnostic only; does not affect the effective heat capacity. Default $10.0$.
- `alpha_c`: Compositional expansivity of the outer-core alloy [dimensionless]. Default $0.0$.
- `c_light`: Light-element mass fraction of the outer core [dimensionless]. Default $0.0$.
- `q_radio`: Core radiogenic power [W]. Default $0.0$.
- `ra_crit_cmb`: Critical Rayleigh number of the mantle-side CMB boundary layer [dimensionless]. Default $450.0$; positive and finite.
- `stratification`: Boolean flag enabling stable layer tracking under subadiabatic conditions. Default `false`.
- `k_core`: Core thermal conductivity [W m$^{-1}$ K$^{-1}$]. Default $130.0$.
- `f_ohm`: Ohmic dissipation fraction for dynamo dissipation. Default $1.0$.

### Diagnostic outputs

In coupled simulations, the core module exports six physical diagnostics to the output helpfile:

1. `core_strat_depth`: Stable layer thickness beneath the CMB [m]. Evaluates to $0.0$ when stratification is disabled or when the core is superadiabatic.
2. `core_dynamo_margin`: Net entropy production rate available to drive a magnetic dynamo [W K$^{-1}$]. Positive values indicate active dynamo generation.
3. `core_B_rms`: Root-mean-square magnetic field strength averaged over the core volume [T], estimated from convective buoyancy power.
4. `core_regime`: Physical core regime integer code: 0 = fully liquid, 1 = bottom-up, 2 = top-down, 3 = snow, 4 = fully frozen.
5. `core_C_eff`: Total effective heat capacity $\tilde{C}(T_\text{cmb})$ [J K$^{-1}$], including secular, latent, and gravitational components.
6. `core_r_icb`: Radius of the solid inner core boundary [m]. Evaluates to $0.0$ before nucleation.

In aragog `SolverOutput`, the per-call core energy change is recorded in `step_dE_core_J` [J], evaluating $\int \tilde{C} dT_\text{cmb}$ over the call.

## Energy conservation and core closure

The core energy closure residual evaluates energy conservation between core internal heat change and CMB heat transport:

$$
r_\text{core} = \frac{|\Delta E_\text{core} + \Delta E(F_\text{cmb}) - Q_\text{radio} \Delta t|}{\max(|\Delta E_\text{core}|, |\Delta E(F_\text{cmb})|)}
$$

where $\Delta E(F_\text{cmb}) = \int F_\text{cmb} A_\text{cmb} dt$ and $\Delta E_\text{core} = \int \tilde{C} dT_\text{core}$.

In strongly stratified regimes with subadiabatic heat flow ($F_\text{cmb} \sim 10^{-3}\text{ W m}^{-2}$), core cooling during an integration call is on the order of millikelvins. Relative error control on the absolute core temperature ($T_\text{core} \approx 5800\text{ K}$) admits a numerical truncation noise floor of $\text{rtol} \times T_\text{core} \approx 0.058\text{ mK}$ per step at $\text{rtol} = 10^{-8}$. In a multi-step call, accumulated truncation noise can reach tens of millikelvins, exceeding the physical cooling signal.

Therefore, core energy closure verification applies to physical regimes where core cooling per integration window satisfies:

$$
|\Delta T_\text{core}| \ge 10^3 \times \text{rtol} \times T_\text{core}
$$

The factor $10^3$ is a derived margin requiring physical cooling to exceed the local truncation noise floor by three orders of magnitude; this requirement ensures that spatial discretization and quadrature errors dominate over numerical truncation noise. For standard tolerances ($\text{rtol} = 10^{-8}$) and core temperatures ($T_\text{core} \approx 5800\text{ K}$), this corresponds to $|\Delta T_\text{core}| \ge 58\text{ mK}$ per call window. Convective regimes with active heat extraction and inner-core growth exhibit cooling rates of $|\Delta T_\text{core}| \approx 8\text{ to } 15\text{ K}$ per window, exceeding the derived margin by more than two orders of magnitude. In convective regimes at standard tolerance ($\text{rtol} = 10^{-8}$), spatial refinement reduces the residual to a numerical floor of $1.3\times 10^{-6}$ to $1.8\times 10^{-6}$; during inner core growth with active latent heat release, $r_\text{core}$ reaches $1.8\times 10^{-7}$. At tighter integrator tolerances ($\text{rtol} \le 10^{-10}$), $r_\text{core}$ drops to $8.0\times 10^{-8}$ to $9.1\times 10^{-8}$, confirming that the residual scales directly with integrator tolerance.

Diagnostic core energy integration uses 32-point Gauss-Legendre quadrature in $T_\text{core}$. For whole-planet energy closure, mantle state-heat quadrature aliases over the solidus kink below $n_\text{quad} = 256$, producing non-monotonic residuals ($1.64\times 10^{-6}$ at $n_\text{quad} = 64$ and $5.68\times 10^{-6}$ at $n_\text{quad} = 128$ at 1025 output points). Evaluating mantle state-heat quadrature with $n_\text{quad} = 512$ provides margin above the aliasing boundary, reducing $r_\text{planet}$ below $2.2\times 10^{-7} \le 1.0\times 10^{-6}$ in all physical cases.

## References

- Anzellini, S., Dewaele, A., Mezouar, M., Loubeyre, P., & Morard, G. (2013). Melting of iron at Earth's inner core boundary based on fast X-ray diffraction. *Science*, 340(6131), 464-466. https://doi.org/10.1126/science.1233514
- Labrosse, S., Poirier, J.-P., & Le Mouël, J.-L. (2001). The age of the inner core. *Earth and Planetary Science Letters*, 190(3-4), 111-123. https://doi.org/10.1016/S0012-821X(01)00387-9
- Nimmo, F. (2015). Energetics of the Core. In G. Schubert (Ed.), *Treatise on Geophysics* (2nd ed., Vol. 9, pp. 27-55). Elsevier. https://doi.org/10.1016/B978-0-444-53802-4.00139-1
