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

State vector length: $N + 2$ (with the CMB entropy gradient $dS/dr|_\text{cmb}$ and $T_\text{cmb}$ as the extra states; the temperature is driven by the staged core-evolution budget of [`aragog.core`](../Reference/api/aragog.core.md)).

The core carries its own physics instead of an isothermal reservoir: the CMB temperature evolves through an effective heat capacity

$$
Q_\text{cmb} = -\tilde{C}(T_\text{cmb})\, \frac{dT_\text{cmb}}{dt} + Q_\text{sources}, \qquad \tilde{C} = \tilde{C}_\text{secular} + \tilde{C}_\text{latent} + \tilde{C}_\text{grav},
$$

where the secular term integrates the mass-weighted adiabat over the closed-form Gaussian core profiles, and the latent and gravitational terms follow inner-core growth through the implicit-function sensitivity of the adiabat-liquidus crossing, activated smoothly at nucleation onset and wound down smoothly at freeze-out completion. The melting curve is the PALEOS iron prescription (Anzellini et al. 2013) with a light-element depression, or the Nimmo (2015) quadratic. All terms are cross-validated against the Leeds `thermal_history` implementation and pinned against Nimmo (2015) Table 2; the entropy side ([`CoreEntropyBudget`](../Reference/api/aragog.core.md)) adds the dynamo criterion and field-strength scaling.

Two properties matter for coupled stability. The reported core temperature is the integrated boundary state, not the lowermost mantle node's EOS read-off, so it does not inherit the node's phase-branch snaps at crystallisation onset. And the budget is smooth with correct derivatives (the boundary solve carries a custom JVP), so anything differentiating through the core state sees the true sensitivities; the CVODE analytic-Jacobian factory covers the mode through `dSdt_core_module` (the budget rides inside the JAX trace), so the production path runs the analytic Jacobian, and a solver without the factory registered falls back to a finite-difference Jacobian that the smooth budget keeps well-conditioned.

The CMB heat flux is the state-derived physical flux: the boundary entropy gradient is its own ODE state, exactly as in `energy_balance`, and the full conductive-plus-convective flux assembly is evaluated from it at the CMB basic node. The boundary-gradient equation is the same SPIDER balance with the isothermal-reservoir factor replaced by $\tilde{C}(T_\text{cmb})$, so the basal mantle boundary can only change entropy as fast as the core's true thermal inertia allows. With the budget in legacy capacity mode (the reservoir constants), the mantle trajectory reproduces `energy_balance` to solver tolerance, which is the mode's regression anchor; the core-side energy booking closes against $\tilde{C}\,\Delta T_\text{cmb}$, which is the guard against a decoupled boundary.

The coupling is rate-continuous, not value-continuous: the flux is set by the mantle-side transport at the boundary, so cooling is mantle-limited and the flux carries no dependence on $T_\text{cmb}$ itself, exactly as in the SPIDER balance it generalises. An initial offset between the core state and the basal cell's EOS temperature therefore persists through the run rather than relaxing, and since the melting curve is evaluated at the core state, the offset shifts inner-core nucleation timing by the corresponding interval. The solver warns at initialisation when the supplied core temperature differs from the basal-cell EOS value by more than 20 percent; keep the two consistent unless the offset is an intended model choice (for example a superheated core after a giant impact).

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

- `rho_cen`: Core central density [kg m$^{-3}$]. Default $12500.0$.
- `length_scale`: Core Gaussian density length scale [m]. Default $7.272\times 10^6$.
- `alpha`: Thermal expansivity [K$^{-1}$]. Default $1.35\times 10^{-5}$.
- `c_p`: Specific heat capacity [J kg$^{-1}$ K$^{-1}$]. Default $840.0$.
- `melting_curve`: Iron melting curve parameterisation (`"iron"` for PALEOS/Anzellini et al. 2013 or `"nimmo"` for Nimmo 2015 quadratic).
- `light_element_fraction`: Initial mass fraction of light elements (e.g. sulfur, silicon, oxygen). Default $0.1$.
- `depression`: Liquidus depression per light element fraction [K]. Default $1.2$.
- `t_m0`, `t_m1`, `t_m2`: Melting temperature polynomial coefficients.
- `ds_fusion`: Entropy of fusion [J kg$^{-1}$ K$^{-1}$]. Default $172.8$.
- `icn_width`: Inner core nucleation smoothing width [K]. Default $10.0$.
- `alpha_c`: Compositional expansivity coefficient. Default $1.0$.
- `c_light`: Light element concentration partitioning coefficient. Default $0.046$.
- `q_radio`: Core radiogenic heat production rate [W kg$^{-1}$]. Default $0.0$.
- `stratification`: Boolean flag enabling stable layer tracking under subadiabatic conditions. Default `false`.
- `k_core`: Core thermal conductivity [W m$^{-1}$ K$^{-1}$]. Default $130.0$.
- `f_ohm`: Ohmic dissipation fraction for dynamo dissipation. Default $1.0$.

### Diagnostic outputs

In coupled simulations, the core module exports six physical diagnostics to the output helpfile:

1. `core_strat_depth`: Stable layer thickness beneath the CMB [m]. Evaluates to $0.0$ when the core is superadiabatic.
2. `core_dynamo_margin`: Net power available to drive a magnetic dynamo [W]. Positive values indicate active dynamo generation.
3. `core_B_rms`: Root-mean-square magnetic field strength at the CMB [T] estimated from convective buoyancy power.
4. `core_regime`: Physical core regime label (`"fully_convective"`, `"stratified"`, `"crystallising"`).
5. `core_C_eff`: Total effective heat capacity $\tilde{C}(T_\text{cmb})$ [J K$^{-1}$], including secular, latent, and gravitational components.
6. `core_r_icb`: Radius of the solid inner core boundary [m]. Evaluates to $0.0$ before nucleation.

In aragog `StepResult`, the per-call core energy change is recorded in `step_dE_core_J` [J], evaluating $\int \tilde{C} dT_\text{cmb}$ across the call.

## Energy conservation and core closure (E1)

The core energy closure residual evaluates energy conservation between core internal heat change and CMB heat transport:

$$
r_\text{core} = \frac{|\Delta E_\text{core} + \Delta E(F_\text{cmb}) - Q_\text{radio} \Delta t|}{\max(|\Delta E_\text{core}|, |\Delta E(F_\text{cmb})|)}
$$

where $\Delta E(F_\text{cmb}) = \int F_\text{cmb} A_\text{cmb} dt$ and $\Delta E_\text{core} = \int \tilde{C} dT_\text{core}$.

In strongly stratified regimes (subadiabatic heat flow where $F_\text{cmb} \sim 10^{-3}\text{ W m}^{-2}$ and core cooling is on the order of millikelvins), the closure metric $r_\text{core}$ is not testable on the standard aragog solver branch. Under `energy_balance` and `core_module`, the CMB basic node copies eddy diffusivity from the interior cell ($\kappa_h[0] = \kappa_h[1] \sim 10^7\text{ m}^2\text{ s}^{-1}$), enforcing an infinite-conductance restoring boundary condition with a stiff coupling eigenvalue $\lambda \approx -0.1\text{ s}^{-1}$ (a 10-second physical timescale).

Because CVODE applies relative error control to the absolute core temperature state ($T_\text{core} \approx 5800\text{ K}$), an integration tolerance of $\text{rtol} = 10^{-8}$ admits a local truncation noise floor of $\text{rtol} \times 5800\text{ K} = 0.058\text{ mK}$ per step. Over an integration call, cumulative truncation noise of $\sim 33\text{ mK}$ swamps a physical cooling signal of $3.5\text{ mK}$. Tightening $\text{rtol}$ to $10^{-12}$ resolves the copied eddy diffusivity, causing un-gated convective flux to chatter and diverge. Formulating the core state as an offset $\Delta T_\text{core} = T_\text{core} - T_\text{core}(0)$ with an absolute tolerance of $10\text{ nK}$ collapses the CVODE step size below 3 seconds at $t = 0.078\text{ yr}$ as the solver attempts to resolve the 10-second thermal feedback, triggering linear system setup failure.

Therefore, the core energy closure (E1) acceptance is defined on testable physical regimes where core cooling per integration window satisfies:

$$
|\Delta T_\text{core}| \ge 10^3 \times \text{rtol} \times T_\text{core}
$$

The factor $10^3$ is a derived margin that requires physical cooling to exceed the local truncation noise floor $\text{rtol} \times T_\text{core}$ by three orders of magnitude, rather than a tuned empirical threshold. This margin ensures that discretization and quadrature errors dominate over numerical floating-point noise. For standard tolerances ($\text{rtol} = 10^{-8}$) and core temperatures ($T_\text{core} \approx 5800\text{ K}$), this corresponds to $|\Delta T_\text{core}| \ge 58\text{ mK}$ per call window. Convective regimes with active heat extraction and inner-core growth exhibit cooling rates of $|\Delta T_\text{core}| \approx 8\text{ to } 15\text{ K}$ per window, exceeding the derived margin by more than two orders of magnitude. In these regimes, the absolute temperature slot converges quadratically with output refinement ($16\times$ to $19\times$ reduction per $4\times$ points, with a continuum Richardson limit below $10^{-6}$), reaching $r_\text{core} \le 1.3\times 10^{-6}$ at 1025 points in purely convective cases and $1.8\times 10^{-7}$ during inner core growth.

Quadrature in the diagnostic core energy integral uses 32-point Gauss-Legendre quadrature in $T_\text{core}$, which is independent of mantle state-heat quadrature ($n_\text{quad} = 16$). For whole-planet energy closure (E2), increasing $n_\text{quad}$ from 16 to 64 in `_step_heat_content` reduces quadrature truncation across the solidus kink by more than $25\times$ (from $8.6\times 10^{-5}$ to $2.1\times 10^{-6}$ at 1025 output points), while $n_\text{quad} = 16$ is retained as the default in routine runs.

## References

- Anzellini, S., Dewaele, A., Mezouar, M., Loubeyre, P., & Morard, G. (2013). Melting of iron at Earth's inner core boundary based on fast X-ray diffraction. *Science*, 340(6131), 464-466. https://doi.org/10.1126/science.1233514
- Labrosse, S., Poirier, J.-P., & Le Mouël, J.-L. (2001). The age of the inner core. *Earth and Planetary Science Letters*, 190(3-4), 111-123. https://doi.org/10.1016/S0012-821X(01)00387-9
- Nimmo, F. (2015). Energetics of the Core. In G. Schubert (Ed.), *Treatise on Geophysics* (2nd ed., Vol. 9, pp. 27-55). Elsevier. https://doi.org/10.1016/B978-0-444-53802-4.00139-1
