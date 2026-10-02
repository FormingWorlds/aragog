# Solid-State Convection (Step 1: Stagnant Lid)

Aragog simulates planetary mantle evolution in both molten and solid states. As a magma ocean crystallises and the local melt fraction drops below the rheological transition threshold ($\phi_\text{rheo} = 0.40$), the mantle transitions from liquid-state turbulent magma flow to solid-state creep. In Step 1 of solid-state convection, Aragog implements Arrhenius diffusion creep with pressure-dependent activation enthalpy, stagnant lid boundary-layer parameterisations, and energy conservation closures within a one-dimensional radial mixing-length framework. Plastic yielding and mobile lid dynamics are deferred to Step 2.

---

## 1. Arrhenius Diffusion Creep

In the solid regime ($\phi < \phi_\text{rheo}$), solid mantle deformation occurs through diffusion creep. The diffusion-creep viscosity follows an Arrhenius relation referenced to a surface reference state $(T_\text{ref}, P = 0)$:

$$
\eta_\text{diff}(T, P) = \eta_\text{solid} \cdot f_\text{water} \cdot \exp\left( \frac{H(P)}{R T} - \frac{H(0)}{R T_\text{ref}} \right)
$$

where:

- $\eta_\text{solid}$ is the reference solid viscosity at temperature $T_\text{ref} = 1600\text{ K}$ and zero pressure (`viscosity` in `[phase_solid]`, default $10^{21}\text{ Pa s}$; source: `src/aragog/parser.py:_PhaseParameters`).
- $f_\text{water}$ is the hydration weakening prefactor (`water_prefactor`, default $1.0$; source: `src/aragog/rheology.py:compute_diffusion_creep_viscosity`).
- $R = 8.314462618\text{ J mol}^{-1}\text{ K}^{-1}$ is the universal gas constant (source: `scipy.constants.R`).
- $H(P)$ is the activation enthalpy.

### Saturating Activation Enthalpy

Under a constant activation volume ($V_a$), activation enthalpy grows linearly with pressure as $H(P) = E_a + P V_a$. In deep planetary mantles, linear extrapolation to core-mantle boundary pressures ($135\text{ GPa}$ in Earth) produces activation enthalpies approaching $1000\text{ kJ mol}^{-1}$ and viscosity increases exceeding nine orders of magnitude along an adiabat. This locks the deep mantle into an unphysical rigid state.

Mineral physics measurements and ab initio calculations demonstrate that the effective activation volume decreases markedly under compression (Yamazaki and Karato, 2001, Am. Mineral. 86, 385; Karato, 2008, Deformation of Earth Materials, Cambridge Univ. Press). Aragog models this pressure saturation via an exponential decay of activation volume (Tackley et al., 2013, Icarus 225, 50, eqs. 2-3):

$$
V_a(P) = V_0 \exp\left( -\frac{P}{P_\text{decay}} \right)
$$

Integrating $\partial H / \partial P = V_a(P)$ from zero pressure gives the saturating activation enthalpy:

$$
H(P) = E_a + V_0 P_\text{decay} \left( 1 - \exp\left( -\frac{P}{P_\text{decay}} \right) \right)
$$

where:

- $E_a$ is the zero-pressure activation energy (`activation_energy`, default $300\text{ kJ mol}^{-1}$; source: `src/aragog/parser.py:_SolidRheologyParams`).
- $V_0$ is the zero-pressure activation volume (`activation_volume`, default $5 \times 10^{-6}\text{ m}^3\text{ mol}^{-1}$; source: `src/aragog/parser.py:_SolidRheologyParams`).
- $P_\text{decay}$ is the characteristic pressure scale for activation volume decay (`activation_volume_decay_pressure`, default $\infty$; source: `src/aragog/parser.py:_SolidRheologyParams`).

When $P_\text{decay} \to \infty$, $H(P)$ reduces to the linear form $H(P) = E_a + P V_0$. When $V_0 = 0$, $H(P)$ reduces to the temperature-only Arrhenius law $H(P) = E_a$. Baseline parameters ($E_a = 300\text{ kJ mol}^{-1}$, $V_0 = 5 \times 10^{-6}\text{ m}^3\text{ mol}^{-1}$) reflect dry olivine diffusion creep (Karato and Wu, 1993, Science 260, 771; Hirth and Kohlstedt, 2003, Geophys. Monogr. 138, 83).

### Numerical Implementation

Evaluating the expression $V_0 P_\text{decay} (1 - \exp(-P / P_\text{decay}))$ directly when $P_\text{decay} = \infty$ yields an indeterminate $\infty \times 0$ form. In numerical kernels, the enthalpy is evaluated using the vectorised array-level form:

$$
H(P) = E_a + V_0 \cdot \operatorname{where}(\operatorname{isinf}(P_\text{decay}), P, -P_\text{safe} \cdot \operatorname{expm1}(-P / P_\text{safe}))
$$

with $P_\text{safe} = \operatorname{where}(\operatorname{isinf}(P_\text{decay}), 1.0\text{ Pa}, P_\text{decay})$. This form guarantees float64 accuracy for both finite and infinite decay scales (source: `src/aragog/rheology.py:compute_activation_enthalpy` and `src/aragog/jax/rheology.py:activation_enthalpy`).

---

## 2. Regime of Validity

Boundary-layer stagnant lid scaling requires two asymptotic conditions (Solomatov, 1995, Phys. Fluids 7, 266):

1. Small temperature scale relative to interior temperature:
   $$
   \frac{R T_i}{E_a} \ll 1
   $$
   (typically $\approx 0.05$ to $0.10$ for mantle temperatures of $1600\text{ K}$ to $3500\text{ K}$).
2. Viscosity contrast through the lithosphere exceeding $10^4$, quantified by the Frank-Kamenetskii contrast parameter:
   $$
   \theta = \frac{H(P_{T_i}) (T_i - T_\text{surf})}{R T_i^2} \ge 9
   $$

The true Arrhenius viscosity contrast $\eta_\text{diff}(T_\text{surf}, P_\text{surf}) / \eta_\text{diff}(T_i, P_{T_i})$ is substantially larger than $\exp(\theta)$ because of non-linear Arrhenius curvature. Aragog reports both $\theta$ and the true contrast as diagnostics. If $\theta < 9$, the convective system operates outside the asymptotic stagnant-lid regime (Solomatov, 1995).

---

## 3. Stagnant Lid Formulations

Step 1 evaluates two distinct formulations for stagnant-lid heat transport:

### Formulation A: MLT-Emergent Stagnant Lid

In Formulation A, the stagnant lid emerges directly from the mixing-length equations without an external parameterised boundary layer switch. In regions where temperatures are cold, Arrhenius diffusion-creep viscosity $\eta_\text{diff}$ increases by many orders of magnitude. The mixing-length convective velocity scales inversely with viscosity ($v \propto 1 / \eta$), which causes convective heat transport $F_\text{conv} = \rho c_p \kappa_h |dT/dr - (dT/dr)_S|$ to drop naturally to zero. Heat transport through the cold lid is then governed purely by molecular conduction $\kappa |dT/dr|$.

The transition between convective interior and conductive lid depends on the mixing-length profile. Setting `mixing_length_profile = 'nearest_boundary'` allows a stagnant lid to form, whereas a constant mixing length (`'constant'`) fails to establish a stable lid profile (Deschamps and Sotin, 2000, Geophys. J. Int. 143, 204).

### Formulation B: Parameterised Column Lid

Formulation B couples a parameterised boundary-layer column model to the internal mixing-length domain (Thiriet et al., 2019, J. Geophys. Res. Planets 124, 138; Morschhauser et al., 2011, Icarus 216, 82).

1. **Interior Adiabatic Temperature ($T_m$):**
   The interior potential temperature is evaluated by mapping mantle temperatures along the local isentrope to the surface:
   $$
   \theta = T \exp(-I), \quad I(r) = \int_r^{R_\text{top}} \frac{|dT/dr|_S}{T} dr'
   $$
   A softmax-weighted mean with a $1\text{ K}$ scale isolates the hottest adiabat $\theta_m$, giving:
   $$
   T_m = \theta_m \exp(I(R_l))
   $$
   where $R_l$ is the radial position of the lid base.

2. **Lid Base Temperature ($T_l$):**
   The temperature at the base of the stagnant lid follows Frank-Kamenetskii boundary-layer scaling:
   $$
   T_l = T_m - a_\text{rh} \frac{R T_m^2}{H(P(R_l))}
   $$
   where $a_\text{rh} = 2.54$ is the rheological temperature contrast coefficient (Thiriet et al., 2019, eq. 15; sensitivity value $a_\text{rh} = 2.9$, Morschhauser et al., 2011, eq. 8).

3. **Rheological Rayleigh Number ($\text{Ra}_\text{rh}$):**
   The Rayleigh number governing convective vigor beneath the lid is defined as:
   $$
   \text{Ra}_\text{rh} = \frac{\alpha \rho g (T_c - T_l) (R_l - R_c)^3}{\kappa \eta(T_m)}
   $$
   where $T_c$ is the core-mantle boundary temperature, $R_c$ is core radius, and $\eta(T_m)$ is interior viscosity (Thiriet et al., 2019, eq. 14). Where the interior is hotter than the core, $T_c - T_l$ is replaced by $\max(T_c, T_m) - T_l$.

4. **Rheological Sublayer Thickness ($\delta_u$):**
   The convective sublayer thickness accommodating the boundary layer drop is:
   $$
   \delta_u = (R_l - R_c) \left( \frac{\text{Ra}_\text{crit}}{\text{Ra}_\text{rh}} \right)^\beta
   $$
   with critical Rayleigh number $\text{Ra}_\text{crit} = 450$ and scaling exponent $\beta = 0.335$ (Thiriet et al., 2019, eq. 13; sensitivity pair $\text{Ra}_\text{crit} = 450, \beta = 1/3$, Morschhauser et al., 2011, eq. 10).

5. **Conductive Suppression Weight ($w_\text{cond}$):**
   Convective eddy diffusivity and flux are multiplied by $1 - w_\text{cond}$, where:
   $$
   w_\text{cond} = \frac{1}{2} \left( 1 + \tanh\left( \frac{r - (R_l - \delta_u)}{\Delta r} \right) \right) w_\text{solid}
   $$
   This enforces pure molecular conduction in the lid and its sublayer ($r > R_l - \delta_u$), while restoring standard mixing-length convection in the deeper interior.

---

## 4. Numerical Solver Stability and Lagged State

Coupling parameterised boundary-layer equations directly into implicit ODE solvers (such as CVODE) introduces dense non-local dependencies into the Jacobian matrix. When the boundary layer thickness $R_l$ and sublayer scale $\delta_u$ depend non-locally on integral interior states, standard banded linear solvers fail to converge.

To guarantee numerical stability, Aragog lags parameterised column states ($T_m$, $R_l$, $\delta_u$, $w_\text{solid}$). These quantities are updated at the start of each solver call and held constant during internal sub-stepping. Numerical experiments demonstrate that lagging produces negligible time-step dependence: solutions computed with $10^7\text{ yr}$ calls differ from $10^6\text{ yr}$ calls by less than $0.34\text{ K}$ in mean interior temperature and less than $0.08\text{ mW m}^{-2}$ in surface heat flux.

---

## 5. Lid Onset in Crystallising Mantles

In an evolving magma ocean, the transition to stagnant lid convection must occur smoothly. Setting an unramped step threshold at a single radial node triggers abrupt surface flux drops (exceeding $80\%$ in a single time step) and solver convergence failure.

Aragog parameterises lid onset through the solid fraction at the lid base:

$$
w_\text{solid} = 1 - \frac{1}{2} \left( 1 + \tanh\left( \frac{\phi(R_l) - \phi_\text{rheo}}{\phi_\text{width}} \right) \right)
$$

with $\phi_\text{rheo} = 0.40$ and $\phi_\text{width} = 0.05$ (source: `src/aragog/parser.py:_PhaseMixedParameters`). The stagnant lid forms progressively as the melt fraction at the lid base falls below the rheological transition threshold, preventing artificial thermal shocks.

---

## 6. Interaction with the Eddy-Diffusivity Floor

PROTEUS applies an eddy-diffusivity floor $\kappa_{h,\text{floor}} = 10\text{ m}^2\text{ s}^{-1}$ in coupled magma ocean simulations to prevent numerical stagnation in convective liquid regions. The floor is ramped through the melt fraction:

$$
f(\phi) = \frac{1}{2} \left( 1 + \tanh\left( \frac{\phi - \phi_\text{rheo}}{\phi_\text{width}} \right) \right)
$$

In a cold conductive lid, the entropy gradient is strongly negative ($\partial S / \partial r < 0$). Applying an unmasked eddy-diffusivity floor in this region would introduce artificial thermal diffusivity five orders of magnitude larger than molecular thermal conductivity ($\kappa \sim 10^{-6}\text{ m}^2\text{ s}^{-1}$), destroying stagnant-lid thermal profiles.

When solid rheology is enabled, Aragog masks the eddy-diffusivity floor inside the lid:

$$
\kappa_{h,\text{floor,eff}} = \kappa_{h,\text{floor}} \cdot f(\phi) \cdot (1 - w_\text{lid})
$$

Inside the cold lid ($w_\text{lid} \to 1$), the floor is suppressed to zero, preserving molecular conduction. In the convective interior and magma ocean ($w_\text{lid} \to 0$), the floor remains active.

---

## 7. Boundary Conditions and Conservation

### Surface Boundary: Conductive Skin (Outer BC 6) and Cutoff

Outer boundary condition 6 implements a conductive surface skin boundary condition coupled to a table-edge cutoff (source: `src/aragog/solver/entropy_solver.py:_skin_surface_flux` and `src/aragog/surface_skin.py`). The surface temperature $T_s$ balances radiative loss against conduction through the outermost half-cell:

$$
\epsilon \sigma (T_s^4 - T_\text{eq}^4) = \frac{k_\text{top}}{\Delta r_\text{half}} (T_\text{top} - T_s)
$$

where $\Delta r_\text{half} = 0.5 (r_{N-1} - r_{N-2})$ and $\sigma = 5.670374419 \times 10^{-8}\text{ W m}^{-2}\text{ K}^{-4}$.

Evaluating this condition on a uniform radial grid ($N = 100$, uniform cell $\approx 29\text{ km}$, $\Delta r_\text{half} \approx 14.5\text{ km}$) severely chokes the conductive flux ($0.21\text{ W m}^{-2}$ during magma ocean cooling). The configuration loader strictly enforces the BC 6 surface refinement rule in `src/aragog/parser.py:Parameters.__post_init__`: configuring `outer_boundary_condition = 6` with `mesh.surface_cell_thickness <= 0.0` raises a `ValueError` stating that `outer_boundary_condition = 6 (conductive surface skin) requires [mesh] surface_cell_thickness > 0`. This validation rule requires opt-in surface cell refinement (for example, `surface_cell_thickness = 1000.0` m) to guarantee that the skin layer is physically resolved.

To prevent surface cells from leaving the lower temperature bound of tabulated equations of state ($T \approx 320\text{ K}$ in SPIDER solid P-S tables), the outward flux is scaled near the table edge:

$$
F_\text{out} = F_\text{nominal} \cdot \frac{1}{2} \left( 1 + \tanh\left( \frac{S - S_\text{edge} - 100}{30} \right) \right)
$$

This cutoff holds the top cell on the valid equation-of-state domain while maintaining outward heat loss governed by lid conduction.

### Core Boundary: Conserving Core Coupling

At the core-mantle boundary, Aragog couples mantle heat loss to core energetics:

- Inner BC 1 (`quasi_steady`): Solves core thermal evolution with single-cell heating partitioning. In this boundary condition, bottom-cell radiogenic heating is accounted for once in the boundary layer flux without spurious double-counting (source: `src/aragog/solver/entropy_solver.py:_step_dE_components`).
- Inner BC 3 (Fixed CMB temperature): Implements conductive heat transfer through the inner half-cell from a specified CMB temperature $T_\text{cmb}$:
  $$
  F_\text{cmb} = \frac{k_\text{bot}}{\Delta r_\text{half,bot}} (T_\text{cell,0} - T_\text{cmb})
  $$

---

## 8. Diagnostics and Behavior in Mush

### Thermal Expansivity Clamp (F21)

In cold, partially solidified regions near the lid base, tabulated equations of state can report raw negative thermal expansivities ($\alpha < 0$) when extrapolating near table boundaries. Aragog clamps $\alpha$ to a positive floor ($\approx 10^{-13}\text{ K}^{-1}$; source: `src/aragog/eos/entropy.py`). While this clamp prevents solver failure, local thermal buoyancy vanishes where the clamp operates, which affects mixing-length convective velocities in mushy boundary cells.

### Output Diagnostics

The table below describes solid-state convection diagnostics recorded in NetCDF snapshot files:

| Diagnostic Name | Unit | Physical Meaning | Source Implementation |
|:---|:---:|:---|:---|
| `lid_thickness` | m | Physical thickness of the stagnant lid $d_\text{lid}$ | `src/aragog/rheology_lid.py` |
| `lid_base_temperature` | K | Temperature $T_\text{lid}$ at the base of the lid | `src/aragog/rheology_lid.py` |
| `interior_temperature` | K | Representative convective interior temperature $T_m$ or $T_i$ | `src/aragog/rheology_lid.py` |
| `lid_stress` | Pa | Convective driving shear stress $\tau_d$ | `src/aragog/rheology_lid.py` |
| `theta` | - | Frank-Kamenetskii contrast parameter $\theta$ | `src/aragog/rheology.py` |
| `lid_regime` | - | Lid regime indicator (0 none, 1 stagnant, 2 mobile) | `src/aragog/rheology_lid.py` |
| `energy_residual` | W | Global discrete energy conservation residual | `src/aragog/solver/entropy_solver.py` |

---

## 9. Benchmark Suite and Published Baselines

Aragog validates Step 1 stagnant-lid convection against published geodynamic benchmarks:

| ID | Reference | Physical System | Key Compared Quantities | Target Tolerances |
|:---:|:---|:---|:---|:---:|
| **B1** | Korenaga (2009), Table 1 | 2D isoviscous/Arrhenius convection, internal heating, insulated base | Nusselt number $\text{Nu}$, interior temperature $T_i$ | $|\text{Nu}/\text{Nu}_\text{ref} - 1| \le 10\%$ |
| **B2** | Thiriet et al. (2019), Mars1 | 3D spherical cooling planet, decaying radionuclides, core balance | Mean mantle $T$, surface flux $F_\text{surf}$, CMB flux $F_\text{cmb}$, $T_\text{cmb}$ | Within 1 unit of eq. 17 ($20\text{ K}$, $3\text{ mW m}^{-2}$) |
| **B3** | Deschamps and Vilella (2021) | 3D spherical shells, FK viscosity, mixed heating | Non-dimensional $T_m$, surface flux $\Phi_\text{top}$, lid thickness $d_\text{lid}$ | $T_m \pm 0.03$, $\Phi_\text{top} \pm 15\%$, $d_\text{lid} \pm 20\%$ |
| **B4** | Morschhauser et al. (2011) | 0D parameterised thermal evolution | Evolution trajectories against independent 0D reference | Match 0D trajectory within B2 tolerances |
| **B5** | Standalone verification | Tables mode evolution from magma ocean state (`mo_15099yr.npz`) | Numerical stability during lid onset, CVODE step counts | No solver stalls, no out-of-bounds EOS errors |
| **B6** | Euen et al. (2023) | 2D spherical shell, basal heating, ASPECT benchmark | Mean mantle temperature, surface and basal Nusselt numbers | Documented in Section 10 |

---

## 10. Benchmark B6 (Euen et al. 2023) Results and Mechanism

Evaluation of the 2D spherical shell benchmark cases of Euen et al. (2023, Geophys. J. Int. 234, 1557) computed with ASPECT indicates that Aragog produces mean non-dimensional temperatures of $0.59$ to $0.63$ in all cases, whereas ASPECT reports $0.17$ to $0.24$ (except benchmark case A7 at $0.515$).

Systematic numerical and physical analysis establishes the governing mechanisms:

1. **Numerical Discretisation**:
   Spatial grid refinement over $N = 50, 100, 200, 400$ nodes demonstrates monotonic spatial convergence with variation under $0.5\%$. Pure conduction solutions ($\text{Ra} = 1$) reproduce the analytical spherical profile within $10^{-4}$. The temperature discrepancies reflect physical parameterisation choices rather than numerical discretisation artifacts.
2. **CMB Boundary Layer Law Interaction**:
   Benchmark B6 features purely basal heating without internal heat sources. In Aragog, activating the Deschamps and Sotin (2000) CMB boundary layer law in combination with a reduced top mixing-length slope ($s_\text{top} = 0.22$) forces interior temperatures upward toward the core temperature. In benchmark case A1 at $N=100$:
   - Combined configuration (law on, $s_\text{top} = 0.22$): mean temperature is $0.596$.
   - Standard MLT baseline (law off, $s_\text{top} = 1.0$): mean temperature is $0.301$.
   - Published ASPECT reference: mean temperature is $0.216$.
   Decomposition shows that the DS2000 CMB boundary layer law contributes $+0.271$ to the interior temperature offset, while the reduced top slope contributes $+0.112$.
3. **Physical Interpretation**:
   The DS2000 parameterisation is calibrated on mixed-heating and internally heated regimes (B1 and B3). In purely basal-heated convection, applying the stagnant-lid boundary layer law at the core boundary overestimates convective heat retention, driving the 1D model interior toward the basal temperature.

---

## 11. heat_budget Cross-Code Comparison

Cross-code comparison against the parameterised thermal evolution model `heat_budget` (Thiriet et al., 2019) for 30 planetary cooling cases ($M \in \{1, 2, 4\} M_\text{Earth}$, $\eta_0 \in \{10^{19}, 10^{21}\}\text{ Pa s}$, epochs $0.5$ to $4.5\text{ Gyr}$) identifies two key physical differences:

1. **Radionuclide Heating Rate**:
   Volumetric radiogenic heating in Aragog is parameterised through radionuclide concentrations converted to mass fractions (`src/aragog/parser.py`), generating a reference initial heating rate of $24\text{ pW kg}^{-1}$.
2. **Deep Mantle Viscosity Saturation**:
   In Aragog, the activation volume is applied at full local lithostatic pressure throughout the radial column. In massive planets ($2 M_\text{Earth}$ and $4 M_\text{Earth}$), pressure in the deep lower mantle exceeds $100\text{ GPa}$, driving diffusion-creep viscosity to the numerical ceiling ($10^{40}\text{ Pa s}$). This halts convective heat extraction from the deep interior, causing radiogenic heat to accumulate beneath the core-mantle boundary and maintaining bottom temperatures near $3050\text{ K}$ at $4.5\text{ Gyr}$. In contrast, 0D parameterised models evaluate mantle viscosity at a single shallow reference pressure, allowing continuous mantle cooling.

---

## 12. Solid Rheology Parameter Reference

The table below lists all configuration parameters for solid-state convection in Aragog:

| Parameter Name | Data Type | Default Value | Physical Units | Functional Description |
|:---|:---:|:---:|:---:|:---|
| `enabled` | `bool` | `False` | dimensionless | Master switch to enable solid-state rheology |
| `activation_energy` | `float` | `300000.0` | $\text{J mol}^{-1}$ | Zero-pressure activation energy $E_a$ |
| `activation_volume` | `float` | `5.0e-6` | $\text{m}^3\text{ mol}^{-1}$ | Zero-pressure activation volume $V_0$ |
| `activation_volume_decay_pressure` | `float` | `inf` | $\text{Pa}$ | Pressure decay scale $P_\text{decay}$ for $V_a(P)$ saturation |
| `arrhenius_t_ref` | `float` | `1600.0` | $\text{K}$ | Reference temperature $T_\text{ref}$ for diffusion creep |
| `viscosity_max_log10` | `float` | `40.0` | $\log_{10}(\text{Pa s})$ | Numerical ceiling on solid mantle viscosity |
| `water_prefactor` | `float` | `1.0` | dimensionless | Hydration weakening multiplier $f_\text{water}$ |
| `yield_stress_c` | `float` | `50.0e6` | $\text{Pa}$ | Cohesion intercept for Byerlee plastic yield stress |
| `yield_stress_mu` | `float` | `0.6` | dimensionless | Friction coefficient for Byerlee plastic yield stress |
| `yield_stress_max` | `float` | `500.0e6` | $\text{Pa}$ | Maximum ceiling on Byerlee plastic yield stress |
| `stress_closure_mode` | `str` | `'lid'` | string | Stress closure mode (`'lid'` or `'local'`) |
| `lid_base_mode` | `str` | `'rheological'` | string | Lid base selection mode (`'rheological'` or `'fixed'`) |
| `lid_base_temperature` | `float` | `1400.0` | $\text{K}$ | Fixed isotherm temperature for lid base when mode is `'fixed'` |
| `lid_contrast_coeff` | `float` | `2.2` | dimensionless | Rheological contrast coefficient $a_\text{rh}$ for lid base |
| `lid_mask_width_cells` | `float` | `1.0` | cells | Smoothing width for hyperbolic tangent lid mask |
| `interior_flux_fraction` | `float` | `0.05` | dimensionless | Threshold convective flux fraction defining interior boundary |
| `phi_visc_single` | `float` | `0.5` | dimensionless | Single-phase cutoff melt fraction in mushy viscosity blending |
| `mlt_top_slope` | `float` | `0.22` | dimensionless | Near-surface slope of viscous-branch mixing length $l_v$ |
| `mlt_bottom_slope` | `float` | `1.0` | dimensionless | Near-CMB slope of viscous-branch mixing length $l_v$ |
| `max_step_const_mode` | `float` | `100.0` | $\text{yr}$ | Maximum solver time step in constant-properties mode |

*Note: Plastic yielding parameters (`yield_stress_c`, `yield_stress_mu`, `yield_stress_max`) govern lithospheric yielding and are deferred to Step 2.*
