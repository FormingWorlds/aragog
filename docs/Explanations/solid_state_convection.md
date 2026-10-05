# Solid-State Convection

Aragog simulates planetary mantle evolution in both molten and solid states. As the mantle crystallises and the melt fraction drops below the rheological transition threshold ($\phi_\text{rheo} = 0.4$), the interior transitions from liquid-state turbulent magma flow to solid-state creep. Aragog couples Arrhenius diffusion creep, Byerlee plastic yielding, and boundary-layer stress closure to model stagnant-lid and mobile-lid convection regimes within a 1D mixing-length framework.

## 1. Arrhenius Diffusion Creep

In the solid regime ($\phi < \phi_\text{rheo}$), the baseline solid matrix deforms by diffusion creep. The diffusion-creep viscosity follows an Arrhenius relation referenced to a surface reference state $(T_\text{ref}, P = 0)$:

$$
\eta_\text{diff}(T, P) = \eta_\text{solid} \cdot f_\text{water} \cdot \exp\left( \frac{H(P)}{R T} - \frac{H(0)}{R T_\text{ref}} \right)
$$

where:

- $\eta_\text{solid}$ is the reference solid viscosity at temperature $T_\text{ref} = 1600\text{ K}$ and zero pressure (`viscosity_solid`, default $10^{21}\text{ Pa s}$).
- $f_\text{water}$ is the hydration weakening prefactor (`water_prefactor`, default $1.0$).
- $R = 8.314462618\text{ J mol}^{-1}\text{ K}^{-1}$ is the universal gas constant.
- $H(P)$ is the activation enthalpy.

### Saturating Activation Enthalpy

Under constant activation volume ($V_a$), activation enthalpy grows linearly with pressure as $H(P) = E_a + P V_a$. In deep planetary mantles, linear extrapolation to core-mantle boundary pressures ($135\text{ GPa}$ in Earth) produces activation enthalpies approaching $1000\text{ kJ mol}^{-1}$ and viscosity increases exceeding nine orders of magnitude along an adiabat. This locks the deep mantle into an unphysical rigid state.

Mineral physics measurements and ab initio calculations demonstrate that the effective activation volume decreases markedly under compression (Yamazaki and Karato, 2001; Karato, 2008). Aragog models this pressure saturation via an exponential decay of activation volume:

$$
V_a(P) = V_0 \exp\left( -\frac{P}{P_\text{decay}} \right)
$$

Integrating $\partial H / \partial P = V_a(P)$ from zero pressure gives the saturating activation enthalpy:

$$
H(P) = E_a + V_0 P_\text{decay} \left( 1 - \exp\left( -\frac{P}{P_\text{decay}} \right) \right)
$$

where:

- $E_a$ is the zero-pressure activation energy (`activation_energy`, default $300\text{ kJ mol}^{-1}$).
- $V_0$ is the zero-pressure activation volume (`activation_volume`, default $5 \times 10^{-6}\text{ m}^3\text{ mol}^{-1}$).
- $P_\text{decay}$ is the characteristic pressure scale for activation volume decay (`activation_volume_decay_pressure`, default $\infty$).

When $P_\text{decay} \to \infty$, $H(P)$ reduces to the linear form $H(P) = E_a + P V_0$. When $V_0 = 0$, $H(P)$ reduces to the temperature-only Arrhenius law $H(P) = E_a$. The default $P_\text{decay} = \infty$ preserves compatibility with linear activation volume baselines; setting $P_\text{decay} \approx 60\text{ GPa}$ is recommended for deep mantle convection studies.

### Numerical Evaluation

Evaluating the expression $V_0 P_\text{decay} (1 - \exp(-P / P_\text{decay}))$ directly when $P_\text{decay} = \infty$ yields an indeterminate $\infty \times 0$ form. In numerical kernels, the enthalpy is evaluated using the vectorised array-level form:

$$
H(P) = E_a + V_0 \cdot \operatorname{where}(\operatorname{isinf}(P_\text{decay}), P, -P_\text{safe} \cdot \operatorname{expm1}(-P / P_\text{safe}))
$$

with $P_\text{safe} = \operatorname{where}(\operatorname{isinf}(P_\text{decay}), 1.0\text{ Pa}, P_\text{decay})$. This form guarantees float64 accuracy for both finite and infinite decay scales.

### Viscosity Values and Mantle Contrast

The table below reports activation enthalpy $H(P)$ and normalised viscosity $\eta_\text{diff} / \eta_\text{solid}$ at representative mantle pressures:

| Pressure $P$ (GPa) | $H(P)$ at $P_\text{decay} = 60\text{ GPa}$ (kJ/mol) | $T = 1600\text{ K}$ | $T = 3000\text{ K}$ | $T = 3500\text{ K}$ | $H(P)$ at $P_\text{decay} = \infty$ (kJ/mol) | $T = 1600\text{ K}$ | $T = 3000\text{ K}$ | $T = 3500\text{ K}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 300.0 | $1.000$ | $2.689 \times 10^{-5}$ | $4.824 \times 10^{-6}$ | 300.0 | $1.000$ | $2.689 \times 10^{-5}$ | $4.824 \times 10^{-6}$ |
| 25 | 402.2 | $2.174 \times 10^{3}$ | $1.620 \times 10^{-3}$ | $1.618 \times 10^{-4}$ | 425.0 | $1.204 \times 10^{4}$ | $4.036 \times 10^{-3}$ | $3.539 \times 10^{-4}$ |
| 60 | 489.6 | $1.552 \times 10^{6}$ | $5.387 \times 10^{-2}$ | $3.262 \times 10^{-3}$ | 600.0 | $6.220 \times 10^{9}$ | $4.497 \times 10^{0}$ | $1.447 \times 10^{-1}$ |
| 135 | 568.4 | $5.775 \times 10^{8}$ | $1.266 \times 10^{0}$ | $4.882 \times 10^{-2}$ | 975.0 | $1.087 \times 10^{22}$ | $1.521 \times 10^{7}$ | $5.715 \times 10^{4}$ |

Geophysical inversions of post-glacial rebound, geoid anomalies, and mantle convection constrain the radial viscosity contrast throughout the whole mantle to a factor of approximately 10 to 100 between the upper mantle and the deep lower mantle (Mitrovica and Forte, 2004; Rudolph et al., 2015; Lau et al., 2016). Along an Earth-like mantle geotherm from $(P = 5\text{ GPa}, T = 1700\text{ K})$ to $(P = 135\text{ GPa}, T = 2600\text{ K})$:

- With $P_\text{decay} = 60\text{ GPa}$, the saturating law yields $\eta_\text{diff}(135\text{ GPa}) / \eta_\text{diff}(5\text{ GPa}) \approx 29.1$, consistent with geodynamic constraints.
- With $P_\text{decay} = \infty$, the constant-$V_a$ law yields a contrast of $4.00 \times 10^{9}$, which overestimates lower-mantle viscosity by seven orders of magnitude.

The model rheology is single-phase and does not include discontinuous jumps from phase changes (such as the ringwoodite to bridgmanite plus ferropericlase transition). Solid-solid phase transformations introduce an estimated factor of 3 to 10 increase in lower-mantle viscosity (Tackley, 1996; Yamazaki and Karato, 2001). The continuous model contrast along the adiabat must therefore remain below the total observed contrast to leave room for phase transition increments. The formulation acts as a Newtonian diffusion-creep proxy at fixed water content and fixed grain size (both absorbed into $\eta_\text{solid}$). Baseline parameters ($E_a = 300\text{ kJ mol}^{-1}$, $V_0 = 5 \times 10^{-6}\text{ m}^3\text{ mol}^{-1}$) reflect dry olivine diffusion creep (Karato and Wu, 1993; Hirth and Kohlstedt, 2003).

## 2. Plastic Yielding Criterion

To model lithospheric failure and mobile-lid convection, Aragog implements a Byerlee frictional yield criterion with a ductile stress ceiling (Byerlee, 1978; Moresi and Solomatov, 1998; Tackley, 2000):

$$
\tau_y(P) = \min(c + \mu P, \tau_\text{max})
$$

where:

- $c$ is the lithospheric cohesion (`yield_stress_c`, default $50\text{ MPa}$).
- $\mu$ is the friction coefficient (`yield_stress_mu`, default $0.6$).
- $P$ is the lithostatic pressure.
- $\tau_\text{max}$ is the yield stress ceiling (`yield_stress_max`, default $500\text{ MPa}$).

### Invariant Conventions

Tensor invariants follow the second invariants of deviatoric stress and strain rate (Moresi and Solomatov, 1998; Tackley, 2000):

$$
\tau = \sqrt{\frac{1}{2} \tau_{ij} \tau_{ij}}, \quad \dot{\epsilon} = \sqrt{\frac{1}{2} \dot{\epsilon}_{ij} \dot{\epsilon}_{ij}}
$$

Under simple shear flow with velocity gradient $\partial v / \partial y = v / l$, the strain rate tensor has components $\dot{\epsilon}_{xy} = \dot{\epsilon}_{yx} = \frac{1}{2} v / l$, which yields $\dot{\epsilon} = v / (2 l)$. The factor of 2 enters directly into the simple-shear strain rate relation.

### Physical Validity and Ceiling Depth

Byerlee frictional failure governs brittle deformation at temperatures below approximately $900\text{ K}$ to $1100\text{ K}$ ($600^\circ\text{C}$ to $800^\circ\text{C}$). At higher temperatures and deeper mantle levels, ductile deformation mechanisms limit shear strength. The ceiling $\tau_\text{max}$ acts as a proxy for this ductile limit.

With $c = 50\text{ MPa}$ and $\mu = 0.6$, the yield stress reaches the $500\text{ MPa}$ ceiling at pressure:

$$
P_\text{ceil} = \frac{\tau_\text{max} - c}{\mu} = \frac{500\text{ MPa} - 50\text{ MPa}}{0.6} = 750\text{ MPa} = 0.75\text{ GPa}
$$

On Earth, $0.75\text{ GPa}$ corresponds to approximately $23\text{ km}$ depth. Because realistic thermal lithospheres are thicker than $23\text{ km}$, the yield stress at the base of the lid is typically set by the ceiling $\tau_\text{max}$. The scalar $\tau_{y,\text{lid}}$ denotes the yield stress evaluated at the lid base pressure and governs lid mobilisation. The local profile $\tau_y(P)$ is retained as an output diagnostic.

## 3. Stagnant Lid Identification

In a vigorously convecting interior beneath a cold surface, convective heat transport dominates in the interior, whereas molecular conduction governs heat loss through the cold surface lid.

### Interior Convective State

Aragog identifies the convecting mantle region from the convective heat flux. The interior temperature $T_i$ and pressure $P_{T_i}$ are defined through a smooth sigmoid selection on the convective flux fraction $f_k = (F_\text{conv} / F_\text{tot})_k$ relative to the interior flux threshold $f_\text{conv,min}$ (`interior_flux_fraction`, default $0.05$). For radial cells indexed $k = 1, \dots, N$ from surface to core-mantle boundary, the local convective weight is:

$$
w_{\text{conv}, k} = \frac{1}{2} \left( 1 + \tanh\left( \frac{f_k - f_\text{conv,min}}{w_\text{conv}} \right) \right)
$$

with transition width $w_\text{conv} = \max(\Delta f_\text{cell}, 10^{-4})$ floored to maintain a finite non-zero width in uniform flux regions. The shallowest convective boundary is isolated by the cumulative product indicator:

$$
W_k = w_{\text{conv}, k} \prod_{j < k} (1 - w_{\text{conv}, j})
$$

Normalising by the sum $\sum_m W_m$ yields the boundary distribution $\tilde{W}_k = W_k / (\sum_m W_m + 10^{-12})$, from which interior state variables are evaluated smoothly:

$$
T_i = \sum_{k=1}^N \tilde{W}_k T_k, \quad P_{T_i} = \sum_{k=1}^N \tilde{W}_k P_k
$$

An overall convective activity weight $w_\text{active} = \frac{1}{2}(1 + \tanh((\sum_k w_{\text{conv}, k} - 1.0) / 0.5))$ modulates the closure. If $f_k < f_\text{conv,min}$ throughout the mantle, $w_\text{active} \to 0$, no lid is delimited, and yielding remains inactive.

### Rheological Temperature Scale

The local temperature scale over which diffusion-creep viscosity changes by a factor of $e$ is the Frank-Kamenetskii rheological temperature scale:

$$
\Delta T_\text{rh} = \frac{R T_i^2}{H(P_{T_i})}
$$

This scale is evaluated at the pressure $P_{T_i}$ of the interior node to avoid circular dependence on the lid thickness.

### Asymptotic Validity Conditions

Boundary-layer stagnant lid scaling requires two asymptotic conditions (Solomatov, 1995):

1. Small temperature scale relative to interior temperature: $R T_i / E_a \ll 1$ (typically $\approx 0.1$ at $3500\text{ K}$).
2. Viscosity contrast through the lithosphere exceeding $10^4$, quantified by the Frank-Kamenetskii contrast parameter:

    $$
    \theta = \frac{H(P_{T_i}) (T_i - T_\text{surf})}{R T_i^2} \ge 9
    $$

The true Arrhenius viscosity contrast $\eta_\text{diff}(T_\text{surf}, P_\text{surf}) / \eta_\text{diff}(T_i, P_{T_i})$ is substantially larger than $\exp(\theta)$ because of non-linear Arrhenius curvature. Aragog reports both $\theta$ and the true contrast. If $\theta < 9$, the convective regime sits outside the asymptotic stagnant-lid window.

### Lid Base and Sublayer Thickness

The parameter `lid_base_mode` governs the definition of the stagnant lid base temperature $T_\text{lid}$:

- `'fixed'`: The lid base is defined by a user-specified isotherm $T_\text{lid} = T_\text{base}$ set by `lid_base_temperature` (default $1400.0\text{ K}$).
- `'rheological'`: The lid base is evaluated dynamically from the Frank-Kamenetskii temperature scale:

    $$
    T_\text{lid} = T_i - a \Delta T_\text{rh}
    $$

    where $a = 2.2$ is the lid contrast coefficient (`lid_contrast_coeff`). This coefficient is derived from boundary-layer theory for steady-state Newtonian convection with exponential temperature-dependent viscosity in the asymptotic limit $\theta \gg 1$ (Solomatov, 1995; Tackley, 2000).

The physical lid base location is identified as the shallowest depth among:

1. The rheological isotherm where $T = T_\text{lid}$.
2. The intersection of the geotherm with the mantle solidus.
3. The rheological crystallisation front where $\phi = \phi_\text{rheo} = 0.4$.

The thickness of the rheological sublayer that accommodates deformation at the lid base is:

$$
\delta_\text{rh} = \frac{\Delta T_\text{rh}}{|dT/dr|_\text{lid\_base}}
$$

To prevent division by zero in isothermal or near-adiabatic profiles, the local temperature gradient is floored at a minimum positive value corresponding to conductive balance across a single cell.

### Smooth Lid Mask and Resolution

In numerical discretisations with 80 radial cells for a $2900\text{ km}$ mantle, the cell spacing is approximately $36\text{ km}$. The rheological sublayer thickness $\delta_\text{rh}$ is typically 5 to $10\text{ km}$, which falls below the single-cell width. To avoid grid-scale sensitivity, the closure evaluates $\delta_\text{rh}$ analytically from the local temperature gradient, rather than from discrete grid differences. In addition, the diagnostics record the integer cell count across the physical lid thickness $d_\text{lid}$ to track numerical resolution across the lid.

To avoid non-differentiable jumps in solver Jacobians, nodes belonging to the lid are weighted by a smooth hyperbolic tangent mask:

$$
w_\text{lid} = \frac{1}{2} \left( 1 - \tanh\left( \frac{T - T_\text{lid}}{w_\text{mask} \Delta T_\text{cell}} \right) \right)
$$

where $w_\text{mask}$ is the mask width parameter (`lid_mask_width_cells`, default $1.0$), and $\Delta T_\text{cell} = \max(|T_{k+1} - T_k|, 1.0\text{ K})$ is the local temperature step floored at $1\text{ K}$. Inside the cold lid ($T < T_\text{lid}$), $w_\text{lid} \to 1$. In the warm convecting interior ($T > T_\text{lid}$), $w_\text{lid} \to 0$.

## 4. Convective Driving Stress

Convective flow beneath the stagnant lid exerts a driving shear stress on the base of the lid. Aragog implements the stagnant-lid convective shear stress scaling of Foley and Bercovici (2014, eqs. 26 and 28, pp. 586-588):

$$
\tau_d = \frac{2 \mu_i v_m}{d}
$$

where $d = r_\text{outer} - r_\text{inner}$ is the convective layer thickness, and $\mu_i = \eta_\text{diff}(T_i, P_{T_i})$ is the interior dynamic viscosity, evaluated at the representative convective interior temperature $T_i$, and pressure $P_{T_i}$. The factor of 2 arises from horizontal shear between the lid base ($u = v_m$), and mid-depth ($u = 0$ at $z = d/2$), as derived by Foley and Bercovici (2014, p. 584, lines 410-412; p. 588, lines 871-872).

The interior convective velocity $v_m$ follows the boundary-layer scaling for bottom-heated stagnant-lid convection (Dumoulin et al. 1999, p. 12760, sec. 2; Foley and Bercovici 2014, p. 586-587, eq. 26):

$$
v_m = \frac{\kappa}{d} C_4 \left( \frac{\mathrm{Ra}_\text{eff} a_\text{rh}}{\theta} \right)^{2/3}
$$

where $C_4 = 0.125$ is the empirical prefactor calibrated by Foley and Bercovici (2014, sec. 4.2, p. 589, line 1080), $\kappa = k / (\rho C_p)$ is the thermal diffusivity, and $\theta = H(P_{T_i}) (T_i - T_\text{surf}) / (R T_i^2)$ is the effective Frank-Kamenetskii parameter (Foley and Bercovici 2014, sec. 2.2, p. 582, line 287) evaluated with activation enthalpy $H(P_{T_i}) = E_a + P_{T_i} V_a$. In the numerical implementation, the thermodynamic properties $\rho$, $g$, $\alpha$, and $\kappa$ are evaluated as column-averaged layer quantities. The effective Rayleigh number is defined at the interior viscosity $\mu_i$:

$$
\mathrm{Ra}_\text{eff} = \frac{\rho g \alpha (T_i - T_\text{surf}) d^3}{\kappa \mu_i}
$$

Because both $\mathrm{Ra}_\text{eff}$ and $\theta$ scale linearly with the total temperature drop $\Delta T = T_i - T_\text{surf}$, the ratio $\mathrm{Ra}_\text{eff} a_\text{rh} / \theta$ depends only on the rheological temperature scale $\Delta T_\text{rh} = a_\text{rh} R T_i^2 / H(P_{T_i})$. This dependence produces scale invariance with respect to the total layer temperature contrast. Consequently, convective driving stress $\tau_d$ does not vanish through $\Delta T \to 0$ alone; instead, it is modulated by the convective activity indicator $w_\text{active}$.

### 4.1 Heating Mode and Evolutionary Context

The $2/3$ velocity scaling law applies to bottom-heated stagnant-lid convection (Dumoulin et al. 1999; Foley and Bercovici 2014, p. 587, lines 850-855). In planetary evolution, cooling terrestrial planets start with a superheated core, creating a large core-mantle boundary temperature difference that drives strong basal heating during early post-magma-ocean evolution (Thiriet et al. 2019, pp. 139-140, sec. 1).

As the core cools, and radiogenic heating decays, the mantle heating regime transitions from bottom-dominated to predominantly internal heating (Thiriet et al. 2019, p. 140). For purely internally heated convection, Solomatov and Moresi (2000, p. 21804, eq. 13; p. 21805, Table 8) obtain a $1/2$ velocity scaling exponent:

$$
v_m = a_u \frac{\kappa}{d} \left( \frac{\mathrm{Ra}_i a_\text{rh}}{\theta} \right)^{1/2}
$$

with $a_u = 0.385 \pm 0.003$, for $n = 1$, and $a_\text{rh} = 2.4 \pm 0.2$. In both the bottom-heated and internally heated scalings, the temperature contrast $\Delta T$ enters the definitions of Rayleigh number and Frank-Kamenetskii parameter symmetrically, cancelling in the ratio $\mathrm{Ra} / \theta$. For an Earth reference state, the internally heated scaling predicts convective driving shear stress $\tau_d \approx 0.058\text{ MPa}$ (or $0.038\text{ MPa}$ with unscaled $a_\text{rh} = 1.0$), approximately four times lower than the bottom-heated estimate ($0.232\text{ MPa}$). Consequently, evaluating $\tau_d$ with the bottom-heated $2/3$ scaling represents an upper estimate on convective shear stress during later, internally heated epochs.

### 4.2 Validated Parameter Ranges and Mushy Extrapolation

Foley and Bercovici (2014, sec. 4.2, p. 589; Table 2, p. 593; Table 3, p. 594; Figs. 8-9) calibrate and validate the scaling laws over:

$$
\mathrm{Ra}_0 \in [10^5, 5 \times 10^7]
$$

In a partially molten, or mushy mantle, where bulk viscosity drops to $\eta_\text{bulk} \sim 10^2 - 10^7\text{ Pa s}$, the effective Rayleigh number reaches $\mathrm{Ra}_\text{eff} \sim 10^{19} - 10^{25}$, well above the numerical calibration domain of solid-state boundary-layer theory. Applying the scaling to mushy interiors represents an extrapolation, and Solomatov and Moresi (2000, sec. 5.3, p. 21803) note that the velocity scaling law can change at very high Rayleigh numbers, when convective plumes disconnect from the lid base.

### 4.3 Rheological Parameter Calibration

In the velocity scaling, $a_\text{rh} = 1.3$ and $C_4 = 0.125$ are the empirical calibration constants fitted jointly by Foley and Bercovici (2014, sec. 4.2, p. 589, line 1080) for their theory-curve constant-healing ($E_h = 0$) damage models. Every numerical model in Foley and Bercovici (2014) includes dynamic grain damage; FB2014 provides no pure damage-free numerical fit. The calibrated constant $a_\text{rh} = 1.3$ is lower than the standard stagnant-lid value ($a_\text{rh} \approx 2$; Solomatov and Moresi 2000; Korenaga 2009, cited in Foley and Bercovici 2014, p. 586) because less effective damage means less of the high-viscosity lid participates in convection (Foley and Bercovici 2014, p. 586). For temperature-dependent healing, Foley and Bercovici (2014, sec. 5.2, p. 594-595) find $a_\text{rh} \approx 1.82$.

Aragog adopts $C_4 = 0.125$ and $a_\text{rh} = 1.3$ as a unified parameter package calibrated against sub-lid stresses. In the Foley and Bercovici (2014) theory, grain damage enters solely through the effective viscosity $\mu_\text{eff} = \mu_i (A_i / A_0)^{-m}$, which reduces to the single-phase or mixture viscosity when damage is absent ($A_i = A_0$). The convective interior velocity scaling thus holds without damage mechanics. The sensitivity to $a_\text{rh}$ is moderate: adopting $a_\text{rh} = 2.0$ instead of $1.3$ would increase convective velocity $v_m$ and driving stress $\tau_d$ by a factor of $(2.0 / 1.3)^{2/3} \approx 1.33$.

This parameter enters the interior convective velocity $v_m$, and differs on purpose from the configured `lid_contrast_coeff` ($a = 2.2$; Solomatov 1995; Solomatov and Moresi 2000, sec. 5.1, p. 21800), which defines the thermal boundary isotherm of the rigid lid ($T_\text{lid} = T_i - a \Delta T_\text{rh}$).

### 4.4 Diagnostic Quantities and Force Balance

The model outputs the rheological sublayer buoyancy stress as a diagnostic:

$$
\tau_\text{buoy} = \rho g \alpha \Delta T_\text{rh} \delta_\text{rh}
$$

where $\delta_\text{rh}$ is the rheological sublayer thickness. The sublayer buoyancy stress $\tau_\text{buoy}$ and the force-balance ratio $\tau_d / \tau_\text{buoy}$ are tracked in the stagnant lid diagnostic state dictionary `lid_state`. For the Earth reference state ($\rho = 3300\text{ kg m}^{-3}$, $g = 9.81\text{ m s}^{-2}$, $\alpha = 3 \times 10^{-5}\text{ K}^{-1}$, $T_i = 1600\text{ K}$, $\Delta T_\text{rh} \approx 71\text{ K}$, $\delta_\text{rh} \approx 30.8\text{ km}$), $\tau_\text{buoy} \approx 2.12\text{ MPa}$, $\tau_d \approx 0.232\text{ MPa}$, and $\tau_d / \tau_\text{buoy} \approx 0.11$. In addition, the shifted soft-maximum convective velocity $v_i$, evaluated over the upper mantle, is retained as a diagnostic profile quantity.

## 5. Minimum Yield Closure

The effective solid viscosity combines diffusion creep and plastic yielding through the minimum formulation implemented in `aragog.rheology_lid.compute_effective_viscosity` (Moresi and Solomatov 1998, eqs. 13-14, p. 672; Tackley 2000, eq. 8, p. 4):

$$
\eta_\text{eff} = \min(\eta_\text{diff}, \eta_y) = \eta_\text{diff} \min\left(1, \frac{\tau_{y,\text{lid}}}{\tau_d}\right)
$$
where Moresi and Solomatov (1998, eqs. 13-14, p. 672) define:
$$
\eta_\text{yield} = \frac{\tau_\text{yield}}{D}, \quad \eta = \begin{cases} \eta_\text{creep}, & \tau_\text{creep} < \tau_\text{yield} \\ \eta_\text{yield}, & \tau_\text{creep} \ge \tau_\text{yield} \end{cases}
$$
Here $D = 2 \dot{\epsilon}_\text{lid} = \tau_d / \eta_\text{diff}$ is the convective strain-rate invariant of the lid, giving $\eta_y = \tau_{y,\text{lid}} \eta_\text{diff} / \tau_d$. Below yield ($\tau_d < \tau_{y,\text{lid}}$), $\eta_\text{eff} = \eta_\text{diff}$ exactly, preserving rigid stagnant-lid strength. Above yield ($\tau_d \ge \tau_{y,\text{lid}}$), the lid softens in proportion to $\tau_{y,\text{lid}} / \tau_d$. An alternative harmonic-mean formulation $\eta_\text{eff} = \eta_\text{diff} \eta_y / (\eta_\text{diff} + \eta_y)$ was investigated by Foley and Becker (2009, eqs. 7-8, p. 3) and Foley and Bercovici (2014, sec. 8.2).

The effective solid viscosity through the radial column is determined by applying the closure viscosity within the cold boundary layer using the smooth lid mask:

$$
\log_{10} \eta_\text{solid} = w_\text{lid} \log_{10} \eta_\text{eff} + (1 - w_\text{lid}) \log_{10} \eta_\text{diff}
$$

Inside the cold lid ($w_\text{lid} \to 1$), $\eta_\text{solid} \to \eta_\text{eff}$. In the warm convective interior ($w_\text{lid} \to 0$), $\eta_\text{solid} \to \eta_\text{diff}$. This formulation prevents lid yielding from altering the interior convective mantle (Foley and Becker 2009, sec. 3.2.2).

The rheological temperature scale $\Delta T_\text{rh} = a R T_i^2 / E$ governs the sublayer temperature drop and thickness. Aragog sets the default lid contrast coefficient to $a = 2.2$ (Solomatov 1995; Tackley 2000). While Foley and Bercovici (2014) adopt $a_\text{rh} = 1.3$ to $1.82$ to account for dynamic grain damage softening, and parameterised models such as Foley and Smye (2018) use $a_\text{rh} = 2.5$, the value $a = 2.2$ represents the standard asymptotic coefficient for Newtonian diffusion creep without grain damage.

Because the minimum closure is continuous at $\tau_d = \tau_{y,\text{lid}}$, $\eta_\text{eff}$ is everywhere finite and strictly non-increasing with driving stress. In one dimension, a mobile lid represents convective lid thinning at the interior strain rate rather than horizontal plate subduction.

With the lid creep strain rate, a yielded lid weakens only by the factor $\tau_{y,\text{lid}} / \tau_d$. Consequently, plastic yielding produces a negligible effect on the thermal solution for moderate convective driving stresses. In the `cold_top_lid` test fixture, the top node viscosity decreases from $8.3 \times 10^{24}\text{ Pa s}$ to $6.9 \times 10^{24}\text{ Pa s}$ when $\tau_d / \tau_{y,\text{lid}}$ increases up to $1.2$. Across stress ratios between $0.8$ and $1.2$, the maximum absolute entropy change $\max |\Delta S|$ is $0\text{ J kg}^{-1}\text{ K}^{-1}$ over $0.1\text{ yr}$. Mobilising a cold lid with diffusion creep viscosity $\eta_\text{diff} \sim 10^{24}\text{ Pa s}$ requires a stress ratio $\tau_d / \tau_{y,\text{lid}}$ of order $10^6$ (an estimate).

When convective vigor ceases ($v_i \to 0$), convective heat transport drops below $f_\text{conv,min}$. In this limit, $w_\text{active} \to 0$, the lid closure remains inactive, and no division by zero occurs.

The closure is explicit and non-iterative, guaranteeing deterministic evaluation and stable derivatives in numerical solvers.

## 6. Interaction with the Eddy-Diffusivity Floor

PROTEUS enforces an eddy-diffusivity floor $\kappa_{h,\text{floor}} = 10\text{ m}^2\text{ s}^{-1}$ in coupled runs to prevent numerical stagnation in the fluid magma ocean. The floor is ramped through the melt fraction:

$$
f(\phi) = \frac{1}{2} \left( 1 + \tanh\left( \frac{\phi - \phi_\text{rheo}}{\phi_\text{width}} \right) \right)
$$

With $\phi_\text{rheo} = 0.4$ and $\phi_\text{width} = 0.2$, the ramp evaluates to $f(0) = 0.018$ at zero melt fraction. In the solid state, this produces an effective diffusivity floor of $0.18\text{ m}^2\text{ s}^{-1}$.

Because a conductive lid exhibits a negative entropy gradient ($dS/dr < 0$), an unmasked eddy-diffusivity floor would inject an unphysical diffusivity five orders of magnitude larger than molecular thermal diffusivity ($\kappa \sim 10^{-6}\text{ m}^2\text{ s}^{-1}$). This would artificially homogenise the thermal lithosphere and destroy stagnant-lid formation.

When solid rheology is enabled, Aragog masks the eddy-diffusivity floor inside the lid:

$$
\kappa_{h,\text{floor,eff}} = \kappa_{h,\text{floor}} \cdot f(\phi) \cdot (1 - w_\text{lid})
$$

Inside the cold lid ($w_\text{lid} \to 1$), the floor is completely suppressed, preserving true molecular conduction. In the convective interior and the molten magma ocean ($w_\text{lid} \to 0$), the floor remains fully active.

## 7. Stress Closure Modes

The geometry of the strain rate closure is controlled by `stress_closure_mode`:

- `lid`:
    Boundary-layer stress closure. The driving stress $\tau_d = \eta_i v_i / \delta_\text{rh}$ is evaluated on the rheological sublayer. Yielding mobilises the lid nodes when convective vigor is sufficient.

- `local`:
    Local mixing-length strain rate closure. The strain rate is evaluated directly from local cell velocity and mixing length using the second invariant convention:

    $$
    \dot{\epsilon}_\text{local}(r) = \frac{|v_\text{unyielded}(r)|}{2 l(r)}
    $$

    Because unyielded Arrhenius viscosity inside a cold lid is very high, $v_\text{unyielded} \approx 0$ and $\dot{\epsilon}_\text{local} \approx 0$. Consequently, `local` mode cannot yield the cold lithosphere and operates solely as an interior weakening mechanism in vigorously convecting deep regions.

- `global`:
    Unsupported mode. Configuration validation rejects `global` and instructs users to specify `lid`.

In the baseline parameter schema, `stress_closure_mode` defaults to `'lid'` to activate the boundary-layer convective closure for stagnant-lid and mobile-lid regimes. Setting `stress_closure_mode = 'local'` limits yielding node-by-node based on the local convective strain rate.

## 8. Viscosity Blending

In partially molten regions ($0 < \phi < \phi_\text{rheo}$), Aragog evaluates viscosity in two steps:

1. **Melt fraction blend:**
    The effective solid viscosity $\eta_\text{eff}$ is blended smoothly with the liquid melt viscosity $\eta_\text{liquid}$ across the rheological transition threshold ($\phi_\text{rheo} = 0.4$) with transition width $\phi_\text{width}$ using a hyperbolic tangent weight:

    $$
    w = \frac{1}{2} \left( 1 + \tanh\left( \frac{\phi - \phi_\text{rheo}}{\phi_\text{width}} \right) \right)
    $$

    $$
    \log_{10} \eta_\text{mixed} = (1 - w) \log_{10} \eta_\text{eff} + w \log_{10} \eta_\text{liquid}
    $$

2. **Material property smoothing:**
    The mixed viscosity is combined with the single-phase branch:

    $$
    \log_{10} \eta_\text{single} = \begin{cases}
    \log_{10} \eta_\text{liquid}, & \text{if } \phi > \phi_\text{visc\_single} \\
    \log_{10} \eta_\text{eff}, & \text{otherwise}
    \end{cases}
    $$

    using the equation-of-state smoothing weight $w_\text{smth}$:

    $$
    \log_{10} \eta_\text{final} = w_\text{smth} \log_{10} \eta_\text{mixed} + (1 - w_\text{smth}) \log_{10} \eta_\text{single}
    $$

The two-stage logarithmic mixing rule originates from SPIDER numerical rheology formulation (`util.c`; Solomatov, 1995) to ensure smooth interpolation across mushy aggregate transitions.

The parameter `phi_visc_single` (default $0.5$) governs the single-phase cutoff threshold. It is strictly isolated to this viscosity blending routine (`entropy_phase.py:477`, `:486` and `jax/phase.py:710-716`). All twelve other $\phi > 0.5$ sites in the codebase remain unchanged at their independent physical thresholds:

- Equation-of-state and thermodynamic phase lookups: `eos/entropy.py:1098`, `:1178`; `entropy_phase.py:302`, `:373-374`, `:397`, `:408`; `jax/eos.py:662`, `:721`, `:729`; `jax/phase.py:497`.
- Convective mixing-flux factors: `solver/entropy_state.py:167`; `jax/phase.py:571`.

### Mass Fraction versus Volume Fraction Conventions

In Aragog, the melt fraction $\phi$ is tracked as a mass fraction ($w_\text{melt}$). In contrast, laboratory and geodynamic literature commonly parameterise the rheological transition in terms of melt volume fraction ($\phi_v$). For a 15% density contrast between solid matrix ($\rho_s \approx 3300\text{ kg m}^{-3}$) and basaltic melt ($\rho_l \approx 2800\text{ kg m}^{-3}$), a volume fraction of $\phi_v = 0.30$ corresponds to a mass fraction of $\phi_m \approx 0.267$:

$$
\phi_m = \frac{\phi_v \rho_l}{\phi_v \rho_l + (1 - \phi_v) \rho_s}
$$

(or $\phi_m \approx 0.28$ for an idealised 10% contrast where $\rho_l = 0.9 \rho_s$). The nominal threshold $\phi_\text{rheo} = 0.40$ in Aragog is defined directly in solver mass fraction.

## 9. Asymptotic Limits

The closure satisfies the following asymptotic limits:

1. **Zero driving stress ($\tau_d \to 0$):**
   $\tau_d / \tau_{y,\text{lid}} \to 0 \implies \eta_y = \eta_\text{diff}$, $\eta_\text{below} = \eta_\text{diff}$, $w_y = 0$, and $\eta_\text{eff} = \eta_\text{diff}$. The system exhibits pure Arrhenius diffusion creep without plastic weakening.
2. **Infinite yield stress ($\tau_y \to \infty$):**
   $\tau_d / \tau_{y,\text{lid}} \to 0 \implies \eta_y = \eta_\text{diff}$, $\eta_\text{below} = \eta_\text{diff}$, yielding never activates, and the lid remains permanently stagnant at diffusion-creep viscosity.
3. **Zero activation volume ($V_0 = 0$):**
   $H(P) = E_a$, recovering purely temperature-dependent Arrhenius creep.
4. **Infinite decay pressure ($P_\text{decay} \to \infty$):**
   $H(P) = E_a + P V_0$, recovering the linear activation volume formulation.
5. **Infinite Frank-Kamenetskii contrast ($\theta \to \infty$):**
   Recovers the asymptotic stagnant-lid regime of Solomatov (1995).
6. **No convective interior ($F_\text{conv} / F_\text{tot} < f_\text{conv,min}$ everywhere):**
   $w_\text{active} \to 0$, no interior node $T_i$ is delimited, and yielding remains inactive throughout the column.
7. **Wholly molten mantle ($\phi > \phi_\text{rheo}$ everywhere):**
   The mantle resides in the liquid regime; solid rheology and yielding are inactive.
8. **Disabled rheology (`enabled = false`):**
   Solvers bypass Arrhenius creep and plastic yielding, using constant solid viscosity $\eta_\text{solid} = 10^{\text{log10\_visc\_solid}}$.
9. **Extreme driving stress ($\tau_d / \tau_{y,\text{lid}} \gg 1$):**
   $w_y \to 1$ and $\eta_\text{eff} \to \eta_\text{yielded} = \min(\eta_\text{lid}, \eta_\text{diff})$, transitioning the column into fully mobile convective thinning.

## 10. Output Diagnostics and Conservation

### Reconciled Diagnostic Columns

The table below reconciles diagnostic quantities across model helpfiles and NetCDF snapshot files:

| Diagnostic Column | Description | Unit | Definition |
|:---|:---|:---:|:---|
| `T_pot` | Mantle potential temperature | K | Staggered temperature at shallowest node where $J_\text{conv} > J_\text{cond}$ |
| `boundary_layer_thickness` | Configured boundary layer thickness | m | Static configured surface boundary layer thickness (`surface_d`) |
| `RF_depth` | Rheological front depth | - | Dimensionless depth ($1 - r_\text{rf} / R_\text{outer}$) where melt fraction equals $\phi_\text{rheo} = 0.4$ |
| `visc_stag` | Staggered viscosity profile | Pa s | Radial viscosity profile on staggered nodes (recorded as `log10visc_s`) |
| `lid_thickness` | Diagnostic lid thickness | m | Physical thickness of the stagnant lid $d_\text{lid}$ |
| `lid_base_temperature` | Lid base temperature | K | Temperature $T_\text{lid}$ at the base of the stagnant lid |
| `lid_regime` | Convective regime indicator | - | $0.0$ for no lid (or liquid surface), $1.0$ for stagnant lid, $2.0$ for mobile lid |
| `tau_d` | Convective driving stress | Pa | Shear stress $\tau_d$ acting on the lid base |
| `tau_y_lid` | Lid base yield stress | Pa | Effective Byerlee yield stress $\tau_{y,\text{lid}}$ |
| `theta` | Frank-Kamenetskii contrast | - | Temperature contrast parameter $H(P_{T_i})(T_i - T_\text{surf}) / (R T_i^2)$ |
| `eta_contrast` | True Arrhenius contrast | - | Ratio $\eta_\text{surf} / \mu_i$ of surface Arrhenius viscosity to interior convecting viscosity $\mu_i$ |
| `interior_temperature` | Interior convective temperature | K | Representative interior temperature $T_i$ |
| `lid_cell_count` | Numerical lid resolution | - | Integer count of discrete radial cells spanning $d_\text{lid}$ |
| `energy_residual` | Energy conservation residual | J | Cumulative discrete energy balance check across mantle volume |

Established columns (`T_pot`, `boundary_layer_thickness`, and `RF_depth`) map to existing helpfile scan tables. The diagnostic `lid_thickness` denotes the physical stagnant lid thickness $d_\text{lid}$ dynamically determined from the rheological isotherm $T_\text{lid}$, solidus, or rheological front, differing from the static configured parameter `boundary_layer_thickness`. In-memory diagnostics report NaN for inactive rheology and stagnant lid quantities. When rheology is disabled, the rheological profiles ($\eta_\text{diff\_b}$, $\tau_\text{y\_b}$) and stagnant lid diagnostics ($d_\text{lid}$, $T_\text{lid}$, $T_i$, $\tau_d$, $\theta$, $\text{lid\_regime}$) are omitted from NetCDF snapshot files (`_int.nc`) so that inactive modules export only finite variables.

### Energy Conservation

Aragog integrates the energy equation in entropy formulation. The discrete energy balance residual:

$$
\Delta E_\text{res} = \int \rho T \frac{\partial S}{\partial t} dV - (H_\text{rad} + H_\text{tidal} - F_\text{surf} A_\text{surf} + F_\text{cmb} A_\text{cmb})
$$

is a numerical bookkeeping check. The entropy derivative $\partial S / \partial t$ is taken directly from the governing right-hand-side evaluations rather than discrete time differencing. Mixing-length theory does not incorporate viscous or plastic dissipation heating terms. Dissipation heating is therefore omitted from the thermal energy budget.

The diagnostic $\eta_\text{diff\_b}$ reflects pure solid diffusion creep and does not account for matrix breakdown or melt-lubricated granular flow occurring in partially molten aggregates.

## 11. Interface Boundary

Aragog defines explicit interfaces for solid-state rheological exchange: `eta_diff_b` is exported as a runtime radial array on the basic mesh, and `water_prefactor` is accepted as an optional radial array. These interfaces represent current physical fields and maintain strict modular separation from external atmospheric and volatile ledgers.

## 12. Rheology Parameters

All solid-state convection parameters are managed by `SolidRheologyParams` in Aragog and mirrored under `[interior_energetics.aragog.rheology]` in PROTEUS:

| `field` | unit | default as repr() | aragog TOML path | PROTEUS TOML path |
|:---|:---:|:---:|:---|:---|
| `enabled` | - | `False` | `[phase_solid].enabled` | `[interior_energetics.aragog.rheology].enabled` |
| `activation_energy` | J/mol | `300000.0` | `[phase_solid].activation_energy` | `[interior_energetics.aragog.rheology].activation_energy` |
| `activation_volume` | m^3/mol | `5e-06` | `[phase_solid].activation_volume` | `[interior_energetics.aragog.rheology].activation_volume` |
| `activation_volume_decay_pressure` | Pa | `inf` | `[phase_solid].activation_volume_decay_pressure` | `[interior_energetics.aragog.rheology].activation_volume_decay_pressure` |
| `arrhenius_t_ref` | K | `1600.0` | `[phase_solid].arrhenius_t_ref` | `[interior_energetics.aragog.rheology].arrhenius_t_ref` |
| `viscosity_max_log10` | log10 Pa s | `40.0` | `[phase_solid].viscosity_max_log10` | `[interior_energetics.aragog.rheology].viscosity_max_log10` |
| `water_prefactor` | - | `1.0` | `[phase_solid].water_prefactor` | `[interior_energetics.aragog.rheology].water_prefactor` |
| `yield_stress_c` | Pa | `50000000.0` | `[phase_solid].yield_stress_c` | `[interior_energetics.aragog.rheology].yield_stress_c` |
| `yield_stress_mu` | - | `0.6` | `[phase_solid].yield_stress_mu` | `[interior_energetics.aragog.rheology].yield_stress_mu` |
| `yield_stress_max` | Pa | `500000000.0` | `[phase_solid].yield_stress_max` | `[interior_energetics.aragog.rheology].yield_stress_max` |
| `stress_closure_mode` | - | `'lid'` | `[phase_solid].stress_closure_mode` | `[interior_energetics.aragog.rheology].stress_closure_mode` |
| `interior_flux_fraction` | - | `0.05` | `[phase_solid].interior_flux_fraction` | `[interior_energetics.aragog.rheology].interior_flux_fraction` |
| `lid_base_mode` | - | `'rheological'` | `[phase_solid].lid_base_mode` | `[interior_energetics.aragog.rheology].lid_base_mode` |
| `lid_base_temperature` | K | `1400.0` | `[phase_solid].lid_base_temperature` | `[interior_energetics.aragog.rheology].lid_base_temperature` |
| `lid_contrast_coeff` | - | `2.2` | `[phase_solid].lid_contrast_coeff` | `[interior_energetics.aragog.rheology].lid_contrast_coeff` |
| `lid_mask_width_cells` | - | `1.0` | `[phase_solid].lid_mask_width_cells` | `[interior_energetics.aragog.rheology].lid_mask_width_cells` |
| `phi_visc_single` | - | `0.5` | `[phase_solid].phi_visc_single` | `[interior_energetics.aragog.rheology].phi_visc_single` |
