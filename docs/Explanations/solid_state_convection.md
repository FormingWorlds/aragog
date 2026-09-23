# Solid-State Convection

Aragog simulates planetary mantle evolution in both molten and solid states. As the mantle crystallises and the melt fraction drops below the rheological transition threshold ($\phi_\text{rheo} = 0.4$), the interior transitions from liquid-state turbulent magma flow to solid-state creep. Aragog couples Arrhenius diffusion creep, Byerlee plastic yielding, and boundary-layer stress closure to model stagnant-lid and mobile-lid convection regimes within a 1D mixing-length framework.

## 1. Arrhenius Diffusion Creep

In the solid regime ($\phi < \phi_\text{rheo}$), the baseline solid matrix deforms by diffusion creep. The diffusion-creep viscosity follows an Arrhenius relation referenced to a surface reference state $(T_\text{ref}, P = 0)$:

$$
\eta_\text{diff}(T, P) = \eta_\text{solid} \cdot f_\text{water} \cdot \exp\left( \frac{H(P)}{R T} - \frac{H(0)}{R T_\text{ref}} \right)
$$

where:
- $\eta_\text{solid}$ is the reference solid viscosity at temperature $T_\text{ref} = 1600\text{ K}$ and zero pressure (`viscosity_solid`, default $10^{20}\text{ Pa s}$).
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

When $P_\text{decay} \to \infty$, $H(P)$ reduces to the linear form $H(P) = E_a + P V_0$. When $V_0 = 0$, $H(P)$ reduces to the temperature-only Arrhenius law $H(P) = E_a$.

### Numerical Evaluation

Evaluating the expression $V_0 P_\text{decay} (1 - \exp(-P / P_\text{decay}))$ directly when $P_\text{decay} = \infty$ yields an indeterminate $\infty \times 0$ form. In numerical kernels, the enthalpy is evaluated using the vectorized array-level form:

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

The model rheology is single-phase and does not include discontinuous jumps from phase changes (such as the ringwoodite to bridgmanite plus ferropericlase transition). The continuous model contrast along the adiabat must therefore remain below the total observed contrast to leave room for phase transition increments (Tackley, 1996). The formulation acts as a Newtonian diffusion-creep proxy at fixed water content and fixed grain size (both absorbed into $\eta_\text{solid}$). Defaults reflect dry olivine diffusion creep (Hirth and Kohlstedt, 2003).

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

Aragog identifies the convecting mantle region from the convective heat flux. The interior temperature $T_i$ and pressure $P_{T_i}$ are defined at the shallowest radial node where the convective heat flux exceeds a prescribed fraction of the total heat flux:

$$
\frac{F_\text{conv}}{F_\text{tot}} \ge f_\text{conv,min}
$$

where $f_\text{conv,min}$ is the interior flux fraction (`interior_flux_fraction`, default $0.05$). If no node satisfies this condition, no convective interior exists, no lid is delimited, and yielding remains inactive.

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

The diagnostic temperature marking the base of the stagnant lid is:

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

where $w_\text{mask}$ is the mask width parameter (`lid_mask_width_cells`, default $1.0$), and $\Delta T_\text{cell}$ is the local temperature step between adjacent radial nodes. Inside the cold lid ($T < T_\text{lid}$), $w_\text{lid} \to 1$. In the warm convecting interior ($T > T_\text{lid}$), $w_\text{lid} \to 0$.

## 4. Convective Driving Stress

Convective flow beneath the lid exerts a driving shear stress on the rheological sublayer at the lid base. In mixing-length theory, the convective velocity scales as $v \sim l^3 / \nu$, where $\nu = \eta / \rho$ is kinematic viscosity. The convective strain rate is $\dot{\epsilon} = v / (2 l)$. Consequently, the product $2 \eta \dot{\epsilon} \sim \rho v l$ is determined by convective buoyancy and is independent of viscosity. The convective driving stress $\tau_d$ is therefore stress-controlled:

$$
\tau_d = \frac{\eta_i v_i}{\delta_\text{rh}}
$$

where $\eta_i = \eta_\text{diff}(T_i, P_{T_i})$ is the interior solid viscosity, and $v_i$ is the convective velocity representative of the upper convecting mantle.

To prevent localised spikes or lower boundary layer velocities from contaminating $v_i$, Aragog evaluates $v_i$ as a smooth log-sum-exp maximum of the mixing-length velocity over the upper half of the convecting region (from the $T_i$ node to mid-depth of the convective domain).

The model also records the alternative stress scale evaluated over the full lid thickness $d_\text{lid}$:

$$
\tau_{d,\text{lid}} = \frac{\eta_i v_i}{d_\text{lid}}
$$

along with the sublayer buoyancy stress:

$$
\tau_\text{buoy} = \rho g \alpha \Delta T_\text{rh} \delta_\text{rh}
$$

The ratio $\tau_d / \tau_\text{buoy}$ is tracked as a diagnostic indicator of convective force balance.

## 5. Stress Closure and Over-Yield Regime Switch

In a stress-controlled system, the effective viscosity cannot be determined by an unconstrained harmonic mean. With prescribed driving stress $\tau_d$ and strain rate $\dot{\epsilon} = \tau_d / (2 \eta_\text{eff})$, substituting into the traditional harmonic blend gives:

$$
\frac{1}{\eta_\text{eff}} = \frac{1}{\eta_\text{diff}} + \frac{2 \dot{\epsilon}}{\tau_{y,\text{lid}}} = \frac{1}{\eta_\text{diff}} + \frac{\tau_d}{\eta_\text{eff} \tau_{y,\text{lid}}}
$$

Rearranging terms:

$$
\frac{1}{\eta_\text{eff}} \left( 1 - \frac{\tau_d}{\tau_{y,\text{lid}}} \right) = \frac{1}{\eta_\text{diff}} \implies \eta_\text{eff} = \eta_\text{diff} \left( 1 - \frac{\tau_d}{\tau_{y,\text{lid}}} \right)
$$

This expression possesses no positive real root when $\tau_d \ge \tau_{y,\text{lid}}$. Traditional iterative solvers diverge above yield not because of slow convergence, but because no solution exists to the equation.

### Two-Branch Regime Switch

To resolve this limitation, Aragog formulates yielding as a regime switch between a stagnant lid and a mobile convective lid:

1. **Sub-yield branch ($\tau_d < \tau_{y,\text{lid}}$):**
   The effective viscosity follows the closed harmonic form, floored at the mobile-lid viscosity $\eta_\text{lid}$ to maintain strict positivity and continuity:

$$
\eta_\text{below} = \max\left( \eta_\text{diff} \left( 1 - \frac{\tau_d}{\tau_{y,\text{lid}}} \right), \eta_\text{lid} \right)
$$

2. **Yielded branch ($\tau_d \ge \tau_{y,\text{lid}}$):**
   When the driving stress reaches or exceeds the yield stress, the lid yields plastically and deforms at the interior convective strain rate $\dot{\epsilon}_i = v_i / (2 \delta_\text{rh})$. The yielded lid viscosity is:

$$
\eta_\text{lid} = \frac{\tau_{y,\text{lid}}}{2 \dot{\epsilon}_i} = \frac{\tau_{y,\text{lid}} \delta_\text{rh}}{v_i}
$$

At exact yield ($\tau_d = \tau_{y,\text{lid}}$), substituting $\tau_d = \eta_i v_i / \delta_\text{rh}$ yields $\eta_\text{lid} = \eta_i$. The lid viscosity matches the interior viscosity at the transition.

### Continuous Regime Blend

The transition between branches is blended smoothly in logarithmic viscosity space using a hyperbolic tangent switch:

$$
w_y = \frac{1}{2} \left( 1 + \tanh\left( \frac{\tau_d / \tau_{y,\text{lid}} - 1}{w_\text{yield}} \right) \right)
$$

$$
\log_{10} \eta_\text{eff} = (1 - w_y) \log_{10} \eta_\text{below} + w_y \log_{10} \eta_\text{lid}
$$

where $w_\text{yield}$ is the transition width parameter (`yield_switch_width`, default $0.1$).

Because both branches evaluate to $\eta_\text{lid}$ at yield, $\eta_\text{eff}(\tau_d)$ is continuous, everywhere finite, and monotonically non-increasing over the full range $\tau_d / \tau_{y,\text{lid}} \in [0, 10]$. Plastic yielding is applied exclusively to nodes within the stagnant lid ($w_\text{lid} > 0$). In one dimension, a mobile lid represents convective lid thinning at the interior strain rate rather than horizontal plate subduction.

When convective vigor ceases ($v_i \to 0$), convective heat transport drops below $f_\text{conv,min}$. In this limit, no convective interior is delimited, the lid closure remains inactive, and no division by zero occurs.

The closure is explicit and non-iterative, guaranteeing deterministic evaluation and exact differentiability in numerical solvers.

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
  Obsolete mode. Configuration validation rejects `global` and instructs users to specify `lid`.

In the baseline parameter schema, `stress_closure_mode` defaults to `'local'`; upon enabling the boundary-layer convective closure, `'lid'` is the operational mode.

## 8. Two-Stage Viscosity Blending

In partially molten regions ($0 < \phi < \phi_\text{rheo}$), Aragog evaluates viscosity in two stages:

1. **Stage 1 (Melt fraction blend):**
   The effective solid viscosity $\eta_\text{eff}$ is blended smoothly with the liquid melt viscosity $\eta_\text{liquid}$ across the rheological transition threshold ($\phi_\text{rheo} = 0.4$) with transition width $\phi_\text{width}$ using a hyperbolic tangent weight:

$$
w = \frac{1}{2} \left( 1 + \tanh\left( \frac{\phi - \phi_\text{rheo}}{\phi_\text{width}} \right) \right)
$$

$$
\log_{10} \eta_\text{mixed} = (1 - w) \log_{10} \eta_\text{eff} + w \log_{10} \eta_\text{liquid}
$$

2. **Stage 2 (Material property smoothing):**
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

The parameter `phi_visc_single` (default $0.5$) governs the single-phase cutoff threshold. It is isolated to this viscosity blending routine and is distinct from the other twelve $\phi > 0.5$ sites in equation-of-state tables and thermal conductivity models, which remain untouched.

## 9. Asymptotic Limits

The closure satisfies the following asymptotic limits:

1. **Zero driving stress ($\tau_d \to 0$):**
   $\tau_d / \tau_{y,\text{lid}} = 0 \implies w_y = 0$ and $\eta_\text{eff} = \eta_\text{diff}$. The system exhibits pure Arrhenius conduction without yielding.
2. **Infinite yield stress ($\tau_y \to \infty$):**
   $\tau_d / \tau_{y,\text{lid}} \to 0$, yielding never activates, and the lid remains permanently stagnant.
3. **Zero activation volume ($V_0 = 0$):**
   $H(P) = E_a$, recovering purely temperature-dependent Arrhenius creep.
4. **Infinite decay pressure ($P_\text{decay} \to \infty$):**
   $H(P) = E_a + P V_0$, recovering the linear activation volume formulation.
5. **Infinite Frank-Kamenetskii contrast ($\theta \to \infty$):**
   Recovers the asymptotic stagnant-lid regime of Solomatov (1995).
6. **No convective interior ($F_\text{conv} / F_\text{tot} < f_\text{conv,min}$ everywhere):**
   No interior node $T_i$ is found; no stagnant lid is delimited, and yielding remains inactive throughout the column.
7. **Wholly molten mantle ($\phi > \phi_\text{rheo}$ everywhere):**
   The mantle resides in the liquid regime; solid rheology and yielding are inactive.
8. **Disabled rheology (`enabled = false`):**
   Solvers bypass Arrhenius creep and plastic yielding, using constant solid viscosity $\eta_\text{solid} = 10^{\text{log10\_visc\_solid}}$.
9. **Extreme driving stress ($\tau_d / \tau_{y,\text{lid}} \gg 1$):**
   $w_y \to 1$ and $\eta_\text{eff} \to \eta_\text{lid} = \tau_{y,\text{lid}} / (2 \dot{\epsilon}_i)$, transitioning the column into fully mobile convective thinning.

## 10. Output Diagnostics and Conservation

### Reconciled Diagnostic Columns

The table below reconciles diagnostic quantities across model helpfiles and NetCDF snapshot files:

| Diagnostic Column | Description | Unit | Definition |
|:---|:---|:---:|:---|
| `T_pot` | Mantle potential temperature | K | Mass-weighted convective interior temperature |
| `boundary_layer_thickness` | Thermal boundary layer thickness | m | Depth where conductive geotherm meets interior adiabat |
| `RF_depth` | Rheological front depth | m | Depth where melt fraction equals $\phi_\text{rheo} = 0.4$ |
| `visc_stag` | Stagnant lid viscosity | Pa s | Viscosity at the cold surface lid (recorded as `log10visc_s`) |
| `lid_thickness` | Diagnostic lid thickness | m | Physical thickness of the stagnant lid $d_\text{lid}$ |
| `lid_base_temperature` | Lid base temperature | K | Temperature $T_\text{lid}$ at the base of the stagnant lid |
| `lid_regime` | Convective regime indicator | - | $0.0$ for stagnant lid, $1.0$ for mobile lid |
| `tau_d` | Convective driving stress | Pa | Shear stress $\tau_d$ acting on the lid base |
| `tau_y_lid` | Lid base yield stress | Pa | Effective Byerlee yield stress $\tau_{y,\text{lid}}$ |

Established columns (`T_pot`, `boundary_layer_thickness`, and `RF_depth`) map to existing helpfile scan tables. Convective closure variables are exported unconditionally in NetCDF snapshot files (`_int.nc`) for model verification. When rheology is disabled, the basic diffusion creep diagnostic array $\eta_\text{diff\_b}$ reports $10^{\text{log10\_visc\_solid}}$ at all nodes.

### Energy Conservation

Aragog integrates the energy equation in entropy formulation. The discrete energy balance residual:

$$
\Delta E_\text{res} = \int \rho T \frac{\partial S}{\partial t} dV - (H_\text{rad} - F_\text{surf} A_\text{surf} + F_\text{cmb} A_\text{cmb})
$$

telescopes to machine precision over finite-volume cells. Mixing-length theory does not incorporate viscous or plastic dissipation heating terms. Dissipation heating is therefore omitted from the thermal energy budget.

The diagnostic $\eta_\text{diff\_b}$ reflects pure solid diffusion creep and does not account for matrix breakdown or melt-lubricated granular flow occurring in partially molten aggregates.

## 11. Interface Boundary

Aragog exports `eta_diff_b` and accepts `water_prefactor` as runtime arrays on the basic numerical mesh. These interfaces describe current physical state variables and maintain strict independence from external coupling modules.

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
| `yield_switch_width` | - | `0.1` | `[phase_solid].yield_switch_width` | `[interior_energetics.aragog.rheology].yield_switch_width` |
| `stress_closure_mode` | - | `'local'` | `[phase_solid].stress_closure_mode` | `[interior_energetics.aragog.rheology].stress_closure_mode` |
| `interior_flux_fraction` | - | `0.05` | `[phase_solid].interior_flux_fraction` | `[interior_energetics.aragog.rheology].interior_flux_fraction` |
| `lid_base_mode` | - | `'fixed'` | `[phase_solid].lid_base_mode` | `[interior_energetics.aragog.rheology].lid_base_mode` |
| `lid_base_temperature` | K | `1400.0` | `[phase_solid].lid_base_temperature` | `[interior_energetics.aragog.rheology].lid_base_temperature` |
| `lid_contrast_coeff` | - | `2.2` | `[phase_solid].lid_contrast_coeff` | `[interior_energetics.aragog.rheology].lid_contrast_coeff` |
| `lid_mask_width_cells` | - | `1.0` | `[phase_solid].lid_mask_width_cells` | `[interior_energetics.aragog.rheology].lid_mask_width_cells` |
| `phi_visc_single` | - | `0.5` | `[phase_solid].phi_visc_single` | `[interior_energetics.aragog.rheology].phi_visc_single` |
