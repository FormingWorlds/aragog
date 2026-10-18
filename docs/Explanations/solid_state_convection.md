# Solid-State Convection and Lithospheric Rheology

Aragog models thermal evolution from a molten magma ocean through a partially molten mush to a solidified rocky mantle. In the solid regime, mantle viscosity varies by many orders of magnitude with temperature and pressure, producing a cold, high-viscosity conductive boundary layer (stagnant lid) unless convective driving stresses exceed the lithospheric yield strength and mobilise the surface boundary layer. The modules `aragog.rheology` and `aragog.rheology_lid` provide the standalone Arrhenius creep, plastic yield, stagnant-lid diagnostic, and effective-viscosity kernels documented below.

**Status:** The functions in `aragog.rheology` and `aragog.rheology_lid` are standalone kernels with no caller in the time-stepping solver (`aragog.jax.phase.compute_mlt` and `aragog.solver.entropy_state.apply_kappah_floor` do not call them). On `SolidRheologyParams`, the fields `enabled`, `stress_closure_mode`, and `phi_visc_single` are validated and stored on the parameter container, and no kernel reads them directly (callers pass `stress_closure_mode` explicitly to `compute_effective_viscosity`).

## 1. Homologous-Temperature Diffusion Creep

In `aragog.rheology`, solid-state diffusion creep viscosity is evaluated using the homologous-temperature Arrhenius formulation (Yamazaki and Karato, 2001, eqs. 4-7; Abe, 1997, Table 1):

$$
\eta_\text{diff}(T, P) = \eta_\text{solid} \cdot f_\text{water} \cdot \exp\left( \frac{H(P)}{R T} - \frac{H(0)}{R T_\text{ref}} \right) = \eta_\text{solid} \cdot f_\text{water} \cdot \exp\left[ g_\text{hom} \left( \frac{T_\text{sol}(P)}{T} - \frac{T_\text{sol}(0)}{T_\text{ref}} \right) \right]
$$

where:

- $\eta_\text{solid}$ is the reference solid viscosity at temperature $T_\text{ref} = 1600\text{ K}$ and zero pressure (`viscosity_solid`, default $10^{21}\text{ Pa s}$).
- $f_\text{water}$ is the hydration weakening prefactor (`water_prefactor`, default $1.0$).
- $R = 8.314462618\text{ J mol}^{-1}\text{ K}^{-1}$ is the universal gas constant.
- $g_\text{hom}$ is the dimensionless homologous activation parameter (`g_homologous`, default $25.0$).
- $T_\text{sol}(P)$ is the local mantle solidus temperature at pressure $P$.
- $H(P) = g_\text{hom} R T_\text{sol}(P)$ is the pressure-dependent activation enthalpy.

### Homologous Activation Enthalpy and Linear Fallback

Under constant activation volume ($V_a$), activation enthalpy grows linearly with pressure as $H(P) = E_a + P V_a$. In deep planetary mantles, linear extrapolation to core-mantle boundary pressures ($135\text{ GPa}$ in Earth) produces viscosity increases exceeding nine orders of magnitude along an adiabat, locking the deep mantle into an unphysical rigid state.

Mineral physics measurements and melting-point scaling relations show that the effective activation enthalpy scales with the local melting (solidus) temperature $T_\text{sol}(P)$, such that the activation volume $V_a(P) = \partial H / \partial P = g_\text{hom} R (dT_\text{sol} / dP)$ decreases naturally as the solidus slope flattens at high pressure (Yamazaki and Karato, 2001). Along any constant homologous fraction $T(P) = f \cdot T_\text{sol}(P)$, the viscosity ratio relative to the solidus is invariant with pressure:

$$
\frac{\eta_\text{diff}(f \cdot T_\text{sol}(P), P)}{\eta_\text{diff}(T_\text{sol}(P), P)} = \exp\left[ g_\text{hom} \left( \frac{1}{f} - 1 \right) \right]
$$

For $f = 0.9$, this ratio equals $\exp(g_\text{hom} / 9) = 16.083$ at $g_\text{hom} = 25$ (and $4.738$ at $g_\text{hom} = 14$) at all depths.

When an explicit solidus profile $T_\text{sol}(P)$ is supplied to `compute_arrhenius_viscosity` or `compute_stagnant_lid_state`, the caller must also supply the zero-pressure reference solidus $T_\text{sol}(0)$ (`t_sol_0`) from the same solidus source. When `solidus_temperature` is `None` (such as in table-free constant-property benchmarks), `compute_arrhenius_enthalpy`, `compute_arrhenius_viscosity`, and `compute_stagnant_lid_state` evaluate the linear solidus fallback:

$$
T_\text{sol}(P) = T_{\text{sol},0} + \frac{dT_\text{sol}}{dP} P
$$

with reference zero-pressure solidus $T_{\text{sol},0} = 1400.0\text{ K}$ (`t_sol_0`) and linear solidus slope $dT_\text{sol} / dP = 2.4 \times 10^{-8}\text{ K Pa}^{-1}$ (`dt_sol_dp`, corresponding to $24\text{ K GPa}^{-1}$). At $g_\text{hom} = 25.0$ and $T_{\text{sol},0} = 1400.0\text{ K}$, the zero-pressure activation enthalpy is $H(0) = 291\text{ kJ mol}^{-1}$, close to the standard olivine diffusion-creep activation energy $E_a = 300\text{ kJ mol}^{-1}$ (Karato and Wu, 1993). The linear defaults $T_{\text{sol},0} = 1400.0\text{ K}$ and $dT_\text{sol}/dP = 24\text{ K GPa}^{-1}$ approximate the whole-mantle secant of the SPIDER phase-boundary solidus table ($1429\text{ K}$ at $0\text{ GPa}$ to $4595\text{ K}$ at $135\text{ GPa}$, slope $23.5\text{ K GPa}^{-1}$); because the physical solidus is concave downward, the linear form is about $30\%$ low at $25\text{ GPa}$ ($2000\text{ K}$ versus $2969\text{ K}$) and serves table-free constant-property benchmarks only. To prevent float64 overflow in cold conductive surface layers, `compute_arrhenius_viscosity` clips the combined viscosity in $\log_{10}$ space at `viscosity_max_log10` (default $40.0$) after applying `water_prefactor`, and returns `NaN` for non-positive temperatures ($T \le 0$).

### Enthalpy Values on the SPIDER Solidus

The table below reports the solidus temperature $T_\text{sol}(P)$ from the SPIDER equation-of-state table (`solidus_P-S.dat`) and the resulting homologous activation enthalpy $H(P) = g_\text{hom} R T_\text{sol}(P)$ over terrestrial mantle pressures:

| Pressure $P$ (GPa) | $T_\text{sol}(P)$ (K) | $H(P)$ at $g_\text{hom} = 14$ (kJ/mol) | $H(P)$ at $g_\text{hom} = 25$ (kJ/mol) | $\eta(0.9 T_\text{sol}) / \eta(T_\text{sol})$ at $g_\text{hom} = 25$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 1428.9 | 166.3 | 297.0 | 16.083 |
| 25 | 2969.0 | 345.6 | 617.1 | 16.083 |
| 50 | 3668.1 | 427.0 | 762.5 | 16.083 |
| 100 | 4270.5 | 497.1 | 887.7 | 16.083 |
| 135 | 4595.2 | 534.9 | 955.2 | 16.083 |

### Viscosity-Contrast-Dependent Mixing-Length Slope

In the viscous mixing-length formulation, the effective mixing length $l_v(r)$ near the mantle boundaries is given by:

$$
l_v(r) = \min\big( s_\text{bot} (r - r_\text{in}), \; s_\text{top}(\theta) (r_\text{out} - r) \big)
$$

where $s_\text{bot}$ is the lower boundary slope (`mlt_bottom_slope`, default $1.0$) and $s_\text{top}(\theta)$ is the viscosity-contrast-dependent upper boundary slope evaluated by `compute_mlt_top_slope`:

$$
s_\text{top}(\theta) = s_{\text{top},\infty} + (1 - s_{\text{top},\infty}) \exp\big( -c_\theta \max(\theta, 0) \big)
$$

with decay rate $c_\theta = 0.15$ (`c_theta`) and stagnant-lid asymptotic slope $s_{\text{top},\infty} = 0.22$ (`s_top_inf`, default $0.22$). In the isoviscous limit ($\theta \le 0$), $s_\text{top}(0) = 1.0$, recovering the symmetric mixing-length profile. In the strong stagnant-lid regime ($\theta \gg 8$), $s_\text{top}(\theta)$ approaches $s_{\text{top},\infty}$. Because the viscous eddy diffusivity scales as $l^4$, `viscous_mixing_length_factor` multiplies the viscous velocity and diffusivity by $(l_v / l)^4$ at interior nodes while keeping $1.0$ at the boundary nodes where $l = 0$.

## 2. Plastic Yielding Criterion

To model lithospheric failure and mobile-lid convection, `compute_yield_stress` implements a Byerlee frictional yield criterion with a ductile stress ceiling (Byerlee, 1978; Moresi and Solomatov, 1998, p. 672, eq. 9; Tackley, 2000, p. 4, eq. 9):

$$
\tau_y(P) = \max\big(0, \; \min(c + \mu P, \tau_\text{max})\big)
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

Under simple shear flow with velocity gradient $\partial v / \partial y = v / l$, the strain rate tensor has components $\dot{\epsilon}_{xy} = \dot{\epsilon}_{yx} = \frac{1}{2} v / l$, which yields $\dot{\epsilon} = v / (2 l)$ and second invariant $D = 2 \dot{\epsilon} = v / l$.

### Physical Validity and Ceiling Depth

Byerlee frictional failure governs brittle deformation in the cold shallow lithosphere. At higher temperatures and deeper mantle levels, ductile deformation mechanisms limit shear strength, and the ceiling $\tau_\text{max}$ acts as a proxy for this ductile limit (Moresi and Solomatov, 1998, p. 672).

With $c = 50\text{ MPa}$ and $\mu = 0.6$, the yield stress reaches the $500\text{ MPa}$ ceiling at pressure:

$$
P_\text{ceil} = \frac{\tau_\text{max} - c}{\mu} = \frac{500\text{ MPa} - 50\text{ MPa}}{0.6} = 750\text{ MPa} = 0.75\text{ GPa}
$$

On Earth, $0.75\text{ GPa}$ corresponds to approximately $23\text{ km}$ depth. Because thermal lithospheres are typically thicker than $23\text{ km}$, the yield stress at the base of the lid is commonly set by the ceiling $\tau_\text{max}$. The scalar $\tau_{y,\text{lid}}$ denotes the yield stress evaluated at the lid base pressure and governs lid mobilisation. The local profile $\tau_y(P)$ is evaluated by `compute_yield_stress`.

## 3. Stagnant Lid Identification

In a convecting interior beneath a cold surface, convective heat transport dominates in the interior, whereas molecular conduction governs heat loss through the cold surface lid.

### Interior Convective State

`compute_stagnant_lid_state` identifies convective activity and the convecting interior state from the convective flux fraction $f_k = \max(F_{\text{conv}, k}, 0) / |F_{\text{tot}, k}|$. Convective activity across the column is measured relative to the interior flux threshold $f_\text{conv,min} \in [0.05, 0.5]$ (`interior_flux_fraction`, default $0.05$) with a fixed transition width of $0.01$:

$$
w_{\text{conv}, k} = \frac{1}{2} \left( 1 + \tanh\left( \frac{f_k - f_\text{conv,min}}{0.01} \right) \right)
$$

so that a zero-flux conductive node ($f_k = 0$) carries a weight of at most $\frac{1}{2}(1 - \tanh(5)) \approx 4.5 \times 10^{-5}$. From the interior peak convective flux fraction $f_\text{int,max} = \max_{1 < k < N} f_k$, `compute_stagnant_lid_state` evaluates the mesh-independent convective activity indicator $w_\text{active} = \frac{1}{2}(1 + \tanh((f_\text{int,max} - f_\text{conv,min}) / 0.01))$ and the shallowest-crossing boundary sum $W_\text{sum} = \sum_k w_{\text{conv}, k} \prod_{j < k} (1 - w_{\text{conv}, j})$ (with $w_\text{weight} = \tanh(W_\text{sum} / 10^{-6})$ and `is_convecting` requiring $W_\text{sum} > 10^{-6}$ and $w_\text{active} > 0.05$).

To evaluate the interior temperature $T_i$, pressure $P_{T_i}$, radius $r_i$, and interior viscosity $\mu_i$ where convection dominates relative to the interior peak $f_\text{int,max} = \max_{1 < k < N} f_k$ (with scale $f_\text{int,scale} = \max(f_\text{int,max}, f_\text{conv,min})$ and threshold $f_{T_i} = 0.9 f_\text{int,scale}$), `compute_stagnant_lid_state` forms the interior selector on non-boundary nodes ($1 < k < N$). For an error-function boundary-layer profile $T(z) = 300 + 1300\,\mathrm{erf}(z / \delta)$ with a $1300\text{ K}$ surface-to-interior temperature contrast and $\Delta T_\text{rh} = 70.95\text{ K}$ ($H = 300\text{ kJ mol}^{-1}$ at $1600\text{ K}$), the onset threshold $f = 0.05$ ($z / \delta = 0.2265$, $T = 626.6\text{ K}$) sits $13.7\,\Delta T_\text{rh}$ below the interior, whereas $f = 0.9$ ($z / \delta = 1.517$, $T = 1558.6\text{ K}$) sits $0.584\,\Delta T_\text{rh}$ below the interior:

$$
w_{T_i, k} = \frac{1}{2} \left( 1 + \tanh\left( \frac{f_k - f_{T_i}}{0.02 f_\text{int,scale}} \right) \right) w_{\text{conv}, k} I_{1 < k < N}
$$

The shallowest convection-dominated node from the surface is isolated by the cumulative product indicator $W_k = w_{T_i, k} \prod_{j < k} (1 - w_{T_i, j})$ and normalised to $\tilde{W}_k$ (falling back when $\sum_m W_m \to 0$ to the shallowest interior node with active convection from $w_\text{conv,int} = \max(w_{\text{conv}, k} - w_{\text{conv}, 0}, 0) I_{1 < k < N}$, or to the deepest interior node $k = 2$ when no interior node convects), from which interior state variables are evaluated smoothly:

$$
T_i = \sum_{k=1}^N \tilde{W}_k T_k, \quad P_{T_i} = \sum_{k=1}^N \tilde{W}_k P_k
$$

Because the selector places $T_i$ at the node under the lid, its accuracy is bounded by the radial mesh resolution across the thermal boundary layer: when the lid is one cell thick (for example on a 100-node post-magma-ocean profile where $T_i = 1423.0\text{ K}$ against $1736.9\text{ K}$ one node deeper), $T_i$ lies in the rheological sublayer.

When an optional mixture viscosity profile `viscosity_mixture` ($\eta_\text{mix}$) is supplied to `compute_stagnant_lid_state`, the convecting interior viscosity $\mu_i$ is evaluated using the smooth activity weight $w_\text{weight}$ in logarithmic space:

$$
\log_{10} \mu_i = w_\text{weight} \sum_{k=1}^N \tilde{W}_k \log_{10} \eta_{\text{mix}, k} + (1 - w_\text{weight}) \log_{10} \eta_\text{diff}(T_i, P_{T_i})
$$

When `viscosity_mixture` is `None`, $\mu_i$ equals the solid Arrhenius interior viscosity $\eta_i = \eta_\text{diff}(T_i, P_{T_i})$. When $f_\text{int,max}$ lies more than $0.01\,\mathrm{arctanh}(0.9) \approx 0.0147$ below $f_\text{conv,min}$ (for example $f_k \le 0.035$ at $f_\text{conv,min} = 0.05$), $w_\text{active} \le 0.05$, `is_convecting` and `has_lid` are `False`, no lid is delimited, and yielding remains inactive across any mesh resolution $N$.

### Rheological Temperature Scale

The local temperature scale over which diffusion-creep viscosity changes by a factor of $e$ is the Frank-Kamenetskii rheological temperature scale:

$$
\Delta T_\text{rh} = \frac{R T_i^2}{H(P_{T_i})}
$$

This scale is evaluated at the pressure $P_{T_i}$ of the interior node to avoid circular dependence on the lid thickness.

### Asymptotic Validity Conditions

Boundary-layer stagnant-lid scaling requires two asymptotic conditions (Solomatov, 1995; Foley and Bercovici, 2014, pp. 582, 586):

1. Small rheological temperature scale relative to interior temperature: $\Delta T_\text{rh} / T_i = 1 / (g_\text{hom} T_\text{sol}(P_{T_i}) / T_i) \ll 1$.
2. Viscosity contrast across the cold boundary layer exceeding $\exp(8) \approx 3 \times 10^3$ and across the convecting layer exceeding $\sim 10^4$ (Solomatov, 1995, p. 268, sec. V; p. 270, sec. VI, Fig. 4), quantified by the Frank-Kamenetskii contrast parameter:

    $$
    \theta = \frac{H(P_{T_i}) (T_i - T_\text{surf})}{R T_i^2} = \frac{T_i - T_\text{surf}}{\Delta T_\text{rh}} \ge 9
    $$

The surface-to-interior viscosity contrast $\eta_\text{diff}(T_\text{surf}, P_\text{surf}) / \mu_i$ is substantially larger than $\exp(\theta)$ because of non-linear Arrhenius curvature and partial melt in mushy interiors. `compute_stagnant_lid_state` reports both $\theta$ and `eta_contrast`. If $\theta < 9$, the convective regime lies outside the asymptotic stagnant-lid window.

### Lid Base and Sublayer Thickness

The parameter `lid_base_mode` governs the definition of the stagnant lid base temperature $T_{\text{lid,iso}}$:

- `'fixed'`: The lid base is defined by a user-specified isotherm $T_{\text{lid,iso}} = T_\text{base}$ set by `lid_base_temperature` (default $1400.0\text{ K}$).
- `'rheological'`: The lid base is evaluated dynamically from the Frank-Kamenetskii temperature scale:

    $$
    T_{\text{lid,iso}} = T_i - a \Delta T_\text{rh}
    $$

    where $a = 2.2$ is the lid contrast coefficient (`lid_contrast_coeff`). This coefficient comes from boundary-layer theory for steady-state Newtonian convection with exponential temperature-dependent viscosity in the asymptotic limit $\theta \gg 1$ (Solomatov, 1995; Tackley, 2000).

The physical lid base location is identified as the shallowest depth among:

1. The rheological isotherm where $T = T_{\text{lid,iso}}$.
2. The intersection of the geotherm with the mantle solidus.
3. The rheological crystallisation front where $\phi = \phi_\text{rheo} = 0.4$.

When the entire column lies below the lid-base criterion so that the interior boundary indicator sum $W_\text{base,sum} \to 0$, `r_lid_base` anchors at the inner boundary radius `r[0]` (the core-mantle boundary), giving $d_\text{lid} = r_\text{outer} - r_\text{inner}$.

The thickness of the rheological sublayer that accommodates deformation at the lid base is:

$$
\delta_\text{rh} = \frac{\Delta T_\text{rh}}{|dT/dr|_\text{lid\_base}}
$$

where $|dT/dr|_\text{lid\_base}$ is the $\tilde{W}_{\text{base}, k}$-weighted radial temperature gradient evaluated from nodal central differences (`xp.gradient(T, r)`). To prevent division by zero in isothermal or near-adiabatic profiles, the nodal temperature gradient is floored at a minimum positive value corresponding to a $1\text{ K}$ difference over the minimum grid spacing.

### Smooth Lid Mask and Resolution

Because $|dT/dr|_\text{lid\_base}$ is computed by a nodal central difference (`xp.gradient(T, r)`), $\delta_\text{rh}$ depends on the radial mesh spacing around the lid base. In addition, the `lid_state` dictionary records `lid_cell_count` ($\sum_k w_{\text{lid}, k}$) over the physical lid thickness $d_\text{lid}$ to track numerical resolution within the lid.

Nodes belonging to the lid are weighted by a smooth hyperbolic tangent mask:

$$
w_\text{lid} = (1 - w_\text{interior}) \cdot w_\text{active}
$$

where $w_\text{interior} = \max(w_\text{hot,iso}, w_\text{hot,sol}, w_{\text{hot},\phi})$ transitions smoothly over the local temperature step $\Delta T_\text{cell} = \max(|\nabla T|, 1.0\text{ K})$ scaled by `lid_mask_width_cells` (default $1.0$). Inside the cold solid lid, $w_\text{lid} \to 1$. In the warm or partially molten interior, $w_\text{lid} \to 0$.

## 4. Convective Driving Stress

Convective flow beneath the stagnant lid exerts a driving shear stress on the base of the lid. `compute_stagnant_lid_state` implements the stagnant-lid convective shear stress scaling of Foley and Bercovici (2014, eqs. 26 and 28, pp. 586-588, doi:10.1093/gji/ggu275):

$$
\tau_d = \frac{2 \mu_i v_m}{d} w_\text{active}
$$

where $d = r_\text{outer} - r_\text{inner}$ is the mantle thickness and $\mu_i$ is the convecting interior dynamic viscosity. The factor of 2 arises from horizontal shear between the lid base ($u = v_m$) and mid-depth ($u = 0$ at $z = d/2$), as derived by Foley and Bercovici (2014, pp. 584, 588).

The interior convective velocity $v_m$ follows the boundary-layer scaling for bottom-heated stagnant-lid convection (Dumoulin et al., 1999, p. 12760, sec. 2; Foley and Bercovici, 2014, pp. 586-587, eq. 26):

$$
v_m = \frac{\kappa_i}{d} C_4 \left( \frac{\mathrm{Ra}_\text{eff} a_\text{rh}}{\theta} \right)^{2/3}
$$

where $C_4 = 0.125$ is the empirical prefactor calibrated by Foley and Bercovici (2014, sec. 4.2, p. 589), $a_\text{rh} = 1.3$ is the rheological sublayer constant fitted jointly with $C_4$, and $\rho_i, g_i, \alpha_i, \kappa_i$ are evaluated as $\tilde{W}_k$-weighted means at the top of the convecting layer. The effective Rayleigh number is defined at the convecting interior viscosity $\mu_i$:

$$
\mathrm{Ra}_\text{eff} = \frac{\rho_i g_i \alpha_i (T_i - T_\text{surf}) d^3}{\kappa_i \mu_i}
$$

Because both $\mathrm{Ra}_\text{eff}$ and $\theta$ scale linearly with the temperature drop $\Delta T = T_i - T_\text{surf}$, the ratio $\mathrm{Ra}_\text{eff} a_\text{rh} / \theta$ is proportional to $a_\text{rh} \Delta T_\text{rh}$ where $\Delta T_\text{rh} = R T_i^2 / H(P_{T_i})$. When convection is inactive or reaches the surface such that no physical conductive lid forms ($d_\text{lid} \le 1\text{ m}$ or $T_i - T_\text{surf} \le 10^{-3}\text{ K}$), `compute_stagnant_lid_state` sets $d_\text{lid}$, $\mathrm{Ra}_\text{eff}$, $\mathrm{Ra}_\text{rh}$, $v_m$, $\tau_d$, $\tau_{d,\text{lid}}$, $\tau_\text{buoy}$, $\tau_d / \tau_\text{buoy}$, $w_y$, and `lid_regime` to zero, preventing numerical floor cancellation artifacts when $\Delta T \to 0$.

### 4.1 Heating Mode and Evolutionary Context

The $2/3$ velocity scaling law applies to bottom-heated stagnant-lid convection (Dumoulin et al., 1999; Foley and Bercovici, 2014, p. 587). In planetary evolution, cooling terrestrial planets start with a superheated core, creating a large core-mantle boundary temperature difference that drives strong basal heating during early post-magma-ocean evolution (Thiriet et al., 2019, pp. 139-140, sec. 1).

As the core cools and radiogenic heating decays, the mantle heating regime transitions from bottom-dominated to predominantly internal heating (Thiriet et al., 2019, p. 140). For purely internally heated convection, Solomatov and Moresi (2000, p. 21804, eq. 13; p. 21805, Table 8) obtain a $1/2$ velocity scaling exponent:

$$
v_m = a_u \frac{\kappa}{d} \left( \frac{\mathrm{Ra}_i a_\text{rh}}{\theta} \right)^{1/2}
$$

with $a_u = 0.385 \pm 0.003$ for Newtonian rheology ($n = 1$) and $a_\text{rh} = 2.4 \pm 0.2$. In both the bottom-heated and internally heated scalings, the temperature contrast $\Delta T$ enters the definitions of Rayleigh number and Frank-Kamenetskii parameter symmetrically, cancelling in the ratio $\mathrm{Ra} / \theta$. For the Earth reference state ($\rho_i = 3300\text{ kg m}^{-3}$, $g_i = 9.81\text{ m s}^{-2}$, $\alpha_i = 3 \times 10^{-5}\text{ K}^{-1}$, $T_i = 1600\text{ K}$, $T_\text{surf} = 300\text{ K}$, $d = 2.89 \times 10^6\text{ m}$, $\kappa_i = 10^{-6}\text{ m}^2\text{ s}^{-1}$, $H = 300\text{ kJ mol}^{-1}$, $\mu_i = 10^{20}\text{ Pa s}$, giving $\theta = 18.32$ and $\mathrm{Ra}_i = 3.05 \times 10^8$), the internally heated scaling predicts $v_m = 8.42 \times 10^{-10}\text{ m s}^{-1}$ and convective driving shear stress $\tau_d = 2 \mu_i v_m / d \approx 0.058\text{ MPa}$ at $a_\text{rh} = 2.4$ (or $v_m = 5.43 \times 10^{-10}\text{ m s}^{-1}$ and $\tau_d \approx 0.038\text{ MPa}$ with $a_\text{rh} = 1.0$), approximately four times lower than the bottom-heated estimate ($v_m = 3.36 \times 10^{-9}\text{ m s}^{-1}$, $\tau_d \approx 0.232\text{ MPa}$ at $C_4 = 0.125$, $a_\text{rh} = 1.3$). Consequently, evaluating $\tau_d$ with the bottom-heated $2/3$ scaling represents an upper estimate on convective shear stress during later, internally heated epochs.

### 4.2 Validated Parameter Ranges and Mushy Extrapolation

Foley and Bercovici (2014, sec. 4.2, p. 589; Table 2, p. 593; Table 3, p. 594; Figs. 8-9) calibrate and validate the scaling laws over:

$$
\mathrm{Ra}_0 \in [10^5, \; 5 \times 10^7]
$$

In a partially molten or mushy mantle where the two-phase log-viscosity blend (`viscosity_solid` $= 10^{21}\text{ Pa s}$, `viscosity_liquid` $= 10^2\text{ Pa s}$ in standard two-phase configurations) reduces the interior viscosity to $\mu_i \sim 10^2\text{ to }10^7\text{ Pa s}$ (or $\sim 10^3\text{ to }10^9\text{ Pa s}$ near the rheological transition), evaluating $\mathrm{Ra}_\text{eff}$ with the Earth reference inputs above ($\rho_i g_i \alpha_i (T_i - T_\text{surf}) d^3 / \kappa_i = 3.05 \times 10^{28}\text{ Pa s}$ at $\Delta T = 1300\text{ K}$) gives $\mathrm{Ra}_\text{eff} = 3.05 \times 10^{21}\text{ to }3.05 \times 10^{26}$ (or $\sim 10^{19}\text{ to }10^{25}$ for $\mu_i \sim 10^3\text{ to }10^9\text{ Pa s}$), well above the numerical calibration domain of solid-state boundary-layer theory. Applying the scaling to mushy interiors represents an extrapolation, and Solomatov and Moresi (2000, sec. 5.3, p. 21803) note that the velocity scaling law can change at very high Rayleigh numbers when convective plumes disconnect from the lid base.

### 4.3 Rheological Parameter Calibration

In the velocity scaling, $a_\text{rh} = 1.3$ and $C_4 = 0.125$ are the empirical calibration constants fitted jointly by Foley and Bercovici (2014, sec. 4.2, p. 589) for their theory-curve constant-healing ($E_h = 0$) damage models. Every numerical model in Foley and Bercovici (2014) includes dynamic grain damage; the paper provides no pure damage-free numerical fit. The calibrated constant $a_\text{rh} = 1.3$ is lower than the standard stagnant-lid value ($a_\text{rh} \approx 2$; Solomatov and Moresi, 2000; Korenaga, 2009, cited in Foley and Bercovici, 2014, p. 586) because less effective damage means less of the high-viscosity lid participates in convection (Foley and Bercovici, 2014, p. 586). For temperature-dependent healing, Foley and Bercovici (2014, sec. 5.2, pp. 594-595) find $a_\text{rh} \approx 1.82$.

`compute_stagnant_lid_state` adopts $C_4 = 0.125$ and $a_\text{rh} = 1.3$ as a unified parameter package calibrated against sub-lid stresses. In the Foley and Bercovici (2014) theory, grain damage enters solely through the effective viscosity $\mu_\text{eff} = \mu_i (A_i / A_0)^{-m}$, which reduces to the single-phase or mixture viscosity when damage is absent ($A_i = A_0$). The convective interior velocity scaling thus holds without damage mechanics. The sensitivity to $a_\text{rh}$ is moderate: adopting $a_\text{rh} = 2.0$ instead of $1.3$ increases convective velocity $v_m$ and driving stress $\tau_d$ by a factor of $(2.0 / 1.3)^{2/3} \approx 1.33$.

This sublayer parameter $a_\text{rh} = 1.3$ enters the interior convective velocity $v_m$ and differs on purpose from the configured `lid_contrast_coeff` ($a = 2.2$; Solomatov, 1995; Solomatov and Moresi, 2000, sec. 5.1, p. 21800), which defines the thermal boundary isotherm of the rigid lid ($T_{\text{lid,iso}} = T_i - a \Delta T_\text{rh}$).

### 4.4 Diagnostic Quantities and Force Balance

`compute_stagnant_lid_state` evaluates the rheological sublayer buoyancy stress as a diagnostic entry in `lid_state`:

$$
\tau_\text{buoy} = \rho_i g_i \alpha_i \Delta T_\text{rh} \delta_\text{rh} w_\text{active}
$$

where $\delta_\text{rh}$ is the rheological sublayer thickness and $\Delta T_\text{rh} = R T_i^2 / H(P_{T_i})$. The sublayer buoyancy stress $\tau_\text{buoy}$ and the force-balance ratio $\tau_d / \tau_\text{buoy}$ are returned in the stagnant-lid diagnostic state dictionary `lid_state`. For the Earth reference state ($\rho = 3300\text{ kg m}^{-3}$, $g = 9.81\text{ m s}^{-2}$, $\alpha = 3 \times 10^{-5}\text{ K}^{-1}$, $T_i = 1600\text{ K}$, $T_\text{surf} = 300\text{ K}$, $d = 2.89 \times 10^6\text{ m}$, $\kappa = 10^{-6}\text{ m}^2\text{ s}^{-1}$, $H = 300\text{ kJ mol}^{-1}$, $\eta_i = 10^{20}\text{ Pa s}$, $\theta \approx 18.32$, $\Delta T_\text{rh} \approx 71.0\text{ K}$, $\tau_d \approx 0.232\text{ MPa}$), a 50-node piecewise-linear reference profile (with resolved lid thickness $d_\text{lid} = 378.4\text{ km}$) gives $\delta_\text{rh} \approx 30.8\text{ km}$ ($30791\text{ m}$), $\tau_\text{buoy} \approx 2.12\text{ MPa}$, and $\tau_d / \tau_\text{buoy} \approx 0.1095$ because `xp.gradient(T, r)` spans the isothermal-to-conductive corner; the boundary-layer scaling values across the 7 conductive intervals ($d_\text{lid,cond} = 7 \times 58.98\text{ km} = 412.9\text{ km}$, $\delta_\text{rh} = d_\text{lid,cond} / \theta \approx 22.5\text{ km}$) are $\tau_\text{buoy} \approx 1.55\text{ MPa}$ and $\tau_d / \tau_\text{buoy} \approx 0.150$. In addition, the shifted soft-maximum convective velocity $v_i$, evaluated over the upper mantle using `SOFTMAX_VELOCITY_SCALE` ($3.17 \times 10^{-12}\text{ m s}^{-1}$), is returned as a diagnostic quantity in `lid_state`.

## 5. Minimum Yield Closure

The effective solid viscosity combines diffusion creep and plastic yielding through the minimum formulation implemented in `aragog.rheology_lid.compute_effective_viscosity` (Moresi and Solomatov, 1998, p. 672, eqs. 13-14, doi:10.1046/j.1365-246X.1998.00531.x; Tackley, 2000, p. 4, eq. 8, doi:10.1029/2000GC000036):

$$
\eta_\text{yield} = \frac{\tau_\text{yield}}{D}, \quad \eta_\text{eff} = \begin{cases} \eta_\text{diff}, & \tau < \tau_\text{yield} \\ \min(\eta_\text{diff}, \eta_\text{yield}), & \tau \ge \tau_\text{yield} \end{cases}
$$

In boundary-layer `'lid'` mode, `compute_effective_viscosity` requires the convecting interior viscosity $\eta_i$, the driving stress $\tau_d$, and the lid yield stress $\tau_{y,\text{lid}}$. The second invariant of the convective strain rate acting on the boundary layer is set by the interior convective flow scale $D_\text{conv} = 2 \dot{\epsilon}_\text{conv} = 2 v_m / d = \tau_d / \eta_i$ (Foley and Bercovici, 2014, p. 588, eq. 28). Substituting $D_\text{conv} = \tau_d / \eta_i$ gives the plastic yield viscosity:

$$
\eta_\text{yield} = \eta_i \frac{\tau_{y,\text{lid}}}{\tau_d}, \quad \eta_\text{eff}(r) = \begin{cases} \eta_\text{diff}(r), & \tau_d \le \tau_{y,\text{lid}} \\ \min\left( \eta_\text{diff}(r), \; \eta_i \frac{\tau_{y,\text{lid}}}{\tau_d} \right), & \tau_d > \tau_{y,\text{lid}} \end{cases}
$$

Below and at yield ($\tau_d \le \tau_{y,\text{lid}}$), $\eta_\text{eff}(r) = \eta_\text{diff}(r)$ exactly, preserving the rigid conductive stagnant lid. For standard Byerlee rock friction ($c = 50\text{ MPa}$, $\mu = 0.6$, $\tau_\text{max} = 500\text{ MPa}$), $\tau_{y,\text{lid}} = 500\text{ MPa}$ exceeds the Earth reference driving stress ($\tau_d \approx 0.232\text{ MPa}$) by more than three orders of magnitude, so the lid remains unyielded. Above yield ($\tau_d > \tau_{y,\text{lid}}$), $\eta_\text{yield} = \eta_i (\tau_{y,\text{lid}} / \tau_d) < \eta_i$, lowering the yielded boundary-layer viscosity below the interior convective viscosity and mobilising the surface boundary layer into thin-boundary-layer convection. For example, at a cold surface ($T_\text{surf} = 300\text{ K}$, $\eta_\text{diff} = 10^{40}\text{ Pa s}$, $\eta_i = 10^{21}\text{ Pa s}$) with $\tau_d / \tau_{y,\text{lid}} = 1.2$, the yielded viscosity drops to $\eta_\text{eff} = 8.33 \times 10^{20}\text{ Pa s} < \eta_i$, whereas evaluating the strain rate from the cold-lid creep viscosity ($\eta_\text{diff} \tau_{y,\text{lid}} / \tau_d$) would leave $\eta_\text{eff} = 8.33 \times 10^{39}\text{ Pa s}$. An alternative harmonic-mean formulation $\eta_\text{eff} = \eta_\text{diff} \eta_\text{yield} / (\eta_\text{diff} + \eta_\text{yield})$ is discussed by Foley and Becker (2009, p. 3, eqs. 7-8) and Foley and Bercovici (2014, sec. 8.2).

The effective solid viscosity through the radial column is determined by applying the closure viscosity within the cold boundary layer using the smooth lid mask:

$$
\log_{10} \eta_\text{solid} = w_\text{lid} \log_{10} \eta_\text{eff} + (1 - w_\text{lid}) \log_{10} \eta_\text{diff}
$$

Inside the cold lid ($w_\text{lid} \to 1$), $\eta_\text{solid} \to \eta_\text{eff}$. In the warm convective interior ($w_\text{lid} \to 0$), $\eta_\text{solid} \to \eta_\text{diff}$. This formulation restricts plastic yielding to the cold boundary layer without altering the interior convective mantle (Foley and Becker, 2009, sec. 3.2.2).

The minimum closure is everywhere finite and non-increasing with driving stress (constant for $\tau_d \le \tau_{y,\text{lid}}$ and decreasing as $\tau_{y,\text{lid}} / \tau_d$ for $\tau_d > \tau_{y,\text{lid}}$). In `'local'` mode, $\min(\eta_\text{diff}, \eta_\text{yield})$ is continuous at $\tau_\text{creep} = \tau_y$ with a derivative jump; in `'lid'` mode, $\eta_\text{eff}$ is smooth within the unyielded ($\tau_d \le \tau_{y,\text{lid}}$) and yielded ($\tau_d > \tau_{y,\text{lid}}$) branches and jumps at $\tau_d = \tau_{y,\text{lid}}$ from $\eta_\text{diff}$ to $\min(\eta_\text{diff}, \eta_i \tau_{y,\text{lid}} / \tau_d)$ when $\eta_i < \eta_\text{diff}$. To prevent zero viscosity when zero cohesion and friction ($c = 0, \mu = 0$) are configured at zero pressure, `compute_effective_viscosity` floors the yield stress at `COHESION_FLOOR = 1.0e-6 Pa`.

When convective vigor ceases ($v_i \to 0$), convective heat transport drops below $f_\text{conv,min}$. In this limit, $w_\text{active} \to 0$, $\tau_d = 0$, the lid closure remains inactive, and no division by zero occurs.

## 6. Interaction with the Eddy-Diffusivity Floor

In `aragog.jax.phase.compute_mlt` and `aragog.solver.entropy_state.apply_kappah_floor`, the thermal eddy diffusivity $\kappa_h$ is floored in convectively unstable cells ($\partial S / \partial r < 0$) by a melt-fraction-modulated ramp without a lid mask:

$$
f(\phi) = \frac{1}{2} \left( 1 + \tanh\left( \frac{\phi - \phi_\text{rheo}}{\phi_\text{width}} \right) \right)
$$

$$
\kappa_{h,\text{floor,eff}} = \kappa_{h,\text{floor}} \cdot f(\phi) \cdot I_\text{conv}, \quad \kappa_h = \max\big(\kappa_{h,\text{MLT}}, \; \kappa_{h,\text{floor,eff}}\big)
$$

where $I_\text{conv} = 1$ when $\partial S / \partial r < 0$ and $0$ in stably stratified cells ($\partial S / \partial r \ge 0$).

In standalone Aragog configurations, the transition width defaults to `rheological_transition_width = 0.15` ($\phi_\text{width} = 0.15$ at $\phi_\text{rheo} = 0.4$, where $f(0) = 0.004805$), and `kappah_floor` defaults to $10.0\text{ m}^2\text{ s}^{-1}$ on `EnergyConfig` and `_EnergyParameters` (while the direct `PhaseParams.__init__` fallback in `aragog.jax.phase` is $0.0\text{ m}^2\text{ s}^{-1}$). In coupled PROTEUS runs (`proteus.config._interior.Aragog`), the defaults are `rfront_loc = 0.4` ($\phi_\text{rheo} = 0.4$), `rfront_wid = 0.2` ($\phi_\text{width} = 0.2$, where $f(0) = 0.018$), and `kappah_floor = 10.0` ($\kappa_{h,\text{floor}} = 10.0\text{ m}^2\text{ s}^{-1}$). At $\kappa_{h,\text{floor}} = 10.0\text{ m}^2\text{ s}^{-1}$ and zero melt fraction ($\phi = 0$), $\kappa_{h,\text{floor,eff}}$ evaluates to $0.048\text{ m}^2\text{ s}^{-1}$ in convectively unstable solid cells at the standalone width $\phi_\text{width} = 0.15$ and $0.18\text{ m}^2\text{ s}^{-1}$ at the PROTEUS width $\phi_\text{width} = 0.2$, exceeding molecular thermal diffusivity ($\kappa \sim 10^{-6}\text{ m}^2\text{ s}^{-1}$) by four to five orders of magnitude. In stably stratified cells ($\partial S / \partial r \ge 0$), $I_\text{conv} = 0$ and $\kappa_{h,\text{floor,eff}} = 0$.

## 7. Stress Closure Modes

The geometry of the strain rate closure in `compute_effective_viscosity` is controlled by `stress_closure_mode`, and cross-mode keyword arguments are rejected:

- `lid`:
    Boundary-layer stress closure. Requires `eta_i`, `tau_d`, and `tau_y_lid` (rejecting local-mode arguments `tau_y`, `tau_y_profile`, `strain_rate`, `mixing_length`, and `unyielded_velocity`). The driving stress $\tau_d = (2 \mu_i v_m / d) w_\text{active}$ is evaluated from the convecting interior scaling, and plastic yield viscosity $\eta_\text{yield} = \eta_i (\tau_{y,\text{lid}} / \tau_d)$ mobilises the lid nodes when $\tau_d > \tau_{y,\text{lid}}$.

- `local`:
    Local mixing-length strain rate closure (rejecting lid-mode arguments `eta_i`, `tau_d`, `tau_y_lid`, and `w_lid`). The strain rate is evaluated directly from local cell velocity and mixing length using the second invariant convention:

    $$
    \dot{\epsilon}_\text{local}(r) = \frac{|v_\text{unyielded}(r)|}{2 \max(l(r), 10^{-15}\text{ m})}
    $$

    and $\eta_\text{eff}(r) = \min\big(\eta_\text{diff}(r), \; \max(\tau_y(r), 10^{-6}\text{ Pa}) / (2 \dot{\epsilon}_\text{local}(r))\big)$. Because unyielded Arrhenius viscosity inside a cold lid is very high, $v_\text{unyielded} \approx 0$ and $\dot{\epsilon}_\text{local} \approx 0$. Consequently, `local` mode cannot yield the cold lithosphere and operates solely as an interior weakening mechanism in vigorously convecting deep regions.

- `global`:
    Unsupported mode. Configuration validation rejects `global` and instructs users to specify `lid`.

In `SolidRheologyParams`, `stress_closure_mode` defaults to `'lid'`; callers pass `stress_closure_mode` to `compute_effective_viscosity` to select the boundary-layer closure (`'lid'`) or the local strain-rate closure (`'local'`).

## 8. Asymptotic Limits

The kernels in `aragog.rheology` and `aragog.rheology_lid` satisfy the following asymptotic limits:

1. **Zero driving stress ($\tau_d \to 0$):**
   $\tau_d / \tau_{y,\text{lid}} \to 0 \implies w_y \approx 0$ ($\frac{1}{2}(1 - \tanh 10) \approx 2.1 \times 10^{-9}$ when a lid is present, or $0$ when `has_lid` is `False`) and $\eta_\text{eff} = \eta_\text{diff}$. The system exhibits pure Arrhenius diffusion creep without plastic weakening.
2. **Infinite yield stress ($\tau_y \to \infty$):**
   $\tau_d / \tau_{y,\text{lid}} \to 0 \implies \eta_\text{eff} = \eta_\text{diff}$, yielding never activates, and the lid remains stagnant at diffusion-creep viscosity.
3. **Small homologous activation parameter ($g_\text{hom} \to 0^+$):**
   `SolidRheologyParams` requires `g_homologous > 0` when `enabled = True` in both `'rheological'` and `'fixed'` lid modes. In `compute_arrhenius_viscosity`, taking $g_\text{hom} \to 0$ gives $H(P) \to 0$ and recovers an isoviscous solid mantle $\eta_\text{diff} = \eta_\text{solid} f_\text{water}$.
4. **Zero solidus pressure slope ($dT_\text{sol} / dP = 0$):**
   $H(P) = g_\text{hom} R T_{\text{sol},0}$, recovering temperature-only Arrhenius creep with constant activation energy.
5. **Infinite Frank-Kamenetskii contrast ($\theta \to \infty$):**
   Recovers the asymptotic stagnant-lid regime of Solomatov (1995) with upper mixing-length slope $s_\text{top}(\theta) \to s_{\text{top},\infty}$.
6. **Sub-threshold interior convection ($f_\text{int,max} \ll f_\text{conv,min}$) or no conductive lid ($d_\text{lid} = 0$):**
   When $f_\text{int,max}$ lies more than $0.01\,\mathrm{arctanh}(0.9) \approx 0.0147$ below $f_\text{conv,min}$ (for example $f_k \le 0.035$ at $f_\text{conv,min} = 0.05$), $w_\text{active} \le 0.05$, `is_convecting` and `has_lid` are `False`, $\tau_d = 0$, $v_m = 0$, $\mathrm{Ra}_\text{eff} = 0$, and yielding remains inactive across any mesh resolution $N$.
7. **Wholly molten mantle ($\phi > \phi_\text{rheo}$ everywhere):**
   In `compute_stagnant_lid_state`, $w_\text{interior} = 1$ throughout the column so the lid-base boundary weight $\tilde{W}_\text{base}$ concentrates at the surface node $r[-1]$, giving $T_\text{lid} = T_\text{surf}$, $r_\text{lid,base} = r[-1]$, $d_\text{lid} = 0$, $\tau_d = 0$, and `lid_regime = 0.0`.
8. **High driving stress ($\tau_d / \tau_{y,\text{lid}} > 1$):**
   $w_y \to 1$ and $\eta_\text{eff} \to \min(\eta_\text{diff}, \eta_i \tau_{y,\text{lid}} / \tau_d) < \eta_i$, mobilising the cold surface boundary layer into thin-boundary-layer convection.

## 9. Rheology Parameters

Solid-state convection parameters are stored on `SolidRheologyParams` in `aragog.rheology` and exposed on `PhaseConfig.rheology` in `aragog.config.phases`:

| `field` | unit | default as repr() |
|:---|:---:|:---:|
| `enabled` | - | `False` |
| `g_homologous` | - | `25.0` |
| `t_sol_0` | K | `1400.0` |
| `dt_sol_dp` | K/Pa | `2.4e-08` |
| `arrhenius_t_ref` | K | `1600.0` |
| `viscosity_max_log10` | log10 Pa s | `40.0` |
| `water_prefactor` | - | `1.0` |
| `yield_stress_c` | Pa | `50000000.0` |
| `yield_stress_mu` | - | `0.6` |
| `yield_stress_max` | Pa | `500000000.0` |
| `stress_closure_mode` | - | `'lid'` |
| `interior_flux_fraction` | - | `0.05` |
| `lid_base_mode` | - | `'rheological'` |
| `lid_base_temperature` | K | `1400.0` |
| `lid_contrast_coeff` | - | `2.2` |
| `lid_mask_width_cells` | - | `1.0` |
| `phi_visc_single` | - | `0.5` |
| `mlt_bottom_slope` | - | `1.0` |
| `s_top_inf` | - | `0.22` |
| `c_theta` | - | `0.15` |
