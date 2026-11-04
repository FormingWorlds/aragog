# Solid-State Convection and Lithospheric Rheology

Aragog follows a mantle from a molten magma ocean, through a partly molten mush, to solid rock. This page explains what happens in the last stage. Solid rock still flows, very slowly, so the mantle can convect while it is solid. A cold, stiff layer can form at the top. This layer is the stagnant lid, and it controls how fast the planet loses heat.

The modules `aragog.rheology` and `aragog.rheology_lid` hold this physics. The solver uses them only when `rheology.enabled` is `True`. The default is `False`. Sections 1 to 8 explain the physics, section 9 lists the limits, section 10 lists the parameters, and section 11 holds the code details.

## The physics in short

**Viscosity varies by many orders of magnitude.** Viscosity measures how hard it is to deform a material. In solid rock it depends strongly on temperature and pressure. For example, rock at 90 % of its solidus temperature is about 16 times more viscous than rock at its solidus temperature ($g_\text{hom} = 25$, a parameter of the creep law in section 1). The solidus is the temperature where rock starts to melt.

**A stagnant lid forms at the cold surface.** The coldest rock is too stiff to take part in convection. It stays in place and carries heat mainly by conduction. The warm mantle below it convects.

**The lid can yield.** Convection below the lid pushes on it with a driving stress. Rock has a limited strength, the yield stress. If the driving stress is larger than the yield stress of the lid, the lid fails and joins the flow. In the Earth example of sections 4.4 and 5, the yield stress is about 280 times larger than the driving stress, so the lid stays in place (section 5).

**What `rheology.enabled` changes in a run:**

| Part of the model | `enabled = False` (default) | `enabled = True` | Section |
|:---|:---|:---|:---:|
| Solid viscosity | Constant at `viscosity_solid` | Creep law: depends on temperature and pressure | 1 |
| Melt fraction and viscosity with `const_properties = True` | $\phi = 1$ (liquid), viscosity `10**const_log10visc` | $\phi = 0$ (solid), creep viscosity | - |
| Viscosity in the flux calculation | Calculated once | Calculated in two passes: the second pass uses the effective viscosity from the first pass (the lid state in `'lid'` mode, the local yield viscosity in `'local'` mode) | 3, 5, 8 |
| Top slope of the mixing length | Not changed | Reduced near the surface | 7 |
| Eddy-diffusivity floor in a fully solid cell | Small value above zero | Exactly zero, so a conductive lid can form | 6 |
| Check on each accepted state | With a table equation of state, a call that advances fails if a cell entropy leaves the table | With a table equation of state, the run stops with `RuntimeError` if a cell entropy leaves the table in an accepted state. With any equation of state, it also stops on a non-finite or out-of-range viscosity and, in `'lid'` mode, on a bad lid diagnostic. On the JAX path these viscosity and lid checks run on the last state of each call | 6 |

With `enabled = False` the solver skips the rheology kernels and reproduces the baseline trajectory without this physics.

## 1. Viscosity of Solid Rock: Diffusion Creep

**How does solid viscosity depend on temperature and pressure?** Solid rock flows by diffusion creep: atoms move slowly through the crystal grains. The hotter the rock is relative to its solidus temperature, the faster this is. Aragog therefore writes viscosity as a function of $T / T_\text{sol}$, the temperature divided by the solidus temperature. This ratio is the homologous temperature.

### The creep law

The law multiplies a reference viscosity by an exponential. The exponent is large when the rock is cold relative to its solidus, and it is zero at the reference state. This is the homologous-temperature Arrhenius form of Yamazaki and Karato (2001)[^cite-yamazaki2001] (eqs. 4-7) and Abe (1997)[^cite-abe1997] (Table 1):

$$
\eta_\text{diff}(T, P) = \eta_\text{solid} \cdot f_\text{water} \cdot \exp\left( \frac{H(P)}{R T} - \frac{H(0)}{R T_\text{ref}} \right) = \eta_\text{solid} \cdot f_\text{water} \cdot \exp\left[ g_\text{hom} \left( \frac{T_\text{sol}(P)}{T} - \frac{T_\text{sol}(0)}{T_\text{ref}} \right) \right]
$$

| Symbol | Meaning | Code name | Default |
|:---|:---|:---|:---|
| $\eta_\text{solid}$ | Reference solid viscosity | `viscosity_solid` | $10^{21}\text{ Pa s}$ |
| $f_\text{water}$ | Hydration weakening prefactor | `water_prefactor` | $1.0$ |
| $R$ | Universal gas constant | `R_GAS` | $8.314462618\text{ J mol}^{-1}\text{ K}^{-1}$ |
| $g_\text{hom}$ | Dimensionless homologous activation parameter | `g_homologous` | $25.0$ |
| $T_\text{sol}(P)$ | Local solidus temperature of the mantle at pressure $P$ | see below | see below |
| $H(P)$ | Activation enthalpy, $H(P) = g_\text{hom} R T_\text{sol}(P)$ | | |
| $T_\text{ref}$ | Reference temperature | `arrhenius_t_ref` | $1600\text{ K}$ |

**What the reference means.** For the linear solidus, $\eta_\text{solid} f_\text{water}$ equals $\eta_\text{diff}$ at $(T_\text{ref} = 1600\text{ K}, P = 0)$. When a solidus table is supplied, $\eta_\text{solid}$ is referenced to `params.t_sol_0` ($T_{\text{sol},0}$), not to the $P = 0$ value of the table. The table $\eta_\text{diff}(T, 0)$ then differs from the linear-solidus $\eta_\text{diff}(T, 0)$ at the same $T$ by the ratio $\exp\big(g_\text{hom}(T_\text{sol}(0) - T_{\text{sol},0}) / T\big)$. The table below gives this ratio at $T = 1600\text{ K}$ and $g_\text{hom} = 25$.

| Table | $T_\text{sol}(0)$ against $T_{\text{sol},0} = 1400\text{ K}$ | Ratio | Result at $\eta_\text{solid} = 10^{21}\text{ Pa s}$ |
|:---|:---:|:---:|:---:|
| Test table of the continuous integration (CI) | $T_{\text{sol,table}}(0) = 1439.230032\text{ K}$ | $1.8459\times$ | $\eta_\text{diff}(1600\text{ K}, 0) = 1.8459 \times 10^{21}\text{ Pa s}$ |
| PALEOS value | $1464.8\text{ K}$ | $2.752\times$, or $2.75\times$ | $\eta_\text{diff}(1600\text{ K}, 0) = 2.752 \times 10^{21}\text{ Pa s}$ |

### The Activation Enthalpy and the Solidus

**Why the enthalpy follows the solidus.** With a constant activation volume $V_a$, the activation enthalpy grows linearly with pressure: $H(P) = E_a + P V_a$. If this line is extended to core-mantle boundary pressures ($135\text{ GPa}$ in Earth), viscosity along an adiabat rises by more than nine orders of magnitude. The deep mantle would then be rigid, which is not physical.

Mineral physics measurements and melting-point scaling relations show that the effective activation enthalpy scales with the solidus temperature $T_\text{sol}(P)$ (Yamazaki and Karato, 2001). The activation volume $V_a(P) = \partial H / \partial P = g_\text{hom} R (dT_\text{sol} / dP)$ then decreases as the solidus slope flattens at high pressure.

One result is simple. Take any constant fraction $f$ of the solidus, $T(P) = f \cdot T_\text{sol}(P)$. The viscosity at that temperature, relative to the viscosity on the solidus, does not depend on pressure:

$$
\frac{\eta_\text{diff}(f \cdot T_\text{sol}(P), P)}{\eta_\text{diff}(T_\text{sol}(P), P)} = \exp\left[ g_\text{hom} \left( \frac{1}{f} - 1 \right) \right]
$$

For $f = 0.9$, this ratio equals $\exp(g_\text{hom} / 9) = 16.083$ at $g_\text{hom} = 25$ and $4.738$ at $g_\text{hom} = 14$, at all depths.

**A solidus from a table.** Tabulated `EntropyEOS` runs supply an explicit solidus profile $T_\text{sol}(P)$ to `compute_arrhenius_viscosity` or `compute_stagnant_lid_state`, such as the phase-boundary solidus from the PALEOS $\mathrm{MgSiO_3}$ equation-of-state tables. `compute_arrhenius_viscosity` then requires the zero-pressure reference solidus $T_{\text{sol},0}$ (`t_sol_0`). `compute_stagnant_lid_state` reads `params.t_sol_0` directly. Coupled and tabulated PALEOS runs use $T_\text{sol}(P) = 0.8\,T_\text{liq}(P)$ from the PALEOS $\mathrm{MgSiO_3}$ melting curve.

**A solidus as a straight line.** When `solidus_temperature` is `None`, as in table-free constant-property benchmarks, `compute_arrhenius_enthalpy`, `compute_arrhenius_viscosity` and `compute_stagnant_lid_state` use a linear fallback:

$$
T_\text{sol}(P) = T_{\text{sol},0} + \frac{dT_\text{sol}}{dP} P
$$

The defaults are $T_{\text{sol},0} = 1400.0\text{ K}$ (`t_sol_0`) and $dT_\text{sol} / dP = 2.4 \times 10^{-8}\text{ K Pa}^{-1}$ (`dt_sol_dp`, which is $24\text{ K GPa}^{-1}$). At $g_\text{hom} = 25.0$ and $T_{\text{sol},0} = 1400.0\text{ K}$, the zero-pressure activation enthalpy is $H(0) = 291\text{ kJ mol}^{-1}$. This is close to the standard olivine diffusion-creep activation energy $E_a = 300\text{ kJ mol}^{-1}$ (Karato and Wu, 1993)[^cite-karato1993].

**How far the line is from the PALEOS solidus.** The straight line is for table-free constant-property benchmarks only. The physical PALEOS solidus is concave downward in pressure, so one line cannot follow it. The PALEOS curve is $1464.8\text{ K}$ at $0\text{ GPa}$, $1563.0\text{ K}$ at $1\text{ GPa}$ and $4754.8\text{ K}$ at $135\text{ GPa}$. Its whole-mantle secant slope is $24.4\text{ K GPa}^{-1}$, and its high-pressure slope is $15.3\text{ K GPa}^{-1}$ between $25$ and $135\text{ GPa}$.

| Pressure | Linear default | PALEOS | Linear form is low by |
|:---:|:---:|:---:|:---:|
| $0\text{ GPa}$ | $1400.0\text{ K}$ | $1464.8\text{ K}$ | $4.4\%$ |
| $25\text{ GPa}$ | $2000.0\text{ K}$ | $3067.0\text{ K}$ | $34.8\%$ |
| $135\text{ GPa}$ | $4640.0\text{ K}$ | $4754.8\text{ K}$ | $2.4\%$ |

The table below compares 3 straight lines with the PALEOS solidus over $0\text{ to }135\text{ GPa}$.

| Line | Largest deviation | Root-mean-square (RMS) deviation | Crossings of the solidus |
|:---|:---|:---:|:---|
| Linear default | $1108.1\text{ K}$ ($-1108.1\text{ K}$ at $36.7\text{ GPa}$); largest relative deviation $-35.0\%$ at $20.5\text{ GPa}$ ($-35.04\%$ at $20.495\text{ GPa}$) | $817.6\text{ K}$ | None |
| Minimax line $1979.7\text{ K} + 24.37\text{ K GPa}^{-1} P$ | $514.9\text{ K}$ | $350.6\text{ K}$ | $6.1\text{ GPa}$ and $98.5\text{ GPa}$ ($6.15\text{ GPa}$ and $98.54\text{ GPa}$) |
| Ordinary least-squares line $2489.3\text{ K} + 19.10\text{ K GPa}^{-1} P$ | $1024.5\text{ K}$ at $0\text{ GPa}$ | $239.0\text{ K}$ | $18.8\text{ GPa}$ and $99.9\text{ GPa}$ ($18.82\text{ GPa}$ and $99.89\text{ GPa}$) |

**The viscosity ceiling.** A cold conductive surface layer can give a viscosity that overflows float64. To prevent this, `compute_arrhenius_viscosity` applies a smooth $\log_{10}$ softplus ceiling at `viscosity_max_log10` (default $40.0$, width `VISCOSITY_CEILING_LOG10_WIDTH = 0.05`), after it applies `water_prefactor`. In this function, a non-positive temperature ($T \le 0$) raises `ValueError` on the eager NumPy path and returns `NaN` on the JAX path.

### Enthalpy Values on the PALEOS Solidus

This table gives the solidus temperature $T_\text{sol}(P) = 0.8\,T_\text{liq}(P)$ from the PALEOS $\mathrm{MgSiO_3}$ melting curve (`mushy_zone_factor = 0.8`, produced by `tools/verification/compute_solidus_reference_values.py`). It also gives the resulting activation enthalpy $H(P) = g_\text{hom} R T_\text{sol}(P)$ over terrestrial mantle pressures.

| Pressure $P$ (GPa) | $T_\text{sol}(P)$ (K) | $H(P)$ at $g_\text{hom} = 14$ (kJ/mol) | $H(P)$ at $g_\text{hom} = 25$ (kJ/mol) | $\eta(0.9 T_\text{sol}) / \eta(T_\text{sol})$ at $g_\text{hom} = 25$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 1464.8 | 170.5 | 304.5 | 16.083 |
| 25 | 3067.0 | 357.0 | 637.5 | 16.083 |
| 50 | 3672.7 | 427.5 | 763.4 | 16.083 |
| 100 | 4397.9 | 511.9 | 914.2 | 16.083 |
| 135 | 4754.8 | 553.5 | 988.3 | 16.083 |

### Optional Saturating Integral Activation Law

**How does activation enthalpy change with pressure in a deep mantle, and when does it stop rising?** In a shallow mantle, pressure increases the energy that rock needs to deform by diffusion creep. Inside massive rocky planets, mineral physics experiments show that this pressure effect becomes weaker. Aragog has an optional law that smoothly caps the activation enthalpy at deep mantle pressures.

To use this law, set `activation_energy` ($E_a$) on `SolidRheologyParams`, `compute_arrhenius_enthalpy` or `compute_arrhenius_viscosity`. The solver then calculates activation enthalpy from pressure, not from the homologous melting curve.

| Symbol | Meaning | Unit | Code name | Default |
|:---|:---|:---|:---|:---|
| $P$ | Hydrostatic pressure | $\text{Pa}$ | | |
| $H(P)$ | Activation enthalpy | $\text{J mol}^{-1}$ | | |
| $E_a$ | Zero-pressure activation energy | $\text{J mol}^{-1}$ | `activation_energy` | `None` |
| $V_0$ | Zero-pressure activation volume | $\text{m}^3\text{ mol}^{-1}$ | `activation_volume` | $0.0\text{ m}^3\text{ mol}^{-1}$ |
| $P_\text{decay}$ | Decay pressure | $\text{Pa}$ | `activation_volume_decay_pressure` | $+\infty$ |
| $p'$ | Dummy pressure integration variable | $\text{Pa}$ | | |

The activation volume decays exponentially with pressure. The law integrates this decay over pressure:

$$
H(P) = E_a + \int_0^P V_0 \exp\left(-\frac{p'}{P_\text{decay}}\right) dp' = E_a + V_0 P_\text{decay} \left(1 - \exp\left(-\frac{P}{P_\text{decay}}\right)\right)
$$

At low pressure, where $P / P_\text{decay} < 10^{-8}$, Aragog uses a second-order Taylor series for the pressure term. This avoids a loss of precision in the floating-point subtraction:

$$
P \left(1 - \frac{P}{2 P_\text{decay}}\right)
$$

The law has five limits:

- When $P_\text{decay} = +\infty$, it gives the linear activation volume law $H_\text{lin}(P) = E_a + V_0 P$ (Karato and Wu, 1993; Armann and Tackley, 2012[^cite-armann2012]) to 0 units in the last place (ulp).
- When $V_0 = 0$, it gives the isobaric Arrhenius law $H(P) = E_a$ (Korenaga, 2009[^cite-korenaga2009], Table 1) to 0 ulp.
- The local activation volume $\partial H / \partial P = V_0 \exp(-P / P_\text{decay}) \ge 0$ is non-negative at all pressures.
- As $P \to \infty$, $H(P)$ rises monotonically toward the constant ceiling $E_a + V_0 P_\text{decay}$.
- For finite $P_\text{decay} > 0$, the Lagrange remainder theorem bounds the departure from the linear law:

$$
0 \le \mathrm{d}H(P) = H_\text{lin}(P) - H(P) \le \text{Bound}(P) = \frac{V_0 P^2}{2 P_\text{decay}}
$$

**Worked example.** Take the core-mantle boundary (CMB) of Earth at $P = 135\text{ GPa}$, with $E_a = 300\text{ kJ mol}^{-1}$, $V_0 = 6.0\text{ cm}^3\text{ mol}^{-1}$ and $P_\text{decay} = 200\text{ GPa}$.

| Quantity | Value |
|:---|:---:|
| Linear law, $H_\text{lin}$ | $1110.0\text{ kJ mol}^{-1}$ |
| Saturating law, $H$ | $889.012\text{ kJ mol}^{-1}$ |
| Reduction, $\mathrm{d}H$ | $220.988\text{ kJ mol}^{-1}$ |
| Lagrange bound, $\text{Bound}$ | $273.375\text{ kJ mol}^{-1}$ |

The reduction is below the bound, as the theorem requires. For comparison, the upper-mantle parameter set of Tackley et al. (2013)[^cite-tackley2013] (Table 2) has $E_0 = 300\text{ kJ mol}^{-1}$, $V_0 = 5.0\text{ cm}^3\text{ mol}^{-1}$ and $P_\text{decay} = \infty$. At $P = 135\text{ GPa}$, this linear reference gives $H(135\text{ GPa}) = 975.0\text{ kJ mol}^{-1}$. Tackley et al. (2013, Table 2) also give another form, the secant form $E_0 + P V_0 \exp(-P / P_\text{decay})$. Its parameter sets are not valid parameter sets of this integral law.

**Bounds.** Configuration validation checks these fields:

- `activation_energy` ($E_a$) must be non-negative ($E_a \ge 0\text{ J mol}^{-1}$). A negative value raises `ValueError`. With `rheology.enabled = True`, the law requires $E_a > 0$.
- `activation_volume` ($V_0$) must be non-negative ($V_0 \ge 0\text{ m}^3\text{ mol}^{-1}$). A negative value raises `ValueError`.
- `activation_volume_decay_pressure` ($P_\text{decay}$) must be strictly positive ($P_\text{decay} > 0\text{ Pa}$). A zero or negative value raises `ValueError`.

**Where this is checked.** [Saturating activation law](../Validation/solid_state_convection.md#the-saturating-activation-law-gives-its-limits).

## 2. Yield Stress of the Rock

**How strong is the rock?** At low pressure, cold rock fails by friction along fractures. Its strength then rises with pressure. Deeper and hotter rock deforms in a ductile way, and this limits its strength. `compute_yield_stress` combines both effects. It uses a Byerlee frictional criterion (Byerlee, 1978[^cite-byerlee1978]; Moresi and Solomatov, 1998[^cite-moresi1998], p. 672, eq. 9) with a ductile ceiling (Tackley, 2000[^cite-tackley2000], p. 8, eq. 7):

$$
\tau_y(P) = \max\big(0, \; \min(c + \mu P, \tau_\text{max})\big)
$$

| Symbol | Meaning | Code name | Default |
|:---|:---|:---|:---|
| $c$ | Lithospheric cohesion | `yield_stress_c` | $50\text{ MPa}$ |
| $\mu$ | Friction coefficient | `yield_stress_mu` | $0.6$ |
| $P$ | Lithostatic pressure | | |
| $\tau_\text{max}$ | Yield stress ceiling | `yield_stress_max` | $500\text{ MPa}$ |

### Stress and Strain-Rate Invariants

Stress and strain rate are tensors, and the closures use one number for each. These numbers are the second invariants of the deviatoric stress and strain rate (Moresi and Solomatov, 1998; Tackley, 2000):

$$
\tau = \sqrt{\frac{1}{2} \tau_{ij} \tau_{ij}}, \quad \dot{\epsilon} = \sqrt{\frac{1}{2} \dot{\epsilon}_{ij} \dot{\epsilon}_{ij}}
$$

Under simple shear with velocity gradient $\partial v / \partial y = v / l$, the strain rate tensor has components $\dot{\epsilon}_{xy} = \dot{\epsilon}_{yx} = \frac{1}{2} v / l$. This gives $\dot{\epsilon} = v / (2 l)$ and second invariant $D = 2 \dot{\epsilon} = v / l$.

### The Ductile Ceiling and Its Depth

Byerlee frictional failure governs brittle deformation in the cold shallow lithosphere. At higher temperatures and greater depths, ductile mechanisms limit the shear strength. The ceiling $\tau_\text{max}$ stands in for this ductile limit (Tackley, 2000, p. 8).

With $c = 50\text{ MPa}$ and $\mu = 0.6$, the yield stress reaches the $500\text{ MPa}$ ceiling at this pressure:

$$
P_\text{ceil} = \frac{\tau_\text{max} - c}{\mu} = \frac{500\text{ MPa} - 50\text{ MPa}}{0.6} = 750\text{ MPa} = 0.75\text{ GPa}
$$

On Earth, $0.75\text{ GPa}$ is at a depth of about $23\text{ km}$. Thermal lithospheres are typically thicker than $23\text{ km}$, so the ceiling $\tau_\text{max}$ commonly sets the yield stress at the base of the lid. The scalar $\tau_{y,\text{lid}}$ is the yield stress at the lid-base pressure, and it decides whether the lid moves. `compute_yield_stress` calculates the local profile $\tau_y(P)$.

## 3. Finding the Stagnant Lid

**Where is the lid, and how thick is it?** Below a cold surface, convection carries the heat in the interior, and conduction carries it through the cold lid. `compute_stagnant_lid_state` finds the top of the convecting layer. From this it gets the interior temperature, the base of the lid and the lid thickness. The weights behind this search are in [section 11](#11-code-details).

### The Convecting Interior

The code measures convection with the convective flux fraction $f_k$ at each node $k$: the convective flux divided by the total flux. It compares $f_k$ with a threshold $f_\text{conv,min} \in [0.05, 0.5]$ (`interior_flux_fraction`, default $0.05$) over a fixed transition width of $0.01$:

$$
w_{\text{conv}, k} = \frac{1}{2} \left( 1 + \tanh\left( \frac{f_k - f_\text{conv,min}}{0.01} \right) \right)
$$

A node with weight near 1 convects. A zero-flux conductive node ($f_k = 0$) has a weight of at most $w_\text{conv}(f = 0) = \frac{1}{2}(1 - \tanh(5)) \approx 4.5 \times 10^{-5}$.

**The interior temperature.** The interior temperature $T_i$ is the temperature near the top of the convecting layer, at the shallowest node where $f_k$ reaches $0.9$ of its interior peak $f_\text{int,max}$. The value $0.9$ places $T_i$ close to the true interior, as this example shows. Take a conductive error-function profile $T(z) = 300 + 1300\,\mathrm{erf}(z / \delta)$, with flux-fraction profile $f(z) = 1 - \exp(-(z / \delta)^2)$, a $1300\text{ K}$ surface-to-interior contrast, and $\Delta T_\text{rh} = 70.95\text{ K}$ ($H = 300\text{ kJ mol}^{-1}$ at $1600\text{ K}$).

| Threshold | Depth $z / \delta$ | Temperature | Temperature below the interior value |
|:---:|:---:|:---:|:---:|
| Onset, $f = 0.05$ | $0.2265$ | $626.6\text{ K}$ | $13.7\,\Delta T_\text{rh}$ |
| Selector, $f = 0.9$ | $1.517$ | $1558.6\text{ K}$ | $0.584\,\Delta T_\text{rh}$ |

From the same selection the code takes the pressure $P_{T_i}$, the radius $r_i$ and the interior viscosity $\mu_i$.

**Too little convection.** The weight $w_\text{active}$ is a smooth value from 0 to 1, near 1 when convection is active (section 11). If convection is too weak, there is no lid: when $f_\text{int,max}$ lies more than $0.01\,\mathrm{arctanh}(0.9) \approx 0.0147$ below $f_\text{conv,min}$ (that is, $f_\text{int,max} \le 0.035278$ at $f_\text{conv,min} = 0.05$), $w_\text{active} \le 0.05$. `is_convecting` and `has_lid` are then `False`, no lid is delimited, and yielding stays inactive at any mesh resolution $N$.

### Rheological Temperature Scale

Diffusion-creep viscosity changes by a factor of $e$ over a temperature interval. This interval is the Frank-Kamenetskii rheological temperature scale:

$$
\Delta T_\text{rh} = \frac{R T_i^2}{H(P_{T_i})}
$$

The code evaluates this scale at the pressure $P_{T_i}$ of the interior node. This avoids a circular dependence on the lid thickness.

### When the Lid Scaling Holds

The boundary-layer scaling of a stagnant lid is valid under 2 conditions (Solomatov, 1995[^cite-solomatov1995]; Foley and Bercovici, 2014[^cite-foley2014], pp. 582, 586):

1. The rheological temperature scale is small relative to the interior temperature: $\Delta T_\text{rh} / T_i = 1 / (g_\text{hom} T_\text{sol}(P_{T_i}) / T_i) \ll 1$.
2. The viscosity contrast across the cold boundary layer exceeds $\exp(8) \approx 3 \times 10^3$, and the contrast across the convecting layer exceeds $\sim 10^4$ (Solomatov, 1995, p. 268, sec. V; p. 270, sec. VI, Fig. 4). The Frank-Kamenetskii contrast parameter measures this:

    $$
    \theta = \frac{H(P_{T_i}) (T_i - T_\text{surf})}{R T_i^2} = \frac{T_i - T_\text{surf}}{\Delta T_\text{rh}} \ge 9
    $$

If $\theta < 9$, the convective regime lies outside the asymptotic stagnant-lid window. The surface-to-interior viscosity contrast $\eta_\text{diff}(T_\text{surf}, P_\text{surf}) / \mu_i$ is substantially larger than $\exp(\theta)$, because of the curvature of the Arrhenius law and partial melt in mushy interiors. `compute_stagnant_lid_state` reports both $\theta$ and `eta_contrast`.

### Lid Base and Sublayer Thickness

**The temperature at the lid base.** The parameter `lid_base_mode` sets the lid base temperature $T_{\text{lid,iso}}$:

- `'fixed'`: a set isotherm, $T_{\text{lid,iso}} = T_\text{base}$ from `lid_base_temperature` (default $1400.0\text{ K}$). A node with $f \ge 1.2 f_\text{conv,min}$ is never part of the lid in this mode (details in section 11).
- `'rheological'`: a value from the temperature scale above:

    $$
    T_{\text{lid,iso}} = T_i - a \Delta T_\text{rh}
    $$

    Here $a = 2.2$ is the lid contrast coefficient (`lid_contrast_coeff`). This coefficient comes from boundary-layer theory for steady Newtonian convection with exponential temperature-dependent viscosity in the limit $\theta \gg 1$ (Solomatov, 1995; Tackley, 2000).

**Where the lid base is.** The lid base is the shallowest of these 3 depths:

1. The rheological isotherm, where $T = T_{\text{lid,iso}}$.
2. The solidus crossing: the shallower of 2 places, where the melt fraction is near $\phi = 0.01$ and, when a solidus profile is supplied, where $T = T_\text{sol}$.
3. The rheological crystallisation front, where $\phi = \phi_\text{rheo} = 0.4$.

A column is the 1-D radial profile of the mantle. Item 2 has a consequence: a column with $\phi \ge 0.012$ at every node has no lid at any mesh resolution (section 9, item 7).

**Special columns.** A column can be non-convecting and below the lid-base criterion at every node, so that no node is marked as interior (the indicator sum $W_\text{base,sum} \to 0$). `r_lid_base` then sits at the inner boundary radius `r[0]` (the core-mantle boundary), which gives $d_\text{lid,raw} = (r_\text{outer} - r_\text{inner}) w_\text{active}$, and $d_\text{lid} = 0$ when $w_\text{active} = 0$. In `'fixed'` mode, a column colder than $T_\text{base}$ that convects on nodes $0 \dots k$ and is conductive above places `r_lid_base` at $r_k$. If no node is non-convecting, `r_lid_base` sits at `r[-1]` with $d_\text{lid,raw} = 0$, $d_\text{lid} = 0$ and `has_lid = False`.

**The sublayer.** Deformation at the lid base takes place in a thin rheological sublayer. Its thickness is the temperature scale divided by the temperature gradient:

$$
\delta_\text{rh} = \frac{\Delta T_\text{rh}}{|dT/dr|_\mathrm{lid\_base}}
$$

The gradient $|dT/dr|_\mathrm{lid\_base}$ is a weighted value from nodal central differences (`xp.gradient(T, r)`). To prevent a division by zero in isothermal or near-adiabatic profiles, the code floors the nodal gradient at a small positive value: a $1\text{ K}$ difference over the minimum grid spacing.

### Lid Mask and Mesh Resolution

Inside the cold solid lid, the mask $w_\text{lid} \to 1$. In the warm or partly molten interior, $w_\text{lid} \to 0$. The mask is smooth, and section 11 gives its formula.

**The mesh matters.** The central difference makes $\delta_\text{rh}$ depend on the radial mesh spacing around the lid base. The `lid_state` dictionary records `lid_cell_count` ($\sum_k w_{\text{lid}, k}$) over the physical lid thickness $d_\text{lid}$. A `lid_cell_count` of about 2 or less shows a lid whose thickness the grid sets. The surface heat flux after the lid forms then depends on the size of the top cell. Resolving such a lid needs a finer mesh near the surface.

The value of $T_i$ has the same limit, because the selector places $T_i$ at the node under the lid. When the lid is one cell thick, $T_i$ lies in the rheological sublayer. An example is a 100-node post-magma-ocean profile with $T_i = 1423.0\text{ K}$, against $1788.6\text{ K}$ one node deeper.

## 4. Convective Driving Stress

**How hard does convection push on the lid?** Flow below the lid exerts a shear stress on its base. This is the driving stress. It equals a viscosity times a velocity gradient, so the code needs the interior viscosity and the convective velocity. `compute_stagnant_lid_state` uses the scaling of Foley and Bercovici (2014) (eqs. 26 and 28, pp. 586-588):

$$
\tau_d = \frac{2 \mu_i v_m}{d} w_\text{active}, \quad \tau_{d,\text{lid}} = \frac{2 \mu_i v_m}{d_\text{geom}} w_\text{active}
$$

| Symbol | Meaning |
|:---|:---|
| $d = r_\text{outer} - r_\text{inner}$ | Mantle thickness |
| $\mu_i$ (`eta_i` and `mu_i` in `lid_state`) | Dynamic viscosity of the convecting interior. In a solver run `viscosity_mixture` is always supplied, so `lid_state['eta_i']` holds $\mu_i$ (section 11) |
| $v_m$ | Interior convective velocity |
| $\tau_{d,\text{lid}}$ | Driving stress scaled by the geometric lid thickness; the solver passes it to `compute_effective_viscosity` |
| $d_\text{geom}$ | Geometric lid thickness with a smooth floor (section 11); the floor uses the top cell width $\Delta r_\text{top}$ |

The factor of 2 comes from horizontal shear between the lid base ($u = v_m$) and mid-depth ($u = 0$ at $z = d/2$), as Foley and Bercovici (2014, pp. 584, 588) derive. $\tau_d$ divides by the full mantle thickness, while $\tau_{d,\text{lid}}$ divides by $d_\text{geom}$. Their ratio is $\tau_{d,\text{lid}} / \tau_d = d / d_\text{geom}$. This ratio is bounded above by $d / \max(\delta_\text{rh}, \Delta r_\text{top})$ on incipient lids, and it vanishes in non-lid columns where $w_\text{lid,present} = 0$.

**The velocity.** The interior velocity follows the boundary-layer scaling for bottom-heated stagnant-lid convection (Foley and Bercovici, 2014, pp. 586-587, eq. 26). Foley and Bercovici note that Solomatov and Moresi (2000)[^cite-solomatov2000] state that this scaling also fits the bottom-heated experiments of Dumoulin et al. (1999)[^cite-dumoulin1999]:

$$
v_m = \frac{\kappa_i}{d} C_4 \left( \frac{\mathrm{Ra}_\text{eff} a_\text{rh}}{\theta} \right)^{2/3}
$$

Here $C_4 = 0.125$ is the empirical prefactor calibrated by Foley and Bercovici (2014, sec. 4.2, p. 589), and $a_\text{rh} = 1.3$ is the rheological sublayer constant fitted jointly with $C_4$. The code takes $\rho_i, g_i, \alpha_i, \kappa_i$ as weighted means at the top of the convecting layer. The effective Rayleigh number measures the vigour of convection. It uses the interior viscosity $\mu_i$:

$$
\mathrm{Ra}_\text{eff} = \frac{\rho_i g_i \alpha_i (T_i - T_\text{surf}) d^3}{\kappa_i \mu_i}
$$

Both $\mathrm{Ra}_\text{eff}$ and $\theta$ scale linearly with the temperature drop $\Delta T = T_i - T_\text{surf}$. The ratio $\mathrm{Ra}_\text{eff} a_\text{rh} / \theta$ is therefore proportional to $a_\text{rh} \Delta T_\text{rh}$, where $\Delta T_\text{rh} = R T_i^2 / H(P_{T_i})$. The surface temperature therefore cancels in this ratio.

**No lid, no stress.** Convection can be inactive or can reach the surface, so that no conductive lid forms ($d_\text{lid,raw} \le 1\text{ m}$ or $T_i - T_\text{surf} \le 10^{-3}\text{ K}$). A smooth weight $w_\text{lid,present}$ then takes the lid diagnostics to zero. The weight keeps them continuous across lid onset. Section 11 gives its formula.

### 4.1 Bottom Heating and Internal Heating

**Does this velocity law hold at all stages of cooling?** It is for bottom heating, so it is an upper estimate later on. The $2/3$ velocity law applies to bottom-heated stagnant-lid convection (Dumoulin et al., 1999; Foley and Bercovici, 2014, p. 587). A cooling terrestrial planet starts with a superheated core. The large temperature difference at the core-mantle boundary drives strong basal heating during early post-magma-ocean evolution (Thiriet et al., 2019[^cite-thiriet2019], pp. 139-140, sec. 1).

As the core cools and radiogenic heating decays, the mantle moves from mainly bottom heating to mainly internal heating (Thiriet et al., 2019, p. 140). For purely internally heated convection, Solomatov and Moresi (2000) (p. 21801, eq. 13; p. 21804, eq. 28; p. 21805, Table 8) obtain an exponent of $1/2$:

$$
v_m = a_u \frac{\kappa}{d} \left( \frac{\mathrm{Ra}_i a_\text{rh}}{\theta} \right)^{1/2}
$$

Here $a_u = 0.385 \pm 0.003$ for Newtonian rheology ($n = 1$) and $a_\text{rh} = 2.4$. $\mathrm{Ra}_i$ is the internal Rayleigh number. In the worked example it has the value of $\mathrm{Ra}_\text{eff}$ of section 4.

**Worked example.** Take the Earth reference state: $\rho_i = 3300\text{ kg m}^{-3}$, $g_i = 9.81\text{ m s}^{-2}$, $\alpha_i = 3 \times 10^{-5}\text{ K}^{-1}$, $T_i = 1600\text{ K}$, $T_\text{surf} = 300\text{ K}$, $d = 2.89 \times 10^6\text{ m}$, $\kappa_i = 10^{-6}\text{ m}^2\text{ s}^{-1}$, $H = 300\text{ kJ mol}^{-1}$ and $\mu_i = 10^{20}\text{ Pa s}$. This gives $\theta = 18.32$ and $\mathrm{Ra}_i = 3.05 \times 10^8$.

| Scaling | Constants | $v_m$ | $\tau_d = 2 \mu_i v_m / d$ |
|:---|:---|:---:|:---:|
| Bottom-heated | $C_4 = 0.125$, $a_\text{rh} = 1.3$ | $3.36 \times 10^{-9}\text{ m s}^{-1}$ | $\approx 0.232\text{ MPa}$ |
| Internally heated | $a_u = 0.385$, $a_\text{rh} = 2.4$ | $8.42 \times 10^{-10}\text{ m s}^{-1}$ | $\approx 0.058\text{ MPa}$ |
| Internally heated | $a_u = 0.385$, $a_\text{rh} = 1.0$ | $5.43 \times 10^{-10}\text{ m s}^{-1}$ | $\approx 0.038\text{ MPa}$ |

At $a_\text{rh} = 2.4$, the internally heated stress is approximately four times lower than the bottom-heated estimate. At $a_\text{rh} = 1.0$ it is about six times lower. The bottom-heated $2/3$ scaling is therefore an upper estimate of the convective shear stress during later, internally heated epochs.

### 4.2 Tested Range and Mushy Mantles

**Over what range was the law tested?** Over $10^5 \le \mathrm{Ra}_0 \le 5 \times 10^7$, far below the values of a mushy mantle. Foley and Bercovici (2014, sec. 4.2, p. 589; Tables 2 and 3, p. 593; Figs. 8-9) calibrate and validate the scaling laws over this range of their reference Rayleigh number $\mathrm{Ra}_0$:

$$
\mathrm{Ra}_0 \in [10^5, \; 5 \times 10^7]
$$

A partly molten mantle lies far outside this range. There, the two-phase log-viscosity blend (`viscosity_solid` $= 10^{21}\text{ Pa s}$, `viscosity_liquid` $= 10^2\text{ Pa s}$ in standard two-phase configurations) reduces the interior viscosity. The Earth reference inputs above give $\rho_i g_i \alpha_i (T_i - T_\text{surf}) d^3 / \kappa_i = 3.05 \times 10^{28}\text{ Pa s}$ at $\Delta T = 1300\text{ K}$.

| Interior viscosity $\mu_i$ | $\mathrm{Ra}_\text{eff}$ |
|:---|:---|
| $\sim 10^2\text{ to }10^7\text{ Pa s}$ | $3.05 \times 10^{26}\text{ to }3.05 \times 10^{21}$ |
| $\sim 10^3\text{ to }10^9\text{ Pa s}$ (near the rheological transition) | $\sim 10^{25}\text{ to }10^{19}$ |

The scaling in a mushy interior is therefore an extrapolation. Solomatov and Moresi (2000, sec. 5.4, p. 21802) note that the velocity law can change at very high Rayleigh numbers, when convective plumes disconnect from the lid base.

### 4.3 Where the Constants Come From

**Where do the constants come from?** From a joint fit to models with grain damage. The constants $a_\text{rh} = 1.3$ and $C_4 = 0.125$ are empirical. Foley and Bercovici (2014, sec. 4.2, p. 589) fit them jointly for their theory-curve constant-healing ($E_h = 0$) damage models. Every numerical model in that paper includes dynamic grain damage, and the paper gives no pure damage-free numerical fit. For temperature-dependent healing, Foley and Bercovici (2014, sec. 5.2, pp. 594-595) find $a_\text{rh} \approx 1.82$.

The value $a_\text{rh} = 1.3$ is lower than the standard stagnant-lid value ($a_\text{rh} \approx 2$; Solomatov and Moresi, 2000; Korenaga, 2009, cited in Foley and Bercovici, 2014, p. 586). The reason is that less effective damage means less of the high-viscosity lid takes part in convection (Foley and Bercovici, 2014, p. 586).

`compute_stagnant_lid_state` adopts $C_4 = 0.125$ and $a_\text{rh} = 1.3$ as one parameter package, calibrated against sub-lid stresses. In Foley and Bercovici (2014), grain damage enters only through the effective viscosity $\mu_\text{eff} = \mu_i (A_i / A_0)^{-m}$. This reduces to the single-phase or mixture viscosity when there is no damage ($A_i = A_0$). The velocity scaling therefore holds without damage mechanics.

The sensitivity to $a_\text{rh}$ is moderate. With $a_\text{rh} = 2.0$ instead of $1.3$, $v_m$ and $\tau_d$ increase by a factor of $(2.0 / 1.3)^{2/3} \approx 1.33$.

Do not confuse $a_\text{rh}$ with `lid_contrast_coeff`. The sublayer parameter $a_\text{rh} = 1.3$ enters the velocity $v_m$. The configured `lid_contrast_coeff` ($a = 2.2$; Solomatov, 1995; Solomatov and Moresi, 2000, sec. 5.1, pp. 21799-21800) sets the isotherm of the rigid lid ($T_{\text{lid,iso}} = T_i - a \Delta T_\text{rh}$). The two values differ on purpose.

### 4.4 Buoyancy Stress of the Sublayer

**How does the driving stress compare with the buoyancy of the sublayer?** In the Earth example below, the ratio is about 0.1 to 0.15. `compute_stagnant_lid_state` also calculates the buoyancy stress of the sublayer. This is a diagnostic entry in `lid_state`:

$$
\tau_\text{buoy} = \rho_i g_i \alpha_i \Delta T_\text{rh} \delta_\text{rh} w_\text{active}
$$

Here $\delta_\text{rh}$ is the sublayer thickness and $\Delta T_\text{rh} = R T_i^2 / H(P_{T_i})$. `lid_state` also holds the force-balance ratio $\tau_d / \tau_\text{buoy}$.

**Worked example.** Take the Earth reference state of section 4.1, with $\mu_i = 10^{20}\text{ Pa s}$, $\theta \approx 18.32$, $\Delta T_\text{rh} \approx 71.0\text{ K}$ and $\tau_d \approx 0.232\text{ MPa}$.

| Evaluation | Lid thickness | $\delta_\text{rh}$ | $\tau_\text{buoy}$ | $\tau_d / \tau_\text{buoy}$ |
|:---|:---:|:---:|:---:|:---:|
| 50-node piecewise-linear reference profile | $d_\text{lid} = 378.4\text{ km}$ | $\approx 30.8\text{ km}$ ($30791\text{ m}$) | $\approx 2.12\text{ MPa}$ | $\tau_d / \tau_\text{buoy} \approx 0.1095$ |
| Boundary-layer scaling across the 7 conductive intervals | $d_\text{lid,cond} = 7 \times 58.98\text{ km} = 412.9\text{ km}$ | $\delta_\text{rh} = d_\text{lid,cond} / \theta \approx 22.5\text{ km}$ | $\approx 1.55\text{ MPa}$ | $\tau_d / \tau_\text{buoy} \approx 0.150$ |

The two rows differ because `xp.gradient(T, r)` spans the isothermal-to-conductive corner on the 50-node profile. In addition, `lid_state` holds the shifted soft-maximum convective velocity $v_i$ as a diagnostic, evaluated over the upper mantle with `SOFTMAX_VELOCITY_SCALE` ($3.17 \times 10^{-12}\text{ m s}^{-1}$).

## 5. When the Lid Yields

**What happens when the stress reaches the yield stress?** Below the yield stress, the rock follows the creep law. At the yield stress, the rock cannot carry more stress, so its effective viscosity falls. `aragog.rheology_lid.compute_effective_viscosity` combines creep and yielding with a smooth minimum (Moresi and Solomatov, 1998, p. 672, eqs. 13-14; Tackley, 2000, p. 8, eq. 8):

$$
\eta_\text{yield} = \frac{\tau_\text{yield}}{D}, \quad \eta_\text{eff} \approx \begin{cases} \eta_\text{diff}, & \tau < \tau_\text{yield} \\ \min(\eta_\text{diff}, \eta_\text{yield}), & \tau \ge \tau_\text{yield} \end{cases}
$$

**The form in `'lid'` mode.** This mode needs the interior viscosity $\mu_i$ (argument `eta_i`), the driving stress $\tau_d$ and the lid yield stress $\tau_{y,\text{lid}}$. In solver pass 2 the driving stress is $\tau_{d,\text{lid}}$ of section 4. It divides by the lid thickness $d_\text{geom}$, not by the mantle thickness $d$. The strain rate invariant on the lid is $D_\text{lid} = 2 \dot{\epsilon}_\text{lid} = 2 v_m w_\text{active} / d_\text{geom} = \tau_{d,\text{lid}} / \mu_i$ (Foley and Bercovici, 2014, p. 588, eq. 28, with $d_\text{geom}$ in place of $d$).

Putting this in gives the plastic branch $\log_{10}\eta_{y,\text{branch}}(r) = \mathrm{softmin}\big(\log_{10}\eta_\text{diff}(r), \; \log_{10}(\mu_i \tau_{y,\text{lid}} / \tau_{d,\text{lid}}), \; \mathrm{LOCAL\_CAP\_LOG10\_WIDTH}\big)$. The code blends this branch with creep, in $\log_{10}$ space, across $\tau_{d,\text{lid}} = \tau_{y,\text{lid}}$. The blend has the linear stress-ratio width `YIELD_BLEND_RATIO_WIDTH = 0.025`:

$$
w_b = \frac{1}{2}\left(1 + \tanh\left(\frac{\tau_{d,\text{lid}} / \tau_{y,\text{lid}} - 1}{\mathrm{YIELD\_BLEND\_RATIO\_WIDTH}}\right)\right), \quad \log_{10}\eta_\text{eff}(r) = (1 - w_b)\log_{10}\eta_\text{diff}(r) + w_b \log_{10}\eta_{y,\text{branch}}(r)
$$

**Well below yield.** When $\tau_{d,\text{lid}} \ll \tau_{y,\text{lid}}$, $w_b \to 0$ and $\eta_\text{eff}(r) \to \eta_\text{diff}(r)$. The lid stays rigid and conductive. For example, standard Byerlee rock friction ($c = 50\text{ MPa}$, $\mu = 0.6$, $\tau_\text{max} = 500\text{ MPa}$) gives $\tau_{y,\text{lid}} = 500\text{ MPa}$. On the 50-node Earth reference profile of section 4.4 ($d_\text{geom} = 378.4\text{ km}$), $\tau_{d,\text{lid}} \approx 1.77\text{ MPa}$. The yield stress exceeds it by a factor of about 280, or 2.45 orders of magnitude, so the lid does not yield.

**Above yield.** When $\tau_{d,\text{lid}} > \tau_{y,\text{lid}}$, $w_b \to 1$ and $\eta_\text{yield} = \mu_i (\tau_{y,\text{lid}} / \tau_{d,\text{lid}}) < \mu_i$. The yielded boundary layer is then less viscous than the convecting interior, and the surface boundary layer moves into thin-boundary-layer convection.

For example, take a cold surface ($T_\text{surf} = 300\text{ K}$, $\eta_\text{diff} = 10^{40}\text{ Pa s}$, $\mu_i = 10^{21}\text{ Pa s}$) with $\tau_{d,\text{lid}} / \tau_{y,\text{lid}} = 1.2$. The yielded viscosity is $\eta_\text{eff} = 8.33 \times 10^{20}\text{ Pa s} < \mu_i$. If the code took the strain rate from the cold-lid creep viscosity ($\eta_\text{diff} \tau_{y,\text{lid}} / \tau_{d,\text{lid}}$), the result would be $\eta_\text{eff} = 8.33 \times 10^{39}\text{ Pa s}$.

An alternative harmonic-mean form $\eta_\text{eff} = \eta_\text{diff} \eta_\text{yield} / (\eta_\text{diff} + \eta_\text{yield})$ is discussed by Foley and Becker (2009)[^cite-foley2009] (p. 3, eqs. 7-8) and Foley and Bercovici (2014, sec. 8.2).

**Only the lid yields.** The code applies the closure viscosity only in the cold boundary layer, through the smooth lid mask:

$$
\log_{10} \eta_\text{solid} = w_\text{lid} \log_{10} \eta_\text{eff} + (1 - w_\text{lid}) \log_{10} \eta_\text{diff}
$$

Inside the cold lid ($w_\text{lid} \to 1$), $\eta_\text{solid} \to \eta_\text{eff}$. In the warm convective interior ($w_\text{lid} \to 0$), $\eta_\text{solid} \to \eta_\text{diff}$. This restricts plastic yielding to the cold boundary layer, without altering the interior convective mantle (Foley and Becker, 2009, sec. 3.2.2).

**When convection stops.** As $v_i \to 0$, convective heat transport drops below $f_\text{conv,min}$. In this limit, $w_\text{active} \to 0$ and $\tau_d = 0$. The lid closure is inactive, and no division by zero occurs.

The closure is everywhere finite, $C^\infty$ smooth, and non-increasing with driving stress. Section 11 gives the floors of the closure.

## 6. Interaction with the Eddy-Diffusivity Floor

**Why does a solid lid need the floor to vanish?** Aragog can lift $\kappa_h$, the thermal eddy diffusivity of mixing-length convection (MLT) in $\text{m}^2\text{ s}^{-1}$, to a minimum value, the floor. In a solid lid, a floor above zero is artificial convective heat transport. When `rheology.enabled` is `True`, the floor is therefore exactly zero at $\phi = 0$.

**The ramp.** The floor follows the melt fraction $\phi$ through a $\tanh$ ramp $f(\phi)$. $f_\text{zero}(\phi)$ is the same ramp, shifted and scaled so that it is exactly 0 at $\phi = 0$ and 1 at $\phi = 1$:

$$
f(\phi) = \frac{1}{2} \left( 1 + \tanh\left( \frac{\phi - \phi_\text{rheo}}{\phi_\text{width}} \right) \right), \quad f_\text{zero}(\phi) = \mathrm{clip}\left( \frac{f(\phi) - f(0)}{\max(f(1) - f(0), 10^{-30})}, \; 0, \; 1 \right)
$$

**The floor.** `aragog.jax.phase.compute_mlt` and `aragog.solver.entropy_state.apply_kappah_floor` apply the floor without a lid mask:

$$
\kappa_{h,\text{floor,eff}} = \kappa_{h,\text{floor}} \cdot f_\text{floor}(\phi) \cdot I_\text{conv}, \quad \kappa_h = \max\big(\kappa_{h,\text{MLT}}, \; \kappa_{h,\text{floor,eff}}\big)
$$

- $f_\text{floor}(\phi) = f_\text{zero}(\phi)$ when `rheology.enabled` is `True`, and $f_\text{floor}(\phi) = f(\phi)$ when it is `False`.
- $I_\text{conv} = 1$ in cells with $\partial S / \partial r < D_\text{thr}$, and $0$ in cells with $\partial S / \partial r \ge D_\text{thr}$. Cells with $I_\text{conv} = 0$ have $\kappa_{h,\text{floor,eff}} = 0$.
- $D_\text{thr}$ is the convective threshold, `energy.convective_threshold` on `EnergyConfig` and `PhaseParams.convective_threshold`. Its default on `EnergyConfig` and `PhaseParams.__init__` is $0.0\text{ J kg}^{-1}\text{ K}^{-1}\text{ m}^{-1}$. When $D_\text{thr} = 0$, the first group of cells is convectively unstable and the second is stably stratified.

**Defaults.**

| Setting | Standalone Aragog | Coupled PROTEUS (`proteus.config._interior.Aragog`) |
|:---|:---|:---|
| $\phi_\text{rheo}$ | $\phi_\text{rheo} = 0.4$ | `rfront_loc = 0.4` |
| $\phi_\text{width}$ | `rheological_transition_width = 0.15` | `rfront_wid = 0.2` |
| $f(0)$ | $0.004805$ | $0.018$ |
| $f(1) - f(0)$ | $0.99486$ | $0.97954$ |
| $\kappa_{h,\text{floor}}$ | `kappah_floor` $= 10.0\text{ m}^2\text{ s}^{-1}$ on `EnergyConfig` and `_EnergyParameters` | `kappah_floor = 10.0` |

The direct `PhaseParams.__init__` fallback in `aragog.jax.phase` is $0.0\text{ m}^2\text{ s}^{-1}$.

**What this means for a fully solid cell ($\phi = 0$).**

| Case | Floor weight at $\phi = 0$ | $\kappa_{h,\text{floor,eff}}$ in a cell with $\partial S / \partial r < D_\text{thr}$, at $\kappa_{h,\text{floor}} = 10\text{ m}^2\text{ s}^{-1}$ |
|:---|:---:|:---:|
| `rheology.enabled = False`, width 0.15 | 0.0048 | $0.048\text{ m}^2\text{ s}^{-1}$ |
| `rheology.enabled = False`, width 0.2 | 0.018 | $0.18\text{ m}^2\text{ s}^{-1}$ |
| `rheology.enabled = True`, either width | $f_\text{zero}(0) = 0.0$ | $0.0\text{ m}^2\text{ s}^{-1}$ |

Without rheology, the floor exceeds the molecular thermal diffusivity ($\kappa \sim 10^{-6}\text{ m}^2\text{ s}^{-1}$) by four to five orders of magnitude. With rheology, an unstable solid cell carries heat only by mixing-length convection and conduction, and a conductive lid can form at the cold top. The molten layers keep the floor: $f_\text{zero}(1) = 1.0$ and $\kappa_{h,\text{floor,eff}}(\phi = 1) = 10.0\text{ m}^2\text{ s}^{-1}$.

**The surface can cool to the lowest temperature of the table.** The floor with $f_\text{zero}$ removes artificial convective heat transport in solid cells, so the surface cell cools toward the radiative equilibrium temperature $T_\text{eqm}$. With `rheology.enabled = True` and a tabular entropy equation of state (`const_properties = False`), `EntropySolver._check_rheology_invariants` requires the entropy $S$ on the staggered mesh of every accepted state to stay inside the table bounds $[S_\text{min}, S_\text{max}]$. If a state leaves the table, it raises `RuntimeError` and names the node index, the time $t$ and the entropy $S$.

The standard test table of the continuous integration (`test-data-v1`) has $S \in [-1606.28, 19734.29]\text{ J kg}^{-1}\text{ K}^{-1}$. Its lowest temperatures are $T_\text{min}(P = 10^5\text{ Pa}) = 320.23\text{ K}$ at the surface and $T_\text{min}(P = 0.218\text{ GPa}) = 320.82\text{ K}$ at the top staggered cell of `cfg/abe_solid.toml`. The validity domain requires the surface temperature to stay above about $320\text{ K}$ and $T_\text{eqm} \ge 325\text{ K}$.

| $T_\text{eqm}$ with `cfg/abe_solid.toml` | Result |
|:---|:---|
| $350\text{ K}$ | Inside the table, $\min S_\text{stag} = -326.01\text{ J kg}^{-1}\text{ K}^{-1}$ |
| $325\text{ K}$ | Inside the table, $\min S_\text{stag} = -392.51\text{ J kg}^{-1}\text{ K}^{-1}$ |
| $273\text{ K}$ | The top cell crosses below $S_\text{min} = -1606.28\text{ J kg}^{-1}\text{ K}^{-1}$ at about $17\text{ to }18\text{ kyr}$ ($17.44\text{ kyr}$ in the ODE energy trace, $18.08\text{ kyr}$ at the accepted segment endpoint), and `_check_rheology_invariants` stops the run |

For a surface that cools below $320\text{ K}$ (such as $T_\text{eqm} = 273\text{ K}$), the Zalmoxis low-temperature PALEOS solid extension extends the solid equation-of-state (EOS) table down to $50\text{ K}$ for pressures up to $12\text{ GPa}$. It is `load_paleos_with_lowt_extension` in `zalmoxis.eos_export` and `tools/converters/paleos_lowt_extension.py`. In PROTEUS it is enabled with `interior_struct.zalmoxis.lowt_extension = true`, with the table `paleos_mgsio3_tables_pt_proteus_solid_lowT.dat`.

## 7. Mixing Length in the Solid

**How does a stiff lid change convection near the surface?** Mixing-length theory needs a length, the distance that a convecting parcel travels. A cold, stiff lid shortens this length near the top boundary. Aragog represents this with a top slope that falls as the viscosity contrast $\theta$ rises.

### Viscosity-Contrast-Dependent Mixing-Length Slope

The effective mixing length $l_v(r)$ near the mantle boundaries is the smaller of 2 lines, one from each boundary:

$$
l_v(r) = \min\big( s_\text{bot} (r - r_\text{in}), \; s_\text{top}(\theta) (r_\text{out} - r) \big)
$$

Here $s_\text{bot}$ is the lower boundary slope (`mlt_bottom_slope`, default $1.0$). `compute_mlt_top_slope` calculates the upper boundary slope:

$$
s_\text{top}(\theta) = s_{\text{top},\infty} + (1 - s_{\text{top},\infty}) \exp\big( -c_\theta \max(\theta, 0) \big)
$$

The decay rate is $c_\theta = 0.15$ (`c_theta`), and the stagnant-lid slope is $s_{\text{top},\infty} = 0.22$ (`s_top_inf`). `SolidRheologyParams`, `compute_mlt_top_slope` and `tests/test_mlt_calibration.py` use these defaults. They give 3 values:

- The exact isoviscous limit: $s_\text{top}(\theta \le 0) = 1.0$.
- An intermediate value: $s_\text{top}(15) = 0.22 + 0.78 \exp(-2.25) = 0.3022113951582542$.
- The stagnant-lid limit: $s_\text{top}(\theta \to \infty) \to 0.22$. This is the limit of the strong stagnant-lid regime, $\theta \gg 8$.

**How the slope acts on the flux.** The viscous eddy diffusivity scales as $l^4$ and the viscous velocity scales as $l^3$. `viscous_mixing_length_factor` therefore returns $(l_v / l)^4$. At interior nodes, the solver multiplies the viscous velocity by $(l_v / l)^3$ and the viscous eddy diffusivity by $(l_v / l)^4$. The factor stays $1.0$ at the boundary nodes, where $l = 0$.

With `stress_closure_mode = 'local'` the solver passes no $\theta$ to this factor. The upper slope is then $s_{\text{top},\infty}$ at every $\theta$, so `c_theta` has no effect.

**Known limit.** A single upper slope without a spherical-curvature geometry factor does not match both the planar and the spherical benchmark sets. [Top slope against published grids](../Validation/solid_state_convection.md#the-top-slope-against-published-convection-grids) gives the conditions and the numbers.

## 8. Stress Closure Modes

**Which strain rate does the closure use?** Either the one from the convecting interior (`'lid'`) or the one from the local cell (`'local'`). `stress_closure_mode` sets this. In `SolidRheologyParams` it defaults to `'lid'`. Callers pass it to `compute_effective_viscosity`.

| Mode | Strain rate from | Result |
|:---|:---|:---|
| Lid (`'lid'`) | Convecting interior scaling (section 4) | Yielding moves the whole lid |
| Local (`'local'`) | Local cell velocity and mixing length | Yielding raises the local convective velocity |
| Global (`global`) | Unsupported | Configuration validation rejects it and tells the user to specify `lid` |

### The `lid` mode

This is the boundary-layer closure. In solver pass 2 the driving stress is $\tau_{d,\text{lid}} = (2 \mu_i v_m / d_\text{geom}) w_\text{active}$ from section 4. The plastic yield viscosity $\eta_\text{yield} = \mu_i (\tau_{y,\text{lid}} / \tau_{d,\text{lid}})$ mobilises the lid nodes when $\tau_{d,\text{lid}} > \tau_{y,\text{lid}}$.

### The `local` mode

The strain rate comes from the local cell velocity and mixing length, with the second invariant convention of section 2:

$$
\dot{\epsilon}_\text{local}(r) = \frac{|v_\text{unyielded}(r)|}{2 \max(l(r), 10^{-15}\text{ m})}
$$

In this mode, yielding is not a stress cap. The two-pass mixing-length solver works in these steps:

1. Pass 1 calculates the viscous velocity $v_{\text{visc},1} = \rho g \alpha T (-\partial S / \partial r)_\text{eff} l_v^3 / (18 C_p \eta_\text{diff})$ (Bower et al., 2018[^cite-bower2018], eq. 5a; p. 4 of arXiv:1711.07303v1).
2. The pass-1 viscous shear stress is $\tau_1 = 2 \eta_\text{diff} \dot{\epsilon}_\text{local} = \eta_\text{diff} |v_{\text{visc},1}| / l_v = \rho g \alpha T (-\partial S / \partial r)_\text{eff} l_v^2 / (18 C_p) = \tau_{\text{buoy},1}$. This depends on the local buoyancy and mixing length, not on $\eta_\text{diff}$.
3. Where $\tau_{\text{buoy},1} > \tau_y(r)$, pass 1 reduces the viscosity to $\eta_\text{eff}(r) \approx \eta_\text{diff}(r) \tau_y(r) / \tau_{\text{buoy},1}$.
4. Pass 2 scales the viscous velocity $v_{\text{visc},2}$ and the thermal eddy diffusivity $\kappa_h$ up by the factor $\tau_{\text{buoy},1} / \tau_y(r)$. In the viscous regime ($\mathrm{Re} \ll \mathrm{Re}_\text{crit}$), the stress stays at $\tau_2 = \eta_\text{eff} |v_{\text{visc},2}| / l_v = \tau_{\text{buoy},1}$.

Thus in `'local'` mode yielding scales the convective velocity by $\tau_{\text{buoy},1} / \tau_y$. Both passes use the upper mixing-length slope $s_{\text{top},\infty}$ with no $\theta$, so `c_theta` has no effect.

## 9. Limits of the Model

**What do the kernels give when one quantity goes to a limit?** The kernels in `aragog.rheology` and `aragog.rheology_lid` have these limits.

1. **Zero driving stress ($\tau_{d,\text{lid}} \to 0$):** $\tau_{d,\text{lid}} / \tau_{y,\text{lid}} \to 0 \implies w_y \approx 0$ ($\frac{1}{2}(1 - \tanh 10) \approx 2.1 \times 10^{-9}$ with a lid, or $0$ when `has_lid` is `False`) and $\eta_\text{eff} = \eta_\text{diff}$. This is pure Arrhenius diffusion creep without plastic weakening.
2. **Infinite yield stress ($\tau_y \to \infty$):** $\tau_{d,\text{lid}} / \tau_{y,\text{lid}} \to 0 \implies \eta_\text{eff} = \eta_\text{diff}$. Yielding stays inactive, and the lid stays stagnant at diffusion-creep viscosity.
3. **Small homologous activation parameter ($g_\text{hom} \to 0^+$):** `SolidRheologyParams` requires `g_homologous > 0` when `enabled = True` in both `'rheological'` and `'fixed'` lid modes. In `compute_arrhenius_viscosity`, $g_\text{hom} \to 0$ gives $H(P) \to 0$ and an isoviscous solid mantle $\eta_\text{diff} = \eta_\text{solid} f_\text{water}$.
4. **Zero solidus pressure slope ($dT_\text{sol} / dP = 0$):** $H(P) = g_\text{hom} R T_{\text{sol},0}$. This is temperature-only Arrhenius creep with constant activation energy.
5. **Infinite Frank-Kamenetskii contrast ($\theta \to \infty$):** the asymptotic stagnant-lid regime of Solomatov (1995), with upper slope $s_\text{top}(\theta) \to s_{\text{top},\infty}$.
6. **Sub-threshold interior convection ($f_\text{int,max} \ll f_\text{conv,min}$) or no conductive lid ($d_\text{lid,raw} \to 0$):** when $f_\text{int,max}$ lies more than $0.01\,\mathrm{arctanh}(0.9) \approx 0.0147$ below $f_\text{conv,min}$ (for example $f_k \le 0.035$ at $f_\text{conv,min} = 0.05$), $w_\text{active} \le 0.05$. `is_convecting` and `has_lid` are `False`. The weight $w_\text{lid,present} = w_d w_T w_\mathrm{act\_gate} w_\mathrm{sum\_gate} w_\text{weight}$ smoothly takes the boundary-layer diagnostics ($\tau_d$, $v_m$, $\mathrm{Ra}_\text{eff}$, $d_\text{lid}$, $w_y$, `lid_regime`) to zero. Yielding stays inactive at any mesh resolution $N$.
7. **Melt at every node ($\phi \ge 0.012$ at every node, with or without a solidus table):** the solidus ramp $w_{\text{sol},\phi}$ of section 11 is 1, so $w_\text{hot,sol} = 1$ and $w_\text{interior} = 1$ identically through the column, however steep the temperature gradient is. The lid-base weight $\tilde{W}_\text{base}$ places $r_\text{lid,base}$ strictly at the surface node $r[-1]$. This gives $d_\text{lid,raw} = 0$, `has_lid = False`, $d_\text{lid} = 0$, $\tau_d = 0$ and `lid_regime = 0.0` at all mesh resolutions $N$, including $N \in \{3, 10, 50, 100, 200\}$ on both Earth and $9000\text{ km}$ mantle spans.
8. **High driving stress ($\tau_{d,\text{lid}} / \tau_{y,\text{lid}} > 1$):** $w_y \to 1$ and $\eta_\text{eff} \to \min(\eta_\text{diff}, \mu_i \tau_{y,\text{lid}} / \tau_{d,\text{lid}}) < \mu_i$. The cold surface boundary layer moves into thin-boundary-layer convection.

## 10. Parameters

**What can a user set?** `SolidRheologyParams` in `aragog.rheology` holds the parameters. `PhaseConfig.rheology` in `aragog.config.phases` exposes them.

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
| `activation_energy` | J/mol | `None` |
| `activation_volume` | m^3/mol | `0.0` |
| `activation_volume_decay_pressure` | Pa | `inf` |

## 11. Code Details

This section is for a reader who works on the kernels. It holds the weights, floors and switches behind sections 3 to 8. You do not need it to understand the physics. Open a box to see its content.

### How the Solver Calls the Kernels

??? note "Two-pass update and the bypass"
    When `rheology.enabled` is `True`, the solver evaluates the kernels in `aragog.rheology` and `aragog.rheology_lid` in a two-pass update: `EntropyState.update` on the NumPy path and `aragog.jax.phase.compute_fluxes` on the JAX path. Solver pass 2 passes $\tau_{d,\text{lid}}$ as `tau_d` to `compute_effective_viscosity`.

    When `rheology.enabled` is `False` (the default), the solver bypasses the kernels and reproduces the baseline liquid-parameterised trajectory.

### Switches, Floors and Caps by Function

??? note "`aragog.rheology`"
    | Function | Switch, floor or cap |
    |:---|:---|
    | `compute_mlt_top_slope` | $\max(\theta, 0)$ |
    | `viscous_mixing_length_factor` | $l_v = \min(s_\text{bot}(r - r_\text{in}), s_\text{top}(\theta)(r_\text{out} - r))$ and the $l > 0$ branch |
    | `compute_arrhenius_viscosity` | Lower exponent clip $-700$; smooth $\log_{10}$ softplus ceiling `viscosity_max_log10` (width `VISCOSITY_CEILING_LOG10_WIDTH = 0.05`) |
    | `compute_yield_stress` | $\max(0, \min(c + \mu P, \tau_\text{max}))$ |
    | `compute_strain_rate_local` | $\lvert v \rvert / (2 \max(l_\text{mix}, \epsilon))$ |

??? note "`aragog.rheology_lid`"
    | Place | Switch, floor or cap |
    |:---|:---|
    | `_boundary_weights` | Boltzmann-weighted smooth maximum (`SMOOTH_MAX_REL_WIDTH = 0.02`); $\max(W_\text{sum}, 10^{-12})$ denominator |
    | Flux fraction | Smooth softplus 5-node local interior RMS $\lvert F_\text{tot} \rvert$ floor (`F_TOT_FLOOR_FRAC = 1e-3`, smooth floor at $10^{-12}$, continuous in every $F_k$); $[0, 1]$ clip of $f_k$ |
    | Lid presence | Boolean `has_lid` flag; smooth product weight $w_\text{lid,present} = w_d w_T w_\mathrm{act\_gate} w_\mathrm{sum\_gate} w_\text{weight}$ (`NO_LID_THICKNESS_WIDTH = 0.25`, `NO_LID_DT_WIDTH = 2.5e-4`, `CONV_ACTIVE_GATE_WIDTH = 0.0025`, `CONV_SUM_GATE_WIDTH = 2.5e-7`) |
    | `_melt_ramp` and $w_\text{solid,surf}$ | Compactly supported $C^2$ quintic melt-fraction ramp $(t^3)(10 - 15 t + 6 t^2)$ |
    | `scale` and `scale_mask` | Smooth softplus floors and caps (`SCALE_FLOOR_WIDTH = 0.005`, `SCALE_MASK_REL_WIDTH = 0.005`) |
    | $\delta_\text{rh}$ | Radial temperature-gradient floor $\max(\lvert dT/dr \rvert, 1 / \max(\Delta r_\text{min}, 1\text{ m}))$; mantle-thickness cap $\min(\delta_\text{rh}, r_\text{top} - r_\text{bot})$ |
    | `compute_effective_viscosity` | `COHESION_FLOOR` ($10^{-6}\text{ Pa}$) on yield stress; smooth $\tanh$ yield transition in `'lid'` mode (`YIELD_BLEND_RATIO_WIDTH = 0.025`); regularized strain-rate floor (`LOCAL_STRAIN_RATE_FLOOR = 1e-100`) and smooth $\log_{10}$ softplus cap (`LOCAL_CAP_LOG10_WIDTH = 0.01`) in `'local'` mode |

### Interior Selector

??? note "Convective flux fraction and its floor"
    The fraction is $f_k = \max(F_{\text{conv}, k}, 0) / |F_{\text{tot}, k}|_\text{safe}$. Here $|F_{\text{tot}, k}|_\text{safe}$ applies a smooth softplus floor against $F_{\text{tot,floor}, k} = \mathrm{softmax}(10^{-3} F_{\text{loc,rms}, k}, 10^{-12})$. $F_{\text{loc,rms}, k}$ is the 5-node local interior RMS of $F_\text{tot}$.

    Boundary nodes $k = 0$ and $k = N - 1$ are replaced by their adjacent interior nodes $F_{\text{tot}, 1}$ and $F_{\text{tot}, N - 2}$ before 2-node edge replication.

    This floor is continuous in every $F_k$. It differs from a scalar $10^{-12}$ floor only at nodes where $F_{\text{conv}, k} > 0$ and $|F_{\text{tot}, k}| < 0.0014 F_{\text{loc,rms}, k}$ when $10^{-3} F_\text{loc,rms} \gg 10^{-12}\text{ W/m}^2$.

??? note "Activity indicator and boundary sum"
    The interior peak is $f_\text{int,max} = \max_{1 < k < N} f_k$. The baseline-subtracted interior indicator is $w_{\text{conv,int}, k} = \max(w_{\text{conv}, k} - w_\text{conv}(f = 0), 0) I_{1 < k < N}$.

    From these, `compute_stagnant_lid_state` calculates the mesh-independent activity indicator $w_\text{active} = \frac{1}{2}(1 + \tanh((f_\text{int,max} - f_\text{conv,min}) / 0.01))$. It also calculates the shallowest-crossing boundary sum $W_\text{sum} = \sum_k w_{\text{conv,int}, k} \prod_{j > k} (1 - w_{\text{conv,int}, j})$, scanned from the outermost surface node $k = N$ inward.

    The mesh-independent transition weight is $w_\text{weight} = \tanh(\text{smax}(w_{\text{conv,int}}) / 10^{-6})$. `is_convecting` requires $W_\text{sum} > 10^{-6}$ and $w_\text{active} > 0.05$.

??? note "Selection of the interior node"
    The scale is $f_\text{int,scale} = \max(f_\text{int,max}, 10^{-12})$, and the threshold is $f_{T_i} = 0.9 f_\text{int,max}$. The selector on non-boundary nodes ($1 < k < N$) is:

    $$
    w_{T_i, k} = \frac{1}{2} \left( 1 + \tanh\left( \frac{f_k - f_{T_i}}{0.02 f_\text{int,scale}} \right) \right) I_{1 < k < N}
    $$

    The surface-down cumulative product $W_k = w_{T_i, k} \prod_{j > k} (1 - w_{T_i, j})$ over shallower nodes $j > k$ isolates the shallowest convection-dominated node. It is normalised to $\tilde{W}_{T_i, k}$ and blended by $\tilde{W}_k = w_\text{weight} \tilde{W}_{T_i, k} + (1 - w_\text{weight}) \tilde{W}_{\text{conv}, k}$. $\tilde{W}_{\text{conv}, k}$ is the shallowest-crossing weight of $w_{\text{conv,int}, k}$, with fallback to the deepest interior node $k = 2$. The weighted means are:

    $$
    T_i = \sum_{k=1}^N \tilde{W}_k T_k, \quad P_{T_i} = \sum_{k=1}^N \tilde{W}_k P_k
    $$

??? note "Weak convection"
    These values hold at the default $f_\text{conv,min} = 0.05$ (`interior_flux_fraction = 0.05`).

    | Interior peak | $w_\text{weight}$ | Result |
    |:---|:---|:---|
    | $f_\text{int,max} < 0.05 + 0.01\,\mathrm{arctanh}(-0.9) = 0.035278$, with $\text{smax}(w_{\text{conv,int}}) \gg 10^{-6}$ | $w_\text{weight} \approx 1$; $w_\text{weight} \ge 1 - 4 \times 10^{-9}$ at $f_\text{int,max} \ge 10^{-3}$ | $T_i$ still selects the top-of-convection node near $0.9 f_\text{int,max}$ |
    | Near $f_\text{int,max} \approx 6 \times 10^{-5}$: $f_\text{int,max} = 6.01 \times 10^{-5}$ | $0.5$ | Transition band |
    | $f_\text{int,max} = 10^{-4}$, where $\text{smax}(w_{\text{conv,int}}) = 9.17 \times 10^{-7}$ | $0.7245$ | Transition band |
    | $f_\text{int,max} \to 0$, with $\text{smax}(w_{\text{conv,int}}) \ll 10^{-6}$ | $w_\text{weight} \to 0$ | $T_i$ relaxes smoothly to the interior fallback node $T_2$ |

    These transition values do not hold at other thresholds, because $w_\text{conv}(f) - w_\text{conv}(0)$ scales with $\mathrm{sech}^2(f_\text{conv,min} / 0.01)$. For example, at $f_\text{conv,min} = 0.20$, $w_\text{conv}(f) - w_\text{conv}(0)$ underflows to $0.0$ in IEEE 754 float64 for $f \le 0.0101$, and $w_\text{weight}$ reaches $0.5$ near $f_\text{int,max} \approx 0.128$.

??? note "Interior viscosity with a mixture viscosity"
    When a mixture viscosity profile `viscosity_mixture` ($\eta_\text{mix}$) is supplied to `compute_stagnant_lid_state`, the code calculates $\mu_i$ with the smooth weight $w_\text{weight}$ in logarithmic space:

    $$
    \log_{10} \mu_i = w_\text{weight} \sum_{k=1}^N \tilde{W}_k \log_{10} \eta_{\text{mix}, k} + (1 - w_\text{weight}) \log_{10} \eta_\text{diff}(T_i, P_{T_i})
    $$

    When `viscosity_mixture` is `None`, $\mu_i$ equals the solid Arrhenius interior viscosity $\eta_i = \eta_\text{diff}(T_i, P_{T_i})$. `lid_state` returns the same value as `eta_i` and `mu_i`, and it returns the Arrhenius value as `eta_solid_arrhenius`.

#### Geotherm Endpoint Blending and Smooth Ramp Weighting

??? note "Below the threshold: stay on the geotherm"
    This applies when $w_\text{weight} < 1$ on a strictly decreasing radial temperature profile ($T_{k+1} < T_k - 10^{-12}\text{ K}$). $P_{T_i}$ and $r_i$ are then evaluated along the monotone geotherm $(T(r), P(r), r)$ at $T_i$.

    The endpoint-consistent correction is $w_\text{eff} = \mathrm{clip}((T_0 - T_i) / (T_0 - T_1), 0, 1)$, between $(T_0, P_0, r_0) = (T_2, P_2, r_2)$ and $(T_1, P_1, r_1) = \sum_k \tilde{W}_{T_i, k} (T_k, P_k, r_k)$. $\Delta T_\text{rh}$, $\theta$ and $\log_{10}\eta_\text{arr}$ blend with $w_\text{eff}$ between $(T_0, P_0)$ and $(T_1, P_1)$. The sub-threshold state therefore stays on the geotherm and does not blend linearly off it.

    `P_T_i` and `r_i` are diagnostics that no solver term reads. They are on the geotherm for a monotone $T$ profile and are the $\tilde{W}_k$ mean otherwise. The two can differ by tens of GPa across the monotonicity switch.

### Lid-Base Locator and Mask

??? note "Solidus ramp"
    The melt-fraction indicator is the compactly supported $C^2$ quintic ramp $w_{\text{sol},\phi} = t^3(10 - 15 t + 6 t^2)$ with $t = \mathrm{clip}((\phi - 0.008) / 0.004, 0, 1)$. It is centered at $\phi = 0.01$ (`SOLIDUS_MELT_FRACTION_THRESHOLD = 0.01`) with half-width $0.002$ (`SOLIDUS_MELT_FRACTION_WIDTH = 0.002`).

    With a supplied $T_\text{sol}$, the code takes the elementwise maximum with the thermal solidus `tanh` ramp, the indicator of $T = T_\text{sol}$. For $\phi \ge 0.012$, $w_{\text{sol},\phi} = 1$ identically. Any column with $\phi \ge 0.012$ at every node therefore has $r_\text{lid,base}$ strictly at $r[-1]$, with $d_\text{lid} = 0$ and `has_lid = False` for every mesh resolution $N$.

    For $\phi \le 0.008$, $w_{\text{sol},\phi} = 0$ identically, so the transition is confined to the compact interval $\phi \in [0.008, 0.012]$. The ramp bounds the raw lid thickness by $d_\text{lid,raw} \le w_\text{active} \sum_{k=0}^{N-2} (1 - w_{\text{interior},k+1}) \Delta r_k$.

??? note "Locator and capped mask"
    The lid-base locator $\tilde{W}_{\text{base}, k}$ scans the cell-width interior indicator $w_\text{interior} = \max(w_\text{hot,iso}, w_\text{hot,sol}, w_\text{hot,phi})$ from the surface node inward. Its scale has the smooth softplus floor $\mathrm{scale\_cell} = \mathrm{lid\_mask\_width\_cells} \cdot \text{softmax}(|\nabla T|, 1.0\text{ K}, \mathrm{SCALE\_FLOOR\_WIDTH})$.

    The nodal lid mask uses the capped-width indicator $w_\text{interior,mask} = \max(w_\text{hot,iso,mask}, w_\text{hot,sol,mask}, w_\text{hot,phi})$. Its scale has the smooth softplus cap $\mathrm{scale\_mask} = \text{softmin}(\mathrm{scale\_cell},\, \mathrm{scale\_cap},\, \mathrm{SCALE\_MASK\_REL\_WIDTH} \cdot \mathrm{scale\_cap})$, where $\mathrm{scale\_cap} = \mathrm{lid\_mask\_width\_cells} \cdot \text{softmax}(\Delta T_\text{rh}, 1.0\text{ K}, \mathrm{SCALE\_FLOOR\_WIDTH})$. The mask is:

    $$
    w_\text{lid} = (1 - w_\text{interior,mask}) \cdot w_\text{active}
    $$

    Using $w_\text{interior,mask}$ makes steep single-cell surface thermal boundary layers saturate at $w_\text{lid}(\text{surface}) \to 1$.

    For $\phi \ge 0.012$, $w_{\text{sol},\phi} = 1$ and $w_\text{lid} = 0$ identically. Within the band $0.008 < \phi < 0.012$, $w_\text{lid} \le 1 - w_{\text{sol},\phi}$, which equals $0.5$ at $\phi = 0.010$ and $0$ at $\phi = 0.012$.

??? note "The `'fixed'` lid-base mode"
    The non-convective weight is $w_\text{nonconv} = 1 - w_{\text{melt,compact}}(f,\, f_\text{conv,min},\, 0.2 f_\text{conv,min})$, written also as $w_\text{nonconv} = 1 - w_\text{ramp}(f, f_\text{conv,min}, 0.2 f_\text{conv,min})$. It multiplies both $1 - w_\text{hot,iso}$ and $1 - w_\text{hot,iso,mask}$, through $1 - (1 - w_\text{hot,iso}) w_\text{nonconv}$ and $1 - (1 - w_\text{hot,iso,mask}) w_\text{nonconv}$.

    Convecting nodes ($f \ge 1.2 f_\text{conv,min}$) have $w_\text{nonconv} = 0$ and $w_\text{hot,iso} = 1$, so they cannot be lid ($w_\text{lid} = 0$), even when $T < T_\text{base}$. When no node is non-convecting ($w_\text{nonconv} = 0$ everywhere), $w_\text{interior} = 1$ at every node.

### Size of the Effect and Round-Off Spread

The effect of the rheology on the trajectory, and the spread of 5 perturbed starts.

??? note "Numbers for 2 configurations"
    The perturbed starts are $S_0(r)(1 + k \cdot 10^{-12})$ ($k \in \{-2, -1, 0, +1, +2\}$) over $100\text{ yr}$. The case is [Round-off perturbation spread with solid rheology](../Validation/numerics.md#round-off-perturbation-spread-with-solid-rheology).

    | Quantity | Active-solid variant of `cfg/abe_solid.toml` | Default `cfg/abe_solid.toml` |
    |:---|:---|:---|
    | $S_\text{init}$ | $2500\text{ J kg}^{-1}\text{ K}^{-1}$ | $2900\text{ J kg}^{-1}\text{ K}^{-1}$ |
    | $\kappa_{h,\text{floor}}$ | $0\text{ m}^2\text{ s}^{-1}$ | $10.0\text{ m}^2\text{ s}^{-1}$ |
    | $\eta_\text{solid}$ | $10^{18}\text{ Pa s}$ | $10^{21}\text{ Pa s}$ |
    | Spread $\max_r \Delta T_\text{on}(r)$ | $2.128 \times 10^{-5}\text{ K}$ (test bound $3.5 \times 10^{-5}\text{ K}$) | $5.946 \times 10^{-7}\text{ K}$ |
    | Spread $\max_r \Delta T_\text{off}(r)$ | $1.232 \times 10^{-5}\text{ K}$ | $1.089 \times 10^{-6}\text{ K}$ |

    On the active-solid variant, the rheology shifts the end-time surface temperature from $481.618\text{ K}$ to $482.079\text{ K}$. This is a $0.461\text{ K}$ change at the surface ($9.570 \times 10^{-4}$ relative). The largest change in the column is $7.9874\text{ K}$ ($2.205 \times 10^{-3}$ of the peak mantle temperature). The spread stays smooth across the radial grid.

    On the default configuration, solid rheology changes the final basic temperatures by $\max_r |T_\text{on} - T_\text{off}| = 7.534\text{ K}$ ($2.052 \times 10^{-3}$ relative). The floor switch alone gives $7.485\text{ K}$. The viscosity law added to it gives at most $0.199\text{ K}$ more. These are maxima over the radius, so they do not add. Section 6 explains the zero-anchored floor ramp.

### Driving-Stress Details

??? note "Geometric lid thickness"
    The diagnostic is `tau_d_lid = w_lid_present * w_active * 2 * eta_i * v_m_raw / d_geom`, where `eta_i` holds $\mu_i$. The velocity is $v_m = w_\text{lid,present} v_{m,\text{raw}}$. The smooth geometric length floor is $d_\text{geom} = \mathrm{softmax}(r_\text{top} - r_\text{lid,base},\, d_\text{geom,floor},\, 0.01\,d_\text{geom,floor})$, with $d_\text{geom,floor} = \mathrm{softmax}(\delta_\text{rh},\, \Delta r_\text{top},\, 0.01\,\Delta r_\text{top})$ and $\Delta r_\text{top} = \max(r_{-1} - r_{-2}, 10^{-6}\text{ m})$.

    `lid_state` returns both the raw lid thickness `d_lid_raw` and the weighted `d_lid`. It takes the yield indicator `w_y` from $\tau_{d,\text{lid,raw}} / \max(\tau_{y,\text{lid}}, 10^{-6}\text{ Pa})$.

??? note "Lid-presence weight"
    `compute_stagnant_lid_state` multiplies $d_\text{lid}$, $\mathrm{Ra}_\text{eff}$, $\mathrm{Ra}_\text{rh}$, $v_m$, $\tau_d$, $\tau_{d,\text{lid}}$, $\tau_\text{buoy}$, $\tau_d / \tau_\text{buoy}$, $w_y$ and `lid_regime` by the weight $w_\text{lid,present} = w_d w_T w_\mathrm{act\_gate} w_\mathrm{sum\_gate} w_\text{weight}$.

    $w_d$ and $w_\mathrm{sum\_gate}$ are $C^0$ with a one-sided slope at the domain edge $d_\text{lid,raw} \ge 0$ and $W_\text{sum} \ge 0$. $w_T$ uses the $C^1$ positive part $\mathrm{where}(\Delta T_\text{lid} > 0, \Delta T_\text{lid} \tanh(\Delta T_\text{lid} / (0.3\,\mathrm{NO\_LID\_DT\_WIDTH})), 0)$. So $w_\text{lid,present}$ vanishes identically when $d_\text{lid,raw} \le 0$, $T_i - T_\text{surf} \le 0$ or $W_\text{sum} \le 0$. This prevents numerical floor cancellation artifacts when $\Delta T \to 0$ and keeps all diagnostics continuous across lid onset.

    $T_i$, $\eta_i$, $\mu_i$ and $\theta$ serve as interior fallback diagnostics. All solver-facing outputs vary continuously across uniform convective flux fractions $f \in [0, 10^{-3}]$.

??? note "Non-smooth switches that stay on purpose"
    These switches serve boolean diagnostics and physical bounds. The 5-node local interior RMS $|F_\text{tot}|$ floor is not among them, because it is continuous in every $F_k$.

    - Flags: `is_convecting`, `has_lid`, the `is_convecting` condition on the diagnostic $v_i$, `dT_ep_nz` (`|T_1 - T_0| > 1e-12`), `is_mono_T` (`all(diff(T) <= 0)`), `if_lid` (`where(w_lid_present >= 1 - 1e-12)`).
    - Selectors and clips: the cap of $f$ at 1, `f_int_max = max(f * interior_node_mask)`, the `w_eff` clip to $[0, 1]$, the `1 - 1e-15` endpoint selectors on `w_eff` and `w_weight`, `f_int_scale`, `max(w_conv - w_conv_0, 0)`.
    - Lower bounds: `max(T_0, 1.0)`, `max(T_1, 1.0)`, `max(H_0, 1e-6)`, `max(H_1, 1e-6)`, `DeltaT_conv = max(T_i - T_surf, 1e-6)`, `max(theta, 1e-6)`, `max(Ra_rh_raw, 1e-30)`, the `COHESION_FLOOR` clamp on yield stress, the $10^{-30}$ denominators.
    - Geometry: the elementwise maxima in $w_\text{interior}$ and $w_\text{hot,sol}$, the temperature-gradient floor `dTdr_floor` with `max(dr_min, 1.0)` in $\delta_\text{rh}$, the mantle-thickness cap on $\delta_\text{rh}$, `d_geom`, the max in `d_lid_raw`, the $v_{m,\text{raw}}$ branches.

### Closure Details

??? note "Floors in `compute_effective_viscosity`"
    In `'local'` mode, the code calculates $\eta_y(r) = \max(\tau_y(r), 10^{-6}\text{ Pa}) / (2 \sqrt{\dot{\epsilon}_\text{local}(r)^2 + 10^{-200}})$. It applies a smooth $\log_{10}$ softplus cap of width `LOCAL_CAP_LOG10_WIDTH = 0.01` between $\log_{10}\eta_\text{diff}(r)$ and $\log_{10}\eta_y(r)$, that is $\log_{10}\big(\max(\tau_y(r), 10^{-6}\text{ Pa}) / (2 \sqrt{\dot{\epsilon}_\text{local}(r)^2 + 10^{-200}})\big)$. It returns $\eta_\text{diff}(r)$ when $\tau_y(r) = +\infty$ or $\dot{\epsilon}_\text{local}(r) \le 0$.

    In `'lid'` mode, $\log_{10}\eta_\text{eff}(r)$ blends smoothly across $\tau_{d,\text{lid}} = \tau_{y,\text{lid}}$ with width `YIELD_BLEND_RATIO_WIDTH = 0.025`.

    Zero cohesion and friction ($c = 0, \mu = 0$) at zero pressure would give zero viscosity. To prevent this, and to keep `jax.grad` finite when the yield stress is $+\infty$, the function floors a finite yield stress at `COHESION_FLOOR = 1.0e-6 Pa`. It substitutes a finite placeholder inside `where` before the ratio when the yield stress is $+\infty$, and it returns $\eta_\text{diff}$ in the outer `where`. It does not cap finite yield stresses or high diffusion viscosities.

??? note "Arguments of `compute_effective_viscosity`"
    `stress_closure_mode` sets the geometry of the strain rate, and the function rejects keyword arguments of the other mode.

    - `'lid'` requires `eta_i`, `tau_d` and `tau_y_lid`. It rejects the local-mode arguments `tau_y`, `tau_y_profile`, `strain_rate`, `mixing_length` and `unyielded_velocity`.
    - `'local'` rejects the lid-mode arguments `eta_i`, `tau_d`, `tau_y_lid` and `w_lid`.
    - Negative values of `tau_y_lid`, `tau_y` or `tau_y_profile`, including $-\infty$, raise `ValueError` on the eager NumPy path and return `NaN` on the JAX path. The value $0$ is floored at `COHESION_FLOOR = 1.0e-6 Pa`, and $+\infty$ is accepted.

## Where This Is Checked

[Validation: solid-state convection](../Validation/solid_state_convection.md) has the activation law, the top slope, the 0-D stagnant-lid comparison and the internal-heating exponents. [Validation: numerics](../Validation/numerics.md) has the solver wiring cases. [Tutorial: solid-state convection](../Tutorials/solid_state_convection.md) shows how to use these kernels.

[^cite-abe1997]: Y. Abe, *[Thermal and chemical evolution of the terrestrial magma ocean](https://doi.org/10.1016/S0031-9201(96)03229-3)*, Physics of the Earth and Planetary Interiors, 100, 27–39, 1997. [SciX](https://scixplorer.org/abs/1997PEPI..100...27A/abstract).
[^cite-armann2012]: M. Armann, P. J. Tackley, *[Simulating the thermochemical magmatic and tectonic evolution of Venus's mantle and lithosphere: Two-dimensional models](https://doi.org/10.1029/2012JE004231)*, Journal of Geophysical Research: Planets, 117, E12003, 2012. [SciX](https://scixplorer.org/abs/2012JGRE..11712003A/abstract).
[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49–62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
[^cite-byerlee1978]: J. Byerlee, *[Friction of rocks](https://doi.org/10.1007/BF00876528)*, Pure and Applied Geophysics, 116, 615–626, 1978. [SciX](https://scixplorer.org/abs/1978PApGe.116..615B/abstract).
[^cite-dumoulin1999]: C. Dumoulin, M.-P. Doin, L. Fleitout, *[Heat transport in stagnant lid convection with temperature- and pressure-dependent Newtonian or non-Newtonian rheology](https://doi.org/10.1029/1999JB900110)*, Journal of Geophysical Research, 104, 12759–12777, 1999. [SciX](https://scixplorer.org/abs/1999JGR...10412759D/abstract).
[^cite-foley2009]: B. J. Foley, T. W. Becker, *[Generation of plate-like behavior and mantle heterogeneity from a spherical, viscoplastic convection model](https://doi.org/10.1029/2009GC002378)*, Geochemistry, Geophysics, Geosystems, 10, Q08001, 2009. [SciX](https://scixplorer.org/abs/2009GGG....10.8001F/abstract).
[^cite-foley2014]: B. J. Foley, D. Bercovici, *[Scaling laws for convection with temperature-dependent viscosity and grain-damage](https://doi.org/10.1093/gji/ggu275)*, Geophysical Journal International, 199, 580–603, 2014. [SciX](https://scixplorer.org/abs/2014GeoJI.199..580F/abstract).
[^cite-karato1993]: S.-I. Karato, P. Wu, *[Rheology of the Upper Mantle: A Synthesis](https://doi.org/10.1126/science.260.5109.771)*, Science, 260, 771–778, 1993. [SciX](https://scixplorer.org/abs/1993Sci...260..771K/abstract).
[^cite-korenaga2009]: J. Korenaga, *[Scaling of stagnant-lid convection with Arrhenius rheology and the effects of mantle melting](https://doi.org/10.1111/j.1365-246X.2009.04272.x)*, Geophysical Journal International, 179, 154–170, 2009. [SciX](https://scixplorer.org/abs/2009GeoJI.179..154K/abstract).
[^cite-moresi1998]: L. Moresi, V. Solomatov, *[Mantle convection with a brittle lithosphere: thoughts on the global tectonic styles of the Earth and Venus](https://doi.org/10.1046/j.1365-246X.1998.00521.x)*, Geophysical Journal International, 133, 669–682, 1998. [SciX](https://scixplorer.org/abs/1998GeoJI.133..669M/abstract).
[^cite-solomatov1995]: V. S. Solomatov, *[Scaling of temperature- and stress-dependent viscosity convection](https://doi.org/10.1063/1.868624)*, Physics of Fluids, 7, 266–274, 1995. [SciX](https://scixplorer.org/abs/1995PhFl....7..266S/abstract).
[^cite-solomatov2000]: V. S. Solomatov, L.-N. Moresi, *[Scaling of time-dependent stagnant lid convection: Application to small-scale convection on Earth and other terrestrial planets](https://doi.org/10.1029/2000JB900197)*, Journal of Geophysical Research, 105, 21795–21817, 2000. [SciX](https://scixplorer.org/abs/2000JGR...10521795S/abstract).
[^cite-tackley2000]: P. J. Tackley, *[Self-consistent generation of tectonic plates in time-dependent, three-dimensional mantle convection simulations](https://doi.org/10.1029/2000GC000036)*, Geochemistry, Geophysics, Geosystems, 1, 1021, 2000. [SciX](https://scixplorer.org/abs/2000GGG.....1.1021T/abstract).
[^cite-tackley2013]: P. J. Tackley, M. Ammann, J. P. Brodholt, D. P. Dobson, D. Valencia, *[Mantle dynamics in super-Earths: Post-perovskite rheology and self-regulation of viscosity](https://doi.org/10.1016/j.icarus.2013.03.013)*, Icarus, 225, 50–61, 2013. [SciX](https://scixplorer.org/abs/2013Icar..225...50T/abstract).
[^cite-thiriet2019]: M. Thiriet, D. Breuer, C. Michaut, A.-C. Plesa, *[Scaling laws of convection for cooling planets in a stagnant lid regime](https://doi.org/10.1016/j.pepi.2018.11.003)*, Physics of the Earth and Planetary Interiors, 286, 138–153, 2019. [SciX](https://scixplorer.org/abs/2019PEPI..286..138T/abstract).
[^cite-yamazaki2001]: D. Yamazaki, S.-I. Karato, *[Some mineral physics constraints on the rheology and geothermal structure of Earth's lower mantle](https://doi.org/10.2138/am-2001-0401)*, American Mineralogist, 86, 385–391, 2001. [SciX](https://scixplorer.org/abs/2001AmMin..86..385Y/abstract).
