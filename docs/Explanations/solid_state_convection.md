# Solid-State Convection and Lithospheric Rheology

Aragog models thermal evolution from a molten magma ocean through a partially molten mush to a solidified rocky mantle. In the solid regime, mantle viscosity varies by many orders of magnitude with temperature and pressure, producing a cold, high-viscosity conductive boundary layer (stagnant lid) unless convective driving stresses exceed the lithospheric yield strength and mobilise the surface boundary layer. The modules `aragog.rheology` and `aragog.rheology_lid` provide the standalone Arrhenius creep, plastic yield, stagnant-lid diagnostic, and effective-viscosity kernels documented below.

**Status:** When `rheology.enabled` is `True`, the time-stepping solver evaluates the kernels in `aragog.rheology` and `aragog.rheology_lid` in a two-pass update (`EntropyState.update` on the NumPy path and `aragog.jax.phase.compute_fluxes` on the JAX path); when `rheology.enabled` is `False` (the default), the rheology kernels are bypassed and the solver reproduces the baseline liquid-parameterised trajectory. Remaining switches, positivity floors, caps, and derivative kinks in these standalone kernels include: in `aragog.rheology`, $\max(\theta, 0)$ in `compute_mlt_top_slope`, $l_v = \min(s_\text{bot}(r - r_\text{in}), s_\text{top}(\theta)(r_\text{out} - r))$ and the $l > 0$ branch in `viscous_mixing_length_factor`, the lower exponent clip $-700$ and smooth $\log_{10}$ softplus viscosity ceiling `viscosity_max_log10` (width `VISCOSITY_CEILING_LOG10_WIDTH = 0.05`) in `compute_arrhenius_viscosity`, $\max(0, \min(c + \mu P, \tau_\text{max}))$ in `compute_yield_stress`, and $|v| / (2 \max(l_\text{mix}, \epsilon))$ in `compute_strain_rate_local`; in `aragog.rheology_lid`, the Boltzmann-weighted smooth maximum (`SMOOTH_MAX_REL_WIDTH = 0.02`) and $\max(W_\text{sum}, 10^{-12})$ denominator inside `_boundary_weights`, the smooth softplus 5-node local interior RMS $|F_\text{tot}|$ floor (`F_TOT_FLOOR_FRAC = 1e-3`, smooth floor at $10^{-12}$, continuous in every $F_k$) and $[0, 1]$ clip of $f_k$, the boolean `has_lid` flag and the smooth product weight $w_\text{lid,present} = w_d w_T w_\text{act\_gate} w_\text{sum\_gate} w_\text{weight}$ (`NO_LID_THICKNESS_WIDTH = 0.25`, `NO_LID_DT_WIDTH = 2.5e-4`, `CONV_ACTIVE_GATE_WIDTH = 0.0025`, `CONV_SUM_GATE_WIDTH = 2.5e-7`), the compactly supported $C^2$ quintic melt-fraction ramp $(t^3)(10 - 15 t + 6 t^2)$ in `_melt_ramp` and $w_\text{solid,surf}$, the smooth softplus floors and caps in `scale` and `scale_mask` (`SCALE_FLOOR_WIDTH = 0.005`, `SCALE_MASK_REL_WIDTH = 0.005`), the radial temperature-gradient floor $\max(|dT/dr|, 1 / \max(\Delta r_\text{min}, 1\text{ m}))$ and mantle-thickness cap $\min(\delta_\text{rh}, r_\text{top} - r_\text{bot})$, and in `compute_effective_viscosity`, the `COHESION_FLOOR` ($10^{-6}\text{ Pa}$) floor on yield stress, the smooth $\tanh$ yield transition in `'lid'` mode (`YIELD_BLEND_RATIO_WIDTH = 0.025`), and the regularized strain-rate floor (`LOCAL_STRAIN_RATE_FLOOR = 1e-100`) and smooth $\log_{10}$ softplus cap (`LOCAL_CAP_LOG10_WIDTH = 0.01`) in `'local'` mode.

## 1. Homologous-Temperature Diffusion Creep

In `aragog.rheology`, solid-state diffusion creep viscosity is evaluated using the homologous-temperature Arrhenius formulation (Yamazaki and Karato, 2001, eqs. 4-7; Abe, 1997, Table 1):

$$
\eta_\text{diff}(T, P) = \eta_\text{solid} \cdot f_\text{water} \cdot \exp\left( \frac{H(P)}{R T} - \frac{H(0)}{R T_\text{ref}} \right) = \eta_\text{solid} \cdot f_\text{water} \cdot \exp\left[ g_\text{hom} \left( \frac{T_\text{sol}(P)}{T} - \frac{T_\text{sol}(0)}{T_\text{ref}} \right) \right]
$$

where:

- $\eta_\text{solid}$ is the reference solid viscosity (`viscosity_solid`, default $10^{21}\text{ Pa s}$), defined such that $\eta_\text{solid} f_\text{water}$ equals $\eta_\text{diff}$ at $(T_\text{ref} = 1600\text{ K}, P = 0)$ for the linear solidus; when a solidus table is supplied, $\eta_\text{solid}$ is referenced to `params.t_sol_0` ($T_{\text{sol},0}$), so the table $\eta_\text{diff}(T, 0)$ scales relative to the linear-solidus $\eta_\text{diff}(T, 0)$ at the same temperature $T$ by the ratio $\exp\big(g_\text{hom}(T_\text{sol}(0) - T_{\text{sol},0}) / T\big)$ ($1.8459\times$, or $\eta_\text{diff}(1600\text{ K}, 0) = 1.8459 \times 10^{21}\text{ Pa s}$ at $\eta_\text{solid} = 10^{21}\text{ Pa s}$, for the CI test table where $T_{\text{sol,table}}(0) = 1439.230032\text{ K}$ against $T_{\text{sol},0} = 1400\text{ K}$, and $2.752\times$, or $2.75\times$, at $T = 1600\text{ K}$ for $1464.8\text{ K}$ against $1400\text{ K}$).
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

When an explicit solidus profile $T_\text{sol}(P)$ is supplied to `compute_arrhenius_viscosity` or `compute_stagnant_lid_state` (such as the PALEOS $\mathrm{MgSiO_3}$ phase-boundary solidus in tabulated `EntropyEOS` runs), `compute_arrhenius_viscosity` requires the zero-pressure reference solidus $T_{\text{sol},0}$ (`t_sol_0`) and `compute_stagnant_lid_state` reads `params.t_sol_0` directly; because the reference term is evaluated at `params.t_sol_0` rather than at the table's $P = 0$ value $T_\text{sol}(0)$, $\eta_\text{solid} f_\text{water}$ is referenced to `params.t_sol_0`, and the table $\eta_\text{diff}(T, 0)$ scales relative to the linear-solidus $\eta_\text{diff}(T, 0)$ at the same temperature $T$ by the ratio $\exp\big(g_\text{hom}(T_\text{sol}(0) - T_{\text{sol},0}) / T\big)$ ($1.8459\times$, giving $\eta_\text{diff}(1600\text{ K}, 0) = 1.8459 \times 10^{21}\text{ Pa s}$ at $\eta_\text{solid} = 10^{21}\text{ Pa s}$, for the CI test table where $T_{\text{sol,table}}(0) = 1439.230032\text{ K}$ against $T_{\text{sol},0} = 1400\text{ K}$, and $2.752\times$, or $2.75\times$, at $T = 1600\text{ K}$ and $g_\text{hom} = 25$ for the $1464.8\text{ K}$ PALEOS value against $1400\text{ K}$). When `solidus_temperature` is `None` (such as in table-free constant-property benchmarks), `compute_arrhenius_enthalpy`, `compute_arrhenius_viscosity`, and `compute_stagnant_lid_state` evaluate the linear solidus fallback:

$$
T_\text{sol}(P) = T_{\text{sol},0} + \frac{dT_\text{sol}}{dP} P
$$

with reference zero-pressure solidus $T_{\text{sol},0} = 1400.0\text{ K}$ (`t_sol_0`) and linear solidus slope $dT_\text{sol} / dP = 2.4 \times 10^{-8}\text{ K Pa}^{-1}$ (`dt_sol_dp`, corresponding to $24\text{ K GPa}^{-1}$). At $g_\text{hom} = 25.0$ and $T_{\text{sol},0} = 1400.0\text{ K}$, the zero-pressure activation enthalpy is $H(0) = 291\text{ kJ mol}^{-1}$, close to the standard olivine diffusion-creep activation energy $E_a = 300\text{ kJ mol}^{-1}$ (Karato and Wu, 1993). Coupled and tabulated PALEOS runs evaluate the phase-boundary solidus $T_\text{sol}(P) = 0.8\,T_\text{liq}(P)$ from the PALEOS $\mathrm{MgSiO_3}$ melting curve ($1464.8\text{ K}$ at $0\text{ GPa}$, $1563.0\text{ K}$ at $1\text{ GPa}$, and $4754.8\text{ K}$ at $135\text{ GPa}$, whole-mantle secant slope $24.4\text{ K GPa}^{-1}$, high-pressure slope $15.3\text{ K GPa}^{-1}$ between $25$ and $135\text{ GPa}$). The linear defaults $T_{\text{sol},0} = 1400.0\text{ K}$ and $dT_\text{sol}/dP = 24\text{ K GPa}^{-1}$ provide a single-slope benchmark form; because the physical PALEOS solidus is concave downward in pressure, the linear form is $4.4\%$ low at $0\text{ GPa}$ ($1400.0\text{ K}$ versus $1464.8\text{ K}$), $34.8\%$ low at $25\text{ GPa}$ ($2000.0\text{ K}$ versus $3067.0\text{ K}$), and $2.4\%$ low at $135\text{ GPa}$ ($4640.0\text{ K}$ versus $4754.8\text{ K}$), reaching a maximum absolute deviation of $1108.1\text{ K}$ ($-1108.1\text{ K}$ at $36.7\text{ GPa}$, maximum relative deviation $-35.0\%$ at $20.5\text{ GPa}$ ($-35.04\%$ at $20.495\text{ GPa}$), RMS deviation $817.6\text{ K}$, and no crossings over $0\text{ to }135\text{ GPa}$; even the minimax linear line $1979.7\text{ K} + 24.37\text{ K GPa}^{-1} P$ deviates by up to $514.9\text{ K}$ with RMS $350.6\text{ K}$ and crossings at $6.1\text{ GPa}$ and $98.5\text{ GPa}$ ($6.15\text{ GPa}$ and $98.54\text{ GPa}$), and the ordinary least-squares line $2489.3\text{ K} + 19.10\text{ K GPa}^{-1} P$ deviates by up to $1024.5\text{ K}$ at $0\text{ GPa}$ with RMS $239.0\text{ K}$ and crossings at $18.8\text{ GPa}$ and $99.9\text{ GPa}$ ($18.82\text{ GPa}$ and $99.89\text{ GPa}$)), and serves table-free constant-property benchmarks only. To prevent float64 overflow in cold conductive surface layers, `compute_arrhenius_viscosity` applies a smooth $\log_{10}$ softplus ceiling at `viscosity_max_log10` (default $40.0$, width `VISCOSITY_CEILING_LOG10_WIDTH = 0.05`) after applying `water_prefactor`; non-positive temperatures ($T \le 0$) raise `ValueError` on the eager NumPy path and return `NaN` on the JAX path.

### Enthalpy Values on the PALEOS Solidus

The table below reports the solidus temperature $T_\text{sol}(P) = 0.8\,T_\text{liq}(P)$ from the PALEOS $\mathrm{MgSiO_3}$ melting curve (`mushy_zone_factor = 0.8`, produced by `tools/verification/compute_solidus_reference_values.py`) and the resulting homologous activation enthalpy $H(P) = g_\text{hom} R T_\text{sol}(P)$ over terrestrial mantle pressures:

| Pressure $P$ (GPa) | $T_\text{sol}(P)$ (K) | $H(P)$ at $g_\text{hom} = 14$ (kJ/mol) | $H(P)$ at $g_\text{hom} = 25$ (kJ/mol) | $\eta(0.9 T_\text{sol}) / \eta(T_\text{sol})$ at $g_\text{hom} = 25$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 1464.8 | 170.5 | 304.5 | 16.083 |
| 25 | 3067.0 | 357.0 | 637.5 | 16.083 |
| 50 | 3672.7 | 427.5 | 763.4 | 16.083 |
| 100 | 4397.9 | 511.9 | 914.2 | 16.083 |
| 135 | 4754.8 | 553.5 | 988.3 | 16.083 |

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

`compute_stagnant_lid_state` identifies convective activity and the convecting interior state from the convective flux fraction $f_k = \max(F_{\text{conv}, k}, 0) / |F_{\text{tot}, k}|_\text{safe}$, where $|F_{\text{tot}, k}|_\text{safe}$ applies a smooth softplus floor against $F_{\text{tot,floor}, k} = \mathrm{softmax}(10^{-3} F_{\text{loc,rms}, k}, 10^{-12})$ using the 5-node local interior RMS $F_{\text{loc,rms}, k}$ of $F_\text{tot}$ (with boundary nodes $k = 0$ and $k = N - 1$ replaced by their adjacent interior nodes $F_{\text{tot}, 1}$ and $F_{\text{tot}, N - 2}$ before 2-node edge replication; this floor is continuous in every $F_k$ and differs from a scalar $10^{-12}$ floor only at nodes where $F_{\text{conv}, k} > 0$ and $|F_{\text{tot}, k}| < 0.0014 F_{\text{loc,rms}, k}$ when $10^{-3} F_\text{loc,rms} \gg 10^{-12}\text{ W/m}^2$). Convective activity across the column is measured relative to the interior flux threshold $f_\text{conv,min} \in [0.05, 0.5]$ (`interior_flux_fraction`, default $0.05$) with a fixed transition width of $0.01$:

$$
w_{\text{conv}, k} = \frac{1}{2} \left( 1 + \tanh\left( \frac{f_k - f_\text{conv,min}}{0.01} \right) \right)
$$

so that a zero-flux conductive node ($f_k = 0$) carries a weight of at most $w_\text{conv}(f = 0) = \frac{1}{2}(1 - \tanh(5)) \approx 4.5 \times 10^{-5}$. From the interior peak convective flux fraction $f_\text{int,max} = \max_{1 < k < N} f_k$ and the baseline-subtracted interior convective indicator $w_{\text{conv,int}, k} = \max(w_{\text{conv}, k} - w_\text{conv}(f = 0), 0) I_{1 < k < N}$, `compute_stagnant_lid_state` evaluates the mesh-independent convective activity indicator $w_\text{active} = \frac{1}{2}(1 + \tanh((f_\text{int,max} - f_\text{conv,min}) / 0.01))$ and the shallowest-crossing boundary sum $W_\text{sum} = \sum_k w_{\text{conv,int}, k} \prod_{j > k} (1 - w_{\text{conv,int}, j})$ scanned from the outermost surface node $k = N$ inward (with mesh-independent transition weight $w_\text{weight} = \tanh(\text{smax}(w_{\text{conv,int}}) / 10^{-6})$ and `is_convecting` requiring $W_\text{sum} > 10^{-6}$ and $w_\text{active} > 0.05$).

To evaluate the interior temperature $T_i$, pressure $P_{T_i}$, radius $r_i$, and interior viscosity $\mu_i$ where convection dominates relative to the interior peak $f_\text{int,max} = \max_{1 < k < N} f_k$ (with scale $f_\text{int,scale} = \max(f_\text{int,max}, 10^{-12})$ and threshold $f_{T_i} = 0.9 f_\text{int,max}$), `compute_stagnant_lid_state` forms the interior selector on non-boundary nodes ($1 < k < N$). For a conductive error-function boundary-layer profile $T(z) = 300 + 1300\,\mathrm{erf}(z / \delta)$ with convective flux-fraction profile $f(z) = 1 - \exp(-(z / \delta)^2)$, a $1300\text{ K}$ surface-to-interior temperature contrast, and $\Delta T_\text{rh} = 70.95\text{ K}$ ($H = 300\text{ kJ mol}^{-1}$ at $1600\text{ K}$), the onset threshold $f = 0.05$ ($z / \delta = 0.2265$, $T = 626.6\text{ K}$) sits $13.7\,\Delta T_\text{rh}$ below the interior, whereas $f = 0.9$ ($z / \delta = 1.517$, $T = 1558.6\text{ K}$) sits $0.584\,\Delta T_\text{rh}$ below the interior:

$$
w_{T_i, k} = \frac{1}{2} \left( 1 + \tanh\left( \frac{f_k - f_{T_i}}{0.02 f_\text{int,scale}} \right) \right) I_{1 < k < N}
$$

The shallowest convection-dominated node from the surface is isolated by the surface-down cumulative product indicator $W_k = w_{T_i, k} \prod_{j > k} (1 - w_{T_i, j})$ over shallower nodes $j > k$, normalised to $\tilde{W}_{T_i, k}$, and blended via $\tilde{W}_k = w_\text{weight} \tilde{W}_{T_i, k} + (1 - w_\text{weight}) \tilde{W}_{\text{conv}, k}$ (where $\tilde{W}_{\text{conv}, k}$ is the shallowest-crossing weight of $w_{\text{conv,int}, k}$ with fallback to the deepest interior node $k = 2$). At the default $f_\text{conv,min} = 0.05$ (`interior_flux_fraction = 0.05`), for weak convective fluxes below the lid onset ($f_\text{int,max} < 0.05 + 0.01\,\mathrm{arctanh}(-0.9) = 0.035278$) where $\text{smax}(w_{\text{conv,int}}) \gg 10^{-6}$ (for example $f_\text{int,max} \ge 10^{-3}$, where $w_\text{weight} \ge 1 - 4 \times 10^{-9}$), $w_\text{weight} \approx 1$ and $T_i$ still selects the top-of-convection node near $0.9 f_\text{int,max}$; over the transition band near $f_\text{int,max} \approx 6 \times 10^{-5}$ ($w_\text{weight} = 0.5$ at $f_\text{int,max} = 6.01 \times 10^{-5}$, and $w_\text{weight} = 0.7245$ at $f_\text{int,max} = 10^{-4}$ where $\text{smax}(w_{\text{conv,int}}) = 9.17 \times 10^{-7}$) and as $f_\text{int,max} \to 0$ ($\text{smax}(w_{\text{conv,int}}) \ll 10^{-6}$), $w_\text{weight} \to 0$ and $T_i$ relaxes smoothly to the interior fallback node $T_2$ (these $f_\text{int,max}$ transition values hold at $f_\text{conv,min} = 0.05$ only, because $w_\text{conv}(f) - w_\text{conv}(0)$ scales with $\mathrm{sech}^2(f_\text{conv,min} / 0.01)$; for example at $f_\text{conv,min} = 0.20$, $w_\text{conv}(f) - w_\text{conv}(0)$ underflows to $0.0$ in IEEE 754 float64 for $f \le 0.0101$ and $w_\text{weight}$ reaches $0.5$ near $f_\text{int,max} \approx 0.128$):

$$
T_i = \sum_{k=1}^N \tilde{W}_k T_k, \quad P_{T_i} = \sum_{k=1}^N \tilde{W}_k P_k
$$

Because the selector places $T_i$ at the node under the lid, its accuracy is bounded by the radial mesh resolution across the thermal boundary layer: when the lid is one cell thick (for example on a 100-node post-magma-ocean profile where $T_i = 1423.0\text{ K}$ against $1788.6\text{ K}$ one node deeper), $T_i$ lies in the rheological sublayer.

When an optional mixture viscosity profile `viscosity_mixture` ($\eta_\text{mix}$) is supplied to `compute_stagnant_lid_state`, the convecting interior viscosity $\mu_i$ is evaluated using the smooth activity weight $w_\text{weight}$ in logarithmic space:

$$
\log_{10} \mu_i = w_\text{weight} \sum_{k=1}^N \tilde{W}_k \log_{10} \eta_{\text{mix}, k} + (1 - w_\text{weight}) \log_{10} \eta_\text{diff}(T_i, P_{T_i})
$$

When `viscosity_mixture` is `None`, $\mu_i$ equals the solid Arrhenius interior viscosity $\eta_i = \eta_\text{diff}(T_i, P_{T_i})$. When $f_\text{int,max}$ lies more than $0.01\,\mathrm{arctanh}(0.9) \approx 0.0147$ below $f_\text{conv,min}$ (that is, $f_\text{int,max} \le 0.035278$ at $f_\text{conv,min} = 0.05$), $w_\text{active} \le 0.05$, `is_convecting` and `has_lid` are `False`, no lid is delimited, and yielding remains inactive across any mesh resolution $N$.

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
2. The solidus crossing, evaluated from the compactly supported $C^2$ quintic melt-fraction ramp $w_{\text{sol},\phi} = t^3(10 - 15 t + 6 t^2)$ with $t = \mathrm{clip}((\phi - 0.008) / 0.004, 0, 1)$ centered at $\phi = 0.01$ (`SOLIDUS_MELT_FRACTION_THRESHOLD`) with half-width $0.002$ (`SOLIDUS_MELT_FRACTION_WIDTH`) and, when a solidus profile $T_\text{sol}$ is supplied, the elementwise maximum with the thermal solidus indicator $T = T_\text{sol}$ (because $w_{\text{sol},\phi} = 1$ identically for $\phi \ge 0.012$, any column with $\phi \ge 0.012$ at every node places $r_\text{lid,base}$ strictly at $r[-1]$ with $d_\text{lid} = 0$ and `has_lid = False` for every mesh resolution $N$; see Section 8 item 7).
3. The rheological crystallisation front where $\phi = \phi_\text{rheo} = 0.4$.

When the entire column lies below the lid-base criterion so that the interior boundary indicator sum $W_\text{base,sum} \to 0$, `r_lid_base` anchors at the inner boundary radius `r[0]` (the core-mantle boundary), giving $d_\text{lid} = r_\text{outer} - r_\text{inner}$.

The thickness of the rheological sublayer that accommodates deformation at the lid base is:

$$
\delta_\text{rh} = \frac{\Delta T_\text{rh}}{|dT/dr|_\text{lid\_base}}
$$

where $|dT/dr|_\text{lid\_base}$ is the $\tilde{W}_{\text{base}, k}$-weighted radial temperature gradient evaluated from nodal central differences (`xp.gradient(T, r)`). To prevent division by zero in isothermal or near-adiabatic profiles, the nodal temperature gradient is floored at a minimum positive value corresponding to a $1\text{ K}$ difference over the minimum grid spacing.

### Smooth Lid Mask and Resolution

Because $|dT/dr|_\text{lid\_base}$ is computed by a nodal central difference (`xp.gradient(T, r)`), $\delta_\text{rh}$ depends on the radial mesh spacing around the lid base. In addition, the `lid_state` dictionary records `lid_cell_count` ($\sum_k w_{\text{lid}, k}$) over the physical lid thickness $d_\text{lid}$ to track numerical resolution within the lid.

The lid-base locator $\tilde{W}_{\text{base}, k}$ scans the cell-width interior indicator $w_\text{interior} = \max(w_\text{hot,iso}, w_\text{hot,sol}, w_\text{hot,phi})$ (with smooth softplus floor $\text{scale\_cell} = \text{lid\_mask\_width\_cells} \cdot \text{softmax}(|\nabla T|, 1.0\text{ K}, \text{SCALE\_FLOOR\_WIDTH})$) from the surface node inward, whereas the nodal lid mask uses the capped-width indicator $w_\text{interior,mask} = \max(w_\text{hot,iso,mask}, w_\text{hot,sol,mask}, w_\text{hot,phi})$ evaluated with the smooth softplus cap $\text{scale\_mask} = \text{softmin}(\text{scale\_cell},\, \text{scale\_cap},\, \text{SCALE\_MASK\_REL\_WIDTH} \cdot \text{scale\_cap})$ where $\text{scale\_cap} = \text{lid\_mask\_width\_cells} \cdot \text{softmax}(\Delta T_\text{rh}, 1.0\text{ K}, \text{SCALE\_FLOOR\_WIDTH})$:

$$
w_\text{lid} = (1 - w_\text{interior,mask}) \cdot w_\text{active}
$$

Using $w_\text{interior,mask}$ in $w_\text{lid}$ ensures that steep single-cell surface thermal boundary layers saturate at $w_\text{lid}(\text{surface}) \to 1$. Inside the cold solid lid, $w_\text{lid} \to 1$. In the warm or partially molten interior, $w_\text{lid} \to 0$ (for $\phi \ge 0.012$, $w_{\text{sol},\phi} = 1$ and $w_\text{lid} = 0$ identically; within the compact transition band $0.008 < \phi < 0.012$, $w_\text{lid} \le 1 - w_{\text{sol},\phi}$, which equals $0.5$ at $\phi = 0.010$ and $0$ at $\phi = 0.012$). Under round-off perturbations $S_0(r)(1 + k \cdot 10^{-12})$ ($k \in \{-2, -1, 0, +1, +2\}$) over $100\text{ yr}$ on the active-solid variant of `cfg/abe_solid.toml` ($S_\text{init} = 2500\text{ J kg}^{-1}\text{ K}^{-1}$, $\kappa_{h,\text{floor}} = 0\text{ m}^2\text{ s}^{-1}$, $\eta_\text{solid} = 10^{18}\text{ Pa s}$, where the rheology shifts the end-time surface temperature from $481.620\text{ K}$ to $482.081\text{ K}$, a $0.461\text{ K}$ or $2.205 \times 10^{-3}$ relative change across the column), the trajectory spread remains smooth across the radial grid with $\max_r \Delta T_\text{on}(r) = 1.751 \times 10^{-5}\text{ K}$ against $\max_r \Delta T_\text{off}(r) = 3.439 \times 10^{-6}\text{ K}$ ($\le 0.05\text{ K}$), whereas on the default `cfg/abe_solid.toml` ($S_\text{init} = 2900\text{ J kg}^{-1}\text{ K}^{-1}$, $\kappa_{h,\text{floor}} = 10.0\text{ m}^2\text{ s}^{-1}$, $\eta_\text{solid} = 10^{21}\text{ Pa s}$) the floor $\kappa_{h,\text{floor}} f(0) = 0.04805\text{ m}^2\text{ s}^{-1}$ bounds solid convection and limits on-vs-off basic temperature deviations to $\max_r |T_\text{on} - T_\text{off}| = 0.194\text{ K}$ ($5.280 \times 10^{-5}$ relative), with perturbation spreads $\max_r \Delta T_\text{on}(r) = 5.557 \times 10^{-7}\text{ K}$ and $\max_r \Delta T_\text{off}(r) = 1.035 \times 10^{-6}\text{ K}$.

## 4. Convective Driving Stress

Convective flow beneath the stagnant lid exerts a driving shear stress on the base of the lid. `compute_stagnant_lid_state` implements the stagnant-lid convective shear stress scaling of Foley and Bercovici (2014, eqs. 26 and 28, pp. 586-588, doi:10.1093/gji/ggu275):

$$
\tau_d = \frac{2 \mu_i v_m}{d} w_\text{active}, \quad \tau_{d,\text{lid}} = \frac{2 \eta_i v_m}{\max(r_\text{top} - r_\text{lid,base}, 10^{-6}\text{ m})} w_\text{active}
$$

where $d = r_\text{outer} - r_\text{inner}$ is the mantle thickness, $\mu_i$ (`eta_i` in `lid_state`) is the convecting interior dynamic viscosity, $v_m = w_\text{lid,present} v_{m,\text{raw}}$ is the smoothly gated interior convective velocity, and $\tau_{d,\text{lid}}$ is the geometric-lid-thickness-scaled stress diagnostic (`tau_d_lid = w_lid_present * w_active * 2 * eta_i * v_m_raw / max(r_top - r_lid_base, 1e-6 m)`). The factor of 2 arises from horizontal shear between the lid base ($u = v_m$) and mid-depth ($u = 0$ at $z = d/2$), as derived by Foley and Bercovici (2014, pp. 584, 588). Because the solver-facing driving stress $\tau_d$ divides by the full mantle thickness $d$, it remains bounded as $d_\text{lid} \to 1\text{ m}^+$, whereas `tau_d_lid` divides by $d_\text{geom} = \max(r_\text{top} - r_\text{lid,base}, 10^{-6}\text{ m})$ and `has_lid` turns on at $d_\text{lid,raw} > 1\text{ m}$ (`NO_LID_THICKNESS_THRESHOLD = 1.0`): the $10^{-6}\text{ m}$ floor is therefore inactive whenever `has_lid` is `True`, vanishes in non-lid columns where $w_d = 0$, and $\tau_{d,\text{lid}} / \tau_d = d / d_\text{geom}$ scales as $1 / d_\text{lid}$ down to $d_\text{geom} = 1\text{ m}$.

The interior convective velocity $v_m$ follows the boundary-layer scaling for bottom-heated stagnant-lid convection (Dumoulin et al., 1999, p. 12760, sec. 2; Foley and Bercovici, 2014, pp. 586-587, eq. 26):

$$
v_m = \frac{\kappa_i}{d} C_4 \left( \frac{\mathrm{Ra}_\text{eff} a_\text{rh}}{\theta} \right)^{2/3}
$$

where $C_4 = 0.125$ is the empirical prefactor calibrated by Foley and Bercovici (2014, sec. 4.2, p. 589), $a_\text{rh} = 1.3$ is the rheological sublayer constant fitted jointly with $C_4$, and $\rho_i, g_i, \alpha_i, \kappa_i$ are evaluated as $\tilde{W}_k$-weighted means at the top of the convecting layer. The effective Rayleigh number is defined at the convecting interior viscosity $\mu_i$:

$$
\mathrm{Ra}_\text{eff} = \frac{\rho_i g_i \alpha_i (T_i - T_\text{surf}) d^3}{\kappa_i \mu_i}
$$

Because both $\mathrm{Ra}_\text{eff}$ and $\theta$ scale linearly with the temperature drop $\Delta T = T_i - T_\text{surf}$, the ratio $\mathrm{Ra}_\text{eff} a_\text{rh} / \theta$ is proportional to $a_\text{rh} \Delta T_\text{rh}$ where $\Delta T_\text{rh} = R T_i^2 / H(P_{T_i})$. When convection is inactive or reaches the surface such that no physical conductive lid forms ($d_\text{lid,raw} \le 1\text{ m}$ or $T_i - T_\text{surf} \le 10^{-3}\text{ K}$), `compute_stagnant_lid_state` smoothly gates $d_\text{lid}$, $\mathrm{Ra}_\text{eff}$, $\mathrm{Ra}_\text{rh}$, $v_m$, $\tau_d$, $\tau_{d,\text{lid}}$, $\tau_\text{buoy}$, $\tau_d / \tau_\text{buoy}$, $w_y$, and `lid_regime` by $w_\text{lid,present} = w_d w_T w_\text{act,gate} w_\text{sum,gate} w_\text{weight}$ (where $w_d$ and $w_\text{sum,gate}$ are $C^0$ with a one-sided slope at the domain edge $d_\text{lid,raw} \ge 0$ and $W_\text{sum} \ge 0$, and $w_T$ evaluates the $C^1$ positive part $\mathrm{where}(\Delta T_\text{lid} > 0, \Delta T_\text{lid} \tanh(\Delta T_\text{lid} / (0.3\,\mathrm{NO\_LID\_DT\_WIDTH})), 0)$ so $w_\text{lid,present}$ vanishes identically when $d_\text{lid,raw} \le 0$, $T_i - T_\text{surf} \le 0$, or $W_\text{sum} \le 0$, preventing numerical floor cancellation artifacts when $\Delta T \to 0$ and keeping all diagnostics continuous across lid onset), while $T_i$, $\eta_i$, $\mu_i$, and $\theta$ serve as interior fallback diagnostics and all solver-facing outputs vary continuously across uniform convective flux fractions $f \in [0, 10^{-3}]$. Intentional non-smooth switches retained for boolean diagnostics and physical bounds include `is_convecting`, `has_lid`, the `is_convecting` gate on the diagnostic $v_i$, the cap of $f$ at 1, `f_int_max = max(f * interior_node_mask)`, `max(T_i, 1.0)`, `max(H_T_i, 1e-6)`, `DeltaT_conv = max(T_i - T_surf, 1e-6)`, `max(theta, 1e-6)`, `max(Ra_rh_raw, 1e-30)`, the elementwise maxima in $w_\text{interior}$ and $w_\text{hot,sol}$, the temperature-gradient floor `dTdr_floor` with `max(dr_min, 1.0)` in $\delta_\text{rh}$, the mantle-thickness cap on $\delta_\text{rh}$, the `COHESION_FLOOR` clamp on yield stress, `if_lid` (`where(w_lid_present >= 1 - 1e-12)`), `d_geom`, the max in `d_lid_raw`, `max(w_conv - w_conv_0, 0)`, `f_int_scale`, the $v_{m,\text{raw}}$ branches, and the $10^{-30}$ denominators (while the 5-node local interior RMS $|F_\text{tot}|$ floor is continuous in every $F_k$).

### 4.1 Heating Mode and Evolutionary Context

The $2/3$ velocity scaling law applies to bottom-heated stagnant-lid convection (Dumoulin et al., 1999; Foley and Bercovici, 2014, p. 587). In planetary evolution, cooling terrestrial planets start with a superheated core, creating a large core-mantle boundary temperature difference that drives strong basal heating during early post-magma-ocean evolution (Thiriet et al., 2019, pp. 139-140, sec. 1).

As the core cools and radiogenic heating decays, the mantle heating regime transitions from bottom-dominated to predominantly internal heating (Thiriet et al., 2019, p. 140). For purely internally heated convection, Solomatov and Moresi (2000, p. 21804, eq. 13; p. 21805, Table 8) obtain a $1/2$ velocity scaling exponent:

$$
v_m = a_u \frac{\kappa}{d} \left( \frac{\mathrm{Ra}_i a_\text{rh}}{\theta} \right)^{1/2}
$$

with $a_u = 0.385 \pm 0.003$ for Newtonian rheology ($n = 1$) and $a_\text{rh} = 2.4$. In both the bottom-heated and internally heated scalings, the temperature contrast $\Delta T$ enters the definitions of Rayleigh number and Frank-Kamenetskii parameter symmetrically, cancelling in the ratio $\mathrm{Ra} / \theta$. For the Earth reference state ($\rho_i = 3300\text{ kg m}^{-3}$, $g_i = 9.81\text{ m s}^{-2}$, $\alpha_i = 3 \times 10^{-5}\text{ K}^{-1}$, $T_i = 1600\text{ K}$, $T_\text{surf} = 300\text{ K}$, $d = 2.89 \times 10^6\text{ m}$, $\kappa_i = 10^{-6}\text{ m}^2\text{ s}^{-1}$, $H = 300\text{ kJ mol}^{-1}$, $\mu_i = 10^{20}\text{ Pa s}$, giving $\theta = 18.32$ and $\mathrm{Ra}_i = 3.05 \times 10^8$), the internally heated scaling predicts $v_m = 8.42 \times 10^{-10}\text{ m s}^{-1}$ and convective driving shear stress $\tau_d = 2 \mu_i v_m / d \approx 0.058\text{ MPa}$ at $a_\text{rh} = 2.4$ (or $v_m = 5.43 \times 10^{-10}\text{ m s}^{-1}$ and $\tau_d \approx 0.038\text{ MPa}$ with $a_\text{rh} = 1.0$), approximately four times lower than the bottom-heated estimate ($v_m = 3.36 \times 10^{-9}\text{ m s}^{-1}$, $\tau_d \approx 0.232\text{ MPa}$ at $C_4 = 0.125$, $a_\text{rh} = 1.3$). Consequently, evaluating $\tau_d$ with the bottom-heated $2/3$ scaling represents an upper estimate on convective shear stress during later, internally heated epochs.

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

The effective solid viscosity combines diffusion creep and plastic yielding through the smooth minimum formulation implemented in `aragog.rheology_lid.compute_effective_viscosity` (Moresi and Solomatov, 1998, p. 672, eqs. 13-14, doi:10.1046/j.1365-246X.1998.00531.x; Tackley, 2000, p. 4, eq. 8, doi:10.1029/2000GC000036):

$$
\eta_\text{yield} = \frac{\tau_\text{yield}}{D}, \quad \eta_\text{eff} \approx \begin{cases} \eta_\text{diff}, & \tau < \tau_\text{yield} \\ \min(\eta_\text{diff}, \eta_\text{yield}), & \tau \ge \tau_\text{yield} \end{cases}
$$

In boundary-layer `'lid'` mode, `compute_effective_viscosity` requires the convecting interior viscosity $\eta_i$, the driving stress $\tau_d$, and the lid yield stress $\tau_{y,\text{lid}}$. The second invariant of the convective strain rate acting on the boundary layer is set by the interior convective flow scale $D_\text{conv} = 2 \dot{\epsilon}_\text{conv} = 2 v_m / d = \tau_d / \eta_i$ (Foley and Bercovici, 2014, p. 588, eq. 28). Substituting $D_\text{conv} = \tau_d / \eta_i$ gives the plastic yield viscosity branch $\log_{10}\eta_{y,\text{branch}}(r) = \mathrm{softmin}\big(\log_{10}\eta_\text{diff}(r), \; \log_{10}(\eta_i \tau_{y,\text{lid}} / \tau_d), \; \mathrm{LOCAL\_CAP\_LOG10\_WIDTH}\big)$, which is blended smoothly in $\log_{10}$ space across $\tau_d = \tau_{y,\text{lid}}$ with linear stress-ratio width `YIELD_BLEND_RATIO_WIDTH = 0.025`:

$$
w_b = \frac{1}{2}\left(1 + \tanh\left(\frac{\tau_d / \tau_{y,\text{lid}} - 1}{\mathrm{YIELD\_BLEND\_RATIO\_WIDTH}}\right)\right), \quad \log_{10}\eta_\text{eff}(r) = (1 - w_b)\log_{10}\eta_\text{diff}(r) + w_b \log_{10}\eta_{y,\text{branch}}(r)
$$

Well below yield ($\tau_d \ll \tau_{y,\text{lid}}$), $w_b \to 0$ and $\eta_\text{eff}(r) \to \eta_\text{diff}(r)$, preserving the rigid conductive stagnant lid. For standard Byerlee rock friction ($c = 50\text{ MPa}$, $\mu = 0.6$, $\tau_\text{max} = 500\text{ MPa}$), $\tau_{y,\text{lid}} = 500\text{ MPa}$ exceeds the Earth reference driving stress ($\tau_d \approx 0.232\text{ MPa}$) by more than three orders of magnitude, so the lid remains unyielded. Above yield ($\tau_d > \tau_{y,\text{lid}}$), $w_b \to 1$ and $\eta_\text{yield} = \eta_i (\tau_{y,\text{lid}} / \tau_d) < \eta_i$, lowering the yielded boundary-layer viscosity below the interior convective viscosity and mobilising the surface boundary layer into thin-boundary-layer convection. For example, at a cold surface ($T_\text{surf} = 300\text{ K}$, $\eta_\text{diff} = 10^{40}\text{ Pa s}$, $\eta_i = 10^{21}\text{ Pa s}$) with $\tau_d / \tau_{y,\text{lid}} = 1.2$, the yielded viscosity drops to $\eta_\text{eff} = 8.33 \times 10^{20}\text{ Pa s} < \eta_i$, whereas evaluating the strain rate from the cold-lid creep viscosity ($\eta_\text{diff} \tau_{y,\text{lid}} / \tau_d$) would leave $\eta_\text{eff} = 8.33 \times 10^{39}\text{ Pa s}$. An alternative harmonic-mean formulation $\eta_\text{eff} = \eta_\text{diff} \eta_\text{yield} / (\eta_\text{diff} + \eta_\text{yield})$ is discussed by Foley and Becker (2009, p. 3, eqs. 7-8) and Foley and Bercovici (2014, sec. 8.2).

The effective solid viscosity through the radial column is determined by applying the closure viscosity within the cold boundary layer using the smooth lid mask:

$$
\log_{10} \eta_\text{solid} = w_\text{lid} \log_{10} \eta_\text{eff} + (1 - w_\text{lid}) \log_{10} \eta_\text{diff}
$$

Inside the cold lid ($w_\text{lid} \to 1$), $\eta_\text{solid} \to \eta_\text{eff}$. In the warm convective interior ($w_\text{lid} \to 0$), $\eta_\text{solid} \to \eta_\text{diff}$. This formulation restricts plastic yielding to the cold boundary layer without altering the interior convective mantle (Foley and Becker, 2009, sec. 3.2.2).

The smooth minimum closure is everywhere finite, $C^\infty$ smooth, and non-increasing with driving stress. In `'local'` mode, `compute_effective_viscosity` evaluates $\eta_y(r) = \max(\tau_y(r), 10^{-6}\text{ Pa}) / (2 \sqrt{\dot{\epsilon}_\text{local}(r)^2 + 10^{-200}})$ and applies a smooth $\log_{10}$ softplus cap of width `LOCAL_CAP_LOG10_WIDTH = 0.01` between $\log_{10}\eta_\text{diff}(r)$ and $\log_{10}\eta_y(r)$; in `'lid'` mode, $\log_{10}\eta_\text{eff}(r)$ blends smoothly across $\tau_d = \tau_{y,\text{lid}}$ with width `YIELD_BLEND_RATIO_WIDTH = 0.025`. To prevent zero viscosity when zero cohesion and friction ($c = 0, \mu = 0$) are configured at zero pressure and to keep `jax.grad` finite when yield stress is $+\infty$, `compute_effective_viscosity` floors finite yield stress at `COHESION_FLOOR = 1.0e-6 Pa` and substitutes a finite placeholder inside `where` before the ratio when yield stress is $+\infty$, returning $\eta_\text{diff}$ in the outer `where` without capping finite yield stresses or high diffusion viscosities.

When convective vigor ceases ($v_i \to 0$), convective heat transport drops below $f_\text{conv,min}$. In this limit, $w_\text{active} \to 0$, $\tau_d = 0$, the lid closure remains inactive, and no division by zero occurs.

## 6. Interaction with the Eddy-Diffusivity Floor

In `aragog.jax.phase.compute_mlt` and `aragog.solver.entropy_state.apply_kappah_floor`, the thermal eddy diffusivity $\kappa_h$ is floored in cells with $\partial S / \partial r < D_\text{thr}$ (where $D_\text{thr}$ is the convective threshold parameter configured via `energy.convective_threshold` on `EnergyConfig` / `PhaseParams.convective_threshold`, defaulting to $0.0\text{ J kg}^{-1}\text{ K}^{-1}\text{ m}^{-1}$ on `EnergyConfig` and `PhaseParams.__init__`; convectively unstable when $D_\text{thr} = 0$) by a melt-fraction-modulated ramp without a lid mask:

$$
f(\phi) = \frac{1}{2} \left( 1 + \tanh\left( \frac{\phi - \phi_\text{rheo}}{\phi_\text{width}} \right) \right)
$$

$$
\kappa_{h,\text{floor,eff}} = \kappa_{h,\text{floor}} \cdot f(\phi) \cdot I_\text{conv}, \quad \kappa_h = \max\big(\kappa_{h,\text{MLT}}, \; \kappa_{h,\text{floor,eff}}\big)
$$

where $I_\text{conv} = 1$ in cells with $\partial S / \partial r < D_\text{thr}$ (convectively unstable when $D_\text{thr} = 0$) and $0$ in cells with $\partial S / \partial r \ge D_\text{thr}$ (stably stratified when $D_\text{thr} = 0$).

In standalone Aragog configurations, the transition width defaults to `rheological_transition_width = 0.15` ($\phi_\text{width} = 0.15$ at $\phi_\text{rheo} = 0.4$, where $f(0) = 0.004805$), and `kappah_floor` defaults to $10.0\text{ m}^2\text{ s}^{-1}$ on `EnergyConfig` and `_EnergyParameters` (while the direct `PhaseParams.__init__` fallback in `aragog.jax.phase` is $0.0\text{ m}^2\text{ s}^{-1}$). In coupled PROTEUS runs (`proteus.config._interior.Aragog`), the defaults are `rfront_loc = 0.4` ($\phi_\text{rheo} = 0.4$), `rfront_wid = 0.2` ($\phi_\text{width} = 0.2$, where $f(0) = 0.018$), and `kappah_floor = 10.0` ($\kappa_{h,\text{floor}} = 10.0\text{ m}^2\text{ s}^{-1}$). At $\kappa_{h,\text{floor}} = 10.0\text{ m}^2\text{ s}^{-1}$ and zero melt fraction ($\phi = 0$), $\kappa_{h,\text{floor,eff}}$ evaluates to $0.048\text{ m}^2\text{ s}^{-1}$ in solid cells with $\partial S / \partial r < D_\text{thr}$ (convectively unstable when $D_\text{thr} = 0$) at the standalone width $\phi_\text{width} = 0.15$ and $0.18\text{ m}^2\text{ s}^{-1}$ at the PROTEUS width $\phi_\text{width} = 0.2$, exceeding molecular thermal diffusivity ($\kappa \sim 10^{-6}\text{ m}^2\text{ s}^{-1}$) by four to five orders of magnitude. In cells with $\partial S / \partial r \ge D_\text{thr}$ (stably stratified when $D_\text{thr} = 0$), $I_\text{conv} = 0$ and $\kappa_{h,\text{floor,eff}} = 0$.

## 7. Stress Closure Modes

The geometry of the strain rate closure in `compute_effective_viscosity` is controlled by `stress_closure_mode`, and cross-mode keyword arguments are rejected (negative values of `tau_y_lid`, `tau_y`, or `tau_y_profile`, including $-\infty$, raise `ValueError` on the eager NumPy path and return `NaN` on the JAX path, whereas $0$ is floored at `COHESION_FLOOR = 1.0e-6 Pa` and $+\infty$ is accepted):

- `lid`:
    Boundary-layer stress closure. Requires `eta_i`, `tau_d`, and `tau_y_lid` (rejecting local-mode arguments `tau_y`, `tau_y_profile`, `strain_rate`, `mixing_length`, and `unyielded_velocity`). The driving stress $\tau_d = (2 \mu_i v_m / d) w_\text{active}$ is evaluated from the convecting interior scaling, and plastic yield viscosity $\eta_\text{yield} = \eta_i (\tau_{y,\text{lid}} / \tau_d)$ mobilises the lid nodes when $\tau_d > \tau_{y,\text{lid}}$.

- `local`:
    Local mixing-length strain rate closure (rejecting lid-mode arguments `eta_i`, `tau_d`, `tau_y_lid`, and `w_lid`). The strain rate is evaluated directly from local cell velocity and mixing length using the second invariant convention:

    $$
    \dot{\epsilon}_\text{local}(r) = \frac{|v_\text{unyielded}(r)|}{2 \max(l(r), 10^{-15}\text{ m})}
    $$

    and $\log_{10}\eta_\text{eff}(r)$ applies a smooth softplus cap of width `LOCAL_CAP_LOG10_WIDTH = 0.01` between $\log_{10}\eta_\text{diff}(r)$ and $\log_{10}\big(\max(\tau_y(r), 10^{-6}\text{ Pa}) / (2 \sqrt{\dot{\epsilon}_\text{local}(r)^2 + 10^{-200}})\big)$ (returning $\eta_\text{diff}(r)$ when $\tau_y(r) = +\infty$ or $\dot{\epsilon}_\text{local}(r) \le 0$). In the two-pass mixing-length solver, the pass-1 viscous velocity is $v_{\text{visc},1} = \rho g \alpha T (-\partial S / \partial r)_\text{eff} l_v^3 / (18 \eta_\text{diff})$, so the pass-1 viscous shear stress $\tau_1 = 2 \eta_\text{diff} \dot{\epsilon}_\text{local} = \eta_\text{diff} |v_{\text{visc},1}| / l_v = \rho g \alpha T (-\partial S / \partial r)_\text{eff} l_v^2 / (18 C_p) = \tau_{\text{buoy},1}$ depends on the local buoyancy driving force and mixing length rather than on $\eta_\text{diff}$. Where $\tau_{\text{buoy},1} > \tau_y(r)$, pass 1 reduces $\eta_\text{eff}(r) \approx \eta_\text{diff}(r) \tau_y(r) / \tau_{\text{buoy},1}$, and pass 2 scales the viscous velocity $v_{\text{visc},2}$ and thermal eddy diffusivity $\kappa_h$ upward by the factor $\tau_{\text{buoy},1} / \tau_y(r)$ while keeping $\tau_2 = \eta_\text{eff} |v_{\text{visc},2}| / l_v = \tau_{\text{buoy},1}$ in the viscous regime ($\mathrm{Re} \ll \mathrm{Re}_\text{crit}$). Thus in `'local'` mode yielding scales the convective velocity by $\tau_{\text{buoy},1} / \tau_y$ and is not a stress cap.

- `global`:
    Unsupported mode. Configuration validation rejects `global` and instructs users to specify `lid`.

In `SolidRheologyParams`, `stress_closure_mode` defaults to `'lid'`; callers pass `stress_closure_mode` to `compute_effective_viscosity` to select the boundary-layer closure (`'lid'`) or the local strain-rate closure (`'local'`).

## 8. Asymptotic Limits

The kernels in `aragog.rheology` and `aragog.rheology_lid` satisfy the following asymptotic limits:

1. **Zero driving stress ($\tau_d \to 0$):**
   $\tau_d / \tau_{y,\text{lid}} \to 0 \implies w_y \approx 0$ ($\frac{1}{2}(1 - \tanh 10) \approx 2.1 \times 10^{-9}$ when a lid is present, or $0$ when `has_lid` is `False`) and $\eta_\text{eff} = \eta_\text{diff}$. The system exhibits pure Arrhenius diffusion creep without plastic weakening.
2. **Infinite yield stress ($\tau_y \to \infty$):**
   $\tau_d / \tau_{y,\text{lid}} \to 0 \implies \eta_\text{eff} = \eta_\text{diff}$, yielding stays inactive, and the lid remains stagnant at diffusion-creep viscosity.
3. **Small homologous activation parameter ($g_\text{hom} \to 0^+$):**
   `SolidRheologyParams` requires `g_homologous > 0` when `enabled = True` in both `'rheological'` and `'fixed'` lid modes. In `compute_arrhenius_viscosity`, taking $g_\text{hom} \to 0$ gives $H(P) \to 0$ and recovers an isoviscous solid mantle $\eta_\text{diff} = \eta_\text{solid} f_\text{water}$.
4. **Zero solidus pressure slope ($dT_\text{sol} / dP = 0$):**
   $H(P) = g_\text{hom} R T_{\text{sol},0}$, recovering temperature-only Arrhenius creep with constant activation energy.
5. **Infinite Frank-Kamenetskii contrast ($\theta \to \infty$):**
   Recovers the asymptotic stagnant-lid regime of Solomatov (1995) with upper mixing-length slope $s_\text{top}(\theta) \to s_{\text{top},\infty}$.
6. **Sub-threshold interior convection ($f_\text{int,max} \ll f_\text{conv,min}$) or no conductive lid ($d_\text{lid,raw} \to 0$):**
   When $f_\text{int,max}$ lies more than $0.01\,\mathrm{arctanh}(0.9) \approx 0.0147$ below $f_\text{conv,min}$ (for example $f_k \le 0.035$ at $f_\text{conv,min} = 0.05$), $w_\text{active} \le 0.05$, `is_convecting` and `has_lid` are `False`, boundary-layer diagnostics ($\tau_d$, $v_m$, $\mathrm{Ra}_\text{eff}$, $d_\text{lid}$, $w_y$, `lid_regime`) are smoothly gated by $w_\text{lid,present} = w_d w_T w_\text{act,gate} w_\text{sum,gate} w_\text{weight}$, and yielding remains inactive across any mesh resolution $N$.
7. **Wholly molten mantle ($\phi \ge 0.012$ at every node, with or without a solidus table) and the compact $[0.008, 0.012]$ solidus ramp:**
   In `compute_stagnant_lid_state`, both the table-free and table-supplied solidus indicators include the compactly supported $C^2$ quintic melt-fraction ramp $w_{\text{sol},\phi} = t^3(10 - 15 t + 6 t^2)$ with $t = \mathrm{clip}((\phi - 0.008) / 0.004, 0, 1)$ centered at `SOLIDUS_MELT_FRACTION_THRESHOLD = 0.01` with half-width `SOLIDUS_MELT_FRACTION_WIDTH = 0.002` (taking the elementwise maximum with the thermal solidus `tanh` ramp when `solidus_temperature` is supplied), which bounds the raw lid thickness by $d_\text{lid,raw} \le w_\text{active} \sum_{k=0}^{N-2} (1 - w_{\text{interior},k+1}) \Delta r_k$. When $\phi \ge 0.012$ at every node, $w_{\text{sol},\phi} = 1$, $w_\text{hot,sol} = 1$, and $w_\text{interior} = 1$ identically throughout the column regardless of whether a solidus table is supplied or how steep the cell-to-cell temperature gradient is, so the lid-base boundary weight $\tilde{W}_\text{base}$ places $r_\text{lid,base}$ strictly at the surface node $r[-1]$, giving $d_\text{lid,raw} = 0$, `has_lid = False`, $d_\text{lid} = 0$, $\tau_d = 0$, and `lid_regime = 0.0` across all mesh resolutions $N$ (including $N \in \{3, 10, 50, 100, 200\}$ on both Earth and $9000\text{ km}$ mantle spans). For $\phi \le 0.008$, $w_{\text{sol},\phi} = 0$ identically, confining the melt-fraction solidus transition strictly to the compact interval $\phi \in [0.008, 0.012]$.
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
