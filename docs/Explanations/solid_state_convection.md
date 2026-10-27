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

### Optional Saturating Integral Activation Law

#### How does activation enthalpy change with pressure in deep planetary mantles, and when does it reach a constant ceiling?

In shallow mantles, pressure increases the energy that mantle rock requires to deform by solid-state diffusion creep.
At high pressures inside massive rocky planets, mineral physics experiments show that this pressure sensitivity decreases.
Aragog provides an optional saturating integral law that smoothly caps the activation enthalpy at deep mantle pressures.

A user activates this formulation when they set `activation_energy` ($E_a$) on `SolidRheologyParams`, `compute_arrhenius_enthalpy`, or `compute_arrhenius_viscosity`.
When set, the solver evaluates activation enthalpy from pressure rather than the homologous melting curve.

Here:
- $P$ is the hydrostatic pressure in pascals ($\text{Pa}$).
- $H(P)$ is the activation enthalpy in joules per mole ($\text{J mol}^{-1}$).
- $E_a$ is the zero-pressure activation energy in joules per mole ($\text{J mol}^{-1}$, code parameter `activation_energy`).
- $V_0$ is the zero-pressure activation volume in cubic metres per mole ($\text{m}^3\text{ mol}^{-1}$, code parameter `activation_volume`, default $0.0\text{ m}^3\text{ mol}^{-1}$).
- $P_\text{decay}$ is the decay pressure in pascals ($\text{Pa}$, code parameter `activation_volume_decay_pressure`, default $+\infty$).
- $p'$ is the dummy pressure integration variable in pascals ($\text{Pa}$).

The integral activation law integrates an exponential decay of activation volume over pressure:

$$
H(P) = E_a + \int_0^P V_0 \exp\left(-\frac{p'}{P_\text{decay}}\right) dp' = E_a + V_0 P_\text{decay} \left(1 - \exp\left(-\frac{P}{P_\text{decay}}\right)\right)
$$

At low pressures where the pressure ratio satisfies $P / P_\text{decay} < 10^{-8}$, Aragog computes the effective pressure term with a second-order Taylor series:

$$
P \left(1 - \frac{P}{2 P_\text{decay}}\right)
$$

This Taylor series avoids catastrophic loss of precision from floating-point subtraction in floating-point arithmetic.

The formulation satisfies five physical and mathematical limits:

- When $P_\text{decay} = +\infty$, the formula recovers the linear activation volume law $H(P) = E_a + V_0 P$ (Karato and Wu, 1993; Armann and Tackley, 2012) to 0 units in the last place (ulp).
- When $V_0 = 0$, the formula recovers the isobaric Arrhenius law $H(P) = E_a$ (Korenaga, 2009, Table 1) to 0 ulp.
- The local differential activation volume $\partial H / \partial P = V_0 \exp(-P / P_\text{decay}) \ge 0$ is non-negative at all pressures.
- As pressure approaches infinity ($P \to \infty$), $H(P)$ increases monotonically toward the constant ceiling $E_a + V_0 P_\text{decay}$.
- For finite positive decay pressure $P_\text{decay} > 0$, the Lagrange remainder theorem bounds the departure from the linear law:

$$
0 \le \mathrm{d}H(P) = H_\text{lin}(P) - H(P) \le \text{Bound}(P) = \frac{V_0 P^2}{2 P_\text{decay}}
$$

#### Worked examples with real numbers

Consider the core-mantle boundary (CMB) of Earth at pressure $P = 135\text{ GPa}$.
Assume $E_a = 300\text{ kJ mol}^{-1}$, $V_0 = 6.0\text{ cm}^3\text{ mol}^{-1}$, and $P_\text{decay} = 200\text{ GPa}$.
Under these parameters:
- The linear activation volume law gives $H_\text{lin} = 1110.0\text{ kJ mol}^{-1}$.
- The saturating integral law gives $H = 889.012\text{ kJ mol}^{-1}$.
- The reduction in activation enthalpy is $\mathrm{d}H = 220.988\text{ kJ mol}^{-1}$.
- This reduction satisfies the theoretical Lagrange remainder bound $\text{Bound} = 273.375\text{ kJ mol}^{-1}$.

For comparison, the upper-mantle parameter set from Tackley et al. (2013, Table 2) specifies $E_0 = 300\text{ kJ mol}^{-1}$, $V_0 = 5.0\text{ cm}^3\text{ mol}^{-1}$, and $P_\text{decay} = \infty$.
At Earth core-mantle boundary pressure $P = 135\text{ GPa}$, that linear reference gives $H(135\text{ GPa}) = 975.0\text{ kJ mol}^{-1}$.

![Saturating integral activation enthalpy limits and convergence](../assets/validation/ssc_activation_law_limits.svg#only-light){ width="80%" }
![Saturating integral activation enthalpy limits and convergence](../assets/validation/ssc_activation_law_limits_dark.svg#only-dark){ width="80%" }

**How to read the plot.**
Panel (a) plots $H(P)$ over pressure $P \in [0, 1000]\text{ GPa}$ for $E_a = 300\text{ kJ mol}^{-1}$ and $V_0 = 5.0\text{ cm}^3\text{ mol}^{-1}$.
It compares curves for $P_\text{decay} \in \{200, 1100, 1610, \infty\}\text{ GPa}$ against the $P_\text{decay} = 200\text{ GPa}$ secant curve and the $V_0 = 0$ isobaric limit.
Panel (b) plots the enthalpy difference $\mathrm{d}H = H_\text{lin}(P) - H_\text{int}(P)$ against the analytical Lagrange bound $\text{Bound} = V_0 P^2 / (2 P_\text{decay})$.
The calculation evaluates Earth core-mantle boundary pressure $P = 135\text{ GPa}$ over decay pressures $P_\text{decay} \in [10^2, 10^6]\text{ GPa}$ ($E_a = 300\text{ kJ mol}^{-1}$, $V_0 = 5.0\text{ cm}^3\text{ mol}^{-1}$, where $V_0 P^2 / 2 = 45562.5\text{ kJ mol}^{-1}\text{ GPa}$).
The secant-form parameter sets of Tackley et al. (2013, Table 2) for $E_0 + P V_0 \exp(-P / P_\text{decay})$ are not valid parameter sets of this integral law.

#### Bounds and parameter validation

- `activation_energy` ($E_a$): must be non-negative ($E_a \ge 0\text{ J mol}^{-1}$); configuration validation raises `ValueError` if negative, and requires $E_a > 0$ when `rheology.enabled = True`.
- `activation_volume` ($V_0$): must be non-negative ($V_0 \ge 0\text{ m}^3\text{ mol}^{-1}$, default $0.0\text{ m}^3\text{ mol}^{-1}$); configuration validation raises `ValueError` if negative.
- `activation_volume_decay_pressure` ($P_\text{decay}$): must be strictly positive ($P_\text{decay} > 0\text{ Pa}$, default $+\infty$); configuration validation raises `ValueError` if zero or negative.
- Low-pressure series: When $P / P_\text{decay} < 10^{-8}$, the code switches to the second-order Taylor series.

**Where this is checked.** This activation law is checked in `tests/test_rheology_verification.py::test_saturating_activation_enthalpy_limits`.

??? info "Test details"
    - Reference: Linear activation-volume law $H_\text{lin}(P) = E_a + V_0 P$ of Karato and Wu (1993, p. 772, eq. 1), Armann and Tackley (2012, p. 4, eq. 1: $E_a = 300\text{ kJ mol}^{-1}$, $V_0 = 6.0\text{ cm}^3\text{ mol}^{-1}$), and Tackley et al. (2013, p. 4, eqs. 2-3; p. 19, Table 2 upper-mantle row: $E_0 = 300\text{ kJ mol}^{-1}$, $V_0 = 5.0\text{ cm}^3\text{ mol}^{-1}$, $p_\text{decay} = \infty$), and isobaric limit $H(P) = E_a$ at $V_0 = 0$.
    - Tolerance: Bitwise equality (`0.0` relative, `0` ulp) for both `compute_arrhenius_enthalpy` and `compute_arrhenius_viscosity` at $P_\text{decay} = \infty$ and at $V_0 = 0$ over $P \in [0, 1000]\text{ GPa}$ in both NumPy and JAX; `jax.grad` derivative $\partial H / \partial P = V_0 \exp(-P / P_\text{decay})$ to $10^{-12}$ relative; for finite $P_\text{decay} > 0$, analytical Lagrange remainder bound $0 \le \mathrm{d}H(P) = H_\text{lin}(P) - H(P) \le \text{Bound}(P) = V_0 P^2 / (2 P_\text{decay})$ (evaluated over $P \in [0, 1000]\text{ GPa}$ in panel a and at $P = 135\text{ GPa}$ over $P_\text{decay} \in [10^2, 10^6]\text{ GPa}$ in panel b).
    - Test id: `tests/test_rheology_verification.py::test_saturating_activation_enthalpy_limits`
    - Command: `pytest tests/test_rheology_verification.py -k test_saturating_activation_enthalpy_limits` (add `--validation-plots` to write the plot pair)

### Viscosity-Contrast-Dependent Mixing-Length Slope

In the viscous mixing-length formulation, the effective mixing length $l_v(r)$ near the mantle boundaries is given by:

$$
l_v(r) = \min\big( s_\text{bot} (r - r_\text{in}), \; s_\text{top}(\theta) (r_\text{out} - r) \big)
$$

where $s_\text{bot}$ is the lower boundary slope (`mlt_bottom_slope`, default $1.0$) and $s_\text{top}(\theta)$ is the viscosity-contrast-dependent upper boundary slope evaluated by `compute_mlt_top_slope`:

$$
s_\text{top}(\theta) = s_{\text{top},\infty} + (1 - s_{\text{top},\infty}) \exp\big( -c_\theta \max(\theta, 0) \big)
$$

with decay rate $c_\theta = 0.15$ (`c_theta`) and stagnant-lid asymptotic slope $s_{\text{top},\infty} = 0.22$ (`s_top_inf`, default $0.22$). Because the viscous eddy diffusivity scales as $l^4$ and the viscous velocity scales as $l^3$, `viscous_mixing_length_factor` returns $(l_v / l)^4$, multiplying the viscous velocity by $(l_v / l)^3$ and the viscous eddy diffusivity by $(l_v / l)^4$ at interior nodes while keeping $1.0$ at the boundary nodes where $l = 0$.

#### How well does the contrast-dependent mixing-length slope match published 2-D and 3-D convection benchmark grids?

Viscous mixing-length theory uses the exponential slope $s_\text{top}(\theta)$ to represent how a cold, high-viscosity surface lid reduces the convective mixing length near the upper boundary. In `SolidRheologyParams`, `compute_mlt_top_slope`, and `tests/test_mlt_calibration.py`, the default parameters are $s_{\text{top},\infty} = 0.22$ (`s_top_inf`) and $c_\theta = 0.15$ (`c_theta`), which satisfy the exact isoviscous limit $s_\text{top}(\theta \le 0) = 1.0$, the intermediate evaluation $s_\text{top}(15) = 0.22 + 0.78 \exp(-2.25) = 0.3022113951582542$, and the asymptotic stagnant-lid limit $s_\text{top}(\theta \to \infty) \to 0.22$ (entered in the strong stagnant-lid regime $\theta \gg 8$).

Steady-state 1-D evaluations across published stagnant-lid convection benchmarks are recorded in `tests/reference/ssc_mlt_top_slope_calibration.json` (generated by `python tools/verification/evaluate_mlt_top_slope_calibration.py --full` with `stress_closure_mode = 'local'`, $s_\text{top}$ frozen at each table's nominal $\theta$ via `c_theta = 0`, a half-cell conductive boundary-flux patch on outer BC 5 and inner BC 3 via `_patched_state_update`, and a Frank-Kamenetskii viscosity override during steady-state integration on the spherical cases; `--check` recomputes `E340` and `f60_H4` live and verifies all stored rows and summary statistics):

1. **Planar internally heated Newtonian stagnant-lid convection (Korenaga, 2009, Table 1, 12 cases):** Evaluated on an $N = 100$ uniform planar Boussinesq grid against a $10\%$ relative tolerance on the Nusselt number $\mathrm{Nu}$ ($r_\text{norm} = |\mathrm{Nu} - \mathrm{Nu}_\text{ref}| / (0.10 \, \mathrm{Nu}_\text{ref})$). At the chosen point $(s_{\text{top},\infty}, c_\theta) = (0.25, 0.25)$, all $12$ of $12$ cases lie within the $10\%$ tolerance, with maximum normalized residual $\max |r_\text{norm}| = 0.9839$ on case `E340` ($\mathrm{Nu} = 3.3391$ versus $\mathrm{Nu}_\text{ref} = 3.04$, relative difference $+9.84\%$) and mean-square normalized residual $J_\text{Korenaga} = 0.8923$. At the default pair $(s_{\text{top},\infty}, c_\theta) = (0.22, 0.15)$, $12$ of $12$ cases exceed the $10\%$ tolerance, with $\max |r_\text{norm}| = 1.4062$ on case `E210` and $J_\text{Korenaga} = 1.3883$.
2. **Spherical mixed-heating stagnant-lid convection (Deschamps and Vilella, 2021, Table 1, 24 cases, 72 observables):** Evaluated on an $N = 100$ spherical-shell Boussinesq grid ($f = r_\text{in} / r_\text{out} \in \{0.60, 0.70, 0.75, 0.80, 0.85\}$) against normalized tolerances of $0.03$ absolute on non-dimensional interior temperature $T_m$, $15\%$ relative on top heat flow $\Phi_\text{top}$, and $20\%$ relative on stagnant-lid thickness $d_\text{lid}$. At $(s_{\text{top},\infty}, c_\theta) = (0.25, 0.25)$, $34$ of $72$ residuals exceed $1.0$ over the $24$ spherical cases, with worst normalized residuals $\max |r_\Phi| = 2.1909$ on `f70_1e7_H2` ($r_\Phi = -2.1909$), $\max |r_{T_m}| = 1.9421$ on `f80_H2` ($r_{T_m} = -1.9421$), and $\max |r_d| = 1.3195$ on `f80_10_1e6_H10` ($r_d = -1.3195$), yielding mean-square normalized residual $J_\text{DV} = 1.0570$ and combined mean-square normalized residual $J_\text{all} = 1.0334$ over all $84$ observables ($36$ total cases). At the default pair $(s_{\text{top},\infty}, c_\theta) = (0.22, 0.15)$, $42$ of $72$ spherical residuals exceed $1.0$ ($\max |r_{T_m}| = 2.4116$ on `f80_H2`, $\max |r_\Phi| = 2.0141$ on `f70_1e7_H2`, $\max |r_d| = 1.5597$ on `f85_10_1e6_H8`, $J_\text{DV} = 1.4659$, and combined $J_\text{all} = 1.4549$ over all $84$ observables).

**Known limit.** While the 1-D mixing-length closure with local closure and $s_\text{top}$ frozen at the tabulated $\theta$ at the chosen point $(s_{\text{top},\infty}, c_\theta) = (0.25, 0.25)$ reproduces all $12$ planar internally heated cases of Korenaga (2009, Table 1) within the $10\%$ $\mathrm{Nu}$ tolerance ($\max |r_\text{norm}| = 0.9839$), it does not satisfy the spherical stagnant-lid benchmark suite of Deschamps and Vilella (2021, Table 1), where $34$ of $72$ observables exceed their tolerances ($0.03$ on $T_m$, $15\%$ on $\Phi_\text{top}$, $20\%$ on $d_\text{lid}$). A single contrast-dependent upper slope $s_\text{top}(\theta)$ without a spherical-curvature geometry factor cannot simultaneously match both planar internally heated and spherical mixed-heating stagnant-lid scaling across core-to-planet radius ratios $f \in [0.60, 0.85]$.

## 2. Plastic Yielding Criterion

To model lithospheric failure and mobile-lid convection, `compute_yield_stress` implements a Byerlee frictional yield criterion with a ductile stress ceiling (Byerlee, 1978; Moresi and Solomatov, 1998, p. 672, eq. 9; Tackley, 2000, p. 8, eq. 7):

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

#### Geotherm Endpoint Blending and Smooth Ramp Weighting

When $w_\text{weight} < 1$ on a strictly decreasing radial temperature profile ($T_{k+1} < T_k - 10^{-12}\text{ K}$), $P_{T_i}$ and $r_i$ are evaluated along the monotone geotherm $(T(r), P(r), r)$ at $T_i$ with endpoint-consistent correction $w_\text{eff} = \mathrm{clip}((T_0 - T_i) / (T_0 - T_1), 0, 1)$ between $(T_0, P_0, r_0) = (T_2, P_2, r_2)$ and $(T_1, P_1, r_1) = \sum_k \tilde{W}_{T_i, k} (T_k, P_k, r_k)$, and $\Delta T_\text{rh}$, $\theta$, and $\log_{10}\eta_\text{arr}$ blend with $w_\text{eff}$ between $(T_0, P_0)$ and $(T_1, P_1)$ so the sub-threshold state stays on the geotherm rather than linearly blending off it (`P_T_i` and `r_i` are diagnostics that no solver term reads, evaluated on the geotherm for a monotone $T$ profile and as the $\tilde{W}_k$ mean otherwise, and the two can differ by tens of GPa across the monotonicity switch). Because the selector places $T_i$ at the node under the lid, its accuracy is bounded by the radial mesh resolution across the thermal boundary layer: when the lid is one cell thick (for example on a 100-node post-magma-ocean profile where $T_i = 1423.0\text{ K}$ against $1788.6\text{ K}$ one node deeper), $T_i$ lies in the rheological sublayer.

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

- `'fixed'`: The lid base is defined by a user-specified isotherm $T_{\text{lid,iso}} = T_\text{base}$ set by `lid_base_temperature` (default $1400.0\text{ K}$), multiplied by the non-convective weight $w_\text{nonconv} = 1 - w_{\text{melt,compact}}(f,\, f_\text{conv,min},\, 0.2 f_\text{conv,min})$ on both $1 - w_\text{hot,iso}$ and $1 - w_\text{hot,iso,mask}$ so that a convecting node is never part of the lid.
- `'rheological'`: The lid base is evaluated dynamically from the Frank-Kamenetskii temperature scale:

    $$
    T_{\text{lid,iso}} = T_i - a \Delta T_\text{rh}
    $$

    where $a = 2.2$ is the lid contrast coefficient (`lid_contrast_coeff`). This coefficient comes from boundary-layer theory for steady-state Newtonian convection with exponential temperature-dependent viscosity in the asymptotic limit $\theta \gg 1$ (Solomatov, 1995; Tackley, 2000).

The physical lid base location is identified as the shallowest depth among:

1. The rheological isotherm where $T = T_{\text{lid,iso}}$.
2. The solidus crossing, evaluated from the compactly supported $C^2$ quintic melt-fraction ramp $w_{\text{sol},\phi} = t^3(10 - 15 t + 6 t^2)$ with $t = \mathrm{clip}((\phi - 0.008) / 0.004, 0, 1)$ centered at $\phi = 0.01$ (`SOLIDUS_MELT_FRACTION_THRESHOLD`) with half-width $0.002$ (`SOLIDUS_MELT_FRACTION_WIDTH`) and, when a solidus profile $T_\text{sol}$ is supplied, the elementwise maximum with the thermal solidus indicator $T = T_\text{sol}$ (because $w_{\text{sol},\phi} = 1$ identically for $\phi \ge 0.012$, any column with $\phi \ge 0.012$ at every node places $r_\text{lid,base}$ strictly at $r[-1]$ with $d_\text{lid} = 0$ and `has_lid = False` for every mesh resolution $N$; see Section 8 item 7).
3. The rheological crystallisation front where $\phi = \phi_\text{rheo} = 0.4$.

When the entire column is non-convecting and lies below the lid-base criterion so that the interior boundary indicator sum $W_\text{base,sum} \to 0$, `r_lid_base` anchors at the inner boundary radius `r[0]` (the core-mantle boundary), giving $d_\text{lid,raw} = (r_\text{outer} - r_\text{inner}) w_\text{active}$ (and $d_\text{lid} = 0$ when $w_\text{active} = 0$). In `'fixed'` mode, because convecting nodes have $w_\text{nonconv} = 0$ and therefore $w_\text{hot,iso} = 1$, a column colder than $T_\text{base}$ that convects on nodes $0 \dots k$ and is conductive above places `r_lid_base` at $r_k$; when no node is non-convecting ($w_\text{nonconv} = 0$ everywhere), $w_\text{interior} = 1$ at every node and `r_lid_base` sits at `r[-1]` with $d_\text{lid,raw} = 0$, $d_\text{lid} = 0$, and `has_lid = False`.

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

Using $w_\text{interior,mask}$ in $w_\text{lid}$ ensures that steep single-cell surface thermal boundary layers saturate at $w_\text{lid}(\text{surface}) \to 1$. In `'fixed'` lid-base mode, both `w_hot_iso` and `w_hot_iso_mask` are combined with the compactly supported convective suppression ramp $w_\text{nonconv} = 1 - w_\text{ramp}(f, f_\text{conv,min}, 0.2 f_\text{conv,min})$ via $1 - (1 - w_\text{hot,iso}) w_\text{nonconv}$ and $1 - (1 - w_\text{hot,iso,mask}) w_\text{nonconv}$, so convecting nodes ($f \ge 1.2 f_\text{conv,min}$) cannot be marked as lid ($w_\text{lid} = 0$ and $w_\text{hot,iso} = 1$) even when $T < T_\text{base}$. Inside the cold solid lid, $w_\text{lid} \to 1$. In the warm or partially molten interior, $w_\text{lid} \to 0$ (for $\phi \ge 0.012$, $w_{\text{sol},\phi} = 1$ and $w_\text{lid} = 0$ identically; within the compact transition band $0.008 < \phi < 0.012$, $w_\text{lid} \le 1 - w_{\text{sol},\phi}$, which equals $0.5$ at $\phi = 0.010$ and $0$ at $\phi = 0.012$). Under round-off perturbations $S_0(r)(1 + k \cdot 10^{-12})$ ($k \in \{-2, -1, 0, +1, +2\}$) over $100\text{ yr}$ on the active-solid variant of `cfg/abe_solid.toml` ($S_\text{init} = 2500\text{ J kg}^{-1}\text{ K}^{-1}$, $\kappa_{h,\text{floor}} = 0\text{ m}^2\text{ s}^{-1}$, $\eta_\text{solid} = 10^{18}\text{ Pa s}$, where the rheology shifts the end-time surface temperature from $481.620\text{ K}$ to $482.081\text{ K}$, a $0.461\text{ K}$ or $2.205 \times 10^{-3}$ relative change across the column), the trajectory spread remains smooth across the radial grid with $\max_r \Delta T_\text{on}(r) = 1.751 \times 10^{-5}\text{ K}$ against $\max_r \Delta T_\text{off}(r) = 3.439 \times 10^{-6}\text{ K}$ ($\le 0.05\text{ K}$), whereas on the default `cfg/abe_solid.toml` ($S_\text{init} = 2900\text{ J kg}^{-1}\text{ K}^{-1}$, $\kappa_{h,\text{floor}} = 10.0\text{ m}^2\text{ s}^{-1}$, $\eta_\text{solid} = 10^{21}\text{ Pa s}$) solid rheology changes the final basic temperatures by $\max_r |T_\text{on} - T_\text{off}| = 7.534\text{ K}$ ($2.052 \times 10^{-3}$ relative), with perturbation spreads $\max_r \Delta T_\text{on}(r) = 7.565 \times 10^{-7}\text{ K}$ and $\max_r \Delta T_\text{off}(r) = 1.035 \times 10^{-6}\text{ K}$ (see Section 6 for the zero-anchored floor ramp when `rheology.enabled` is `True`).

## 4. Convective Driving Stress

Convective flow beneath the stagnant lid exerts a driving shear stress on the base of the lid. `compute_stagnant_lid_state` implements the stagnant-lid convective shear stress scaling of Foley and Bercovici (2014, eqs. 26 and 28, pp. 586-588, doi:10.1093/gji/ggu275):

$$
\tau_d = \frac{2 \mu_i v_m}{d} w_\text{active}, \quad \tau_{d,\text{lid}} = \frac{2 \eta_i v_m}{d_\text{geom}} w_\text{active}
$$

where $d = r_\text{outer} - r_\text{inner}$ is the mantle thickness, $\mu_i$ (`eta_i` in `lid_state`) is the convecting interior dynamic viscosity, $v_m = w_\text{lid,present} v_{m,\text{raw}}$ is the interior convective velocity multiplied by $w_\text{lid,present}$, and $\tau_{d,\text{lid}}$ is the geometric-lid-thickness-scaled stress diagnostic (`tau_d_lid = w_lid_present * w_active * 2 * eta_i * v_m_raw / d_geom`, passed as `tau_d` to `compute_effective_viscosity` in solver pass 2) with smooth geometric length floor $d_\text{geom} = \mathrm{softmax}(r_\text{top} - r_\text{lid,base},\, d_\text{geom,floor},\, 0.01\,d_\text{geom,floor})$ and $d_\text{geom,floor} = \mathrm{softmax}(\delta_\text{rh},\, \Delta r_\text{top},\, 0.01\,\Delta r_\text{top})$ where $\Delta r_\text{top} = \max(r_{-1} - r_{-2}, 10^{-6}\text{ m})$. The factor of 2 arises from horizontal shear between the lid base ($u = v_m$) and mid-depth ($u = 0$ at $z = d/2$), as derived by Foley and Bercovici (2014, pp. 584, 588). The `lid_state` dictionary returns both the raw lid thickness `d_lid_raw` and the weighted `d_lid`, and evaluates the yield indicator `w_y` from $\tau_{d,\text{lid,raw}} / \max(\tau_{y,\text{lid}}, 10^{-6}\text{ Pa})$. Because $\tau_d$ divides by the full mantle thickness $d$ while $\tau_{d,\text{lid}}$ divides by $d_\text{geom} \ge \max(\delta_\text{rh}, \Delta r_\text{top})$, the ratio $\tau_{d,\text{lid}} / \tau_d = d / d_\text{geom}$ remains bounded above by $d / \max(\delta_\text{rh}, \Delta r_\text{top})$ on incipient lids and vanishes in non-lid columns where $w_\text{lid,present} = 0$.

The interior convective velocity $v_m$ follows the boundary-layer scaling for bottom-heated stagnant-lid convection (Dumoulin et al., 1999, p. 12760, sec. 2; Foley and Bercovici, 2014, pp. 586-587, eq. 26):

$$
v_m = \frac{\kappa_i}{d} C_4 \left( \frac{\mathrm{Ra}_\text{eff} a_\text{rh}}{\theta} \right)^{2/3}
$$

where $C_4 = 0.125$ is the empirical prefactor calibrated by Foley and Bercovici (2014, sec. 4.2, p. 589), $a_\text{rh} = 1.3$ is the rheological sublayer constant fitted jointly with $C_4$, and $\rho_i, g_i, \alpha_i, \kappa_i$ are evaluated as $\tilde{W}_k$-weighted means at the top of the convecting layer. The effective Rayleigh number is defined at the convecting interior viscosity $\mu_i$:

$$
\mathrm{Ra}_\text{eff} = \frac{\rho_i g_i \alpha_i (T_i - T_\text{surf}) d^3}{\kappa_i \mu_i}
$$

Because both $\mathrm{Ra}_\text{eff}$ and $\theta$ scale linearly with the temperature drop $\Delta T = T_i - T_\text{surf}$, the ratio $\mathrm{Ra}_\text{eff} a_\text{rh} / \theta$ is proportional to $a_\text{rh} \Delta T_\text{rh}$ where $\Delta T_\text{rh} = R T_i^2 / H(P_{T_i})$. When convection is inactive or reaches the surface such that no physical conductive lid forms ($d_\text{lid,raw} \le 1\text{ m}$ or $T_i - T_\text{surf} \le 10^{-3}\text{ K}$), `compute_stagnant_lid_state` multiplies $d_\text{lid}$, $\mathrm{Ra}_\text{eff}$, $\mathrm{Ra}_\text{rh}$, $v_m$, $\tau_d$, $\tau_{d,\text{lid}}$, $\tau_\text{buoy}$, $\tau_d / \tau_\text{buoy}$, $w_y$, and `lid_regime` by the weight $w_\text{lid,present} = w_d w_T w_\text{act\_gate} w_\text{sum\_gate} w_\text{weight}$ (where $w_d$ and $w_\text{sum\_gate}$ are $C^0$ with a one-sided slope at the domain edge $d_\text{lid,raw} \ge 0$ and $W_\text{sum} \ge 0$, and $w_T$ evaluates the $C^1$ positive part $\mathrm{where}(\Delta T_\text{lid} > 0, \Delta T_\text{lid} \tanh(\Delta T_\text{lid} / (0.3\,\mathrm{NO\_LID\_DT\_WIDTH})), 0)$ so $w_\text{lid,present}$ vanishes identically when $d_\text{lid,raw} \le 0$, $T_i - T_\text{surf} \le 0$, or $W_\text{sum} \le 0$, preventing numerical floor cancellation artifacts when $\Delta T \to 0$ and keeping all diagnostics continuous across lid onset), while $T_i$, $\eta_i$, $\mu_i$, and $\theta$ serve as interior fallback diagnostics and all solver-facing outputs vary continuously across uniform convective flux fractions $f \in [0, 10^{-3}]$. Intentional non-smooth switches retained for boolean diagnostics and physical bounds include `is_convecting`, `has_lid`, the `is_convecting` condition on the diagnostic $v_i$, the cap of $f$ at 1, `f_int_max = max(f * interior_node_mask)`, `max(T_0, 1.0)`, `max(T_1, 1.0)`, `max(H_0, 1e-6)`, `max(H_1, 1e-6)`, the `w_eff` clip to $[0, 1]$, `dT_ep_nz` (`|T_1 - T_0| > 1e-12`), `is_mono_T` (`all(diff(T) <= 0)`), the `1 - 1e-15` endpoint selectors on `w_eff` and `w_weight`, `DeltaT_conv = max(T_i - T_surf, 1e-6)`, `max(theta, 1e-6)`, `max(Ra_rh_raw, 1e-30)`, the elementwise maxima in $w_\text{interior}$ and $w_\text{hot,sol}$, the temperature-gradient floor `dTdr_floor` with `max(dr_min, 1.0)` in $\delta_\text{rh}$, the mantle-thickness cap on $\delta_\text{rh}$, the `COHESION_FLOOR` clamp on yield stress, `if_lid` (`where(w_lid_present >= 1 - 1e-12)`), `d_geom`, the max in `d_lid_raw`, `max(w_conv - w_conv_0, 0)`, `f_int_scale`, the $v_{m,\text{raw}}$ branches, and the $10^{-30}$ denominators (while the 5-node local interior RMS $|F_\text{tot}|$ floor is continuous in every $F_k$).

### 4.1 Heating Mode and Evolutionary Context

The $2/3$ velocity scaling law applies to bottom-heated stagnant-lid convection (Dumoulin et al., 1999; Foley and Bercovici, 2014, p. 587). In planetary evolution, cooling terrestrial planets start with a superheated core, creating a large core-mantle boundary temperature difference that drives strong basal heating during early post-magma-ocean evolution (Thiriet et al., 2019, pp. 139-140, sec. 1).

As the core cools and radiogenic heating decays, the mantle heating regime transitions from bottom-dominated to predominantly internal heating (Thiriet et al., 2019, p. 140). For purely internally heated convection, Solomatov and Moresi (2000, p. 21801, eq. 13; p. 21804, eq. 28; p. 21805, Table 8) obtain a $1/2$ velocity scaling exponent:

$$
v_m = a_u \frac{\kappa}{d} \left( \frac{\mathrm{Ra}_i a_\text{rh}}{\theta} \right)^{1/2}
$$

with $a_u = 0.385 \pm 0.003$ for Newtonian rheology ($n = 1$) and $a_\text{rh} = 2.4$. In both the bottom-heated and internally heated scalings, the temperature contrast $\Delta T$ enters the definitions of Rayleigh number and Frank-Kamenetskii parameter symmetrically, cancelling in the ratio $\mathrm{Ra} / \theta$. For the Earth reference state ($\rho_i = 3300\text{ kg m}^{-3}$, $g_i = 9.81\text{ m s}^{-2}$, $\alpha_i = 3 \times 10^{-5}\text{ K}^{-1}$, $T_i = 1600\text{ K}$, $T_\text{surf} = 300\text{ K}$, $d = 2.89 \times 10^6\text{ m}$, $\kappa_i = 10^{-6}\text{ m}^2\text{ s}^{-1}$, $H = 300\text{ kJ mol}^{-1}$, $\mu_i = 10^{20}\text{ Pa s}$, giving $\theta = 18.32$ and $\mathrm{Ra}_i = 3.05 \times 10^8$), the internally heated scaling predicts $v_m = 8.42 \times 10^{-10}\text{ m s}^{-1}$ and convective driving shear stress $\tau_d = 2 \mu_i v_m / d \approx 0.058\text{ MPa}$ at $a_\text{rh} = 2.4$ (or $v_m = 5.43 \times 10^{-10}\text{ m s}^{-1}$ and $\tau_d \approx 0.038\text{ MPa}$ with $a_\text{rh} = 1.0$), approximately four times lower than the bottom-heated estimate ($v_m = 3.36 \times 10^{-9}\text{ m s}^{-1}$, $\tau_d \approx 0.232\text{ MPa}$ at $C_4 = 0.125$, $a_\text{rh} = 1.3$). Consequently, evaluating $\tau_d$ with the bottom-heated $2/3$ scaling represents an upper estimate on convective shear stress during later, internally heated epochs.

### 4.2 Validated Parameter Ranges and Mushy Extrapolation

Foley and Bercovici (2014, sec. 4.2, p. 589; Table 2, p. 593; Table 3, p. 594; Figs. 8-9) calibrate and validate the scaling laws over:

$$
\mathrm{Ra}_0 \in [10^5, \; 5 \times 10^7]
$$

In a partially molten or mushy mantle where the two-phase log-viscosity blend (`viscosity_solid` $= 10^{21}\text{ Pa s}$, `viscosity_liquid` $= 10^2\text{ Pa s}$ in standard two-phase configurations) reduces the interior viscosity to $\mu_i \sim 10^2\text{ to }10^7\text{ Pa s}$ (or $\sim 10^3\text{ to }10^9\text{ Pa s}$ near the rheological transition), evaluating $\mathrm{Ra}_\text{eff}$ with the Earth reference inputs above ($\rho_i g_i \alpha_i (T_i - T_\text{surf}) d^3 / \kappa_i = 3.05 \times 10^{28}\text{ Pa s}$ at $\Delta T = 1300\text{ K}$) gives $\mathrm{Ra}_\text{eff} = 3.05 \times 10^{21}\text{ to }3.05 \times 10^{26}$ (or $\sim 10^{19}\text{ to }10^{25}$ for $\mu_i \sim 10^3\text{ to }10^9\text{ Pa s}$), well above the numerical calibration domain of solid-state boundary-layer theory. Applying the scaling to mushy interiors represents an extrapolation, and Solomatov and Moresi (2000, sec. 5.4, p. 21802) note that the velocity scaling law can change at very high Rayleigh numbers when convective plumes disconnect from the lid base.

### 4.3 Rheological Parameter Calibration

In the velocity scaling, $a_\text{rh} = 1.3$ and $C_4 = 0.125$ are the empirical calibration constants fitted jointly by Foley and Bercovici (2014, sec. 4.2, p. 589) for their theory-curve constant-healing ($E_h = 0$) damage models. Every numerical model in Foley and Bercovici (2014) includes dynamic grain damage; the paper provides no pure damage-free numerical fit. The calibrated constant $a_\text{rh} = 1.3$ is lower than the standard stagnant-lid value ($a_\text{rh} \approx 2$; Solomatov and Moresi, 2000; Korenaga, 2009, cited in Foley and Bercovici, 2014, p. 586) because less effective damage means less of the high-viscosity lid participates in convection (Foley and Bercovici, 2014, p. 586). For temperature-dependent healing, Foley and Bercovici (2014, sec. 5.2, pp. 594-595) find $a_\text{rh} \approx 1.82$.

`compute_stagnant_lid_state` adopts $C_4 = 0.125$ and $a_\text{rh} = 1.3$ as a unified parameter package calibrated against sub-lid stresses. In the Foley and Bercovici (2014) theory, grain damage enters solely through the effective viscosity $\mu_\text{eff} = \mu_i (A_i / A_0)^{-m}$, which reduces to the single-phase or mixture viscosity when damage is absent ($A_i = A_0$). The convective interior velocity scaling thus holds without damage mechanics. The sensitivity to $a_\text{rh}$ is moderate: adopting $a_\text{rh} = 2.0$ instead of $1.3$ increases convective velocity $v_m$ and driving stress $\tau_d$ by a factor of $(2.0 / 1.3)^{2/3} \approx 1.33$.

This sublayer parameter $a_\text{rh} = 1.3$ enters the interior convective velocity $v_m$ and differs on purpose from the configured `lid_contrast_coeff` ($a = 2.2$; Solomatov, 1995; Solomatov and Moresi, 2000, sec. 5.1, pp. 21799-21800), which defines the thermal boundary isotherm of the rigid lid ($T_{\text{lid,iso}} = T_i - a \Delta T_\text{rh}$).

### 4.4 Diagnostic Quantities and Force Balance

`compute_stagnant_lid_state` evaluates the rheological sublayer buoyancy stress as a diagnostic entry in `lid_state`:

$$
\tau_\text{buoy} = \rho_i g_i \alpha_i \Delta T_\text{rh} \delta_\text{rh} w_\text{active}
$$

where $\delta_\text{rh}$ is the rheological sublayer thickness and $\Delta T_\text{rh} = R T_i^2 / H(P_{T_i})$. The sublayer buoyancy stress $\tau_\text{buoy}$ and the force-balance ratio $\tau_d / \tau_\text{buoy}$ are returned in the stagnant-lid diagnostic state dictionary `lid_state`. For the Earth reference state ($\rho = 3300\text{ kg m}^{-3}$, $g = 9.81\text{ m s}^{-2}$, $\alpha = 3 \times 10^{-5}\text{ K}^{-1}$, $T_i = 1600\text{ K}$, $T_\text{surf} = 300\text{ K}$, $d = 2.89 \times 10^6\text{ m}$, $\kappa = 10^{-6}\text{ m}^2\text{ s}^{-1}$, $H = 300\text{ kJ mol}^{-1}$, $\eta_i = 10^{20}\text{ Pa s}$, $\theta \approx 18.32$, $\Delta T_\text{rh} \approx 71.0\text{ K}$, $\tau_d \approx 0.232\text{ MPa}$), a 50-node piecewise-linear reference profile (with resolved lid thickness $d_\text{lid} = 378.4\text{ km}$) gives $\delta_\text{rh} \approx 30.8\text{ km}$ ($30791\text{ m}$), $\tau_\text{buoy} \approx 2.12\text{ MPa}$, and $\tau_d / \tau_\text{buoy} \approx 0.1095$ because `xp.gradient(T, r)` spans the isothermal-to-conductive corner; the boundary-layer scaling values across the 7 conductive intervals ($d_\text{lid,cond} = 7 \times 58.98\text{ km} = 412.9\text{ km}$, $\delta_\text{rh} = d_\text{lid,cond} / \theta \approx 22.5\text{ km}$) are $\tau_\text{buoy} \approx 1.55\text{ MPa}$ and $\tau_d / \tau_\text{buoy} \approx 0.150$. In addition, the shifted soft-maximum convective velocity $v_i$, evaluated over the upper mantle using `SOFTMAX_VELOCITY_SCALE` ($3.17 \times 10^{-12}\text{ m s}^{-1}$), is returned as a diagnostic quantity in `lid_state`.

## 5. Minimum Yield Closure

The effective solid viscosity combines diffusion creep and plastic yielding through the smooth minimum formulation implemented in `aragog.rheology_lid.compute_effective_viscosity` (Moresi and Solomatov, 1998, p. 672, eqs. 13-14, doi:10.1046/j.1365-246X.1998.00531.x; Tackley, 2000, p. 8, eq. 8, doi:10.1029/2000GC000036):

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
f(\phi) = \frac{1}{2} \left( 1 + \tanh\left( \frac{\phi - \phi_\text{rheo}}{\phi_\text{width}} \right) \right), \quad f_\text{zero}(\phi) = \mathrm{clip}\left( \frac{f(\phi) - f(0)}{\max(f(1) - f(0), 10^{-30})}, \; 0, \; 1 \right)
$$

$$
\kappa_{h,\text{floor,eff}} = \kappa_{h,\text{floor}} \cdot f_\text{floor}(\phi) \cdot I_\text{conv}, \quad \kappa_h = \max\big(\kappa_{h,\text{MLT}}, \; \kappa_{h,\text{floor,eff}}\big)
$$

where $f_\text{floor}(\phi) = f_\text{zero}(\phi)$ when `rheology.enabled` is `True` and $f_\text{floor}(\phi) = f(\phi)$ when `rheology.enabled` is `False`, and $I_\text{conv} = 1$ in cells with $\partial S / \partial r < D_\text{thr}$ (convectively unstable when $D_\text{thr} = 0$) and $0$ in cells with $\partial S / \partial r \ge D_\text{thr}$ (stably stratified when $D_\text{thr} = 0$).

In standalone Aragog configurations, the transition width defaults to `rheological_transition_width = 0.15` ($\phi_\text{width} = 0.15$ at $\phi_\text{rheo} = 0.4$, where $f(0) = 0.004805$), and `kappah_floor` defaults to $10.0\text{ m}^2\text{ s}^{-1}$ on `EnergyConfig` and `_EnergyParameters` (while the direct `PhaseParams.__init__` fallback in `aragog.jax.phase` is $0.0\text{ m}^2\text{ s}^{-1}$). In coupled PROTEUS runs (`proteus.config._interior.Aragog`), the defaults are `rfront_loc = 0.4` ($\phi_\text{rheo} = 0.4$), `rfront_wid = 0.2` ($\phi_\text{width} = 0.2$, where $f(0) = 0.018$), and `kappah_floor = 10.0` ($\kappa_{h,\text{floor}} = 10.0\text{ m}^2\text{ s}^{-1}$). With the unanchored ramp (`rheology.enabled = False`), at $\kappa_{h,\text{floor}} = 10.0\text{ m}^2\text{ s}^{-1}$ and zero melt fraction ($\phi = 0$), $\kappa_{h,\text{floor,eff}}$ evaluates to $0.048\text{ m}^2\text{ s}^{-1}$ in solid cells with $\partial S / \partial r < D_\text{thr}$ at the standalone width $\phi_\text{width} = 0.15$ and $0.18\text{ m}^2\text{ s}^{-1}$ at the PROTEUS width $\phi_\text{width} = 0.2$, exceeding molecular thermal diffusivity ($\kappa \sim 10^{-6}\text{ m}^2\text{ s}^{-1}$) by four to five orders of magnitude. When `rheology.enabled` is `True`, the zero-anchored ramp $f_\text{zero}(\phi)$ satisfies $f_\text{zero}(0) = 0.0$ and $f_\text{zero}(1) = 1.0$ identically at both $\phi_\text{width} = 0.15$ ($f(1) - f(0) = 0.99486$) and $\phi_\text{width} = 0.20$ ($f(1) - f(0) = 0.97954$), so $\kappa_{h,\text{floor,eff}}(\phi = 0) = 0.0\text{ m}^2\text{ s}^{-1}$ in solid cells while preserving the floor in molten layers ($\kappa_{h,\text{floor,eff}}(\phi = 1) = 10.0\text{ m}^2\text{ s}^{-1}$). In cells with $\partial S / \partial r \ge D_\text{thr}$ (stably stratified when $D_\text{thr} = 0$), $I_\text{conv} = 0$ and $\kappa_{h,\text{floor,eff}} = 0$.

When `rheology.enabled` is `True` with a tabular entropy equation of state (`const_properties = False`), `EntropySolver._check_rheology_invariants` requires every accepted state's staggered entropy $S$ to remain inside the loaded table bounds $[S_\text{min}, S_\text{max}]$ (raising `RuntimeError` naming the node index, time $t$, and entropy $S$ if any accepted state leaves the table). Because the zero-anchored floor removes artificial convective heat transport across solid stagnant lids ($\kappa_{h,\text{floor,eff}}(\phi = 0) = 0$), the surface cell cools toward the radiative boundary equilibrium temperature $T_\text{eqm}$. With the standard CI test table (`test-data-v1`, where $S \in [-1606.28, 19734.29]\text{ J kg}^{-1}\text{ K}^{-1}$ and the lowest table temperatures are $T_\text{min}(P = 10^5\text{ Pa}) = 320.23\text{ K}$ at the surface and $T_\text{min}(P = 0.218\text{ GPa}) = 320.82\text{ K}$ at the top staggered cell of `cfg/abe_solid.toml`), the validity domain requires the surface temperature to remain above about $320\text{ K}$ and the radiative equilibrium temperature to satisfy $T_\text{eqm} \ge 325\text{ K}$ (at $T_\text{eqm} = 325\text{ K}$ and $350\text{ K}$, `cfg/abe_solid.toml` remains strictly inside the table with $\min S_\text{stag} = -392.51\text{ J kg}^{-1}\text{ K}^{-1}$ and $-326.01\text{ J kg}^{-1}\text{ K}^{-1}$, whereas at $T_\text{eqm} = 273\text{ K}$ the top cell crosses below $S_\text{min} = -1606.28\text{ J kg}^{-1}\text{ K}^{-1}$ at about $17\text{ to }18\text{ kyr}$, specifically $17.44\text{ kyr}$ in the ODE energy trace and $18.08\text{ kyr}$ at the accepted segment endpoint, and is stopped by `_check_rheology_invariants`). For simulations whose surface cools below $320\text{ K}$ (such as $T_\text{eqm} = 273\text{ K}$), the Zalmoxis low-temperature PALEOS solid extension (`load_paleos_with_lowt_extension` in `zalmoxis.eos_export` and `tools/converters/paleos_lowt_extension.py`, enabled in PROTEUS via `interior_struct.zalmoxis.lowt_extension = true` with `paleos_mgsio3_tables_pt_proteus_solid_lowT.dat`) extends the solid EOS table down to $50\text{ K}$ for pressures up to $12\text{ GPa}$.

## 7. Stress Closure Modes

The geometry of the strain rate closure in `compute_effective_viscosity` is controlled by `stress_closure_mode`, and cross-mode keyword arguments are rejected (negative values of `tau_y_lid`, `tau_y`, or `tau_y_profile`, including $-\infty$, raise `ValueError` on the eager NumPy path and return `NaN` on the JAX path, whereas $0$ is floored at `COHESION_FLOOR = 1.0e-6 Pa` and $+\infty$ is accepted):

- `lid`:
    Boundary-layer stress closure. Requires `eta_i`, `tau_d`, and `tau_y_lid` (rejecting local-mode arguments `tau_y`, `tau_y_profile`, `strain_rate`, `mixing_length`, and `unyielded_velocity`). The driving stress $\tau_d = (2 \mu_i v_m / d) w_\text{active}$ is evaluated from the convecting interior scaling, and plastic yield viscosity $\eta_\text{yield} = \eta_i (\tau_{y,\text{lid}} / \tau_d)$ mobilises the lid nodes when $\tau_d > \tau_{y,\text{lid}}$.

- `local`:
    Local mixing-length strain rate closure (rejecting lid-mode arguments `eta_i`, `tau_d`, `tau_y_lid`, and `w_lid`). The strain rate is evaluated directly from local cell velocity and mixing length using the second invariant convention:

    $$
    \dot{\epsilon}_\text{local}(r) = \frac{|v_\text{unyielded}(r)|}{2 \max(l(r), 10^{-15}\text{ m})}
    $$

    and $\log_{10}\eta_\text{eff}(r)$ applies a smooth softplus cap of width `LOCAL_CAP_LOG10_WIDTH = 0.01` between $\log_{10}\eta_\text{diff}(r)$ and $\log_{10}\big(\max(\tau_y(r), 10^{-6}\text{ Pa}) / (2 \sqrt{\dot{\epsilon}_\text{local}(r)^2 + 10^{-200}})\big)$ (returning $\eta_\text{diff}(r)$ when $\tau_y(r) = +\infty$ or $\dot{\epsilon}_\text{local}(r) \le 0$). In the two-pass mixing-length solver, the pass-1 viscous velocity is $v_{\text{visc},1} = \rho g \alpha T (-\partial S / \partial r)_\text{eff} l_v^3 / (18 \eta_\text{diff})$ (Bower et al., 2018, p. 4, eq. 5a), so the pass-1 viscous shear stress $\tau_1 = 2 \eta_\text{diff} \dot{\epsilon}_\text{local} = \eta_\text{diff} |v_{\text{visc},1}| / l_v = \rho g \alpha T (-\partial S / \partial r)_\text{eff} l_v^2 / (18 C_p) = \tau_{\text{buoy},1}$ depends on the local buoyancy driving force and mixing length rather than on $\eta_\text{diff}$. Where $\tau_{\text{buoy},1} > \tau_y(r)$, pass 1 reduces $\eta_\text{eff}(r) \approx \eta_\text{diff}(r) \tau_y(r) / \tau_{\text{buoy},1}$, and pass 2 scales the viscous velocity $v_{\text{visc},2}$ and thermal eddy diffusivity $\kappa_h$ upward by the factor $\tau_{\text{buoy},1} / \tau_y(r)$ while keeping $\tau_2 = \eta_\text{eff} |v_{\text{visc},2}| / l_v = \tau_{\text{buoy},1}$ in the viscous regime ($\mathrm{Re} \ll \mathrm{Re}_\text{crit}$). Thus in `'local'` mode yielding scales the convective velocity by $\tau_{\text{buoy},1} / \tau_y$ and is not a stress cap.

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
| `activation_energy` | J/mol | `None` |
| `activation_volume` | m^3/mol | `0.0` |
| `activation_volume_decay_pressure` | Pa | `inf` |

## 10. Stagnant-Lid Trajectory and Yield-Thinning Comparison Against the 0-D Reference Model

### How closely does the 1-D stagnant-lid solver follow a 0-D boundary-layer cooling model and plastic yield-thinning law?

In planetary thermal-history models, a simplified zero-dimensional (0-D) energy-balance ordinary differential equation (ODE) is often used to estimate interior cooling and stagnant-lid growth.
We compare Aragog's one-dimensional (1-D) radial finite-volume solver (`EntropySolver`) against the standalone 0-D stagnant-lid ODE in `tests/reference/stagnant_lid_0d.py` over $4.50\text{ Gyr}$ of secular cooling and across steady-state plastic yield-thinning sweeps (`tests/reference/ssc_0d_reference_benchmarks.json`, regenerated with `python tools/verification/generate_ssc_0d_reference_benchmarks.py --full` and verified with `--check`).
The 1-D solver fails the pre-run 0-D reference tolerances on both the $4.50\text{ Gyr}$ cooling trajectory ($5\%$ on interior temperature $T_i$, $22\%$ on lid thickness $d_\text{lid}$) and the steady-state yield-ceiling thinning ratio ($11\%$ at yield stress $\tau_{y,0} = 0.5\,\tau_{d,\text{lid}}$), consistent with differences between 1-D spatially resolved radial heat transport and the 0-D bulk energy balance and with plastic yielding mobilising convective transport only below the top two radial nodes.

**Why this check exists.**
If the 1-D solver's stagnant-lid geometry, convective mixing-length slope, or plastic yield-stress blend were miswired, the model would predict the wrong mantle cooling rate or silently ignore lithospheric yielding.
This test catches three classes of code faults, each verified by a mutation or perturbation probe:

- Loss of plastic yield mobilisation inside the stagnant lid: mutating the unyielded bypass in `src/aragog/rheology_lid.py` (`unyielded = (td <= 0.0) | inf_ty | True`) causes the live yielded steady-state evaluation (`E300` at $N = 100$ and $\tau_{y,0} = 0.5\,\tau_{d,\text{lid}}$ with fixed internal heating) to drop from $46$ yielding basic nodes to $0$ (failing the $\ge 40$ node assertion) and raises the steady-state entropy residual $\max |\mathrm{d}S/\mathrm{d}t|$ to $2.356 \times 10^{-4}\text{ J kg}^{-1}\text{ K}^{-1}\text{ yr}^{-1}$, above the $10^{-9}\text{ J kg}^{-1}\text{ K}^{-1}\text{ yr}^{-1}$ ceiling.
- Perturbations to the 1-D mixing-length top-slope asymptote or rheological lid-base contrast: perturbing $(s_{\text{top},\infty}, a_\text{lid})$ from $(0.25, 2.2)$ to $(0.26, 2.3)$ in the live $4.50\text{ Gyr}$ 1-D right-hand side evaluation shifts the Pass-1 isotherm lid thickness from $540.90\text{ km}$ to $534.13\text{ km}$ ($1.252\%$ relative), shifts the peak convective heat flux by $16.74\%$, and shifts the entropy time derivative by up to $161.6\%$, failing the $10^{-7}$ pinned tolerance.
- Perturbations to the 0-D reference Nusselt prefactor or parameter defaults: perturbing $a_\text{Nu}$ by $1\%$ ($1.01\,a_\text{Nu}$) shifts the 0-D interior temperature trajectory by up to $9.217 \times 10^{-4}$ relative (failing the $10^{-7}$ pinned tolerance), while explicit standalone default constants in `tests/reference/stagnant_lid_0d.py` are asserted by parity tests to match `SolidRheologyParams` defaults in `src/aragog/rheology.py`, forcing lockstep between the 0-D reference and solver defaults.
- Limit of independence: while `stagnant_lid_0d.py` integrates a separate 0-D ODE without calling `EntropySolver`, both the 0-D Nusselt prefactor $a_\text{Nu} = 0.5954$ (`0.5954345503010264`) and Aragog's 1-D mixing-length top-slope calibration $(s_{\text{top},\infty}, c_\theta)$ ([Viscosity-Contrast-Dependent Mixing-Length Slope](#viscosity-contrast-dependent-mixing-length-slope)) rest on the same 12 Newtonian ($n = 1$) rows of Korenaga (2009, p. 157, Table 1), so the 0-D reference and the calibrated 1-D closure share that calibration dataset and are not independent of it; a systematic bias in those 12 benchmark rows affects both models and is not caught by comparing them.

**What it tests.**
Here:
- $T_i$ is the sub-lid interior temperature in kelvins ($\text{K}$).
- $d_\text{lid}$ is the conductive stagnant-lid thickness in metres or kilometres ($\text{m}$ or $\text{km}$).
- $F_\text{top}$ is the surface conductive heat flux in watts per square metre ($\text{W m}^{-2}$).
- $\mathrm{Nu}$ is the dimensionless Nusselt number, the ratio of total surface heat loss to basal-to-surface conductive heat loss.
- $\mathrm{Ra}_i$ is the dimensionless internal Rayleigh number evaluated at the interior viscosity $\eta(T_i, P_i)$.
- $\theta$ is the dimensionless Frank-Kamenetskii viscosity-contrast parameter across the temperature drop $T_i - T_\text{surf}$.
- $\tau_{d,\text{lid}}$ is the convective driving stress at the lid base in pascals ($\text{Pa}$), and $\tau_{y,0}$ is the surface yield-stress cohesion ceiling in pascals ($\text{Pa}$).
- $R_\text{thin} = d_\text{lid}(\tau_{y,0}) / d_\text{lid}(\text{unyielded})$ is the dimensionless lid-thinning ratio when plastic yielding is active.

In the 0-D reference ODE, the stagnant-lid Nusselt prefactor $a_\text{Nu} = 0.5954$ (`0.5954345503010264`) is the arithmetic mean of $a_\text{Nu} = \mathrm{Nu}_\text{ref}\,\theta^{4/3}\,\mathrm{Ra}_i^{-1/3}$ over the 12 Newtonian Arrhenius 2-D stagnant-lid runs of Korenaga (2009, p. 157, Table 1; $\min = 0.5025$, $\max = 0.6862$, standard deviation $0.0651$; `a_nu_provenance` in `tests/reference/ssc_0d_reference_benchmarks.json`).
To provide a reference prefactor that does not use the Korenaga (2009) Table 1 rows, the test also evaluates the 0-D ODE with the published prefactor $a_\text{Nu} = 0.53$ of Solomatov and Moresi (2000, p. 21803, Table 6; p. 21804, eq. 23), which gives $T_{i,\text{0D}}(4.50\text{ Gyr}) = 1329.8\text{ K}$, $d_{\text{lid,0D}}(4.50\text{ Gyr}) = 899.2\text{ km}$, maximum $T_i$ relative deviation $7.78\%$ at $4.50\text{ Gyr}$ ($6.52\%$ on its strict pre-clamp window $t < 3.6958\text{ Gyr}$), and post-relaxation $d_\text{lid}$ relative deviation $46.97\%$ pre-clamp and full-run (for the printed prefactor $a_\text{Nu} = 0.57$ of Korenaga, 2009, p. 158, eq. 29; p. 159, Fig. 4a, $T_{i,\text{0D}}(4.50\text{ Gyr}) = 1320.9\text{ K}$ and the maximum $T_i$ relative deviation is $7.16\%$ at $4.50\text{ Gyr}$ and $5.89\%$ on its strict pre-clamp window $t < 3.6958\text{ Gyr}$, with $46.94\%$ pre-clamp on $d_\text{lid}$).
Both cases run with constant material properties (`eos_method = 1`, `const_properties = True`, no equation-of-state table read).
In `--full` mode, `tools/verification/generate_ssc_0d_reference_benchmarks.py` temporarily replaces `EntropySolver._solve_cvode` with an unbounded-step wrapper (`max_step = np.inf`) to integrate the multi-Gyr trajectories without stopping at the $100\text{ yr}$ phase-boundary step cap, because the solver caps steps at $100\text{ yr}$ after solidification.
The steady-state `E300` runs prescribe the surface temperature (`outer_boundary_condition = 5`): the solver sets the surface node to the entropy whose temperature is $T_\text{surf} = 273.0\text{ K}$ and takes the surface gradient from the half-cell temperature gradient, so the top conductive heat flux is $F_\text{top} = k (T_\text{cell} - T_\text{surf}) / (\Delta r / 2)$ and the stagnant-lid closure reads $T_\text{surf}$; `test_rheology_verification_smoke.py` evaluates the stored `E300` steady states with `outer_boundary_condition = 4` using the stored surface flux.
By contrast, the $400\text{ Myr}$ and $4.50\text{ Gyr}$ secular-cooling trajectories run with `outer_boundary_condition = 1` (grey-body radiative boundary condition with emissivity $\varepsilon = 1$ and equilibrium temperature $T_\text{eq} = 300\text{ K}$) in a single continuous integration over $[0, t_\text{end}]$ and do not use the BC 5 patch.
In `--check` mode, `generate_ssc_0d_reference_benchmarks.py` recomputes live all four 0-D ODE trajectories, `a_nu_provenance`, `yield_sweep_0d_at_1600K`, the 0-D fixed-$T_i$ and fixed-$H$ yield-thinning ratios, the live 1-D `segment_25Myr_spread_5starts` trajectory integration, the live 1-D `10Myr` yield integration for `unyielded_500MPa` and `below_0p5x_tau_d_lid`, and live 1-D `EntropySolver._dSdt_single` evaluations on the four `endpoint_states` profiles, the five `0Myr` `live_yield_states` profiles, and the three `E300` `steady_yield_thinning` profiles, while checking that the stored five-start spread statistics in `segment_25Myr_spread_5starts` satisfy $0 < \text{spread} < 10^{-7}$ (bounded rather than recomputed from the five perturbed starts) and checking the remaining stored 1-D trajectory points, the `5Myr`, `20Myr`, `30Myr`, and `yield_sweep_1d_equal_Ti` yield horizons, and the remaining `E300`, `E340`, and `E270` steady-state rows for internal algebraic consistency.

1. **Secular cooling trajectory over $4.5\text{ Gyr}$ and 0-D validity window:**
   We cool the constant-property spherical mantle configuration of `cfg/abe_solid.toml` ($r_\text{out} = 6371\text{ km}$, $r_\text{in} = 5371\text{ km}$, $N = 100$, density $\rho = 4200\text{ kg m}^{-3}$, specific heat capacity $c_p = 1000\text{ J kg}^{-1}\text{ K}^{-1}$, thermal expansivity $\alpha = 1 \times 10^{-5}\text{ K}^{-1}$, thermal conductivity $k = 4.0\text{ W m}^{-1}\text{ K}^{-1}$, reference interior pressure $P_i = 4.1202\text{ GPa}$, reference solid viscosity $\eta_\text{solid} = 10^{18}\text{ Pa s}$, homologous exponent $g_\text{hom} = 25.0$, surface temperature $T_\text{surf} = 300\text{ K}$, basal heat flux $F_\text{bot} = 0$, unyielded yield stress $\tau_y = 500\text{ MPa}$) from $T_\text{init} = 1800\text{ K}$ to $4.50\text{ Gyr}$ at $(s_{\text{top},\infty}, c_\theta) = (0.25, 0.25)$.
   The 0-D reference ODE integrates $-c_\text{eff}\,\mathrm{d}T_i/\mathrm{d}t = 4\pi r_\text{out}^2 F_\text{top}$ with bulk heat capacity $c_\text{eff} = m_\text{mantle} c_p$ over the entire spherical shell (treating the whole mantle as cooling at the single rate $\mathrm{d}T_i/\mathrm{d}t$ rather than evolving a spatially resolved radial temperature profile $T(r, t)$ across the conductive lid and convecting interior) and clamps $\mathrm{Nu} = \max(\mathrm{Nu}_\text{unyielded}, 1.0)$, while scaling the spherical surface heat flux by the conductive shell factor $(r_\text{out} - \delta_{l,\text{eff}}) / r_\text{out}$ evaluated at the total thermal boundary-layer thickness $\delta_{l,\text{eff}} = d_\text{lid,eff} / (1 - a_\text{lid}/\theta)$.
   Along the 0-D cooling trajectory ($a_\text{Nu} = 0.5954$), the unyielded Nusselt number $\mathrm{Nu}_\text{unyielded}$ decreases from $16.328$ at $t = 0\text{ yr}$ below $1.0$ at $t = 3.6958\text{ Gyr}$ ($\mathrm{Nu}_\text{unyielded}(3.6324\text{ Gyr}) = 1.0074$ and $\mathrm{Nu}_\text{unyielded}(3.6958\text{ Gyr}) = 0.9917$, decaying to $0.8121$ at $4.50\text{ Gyr}$).
   Once the $\mathrm{Nu} = 1.0$ clamp activates at $3.6958\text{ Gyr}$, the reported 0-D lid thickness follows $d_\text{lid,0D} = (1 - a_\text{lid}/\theta)\,D / \mathrm{Nu} = (1 - 2.2/\theta)\,D$, creeping upward from $898.62\text{ km}$ at $3.6958\text{ Gyr}$ ($\theta = 21.70$; up from $891.89\text{ km}$ at $3.6324\text{ Gyr}$) to $899.94\text{ km}$ ($899944.45\text{ m}$) at $4.50\text{ Gyr}$ ($\theta = 21.99$).
   Within the strict $\mathrm{Nu}_\text{unyielded} > 1$ validity window ($t < 3.6958\text{ Gyr}$), the post-relaxation ($t \ge 25\text{ Myr}$) isotherm lid-thickness relative deviation reaches $46.93\%$ ($0.4693$ at $3.6324\text{ Gyr}$, crossing the $22\%$ bound at $413.0\text{ Myr}$; $46.73\%$ at the $3.6958\text{ Gyr}$ clamp sample, $21.61\%$ over $25\text{ to }400\text{ Myr}$, and $100.0\%$ at $t = 0\text{ yr}$ before boundary-layer relaxation).
   Over the same strict validity window ($t < 3.6958\text{ Gyr}$), the interior temperature relative deviation $|T_{i,\text{1D}} - T_{i,\text{0D}}| / T_{i,\text{0D}}$ crosses the $5\%$ bound at $3.2692\text{ Gyr}$ by linear interpolation ($3.3234\text{ Gyr}$ at the first grid sample above $5\%$, where the error is $5.07\%$).
   It reaches $5.51\%$ inside the valid window ($0.0551$ at $3.6324\text{ Gyr}$; $5.60\%$ at the $3.6958\text{ Gyr}$ clamp sample and $6.78\%$ at $4.50\text{ Gyr}$, where $T_{i,\text{1D}} = 1226.4\text{ K}$ versus $T_{i,\text{0D}} = 1315.6\text{ K}$; within $3.61\%$ over $25\text{ to }400\text{ Myr}$; at the default $(0.22, 0.15)$ the full-run deviations are $5.97\%$ on $T_i$ and $45.28\%$ post-relaxation on $d_\text{lid}$).
   At all 128 post-initial samples, the 1-D radial solver interior temperature $T_{i,\text{1D}}(t)$ is cooler than $T_{i,\text{0D}}(t)$ and the 1-D Pass-1 isotherm lid $d_{\text{lid,1D}}(t)$ is thinner than $d_{\text{lid,0D}}(t)$ ($540.90\text{ km}$ versus $899.94\text{ km}$ at $4.50\text{ Gyr}$, and $478.69\text{ km}$ versus $898.62\text{ km}$ at $3.6958\text{ Gyr}$), with higher 1-D surface heat flux ($F_{\text{top,1D}} / F_{\text{top,0D}} = 2.207$ at $4.50\text{ Gyr}$, $7.56\text{ mW m}^{-2}$ versus $3.42\text{ mW m}^{-2}$).
   Whether this cooling offset is primarily driven by the spatially resolved radial temperature profile across the thickening spherical shell or by the 1-D spherical mixing-length flux formulation has not been isolated by a separate ablation test and remains an untested hypothesis.
2. **Steady-state yield-ceiling lid thinning:**
   We evaluate the planar internally heated steady-state configuration (`E300` at $N = 100$ and $(s_{\text{top},\infty}, c_\theta) = (0.25, 0.25)$, where the unyielded steady state has $T_i = 1666.43\text{ K}$, Pass-1 isotherm $d_\text{lid} = 282.37\text{ km}$, Pass-2 fixed-isotherm depth $435.21\text{ km}$, $F_\text{top} = 2078.91\text{ mW m}^{-2}$, and lid driving stress $\tau_{d,\text{lid}} = 1.956\text{ MPa}$).
   The 0-D boundary-layer scaling predicts lid-thinning ratios $R_\text{thin,0D} = (\tau_{y,0} / \tau_{d,\text{lid}})^{1/3}$ of $1.0000$, $0.7937$, $0.5848$, and $0.4642$ ($1.00$, $0.79$, $0.58$, and $0.46$) for surface cohesion $\tau_{y,0} / \tau_{d,\text{lid}} \in \{2.0, 0.5, 0.2, 0.1\}$.
   The rheological isotherm lid thickness `d_lid` fails this thinning check.
   For $\tau_{y,0} / \tau_{d,\text{lid}} \in \{0.5, 0.2, 0.1\}$, the Pass-1 isotherm ratio $R_\text{thin,iso} = d_\text{lid}(\tau_{y,0}) / d_\text{lid}(\text{unyielded})$ increases to $1.5257$, $1.7395$, and $1.9116$ ($1.53$, $1.74$, and $1.91$, exceeding the 0-D targets by $+92.2\%$, $+197.4\%$, and $+311.8\%$) at fixed internal heating (as enhanced surface heat loss cools the interior to $1557.05\text{ K}$, $1517.88\text{ K}$, and $1486.01\text{ K}$).
   When internal heating is adjusted to hold $T_i = 1666.43\text{ K}$ fixed ($F_\text{top} = 3407.08$, $4168.05$, and $4941.39\text{ mW m}^{-2}$), the Pass-1 isotherm ratio equals $1.0465$, $1.0549$, and $1.0578$ ($1.05$, $1.05$, and $1.06$, exceeding the 0-D targets by $+31.9\%$, $+80.4\%$, and $+127.9\%$).
   In the two-pass solver update, the first pass locates $r_\text{lid,base}$ from the unyielded convective flux fraction and the rheological isotherm $T_\text{lid,iso} = T_i - 2.2\,\Delta T_\text{rh}$.
   The second pass lowers $\eta_\text{eff}$ within the lid and mobilises convective heat transport below $20.20\text{ km}$ depth ($20202.02\text{ m} = 2 \times 10^6\text{ m} / 99$, equal to $2$ radial node spacings on the $N = 100$ mesh).
   Meanwhile, the Pass-2 fixed-isotherm depth reaches $659.29\text{ km}$, $491.18\text{ km}$, and $468.59\text{ km}$ at $\tau_{y,0} / \tau_{d,\text{lid}} \in \{0.5, 0.2, 0.1\}$ at fixed internal heating (ratios $1.5149$, $1.1286$, and $1.0767$ relative to the $435.21\text{ km}$ unyielded Pass-2 baseline) and $392.63\text{ km}$, $225.93\text{ km}$, and $179.61\text{ km}$ at matched $T_i = 1666.43\text{ K}$ (ratios $0.9022$, $0.5191$, and $0.4127$ relative to the $435.21\text{ km}$ unyielded Pass-2 baseline).
   Evaluated on the post-hoc conductive lid-thickness diagnostic $d_\text{lid,cond} = (1 - 2.2 / \theta) k (T_i - T_\text{surf}) / F_\text{top}$ (a diagnostic chosen after the result, not the solver `d_lid` output), the ratios for $\tau_{y,0} / \tau_{d,\text{lid}} \in \{0.5, 0.2, 0.1\}$ are $0.9282$, $0.9022$, and $0.8809$ ($0.93$, $0.90$, and $0.88$) at fixed internal heating, which deviate from the 0-D targets $0.7937$, $0.5848$, and $0.4642$ by $+16.9\%$, $+54.3\%$, and $+89.8\%$ (all three failing the $11\%$ relative bound).
   At matched $T_i = 1666.43\text{ K}$, the conductive diagnostic ratios are $0.6102$, $0.4988$, and $0.4207$ ($0.61$, $0.50$, and $0.42$), which deviate from the 0-D targets by $-23.1\%$, $-14.7\%$, and $-9.4\%$ (failing the $11\%$ bound at $0.5$ and $0.2\,\tau_{d,\text{lid}}$, and within $11\%$ only at $0.1\,\tau_{d,\text{lid}}$).

**How we know the right answer.**
The reference trajectory comes from integrating the 0-D stagnant-lid global energy balance with published Nusselt-Rayleigh scaling laws from Korenaga (2009) and Solomatov and Moresi (2000), while the analytical yield-thinning reference $R_\text{thin,0D} = (\tau_{y,0} / \tau_{d,\text{lid}})^{1/3}$ follows from balancing plastic yield stress against convective boundary-layer stress (Moresi and Solomatov, 1998; Foley and Bercovici, 2014).

**Why it matters.**
Stagnant-lid thickness and plastic yielding control how fast a solid rocky mantle loses heat after magma-ocean solidification and whether lithospheric failure can transition a stagnant lid toward mobile-lid cooling in coupled PROTEUS evolutions.

![0-D stagnant-lid trajectory and yield-thinning comparison](../assets/validation/ssc_0d_reference_comparison.svg#only-light){ width="80%" }
![0-D stagnant-lid trajectory and yield-thinning comparison](../assets/validation/ssc_0d_reference_comparison_dark.svg#only-dark){ width="80%" }

*Figure: Comparison of 1-D radial stagnant-lid cooling (panel a) and steady-state yield-thinning ratios (panel b) against the 0-D boundary-layer reference model.*

**How to read the plot.**
Panel (a) plots interior temperature $T_i$ in kelvins (left axis, solid and dashed lines) and stagnant-lid thickness $d_\text{lid}$ in kilometres (right axis, dash-dotted and dotted lines) against time in gigayears over $4.50\text{ Gyr}$; the 1-D lid remains thinner and the 1-D interior cools faster than the 0-D ODE, whose lid thickness follows $(1 - 2.2/\theta)\,D$ from $898.62\text{ km}$ at $3.6958\text{ Gyr}$ to $899.94\text{ km}$ at $4.50\text{ Gyr}$ once the $\mathrm{Nu} = 1.0$ clamp engages.
Panel (b) plots the lid-thinning ratio $R_\text{thin}$ against $\log_{10}(\tau_{y,0} / \tau_{d,\text{lid}})$; the 0-D cube-root reference decreases monotonically as yield stress drops, whereas the 1-D Pass-1 isotherm ratio stays above $1.0$ and only the post-hoc conductive diagnostic thins toward the reference curve.

**Known limit.**
This comparison fails the derived 0-D reference bounds on the $4.50\text{ Gyr}$ cooling trajectory ($6.78\%$ full-run and $5.51\%$ pre-clamp vs $5\%$ on $T_i$; $46.93\%$ pre-clamp and full-run vs $22\%$ on post-relaxation $d_\text{lid}$) and on the steady-state Pass-1 isotherm yield-thinning ratio ($1.53$ fixed heating and $1.05$ matched $T_i$ vs $0.79$ at $\tau_{y,0} = 0.5\,\tau_{d,\text{lid}}$), and pins the current 1-D and 0-D values to $10^{-7}$ relative.

**Where this is checked.** This comparison is checked in `tests/test_rheology_verification_smoke.py::test_0d_stagnant_lid_reference_and_yield_thinning`.

??? info "Test details"
    - Reference: Zero-dimensional stagnant-lid energy-balance ODE (`tests/reference/stagnant_lid_0d.py`), Korenaga (2009, p. 157, Table 1; p. 158, eq. 29; p. 159, Fig. 4a), Solomatov and Moresi (2000, p. 21803, Table 6; p. 21804, eq. 23), Moresi and Solomatov (1998, p. 672, eqs. 11, 13-14; p. 677, eqs. 25-26), and Foley and Bercovici (2014, p. 588, eqs. 26-27).
    - Tolerance: Pinned to $1e-07$ relative for deterministic single-process evaluation, and $1e-07$ relative on the live $25\text{ Myr}$ 1-D trajectory segment from its measured 5-start round-off spread ($\le 4.28 \times 10^{-9}$ on $T_i$, $\le 5.25 \times 10^{-9}$ on $d_\text{lid}$, and $\le 2.77 \times 10^{-8}$ on $F_\text{top}$; the comparison fails the derived $5\%$ $T_i$, $22\%$ $d_\text{lid}$, and $11\%$ isotherm thinning bounds as documented above).
    - Test id: `tests/test_rheology_verification_smoke.py::test_0d_stagnant_lid_reference_and_yield_thinning`
    - Command: `pytest tests/test_rheology_verification_smoke.py -k test_0d_stagnant_lid_reference_and_yield_thinning` (add `--validation-plots` to write the plot pair; regenerate the benchmark file with `python tools/verification/generate_ssc_0d_reference_benchmarks.py --full`)

## 11. Stagnant-Lid Internal-Heating Scaling Exponents

### Does the 1-D mixing-length closure reproduce published internal-heating stagnant-lid scaling exponents across Rayleigh number and viscosity contrast?

In a stagnant-lid mantle heated from within by radioactive decay, boundary-layer theory and two-dimensional (2-D) convection simulations predict that the Nusselt number scales as a power law $\mathrm{Nu} = a\,\theta^{-\alpha}\,\mathrm{Ra}_i^\beta$ with $\beta \approx 1/3$ and $\alpha \approx 4/3$.
We test Aragog's 1-D viscous mixing-length closure over a 12-case steady-state planar internal-heating sweep (`case_01` to `case_12`, stored in `tests/reference/ssc_scaling_exponents_benchmarks.json`, regenerated with `python tools/verification/generate_ssc_scaling_exponents_benchmarks.py --full` and verified with `--check`, where `--check` evaluates live 1-D `EntropySolver._dSdt_single` and `korenaga_ti_fixed_point` on the five stored profile snapshots `case_01`, `case_03`, `case_06`, `case_07`, and `case_12`, checks the remaining seven rows of `cases_results` for internal algebraic and scaling consistency against `cases_spec`, and recomputes all least-squares regressions and conduction-floor values).
Both the design-value regression ($\beta = 0.2819$, $\alpha = 1.0366$) and the realized-state regression ($\beta_\text{real} = 0.1128$, $\alpha_\text{real} = -0.0786$) fall below the Newtonian ($n = 1$) internal-heating stagnant-lid acceptance window $\beta \in [0.294, 0.350]$ and $\alpha \in [1.294, 1.350]$ of Solomatov and Moresi (2000), consistent with low-vigor cases cooling onto the discrete pure-conduction floor and the exponential top mixing-length slope leveling off at high viscosity contrast.

**Why this check exists.**
If the 1-D mixing-length top-slope formula, the steady-state convective flux evaluation, or the sub-lid interior temperature diagnostic were altered, the scaling of mantle heat loss with radiogenic heating and viscosity contrast would shift silently.
This test catches three classes of code faults, each verified by a mutation or perturbation probe:

- Perturbations to the 1-D mixing-length top-slope asymptote on steady-state profiles: perturbing $s_{\text{top},\infty}$ from $0.25$ to $0.26$ on live steady-state snapshot `case_03` ($\mathrm{Ra}_i = 10^7$, $\theta = 15$) shifts the peak convective heat flux $j_\text{conv,max}$ from $0.8229\text{ W m}^{-2}$ to $0.9504\text{ W m}^{-2}$ ($15.49\%$ relative, failing the $10^{-7}$ pinned tolerance) and raises the steady-state entropy residual $\max |\mathrm{d}S/\mathrm{d}t|$ from $3.388 \times 10^{-11}\text{ J kg}^{-1}\text{ K}^{-1}\text{ yr}^{-1}$ to $1.701 \times 10^{-5}\text{ J kg}^{-1}\text{ K}^{-1}\text{ yr}^{-1}$, above the $10^{-9}\text{ J kg}^{-1}\text{ K}^{-1}\text{ yr}^{-1}$ steady-state residual bound.
- Perturbations to the steady-state Nusselt numbers or least-squares power-law regression: perturbing the first steady-state Nusselt number by $1\%$ ($1.01\,\mathrm{Nu}_1$) shifts the fitted Rayleigh exponent $\beta$ from $0.2819$ (`0.281862`) to $0.2812$ (`0.281166`, a $2.931 \times 10^{-3}$ relative change, failing the $10^{-7}$ pinned tolerance).
- Non-convergence of the Korenaga (2009, p. 156, eq. 20) sub-lid interior temperature fixed-point solver: `korenaga_ti_fixed_point` (`tools/verification/_ssc_helpers.py`) raises `RuntimeError` if the fixed-point iteration does not converge within `100` steps.

**What it tests.**
Here:
- $\mathrm{Ra}_i$ is the design (nominal) internal Rayleigh number evaluated at the nominal interior temperature $T_{i,\text{nom}} = 1773\text{ K}$, and $\mathrm{Ra}_{i,\text{realized,kor}}$ is the realized internal Rayleigh number evaluated at the steady-state sub-lid interior temperature $T_{i,\text{kor}}$ in kelvins ($\text{K}$).
- $\theta$ is the design Frank-Kamenetskii parameter at $T_{i,\text{nom}}$, and $\theta_\text{realized,kor}$ is the realized Frank-Kamenetskii parameter at $T_{i,\text{kor}}$.
- $\Delta T_H = \rho H D^2 / k$ is the internal-heating temperature scale in kelvins ($\text{K}$), where $H$ is the internal heat generation rate in watts per kilogram ($\text{W kg}^{-1}$) and $D$ is the mantle thickness in metres ($\text{m}$).
- $\mathrm{Nu} = \Delta T_H / (T_{i,\text{kor}} - T_\text{surf})$ is the internal-heating Nusselt number, $\beta$ is the Rayleigh scaling exponent, $\alpha$ is the Frank-Kamenetskii scaling exponent, and $a$ is the dimensionless scaling prefactor.

The 12 steady-state runs span design values $\mathrm{Ra}_i \in [10^6, 10^9]$ at fixed $\theta = 15$ and $\theta \in [10, 25]$ at fixed $\mathrm{Ra}_i = 10^7$ on an $N = 100$ mesh with insulated basal boundary $F_\text{bot} = 0$, planar aspect ratio $r_\text{out} = 100\,D$, surface temperature $T_\text{surf} = 273\text{ K}$, nominal interior temperature $T_{i,\text{nom}} = 1773\text{ K}$, calibration $(s_{\text{top},\infty}, c_\theta) = (0.25, 0.25)$, and constant material properties (`eos_method = 1`, `const_properties = True`, no equation-of-state table read).
In `--full` mode, `tools/verification/generate_ssc_scaling_exponents_benchmarks.py` temporarily replaces `EntropySolver._solve_cvode` with an unbounded-step wrapper (`max_step = np.inf`) to integrate the 12 cases to steady state (up to $320\text{ Gyr}$) without stopping at the $100\text{ yr}$ phase-boundary step cap, because the solver caps steps at $100\text{ yr}$ after solidification.
The 12 cases prescribe the surface temperature (`outer_boundary_condition = 5`): the solver sets the surface node to the entropy whose temperature is $T_\text{surf} = 273.0\text{ K}$ and takes the surface gradient from the half-cell temperature gradient, so the top conductive heat flux is $F_\text{top} = k (T_\text{cell} - T_\text{surf}) / (\Delta r / 2)$ and the stagnant-lid closure reads $T_\text{surf}$.
In Solomatov and Moresi (2000, pp. 21797-21798, sec. 3; p. 21800, Table 4 and eqs. 9, 10, 12; p. 21801, Table 5; p. 21803, Table 6 rows 6-8; p. 21805, sec. 7.4), the 12 Newtonian ($n = 1$) internal-heating stagnant-lid experiments span $\theta_H \in [40, 60]$ and $\mathrm{Ra}_{H,0} \in [0.1, 100]$ (effective viscosity-contrast parameter $\theta / \mathrm{Nu} \in [13, 19]$), and Korenaga (2009, p. 157, Table 1; p. 158, eqs. 29-30) fits the Arrhenius internal-heating scaling over $\mathrm{Ra}_i \in [2.41 \times 10^6, 1.89 \times 10^7]$ and $\theta \in [11.7, 19.2]$.
Both the design-value regression and the realized-state regression fall below the Newtonian ($n = 1$) internal-heating stagnant-lid acceptance window $\beta \in [0.294, 0.350]$ and $\alpha \in [1.294, 1.350]$ from Solomatov and Moresi (2000, Tables 5 and 6, around the asymptotic boundary-layer exponents $\beta = 1/3 \approx 0.333$ and $\alpha = 4/3 \approx 1.333$):

1. **Design-value vs realized-state least-squares fits:**
   In each steady-state run, the internal heating rate $H$ and reference viscosity $\eta_{i,\text{nom}}$ are set from the design (nominal) parameters $(\mathrm{Ra}_i, \theta)$ evaluated at $T_{i,\text{nom}} = 1773\text{ K}$ ($\Delta T_\text{nom} = 1500\text{ K}$).
   As the mantle equilibrates, the realized steady state cools to the Korenaga (2009, p. 156, eq. 20) sub-lid interior temperature $T_{i,\text{kor}} < T_{i,\text{nom}}$, which defines the realized parameters (`theta_realized_kor` and `Ra_i_realized_kor` in `tests/reference/ssc_scaling_exponents_benchmarks.json`):

    $$
    \theta_\text{realized,kor} = \frac{E_\text{act}(T_{i,\text{kor}} - T_\text{surf})}{R T_{i,\text{kor}}^2}, \quad \mathrm{Ra}_{i,\text{realized,kor}} = \frac{\alpha \rho g (T_{i,\text{kor}} - T_\text{surf}) D^3}{\kappa \eta_\text{diff}(T_{i,\text{kor}})}
    $$

    Regressing $\ln \mathrm{Nu} = \ln a - \alpha \ln \theta + \beta \ln \mathrm{Ra}_i$ (with $\mathrm{Nu} = \Delta T_H / (T_{i,\text{kor}} - T_\text{surf})$) against the design values $(\mathrm{Ra}_i, \theta)$ over all 12 cases yields $\beta = 0.2819 \pm 0.0134$, $\alpha = 1.0366 \pm 0.0971$, and $a = 0.6415$ ($\ln a = -0.4439 \pm 0.3571$, $R^2 = 0.9849$), with single-sweep ordinary least-squares slopes $\beta_\text{sweep} = 0.2783 \pm 0.0172$ at design $\theta = 15$ (`6` points) and $\alpha_\text{sweep} = 1.0104 \pm 0.0590$ at design $\mathrm{Ra}_i = 10^7$ (`6` points).
    Regressing the same 12 steady-state Nusselt numbers against the realized state variables $(\mathrm{Ra}_{i,\text{realized,kor}}, \theta_\text{realized,kor})$ yields $\beta_\text{real} = 0.1128 \pm 0.0286$, $\alpha_\text{real} = -0.0786 \pm 0.4398$, and $a_\text{real} = 0.6940$ ($\ln a_\text{real} = -0.3653 \pm 1.5581$, $R^2 = 0.7750$; single-sweep slopes $\beta_\text{real,sweep} = 0.1198 \pm 0.0285$ and $\alpha_\text{real,sweep} = 0.8290 \pm 0.0731$).
    Because $T_{i,\text{kor}}$ enters both the dependent variable $\mathrm{Nu} = \Delta T_H / (T_{i,\text{kor}} - T_\text{surf})$ and the realized regressors $(\mathrm{Ra}_{i,\text{realized,kor}}, \theta_\text{realized,kor})$, variations in $T_{i,\text{kor}}$ couple the response and predictor axes in the realized-state regression.
    Arrhenius cooling $T_{i,\text{kor}} < T_{i,\text{nom}}$ increases $\eta_\text{diff}(T_{i,\text{kor}})$ by up to $4.5$ orders of magnitude and shifts 10 of the 12 cases below the $\mathrm{Ra}_{i,\text{realized,kor}} \ge 10^7$ asymptotic stagnant-lid regime:

    | Case | Design $\mathrm{Ra}_i$ | Design $\theta$ | Realized $T_{i,\text{kor}}$ [K] | Realized $\theta_\text{realized,kor}$ | Realized $\mathrm{Ra}_{i,\text{realized,kor}}$ | $\mathrm{Nu}_\text{kor}$ |
    |:---|---:|---:|---:|---:|---:|---:|
    | `case_01` | $1.00 \times 10^6$ | $15.0$ | $1228.14$ | $19.91$ | $2.44 \times 10^2$ | $2.2500$ |
    | `case_02` | $3.16 \times 10^6$ | $15.0$ | $1501.99$ | $17.13$ | $1.06 \times 10^5$ | $2.5666$ |
    | `case_03` | $1.00 \times 10^7$ | $15.0$ | $1613.29$ | $16.19$ | $1.54 \times 10^6$ | $3.4545$ |
    | `case_04` | $3.16 \times 10^7$ | $15.0$ | $1670.69$ | $15.74$ | $9.95 \times 10^6$ | $4.8622$ |
    | `case_05` | $1.00 \times 10^8$ | $15.0$ | $1706.90$ | $15.47$ | $4.81 \times 10^7$ | $6.9565$ |
    | `case_06` | $1.00 \times 10^9$ | $15.0$ | $1750.80$ | $15.16$ | $7.87 \times 10^8$ | $14.5422$ |
    | `case_07` | $1.00 \times 10^7$ | $10.0$ | $1710.27$ | $10.30$ | $6.21 \times 10^6$ | $5.5313$ |
    | `case_08` | $1.00 \times 10^7$ | $13.0$ | $1650.96$ | $13.77$ | $2.95 \times 10^6$ | $4.0664$ |
    | `case_09` | $1.00 \times 10^7$ | $16.0$ | $1594.43$ | $17.43$ | $1.06 \times 10^6$ | $3.2149$ |
    | `case_10` | $1.00 \times 10^7$ | $19.0$ | $1532.72$ | $21.35$ | $2.48 \times 10^5$ | $2.6818$ |
    | `case_11` | $1.00 \times 10^7$ | $22.0$ | $1448.13$ | $25.84$ | $2.29 \times 10^4$ | $2.3644$ |
    | `case_12` | $1.00 \times 10^7$ | $25.0$ | $1313.95$ | $31.59$ | $2.28 \times 10^2$ | $2.2509$ |

2. **Analytical and discrete pure-conduction floor:**
   For a continuous planar conductive layer with uniform internal heating $H$, insulated base ($z = 1$), and fixed surface temperature $T_\text{surf}$ ($z = 0$), the dimensionless conductive profile $u(z) = (T(z) - T_\text{surf}) / \Delta T_H = z - \frac{1}{2}z^2$ has basal peak $u(1) = 1/2$ ($\mathrm{Nu}_\text{peak} = 2.0$) and volume average $\int_0^1 u(z)\,\mathrm{d}z = 1/3$ ($\mathrm{Nu}_\text{vol} = 3.0$).
   Substituting $u(z) = z - \frac{1}{2}z^2$ into the Korenaga (2009, p. 156, eq. 20) sub-lid average $u_i = \frac{1}{1 - d_l}\int_{d_l}^1 (z - \frac{1}{2}z^2)\,\mathrm{d}z = \frac{1}{3} + \frac{1}{3}d_l - \frac{1}{6}d_l^2$ with conductive lid thickness $d_l = u_i$ gives $d_l^2 + 4 d_l - 2 = 0$, with physical root $d_l = \sqrt{6} - 2 \approx 0.4495$ (`0.44948974`) and exact analytic planar Korenaga conduction floor $\mathrm{Nu}_{\text{Korenaga,cond}}^\text{planar} = \frac{1}{\sqrt{6} - 2} = 1 + \frac{\sqrt{6}}{2} \approx 2.2247$ (`2.22474487`).
   Evaluating the same Korenaga fixed point by trapezoidal quadrature on the discrete $N = 100$ staggered grid (`fixed_parameters` in `tests/reference/ssc_scaling_exponents_benchmarks.json`) gives $\mathrm{Nu}_{\text{Korenaga,cond}}^{\text{planar},N=100} = 2.2276$ (`2.22763824`, $d_l = 0.4489$) for the planar parabola and $\mathrm{Nu}_{\text{Korenaga,cond}}^{\text{sph},N=100} = 2.2472$ (`2.24715118`, $d_l = 0.4450$) for the steady finite-volume spherical shell with $r_\text{out} = 100\,D$ and $r_\text{in} = 99\,D$ (`convection = False`).
   In the pinned 12-case sweep, `case_01` ($\mathrm{Nu} = 2.2500$, $T_{i,\text{kor}} = 1228.14\text{ K}$, $\mathrm{Ra}_{i,\text{realized,kor}} = 244.3$) and `case_12` ($\mathrm{Nu} = 2.2509$, $T_{i,\text{kor}} = 1313.95\text{ K}$, $\mathrm{Ra}_{i,\text{realized,kor}} = 227.9$) sit just above the discrete spherical conduction floor $2.2472$ due to weak residual sub-lid convection ($j_\text{conv,max} = 6.27\text{ mW m}^{-2}$ and $12.25\text{ mW m}^{-2}$).
3. **Convecting 7-case subset (diagnostic chosen after the result) and asymptotic $\theta$ limit:**
   Restricting the regression to the 7 cases with $\mathrm{Nu}_\text{est} \ge 2.8$ (`case_03` to `case_09`, a post-hoc diagnostic chosen after the result) gives $\beta = 0.3112 \pm 0.0026$ (single-sweep $\beta_\text{sweep} = 0.3130 \pm 0.0037$ over `4` points at $\theta = 15$, within $[0.294, 0.350]$), $\alpha = 1.1698 \pm 0.0285$ (single-sweep $\alpha_\text{sweep} = 1.1553 \pm 0.0117$ over `3` points at $\mathrm{Ra}_i = 10^7$, `1` degree of freedom), and $a = 0.5414$ ($R^2 = 0.9997$) on the design values $(\mathrm{Ra}_i, \theta)$.
   On the realized values $(\mathrm{Ra}_{i,\text{realized,kor}}, \theta_\text{realized,kor})$, the 7-case subset gives $\beta_\text{real} = 0.2210 \pm 0.0106$, $\alpha_\text{real} = 0.3573 \pm 0.1423$, and $a_\text{real} = 0.3935$ ($R^2 = 0.9911$; single-sweep $\beta_\text{real,sweep} = 0.2324 \pm 0.0139$ over `4` points and $\alpha_\text{real,sweep} = 1.0322 \pm 0.0171$ over `3` points, `1` degree of freedom).
   Even on the convecting subset, $\alpha = 1.1698$ (and $\alpha_\text{real} = 0.3573$) remains below $[1.294, 1.350]$, consistent with the exponential top mixing-length slope $s_\text{top}(\theta) = s_{\text{top},\infty} + (1 - s_{\text{top},\infty})\exp(-c_\theta \theta)$ approaching the constant asymptote $s_{\text{top},\infty} = 0.25$ once $\theta \ge 15$ ($s_\text{top}(15) = 0.2676$ and $s_\text{top}(25) = 0.2514$ at $(0.25, 0.25)$) rather than decaying as $\theta^{-1/3}$.

**How we know the right answer.**
Boundary-layer theory and 2-D stagnant-lid numerical experiments for Newtonian internal heating (Solomatov and Moresi, 2000; Korenaga, 2009; Reese et al., 2005) establish the asymptotic power-law exponents $\beta = 1/3$ and $\alpha = 4/3$, while the pure-conduction floor $\mathrm{Nu}_{\text{Korenaga,cond}}^\text{planar} = 1 + \sqrt{6}/2$ follows analytically from integrating the steady 1-D conduction parabola over the Korenaga sub-lid domain.

**Why it matters.**
Radiogenic heating drives solid-mantle convection over gigayear timescales; knowing where the 1-D mixing-length closure follows or departs from stagnant-lid scaling laws tells users how accurately Aragog predicts deep-mantle temperatures across different rocky planet interiors.

![Stagnant-lid internal-heating scaling exponents](../assets/validation/ssc_stagnant_lid_scalings.svg#only-light){ width="80%" }
![Stagnant-lid internal-heating scaling exponents](../assets/validation/ssc_stagnant_lid_scalings_dark.svg#only-dark){ width="80%" }

*Figure: Steady-state internal-heating Nusselt number scalings against internal Rayleigh number (panel a) and Frank-Kamenetskii viscosity-contrast parameter (panel b).*

**How to read the plot.**
Panel (a) plots $\log_{10}(\mathrm{Nu}\,\theta^{4/3})$ against $\log_{10}(\mathrm{Ra}_i)$ at design $\theta = 15$, and panel (b) plots $\log_{10}(\mathrm{Nu}\,\mathrm{Ra}_i^{-1/3})$ against $\log_{10}(\theta)$ at design $\mathrm{Ra}_i = 10^7$.
Circles show the 12 steady-state 1-D runs, the solid line shows the 12-case ordinary least-squares fit, the dash-dotted line shows the post-hoc 7-case convecting fit, and the dashed line shows the Solomatov and Moresi (2000) asymptotic slope ($\beta = 1/3$ in panel a, $-\alpha = -4/3$ in panel b); low-$\mathrm{Ra}_i$ and high-$\theta$ points flatten onto the conduction floor.

**Known limit.**
Both the 12-case design-parameter fit ($\beta = 0.2819 \pm 0.0134$, $\alpha = 1.0366 \pm 0.0971$) and the realized-state fit ($\beta_\text{real} = 0.1128 \pm 0.0286$, $\alpha_\text{real} = -0.0786 \pm 0.4398$) fail the pre-run acceptance window $\beta \in [0.294, 0.350]$ and $\alpha \in [1.294, 1.350]$, and the test pins the current steady-state values and regressions to $10^{-7}$ relative.

**Where this is checked.** This scaling sweep is checked in `tests/test_mlt_calibration_smoke.py::test_stagnant_lid_scaling_exponents`.

??? info "Test details"
    - Reference: Newtonian ($n = 1$) internal-heating stagnant-lid scaling theory and numerical fits of Solomatov and Moresi (2000, pp. 21797-21798, sec. 3; p. 21800, Table 4 and eqs. 9, 10, 12; p. 21801, Table 5; p. 21803, Table 6 rows 6-8; p. 21805, sec. 7.4), Korenaga (2009, p. 156, eq. 20; p. 157, Table 1; p. 158, eqs. 29-30), and Reese et al. (2005, p. 366, Table 2 and eq. 20).
    - Tolerance: Pinned to $1e-07$ relative for deterministic single-process evaluation (both the 12-case design-value fit $\beta = 0.2819 \pm 0.0134$, $\alpha = 1.0366 \pm 0.0971$ and the realized-state fit $\beta_\text{real} = 0.1128 \pm 0.0286$, $\alpha_\text{real} = -0.0786 \pm 0.4398$ fail the pre-run acceptance window $\beta \in [0.294, 0.350]$ and $\alpha \in [1.294, 1.350]$ as documented above).
    - Test id: `tests/test_mlt_calibration_smoke.py::test_stagnant_lid_scaling_exponents`
    - Command: `pytest tests/test_mlt_calibration_smoke.py -k test_stagnant_lid_scaling_exponents` (add `--validation-plots` to write the plot pair; regenerate the benchmark file with `python tools/verification/generate_ssc_scaling_exponents_benchmarks.py --full`)
