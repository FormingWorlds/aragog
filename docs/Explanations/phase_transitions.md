# Phase transitions

Aragog handles solid, liquid, and mixed (partially molten) phases through the `EntropyPhaseEvaluator`, a single phase-aware wrapper around the loaded `EntropyEOS`. The phase state at each radial node is determined by the local entropy relative to the solidus and liquidus entropies at that pressure, with smooth blending across the phase boundaries.

## Melt fraction

The melt mass fraction in the entropy formulation is

$$
\phi(P, S) = \mathrm{clip}\!\left(\frac{S - S_\mathrm{sol}(P)}{S_\mathrm{liq}(P) - S_\mathrm{sol}(P)},\ 0,\ 1\right),
$$

with $S_\mathrm{sol}(P)$ and $S_\mathrm{liq}(P)$ the solidus and liquidus entropies. The clip is implemented as a smooth (rather than hard) transition so that the BDF Jacobian remains continuous through the solidus and liquidus crossings; the analytic clip is

$$
\mathrm{clip}_\mathrm{smooth}(x) = \tfrac{1}{2}\left[\mathrm{softplus}_\beta(x) - \mathrm{softplus}_\beta(x - 1) - 1\right] + \tfrac{1}{2},
$$

with a sharpness parameter $\beta$ chosen so the smooth and hard clips agree to better than $10^{-3}$ in $\phi$ except in a narrow band of width $\sim 1/\beta$ around 0 and 1.

## Phase evaluator

The production EOS path uses a single evaluator class:

- `EntropyEOS` loads the SPIDER-format pressure-entropy tables (`temperature_solid.dat`, `temperature_melt.dat`, `density_*`, `heat_capacity_*`, `adiabat_temp_grad_*`, `solidus_P-S.dat`, `liquidus_P-S.dat`) and exposes property lookups: `temperature(P, S)`, `density(P, S)`, `melt_fraction(P, S)`, `solidus_entropy(P)`, `liquidus_entropy(P)`, `latent_heat(P)`, `solidus_entropy_dP(P)`, `liquidus_entropy_dP(P)`.
- `EntropyPhaseEvaluator` wraps `EntropyEOS` with phase-mixing logic, viscosity blending, and the cached per-cell properties used by the solver.

There is no separate `SinglePhaseEvaluator` / `MixedPhaseEvaluator` / `CompositePhaseEvaluator` hierarchy in the production path: the EOS table itself spans all three regimes, and the per-cell evaluator selects between solid, mixed, and liquid branches via a smooth two-stage blend (described next).

## Two-stage SPIDER-parity blend

For each cell at $(P, S)$, the phase evaluator computes a blending weight in two stages:

1. **Stage 1 - mushy band weight $\mathrm{smth}(\phi)$.** The phase-boundary smoothing function from [Heat transport](heat_transport.md) defines the weight of the mixed-phase contribution (tanh in the production setting, cubic-Hermite as the fallback). Outside the mushy band $\mathrm{smth}(\phi) = 0$, and the property reduces to the relevant single-phase value.
2. **Stage 2 - SPIDER `combine_matprop`.** Inside the band the property is a smooth combination of the mixed (lever-rule) value and the corresponding single-phase value, weighted by the cached `matprop_smooth_width` ([width of the blend](#width-of-the-blend-at-the-solidus-and-the-liquidus)). This is the SPIDER-parity blend `smth*mixed + (1-smth)*single` that ensures continuity across the solidus and liquidus.

The same two-stage blend is applied to density, heat capacity, thermal expansivity, isentropic temperature gradient, and thermal conductivity. The resulting per-cell properties are used by the conduction, convection, gravitational-separation, and mixing fluxes on the same RHS evaluation.

## Width of the blend at the solidus and the liquidus

How does a property change when the entropy of a node crosses the solidus or the liquidus?

On each side of a phase boundary the evaluator has two values of every property: the two-phase value of the mushy zone and the single-phase value from the solid or the melt table. The weight $\mathrm{smth}$ moves from one value to the other over a width $w$, the setting `matprop_smooth_width` in `[phase_mixed]`. The default is $w = 0.01$, the value PROTEUS uses, so the properties change smoothly at both boundaries.

The weight uses the melt coordinate $g = (S - S_\mathrm{sol}) / (S_\mathrm{liq} - S_\mathrm{sol})$, which is not clipped: $g < 0$ below the solidus and $g > 1$ above the liquidus. SPIDER's `get_smoothing` (`util.c`) gives

$$
\mathrm{smth}(g) =
\begin{cases}
\tfrac{1}{2}\left[1 + \tanh(g / w)\right], & g \le 0.5,\\
\tfrac{1}{2}\left[1 - \tanh((g - 1) / w)\right], & g > 0.5,
\end{cases}
\qquad
X = \mathrm{smth}\,X_\mathrm{two\text{-}phase} + (1 - \mathrm{smth})\,X_\mathrm{single},
$$

for each property $X$. On a boundary ($g = 0$ or $g = 1$) the weight is exactly $\tfrac{1}{2}$, so the property is the mean of the two values.

With $w = 0$ the weight is 1 inside the mushy zone and 0 outside it, so the properties jump at the boundary. Example from the SPIDER test tables at 65.7 GPa: the melting interval is $S_\mathrm{liq} - S_\mathrm{sol}$ = 8148 J/kg/K, so $w = 0.01$ spreads the change over about 81 J/kg/K. At $w = 0$ the heat capacity jumps by a factor of 7.9 to 23 at the solidus, at the 12 pressures from 0 to 146 GPa of the validation mesh. The convective heat flux through a cell face depends on the heat capacity, the expansivity and the viscosity, so the flux jumps when one node crosses the boundary, and the time stepper must take very small steps there.

Limits:

- `matprop_smooth_width` must lie strictly between 0 and 1. A value at or below 0, at or above 1, NaN, an infinity, a bool or a string stops the configuration with a `ValueError` that names the setting.
- More than 0.2 in $g$ from both boundaries, $\tanh$ rounds to 1 in double precision at $w = 0.01$: the weight is exactly 0 or 1, and the properties are the same as at $w = 0$.
- `EntropyPhaseEvaluator` and the JAX `PhaseParams` keep a default of 0 when a test or script calls them directly; a run built from a configuration passes the width of the configuration.
- PROTEUS passes its own setting `interior_energetics.spider.matprop_smooth_width` (default 0.01, also strictly between 0 and 1).

**Where this is checked.** [Validation: Phase transitions](../Validation/phase_transitions.md). The tutorial [Properties at the solidus](../Tutorials/phase_blend.md) evaluates the blend step by step.

## Density and heat capacity in the mushy zone

In the mushy zone the lever-rule density is

$$
\frac{1}{\rho_\mathrm{mix}} = \frac{\phi}{\rho_m} + \frac{1-\phi}{\rho_s},
$$

and the heat capacity follows from $T\,dS = c_p\,dT - (\alpha T/\rho)\,dP$ along the isobaric mushy path. Two `cp_blend` modes are supported:

| `cp_blend` | Formula |
|------------|---------|
| `"latent"` (default; SPIDER parity) | $c_p^\mathrm{mix}(P, S) = (S_\mathrm{liq}(P) - S_\mathrm{sol}(P)) / (T_\mathrm{liq}(P) - T_\mathrm{sol}(P)) \cdot T_\mathrm{mid}$ |
| `"linear"` | Linear interpolation of $c_p^\mathrm{sol}$ and $c_p^\mathrm{liq}$ in $\phi$ |

The latent-blend mode tracks SPIDER's internal `cp_s` to within a few percent along the mushy trajectory, while the linear blend can underestimate by up to a factor of six in the deep mushy zone.

## Latent heat

The latent heat of fusion enters the entropy equation in two places:

- The capacitance $\rho T$ multiplied by $\partial S/\partial t$ already accounts for the energy absorbed or released by melting. There is **no $c_p$ effective spike** at the solidus or liquidus in the entropy formulation, which is what makes it numerically robust through phase transitions where a $T$-form solver would see a divergent effective heat capacity.
- The pressure-dependent latent heat $L(P)$ multiplies the gravitational-separation mass flux to give $F_\mathrm{grav} = j_\mathrm{grav}\,L(P)$. $L(P)$ is read from the EOS-tabulated $T_\mathrm{liq}\,(S_\mathrm{liq} - S_\mathrm{sol})$ at each pressure, not from the legacy `latent_heat_of_fusion` configuration key (which is retained for the constant-properties analytical mode).

## Solidus and liquidus curves

The solidus and liquidus are loaded from two-column $(P, S)$ files named `solidus_P-S.dat` and `liquidus_P-S.dat` in the EOS-table directory. These are not P-T tables: in PROTEUS coupled runs the wrapper derives them from the configured solid-melt P-T file via the EOS, ensuring the P-S curves used by the solver are exactly consistent with the tabulated $T(P, S)$.

The pressure dependence of the curves determines where the mushy zone sits at each depth and, indirectly, the cooling timescale of the mantle: a shallow solidus produces a thin mushy band at high pressure; a steep solidus broadens the band and prolongs cooling.

## Rheological transition

The mantle viscosity changes dramatically (many orders of magnitude) across the rheological transition at the critical melt fraction $\phi_\mathrm{rheo}$. The transition is parameterised as a smooth $\tanh$ in log-viscosity space:

$$
\log_{10}\eta(\phi) = \log_{10}\eta_s + (\log_{10}\eta_m - \log_{10}\eta_s)\,w(\phi),\qquad
w(\phi) = \tfrac{1}{2}\Big[1 + \tanh\!\big((\phi - \phi_\mathrm{rheo})/\Delta_\mathrm{rheo}\big)\Big],
$$

with `rheological_transition_melt_fraction` setting $\phi_\mathrm{rheo}$ and `rheological_transition_width` setting the blend width $\Delta_\mathrm{rheo}$. Below the transition the solid viscosity dominates and convection is sluggish; above it the melt viscosity takes over and convection becomes vigorous, which is what triggers magma-ocean convective overturn.

## Constant-properties analytical mode

Setting `const_properties = true` in `[phase_mixed]` activates an analytical mode that bypasses the EOS tables entirely. Properties become constants set by `const_rho`, `const_Cp`, `const_alpha`, `const_cond`, `const_log10visc`, and the entropy-temperature relation reduces to

$$
T(S) = T_\mathrm{ref}\,\exp\!\left(\frac{S - S_\mathrm{ref}}{C_p}\right),
$$

with `const_T_ref` and `const_S_ref` setting the reference state. This mode is useful for analytic cooling tests and for unit tests where a tabulated EOS is not available; it is not used in production PROTEUS runs.
