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
2. **Stage 2 - SPIDER `combine_matprop`.** Inside the band the property is a smooth combination of the mixed (lever-rule) value and the corresponding single-phase value, weighted by the cached `matprop_smooth_width`. This is the SPIDER-parity blend `smth*mixed + (1-smth)*single` that ensures continuity across the solidus and liquidus.

The same two-stage blend is applied to density, heat capacity, thermal expansivity, isentropic temperature gradient, and thermal conductivity. The resulting per-cell properties are used by the conduction, convection, gravitational-separation, and mixing fluxes on the same RHS evaluation.

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
