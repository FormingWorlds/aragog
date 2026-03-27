# Phase transitions

Aragog handles solid, liquid, and mixed (partially molten) phases through a system of composable phase evaluators. The phase state at each mesh node is determined by the local temperature and pressure relative to the solidus and liquidus.

## Melt fraction

Aragog uses a piecewise-linear melt mass fraction between pressure-dependent solidus and liquidus curves:

$$
\phi =
\begin{cases}
0, & T < T_{\mathrm{sol}}(P),\\[4pt]
\dfrac{T - T_{\mathrm{sol}}}{T_{\mathrm{liq}} - T_{\mathrm{sol}}}, &
T_{\mathrm{sol}} \le T \le T_{\mathrm{liq}},\\[8pt]
1, & T > T_{\mathrm{liq}}(P).
\end{cases}
$$

The solidus and liquidus curves can be specified as either data files (lookup tables) or analytic functions.

## Phase evaluator hierarchy

Aragog's EOS system is organized as a hierarchy of evaluators:

### SinglePhaseEvaluator

Computes material properties (density, viscosity, heat capacity, thermal expansivity, thermal conductivity) for a single end-member phase (solid or liquid). Properties can be constants or lookup tables.

### MixedPhaseEvaluator

Computes properties in the two-phase region by mixing solid and liquid properties according to the melt fraction $\phi$:

- **Density** (harmonic mean):
  $$
  \frac{1}{\rho} = \frac{\phi}{\rho_m} + \frac{1-\phi}{\rho_s}
  $$

- **Thermal conductivity** (linear):
  $$
  \lambda = \phi\lambda_m + (1-\phi)\lambda_s
  $$

- **Effective viscosity**: transitions sharply near a critical melt fraction $\phi_c^\eta$ using a smoothed log-mixing law (tanh weighting). Below the rheological transition, the solid viscosity dominates; above it, the melt viscosity takes over.

- **Effective heat capacity** and **thermal expansivity**: include dominant contributions from latent heat via $\partial\phi/\partial T$, producing large effective values in the mixed-phase region that capture the energy absorbed/released during melting.

### CompositePhaseEvaluator

The composite evaluator assembles the full model by selecting which evaluator to use at each mesh node:

- Below the solidus: use the solid evaluator
- Above the liquidus: use the liquid evaluator
- Between solidus and liquidus: use the mixed-phase evaluator

The transitions are smoothed using a configurable `phase_transition_width` parameter to avoid discontinuities that would cause numerical difficulties for the BDF integrator.

## Latent heat treatment

Latent heat is not handled as an explicit source term. Instead, it enters implicitly through the effective heat capacity in the mixed-phase region. Since $\phi$ depends on $T$, the partial derivative $\partial\phi/\partial T$ contributes a large term to the effective $c_p$:

$$
c_p^{\mathrm{eff}} = c_p + \Delta h \frac{\partial \phi}{\partial T},
$$

where $\Delta h$ is the latent heat of fusion. This approach is numerically stable because the large effective heat capacity simply slows temperature changes in the mixed-phase region, rather than introducing discrete energy injections.

## Solidus and liquidus curves

The solidus and liquidus are specified in the `[phase_mixed]` configuration section. Options include:

- **Lookup tables**: text files with pressure-temperature pairs, interpolated with PCHIP
- **Analytic functions**: hardcoded parametric fits (Simon-Glatzel form)

The pressure dependence of the melting curves determines where the mixed-phase region exists at each depth.

## Rheological transition

The viscosity undergoes a dramatic change (many orders of magnitude) across the rheological transition at melt fraction $\phi_c^\eta$. This is parameterized as a smooth tanh transition in log-viscosity space:

$$
\log_{10}\eta = \log_{10}\eta_s + (\log_{10}\eta_m - \log_{10}\eta_s) \cdot w(\phi),
$$

where $w(\phi)$ is a tanh weight function centered at $\phi_c^\eta$ with configurable width. The rheological transition separates the solid-like regime (high viscosity, slow convection) from the liquid-like regime (low viscosity, vigorous convection).
