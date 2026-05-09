# Two-phase flow in Aragog

A magma ocean is rarely fully molten or fully solid.
Across most of its evolution, the silicate mantle sits in a *mushy* regime where solid grains and silicate melt coexist over a finite depth range.
Aragog represents this regime as a **two-phase mixture** of melt and solid that coevolves at every radial node, rather than as a single-phase fluid with a depth-cutoff between molten and solid layers.
This page explains the conceptual difference between one-phase and two-phase formulations, lists the additional physics terms the two-phase representation enables, and points to the source files that implement them.

## One-phase versus two-phase formulations

In a **one-phase** magma-ocean model, the mantle is treated as a single homogeneous fluid that is either entirely above the liquidus or entirely below the solidus.
A solidification front (a depth or a single state variable) tracks where the transition lives, and the mantle on each side of the front is closed by its own equation of state.
The bulk fluid evolves through a global energy balance, and the latent heat of fusion either appears as an effective heat-capacity spike at the phase boundary or as a global solidification term subtracted from the surface flux.
Most of the boundary-layer-theory (BLT) lineage of magma-ocean codes ([Elkins-Tanton (2008)](https://scixplorer.org/abs/2008E%26PSL.271..181E/abstract)[^cite-elkinstanton2008], [Hamano et al. (2013)](https://scixplorer.org/abs/2013Natur.497..607H/abstract)[^cite-hamano2013], [Schaefer et al. (2016)](https://scixplorer.org/abs/2016ApJ...829...63S/abstract)[^cite-schaefer2016]) is one-phase by construction: the well-mixed assumption that defines BLT is a one-phase assumption.

In a **two-phase** model, both the melt and the solid coexist on the same spatial element, with a melt fraction $\phi(r,t) \in [0,1]$ describing how much of each is present.
Solid and melt have independent equations of state, blended by the lever rule into composite material properties.
Solid grains and melt droplets can move with respect to each other in response to gravity (gravitational separation) and chemical diffusion (chemical mixing of melt fraction).
Phase-change latent heat enters the energy budget as a continuous function of $\phi$, not as a delta function at the solidus or liquidus.
Aragog and [SPIDER](https://github.com/FormingWorlds/SPIDER) ([Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract)[^cite-bower2018]) are two-phase magma-ocean codes; this is the second defining feature, alongside [mixing-length theory](mixing_length.md), that distinguishes them from the BLT lineage.

The trade-off is the usual one: the two-phase representation is necessary to capture the partial-melt physics that controls the rheological transition and the late-stage solidification window, but it costs an extra state variable per cell, two phase-boundary lookups per cell per call, and three additional flux contributions on the right-hand side.

## How Aragog defines the two-phase state

The two-phase state at each radial node is defined by the local pressure $P$ and specific entropy $S$.
The phase-boundary entropies $S_\mathrm{sol}(P)$ (solidus) and $S_\mathrm{liq}(P)$ (liquidus) are tabulated as functions of pressure in the EOS file.
The melt fraction is computed from the *un-truncated* lever rule

$$
g\phi = \frac{S - S_\mathrm{sol}(P)}{S_\mathrm{liq}(P) - S_\mathrm{sol}(P)},\qquad \phi = \mathrm{clip}(g\phi,\ 0,\ 1).
$$

The clipped $\phi$ is the physical melt fraction; the un-truncated $g\phi$ is retained for the smoothing functions that gate the two-phase fluxes (see below).
This single ratio determines how all the other material properties are blended.

| Property | Two-phase blending rule |
|---|---|
| $T$ | $T = \phi\,T_\mathrm{liq}(P) + (1-\phi)\,T_\mathrm{sol}(P)$ (linear lever rule) |
| $\rho$ | $1/\rho = \phi/\rho_\mathrm{liq}(P) + (1-\phi)/\rho_\mathrm{sol}(P)$ (harmonic mean) |
| $\alpha$ | composite expression $(\rho_\mathrm{sol}-\rho_\mathrm{liq})/(\Delta T_\mathrm{phase}\,\rho_\mathrm{mix})$ in `latent` Cp mode |
| $c_p$ | latent-heat-augmented in mushy regions, returning to the per-phase EOS values at $\phi=0$ and $\phi=1$ |
| $\eta$ (viscosity) | $\log_{10}\eta = z\,\log_{10}\eta_\mathrm{liq} + (1-z)\,\log_{10}\eta_\mathrm{sol}$, with the rheological-transition weight $z(\phi)$ from [Costa et al. (2009)](https://scixplorer.org/abs/2009GGG....10.3010C/abstract)[^cite-costa2009] |
| $L(P)$ (latent heat) | $L(P) = T_\mathrm{fus}(P)\,(S_\mathrm{liq}(P) - S_\mathrm{sol}(P))$, EOS-tabulated |

The single-pass evaluation lives in `EntropyPhaseEvaluator._update_eos` (file `src/aragog/eos/entropy_phase.py`), which mirrors SPIDER's `EOSEval_Composite_TwoPhase` (`eos_composite.c:177-290`) so the same blending rules are used in both codes.
The phase-boundary lookups happen exactly once per call, and all derived quantities are computed from the cached intermediates rather than via repeated table calls.

The viscosity blend deserves special note.
[Costa et al. (2009)](https://scixplorer.org/abs/2009GGG....10.3010C/abstract) Figure 2 shows the relative viscosity of crystal-bearing magmas rising by $\sim 10^{10}$ across the rheological transition near the critical melt fraction $\phi_c$, while Aragog's solid/liquid end-member viscosities are $\eta_\mathrm{sol} \sim 10^{21}$ Pa s and $\eta_\mathrm{liq} \sim 10^{-1}$ Pa s by default, giving an end-member contrast of $\sim 10^{22}$ across $\phi$.
This is what makes the partial-melt regime numerically stiff and motivates Aragog's smooth tanh blend over a hard if/else.

## Two-phase flow contributions

In a one-phase formulation, the heat flux is conduction plus convection only.
Two-phase flow adds two additional flux components and one additional source term in the partial-melt regime:

### Gravitational separation of melt and solid

Solid and melt have different densities, so they separate vertically by gravity.
The mass flux of melt (positive upward when melt is buoyant) is

$$
j_\mathrm{grav} = \rho\,\phi(1-\phi)\,v_\mathrm{rel}\,\mathrm{smth}(\phi),\qquad
v_\mathrm{rel} = \frac{(\rho_\mathrm{liq} - \rho_\mathrm{sol})\,g\,K(\phi)}{\eta_\mathrm{liq}}.
$$

The permeability factor $K(\phi)$ spans three asymptotic regimes ([Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract) §2.1):

- **Stokes settling** at high $\phi$: melt is the matrix and solid grains settle as isolated spheres, derived from Stokes' law ([Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract) Eq. 13a).
- **Rumpf-Gupte** at intermediate $\phi$: power-law permeability fit to particle-bed measurements (Rumpf & Gupte, 1971, Chem. Ing. Tech. 43, 367)[^cite-rumpfgupte1971].
- **Blake-Kozeny-Carman** at low $\phi$: melt as a percolating fluid through a near-rigid solid matrix.

The three-regime $\zeta_\mathrm{grav}(\phi)$ formulation that Aragog and SPIDER use is consolidated in Abe (1995)[^cite-abe1995] and reviewed in [Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract) §2.1 Eqs. 13a-c, where the regime transitions are density-ratio-dependent: $\rho_\mathrm{liq}/(11.993\,\rho_\mathrm{sol} + \rho_\mathrm{liq})$ for BKC-to-RG and $\rho_\mathrm{liq}/(0.29624\,\rho_\mathrm{sol} + \rho_\mathrm{liq})$ for RG-to-Stokes.
Aragog hard-codes the equal-density-ratio limits ($\rho_\mathrm{sol} = \rho_\mathrm{liq}$) of these expressions, $\zeta_1 = 0.0769452$ (BKC to RG) and $\zeta_2 = 0.771462$ (RG to Stokes), and blends through them with a tanh switch rather than recomputing per radial node.
The resulting permeability-vs-porosity curve (Figure 1 of [Heat transport](heat_transport.md#permeability-kzeta-across-the-three-abe-regimes)) is JAX-differentiable and shifts the transition location by a few percent in $\phi$ relative to the density-ratio-dependent form.
The corresponding heat flux is

$$
F_\mathrm{grav} = j_\mathrm{grav}\,L(P),
$$

with $L(P)$ the EOS-tabulated, pressure-dependent latent heat of fusion.

`bottom_up_grav_sep = true` adds a SPIDER-parity gate that disables $j_\mathrm{grav}$ below the rheological transition until the solid fraction is interconnected, reflecting the physical fact that melt cannot drain through a fully molten lower mantle.

### Chemical mixing of melt fraction

Convective parcels carry melt fraction across the rheological transition.
When parcels with different $\phi$ mix into different ambient $\phi$, latent heat is released or absorbed as the local lever rule is restored.
[Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract) Eq. 9 writes this as a Fick's-law flux on the melt fraction itself, $F_\mathrm{mix} = -\rho T \Delta S_\mathrm{fus}\,\kappa_h\,\partial\phi/\partial r$.
Aragog uses the SPIDER-parity bracket form, which vanishes at the solidus and liquidus by construction (no $\partial\phi/\partial r$ singularity at phase boundaries) and reduces to the Bower (2018) Eq. 9 form in the strict mushy interior.
The bracket form makes the smoothness of the RHS explicit at the cost of a more complex algebraic form:

$$
F_\mathrm{mix} = -\kappa_c\,\rho\,T_\mathrm{fus}\left[
\frac{\partial S}{\partial r}
- \left(\phi\,\tfrac{\partial S_\mathrm{liq}}{\partial P} + (1-\phi)\,\tfrac{\partial S_\mathrm{sol}}{\partial P}\right)\frac{\partial P}{\partial r}
\right]\mathrm{smth}(\phi).
$$

The bracketed expression is the entropy-gradient excess relative to the linear (lever-rule) interpolation between the solidus and liquidus entropy gradients at the local pressure.
Outside the mushy band $\mathrm{smth}(\phi)$ vanishes and so does the flux.
The chemical eddy diffusivity $\kappa_c$ is the same MLT diffusivity as $\kappa_h$ scaled by the configurable `eddy_diffusivity_chemical`.

### Phase-boundary smoothing

The smoothing function $\mathrm{smth}(\phi)$ is what keeps the two two-phase fluxes ($F_\mathrm{grav}$ and $F_\mathrm{mix}$) differentiable at the solidus and liquidus.
Aragog supports two forms:

| `phase_smoothing` | Formula |
|---|---|
| `"tanh"` (default, SPIDER parity) | Two-branch $\mathrm{get\_smoothing}$ with width `matprop_smooth_width = 0.01`; $\mathrm{smth} \approx 1$ in the mushy interior except for tanh skirts of width $\sim 0.01$ in $g\phi$ at $g\phi = 0$ and $g\phi = 1$ |
| `"cubic_hermite"` | $\mathrm{smth}(g\phi) = 16\,g\phi^2(1-g\phi)^2$ for $g\phi \in [0, 1]$; intermediate-$\phi$ damping, slightly more conservative for residual EOS mismatch |

Without smoothing, the raw $\rho\phi(1-\phi)\,v_\mathrm{rel}$ flux drains the CMB cell off the EOS table edge in a single coupling step at first crystallisation; with smoothing, the flux ramps continuously and the JAX Jacobian remains finite across the transition.

## Why this matters: where one-phase models lose physics

A one-phase magma-ocean model can be tuned to give a reasonable bulk cooling rate during the fully-molten phase, where the well-mixed assumption is well justified and a single Nu-Ra closure captures the surface flux.
The lossy regime is the partial-melt phase: the time window between first crystallisation and final solidification, which for an Earth-mass mantle is several hundred kyr to tens of Myr.
In that window:

- Gravitational separation moves melt up and solid down at speeds comparable to the convective velocity in the lowest-permeability regime, transporting both mass and latent heat that a one-phase model cannot represent.
- Chemical mixing carries entropy in the *opposite* direction to convection in parts of the mushy band, partially cancelling the convective flux on either side of the rheological transition; this is what produces the multi-component flux structure shown in Figure 2 of [Heat transport](heat_transport.md#decomposition-on-a-fully-mushy-state).
- The relative viscosity rises by $\sim 10^{10}$ across the rheological transition ([Costa et al. (2009)](https://scixplorer.org/abs/2009GGG....10.3010C/abstract) Figure 2), well outside any single Ra-Nu fit, so the BLT closure that worked at fully-molten state is no longer accurate.
- The latent heat of solidification, encoded in Aragog as a continuous function of $\phi$ via the lever rule, becomes a c$_p$ delta function in a one-phase model, which is either ignored (incorrect) or imposed as a global term that misrepresents its spatial distribution.

In a coupled atmosphere-interior simulation, this is the regime where atmospheric H$_2$O outgassing, surface volatile budgets, and the timing of solidification are decided.
The one-phase formulation can give the right qualitative answer but will miss the resolved time-dependence and the radial structure that controls the coupling to the atmosphere.

## See also

For the algebraic forms of the four heat-flux components Aragog assembles, see [Heat transport](heat_transport.md).
For the closure that determines $\kappa_h$ in the convective flux, see [Mixing-length theory](mixing_length.md).
For the SPIDER cross-check, see [Aragog vs SPIDER](spider_comparison.md).

[^cite-abe1995]: Yutaka Abe, *Basic equations for evolution of partially molten mantle and core*, in The Earth's Central Part: Its Structure and Dynamics, Terra Sci. Pub. Com., 1995.
[^cite-bower2018]: D. J. Bower, et al., *Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets*, Physics of the Earth and Planetary Interiors, 2018.
[^cite-costa2009]: A. Costa, et al., *A model for the rheology of particle-bearing suspensions and partially molten rocks*, Geochemistry, Geophysics, Geosystems, 2009.
[^cite-elkinstanton2008]: L. T. Elkins-Tanton, *Linked magma ocean solidification and atmospheric growth for Earth and Mars*, Earth and Planetary Science Letters, 2008.
[^cite-hamano2013]: K. Hamano, et al., *Emergence of two types of terrestrial planet on solidification of magma ocean*, Nature, 2013.
[^cite-rumpfgupte1971]: H. C. H. Rumpf; A. R. Gupte, *Einflüsse der Porosität und Korngrößenverteilung im Widerstandsgesetz der Porenströmung*, Chemie Ingenieur Technik, 1971.
[^cite-schaefer2016]: L. Schaefer, et al., *Predictions of the atmospheric composition of GJ 1132b*, The Astrophysical Journal, 2016.
