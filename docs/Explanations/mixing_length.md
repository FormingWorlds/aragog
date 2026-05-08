# Mixing-length theory

Magma-ocean convection cannot be resolved as 3-D unsteady flow inside a 1-D thermal evolution code, so the convective heat flux must be closed by a parameterisation.
Two closures dominate the published literature on terrestrial and exoplanet magma oceans.

1. **Mixing-length theory** (MLT) is a *local* closure: at every radial node an eddy diffusivity $\kappa_h$ is built from the local entropy gradient, gravity, density, viscosity, and a mixing length $l(r)$.
   The radial entropy profile $S(r,t)$ is the prognostic variable and the heat flux at each basic node follows from the local state.
   Aragog and [SPIDER](https://github.com/FormingWorlds/SPIDER) ([Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract)) use MLT.
   Earlier MLT-style magma-ocean work goes back to [Abe (1993)](https://scixplorer.org/abs/1993GMS....74...41A/abstract).

2. **Boundary-layer theory** (BLT), also called *parameterised convection*, is a *global* closure: the interior is taken to be well mixed and near isentropic, and the heat flux out of the layer is fixed by a Nusselt-Rayleigh (Nu-Ra) scaling across a thin thermal boundary layer (TBL) at the surface (and sometimes the core-mantle boundary).
   The radial structure collapses to one or two state variables (a potential temperature, optionally a melt-fraction front).
   This closure underlies the magma-ocean models of [Elkins-Tanton (2008)](https://scixplorer.org/abs/2008E%26PSL.271..181E/abstract), [Hamano et al. (2013)](https://scixplorer.org/abs/2013Natur.497..607H/abstract), [Schaefer et al. (2016)](https://scixplorer.org/abs/2016ApJ...829...63S/abstract), and the parameterised lineage reviewed by [Solomatov (2007)](https://scixplorer.org/abs/2007evea.book...91S/abstract).

This page describes what MLT does inside Aragog, what BLT does in those companion codes, and where the two approaches make different physical statements.
For the algebraic formulas of the four heat flux components Aragog assembles, see [Heat transport](heat_transport.md).
For the SPIDER cross-check, see [Aragog vs SPIDER](spider_comparison.md).

## Mixing-length theory inside Aragog

MLT was developed for stellar interiors by [Vitense (1953)](https://scixplorer.org/abs/1953ZA.....32..135V/abstract) and adapted to terrestrial magma oceans by [Abe (1993)](https://scixplorer.org/abs/1993GMS....74...41A/abstract); the critical Reynolds number $\mathrm{Re}_\mathrm{crit} = 9/8$ that Aragog inherits comes via Abe (1995) and is documented in [Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract) §2.1.
The convective heat flux at a basic node is parameterised as eddy diffusion of entropy:

$$
F_\mathrm{conv} = \rho\,T\,\kappa_h\,\max\!\left(-\frac{\partial S}{\partial r},\ 0\right).
$$

The instability criterion is $\partial S/\partial r < 0$; the $\max(\cdot, 0)$ ramp gates the flux on stably-stratified cells.
The buoyancy driver is the locally *superadiabatic* entropy gradient.
In Aragog's entropy form there is no explicit subtraction of an adiabat: the adiabatic state is $\partial S/\partial r = 0$ by definition, so the entropy gradient itself measures the deviation from the adiabat.

The eddy diffusivity is

$$
\kappa_h = l\,\big[(1-w)\,v_\mathrm{visc} + w\,v_\mathrm{inv}\big],
$$

with viscous and inviscid velocity scales

$$
v_\mathrm{visc} = \frac{\alpha\,g\,T\,(-\partial S/\partial r)\,l^3}{18\,\nu\,c_p},\qquad
v_\mathrm{inv}  = \left[\frac{\alpha\,g\,T\,(-\partial S/\partial r)\,l^2}{16\,c_p}\right]^{1/2},
$$

and a smooth blend $w(\mathrm{Re}) = \tfrac{1}{2}\big[1 + \tanh\!\big((\mathrm{Re}-\mathrm{Re}_\mathrm{crit})/\Delta\big)\big]$ on the cell Reynolds number $\mathrm{Re} = v_\mathrm{visc}\,l/\nu$, with $\mathrm{Re}_\mathrm{crit} = 9/8$ and a narrow blend width $\Delta = 0.01\,\mathrm{Re}_\mathrm{crit}$.
The viscous limit recovers the laminar Stokes-like scaling $\kappa_h \propto l^4/\nu$ at low Re, the inviscid limit recovers the free-fall scaling $\kappa_h \propto l^2 (\alpha g T \,|\partial S/\partial r|/c_p)^{1/2}$ at high Re.
This is the [Abe (1993)](https://scixplorer.org/abs/1993GMS....74...41A/abstract) two-regime form; SPIDER implements it as a hard `if/else` on Re, Aragog as a tanh blend so the JAX Jacobian stays smooth.

The mixing length defaults to `nearest_boundary`,

$$
l(r) = \min(r_\mathrm{top}-r,\ r - r_\mathrm{cmb}),
$$

the classical bounded-eddy profile that vanishes at both boundaries and reaches half the layer thickness at mid-mantle.
A `constant` option (a uniform fraction of the mantle thickness) is also available; see [Heat transport](heat_transport.md#mixing-length-profile).
A phase-modulated diffusivity floor `kappah_floor` activates only where melt fraction is non-trivial, mirroring the SPIDER convention; see [Heat transport](heat_transport.md#phase-modulated-floor).

The implementation is inlined in `EntropyState.update` (numpy path, `src/aragog/solver/entropy_state.py:441-557`) and packaged as `aragog.jax.phase.compute_mlt` (JAX path), with bit-identity tested in `tests/test_jax_entropy.py`.

## Boundary-layer theory in companion codes

BLT replaces the radial profile with one (or a few) lumped variables.
PROTEUS exposes the BLT path via the dedicated boundary-layer interior module (`interior_energetics.module = 'dummy'` with the BL parameterisation enabled), so a coupled run can pick BLT instead of MLT without a separate code change; the two paths share PROTEUS's atmosphere coupling and configuration surface.
The classical magma-ocean closure ([Solomatov (2007)](https://scixplorer.org/abs/2007evea.book...91S/abstract), reviewing [Solomatov & Stevenson (1993)](https://scixplorer.org/abs/1993JGR....98.5375S/abstract) and earlier scalings) writes the surface flux as

$$
F_\mathrm{surf} \;=\; \rho_\mathrm{ml}\,c_p\,\frac{T_\mathrm{pot} - T_\mathrm{surf}}{\delta_\mathrm{TBL}/u_\mathrm{conv}}\;\approx\;k\,\frac{\Delta T_\mathrm{TBL}}{\delta_\mathrm{TBL}},
$$

with $\delta_\mathrm{TBL}$ and $u_\mathrm{conv}$ set by Nu-Ra scaling

$$
\mathrm{Nu} = a\,\mathrm{Ra}^{\beta},\qquad \mathrm{Ra} = \frac{\alpha g \Delta T D^3}{\kappa\,\nu},
$$

with $\beta = 1/3$ for soft turbulence and $\beta \to 2/7$ for hard turbulence in the laboratory regime; for internally heated layers an $\mathrm{Ra}_H$ form is used.
The bulk magma-ocean temperature evolves through a single energy balance,

$$
\rho V c_p\,\frac{dT_\mathrm{pot}}{dt} \;=\; -A\,F_\mathrm{surf} + Q_\mathrm{int},
$$

with $F_\mathrm{surf}$ set by the BL closure (and the atmospheric column above), $Q_\mathrm{int}$ the integrated radiogenic and tidal heating, and the radial structure reconstructed from the assumed adiabat.
[Elkins-Tanton (2008)](https://scixplorer.org/abs/2008E%26PSL.271..181E/abstract) couples this lumped magma-ocean evolution to a grey-body atmosphere with volatile build-up; [Schaefer et al. (2016)](https://scixplorer.org/abs/2016ApJ...829...63S/abstract) uses an equivalent closure and adds water photolysis and hydrogen escape on top.
Both treat the magma ocean's interior as one cell.

## What the two closures actually assume

The conceptual difference is not "MLT vs BLT" as alternative formulas for the same quantity.
The two closures resolve different physical states.

| | MLT (Aragog, SPIDER) | BLT / parameterised (Elkins-Tanton 2008, Schaefer 2016) |
|---|---|---|
| State variable | $S(r,t)$ on a radial mesh | Bulk $T_\mathrm{pot}(t)$ (sometimes plus a solidification front) |
| Where convection is closed | Per cell, locally | Across the upper TBL, globally |
| Adiabaticity of interior | Emerges from the solution; not assumed | Imposed by closure |
| Phase boundaries | Resolved per cell, smoothed at solidus/liquidus | Read from depth-T table along the assumed adiabat |
| Viscous transition | Captured (viscous + inviscid blend on Re) | Usually outside the assumed scaling regime |
| Computational class | Stiff PDE in $S(r,t)$ | Lumped ODE in $T_\mathrm{pot}(t)$ |
| Ra-Nu interpretation | Implicit, recovered as a diagnostic | Explicit, imposed as the closure |
| Required input | $l(r)$, local $\alpha$, $\nu$, $c_p$, $\partial S/\partial r$ | Global Ra (or $T_\mathrm{surf}$ and $\delta_\mathrm{TBL}$) |

A more fundamental difference, also worth noting: MLT closes the convective heat transport on a *single* representative wavenumber of the convective spectrum (typically the most unstable mode, or the one carrying the most energy at saturation), so it captures the contribution of one mode and ignores the rest of the inertial range.
BLT, by contrast, closes the surface heat flux globally through the upper thermal boundary layer thickness, which is set by the *integrated* turbulent transport across all wavenumbers.
As a consequence, MLT typically *under*-estimates the true convective flux (it misses the integrated contribution of the secondary modes) while BLT typically *over*-estimates it (it lumps all transport into a single Ra-Nu scaling that cannot resolve the partial-melt regime).
The two errors point in opposite directions, and there is no general theorem that says either one bounds the truth from the same side across the whole magma-ocean parameter regime.

In the high-$\mathrm{Ra}$, fully-molten, isoviscous limit MLT and BLT converge: MLT in the inviscid regime recovers the same free-fall surface-flux scaling that BLT applies as a single boundary condition.
The two closures stop agreeing when:

- partial melt rheology becomes important (the rheological transition near $\phi \approx 0.4$ has a viscosity jump of $\sim 10^{15}$, well outside any single Ra-Nu fit);
- gravitational separation of melt and solid carries non-negligible mass and entropy flux ($F_\mathrm{grav}$ in [Heat transport](heat_transport.md#gravitational-separation-of-melt));
- the lower BL evolves on the same timescale as the interior (a coupled core-mantle thermal evolution where $T_\mathrm{cmb}$ and the mantle adiabat coevolve);
- the EOS-driven adiabat is non-trivial and an assumed reference adiabat is no longer accurate.

## Advantages of MLT

- **Resolved radial structure.** The full $S(r,t)$ profile is solved for, so solidification fronts, retained melt pockets, partial-melt rheology, and EOS-resolved adiabats are recovered without an assumed reference state.
- **Native handling of phase transitions.** Solidus and liquidus are smoothed per cell within the same flux assembly, so partial-melt physics enters the heat budget continuously rather than as a switch between regimes.
- **Per-component flux diagnostics.** Conduction, convection, gravitational separation, and chemical mixing fluxes ($F_\mathrm{cond}$, $F_\mathrm{conv}$, $F_\mathrm{grav}$, $F_\mathrm{mix}$) are exposed individually and reconstruct the total to floating-point round-off.
- **Two-way coupling to a multi-phase mantle.** Multi-Myr atmosphere-interior runs in which mantle composition feeds back on the lower boundary remain self-consistent because the radial structure that the lower BL sees is the solved mantle, not an imposed adiabat.
- **Captures the viscous transition.** Both viscous (laminar) and inviscid (turbulent) limits are handled inside the same closure via the smooth blend on Re, so the parameterisation extends across the full range from solid to fully molten without piecing together separate scaling fits.

## Advantages of BLT

- **Cheap.** A single ODE in $T_\mathrm{pot}(t)$ runs in seconds, which makes parameter sweeps over many planets and atmosphere-led parameter studies tractable where a stiff PDE solve is not.
- **Direct contact with laboratory scalings.** The Nu-Ra closure matches the form of Rayleigh-Bénard experiments and high-Ra simulations, so closure choices and uncertainties translate directly into experimentally measured exponents.
- **Defensible in the fully-molten high-Ra limit.** When the magma ocean is well-mixed and isoviscous, the BLT closure is the one MLT itself converges to in the inviscid limit, so BLT is as accurate as MLT in that regime at a small fraction of the cost.
- **Compact coupling interface.** A single bulk temperature is easier to hand to an atmospheric model than a full radial profile, which simplifies coupling pipelines where the magma ocean is one element of a larger system.

The two formulations are complementary, not competing.
BLT makes a defensible high-Ra assumption that an MLT solution can validate or invalidate by inspection of the radial profile, and most published MLT runs (including Aragog's) sit deep inside the parameter regime where BLT is also defensible during the fully-molten phase.
MLT becomes necessary, not just preferable, in the partial-melt regime and through the rheological transition.

## Aragog-specific choices

The version of MLT in Aragog inherits five concrete choices that distinguish it from a textbook stellar-interior MLT and from a different magma-ocean MLT implementation.

- **Entropy form, not temperature form.**
  The buoyancy driver is the superadiabatic entropy gradient $\max(-\partial S/\partial r, 0)$, not the superadiabatic temperature gradient.
  The two are equivalent through the thermodynamic identity $\partial T/\partial r|_S = (T/c_p)\,\partial S/\partial r$, but the entropy form keeps the EOS-tabulated $(P,S)$ lookup as the only thermodynamic call inside the RHS.
  An earlier plan to rewrite the prognostic equation in a temperature-only formalism has been deprioritised: the entropy form's tighter coupling to the SPIDER-format P-S property tables and its correct lever-rule blending across the mushy zone outweigh the marginal interpretability gain of a $T(r,t)$ state vector.
  A T-only path remains available through the `const_properties` mode (matching SPIDER's `-use_const_properties` flag) for analytic-mode benchmarks, but it is not the production solver path.

- **Smooth viscous-to-inviscid blend.**
  SPIDER implements the [Abe (1993)](https://scixplorer.org/abs/1993GMS....74...41A/abstract) two-regime form as a hard `if/else` on Re, while Aragog blends the regimes through a $\tanh$ on Re with a narrow width ($\Delta = 0.01\,\mathrm{Re}_\mathrm{crit}$).
  The narrow width keeps the inviscid contribution confined to the convecting regime without leaking into solid layers; widening it has been shown to induce $T_\mathrm{core}$ bistability in test runs.

- **Bounded-eddy mixing length.**
  The default profile $l(r) = \min(r_\mathrm{top}-r,\ r-r_\mathrm{cmb})$ vanishes at both boundaries.
  [Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract) §2.1 motivates this distance-to-nearest-boundary choice as the form for which the calculated convective heat flux best fits experimental measurements (Fig. 3 in Abe 1995; Stothers & Chin 1997).
  A `constant` profile is also available for analytical-mode tests.

- **Phase-modulated diffusivity floor.**
  `kappah_floor` activates only where melt fraction $\phi$ is large enough (tanh ramp around $\phi = 0.4$), so a SPIDER-equivalent floor never spuriously diffuses solid-state cells.
  This avoids the `kappah_floor` interacting with the rheological transition to create a metastable equilibrium for $T_\mathrm{cmb}$.

- **Per-component flux diagnostics.**
  Aragog reports $F_\mathrm{conv}$ separately from the conduction, gravitational-separation, and chemical-mixing fluxes, with reconstruction of the total to floating-point round-off (Figure 2 of [Heat transport](heat_transport.md)).
  This component-wise decomposition is not natural in a BLT formulation, where the single closed-form $F_\mathrm{surf}$ is the only flux available for diagnostics.

## Where to read more

- [Abe (1993)](https://scixplorer.org/abs/1993GMS....74...41A/abstract). *Thermal evolution and chemical differentiation of the terrestrial magma ocean*. Geophysical Monograph 74, American Geophysical Union, pp. 41-54. The two-regime MLT form Aragog and SPIDER both follow.
- Abe, Y. (1995). *Basic equations for evolution of partially molten mantle and core*. In Yukutake, T. (ed.), *The Earth's Central Part: Its Structure and Dynamics*, Terra Sci. Pub. Com., Tokyo, pp. 215-230. Source for $\mathrm{Re}_\mathrm{crit} = 9/8$ via the viscous-inviscid asymptotic match.
- [Abe (1997)](https://scixplorer.org/abs/1997PEPI..100...27A/abstract). *Thermal and chemical evolution of the terrestrial magma ocean*. PEPI 100, 27.
- [Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract). *Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets*. PEPI 274, 49. The SPIDER paper; the entropy-form mantle equation Aragog inherits.
- [Elkins-Tanton (2008)](https://scixplorer.org/abs/2008E%26PSL.271..181E/abstract). *Linked magma ocean solidification and atmospheric growth for Earth and Mars*. EPSL 271, 181. A representative BLT magma-ocean evolution coupled to a grey atmosphere.
- [Hamano et al. (2013)](https://scixplorer.org/abs/2013Natur.497..607H/abstract). *Emergence of two types of terrestrial planet on solidification of magma ocean*. Nature 497, 607.
- [Lichtenberg et al. (2021)](https://scixplorer.org/abs/2021JGRE..12606711L/abstract). *Vertically resolved magma ocean-protoatmosphere evolution*. JGR Planets 126, e06711. SPIDER + atmosphere coupling that Aragog now replaces inside PROTEUS.
- [Schaefer et al. (2016)](https://scixplorer.org/abs/2016ApJ...829...63S/abstract). *Predictions of the atmospheric composition of GJ 1132b*. ApJ 829, 63. A representative BLT magma-ocean coupled to atmospheric photochemistry and escape.
- [Solomatov (2007)](https://scixplorer.org/abs/2007evea.book...91S/abstract). *Magma oceans and primordial mantle differentiation*. In *Evolution of the Earth*, Treatise on Geophysics 9, 91. Review of the Nu-Ra scalings BLT relies on.
- [Solomatov & Stevenson (1993)](https://scixplorer.org/abs/1993JGR....98.5375S/abstract). *Suspension in convective layers and style of differentiation of a terrestrial magma ocean*. JGR 98, 5375.
- [Vitense (1953)](https://scixplorer.org/abs/1953ZA.....32..135V/abstract). *Die Wasserstoffkonvektionszone der Sonne*. Zeitschrift für Astrophysik 32, 135. Origin of the inviscid free-fall MLT velocity scaling that magma-ocean MLT inherits.
