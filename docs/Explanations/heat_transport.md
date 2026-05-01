# Heat transport

Aragog assembles the total radial heat flux at each basic node from four independently configurable contributions:

$$
F_\mathrm{tot} = F_\mathrm{cond} + F_\mathrm{conv} + F_\mathrm{grav} + F_\mathrm{mix}.
$$

Each term is gated by a boolean in the `[energy]` section (`conduction`, `convection`, `gravitational_separation`, `mixing`); a flux that is disabled is identically zero. Internal heating sources (radiogenic, dilatation, tidal) are documented separately in [Energy equation](energy_equation.md).

All flux formulas below use the entropy gradient $\partial S/\partial r$ as the primary driver. Temperature, density, heat capacity, thermal expansivity, and the isentropic temperature gradient $(\partial T/\partial P)_S$ are looked up from the EOS at $(P, S)$ on every RHS evaluation.

## Conduction

The Fourier flux is rewritten in entropy form using the thermodynamic identity $\partial T/\partial r |_S = (T/c_p)\,\partial S/\partial r$ and the EOS-tabulated isentropic temperature gradient:

$$
\boxed{\; F_\mathrm{cond}
= -k\left[\frac{T}{c_p}\,\frac{\partial S}{\partial r} + \left.\frac{\partial T}{\partial P}\right|_S \frac{\partial P}{\partial r}\right]. \;}
$$

The pressure gradient is the configured profile (Adams-Williamson or external mesh file). When the entropy gradient vanishes (an isentropic profile) the conduction flux reduces to the adiabatic part alone, which is non-zero because the EOS provides $(\partial T/\partial P)_S$ directly. In the production path with PALEOS tables, $k$ is read from the EOS as a function of $(P, S)$; the `phase_solid.thermal_conductivity` and `phase_liquid.thermal_conductivity` config keys are kept only for the constant-properties analytical mode.

## Convection (mixing-length theory)

Convection is parameterised as eddy diffusion of entropy:

$$
\boxed{\; F_\mathrm{conv} = \rho T \kappa_h \max\!\left(-\tfrac{\partial S}{\partial r},\ 0\right). \;}
$$

The instability criterion is $\partial S/\partial r < 0$. The $\max(\cdot, 0)$ form is implemented as a smooth approximation (a $C^\infty$ ramp) so that the BDF Jacobian remains continuous through the onset of convection. There is no explicit "superadiabatic gradient" subtraction in the entropy formulation: the adiabat corresponds to $\partial S/\partial r = 0$, so the entropy gradient itself measures departures from the adiabat.

### Eddy diffusivity

$\kappa_h$ is the product of a mixing length $l(r)$ and a regime-dependent velocity scale. Following Abe (1993), Aragog blends the viscous and inviscid limits via a $\tanh$ on the cell Reynolds number:

$$
v_\mathrm{visc} = \frac{\alpha g\,(-\partial S/\partial r)\,T\,l^2}{18\,\nu},\qquad
v_\mathrm{inv} = l\left[\frac{\alpha g\,T\,(-\partial S/\partial r)}{c_p}\right]^{1/2},
$$

$$
\kappa_h = l\,[(1-w)\,v_\mathrm{visc} + w\,v_\mathrm{inv}],\qquad
w = \tfrac{1}{2}\Big[1 + \tanh\!\big((\mathrm{Re} - Re_\mathrm{crit})/\Delta\big)\Big],
$$

with $\mathrm{Re} = v_\mathrm{visc}\,l / \nu$, $Re_\mathrm{crit} = 9/8$, and a narrow blend width $\Delta = 0.01\,Re_\mathrm{crit}$. The narrow blend keeps inviscid mixing confined to the convecting regime; widening it leaks inviscid $\kappa_h$ into the solid phase and induces $T_\mathrm{core}$ bistability.

### Mixing length profile

| `mixing_length_profile` | Formula |
|-------------------------|---------|
| `"nearest_boundary"` | $l(r) = \min(r_\mathrm{top} - r,\ r - r_\mathrm{cmb})$ |
| `"constant"` | $l$ = configured fraction of mantle thickness |

### Phase-modulated floor

A phase-aware floor on $\kappa_h$ is set by `kappah_floor`:

$$
\kappa_h \;\to\; \max\!\big(\kappa_h,\ \kappa_h^\mathrm{floor}(\phi)\big),
$$

where $\kappa_h^\mathrm{floor}(\phi)$ is modulated by melt fraction so that the floor activates only in convectively suppressed solid regions. The thermal and chemical eddy diffusivities also accept independent scale factors:

| Key | Meaning |
|-----|---------|
| `eddy_diffusivity_thermal` | Multiplier on $\kappa_h$. Negative values pin $\kappa_h$ to the absolute value (SPIDER convention) |
| `eddy_diffusivity_chemical` | Multiplier on $\kappa_c$. Negative values pin $\kappa_c$ to the absolute value |

## Gravitational separation of melt

In the partially molten regime, melt and solid separate vertically by gravity. The mass flux is

$$
\boxed{\; j_\mathrm{grav} = \rho\,\phi(1-\phi)\,v_\mathrm{rel}\,\mathrm{smth}(\phi),\qquad
v_\mathrm{rel} = \frac{(\rho_m - \rho_s)\,g\,K(\phi)}{\eta_m}, \;}
$$

with $K(\phi)$ a Stokes-or-Darcy permeability and $\eta_m$ the melt viscosity. The corresponding heat flux is

$$
F_\mathrm{grav} = j_\mathrm{grav}\,L(P),
$$

where $L(P)$ is the EOS-tabulated, pressure-dependent latent heat of fusion.

### Phase-boundary smoothing

The smoothing function $\mathrm{smth}(\phi)$ vanishes outside the mushy band, keeping the flux differentiable at the solidus and liquidus. Two forms are configurable:

| `phase_smoothing` | Formula |
|-------------------|---------|
| `"cubic_hermite"` (default) | $\mathrm{smth}(g\phi) = 16\,g\phi^2(1 - g\phi)^2$ for $g\phi \in [0, 1]$ |
| `"tanh"` | SPIDER-style $\mathrm{get\_smoothing}$ (parity validation) |

Here $g\phi$ is the un-truncated two-phase fraction at the staggered cell below the basic-node interface. Without smoothing, a raw $\rho\phi(1-\phi)v_\mathrm{rel}$ flux drains the CMB cell off the EOS table edge in a single coupling step at first crystallisation.

### Bottom-up gating

When `bottom_up_grav_sep = true`, a SPIDER-parity bottom-up gate disables $j_\mathrm{grav}$ below the rheological transition until the solid fraction is interconnected. This prevents spurious upward percolation of the first solid grains and reflects the physics that melt cannot drain through a fully molten lower mantle.

## Chemical mixing of melt fraction

Chemical mixing acts as a diffusive flux that relaxes the entropy gradient toward the local lever-rule prediction. The SPIDER-parity bracket form is

$$
\boxed{\; F_\mathrm{mix} = -\kappa_c\,\rho\,T_\mathrm{fus}\left[
\frac{\partial S}{\partial r}
- \left(\phi\,\tfrac{\partial S_\mathrm{liq}}{\partial P} + (1-\phi)\,\tfrac{\partial S_\mathrm{sol}}{\partial P}\right)\frac{\partial P}{\partial r}
\right]\mathrm{smth}(\phi). \;}
$$

The bracketed expression is the entropy-gradient excess relative to the linear (lever-rule) interpolation between the solidus and liquidus entropy gradients at the local pressure. Outside the mushy band $\mathrm{smth}(\phi)$ vanishes and so does the flux. The chemical eddy diffusivity $\kappa_c$ is the same MLT diffusivity as $\kappa_h$ scaled by `eddy_diffusivity_chemical`.

The term enters the entropy equation as a heat flux: even though the flux carries melt mass (not energy directly), the latent heat associated with the redistributed melt is what shows up in the energy budget at staggered nodes.

## Internal heating

Internal heating contributes to the entropy equation through the source term $\rho H$ in the integral balance. The three contributions are summarised here; the radiogenic-decay model and the per-isotope configuration are discussed in [Energy equation](energy_equation.md).

- **Radiogenic.** $H_\mathrm{radio} = \sum_i \chi_i \varphi_i \exp(-\ln 2\,(t - t_0)/\tau_{1/2,i})$, time-dependent and (typically) space-uniform.
- **Dilatation $P\,dV$.** Work done when melt of different density is transported across a pressure gradient by chemical mixing or gravitational separation:
  $$
  \Phi_\mathrm{vol} = g\,\left(\frac{1}{\rho_m} - \frac{1}{\rho_s}\right)\,(j_\mathrm{mix} + j_\mathrm{grav}),
  $$
  with $j_\mathrm{mix}$ the convective-mixing mass flux and $j_\mathrm{grav}$ the gravitational-separation mass flux. Only active when both `dilatation = true` and `gravitational_separation = true`.
- **Tidal.** Per-staggered-node array supplied through `tidal_array`; broadcast scalar or length-$N$ array.

## Per-component flux output

The basic-node flux contributions are exposed individually on `SolverOutput`:

| Field | Component |
|-------|-----------|
| `jcond_b` | $F_\mathrm{cond}$ |
| `jconv_b` | $F_\mathrm{conv}$ |
| `jgrav_b` | $F_\mathrm{grav}$ |
| `jmix_b` | $F_\mathrm{mix}$ |
| `heat_flux` | $F_\mathrm{tot}$ |
| `dSdr_b` | $\partial S/\partial r$ at basic nodes |

Per-staggered-node heating is in `heating` (sum of the three contributions).
