# Aragog model overview

Aragog is a 1-D (radial), spherically symmetric thermal evolution model for rocky planetary mantles in solid, fully molten, or partially molten states. The state of the mantle is represented by a single prognostic variable: the **specific entropy $S(r,t)$** at staggered nodes. Temperature, density, melt fraction, heat capacity, thermal expansivity, and the adiabatic gradient are diagnostic quantities derived from $(P, S)$ via a tabulated equation of state. This formulation absorbs the latent heat of fusion into the entropy axis of the EOS table rather than expressing it as a $c_p$ spike across the solidus and liquidus, and so handles phase boundaries without an effective heat-capacity divergence.

## What Aragog solves

### Entropy balance in integral form

The mantle is treated as a spherical shell between the core-mantle boundary (CMB) at $r = r_\mathrm{cmb}$ and the surface at $r = r_\mathrm{top}$. The integral entropy balance over a control volume $V$ with boundary $\partial V$ is

$$
\int_V \rho T \left.\frac{\partial S}{\partial t}\right|_{\xi}\, dV
= -\int_{\partial V} \mathbf{F}\cdot\mathbf{n}\, dS
+ \int_V \rho H\, dV,
$$

where $\mathbf{F}$ is the radial heat flux $[\mathrm{W\,m^{-2}}]$ and $H$ is the specific internal heating rate $[\mathrm{W\,kg^{-1}}]$. The capacitance multiplying $\partial S/\partial t$ is $\rho T$, not $\rho c_p$: the thermodynamic identity $T\,dS = c_p\,dT - (\alpha T/\rho)\,dP$ is enforced by the EOS lookups rather than by an explicit $c_p$ in the time-derivative term. Each finite-volume cell coincides with a material volume of constant mass; species fluxes (melt vs solid) may exist internally but their net mass flux through cell faces is zero.

### Semi-discrete finite-volume form

Discretising the mantle into $N$ radial shells produces, for each staggered node $i$,

$$
(\rho T V)_i \left.\frac{\partial S}{\partial t}\right|_i
= -F_{i+1/2}\,A_{i+1/2} + F_{i-1/2}\,A_{i-1/2} + \rho_i H_i V_i,
$$

with $A_{i\pm 1/2} = 4\pi r_{i\pm 1/2}^2$ and $V_i = \tfrac{4}{3}\pi(r_{i+1/2}^3 - r_{i-1/2}^3)$. Entropy is held at the $N$ staggered nodes (cell centres), fluxes at the $N+1$ basic nodes (cell faces), and $\rho$, $T$ are read from the EOS at the staggered pressures.

For the `quasi_steady` core BC the unknown vector is $[S_0, \ldots, S_{N-1}]$ of length $N$. The `energy_balance` mode appends the boundary entropy gradient $\partial S/\partial r |_\mathrm{cmb}$ as an additional ODE state variable (length $N+1$), and the `gradient` mode replaces the entropy block with its gradient and adds the surface entropy as a closure (length $N+2$).

## Coordinates and geometry

### Radial symmetry

All fields vary with radius only; vector quantities reduce to their radial component.

### Optional mass coordinate $\xi$

Aragog can run on a mesh that is uniform in radius (`mass_coordinates = false`) or uniform in mass coordinate $\xi$ (`mass_coordinates = true`). The mass coordinate is defined by

$$
\xi(r) = \left( 3\int_{r_\mathrm{cmb}}^{r} \frac{\rho^*(r')}{\rho^*_\mathrm{planet}}\,r'^2\,dr' + \xi_\mathrm{cmb}^3 \right)^{1/3},
$$

where $\rho^*(r)$ is a pseudo-density profile derived from the configured pressure-density relation. In mass-coordinate mode the spatial radii at the basic nodes are recovered by a Newton solve at construction time so that $\xi$ is uniformly spaced. Spatial gradients are then converted via

$$
\frac{\partial \psi}{\partial r}
= \frac{\rho^*(r)}{\rho^*_\mathrm{planet}}\left(\frac{r}{\xi}\right)^2 \frac{\partial \psi}{\partial \xi}.
$$

When `mass_coordinates = false`, $\xi$ is identified with $r$ and the chain rule above reduces to the identity.

## Heat fluxes

The total radial heat flux at each basic node is the sum of four configurable contributions:

$$
F_\mathrm{tot} = F_\mathrm{cond} + F_\mathrm{conv} + F_\mathrm{grav} + F_\mathrm{mix}.
$$

### Conduction

The Fourier flux is rewritten in entropy-gradient form using the thermodynamic identity $\partial T/\partial r |_S = (T/c_p)\,\partial S/\partial r$ and the EOS-tabulated isentropic temperature gradient $\partial T/\partial P |_S$:

$$
F_\mathrm{cond}
= -k\left[\frac{T}{c_p}\,\frac{\partial S}{\partial r} + \left.\frac{\partial T}{\partial P}\right|_S \frac{\partial P}{\partial r}\right].
$$

The pressure gradient $\partial P/\partial r$ is taken from the configured pressure profile (Adams-Williamson or external mesh file). When the entropy gradient vanishes the conduction flux reduces to the adiabatic part alone.

### Convection (mixing-length theory)

Convection is parameterised as eddy diffusion of entropy:

$$
F_\mathrm{conv} = \rho T \kappa_h \max\!\left(-\frac{\partial S}{\partial r},\,0\right).
$$

The instability criterion in the entropy formulation is simply $\partial S/\partial r < 0$. The $\max$ form is implemented as a smooth approximation (rather than a hard switch) so that the BDF Jacobian remains continuous through the onset of convection.

The eddy diffusivity $\kappa_h$ is computed from a mixing length $l(r)$ and a regime-dependent velocity scale. The viscous and inviscid limits of Abe (1993) are blended via a $\tanh$ on the cell Reynolds number around $Re_\mathrm{crit} = 9/8$:

$$
\kappa_h = l\,\Big[(1-w)\,v_\mathrm{visc} + w\,v_\mathrm{inv}\Big],\qquad
w = \tfrac{1}{2}\Big[1 + \tanh\!\big((\mathrm{Re} - Re_\mathrm{crit})/\Delta\big)\Big],
$$

with a narrow blend width $\Delta = 0.01\,Re_\mathrm{crit}$. A configurable floor `kappah_floor` sets a phase-modulated lower bound on $\kappa_h$. Mixing lengths are either the distance to the nearest boundary (`mixing_length_profile = "nearest_boundary"`) or a constant fraction of the mantle thickness.

### Gravitational separation of melt

In the partially molten regime, melt and solid separate vertically by gravity. The separation mass flux is

$$
j_\mathrm{grav} = \rho\,\phi(1-\phi)\,v_\mathrm{rel}\,\mathrm{smth}(\phi),\qquad
v_\mathrm{rel} = \frac{(\rho_m - \rho_s)\,g\,K(\phi)}{\eta_m},
$$

where $K(\phi)$ is a Stokes-or-Darcy permeability set by the configured `grain_size`, and $\mathrm{smth}(\phi)$ is a smoothing function that vanishes outside the mushy band. The default cubic-Hermite smoothing $16\,g\phi^2(1-g\phi)^2$ keeps the flux differentiable at the solidus and liquidus; the alternative tanh form reproduces SPIDER's `get_smoothing` for parity validation. The corresponding heat flux is

$$
F_\mathrm{grav} = j_\mathrm{grav}\,L(P),
$$

with $L(P)$ the EOS-tabulated, pressure-dependent latent heat of fusion. When `bottom_up_grav_sep = true`, a SPIDER-parity bottom-up gating disables the separation flux below the rheological transition until the solid fraction is interconnected.

### Chemical mixing of melt fraction

The chemical mixing flux follows the SPIDER-parity bracket form:

$$
F_\mathrm{mix} = -\kappa_c\,\rho\,T_\mathrm{fus}\left[
\frac{\partial S}{\partial r}
- \left(\phi\,\frac{\partial S_\mathrm{liq}}{\partial P} + (1-\phi)\,\frac{\partial S_\mathrm{sol}}{\partial P}\right)\frac{\partial P}{\partial r}
\right]\mathrm{smth}(\phi),
$$

where $\kappa_c$ is the chemical eddy diffusivity and $T_\mathrm{fus}$ is the local fusion temperature. The bracketed expression is the entropy-gradient excess relative to the linear interpolation between the solidus and liquidus entropy gradients along the local pressure profile. Outside the mushy band $\mathrm{smth}(\phi)$ vanishes and so does the flux.

## Internal heating

The specific heating $H$ at staggered nodes is the sum of two contributions:

$$
H = H_\mathrm{radio} + H_\mathrm{tidal}.
$$

- **Radiogenic heating.** A sum over user-configured isotopes,
  $$
  H_\mathrm{radio} = \sum_i \chi_i\,\varphi_i\,\exp\!\left(-\ln 2\,\frac{t-t_0}{\tau_{1/2,i}}\right),
  $$
  with mass fraction $\chi_i$, specific power $\varphi_i$, and half-life $\tau_{1/2,i}$.

- **Tidal heating.** A user-supplied scalar or per-node array passed through the `tidal_array` configuration key.

The volumetric work associated with phase segregation is carried implicitly by the divergence of the $\Delta h$-weighted mass-flux contributions to `_heat_flux` and is not added as a separate volumetric source; see [Heat transport](heat_transport.md#internal-heating).

## Boundary conditions

### Outer (surface) boundary

Five outer BC modes are defined; modes 1, 4, and 5 are implemented in the entropy solver:

| Mode | Description |
|------|-------------|
| 1 | Grey-body atmosphere: $F_\mathrm{top} = \varepsilon\sigma(T_\mathrm{top}^4 - T_\mathrm{eqm}^4)$ with optional UTBL correction |
| 2 | Zahnle steam atmosphere (not implemented) |
| 3 | Atmodeller coupling (not implemented; PROTEUS uses prescribed flux for atmodeller-driven runs) |
| 4 | Prescribed flux, updated per coupling step from the helpfile |
| 5 | Prescribed temperature |

### Inner (CMB) boundary

| Mode | Description |
|------|-------------|
| 1 | Core cooling: flux determined by the selected `core_bc` formulation |
| 2 | Prescribed flux |
| 3 | Prescribed temperature |

The `core_bc` selector chooses how the core energy balance is closed when `inner_boundary_condition = 1`:

| Mode | State vector | Formulation |
|------|--------------|-------------|
| `quasi_steady` | length $N$ | Alpha-factor heat-flux partition between the bottom mantle cell and the core, weighted by heat-capacity ratio. Default. |
| `energy_balance` | length $N+1$ | The CMB entropy gradient is an ODE state variable evolved by SPIDER's `bc.c:76-131` formula; produces SPIDER bit-parity. |
| `gradient` | length $N+2$ | Entropy gradient as the primary state field; $S$ is reconstructed by cumulative integration from the surface inward. |
| `bower2018` | length $N+1$ | $T_\mathrm{core}$ as an ODE state with conduction-only $F_\mathrm{cmb}$. Retained for parity testing only; not recommended. |

## Pressure and density profiles

Pressure $P(r)$ is treated as time-invariant. Two options are provided:

- **Adams-Williamson (`eos_method = 1`).** A pseudo-density $\rho^*(P) = \rho^*_\mathrm{top}\exp(P/K_S)$ closed against $dP/dr = -\rho^*g$ yields a closed-form $P(r)$. The bulk modulus $K_S$ and surface density are configurable.
- **External mesh file (`eos_method = 2`).** A four-column file (radius $r$, pressure $P$, density $\rho$, gravity $g$) is loaded at construction time and interpolated onto the basic and staggered nodes. This is the path used by PROTEUS for Zalmoxis-derived structure: the per-node gravity profile from the file replaces the scalar `gravitational_acceleration`, and the MLT buoyancy cascade picks up the radial dependence of $g$ automatically.

## Numerical methods

### Spatial discretisation

- Staggered finite-volume mesh on $N+1$ basic nodes and $N$ staggered nodes.
- The entropy gradient at basic nodes is the SPIDER-parity centred difference of staggered $S$ in uniform $\xi$-space, then chain-ruled through $d\xi/dr$. Boundary values copy from the nearest interior node; in `energy_balance` mode the CMB boundary value is overridden by the state-vector entry.
- A configurable phase-boundary smoothing function gates the gravitational-separation and chemical-mixing fluxes outside the mushy band. The default is `phase_smoothing = "tanh"` (SPIDER-parity, two-branch tanh of width `matprop_smooth_width`); `phase_smoothing = "cubic_hermite"` is available as an alternative.

### Time integration

The default integrator is SUNDIALS CVODE via `scikits.odes` (`solver_method = "cvode"`), the same modified-Newton, cached-Jacobian solver SPIDER uses. CVODE is paired by default with the JAX-derived analytic Jacobian (`use_jax_jacobian = true`), built by tracing the pure-functional flux computation in `aragog.jax.phase` with `jax.jacrev`. When `scikits.odes` is not installed the solver falls back to scipy `Radau` and emits a warning at solve time; scipy `BDF` is also selectable via `solver_method = "bdf"`.

The state and time are nondimensionalised internally before being passed to the integrator (entropy reference $S_\mathrm{ref}$ and time reference $t_\mathrm{ref}$) to avoid precision loss in the BDF tolerance control. Physical units are restored on output.

A phase-aware step-size policy reduces `max_step` to one year whenever any cell is within $200~\mathrm{J\,kg^{-1}\,K^{-1}}$ of a phase boundary or sits inside the mushy band; once the mantle is fully solid (mean $\phi < 0.01$) the absolute tolerance is relaxed by a factor of ten to avoid integrator stalling.

### Initial condition

Three modes are provided via `initial_condition`:

1. Linear $T(r)$ between `surface_temperature` and `basal_temperature`, mapped to $S(r)$ via the EOS at the staggered pressures.
2. User-supplied $T(r)$ or $S(r)$ from a two-column file (`init_file`).
3. Adiabatic profile, integrated upward from the bottom temperature using $\partial T/\partial r |_S = -g\alpha T/c_p$.

In coupled PROTEUS runs the wrapper bypasses the file-based path and calls `EntropySolver.set_initial_entropy()` directly with the entropy profile inverted from a Zalmoxis $T(r)$ via the EOS.

## Assumptions

Aragog is intended as a fast 1-D mantle thermal evolution model. The core simplifications are:

- Spherical symmetry (1-D radial).
- Local thermal equilibrium between solid and melt (single entropy field).
- Time-invariant pressure and gravity profiles (set at construction; not co-evolved).
- Parameterised convection via mixing-length eddy diffusion of entropy, not explicit momentum equations.
- Mixed-phase physics through a smooth-clipped melt fraction $\phi(P, S)$, EOS-derived latent heat $L(P)$, and SPIDER-parity smoothing of the segregation and mixing fluxes.

## Symbols and units

| Symbol | Unit | Meaning |
|---|---|---|
| $r$ | m | Radius |
| $\xi$ | m | Mass coordinate (uniform-mass mesh option) |
| $t$ | s | Time |
| $S$ | J/kg/K | Specific entropy (prognostic) |
| $T$ | K | Temperature (diagnostic from EOS) |
| $P$ | Pa | Pressure |
| $\rho$ | kg/m³ | Density |
| $\rho_s$, $\rho_m$ | kg/m³ | Solid, liquid densities |
| $c_p$ | J/kg/K | Specific heat capacity |
| $\alpha$ | 1/K | Thermal expansivity |
| $k$ | W/m/K | Thermal conductivity |
| $g$ | m/s² | Gravitational acceleration |
| $F$ | W/m² | Radial heat flux (basic-node) |
| $F_\mathrm{cond}$, $F_\mathrm{conv}$, $F_\mathrm{grav}$, $F_\mathrm{mix}$ | W/m² | Per-component radial heat fluxes |
| $H$ | W/kg | Specific internal heating |
| $\phi$ | -- | Melt mass fraction |
| $S_\mathrm{sol}(P)$, $S_\mathrm{liq}(P)$ | J/kg/K | Solidus and liquidus entropies |
| $L(P)$ | J/kg | Pressure-dependent latent heat of fusion |
| $\kappa_h$, $\kappa_c$ | m²/s | Thermal and chemical eddy diffusivities |
| $l$ | m | Mixing length |
| $\eta$, $\eta_m$ | Pa·s | Dynamic viscosity (mixture, melt) |
| $j_\mathrm{grav}$ | kg/m²/s | Gravitational-separation mass flux |
| $K(\phi)$ | m² | Permeability of the partially molten matrix |
| $K_S$ | Pa | Adiabatic bulk modulus (Adams-Williamson) |
| $\varepsilon$ | -- | Surface emissivity |
| $\sigma$ | W/m²/K⁴ | Stefan-Boltzmann constant |
| $\mathrm{smth}(\phi)$ | -- | Phase-boundary smoothing function (cubic-Hermite or tanh) |

## Code mapping

The formulation maps to the following source modules:

| Component | Source module |
|-----------|---------------|
| Mesh, $\xi$ transform, A-W and external pressure profiles | `aragog.mesh` |
| EOS lookup tables (`EntropyEOS`) and phase evaluator (`EntropyPhaseEvaluator`) | `aragog.eos` |
| Boundary conditions (grey-body, UTBL, prescribed flux/T) | `aragog.solver.boundary` |
| Time integration, CVODE/Radau dispatch, retry-ladder hooks | `aragog.solver.entropy_solver` |
| Per-RHS flux assembly, MLT, gravitational separation, mixing | `aragog.solver.entropy_state` |
| JAX-traceable phase and flux for the analytic Jacobian path | `aragog.jax` |
| Configuration parsing | `aragog.parser`, `aragog.config` |
| Diagnostic helpers (rheological front, global $\phi$) | `aragog.output.diagnostics` |

For the package layout in detail, see [Code architecture](code_architecture.md). For the public API, see the [API reference](../Reference/api/index.md).
