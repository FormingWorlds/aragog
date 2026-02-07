# Aragog model overview

Aragog is a 1‑D (radial), spherically symmetric thermal evolution model for rocky planetary mantles that can be **solid**, **molten**, or **partially molten** (mixed phase). The model assumes **local thermal equilibrium** between solid and melt, so the state is represented by a single prognostic variable: an **effective temperature** $T(r,t)$.

## What Aragog solves

### Integral enthalpy balance

Aragog evolves temperature by enforcing an enthalpy balance over a spherical shell (mantle) between the **core–mantle boundary (CMB)** at $r=r_{\mathrm{cmb}}$ and the **surface** at $r=r_{\mathrm{top}}$. In integral form, for a control volume $V$ with boundary $S$,

$$
\int_V \rho c_p \left.\frac{\partial T}{\partial t}\right|_{\xi}\, dV
=
- \int_S \mathbf{q}\cdot \mathbf{n}\, dS
+ \int_V \Phi\, dV,
$$

where $ \mathbf{q}$ is the heat flux, $\Phi$ the heating rate and $c_p$ the specific heat capacity. The time derivative is taken at **constant mass coordinate** $\xi$ (a Lagrangian-like coordinate; see below).

Each finite-volume cell is assumed to coincide with a **material volume**: its mass is constant and there is **no net mass flux through cell interfaces** in the control-volume sense (species fluxes may exist internally in mixed phase via melt/solid segregation, but their sum is zero).

### Semi-discrete finite-volume form

Discretising the mantle into radial shells (cells), Aragog uses a finite-volume balance per cell $i$:

$$
(\rho c_p V)_i \left.\frac{\partial T}{\partial t}\right|_i
= -\, q_{i+1/2}\, S_{i+1/2} + q_{i-1/2}\, S_{i-1/2} + \Phi_i\, V_i ,
$$

where:

- $S_{i\pm 1/2} = 4\pi r_{i\pm 1/2}^2$ is the spherical area of the outer/inner face,
- $V_i = \frac{4}{3}\pi\left(r_{i+1/2}^3-r_{i-1/2}^3\right)$ is the shell volume,
- $q$ is the **radial heat flux** (positive upward by convention),
- $\Phi$ is a **volumetric heating rate** $[\mathrm{W\,m^{-3}}]$.

The unknown is $T_i(t)$ at cell centers; fluxes are evaluated on faces.

## Coordinates and geometry

### Radial symmetry

All quantities vary with radius only. Vector quantities reduce to their radial component (e.g., $\mathbf{q}\cdot\mathbf{n}\rightarrow q_r$).

### Mass coordinate $\xi$

To track a mesh with constant mass per cell, Aragog can work in a **mass coordinate** $\xi$ (units of length) defined so that equal increments in $\xi$ correspond to equal mass increments (after suitable scaling). A mapping between spatial radius $r$ and $\xi$ is defined using a **pseudo-density** field $\rho^*(r)$:

$$
\xi(r) = \left( 3\int_{r_{\mathrm{cmb}}}^{r} \frac{\rho^*(r')}{\rho^*_{\mathrm{planet}}} r'^2\, dr' + \xi_{\mathrm{cmb}}^3 \right)^{1/3}.
$$

Spatial gradients are converted via

$$
\frac{\partial \psi}{\partial r}
=
\frac{\rho^*(r)}{\rho^*_{\mathrm{planet}}}\left(\frac{r}{\xi}\right)^2
\frac{\partial \psi}{\partial \xi}.
$$

If the mass-coordinate transform is ignored (i.e. on a fixed spatial mesh), the model approximates
$\left.\partial T/\partial t\right|_{\xi} \approx \left.\partial T/\partial t\right|_{r}$.


## Heat fluxes

Aragog represents the total radial heat flux as a sum of four contributions:

$$
q_{\mathrm{tot}} = q_{\mathrm{cd}} + q_{\mathrm{cv}} + q_{\mathrm{cm}} + q_{\mathrm{gm}}.
$$

### Conduction

$$
q_{\mathrm{cd}} = -\lambda\,\frac{\partial T}{\partial r},
$$
with thermal conductivity $\lambda(T,P,\phi)$.

### Parameterised convection (eddy diffusion)

Convection is written as a diffusive flux that relaxes superadiabatic gradients:

$$
q_{\mathrm{cv}} = -\rho c_p \kappa_h\left(
\frac{\partial T}{\partial r} - \left.\frac{\partial T}{\partial r}\right|_S
\right),
$$

where $\kappa_h$ is an **eddy diffusivity** and the adiabatic gradient is

$$
\left.\frac{\partial T}{\partial r}\right|_S = -\frac{g\alpha T}{c_p}.
$$

Convection is only active when the flow is unstable, i.e. the **superadiabatic gradient is negative**:
$$
\left(\frac{\partial T}{\partial r} - \left.\frac{\partial T}{\partial r}\right|_S\right) < 0.
$$

### Enthalpy fluxes from melt transport (mixed phase only)

In partially molten regions, Aragog includes additional enthalpy transport associated with melt redistribution:

$$
q_{\mathrm{cm}} + q_{\mathrm{gm}} = \Delta h\, (j_{\mathrm{cm}} + j_{\mathrm{gm}}),
$$

where $\Delta h$ is latent heat and $j$ are **melt mass fluxes** $[\mathrm{kg\,m^{-2}\,s^{-1}}]$.

- **Convective mixing mass flux** (diffusion of melt fraction):
  $
  j_{\mathrm{cm}} = -\rho \kappa_h \frac{\partial \phi}{\partial r},
  $
  with $\phi$ the melt fraction. 
- **Gravitational separation mass flux** (buoyant percolation/settling):
  $$
  j_{\mathrm{gm}} = \rho\,\phi(1-\phi)\, v_{\mathrm{rel}},
  \qquad
  v_{\mathrm{rel}} = \frac{(\rho_m-\rho_s)gK}{\eta_m},
  $$

where $v_{\mathrm{rel}}$ is the relative velocity between melt and solid; $K$ is the permeability of the mixed-phase region and  $\eta_m$ is the melt viscosity. These melt-transport fluxes are constructed so that melt and solid species fluxes sum to zero (no net mass flux).

## Volumetric heating sources

Aragog allows the following volumetric sources:

$$
\Phi = \Phi_{\mathrm{tidal}} + \Phi_{\mathrm{radio}} + \Phi_{\mathrm{vol}}.
$$

- **Radiogenic heating** is typically space-uniform but time-dependent:
  $$
  \Phi_{\mathrm{radio}} = \sum_i \rho\,\varphi_i\chi_i
  \exp\!\left(-\frac{t-t_0}{\tau_{1/2,i}}\right),
  $$
  where $\varphi_i$, $\chi_i$, and $\tau_{1/2,i}$ are the power per mass, mass fraction, and half-life of isotope $i$.
- **Tidal heating** $\Phi_{\mathrm{tidal}}$ is user-specified (may vary with radius; typically constant in time within a run).
- **Volumetric dilation/compression work** couples to melt transport:
  $$
  \Phi_{\mathrm{vol}} = \rho g\left(\frac{1}{\rho_m}-\frac{1}{\rho_s}\right)(j_{\mathrm{cm}}+j_{\mathrm{gm}}).
  $$


## Thermodynamics and phase state

### Single prognostic variable

Temperature $T$ is the only prognostic variable. All other state variables are diagnostic functions of $T$ and pressure $P(r)$ (and, in mixed phase, melt fraction).

### Melt fraction $\phi$

Aragog uses a piecewise-linear melt mass fraction between a pressure-dependent solidus and liquidus:

$$
\phi =
\begin{cases}
0, & T < T_{\mathrm{sol}}(P),\\[4pt]
\dfrac{T-T_{\mathrm{sol}}}{T_{\mathrm{liq}}-T_{\mathrm{sol}}}, &
T_{\mathrm{sol}} \le T \le T_{\mathrm{liq}},\\[8pt]
1, & T > T_{\mathrm{liq}}(P).
\end{cases}
$$

### Pressure model (pseudo-density)

By default, pressure is computed with an **Adams–Williamson** profile assuming constant gravity and an exponential $\rho^*(P)$ relation:

$$
\rho^*(P) = \rho^*_{\mathrm{top}} \exp\!\left(\frac{P}{B}\right),
\qquad
\frac{dP}{dr} = -\rho^*(P)\, g.
$$

This yields a closed-form $P(r)$. The pseudo-density $\rho^*$ is used to:
- define the pressure field (assumed time-invariant), and
- construct mass-coordinate mappings / constant-mass cells.

It may differ slightly from the actual density $\rho$ used in transport, which varies with phase.


## Transport properties

Thermophysical properties can depend on $T$ and $P$, with separate solid and melt values ($s$ and $m$). In the mixed phase:

- **Mixture density** $\rho$ (harmonic form):
  $$
  \frac{1}{\rho} = \frac{\phi}{\rho_m} + \frac{1-\phi}{\rho_s}.
  $$
- **Mixture conductivity** $\lambda$ (linear):
  $$
  \lambda = \phi\lambda_m + (1-\phi)\lambda_s.
  $$
- **Effective viscosity** transitions sharply near a critical melt fraction $\phi_c^\eta$ using a smoothed log-mixing law.

In mixed phase, $c_p$ and $\alpha$ include dominant contributions from latent heat (via $\partial\phi/\partial T$), producing large effective heat capacity and expansivity.

### Eddy diffusivity $\kappa_h$ (mixing length theory)

$\kappa_h$ is computed from a mixing length $l$ and a velocity scale that depends on flow regime (viscous vs inviscid). A common choice for $l$ is either a constant fraction of mantle thickness or the distance to the nearest boundary:
$$
l(r) = \min(r_{\mathrm{top}}-r,\; r-r_{\mathrm{cmb}}).
$$

## Numerical methods

### Spatial discretisation

- **Finite-volume** in spherical shells.
- **Staggering:** temperature and volumetric terms at cell centers; fluxes at cell faces.
- **Dual-mesh mapping:** linear interpolation to faces and finite differences for gradients in the chosen coordinate.

### Boundary conditions

Aragog supports common thermal boundary types at $r_{\mathrm{cmb}}$ and $r_{\mathrm{top}}$:

- **Neumann (flux) boundary condition:** impose $q_{\mathrm{tot}}$ at a boundary. Boundary state and gradients needed for individual flux components are obtained by linear extrapolation from interior nodes.
- **Dirichlet (temperature) boundary condition:** impose $T$ at a boundary; compute gradients using one-sided differences to recover fluxes.
- **Radiative surface flux (optional):**
  $$
  q_{\mathrm{top}} = \varepsilon\sigma\left(T_{\mathrm{top}}^4 - T_{\mathrm{atm}}^4\right),
  $$
  implemented as an effective flux boundary condition using an extrapolated $T_{\mathrm{top}}$.
- **Core cooling model (optional):** when applying a flux at the CMB, Aragog can estimate $q_{\mathrm{cmb}}$ by coupling the lowermost mantle cell to a lumped core energy balance, assuming $T_{\mathrm{core}}$ is proportional to the lowermost mantle temperature.

### Time integration

The semi-discrete system is advanced using an **implicit, variable-order, variable-step backward differentiation formula (BDF)** method, which is well-suited to the stiffness introduced by conduction, convection, and phase-change effective heat capacities.

### Typical initial condition

A common default is an **adiabatic** initial profile satisfying

$$
\frac{\partial T}{\partial r} = -\frac{g\alpha T}{c_p},
$$

solved numerically when properties vary with depth.

## Assumptions 

Aragog is intended as a computationally efficient mantle thermal evolution tool. Its core assumptions are:

- **Spherical symmetry** (1-D radial).
- **Local thermal equilibrium** between solid and melt (single temperature).
- **Time-invariant pressure field** by default (via pseudo-density and chosen EOS/profile).
- **Parameterised convection** via mixing-length eddy diffusivity, not explicit momentum equations.
- **Mixed-phase physics** represented through melt fraction, latent heat effects, and simplified segregation/mixing fluxes.

## Symbols and units

- **$r$** — Radius (spatial coordinate) \[m\]
- **$\xi$** — Mass coordinate \[m\]
- **$t$** — Time \[s\] (often shown in years in postprocessing)
- **$T$** — Temperature \[K\]
- **$P$** — Pressure \[Pa\]
- **$\rho$** — Density \[kg/m³\]
- **$\rho_s$** — Solid density \[kg/m³\]
- **$\rho_m$** — Melt (liquid) density \[kg/m³\]
- **$c_p$** — Heat capacity \[J/kg/K\]
- **$\alpha$** — Thermal expansivity \[K⁻¹\]
- **$\lambda$** — Thermal conductivity \[W/m/K\]
- **$g$** — Gravitational acceleration \[m/s²\]
- **$\vec{q}$** — Heat flux vector \[W/m²\]
- **$q_{\mathrm{cd}}$** — Conductive heat flux \[W/m²\]
- **$q_{\mathrm{cv}}$** — Convective (eddy-diffusive) heat flux \[W/m²\]
- **$q_{\mathrm{tot}}$** — Total heat flux \[W/m²\]
- **$\Phi$** — Volumetric heating / source term \[W/m³\]
- **$\phi$** — Melt mass fraction \[-\]
- **$T_{\mathrm{sol}}(P)$** — Solidus temperature \[K\]
- **$T_{\mathrm{liq}}(P)$** — Liquidus temperature \[K\]
- **$\kappa_h$** — Eddy diffusivity \[m²/s\]
- **$l$** — Mixing length \[m\]
- **$\eta$** — Dynamic viscosity \[Pa·s\]
- **$\nu$** — Kinematic viscosity \($\nu=\eta/\rho$\) \[m²/s\]
- **$\vec{j}$** — Melt mass flux vector \[kg/m²/s\]
- **$j_{\mathrm{cm}}$** — Convective mixing mass flux \[kg/m²/s\]
- **$j_{\mathrm{gm}}$** — Gravitational separation mass flux \[kg/m²/s\]
- **$v_{\mathrm{rel}}$** — Relative melt–solid velocity \[m/s\]
- **$K$** — Permeability of the partially molten matrix \[m²\]
- **$\Delta h$** — Latent heat of fusion \[J/kg\]
- **$\varepsilon$** — Emissivity \[-\]
- **$\sigma$** — Stefan–Boltzmann constant \[W/m²/K⁴\]
- **$B$** — Adiabatic bulk modulus (used in Adams–Williamson EOS) \[Pa\]


## Code mapping (high level)

### Physical scripts

The formulation corresponds broadly to:

- **Mesh + coordinates + pressure/EOS**: [`mesh.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/mesh.py)

- **Phase and material properties**: [`phase.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/phase.py), [`interfaces.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/interfaces.py)

- **Core model pieces (initial and boundary conditions)**: [`core.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/core.py)

- **Time integration and main solver**: [`solver.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/solver.py)

### Project scripts

Additionally, the code contains:

- **CLI / entry point**: [`cli.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/cli.py)

- **Configuration and parameters**: [`parser.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/parser.py), [`cfg/`](https://github.com/FormingWorlds/aragog/tree/main/aragog/cfg)

  - **Output and plotting**: [`output.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/output.py)

- **Data download (lookup tables)**: [`data.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/data.py)

- **Small utilities**: [`utilities.py`](https://github.com/FormingWorlds/aragog/tree/main/aragog/utilities.py)