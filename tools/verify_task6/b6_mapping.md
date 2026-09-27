# Benchmark B6 Mapping: Euen et al. (2023) to aragog

This document defines the mathematical and physical mapping between the 3-D spherical shell thermal convection benchmark of Euen et al. (2023) and the 1-D radial mixing-length convection solver `aragog`.

## 1. Domain Geometry and Curvature

Euen et al. (2023) define a non-dimensional 3-D spherical shell:
- Inner boundary radius: $r_b = 0.55$
- Outer boundary radius: $r_t = 1.0$
- Shell thickness: $D = r_t - r_b = 0.45$
- Radius ratio: $f = r_b / r_t = 0.55$

In `aragog`, the domain geometry is set by spherical coordinates:
- `mesh.inner_radius`: $r_{\mathrm{cmb}} = r_b \cdot L_0$
- `mesh.outer_radius`: $r_{\mathrm{surf}} = r_t \cdot L_0$
- Shell thickness: $D = r_{\mathrm{surf}} - r_{\mathrm{cmb}} = 0.45 \cdot L_0$
- Radial grid: $n = 100$ cells with uniform or stag/basic staggered spacing.

## 2. Governing Equations and Dimensionless Numbers

Euen et al. (2023) solve the incompressible Boussinesq equations without internal heating:
1. Mass conservation: $\nabla \cdot \mathbf{v} = 0$
2. Momentum conservation: $0 = -\nabla P + \nabla \cdot [\eta (\nabla \mathbf{v} + \nabla \mathbf{v}^T)] + \mathrm{Ra}\, T\, \hat{\mathbf{r}}$
3. Energy conservation: $\frac{\partial T}{\partial t} + \mathbf{v} \cdot \nabla T = \nabla^2 T$

### Rayleigh Number
The Rayleigh number is defined at the mid-temperature $T = 0.5$:
$$\mathrm{Ra} = \frac{\rho_0 \alpha g_0 \Delta T D^3}{\kappa_0 \eta(T=0.5)}$$

Values from Euen et al. (2023) Table 2:
- Case A (e.g. A1, A7): $\mathrm{Ra} = 7 \times 10^3$
- Case C (e.g. C1): $\mathrm{Ra} = 10^5$

### Internal Heating
The heating number is zero ($H = 0.0$). There are no internal heat sources in the domain. All heat enters across the bottom boundary ($r = r_b$) and exits across the top boundary ($r = r_t$).

## 3. Viscosity Law (Frank-Kamenetskii)

Euen et al. (2023) use the non-dimensional Frank-Kamenetskii viscosity:
$$\eta(T) = \exp[E (0.5 - T)]$$
where $\Delta \eta = \exp(E)$ is the total viscosity contrast across the unit temperature span $\Delta T = 1.0$.

For each benchmark case:
- Case A1: $\Delta \eta = 1 \implies E = 0.0$ (isoviscous)
- Case A3: $\Delta \eta = 20 \implies E = \ln(20) \approx 2.995732$
- Case A7: $\Delta \eta = 10^5 \implies E = \ln(10^5) = 5 \ln(10) \approx 11.512925$ (stagnant lid)
- Case C1: $\Delta \eta = 1 \implies E = 0.0$ (isoviscous)
- Case C2: $\Delta \eta = 10 \implies E = \ln(10) \approx 2.302585$
- Case C3: $\Delta \eta = 30 \implies E = \ln(30) \approx 3.401197$

In `aragog`, temperature-dependent viscosity in the mixing-length parameterization uses either an Arrhenius or Frank-Kamenetskii formulation in the solid phase rheology. For B6, the viscosity contrast across the convective layer is governed by the parameter $E$.

## 4. Boundary Conditions

Euen et al. (2023) apply fixed temperatures at both boundaries:
- Top boundary ($r = r_t = 1.0$): $T(r_t) = 0.0$ (non-dimensional)
- Bottom boundary ($r = r_b = 0.55$): $T(r_b) = 1.0$ (non-dimensional)

Mapping to `aragog` boundary options:
- Outer boundary: Dirichlet fixed temperature condition
  - `boundary_conditions.outer_boundary_condition = 5` (fixed surface temperature)
  - `boundary_conditions.outer_boundary_value = T_surf`
- Inner boundary: Dirichlet fixed temperature condition
  - `boundary_conditions.inner_boundary_condition = 3` (fixed core-mantle boundary temperature)
  - `boundary_conditions.inner_boundary_value = T_cmb`
- CMB boundary layer law:
  - `boundary_conditions.cmb_flux_law = 'none'` (fixed temperature BC 3 sets Dirichlet boundary condition directly; Deschamps and Sotin 2000 thermal boundary layer law parameterizes heat flux when core couples to a thermal reservoir, but here the boundary temperature is fixed at $T = 1.0$).
- Core BC mode:
  - `boundary_conditions.core_bc = 'quasi_steady'` (or 'energy_balance' with BC 3).

## 5. Output Diagnostic Definitions

### Nusselt Numbers
Euen et al. (2023) define top and bottom Nusselt numbers in spherical geometry (Eqs. 6 and 7):
$$\mathrm{Nu}_t = \frac{r_t (r_t - r_b)}{r_b} Q_t$$
$$\mathrm{Nu}_b = \frac{r_b (r_t - r_b)}{r_t} Q_b$$
where $Q_t$ and $Q_b$ are the non-dimensional heat fluxes normalized by conductive heat flux across a spherical shell.
In steady state without internal heat sources, global energy conservation requires:
$$r_t^2 Q_t = r_b^2 Q_b \implies \mathrm{Nu}_t = \mathrm{Nu}_b$$

### Mean Temperature
Volume-averaged temperature in the spherical shell (Eq. 9):
$$\langle T \rangle = \frac{1}{\mathcal{V}} \int_{\mathcal{V}} T\, \mathrm{d}\mathcal{V} = \frac{3}{4\pi (r_t^3 - r_b^3)} \int_{r_b}^{r_t} 4\pi r^2 T(r)\, \mathrm{d}r$$

## 6. Assumptions and Inherent 1-D Limitations

### What aragog Can Represent
- Spherically symmetric radial temperature profile $T(r)$.
- Volume-averaged mean temperature $\langle T \rangle$.
- Surface and bottom conductive and convective heat fluxes ($Q_t$, $Q_b$, $\mathrm{Nu}_t$, $\mathrm{Nu}_b$).
- Radial turbulent transport parameterized by mixing-length eddy diffusivity $\kappa_H(r)$.
- Top boundary stagnation and conductive lid thickness when viscosity contrast is large ($E \ge 10$).

### What aragog Cannot Represent
- 3-D velocity field $\mathbf{v}(r, \theta, \phi)$ and convective planform geometry: Euen et al. model 3-D convection cells with tetrahedral planform (Case A) or cubic planform (Case C).
- Root-mean-square velocity $V_{\mathrm{rms}}$: Euen et al. compute 3-D volume-integrated velocity magnitude (Eq. 10). In 1-D mixing-length theory, convective transport is represented as an enhanced effective thermal conductivity $\kappa_H$ without explicitly solving a 3-D vector Navier-Stokes velocity field.
- Non-spherical plumes, upwellings, and downwelling sheets.

### Comparable Quantities
The physically meaningful quantities for quantitative comparison between 3-D simulations and 1-D mixing-length models are:
1. Mean temperature $\langle T \rangle$
2. Top Nusselt number $\mathrm{Nu}_t$
3. Bottom Nusselt number $\mathrm{Nu}_b$
4. Balance between top and bottom heat flux at steady state.
