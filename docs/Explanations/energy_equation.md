# Energy equation

This page describes the governing equation solved by Aragog and its spatial discretization.

## Integral enthalpy balance

Aragog evolves temperature by enforcing an enthalpy balance over a spherical shell (mantle) between the core-mantle boundary (CMB) at $r = r_{\mathrm{cmb}}$ and the surface at $r = r_{\mathrm{top}}$. In integral form, for a control volume $V$ with boundary $S$:

$$
\int_V \rho c_p \left.\frac{\partial T}{\partial t}\right|_{\xi}\, dV
=
- \int_S \mathbf{q}\cdot \mathbf{n}\, dS
+ \int_V \Phi\, dV,
$$

where $\mathbf{q}$ is the heat flux, $\Phi$ the volumetric heating rate, and $c_p$ the specific heat capacity. The time derivative is taken at constant mass coordinate $\xi$ (a Lagrangian-like coordinate).

Each finite-volume cell coincides with a material volume: its mass is constant, and there is no net mass flux through cell interfaces in the control-volume sense. Species fluxes may exist internally in mixed phase via melt/solid segregation, but their sum is zero.

## Semi-discrete finite-volume form

Discretizing the mantle into radial shells (cells), Aragog applies a finite-volume balance per cell $i$:

$$
(\rho c_p V)_i \left.\frac{\partial T}{\partial t}\right|_i
= -\, q_{i+1/2}\, S_{i+1/2} + q_{i-1/2}\, S_{i-1/2} + \Phi_i\, V_i ,
$$

where:

- $S_{i\pm 1/2} = 4\pi r_{i\pm 1/2}^2$ is the spherical area of the outer/inner face,
- $V_i = \frac{4}{3}\pi\left(r_{i+1/2}^3 - r_{i-1/2}^3\right)$ is the shell volume,
- $q$ is the radial heat flux (positive upward by convention),
- $\Phi$ is a volumetric heating rate $[\mathrm{W\,m^{-3}}]$.

The unknown is $T_i(t)$ at cell centers; fluxes are evaluated on cell faces.

## Staggered mesh

Aragog uses a **staggered mesh**:

- **Basic (face) nodes**: $N$ radial positions where fluxes and gradients are evaluated.
- **Staggered (cell-center) nodes**: $N-1$ midpoints where temperature and volumetric terms live.

This arrangement avoids the checkerboard instability that can arise from collocated discretizations.

## Mass coordinate transform

When mass coordinates are enabled (`mass_coordinates = true`), the spatial coordinate is replaced by a mass coordinate $\xi$ defined so that equal increments in $\xi$ correspond to equal mass increments:

$$
\xi(r) = \left( 3\int_{r_{\mathrm{cmb}}}^{r} \frac{\rho^*(r')}{\rho^*_{\mathrm{planet}}} r'^2\, dr' + \xi_{\mathrm{cmb}}^3 \right)^{1/3}.
$$

Spatial gradients are converted via:

$$
\frac{\partial \psi}{\partial r}
=
\frac{\rho^*(r)}{\rho^*_{\mathrm{planet}}}\left(\frac{r}{\xi}\right)^2
\frac{\partial \psi}{\partial \xi}.
$$

The mass coordinate provides uniform resolution in mass, concentrating spatial resolution in high-density regions near the CMB.

If mass coordinates are disabled, the model uses a uniform spatial grid and approximates $\left.\partial T/\partial t\right|_{\xi} \approx \left.\partial T/\partial t\right|_{r}$.

## Time integration

The semi-discrete ODE system is advanced using an **implicit, variable-order, variable-step backward differentiation formula (BDF)** method via scipy's `solve_ivp`. BDF is well-suited to the stiffness introduced by conduction, convection, and phase-change effective heat capacities.

The right-hand side function (`Solver.dTdt`) computes:

1. Face fluxes from the current temperature profile
2. Flux divergence across each cell
3. Capacitance (effective $\rho c_p V$) per cell
4. Heating rate from internal sources (radiogenic, tidal, volumetric)
5. Temperature tendency in K/yr (the ODE is integrated in years)

## Boundary conditions

Aragog supports three types of boundary conditions at each boundary (CMB and surface):

1. **Dirichlet**: impose $T$ at the boundary; compute gradients via one-sided differences.
2. **Neumann**: impose total heat flux $q_{\mathrm{tot}}$ at the boundary; extrapolate boundary state from interior nodes.
3. **Radiative**: impose a radiative surface flux:
   $$
   q_{\mathrm{top}} = \varepsilon\sigma\left(T_{\mathrm{top}}^4 - T_{\mathrm{atm}}^4\right).
   $$

A **core cooling model** is available at the CMB: the lowermost mantle cell is coupled to a lumped core energy balance, estimating $q_{\mathrm{cmb}}$ from the difference between core and mantle temperatures.

## Surface temperature event

The solver monitors the surface temperature and can stop integration early when the change from the initial surface temperature exceeds a configured threshold (`tsurf_poststep_change`). This is used in coupled PROTEUS runs to hand control back to the atmosphere solver.
