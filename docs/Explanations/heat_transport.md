# Heat transport

Aragog represents the total radial heat flux as a sum of four contributions:

$$
q_{\mathrm{tot}} = q_{\mathrm{cd}} + q_{\mathrm{cv}} + q_{\mathrm{cm}} + q_{\mathrm{gm}}.
$$

Each term can be individually enabled or disabled via the `[energy]` section of the configuration.

## Conduction

$$
q_{\mathrm{cd}} = -\lambda\,\frac{\partial T}{\partial r},
$$

with thermal conductivity $\lambda(T, P, \phi)$. In mixed phase, the conductivity is a linear average of solid and liquid values:

$$
\lambda = \phi\lambda_m + (1-\phi)\lambda_s.
$$

## Parameterized convection (eddy diffusion)

Convection is written as a diffusive flux that relaxes superadiabatic gradients:

$$
q_{\mathrm{cv}} = -\rho c_p \kappa_h\left(
\frac{\partial T}{\partial r} - \left.\frac{\partial T}{\partial r}\right|_S
\right),
$$

where $\kappa_h$ is an eddy diffusivity and the adiabatic gradient is:

$$
\left.\frac{\partial T}{\partial r}\right|_S = -\frac{g\alpha T}{c_p}.
$$

Convection is only active when the flow is unstable, i.e. the superadiabatic gradient is negative:

$$
\left(\frac{\partial T}{\partial r} - \left.\frac{\partial T}{\partial r}\right|_S\right) < 0.
$$

### Eddy diffusivity (mixing length theory)

The eddy diffusivity $\kappa_h$ is computed from a mixing length $l$ and a velocity scale that depends on the flow regime. The mixing length profile is configurable:

- **`nearest_boundary`**: $l(r) = \min(r_{\mathrm{top}} - r,\; r - r_{\mathrm{cmb}})$
- **`constant`**: $l = $ constant fraction of mantle thickness

The velocity scale and resulting $\kappa_h$ depend on whether the flow is in the viscous or inviscid regime, determined by the local Rayleigh number relative to the critical Rayleigh number.

## Melt transport fluxes (mixed phase only)

In partially molten regions, Aragog includes additional enthalpy transport associated with melt redistribution:

$$
q_{\mathrm{cm}} + q_{\mathrm{gm}} = \Delta h\, (j_{\mathrm{cm}} + j_{\mathrm{gm}}),
$$

where $\Delta h$ is the latent heat of fusion and $j$ are melt mass fluxes $[\mathrm{kg\,m^{-2}\,s^{-1}}]$.

### Convective mixing mass flux

Mixing homogenizes the melt fraction via diffusion:

$$
j_{\mathrm{cm}} = -\rho \kappa_h \frac{\partial \phi}{\partial r},
$$

with $\phi$ the melt fraction. This term redistributes melt toward regions of lower melt fraction, counteracting gravitational separation.

### Gravitational separation mass flux

Buoyant percolation or settling of melt through the solid matrix:

$$
j_{\mathrm{gm}} = \rho\,\phi(1-\phi)\, v_{\mathrm{rel}},
\qquad
v_{\mathrm{rel}} = \frac{(\rho_m - \rho_s) g K}{\eta_m},
$$

where $v_{\mathrm{rel}}$ is the relative velocity between melt and solid, $K$ is the permeability of the mixed-phase region (depending on grain size and melt fraction), and $\eta_m$ is the melt viscosity.

These melt-transport fluxes are constructed so that melt and solid species fluxes sum to zero (no net mass flux).

## Volumetric heating sources

$$
\Phi = \Phi_{\mathrm{tidal}} + \Phi_{\mathrm{radio}} + \Phi_{\mathrm{vol}}.
$$

### Radiogenic heating

Typically space-uniform but time-dependent:

$$
\Phi_{\mathrm{radio}} = \sum_i \rho\,\varphi_i\chi_i
\exp\!\left(-\frac{t - t_0}{\tau_{1/2,i}}\right),
$$

where $\varphi_i$, $\chi_i$, and $\tau_{1/2,i}$ are the power per mass, mass fraction, and half-life of isotope $i$.

### Tidal heating

$\Phi_{\mathrm{tidal}}$ is user-specified and may vary with radius. It is typically constant in time within a run.

### Volumetric dilation/compression work

Couples to melt transport:

$$
\Phi_{\mathrm{vol}} = \rho g\left(\frac{1}{\rho_m} - \frac{1}{\rho_s}\right)(j_{\mathrm{cm}} + j_{\mathrm{gm}}).
$$

This term accounts for the work done when melt (with different density than solid) moves through the gravitational field.
