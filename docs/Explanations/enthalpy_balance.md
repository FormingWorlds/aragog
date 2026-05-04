# Aragog: model overview

## Enthalpy balance

We start with the enthalpy balance in integral form:

$$\int_V \rho c_p \left.\frac{\partial T}{\partial t}\right|_\xi \, dV = -\oint_S \mathbf{q} \cdot \mathbf{n} \, dS + \int_V \Phi \, dV \tag{1}$$

where $T$ is the effective temperature of the melting medium, assuming thermal equilibrium between solid and melt phases. Mass coordinates $\xi$ are used (see Sec. 1.3), but integration is performed over a physical domain of volume $V$ and surface boundary $S$. Any control volume $V$ is assumed to coincide with a material volume that does not evolve in time — the mass of each volume is constant and there is no mass flux at the interfaces.

Spherical symmetry is assumed and only spatial variations along the radial direction are considered. In what follows, only the radial component of vector quantities is retained. The enthalpy balance, heat fluxes, and heat sources are implemented in Aragog in [`solver.py`](https://github.com/FormingWorlds/aragog/blob/main/aragog/solver.py).

---

## 1.1 Heat fluxes

The total heat flux is defined as the sum of the conductive ($q_{cd}$), convective ($q_{cv}$), convective mixing ($q_{cm}$), and gravitational separation ($q_{gm}$) contributions:

$$q_{tot} = q_{cd} + q_{cv} + q_{cm} + q_{gm} \tag{2}$$

The **conductive flux** is:

$$q_{cd} = -\lambda \frac{\partial T}{\partial r} \tag{3}$$

The **convective flux** is modelled as:

$$q_{cv} = -\rho c_p \kappa_h \left(\frac{\partial T}{\partial r} - \left.\frac{\partial T}{\partial r}\right|_S\right) \tag{4}$$

where the adiabatic temperature gradient (temperature gradient at constant entropy $S$) is:

$$\left.\frac{\partial T}{\partial r}\right|_S = -\frac{g \alpha T}{c_p} \tag{5}$$

The **convective mixing flux** and the **gravitational separation flux** are enthalpy fluxes associated with corresponding mass fluxes:

$$q_{cm} + q_{gm} = \Delta h \left(j_{cm} + j_{gm}\right) \tag{6}$$

where $j_{cm}$ and $j_{gm}$ are mass fluxes associated with the melt (the corresponding fluxes for the solid are $-j_{cm}$ and $-j_{gm}$, such that the sum of species mass fluxes is zero).

The **mass diffusion flux of melt** is:

$$j_{cm} = -\rho \kappa_h \frac{\partial \phi}{\partial r} \tag{7}$$

where $\phi$ is the mass fraction of melt (defined in Sec. 2.1). The mass diffusivity is assumed equal to the eddy diffusivity $\kappa_h$.

The **melt mass flux associated with gravitational separation** is:

$$j_{gm} = \rho \phi (1 - \phi) v_{rel} \tag{8}$$

where $v_{rel}$ is the relative velocity between melt and solid:

$$v_{rel} = \frac{(\rho_m - \rho_s) g K}{\eta_m} \tag{9}$$

Both mass fluxes $j_{cm}$ and $j_{gm}$ are zero outside the mixed-phase region.

---

## 1.2 Heat sources

The volumetric heat sources considered are:

$$\Phi = \Phi_{tidal} + \Phi_{radio} \tag{10}$$

associated with tidal heating and radiogenic heating, respectively.

The **radiogenic heating** is defined as:

$$\Phi_{radio} = \sum_i \rho \phi_i \chi_i \exp\left(-\frac{t - t_0}{\tau^{1/2}_i}\right) \tag{11}$$

where $\phi_i$, $\chi_i$, and $\tau^{1/2}_i$ are the power generation per unit mass, the mass fraction, and the half-life associated with radioisotope $i$, respectively. The radiogenic heating is time-dependent but spatially uniform.

The **tidal heating** volume source must be provided by the user. It can be spatially dependent but is constant in time.

The volumetric work associated with phase segregation is *not* added as a separate explicit source. By definition the enthalpy contrast $\Delta h = \Delta u + P\,\Delta v$, and on a hydrostatic column the divergence of the $\Delta h$-weighted mass-flux contributions to $q_{cm} + q_{gm}$ already carries $+\rho g \Delta v\,(j_{cm} + j_{gm})$ implicitly via the chain rule. Adding it explicitly would double-count (Bower 2018 §3, SPIDER `energy.c`).

---

## 1.3 Mass coordinates

Mass coordinates $\xi$ (in unit length) correspond to a change of variable from the spatial coordinate $r$, such that the mass contained in an element $\Delta\xi$ is constant regardless of depth.

The transformation from spatial to mass coordinates is:

$$\xi(r) = \left[3\int_{r_{cmb}}^{r} \frac{\rho^*(r')}{\rho^*_{planet}} r'^2 \, dr' + \xi_{cmb}^3\right]^{1/3} \tag{13}$$

where $\rho^*_{planet}$ is the volume-averaged density of the planet and $\rho^*$ is the local mantle density. The mass coordinate at the core-mantle boundary (CMB) is defined as:

$$\xi_{cmb} = \left(\frac{\rho^*_{core}}{\rho^*_{planet}}\right)^{1/3} r_{cmb} \tag{14}$$

where $\rho^*_{core}$ is the volume-averaged density of the core. Using the volume-averaged planetary density as a scaling quantity ensures that the spatial coordinate and the mass coordinate coincide at the surface:

$$\xi_{top} = r_{top} \tag{15}$$

which equals the planetary radius. The reverse transformation from mass coordinates back to spatial coordinates is:

$$r(\xi) = \left[3\int_{\xi_{cmb}}^{\xi} \frac{\rho^*_{planet}}{\rho^*(\xi')} \xi'^2 \, d\xi' + r_{cmb}^3\right]^{1/3} \tag{16}$$

The spatial gradient of any quantity $\psi$ is computed according to:

$$\frac{\partial \psi}{\partial r} = \frac{\rho^*(r)}{\rho^*_{planet}} \left(\frac{r}{\xi}\right)^2 \frac{\partial \psi}{\partial \xi} \tag{17}$$

!!! note
    If the spatial mesh is uniform, the mass coordinate mesh will generally not be uniform. The notation $\rho^*$ for density will be clarified in the [next section](thermodynamics.md).

If this transformation is ignored and spatial coordinates are used directly, the convective derivative term in the enthalpy balance is neglected, i.e.:

$$\left.\frac{\partial T}{\partial t}\right|_\xi \simeq \left.\frac{\partial T}{\partial t}\right|_r \tag{18}$$

Mass coordinates are implemented in Aragog in [`mesh.py`](https://github.com/FormingWorlds/aragog/blob/main/aragog/mesh.py).