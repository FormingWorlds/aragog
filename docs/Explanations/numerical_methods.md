# Aragog: model overview

## Numerical methods

The solver for the enthalpy balance is implemented in [`solver.py`](https://github.com/FormingWorlds/aragog/blob/main/aragog/solver.py). Spatial approximation routines (gradients, interpolations) are found in [`mesh.py`](https://github.com/FormingWorlds/aragog/blob/main/aragog/mesh.py). Boundary conditions and the initial condition are implemented in [`core.py`](https://github.com/FormingWorlds/aragog/blob/main/aragog/core.py).

---

## 4.1 Finite-volume method

Spatial integration of Eq. (1) in a finite-volume fashion, over a spherical layer $i$ bounded by radii $r_{i-1/2}$ and $r_{i+1/2}$, gives:

$$(\rho c_p V)_i \left.\frac{\partial T}{\partial t}\right|_i = -q_{i+1/2} S_{i+1/2} + q_{i-1/2} S_{i-1/2} + \Phi_i V_i \tag{47}$$

with $S_{i+1/2} = 4\pi r_{i+1/2}^2$ and $V_i = \frac{4}{3}\pi\left(r_{i+1/2}^3 - r_{i-1/2}^3\right)$.

Volume terms are evaluated at cell centers of the mass coordinate mesh, while surface terms are evaluated at cell boundaries. A **dual mesh** approach maps quantities between staggered nodes (cell centers) and basic nodes (boundaries).

A uniform spatial mesh of constant spacing $\Delta r$ is used between the surface $r_{top}$ and the core-mantle boundary $r_{cmb}$. This mesh is then mapped into mass coordinates as described in Sec. 1.3. The mesh is defined in terms of basic nodes such that the spatial and mass coordinate meshes overlap at cell boundaries ($\xi_{i+1/2} = \xi(r_{i+1/2})$) but not at cell centers ($\xi_i \neq \xi(r_i)$).

Physical quantities at basic nodes (cell boundaries) are approximated from quantities at staggered nodes (cell centers) via simple **linear interpolation**:

$$\psi(\xi_{i+1/2}) = \frac{\Delta\xi_i\, \psi(\xi_{i+1}) + \Delta\xi_{i+1}\, \psi(\xi_i)}{\Delta\xi_i + \Delta\xi_{i+1}} \tag{48}$$

where $\Delta\xi_i = \xi_{i+1/2} - \xi_{i-1/2}$ is the cell width.

**Spatial gradients** at basic nodes are approximated as:

$$\left.\frac{\partial \psi}{\partial \xi}\right|_{\xi_{i+1/2}} = \frac{\psi(\xi_{i+1}) - \psi(\xi_i)}{\xi_{i+1} - \xi_i} \tag{49}$$

!!! Note
     The state of the system at the outer and inner boundaries is not defined by Eqs. (48) and (49); any quantity $\psi$ or $\partial\psi/\partial\xi$ is unknown at $\xi_{cmb}$ and $\xi_{top}$.

The energy balance is implemented in **non-dimensional** form, using reference values for temperature, time, radius, and density, such that the order of magnitude of each physical quantity is close to one.

---

## 4.2 Boundary conditions

### 4.2.1 Neumann boundary condition

A Neumann boundary condition, where the total heat flux $q_{tot}$ is imposed at the inner or outer boundary (or both), applies naturally in a finite-volume formulation. However, the individual components of the heat flux (and the thermal state) remain unknown at the boundary. To estimate these, a **linear extrapolation** from the two closest interior points is applied.

For the core-mantle boundary, with $i = 1$ denoting the lowermost cell:

$$\psi_{cmb} = \frac{2\Delta\xi_1 + \Delta\xi_2}{\Delta\xi_1 + \Delta\xi_2}\,\psi(r_1) - \frac{\Delta\xi_1}{\Delta\xi_1 + \Delta\xi_2}\,\psi(r_2) \tag{50}$$

$$\left.\frac{\partial \psi}{\partial \xi}\right|_{cmb} = \left(\frac{\Delta\xi_2}{\Delta\xi_1} + 1\right)\frac{\psi(\xi_2) - \psi(\xi_1)}{\xi_2 - \xi_1} - \frac{\Delta\xi_2}{\Delta\xi_1}\frac{\psi(\xi_3) - \psi(\xi_2)}{\xi_3 - \xi_2} \tag{51}$$

Similar relationships apply at the top boundary. These provide estimates of the individual flux components, which may not be strictly consistent with the imposed total heat flux value, though this does not affect the time evolution.

At the top of the planet, the heat flux can optionally be expressed as **radiative exchange** between the ground and a blackbody atmosphere:

$$q_{top} = \varepsilon\sigma\left(T_{top}^4 - T_{atm}^4\right) \tag{52}$$

where $\varepsilon$ is the emissivity of the ground. Equation (52) is technically a mixed boundary condition (involving the unknown surface temperature), but it is implemented as a flux boundary condition using a surface temperature extrapolated from the inner node via an expression analogous to Eq. (50).

When `param_utbl = True`, an ultra-thin thermal boundary layer attenuates the radiating temperature: the surface $T_\mathrm{surf}$ is the real cubic root of $b\,T_\mathrm{surf}^3 + T_\mathrm{surf} - T_\mathrm{interior} = 0$ (Bower et al. 2018, Eq. 18), with $b = $ `param_utbl_const` controlling the strength. The numpy path (`solver/boundary.py:_utbl_tsurf`) and the JAX path (`jax/solver.py:_utbl_tsurf_jax`) both use Cardano's formula and agree to within $\sim 10^{-12}$ K.

#### Verification of the UTBL Cardano correction

![UTBL Cardano correction](../figures/vv/fig_05_utbl_cardano.pdf)

**Figure 4.** (a) $T_\mathrm{surf}$ vs $T_\mathrm{interior}$ over the magma-ocean range 1500-5000 K for three values of $b$: $10^{-7}$ (weak), $10^{-6}$ (canonical Bower+2018), $10^{-5}$ (strong). The numpy reference and the JAX-traceable form are visually indistinguishable; their max absolute disagreement across all three $b$ values is $1.8\times 10^{-12}$ K. (b) Cubic residual $|b\,T_\mathrm{surf}^3+T_\mathrm{surf}-T_\mathrm{interior}|$ of the returned root, demonstrating that both implementations satisfy the cubic to within machine epsilon scaled by $T_\mathrm{interior}$.

### 4.2.2 Dirichlet Boundary Condition

When a Dirichlet boundary condition is applied (i.e., the temperature is imposed at a boundary), the thermal state is fully defined at that boundary. The temperature gradient (and the melt fraction gradient) must be computed accordingly to obtain correct heat fluxes.

For the **top boundary**, with $i = N$ being the uppermost cell and $\xi_{top} = \xi_{N+1/2}$:

$$\left.\frac{\partial \psi}{\partial \xi}\right|_{top} = \frac{\psi_{top} - \psi(\xi_N)}{\xi_{top} - \xi_N} \tag{53}$$

For the **core-mantle boundary**, with $i = 1$ being the lowermost cell and $\xi_{cmb} = \xi_{1/2}$:

$$\left.\frac{\partial \psi}{\partial \xi}\right|_{cmb} = \frac{\psi(\xi_1) - \psi_{cmb}}{\xi_1 - \xi_{cmb}} \tag{54}$$

### 4.2.3 Core-cooling model

When using a flux boundary condition at the core-mantle boundary, the following **core cooling model** may be used to estimate $q_{cmb}$.

Starting from the enthalpy balance of the core:

$$(\rho c_p V)_{core} \frac{dT_{core}}{dt} = -S_{cmb}\, q_{cmb} \tag{55}$$

The core temperature $T_{core}$ is assumed to be linearly related to the temperature of the lowermost cell $T_1$, as are their time derivatives:

$$T_{core} \simeq \hat{T}_{core}\, T_1, \qquad \frac{\partial T_{core}}{\partial t} \simeq \hat{T}_{core} \frac{\partial T_1}{\partial t} \tag{56}$$

using the proportionality constant $\hat{T}_{core} = 1.147$ (see Ref. [1]).

The thermal balance at the lowermost cell $i = 1$ gives:

$$(\rho c_p V)_1 \left.\frac{\partial T}{\partial t}\right|_1 = -q_{1+1/2}\, S_{1+1/2} + q_{cmb}\, S_{cmb} \tag{57}$$

Combining Eqs. (55) and (57) with the scaling of Eq. (56) yields the following estimate for the CMB heat flux:

$$q_{cmb} = \frac{S_{1+1/2}}{S_{core}} \left[1 + \frac{(\rho c_p V)_1}{(\rho c_p V)_{core}\,\hat{T}_{core}}\right]^{-1} q_{1+1/2} \tag{58}$$

!!! Note
    Volumetric heating rates have been neglected in Eqs. (55) and (57) in deriving this expression.

---

## 4.3 Time integration

Equation (47) is integrated in time using an **implicit multi-step variable-order backward differentiation** method.

Although any temperature field may be provided as an initial condition, the default is to begin the time integration on an **adiabat**, such that:

$$\frac{\partial T}{\partial r} = \left.\frac{\partial T}{\partial r}\right|_S = -\frac{g\alpha T}{c_p} \tag{59}$$

In the general case, this equation is solved numerically since $g$, $\alpha$, and $c_p$ are spatially dependent. When these properties are constant, the following analytical temperature profile is obtained:

$$T(r) = T_{top} \exp\left(\frac{g\alpha}{c_p}(r_{top} - r)\right) \tag{60}$$

with $T_{top}$ being the surface temperature. This profile is then mapped onto the mass coordinate mesh.