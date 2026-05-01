# Aragog: model overview

## Transport properties

Transport properties in Aragog are implemented in [`phase.py`](https://github.com/FormingWorlds/aragog/blob/main/aragog/phase.py), except for the eddy diffusivity, which is found in [`solver.py`](https://github.com/FormingWorlds/aragog/blob/main/aragog/solver.py).

---

## 3.1 Thermophysical properties

All thermophysical quantities — density $\rho$, conductivity $\lambda$, heat capacity $c_p$, thermal expansion coefficient $\alpha$, and dynamic viscosity $\eta$ — are assumed to be functions of both temperature and pressure. Properties in the melt phase and solid phase are denoted with subscripts $m$ and $s$, respectively.

In the **mixed phase region**, the density is estimated as:

$$\frac{1}{\rho} = \frac{\phi}{\rho_m} + \frac{1 - \phi}{\rho_s} \tag{27}$$

As noted in Sec. 2.2, this density may differ from the pseudo-density $\rho^*$ used to estimate the pressure field.

The **thermal conductivity** in the mixed phase is:

$$\lambda = \phi \lambda_m + (1 - \phi) \lambda_s \tag{28}$$

The **dynamic viscosity** in the mixed phase is formulated to capture the rheological transition, where the aggregate viscosity changes abruptly between the melt and solid viscosity at a critical melt fraction [1]:

$$\log_{10} \eta = z \log_{10}(\eta_m) + (1 - z) \log_{10}(\eta_s) \tag{29}$$

$$z(\phi) = \frac{1}{2}\left[1 + \tanh\left(\frac{\phi - \phi^\eta_c}{\Delta\phi^\eta_w}\right)\right] \tag{30}$$

where $\phi^\eta_c$ is the rheological transition melt fraction and $\Delta\phi^\eta_w$ is the rheological transition width. These are input parameters motivated by geochemical experiments.

The **heat capacity** and **thermal expansivity** in the mixed phase region are expressed following Ref. [2] as:

$$c_p = \left.\frac{\partial h}{\partial T}\right|_P = c_p^0 + \Delta h \left.\frac{\partial \phi}{\partial T}\right|_P \simeq \frac{\Delta h}{T_{liq} - T_{sol}} \tag{31}$$

$$\alpha = \rho \left.\frac{\partial (1/\rho)}{\partial T}\right|_P = \alpha^0 + \frac{\rho}{\rho_m \rho_s} \Delta\rho \left.\frac{\partial \phi}{\partial T}\right|_P \simeq \frac{\rho_s - \rho_m}{\rho(T_{liq} - T_{sol})} \tag{32}$$

where the phase-change terms dominate, such that $c_p \gg c_p^0$ and $\alpha \gg \alpha^0$. The definition of the adiabatic temperature gradient in Eq. (5) still holds in the mixed phase region with the corresponding thermophysical properties.

The **porosity** in the mixed phase is defined as:

$$\zeta = \frac{\rho_s - \rho}{\rho_s - \rho_m} \tag{33}$$

It is related to the melt fraction as:

$$\frac{\rho_m}{\rho}\zeta = \phi, \qquad \frac{\rho_s}{\rho}(1 - \zeta) = (1 - \phi) \tag{34}$$

Because the melt fraction is not a continuous quantity and some properties exhibit a jump at the mixed-phase boundaries, an **additional smoothing** is applied to all thermophysical quantities $\beta$.

Near the interface between the mixed phase and the **melt**:

$$\tilde{\beta} = z_m(\phi^*)\,\beta_m + (1 - z_m(\phi^*))\,\beta \tag{35}$$

$$z_m(\phi^*) = \frac{1}{2}\left[1 + \tanh\left(\frac{\phi^* - 1}{\Delta\phi^*_w}\right)\right] \tag{36}$$

Near the interface between the mixed phase and the **solid**:

$$\tilde{\beta} = z_s(\phi^*)\,\beta + (1 - z_s(\phi^*))\,\beta_s \tag{37}$$

$$z_s(\phi^*) = \frac{1}{2}\left[1 + \tanh\left(\frac{\phi^*}{\Delta\phi^*_w}\right)\right] \tag{38}$$

where $\phi^* = \frac{T - T_{sol}}{T_{liq} - T_{sol}}$ is the extended melt fraction profile (with $\phi^* < 0$ in the solid phase and $\phi^* > 1$ in the melt phase), and $\Delta\phi^*_w$ is the phase transition width. For the kinematic viscosity, this smoothing is applied in logarithmic space.

---

## 3.2 Eddy diffusivity

The eddy diffusivity, which affects the convective flux (Eq. 4) and convective mixing flux (Eq. 7), is derived from **mixing length theory**. It equals the product of a velocity scale and the mixing length, and depends on the flow regime:

$$\kappa_h = \begin{cases}
0 & \text{for } Re = 0 \\
u_{visc}\, l & \text{for } 0 \leq Re \leq 9/8 \\
u_{invis}\, l & \text{for } Re > 9/8
\end{cases} \tag{39}$$

where $l$ is the mixing length and $Re = u_{visc}\,l/\nu$ is the Reynolds number based on the viscous velocity. The velocity scales are:

$$u_{visc} = -\frac{\alpha g l^3}{18\nu}\left(\frac{\partial T}{\partial r} - \left.\frac{\partial T}{\partial r}\right|_S\right) \tag{40}$$

$$u_{invis} = \sqrt{-\frac{\alpha g l^2}{16}\left(\frac{\partial T}{\partial r} - \left.\frac{\partial T}{\partial r}\right|_S\right)} \tag{41}$$

The necessary condition for convection to occur is:

$$\frac{\partial T}{\partial r} - \left.\frac{\partial T}{\partial r}\right|_S < 0 \tag{42}$$

and the velocity is set to zero if this condition is not satisfied.

The **mixing length** $l$ is either set constant according to the size of the domain:

$$l = 0.25\,(r_{top} - r_{cmb}) \tag{43}$$

or set equal to the distance to the nearest boundary:

$$l(r) = \min(r_{top} - r,\; r - r_{cmb}) \tag{44}$$

---

## 3.3 Permeability

The permeability factor $K$ in Eq. (9), which affects the gravitational separation flux, depends on the porosity $\zeta$ and varies with the flow regime [3]:

$$K = \begin{cases}
\dfrac{2}{9} a^2 & \zeta > 0.771462 \quad \text{(Stokes)} \\[8pt]
0.001\, a^2 \dfrac{\zeta^2}{(1 - \zeta)^2} & 0.0769452 \leq \zeta \leq 0.771462 \quad \text{(Blake–Kozeny–Carman)} \\[8pt]
\dfrac{5}{7} a^2 \zeta^{4.5} & \zeta < 0.0769452 \quad \text{(Rumpf–Gupte)}
\end{cases} \tag{45}$$

This expression comes from Ref. [3], where the derivation uses a factor $F(\phi)$ as a function of melt fraction related to the permeability factor by:

$$K = a^2 \frac{(\rho_s - \rho_m)\phi + \rho_m}{\rho\,\phi(1 - \phi)}\,F(\phi) \tag{46}$$

!!! Note
    The condition $\zeta > \beta$ is equivalent to $\phi > \rho_m / (\gamma \rho_s + \rho_m)$ using $\gamma = (1 - \beta)/\beta$.