# Aragog: model overview

## Thermodynamics

Temperature is the only prognostic variable. The pressure is computed following a given equation of state in [`mesh.py`](https://github.com/FormingWorlds/aragog/blob/main/aragog/mesh.py). The melt fraction is calculated as a function of pressure and temperature in [`phase.py`](https://github.com/FormingWorlds/aragog/blob/main/aragog/phase.py).

---

## 2.1 Melt fraction

The mass fraction of the melt $\phi$ is defined as follows:

$$\phi = \begin{cases}
0 & T < T_{sol} \\
\dfrac{T - T_{sol}}{T_{liq} - T_{sol}} & T_{sol} \leq T \leq T_{liq} \\
1 & T > T_{liq}
\end{cases} \tag{19}$$

The mass fraction of the solid is correspondingly $1 - \phi$. The solidus $T_{sol}$ and liquidus $T_{liq}$ temperatures are assumed to depend solely on pressure, with $T_{liq} > T_{sol}$.

---

## 2.2 Pressure

The **Adams–Williamson model** is the default model used to derive the relationship between pressure and radius, assuming constant gravitational acceleration. It is also possible to specify arbitrary relationships between pressure, radius, and pseudo-density from a planetary structure calculation, in discretized form, which additionally allows for spatially dependent gravitational acceleration.

The Adams–Williamson model assumes an exponential relationship between density and pressure:

$$\rho^*(P) = \rho^*_{top} \exp\left(\frac{P}{B}\right) \tag{20}$$

where $B$ is the adiabatic bulk modulus and $\rho^*_{top}$ is the density at the surface. The notation $\rho^*$ emphasizes that this is a **pseudo-density** used to estimate the pressure field and the mass of each control volume (assumed constant in time). It may differ to a limited extent from the actual density $\rho$ used for solving thermal transport, which varies due to phase change. The pressure field is assumed to not vary in time.

A balance between pressure forces and weight gives:

$$\frac{dP}{dr} = -\rho^*(P)\, g \tag{21}$$

Integrating this expression yields the pressure field:

$$P(r) = -B \log\left(1 + \frac{\rho^*_{top}\, g\, (r - r_{top})}{B}\right) \tag{22}$$

Equation (22) assumes zero surface pressure, but the integration can be performed for any surface pressure.

The pseudo-density field as a function of radius is:

$$\rho^*(r) = \frac{\rho^*_{top}\, B}{B + \rho^*_{top}\, g\, (r - r_{top})} \tag{23}$$

The total mass of the mantle $M$ is retrieved by integrating the pseudo-density field from the core-mantle boundary $r_{cmb}$ to the surface $r_{top}$:

$$M = \int_{r_{cmb}}^{r_{top}} \rho^*(r)\, 4\pi r^2 \, dr \tag{24}$$

$$= \int_{r_{cmb}}^{r_{top}} \frac{4\pi B}{g} \frac{r^2}{\beta + r} \, dr, \qquad \beta = \frac{B}{\rho^*_{top}\, g} - r_{top} \tag{25}$$

$$= \frac{4\pi B}{g} \left[-3\beta^2 - 2\beta r + \frac{r^2}{2} + \beta^2 \log|\beta + r|\right]_{r_{cmb}}^{r_{top}} \tag{26}$$