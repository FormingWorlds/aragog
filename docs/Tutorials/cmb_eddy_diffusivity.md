# The eddy diffusivity at the CMB node

What eddy diffusivity does Aragog give the core-mantle boundary (CMB) node, and how does the convective heat flux out of the core follow from it? This tutorial evaluates the rule on 3 kinds of CMB state and then moves the entropy gradient of the node above the CMB through 0.

The CMB node takes the mixing-length eddy diffusivity of node 1 (the node above it) or its own floor, whichever is larger. The blocks below use the JAX function `compute_mlt` with material properties given by hand, so they need no tables of the equation of state (EOS). They need the JAX extra (`pip install -e '.[jax]'`). Run them in one Python session from the root of the repository.

[Eddy diffusivity at the CMB node](../Explanations/heat_transport.md#eddy-diffusivity-at-the-cmb-node) explains the rule, and [its validation](../Validation/cmb_eddy_flux.md) shows the checks.

## 1. The rule in 3 kinds of CMB state

The mesh has 4 basic nodes with a mixing length of 100 km. The mush has the melt fraction 0.6 at the CMB node and 0.4 above it. With `kappah_floor` = 10 m$^2$ s$^{-1}$ the floor is about 9.35 m$^2$ s$^{-1}$ at the CMB node and 5 m$^2$ s$^{-1}$ at node 1.

```python
import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp

from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties, compute_mlt

n = 4  # basic nodes: 0 is the CMB, 1 the node above it
r = jnp.linspace(3.5e6, 6.371e6, n)
p = jnp.linspace(120e9, 1e9, n)
mesh = MeshArrays(
    d_dr_matrix=jnp.zeros((n, n - 1)),
    quantity_matrix=jnp.zeros((n, n - 1)),
    area=jnp.ones(n),
    volume=jnp.ones(n),
    radii_basic=r,
    radii_stag=0.5 * (r[1:] + r[:-1]),
    mixing_length=jnp.full(n, 1.0e5),  # m
    mixing_length_sq=jnp.full(n, 1.0e10),
    mixing_length_cu=jnp.full(n, 1.0e15),
    P_stag=0.5 * (p[1:] + p[:-1]),
    P_basic=p,
    dP_dr_basic=jnp.gradient(p, r),
    gravity=jnp.full(n, 9.81),
)


def phase(phi0, phi1):
    """A mush with the melt fraction phi0 at the CMB node and phi1 above it."""
    ones = jnp.ones(n)
    return PhaseProperties(
        temperature=4000.0 * ones,  # K
        density=4500.0 * ones,  # kg/m^3
        heat_capacity=1000.0 * ones,  # J/kg/K
        thermal_expansivity=3.0e-5 * ones,  # 1/K
        dTdPs=1.0e-8 * ones,  # K/Pa
        melt_fraction=jnp.array([phi0, phi1, 0.4, 0.4]),
        viscosity=1.0e3 * ones,  # Pa s
        kinematic_viscosity=ones,  # m^2/s
        thermal_conductivity=4.0 * ones,  # W/m/K
        latent_heat=4.0e5 * ones,  # J/kg
        capacitance=4500.0 * 4000.0 * ones,
    )


params = PhaseParams(kappah_floor=10.0, phi_rheo=0.4, phi_width=0.15, eddy_diff_thermal=1.0)
states = {
    'stable CMB, node 1 on its floor': (1.0e-4, -1.0e-31),
    'unstable CMB, node 1 on its floor': (-1.0e-4, -1.0e-31),
    'unstable CMB, node 1 stable': (-1.0e-4, 1.0e-4),
}
for name, (g0, g1) in states.items():
    kh, _ = compute_mlt(jnp.array([g0, g1, -1.0e-4, -1.0e-4]), phase(0.6, 0.4), mesh, params)
    print(f'{name:35s} kh[0] = {float(kh[0]):9.3g} m^2/s, kh[1] = {float(kh[1]):9.3g} m^2/s')
```

```text
stable CMB, node 1 on its floor     kh[0] =  3.61e-15 m^2/s, kh[1] =         5 m^2/s
unstable CMB, node 1 on its floor   kh[0] =      9.35 m^2/s, kh[1] =         5 m^2/s
unstable CMB, node 1 stable         kh[0] =      9.35 m^2/s, kh[1] =         0 m^2/s
```

Over a stable CMB gradient the CMB node keeps only the mixing-length part of node 1, here $3.6 \times 10^{-15}$ m$^2$ s$^{-1}$, not the floor of node 1. Over an unstable CMB gradient it keeps its own floor, also when node 1 is stable and has no eddy diffusivity at all.

## 2. The CMB flux where node 1 changes sign

The CMB gradient is now stable, $2.36 \times 10^{-5}$ J kg$^{-1}$ K$^{-1}$ m$^{-1}$, and both nodes have the melt fraction 0.4. The loop moves the gradient of node 1 through 0 and prints the convective CMB flux $F_\mathrm{conv}[0] = -\rho T \kappa_h[0]\,\partial S/\partial r[0]$, next to the flux that a copy of the floored value of node 1 would give.

```python
rho_t = 4500.0 * 4000.0  # kg K / m^3
g0 = 2.36e-5  # stable CMB gradient, J/kg/K/m
for g1 in (-1.0e-12, -1.0e-20, 0.0, 1.0e-20, 1.0e-12):
    kh, _ = compute_mlt(jnp.array([g0, g1, -1.0e-4, -1.0e-4]), phase(0.4, 0.4), mesh, params)
    print(f'dS/dr[1] = {g1:+.0e}: F_conv[0] = {-rho_t * float(kh[0]) * g0:+.3e} W/m^2, '
          f'with the copy of node 1: {-rho_t * float(kh[1]) * g0:+.3e} W/m^2')
```

```text
dS/dr[1] = -1e-12: F_conv[0] = -3.644e+04 W/m^2, with the copy of node 1: -3.644e+04 W/m^2
dS/dr[1] = -1e-20: F_conv[0] = -2.778e-02 W/m^2, with the copy of node 1: -2.124e+03 W/m^2
dS/dr[1] = +0e+00: F_conv[0] = -0.000e+00 W/m^2, with the copy of node 1: -0.000e+00 W/m^2
dS/dr[1] = +1e-20: F_conv[0] = -0.000e+00 W/m^2, with the copy of node 1: -0.000e+00 W/m^2
dS/dr[1] = +1e-12: F_conv[0] = -0.000e+00 W/m^2, with the copy of node 1: -0.000e+00 W/m^2
```

At $-10^{-12}$ node 1 convects above its floor, and both columns agree. At $-10^{-20}$ node 1 sits on its floor: the copy sends $2.1 \times 10^3$ W m$^{-2}$ downward across the stable CMB, while the rule gives $0.03$ W m$^{-2}$ and goes to 0 with the gradient of node 1. So the rule has no step at $\partial S/\partial r[1] = 0$, and the copy steps by the floor flux.
