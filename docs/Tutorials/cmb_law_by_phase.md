# The CMB law by phase

How does the heat flux out of the core change as the mantle above it freezes? This tutorial computes the heat flux across the core-mantle boundary (CMB) of the core module under the 3 settings of `ra_crit_cmb`, the critical Rayleigh number of the boundary layer on the mantle side of the CMB:

- 450, the value of Thiriet et al. (2019)[^cite-thiriet2019] Table 2 for the upper boundary layer;
- none set, eq. 16 of Thiriet et al. (2019), which takes the number from the internal Rayleigh number of the mantle;
- `"phase"`, which blends the two by the melt fraction of the bottom mantle cell: 450 while that cell is liquid, eq. 16 while it is solid.

In the example below the 2 laws differ by a factor of about 10 in the flux. The block needs no tables of the equation of state (EOS): the layer properties are given by hand. Run it from the root of the aragog repository.

[Critical Rayleigh number by phase](../Explanations/core_bc.md#critical-rayleigh-number-by-phase) explains the law, and [its validation](../Validation/core.md#15-critical-rayleigh-number-by-phase) shows the checks.

## 1. The flux of the 3 settings

The layer below is an example, chosen for this tutorial and not taken from a run: a core 600 K hotter than the bottom cell, under a layer with a liquid viscosity. The loop moves the melt fraction $\phi_0$ of the bottom cell across the rheological transition at 0.4 with width 0.2, the PROTEUS defaults.

```python
import jax

jax.config.update('jax_enable_x64', True)

from aragog.core import cmb_boundary_layer_flux, liquid_weight

layer = dict(
    conductivity=4.0,  # W/m/K
    density=5200.0,  # kg/m^3
    heat_capacity=1200.0,  # J/kg/K
    expansivity=1.5e-5,  # 1/K
    viscosity=0.1,  # Pa s, the layer
    gravity=10.5,  # m/s^2
    dr_half=750.0,  # m
    shell_thickness=2.9e6,  # m
    shell_contrast=3500.0,  # K
    interior_viscosity=0.2,  # Pa s, the interior at T_m
)
t_core, t_m = 5400.0, 4800.0  # K
q450 = float(cmb_boundary_layer_flux(t_core, t_m, ra_crit=450.0, **layer))
q16 = float(cmb_boundary_layer_flux(t_core, t_m, ra_crit=None, **layer))
print(f'450: {q450:.4g} W/m^2   eq. 16: {q16:.4g} W/m^2')
for phi in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0):
    s = float(liquid_weight(phi, 0.4, 0.2))
    q = float(
        cmb_boundary_layer_flux(
            t_core, t_m, ra_crit=None, melt_fraction=phi, liquid_band=(0.4, 0.2), **layer
        )
    )
    print(f'phi0 {phi:.1f}   s {s:.4f}   q {q:.4g} W/m^2')
```

```text
450: 6.175e+05 W/m^2   eq. 16: 5.995e+04 W/m^2
phi0 0.0   s 0.0000   q 5.995e+04 W/m^2
phi0 0.2   s 0.0028   q 6.034e+04 W/m^2
phi0 0.3   s 0.0500   q 6.736e+04 W/m^2
phi0 0.4   s 0.5000   q 1.924e+05 W/m^2
phi0 0.5   s 0.9500   q 5.496e+05 W/m^2
phi0 0.6   s 0.9972   q 6.136e+05 W/m^2
phi0 1.0   s 1.0000   q 6.175e+05 W/m^2
```

The weight `s` is 1/2 at the transition, and the flux there is the geometric mean of the two laws: $\sqrt{6.175 \times 10^5 \cdot 5.995 \times 10^4} \approx 1.924 \times 10^5$ W m$^{-2}$. At the band edges 0.3 and 0.5 the weight is 0.05 and 0.95. Outside the band the flux goes quickly to the law of that side: at $\phi_0 = 0.2$ it is 1.006 times that of eq. 16, and at $\phi_0 = 0.6$ 0.994 times that of 450.

A coupled run gives other numbers, quoted here from a 1 Earth-mass run at 5.03 Myr. The layer viscosity $\eta$ (`viscosity` above) is $1.15 \times 10^{19}$ Pa s and the interior viscosity $\eta_i$ (`interior_viscosity`) is $8.5 \times 10^{22}$ Pa s, over a 3670 km mantle. The internal Rayleigh number of the mantle is then $\mathrm{Ra}_i = 1.1 \times 10^{7}$, and eq. 16 gives $\mathrm{Ra}_\text{crit} = 8.5$. The layer is 3.1 km thick and its flux 0.77 W m$^{-2}$, 3.8 times the flux of a fixed 450. There eq. 16 gives the larger flux, because the interior is far more viscous than the layer; in the example above the 450 law does.

## 2. Select the law in a run

In an Aragog configuration the law is a key of the core module:

```toml
[boundary_conditions]
    inner_boundary_condition = 1
    core_bc = "core_module"

[boundary_conditions.core_module_params]
    ra_crit_cmb = "phase"
```

The transition comes from `rheological_transition_melt_fraction` and `rheological_transition_width` of the phase settings, the same pair as the viscosity blend. PROTEUS passes `interior_energetics.aragog.core_module.ra_crit_cmb` to this setting, and the transition from `interior_energetics.rfront_loc` and `rfront_wid`. A transition band $[\phi_\text{rheo} - w/2,\ \phi_\text{rheo} + w/2]$ outside \[0, 1\] is refused.

[^cite-thiriet2019]: M. Thiriet, D. Breuer, C. Michaut, A.-C. Plesa, *[Scaling laws of convection for cooling planets in a stagnant lid regime](https://doi.org/10.1016/j.pepi.2018.11.003)*, Physics of the Earth and Planetary Interiors, 286, 138-153, 2019. [SciX](https://scixplorer.org/abs/2019PEPI..286..138T/abstract).
