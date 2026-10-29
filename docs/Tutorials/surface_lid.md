# A crust on the magma ocean

In this tutorial you switch on the sub-grid surface lid, run a short cooling step of a mantle whose top cell has started to freeze, and read the crust from the solver output. You then follow the crust for a few years and see it thicken as the square root of time.

## What you will learn

- How to switch on the lid with `subgrid_lid = true`.
- How the lid changes the heat the planet loses.
- How to read the crust thickness and its temperatures from `SolverOutput`.

## What you need

- Aragog installed (see [Installation](../How-to/installation.md)).
- The SPIDER-format test tables, with `ARAGOG_TEST_EOS_DIR` pointing at them (see [First run](firstrun.md), step 1). The lid needs equation-of-state tables; it does not run in the constant-properties mode.

## 1. Write the configuration

The mantle is 10 cells thick, from 3480 km to 6371 km radius, with a grey-body surface and no heat flow through the bottom. The lid needs the grey-body surface, `outer_boundary_condition = 1`. The only new line is `subgrid_lid = true`. Nothing else changes: the other lid settings keep their defaults (see [Surface lid](../Explanations/surface_lid.md#settings-and-limits)).

```python
import os
from pathlib import Path

config = """
[solver]
start_time = 0
end_time = 1
atol = 1e-8
rtol = 1e-8

[boundary_conditions]
outer_boundary_condition = 1        # grey-body surface, needed by the lid
outer_boundary_value = 0
inner_boundary_condition = 2        # prescribed flux at the bottom
inner_boundary_value = 0            # W/m^2: no heat crosses the bottom
emissivity = 1
equilibrium_temperature = 255
core_heat_capacity = 880
core_bc = "quasi_steady"
subgrid_lid = true                  # carry a crust thinner than the top cell

[mesh]
outer_radius = 6371000
inner_radius = 3480000
number_of_nodes = 10
mixing_length_profile = "nearest_boundary"
core_density = 10500
eos_method = 1

[energy]
conduction = true
convection = true
gravitational_separation = false
mixing = false
radionuclides = false
tidal = false
solver_method = "radau"
use_jax_jacobian = false

[initial_condition]
initial_condition = 1
surface_temperature = 3500
basal_temperature = 3500

[phase_liquid]
density = 4000
viscosity = 10
heat_capacity = 1000
melt_fraction = 1
thermal_conductivity = 4
thermal_expansivity = 3e-5

[phase_solid]
density = 4200
viscosity = 1e21
heat_capacity = 1000
melt_fraction = 0
thermal_conductivity = 4
thermal_expansivity = 3e-5

[phase_mixed]
latent_heat_of_fusion = 4e5
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = "solidus.dat"
liquidus = "liquidus.dat"
phase = "mixed"
phase_transition_width = 0.01
grain_size = 1.0e-3
"""
Path('surface_lid.toml').write_text(config)
```

## 2. Run with and without the lid

Each cell starts at the entropy 3050 J/kg/K. At the pressure of the top cell this is in the mush, just above the solidus. The function `run` builds a solver from the file, sets the end time and the lid switch, and runs one call.

```python
from aragog.solver import EntropySolver

eos_dir = os.environ['ARAGOG_TEST_EOS_DIR']


def run(end_yr, lid=True):
    solver = EntropySolver.from_file(filename='surface_lid.toml', eos_dir=eos_dir)
    solver.parameters.boundary_conditions.subgrid_lid = lid
    solver.parameters.solver.end_time = end_yr
    solver.initialize()
    solver.set_initial_entropy(3050.0)
    solver.solve()
    return solver.get_state()


on, off = run(1.0), run(1.0, lid=False)
print(f'surface flux with the lid:    {on.heat_flux[-1]:12.1f} W/m^2')
print(f'surface flux without the lid: {off.heat_flux[-1]:12.1f} W/m^2')
print(f'lid thickness {on.lid_delta:.3f} m, weight {on.lid_weight:.3f}')
print(f'temperatures: surface {on.lid_T_surface:.1f} K, mean {on.lid_T_mean:.1f} K, base {on.lid_T_base:.1f} K')
```

The output:

```text
surface flux with the lid:          1519.9 W/m^2
surface flux without the lid:     641018.2 W/m^2
lid thickness 4.888 m, weight 1.000
temperatures: surface 419.7 K, mean 1315.7 K, base 2149.0 K
```

Without the lid, the surface node has the temperature of the hot top cell, and the grey body radiates 6.4 × 10⁵ W/m². With the lid, a crust 4.9 m thick has formed after one year; its surface has cooled to 420 K, and the planet loses 1520 W/m², about 420 times less. The lid base sits at the solidus of the top-cell pressure, 2149 K. The weight 1 says that the lid acts in full.

## 3. Watch the crust grow

A crust that freezes over melt thickens as the square root of time (see [Surface lid](../Validation/surface_lid.md#a-crust-growing-over-melt)). Run to 1, 4 and 16 years:

```python
import numpy as np

for t in (1.0, 4.0, 16.0):
    out = run(t)
    print(f't = {t:4.0f} yr: delta = {out.lid_delta:7.3f} m, delta / sqrt(t) = {out.lid_delta / np.sqrt(t):.3f} m/yr^0.5')
```

```text
t =    1 yr: delta =   4.888 m, delta / sqrt(t) = 4.888 m/yr^0.5
t =    4 yr: delta =   9.883 m, delta / sqrt(t) = 4.942 m/yr^0.5
t =   16 yr: delta =  19.943 m, delta / sqrt(t) = 4.986 m/yr^0.5
```

The thickness grows by a factor 4.08 over a 16-fold time, and $\delta / \sqrt{t}$ changes by 2 %. The lid starts as a thin phantom crust of 0.5 m, not on the textbook profile, so the ratio still rises slowly towards its long-time value.

## What next

- [Surface lid](../Explanations/surface_lid.md) explains the two lid states, the profile and the hand-over to the mesh.
- [Surface lid validation](../Validation/surface_lid.md) compares the lid with textbook solutions.
