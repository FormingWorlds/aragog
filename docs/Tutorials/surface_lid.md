# A crust on the magma ocean

In this tutorial you switch on the sub-grid surface lid, run a short cooling step of a mantle whose top cell has started to freeze, and read the crust from the solver output. You then follow the crust for a few years and see it thicken as the square root of time.

## What you will learn

- How to switch on the lid with `subgrid_lid = true`.
- How the lid changes the heat the planet loses.
- How to read the crust thickness and its temperatures from `SolverOutput`.
- How the lid runs on a constant-property column under a fixed surface temperature, and how to check its energy.

## What you need

- Aragog installed (see [Installation](../How-to/installation.md)).
- The SPIDER-format test tables, with `ARAGOG_TEST_EOS_DIR` pointing at them (see [First run](firstrun.md), step 1). This tutorial uses the equation-of-state tables. The lid also runs in one constant-properties mode (see [Surface lid](../Explanations/surface_lid.md#on-a-constant-property-column)).

## 1. Write the configuration

The mantle has 9 cells (10 nodes), from 3480 km to 6371 km radius, with a grey-body surface and no heat flow through the bottom. The lid runs with this grey-body surface, `outer_boundary_condition = 1`, or with a fixed surface temperature, `5`. The only new line is `subgrid_lid = true`. Nothing else changes: the other lid settings keep their defaults (see [Surface lid](../Explanations/surface_lid.md#settings-and-limits)).

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
outer_boundary_condition = 1        # grey-body surface
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

Each cell starts at the entropy 3050 J/kg/K. At the pressure of the top cell, 7.8 GPa, this is in the mush, with a melt fraction of 0.36. The top cell of this coarse mesh is 380 km thick, and the lid takes its base temperature at the pressure of the top-cell centre: the solidus there is 2149 K, against 1439 K at the surface. The run shows how the lid works, so its crust thickness and temperatures below are numbers of this coarse mesh, not physical values; for a real planet use a mesh with a thin top cell (see [Settings and limits](../Explanations/surface_lid.md#settings-and-limits)). The function `run` builds a solver from the file, sets the end time and the lid switch, and runs one call.

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
surface flux with the lid:          1539.5 W/m^2
surface flux without the lid:     641018.2 W/m^2
lid thickness 4.861 m, weight 1.000
temperatures: surface 420.9 K, mean 1319.8 K, base 2149.0 K
```

Without the lid, the surface node has the temperature of the hot top cell, and the grey body radiates 6.4 × 10⁵ W/m². With the lid, a crust 4.9 m thick has formed after one year; its surface has cooled to 421 K, and the planet loses 1540 W/m², about 416 times less. The lid base sits at the solidus of the top-cell pressure, 2149 K. The weight 1 says that the lid acts in full.

## 3. Watch the crust grow

A crust that freezes over melt thickens as the square root of time (see [Surface lid](../Validation/surface_lid.md#a-crust-growing-over-melt)). Run to 1, 4 and 16 years:

```python
import numpy as np

for t in (1.0, 4.0, 16.0):
    out = run(t)
    print(f't = {t:4.0f} yr: delta = {out.lid_delta:7.3f} m, delta / sqrt(t) = {out.lid_delta / np.sqrt(t):.3f} m/yr^0.5')
```

```text
t =    1 yr: delta =   4.861 m, delta / sqrt(t) = 4.861 m/yr^0.5
t =    4 yr: delta =   9.830 m, delta / sqrt(t) = 4.915 m/yr^0.5
t =   16 yr: delta =  19.840 m, delta / sqrt(t) = 4.960 m/yr^0.5
```

The thickness grows by a factor 4.08 over a 16-fold time, and $\delta / \sqrt{t}$ changes by 2 %. The lid starts as a thin phantom crust of 0.5 m, not on the textbook profile, and the ratio still rises slowly.

## 4. The lid on a constant-property column

The stagnant-lid benchmarks of [Solid-state convection](../Validation/solid_state_convection.md#the-top-slope-against-published-convection-grids) use constant material properties and a fixed surface temperature. The lid runs there in one mode: no equation-of-state tables, the solid rheology on, and the `'lid'` closure, whose stagnant lid gives the base of the crust (see [On a constant-property column](../Explanations/surface_lid.md#on-a-constant-property-column)).

The function below takes the configuration of step 1 and changes it to a shell 300 km thick on 20 nodes (top cell 16 km), with the surface held at 300 K, constant properties and the solid rheology. It passes `entropy_eos=None`, because the lid refuses this mode when tables are attached. The shell starts at 3200 J/kg/K at the bottom and 2800 J/kg/K at the top, 1878 K and 1363 K.

`phase_solid.viscosity = 1e16` Pa s is a demonstration value, chosen so that the stagnant lid of the closure starts thin and the sub-grid lid hands over within a short run. With 1e21 Pa s the lid-base isotherm of the closure already lies about 150 km deep at the start, 9 cells of this mesh, so the lid hands over at once.

```python
import numpy as np
from aragog.config import Config
from aragog.parser import SolidRheologyParams


def run_constant(end_yr):
    config = Config.from_file('surface_lid.toml')
    config.solver.end_time = end_yr
    config.mesh.number_of_nodes = 20
    config.mesh.inner_radius = 6071000.0  # a shell 300 km thick
    bc = config.boundary_conditions
    bc.outer_boundary_condition, bc.outer_boundary_value = 5, 300.0  # surface held at 300 K
    bc.subgrid_lid, bc.lid_handover_cells = True, 1
    pm = config.phase_mixed
    pm.const_properties = True
    pm.const_T_ref, pm.const_S_ref = 1600.0, 3000.0  # T = 1600 K at S = 3000 J/kg/K
    pm.const_rho, pm.const_Cp, pm.const_cond = 3300.0, 1250.0, 3.0
    config.phase_solid.viscosity = 1e16  # a demonstration value, see the text
    config.phase_solid.rheology = SolidRheologyParams(enabled=True, stress_closure_mode='lid')
    solver = EntropySolver(config, entropy_eos=None)  # no tables: the constant-property mode
    solver.initialize()
    solver.set_initial_entropy(np.linspace(3200.0, 2800.0, config.mesh.number_of_nodes - 1))
    solver.solve()
    return solver.get_state()


out = run_constant(2.5e5)
powers = out.step_dE_F_int_J + out.step_dE_F_cmb_J + out.step_dE_Q_radio_J + out.step_dE_Q_tidal_J
print(f'hand-over at {out.lid_handover_time:.4g} yr, lid thickness now {out.lid_delta:.1f} m')
print(f'heat content change {out.step_dE_state_heat_J:.4e} J, integrated powers {powers:.4e} J')
print(f'relative miss {out.step_dE_state_heat_J / powers - 1:.1e}, solver residual below 1e-12: {abs(out.step_solver_residual_J / out.step_dE_F_int_J) < 1e-12}')
assert abs(out.lid_handover_time / 2.417e5 - 1.0) < 1e-3
assert abs(out.step_dE_state_heat_J / powers - 1.0) < 1e-4
```

The output (about 50 s):

```text
hand-over at 2.417e+05 yr, lid thickness now 0.0 m
heat content change -5.9818e+27 J, integrated powers -5.9819e+27 J
relative miss -1.3e-05, solver residual below 1e-12: True
```

What to look at. The crust grows under the cold surface until the isotherm of the stagnant lid has moved below the top cell; then, at 2.4 × 10⁵ years, the lid hands its crust to the mesh, and its thickness is 0. The energy of the shell and the lid together changes by the heat that leaves through the surface: the change of the heat content and the time integral of the surface power agree to 1.3 × 10⁻⁵, and the solver residual is at round-off.

The miss falls as the solver tolerance falls, to 5.6 × 10⁻⁶, 2.4 × 10⁻⁶ and 1.1 × 10⁻⁶ at 10⁻⁹, 10⁻¹⁰ and 10⁻¹¹ against 10⁻⁸ here, which fits an error of the time integration over the steps. The check of the heat of the lid alone, its sub-ledger, is part of the [validation of the lid on the stagnant-lid benchmarks](../Validation/surface_lid.md#the-lid-on-the-stagnant-lid-benchmarks).

## What next

- [Surface lid](../Explanations/surface_lid.md) explains the two lid states, the profile and the hand-over to the mesh.
- [Surface lid validation](../Validation/surface_lid.md) compares the lid with textbook solutions.
