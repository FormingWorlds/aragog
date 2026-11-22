# Closing the core energy at the CMB

In this tutorial you see how Aragog turns the cooling of the core into the entropy change of the node at the core-mantle boundary (CMB), and you check that the core loses exactly the heat that crosses the CMB. The [explanation](../Explanations/core_bc.md#the-entropy-slope-at-the-cmb-node) gives the equations.

## What you will learn

- How to evaluate the slope $(\partial T/\partial S)_P$ of the blended temperature at the CMB node.
- Why the heat capacity $c_p$ of the blend does not give that slope in the mush.
- How to read the ledger check of a solver call.

## What you need

- Aragog installed (see [Installation](../How-to/installation.md)).
- The SPIDER P-S test tables, with `ARAGOG_TEST_EOS_DIR` set to their folder (see [Build tests](../How-to/build_tests.md)).

## 1. Write the configuration

The mantle has 14 cells (15 nodes), from 3480 km to 6371 km radius, with a grey-body surface. The core cools through the CMB (`inner_boundary_condition = 1`) with the default `core_bc = "energy_balance"`. The tight tolerances make the solver error small against the effect you look at.

```python
import os
from pathlib import Path

config = """
[solver]
start_time = 0
end_time = 2
atol = 1e-10
rtol = 1e-10

[boundary_conditions]
outer_boundary_condition = 1        # grey-body surface
outer_boundary_value = 1500
inner_boundary_condition = 1        # the core cools through the CMB
inner_boundary_value = 0
emissivity = 1
equilibrium_temperature = 255
core_heat_capacity = 880
core_bc = "energy_balance"          # the default

[mesh]
outer_radius = 6371000
inner_radius = 3480000
number_of_nodes = 15
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
Path('cmb_closure.toml').write_text(config)
```

## 2. Compare the slope with $c_p/T$ at the CMB node

The evaluator returns the blended temperature, its heat capacity and its slope at a pressure and an entropy. The melt coordinate $g$ is 0 at the solidus and 1 at the liquidus. The product $c_p (\partial T/\partial S)_P / T$ is the rate at which the node would cool, relative to the core, if the closure used $c_p/T$.

```python
import numpy as np

from aragog.solver import EntropySolver

eos_dir = os.environ['ARAGOG_TEST_EOS_DIR']
solver = EntropySolver.from_file(filename='cmb_closure.toml', eos_dir=eos_dir)
solver.initialize()
eos = solver.entropy_eos
P_cmb = float(np.ravel(solver.evaluator.mesh.basic_pressure)[0])
S_sol = float(np.ravel(eos.solidus_entropy(np.array([P_cmb])))[0])
S_liq = float(np.ravel(eos.liquidus_entropy(np.array([P_cmb])))[0])
print(f'CMB pressure {P_cmb / 1e9:.1f} GPa, solidus {S_sol:.0f} J/kg/K, liquidus {S_liq:.0f} J/kg/K')
from aragog.eos.entropy_phase import EntropyPhaseEvaluator

g = np.array([0.1, 0.5, 0.9])  # melt coordinate: 0 at the solidus, 1 at the liquidus
node = EntropyPhaseEvaluator(eos, gravitational_acceleration=10.0, matprop_smooth_width=0.01)
node.set_pressure(np.full(g.size, P_cmb))
node.set_entropy(S_sol + g * (S_liq - S_sol))
node.update()
T, cp, slope = node.temperature(), node.heat_capacity(), node.temperature_slope()
for gi, Ti, ki in zip(g, T, cp * slope / T):
    print(f'g = {gi:.1f}: T = {Ti:.0f} K, c_p/T times the slope = {ki:.3f}')
```

```text
CMB pressure 145.7 GPa, solidus 1365 J/kg/K, liquidus 9759 J/kg/K
g = 0.1: T = 4725 K, c_p/T times the slope = 1.130
g = 0.5: T = 5341 K, c_p/T times the slope = 1.000
g = 0.9: T = 5956 K, c_p/T times the slope = 0.897
```

Near the solidus $c_p/T$ would make the node cool 13 % too fast, and near the liquidus 10 % too slowly. In the middle of the mush the 2 agree. The closure divides by the slope, so the node cools at the core rate everywhere.

## 3. Run one call and read the ledger

Every cell starts at $g = 0.9$ at its own pressure. After the call the solver compares the core heat change with the CMB heat $q$ and stores the result in `solver.cmb_ledger`: the miss `r`, the bound `bound` from the solver tolerances, and `rel`, the difference of the slope of the tabulated temperature from the closure slope.

```python
P_stag = np.ravel(solver.evaluator.mesh.staggered_pressure)
S_sol_c = np.ravel(eos.solidus_entropy(P_stag))
S_liq_c = np.ravel(eos.liquidus_entropy(P_stag))
solver.set_initial_entropy(S_sol_c + 0.9 * (S_liq_c - S_sol_c))
solver.solve()
out = solver.get_state()
check = solver.cmb_ledger
q = float(out.step_dE_F_cmb_J)
print(f'CMB heat of the call: {q:.2e} J')
print(f'ledger miss below 1e-4 of the CMB heat: {abs(check["r"]) < 1e-4 * abs(q)}')
print(f'ledger miss within the solver bound: {abs(check["r"]) <= check["bound"]}')
```

```text
CMB heat of the call: 2.15e+29 J
ledger miss below 1e-4 of the CMB heat: True
ledger miss within the solver bound: True
```

With $c_p/T$ the same call would miss by about 9 % of $q$ (see [the CMB closure ledger across the melt band](../Validation/core.md#16-the-cmb-closure-ledger-across-the-melt-band)). A miss above the bound logs a warning, and a miss above 10 times the bound stops the run.
