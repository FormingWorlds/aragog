# Refining the mesh at the surface

In this tutorial we run the same short mantle cooling case twice: once on the uniform mesh and once with a 1 km top cell. We then compare the top cell, the surface heat flux and the mean magma temperature. [Mesh refinement](../Explanations/mesh_refinement.md) explains how the refined mesh is built.

## Assumptions

- Aragog is installed (see [Installation](../How-to/installation.md)).
- `ARAGOG_TEST_EOS_DIR` points at the SPIDER test tables (step 1 of the [First-run tutorial](firstrun.md) shows how to fetch them).

## 1. Run the two meshes

The script loads the bundled `abe_solid.toml` (1000 km mantle, 100 nodes, mass-coordinate mesh), sets the top cell, shortens the run to 100 years and starts from a uniform entropy of 2900 J/kg/K. Save it as `refined.py`:

```python
import os
from pathlib import Path

import numpy as np

import aragog
from aragog.solver import EntropySolver

cfg = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'
eos_dir = os.environ['ARAGOG_TEST_EOS_DIR']

for top in (0.0, 1000.0):
    solver = EntropySolver.from_file(filename=str(cfg), eos_dir=eos_dir)
    solver.parameters.mesh.surface_cell_thickness = top  # m; 0 = not set
    solver.parameters.solver.end_time = 100.0  # yr
    solver.initialize()
    solver.set_initial_entropy(2900.0)
    solver.solve()
    out = solver.get_state()
    r = np.ravel(solver.evaluator.mesh.basic.radii)
    print(
        f'top cell {r[-1] - r[-2]:8.1f} m: status {out.status}, '
        f'surface flux {np.ravel(out.heat_flux)[-1]:.4e} W/m^2, T_magma {out.T_magma:.1f} K'
    )
```

Run it:

```sh
python refined.py
```

The two runs take about 100 s together on one core. The script prints

```text
top cell  10866.6 m: status 0, surface flux 1.4383e+05 W/m^2, T_magma 1262.7 K
top cell   1000.0 m: status 0, surface flux 1.4530e+05 W/m^2, T_magma 1265.9 K
```

## 2. Read the result

The first line is the uniform mesh: in mass-coordinate mode its top cell is 10.9 km, thicker than the 10.1 km radial average because the density is lowest at the surface. The second line has the 1 km top cell that the config sets. Both runs end with status 0.

After 100 years the surface heat flux on the refined mesh is 1.0 % higher than on the uniform mesh, and the mean magma temperature is 3.2 K higher. The top cell changes the result, so a run with a thin top cell is not the same model as the uniform run. Which top cell is thin enough is the question of the convergence study on the [Validation page](../Validation/mesh_refinement.md).

## 3. Set the cell in a config file

For a config file, set the field in the `[mesh]` table instead of in Python:

```toml
[mesh]
surface_cell_thickness = 1000.0   # m
```

[Refining the mesh towards the surface and the CMB](../How-to/mesh-refinement.md) lists the allowed values and the errors.
