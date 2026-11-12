# Refining the mesh towards the surface and the CMB

This page shows how to make the outermost or innermost mantle cell thinner than the uniform cell. [Mesh refinement](../Explanations/mesh_refinement.md) explains the stretched grid and its limits. The tutorial [Refining the mesh at the surface](../Tutorials/mesh_refinement.md) runs a case on both meshes.

## The default mesh

Without these fields the mesh has a 1000 m CMB cell, and its top cell is the top cell of the uniform mesh of the same coordinate mode (`surface_cell_thickness = "uniform"`). The cells between the two ends follow the stretched grid. For the uniform mesh of earlier versions, set both fields:

```toml
[mesh]
cmb_cell_thickness = 0.0
surface_cell_thickness = 0.0   # or "uniform": the same mesh when the CMB cell is 0
```

A shell whose uniform radial cell is 1000 m or less, for example a 25 km shell with 100 nodes, needs `cmb_cell_thickness = 0` or a value below its uniform cell.

## Set the cell thickness

Add one or both fields to the `[mesh]` table of the config, in metres:

```toml
[mesh]
surface_cell_thickness = 200.0   # outermost cell [m]; 0 = not set
cmb_cell_thickness = 2000.0      # innermost cell [m]; 0 = not set
```

Each value must be 0 or a number of at least 2 m, or `"uniform"` for the surface cell, and below the uniform radial cell, `(outer_radius - inner_radius) / (number_of_nodes - 1)`. A set cell needs at least 4 nodes. With one field set, only that end is refined, and the cells at the other end grow: a 50 m top cell on `abe_solid.toml` makes the CMB cell 38.5 km instead of 9.33 km. Set both fields to hold both ends. The number of nodes does not change. A value set in Python after the config is read gets the same check when the mesh is built.

The fields act in both mesh modes. With `mass_coordinates = true`, the default, the end cells still have the thickness you set, in radius.

## Check the mesh

Build the mesh from the config and read the cell widths from the node radii:

```python
from pathlib import Path

import numpy as np

import aragog
from aragog.mesh import Mesh
from aragog.parser import Parameters

cfg = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'
params = Parameters.from_file(cfg)
params.mesh.surface_cell_thickness = 200.0  # m
params.mesh.cmb_cell_thickness = 2000.0  # m

cells = np.diff(np.ravel(Mesh(params).basic.radii))
print(f'top cell {cells[-1]:.3f} m, CMB cell {cells[0]:.3f} m, largest cell {cells.max():.0f} m')
```

For `abe_solid.toml` (1000 km mantle, 100 nodes, mass-coordinate mode) this prints

```text
top cell 200.000 m, CMB cell 2000.000 m, largest cell 26584 m
```

## Errors

- `mesh.surface_cell_thickness must be 0 (off) or 'uniform' or in [2, ...) m, the uniform radial cell`: the value is not a number or `"uniform"`, is negative, is below 2 m, or is not below the uniform radial cell. A TOML `true` gives this error too. The CMB cell gives the same error without `'uniform'`, for example when the default 1000 m is not below the uniform radial cell of a thin shell.
- `mesh.cmb_cell_thickness = ... m and mesh.surface_cell_thickness = ... m give no refined grid of ... nodes`: the mesh cannot hold the cells. In mass-coordinate mode the value lies between the uniform mass-coordinate end cell and the uniform radial cell. Use a thinner cell.
