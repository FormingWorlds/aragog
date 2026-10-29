# Refining the mesh towards the surface and the CMB

This page shows how to make the outermost or innermost mantle cell thinner than the uniform cell. [Mesh refinement](../Explanations/mesh_refinement.md) explains the stretched grid and its limits.

## Set the cell thickness

Add one or both fields to the `[mesh]` table of the config, in metres:

```toml
[mesh]
surface_cell_thickness = 200.0   # outermost cell [m]; 0 keeps the uniform spacing
cmb_cell_thickness = 2000.0      # innermost cell [m]; 0 keeps the uniform spacing
```

Each value must be 0 or a number above 0 and below the uniform radial cell, `(outer_radius - inner_radius) / (number_of_nodes - 1)`, and at least $10^{-7}$ of the mantle thickness. With one field set, only that end is refined; with both set, both ends are. The number of nodes does not change.

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

- `mesh.surface_cell_thickness must be 0 (off) or in (0, ...) m, the uniform radial cell`: the value is not a number, is negative, or is not below the uniform radial cell. A TOML `true` gives this error too.
- `mesh.cmb_cell_thickness = ... m and mesh.surface_cell_thickness = ... m give no refined grid of ... nodes`: the mesh cannot hold the cells. The value is below $10^{-7}$ of the mantle thickness, or, in mass-coordinate mode, it lies between the uniform mass-coordinate end cell and the uniform radial cell. Use a thinner cell.
