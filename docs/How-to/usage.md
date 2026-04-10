# Usage

## Running Aragog from Python

Aragog's primary interface is Python. The minimal workflow is: load configuration, initialize the solver, solve, and inspect results.

```python
from aragog.config import Config
from aragog.solver import Solver
from aragog.output import Output

# Load configuration from a TOML file
params = Config.from_toml("src/aragog/cfg/abe_solid.toml")

# Create solver and run
solver = Solver(params)
solver.initialize()
solver.solve()

# Inspect results
out = Output(solver)
print("Time span (yr):", out.time_range)
print("Final surface T (K):", out.solution_top_temperature)
print("Global melt fraction:", float(out.melt_fraction_global))
```

## Running from the command line

Aragog also has a CLI entry point:

```console
aragog run src/aragog/cfg/abe_solid.toml
```

Use `aragog --help` to see all available commands, including `download` and `env`.

## Running within PROTEUS

When used as a PROTEUS submodule, Aragog is called through the interior module wrapper. PROTEUS constructs a configuration dictionary from the coupled state and passes it via `Config.from_dict()`. Users do not need to manage Aragog's configuration directly in this mode.

See the [PROTEUS documentation](https://proteus-framework.org/PROTEUS) for details on coupled runs.

## Writing output

### NetCDF files

The `Output` class writes state snapshots to NetCDF files:

```python
out.write_at_time("output.nc", tidx=-1, compress=True)
```

The NetCDF file contains:

**Scalar variables:**

| Variable | Unit | Description |
|----------|------|-------------|
| `time` | yr | Simulation time |
| `phi_global` | -- | Volume-averaged melt fraction |
| `mantle_mass` | kg | Total mantle mass |
| `rheo_front` | -- | Rheological front position (dimensionless) |

**Mesh variables (basic mesh, suffix `_b`):**

| Variable | Unit | Description |
|----------|------|-------------|
| `radius_b` | km | Radial position |
| `pres_b` | GPa | Pressure |
| `temp_b` | K | Temperature |
| `phi_b` | -- | Melt fraction |
| `Fcond_b` | W/m^2 | Conductive heat flux |
| `Fconv_b` | W/m^2 | Convective heat flux |
| `Fgrav_b` | W/m^2 | Gravitational separation heat flux |
| `Fmix_b` | W/m^2 | Convective mixing heat flux |
| `Ftotal_b` | W/m^2 | Total heat flux |
| `log10visc_b` | Pa s | Log10 dynamic viscosity |
| `density_b` | kg/m^3 | Density |
| `heatcap_b` | J/kg/K | Heat capacity |

**Mesh variables (staggered mesh, suffix `_s`):**

| Variable | Unit | Description |
|----------|------|-------------|
| `log10visc_s` | Pa s | Log10 dynamic viscosity |
| `mass_s` | kg | Mass per cell |
| `Hradio_s` | W/kg | Radiogenic heating |
| `Hvol_s` | W/kg | Volumetric dilation/compression heating |
| `Htidal_s` | W/kg | Tidal heating |
| `Htotal_s` | W/kg | Total internal heating |

Inspect a NetCDF file header with:

```console
ncdump -h output.nc
```

### Plotting

The `Output.plot()` method generates a multi-panel figure showing temperature, melt fraction, viscosity, fluxes, and other quantities versus pressure:

```python
import matplotlib.pyplot as plt

out.plot(num_lines=7)
plt.savefig("output_plot.png", dpi=200, bbox_inches="tight")
```

## Resetting the solver

To run multiple configurations without reloading lookup tables, use `reset()`:

```python
solver.parameters = new_params
solver.reset()
solver.solve()
```

This preserves the `PhaseEvaluatorCollection` (and any loaded lookup tables) while reinitializing the mesh, boundary conditions, initial condition, and solver state.

## Downloading lookup table data

Aragog requires lookup table data for thermophysical properties. Download it with:

```console
aragog download all
```

See [Reference: Data](../Reference/data.md) for a list of data files and their sources.
