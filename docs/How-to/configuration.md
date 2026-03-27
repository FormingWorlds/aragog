# Configuration

Aragog reads its input from a single configuration file. The preferred format is **TOML**; the legacy INI format (`.cfg`) is still supported for backward compatibility.

## TOML configuration

A TOML configuration file specifies all model parameters grouped by section. Here is a minimal example for a solid-phase cooling run:

```toml
[scalings]
radius = 6371000
temperature = 4000
density = 4000
time = 31557600000  # 1000 years in seconds

[solver]
start_time = 0
end_time = 1000000000
atol = 1e-9
rtol = 1e-9
tsurf_poststep_change = 30
event_triggering = false

[boundary_conditions]
outer_boundary_condition = 1      # 1 = Dirichlet, 2 = Neumann, 3 = radiative
outer_boundary_value = 1500       # K (Dirichlet) or W/m^2 (Neumann)
inner_boundary_condition = 3
inner_boundary_value = 4000
emissivity = 1
equilibrium_temperature = 273
core_heat_capacity = 880

[mesh]
outer_radius = 6371000            # m
inner_radius = 5371000            # m (core-mantle boundary)
number_of_nodes = 100
mixing_length_profile = "nearest_boundary"
core_density = 10738.332568062382 # kg/m^3
surface_density = 4090            # kg/m^3
gravitational_acceleration = 9.81 # m/s^2
adiabatic_bulk_modulus = 260e9    # Pa

[energy]
conduction = true
convection = true
gravitational_separation = false
mixing = false
radionuclides = false
dilatation = false
tidal = false

[initial_condition]
surface_temperature = 3600        # K
basal_temperature = 4000          # K

[phase_liquid]
density = 4000                    # kg/m^3
viscosity = 1e2                   # Pa s
heat_capacity = 1000              # J/kg/K
melt_fraction = 1
thermal_conductivity = 4          # W/m/K
thermal_expansivity = 1.0e-5      # 1/K

[phase_solid]
density = 4200
viscosity = 1e21
heat_capacity = 1000
melt_fraction = 0
thermal_conductivity = 4
thermal_expansivity = 1.0e-5

[phase_mixed]
latent_heat_of_fusion = 4e6      # J/kg
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = "data/test/solidus_1d_lookup.dat"
liquidus = "data/test/liquidus_1d_lookup.dat"
phase = "solid"                   # "solid", "liquid", or "composite"
phase_transition_width = 0.01
grain_size = 1.0e-3              # m
```

### Radionuclides (optional)

Any number of radionuclide sections can be added. Each section name must start with `radionuclide_`:

```toml
[radionuclide_K40]
name = "K40"
t0_years = 4.55e9
abundance = 1.1668e-4
concentration = 310
heat_production = 2.8761e-5
half_life_years = 1.248e9

[radionuclide_U238]
name = "U238"
t0_years = 4.55e9
abundance = 0.9927955
concentration = 0.031
heat_production = 9.4946e-5
half_life_years = 4.468e9
```

## Configuration sections

### `[scalings]`

Scaling parameters used to non-dimensionalize the governing equations. Typical values are order-of-magnitude characteristic quantities of the problem.

| Key | Unit | Description |
|-----|------|-------------|
| `radius` | m | Characteristic length scale |
| `temperature` | K | Characteristic temperature |
| `density` | kg/m^3 | Characteristic density |
| `time` | s | Characteristic time scale |

### `[solver]`

ODE solver parameters controlling the BDF time integrator.

| Key | Unit | Description |
|-----|------|-------------|
| `start_time` | yr | Start time of integration |
| `end_time` | yr | End time of integration |
| `atol` | -- | Absolute tolerance for BDF solver |
| `rtol` | -- | Relative tolerance for BDF solver |
| `tsurf_poststep_change` | K | Surface temperature change that triggers an event |
| `event_triggering` | bool | Whether the surface temperature event is terminal |

### `[boundary_conditions]`

Thermal boundary conditions at the CMB (inner) and surface (outer).

| Key | Unit | Description |
|-----|------|-------------|
| `outer_boundary_condition` | -- | 1 = Dirichlet (fixed $T$), 2 = Neumann (fixed flux), 3 = radiative |
| `outer_boundary_value` | K or W/m^2 | Value applied at the outer boundary |
| `inner_boundary_condition` | -- | Same options as outer |
| `inner_boundary_value` | K or W/m^2 | Value applied at the inner boundary |
| `emissivity` | -- | Surface emissivity (for radiative BC) |
| `equilibrium_temperature` | K | Radiative equilibrium temperature (for radiative BC) |
| `core_heat_capacity` | J/kg/K | Core heat capacity (for core cooling BC) |

### `[mesh]`

Spatial discretization parameters.

| Key | Unit | Description |
|-----|------|-------------|
| `outer_radius` | m | Planet surface radius |
| `inner_radius` | m | Core-mantle boundary radius |
| `number_of_nodes` | -- | Number of basic mesh nodes |
| `mixing_length_profile` | -- | `"nearest_boundary"` or `"constant"` |
| `core_density` | kg/m^3 | Assumed core density |
| `surface_density` | kg/m^3 | Density at the surface |
| `gravitational_acceleration` | m/s^2 | Constant $g$ (for Adams-Williamson EOS) |
| `adiabatic_bulk_modulus` | Pa | Bulk modulus $B$ (for Adams-Williamson EOS) |
| `mass_coordinates` | bool | Whether to use mass coordinates |
| `eos_method` | int | 1 = Adams-Williamson, 2 = user-defined |

### `[energy]`

Toggle individual heat transport and source terms.

| Key | Type | Description |
|-----|------|-------------|
| `conduction` | bool | Enable conductive heat transport |
| `convection` | bool | Enable parameterized convection |
| `gravitational_separation` | bool | Enable gravitational separation of melt |
| `mixing` | bool | Enable convective mixing of melt fraction |
| `radionuclides` | bool | Enable radiogenic heating |
| `dilatation` | bool | Enable volumetric dilation/compression work |
| `tidal` | bool | Enable tidal heating |

### `[initial_condition]`

| Key | Unit | Description |
|-----|------|-------------|
| `surface_temperature` | K | Temperature at the top of the mantle |
| `basal_temperature` | K | Temperature at the CMB |

The initial profile is an adiabat anchored to the surface temperature by default, with the basal temperature as a fallback constraint.

### `[phase_liquid]` and `[phase_solid]`

Material properties for each end-member phase. All values are constant (no pressure or temperature dependence) unless lookup tables are specified.

| Key | Unit | Description |
|-----|------|-------------|
| `density` | kg/m^3 | Phase density |
| `viscosity` | Pa s | Dynamic viscosity |
| `heat_capacity` | J/kg/K | Specific heat capacity |
| `melt_fraction` | -- | Fixed melt fraction (0 for solid, 1 for liquid) |
| `thermal_conductivity` | W/m/K | Thermal conductivity |
| `thermal_expansivity` | 1/K | Thermal expansivity |

### `[phase_mixed]`

Parameters governing the mixed-phase (two-phase) region.

| Key | Unit | Description |
|-----|------|-------------|
| `latent_heat_of_fusion` | J/kg | Latent heat of melting |
| `rheological_transition_melt_fraction` | -- | Critical melt fraction for rheological transition |
| `rheological_transition_width` | -- | Width of the tanh transition in log-viscosity |
| `solidus` | -- | Path to solidus data file or analytic identifier |
| `liquidus` | -- | Path to liquidus data file or analytic identifier |
| `phase` | -- | `"solid"`, `"liquid"`, or `"composite"` |
| `phase_transition_width` | -- | Smoothing width at solidus/liquidus boundaries |
| `grain_size` | m | Grain size for permeability calculation |

## Loading configuration in Python

```python
from aragog.config import Config

# From a TOML file
params = Config.from_toml("path/to/config.toml")

# From a dictionary (used by the PROTEUS wrapper)
params = Config.from_dict(config_dict)

# Auto-detect format (TOML or INI)
params = Config.from_file("path/to/config.toml")
```

## Legacy INI format

The legacy `.cfg` format is still supported. Example files are in `src/aragog/cfg/`. The INI format uses the same section and key names, but with INI syntax instead of TOML.
