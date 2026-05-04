# Configuration

Aragog reads its input from a single configuration file. The preferred format is **TOML**; the legacy INI format (`.cfg`) is still parsed for back-compatibility.

When Aragog is invoked from PROTEUS the configuration is built programmatically by the PROTEUS wrapper and the file-based path is bypassed; the same key names and value semantics apply in both cases.

## TOML configuration

A TOML configuration file specifies all model parameters grouped by section. Below is a representative coupled-mantle setup; the reference tables that follow document every key.

```toml
[solver]
start_time = 0
end_time = 1.0e9                  # yr
atol = 1e-9
rtol = 1e-9
tsurf_poststep_change = 30
event_triggering = false

[boundary_conditions]
outer_boundary_condition = 4      # 4 = prescribed flux (PROTEUS coupling)
outer_boundary_value = 280.0      # W/m^2
inner_boundary_condition = 1      # 1 = core cooling
inner_boundary_value = 0.0        # ignored when inner_bc=1
emissivity = 1.0
equilibrium_temperature = 273.0
core_heat_capacity = 880.0
tfac_core_avg = 1.147
param_utbl = false
param_utbl_const = 1.0e-7
core_bc = "energy_balance"        # CMB BC mode

[mesh]
outer_radius = 6.371e6            # m
inner_radius = 3.480e6            # m
number_of_nodes = 200
mixing_length_profile = "nearest_boundary"
core_density = 11000.0            # kg/m^3
surface_density = 4000.0          # kg/m^3
gravitational_acceleration = 9.81 # m/s^2 (scalar fallback)
adiabatic_bulk_modulus = 260e9    # Pa
adams_williamson_beta = 0.0
surface_pressure = 0.0            # Pa
mass_coordinates = false
eos_method = 1                    # 1 = Adams-Williamson, 2 = external file
eos_file = ""                     # used when eos_method = 2

[energy]
conduction = true
convection = true
gravitational_separation = true
mixing = true
radionuclides = true
tidal = false
eddy_diffusivity_thermal = 1.0
eddy_diffusivity_chemical = 1.0
kappah_floor = 0.0
bottom_up_grav_sep = true
phase_smoothing = "tanh"           # "tanh" (default, SPIDER-parity) or "cubic_hermite"
solver_method = "cvode"            # "cvode" | "radau" | "bdf"
use_jax_jacobian = true
tidal_array = [0.0]

[initial_condition]
initial_condition = 1              # 1=linear T, 2=user file, 3=adiabat
surface_temperature = 3600.0       # K
basal_temperature = 4000.0         # K
init_file = ""                     # used when initial_condition = 2

[phase_solid]
density = "data/lookup/density_solid.dat"
heat_capacity = "data/lookup/heat_capacity_solid.dat"
melt_fraction = 0.0
thermal_conductivity = 4.0
thermal_expansivity = 1.0e-5
viscosity = 1.0e21

[phase_liquid]
density = "data/lookup/density_melt.dat"
heat_capacity = "data/lookup/heat_capacity_melt.dat"
melt_fraction = 1.0
thermal_conductivity = 4.0
thermal_expansivity = 1.0e-5
viscosity = 1.0e2

[phase_mixed]
latent_heat_of_fusion = 4.0e5
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = "data/lookup/solidus.dat"
liquidus = "data/lookup/liquidus.dat"
phase = "composite"
phase_transition_width = 0.01
grain_size = 1.0e-3
matprop_smooth_width = 0.0
cp_blend = "latent"                # "latent" or "linear"
const_properties = false

[radionuclide_K40]
name = "K40"
t0_years = 4.55e9
abundance = 1.1668e-4
concentration = 310.0
heat_production = 2.8761e-5
half_life_years = 1.248e9
```

Any number of `[radionuclide_*]` sections can be added; each section name must start with `radionuclide_`.

## Configuration sections

### `[scalings]` (legacy)

A `[scalings]` section may still appear in older configuration files. Aragog no longer non-dimensionalises at the configuration level; the parser sets every scaling to 1.0 on load. The solver applies its own internal nondimensionalisation around the integrator (entropy reference $S_\mathrm{ref}$ and time reference $t_\mathrm{ref}$) which is not user-configurable. Any values supplied in `[scalings]` are ignored.

### `[solver]`

Time-integration controls.

| Key | Unit | Description |
|-----|------|-------------|
| `start_time` | yr | Start of integration |
| `end_time` | yr | End of integration |
| `atol` | -- | Absolute tolerance (floored at $10^{-8}$) |
| `rtol` | -- | Relative tolerance |
| `tsurf_poststep_change` | K | Maximum allowed surface-temperature change per coupling step (PROTEUS use) |
| `event_triggering` | bool | Reserved; not consumed by the entropy solver |

### `[boundary_conditions]`

Thermal boundary conditions at the surface and CMB.

| Key | Unit | Description |
|-----|------|-------------|
| `outer_boundary_condition` | int | Surface BC mode. `1` = grey-body atmosphere, `2` = Zahnle steam (not implemented), `3` = atmodeller coupling (not implemented), `4` = prescribed flux, `5` = prescribed temperature |
| `outer_boundary_value` | W/m² or K | Surface flux (modes `1`, `4`) or temperature (mode `5`) |
| `inner_boundary_condition` | int | CMB BC mode. `1` = core cooling, `2` = prescribed flux, `3` = prescribed temperature |
| `inner_boundary_value` | W/m² or K | CMB flux (modes `1`, `2`) or temperature (mode `3`) |
| `emissivity` | -- | Surface emissivity (used in mode `1`) |
| `equilibrium_temperature` | K | Radiative equilibrium temperature (mode `1`) |
| `core_heat_capacity` | J/kg/K | Core specific heat capacity |
| `tfac_core_avg` | -- | Core adiabat correction factor (default 1.147; Bower+2018 Table 2) |
| `param_utbl` | bool | Enable upper-thermal-boundary-layer parameterisation (default false) |
| `param_utbl_const` | K⁻² | UTBL constant in $\Delta T = b\,T^3$ |
| `core_bc` | str | CMB BC mode. `quasi_steady` (alpha-factor, state length $N$), `energy_balance` (SPIDER bit-parity, length $N+1$), `gradient` (entropy gradient as state, length $N+2$), or `bower2018` (parity-only, not recommended). Default: `quasi_steady`. |

### `[mesh]`

Spatial discretisation and pressure-density profile.

| Key | Unit | Description |
|-----|------|-------------|
| `outer_radius` | m | Planet (mantle top) radius |
| `inner_radius` | m | CMB radius |
| `number_of_nodes` | -- | Number of basic-grid nodes (cell faces) |
| `mixing_length_profile` | str | `"nearest_boundary"` or `"constant"` |
| `core_density` | kg/m³ | Mean core density |
| `eos_method` | int | `1` = analytic Adams–Williamson; `2` = external file (`eos_file`) |
| `surface_density` | kg/m³ | Surface mantle density (Adams–Williamson). Default 4000 |
| `gravitational_acceleration` | m/s² | Scalar gravity for Adams–Williamson; superseded by the per-node profile from `eos_file` when `eos_method = 2`. Default 9.81 |
| `adiabatic_bulk_modulus` | Pa | $K_S$ in Adams–Williamson. Default 260e9 |
| `adams_williamson_beta` | -- | A–W exponent $\beta$; `0.0` derives it from $K_S$. Default 0.0 |
| `surface_pressure` | Pa | Atmospheric overburden added to the pressure integration. Default 0.0 |
| `mass_coordinates` | bool | If true, use a SPIDER-parity mass-coordinate grid with Newton-solved spatial radii. Default false |
| `eos_file` | str | Path to a four-column file (`r [m]`, `P [Pa]`, `rho [kg/m³]`, `g [m/s²]`) used when `eos_method = 2`. PROTEUS supplies a Zalmoxis-generated file via this key. |

### `[energy]`

Heat-transport switches, transport parameters, and integrator selection.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `conduction` | bool | -- | Conductive flux $-k\,[(T/c_p)\partial S/\partial r + (\partial T/\partial P)_S\,\partial P/\partial r]$ |
| `convection` | bool | -- | Convective flux $\rho T\kappa_h(-\partial S/\partial r)$ when $\partial S/\partial r < 0$ |
| `gravitational_separation` | bool | -- | Stokes-regime separation of melt and solid; phase-boundary smoothed |
| `mixing` | bool | -- | Chemical mixing flux (SPIDER bracket form involving $\partial S_\mathrm{liq}/\partial P$ and $\partial S_\mathrm{sol}/\partial P$) |
| `radionuclides` | bool | -- | Internal radiogenic heating from any `[radionuclide_*]` sections |
| `tidal` | bool | -- | Tidal heating from `tidal_array` |
| `eddy_diffusivity_thermal` | float | 1.0 | Scalar multiplier on $\kappa_h$. Negative values pin $\kappa_h$ to the absolute value (SPIDER convention) |
| `eddy_diffusivity_chemical` | float | 1.0 | Scalar multiplier on $\kappa_c$. Negative values pin to absolute |
| `kappah_floor` | m²/s | 0.0 | Phase-modulated lower bound on $\kappa_h$ |
| `bottom_up_grav_sep` | bool | true | Apply SPIDER's bottom-up gating on the gravitational-separation flux |
| `phase_smoothing` | str | `"tanh"` | Phase-boundary smoothing. `"tanh"` (default) is SPIDER's two-branch `get_smoothing` with width `matprop_smooth_width = 0.01`; `"cubic_hermite"` is the fallback $16\,g\phi^2(1-g\phi)^2$ form. |
| `solver_method` | str | `"cvode"` | ODE integrator. `"cvode"` selects SUNDIALS CVODE via `scikits.odes` (default); `"radau"` and `"bdf"` use scipy `solve_ivp`. When `scikits.odes` is not installed the solver falls back to Radau with a warning. |
| `use_jax_jacobian` | bool | true | When `solver_method = "cvode"`, install a JAX-traced analytic Jacobian instead of the finite-difference Jacobian. Requires JAX and the PROTEUS-side factory registration; falls back to FD Jacobian otherwise. |
| `tidal_array` | array[float] | `[0.0]` | Per-staggered-node tidal heating in W/kg. Length must be `1` (broadcast scalar) or `number_of_nodes - 1` |

### `[initial_condition]`

Initial entropy profile (mapped from the temperature inputs via the EOS).

| Key | Unit | Description |
|-----|------|-------------|
| `initial_condition` | int | `1` = linear T(r) profile (default), `2` = read from `init_file`, `3` = adiabatic |
| `surface_temperature` | K | Top temperature for modes `1` and `3` |
| `basal_temperature` | K | Bottom temperature for mode `1` |
| `init_file` | str | Path to a two-column `r, T` (or `r, S`) file used when `initial_condition = 2` |

When the PROTEUS wrapper drives Aragog the initial entropy is supplied directly via `EntropySolver.set_initial_entropy()`; the section keys are still parsed but the values are overridden by the call.

### `[phase_liquid]` and `[phase_solid]`

End-member phase properties. Float values mean a constant; string values are file paths to one-column lookup tables.

| Key | Unit | Description |
|-----|------|-------------|
| `density` | kg/m³ | Reference density (or path to lookup) |
| `viscosity` | Pa·s | Dynamic viscosity (or path) |
| `heat_capacity` | J/kg/K | Specific heat capacity |
| `melt_fraction` | -- | Reference melt fraction (`0` for solid, `1` for liquid) |
| `thermal_conductivity` | W/m/K | Thermal conductivity |
| `thermal_expansivity` | 1/K | Thermal expansivity |
| `entropy` | J/kg/K | Optional reference entropy |

In the production PROTEUS path the per-phase keys are not consumed for material properties; the EOS tables provide $\rho$, $c_p$, $\alpha$, $k$, and $T$ as functions of $(P, S)$. The values are kept for the standalone constant-properties path (see `const_properties` below).

### `[phase_mixed]`

Mixed-phase (mushy) parameters and the optional constant-properties analytical mode.

| Key | Unit | Default | Description |
|-----|------|---------|-------------|
| `latent_heat_of_fusion` | J/kg | -- | Reference latent heat (legacy; production runs use $L(P)$ from the EOS) |
| `rheological_transition_melt_fraction` | -- | -- | Critical melt fraction for the viscosity transition |
| `rheological_transition_width` | -- | -- | Width of the tanh viscosity blend |
| `solidus` | str | -- | Path to a solidus reference (legacy T-based; not consumed by the entropy solver) |
| `liquidus` | str | -- | Path to a liquidus reference (legacy T-based) |
| `phase` | str | -- | Phase label |
| `phase_transition_width` | -- | -- | Legacy smoothing width |
| `grain_size` | m | -- | Grain size for the Stokes permeability |
| `matprop_smooth_width` | -- | 0.0 | SPIDER-parity blending width across phase boundaries |
| `cp_blend` | str | `"latent"` | `"latent"` (SPIDER parity) or `"linear"` for the mushy $C_p$ rule |
| `const_properties` | bool | false | Enable the analytic constant-properties path (no EOS tables needed) |
| `const_rho` | kg/m³ | 4000 | Constant density when `const_properties = true` |
| `const_Cp` | J/kg/K | 1000 | Constant heat capacity |
| `const_alpha` | 1/K | 1e-5 | Constant thermal expansivity |
| `const_cond` | W/m/K | 4.0 | Constant thermal conductivity |
| `const_log10visc` | -- | 2.0 | Constant log₁₀ viscosity |
| `const_T_ref` | K | 3500 | Reference temperature for $T(S) = T_\mathrm{ref}\exp((S - S_\mathrm{ref})/C_p)$ |
| `const_S_ref` | J/kg/K | 3000 | Reference entropy in the same expression |

## Loading configuration in Python

```python
from aragog.parser import Parameters

# From a TOML or INI file (auto-detected by suffix)
params = Parameters.from_file("path/to/config.toml")

# From a dictionary (used by PROTEUS via Config.from_dict)
from aragog.config import Config
params = Config.from_dict(config_dict)
```

Both calls return a `Parameters` dataclass; the `Config` facade exists primarily for the dict-driven path used by the PROTEUS wrapper.

## Legacy INI format

The legacy `.cfg` format is still parsed. Example files live under `src/aragog/cfg/`. Keys and section names match the TOML schema; only the surface syntax differs.
