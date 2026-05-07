# `aragog.config`

The `aragog.config` package provides the modern `attrs`-based configuration classes and the `Config` facade used by the PROTEUS wrapper. Each subclass corresponds to one TOML section; the facade composes them into a legacy `Parameters` object that the solver consumes internally.

| Name | Role |
|------|------|
| `Config` | Top-level facade. Static constructors `from_toml(path)`, `from_dict(data)`, and `from_file(*paths)` return a legacy `Parameters` ready for `EntropySolver`. |
| `BoundaryConfig` | Surface and core boundary settings: `outer_boundary_condition`, `inner_boundary_condition`, emissivity, `core_bc` mode (`quasi_steady` default, `energy_balance`, `gradient`, `bower2018`), UTBL toggle. |
| `EnergyConfig` | Physics toggles (conduction, convection, gravitational separation, mixing, radionuclides, tidal), eddy-diffusivity ratio, tidal-array buffer. |
| `InitialConditionConfig` | IC type (linear, user-defined, adiabatic), surface and basal temperatures, init-file path. |
| `MeshConfig` | Mesh geometry, EOS method (Adams-Williamson or user-defined), surface density, gravity, bulk modulus, mass-coordinate flag, surface pressure. |
| `MixedPhaseConfig` | Mushy-zone parameters: latent heat, rheological transition, smoothing widths, solidus/liquidus paths, `cp_blend` strategy. |
| `PhaseConfig` | Single-phase (solid or liquid) properties: density, heat capacity, conductivity, expansivity, viscosity, optional entropy lookup. Each property accepts a float or a path-string lookup. |
| `RadionuclideConfig` | One radioisotope: name, reference time, abundance, concentration, heat production, half-life. Provides `get_heating(time)` (W/kg). |
| `ScalingsConfig` | Vestigial: forced to unity in `__attrs_post_init__`. Retained only for TOML compatibility. |
| `SolverConfig` | ODE driver settings: `start_time`, `end_time`, `atol`, `rtol`, surface-T step cap, event triggering. |

For the TOML field syntax expected on disk, see [How-to: configuration](../../How-to/configuration.md).
For the `Parameters` dataclass that the facade returns, see [`aragog.parser`](aragog.solver.md) (the solver consumes `Parameters`, not `Config`, internally).

## Facade

::: aragog.config.Config

## Boundary conditions

::: aragog.config.boundary.BoundaryConfig

## Energy sources

::: aragog.config.energy.EnergyConfig

## Initial condition

::: aragog.config.initial_condition.InitialConditionConfig

## Mesh

::: aragog.config.mesh.MeshConfig

## Phases (single and mixed)

::: aragog.config.phases.PhaseConfig

::: aragog.config.phases.MixedPhaseConfig

## Radionuclides

::: aragog.config.radionuclides.RadionuclideConfig

## Scalings

::: aragog.config.scalings.ScalingsConfig

## Solver driver

::: aragog.config.solver.SolverConfig
