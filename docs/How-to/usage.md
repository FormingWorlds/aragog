# Usage

Aragog is primarily used as a Python library, either directly via `EntropySolver` or, in coupled simulations, through the PROTEUS framework wrapper. The standalone command-line interface is minimal and intentional: production runs go through the Python API.

## Running Aragog from Python

The minimal workflow is **load configuration → instantiate solver → set initial entropy → solve → read SolverOutput**.

```python
from pathlib import Path
from aragog.solver import EntropySolver

# Build a solver from a TOML configuration file and a directory
# of SPIDER-format pressure-entropy EOS tables.
solver = EntropySolver.from_file(
    filename="src/aragog/cfg/abe_solid.toml",
    eos_dir="data/lookup",
)

solver.initialize()

# Set the initial entropy profile at staggered nodes (J/kg/K).
# A scalar S_init produces a uniform isentropic profile.
solver.set_initial_entropy(2900.0)

# When core_bc = "energy_balance", the CMB entropy gradient is part
# of the state vector. Set its initial value here, or omit the call
# to let the solver derive it from S_init via one-sided FD.
# solver.set_initial_dSdr_cmb(0.0)

solver.solve()

# Retrieve a SolverOutput dataclass with profiles, fluxes, and
# scalar diagnostics.
out = solver.get_state()
print(f"Status:           {out.status}")           # 0 success, -1 failure
print(f"T_magma:          {out.T_magma:.0f} K")    # surface temperature
print(f"T_core:           {out.T_core:.0f} K")
print(f"Phi_global:       {out.Phi_global:.4f}")
print(f"F_heat_total:     {out.F_heat_total:.3e} W/m^2")
```

`EntropySolver.from_file` is a convenience constructor; in PROTEUS, the wrapper builds `Parameters` and `EntropyEOS` programmatically and passes them to the constructor directly.

## What `SolverOutput` contains

`get_state()` returns a `SolverOutput` dataclass. The fields are flat NumPy arrays except where noted:

**Profiles (staggered nodes, length $N$):**

| Field | Unit | Description |
|-------|------|-------------|
| `S_final` | J/kg/K | Specific entropy at the end of the integration |
| `T_stag` | K | Temperature derived from $(P, S)$ via EOS |
| `phi_stag` | -- | Melt fraction |
| `rho_stag` | kg/m³ | Density |
| `visc_stag` | Pa·s | Dynamic viscosity (after the SPIDER-parity solid/liquid blend) |

**Mesh geometry:**

| Field | Unit | Description |
|-------|------|-------------|
| `P_stag` | Pa | Pressure at staggered nodes |
| `r_basic` | m | Basic-node radii (length $N+1$) |
| `r_stag` | m | Staggered-node radii (length $N$) |
| `vol` | m³ | Cell volume per staggered node |
| `mass_stag` | kg | Mass per staggered cell |

**Per-component fluxes (basic nodes, length $N+1$):**

| Field | Unit | Description |
|-------|------|-------------|
| `jcond_b` | W/m² | Conductive flux |
| `jconv_b` | W/m² | Convective (MLT) flux |
| `jgrav_b` | W/m² | Gravitational-separation heat flux |
| `jmix_b` | W/m² | Chemical-mixing heat flux |
| `dSdr_b` | J/kg/K/m | Entropy gradient at basic nodes |
| `phi_basic` | -- | Melt fraction at basic nodes |
| `T_basic` | K | Temperature at basic nodes |
| `cp_basic` | J/kg/K | Heat capacity at basic nodes |
| `rho_basic` | kg/m³ | Density at basic nodes |
| `heat_flux` | W/m² | Total heat flux at basic nodes |
| `heating` | W/kg | Internal heating at staggered nodes (length $N$) |
| `eddy_diff` | m²/s | Thermal eddy diffusivity at basic nodes |
| `cap_stag` | J/m³/K | Capacitance $\rho T$ at staggered nodes |

**Scalar diagnostics:**

| Field | Unit | Description |
|-------|------|-------------|
| `T_magma` | K | Top basic-node temperature; the value passed to the atmosphere module in coupled runs |
| `T_core` | K | Bottom staggered-node temperature (or the integrated `T_core` state in `bower2018` mode) |
| `Phi_global` | -- | Volume-weighted mean melt fraction |
| `Phi_global_vol` | -- | Porosity-weighted mean melt fraction |
| `M_mantle` | kg | Total mantle mass |
| `M_mantle_liquid` | kg | Liquid mantle mass |
| `M_mantle_solid` | kg | Solid mantle mass |
| `RF_depth` | -- | Rheological front depth, $1 - r_\mathrm{rf}/R_\mathrm{surf}$ |
| `E_th` | J | Thermal energy integral, $\sum m_i c_p(P_i, S_i) T_i$ |
| `Cp_eff` | J/kg/K | Mass-weighted mean heat capacity |
| `F_heat_total` | W/m² | Total surface heat flux |
| `dt_actual` | yr | Wall-clock integration span achieved by the solver |
| `status` | int | `0` on success; `-1` on solver failure (drives the PROTEUS retry ladder) |

## Driving Aragog from PROTEUS

When Aragog is invoked as a PROTEUS submodule, the wrapper (`proteus.interior_energetics.aragog.AragogRunner`) constructs `Parameters` from the PROTEUS config object and the current helpfile row, instantiates `EntropySolver` directly, sets the initial entropy from the previous step (or from a Zalmoxis-derived profile on the first iteration), runs `solve()`, and reads the results from `get_state()`.

PROTEUS handles:

- The mapping of PROTEUS config keys (`config.interior_energetics.*`, `config.interior_struct.*`) onto Aragog's `Parameters`.
- The choice of EOS table directory (`output/data/aragog_pt/` for PALEOS-generated tables, or a SPIDER-bundled directory for parity runs).
- The four-column external mesh file (`output/data/{spider,zalmoxis}_mesh.dat`) when `eos_method = 2`.
- A retry ladder around `solve()` that calls `set_initial_dSdr_cmb()` and `_atol_sf` to recover after a `status = -1` failure.

For an end-to-end coupled run, see the [PROTEUS documentation](https://proteus-framework.org/PROTEUS).

## Resetting and re-running the solver

To run multiple configurations or restart from a saved entropy profile, call `reset()` and then `set_initial_entropy()` again:

```python
solver.reset()                 # rebuilds the mesh and BCs from current parameters
solver.set_initial_entropy(S_new)
# Optional in energy_balance mode:
# solver.set_initial_dSdr_cmb(dSdr_cmb_new)
solver.solve()
out = solver.get_state()
```

`reset()` does not change the parameters; mutate `solver.parameters` first if a different time window or boundary value is needed.

## NetCDF output

Aragog itself does not write NetCDF files; that is handled by the PROTEUS wrapper, which packs the `SolverOutput` arrays into the per-iteration `int.nc` snapshot used by the PROTEUS analysis pipeline. See [Inspecting NetCDF output](netcdf.md) for the snapshot schema.

## Logging

`aragog.aragog_file_logger(log_dir=output_dir)` configures a console+file logger that writes `aragog.log` into the supplied directory. The logger is shared between the solver and the EOS layer; the standalone path uses it implicitly.

## Command-line interface

Aragog exposes a `aragog` console entry point with a single `click.Group` and no production subcommands. The CLI is intentionally minimal: data downloads, plotting, and run orchestration live in PROTEUS or in user scripts. Run the Python API from scripts or notebooks for standalone integrations.
