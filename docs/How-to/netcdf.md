# Writing and inspecting NetCDF output

Aragog writes its own NetCDF4 snapshots in standalone use, and PROTEUS-coupled runs additionally produce a per-iteration interior snapshot via the framework wrapper.
The two paths are independent on purpose: the standalone schema captures the full `SolverOutput` dataclass (every scalar diagnostic and every profile), while the PROTEUS schema is a curated subset of that, written under a shorter variable-name convention that the PROTEUS post-processors and `read_last_Sfield` resume helper expect.

This page covers both: how to write a standalone snapshot, and how to inspect either.

## Standalone NetCDF write

`SolverOutput.to_netcdf` writes a self-contained file capturing every field on the dataclass.
For convenience `EntropySolver.write_netcdf` calls `self.get_state().to_netcdf(...)` so a typical script reads:

```python
from pathlib import Path
from aragog.solver import EntropySolver

solver = EntropySolver.from_file(
    filename="src/aragog/cfg/abe_solid.toml",
    eos_dir="/path/to/eos/tables",
)
solver.initialize()
solver.set_initial_dSdr_cmb(0.0)        # only when core_bc='energy_balance'
solver.set_initial_entropy(2900.0)
solver.solve()

# Single call: dumps the final state to a NetCDF4 file.
solver.write_netcdf(Path("output/snapshot.nc"), time=0.0,
                    description="Final state of abe_solid smoke run")
```

The writer mkdirs any missing parent directories, opens the file in `mode='w'` (overwriting any pre-existing file), and stamps the dataset with `aragog_version`, an ISO-8601 UTC `created_utc`, the user-supplied `description`, and the CF-1.8 conventions tag.
Every variable carries a `units` and a `long_name` attribute so the file is interpretable without consulting the source.

To dump an intermediate state (rather than the final), call `to_netcdf` directly on a `SolverOutput`:

```python
out = solver.get_state()
out.to_netcdf("output/snapshot.nc", time=t_yr)
```

When the `time` argument is omitted the writer falls back to `dt_actual` (the per-call integration duration); pass an explicit `time` whenever you have an absolute simulation clock.

### Variables in the standalone schema

Variables are named verbatim after the `SolverOutput` field names, with their units in SI by default.

#### Staggered-node profiles (length $N$)

| Variable | Source field | Unit | Description |
|---|---|---|---|
| `r_stag` | `r_stag` | m | Staggered-node radii |
| `P_stag` | `P_stag` | Pa | Pressure at staggered nodes |
| `S_final` | `S_final` | J kg⁻¹ K⁻¹ | Specific entropy |
| `T_stag` | `T_stag` | K | Temperature |
| `phi_stag` | `phi_stag` | -- | Melt mass fraction |
| `rho_stag` | `rho_stag` | kg m⁻³ | Density |
| `visc_stag` | `visc_stag` | Pa s | Dynamic viscosity |
| `vol` | `vol` | m³ | Per-shell volume |
| `mass_stag` | `mass_stag` | kg | Per-shell mass |
| `heating` | `heating` | W kg⁻¹ | Internal heating (radiogenic + tidal) |
| `cap_stag` | `cap_stag` | kg K m⁻³ | Capacitance $\rho T$ |

#### Basic-node profiles (length $N+1$)

| Variable | Source field | Unit | Description |
|---|---|---|---|
| `r_basic` | `r_basic` | m | Basic-node radii |
| `heat_flux` | `heat_flux` | W m⁻² | Total radial heat flux |
| `eddy_diff` | `eddy_diff` | m² s⁻¹ | Thermal eddy diffusivity ($\kappa_h$) |
| `jcond_b` | `jcond_b` | W m⁻² | Conductive flux |
| `jconv_b` | `jconv_b` | W m⁻² | Convective (MLT) flux |
| `jgrav_b` | `jgrav_b` | W m⁻² | Gravitational-separation flux |
| `jmix_b` | `jmix_b` | W m⁻² | SPIDER-parity phase-mixing flux |
| `dSdr_b` | `dSdr_b` | J kg⁻¹ K⁻¹ m⁻¹ | Radial entropy gradient |
| `phi_basic` | `phi_basic` | -- | Melt fraction at basic nodes |
| `T_basic` | `T_basic` | K | Temperature at basic nodes |
| `cp_basic` | `cp_basic` | J kg⁻¹ K⁻¹ | Heat capacity at basic nodes |
| `rho_basic` | `rho_basic` | kg m⁻³ | Density at basic nodes |

#### Scalars

Time, surface and CMB temperatures, both averaging conventions of the global melt fraction, the energy-balance integrals, and the solver status code:

`time`, `T_magma`, `T_core`, `Phi_global`, `Phi_global_vol`, `M_mantle`, `M_mantle_liquid`, `M_mantle_solid`, `RF_depth`, `E_th`, `E_state`, `E_state_cons`, `Cp_eff`, `F_heat_total`, `F_cmb`, `Q_radio_total`, `Q_tidal_total`, `step_dE_F_int_J`, `step_dE_F_cmb_J`, `step_dE_Q_radio_J`, `step_dE_Q_tidal_J`, `step_dE_Q_radio_cons_J`, `step_dE_Q_tidal_cons_J`, `step_solver_residual_J`, `dt_actual`, `status` (i4).

## Inspecting a snapshot

Either via the netCDF4 CLI, the Python netCDF4 library, or xarray:

```sh
ncdump -h output/snapshot.nc                     # header + variable list
ncdump output/snapshot.nc | less                 # full contents
```

```python
import xarray as xr

ds = xr.open_dataset("output/snapshot.nc")
print(ds)                                          # one-line summary
print(ds["Phi_global"].item())                     # scalar access
print(ds["S_final"].values.shape, ds["r_stag"].values.shape)
```

`ncdump` is bundled with most netCDF4 distributions; on macOS `brew install netcdf` provides it, on Linux `apt install netcdf-bin` or `dnf install netcdf`. `xarray` is the recommended tool for analysis-style work because it keeps coordinates and units alongside the data.

## Plotting from a snapshot

A minimal plot of temperature, melt fraction, and total heat flux versus pressure:

```python
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("output/snapshot.nc")

fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
P_GPa = ds["P_stag"].values / 1e9

axes[0].plot(ds["T_stag"], P_GPa)
axes[0].set_xlabel("Temperature [K]")
axes[0].set_ylabel("Pressure [GPa]")
axes[0].invert_yaxis()

axes[1].plot(ds["phi_stag"], P_GPa)
axes[1].set_xlabel("Melt fraction")
axes[1].invert_yaxis()

axes[2].plot(ds["heat_flux"], ds["r_basic"].values / 1e3)
axes[2].set_xlabel(r"Total heat flux [W m$^{-2}$]")
axes[2].set_ylabel("Radius [km]")
axes[2].set_xscale("symlog", linthresh=1e-3)

fig.tight_layout()
fig.savefig("interior_snapshot.pdf")
```

The per-component flux decomposition lives in the same file as `jcond_b`, `jconv_b`, `jgrav_b`, `jmix_b`; overlay them on the heat-flux panel for the full Figure 2 of [Heat transport](../Explanations/heat_transport.md#decomposition-on-a-fully-mushy-state).

## PROTEUS-coupled snapshots

When Aragog is driven by PROTEUS, the framework wrapper (`proteus.interior_energetics.aragog.AragogRunner._write_output_ncdf`) writes a *separate*, more compact per-iteration snapshot at `output/data/<iter>_int.nc` (where `<iter>` is the integer time in years), using shorter `*_s` and `*_b` variable names that the resume helper recognises.
This schema is independent of `SolverOutput.to_netcdf` so that PROTEUS post-processors and the resume path do not break when the standalone schema evolves.

### PROTEUS variable name convention

Always written:

| Variable | Source field | Unit | Description |
|---|---|---|---|
| `entropy_s` | `S_final` | J kg⁻¹ K⁻¹ | Specific entropy |
| `temp_s` | `T_stag` | K | Temperature derived from $(P, S)$ |
| `phi_s` | `phi_stag` | -- | Melt fraction |
| `density_s` | `rho_stag` | kg m⁻³ | Density |
| `log10visc_s` | $\log_{10}(\texttt{visc\_stag})$ | $\log_{10}$ Pa s | Base-10 dynamic viscosity |
| `pres_s` | `P_stag / 1e9` | GPa | Pressure (note: GPa, not Pa) |
| `radius_s` | `r_stag / 1e3` | km | Staggered-node radii (km) |
| `mass_s` | `mass_stag` | kg | Cell mass |
| `Htotal_s` | `heating` | W kg⁻¹ | Per-node internal heating |
| `radius_b` | `r_basic / 1e3` | km | Basic-node radii (km) |
| `Ftotal_b` | `heat_flux` | W m⁻² | Total radial heat flux |
| `time` | -- | yr | Snapshot time |
| `phi_global` | `Phi_global` | -- | Mass-weighted mean melt fraction |

Written only when `interior_energetics.write_flux_diagnostics = true`:

| Variable | Source field | Unit | Description |
|---|---|---|---|
| `Jcond_b` | `jcond_b` | W m⁻² | Conductive flux |
| `Jconv_b` | `jconv_b` | W m⁻² | Convective (MLT) flux |
| `Jgrav_b` | `jgrav_b` | W m⁻² | Gravitational-separation flux |
| `Jmix_b` | `jmix_b` | W m⁻² | Chemical-mixing flux |
| `dSdr_b` | `dSdr_b` | J kg⁻¹ K⁻¹ m⁻¹ | Entropy gradient |
| `eddy_diff_b` | `eddy_diff` | m² s⁻¹ | Thermal eddy diffusivity ($\kappa_h$) |
| `phi_basic_b` | `phi_basic` | -- | Melt fraction at basic nodes |
| `T_basic_b` | `T_basic` | K | Temperature at basic nodes |
| `cp_basic_b` | `cp_basic` | J kg⁻¹ K⁻¹ | Heat capacity at basic nodes |
| `rho_basic_b` | `rho_basic` | kg m⁻³ | Density at basic nodes |

Per-iteration scalar diagnostics ($T_\mathrm{magma}$, $T_\mathrm{core}$, $\Phi_\mathrm{global}$, $E_\mathrm{th}$, $C_p^\mathrm{eff}$, $F_\mathrm{heat}^\mathrm{total}$, the four `step_dE_*_J` integrals) live in `output/runtime_helpfile.csv` rather than the snapshot, since PROTEUS aggregates them across all interior steps.

### Resuming from a PROTEUS snapshot

PROTEUS itself uses `proteus.interior_energetics.aragog.read_last_Sfield` to re-load the previous step's entropy field at the start of the next iteration. The same helper is the canonical way to resume a run from a saved snapshot:

```python
from proteus.interior_energetics.aragog import read_last_Sfield

S_prev = read_last_Sfield(output_dir="output", time=t_prev)
solver.set_initial_entropy(S_prev)
```

The helper reads `entropy_s` (PROTEUS schema) from the snapshot, validates the array length against the live mesh, and returns the entropy in J/kg/K.
For standalone snapshots written via `to_netcdf`, the equivalent field is `S_final` and you can re-load it with one line of `xarray` or `netCDF4`.
