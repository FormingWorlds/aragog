# Inspecting NetCDF output

Aragog itself does not write NetCDF files.
In standalone use, results come back as a `SolverOutput` dataclass from `EntropySolver.get_state()` (see [Usage](usage.md)).
When Aragog is driven by PROTEUS, the framework wrapper (`proteus.interior_energetics.aragog.AragogRunner._write_output_ncdf`) packs a subset of the `SolverOutput` arrays into a per-iteration interior snapshot at `output/data/<iter>_int.nc`, where `<iter>` is the integer time in years.
This page describes what is in those snapshots and how to inspect them.

The snapshot is intentionally a subset of `SolverOutput`: per-iteration scalar diagnostics ($T_\mathrm{magma}$, $T_\mathrm{core}$, $\Phi_\mathrm{global}$, $E_\mathrm{th}$, $C_p^\mathrm{eff}$, $F_\mathrm{heat}^\mathrm{total}$, the four `step_dE_*_J` integrals) live in `output/runtime_helpfile.csv` instead, since they are time-series quantities that PROTEUS aggregates across all interior steps.

## Inspecting a PROTEUS interior snapshot

```sh
ncdump -h output/data/<iter>_int.nc
```

Or open it with `netCDF4` or `xarray` for analysis and plotting:

```python
import xarray as xr

ds = xr.open_dataset("output/data/<iter>_int.nc")
print(ds)
```

## Default variables

Always written, regardless of `interior_energetics.write_flux_diagnostics`.

### Staggered-node profiles (length $N$)

| Variable | Source field | Unit | Description |
|---|---|---|---|
| `entropy_s` | `S_final` | J/kg/K | Specific entropy |
| `temp_s` | `T_stag` | K | Temperature derived from $(P, S)$ |
| `phi_s` | `phi_stag` | -- | Melt fraction |
| `density_s` | `rho_stag` | kg/m³ | Density |
| `log10visc_s` | `log10(visc_stag)` | log10(Pa s) | Base-10 logarithm of dynamic viscosity |
| `pres_s` | `P_stag / 1e9` | GPa | Pressure (note: GPa, not Pa) |
| `radius_s` | `r_stag / 1e3` | km | Staggered-node radii (note: km, not m) |
| `mass_s` | `mass_stag` | kg | Cell mass |
| `Htotal_s` | `heating` | W/kg | Per-node specific internal heating (radiogenic + tidal) |

### Basic-node fluxes (length $N+1$)

| Variable | Source field | Unit | Description |
|---|---|---|---|
| `radius_b` | `r_basic / 1e3` | km | Basic-node radii (note: km, not m) |
| `Ftotal_b` | `heat_flux` | W/m² | Total radial heat flux |

### Scalars

| Variable | Source field | Unit | Description |
|---|---|---|---|
| `time` | -- | yr | Snapshot time |
| `phi_global` | `Phi_global` | -- | Mass-weighted mean melt fraction |

## Diagnostic variables

Written only when `interior_energetics.write_flux_diagnostics = true`. These are the per-component heat flux decomposition and the basic-node thermodynamic state used for solver post-mortems.

### Basic-node fluxes (length $N+1$)

| Variable | Source field | Unit | Description |
|---|---|---|---|
| `Jcond_b` | `jcond_b` | W/m² | Conductive flux |
| `Jconv_b` | `jconv_b` | W/m² | Convective (MLT) flux |
| `Jgrav_b` | `jgrav_b` | W/m² | Gravitational-separation heat flux |
| `Jmix_b` | `jmix_b` | W/m² | Chemical-mixing heat flux |
| `dSdr_b` | `dSdr_b` | J kg⁻¹ K⁻¹ m⁻¹ | Entropy gradient |
| `eddy_diff_b` | `eddy_diff` | m²/s | Thermal eddy diffusivity (MLT $\kappa_h$) |

### Basic-node thermodynamic state (length $N+1$)

| Variable | Source field | Unit | Description |
|---|---|---|---|
| `phi_basic_b` | `phi_basic` | -- | Melt fraction at basic nodes |
| `T_basic_b` | `T_basic` | K | Temperature at basic nodes |
| `cp_basic_b` | `cp_basic` | J kg⁻¹ K⁻¹ | Heat capacity at basic nodes |
| `rho_basic_b` | `rho_basic` | kg/m³ | Density at basic nodes |

## Plotting from a snapshot

Default snapshot plot of the temperature, melt fraction, and total heat flux profiles versus radius:

```python
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("output/data/<iter>_int.nc")  # replace <iter>

r_stag_km  = ds["radius_s"].values  # already km
r_basic_km = ds["radius_b"].values  # already km

fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)

axes[0].plot(ds["temp_s"], ds["pres_s"])     # pres_s already in GPa
axes[0].set_xlabel("Temperature [K]")
axes[0].set_ylabel("Pressure [GPa]")
axes[0].invert_yaxis()

axes[1].plot(ds["phi_s"], ds["pres_s"])
axes[1].set_xlabel("Melt fraction")
axes[1].invert_yaxis()

axes[2].plot(ds["Ftotal_b"], r_basic_km)
axes[2].set_xlabel(r"Total heat flux [W m$^{-2}$]")
axes[2].set_ylabel("Radius [km]")
axes[2].set_xscale("symlog", linthresh=1e-3)

fig.tight_layout()
fig.savefig("interior_snapshot.pdf")
```

To plot the per-component heat fluxes (Figure 2 of [Heat transport](../Explanations/heat_transport.md#decomposition-on-a-fully-mushy-state)), enable `interior_energetics.write_flux_diagnostics = true` in the PROTEUS config and add:

```python
axes[2].plot(ds["Jcond_b"], r_basic_km, label="conduction")
axes[2].plot(ds["Jconv_b"], r_basic_km, label="convection")
axes[2].plot(ds["Jgrav_b"], r_basic_km, label="gravitational sep.")
axes[2].plot(ds["Jmix_b"],  r_basic_km, label="chemical mixing")
axes[2].legend(loc="best")
```

## Reading the snapshot back into Aragog

PROTEUS itself uses `proteus.interior_energetics.aragog.read_last_Sfield(<outdir>, t)` to re-load the previous step's entropy field at the start of the next iteration. The same helper is the canonical way to resume a run from a saved snapshot:

```python
from proteus.interior_energetics.aragog import read_last_Sfield

S_prev = read_last_Sfield(output_dir="output", time=t_prev)
solver.set_initial_entropy(S_prev)
```

The helper reads `entropy_s` from the snapshot, validates the array length against the live mesh, and returns the entropy in J/kg/K.

## Building your own snapshot from `SolverOutput`

For standalone runs, you can replicate the PROTEUS snapshot schema by hand. The minimal recipe matching `_write_output_ncdf` is:

```python
import netCDF4 as nc
import numpy as np

out = solver.get_state()  # SolverOutput from EntropySolver
n_stag  = len(out.S_final)
n_basic = len(out.r_basic)

ds = nc.Dataset("standalone_snapshot.nc", mode="w")
ds.description = "Aragog standalone snapshot"
ds.createDimension("staggered", n_stag)
ds.createDimension("basic", n_basic)

def add(name, data, dim, units=""):
    v = ds.createVariable(name, np.float64, (dim,))
    v[:] = data
    v.units = units

add("entropy_s",   out.S_final,                              "staggered", "J/kg/K")
add("temp_s",      out.T_stag,                               "staggered", "K")
add("phi_s",       out.phi_stag,                             "staggered", "")
add("radius_s",    out.r_stag / 1e3,                         "staggered", "km")
add("pres_s",      out.P_stag / 1e9,                         "staggered", "GPa")
add("radius_b",    out.r_basic / 1e3,                        "basic",     "km")
add("log10visc_s", np.log10(np.maximum(out.visc_stag, 1e-10)), "staggered", "Pa s")
add("density_s",   out.rho_stag,                             "staggered", "kg m-3")
add("Ftotal_b",    out.heat_flux,                            "basic",     "W m-2")
add("Htotal_s",    out.heating,                              "staggered", "W kg-1")
add("mass_s",      out.mass_stag,                            "staggered", "kg")

ds.close()
```

The default-on variables match the eleven entries above; add the diagnostic-tier variables (`Jcond_b`, `Jconv_b`, `Jgrav_b`, `Jmix_b`, `dSdr_b`, `eddy_diff_b`, `phi_basic_b`, `T_basic_b`, `cp_basic_b`, `rho_basic_b`) by extending the same pattern.
