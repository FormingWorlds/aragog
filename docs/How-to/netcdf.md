# Inspecting NetCDF output

Aragog itself does not write NetCDF files. In standalone use, results come back as a `SolverOutput` dataclass from `EntropySolver.get_state()` (see [Usage](usage.md)). When Aragog is driven by PROTEUS, the framework wrapper packs the same `SolverOutput` arrays into a per-iteration interior snapshot under `output/data/<iter>_int.nc`. This page describes what is in those snapshots and how to inspect them.

## Inspecting a PROTEUS interior snapshot

Each PROTEUS coupling step produces a NetCDF file at `output/data/<iter>_int.nc`. Print its header with:

```sh
ncdump -h output/data/0001_int.nc
```

Or open it with `xarray` for analysis and plotting:

```python
import xarray as xr

ds = xr.open_dataset("output/data/0001_int.nc")
print(ds)
```

The dataset contains the staggered-node profiles, the basic-node fluxes, and the scalar diagnostics from `SolverOutput`, plus the mesh geometry needed to interpret them.

## Variables written by the PROTEUS wrapper

The exact variable list is set by the PROTEUS wrapper, but for a standard coupled run it includes the following groups. All entries derive directly from `SolverOutput` fields described in [Usage](usage.md).

### Staggered-node profiles (length $N$)

| Variable | Source field | Unit | Description |
|----------|-------------|------|-------------|
| `entropy_s` | `S_final` | J/kg/K | Specific entropy |
| `temperature_s` | `T_stag` | K | Temperature derived from $(P, S)$ |
| `melt_fraction_s` | `phi_stag` | -- | Melt fraction |
| `density_s` | `rho_stag` | kg/m³ | Density |
| `viscosity_s` | `visc_stag` | Pa·s | Effective viscosity |
| `pressure_s` | `P_stag` | Pa | Pressure |

### Basic-node fluxes (length $N+1$)

| Variable | Source field | Unit | Description |
|----------|-------------|------|-------------|
| `radius_b` | `r_basic` | m | Basic-node radii |
| `Fcond_b` | `jcond_b` | W/m² | Conductive flux |
| `Fconv_b` | `jconv_b` | W/m² | Convective (MLT) flux |
| `Fgrav_b` | `jgrav_b` | W/m² | Gravitational-separation heat flux |
| `Fmix_b` | `jmix_b` | W/m² | Chemical-mixing heat flux |
| `Ftotal_b` | `heat_flux` | W/m² | Total radial heat flux |
| `dSdr_b` | `dSdr_b` | J/kg/K/m | Entropy gradient |

### Scalar diagnostics

| Variable | Source field | Unit | Description |
|----------|-------------|------|-------------|
| `T_magma` | `T_magma` | K | Top basic-node temperature (surface T) |
| `T_core` | `T_core` | K | Bottom staggered-node temperature |
| `phi_global` | `Phi_global` | -- | Volume-weighted mean melt fraction |
| `M_mantle` | `M_mantle` | kg | Total mantle mass |
| `M_mantle_liquid` | `M_mantle_liquid` | kg | Liquid mantle mass |
| `M_mantle_solid` | `M_mantle_solid` | kg | Solid mantle mass |
| `RF_depth` | `RF_depth` | -- | Rheological front depth |
| `E_th` | `E_th` | J | Thermal energy integral |
| `Cp_eff` | `Cp_eff` | J/kg/K | Mass-weighted mean heat capacity |
| `F_heat_total` | `F_heat_total` | W/m² | Total surface heat flux |

## Plotting from a snapshot

A minimal plot of temperature, melt fraction, and the per-component fluxes versus radius:

```python
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("output/data/0001_int.nc")
r_km = ds["radius_b"].values * 1e-3

fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)

axes[0].plot(ds["temperature_s"], ds["pressure_s"] * 1e-9)
axes[0].set_xlabel("Temperature [K]")
axes[0].set_ylabel("Pressure [GPa]")
axes[0].invert_yaxis()

axes[1].plot(ds["melt_fraction_s"], ds["pressure_s"] * 1e-9)
axes[1].set_xlabel("Melt fraction")
axes[1].invert_yaxis()

axes[2].plot(ds["Fcond_b"], r_km, label="conduction")
axes[2].plot(ds["Fconv_b"], r_km, label="convection")
axes[2].plot(ds["Fgrav_b"], r_km, label="gravitational sep.")
axes[2].plot(ds["Fmix_b"], r_km, label="chemical mixing")
axes[2].set_xlabel(r"Heat flux [W m$^{-2}$]")
axes[2].set_xscale("symlog", linthresh=1e-3)
axes[2].legend(loc="best")

fig.tight_layout()
fig.savefig("interior_snapshot.pdf")
```

## Building your own NetCDF from `SolverOutput`

For standalone runs, you can write the same schema yourself with `xarray`:

```python
import xarray as xr
import numpy as np

out = solver.get_state()           # SolverOutput from EntropySolver

ds = xr.Dataset(
    data_vars={
        "entropy_s":      (["s"], out.S_final),
        "temperature_s":  (["s"], out.T_stag),
        "melt_fraction_s":(["s"], out.phi_stag),
        "density_s":      (["s"], out.rho_stag),
        "viscosity_s":    (["s"], out.visc_stag),
        "pressure_s":     (["s"], out.P_stag),
        "Fcond_b":        (["b"], out.jcond_b),
        "Fconv_b":        (["b"], out.jconv_b),
        "Fgrav_b":        (["b"], out.jgrav_b),
        "Fmix_b":         (["b"], out.jmix_b),
        "Ftotal_b":       (["b"], out.heat_flux),
        "dSdr_b":         (["b"], out.dSdr_b),
    },
    coords={
        "s": (["s"], out.r_stag),
        "b": (["b"], out.r_basic),
    },
    attrs={
        "T_magma":      float(out.T_magma),
        "T_core":       float(out.T_core),
        "Phi_global":   float(out.Phi_global),
        "F_heat_total": float(out.F_heat_total),
        "status":       int(out.status),
    },
)
ds.to_netcdf("standalone_snapshot.nc")
```

The PROTEUS wrapper goes through additional steps (per-step time stamps, attribute metadata, optional compression) that this minimal snippet skips.
