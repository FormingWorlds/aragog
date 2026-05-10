# `aragog.output`

The `aragog.output` package exposes standalone diagnostic functions used by the entropy solver and by post-processing code. The primary output contract for callers is the `SolverOutput` dataclass returned by `EntropySolver.get_state()`; this module hosts auxiliary diagnostics that operate on a `SolverOutput` or its constituent arrays.

| Name | Role |
|------|------|
| `total_enthalpy` | EOS-consistent integrated mantle enthalpy $\sum h(P_i, S_i)\,m_i$. Underpins the `E_state` / `E_state_cons` columns reported by `aragog inspect` and by the PROTEUS energy-conservation diagnostics. |
| `volume_average` | Cell-volume-weighted mean of a staggered-node quantity. |
| `rheological_front` | Depth (as fractional radius) of the rheological transition, given a basic-node $\phi$ array, the basic-node radii, and a critical melt fraction $\phi_\mathrm{rheo}$. |

The global mantle melt fraction (mass-weighted $M_\mathrm{liq}/M_\mathrm{mantle}$ and porosity-derived $V_\mathrm{liq}/V_\mathrm{mantle}$) is exposed directly on `SolverOutput` as `Phi_global` and `Phi_global_vol`; no separate diagnostic helper is needed.

Standalone NetCDF output is handled by `SolverOutput.to_netcdf(path)` (also reachable as `EntropySolver.write_netcdf(path)` and as `aragog run --out path.nc`); the PROTEUS wrapper builds its own per-iteration `int.nc` snapshot independently.

::: aragog.output
