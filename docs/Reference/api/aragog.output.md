# `aragog.output`

The `aragog.output` package exposes two standalone diagnostic functions used by the entropy solver and by post-processing code. The primary output contract for callers is the `SolverOutput` dataclass returned by `EntropySolver.get_state()`; this module hosts auxiliary diagnostics that operate on a `SolverOutput` or its constituent arrays.

| Name | Role |
|------|------|
| `melt_fraction_global` | Volume-weighted (or porosity-weighted) global melt fraction from a staggered-node $\phi$ array and the cell volumes. |
| `rheological_front` | Depth (as fractional radius) of the rheological transition, given a basic-node $\phi$ array, the basic-node radii, and a critical melt fraction $\phi_\mathrm{rheo}$. |

There is no `Output` class in this package: NetCDF writing, multi-panel plotting, and the time-axis snapshot machinery are handled by the PROTEUS wrapper, not by Aragog itself. Standalone callers can construct equivalent output by reading the arrays in `SolverOutput` directly.

::: aragog.output
