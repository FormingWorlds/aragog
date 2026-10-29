# `aragog.surface_lid`

The `aragog.surface_lid` module holds the sub-grid surface lid: a conductive crust thinner than the top cell, carried as its thickness and its energy per area, with the temperature profile of the one-phase Stefan problem. The functions are pure; the solver evaluates the material properties and calls them in every right-hand side when `subgrid_lid` is on.

| Name | Role |
|------|------|
| `shape` | Slopes at the top and the base and the mean of the profile $f_s$ for the shape parameter $s$. |
| `surface_state` | Surface temperature, surface flux and base conduction of a lid under a grey-body surface. |
| `fixed_surface_state` | The same for a fixed surface temperature. |
| `convective_base_flux` | Boundary-layer flux of the melt under the lid at the critical Rayleigh number. |
| `stefan_rate` | Growth rate of the lid base, with the growth stop and the melt-back stop. |
| `existence_weight` | Weight of the lid from the thickness of a steady crust. |
| `lid_rates` | Rates of the two lid states, the surface loss and the flux out of the melt. |

For the physics, see [Surface lid](../../Explanations/surface_lid.md).

::: aragog.surface_lid
