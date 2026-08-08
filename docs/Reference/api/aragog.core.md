# `aragog.core`

The `aragog.core` package carries the staged core-evolution module: the core as a component with its own state rather than an isothermal reservoir. Radial structure uses the closed-form Gaussian profile family (Labrosse et al. 2001; Nimmo 2015), so every budget term is an analytic integral or a fixed-order quadrature and the module stays at ODE cost. All evaluation runs through `jax.numpy` at 64-bit precision and is jit- and grad-safe.

| Name | Role |
|------|------|
| `GaussianCoreProfiles` | Density, enclosed mass, gravity (exact erf form), hydrostatic pressure, and the adiabat of the core. |
| `IronMeltingCurve` | Pure-iron melting curve (Anzellini et al. 2013, the PALEOS prescription) with a multiplicative light-element depression. |
| `CoreEnergyBudget` | Effective-heat-capacity energy balance: secular cooling plus smoothed inner-core-nucleation latent heat; a `legacy` mode reproduces the isothermal-reservoir closure exactly. |

::: aragog.core
