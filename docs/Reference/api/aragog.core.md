# `aragog.core`

The `aragog.core` package carries the staged core-evolution module: the core as a component with its own state rather than an isothermal reservoir. Radial structure uses the closed-form Gaussian profile family (Labrosse et al. 2001; Nimmo 2015), so every budget term is an analytic integral or a fixed-order quadrature and the module stays at ODE cost. All evaluation runs through `jax.numpy` at 64-bit precision and is jit- and grad-safe.

| Name | Role |
|------|------|
| `GaussianCoreProfiles` | Density, enclosed mass, gravity (exact erf form), hydrostatic pressure (quadrature or the printed Labrosse closed form), gravitational potential, and the adiabat of the core. |
| `IronMeltingCurve` | Pure-iron melting curve (Anzellini et al. 2013, the PALEOS prescription) with a multiplicative light-element depression. |
| `QuadraticMeltingCurve` | The quadratic-in-pressure parameterisation of Nimmo (2015, Eq. 6), interchangeable with the iron curve; the form benchmark models are defined in. |
| `CoreEnergyBudget` | Effective-heat-capacity energy balance: secular cooling, smoothed inner-core-nucleation latent heat, and the gravitational energy of light-element rejection; a `legacy` mode reproduces the isothermal-reservoir closure exactly. |
| `CoreEntropyBudget` | Entropy balance on top of the energy budget: cooling, latent, gravitational, and radiogenic sources against the conduction sink (closed form for the Gaussian adiabat), the dynamo entropy margin, and field strength via the Christensen, Holzwarth & Reiners (2009) energy-flux scaling. |

The budget terms are cross-validated against the open-source Leeds `thermal_history` implementation (secular exact, boundary terms to 0.5%), and the profile and melting machinery is pinned against Nimmo (2015) Table 2 (adiabatic length scales, ICB temperatures, melting gradients, adiabatic heat flows). This stage assumes bottom-up crystallization; parameter sets whose melting curve dips below the adiabat at the CMB (a top-down or snow topology) shut off the boundary terms rather than emitting heat from an ill-defined boundary.

::: aragog.core
