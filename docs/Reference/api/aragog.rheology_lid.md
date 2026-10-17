# `aragog.rheology_lid`

The `aragog.rheology_lid` module implements stagnant-lid identification, boundary-layer convective stress scaling (Foley and Bercovici, 2014), and the minimum plastic yield viscosity closure (Moresi and Solomatov, 1998; Tackley, 2000). Both functions accept an `xp` array namespace (`numpy` or `jax.numpy`) and run at 64-bit precision.

| Name | Role |
|------|------|
| `COHESION_FLOOR` | Minimum positive yield-stress floor ($10^{-6}\text{ Pa}$) preventing zero viscosity when cohesion and friction are zero at zero pressure. |
| `compute_stagnant_lid_state` | Identify the convecting interior state ($T_i, P_{T_i}, \eta_i, \mu_i$), stagnant-lid base and thickness ($r_\mathrm{lid,base}, d_\mathrm{lid}, \delta_\mathrm{rh}$), Frank-Kamenetskii contrast $\theta$, convective velocity $v_m$, driving stress $\tau_d$, lid yield stress $\tau_{y,\mathrm{lid}}$, smooth lid mask $w_\mathrm{lid}$, and regime indicator `lid_regime`. |
| `compute_effective_viscosity` | Combine diffusion-creep viscosity and plastic yield viscosity via the minimum closure in boundary-layer `'lid'` mode or local mixing-length `'local'` mode. |

For the physical derivations and asymptotic regimes, see [Solid-state convection and lithospheric rheology](../../Explanations/solid_state_convection.md).

::: aragog.rheology_lid
