# `aragog.rheology`

The `aragog.rheology` module provides the homologous-temperature Arrhenius diffusion-creep viscosity kernel, the Byerlee plastic yield-stress criterion, the viscosity-contrast-dependent mixing-length factor, and the `SolidRheologyParams` configuration dataclass. All kernels accept an `xp` array namespace (`numpy` or `jax.numpy`) and run at 64-bit precision.

| Name | Role |
|------|------|
| `SolidRheologyParams` | Validated configuration dataclass for solid-state Arrhenius creep, Byerlee yielding, stagnant-lid diagnostics, and mixing-length calibration. |
| `merge_rheology` | Return a `SolidRheologyParams` instance with keyword field overrides applied. |
| `delegate_rheology` | Class decorator exposing `SolidRheologyParams` fields as read-only properties reading `self.rheology`. |
| `compute_arrhenius_enthalpy` | Pressure-dependent homologous activation enthalpy $H(P) = g_\mathrm{hom} R T_\mathrm{sol}(P)$ (J/mol). |
| `compute_arrhenius_viscosity` | Homologous-temperature Arrhenius diffusion-creep viscosity $\eta_\mathrm{diff}(T, P)$ (Pa s), clipped in $\log_{10}$ space after applying `water_prefactor`. |
| `compute_yield_stress` | Byerlee frictional yield stress $\tau_y(P) = \max(0, \min(c + \mu P, \tau_\mathrm{max}))$ (Pa). |
| `compute_strain_rate_local` | Second invariant of the local convective strain-rate tensor $\dot{\epsilon}_\mathrm{local} = |v_\mathrm{unyielded}| / (2 \max(l, 10^{-15}\text{ m}))$ ($\text{s}^{-1}$). |
| `compute_mlt_top_slope` | Viscosity-contrast-dependent upper mixing-length slope $s_\mathrm{top}(\theta) = s_{\mathrm{top},\infty} + (1 - s_{\mathrm{top},\infty}) \exp(-c_\theta \max(\theta, 0))$. |
| `viscous_mixing_length_factor` | Fourth-power mixing-length scaling factor $(l_v / l)^4$ for the viscous branch of mixing-length theory. |

For the physical derivations and asymptotic regimes, see [Solid-state convection and lithospheric rheology](../../Explanations/solid_state_convection.md).

::: aragog.rheology
