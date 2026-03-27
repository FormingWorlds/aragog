# Aragog

**Aragog** is a 1-D spherically symmetric interior thermal evolution solver for rocky planetary mantles, part of the [PROTEUS](https://proteus-framework.org/PROTEUS) coupled atmosphere-interior evolution framework.

Aragog computes the time evolution of a radial temperature profile inside a spherical shell (mantle) using a staggered finite-volume mesh and scipy's BDF integrator. It handles solid, partially molten, and fully molten regimes through composable phase evaluators, with conduction, parameterized convection (mixing length theory), gravitational separation, and convective mixing of melt.

!!! tip "New to Aragog?"
    See the **[Getting Started guide](getting_started.md)** for installation, first run, and basic usage.

## Features

- **Temperature-pressure formalism**: single prognostic variable ($T$) with diagnostic properties from EOS
- **Staggered finite-volume mesh**: cell-center temperatures, face fluxes, with optional mass-coordinate transform
- **BDF time integration**: implicit variable-order solver (scipy `solve_ivp`) handles stiff conduction and phase-change problems
- **Composable phase evaluators**: separate solid, liquid, and mixed-phase evaluators assembled into a composite evaluator that switches per cell
- **Configurable heat transport**: conduction, convective mixing (MLT with viscous/inviscid regimes), gravitational separation, tidal and radiogenic heating
- **TOML configuration**: modern TOML config files with attrs-based validation, plus legacy INI support
- **NetCDF output**: mesh variables, fluxes, heating sources, and scalar diagnostics at any time step

!!! info "PROTEUS framework"
    When used within PROTEUS, Aragog is called at every coupling timestep to update the mantle temperature profile, surface temperature, and heat fluxes. The documentation for PROTEUS can be found [here](https://proteus-framework.org/PROTEUS).

## Quick links

<div class="grid cards" markdown>

-   :material-download: **Install**

    [Go to installation guide](How-to/installation.md)

-   :material-tune: **Configure**

    [Go to configuration](How-to/configuration.md)

-   :material-rocket-launch: **Use Aragog**

    [Go to usage](How-to/usage.md)

-   :material-book-open-variant: **Understand the model**

    [Go to model overview](Explanations/model.md)

-   :material-code-braces: **Browse the API**

    [Go to API reference](Reference/api/index.md)

-   :material-github: **Contribute / browse code**

    [Go to source code](https://github.com/FormingWorlds/aragog)

-   :material-bug: **Raise an issue**

    [Go to issues](https://github.com/FormingWorlds/aragog/issues)

-   :material-email: **Get in touch**

    [Go to contact](Community/contact.md)

</div>

## Citation

If you use Aragog, please cite:

Bower, D.J., Sanan, P., & Wolf, A.S. (2018). *Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets*. **Physics of the Earth and Planetary Interiors**, 274, 49--62. [https://doi.org/10.1016/j.pepi.2017.11.004](https://doi.org/10.1016/j.pepi.2017.11.004)

## Code availability

- [PROTEUS distribution (Forming Worlds)](https://github.com/FormingWorlds/aragog)
- [Upstream repository (ExPlanetology)](https://github.com/ExPlanetology/aragog)

If you plan to contribute to Aragog, please read our [Code of Conduct](Community/CODE_OF_CONDUCT.md) and [contributing guidelines](Community/CONTRIBUTING.md).
If you are running into problems, please do not hesitate to raise an [Issue](https://github.com/FormingWorlds/aragog/issues).

## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See [the included license](https://github.com/FormingWorlds/aragog/blob/main/LICENSE).
