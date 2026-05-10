# Aragog

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/docs.yaml?branch=main&label=Docs)](https://proteus-framework.org/aragog)
[![codecov](https://img.shields.io/codecov/c/github/FormingWorlds/aragog?label=coverage&logo=codecov)](https://app.codecov.io/gh/FormingWorlds/aragog)
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/ci_tests.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/nightly.yml)

**Aragog** is a 1-D, two-phase, spherically symmetric interior dynamics solver with a mixing-length convective closure, part of the [PROTEUS](https://proteus-framework.org/PROTEUS) coupled atmosphere-interior evolution framework.

Two design choices define Aragog's approach:

- **[Mixing-length theory](Explanations/mixing_length.md) (MLT)** closes the convective heat flux *locally* at every radial node. The full radial entropy profile $S(r,t)$ is the prognostic variable, so solidification fronts, retained melt pockets, and EOS-resolved adiabats are recovered without an assumed reference state.
- **[Two-phase flow](Explanations/two_phase_flow.md)** represents the mushy mantle as a coexisting solid + melt mixture at every node. This activates gravitational separation of melt and solid, chemical mixing of melt fraction across the rheological transition, and a continuous (lever-rule) treatment of latent heat through the partial-melt regime.

Together, MLT and two-phase flow let Aragog resolve the partial-melt window between first crystallisation and final solidification, where atmospheric outgassing, surface volatile budgets, and the timing of solidification are shaped by the coupled mantle-atmosphere evolution.
The mantle is integrated as a stiff initial-value problem with SUNDIALS CVODE (default) plus a JAX-derived analytic Jacobian, with scipy `Radau` and `BDF` available as fall-backs; conduction, convection (MLT), gravitational separation, chemical mixing, radiogenic heating, and tidal heating each contribute as configurable flux or source terms.

!!! note "Forming Worlds fork"
    This documentation describes the version of Aragog integrated into the [PROTEUS framework](https://proteus-framework.org/PROTEUS). For the original project, see [ExPlanetology](https://aragog.readthedocs.io).

!!! tip "New to Aragog?"
    See the **[Getting Started guide](getting_started.md)** for installation, first run, and basic usage.

!!! info "Standalone or PROTEUS-integrated?"
    Aragog is used in two distinct ways: as a standalone Python library, or as the interior energetics module inside the PROTEUS coupled framework. The two paths share the same numerical core but differ at the configuration boundary. See [Standalone usage](How-to/usage-paths.md) or [Coupling Aragog to PROTEUS](How-to/proteus_coupling.md) before picking a How-to or Tutorial.

## Features

- **Entropy formulation**: the prognostic variable is specific entropy $S(r,t)$. Temperature, density, melt fraction, heat capacity, thermal expansivity, and adiabatic gradient are looked up from a $(P, S)$ EOS table on each call, so the latent heat of fusion is encoded in the table rather than expressed as a $c_p$ spike across the solidus and liquidus.
- **Staggered finite-volume mesh**: entropy at cell centres (staggered nodes), fluxes at cell faces (basic nodes). The radial coordinate is either uniform in radius or uniform in mass coordinate, with a Newton solve for the spatial radii in the mass-coordinate variant.
- **Stiff implicit time integration**: SUNDIALS CVODE via `scikits.odes` is the default (`solver_method = "cvode"`), the same modified-Newton, cached-Jacobian solver SPIDER uses; scipy `Radau` and `BDF` are available as fall-backs (`solver_method = "radau"` / `"bdf"`). A phase-aware `max_step` reduction caps the step size near solidus and liquidus crossings.
- **Multiple core boundary conditions**: ``quasi_steady`` (heat-capacity-weighted flux partition, state vector length $N$), ``energy_balance`` (SPIDER-parity ODE evolution of the CMB entropy gradient, length $N+1$), ``gradient`` (entropy gradient as the primary state field, length $N+2$), and a ``bower2018`` mode for parity testing.
- **Single phase evaluator backed by P-S tables**: `EntropyPhaseEvaluator` wraps the loaded `EntropyEOS` and reproduces SPIDER's ``EOSEval_Composite_TwoPhase`` blending rules between solid, mixed, and liquid regimes via cubic-Hermite or tanh smoothing across the phase boundaries.
- **Configurable heat transport**: conduction, convection (MLT with smooth viscous/inviscid blend at $Re_\mathrm{crit} = 9/8$), gravitational separation of melt, chemical mixing flux (SPIDER bracket form), radiogenic heating, and tidal heating. Each is independently switchable.
- **JAX-accelerated path**: optional analytic Jacobian for CVODE, built by tracing the pure-functional `compute_fluxes` in `aragog.jax.phase` with `jax.jacrev`. Removes the $O(N)$ finite-difference RHS evaluations per Jacobian build.
- **TOML configuration**: attrs-validated TOML config files. The INI (`.cfg`) format is also accepted; TOML is recommended for new work.
- **Programmatic output via `SolverOutput`**: a dataclass returned by `EntropySolver.get_state()` carries the staggered-node profiles ($S$, $T$, $\phi$, $\rho$, $\eta$), basic-node fluxes ($F_\mathrm{cond}$, $F_\mathrm{conv}$, $F_\mathrm{grav}$, $F_\mathrm{mix}$, $\partial S/\partial r$), scalar diagnostics ($T_\mathrm{magma}$, $T_\mathrm{core}$, $\Phi_\mathrm{global}$, $E_\mathrm{th}$, $C_p^\mathrm{eff}$, $F_\mathrm{heat}^\mathrm{total}$), and the integration status flag used by the PROTEUS retry ladder.

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

If you use Aragog in published work, please cite the original numerical method paper and, where applicable, the PALEOS multiphase EOS framework that supplies the production-run $(P, S)$ tables:

- [Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract). *Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets*. **Physics of the Earth and Planetary Interiors**, 274, 49 to 62.
- [Attia et al. (2026)](https://scixplorer.org/abs/2026arXiv260503741A/abstract). *PALEOS: Multiphase Equations of State and Mass-Radius Relations for Exoplanet Interiors* (submitted to A&A; arXiv:2605.03741).

## Code availability

- [PROTEUS distribution (Forming Worlds)](https://github.com/FormingWorlds/aragog)
- [Upstream repository (ExPlanetology)](https://github.com/ExPlanetology/aragog)

If you plan to contribute to Aragog, please read our [Code of Conduct](Community/CODE_OF_CONDUCT.md) and [contributing guidelines](Community/CONTRIBUTING.md).
If you are running into problems, please do not hesitate to raise an [Issue](https://github.com/FormingWorlds/aragog/issues).

## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See [the included license](https://github.com/FormingWorlds/aragog/blob/main/LICENSE).
