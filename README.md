# Aragog

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![docs](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/docs.yaml?branch=main&label=docs)](https://proteus-framework.org/aragog)
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/ci_tests.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/FormingWorlds/aragog/graph/badge.svg)](https://codecov.io/gh/FormingWorlds/aragog)

**1-D entropy-form interior thermal evolution solver for rocky planetary mantles.**

Aragog integrates the spherically symmetric specific-entropy equation for a partially molten silicate mantle from the core-mantle boundary to the surface. Temperature, density, melt fraction, heat capacity, thermal expansivity, and the adiabatic gradient are all diagnostic quantities derived from $(P, S)$ via tabulated equations of state, so phase transitions are handled without an effective $c_p$ divergence at the solidus or liquidus. Aragog is part of the [PROTEUS](https://proteus-framework.org/PROTEUS) coupled atmosphere-interior evolution framework and is its production interior-energetics backend.

- Documentation: <https://proteus-framework.org/aragog>
- Source code: <https://github.com/FormingWorlds/aragog>

## Features

- **Entropy-form magma-ocean solver.** Specific entropy $S(r,t)$ at staggered nodes is the only state variable; $T$, $\rho$, $\phi$, $c_p$, $\alpha$, $(\partial T/\partial P)_S$ are read from PALEOS or SPIDER-format pressure-entropy tables.
- **Production-grade integrator.** SUNDIALS CVODE via `scikits.odes` is the default (`solver_method = "cvode"`), paired with a JAX-derived analytic Jacobian (`use_jax_jacobian = true`) for production-tolerance coupled runs. scipy `Radau` and `BDF` are available as fall-backs (`solver_method = "radau"` / `"bdf"`); standalone installs without `scikits.odes` or JAX fall back automatically.
- **SPIDER bit-parity boundary conditions.** Default `core_bc = "energy_balance"` evolves $dS/dr|_\mathrm{cmb}$ as an extra ODE state, mirroring SPIDER's `bc.c:76-131`. Three other modes (`quasi_steady`, `gradient`, `bower2018`) are available for parity testing and quick exploration.
- **Per-call mass-weighted $\Delta\Phi$ cap.** SUNDIALS root function returns at the exact step where the global melt-fraction change first reaches the configured limit; required at the rheological transition where any rate estimate from $t = 0$ overshoots within the call window and stalls the adaptive $dt$.
- **Coupling to Zalmoxis.** External P-T tables, mesh, and per-node gravity profiles read from the structure solver are accepted via `eos_method = 2` and `mass_coordinates = true`, so the magma-ocean solve and the structure solve share a single self-consistent mantle.
- **Six radionuclides per [Ruedas (2017)](https://scixplorer.org/abs/2017GGG....18.3530R/abstract).** $^{40}\mathrm{K}$, $^{232}\mathrm{Th}$, $^{235}\mathrm{U}$, $^{238}\mathrm{U}$ for present-day heating; $^{26}\mathrm{Al}$ and $^{60}\mathrm{Fe}$ available with the same parser interface for early-Solar-System studies.

## Quick start

### Install

```sh
git clone git@github.com:FormingWorlds/aragog.git
cd aragog
pip install -e ".[jax,test,docs]"
```

The optional extras are: `jax` (JAX, equinox, scikits-odes-sundials for the production CVODE+JAX path), `test` (pytest with xdist + cov), `docs` (Zensical + mkdocstrings for building the doc site). Plain `pip install -e .` works for an inspection-only install.

A PyPI release is available as `fwl-aragog`:

```sh
pip install fwl-aragog
```

### Equation-of-state tables

Aragog requires SPIDER-format pressure-entropy lookup tables. Inside PROTEUS the tables are produced on the fly by Zalmoxis or distributed with SPIDER; for a standalone install you can download the bundled set from the OSF repository:

```sh
aragog download all
```

By default the tables go to a platform-dependent cache directory (`aragog env` prints the path). Override it via `FWL_DATA`:

```sh
export FWL_DATA=/your/data/path
```

### Run a smoke integration

```python
from aragog.config import Config
from aragog.solver import EntropySolver

params = Config.from_file("input/abe_mixed.cfg").to_parameters()
solver = EntropySolver(params)
solver.initialize()
solver.set_initial_entropy(3300.0)
solver.solve()
out = solver.get_state()
```

The bundled `abe_mixed.cfg` uses production defaults: SUNDIALS CVODE, JAX analytic Jacobian, `core_bc = "energy_balance"`, `mass_coordinates = true`, `phase_smoothing = "tanh"`, `kappah_floor = 10` m$^2$/s, and the four long-lived radionuclides ($^{40}\mathrm{K}$, $^{232}\mathrm{Th}$, $^{235}\mathrm{U}$, $^{238}\mathrm{U}$) with Earth-mantle concentrations from [Turcotte & Schubert (2002)](https://scixplorer.org/abs/2002gdyn.book.....T/abstract).

## Test suite

```sh
pytest -m unit                           # ~2 min on a workstation
pytest -m "unit or smoke or slow"        # full nightly suite, 10 min wall + EOS-table data
```

Coverage is enforced at 85% by `[tool.coverage.report]` in `pyproject.toml`; the nightly workflow uploads to Codecov.

## PROTEUS integration

When PROTEUS drives Aragog, the configuration lives in PROTEUS's TOML schema, not in `input/abe_mixed.cfg`. The PROTEUS-side attrs class is `proteus.config._interior.Aragog`; the wrapper at `src/proteus/interior_energetics/aragog.py` translates PROTEUS settings into Aragog `Parameters` and registers the JAX CVODE factory.

Recommended PROTEUS-side knobs (in priority order):

1. `interior_struct.module = "zalmoxis"` + `interior_energetics.module = "aragog"` for new production runs.
2. `interior_energetics.aragog.core_bc = "energy_balance"` (default).
3. `interior_energetics.aragog.backend = "jax"` (default; the wrapper translates to `use_jax_jacobian = true`).
4. `interior_energetics.aragog.solver_method = "cvode"` (default).
5. `interior_energetics.aragog.phi_step_cap = 0.05` for typical evolution; leave at 0.0 unless mushy-zone melt-fraction oscillations show up early.

Full theory and the prioritised-settings table live in [`docs/Explanations/`](https://proteus-framework.org/aragog/Explanations/).

## Citation

If you use Aragog (or the original [SPIDER code](https://github.com/djbower/spider)) please cite:

- [Bower et al. (2018)](https://scixplorer.org/abs/2018PEPI..274...49B/abstract). Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets. *Physics of the Earth and Planetary Interiors*, **274**, 49 to 62.

The PALEOS pressure-entropy tables used by the production path are described in [Attia et al. (2026)](https://scixplorer.org/abs/2026arXiv260503741A/abstract), *PALEOS: Multiphase Equations of State and Mass-Radius Relations for Exoplanet Interiors* (submitted to A&A; arXiv:2605.03741).
