# Aragog

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/docs.yaml?branch=main&label=Docs)](https://proteus-framework.org/aragog)
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/ci_tests.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/aragog/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/aragog/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/FormingWorlds/aragog/graph/badge.svg)](https://codecov.io/gh/FormingWorlds/aragog)

**1-D two-phase interior dynamics solver with mixing-length convective closure.**

Aragog integrates the spherically symmetric specific-entropy equation for a partially molten silicate mantle from the core-mantle boundary to the surface. Two design choices define Aragog's approach:

- **Mixing-length theory (MLT)** closes the convective heat flux *locally* at every radial node. The full radial entropy profile $S(r,t)$ is the prognostic variable, so solidification fronts, retained melt pockets, and EOS-resolved adiabats are recovered without an assumed reference state. See [Mixing-length theory](https://proteus-framework.org/aragog/Explanations/mixing_length).
- **Two-phase flow** treats the mushy mantle as a coexisting solid + melt mixture at every node. This activates gravitational separation of melt and solid, chemical mixing of melt fraction across the rheological transition, and a continuous (lever-rule) treatment of latent heat through the partial-melt regime. See [Two-phase flow in Aragog](https://proteus-framework.org/aragog/Explanations/two_phase_flow).

Together, MLT and two-phase flow let Aragog resolve the partial-melt window between first crystallisation and final solidification, where atmospheric outgassing, surface volatile budgets, and the timing of solidification are shaped by the coupled mantle-atmosphere evolution.
Aragog is part of the [PROTEUS](https://proteus-framework.org/PROTEUS) coupled atmosphere-interior evolution framework and is its production interior-energetics backend.

- Documentation: <https://proteus-framework.org/aragog>
- Source code: <https://github.com/FormingWorlds/aragog>

## Features

- **Entropy-form, two-phase interior dynamics solver.** Specific entropy $S(r,t)$ at staggered nodes is the only state variable; $T$, $\rho$, $\phi$, $c_p$, $\alpha$, $(\partial T/\partial P)_S$ are read from PALEOS or SPIDER-format pressure-entropy tables. Lever-rule blending, gravitational separation, and chemical mixing all enter the energy budget continuously across the solidus and liquidus.
- **Mixing-length theory with smooth viscous-to-inviscid blend.** Convective eddy diffusivity $\kappa_h$ is built per cell from the local entropy gradient, gravity, density, viscosity, and a mixing length $l(r)$, with a tanh blend at $\mathrm{Re}_\mathrm{crit} = 9/8$. Captures both the laminar and turbulent regimes inside one closure.
- **Production-grade integrator.** SUNDIALS CVODE via `scikits.odes` is the default (`solver_method = "cvode"`), paired with a JAX-derived analytic Jacobian (`use_jax_jacobian = true`) for production-tolerance coupled runs. scipy `Radau` and `BDF` are available as fall-backs (`solver_method = "radau"` / `"bdf"`); standalone installs without `scikits.odes` or JAX fall back automatically.
- **SPIDER bit-parity boundary conditions.** Default `core_bc = "energy_balance"` evolves $dS/dr|_\mathrm{cmb}$ as an extra ODE state, mirroring SPIDER's `bc.c:76-131`. Three other modes (`quasi_steady`, `gradient`, `bower2018`) are available for parity testing and quick exploration.
- **Per-call mass-weighted $\Delta\Phi$ cap.** SUNDIALS root function returns at the exact step where the global melt-fraction change first reaches the configured limit; required at the rheological transition where any rate estimate from $t = 0$ overshoots within the call window and stalls the adaptive $dt$.
- **Coupling to Zalmoxis.** External P-T tables, mesh, and per-node gravity profiles read from the structure solver are accepted via `eos_method = 2` and `mass_coordinates = true`, so the interior dynamics solve and the structure solve share a single self-consistent mantle.
- **Configurable radiogenic and tidal heating.** Six radionuclides ($^{40}\mathrm{K}$, $^{232}\mathrm{Th}$, $^{235}\mathrm{U}$, $^{238}\mathrm{U}$ for present-day heating; $^{26}\mathrm{Al}$ and $^{60}\mathrm{Fe}$ for early-Solar-System studies) plus a per-staggered-node tidal-heating array.

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

Aragog requires SPIDER-format pressure-entropy lookup tables on disk. Inside a PROTEUS coupled run the tables are produced on the fly by Zalmoxis from PALEOS; for standalone use, point `eos_dir` at any directory containing the ten required files (see [Reference: data](https://proteus-framework.org/aragog/Reference/data) for the schema and the canonical file list).

`FWL_DATA` is honoured as the default data root if it is set:

```sh
export FWL_DATA=/your/data/path
```

### Run a smoke integration

```python
from pathlib import Path

from aragog import aragog_file_logger
from aragog.solver import EntropySolver

aragog_file_logger(log_dir=str(Path.cwd()))

solver = EntropySolver.from_file(
    filename="src/aragog/cfg/abe_solid.toml",
    eos_dir="path/to/eos/tables",
)
solver.initialize()
solver.set_initial_entropy(2900.0)
solver.solve()

out = solver.get_state()
print(f"Status: {out.status}")
print(f"T_magma: {out.T_magma:.0f} K, T_core: {out.T_core:.0f} K, Phi: {out.Phi_global:.4f}")
```

`EntropySolver.from_file` builds `Parameters` from the TOML file and loads the EOS tables in one call. The bundled configs in `src/aragog/cfg/` are short standalone smoke setups; for a full walkthrough including the `SolverOutput` dataclass fields, see the [first-run tutorial](https://proteus-framework.org/aragog/Tutorials/firstrun). The production-tolerance defaults that PROTEUS uses (CVODE + JAX, `core_bc = "energy_balance"`, `mass_coordinates = true`, `phase_smoothing = "tanh"`, `kappah_floor = 10` m$^2$/s, the six-isotope radionuclide cocktail) are set on the PROTEUS side, not in `abe_solid.toml` or `abe_mixed.cfg`.

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
