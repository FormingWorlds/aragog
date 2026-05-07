# Aragog

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://github.com/FormingWorlds/aragog/actions/workflows/docs.yaml/badge.svg)](https://proteus-framework.org/aragog)
[![Tests](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml/badge.svg)](https://github.com/FormingWorlds/aragog/actions/workflows/ci_tests.yml)
[![codecov](https://codecov.io/gh/FormingWorlds/aragog/graph/badge.svg)](https://codecov.io/gh/FormingWorlds/aragog)

**1-D entropy-form interior thermal evolution solver for rocky planetary mantles.**

Aragog integrates the spherically symmetric specific-entropy equation for a partially molten silicate mantle from the core-mantle boundary to the surface. Temperature, density, melt fraction, heat capacity, thermal expansivity, and the adiabatic gradient are all diagnostic quantities derived from $(P, S)$ via tabulated equations of state, so phase transitions are handled without an effective $c_p$ divergence at the solidus or liquidus. Aragog is part of the [PROTEUS](https://proteus-framework.org/PROTEUS) coupled atmosphere-interior evolution framework and is the production CHILI interior backend.

- Documentation: <https://proteus-framework.org/aragog>
- Source code: <https://github.com/FormingWorlds/aragog>

## Features

- **Entropy-form magma-ocean solver.** Specific entropy $S(r,t)$ at staggered nodes is the only state variable; $T$, $\rho$, $\phi$, $c_p$, $\alpha$, $(\partial T/\partial P)_S$ are read from PALEOS or SPIDER-format pressure-entropy tables.
- **Production-grade integrator.** SUNDIALS CVODE via `scikits.odes` with a JAX-derived analytic Jacobian (`jax.jacrev`) installed through a factory registered by the PROTEUS wrapper. Standalone runs that do not need the JAX path fall back to scipy `Radau` cleanly.
- **SPIDER bit-parity boundary conditions.** Default `core_bc = "energy_balance"` evolves $dS/dr|_\mathrm{cmb}$ as an extra ODE state, mirroring SPIDER's `bc.c:76-131`. Three other modes (`quasi_steady`, `gradient`, `bower2018`) are available for parity testing and quick exploration.
- **Per-call mass-weighted $\Delta\Phi$ cap.** SUNDIALS root function returns at the exact step where the global melt-fraction change first reaches the configured limit; required at the rheological transition where any rate estimate from $t = 0$ overshoots within the call window and stalls the adaptive $dt$.
- **Coupling to Zalmoxis.** External P-T tables, mesh, and per-node gravity profiles read from the structure solver are accepted via `eos_method = 2` and `mass_coordinates = true`, so the magma-ocean solve and the structure solve share a single self-consistent mantle.
- **Six radionuclides per Ruedas (2017).** $^{40}\mathrm{K}$, $^{232}\mathrm{Th}$, $^{235}\mathrm{U}$, $^{238}\mathrm{U}$ for present-day heating; $^{26}\mathrm{Al}$ and $^{60}\mathrm{Fe}$ available with the same parser interface for early-Solar-System studies.

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

The bundled `abe_mixed.cfg` uses production CHILI defaults: SUNDIALS CVODE, JAX analytic Jacobian, `core_bc = "energy_balance"`, `mass_coordinates = true`, `phase_smoothing = "tanh"`, `kappah_floor = 10` m$^2$/s, and the four long-lived radionuclides ($^{40}\mathrm{K}$, $^{232}\mathrm{Th}$, $^{235}\mathrm{U}$, $^{238}\mathrm{U}$) with Earth-mantle concentrations from Turcotte & Schubert (2014).

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

- Bower, D.J., P. Sanan, and A.S. Wolf (2018), Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets, *Phys. Earth Planet. Inter.*, **274**, 49-62, doi: <https://doi.org/10.1016/j.pepi.2017.11.004>. (open access at <https://arxiv.org/abs/1711.07303>; EarthArXiv mirror at <https://eartharxiv.org/k6tgf>).

The PALEOS pressure-entropy tables used by the production CHILI path are from Attia, M., Lichtenberg, T., Werlen, A., Bonati, I., Bower, D., et al. (2026), *PALEOS: A planetary entropy and structure model* (manuscript in preparation). Radioactive heat-production data are from Ruedas, T. (2017), *Geochem. Geophys. Geosyst.* **18**(9), 3530-3541, doi: <https://doi.org/10.1002/2017GC006997>.
