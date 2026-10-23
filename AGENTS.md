# Aragog agent instructions

Aragog is the interior energetics module of the PROTEUS ecosystem: a 1-D, spherically symmetric, two-phase entropy solver for the mantle, with a mixing-length closure of the convective flux at every node and the partial melt treated as a mixture of solid and melt. PROTEUS selects it as `interior_energetics.module = 'aragog'`. Before a first edit:

- Run every command from the repository root. Tests for `src/aragog/<package>/<file>.py` go in `tests/`; the test rules are in `tests/AGENTS.md`.
- These commands decide whether a change is ready (PR CI runs the unit tier on Linux and macOS with Python 3.12 and 3.13; the nightly runs every tier):

```bash
pip install -e '.[test,jax]'        # jax: CVODE, JAX and equinox, the production solver path
export ARAGOG_TEST_EOS_DIR=...       # SPIDER P-S tables; without them the EOS tests skip
pytest -m "unit and not slow" -n auto -o "addopts="
ruff check src/ tests/
ruff format --check src/ tests/
bash tools/validate_test_structure.sh
python tools/agents/check_agents_md.py
```

The CI takes the test tables from the release asset `test-data-v1` of this repository (`spider_eos_test_data.tar.gz`). Docs: `pip install -e '.[docs]'`, then `zensical build --clean` (Zensical, not raw mkdocs). Verification figures: `python tools/verification/figures/verify_<topic>.py` (or `aragog vnv`) and `python tools/verification/run_core_verification.py`; three figure scripts read a PROTEUS fixture and need `PROTEUS_SCRIPTS` (`tools/verification/figures/README.md`).

<!-- fwl-core:begin sha256=1adfa510a4f117ec -->
## PROTEUS ecosystem rules

PROTEUS couples separate module repositories into one model of the evolution of rocky planets and their atmospheres. Its config selects one module per process; the config values are the lowercase names:

- `interior_energetics.module`: `aragog`, `spider` (or the built-in `boundary`); `interior_struct.module`: `zalmoxis`, `spider`
- `atmos_clim.module`: `agni`, `janus`, both with SOCRATES spectral radiative transfer
- `outgas.module`: `calliope`, `atmodeller`; `atmos_chem.module`: `vulcan`
- `star.module`: `mors`; `escape.module`: `zephyrus`, `boreas`; `orbit.module` (tides): `obliqua`, `lovepy`
- fwl-io downloads the reference data the modules use

A module change to anything PROTEUS calls or reads (a function signature, a config key, an output column, a unit) can break the coupled model while the module's own tests pass. Search `src/proteus/` in PROTEUS for the name before you change it, and name the affected PROTEUS call sites in the pull request.

### Physics and numerics

- Units differ between and within modules: MORS and VULCAN use cgs; CALLIOPE and atmodeller work with pressures in bar; AGNI takes surface pressure in bar at setup and holds Pa inside; PROTEUS states the unit of every `hf_row` key in `GetHelpfileKeys` (`src/proteus/utils/coupler.py`: atmospheric pressures in bar, interior pressures in Pa, model time in years, orbital periods in s), and its config reference pages state each key's unit (for example stellar mass in M_sun, initial partial surface pressures in bar, stellar age in Gyr). State the unit of every physical quantity in its docstring and convert explicitly at the boundary: a unit mismatch between two modules passes the tests of both and shows only in the coupled run.
- A physics test must fail for the most plausible wrong formula: check a conservation law, a bound, a monotonicity, or a published or analytical value, and assert that the wrong result falls outside the tolerance.
- Take each physical constant from one source per repository (in Python `scipy.constants` or the module's constants file). Two retyped values of one constant differ at round-off and hide real differences between code paths.
- Do not loosen a solver tolerance, a conservation check or a clamp to make a run or a test pass. Find the cause; a loosened check also hides the next defect.

### Validation

A new or changed physics routine gets the validation cases named in its confirmed plan: an analytical law, a published figure or table, or a cross-code check. Each case is a `reference_pinned` test that runs the real code on real data, with no mocks, asserts the tolerance, and writes its light and dark plot through the `validation_plot` fixture of proteus-mpl. `pytest -m reference_pinned --validation-plots` makes all plots again. Line plots are SVG, maps are AVIF; each file is at most 60 kB, under `docs/assets/validation/`. Each case has a subsection on its page in the Validation tab, with the reference, the tolerance, the test id and the command. Data read from a published figure comes from 2 independent figlab reads that agree within the stated error, as a CSV with its provenance under `tests/validation/data/`.

### Code

- Keep an inline comment block to 2 lines or fewer, and never more than 4, because a long block drifts from the code it describes. An explanation that needs more goes in the docstring, the commit message or the pull-request description.

### Branches and pull requests

- Work on a feature branch `<initials>/<short-description>`; `main` changes only through a reviewed pull request.
- Fill every section of the repository's pull-request template, where it has one.
- Before you push, run the checks listed at the top of this `AGENTS.md` and in `tests/AGENTS.md`.

### Where knowledge goes

Rules for every contributor go in the `AGENTS.md` files, the reason for a change in its commit message and pull-request description, and the scientific validation of a module in its `docs/Validation/` pages, where the repository has them. Do not add memory, notes or decision-log files to the repository: nobody maintains them, and they go stale.
<!-- fwl-core:end -->

<!-- fwl-voice:begin sha256=b4fe24baeef8b734 -->
### Commit messages and public text

Commit messages, pull-request text, code comments, docstrings, test names, test skip reasons, parametrize ids, log strings that ship with the code, and CI job and step names describe the change and the current state of the code. They name no tool used to write the change, carry no tool-attribution trailer, use no internal plan, phase or work-group labels, and use no em or en dashes (a page range in a citation is the exception).
<!-- fwl-voice:end -->

## Architecture

`src/aragog/`, by package:

- `solver/`: `entropy_solver.py` (`EntropySolver`, its `from_file` constructor and the CVODE and scipy integration), `entropy_state.py` (state and right-hand side of the NumPy path), `boundary.py` (surface and CMB boundary conditions), `cvode_jax.py` (CVODE with the JAX right-hand side and analytic Jacobian, the production path).
- `eos/`: `entropy.py` (P-S table loader), `entropy_phase.py` (two-phase evaluator), `consistency.py`, `table_cache.py`. `jax/`: the JAX counterparts of the EOS, the phase evaluator and the fluxes, and `nondim.py`, shared by both paths.
- `core/`: the core module (`core_bc = 'core_module'`): profiles, budget, entropy and dynamo criterion, melting curve, CMB boundary layer, stratification and the resolved layer.
- `config/` (attrs configuration classes), `parser.py` (the dataclass parser of the standalone configs in `cfg/`), `mesh/` (fixed mesh, pressure EOS), `output/` (diagnostics), `rheology.py` and `rheology_lid.py` (solid-state rheology), `cli.py` (the `aragog` command: `run`, `inspect`, `validate`, `show-config`, `new`, `list-configs`, `vnv`).

The state is the entropy at the mesh nodes, plus extra entries set by the core boundary condition: `core_bc` is one of `quasi_steady`, `energy_balance`, `gradient`, `bower2018`, `core_module` (`config/boundary.py`; `docs/Explanations/core_bc.md`). The heat flux is conduction, mixing-length convection (viscous and inviscid limits blended at the cell Reynolds number `RE_CRIT = 9/8`), gravitational separation (permeability regimes switching at the porosities 0.0769452 and 0.771462) and chemical mixing, with radiogenic and tidal heating. The default `solver_method = 'cvode'` with `use_jax_jacobian = true` integrates with SUNDIALS CVODE and a `jax.jacrev` Jacobian; without scikits-odes-sundials the solver falls back to scipy Radau with a warning, and without a JAX factory CVODE uses its finite-difference Jacobian. `phi_step_cap > 0` stops each call where the mean melt fraction has changed by the cap (`docs/How-to/phi-step-cap.md`).

## PROTEUS and Zalmoxis coupling

- PROTEUS configures Aragog from its own schema (`proteus.config._interior.Aragog`) and runs it through `proteus.interior_energetics.aragog.AragogRunner`, which calls `EntropySolver` directly; the CLI and the bundled `cfg/` files are for standalone runs only. The `hf_row` keys it reads and writes are in `docs/Explanations/proteus_coupling.md`, the recommended settings in `docs/How-to/proteus_coupling.md`.
- With `interior_struct.module = 'zalmoxis'`, the mesh comes from `<outdir>/data/zalmoxis_output.dat`, and the P-S tables written by Zalmoxis must stay a rectangular grid in (P, S), since the loader builds a `RegularGridInterpolator` on it (`eos/entropy.py`).
- A resumed run compares the saved mesh radii with the mesh to `max(1 m, 1e-9 * span)`; if it fails, regenerate the mesh with the structure module.

## Conventions

- Imports go to the defining module (`from aragog.solver import EntropySolver`, `from aragog.eos.entropy import EntropyEOS`); there are no compatibility shims. Every non-empty `.py` file starts with `from __future__ import annotations`; ruff enforces it (`I002`, `required-imports` in `pyproject.toml`; CI lints `src/` and `tests/`). `aragog/__init__.py` re-exports `EntropySolver` at its end (`# noqa: E402`) to break the import cycle with `aragog.solver`; keep it there.
- The version comes from the latest git tag through setuptools-scm (CalVer); `src/aragog/_version.py` is generated and ignored. Releases: `docs/How-to/releasing.md`.
- Style: ruff with single quotes and a line length of 96, NumPy docstrings, comments that say why. Citations in the docs prefer SciX URLs, with the DOI as fallback; never invent a bibcode.
- Docs: Zensical with `use_directory_urls: false`; a new page needs its file and its `mkdocs.yml` nav entry; a new module gets a stub under `docs/Reference/api/`. The Validation tab (`docs/Validation/`) holds the core and numerics verification pages and the overview of the `reference_pinned` cases; the script figures are in `docs/figures/vv/` (PNG on the site, PDF beside it), the plot pairs of the tests in `docs/assets/validation/`. `tests/test_core_verification_page.py` checks every tagged number of `docs/Validation/core.md` against `docs/figures/vv/core_verification_values.json`.

## Agent files

`AGENTS.md` and `tests/AGENTS.md` hold the instructions; each `CLAUDE.md` holds only `@AGENTS.md`, and `.github/copilot-instructions.md` is a short summary for review tools. The shared blocks come from PROTEUS (`tools/agents/sync_core.py`); edit them there, not here. `python tools/agents/check_agents_md.py` checks the block hashes, the `CLAUDE.md` files and the size caps, and CI runs it.
