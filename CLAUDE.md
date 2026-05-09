# Aragog AI Agent Guidelines

Aragog is the **interior energetics module** of the PROTEUS ecosystem.
It is a 1-D, two-phase, spherically symmetric magma-ocean solver with a mixing-length convective closure.
Two design choices set Aragog apart from the boundary-layer-theory (BLT) lineage of magma-ocean codes:

1. Mixing-length theory (MLT) closes the convective heat flux *locally* at every radial node, not globally through a Nusselt-Rayleigh scaling across a thin upper thermal boundary layer.
2. The partial-melt mantle is treated as a coexisting solid + melt mixture, not a single-phase fluid with a depth cutoff.

Together these resolve the partial-melt window between first crystallisation and final solidification.
Ecosystem-wide guidelines live in `../CLAUDE.md` (PROTEUS-level).
End-user documentation is in `docs/` (Zensical site at `proteus-framework.org/aragog/`).

## Quick reference

```bash
# Activate the production environment
conda activate proteus

# Tests (fast unit tier, mirrors CI)
pytest -m "unit and not slow" -n auto

# Full nightly tier with coverage
pytest -m "unit or smoke or slow" -n auto --cov=src/aragog

# Lint and format (CI canonical)
ruff check src/ tests/ tools/ && ruff format --check src/ tests/ tools/

# Documentation (Zensical, NOT raw mkdocs)
zensical serve              # live reload on docs/ and src/
zensical build --clean      # full build for CI parity

# Regenerate V&V figures
python tools/verification/figures/verify_radio_decay.py
# Three figures need a PROTEUS-side fixture:
export PROTEUS_SCRIPTS=$HOME/git/PROTEUS/scripts
python tools/verification/figures/verify_rhs_parity.py
```

## Environment

- The `proteus` conda env ships Python 3.12, numpy ≥ 2.0, JAX, equinox, and the SUNDIALS / `scikits-odes-sundials` stack. All production paths assume float64 (`jax.config.update('jax_enable_x64', True)` is set at JAX-import time inside Aragog).
- `FWL_DATA` may point at the SPIDER P-S EOS-table cache for standalone runs. PROTEUS-coupled runs do not need it; Zalmoxis writes the tables on the fly.
- Optional install groups (`pyproject.toml`):
  - `[jax]` — `scikits-odes-sundials`, `jax`, `equinox`. **Required for the production solver path.** A bare `pip install -e .` falls back to scipy `Radau`/`BDF` with a finite-difference Jacobian, which is correct for short tests but step-size-fragile on multi-Myr cooling runs.
  - `[test]` — `pytest`, `pytest-xdist`, `pytest-dependency`.
  - `[docs]` — `zensical`, `mkdocstrings[python]`, `pymdown-extensions`, `mkdocs-material`, `markdown-include`.

## Project layout

```
aragog/
  src/aragog/
    __init__.py              # Logger setup, CFG_DATA, EntropySolver re-export
    cli.py                   # click subcommands: run, inspect, validate, show-config, new, list-configs, vnv
    parser.py                # Legacy dataclass-based TOML/dict config parser
    utilities.py             # tanh_weight, smooth helpers
    cfg/                     # Bundled standalone configs (abe_solid.toml, abe_mixed.cfg, ...)
    config/                  # Attrs-based config classes (newer path)
      boundary.py energy.py initial_condition.py mesh.py
      phases.py radionuclides.py solver.py
    eos/
      entropy.py             # SPIDER-format P-S table loader (RegularGridInterpolator)
      entropy_phase.py       # Two-phase blender (lever rule for T, harmonic mean for rho,
                             # latent-augmented Cp, Costa 2009 viscosity log-blend)
    jax/
      eos.py                 # JAX bilinear EOS lookup (parity with numpy to float64 epsilon)
      nondim.py              # Non-dimensionalisation helpers
      phase.py               # JAX flux assembly: compute_mlt, compute_fluxes, compute_phase_state
      solver.py              # JAX RHS dSdt_energy_balance, make_radio_heating_fn, _utbl_tsurf_jax
    mesh/
      fixed_mesh.py          # Static finite-volume mesh; mixing_length(r) = nearest_boundary
      pressure_eos.py        # AdamsWilliamsonEOS (eos_method=1), UserDefinedEOS (eos_method=2)
    output/
      diagnostics.py         # SolverOutput dataclass, NetCDF writer (PROTEUS uses its own writer)
    solver/
      boundary.py            # Outer BC modes 1/4/5; CMB BCs; UTBL Cardano cubic
      cvode_jax.py           # build_jax_rhs_and_jacobian factory (the "Option Z" production path)
      entropy_solver.py      # EntropySolver class; from_file constructor; retry helpers
      entropy_state.py       # EntropyState + RHS assembly (numpy path), inlined MLT block
  tests/                     # ~366 unit + ~40 smoke + ~3 slow
    conftest.py              # `helper` fixture; per-file fixtures for EOS-dependent tests
  tools/
    verification/            # Existing first-principles verification runners
      figures/                # Reproducible V&V figure scripts (paired with docs/figures/vv/)
        _style.py             # Shared matplotlib style helper
        verify_*.py           # 7 scripts producing fig_NN_*.pdf and .png
        README.md
  docs/                      # Zensical sources (Diataxis: How-to, Tutorials, Explanations, Reference, Community)
  data/                      # Reference data for tests (e.g. abe_*.dat)
  output/                    # Run outputs (gitignored)
  mkdocs.yml                 # Nav (consumed by Zensical)
  pyproject.toml             # PEP 621 setuptools; ruff and pytest config
```

## Solver architecture

Two-phase entropy ODE on a 1-D radial mesh.
The state vector length depends on the core BC mode:

- `core_bc = 'energy_balance'` (default; SPIDER bit-parity): length $N+1$ where the extra state is $dS/dr|_\mathrm{cmb}$.
- `core_bc = 'quasi_steady'`: length $N$.
- `core_bc = 'gradient'`: length $N+2$.
- `core_bc = 'bower2018'`: experimental.

The right-hand side assembles four heat-flux components plus internal heating:

```
F_total = F_cond + F_conv + F_grav + F_mix
H_total = H_radio + H_tidal
```

- `F_cond` follows Fourier's law in entropy form using EOS-tabulated $(\partial T/\partial P)_S$.
- `F_conv` is mixing-length theory: $F_\mathrm{conv} = \rho T \kappa_h \max(-\partial S/\partial r, 0)$ with $\kappa_h$ blended between viscous (Stokes-like) and inviscid (free-fall) limits via a tanh on the cell Reynolds number at $\mathrm{Re}_\mathrm{crit} = 9/8$. The MLT block is inlined in `EntropyState.update` (numpy, `entropy_state.py:441-557`) and packaged as `aragog.jax.phase.compute_mlt` (JAX).
- `F_grav` is the gravitational-separation flux. Three permeability regimes (Stokes / Rumpf-Gupte / BKC) with tanh transitions at the equal-density-ratio limits $\zeta_1 = 0.0769452$ and $\zeta_2 = 0.771462$ (Bower 2018 Eqs. 13a-c).
- `F_mix` is the SPIDER-parity bracket form of chemical mixing.

**Production solver path: SUNDIALS CVODE + JAX analytic Jacobian.**
The `[jax]` extra installs `scikits-odes-sundials`, `jax`, and `equinox`.
The default behaviour with `solver_method = 'cvode'` and `solver.use_jax_jacobian = true` registers a JAX `jax.jacrev` factory that builds the Jacobian in a single backward pass instead of the $O(N)$ RHS calls a finite-difference Jacobian needs.
Without `scikits-odes-sundials`, the solver falls back to scipy `Radau` / `BDF`; without JAX with `use_jax_jacobian = true`, CVODE falls back to its own finite-difference Jacobian.

**Per-call $|\Delta\Phi|$ cap.**
`phi_step_cap > 0` registers a SUNDIALS root function $g(t, y) = \mathrm{cap} - |\Phi_\mathrm{global}(t,y) - \Phi_\mathrm{global}(t_\mathrm{start})|$ that returns CVODE control at the exact step where the mass-weighted mean melt fraction first changes by `cap`.
The scipy fallbacks register the equivalent logic as a `solve_ivp` event.
Production CHILI runs use `phi_step_cap = 0.05`.

## Standalone execution

```python
from pathlib import Path
from aragog import aragog_file_logger
from aragog.solver import EntropySolver

aragog_file_logger(log_dir=str(Path.cwd()))

solver = EntropySolver.from_file(
    filename='src/aragog/cfg/abe_solid.toml',
    eos_dir='/path/to/eos/tables',
)
solver.initialize()
# When core_bc='energy_balance', set the CMB gradient BEFORE the entropy IC:
# solver.set_initial_dSdr_cmb(0.0)
solver.set_initial_entropy(2900.0)
solver.solve()

out = solver.get_state()  # SolverOutput dataclass
```

The bundled `cfg/abe_solid.toml` and `cfg/abe_mixed.cfg` are short standalone smoke setups.
The CLI (`aragog`) wraps the same path with seven subcommands: `run`, `inspect`, `validate`, `show-config`, `new`, `list-configs`, `vnv`. `aragog run` accepts `--set <key.path>=<value>` overrides without editing the TOML on disk; see `docs/Reference/cli.md` for the full reference. The CLI is a thin wrapper over the Python API; PROTEUS-coupled runs bypass it and call `EntropySolver` directly via `AragogRunner`.

## PROTEUS integration

When PROTEUS drives Aragog the configuration lives in PROTEUS's TOML schema, not in Aragog's bundled configs.
The PROTEUS-side attrs class is `proteus.config._interior.Aragog`, and the wrapper is `proteus.interior_energetics.aragog.AragogRunner`.

Two call sites in PROTEUS:

- **Init solve**: `AragogRunner.setup_solver` at `aragog.py:293`, called once before the main loop.
- **Per-iteration evolve**: `AragogRunner.run_solver` at `aragog.py:1509`, called every coupling step. The helpfile-output assembly is `_build_helpfile_output` at `aragog.py:1740`.

`hf_row` exchange:

- *Reads*: `Time`, `F_atm`, `T_eqm`, `R_int`, `R_core`, `gravity`, `P_surf`, `core_density`, `core_heatcap`, `M_core`, `ini_entropy` (first call only).
- *Writes*: scalar mantle state — `T_magma`, `T_core`, `Phi_global`, `Phi_global_vol`, `RF_depth`, `T_pot`, `M_mantle`, `M_mantle_liquid`, `M_mantle_solid`, `Cp_eff`, `E_th_mantle`. Echo-back / passthrough — `M_core`, `F_int`. Source powers — `F_radio`, `F_tidal`, `F_cmb`, `Q_radio_W`, `Q_tidal_W`. Energy-conservation primitives — `E_state_J`, `E_state_cons_J`, `step_dE_F_int_J`, `step_dE_F_cmb_J`, `step_dE_Q_radio_J`, `step_dE_Q_tidal_J`, `step_dE_Q_radio_cons_J`, `step_dE_Q_tidal_cons_J`, `step_solver_residual_J`. The cumulative residuals (`dE_predicted_cons_J`, `E_residual_cons_J`, `E_residual_cons_frac`, `solver_residual_J`) are computed in `proteus.utils.coupler._populate_energy_residual` from the per-call writes above. `T_surf` and `dt_actual` are NOT written by the Aragog wrapper: the former is set by `proteus.atmos_clim.wrapper`, the latter is the consumed `out.dt_actual` returned to the caller as the simulated time step.
- *Echo-back*: the wrapper recomputes `core_density` from the on-disk Zalmoxis mesh and `hf_row['M_core']` at every solve entry (`resolve_core_density` at `aragog.py:66`), mirroring SPIDER's `-rho_core` re-derivation. This survives mesh-blending fall-backs and stale-cache cases.

Recommended PROTEUS-side knobs (in priority order):

1. `interior_struct.module = 'zalmoxis'` + `interior_energetics.module = 'aragog'` for new production runs.
2. `interior_energetics.aragog.backend = 'jax'` (default).
3. `interior_energetics.aragog.core_bc = 'energy_balance'` (default; SPIDER bit-parity).
4. `interior_energetics.aragog.solver_method = 'cvode'` (default).
5. `interior_energetics.aragog.mass_coordinates = true` (default; required for `energy_balance`).
6. `interior_energetics.aragog.phi_step_cap = 0.05` for typical evolution; leave at `0.0` unless mushy-zone melt-fraction oscillations show up early.
7. `interior_energetics.aragog.surface_bc_mode = 'flux'` (default) for atmosphere-driven flux BC; `'grey_body'` selects mode 1 (Aragog re-evaluates $\varepsilon\sigma(T_\mathrm{top}^4 - T_\mathrm{eqm}^4)$ per CVODE sub-step).
8. `interior_energetics.{rtol, atol} = 1e-10` (schema defaults); `atol_temperature_equivalent = 1e-8` matches SPIDER.
9. `planet.prevent_warming = false`. Must stay false; the clamp is energy-non-conserving and produces a spurious $T_\mathrm{magma}$ plateau when true.

Pitfalls:

- *Deleted `dilatation` field*: removed on 2026-05-04 along with the explicit Φ_vol volumetric-work source (it was a divergence double-count). Any config that sets `dilatation = true` or `dilatation = false` is rejected by the schema validator. Negative-regression tests guard against re-introduction of `Q_dil_total`, `step_dE_Q_dil_J`, and the schema field.
- *Resume + ULP radius drift*: a resumed run can hit single-ULP mismatch between saved Zalmoxis radii and Aragog's mesh bounds at `solver/entropy_solver.py:91-110`. The validator tolerates `max(1 m, 1e-9 * span)` to absorb this; if it trips, regenerate the mesh by re-running the structure module.
- *The PROTEUS wrapper bypasses Aragog's standalone CLI and TOML loader.* `AragogRunner` builds `Parameters` directly from the PROTEUS schema. The bundled `abe_*.{cfg,toml}` configs are only for standalone work.

## Zalmoxis coupling

When `interior_struct.module = 'zalmoxis'` and `interior_energetics.module = 'aragog'`, two artefacts cross the Zalmoxis → Aragog boundary on every successful structure re-solve:

1. **Mesh file** `<outdir>/data/zalmoxis_output.dat`: five-column TSV (`r`, `P`, `rho`, `g`, `T`). Aragog reads only the first four columns inside `solver.reset()` to rebuild its mass-coordinate mesh; the temperature column is a Zalmoxis diagnostic and is ignored on the Aragog side.
2. **Aragog P-T tables** in `<outdir>/data/aragog_pt/`: `{density, temperature, heat_capacity, adiabat_temp_grad, thermal_exp}_{melt, solid}.dat`. Written by `zalmoxis.eos_export.generate_aragog_pt_tables` (or `_2phase` for two-phase PALEOS), default resolution $200 \times 200$, P-range $[10^5, \min(10^{13},\ 150 M_\oplus + 200\,\mathrm{GPa})]$ Pa.

Aragog's table loader builds `RegularGridInterpolator((P, S), ...)` over the rectangular grid (the directory name `aragog_pt/` is a legacy label; the contents follow the SPIDER P-S convention).
Phase-filtering the input PALEOS table breaks the rectangularity assumption and silently drops scipy into unstructured (linear-ND) interpolation, ~100x slower per call.

## Import conventions

- No backward-compatibility shims; imports are direct to the actual module.
  - `from aragog import aragog_file_logger`
  - `from aragog.solver import EntropySolver`
  - `from aragog.eos.entropy import EntropyEOS`
  - `from aragog.eos.entropy_phase import EntropyPhaseEvaluator`
  - `from aragog.config import Parameters`  (or `aragog.parser` for the legacy dataclass path)
  - `from aragog.jax.solver import dSdt_energy_balance, make_radio_heating_fn`
  - `from aragog.jax.phase import compute_mlt, compute_fluxes`
- `from __future__ import annotations` is required at the top of every `.py` file (enforced by ruff isort).
- `aragog/__init__.py` re-exports `EntropySolver` at the bottom of the file with `# noqa: E402` to break the circular import with `aragog.solver`. Do not move it to the top.
- `__version__` should match `pyproject.toml`. Both are CalVer (`YY.MM.DD`).

## Testing

- ~366 unit, ~40 smoke, ~3 slow tests. Markers from `pyproject.toml [tool.pytest.ini_options]`:
  - `unit`: < 100 ms, no real solver call.
  - `smoke`: one full `EntropySolver.solve()` at relaxed tolerance.
  - `slow`: multi-Myr runs and tolerance-convergence studies.
- CI canonical invocations:
  - Push / PR (`.github/workflows/ci_tests.yml`): `pytest -m "unit and not slow" -n auto` on Ubuntu + macOS × Python 3.11/3.12/3.13.
  - Nightly (`.github/workflows/nightly.yml`, 02:30 UTC): `pytest -m "unit or smoke or slow" -n auto --cov=src/aragog` with the 85% Codecov floor enforced via `[tool.coverage.report].fail_under`.
- The `helper` fixture (in `tests/conftest.py`) provides path-resolution helpers; module-local fixtures handle EOS-dependent setup.
- Use `pytest.approx` or `np.testing.assert_allclose` for float comparisons, never `==`.
- Tests without a marker are invisible to CI. Add `@pytest.mark.unit` (or another tier) on every new test.
- For the LLM-prompt template that encodes these rules, see `docs/How-to/build_tests.md`.

## Documentation (Zensical)

- The site builds with **Zensical**, not raw mkdocs. Use `zensical serve` and `zensical build --clean`. Raw `mkdocs serve` may fail on theme resolution.
- Live reload: `zensical serve` watches `docs/` and `src/` automatically.
- URLs use `use_directory_urls: false` (pages are `*.html`).
- Nav is defined in `mkdocs.yml`. New pages need both the file and the nav entry.
- mkdocstrings `:::` directives auto-generate API pages from docstrings; add a stub under `docs/Reference/api/` for any new module.
- V&V figures live in `docs/figures/vv/fig_NN_<topic>.{pdf,png}` and are reproducible from `tools/verification/figures/verify_<topic>.py`. PNGs are embedded in the docs site; PDFs are the print master. Both are tracked.

## Code style

- `ruff` for linting and formatting (`pyproject.toml [tool.ruff]`).
- Single quotes (configured in ruff).
- `from __future__ import annotations` required in all files.
- Line length 96 (prefer < 92).
- Run `ruff format` AND `ruff check` before committing; the local ruff (0.12.x) lags CI ruff (0.15.x) and `format` does NOT catch lint rules like `E402` misplaced imports.
- NumPy-style docstrings (Parameters / Returns / Raises / Notes).
- Comments should explain WHY, not WHAT. No project-tracking labels (T1.x, Stage X, dates of past changes); the development history belongs in commit messages, not in inline comments.
- Citations in user-facing docs prefer SciX URLs (`https://scixplorer.org/abs/<bibcode>/abstract`); use DOI as a fallback. Never fabricate bibcodes — for un-indexed references (book chapters, German-language papers without ADS entries), cite as plain author / year / journal / page.
