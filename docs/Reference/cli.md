# Command-line interface

The `aragog` console script (registered under `[project.scripts]` in `pyproject.toml`) is the standalone entry point.
It is a thin wrapper over the Python API: anything the CLI does is also reachable via `from aragog.solver import EntropySolver`.
PROTEUS-coupled runs do *not* use the CLI; the `proteus` driver builds `Parameters` programmatically and calls `EntropySolver` directly via `AragogRunner`.

## Group flags

```
aragog --version    # bare aragog version, e.g. 'aragog 26.05.08'
aragog --versions   # multi-line block: aragog + numpy / scipy / jax / scikits.odes / netCDF4
```

The `--versions` block is the recommended attachment for any bug report.

## Subcommands

| Subcommand | Purpose |
|---|---|
| `aragog new` | Scaffold a new TOML config from a bundled template. |
| `aragog list-configs` | Enumerate the bundled `cfg/abe_*.{toml,cfg}` examples. |
| `aragog validate` | Parse a config and report errors without solving. |
| `aragog show-config` | Dump the resolved `Parameters` tree as JSON. |
| `aragog run` | Solve a configured run end-to-end and write a NetCDF snapshot. |
| `aragog inspect` | Print key diagnostics from a `SolverOutput` NetCDF. |
| `aragog vnv` | Run a verification-figure script under `tools/verification/figures/`. |

Each subcommand is described below.

### `aragog new`

```
aragog new <name> [--from <template>] [--force]
```

Copies a bundled `cfg/abe_*.{toml,cfg}` template to the cwd as `<name>.toml`.
Default template is `abe_solid` (the canonical solid-phase cooling smoke); use `--from abe_mixed` (or any other name from `aragog list-configs`) to pick a different starting point.
The `.toml` suffix is appended automatically when omitted.
Refuses to overwrite an existing destination unless `--force` is passed.

```bash
aragog new earth_smoke              # writes ./earth_smoke.toml from abe_solid
aragog new earth_mixed --from abe_mixed
```

### `aragog list-configs`

```
aragog list-configs
```

Prints each bundled template in `src/aragog/cfg/` together with its first-comment-line summary.
Reads via `importlib.resources`, so it works in both editable and wheel installs.

### `aragog validate`

```
aragog validate <config>
```

Loads the configuration through `Parameters.from_file`, runs `Parameters.__post_init__` (BC normalise, optional EOS file load, ppm-to-mass-fraction conversion for radionuclides), and reports the first parser error.
The EOS tables and the solver are *not* touched.
Use this before launching a long CHILI run to catch typos, missing files, IBC=99-style dispatch errors, and the `[scalings]` strict-rejection.

On success, prints a one-line summary:

```
OK earth_smoke.toml: core_bc=energy_balance, IBC=2, OBC=1, eos_method=1, IC_method=1, radionuclides=4
```

On failure, raises a `ClickException` with `configuration error in <path>: <message>` and a non-zero exit code.

### `aragog show-config`

```
aragog show-config <config> [--indent N]
```

Dumps the resolved `Parameters` dataclass tree (post-`__post_init__`) as JSON on stdout.
Use for diff'ing PROTEUS-extracted configs against hand-crafted ones, for machine-readable extraction with `jq`, or for archival snapshots of the resolved schema.

```bash
aragog show-config earth_smoke.toml | jq .energy.kappah_floor
aragog show-config earth_smoke.toml --indent 0 > resolved.json   # one-line, jq-friendly
```

The walker uses `dataclasses.asdict` plus a numpy-array-to-list pass, so `energy.tidal_array` and any optional `mesh.eos_*` profile arrays serialise cleanly.

### `aragog run`

```
aragog run <config.toml> --eos-dir <path> --initial-entropy <S0> [options]
```

Solves a configured run end-to-end and writes a NetCDF snapshot.
Mirrors the Python recipe in [Tutorials: First run](../Tutorials/firstrun.md): load `Parameters`, initialise the solver, set the initial-condition state vector, solve, and call `SolverOutput.to_netcdf`.

#### Options

| Option | Type | Description |
|---|---|---|
| `--eos-dir PATH` | directory | SPIDER-format P-S EOS tables. Defaults to `$FWL_DATA/spider/lookup-fs` if the env var is set. |
| `--initial-entropy FLOAT` | J/kg/K | Initial isentropic profile. Required. |
| `--initial-dsdr-cmb FLOAT` | J/kg/K/m | Initial CMB entropy gradient. Used only when `core_bc = "energy_balance"`. Default 0. |
| `--out PATH` | file | NetCDF output path. Defaults to `<config-stem>.nc` in the cwd. |
| `--log-dir PATH` | directory | Directory for `aragog.log`. Defaults to the cwd. |
| `--log-level {debug,info,warning,error}` | level | Console log level. Default `info`. |
| `--set KEY.PATH=VALUE` | repeatable | Override a config field. See below. |

#### `--set` overrides

Repeatable flag that overrides any field in the resolved configuration tree without editing the TOML on disk.
Type-coercion order: `int` → `float` → `bool` (`'true'`/`'false'`) → JSON (`[...]`, `{...}`, `"..."`) → bare string.

```bash
# Tighten tolerances and bump kappah_floor for one run.
aragog run earth_smoke.toml \
    --eos-dir $FWL_DATA/spider/lookup-fs \
    --initial-entropy 2900.0 \
    --set energy.kappah_floor=20.0 \
    --set solver.atol=1e-11

# Drive a tidal-heating array via JSON.
aragog run earth_smoke.toml \
    --eos-dir $FWL_DATA/spider/lookup-fs \
    --initial-entropy 2900.0 \
    --set 'energy.tidal_array=[1e-9, 2e-9, 3e-9]'
```

Constraints:

- `--set` requires a `.toml` config (uses `tomllib`); legacy `.cfg` INI is rejected with a clear error.
- Unknown intermediate sections (e.g. `--set atmos.X=...`) raise immediately, before any solver construction.
- Unknown leaf keys (e.g. `--set energy.kapaha_floor=...` with the typo) surface as `UsageError: after applying --set overrides, the resolved config has unknown / mismatched fields: ...`.

### `aragog inspect`

```
aragog inspect <file.nc> [--json]
```

Reads a NetCDF snapshot produced by `SolverOutput.to_netcdf` (or by `aragog run --out`) and prints key diagnostics.

Default output is human-readable: scalar lines aligned in two columns with units in the label, scientific notation for very large / small values, plus a `S_final` / `T_basic` profile-range block (min / max / finite-count).

```
snapshot: out.nc
  aragog: 26.05.08
  created: 2026-05-09T00:00:00+00:00
  description: Aragog run from earth_smoke.toml
  dimensions: staggered=120, basic=121

  status              0  solver status (0 = success)
  time         1.0000e+06  simulation time [yr]
  T_magma         3500.0000  surface (magma) temperature [K]
  T_core          4200.0000  CMB temperature [K]
  Phi_global         0.4200  mass-weighted melt fraction [-]
  ...

  S_final: min=2.9000e+03, max=3.0000e+03 (120/120 finite)
  T_basic: min=1.5000e+03, max=4.5000e+03 (121/121 finite)
```

`--json` emits the same payload as a single JSON document keyed `{path, dimensions, attrs, scalars, profiles}`, suitable for `jq` post-processing or CI assertions.

```bash
aragog inspect out.nc --json | jq .scalars.T_magma
```

The command is robust to all-NaN profiles (e.g. `E_state` on the `const_properties` path) and to older snapshots missing some scalars; absent fields are silently skipped.

### `aragog vnv`

```
aragog vnv [<topic>] [--list]
```

Runs a verification-figure script under `tools/verification/figures/verify_<topic>.py`.
Bare invocation or `--list` enumerates the available scripts (`radio_decay`, `permeability`, `rhs_parity`, `eos_bilinear_jacobian`, `flux_decomposition`, `mass_coord_jacobian`, `utbl_cardano`).

```bash
aragog vnv --list
aragog vnv radio_decay
```

The V&V scripts are not packaged inside the `aragog` distribution; they live under the source tree and require a checkout. A non-editable `pip install aragog` install will print a clear "no V&V scripts found" error.
A script `main()` that raises is wrapped in a `ClickException` so the CLI exits non-zero with a clean message rather than a raw traceback.

## Recommended workflows

### First run (CLI-only)

```bash
aragog new my_first --from abe_solid
aragog validate my_first.toml
aragog run my_first.toml \
    --eos-dir $FWL_DATA/spider/lookup-fs \
    --initial-entropy 2900.0
aragog inspect my_first.nc
```

### Debugging a stuck PROTEUS-coupled run

```bash
# Extract the resolved config that PROTEUS built, save it to disk, then validate.
aragog show-config saved_from_proteus.toml > resolved.json
aragog validate saved_from_proteus.toml

# Try a single override to test a hypothesis without editing the TOML.
aragog run saved_from_proteus.toml \
    --eos-dir <eos> --initial-entropy <S0> \
    --set solver.atol=1e-11 \
    --out probe.nc
aragog inspect probe.nc
```

### Diff'ing two configs

```bash
aragog show-config a.toml --indent 2 > a.json
aragog show-config b.toml --indent 2 > b.json
diff a.json b.json
```
