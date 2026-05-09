"""Aragog command-line entry point.

The ``aragog`` script (registered under ``[project.scripts]`` in
``pyproject.toml``) dispatches to seven subcommands:

* ``aragog new`` scaffolds a new TOML config from a bundled template.
* ``aragog list-configs`` enumerates the bundled
  ``cfg/abe_*.{toml,cfg}`` examples.
* ``aragog validate`` parses a config and reports errors without
  solving (no EOS load, no integrator).
* ``aragog show-config`` dumps the resolved ``Parameters`` tree
  as JSON for diff'ing or ``jq``-style extraction.
* ``aragog run`` solves a configured run end-to-end and writes a
  NetCDF snapshot. Supports ``--set <key.path>=<value>`` overrides
  without editing the TOML on disk.
* ``aragog inspect`` prints key diagnostics from a ``SolverOutput``
  NetCDF (``--json`` for machine-readable output).
* ``aragog vnv`` runs a verification-figure script under
  ``tools/verification/figures/`` by topic name.

The CLI is a thin wrapper over the Python API; anything reachable
here is also reachable via ``from aragog.solver import EntropySolver``.
PROTEUS-coupled runs do not use this CLI; the ``proteus`` driver
calls ``EntropySolver`` directly via ``AragogRunner``.

For the full subcommand reference see ``docs/Reference/cli.md``.
"""

from __future__ import annotations

import importlib
import importlib.resources
import importlib.util
import logging
import os
import sys
from importlib.resources.abc import Traversable
from pathlib import Path

import click

from aragog import __version__ as _ARAGOG_VERSION

logger = logging.getLogger('fwl.' + __name__)


def _version_message() -> str:
    """Return a multi-line version block: aragog + key dependencies.

    Lists the installed versions of the libraries that materially
    affect solver behaviour (numpy, scipy, JAX, scikits.odes,
    netCDF4). Helps users (and bug reports) pin down a build before
    debugging a numerical issue.
    """
    lines = [f'aragog {_ARAGOG_VERSION}']
    for name in ('numpy', 'scipy', 'jax', 'scikits.odes', 'netCDF4'):
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, '__version__', 'unknown')
        except Exception:
            ver = 'not installed'
        lines.append(f'  {name}: {ver}')
    return '\n'.join(lines)


@click.group()
@click.version_option(
    version=_ARAGOG_VERSION,
    message='%(prog)s %(version)s',
    help='Print the bare aragog version (one line) and exit.',
)
@click.option(
    '--versions',
    'show_versions',
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=lambda ctx, param, value: (
        click.echo(_version_message()) or ctx.exit(0) if value else None
    ),
    help=(
        'Print aragog version + numpy/scipy/jax/scikits.odes/netCDF4 versions '
        'and exit (multi-line; recommended attachment for bug reports).'
    ),
)
def cli() -> None:
    """Aragog: 1-D entropy-form magma-ocean solver."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')


def _resolve_log_level(name: str) -> int:
    return getattr(logging, name.upper())


def _bundled_cfg_dir() -> Traversable:
    """Return a Traversable handle to ``aragog/cfg/`` (works for both
    editable installs and wheel installs)."""
    return importlib.resources.files('aragog').joinpath('cfg')


def _vnv_figures_dir() -> Path:
    """Return the on-disk path to ``tools/verification/figures/``.

    The V&V figure scripts are not packaged inside the ``aragog``
    distribution; they live under the source tree and are expected
    to be run from a checkout. Resolves relative to the repo root,
    walking up from the ``aragog/cli.py`` file.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / 'tools' / 'verification' / 'figures'


def _coerce_value(raw: str):
    """Coerce a CLI override string to a Python value.

    Heuristic ordering: int -> float -> bool ('true'/'false') ->
    JSON (`[...]`, `{...}`, or a quoted string) -> bare string.
    The order matters: ``20`` becomes int 20, ``20.0`` becomes
    float 20.0, ``true`` becomes True, ``[1e-9, 2e-9]`` becomes
    a list of floats. Anything that survives the four typed
    parses is kept as a string; this is the right behaviour for
    file paths and string-typed config fields like
    ``mixing_length_profile = 'nearest_boundary'``.

    Note: bare ``null`` returns the string ``'null'``, not Python
    ``None``. The JSON path triggers only on ``[``, ``{``, or ``"``
    prefixes (otherwise a string like ``'null'`` would be silently
    promoted from a path-like value to None). To clear a field to
    None, edit the TOML directly; ``--set`` is not the right
    surface for that.
    """
    import json

    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.lower() == 'true':
        return True
    if raw.lower() == 'false':
        return False
    if raw.startswith(('[', '{', '"')):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _apply_overrides(data: dict, overrides: tuple[str, ...]) -> dict:
    """Apply ``--set <dotted.key>=<value>`` overrides to a nested dict.

    Returns a deep-copied dict with the overrides applied. Each
    override must be of the form ``key.path=value``; the dotted
    path is walked through nested dicts and the final segment is
    overwritten with the type-coerced value.

    Raises ``click.UsageError`` if the spec is malformed (no ``=``,
    empty path segment) or if an intermediate path segment is
    missing or non-dict in the input. Does NOT validate the leaf
    key against the dataclass schema; an unknown leaf key surfaces
    later as a TypeError at ``Config.from_dict`` construction
    time, which the caller wraps with a clearer message.
    """
    import copy

    out = copy.deepcopy(data)
    for spec in overrides:
        if '=' not in spec:
            raise click.UsageError(f'--set requires KEY=VALUE; got {spec!r} (no "=" found).')
        key_path, _, raw_value = spec.partition('=')
        keys = [k for k in key_path.split('.') if k]
        if not keys or len(keys) != len(key_path.split('.')):
            raise click.UsageError(
                f'--set key path is malformed: {key_path!r} (empty segment).'
            )
        if len(keys) < 2:
            # A single-segment path (e.g. --set energy=20.0) would
            # overwrite the entire `energy` section dict with a scalar,
            # surfacing later as a TypeError from
            # `_EnergyParameters(**20.0)`. That error message
            # ("argument of type 'float' is not iterable") is more
            # confusing than naming the malformed key path here.
            raise click.UsageError(
                f'--set key path must contain at least one dot separator '
                f'(e.g. energy.kappah_floor=20.0); got {key_path!r}. '
                'Setting an entire section with a single value is not '
                'supported.'
            )
        target = out
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                raise click.UsageError(
                    f'--set {key_path!r}: section {k!r} not found in '
                    'config or is not a section.'
                )
            target = target[k]
        target[keys[-1]] = _coerce_value(raw_value)
    return out


# ---------------------------------------------------------------------------
# aragog run
# ---------------------------------------------------------------------------


@cli.command(name='run')
@click.argument('config', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    '--eos-dir',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        'Directory with SPIDER-format P-S EOS tables. '
        'Defaults to ``$FWL_DATA/spider/lookup-fs`` if the environment '
        'variable is set; otherwise required.'
    ),
)
@click.option(
    '--initial-entropy',
    type=float,
    default=None,
    help=(
        'Initial isentropic profile [J/kg/K]. Required unless the '
        'config supplies a usable initial condition.'
    ),
)
@click.option(
    '--initial-dsdr-cmb',
    type=float,
    default=0.0,
    show_default=True,
    help=(
        'Initial dS/dr at the CMB [J/kg/K/m]. Used only when '
        '``boundary_conditions.core_bc = "energy_balance"`` (the '
        'default), where the CMB entropy gradient is an extended '
        'state variable.'
    ),
)
@click.option(
    '--out',
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        'NetCDF output path. Defaults to ``<config-stem>.nc`` in the current working directory.'
    ),
)
@click.option(
    '--log-dir',
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help='Directory for ``aragog.log``. Defaults to the cwd.',
)
@click.option(
    '--log-level',
    type=click.Choice(_LOG_LEVELS, case_sensitive=False),
    default='INFO',
    show_default=True,
    help='Console log level.',
)
@click.option(
    '--set',
    'set_overrides',
    multiple=True,
    metavar='KEY.PATH=VALUE',
    help=(
        'Override a configuration field, e.g. '
        '`--set energy.kappah_floor=20.0`. Repeatable. Type '
        "coercion order: int, float, bool ('true'/'false'), "
        'JSON (`[...]`, `{...}`, `"..."`), then bare string. '
        'Requires a .toml config; .cfg legacy INI is not supported.'
    ),
)
def run(
    config: Path,
    eos_dir: Path | None,
    initial_entropy: float | None,
    initial_dsdr_cmb: float,
    out: Path | None,
    log_dir: Path | None,
    log_level: str,
    set_overrides: tuple[str, ...],
) -> None:
    """Solve a configured run and write a NetCDF snapshot.

    CONFIG is the TOML configuration file (e.g.
    ``src/aragog/cfg/abe_solid.toml``).

    The run path mirrors the standalone Python recipe in
    ``docs/Tutorials/firstrun.md``:

        EntropySolver.from_file(CONFIG, eos_dir=...)
        solver.initialize()
        solver.set_initial_dSdr_cmb(...)        # if energy_balance
        solver.set_initial_entropy(...)
        solver.solve()
        solver.get_state().to_netcdf(...)
    """
    from aragog import aragog_file_logger
    from aragog.solver import EntropySolver

    if eos_dir is None:
        env_eos = os.environ.get('FWL_DATA')
        if env_eos:
            eos_dir = Path(env_eos) / 'spider' / 'lookup-fs'
        if eos_dir is None or not eos_dir.is_dir():
            raise click.UsageError(
                '--eos-dir not set and $FWL_DATA/spider/lookup-fs not '
                'usable; pass --eos-dir explicitly.'
            )

    log_dir_resolved = (log_dir or Path.cwd()).resolve()
    log_dir_resolved.mkdir(parents=True, exist_ok=True)
    aragog_file_logger(
        console_level=_resolve_log_level(log_level),
        file_level=logging.DEBUG,
        log_dir=str(log_dir_resolved),
    )

    logger.info('aragog run: config=%s eos_dir=%s', config, eos_dir)

    if set_overrides:
        if config.suffix.lower() != '.toml':
            raise click.UsageError(
                f'--set requires a .toml configuration; got {config.suffix!r}. '
                'Convert the .cfg file to TOML or drop --set.'
            )
        if sys.version_info < (3, 11):
            import tomli as tomllib
        else:
            import tomllib

        from aragog.config import Config
        from aragog.eos.entropy import EntropyEOS

        with config.open('rb') as fh:
            data = tomllib.load(fh)
        data = _apply_overrides(data, set_overrides)
        try:
            parameters = Config.from_dict(data)
        except TypeError as exc:
            raise click.UsageError(
                f'after applying --set overrides, the resolved config has '
                f'unknown / mismatched fields: {exc}.'
            ) from exc
        entropy_eos = EntropyEOS(Path(eos_dir))
        solver = EntropySolver(parameters, entropy_eos)
        logger.info(
            'aragog run: applied %d --set override(s) on top of %s',
            len(set_overrides),
            config.name,
        )
    else:
        solver = EntropySolver.from_file(filename=str(config), eos_dir=str(eos_dir))
    solver.initialize()

    core_bc = getattr(solver.parameters.boundary_conditions, 'core_bc', 'energy_balance')
    if core_bc == 'energy_balance':
        solver.set_initial_dSdr_cmb(initial_dsdr_cmb)
    elif initial_dsdr_cmb != 0.0:
        logger.warning(
            '--initial-dsdr-cmb=%g ignored: core_bc=%r does not use the '
            'CMB entropy gradient as a state variable. Pass core_bc='
            "'energy_balance' to make this option meaningful.",
            initial_dsdr_cmb,
            core_bc,
        )

    if initial_entropy is None:
        raise click.UsageError(
            '--initial-entropy is required: pass an isentropic value '
            'in J/kg/K (e.g. 2900.0 for an early-Earth-like state).'
        )
    solver.set_initial_entropy(initial_entropy)

    solver.solve()

    out_path = (out or Path.cwd() / f'{config.stem}.nc').resolve()
    state = solver.get_state()
    state.to_netcdf(
        out_path,
        description=f'Aragog run from {config.name}',
    )
    click.echo(f'wrote {out_path}')


# ---------------------------------------------------------------------------
# aragog list-configs
# ---------------------------------------------------------------------------


@cli.command(name='list-configs')
def list_configs() -> None:
    """List configurations bundled under ``src/aragog/cfg/``."""
    cfg_dir = _bundled_cfg_dir()
    suffixes = ('.toml', '.cfg')
    rows: list[tuple[str, str]] = []
    for entry in sorted(cfg_dir.iterdir(), key=lambda p: p.name):
        name = entry.name
        if not name.endswith(suffixes):
            continue
        summary = _first_comment_line(entry)
        rows.append((name, summary))
    if not rows:
        click.echo('(no bundled configs found)')
        return
    width = max(len(r[0]) for r in rows)
    for name, summary in rows:
        click.echo(f'  {name:<{width}}  {summary}')


def _first_comment_line(entry: Traversable) -> str:
    """Return the first '#'-prefixed line from a TOML/cfg file as a
    one-line summary, stripped of leading '#' and whitespace. Returns
    an empty string when the file has no leading comment."""
    try:
        text = entry.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return ''
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            return line.lstrip('#').strip()
        # Stop scanning on the first non-comment, non-blank line.
        break
    return ''


# ---------------------------------------------------------------------------
# aragog inspect
# ---------------------------------------------------------------------------


# Scalar diagnostics surfaced by `inspect`. Order matters for human
# output: status first (so a failed run is impossible to miss), then
# the dominant state variables, then heat-balance terms.
_INSPECT_SCALARS: tuple[tuple[str, str], ...] = (
    ('status', 'solver status (0 = success)'),
    ('time', 'simulation time [yr]'),
    ('dt_actual', 'integration interval [yr]'),
    ('T_magma', 'surface (magma) temperature [K]'),
    ('T_core', 'CMB temperature [K]'),
    ('Phi_global', 'mass-weighted melt fraction [-]'),
    ('Phi_global_vol', 'volume-weighted melt fraction [-]'),
    ('M_mantle', 'mantle mass [kg]'),
    ('F_heat_total', 'total internal heat flux [W/m^2]'),
    ('F_cmb', 'CMB heat flux [W/m^2]'),
    ('Q_radio_total', 'radiogenic power [W]'),
    ('Q_tidal_total', 'tidal power [W]'),
    ('E_th', 'thermal energy proxy [J]'),
    ('E_state', 'EOS-consistent enthalpy [J]'),
    ('E_state_cons', 'frozen-mass enthalpy [J]'),
    ('Cp_eff', 'mass-weighted Cp [J/kg/K]'),
    ('RF_depth', 'rheological-front depth [-]'),
)

_INSPECT_GLOBAL_ATTRS: tuple[str, ...] = (
    'description',
    'aragog_version',
    'created_utc',
    'Conventions',
)


@cli.command(name='inspect')
@click.argument(
    'snapshot',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    '--json',
    'as_json',
    is_flag=True,
    default=False,
    help='Emit a machine-readable JSON document instead of the human-readable summary.',
)
def inspect_cmd(snapshot: Path, as_json: bool) -> None:
    """Print key diagnostics from a SolverOutput NetCDF snapshot.

    Reads a NetCDF file produced by ``EntropySolver.get_state().to_netcdf()``
    (or the equivalent ``aragog run --out`` artefact) and prints the
    scalar status fields and the key heat-balance / state diagnostics.

    Use ``--json`` for a machine-readable document suitable for piping
    into ``jq``, capturing in CI assertions, or archiving alongside
    plots.
    """
    import netCDF4 as nc
    import numpy as np

    payload: dict[str, object] = {'path': str(snapshot)}

    try:
        ds = nc.Dataset(snapshot, mode='r')
    except OSError as exc:
        raise click.ClickException(
            f'could not open {snapshot} as a NetCDF file: {exc}'
        ) from exc

    try:
        # Mesh dimensions.
        dims = {name: int(d.size) for name, d in ds.dimensions.items()}
        payload['dimensions'] = dims

        # Global attributes.
        attrs = {}
        for name in _INSPECT_GLOBAL_ATTRS:
            if name in ds.ncattrs():
                attrs[name] = ds.getncattr(name)
        payload['attrs'] = attrs

        # Scalar variables. Skip silently when a field is absent so an
        # older snapshot from a pre-Cp_eff version still inspects
        # cleanly. Production NetCDF writes E_state and E_state_cons
        # with ``_FillValue=NaN``; netCDF4 returns those reads as a
        # masked array whose ``float()`` is ``numpy.ma.masked`` rather
        # than raising. Disable auto-masking so a NaN-filled scalar
        # round-trips as a Python float (NaN), then map NaN -> None.
        scalars: dict[str, object] = {}
        for name, _label in _INSPECT_SCALARS:
            if name not in ds.variables:
                continue
            var = ds.variables[name]
            var.set_auto_mask(False)
            raw = np.asarray(var[...]).reshape(-1)
            value = raw[0] if raw.size else None
            if value is None or (isinstance(value, float) and np.isnan(value)):
                scalars[name] = None
            else:
                try:
                    scalars[name] = float(value)
                except (TypeError, ValueError):
                    scalars[name] = None
        payload['scalars'] = scalars

        # Profile shape sanity: report min/max of S_final and T_basic
        # so an empty / NaN-only snapshot is obvious from the summary.
        # Same auto-mask hazard applies to profiles whose schema sets
        # ``_FillValue``; ``np.isfinite`` on a masked array reads the
        # underlying fill value (often a finite sentinel) and
        # mis-counts the finite-cell tally. Disable masking and
        # sentinel-fill the array with NaN before the isfinite check.
        profiles: dict[str, object] = {}
        for name in ('S_final', 'T_basic'):
            if name not in ds.variables:
                continue
            var = ds.variables[name]
            var.set_auto_mask(False)
            arr = np.asarray(var[:]).astype('f8')
            finite = arr[np.isfinite(arr)]
            if finite.size:
                profiles[name] = {
                    'min': float(finite.min()),
                    'max': float(finite.max()),
                    'n_finite': int(finite.size),
                    'n_total': int(arr.size),
                }
            else:
                profiles[name] = {
                    'min': None,
                    'max': None,
                    'n_finite': 0,
                    'n_total': int(arr.size),
                }
        payload['profiles'] = profiles
    finally:
        ds.close()

    if as_json:
        import json

        click.echo(json.dumps(payload, indent=2, default=str))
        return

    # Human-readable summary.
    lines: list[str] = []
    lines.append(f'snapshot: {snapshot}')
    if attrs.get('aragog_version'):
        lines.append(f'  aragog: {attrs["aragog_version"]}')
    if attrs.get('created_utc'):
        lines.append(f'  created: {attrs["created_utc"]}')
    if attrs.get('description'):
        lines.append(f'  description: {attrs["description"]}')
    lines.append('  dimensions: ' + ', '.join(f'{name}={size}' for name, size in dims.items()))
    lines.append('')
    label_width = max(len(name) for name, _ in _INSPECT_SCALARS) + 2
    for name, label in _INSPECT_SCALARS:
        if name not in scalars:
            continue
        v = scalars[name]
        if v is None:
            formatted = 'n/a'
        elif name == 'status':
            formatted = f'{int(v)}'
        elif abs(v) >= 1e4 or (0 < abs(v) < 1e-2):
            formatted = f'{v:.4e}'
        else:
            formatted = f'{v:.4f}'
        lines.append(f'  {name:<{label_width}} {formatted:>14}  {label}')
    if profiles:
        lines.append('')
        for name, info in profiles.items():
            if info['min'] is None:
                lines.append(f'  {name}: all NaN ({info["n_total"]} nodes)')
            else:
                lines.append(
                    f'  {name}: min={info["min"]:.4e}, max={info["max"]:.4e} '
                    f'({info["n_finite"]}/{info["n_total"]} finite)'
                )
    click.echo('\n'.join(lines))


# ---------------------------------------------------------------------------
# aragog show-config
# ---------------------------------------------------------------------------


def _serialize_params(params) -> dict:
    """Convert the resolved Parameters dataclass tree to a JSON-safe dict.

    Walks the dataclass with dataclasses.asdict, then post-processes
    numpy-array fields (notably energy.tidal_array and the optional
    mesh.eos_* arrays loaded by __post_init__) into Python lists so
    json.dumps does not choke. Skips fields with leading underscore.
    """
    import dataclasses

    import numpy as np

    def _convert(value):
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items() if not k.startswith('_')}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if isinstance(value, tuple):
            return [_convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    return _convert(dataclasses.asdict(params))


@cli.command(name='show-config')
@click.argument('config', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    '--indent',
    type=int,
    default=2,
    show_default=True,
    help='JSON indent. Pass 0 for one-line output (jq-friendly).',
)
def show_config(config: Path, indent: int) -> None:
    """Dump the resolved Parameters dataclass tree as JSON.

    Loads the configuration through Parameters.from_file (same path
    as ``aragog run`` and ``aragog validate``), runs
    __post_init__, and serialises the resulting tree to JSON on
    stdout. Useful for diff'ing PROTEUS-extracted configs against
    hand-crafted ones, or for machine-readable extraction
    (`aragog show-config cfg.toml | jq .energy.kappah_floor`).
    """
    import json

    from aragog.parser import Parameters

    try:
        params = Parameters.from_file(config)
    except Exception as exc:
        raise click.ClickException(f'configuration error in {config}: {exc}') from exc

    payload = _serialize_params(params)
    click.echo(json.dumps(payload, indent=indent if indent > 0 else None, default=str))


# ---------------------------------------------------------------------------
# aragog validate
# ---------------------------------------------------------------------------


@cli.command(name='validate')
@click.argument('config', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(config: Path) -> None:
    """Parse a configuration file and report errors without solving.

    Loads the TOML/INI through Parameters.from_file, runs
    Parameters.__post_init__ (which dispatches the BC normalise,
    optionally loads an external EOS file, and converts radionuclide
    concentrations from ppm), and reports the first ValueError or
    other parser exception. EOS-table loading and the solver are not
    touched. Use this before launching a long CHILI run to catch
    typos, missing files, IBC=99-style dispatch errors, and the
    [scalings] strict-rejection.
    """
    from aragog.parser import Parameters

    try:
        params = Parameters.from_file(config)
    except Exception as exc:
        raise click.ClickException(f'configuration error in {config}: {exc}') from exc

    n_radio = len(params.radionuclides)
    click.echo(
        f'OK {config}: '
        f'core_bc={params.boundary_conditions.core_bc}, '
        f'IBC={params.boundary_conditions.inner_boundary_condition}, '
        f'OBC={params.boundary_conditions.outer_boundary_condition}, '
        f'eos_method={params.mesh.eos_method}, '
        f'IC_method={params.initial_condition.initial_condition}, '
        f'radionuclides={n_radio}'
    )


# ---------------------------------------------------------------------------
# aragog new
# ---------------------------------------------------------------------------


_DEFAULT_TEMPLATE = 'abe_solid'


@cli.command(name='new')
@click.argument('name')
@click.option(
    '--from',
    'template',
    default=_DEFAULT_TEMPLATE,
    show_default=True,
    help=(
        'Bundled template to copy. Run `aragog list-configs` to see '
        'available templates. Match by stem (without the .toml/.cfg '
        'suffix); .toml is preferred when both exist.'
    ),
)
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Overwrite the destination if it already exists.',
)
def new(name: str, template: str, force: bool) -> None:
    """Scaffold a new TOML config in the cwd by copying a bundled template.

    NAME is the destination filename (with or without a `.toml`
    extension). The file is written to the current working directory.
    """
    cfg_dir = _bundled_cfg_dir()

    # Prefer .toml over .cfg when both exist for the same stem; .cfg
    # is the legacy INI flavour and is not recommended for new files.
    candidates = [f'{template}.toml', f'{template}.cfg']
    src: Traversable | None = None
    for candidate in candidates:
        entry = cfg_dir.joinpath(candidate)
        if entry.is_file():
            src = entry
            break
    if src is None:
        available = sorted(
            p.name for p in cfg_dir.iterdir() if p.name.endswith(('.toml', '.cfg'))
        )
        raise click.UsageError(
            f"unknown template '{template}'. Available templates: {', '.join(available)}."
        )

    dest_name = name if name.endswith('.toml') else f'{name}.toml'
    dest = Path.cwd() / dest_name
    if dest.exists() and not force:
        raise click.UsageError(
            f'{dest} already exists. Pass --force to overwrite, or pick a different name.'
        )

    dest.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
    click.echo(f'wrote {dest} (from template {template})')


# ---------------------------------------------------------------------------
# aragog vnv
# ---------------------------------------------------------------------------


@cli.command(name='vnv')
@click.argument('topic', required=False)
@click.option(
    '--list',
    'list_topics',
    is_flag=True,
    default=False,
    help='List available V&V topics and exit.',
)
def vnv(topic: str | None, list_topics: bool) -> None:
    """Run a verification-figure script by topic name.

    TOPIC is the suffix of ``tools/verification/figures/verify_<TOPIC>.py``.
    Pass ``--list`` (or omit TOPIC) to enumerate available scripts.
    """
    figures_dir = _vnv_figures_dir()
    available = sorted(p.stem.removeprefix('verify_') for p in figures_dir.glob('verify_*.py'))

    if list_topics or topic is None:
        if not available:
            # Non-zero exit so a CI script that calls `aragog vnv --list`
            # to enumerate topics can detect the missing-checkout case
            # rather than reading the stdout sentinel string.
            raise click.ClickException(
                'no V&V scripts found; this command requires a source '
                'checkout (tools/verification/figures/) on the import '
                'path of the installed aragog package.'
            )
        click.echo('Available V&V topics:')
        for name in available:
            click.echo(f'  {name}')
        return

    if topic not in available:
        raise click.UsageError(
            f"unknown V&V topic '{topic}'. Run 'aragog vnv --list' to see available topics."
        )

    script = figures_dir / f'verify_{topic}.py'
    spec = importlib.util.spec_from_file_location(f'aragog._vnv_{topic}', script)
    if spec is None or spec.loader is None:
        raise click.ClickException(f'could not load {script}')
    module = importlib.util.module_from_spec(spec)
    # Register on sys.modules so the script can resolve relative imports
    # against its own module name; clean up after exec_module so a
    # second `aragog vnv` invocation in the same process does not reuse
    # the cached module (which would skip the script's main() side
    # effects on the second call).
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        if not hasattr(module, 'main'):
            raise click.ClickException(f"V&V script '{script.name}' has no main() entry point.")
        try:
            module.main()
        except click.ClickException:
            raise
        except Exception as exc:
            # Convert any uncaught exception from the V&V script into a
            # ClickException so the CLI exits with a non-zero status and
            # a clean message instead of a raw traceback that callers
            # would have to scrape.
            raise click.ClickException(f"V&V script '{script.name}' failed: {exc}") from exc
    finally:
        sys.modules.pop(spec.name, None)


if __name__ == '__main__':
    cli()
