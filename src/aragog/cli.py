"""Aragog command-line entry point.

The ``aragog`` script (registered under ``[project.scripts]`` in
``pyproject.toml``) dispatches to three subcommands:

* ``aragog run`` solves a configured run end-to-end and writes a
  NetCDF snapshot of the final state.
* ``aragog list-configs`` enumerates the configurations bundled
  under ``src/aragog/cfg/``.
* ``aragog vnv`` runs a single verification-figure script under
  ``tools/verification/figures/`` by topic name.

PROTEUS-coupled runs do not use this CLI; the ``proteus`` driver
calls ``EntropySolver`` directly via ``AragogRunner``.
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
    help='Print aragog and key-dependency versions and exit.',
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
    help='Print aragog plus key-dependency versions and exit.',
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
def run(
    config: Path,
    eos_dir: Path | None,
    initial_entropy: float | None,
    initial_dsdr_cmb: float,
    out: Path | None,
    log_dir: Path | None,
    log_level: str,
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
