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

logger = logging.getLogger('fwl.' + __name__)


@click.group()
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
            click.echo('(no V&V scripts found; this command requires a source checkout)')
            return
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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if hasattr(module, 'main'):
        module.main()
    else:
        raise click.ClickException(f"V&V script '{script.name}' has no main() entry point.")


if __name__ == '__main__':
    cli()
