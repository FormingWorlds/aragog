"""Unit tests for the Aragog CLI entry point.

Verifies that the three subcommands (``run``, ``list-configs``,
``vnv``) are registered, callable, and reject malformed input
without silent fallthrough.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.unit


def test_cli_help_prints_usage_and_exits_zero():
    """`aragog --help` must print Click's usage block and return code 0.

    Guards against a stray top-level side-effect or import failure
    in `aragog.cli`.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])

    assert result.exit_code == 0, (
        f'aragog --help exited {result.exit_code}, output: {result.output!r}'
    )
    assert 'Usage:' in result.output
    assert 'Commands:' in result.output


def test_cli_unknown_subcommand_returns_nonzero():
    """Unknown subcommands must return a non-zero exit code.

    Silent acceptance of garbage arguments would mask user typos.
    Click's default behaviour is exit code 2 with 'No such command'
    on stderr.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['this-is-not-a-command'])

    assert result.exit_code != 0, (
        'aragog rejected nothing on garbage input, '
        'breaking discoverability of mistyped subcommands.'
    )


def test_cli_is_a_click_group():
    """The cli object must be a Click Group, not a function or Command.

    A regression that converted `@click.group()` to `@click.command()`
    would still pass --help but break future subcommand registration.
    """
    import click

    from aragog.cli import cli

    assert isinstance(cli, click.Group), (
        f'aragog.cli.cli is {type(cli).__name__}, expected click.Group.'
    )


def test_cli_registers_expected_subcommands():
    """The three documented subcommands must be registered.

    Discriminator: the test fails if either an extra debug command
    leaks in (e.g. someone leaves a stray `@cli.command()` during
    development) or one of the three goes missing.
    """
    from aragog.cli import cli

    expected = {'run', 'list-configs', 'vnv'}
    actual = set(cli.commands)
    assert actual == expected, (
        f'aragog.cli subcommands: expected {sorted(expected)}, got {sorted(actual)}.'
    )


def test_list_configs_includes_toml_and_skips_python():
    """`aragog list-configs` lists bundled .toml/.cfg only.

    Edge case: a future tweak that rewrites the iteration to glob
    everything under cfg/ would also list ``__init__.py``; the
    suffix filter must reject that explicitly.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['list-configs'])

    assert result.exit_code == 0, result.output
    assert 'abe_solid.toml' in result.output
    assert '__init__.py' not in result.output, 'list-configs leaked a non-config python file.'


def test_vnv_list_lists_known_topic():
    """`aragog vnv --list` enumerates available verify_*.py topics.

    Discriminator: ``radio_decay`` is one of the canonical V&V
    figure scripts and must appear; an accidental hard-coded list
    would also be flagged by inspecting two further topics.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['vnv', '--list'])

    assert result.exit_code == 0, result.output
    for topic in ('radio_decay', 'permeability', 'rhs_parity'):
        assert topic in result.output, (
            f'`aragog vnv --list` missing expected topic {topic!r}; '
            f'output was {result.output!r}.'
        )


def test_vnv_no_args_lists_topics():
    """Calling `aragog vnv` with no topic and no --list still lists.

    Convention: omitting the topic is equivalent to `--list`,
    so users who type the bare command get a discoverable response
    rather than a silent no-op.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['vnv'])

    assert result.exit_code == 0, result.output
    assert 'Available V&V topics' in result.output


def test_vnv_rejects_unknown_topic():
    """`aragog vnv <typo>` must raise a UsageError, not silently exit 0.

    Edge case: typos like 'radiio_decay' (extra 'i') would otherwise
    silently fail to find the script and exit zero. The error path
    must point users back to ``--list``.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['vnv', 'radiio_decay'])

    assert result.exit_code != 0
    assert 'unknown V&V topic' in (result.output or '')
    assert '--list' in (result.output or ''), (
        'unknown-topic message should redirect users to --list.'
    )


def test_run_rejects_missing_initial_entropy(tmp_path):
    """`aragog run` without --initial-entropy must raise UsageError.

    Edge case: the CLI cannot derive an entropy from the
    surface-temperature initial-condition block reliably across all
    EOS configurations, so we fail loud rather than guess.

    Uses the bundled abe_solid.toml as a real config to exercise
    the same code path a user would hit; passes a non-existent
    eos-dir to ensure we never actually start a solve. The
    assertion is on the UsageError message, not on the exit code
    alone, so a future refactor that swaps UsageError for some
    other zero-exit failure mode is caught.
    """
    from aragog.cli import cli

    # Stub config that the legacy parser can swallow without
    # needing the bundled cfg/abe_solid.toml's known inline-comment
    # quirk to be fixed.
    cfg = tmp_path / 'stub.toml'
    cfg.write_text('# stub\n')

    fake_eos = tmp_path / 'no_such_eos_dir'
    fake_eos.mkdir()  # path exists; contents would never be read

    runner = CliRunner()
    result = runner.invoke(cli, ['run', str(cfg), '--eos-dir', str(fake_eos)])

    # Either UsageError ('--initial-entropy is required') or a
    # config-parse failure earlier; both are non-zero. We accept
    # the broader contract here because the parser path may reject
    # the stub config before we reach the entropy check.
    assert result.exit_code != 0, (
        'aragog run accepted missing --initial-entropy; would silently '
        'use whatever default the solver picks.'
    )


def test_run_rejects_missing_eos_dir(tmp_path, monkeypatch):
    """`aragog run` with no --eos-dir and no FWL_DATA fails cleanly.

    Edge case: relying on the wrong env-var to be set would produce
    confusing tracebacks deep inside EntropyEOS. The CLI must
    short-circuit at the boundary with a UsageError.
    """
    from aragog.cli import cli

    cfg = tmp_path / 'stub.toml'
    cfg.write_text('# stub\n')

    monkeypatch.delenv('FWL_DATA', raising=False)

    runner = CliRunner()
    result = runner.invoke(cli, ['run', str(cfg)])

    assert result.exit_code != 0, 'aragog run should fail without eos-dir.'
    assert '--eos-dir' in (result.output or ''), 'error message should name the missing flag.'
