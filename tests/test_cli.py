"""Unit tests for the Aragog CLI entry point.

The CLI is intentionally minimal: a single Click group with no
subcommands. These tests verify it is importable, callable, and
fails gracefully on bad invocations rather than silently doing
nothing.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.unit


def test_cli_help_prints_usage_and_exits_zero():
    """`aragog --help` must print Click's usage block and return code 0.

    This guards against the CLI being broken by a stray top-level
    side-effect or import failure in `aragog.cli`.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])

    assert result.exit_code == 0, (
        f'aragog --help exited {result.exit_code}, output: {result.output!r}'
    )
    assert 'Usage:' in result.output
    assert 'Options' in result.output


def test_cli_unknown_subcommand_returns_nonzero():
    """An unknown subcommand must return a non-zero exit code.

    Edge case: silent acceptance of garbage arguments would mask
    user typos. Click's default behaviour is exit code 2 with
    'No such command' on stderr.
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

    Discriminator: a regression that converted `@click.group()` to
    `@click.command()` would still be importable and pass the help
    test, but would silently break future subcommand registration.
    """
    import click

    from aragog.cli import cli

    assert isinstance(cli, click.Group), (
        f'aragog.cli.cli is {type(cli).__name__}, expected click.Group; '
        f'subcommand registration via @cli.command() would be impossible.'
    )


def test_cli_no_subcommands_registered_yet():
    """The CLI ships without subcommands by design (usage.md).

    If subcommands are added in the future this test must be
    updated; until then it documents the intentional minimality
    and discriminates against accidental registration of a
    debug subcommand that leaks into a release.
    """
    from aragog.cli import cli

    assert cli.commands == {}, (
        f'aragog.cli has unexpected subcommands: {sorted(cli.commands)}; '
        f'verify whether they are intentional and documented.'
    )
