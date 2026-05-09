"""Unit tests for the Aragog CLI entry point.

Verifies that the three subcommands (``run``, ``list-configs``,
``vnv``) are registered, callable, and reject malformed input
without silent fallthrough.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.unit


def test_cli_version_flag_prints_aragog_version():
    """`aragog --version` must print the package version and exit 0.

    Discriminator: the printed version must match
    ``aragog.__version__`` exactly. A regression that hard-coded a
    stale version in cli.py would surface here.
    """
    from aragog import __version__
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])

    assert result.exit_code == 0, result.output
    assert __version__ in result.output, (
        f'aragog --version output {result.output!r} does not match '
        f'aragog.__version__ = {__version__!r}.'
    )


def test_cli_versions_flag_lists_dependencies():
    """`aragog --versions` (plural) must print the aragog version
    plus key dependency versions (numpy, scipy, jax) so bug reports
    are reproducible.

    Edge case: in an environment with a missing optional dependency
    (e.g. JAX), the line must still appear with 'not installed' or
    'unknown' rather than crashing the CLI. We can't reliably
    uninstall jax in a test, but we can assert the multi-line
    output structure.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['--versions'])

    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0].startswith('aragog '), (
        f'first line must start with "aragog "; got {lines[0]!r}.'
    )
    # Each dependency line is indented with two spaces and contains a colon.
    dep_names = {ln.strip().split(':', 1)[0] for ln in lines[1:] if ':' in ln}
    assert {'numpy', 'scipy'} <= dep_names, (
        f'expected numpy and scipy in dependency block; got {dep_names}.'
    )


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
    """The documented subcommands must be registered.

    Discriminator: the test fails if either an extra debug command
    leaks in (e.g. someone leaves a stray `@cli.command()` during
    development) or one of the documented commands goes missing.
    Tier 1 documented set: run, list-configs, new, vnv.
    """
    from aragog.cli import cli

    expected = {'run', 'list-configs', 'new', 'vnv'}
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


def test_new_scaffolds_default_template(tmp_path, monkeypatch):
    """`aragog new my_run` writes a TOML file in the cwd by copying the
    default abe_solid template.

    Discriminator: the destination file must exist after invocation
    AND its contents must equal the template's contents byte-for-byte
    (catches a regression that wrote an empty file or a stale stub).
    """
    from aragog.cli import cli

    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ['new', 'my_run'])

    assert result.exit_code == 0, result.output
    dest = tmp_path / 'my_run.toml'
    assert dest.is_file(), f'expected {dest} to exist; output={result.output!r}'

    # Compare against bundled abe_solid.toml byte-for-byte.
    import importlib.resources

    template = (
        importlib.resources.files('aragog')
        .joinpath('cfg/abe_solid.toml')
        .read_text(encoding='utf-8')
    )
    assert dest.read_text(encoding='utf-8') == template, (
        'scaffolded file diverges from the bundled template; the copy is broken.'
    )


def test_new_appends_toml_suffix_when_missing(tmp_path, monkeypatch):
    """`aragog new foo` writes foo.toml; `aragog new foo.toml` writes
    foo.toml. Edge case: the suffix logic must be idempotent.
    """
    from aragog.cli import cli

    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(cli, ['new', 'no_suffix'])
    runner.invoke(cli, ['new', 'with_suffix.toml'])

    assert (tmp_path / 'no_suffix.toml').is_file()
    assert (tmp_path / 'with_suffix.toml').is_file()
    assert not (tmp_path / 'with_suffix.toml.toml').exists(), (
        'aragog new doubled the .toml suffix; the suffix logic is not idempotent.'
    )


def test_new_rejects_unknown_template(tmp_path, monkeypatch):
    """`aragog new foo --from nonexistent` must fail with a clear list
    of available templates, not silently produce an empty file.
    """
    from aragog.cli import cli

    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ['new', 'foo', '--from', 'nonexistent_template'])

    assert result.exit_code != 0
    assert 'unknown template' in (result.output or '')
    assert 'abe_solid.toml' in (result.output or ''), (
        'available templates list should include the canonical abe_solid.toml; '
        f'got output={result.output!r}.'
    )
    assert not (tmp_path / 'foo.toml').exists(), (
        'aragog new wrote a file even though the template was unknown.'
    )


def test_new_refuses_overwrite_without_force(tmp_path, monkeypatch):
    """When the destination exists, `aragog new` must refuse to
    overwrite unless `--force` is passed.

    Edge case: a user running `aragog new my_run` twice in a row
    must not silently lose their edits to the first file.
    """
    from aragog.cli import cli

    monkeypatch.chdir(tmp_path)
    existing = tmp_path / 'my_run.toml'
    existing.write_text('# user edits\n', encoding='utf-8')

    runner = CliRunner()
    result = runner.invoke(cli, ['new', 'my_run'])

    assert result.exit_code != 0
    assert '--force' in (result.output or ''), (
        'overwrite-refusal message should suggest the --force escape hatch.'
    )
    assert existing.read_text(encoding='utf-8') == '# user edits\n', (
        'aragog new clobbered an existing file without --force.'
    )

    # With --force the overwrite goes through.
    result = runner.invoke(cli, ['new', 'my_run', '--force'])
    assert result.exit_code == 0
    assert existing.read_text(encoding='utf-8') != '# user edits\n', (
        '--force did not actually overwrite the file.'
    )


def test_run_rejects_malformed_config(tmp_path):
    """`aragog run` rejects a malformed config before reaching the solve.

    Edge case: a near-empty TOML fails inside Parameters.from_file
    (missing required sections). The CLI must surface that as a
    non-zero exit, not catch and swallow the error.

    This test deliberately uses a stub config so the failure happens
    at parser load time. The companion test
    test_run_rejects_missing_initial_entropy_with_valid_config
    exercises the --initial-entropy guard with a parseable config.
    """
    from aragog.cli import cli

    cfg = tmp_path / 'stub.toml'
    cfg.write_text('# stub\n')

    fake_eos = tmp_path / 'no_such_eos_dir'
    fake_eos.mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, ['run', str(cfg), '--eos-dir', str(fake_eos)])

    assert result.exit_code != 0, (
        'aragog run silently accepted a malformed config; the user '
        'would never see the parser error.'
    )


def test_run_rejects_missing_initial_entropy_with_valid_config(tmp_path, monkeypatch):
    """`aragog run` raises UsageError when --initial-entropy is omitted
    on an otherwise-valid run.

    Mocks ``EntropySolver.from_file`` so the parser/EOS layer is
    bypassed and the guard at the ``initial_entropy is None`` check
    is the only failure surface. Without the mock the parser would
    reject a stub config first, masking the guard. This test
    discriminates against a future refactor that drops the
    UsageError check (e.g., silently picking a default S_0).
    """
    from aragog.cli import cli

    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# placeholder\n')
    eos = tmp_path / 'eos'
    eos.mkdir()

    class _StubSolver:
        parameters = type(
            'P',
            (),
            {'boundary_conditions': type('BC', (), {'core_bc': 'energy_balance'})()},
        )()

        def initialize(self):
            pass

        def set_initial_dSdr_cmb(self, _):
            pass

    import aragog.solver as _solver_mod

    monkeypatch.setattr(
        _solver_mod.EntropySolver,
        'from_file',
        classmethod(lambda cls, **kw: _StubSolver()),
        raising=True,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ['run', str(cfg), '--eos-dir', str(eos)])

    assert result.exit_code != 0
    assert '--initial-entropy is required' in (result.output or ''), (
        'CLI must surface the guard message verbatim so users know which '
        f'flag to pass; got output={result.output!r}.'
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
