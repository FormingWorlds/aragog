"""Branch coverage for ``aragog vnv`` error / edge paths.

The existing ``test_cli.py`` covers the happy paths (``--list``,
unknown topic, no args). The remaining gaps (lines 903, 918-922,
928-942 in ``cli.py``) are:

* ``--list`` with no scripts on disk (race condition / wheel-only
  install).
* Module-spec-load failure for a syntactically broken verify_*.py.
* Missing ``main()`` symbol.
* Uncaught exception inside ``main()`` (must surface as
  ClickException, not a raw traceback).

These are all controllable from a tmp_path without poking the
production ``tools/verification/figures/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.unit


def _patch_vnv_dir(monkeypatch, tmp_path: Path) -> Path:
    """Swap the V&V figures directory for ``tmp_path``."""
    import aragog.cli as cli_mod

    monkeypatch.setattr(cli_mod, '_vnv_figures_dir', lambda: tmp_path)
    return tmp_path


def test_vnv_list_with_empty_directory_raises_click_exception(monkeypatch, tmp_path):
    """When the figures directory exists but contains no
    ``verify_*.py``, ``--list`` must surface a ClickException so a
    CI script can detect the missing-checkout case via exit code,
    not by reading the stdout sentinel string.
    """
    from aragog.cli import cli

    _patch_vnv_dir(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ['vnv', '--list'])
    assert result.exit_code != 0, 'empty figures dir must produce non-zero exit'
    assert 'no V&V scripts found' in (result.output or '')
    assert 'tools/verification/figures' in (result.output or ''), (
        'error message must point users to the expected source location'
    )


def test_vnv_runs_main_from_user_supplied_module(monkeypatch, tmp_path):
    """A well-formed verify_<topic>.py with a ``main()`` function must
    be loaded, executed, and produce a zero-exit ``aragog vnv`` run.

    Discriminator: ``main()`` writes a sentinel file; the assertion
    confirms the side effect actually happened (a regression that
    skipped the ``module.main()`` call would leave the sentinel
    absent).
    """
    from aragog.cli import cli

    figures_dir = _patch_vnv_dir(monkeypatch, tmp_path)
    sentinel = tmp_path / 'sentinel.txt'
    (figures_dir / 'verify_demo.py').write_text(
        f"""
from pathlib import Path

def main():
    Path({str(sentinel)!r}).write_text('ran')
""",
        encoding='utf-8',
    )

    runner = CliRunner()
    result = runner.invoke(cli, ['vnv', 'demo'])
    assert result.exit_code == 0, f'vnv demo failed unexpectedly: {result.output!r}'
    assert sentinel.exists(), (
        'main() did not produce its side effect; module.main() not invoked'
    )
    assert sentinel.read_text() == 'ran'


def test_vnv_rejects_module_without_main(monkeypatch, tmp_path):
    """A verify_<topic>.py that imports cleanly but defines no
    ``main()`` must raise ClickException with a clear message.

    Edge case: a test author might write the script as a plain
    procedural file with side effects at import time. The CLI must
    still flag the missing entry point.
    """
    from aragog.cli import cli

    figures_dir = _patch_vnv_dir(monkeypatch, tmp_path)
    (figures_dir / 'verify_no_main.py').write_text('# nothing here\n', encoding='utf-8')

    runner = CliRunner()
    result = runner.invoke(cli, ['vnv', 'no_main'])
    assert result.exit_code != 0
    assert 'no main() entry point' in (result.output or '')


def test_vnv_wraps_main_exception_in_click_exception(monkeypatch, tmp_path):
    """Uncaught exceptions inside ``main()`` must be re-raised as
    ClickException so the CLI exits non-zero with a clean message.

    Discriminator: the wrapper deliberately preserves ClickException
    re-raises (so a script can raise a UsageError that surfaces with
    its own message), but converts every other Exception into a
    ClickException with the script name. We test both: (a) a
    ClickException-raising script must propagate verbatim, (b) a
    plain ValueError-raising script must be wrapped.
    """
    from aragog.cli import cli

    figures_dir = _patch_vnv_dir(monkeypatch, tmp_path)
    (figures_dir / 'verify_boom.py').write_text(
        "def main():\n    raise ValueError('intentional explosion')\n",
        encoding='utf-8',
    )
    (figures_dir / 'verify_user_error.py').write_text(
        "import click\ndef main():\n    raise click.UsageError('please pass --foo')\n",
        encoding='utf-8',
    )

    runner = CliRunner()

    # Plain exception is wrapped.
    result = runner.invoke(cli, ['vnv', 'boom'])
    assert result.exit_code != 0
    out = (result.output or '').lower()
    assert 'verify_boom.py' in out and 'failed' in out, (
        f'plain exception not wrapped with script name: {result.output!r}'
    )

    # ClickException is preserved verbatim.
    result_user = runner.invoke(cli, ['vnv', 'user_error'])
    assert result_user.exit_code != 0
    assert 'please pass --foo' in (result_user.output or '')


def test_vnv_module_cleanup_re_runs_main_on_second_invocation(monkeypatch, tmp_path):
    """After ``aragog vnv`` runs, ``sys.modules`` must NOT cache the
    script under its synthetic ``aragog._vnv_<topic>`` name. A second
    invocation in the same Python process must re-execute ``main()``.

    Discriminator: a regression that left the module on ``sys.modules``
    would skip the script's main() side effects on the second call
    because Python sees the entry as already-imported.
    """
    from aragog.cli import cli

    figures_dir = _patch_vnv_dir(monkeypatch, tmp_path)
    counter_path = tmp_path / 'counter.txt'
    counter_path.write_text('0', encoding='utf-8')
    (figures_dir / 'verify_counter.py').write_text(
        f"""
from pathlib import Path
def main():
    p = Path({str(counter_path)!r})
    n = int(p.read_text() or 0)
    p.write_text(str(n + 1))
""",
        encoding='utf-8',
    )

    runner = CliRunner()
    runner.invoke(cli, ['vnv', 'counter'])
    runner.invoke(cli, ['vnv', 'counter'])
    n_runs = int(counter_path.read_text())
    assert n_runs == 2, (
        f'main() ran {n_runs} times across two invocations; expected 2. '
        'sys.modules cache likely leaked between calls.'
    )


def test_coerce_value_falls_through_to_string_when_json_malformed():
    """When the input starts with ``[``/``{``/``"`` but is invalid
    JSON, the coercion must fall through to the bare-string branch
    rather than raising.

    Edge case: passing ``--set energy.kappah_floor='[1, 2'`` (a
    truncated list) should keep the raw string, surfacing the type
    error downstream where the user can fix it; raising here would
    obscure the source of the malformed value.
    """
    from aragog.cli import _coerce_value

    assert _coerce_value('[broken') == '[broken'
    assert _coerce_value('{not, json}') == '{not, json}'
    # Quoted-prefix that's not actually JSON: same fallback.
    assert _coerce_value('"unterminated') == '"unterminated'
