"""The aragog package logger starts every test at its import state.

``aragog run`` configures the ``fwl.aragog`` logger for the rest of the process (DEBUG level, a
console and a file handler). A test that calls it through ``CliRunner`` must not pass that state to
the tests that run after it in the same process.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(150)]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE = 'tests/test_logger_isolation.py::test_the_package_logger_starts_at_its_import_state'


def test_the_package_logger_starts_at_its_import_state():
    """The package logger has no level and only the NullHandler that ``aragog`` adds at import."""
    package_logger = logging.getLogger('fwl.aragog')
    assert package_logger.level == logging.NOTSET
    assert [type(h) for h in package_logger.handlers] == [logging.NullHandler]
    assert not package_logger.isEnabledFor(logging.DEBUG)


# One test per file that calls ``aragog run`` through CliRunner and leaves the logger configured.
_CLI_RUN_TESTS = [
    'tests/test_cli.py::test_run_set_overrides_rejected_for_cfg_files',
    'tests/test_cli_exit_code.py::test_exit_code_follows_solver_status[0-0]',
    'tests/test_cli_more_branches.py::test_run_dispatches_initial_dsdr_cmb_for_gradient_state_modes[core_module]',
    'tests/test_config_data_paths.py::test_cli_missing_data_file_is_a_usage_error[overrides1]',
]


def test_a_cli_run_leaves_no_logger_state_for_the_next_test():
    """The probe passes after CLI runs in the same process, so each test sees the import state."""
    cmd = [sys.executable, '-m', 'pytest', *_CLI_RUN_TESTS, _PROBE]
    cmd += [
        '-o',
        'addopts=',
        '-p',
        'no:cacheprovider',
        '-p',
        'no:xdist',
        '-p',
        'no:randomly',
        '-q',
    ]
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=120)
    tail = proc.stdout[-3000:]
    assert f'{len(_CLI_RUN_TESTS) + 1} passed' in tail, tail
    assert proc.returncode == 0, tail
