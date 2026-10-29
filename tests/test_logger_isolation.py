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

import aragog
from tests.conftest import _package_logger_state, _restore_package_logger

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


def test_restoring_the_logger_closes_the_handlers_a_cli_run_added(tmp_path):
    """The file handler of ``aragog_file_logger`` is closed and the import handler comes back."""
    saved = _package_logger_state()
    package_logger = logging.getLogger('fwl.aragog')
    import_handler = package_logger.handlers[0]

    aragog.aragog_file_logger(log_dir=str(tmp_path))
    file_handlers = [h for h in package_logger.handlers if isinstance(h, logging.FileHandler)]
    assert package_logger.level == logging.DEBUG
    assert len(file_handlers) == 1
    assert import_handler not in package_logger.handlers
    assert (tmp_path / 'aragog.log').exists()

    _restore_package_logger(saved)
    assert file_handlers[0].stream is None
    assert package_logger.handlers == [import_handler]
    assert package_logger.level == logging.NOTSET
    assert not package_logger.isEnabledFor(logging.DEBUG)


def test_restoring_keeps_a_handler_that_was_there_before(tmp_path):
    """A handler present when the state was taken stays open; level and flags come back."""
    package_logger = logging.getLogger('fwl.aragog')
    before = logging.FileHandler(tmp_path / 'before.log')
    package_logger.addHandler(before)
    package_logger.setLevel(logging.INFO)
    saved = _package_logger_state()

    package_logger.addHandler(logging.FileHandler(tmp_path / 'during.log'))
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False
    package_logger.disabled = True
    _restore_package_logger(saved)

    assert before in package_logger.handlers
    assert before.stream is not None
    assert len(package_logger.handlers) == 2
    assert package_logger.level == logging.INFO
    assert package_logger.propagate is True
    assert package_logger.disabled is False
