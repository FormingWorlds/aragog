"""CLI runs in one pytest process leave the package logger at its import state for the next test."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE = 'tests/test_logger_isolation.py::test_the_package_logger_starts_at_its_import_state'
# Leaves the logger configured in the probe's own module, so a fixture of wider scope fails.
_SAME_MODULE = (
    'tests/test_logger_isolation.py::test_aragog_file_logger_configures_the_package_logger'
)
# One test per file that calls ``aragog run`` through CliRunner and configures the logger.
_CLI_RUN_TESTS = [
    'tests/test_cli.py::test_run_set_overrides_rejected_for_cfg_files',
    'tests/test_cli_exit_code.py::test_exit_code_follows_solver_status[0-0]',
    'tests/test_cli_more_branches.py::test_run_dispatches_initial_dsdr_cmb_for_gradient_state_modes[core_module]',
    'tests/test_config_data_paths.py::test_cli_missing_data_file_is_a_usage_error[overrides1]',
]


def test_the_probe_passes_after_cli_runs_in_the_same_process(tmp_path):
    """Each CLI test and then the probe pass; a renamed CLI test shows as 'not found' in the output."""
    nodes = [str(_REPO_ROOT / n) for n in [*_CLI_RUN_TESTS, _SAME_MODULE, _PROBE]]
    cmd = [sys.executable, '-m', 'pytest', *nodes, '-c', str(_REPO_ROOT / 'pyproject.toml')]
    cmd += ['-o', 'addopts=', '-p', 'no:cacheprovider', '-p', 'no:xdist', '-rA', '-q']
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=50)
    out = proc.stdout + proc.stderr
    passed = [line.split(' ', 1)[1] for line in out.splitlines() if line.startswith('PASSED ')]
    missing = [
        n
        for n in [*_CLI_RUN_TESTS, _SAME_MODULE, _PROBE]
        if not any(p.endswith(n) for p in passed)
    ]
    assert not missing, out[-3000:]
    assert len(passed) == len(_CLI_RUN_TESTS) + 2, out[-3000:]
    assert proc.returncode == 0, out[-3000:]
