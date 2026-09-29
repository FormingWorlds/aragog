"""``aragog run`` exits non-zero when the CVODE integration fails."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from aragog.cli import cli

from .test_phi_step_cap_armed_smoke import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, needs_eos]

CFG = Path(__file__).resolve().parents[1] / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml'


def _run(tmp_path, override):
    args = ['run', str(CFG), '--eos-dir', str(EOS_DIR), '--out', str(tmp_path / 'o.nc')]
    return CliRunner().invoke(cli, args + ['--set', override])


def test_failed_integration_exits_non_zero(tmp_path):
    result = _run(tmp_path, 'solver.max_steps=5')
    assert result.exit_code == 1, result.output
    assert 'integration failed (status=-1)' in result.output
    assert (tmp_path / 'o.nc').is_file()


def test_successful_run_exits_zero(tmp_path):
    result = _run(tmp_path, 'solver.end_time=1.0')
    assert result.exit_code == 0, result.output
    assert 'integration failed' not in result.output
