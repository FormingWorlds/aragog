"""``aragog run`` exits non-zero when the integration fails."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from aragog.cli import cli

from .test_phi_step_cap_armed_smoke import EOS_DIR, needs_eos

CFG = Path(__file__).resolve().parents[1] / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml'


def _run(tmp_path, *overrides):
    args = ['run', str(CFG), '--eos-dir', str(EOS_DIR), '--out', str(tmp_path / 'o.nc')]
    return CliRunner().invoke(cli, args + [a for o in overrides for a in ('--set', o)])


@pytest.mark.unit
@pytest.mark.parametrize('status, code', [(-1, 1), (0, 0), (1, 0)])
def test_exit_code_follows_solver_status(tmp_path, monkeypatch, status, code):
    written = []

    class _Solver:
        solution = SimpleNamespace(message='stub message')

        def __init__(self, parameters, entropy_eos):
            self.parameters = parameters

        def get_state(self):
            return SimpleNamespace(status=status, to_netcdf=lambda p, **k: written.append(p))

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr('aragog.solver.EntropySolver', _Solver)
    monkeypatch.setattr('aragog.eos.entropy.EntropyEOS', lambda *a, **k: object())
    result = CliRunner().invoke(
        cli,
        [
            'run',
            str(CFG),
            '--eos-dir',
            str(tmp_path),
            '--initial-entropy',
            '2900',
            '--out',
            str(tmp_path / 'o.nc'),
            '--set',
            'solver.atol=1e-10',
        ],
    )
    assert result.exit_code == code, result.output
    assert written == [tmp_path / 'o.nc']
    assert ('stub message; partial state written to' in result.output) == (code == 1)


@pytest.mark.smoke
@needs_eos
def test_cvode_failure_exits_non_zero(tmp_path):
    pytest.importorskip('scikits_odes_sundials')  # max_steps acts on the CVODE path only
    result = _run(tmp_path, 'solver.max_steps=5')
    assert result.exit_code == 1, result.output
    assert 'partial state written to' in result.output
    assert (tmp_path / 'o.nc').is_file()


@pytest.mark.smoke
@needs_eos
def test_successful_run_exits_zero(tmp_path):
    result = _run(tmp_path, 'solver.end_time=1.0')
    assert result.exit_code == 0, result.output
    assert 'integration failed' not in result.output
