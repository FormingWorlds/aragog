"""``aragog run`` exits non-zero when the integration fails."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from aragog.cli import cli
from aragog.solver.entropy_solver import SolverOutput

from .test_phi_step_cap_armed_smoke import EOS_DIR, needs_eos

CFG = Path(__file__).resolve().parents[1] / 'src' / 'aragog' / 'cfg' / 'abe_solid.toml'


# CI (ci_tests.yml, nightly.yml) runs these against the test-data-v1 release tables
# (spider_eos_test_data.tar.gz); the explicit initial entropy lies inside that table range.
S_INIT = '2900'


def _run(tmp_path, *overrides, eos_dir=EOS_DIR, extra=('--initial-entropy', S_INIT)):
    args = ['run', str(CFG), '--eos-dir', str(eos_dir), '--out', str(tmp_path / 'o.nc'), *extra]
    return CliRunner().invoke(cli, args + [a for o in overrides for a in ('--set', o)])


def _stub_solver(monkeypatch, status, *, solution):
    """Replace the solver with one that returns a real SolverOutput with ``status``."""
    written = []
    out = SolverOutput(**{f.name: 0 for f in fields(SolverOutput)} | {'status': status})
    monkeypatch.setattr(SolverOutput, 'to_netcdf', lambda self, p, **k: written.append(p))

    class _Solver:
        def __init__(self, parameters, entropy_eos):
            self.parameters, self.solution = parameters, solution

        def get_state(self):
            return out

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr('aragog.solver.EntropySolver', _Solver)
    monkeypatch.setattr('aragog.eos.entropy.EntropyEOS', lambda *a, **k: object())
    return written


@pytest.mark.unit
@pytest.mark.parametrize('status, code', [(-1, 1), (0, 0), (1, 0)])
def test_exit_code_follows_solver_status(tmp_path, monkeypatch, status, code):
    written = _stub_solver(
        monkeypatch, status, solution=SimpleNamespace(message='stub message')
    )
    result = _run(tmp_path, 'solver.atol=1e-10', eos_dir=tmp_path)
    assert result.exit_code == code, result.output
    assert written == [tmp_path / 'o.nc']
    failed = (
        'integration failed (status=-1): stub message; wrote the state at the last successful'
    )
    assert (failed in result.output) == (code == 1)


@pytest.mark.unit
def test_failure_without_a_solution_exits_one(tmp_path, monkeypatch):
    written = _stub_solver(monkeypatch, -1, solution=None)
    result = _run(tmp_path, 'solver.atol=1e-10', eos_dir=tmp_path)
    assert result.exit_code == 1, result.output
    assert 'integration failed (status=-1): ; wrote the state' in result.output
    assert written == [tmp_path / 'o.nc']


@pytest.mark.unit
@pytest.mark.parametrize(
    'status, failed', [(-2, True), (-1, True), (0, False), (1, False), (2, True)]
)
def test_solver_output_failed(status, failed):
    out = SolverOutput(**{f.name: 0 for f in fields(SolverOutput)} | {'status': status})
    assert out.failed is failed


@pytest.mark.smoke
@needs_eos
def test_cvode_failure_exits_non_zero(tmp_path):
    pytest.importorskip('scikits_odes_sundials')  # max_steps acts on the CVODE path only
    import netCDF4

    result = _run(tmp_path, 'solver.max_steps=5')
    assert result.exit_code == 1, result.output
    assert 'integration failed (status=-1)' in result.output
    assert 'last successful output time' in result.output
    with netCDF4.Dataset(tmp_path / 'o.nc') as d:
        assert int(d['status'][:]) == -1
        assert float(d['time'][:]) == 0.0  # CVODE fails before the first output time
        floats = [np.asarray(d[k][:]) for k in d.variables if d[k].dtype.kind == 'f']
        assert floats and all(np.all(np.isfinite(v)) for v in floats)


@pytest.mark.smoke
@needs_eos
def test_successful_run_exits_zero(tmp_path):
    result = _run(tmp_path, 'solver.end_time=1.0')
    assert result.exit_code == 0, result.output
    assert 'integration failed' not in result.output


@pytest.mark.smoke
@needs_eos
def test_step_cap_stop_exits_zero_with_early_snapshot(tmp_path):
    import netCDF4

    result = _run(tmp_path, 'solver.end_time=1000.0', 'energy.phi_step_cap=0.01', extra=())
    assert result.exit_code == 0, result.output
    with netCDF4.Dataset(tmp_path / 'o.nc') as d:
        assert int(d['status'][:]) == 0
        assert 0.0 < float(d['time'][:]) < 1000.0


@pytest.mark.smoke
@needs_eos
def test_scipy_failure_exits_non_zero(tmp_path, monkeypatch):
    from scipy.optimize import OptimizeResult

    def _failed_solve_ivp(fun, t_span, y0, **kw):
        y = np.asarray(y0, dtype=float).reshape(-1, 1)
        return OptimizeResult(t=np.array([t_span[0]]), y=y, status=-1, message='stub', nfev=1)

    monkeypatch.setattr('aragog.solver.entropy_solver.solve_ivp', _failed_solve_ivp)
    result = _run(tmp_path, 'energy.solver_method=bdf')
    assert result.exit_code == 1, result.output
    assert 'integration failed (status=-1): stub; wrote the state' in result.output
