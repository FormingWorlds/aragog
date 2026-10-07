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


# The nightly CI run (nightly.yml) runs the smoke tests against the test-data-v1 release tables
# (spider_eos_test_data.tar.gz); the explicit initial entropy lies inside that table range.
S_INIT = '2900'


def _run(tmp_path, *overrides, eos_dir=EOS_DIR, extra=('--initial-entropy', S_INIT)):
    args = ['run', str(CFG), '--eos-dir', str(eos_dir), '--out', str(tmp_path / 'o.nc'), *extra]
    return CliRunner().invoke(cli, args + [a for o in overrides for a in ('--set', o)])


def _output(status):
    return SolverOutput(**{f.name: 0 for f in fields(SolverOutput)} | {'status': status})


def _stub_solver(monkeypatch, status, *, solution):
    """Replace the solver with one that returns a real SolverOutput with ``status``."""
    written = []
    out = _output(status)
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
@pytest.mark.parametrize('status, code', [(-1, 1), (0, 0), (1, 0), (2, 1)])
def test_exit_code_follows_solver_status(tmp_path, monkeypatch, status, code):
    written = _stub_solver(
        monkeypatch, status, solution=SimpleNamespace(message='stub message')
    )
    result = _run(tmp_path, 'solver.atol=1e-10', eos_dir=tmp_path)
    assert result.exit_code == code, result.output
    assert written == [tmp_path / 'o.nc']
    failed = f'integration failed (status={status}): stub message; wrote the state at the last successful'
    assert (failed in result.output) == (code == 1)


@pytest.mark.unit
@pytest.mark.parametrize(
    'status, failed', [(-2, True), (-1, True), (0, False), (1, False), (2, True)]
)
def test_solver_output_failed(status, failed):
    assert _output(status).failed is failed


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
        nan_ok = {'core_layer_base', 'core_T_top'}  # NaN without a stratified core
        floats = [
            np.asarray(d[k][:])
            for k in d.variables
            if d[k].dtype.kind == 'f' and k not in nan_ok
        ]
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
    pytest.importorskip('scikits_odes_sundials')  # the CVODE root stop reports status 0
    import netCDF4

    result = _run(tmp_path, 'solver.end_time=1000.0', 'energy.phi_step_cap=0.01', extra=())
    assert result.exit_code == 0, result.output
    with netCDF4.Dataset(tmp_path / 'o.nc') as d:
        assert int(d['status'][:]) == 0
        assert 0.0 < float(d['time'][:]) < 1000.0


@pytest.mark.smoke
@needs_eos
def test_scipy_failure_exits_non_zero(tmp_path, monkeypatch):
    import netCDF4
    from scipy.optimize import OptimizeResult

    def _failed_solve_ivp(fun, t_span, y0, **kw):
        y = np.asarray(y0, dtype=float).reshape(-1, 1)
        return OptimizeResult(t=np.array([t_span[0]]), y=y, status=-1, message='stub', nfev=1)

    monkeypatch.setattr('aragog.solver.entropy_solver.solve_ivp', _failed_solve_ivp)
    result = _run(tmp_path, 'energy.solver_method=bdf')
    assert result.exit_code == 1, result.output
    assert 'integration failed (status=-1): stub; wrote the state' in result.output
    with netCDF4.Dataset(tmp_path / 'o.nc') as d:
        assert int(d['status'][:]) == -1


def _stub_solve_ivp(status):
    """A scipy ``solve_ivp`` stand-in that returns ``status`` after half the span."""
    from scipy.optimize import OptimizeResult

    def solve_ivp(fun, t_span, y0, **kw):
        y = np.asarray(y0, dtype=float).reshape(-1, 1)
        t = np.array([t_span[0], 0.5 * (t_span[0] + t_span[1])])
        return OptimizeResult(t=t, y=np.hstack([y, y]), status=status, message='stub', nfev=1)

    return solve_ivp


@pytest.mark.smoke
@needs_eos
def test_scipy_step_cap_stop_exits_zero(tmp_path, monkeypatch):
    import netCDF4

    monkeypatch.setattr('aragog.solver.entropy_solver.solve_ivp', _stub_solve_ivp(1))
    result = _run(tmp_path, 'energy.solver_method=bdf', 'solver.end_time=10.0')
    assert result.exit_code == 0, result.output
    with netCDF4.Dataset(tmp_path / 'o.nc') as d:
        assert int(d['status'][:]) == 1
        assert d['status'].long_name == (
            'Solver status (0 success, 1 stop at a step-cap event, other values failure)'
        )


@pytest.mark.smoke
@needs_eos
@pytest.mark.parametrize('status, stop', [(-1, True), (1, False)])
def test_solve_sets_stop_early_from_the_status(monkeypatch, status, stop):
    from aragog.solver import EntropySolver

    monkeypatch.setattr('aragog.solver.entropy_solver.solve_ivp', _stub_solve_ivp(status))
    solver = EntropySolver.from_file(str(CFG), eos_dir=str(EOS_DIR))
    solver.parameters.energy.solver_method = 'bdf'
    solver.initialize()
    solver.set_initial_entropy(float(S_INIT))
    solver.stop_early = not stop
    solver.solve()
    assert solver.stop_early is stop
