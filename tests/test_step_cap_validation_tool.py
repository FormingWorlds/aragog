"""``tools/verification/run_step_cap_validation.py``: segment counts, case setup, checks, refusals."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from .conftest import EOS_DIR, needs_eos

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location(
    'run_step_cap_validation',
    Path(__file__).resolve().parents[1]
    / 'tools'
    / 'verification'
    / 'run_step_cap_validation.py',
)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

# Two cells, solidus 100 and liquidus 200 J/(kg K), stiff-zone half-width 10 J/(kg K).
S_SOL, S_LIQ, MASS, DELTA = np.full(2, 100.0), np.full(2, 200.0), np.ones(2), 10.0


def test_subsolidus_when_no_cell_is_above_the_solidus():
    """All cells at or below the solidus count as sub-solidus; no cell is a slow landing."""
    assert tool.regime(np.array([90.0, 95.0]), -np.ones(2), S_LIQ, S_SOL, MASS, DELTA) == (
        False,
        True,
    )


def test_slow_landing_needs_a_slow_binding_cell_near_a_boundary():
    """A cell 3 J/(kg K) above the solidus binds; it is a slow landing only below 0.05 delta/yr."""
    S = np.array([103.0, 150.0])
    assert tool.regime(S, np.full(2, -0.1), S_LIQ, S_SOL, MASS, DELTA) == (True, False)
    assert tool.regime(S, np.full(2, -1.0), S_LIQ, S_SOL, MASS, DELTA) == (False, False)


def test_regime_takes_the_binding_cell_of_the_controller():
    """The cell that sets the controller step decides the count, not the raw distance."""
    S, dSdt, mass = np.array([100.1, 191.0]), np.array([-0.4, -1.0]), np.array([1.0, 0.0])
    t_c = tool.es._rate_phase_boundary_times(S, dSdt, S_LIQ, S_SOL, mass, DELTA)
    np.testing.assert_allclose(t_c, [12.5, 9.0])  # 0.5 delta / 0.4 against 9 / 1
    assert tool.regime(S, dSdt, S_LIQ, S_SOL, mass, DELTA) == (False, False)


def test_regime_counter_counts_calls_and_restores_the_cap():
    """The counter passes the cap value through, counts each call, and restores the function."""
    original = tool.es._rate_phase_boundary_max_step
    args = (np.array([103.0, 150.0]), np.full(2, -0.1), S_LIQ, S_SOL, MASS, DELTA)
    with tool.regime_counter() as counts:
        assert tool.es._rate_phase_boundary_max_step(*args) == original(*args)
        tool.es._rate_phase_boundary_max_step(np.array([90.0, 95.0]), *args[1:])
    assert counts == {'segments': 2, 'slow_landing': 1, 'subsolidus': 1}
    assert tool.es._rate_phase_boundary_max_step is original
    with pytest.raises(RuntimeError), tool.regime_counter():
        raise RuntimeError
    assert tool.es._rate_phase_boundary_max_step is original


def _row(mode, m, T, slow=0, S=(10.0, 20.0)):
    return dict(
        mode=mode,
        tol=1e-8,
        m=m,
        steps=1,
        wall_s=0.0,
        T_end=T,
        S_min=S[0],
        S_max=S[1],
        segments=1,
        slow_landing=slow,
        subsolidus=0,
        inf_caps=m + 2,
    )


class _Table:
    _tables = {'temperature_solid': {'S': np.array([0.0, 100.0])}}

    def __init__(self, _eos_dir):
        pass


def test_run_v3_metrics_come_from_the_m0_start_and_the_reference(monkeypatch):
    """The error is m = 0 against 'fixed'; the envelope is the largest m != 0 offset from m = 0."""
    T = {-2: [1.0, 2.0], -1: [1.0, 2.5], 0: [1.0, 2.0], 1: [1.0, 2.0], 2: [0.75, 2.0]}
    rows = {('fixed', 0): _row('fixed', 0, [1.0, 3.0])}
    rows.update({('rate', m): _row('rate', m, T[m], slow=int(m == 0)) for m in T})
    monkeypatch.setattr(tool, 'v3_run', lambda task: rows[task[2], task[4]])
    monkeypatch.setattr(tool, 'EntropyEOS', _Table)
    metrics = tool.run_v3('eos', 'v3_r3', workers=1)['metrics']
    assert metrics['max_dT_rate_vs_ref_K'] == 1.0
    assert metrics['round_off_envelope_K'] == 0.5
    assert metrics['slow_landing_segments'] == 1
    assert (metrics['steps'], metrics['inf_cap_segments']) == (1, 2)
    assert metrics['inside_table'] is True
    rows['rate', 2]['S_max'] = 101.0
    assert tool.run_v3('eos', 'v3_r3', workers=1)['metrics']['inside_table'] is False


def test_main_writes_the_case_result(monkeypatch, tmp_path):
    """main() passes the width to the v3 run and writes the result with the case name."""
    seen = {}

    def fake_run_v3(eos_dir, name, workers, width):
        seen.update(eos_dir=eos_dir, name=name, workers=workers, width=width)
        ok = dict(max_dT_rate_vs_ref_K=0.0, inside_table=True, steps=1, inf_cap_segments=1)
        return dict(rows=[], metrics=ok)

    monkeypatch.setattr(tool, 'run_v3', fake_run_v3)
    monkeypatch.setenv('ARAGOG_TEST_EOS_DIR', 'E')  # main() sets it; monkeypatch restores it
    out = tmp_path / 'r.json'
    args = ['--case', 'v3_r4', '--eos-dir', 'E', '--workers', '3', '--width', '0', '--output']
    assert tool.main([*args, str(out)]) == 0
    assert seen == {'eos_dir': 'E', 'name': 'v3_r4', 'workers': 3, 'width': 0.0}
    assert json.loads(out.read_text())['case'] == 'v3_r4'


def test_check_names_each_failed_tolerance():
    """check() fails a v3 error above the bound, a v4 failure, and an entropy outside the table."""
    m = dict(
        max_dT_rate_vs_ref_K=tool.V3_BOUND_K, inside_table=True, steps=250, inf_cap_segments=0
    )
    assert tool.check('v3_r3', {'metrics': m}) == []
    assert tool.check('v3_r4', {'metrics': m}) == ['no inf cap']
    m.update(max_dT_rate_vs_ref_K=1.01 * tool.V3_BOUND_K, inside_table=False, steps=1501)
    assert tool.check('v3_r4', {'metrics': {**m, 'inf_cap_segments': 1}}) == [
        f'|dT| > {tool.V3_BOUND_K} K',
        'steps > 1500',
        'inside_table',
    ]
    assert tool.check('v3_r3', {'metrics': {**m, 'steps': 251}})[1] == 'steps > 250'
    v4 = dict(status=0, passed_stall_point=True, solidified=True, inside_table=True)
    assert tool.check('v4_rheo_on', v4) == []
    v4.update(status=-1, solidified=False)
    assert tool.check('v4_rheo_off', v4) == ['status -1', 'solidified']
    v4.update(status=0, solidified=True, passed_stall_point=False)
    assert tool.check('v4_rheo_off', v4) == ['passed_stall_point']


def test_main_check_exits_1_on_a_failed_tolerance(monkeypatch, tmp_path, capsys):
    """With --check a failed tolerance gives exit 1; without it the run still exits 0."""
    bad = {'max_dT_rate_vs_ref_K': 1.0, 'inside_table': True, 'steps': 1, 'inf_cap_segments': 0}
    monkeypatch.setattr(tool, 'run_v3', lambda *a: dict(rows=[], metrics=bad))
    monkeypatch.setenv('ARAGOG_TEST_EOS_DIR', 'E')
    args = ['--case', 'v3_r3', '--eos-dir', 'E', '--output', str(tmp_path / 'r.json')]
    assert tool.main([*args, '--check']) == 1
    assert 'v3_r3: FAIL |dT|' in capsys.readouterr().out
    assert tool.main(args) == 0


def test_stop_after_stall_keeps_a_failed_status():
    """Only a successful solid segment past the stall ends the loop; a failure stays a failure."""
    import types

    def res(status):
        return types.SimpleNamespace(status=status, cvode_flag=2)

    solid = {'phi_max': 0.0, 't_yr': 2.0}
    assert tool.stop_after_stall(res(0), solid, 1.0).cvode_flag == 0
    failed = tool.stop_after_stall(res(-3), solid, 1.0)
    assert (failed.status, failed.cvode_flag) == (-3, 2)
    assert tool.stop_after_stall(res(0), solid, 3.0).cvode_flag == 2
    assert tool.stop_after_stall(res(0), {'phi_max': 0.1, 't_yr': 2.0}, 1.0).cvode_flag == 2


def test_main_runs_a_v4_case_and_checks_it(monkeypatch, tmp_path, capsys):
    """main() passes the end time to the v4 run, writes its result and exits 1 on a failure."""
    seen = {}

    def fake_run_v4(eos_dir, name, end_time):
        seen.update(eos_dir=eos_dir, name=name, end_time=end_time)
        return dict(status=-1, passed_stall_point=True, solidified=True, inside_table=True)

    monkeypatch.setattr(tool, 'run_v4', fake_run_v4)
    monkeypatch.setenv('ARAGOG_TEST_EOS_DIR', 'E')
    out = tmp_path / 'v4.json'
    args = ['--case', 'v4_rheo_on', '--eos-dir', 'E', '--end-time', '5', '--output', str(out)]
    assert tool.main([*args, '--check']) == 1
    assert 'v4_rheo_on: FAIL status -1' in capsys.readouterr().out
    assert seen == {'eos_dir': 'E', 'name': 'v4_rheo_on', 'end_time': 5.0}
    assert json.loads(out.read_text())['status'] == -1


def test_main_needs_an_eos_dir(monkeypatch):
    """Without --eos-dir and ARAGOG_TEST_EOS_DIR the command exits with a usage error."""
    monkeypatch.delenv('ARAGOG_TEST_EOS_DIR', raising=False)
    with pytest.raises(SystemExit) as exc:
        tool.main(['--case', 'v3_r3'])
    assert exc.value.code == 2


def test_v4_refuses_a_table_that_is_not_the_paleos_low_t_set(tmp_path):
    """The v4 cases stop before any solve when temperature_solid.dat has another md5."""
    (tmp_path / 'temperature_solid.dat').write_text('not the table\n')
    with pytest.raises(SystemExit, match='needs the PALEOS low-T table set'):
        tool.run_v4(tmp_path, 'v4_rheo_off', 1.0)


@needs_eos
def test_v3_starts_place_the_cells_against_the_solidus():
    """v3_r3 puts cells 4 and 5 above the stiff zone and the rest 15 below; v3_r4 is uniform."""
    eos = tool.EntropyEOS(str(EOS_DIR))
    S_sol = np.asarray(
        eos.solidus_entropy(tool._v3_solver(eos, tool.V3['v3_r3'], 'rate', 1e-8)._P_stag_flat)
    ).ravel()
    offset = tool.v3_start(eos, 'v3_r3') - S_sol
    np.testing.assert_allclose(offset[[4, 5]], [11.0, 12.5], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.delete(offset, [4, 5]), -15.0, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(
        tool.v3_start(eos, 'v3_r4'), np.full_like(S_sol, S_sol[-1] + 5.0)
    )


@needs_eos
def test_v3_cases_run_at_width_zero_unless_replaced():
    """The v3 controller cases fix matprop_smooth_width = 0; --width replaces it."""
    eos = tool.EntropyEOS(str(EOS_DIR))
    for name in tool.V3:
        s = tool._v3_solver(eos, tool.V3[name], 'rate', 1e-8)
        assert s.parameters.phase_mixed.matprop_smooth_width == 0.0
        s = tool._v3_solver(eos, tool.V3[name], 'rate', 1e-8, width=0.01)
        assert s.parameters.phase_mixed.matprop_smooth_width == 0.01


_spec_v5b = importlib.util.spec_from_file_location(
    'run_v5b_proteus',
    Path(__file__).resolve().parents[1] / 'tools' / 'verification' / 'run_v5b_proteus.py',
)
v5b = importlib.util.module_from_spec(_spec_v5b)
_spec_v5b.loader.exec_module(v5b)


def _fake_proteus(monkeypatch, **attrs):
    import types

    pkg, utils, term = (
        types.ModuleType(n) for n in ('proteus', 'proteus.utils', 'proteus.utils.terminate')
    )
    pkg.utils, utils.terminate = utils, term
    for name, value in attrs.items():
        setattr(term, name, value)
    for mod in (pkg, utils, term):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return term


def test_v5b_driver_refuses_without_the_stop_hook(monkeypatch):
    """A PROTEUS without check_termination stops the driver before any run."""
    _fake_proteus(monkeypatch)
    with pytest.raises(SystemExit, match='check_termination is missing'):
        v5b.install(reference=False)


def test_v5b_stop_rule_ends_at_the_third_step_with_phi_zero(monkeypatch):
    """The patched stop ends the run at the 3rd step with Phi_global = 0; others go to PROTEUS."""
    import types

    import aragog.solver.entropy_solver as es

    monkeypatch.setattr(es.EntropySolver, '_solve_cvode', es.EntropySolver._solve_cvode)
    term = _fake_proteus(monkeypatch, check_termination=lambda handler: 'proteus')
    solid, _ = v5b.install(reference=False)
    handler = types.SimpleNamespace(hf_row={'Phi_global': 0.5})
    assert term.check_termination(handler) == 'proteus'
    handler.hf_row['Phi_global'] = 0.0
    assert [term.check_termination(handler) for _ in range(3)] == ['proteus', 'proteus', True]
    assert solid == [3] and handler.finished_both and handler.finished_prev


def _recording_solver(monkeypatch):
    """Patch _solve_cvode with a recorder on a solver whose state scale is [2, 4] and t_ref 10."""
    import types

    import aragog.solver.entropy_solver as es

    calls = []

    def solve_cvode(self, start_time, end_time, y0, atol, rtol, max_step, **kw):
        calls.append((np.asarray(y0, dtype=float), atol, rtol, max_step))
        return {'cvode_nst': 7}

    monkeypatch.setattr(es.EntropySolver, '_solve_cvode', solve_cvode)
    _fake_proteus(monkeypatch, check_termination=lambda handler: False)
    scales = types.SimpleNamespace(state_scale=np.array([2.0, 4.0]), t_ref=10.0)
    return es.EntropySolver, types.SimpleNamespace(_build_nondim_scales=lambda: scales), calls


def test_v5b_driver_counts_steps_and_keeps_the_run_settings(monkeypatch):
    """Without --reference and --s0-eps each call keeps its state, tolerances and max_step."""
    cls, solver, calls = _recording_solver(monkeypatch)
    _, steps = v5b.install(reference=False)
    y0 = np.array([100.0, 200.0])
    cls._solve_cvode(solver, 0.0, 1.0, y0, 1e-8, 1e-8, 5.0)
    cls._solve_cvode(
        solver, start_time=1.0, end_time=2.0, y0=y0, atol=2e-8, rtol=1e-9, max_step=3.0
    )
    assert [c[1:] for c in calls] == [(1e-8, 1e-8, 5.0), (2e-8, 1e-9, 3.0)] and steps == [14]
    assert all(np.array_equal(c[0], y0) for c in calls)


def test_v5b_reference_is_strict_in_every_call_and_s0_eps_moves_the_first_state(monkeypatch):
    """--reference sets 1 yr steps and tolerances; --s0-eps scales the first call that spans time."""
    cls, solver, calls = _recording_solver(monkeypatch)
    v5b.install(reference=True, s0_eps=1e-3, reference_rtol=1e-11)
    cls._solve_cvode(solver, 0.0, 0.0, np.array([100.0, 200.0]), 1e-6, 1e-6, 5.0)
    for max_step in (5.0, np.inf):
        cls._solve_cvode(solver, 0.0, 1.0, np.array([100.0, 200.0]), 1e-6, 1e-6, max_step)
    for _, atol, rtol, max_step in calls:
        np.testing.assert_allclose(atol, [5e-12, 2.5e-12], rtol=1e-15)
        assert (rtol, max_step) == (1e-11, 0.1)
    assert calls[0][0].tolist() == calls[2][0].tolist() == [100.0, 200.0]
    np.testing.assert_allclose(calls[1][0], [100.1, 200.2], rtol=1e-15)


def test_v5b_main_writes_the_result_with_steps_and_wall(monkeypatch, tmp_path):
    """main() writes the config with the output path, runs PROTEUS, and records the run."""
    import types

    import aragog.solver.entropy_solver as es

    monkeypatch.setattr(es.EntropySolver, '_solve_cvode', es.EntropySolver._solve_cvode)
    _fake_proteus(monkeypatch, check_termination=lambda handler: False)
    seen = {}

    class Rows(list):
        @property
        def iloc(self):
            return self

    class Runner:
        def __init__(self, config_path):
            seen['config'] = Path(config_path).read_text()

        def start(self, offline):
            seen['offline'] = offline
            self.hf_all = Rows([dict.fromkeys(v5b.COLUMNS, 1.0)] * 2)

    sys.modules['proteus'].Proteus = Runner
    dumps = lambda d: f'path = {d["params"]["out"]["path"]!r}\n'  # noqa: E731
    monkeypatch.setitem(sys.modules, 'toml', types.SimpleNamespace(dumps=dumps))
    argv = ['--reference', '--reference-rtol', '1e-11', '--s0-eps=1e-12']
    assert v5b.main([*argv, '--output-dir', str(tmp_path)]) == 0
    res = json.loads((tmp_path / 'v5b_result.json').read_text())
    assert (res['reference'], res['reference_rtol'], res['s0_eps']) == (True, 1e-11, 1e-12)
    assert (res['cvode_steps'], res['rows'], res['solid_steps'], res['T_magma']) == (
        0,
        2,
        0,
        1.0,
    )
    assert res['wall_s'] >= 0.0 and seen['offline'] is True
    assert str(tmp_path / 'output') in seen['config']


def test_v4_cases_use_their_end_times(monkeypatch, tmp_path):
    """v4_rheo_off ends at 1.2e8 yr, v4_rheo_on at 1e9 yr; --end-time replaces both."""
    import hashlib

    (tmp_path / 'temperature_solid.dat').write_text('table\n')
    monkeypatch.setattr(tool, 'PALEOS_LOWT_MD5', hashlib.md5(b'table\n').hexdigest())
    monkeypatch.setattr(tool, 'EntropyEOS', lambda _eos_dir: object())
    seen = []

    class Built(Exception):
        pass

    def solver(p, entropy_eos):
        seen.append(p.solver.end_time)
        raise Built

    monkeypatch.setattr(tool, 'EntropySolver', solver)
    for name in ('v4_rheo_off', 'v4_rheo_on'):
        with pytest.raises(Built):
            tool.run_v4(tmp_path, name)
    with pytest.raises(Built):
        tool.run_v4(tmp_path, 'v4_rheo_off', 5.0)
    assert seen == [1.2e8, 1.0e9, 5.0]
