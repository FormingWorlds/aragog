"""``tools/verification/run_step_cap_validation.py``: regime counts, case setup and refusals."""

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


def test_regime_4_when_no_cell_is_above_the_solidus():
    """All cells at or below the solidus give Regime 4; the binding cell sits at 0.5 delta."""
    assert tool.regime(np.array([90.0, 95.0]), -np.ones(2), S_LIQ, S_SOL, MASS, DELTA) == (
        False,
        True,
    )


def test_regime_3_needs_a_slow_binding_cell_near_a_boundary():
    """A cell 3 J/(kg K) above the solidus binds; it is Regime 3 only below 0.05 delta per yr."""
    S = np.array([103.0, 150.0])
    assert tool.regime(S, np.full(2, -0.1), S_LIQ, S_SOL, MASS, DELTA) == (True, False)
    assert tool.regime(S, np.full(2, -1.0), S_LIQ, S_SOL, MASS, DELTA) == (False, False)


def test_regime_counter_counts_calls_and_restores_the_cap():
    """The counter passes the cap value through, counts each call, and restores the function."""
    original = tool.es._rate_phase_boundary_max_step
    args = (np.array([103.0, 150.0]), np.full(2, -0.1), S_LIQ, S_SOL, MASS, DELTA)
    with tool.regime_counter() as counts:
        assert tool.es._rate_phase_boundary_max_step(*args) == original(*args)
        tool.es._rate_phase_boundary_max_step(np.array([90.0, 95.0]), *args[1:])
    assert counts == {'segments': 2, 'regime_3': 1, 'regime_4': 1}
    assert tool.es._rate_phase_boundary_max_step is original
    with pytest.raises(RuntimeError), tool.regime_counter():
        raise RuntimeError
    assert tool.es._rate_phase_boundary_max_step is original


def _row(mode, m, T, regime_3=0, S=(10.0, 20.0)):
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
        regime_3=regime_3,
        regime_4=0,
    )


class _Table:
    _tables = {'temperature_solid': {'S': np.array([0.0, 100.0])}}

    def __init__(self, _eos_dir):
        pass


def test_run_v3_metrics_come_from_the_m0_start_and_the_reference(monkeypatch):
    """The error is m = 0 against 'fixed'; the envelope is the largest m != 0 offset from m = 0."""
    T = {-2: [1.0, 2.0], -1: [1.0, 2.5], 0: [1.0, 2.0], 1: [1.0, 2.0], 2: [0.75, 2.0]}
    rows = {('fixed', 0): _row('fixed', 0, [1.0, 3.0])}
    rows.update({('rate', m): _row('rate', m, T[m], regime_3=int(m == 0)) for m in T})
    monkeypatch.setattr(tool, 'v3_run', lambda task: rows[task[2], task[4]])
    monkeypatch.setattr(tool, 'EntropyEOS', _Table)
    metrics = tool.run_v3('eos', 'v3_r3', workers=1)['metrics']
    assert metrics['max_dT_rate_vs_ref_K'] == 1.0
    assert metrics['round_off_envelope_K'] == 0.5
    assert metrics['regime_3_segments'] == 1
    assert metrics['inside_table'] is True
    rows['rate', 2]['S_max'] = 101.0
    assert tool.run_v3('eos', 'v3_r3', workers=1)['metrics']['inside_table'] is False


def test_main_writes_the_case_result(monkeypatch, tmp_path):
    """main() passes the width to the v3 run and writes the result with the case name."""
    seen = {}

    def fake_run_v3(eos_dir, name, workers, width):
        seen.update(eos_dir=eos_dir, name=name, workers=workers, width=width)
        return dict(rows=[], metrics={'max_dT_rate_vs_ref_K': 0.0})

    monkeypatch.setattr(tool, 'run_v3', fake_run_v3)
    monkeypatch.setenv('ARAGOG_TEST_EOS_DIR', 'E')  # main() sets it; monkeypatch restores it
    out = tmp_path / 'r.json'
    args = ['--case', 'v3_r4', '--eos-dir', 'E', '--workers', '3', '--width', '0', '--output']
    assert tool.main([*args, str(out)]) == 0
    assert seen == {'eos_dir': 'E', 'name': 'v3_r4', 'workers': 3, 'width': 0.0}
    assert json.loads(out.read_text())['case'] == 'v3_r4'


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
    """v3_r3 puts cells 4 and 5 above the stiff zone and the rest 10 below; v3_r4 is uniform."""
    eos = tool.EntropyEOS(str(EOS_DIR))
    S_sol = np.asarray(
        eos.solidus_entropy(tool._v3_solver(eos, tool.V3['v3_r3'], 'rate', 1e-8)._P_stag_flat)
    ).ravel()
    offset = tool.v3_start(eos, 'v3_r3') - S_sol
    np.testing.assert_allclose(offset[[4, 5]], [11.0, 12.5], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.delete(offset, [4, 5]), -10.0, rtol=0, atol=1e-9)
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


def test_v5b_stop_rule_ends_three_steps_after_phi_reaches_zero(monkeypatch):
    """The patched stop ends the run at the 3rd step with Phi_global = 0; others go to PROTEUS."""
    import types

    import aragog.solver.entropy_solver as es

    monkeypatch.setattr(es.EntropySolver, '_solve_cvode', es.EntropySolver._solve_cvode)
    original = es.EntropySolver._solve_cvode
    term = _fake_proteus(monkeypatch, check_termination=lambda handler: 'proteus')
    solid = v5b.install(reference=False)
    assert es.EntropySolver._solve_cvode is original
    handler = types.SimpleNamespace(hf_row={'Phi_global': 0.5})
    assert term.check_termination(handler) == 'proteus'
    handler.hf_row['Phi_global'] = 0.0
    assert [term.check_termination(handler) for _ in range(3)] == ['proteus', 'proteus', True]
    assert solid == [3] and handler.finished_both and handler.finished_prev
    v5b.install(reference=True)
    assert es.EntropySolver._solve_cvode is not original
