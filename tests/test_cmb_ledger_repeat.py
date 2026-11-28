"""A call that the CMB ledger check refuses is solved once more at rtol 1e-10.

The 15-node solver of ``tests/test_cmb_closure_ledger.py`` runs one Radau call from a mush
profile. A closure with ``c_p/T`` of the blend in place of the slope misses the ledger by about
40 B at rtol 1e-8 and 22 B at rtol 1e-10; the exact closure stays at about 0.2 B and 0.02 B.
"""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import needs_eos
from tests.test_cmb_closure_ledger import _solver

RTOL = 1.0e-8


def _blend_cp(self):
    """``c_p`` of the node blend, the closure that misses the ledger in the mush."""
    return float(np.asarray(self.state.phase_basic.heat_capacity()).flat[0])


def _closure(monkeypatch, fault):
    """Set the CMB closure: ``fault(rtol)`` True gives the blend ``c_p``, else the exact one."""
    from aragog.solver.entropy_solver import EntropySolver

    exact = EntropySolver._cmb_closure_cp
    monkeypatch.setattr(
        EntropySolver,
        '_cmb_closure_cp',
        lambda self: _blend_cp(self) if fault(self.parameters.solver.rtol) else exact(self),
    )


def _record_starts(monkeypatch):
    """Record t_span, y0 and rtol of every scipy integrator call."""
    from aragog.solver import entropy_solver as es

    calls, run = [], es.solve_ivp

    def recorded(fun, t_span, y0, **kw):
        calls.append((tuple(t_span), np.array(y0, dtype=float), kw['rtol']))
        return run(fun, t_span, y0, **kw)

    monkeypatch.setattr(es, 'solve_ivp', recorded)
    return calls


def _call(rtol=RTOL, end_time=20.0):
    """Solver with one accepted 2-yr call, set for a call to ``end_time`` at ``rtol``."""
    solver = _solver(0.9, end_time=2.0, solver_method='radau')
    solver.parameters.solver.rtol = rtol
    solver.solve()
    solver.parameters.solver.end_time = end_time
    return solver


@pytest.mark.physics_invariant
@pytest.mark.smoke
@needs_eos
@pytest.mark.timeout(600)
def test_a_refused_call_passes_on_the_repeat_and_the_run_keeps_it(monkeypatch, caplog):
    """The call misses by more than 10 B at rtol 1e-8; its repeat at 1e-10 passes. The run keeps
    the repeat, which equals a solve at 1e-10 from the call start, and the rtol is back at 1e-8."""
    solver = _call()
    _closure(monkeypatch, lambda rtol: rtol > 1.0e-10)
    direct = copy.deepcopy(solver)
    direct.parameters.solver.rtol = 1.0e-10
    direct.solve()
    with caplog.at_level(logging.WARNING, logger='fwl.aragog'):
        solver.solve()
    assert solver.parameters.solver.rtol == RTOL
    np.testing.assert_array_equal(solver._solution.y, direct._solution.y)
    assert solver._solution.t[-1] == pytest.approx(20.0)
    ledger = solver.cmb_ledger
    assert abs(ledger['r']) <= ledger['bound']
    refused = ledger['repeat_of']
    assert refused['rtol'] == RTOL and abs(refused['r']) > 10.0 * refused['bound']
    assert solver._solution.cmb_ledger_repeat is True
    assert solver._cmb_ledger_repeats == {'repeated': 1, 'failed': 0}
    assert solver._cmb_ledger_counts == {'missed': 0, 'excluded': 0}
    lines = [r.getMessage() for r in caplog.records if 'repeat at rtol' in r.getMessage()]
    assert len(lines) == 1 and 'passed' in lines[0], lines


@pytest.mark.physics_invariant
@pytest.mark.smoke
@needs_eos
@pytest.mark.timeout(600)
def test_a_call_that_fails_twice_raises_and_keeps_the_call_start(monkeypatch, caplog):
    """The blend closure misses at both tolerances: the call raises the ledger miss with both
    misses, the solver holds the last accepted state and the rtol of the run, and the call
    counts once as missed."""
    from aragog.solver.entropy_solver import CMBClosureError, CMBLedgerMissError

    solver = _call()
    _closure(monkeypatch, lambda rtol: True)
    accepted = solver.get_current_dSdr_cmb()
    with (
        caplog.at_level(logging.WARNING, logger='fwl.aragog'),
        pytest.raises(CMBLedgerMissError, match='repeat at rtol 1e-10') as info,
    ):
        solver.solve()
    assert isinstance(info.value, CMBClosureError)
    assert isinstance(info.value.__cause__, CMBLedgerMissError)
    assert solver.get_current_dSdr_cmb() == accepted
    assert solver.parameters.solver.rtol == RTOL
    assert solver._cmb_ledger_repeats == {'repeated': 1, 'failed': 1}
    assert solver._cmb_ledger_counts['missed'] == 1
    lines = [r.getMessage() for r in caplog.records if 'repeat at rtol' in r.getMessage()]
    assert len(lines) == 1 and 'failed' in lines[0], lines


@pytest.mark.smoke
@needs_eos
@pytest.mark.timeout(600)
def test_the_repeat_starts_from_the_call_start(monkeypatch):
    """Both attempts start the integrator from the same time span and the same state."""
    solver = _call()
    _closure(monkeypatch, lambda rtol: rtol > 1.0e-10)
    calls = _record_starts(monkeypatch)
    solver.solve()
    assert [c[2] for c in calls] == [RTOL, 1.0e-10]
    assert calls[0][0] == calls[1][0]
    np.testing.assert_array_equal(calls[0][1], calls[1][1])


@pytest.mark.smoke
@needs_eos
@pytest.mark.timeout(600)
def test_the_repeat_starts_from_the_call_start_with_the_lid(monkeypatch):
    """With the sub-grid lid and a lid set before the call, both attempts start from the same
    state, the lid slots included, and the kept solution starts from the set lid. The CMB heat
    of this call is far below its bound, so the check raises the miss at rtol 1e-8 by hand."""
    from aragog.solver.entropy_solver import EntropySolver
    from tests.conftest import EOS_DIR, entropy_eos_copy
    from tests.test_entropy_solver_bc_dispatch_smoke import _build

    p = _build(core_bc='energy_balance', outer_bc=1, inner_bc=1, n_nodes=10, end_time=1.0)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.normalize()
    p.solver.solver_method, p.solver.rtol = 'radau', RTOL
    solver = EntropySolver(p, entropy_eos=entropy_eos_copy(EOS_DIR))
    solver.initialize()
    solver.set_initial_entropy(3050.0)
    solver.set_initial_lid(2.0, -1.0e9)
    _refused_at_1e8(monkeypatch)
    calls = _record_starts(monkeypatch)
    solver.solve()
    assert solver._lid_on and solver._cmb_ledger_repeats == {'repeated': 1, 'failed': 0}
    assert [c[2] for c in calls] == [RTOL, 1.0e-10]
    assert calls[0][0] == calls[1][0]
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    np.testing.assert_array_equal(solver._solution.y[-2:, 0], [2.0, -1.0e9])


def _refused_at_1e8(monkeypatch):
    """Let the ledger check raise its miss after the real check while the rtol is above 1e-10."""
    from aragog.solver.entropy_solver import CMBLedgerMissError, EntropySolver

    check = EntropySolver._check_cmb_ledger

    def refuse(self, *args, **kw):
        check(self, *args, **kw)
        if self.parameters.solver.rtol > 1.0e-10:
            raise CMBLedgerMissError('CMB closure fault in the probe call')

    monkeypatch.setattr(EntropySolver, '_check_cmb_ledger', refuse)


@pytest.mark.smoke
@needs_eos
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ('core_bc', 'method'), [('energy_balance', 'cvode'), ('core_module', 'radau')]
)
def test_the_kept_repeat_equals_a_solve_at_1e10_on_cvode_and_core_module(
    monkeypatch, core_bc, method
):
    """On the CVODE integrator and with the core_module ledger (in K), a call refused at rtol
    1e-8 keeps a repeat equal to a solve at 1e-10 from the same start."""
    from aragog.solver.entropy_solver import EntropySolver
    from tests.conftest import EOS_DIR, entropy_eos_copy
    from tests.test_cmb_closure_ledger import _profile
    from tests.test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build

    if method == 'cvode':
        pytest.importorskip('scikits_odes_sundials')
        solver = _solver(0.9, end_time=20.0, solver_method='cvode', use_jax_jacobian=False)
    else:
        eos = entropy_eos_copy(EOS_DIR)
        solver = _build(
            'core_module', eos, CORE_MODULE_PARAMS, end_time=2.0, solver_method=method
        )
        S = _profile(solver, eos, 0.9)
        solver.set_initial_entropy(S)
        t_m = float(np.asarray(eos.temperature(solver._P_basic_flat[:1], S[:1])).item())
        solver.set_initial_core_temperature(t_m + 50.0)
        solver.set_initial_entropy(S)
    assert isinstance(solver, EntropySolver) and solver._core_bc == core_bc
    solver.parameters.solver.rtol = RTOL
    _refused_at_1e8(monkeypatch)
    direct = copy.deepcopy(solver)
    direct.parameters.solver.rtol = 1.0e-10
    direct.solve()
    solver.solve()
    assert solver._cmb_ledger_repeats == {'repeated': 1, 'failed': 0}
    assert solver.parameters.solver.rtol == RTOL
    np.testing.assert_array_equal(solver._solution.y, direct._solution.y)
    assert solver.cmb_ledger['repeat_of']['rtol'] == RTOL


@pytest.mark.smoke
@needs_eos
@pytest.mark.timeout(900)
def test_the_repeat_counters_add_up_over_calls(monkeypatch):
    """2 refused calls that pass on repeat count 2 repeats; a call that passes at once adds
    none and keeps ``cmb_ledger_repeat`` False."""
    fault = {'on': True}
    solver = _call(end_time=10.0)
    _closure(monkeypatch, lambda rtol: fault['on'] and rtol > 1.0e-10)
    solver.solve()
    solver.parameters.solver.end_time = 20.0
    solver.solve()
    assert solver._cmb_ledger_repeats == {'repeated': 2, 'failed': 0}
    fault['on'] = False
    solver.parameters.solver.end_time = 25.0
    solver.solve()
    assert solver._cmb_ledger_repeats == {'repeated': 2, 'failed': 0}
    assert solver._solution.cmb_ledger_repeat is False
    assert 'repeat_of' not in solver.cmb_ledger


@pytest.mark.smoke
@needs_eos
@pytest.mark.timeout(600)
@pytest.mark.parametrize('rtol', [1.0e-10, 5.0e-11])
def test_no_repeat_at_or_below_the_repeat_rtol(monkeypatch, rtol):
    """A call at rtol 1e-10 or below raises its ledger miss at once."""
    from aragog.solver.entropy_solver import CMBLedgerMissError

    solver = _call(rtol=rtol)
    _closure(monkeypatch, lambda r: True)
    calls = _record_starts(monkeypatch)
    with pytest.raises(CMBLedgerMissError, match='CMB closure fault') as info:
        solver.solve()
    assert info.value.__cause__ is None and len(calls) == 1
    assert solver._cmb_ledger_repeats == {'repeated': 0, 'failed': 0}
    assert solver.parameters.solver.rtol == rtol


@pytest.mark.smoke
@needs_eos
@pytest.mark.timeout(600)
def test_no_repeat_for_a_slope_fault(monkeypatch):
    """A slope fault is not a ledger miss: it raises at once, with no repeat."""
    from aragog.solver import entropy_solver as es

    solver = _call()

    def faulty(self):
        raise es.CMBClosureError('CMB fault at the probe')

    monkeypatch.setattr(es.EntropySolver, '_cmb_closure_cp', faulty)
    calls = _record_starts(monkeypatch)
    with pytest.raises(es.CMBClosureError, match='at the probe') as info:
        solver.solve()
    assert not isinstance(info.value, es.CMBLedgerMissError) and len(calls) == 1
    assert solver._cmb_ledger_repeats == {'repeated': 0, 'failed': 0}


@pytest.mark.smoke
@needs_eos
@pytest.mark.timeout(600)
@pytest.mark.parametrize('kind', ['slope', 'status', 'range', 'raise'])
def test_a_repeat_that_fails_another_way_raises_the_miss(monkeypatch, kind):
    """A slope fault, a failed integrator status, a state outside the EOS tables or another error
    in the repeat: the call raises the ledger miss chained to it, from the call start, with the
    rtol, tolerances and stop flag of the refused attempt and its ledger terms."""
    from scipy.optimize import OptimizeResult

    from aragog.solver import entropy_solver as es

    exact = es.EntropySolver._cmb_closure_cp

    def closure(self):
        if self.parameters.solver.rtol > 1.0e-10:
            return _blend_cp(self)
        if kind == 'slope':
            raise es.CMBClosureError('CMB closure: node slope (dT/dS)_P = -1 at the probe')
        return exact(self)

    solver = _call()
    monkeypatch.setattr(es.EntropySolver, '_cmb_closure_cp', closure)
    accepted = solver.get_current_dSdr_cmb()
    run = es.solve_ivp

    def integrator(fun, t_span, y0, **kw):
        if kw['rtol'] > 1.0e-10 or kind == 'slope':
            return run(fun, t_span, y0, **kw)
        if kind == 'raise':
            raise RuntimeError('integrator probe')
        if kind == 'range':
            raise es.AcceptedStateRangeError('range probe')
        y = np.asarray(y0, dtype=float).reshape(-1, 1)
        return OptimizeResult(t=np.array(t_span[:1]), y=y, status=-1, message='probe', nfev=1)

    monkeypatch.setattr(es, 'solve_ivp', integrator)
    with pytest.raises(es.CMBLedgerMissError, match='repeat at rtol 1e-10') as info:
        solver.solve()
    assert info.value.__cause__ is not None
    assert solver.get_current_dSdr_cmb() == accepted
    assert solver.parameters.solver.rtol == RTOL and solver._call_tol[1] == RTOL
    assert solver.stop_early is False
    ledger = solver.cmb_ledger
    assert abs(ledger['r']) > 10.0 * ledger['bound'] and ledger['rtol'] == RTOL
    assert 'repeat' in ledger
    assert solver._cmb_ledger_repeats == {'repeated': 1, 'failed': 1}


def _stub_solver(rtol, outcomes):
    """Object with the attributes ``EntropySolver.solve`` reads, and a stub of the call body
    that acts by ``outcomes`` (one per attempt: 'pass', 'miss', 'stop' or 'error')."""
    from aragog.solver.entropy_solver import CMBLedgerMissError, EntropySolver

    start = SimpleNamespace(tag='start')
    stub = EntropySolver.__new__(EntropySolver)
    vars(stub).update(
        parameters=SimpleNamespace(solver=SimpleNamespace(rtol=rtol, end_time=1.0)),
        _solution=start,
        _S0=np.arange(4.0),
        _cmb_ledger_counts={'missed': 0, 'excluded': 0},
        _cmb_ledger_repeats={'repeated': 0, 'failed': 0},
        cmb_ledger=None,
        stop_early=False,
        _call_tol=(0.0, rtol),
        _lid_call_tol=(0.0, rtol),
        seen=[],
    )

    def body(_floor_discard=None, _start=None):
        kind = outcomes[len(stub.seen)]
        stub.seen.append((stub.parameters.solver.rtol, _start))
        stub.stop_early = kind == 'stop'
        stub._call_tol = stub._lid_call_tol = (0.0, stub.parameters.solver.rtol)
        stub.cmb_ledger = {'r': 1.0, 'bound': 1.0 if kind != 'miss' else 0.01}
        if kind == 'miss':
            stub._cmb_ledger_counts['missed'] += 1
            stub._solution = _start
            raise CMBLedgerMissError('CMB closure fault in the probe call')
        if kind == 'error':
            stub._solution = _start
            raise ValueError('probe')
        stub._solution = SimpleNamespace(tag='new', t=np.array([0.0, 1.0]))

    stub._solve_call = body
    return stub, start


@pytest.mark.unit
@settings(max_examples=300, deadline=None)
@given(
    rtol=st.one_of(
        st.floats(1.0e-12, 1.0e-6),
        st.sampled_from([1.0e-10, np.nextafter(1.0e-10, 1.0), np.nextafter(1.0e-10, 0.0)]),
    ),
    first=st.sampled_from(['pass', 'miss']),
    second=st.sampled_from(['pass', 'miss', 'stop', 'error']),
)
def test_the_repeat_restores_the_rtol_and_counts_each_call_once(rtol, first, second):
    """Over the rtol range and every outcome pair: the rtol is restored, a repeat runs only
    after a miss above the repeat rtol and starts from the call start, the counters agree
    with the outcome, and a raise leaves the call-start solution."""
    from aragog.solver.entropy_solver import CMBLedgerMissError, EntropySolver

    stub, start = _stub_solver(rtol, [first, second])
    repeat = bool(first == 'miss' and rtol > 1.0e-10)
    fails = first == 'miss' and (not repeat or second != 'pass')
    if fails:
        with pytest.raises(CMBLedgerMissError):
            EntropySolver.solve(stub)
        assert stub._solution is start and stub.stop_early is False
        assert stub._call_tol[1] == rtol
    else:
        EntropySolver.solve(stub)
        assert stub._solution.tag == 'new'
        assert stub._solution.cmb_ledger_repeat is repeat
    assert stub.parameters.solver.rtol == rtol
    assert [s[0] for s in stub.seen] == ([rtol, 1.0e-10] if repeat else [rtol])
    assert all(s[1] is start for s in stub.seen)
    assert stub._cmb_ledger_repeats == {
        'repeated': int(repeat),
        'failed': int(repeat and second != 'pass'),
    }
    assert stub._cmb_ledger_counts['missed'] == int(fails)


@pytest.mark.unit
def test_a_changed_start_state_stops_the_repeat():
    """The repeat refuses to run when the state vector changed during the refused attempt."""
    from aragog.solver.entropy_solver import EntropySolver

    stub, _ = _stub_solver(RTOL, ['miss', 'pass'])
    body = stub._solve_call

    def moves_the_start(*args):
        stub._S0 = stub._S0 + 1.0
        body(*args)

    stub._solve_call = moves_the_start
    with pytest.raises(RuntimeError, match='start state'):
        EntropySolver.solve(stub)
    assert len(stub.seen) == 1
