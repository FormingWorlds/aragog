"""The EOS floor: a root ends a call before the surface entropy leaves the entropy tables."""

from __future__ import annotations

import dataclasses
import logging
import warnings

import numpy as np
import pytest

from aragog.solver import entropy_solver as es
from aragog.solver.entropy_solver import (
    EntropySolver,
    _floor_entropy,
    _PhaseBoundarySegmentRoot,
    _PhiCapRootFunction,
)

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos
from .test_phi_step_cap_armed_smoke import _build_mushy_parameters, _pick_mushy_S

pytestmark = [pytest.mark.smoke, needs_eos]

_FLUX = 1.0e6  # prescribed surface flux [W/m^2]; drains the top cell of the cold start in ~3 yr
_TOL = 1.0e-4  # 1e-6 of the default margin


@pytest.fixture(scope='module')
def shared_eos():
    return entropy_eos_copy(EOS_DIR)


def _weights(s):
    return np.asarray(s.evaluator.mesh.quantity_at_basic_nodes(np.eye(s._n_stag)))[-1]


def _basic(s):
    """Basic-node entropies above the CMB node (not watched by the floor) of every state after
    the call start of the last solve."""
    sol = s._solution
    S = np.asarray(sol.y, dtype=float)[: s._n_stag, np.asarray(sol.t) > sol.t[0]]
    return np.asarray(s.evaluator.mesh.quantity_at_basic_nodes(S))[1:]


def _floor_series(s):
    """The floor entropy at every output point of the last solve."""
    S = np.asarray(s._solution.y, dtype=float)[: s._n_stag]
    return np.array([_floor_entropy(_weights(s), S[:, k]) for k in range(S.shape[1])])


def _drained(
    eos,
    S0,
    *,
    flux=_FLUX,
    margin=100.0,
    method='cvode',
    core='quasi_steady',
    mode=None,
    end_time=10.0,
    start_time=0.0,
    **caps,
):
    """A cold start at entropy ``S0`` (scalar or per cell) under a prescribed surface flux, over
    ``end_time`` years from ``start_time``."""
    p = _build_mushy_parameters(solver_method=method, end_time=start_time + end_time)
    p.solver.start_time = start_time
    p.boundary_conditions.outer_boundary_condition = 4
    p.boundary_conditions.outer_boundary_value = flux
    p.boundary_conditions.core_bc = core
    p.energy = dataclasses.replace(
        p.energy,
        **{'phi_step_cap': None, **caps},
        eos_floor_margin=margin,
        phase_boundary_cap=mode,
    )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(np.broadcast_to(np.asarray(S0, dtype=float), s._n_stag).copy())
    s.solve()
    if (
        margin is not None and margin > 0.0
    ):  # the floor keeps basic nodes and cells in the table
        sol, n = s._solution, s._n_stag
        trace = sol.get('energy_trace')
        assert np.all(_basic(s) >= float(eos.S_min)) and _cells(s).min() >= float(eos.S_min)
        assert trace is None or np.asarray(trace[1])[:n].min() >= float(eos.S_min)
    return s, _floor_series(s)


def _floor_logs(caplog):
    return [
        (r.levelno, r.getMessage()) for r in caplog.records if 'EOS floor' in r.getMessage()
    ]


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_floor_root_ends_the_call_at_the_margin(shared_eos, caplog, method):
    """CVODE (status 0) and scipy (status 1) stop at S_min + 100 before the call end with one
    WARNING; without the floor the same call drains below the table."""
    S_min = float(shared_eos.S_min)
    with caplog.at_level(logging.INFO):
        s, f = _drained(shared_eos, S_min + 400.0, method=method)
    sol = s._solution
    assert sol.status == (0 if method == 'cvode' else 1) and 0.0 < sol.t[-1] < 10.0
    assert f[-1] == pytest.approx(S_min + 100.0, abs=1.0e-3) and f.min() >= S_min + 100.0 - 1e-3
    logs = _floor_logs(caplog)
    assert len(logs) == 1 and logs[0][0] == logging.WARNING
    assert f'level={S_min + 100.0:.6g}' in logs[0][1]
    if method == 'cvode':
        assert sol.cap_label == 'eos_floor' and sol.cap_value == S_min + 100.0
    s_off, f_off = _drained(shared_eos, S_min + 400.0, margin=None, method=method)
    assert s_off._solution.t[-1] == pytest.approx(10.0) and f_off[-1] < S_min


@pytest.mark.parametrize(
    ('core', 'mode', 'stratified', 'offset'),
    [
        ('energy_balance', 'fixed', False, 400.0),
        ('quasi_steady', 'rate', True, 400.0),
        ('energy_balance', 'rate', True, 400.0),
        ('quasi_steady', None, False, 80.0),
    ],
)
def test_floor_stops_each_core_and_rate_segments_at_its_level(
    shared_eos, core, mode, stratified, offset
):
    """The floor stops at S_min + 100, or S_min + 50 for a start below S_min + 100 (+ tol), on
    both cores; a start from the mushy band down to S_min + offset runs rate-cap segments."""
    S_min = float(shared_eos.S_min)
    S0 = S_min + offset
    if stratified:
        S0 = np.linspace(_pick_mushy_S(shared_eos), S_min + offset, 11)
    s, f = _drained(shared_eos, S0, core=core, mode=mode)
    s0 = _floor_entropy(_weights(s), np.broadcast_to(S0, s._n_stag))
    level = S_min + (100.0 if s0 > S_min + 100.0 + _TOL else 50.0)
    assert s._solution.cap_label == 'eos_floor' and 0.0 < s._solution.t[-1] < 10.0
    assert f[-1] == pytest.approx(level, abs=1.0e-3)
    assert bool(s._solution.get('segments')) is stratified


@pytest.mark.parametrize(
    ('offset', 'level'), [(100.0 + 5e-5, 50.0), (100.0 + 1e-3, 100.0), (50.0 + 1e-3, 50.0)]
)
def test_floor_level_band_is_one_millionth_of_the_margin(shared_eos, offset, level):
    """A start within 1e-6 m above a level takes the next one down; 1e-3 above it keeps it."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + offset)
    assert s._solution.cap_label == 'eos_floor' and f[-1] == pytest.approx(
        S_min + level, abs=1e-6
    )


def test_a_persistent_drain_ends_after_two_floor_stops_and_a_pause_does_not(shared_eos):
    """Calls under the same drain stop at S_min + 100, then S_min + 50, then are refused; a call
    without the drain after that advances its full span."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + 400.0)
    ends, t, status = [f[-1]], float(s._solution.t[-1]), []
    for flux in (_FLUX, _FLUX, 0.0):
        S = np.array(s._solution.y[: s._n_stag, -1])
        s.parameters.solver.start_time, s.parameters.solver.end_time = t, t + 10.0
        s.parameters.boundary_conditions.outer_boundary_value = flux
        s.reset()
        s.set_initial_entropy(S)
        s.solve()
        ends.append(_floor_series(s)[-1])
        status.append((s._solution.status, float(s._solution.t[-1]) - t))
        t = float(s._solution.t[-1])
    assert ends[0] == pytest.approx(S_min + 100.0, abs=1.0e-3)
    assert ends[1] == pytest.approx(S_min + 50.0, abs=1.0e-3)
    assert status[1] == (1, 0.0) and ends[2] == ends[1]
    assert status[2] == (0, pytest.approx(10.0)) and ends[3] >= ends[2]


@pytest.mark.parametrize(
    'where',
    ['below', 'at', 'one ulp above', 'at the band edge', 'at half the margin', 'in the band'],
)
@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_floor_refuses_a_draining_start_at_or_below_half_the_margin(
    shared_eos, caplog, method, where
):
    """Below S_min, at it, 1 ulp above it, at S_min + 50 or within 1e-6 m above it, a draining call does
    not advance: status 1, dt_actual 0, one named warning; the log names the requested span."""
    S_min = float(shared_eos.S_min)
    S0 = {
        'below': S_min - 10.0,
        'at': np.r_[S_min, _steep_top(shared_eos, [400.0] * 3)[1:]],  # a cell exactly at S_min
        'one ulp above': np.nextafter(S_min, np.inf),
        'at the band edge': S_min + 2.0e-6 * 100.0,
        'at half the margin': S_min + 50.0,
        'in the band': S_min + 50.0 + 5e-5,
    }[where]
    with caplog.at_level(logging.INFO):
        s, _ = _drained(shared_eos, S0, method=method)
    sol = s._solution
    assert sol.status == 1 and sol.t[-1] == sol.t[0] == 0.0
    assert s.get_state().dt_actual == 0.0
    messages = [r.getMessage() for r in caplog.records]
    refusals = [r for r in caplog.records if 'the call does not advance' in r.getMessage()]
    assert len(refusals) == 1 and refusals[0].levelno == logging.WARNING
    text = {'below': 'at or below the table minimum', 'at': 'at or below the table minimum'}
    text['one ulp above'] = 'no level fits'
    text['at the band edge'] = 'counts as at the table'
    text = text.get(where, 'falls too fast')
    assert text in refusals[0].getMessage() and sol.message.startswith('EOS floor:')
    assert 'J/kg/K from the table minimum' in refusals[0].getMessage()
    assert any('stopped 10.0 yr before end_time' in m for m in messages)


@pytest.mark.parametrize('flux', [0.0, -1.0e5])
def test_floor_lets_a_start_in_the_band_advance_when_it_does_not_drain(shared_eos, flux):
    """At S_min + 40 a call with no or an inward surface flux advances its full span."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + 40.0, flux=flux)
    assert s._solution.status == 0 and s._solution.t[-1] == pytest.approx(10.0)
    assert f.min() > S_min


def test_floor_leaves_a_non_finite_start_to_the_table_guard(shared_eos):
    """A NaN start raises the same error with the floor as without it, not a floor refusal."""
    S_min = float(shared_eos.S_min)
    errors = []
    for margin in (100.0, None):
        S0 = np.full(11, S_min + 400.0)  # 12 nodes, 11 cells
        S0[3] = np.nan
        with pytest.raises(Exception) as excinfo:
            _drained(shared_eos, S0, margin=margin)
        errors.append(type(excinfo.value))
    assert errors[0] is errors[1]


def test_floor_margin_must_be_below_the_table_span(shared_eos):
    """A margin as large as the table entropy span is refused by name at solve."""
    span = float(shared_eos.S_max - shared_eos.S_min)
    with pytest.raises(ValueError, match='eos_floor_margin'):
        _drained(shared_eos, float(shared_eos.S_min) + 400.0, margin=span)


@pytest.mark.parametrize('phi_step_cap', [None, 0.05])
@pytest.mark.parametrize('method', ['cvode', 'bdf'])
@pytest.mark.parametrize('mode', ['fixed', 'rate'])
@pytest.mark.parametrize('core', ['quasi_steady', 'energy_balance'])
def test_floor_on_and_off_are_bit_equal_when_it_does_not_fire(
    shared_eos, monkeypatch, method, mode, core, phi_step_cap
):
    """A call that never comes near the table edge takes the same steps with the floor armed as
    without it (also with no other cap): output times, states and energy integrals are equal."""
    built, real = [], es._PhiCapRootFunction.__init__
    monkeypatch.setattr(
        es._PhiCapRootFunction,
        '__init__',
        lambda self, *a, **k: built.append(1) or real(self, *a, **k),
    )
    runs = []
    for margin in (100.0, None):
        built.clear()
        p = _build_mushy_parameters(solver_method=method, end_time=2.0)
        p.boundary_conditions.core_bc = core
        p.energy = dataclasses.replace(
            p.energy,
            eos_floor_margin=margin,
            phase_boundary_cap=mode,
            phi_step_cap=phi_step_cap,
        )
        s = EntropySolver(p, entropy_eos=shared_eos)
        s.initialize()
        s.set_initial_entropy(_pick_mushy_S(shared_eos))
        s.solve()
        runs.append((s._solution, bool(built)))
    (on, armed_on), (off, armed_off) = runs
    assert armed_on and armed_off is (phi_step_cap is not None)
    assert on.status == off.status == 0 and not getattr(on, 'cap_fired', False)
    assert np.array_equal(on.t, off.t) and np.array_equal(on.y, off.y)
    assert on.energy_integrals == off.energy_integrals


def test_floor_component_takes_the_lower_of_the_top_node_and_the_cells():
    """g is the smaller of w @ S and min(S) minus the level, and it names the floor; the
    exception sentinel stays positive with no other cap armed."""
    w = np.array([0.0, -0.5, 1.5])
    kw = dict(
        eos=None, P_stag=np.zeros(3), volume=np.ones(3), n_stag=3, phi0_global=0.0, cap=0.0
    )
    root = _PhiCapRootFunction(
        **kw, state_scale=np.ones(3), floor_weights=w, floor_level=200.0, floor_margin=100.0
    )
    g = np.zeros(1)
    for S, expected in (
        ([500.0, 400.0, 300.0], 250.0 - 200.0),
        ([500.0, 300.0, 400.0], 300.0 - 200.0),
    ):
        root.evaluate(0.0, np.array(S), g)
        assert g[0] == pytest.approx(expected) and root.binding_cap == 'eos_floor'
    root.evaluate(0.0, None, g)
    assert g[0] == 100.0 and root.binding_cap is None


def _segment_roots(floor_weights):
    cap = _PhiCapRootFunction(
        eos=None, P_stag=np.zeros(2), volume=np.ones(2), n_stag=2, phi0_global=0.0, cap=0.0,
        state_scale=np.ones(2), floor_weights=floor_weights, floor_level=200.0, floor_margin=100.0,
    )  # fmt: skip
    y0 = np.array([5000.0, 5000.0])
    return cap, _PhaseBoundarySegmentRoot(
        np.full(3, 6000.0), np.full(3, 4000.0), y0, 10.0, None, np.ones(2), 2, cap=cap
    )


def test_segment_root_names_the_floor_as_the_cap():
    """At a floor root the segment roots report 'cap', which ends a rate-cap call; a floor-only
    cap whose evaluate raised is scaled by its margin, without a division by zero."""
    _, roots = _segment_roots(np.array([-0.5, 1.5]))
    assert roots.names[0] == 'cap' and roots.fired(0.0, np.array([250.0, 200.0])) == 'cap'
    bad, roots = _segment_roots(np.ones(3))
    # warnings as errors: a zero scale would warn on division
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert roots.fired(0.0, np.array([5000.0, 5000.0])) in roots.names
    assert bad.binding_cap is None


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
@pytest.mark.parametrize(
    'caps',
    [{'phi_step_cap': 0.05}, {'temperature_step_cap': 1.0e4}, {'entropy_step_cap': 1.0e5}],
)
def test_floor_fires_and_refuses_with_another_cap_armed(shared_eos, method, caps):
    """With a phi, temperature or entropy step cap armed, the floor still ends a drained call at
    S_min + 100 (named eos_floor on CVODE) and refuses a draining start at S_min + 50."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + 400.0, method=method, **caps)
    assert 0.0 < s._solution.t[-1] < 10.0
    assert f[-1] == pytest.approx(S_min + 100.0, abs=1.0e-3)
    if method == 'cvode':
        assert s._solution.cap_label == 'eos_floor'
    s, _ = _drained(shared_eos, S_min + 50.0, method=method, **caps)
    assert s._solution.status == 1 and s.get_state().dt_actual == 0.0


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_zero_span_call_is_not_refused(shared_eos, caplog, method):
    """A call with start == end (a static initial call) skips the floor: a falling start at the
    table returns status 0 without a floor log."""
    with caplog.at_level(logging.INFO):
        s, _ = _drained(
            shared_eos, float(shared_eos.S_min) + 1.0e-5, method=method, end_time=0.0
        )
    assert s._solution.status == 0 and not _floor_logs(caplog)


def test_a_top_node_below_the_table_with_every_cell_inside_is_refused(shared_eos, caplog):
    """A steep top gradient extrapolates the top basic node below S_min while every cell is
    inside the table: under a slow drain the call does not advance."""
    S_min = float(shared_eos.S_min)
    p = _build_mushy_parameters(solver_method='cvode', end_time=10.0)
    s = EntropySolver(p, entropy_eos=shared_eos)
    s.initialize()
    S0 = np.full(s._n_stag, S_min + 2000.0)
    S0[-2:] = S_min + 236.0, S_min + 40.0
    assert _weights(s) @ S0 < S_min  # precondition: the extrapolated top node is out
    with caplog.at_level(logging.INFO):
        s, _ = _drained(shared_eos, S0, flux=50.0)
    assert s._solution.status == 1 and s.get_state().dt_actual == 0.0
    refusals = [r for r in caplog.records if 'the call does not advance' in r.getMessage()]
    assert len(refusals) == 1 and refusals[0].levelno == logging.WARNING
    assert s._solution.message.startswith('EOS floor:')


@pytest.mark.parametrize(('core', 'end_time'), [('quasi_steady', 0.0), ('gradient', 10.0)])
def test_the_margin_is_checked_also_without_span_or_on_the_gradient_core(
    shared_eos, core, end_time
):
    """The margin check against the table span runs also on a call with no span and on the
    gradient core, where the floor is not armed."""
    span = float(shared_eos.S_max) - float(shared_eos.S_min)
    with pytest.raises(ValueError, match='eos_floor_margin'):
        _drained(
            shared_eos,
            float(shared_eos.S_min) + 400.0,
            margin=span,
            core=core,
            end_time=end_time,
        )


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_the_floor_log_names_the_half_margin_level(shared_eos, caplog, method):
    """A start between S_min + m/2 and S_min + m stops at S_min + 50, and the log names it."""
    S_min = float(shared_eos.S_min)
    with caplog.at_level(logging.INFO):
        _, f = _drained(shared_eos, S_min + 80.0, method=method)
    logs = _floor_logs(caplog)
    assert f[-1] == pytest.approx(S_min + 50.0, abs=1.0e-3)
    assert len(logs) == 1 and f'level={S_min + 50.0:.6g}' in logs[0][1]


def test_an_entropy_cap_that_binds_first_ends_the_call_before_the_floor(shared_eos):
    """With a 50 J/kg/K entropy cap and the floor 300 J/kg/K away, the entropy cap ends the
    call and the floor does not."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + 400.0, entropy_step_cap=50.0)
    assert s._solution.cap_label == 'entropy' and f[-1] > S_min + 100.0


def _cells(s):
    return np.asarray(s._solution.y, dtype=float)[: s._n_stag]


def _steep_top(eos, top3):
    """Interior at S_min + 2000 with the top three cells at S_min + ``top3``."""
    p = _build_mushy_parameters(solver_method='cvode', end_time=10.0)
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    S0 = np.full(s._n_stag, float(eos.S_min) + 2000.0)
    S0[-3:] = float(eos.S_min) + np.asarray(top3, dtype=float)
    return S0


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
@pytest.mark.parametrize(
    ('top3', 'flux', 'end_time'),
    [([2000.0, 236.0, 40.0], 0.0, 10.0), ([40.0, 2000.0, 40.0], 860.0, 2.0e4)],
    ids=['warming top cell', 'rising node'],
)
def test_a_top_node_below_the_table_is_refused_when_it_or_the_top_cell_rises(
    shared_eos, caplog, method, top3, flux, end_time
):
    """Top node at S_min - 58 under flux 0 (it falls only by extrapolation while the top cell
    warms) or at S_min - 940 under 860 W/m^2 (it rises while the top cell falls): the call does
    not advance, with the named refusal."""
    S0 = _steep_top(shared_eos, top3)
    with caplog.at_level(logging.INFO):
        s, _ = _drained(shared_eos, S0, flux=flux, end_time=end_time, method=method)
    assert s._solution.status == 1 and s.get_state().dt_actual == 0.0
    refusals = [r for r in caplog.records if 'the call does not advance' in r.getMessage()]
    assert len(refusals) == 1 and 'at or below the table minimum' in refusals[0].getMessage()
    assert s._solution.message.startswith('EOS floor:')


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_top_node_in_the_table_band_falling_by_extrapolation_is_refused(
    shared_eos, caplog, method
):
    """Top node at S_min + 1e-4 under flux 0, falling only by extrapolation while the top cell
    warms: inside S_min + 2e-6 m the node's own rate counts, so the call is refused at once."""
    S0 = _steep_top(shared_eos, [2000.0, 120.0 - 2e-4, 40.0])
    with caplog.at_level(logging.INFO):
        s, _ = _drained(shared_eos, S0, flux=0.0, method=method)
    assert s._solution.status == 1 and s.get_state().dt_actual == 0.0
    refusals = [r for r in caplog.records if 'the call does not advance' in r.getMessage()]
    assert len(refusals) == 1 and 'counts as at the table' in refusals[0].getMessage()


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_node_falling_by_extrapolation_halves_its_distance_then_is_refused(
    shared_eos, method
):
    """Top node at S_min + 1e-2 under flux 0, falling only by extrapolation while the top cell
    warms: each call stops at half the node's distance until the start is inside S_min + 2e-6 m,
    at most ceil(log2(1e-2 / 2e-4)) = 6 calls, and that start is refused."""
    S_min = float(shared_eos.S_min)
    S0, t0, advancing = _steep_top(shared_eos, [2000.0, 120.0 - 2e-2, 40.0]), 0.0, 0
    for _ in range(10):
        s, _ = _drained(shared_eos, S0, flux=0.0, method=method, start_time=t0)
        if s.get_state().dt_actual == 0.0:
            break
        advancing += 1
        S0, t0 = _cells(s)[:, -1], float(s._solution.t[-1])
    assert s._solution.status == 1 and 1 <= advancing <= 6
    assert S_min < _weights(s) @ S0 <= S_min + 2e-4  # the refused start is in the band


T0S = [0.0, 1.0e6, 1.0e9, 4.5e9]  # call start times [yr]; the root error grows with the time


def _stopped_or_discarded(s, caplog, level):
    """The call stopped with the top node at most at ``level`` (the root error grows with the
    time) and inside the table (checked in ``_drained``), or it was discarded: status 1, no time
    advanced and the warning that names the discard."""
    if s.get_state().dt_actual > 0.0:
        assert _basic(s)[-1, -1] <= level + 1e-7
    else:
        assert s._solution.status == 1 and 'discarded' in caplog.text


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
@pytest.mark.parametrize('start_time', T0S)
def test_a_cell_at_the_table_that_does_not_fall_still_guards_the_others(
    shared_eos, caplog, method, start_time
):
    """A mid cell 1e-5 above S_min, every other cell at S_min + 400, 1e6 W/m^2: the start arms
    half the mid cell's distance, and the root stops the top node, extrapolated
    from the draining top cell, there instead of letting it leave the table."""
    S_min = float(shared_eos.S_min)
    S0 = np.full(len(_steep_top(shared_eos, [400.0] * 3)), S_min + 400.0)
    S0[5] = S_min + 1e-5  # a mid cell: no extrapolated basic node leaves the table
    with caplog.at_level(logging.INFO):
        s, _ = _drained(shared_eos, S0, method=method, start_time=start_time)
    assert s._solution.t[-1] < start_time + 10.0
    _stopped_or_discarded(s, caplog, S_min + 5e-6)
    if start_time == 0.0:
        assert s.get_state().dt_actual > 0.0


def _mid_cell_at(eos, d):
    """Every cell at S_min + 400 except a mid cell at S_min + d (no extrapolated node leaves)."""
    S0 = np.full(len(_steep_top(eos, [400.0] * 3)), float(eos.S_min) + 400.0)
    S0[5] = float(eos.S_min) + d
    return S0


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
@pytest.mark.parametrize('start_time', T0S)
@pytest.mark.parametrize('d', [1e-3, 1e-6, 1e-8])
def test_a_late_call_near_the_table_returns_no_state_below_it(
    shared_eos, method, start_time, d
):
    """A mid cell S_min + d under 1e6 W/m^2 from 0 to 4.5e9 yr, where the root location error
    of the integrator grows with the time: no returned state or trace node has a basic node above
    the CMB node or a cell below S_min (checked in ``_drained``), whether the call stops or not."""
    _drained(shared_eos, _mid_cell_at(shared_eos, d), method=method, start_time=start_time)


@pytest.mark.parametrize(('method', 'start_time'), [('bdf', 9.0e9), ('radau', 1.2345678e9)])
def test_a_late_scipy_call_near_the_table_returns_no_state_below_it(
    shared_eos, method, start_time
):
    """The scipy integrators, whose event location error also grows with the time, at start
    times where a stop near the table lands below it."""
    _drained(shared_eos, _mid_cell_at(shared_eos, 1e-6), method=method, start_time=start_time)


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_warming_start_with_no_room_for_a_level_names_that(shared_eos, caplog, method):
    """A mid cell one ulp above S_min that warms leaves no level between S_min and the start:
    the call is refused with a warning that says so, not that the entropy falls."""
    S_min = float(shared_eos.S_min)
    with caplog.at_level(logging.INFO):
        s, _ = _drained(
            shared_eos,
            _mid_cell_at(shared_eos, 0.0)
            + np.eye(11)[5] * (np.nextafter(S_min, np.inf) - S_min),
            flux=0.0,
            method=method,
        )
    assert s._solution.status == 1 and s.get_state().dt_actual == 0.0
    refusals = [r for r in caplog.records if 'the call does not advance' in r.getMessage()]
    assert len(refusals) == 1 and 'no level fits' in refusals[0].getMessage()


@pytest.mark.parametrize(
    ('method', 'start_time'), [('cvode', 1.0e9), ('bdf', 9.0e9), ('radau', 1.2345678e9)]
)
def test_a_discarded_call_keeps_the_start_state_time_and_energy_ledger(
    shared_eos, caplog, method, start_time
):
    """A mid cell 1e-6 above S_min under 1e6 W/m^2 at a late start time, where the stop lands
    below the table: the call is discarded with a warning that gives the level and the entropy
    as offsets from S_min, and its state, time, energy trace and per-call energy ledger equal
    those of a call with no span from the same start."""
    S0 = _mid_cell_at(shared_eos, 1e-6)
    with caplog.at_level(logging.INFO):
        s, _ = _drained(shared_eos, S0, method=method, start_time=start_time)
    z, _ = _drained(shared_eos, S0, method=method, start_time=start_time, end_time=0.0)
    assert 'put the floor entropy at S_min -' in caplog.text
    sol, st, zt = s._solution, s.get_state(), z.get_state()
    assert sol.status == 1 and st.dt_actual == 0.0 and 'discarded' in caplog.text
    assert np.array_equal(_cells(s)[:, -1], S0) and np.all(np.asarray(sol.t) == sol.t[0])
    assert np.all(np.asarray((sol.get('energy_trace') or (sol.t, sol.y))[0]) == sol.t[0])
    ledger = [k for k in vars(st) if k.startswith('step_dE_')]
    assert ledger and all(getattr(st, k) == getattr(zt, k) for k in ledger)


@pytest.mark.parametrize('start_time', [1.0e8, 54522722.0])
def test_a_floor_stop_at_the_call_start_is_a_refusal(
    shared_eos, caplog, monkeypatch, start_time
):
    """A level 1e-12 below a draining start at 1e8 yr, or at 54522722 yr, which comes back one
    ulp later through the time scale: the scipy event stops the call at its start, which is
    logged and returned as a refusal, not as a silent zero step."""
    monkeypatch.setattr(
        es, '_floor_start_level', lambda w, S, *_: (es._floor_entropy(w, S) - 1e-12, False)
    )
    with caplog.at_level(logging.INFO):
        s, _ = _drained(
            shared_eos, float(shared_eos.S_min) + 400.0, method='bdf', start_time=start_time
        )
    refusals = [r for r in caplog.records if 'the call does not advance' in r.getMessage()]
    assert s._solution.status == 1 and s.get_state().dt_actual == 0.0 and len(refusals) == 1
    assert 'fell at the call start' in refusals[0].getMessage()
