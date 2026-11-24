"""``aragog.solver.energy_stream``: the per-call energy integrals and node checks in chunks.

Checks that the node filter of a CVODE call equals the stored-list filter, including a root
step whose end lies past the call end, and passes no node past an output time CVODE has not
returned; that the joined nodes of a solve are deduplicated and converted as the stored trace
was; that every consumer gives with 4-node chunks what it gives on the whole trace when its
extreme or its fault lies in a later chunk (the trapezoid, the core heat tolerance, the CMB
ledger, the floor minimum, the table guard, the regime check, the lid weight maximum); that
the checks raise in the order of the stored trace; that a batch failure in a later chunk falls
back from that chunk on and names the earlier steps; that a failed check is raised after the
solve, not inside a callback; and that the memory of the recording does not grow with the node
count.
"""

from __future__ import annotations

import re
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from aragog.solver import energy_stream as st
from aragog.solver import entropy_solver as es

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _list_filter(nodes, t, y, start=True):
    """The node filter of a stored list: increasing times up to the call end, bracketed by the
    start and end states."""
    ts, ys = [float(t[0])], [y[:, 0]]
    for tn, yn in nodes:
        if ts[-1] < tn <= t[-1]:
            ts.append(tn)
            ys.append(yn)
    if ts[-1] < t[-1]:
        ts.append(float(t[-1]))
        ys.append(y[:, -1])
    if not start:
        ts, ys = ts[1:], ys[1:]
    return np.array(ts), np.array(ys).reshape(len(ts), -1).T


def _cvode_like_nodes(rng, t0, n, root):
    """Root-function times of a CVODE call: increasing step ends with output points behind
    them, and with ``root`` a last step past the root followed by root-search points around
    it. Returns the times, the call end and the output times."""
    ends = t0 + np.cumsum(rng.uniform(1e-3, 4e-3, n))
    times, t_out = [], [t0]
    for k, te in enumerate(ends):
        times.append(float(te))
        if k % 7 == 3:  # an output point inside the step just taken
            times.append(float(te - 0.3 * (te - (ends[k - 1] if k else t0))))
            t_out.append(times[-1])
    if root:
        t_end = float(ends[-1] + 2e-3)
        times.append(t_end + 1.5e-3)  # the step that crosses the root
        times += [t_end + 7e-4, t_end - 4e-4, t_end + 2e-4, t_end - 1e-4, t_end]
    else:
        t_end = float(ends[-1] + 1e-3)
        times.append(t_end + 2e-3)  # CVODE steps past the call end and interpolates back
        times.append(t_end)
    t_out.append(t_end + (1e-2 if root else 0.0))
    return times, t_end, np.array(t_out)


@pytest.mark.parametrize('root', [False, True])
@pytest.mark.parametrize('chunk', [3, 5, 2048])
def test_call_nodes_equal_the_list_filter_across_chunk_and_tail_boundaries(
    monkeypatch, root, chunk
):
    """Two joined calls of 300 steps, with output points and (``root``) a root step whose end
    lies past the call end, give the nodes of the stored-list filter bit for bit."""
    monkeypatch.setattr(st, 'CHUNK', chunk)
    rng = np.random.default_rng(chunk + root)
    sink = st.ListSink()
    expected_t, expected_y, t0 = [], [], 0.0
    y_end = None
    for start in (True, False):  # the second call continues the first one's end node
        times, t_end, t_out = _cvode_like_nodes(rng, t0, 300, root)
        nodes = [(tn, rng.normal(size=3)) for tn in times]
        y0 = rng.normal(size=3) if start else y_end
        t = np.array([t0, t_end])
        y = np.column_stack([y0, rng.normal(size=3)])
        call = st.CallNodes(sink, t0, y0, t_out, start=start)
        for tn, yn in nodes:
            call.record(tn, yn)
            assert sink.t[-1] <= t_end  # no node past the call end before the call returns
        call.end(t, y)
        ts, ys = _list_filter(nodes, t, y, start=start)
        assert call.count == ts.size
        expected_t.append(ts)
        expected_y.append(ys)
        t0, y_end = t_end, y[:, -1]
    got_t, got_y = sink.arrays()
    np.testing.assert_array_equal(got_t, np.concatenate(expected_t))
    np.testing.assert_array_equal(got_y, np.concatenate(expected_y, axis=1))
    assert np.all(np.diff(got_t) > 0)
    # the nodes past the call end were dropped: the call end is the last node of each call
    assert got_t[-1] == pytest.approx(t0, abs=0)


def test_a_call_without_nodes_and_without_a_start_passes_only_its_end():
    """A joined call with no root-function nodes passes its end state only; a call of zero
    length passes nothing, and an empty sink has no trace."""
    sink = st.ListSink()
    assert sink.arrays() is None
    call = st.CallNodes(sink, 2.0, np.zeros(2), [2.0, 3.0], start=False)
    call.end(np.array([2.0, 3.0]), np.array([[0.0, 1.0], [0.0, 2.0]]))
    t, y = sink.arrays()
    np.testing.assert_array_equal(t, [3.0])
    np.testing.assert_array_equal(y, [[1.0], [2.0]])
    empty = st.CallNodes(sink, 3.0, np.zeros(2), [3.0], start=False)
    empty.end(np.array([3.0, 3.0]), np.ones((2, 2)))
    assert empty.count == 0 and sink.arrays()[0].size == 1


class _Collect:
    def __init__(self):
        self.chunks = []

    def add(self, t, y, change=None):
        self.chunks.append((t.copy(), y.copy(), change.copy()))


def test_solve_nodes_deduplicate_convert_and_chunk_as_the_stored_trace(monkeypatch):
    """Nodes whose nondimensional times round to one year keep the last; equal nondimensional
    times (a lid hand-over) are both kept; each state is ``y scale + offset``, the change rows
    ``y scale``, and the lid energy is 0 where the thickness is 0; 3-node chunks change none
    of it."""
    monkeypatch.setattr(st, 'CHUNK', 3)
    t_ref = 0.003168808781402895
    a = b = 1e8 / t_ref
    while b * t_ref != a * t_ref or a == b:  # two distinct times that round to one year
        a, b = b, np.nextafter(b, np.inf)
    t_nd = np.array([a - 1e6, a - 5e5, a, b, b, a + 1e6, a + 2e6])
    rng = np.random.default_rng(3)
    y_nd = rng.normal(size=(4, t_nd.size))
    y_nd[2, 5] = 0.0  # the lid thickness row: handed over at node 5
    scale, offset = np.array([10.0, 3.0, 2.0, 5.0]), np.array([0.0, 0.0, 0.0, 7.0])
    sink = _Collect()
    nodes = st.SolveNodes([sink], t_ref, scale, offset, change_from=3, lid=True, keep=True)
    for tn, yn in zip(t_nd, y_nd.T, strict=True):
        nodes.add(float(tn), yn)
    nodes.close()
    t = np.concatenate([c[0] for c in sink.chunks])
    y = np.concatenate([c[1] for c in sink.chunks], axis=1)
    change = np.concatenate([c[2] for c in sink.chunks], axis=1)
    keep = [0, 1, 3, 4, 5, 6]  # node 2 rounds to the year of nodes 3 and 4
    ref = y_nd[:, keep] * scale[:, None] + offset[:, None]
    ref[-1] = np.where(ref[-2] == 0.0, 0.0, ref[-1])
    np.testing.assert_array_equal(t, t_nd[keep] * t_ref)
    np.testing.assert_array_equal(y, ref)
    np.testing.assert_array_equal(change, y_nd[3:, keep] * scale[3:, None])
    assert y[-1, 4] == 0.0 and y_nd[-1, 5] * 5.0 + 7.0 != 0.0  # the zeroing acted
    assert [c[0].size for c in sink.chunks] == [3, 3] and nodes.n == 6
    (t_kept, y_kept), scaled = nodes.kept()
    np.testing.assert_array_equal(t_kept, t)
    np.testing.assert_array_equal(scaled, y_nd[:, keep] * scale[:, None])


def _ledger_solver(powers):
    """A solver with the energy ledger and ``_step_powers`` at each node, no EOS work."""
    s = es.EntropySolver.__new__(es.EntropySolver)
    s.entropy_eos, s._n_stag, s._core_bc = object(), 1, 'quasi_steady'
    s._r_basic_flat = np.array([1.0, 2.0])
    s.state = SimpleNamespace(_pb_cache_hits=0, _pb_cache_misses=0)
    s._stag_entropy = lambda y: np.asarray(y[:1])
    s._step_heat_content = lambda a, b: 0.0
    s._step_powers = powers
    s._warned = set()
    s.parameters = SimpleNamespace(solver=SimpleNamespace(strict_accepted_range=False))
    return s


def test_chunked_trapezoid_equals_the_exact_integral_of_a_linear_power(monkeypatch):
    """A power linear in time is integrated exactly by the trapezoid: over 1000 uneven nodes
    in 7-node chunks every column equals its integral to 1e-13; dropping the interval between
    two chunks would lose about 142 of the 999 intervals, far outside the tolerance."""
    rate = np.array([1.0, -2.0, 3.0, 0.5, 4.0, -1.0, 1e-3])

    def powers(t, y):
        return rate * (1.0 + t)

    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, 50.0, 1000))
    t[0], t[-1] = 0.0, 50.0
    s = _ledger_solver(powers)
    s._solution = OptimizeResult(t=t, y=np.vstack([t, 2 * t]))
    exact = rate * (50.0 + 0.5 * 50.0**2) * st.SECS_PER_YEAR
    monkeypatch.setattr(st, 'CHUNK', 7)
    out = s._compute_step_energy_integrals()
    keys = ('F_int', 'F_cmb', 'Q_radio', 'Q_tidal', 'Q_radio_cons', 'Q_tidal_cons')
    np.testing.assert_allclose([out[k] for k in keys], exact[:6], rtol=1e-13)
    assert out['solver_residual'] == pytest.approx(exact[6], rel=1e-13)
    # the step-average CMB flux divides by the CMB area and the whole span
    assert out['F_cmb_step_avg'] == pytest.approx(
        exact[1] / (4.0 * np.pi * 50.0 * st.SECS_PER_YEAR), rel=1e-13
    )
    gap = 0.5 * (powers(t[6], None) + powers(t[7], None))[0] * (t[7] - t[6]) * st.SECS_PER_YEAR
    assert abs(gap) > 1e-6 * abs(exact[0])  # one lost interval would show
    monkeypatch.setattr(st, 'CHUNK', 2048)
    whole = s._compute_step_energy_integrals()
    np.testing.assert_allclose([whole[k] for k in keys], [out[k] for k in keys], rtol=1e-14)


def test_a_failed_check_is_raised_after_the_solve_and_stops_the_powers():
    """A power evaluation that raises at node 10 does not raise from ``add`` (a CVODE callback);
    ``result`` raises it, and no later node is evaluated."""
    calls = []

    def powers(t, y):
        calls.append(t)
        if len(calls) == 10:
            raise RuntimeError('lid invariant violated at node 10')
        return np.zeros(7)

    s = _ledger_solver(powers)
    s._solution = sol = OptimizeResult(t=np.array([0.0, 1.0]), y=np.zeros((2, 2)))
    quad = st.EnergyQuadrature(s)
    t = np.linspace(0.0, 1.0, 30)
    for a in range(0, 30, 8):
        quad.add(t[a : a + 8], np.zeros((2, t[a : a + 8].size)), None)
    assert len(calls) == 10 and quad.n == 30
    with pytest.raises(RuntimeError, match='node 10'):
        quad.result(sol)


def test_the_table_guard_finds_the_first_bad_node_across_chunks(monkeypatch):
    """The guard names the first node with a cell entropy outside the tables, in a later chunk,
    and says when the bad node is the call start."""
    monkeypatch.setattr(st, 'CHUNK', 4)
    s = _ledger_solver(None)
    s._n_stag, s.entropy_eos = 2, SimpleNamespace(S_min=100.0, S_max=200.0)
    s._stag_entropy = lambda y: np.asarray(y[:2])
    y = np.full((2, 10), 150.0)
    y[1, 6], y[0, 8] = 300.0, 50.0  # node 6 is the first outside
    guard = st.TableGuard(s)
    st.feed([guard], np.arange(10.0), y)
    assert guard.bad[0] == 6.0 and 'cell 1 entropy 300' in guard.bad[1]
    assert 'call start' not in guard.bad[1]
    y[0, 0] = 99.0  # 1e-3 J/kg/K tolerance: 1 J/kg/K below is outside
    start = st.TableGuard(s)
    st.feed([start], np.arange(10.0), y)
    assert start.bad[0] == 0.0 and 'at the call start' in start.bad[1]


def test_streamed_recording_memory_does_not_grow_with_the_node_count():
    """100,000 nodes of 80 floats (64 MB of states), with an output time every 100 nodes, pass
    the filter and the chunked conversion with a peak below 12 MB; a stored list of (t, y)
    pairs needs about 840 B a node, 84 MB."""
    n_state, n = 80, 100_000
    y = np.linspace(1.0, 2.0, n_state)
    counter = SimpleNamespace(n=0, add=None)

    def count(t, ys, change=None):
        counter.n += t.size

    counter.add = count
    tracemalloc.start()
    try:
        nodes = st.SolveNodes([counter], 1.0, np.ones(n_state), np.zeros(n_state), n_state)
        call = st.CallNodes(nodes, 0.0, y, np.arange(0.0, n + 1.0, 100.0))
        for i in range(1, n):
            call.record(float(i), y)
        call.end(np.array([0.0, float(n)]), np.column_stack([y, y]))
        nodes.close()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert counter.n == n + 1  # every step, the start and the end
    assert peak < 12e6


def test_a_batch_that_fails_in_a_later_chunk_falls_back_from_that_chunk_on(monkeypatch, caplog):
    """The JAX batch raises on the second of three 8-node chunks: that chunk and the third use
    ``_step_powers`` at each node, the first keeps its batch powers, the fallback is logged once,
    and the integral of a power linear in time is still exact."""
    monkeypatch.setattr(st, 'CHUNK', 8)
    rate = np.array([2.0, -1.0, 0.5, 0.25, 3.0, -2.0, 0.0])
    numpy_nodes, batch_calls = [], []

    def powers(t, y):
        numpy_nodes.append(t)
        return rate * (1.0 + t)

    def batch(t, y):
        batch_calls.append(t.size)
        if len(batch_calls) == 2:
            raise RuntimeError('simulated batch failure')
        return np.outer(1.0 + t, rate)

    t = np.linspace(0.0, 10.0, 20)
    s = _ledger_solver(powers)
    s._cvode_step_powers_batch, s._step_powers_batch = object(), batch
    s._batch_powers_calls = 0
    s._solution = OptimizeResult(t=t, y=np.vstack([t, t]))
    with caplog.at_level('WARNING', logger=es.logger.name):
        out = s._compute_step_energy_integrals()
    assert batch_calls == [8, 8]  # the third chunk does not try the batch again
    np.testing.assert_array_equal(numpy_nodes, t[8:])
    assert sum('batch step powers evaluation failed' in r.message for r in caplog.records) == 1
    named = 'steps 0 to 7 of the call had the batch checks only'
    assert sum(named in r.message for r in caplog.records) == 1
    assert s._batch_powers_calls == 1
    exact = rate[0] * (10.0 + 0.5 * 10.0**2) * st.SECS_PER_YEAR
    assert out['F_int'] == pytest.approx(exact, rel=1e-13)


def test_a_failed_call_passes_no_node_past_its_last_returned_output():
    """CVODE returns the output time 1 and then fails after 59 steps toward 2: those steps reach
    the sink neither during the call nor at its end, as with the stored list (the base keeps
    the nodes up to the returned end); a joined start still counts as the call start."""
    sink = st.ListSink()
    call = st.CallNodes(sink, 0.0, np.zeros(1), [0.0, 1.0, 2.0])
    steps = np.r_[0.4, 0.8, 1.1, 1.1 + np.cumsum(np.full(59, 0.01))]
    for tn in steps:
        call.record(tn, np.array([tn]))
        assert sink.t[-1] <= 1.0
    assert sink.t == [0.0, 0.4, 0.8]  # 1.1 waits: CVODE has not passed a later step
    call.end(np.array([0.0, 1.0]), np.array([[0.0, 1.0]]))
    np.testing.assert_array_equal(sink.arrays()[0], [0.0, 0.4, 0.8, 1.0])
    assert call.count == 4 and call.recorded == steps.size


def test_nodes_at_a_waiting_time_are_dropped_so_the_wait_stays_bounded():
    """10,000 root-function calls at the time of a waiting node never enter the wait list, since
    the stored filter would never keep them; a time below the last passed node is dropped too."""
    sink = st.ListSink()
    call = st.CallNodes(sink, 0.0, np.zeros(1), [0.0, 5.0])
    call.record(1.0, np.ones(1))
    for _ in range(10_000):
        call.record(1.0, np.full(1, 2.0))
    call.record(0.0, np.ones(1))
    assert len(call._wait) == 1 and call.recorded == 10_002
    call.end(np.array([0.0, 5.0]), np.array([[0.0, 3.0]]))
    np.testing.assert_array_equal(sink.arrays()[0], [0.0, 1.0, 5.0])
    np.testing.assert_array_equal(sink.arrays()[1], [[0.0, 1.0, 3.0]])


def test_the_floor_minimum_of_a_later_chunk_and_of_an_earlier_one_survive_the_chunks(
    monkeypatch,
):
    """The floor entropy is lowest in chunk 1 and the cell entropy lowest in chunk 3: both 4-node
    chunks and one chunk give the minimum of the two; a minimum that the last chunk overwrote
    would be 450, outside any tolerance."""
    s = SimpleNamespace(_n_stag=2)
    w = np.array([[0.5, 0.5], [1.0, 0.0]])
    y = np.full((3, 11), 500.0)
    y[:2, 1] = [430.0, 440.0]  # floor entropies 435 and 430 at node 1
    y[1, 9] = 420.0  # a cell at 420 in chunk 3; its floor entropy is 460
    y[:2, 10] = 450.0
    lows = []
    for chunk in (4, 2048):
        monkeypatch.setattr(st, 'CHUNK', chunk)
        floor = st.FloorMin(s, w)
        st.feed([floor], np.arange(11.0), y)
        lows.append(floor.low)
    assert lows == [420.0, 420.0]
    y[1, 9] = 500.0
    floor = st.FloorMin(s, w)
    monkeypatch.setattr(st, 'CHUNK', 4)
    st.feed([floor], np.arange(11.0), y)
    assert floor.low == 430.0 and floor.low < 450.0


@pytest.mark.parametrize('strict', [False, True])
def test_the_strict_table_guard_keeps_the_first_bad_node_of_a_later_chunk(monkeypatch, strict):
    """With ``strict_accepted_range`` the guard keeps an ``AcceptedStateRangeError`` for the
    first bad node (node 6, chunk 2) with the message of the failure; off, it keeps none."""
    monkeypatch.setattr(st, 'CHUNK', 4)
    s = _ledger_solver(None)
    s._n_stag, s.entropy_eos = 2, SimpleNamespace(S_min=100.0, S_max=200.0)
    s.parameters = SimpleNamespace(solver=SimpleNamespace(strict_accepted_range=strict))
    s._stag_entropy = lambda y: np.asarray(y[:2])
    y = np.full((2, 12), 150.0)
    y[0, 6], y[1, 10] = 250.0, 20.0
    guard = st.TableGuard(s)
    st.feed([guard], np.arange(12.0), y)
    assert guard.bad[0] == 6.0
    if strict:
        assert isinstance(guard.error, es.AcceptedStateRangeError)
        assert str(guard.error) == guard.bad[1] and 'cell 0 entropy 250' in guard.bad[1]
    else:
        assert guard.error is None


def test_the_regime_check_keeps_the_first_refused_state_of_a_later_chunk(monkeypatch):
    """The regime refusal of the T_core row (stand-in raising below 2000 K) fires first in chunk
    2 at node 5; the error of chunk 3 (node 9) does not replace it."""
    import aragog.core.regime as regime

    def refuse(budget, t_cmb):
        cold = np.flatnonzero(np.asarray(t_cmb) < 2000.0)
        if cold.size:
            raise ValueError(f'refused at T_core = {t_cmb[cold[0]]:.1f} K')

    monkeypatch.setattr(regime, 'refuse_unmodelled_regime', refuse)
    monkeypatch.setattr(st, 'CHUNK', 4)
    y = np.full((3, 12), 3000.0)
    y[2, 5], y[2, 9] = 1900.0, 1500.0
    check = st.RegimeCheck(SimpleNamespace(_n_stag=1), budget=None)
    st.feed([check], np.arange(12.0), y)
    assert isinstance(check.error, ValueError) and '1900.0 K' in str(check.error)


def _core_solver(stratified):
    """A core_module solver stub: C_eff linear in T_core, the real tolerance scale."""
    s = _ledger_solver(lambda t, y: np.zeros(7))
    s._core_bc, s._n_stag, s._call_tol = 'core_module', 1, (1e-8, 1e-8)
    shell = SimpleNamespace(n_cells=0, heat_change=lambda d: 0.0)
    s._core_module_budget = SimpleNamespace(
        stratification=stratified, shell=shell, heat_change=lambda t0, d: 7.0e26 * d
    )
    s._core_capacity = lambda budget, y: 1.0e27 * (1.0 + 1e-4 * y[2])
    s._core_shell = lambda: None
    s._cmb_node_phase = None
    s.cmb_ledger, s._cmb_ledger_counts = None, {'missed': 0, 'excluded': 0}
    return s


@pytest.mark.physics_invariant
@pytest.mark.parametrize('stratified', [False, True])
def test_core_heat_and_its_tolerance_scale_equal_the_one_chunk_values(monkeypatch, stratified):
    """Over 11 nodes of a falling T_core the core heat change and its tolerance scale with 4-node
    chunks equal the whole-trace formula ``sum_k C_k (atol + rtol |d_k - d_0|)`` to 1e-14; a
    chunk that counted its carried node again would add at least 1e27 atol = 1e19 J per
    chunk boundary, 1e-3 of the scale."""
    t = np.linspace(0.0, 10.0, 11)
    T = 4000.0 - 3.0 * t**1.5
    y = np.vstack([np.full(11, 3000.0), np.zeros(11), T])
    change = (T - 4000.0)[None, :]
    out = []
    for chunk in (4, 2048):
        monkeypatch.setattr(st, 'CHUNK', chunk)
        s = _core_solver(stratified)
        s._solution = OptimizeResult(t=t, y=y)
        quad = st.EnergyQuadrature(s)
        st.feed([quad], t, y, change)
        out.append(quad.result(s._solution))
    d = change[0] - change[0, 0]
    c = 1.0e27 * (1.0 + 1e-4 * T)
    tol = float(np.sum(c[1:] * (1e-8 + 1e-8 * np.abs(d[1:]))))
    assert out[0]['core_tol'] == pytest.approx(tol, rel=1e-14)
    assert out[1]['core_tol'] == pytest.approx(tol, rel=1e-14)
    assert 1e27 * 1e-8 > 1e-4 * tol  # one recounted boundary node is far outside 1e-14
    heat = float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(d))) if stratified else 7e26 * d[-1]
    assert out[0]['core'] == pytest.approx(heat, rel=1e-13) and out[0]['core'] < 0.0


class _Blend:
    """CMB node blend stand-in: T = S^2 / 1000 with the slope ``2 S / 1000 (1 + bump(S))``."""

    def __init__(self, bump):
        self.bump = bump

    def set_pressure(self, P):
        self.P = P

    def set_entropy(self, S):
        self.S = np.asarray(S, dtype=float)

    def update(self):
        pass

    def temperature(self):
        return self.S**2 / 1000.0

    def temperature_slope(self):
        return 2.0 * self.S / 1000.0 * (1.0 + self.bump(self.S))


def _cmb_solver(bump, core_bc='energy_balance'):
    s = _ledger_solver(None)
    s._core_bc, s._n_stag, s._call_tol = core_bc, 2, (1e-8, 1e-8)
    s._cmb_node_phase = _Blend(bump)
    s._cmb_dr_cmb, s._P_basic_flat = 2.0e4, np.array([1.3e11])
    s._core_cp, s._core_tfac, s._core_M = 800.0, 1.1, 1.9e24
    s.cmb_ledger, s._cmb_ledger_counts = None, {'missed': 0, 'excluded': 0}
    return s


def _cmb_nodes(n):
    """Nodes of a cooling call: the CMB node entropy ``S0 + 10`` and its temperature change."""
    t = np.linspace(0.0, 30.0, n) ** 1.2
    S0 = 3000.0 - 2.0 * t
    y = np.vstack([S0, S0 + 5.0, np.full(n, -1e-3)])
    p = 1e16 * (1.0 + 0.3 * np.sin(t))
    T = (S0 + 10.0) ** 2 / 1000.0
    return t, y, p, T - T[0]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('core', [False, True])
@pytest.mark.parametrize('n', [10, 11])
def test_cmb_ledger_terms_of_4_node_chunks_equal_the_whole_trace(n, core):
    """The ledger terms of 4-node chunks equal those of the whole trace (``_check_cmb_ledger``
    on the arrays) for an even and an odd node count: the largest slope error ``rel`` sits in
    the last chunk, and a ``rel`` kept from the first chunks only would be 0.002 against 0.038."""
    t, y, p, dT = _cmb_nodes(n)
    S_late = y[0, -2] - 0.5 * 2.0e4 * y[2, -2]
    bump = lambda S: np.where(np.abs(S - S_late) < 1e-6, 0.04, 0.002)  # noqa: E731
    whole = _cmb_solver(bump, 'core_module' if core else 'energy_balance')
    # the core cools by the heat the CMB carries out, so the ledger closes to round-off
    q = -800.0 * 1.1 * 1.9e24 * dT[-1]
    d = dT if core else None
    whole._check_cmb_ledger(t, y, p, q, d_top=d)
    parts = _cmb_solver(bump, 'core_module' if core else 'energy_balance')
    ledger = st.CMBLedger(parts, core=core)
    for a in range(0, n, 4):
        sl = slice(a, a + 4)
        ledger.add(t[sl], y[:, sl], p=p[sl], d=None if d is None else d[sl])
    parts._check_cmb_ledger(q=q, ledger=ledger)
    ref, got = whole.cmb_ledger, parts.cmb_ledger
    assert got['rel'] == ref['rel'] == pytest.approx(0.04 / 1.04, rel=1e-6)
    assert 0.002 / 1.002 < 0.1 * got['rel']  # the first chunks alone
    assert got['excluded'] is ref['excluded'] is False
    assert got['r'] == pytest.approx(ref['r'], rel=1e-14, abs=1e-30)
    assert got['bound'] == pytest.approx(ref['bound'], rel=1e-14)


def test_a_cmb_node_slope_fault_of_a_later_chunk_is_raised_with_its_time():
    """A node slope of 0 at node 9 (chunk 3) is kept and raised by the check at the end, naming
    that node's time, as the whole-trace check does; the same fault in a call that does not
    advance is not judged."""
    t, y, p, _ = _cmb_nodes(11)
    S9 = y[0, 9] - 0.5 * 2.0e4 * y[2, 9]
    bump = lambda S: np.where(np.abs(S - S9) < 1e-6, -1.0, 0.0)  # noqa: E731
    ledger = st.CMBLedger(_cmb_solver(bump))
    for a in range(0, 11, 4):
        ledger.add(t[a : a + 4], y[:, a : a + 4], p=p[a : a + 4])
    at = re.escape(f'slope (dT/dS)_P = 0 at t = {t[9]:.6e} yr')
    with pytest.raises(es.CMBClosureError, match=at):
        ledger.s._check_cmb_ledger(q=0.0, ledger=ledger)
    with pytest.raises(es.CMBClosureError, match=at):
        _cmb_solver(bump)._check_cmb_ledger(t, y, p, 0.0)
    still = _cmb_solver(bump)
    still._check_cmb_ledger(np.zeros(11), y, p, 0.0)
    assert still.cmb_ledger is None


class _LidSolver(es.EntropySolver):
    """A lid solver stub for the batch checks: every check reads the arrays it is given."""

    _n_lid = 2
    _rheology_enabled = True


def _lid_batch_solver(rheo_bad=None, weights=None):
    s = _LidSolver.__new__(_LidSolver)
    s.entropy_eos, s._n_stag, s._core_bc = object(), 2, 'quasi_steady'
    s._r_basic_flat = np.array([1.0, 2.0, 3.0])
    s.state = SimpleNamespace(_pb_cache_hits=0, _pb_cache_misses=0)
    s._stag_entropy = lambda y: np.asarray(y[:2])
    s._step_heat_content = lambda a, b: 0.0
    s._lid_heat_change = lambda a, b: 0.0
    s._warned, s._batch_powers_calls = set(), 0
    s._cvode_step_powers_batch = object()
    s._step_powers_batch = lambda t, y: np.zeros((t.size, 7))
    s._lid_weights_of = weights or (lambda t, y: np.zeros(t.size))
    s._dSdt_single = lambda t, y: None
    s._check_lid_step = lambda lo, k, t: None
    s._check_rheology_step_invariants = lambda rheo, step_idx, time_yr: None
    s._solid_rheology = lambda: None
    s._lid_active, s._lid_mush = True, True
    s._lid_h_top, s._lid_delta_lo, s._lid_S_sol = 1.0e4, 10.0, 2000.0
    s._lid_const, s._lid_W_of_S = False, SimpleNamespace(x=np.array([1000.0, 4000.0]))
    s._lid_lo_bound = lambda lo: lo - 1.0

    def rheo_range(S, step_idx=0, time_yr=None):
        if rheo_bad is not None and step_idx == rheo_bad:
            raise es.AcceptedStateRangeError(f'rheology invariant violated at step {step_idx}')

    s._check_rheology_eos_range = rheo_range
    return s


def _lid_nodes(n=80):
    t = np.linspace(0.0, 1.0, n)
    y = np.vstack([np.full(n, 3000.0), np.full(n, 3000.0), np.full(n, 100.0), np.ones(n)])
    return t, y


@pytest.mark.parametrize('chunk', [4, 2048])
@pytest.mark.parametrize('rheo_bad', [None, 50])
def test_the_checks_raise_in_the_order_of_the_stored_trace(monkeypatch, chunk, rheo_bad):
    """A thickness fault at step 1 and a floor fault at step 69: the stored trace raised the
    floor fault, since it checks the floor of every node before the thickness, and so do
    4-node chunks. With an EOS range fault at step 50 both raise the
    ``AcceptedStateRangeError``, which PROTEUS retries with a half step, not the lid
    ``RuntimeError``."""
    monkeypatch.setattr(st, 'CHUNK', chunk)
    s = _lid_batch_solver(rheo_bad)
    t, y = _lid_nodes()
    y[2, 1] = 5.0  # thinner than the stop
    y[1, 69] = 1990.0  # top cell below the solidus floor
    s._solution = OptimizeResult(t=t, y=y)
    want = (es.AcceptedStateRangeError, 'step 50') if rheo_bad else (RuntimeError, 'step 69')
    with pytest.raises(want[0], match=want[1]):
        s._compute_step_energy_integrals()


@pytest.mark.parametrize('chunk', [4, 2048])
def test_the_largest_lid_weight_of_a_later_chunk_is_kept(monkeypatch, chunk):
    """The lid weight is largest (0.93) at node 70, in a later chunk than the next largest
    (0.6 at node 2): both chunkings report 0.93, and a value that a later chunk overwrote
    would be 0.1."""
    monkeypatch.setattr(st, 'CHUNK', chunk)
    t, y = _lid_nodes()
    w = np.full(t.size, 0.1)
    w[2], w[70] = 0.6, 0.93
    s = _lid_batch_solver(weights=lambda tc, yc: w[np.searchsorted(t, tc)])
    s._solution = sol = OptimizeResult(t=t, y=y)
    s._compute_step_energy_integrals()
    cap = st._lid.LID_CAP_FRAC * 1.0e4
    assert sol.lid_weight_max == 0.93
    assert sol.lid_cap_excess == pytest.approx(100.0 / cap - 1.0, rel=1e-12)
