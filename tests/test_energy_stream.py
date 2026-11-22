"""``aragog.solver.energy_stream``: the per-call energy integrals and node checks in chunks.

Checks that the node filter of a CVODE call equals the stored-list filter across the chunk and
tail boundaries, including a root step whose end lies past the call end; that the joined nodes
of a solve are deduplicated and converted as the stored trace was; that the chunked trapezoid
equals the analytic integral of a linear power and does not drop the interval between chunks;
that a batch failure in a later chunk falls back to the per-node powers from that chunk on;
that a failed check is raised after the solve, not inside a callback; and that the memory of
the recording does not grow with the node count.
"""

from __future__ import annotations

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
    it. Returns the times and the call end."""
    ends = t0 + np.cumsum(rng.uniform(1e-3, 4e-3, n))
    times = []
    for k, te in enumerate(ends):
        times.append(float(te))
        if k % 7 == 3:  # an output point inside the step just taken
            times.append(float(te - 0.3 * (te - (ends[k - 1] if k else t0))))
    if root:
        t_end = float(ends[-1] + 2e-3)
        times.append(t_end + 1.5e-3)  # the step that crosses the root
        times += [t_end + 7e-4, t_end - 4e-4, t_end + 2e-4, t_end - 1e-4, t_end]
    else:
        t_end = float(ends[-1] + 1e-3)
        times.append(t_end + 2e-3)  # CVODE steps past the call end and interpolates back
        times.append(t_end)
    return times, t_end


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
        times, t_end = _cvode_like_nodes(rng, t0, 300, root)
        nodes = [(tn, rng.normal(size=3)) for tn in times]
        y0 = rng.normal(size=3) if start else y_end
        t = np.array([t0, t_end])
        y = np.column_stack([y0, rng.normal(size=3)])
        call = st.CallNodes(sink, t0, y0, start=start)
        for tn, yn in nodes:
            call.record(tn, yn)
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
    call = st.CallNodes(sink, 2.0, np.zeros(2), start=False)
    call.end(np.array([2.0, 3.0]), np.array([[0.0, 1.0], [0.0, 2.0]]))
    t, y = sink.arrays()
    np.testing.assert_array_equal(t, [3.0])
    np.testing.assert_array_equal(y, [[1.0], [2.0]])
    empty = st.CallNodes(sink, 3.0, np.zeros(2), start=False)
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
    """100,000 nodes of 80 floats (64 MB of states) pass the filter and the chunked conversion
    with a peak below 12 MB; a stored list of (t, y) pairs needs about 840 B a node, 84 MB."""
    n_state, n = 80, 100_000
    y = np.linspace(1.0, 2.0, n_state)
    counter = SimpleNamespace(n=0, add=None)

    def count(t, ys, change=None):
        counter.n += t.size

    counter.add = count
    tracemalloc.start()
    try:
        nodes = st.SolveNodes([counter], 1.0, np.ones(n_state), np.zeros(n_state), n_state)
        call = st.CallNodes(nodes, 0.0, y)
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
    assert s._batch_powers_calls == 1
    exact = rate[0] * (10.0 + 0.5 * 10.0**2) * st.SECS_PER_YEAR
    assert out['F_int'] == pytest.approx(exact, rel=1e-13)
