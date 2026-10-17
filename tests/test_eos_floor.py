"""The EOS floor: a root ends a call before the surface entropy leaves the entropy tables."""

from __future__ import annotations

import dataclasses
import logging
import warnings

import numpy as np
import pytest

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


@pytest.fixture(scope='module')
def shared_eos():
    return entropy_eos_copy(EOS_DIR)


def _floor_series(s):
    """The floor entropy at every output point of the last solve."""
    w = np.asarray(s.evaluator.mesh.quantity_at_basic_nodes(np.eye(s._n_stag)))[-1]
    S = np.asarray(s._solution.y, dtype=float)[: s._n_stag]
    return np.array([_floor_entropy(w, S[:, k]) for k in range(S.shape[1])])


def _drained(
    eos, S0, *, margin=100.0, method='cvode', end_time=10.0, core='quasi_steady', mode=None
):
    """A cold uniform start at entropy ``S0`` under a prescribed outgoing surface flux."""
    p = _build_mushy_parameters(solver_method=method, end_time=end_time)
    p.boundary_conditions.outer_boundary_condition = 4
    p.boundary_conditions.outer_boundary_value = _FLUX
    p.boundary_conditions.core_bc = core
    p.energy = dataclasses.replace(
        p.energy, phi_step_cap=None, eos_floor_margin=margin, phase_boundary_cap=mode
    )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(np.full(s._n_stag, S0))
    s.solve()
    return s, _floor_series(s)


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_floor_root_ends_the_call_at_the_margin(shared_eos, method):
    """CVODE (status 0) and scipy (status 1) stop at S_min + 100 before the call end, and the
    surface entropy never leaves the table; without the floor the same call drains below it."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + 400.0, method=method)
    sol = s._solution
    assert sol.status == (0 if method == 'cvode' else 1)
    assert 0.0 < sol.t[-1] < 10.0
    assert (
        f[-1] == pytest.approx(S_min + 100.0, abs=1.0e-3) and f.min() >= S_min + 100.0 - 1.0e-3
    )
    if method == 'cvode':
        assert sol.cap_label == 'eos_floor' and sol.cap_value == S_min + 100.0
    s_off, f_off = _drained(shared_eos, S_min + 400.0, margin=None, method=method)
    assert s_off._solution.t[-1] == pytest.approx(10.0) and f_off[-1] < S_min


@pytest.mark.parametrize(
    ('core', 'mode'), [('energy_balance', 'fixed'), ('quasi_steady', 'rate')]
)
def test_floor_root_stops_the_energy_balance_core_and_rate_segments(shared_eos, core, mode):
    """The energy_balance core and the rate-cap segments stop at S_min + 100 as well."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + 400.0, core=core, mode=mode)
    assert s._solution.cap_label == 'eos_floor' and 0.0 < s._solution.t[-1] < 10.0
    assert f[-1] == pytest.approx(S_min + 100.0, abs=1.0e-3)


def test_floor_stops_a_start_inside_the_margin_at_half_the_margin(shared_eos):
    """A start at S_min + 80 stops at S_min + 50, the second and last level."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + 80.0)
    assert s._solution.cap_label == 'eos_floor' and 0.0 < s._solution.t[-1] < 10.0
    assert f[-1] == pytest.approx(S_min + 50.0, abs=1.0e-3)


def test_a_persistent_drain_ends_after_two_floor_stops(shared_eos):
    """Calls chained under the same drain stop at S_min + 100, then S_min + 50, then are refused."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + 400.0)
    ends, t = [f[-1]], float(s._solution.t[-1])
    for _ in range(3):
        S = np.array(s._solution.y[: s._n_stag, -1])
        s.parameters.solver.start_time, s.parameters.solver.end_time = t, t + 10.0
        s.reset()
        s.set_initial_entropy(S)
        s.solve()
        ends.append(_floor_series(s)[-1])
        t = float(s._solution.t[-1])
    assert ends[0] == pytest.approx(S_min + 100.0, abs=1.0e-3)
    assert ends[1] == pytest.approx(S_min + 50.0, abs=1.0e-3)
    assert ends[2] == ends[1] and ends[3] == ends[1] and s._solution.status == 1


@pytest.mark.parametrize('where', ['below', 'one ulp above', 'at half the margin'])
@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_floor_refuses_a_start_at_or_below_half_the_margin(shared_eos, caplog, method, where):
    """Below S_min, 1 ulp above it, or at S_min + 50 the call does not advance: status 1,
    dt_actual 0, one named warning, and the log names the full requested span."""
    S_min = float(shared_eos.S_min)
    S0 = {'below': S_min - 10.0, 'one ulp above': np.nextafter(S_min, np.inf)}.get(
        where, S_min + 50.0
    )
    with caplog.at_level(logging.INFO):
        s, _ = _drained(shared_eos, S0, method=method)
    sol = s._solution
    assert sol.status == 1 and sol.t[-1] == sol.t[0] == 0.0
    assert s.get_state().dt_actual == 0.0
    messages = [r.getMessage() for r in caplog.records]
    assert sum('the call does not advance' in m for m in messages) == 1
    assert any('stopped 10.0 yr before end_time' in m for m in messages)


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
@pytest.mark.parametrize('mode', ['fixed', 'rate'])
@pytest.mark.parametrize('core', ['quasi_steady', 'energy_balance'])
def test_floor_on_and_off_are_bit_equal_when_it_does_not_fire(shared_eos, method, mode, core):
    """A call that never comes near the table edge takes the same steps with and without the
    floor: output times, states and energy integrals are identical."""
    runs = []
    for margin in (100.0, None):
        p = _build_mushy_parameters(solver_method=method, end_time=2.0)
        p.boundary_conditions.core_bc = core
        p.energy = dataclasses.replace(
            p.energy, eos_floor_margin=margin, phase_boundary_cap=mode
        )
        s = EntropySolver(p, entropy_eos=shared_eos)
        s.initialize()
        s.set_initial_entropy(_pick_mushy_S(shared_eos))
        s.solve()
        runs.append(s._solution)
    on, off = runs
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


def test_segment_root_names_the_floor_as_the_cap():
    """At a floor root the segment roots report 'cap', which ends a rate-cap call."""
    w = np.array([-0.5, 1.5])
    cap = _PhiCapRootFunction(
        eos=None, P_stag=np.zeros(2), volume=np.ones(2), n_stag=2, phi0_global=0.0, cap=0.0,
        state_scale=np.ones(2), floor_weights=w, floor_level=200.0, floor_margin=100.0,
    )  # fmt: skip
    y0 = np.array([5000.0, 5000.0])
    roots = _PhaseBoundarySegmentRoot(
        np.full(3, 6000.0), np.full(3, 4000.0), y0, 10.0, None, np.ones(2), 2, cap=cap
    )
    assert roots.names[0] == 'cap'
    assert roots.fired(0.0, np.array([250.0, 200.0])) == 'cap'
    bad = _PhiCapRootFunction(
        eos=None, P_stag=np.zeros(2), volume=np.ones(2), n_stag=2, phi0_global=0.0, cap=0.0,
        state_scale=np.ones(2), floor_weights=np.ones(3), floor_level=200.0, floor_margin=100.0,
    )  # fmt: skip
    roots = _PhaseBoundarySegmentRoot(
        np.full(3, 6000.0), np.full(3, 4000.0), y0, 10.0, None, np.ones(2), 2, cap=bad
    )
    with warnings.catch_warnings():
        warnings.simplefilter(
            'error'
        )  # a floor-only cap whose evaluate raised scales by the margin
        assert roots.fired(0.0, np.array([5000.0, 5000.0])) in roots.names
    assert bad.binding_cap is None
