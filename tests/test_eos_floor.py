"""The EOS floor: a root ends a call before the surface entropy leaves the entropy tables."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

from aragog.solver import entropy_solver as es
from aragog.solver.entropy_solver import (
    EntropySolver,
    _floor_entropy,
    _PhaseBoundarySegmentRoot,
    _PhiCapRootFunction,
)
from tests.conftest import entropy_eos_copy

from .conftest import EOS_DIR, needs_eos
from .test_phi_step_cap_armed_smoke import _build_mushy_parameters

pytestmark = [pytest.mark.smoke, needs_eos]

_FLUX = 1.0e6  # prescribed surface flux [W/m^2]; drains the top cell of the cold start in ~3 yr


@pytest.fixture(scope='module')
def shared_eos():
    return entropy_eos_copy(EOS_DIR)


def _drained(eos, S0, *, margin=100.0, method='cvode', end_time=10.0):
    """A cold uniform start at entropy ``S0`` under a prescribed outgoing surface flux."""
    p = _build_mushy_parameters(solver_method=method, end_time=end_time)
    p.boundary_conditions.outer_boundary_condition = 4
    p.boundary_conditions.outer_boundary_value = _FLUX
    p.energy = dataclasses.replace(p.energy, phi_step_cap=None, eos_floor_margin=margin)
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(np.full(s._n_stag, S0))
    s.solve()
    w = np.asarray(s.evaluator.mesh.quantity_at_basic_nodes(np.eye(s._n_stag)))[-1]
    S = np.asarray(s._solution.y, dtype=float)[: s._n_stag]
    return s, np.array([_floor_entropy(w, S[:, k]) for k in range(S.shape[1])])


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


def test_floor_level_is_halfway_when_the_start_is_inside_twice_the_margin(shared_eos):
    """A start at S_min + 50 stops at S_min + 25, not at once or past the table edge."""
    S_min = float(shared_eos.S_min)
    s, f = _drained(shared_eos, S_min + 50.0)
    assert s._solution.cap_label == 'eos_floor' and 0.0 < s._solution.t[-1] < 10.0
    assert f[-1] == pytest.approx(S_min + 25.0, abs=1.0e-3)


@pytest.mark.parametrize('at', [False, True])
def test_floor_refuses_a_start_at_or_below_the_table(shared_eos, caplog, monkeypatch, at):
    """Below S_min, or exactly at it (the floor entropy pinned to S_min), the call does not
    advance: status 1, dt_actual 0, one named warning."""
    S_min = float(shared_eos.S_min)
    if at:
        monkeypatch.setattr(es, '_floor_entropy', lambda w, S: S_min)
    with caplog.at_level(logging.WARNING):
        s, _ = _drained(shared_eos, S_min + 400.0 if at else S_min - 10.0)
    sol = s._solution
    assert sol.status == 1 and sol.t[-1] == sol.t[0] == 0.0
    assert s.get_state().dt_actual == 0.0
    assert any('the call does not advance' in r.getMessage() for r in caplog.records)


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
