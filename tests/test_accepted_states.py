"""Accepted states of a call stay finite and inside the EOS tables, or the call raises."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.solver.entropy_solver import AcceptedStateError, _check_accepted_states

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos
from .test_eos_floor import _FLUX, _drained

S_MIN, S_MAX = 100.0, 3000.0
W = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, -0.5, 1.5]])
T = np.array([0.0, 1.0, 2.0, 3.0])  # node times [yr]; the first is the call start


def _check(S):
    _check_accepted_states(T, np.asarray(S, float), W, S_MIN, S_MAX)


def _inside():
    return np.full((3, 4), 1000.0)


@pytest.mark.unit
def test_states_inside_the_tables_pass():
    _check(_inside())


@pytest.mark.unit
@pytest.mark.parametrize(
    ('cell', 'value', 'bound'), [(1, S_MIN - 1.0, S_MIN), (2, S_MAX + 1.0, S_MAX)]
)
def test_a_cell_outside_the_table_names_the_cell_time_value_and_bound(cell, value, bound):
    S = _inside()
    S[cell, 2] = value
    with pytest.raises(AcceptedStateError) as e:
        _check(S)
    err = e.value
    assert (err.check, err.node, err.time, err.value, err.bound) == (
        'entropy_table',
        f'cell {cell}',
        2.0,
        value,
        bound,
    )
    assert isinstance(err, ValueError)


@pytest.mark.unit
def test_a_basic_node_below_the_table_raises_with_every_cell_inside():
    """The top basic node extrapolates the top two cells, so it leaves the table first."""
    S = _inside()
    S[1, 3], S[2, 3] = S_MIN + 3.0, S_MIN + 0.5
    with pytest.raises(AcceptedStateError) as e:
        _check(S)
    assert (e.value.check, e.value.node, e.value.time) == ('entropy_table', 'basic 3', 3.0)
    assert e.value.value == pytest.approx(S_MIN - 0.75)


@pytest.mark.unit
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_a_non_finite_state_raises(bad):
    S = _inside()
    S[0, 1] = bad
    with pytest.raises(AcceptedStateError) as e:
        _check(S)
    assert (e.value.check, e.value.node, e.value.time) == ('finite_state', 'cell 0', 1.0)


@pytest.mark.unit
def test_the_call_start_is_not_checked():
    """The first node is the previous call's end, checked there."""
    S = _inside()
    S[0, 0] = S_MIN - 50.0
    _check(S)


@pytest.mark.unit
def test_the_earliest_offending_state_is_reported():
    S = _inside()
    S[2, 3], S[0, 2] = S_MIN - 9.0, S_MAX + 9.0
    with pytest.raises(AcceptedStateError) as e:
        _check(S)
    assert (e.value.node, e.value.time) == ('cell 0', 2.0)


@pytest.mark.unit
@pytest.mark.parametrize('edge', [S_MIN, S_MAX])
def test_the_table_edge_itself_is_inside(edge):
    S = _inside()
    S[:, 1:] = edge
    _check(S)


@pytest.mark.unit
@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(S_MIN, S_MAX), min_size=9, max_size=9),
    st.integers(0, 2),
    st.integers(1, 3),
    st.floats(1e-9, 1e6),
    st.booleans(),
)
def test_a_single_entry_outside_raises_at_its_time(values, cell, col, step, low):
    """Inside states never raise under interpolating weights; one entry pushed out raises."""
    w = W[:3]
    S = np.vstack([np.full(3, 1000.0), np.reshape(values, (3, 3)).T]).T
    _check_accepted_states(T, S, w, S_MIN, S_MAX)
    S[cell, col] = S_MIN - step if low else S_MAX + step
    with pytest.raises(AcceptedStateError) as e:
        _check_accepted_states(T, S, w, S_MIN, S_MAX)
    assert e.value.time == T[col] and e.value.check == 'entropy_table'


@pytest.fixture(scope='module')
def shared_eos():
    return entropy_eos_copy(EOS_DIR)


@pytest.mark.smoke
@needs_eos
@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_drain_without_the_floor_raises_instead_of_clamping(shared_eos, method):
    S_min = float(shared_eos.S_min)
    with pytest.raises(AcceptedStateError) as e:
        _drained(shared_eos, S_min + 400.0, margin=None, method=method)
    assert e.value.check == 'entropy_table' and e.value.bound == S_min
    assert e.value.value < S_min and 0.0 < e.value.time <= 10.0


@pytest.mark.smoke
@needs_eos
@pytest.mark.parametrize('method', ['cvode', 'bdf'])
@pytest.mark.parametrize('flux', [_FLUX, 0.0])
def test_a_call_inside_the_tables_does_not_raise(shared_eos, method, flux):
    """The floor keeps a drained call inside; a call without the drain stays inside."""
    s, f = _drained(shared_eos, float(shared_eos.S_min) + 400.0, flux=flux, method=method)
    assert f.min() >= float(shared_eos.S_min)
