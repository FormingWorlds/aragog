"""Accepted states of a call stay finite and inside the EOS tables, or the call raises."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.solver.entropy_solver import (
    AcceptedStateError,
    _check_accepted_states,
    _check_end_state,
)

pytestmark = pytest.mark.unit

S_MIN, S_MAX = 100.0, 3000.0
W = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, -0.5, 1.5]])
T = np.array([0.0, 1.0, 2.0, 3.0])  # node times [yr]; the first is the call start


def _check(S):
    _check_accepted_states(T, np.asarray(S, float), W, S_MIN, S_MAX)


def _passes(t, S):
    """True when the states pass, so a wrong raise fails an assertion."""
    try:
        _check_accepted_states(t, np.asarray(S, float), W, S_MIN, S_MAX)
    except AcceptedStateError:
        return False
    return True


def _inside():
    return np.full((3, 4), 1000.0)


def test_states_inside_the_tables_pass():
    assert _passes(T, _inside())


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


def test_a_basic_node_below_the_table_raises_with_every_cell_inside():
    """The top basic node extrapolates the top two cells, so it leaves the table first."""
    S = _inside()
    S[1, 3], S[2, 3] = S_MIN + 3.0, S_MIN + 0.5
    with pytest.raises(AcceptedStateError) as e:
        _check(S)
    assert (e.value.check, e.value.node, e.value.time) == ('entropy_table', 'basic 3', 3.0)
    assert e.value.value == pytest.approx(S_MIN - 0.75)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_a_non_finite_state_raises(bad):
    S = _inside()
    S[0, 1] = bad
    with pytest.raises(AcceptedStateError) as e:
        _check(S)
    assert (e.value.check, e.value.node, e.value.time) == ('finite_state', 'cell 0', 1.0)


def test_the_call_start_is_not_checked():
    """The first node is the previous call's end, checked there."""
    S = _inside()
    S[0, 0] = S_MIN - 50.0
    assert _passes(T, S)


def test_nodes_at_the_call_start_time_are_not_checked():
    """A zero-span call returns its start state twice; neither copy is checked."""
    S = np.full((3, 2), S_MIN - 10.0)
    assert _passes(np.zeros(2), S)
    S = np.hstack([S, np.full((3, 1), S_MIN - 10.0)])
    with pytest.raises(AcceptedStateError) as e:
        _check_accepted_states(np.array([0.0, 0.0, 1.0]), S, W, S_MIN, S_MAX)
    assert e.value.time == 1.0


def test_the_earliest_offending_state_is_reported():
    S = _inside()
    S[2, 3], S[0, 2] = S_MIN - 9.0, S_MAX + 9.0
    with pytest.raises(AcceptedStateError) as e:
        _check(S)
    assert (e.value.node, e.value.time) == ('cell 0', 2.0)


@pytest.mark.parametrize('edge', [S_MIN, S_MAX])
def test_the_table_edge_itself_is_inside(edge):
    S = _inside()
    S[:, 1:] = edge
    assert _passes(T, S)


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


def test_an_end_state_with_positive_temperatures_and_melt_fractions_in_0_1_passes():
    _check_end_state(5.0, np.full(3, 300.0), np.full(4, 300.0), np.array([0.0, 0.5, 1.0]))


@pytest.mark.parametrize(
    ('T_stag', 'T_basic', 'phi', 'expected'),
    [
        ([300.0, 0.0, 300.0], [300.0] * 4, [0.5] * 3, ('end_temperature', 'cell 1', 0.0, 0.0)),
        ([300.0] * 3, [300.0, 300.0, -1.0, 300.0], [0.5] * 3, ('end_temperature', 'basic 2', -1.0, 0.0)),
        ([300.0] * 3, [300.0] * 4, [0.5, 1.0 + 1e-12, 0.5], ('end_melt_fraction', 'cell 1', 1.0 + 1e-12, 1.0)),
        ([300.0] * 3, [300.0] * 4, [-1e-12, 0.5, 0.5], ('end_melt_fraction', 'cell 0', -1e-12, 0.0)),
    ],
)  # fmt: skip
def test_an_end_state_outside_its_bounds_names_the_check(T_stag, T_basic, phi, expected):
    with pytest.raises(AcceptedStateError) as e:
        _check_end_state(5.0, np.array(T_stag), np.array(T_basic), np.array(phi))
    assert (e.value.check, e.value.node, e.value.value, e.value.bound) == expected
    assert e.value.time == 5.0


@pytest.mark.parametrize('which', ['T_stag', 'T_basic', 'phi'])
def test_a_non_finite_end_state_raises(which):
    arrays = {'T_stag': np.full(3, 300.0), 'T_basic': np.full(4, 300.0), 'phi': np.full(3, 0.5)}
    arrays[which][0] = np.nan
    with pytest.raises(AcceptedStateError) as e:
        _check_end_state(5.0, arrays['T_stag'], arrays['T_basic'], arrays['phi'])
    assert e.value.check == ('end_melt_fraction' if which == 'phi' else 'end_temperature')


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_a_non_finite_end_value_has_a_nan_bound(bad):
    """The class contract: the bound of a non-finite value is NaN."""
    with pytest.raises(AcceptedStateError) as e:
        _check_end_state(1.0, [bad, 300.0], [300.0, 300.0], [0.5, 0.5])
    assert np.isnan(e.value.bound)


def test_the_error_survives_pickling():
    e = AcceptedStateError('entropy_table', 'cell 2', 3.0, 99.0, 100.0)
    f = pickle.loads(pickle.dumps(e))
    assert (f.check, f.node, f.time, f.value, f.bound) == (
        'entropy_table',
        'cell 2',
        3.0,
        99.0,
        100.0,
    )


def test_the_call_start_is_checked_when_asked():
    """With skip_start=False the start column is checked too (a first call, a resume, a re-melt
    or a structure update)."""
    S = _inside()
    S[0, 0] = S_MIN - 50.0
    with pytest.raises(AcceptedStateError) as e:
        _check_accepted_states(T, S, W, S_MIN, S_MAX, skip_start=False)
    assert (e.value.node, e.value.time) == ('cell 0', 0.0)
