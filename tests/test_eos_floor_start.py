"""Unit tests of the EOS floor start decision, a pure function of the start state and its rate."""

from __future__ import annotations

import numpy as np
import pytest

from aragog.solver.entropy_solver import _floor_start_level, _phi_cap_event_factory

pytestmark = pytest.mark.unit

S_MIN, M, DT = 100.0, 100.0, 10.0  # table minimum, margin [J/kg/K], call span [yr]
W = np.array([0.0, -0.5, 1.5])  # top basic node from the top two cells


def _start(S, rate):
    """Level and refusal for state ``S`` with per-cell rate ``rate`` [J/kg/K/yr]."""
    return _floor_start_level(
        W, np.asarray(S, float), lambda: np.asarray(rate, float), S_MIN, M, DT
    )


def _no_rate():
    raise AssertionError('the rate must not be evaluated above S_min + m/2')


@pytest.mark.parametrize(
    ('offset', 'level'),
    [(400.0, 200.0), (100.0 + 1e-3, 200.0), (100.0 + 5e-5, 150.0), (80.0, 150.0)],
)
def test_the_two_fixed_levels_need_no_rate(offset, level):
    """Above S_min + m/2 + 1e-6 m the level is S_min + m or S_min + m/2, without an RHS call."""
    S = np.full(3, S_MIN + offset)
    assert _floor_start_level(W, S, _no_rate, S_MIN, M, DT) == (level, False)


@pytest.mark.parametrize('offset', [40.0, 40.000001, 36.7468, 41.2025])
def test_a_tie_between_the_top_node_and_a_cell_does_not_decide(offset):
    """A uniform start in the band, whose top node equals the cells up to rounding, arms half its
    distance under a warming top and a slowly cooling bottom cell, at every offset."""
    level, refuse = _start(np.full(3, S_MIN + offset), [-1.1e-8, 0.0, 3.4e-8])
    s0 = min(float(W @ np.full(3, S_MIN + offset)), S_MIN + offset)
    assert not refuse and level == pytest.approx(0.5 * (s0 + S_MIN), abs=1e-9)


def test_a_slow_outward_drain_in_the_band_advances():
    """A top node in the band losing 5e-5 J/kg/K/yr cannot reach half its distance in the call."""
    S = np.array([S_MIN + 60.0, S_MIN + 50.0, S_MIN + 40.0 + 2.7 / 1.5])
    level, refuse = _start(S, [0.0, 0.0, -5e-5 / 1.5])
    assert not refuse and level == pytest.approx(0.5 * (float(W @ S) + S_MIN))


def test_a_fast_drain_in_the_band_is_refused():
    """A drain that reaches half the distance within the call span is refused."""
    assert _start(np.full(3, S_MIN + 40.0), [-10.0, -10.0, -10.0]) == (None, True)


@pytest.mark.parametrize('offset', [-10.0, 0.0, 1e-4])
@pytest.mark.parametrize(('rate', 'refuse'), [(0.0, False), (1.0, False), (-1.0, True)])
def test_a_start_at_or_below_the_table_advances_unless_it_falls(offset, rate, refuse):
    """At or below S_min + 2e-6 m no level is armed: a warming or steady start advances with the
    floor off, a falling one is refused."""
    assert _start(np.full(3, S_MIN + offset), [rate] * 3) == (None, refuse)


def test_the_rate_of_the_lowest_component_decides():
    """With a deep cell lower than the top node, the deep cell's rate decides; the top node's does not."""
    S = np.array([S_MIN + 30.0, S_MIN + 49.0, S_MIN + 49.0])
    assert _start(S, [-10.0, 0.0, 1.0]) == (None, True)
    level, refuse = _start(S, [1.0, 0.0, -10.0])
    assert not refuse and level == pytest.approx(0.5 * (S_MIN + 30.0 + S_MIN))


@pytest.mark.parametrize(
    ('S', 'rate', 'expected'),
    [
        ([np.nan, S_MIN + 40.0, S_MIN + 40.0], [0.0] * 3, (None, False)),
        ([np.inf, S_MIN + 40.0, S_MIN + 40.0], [0.0] * 3, (None, False)),
        ([S_MIN + 40.0] * 3, [np.nan] * 3, (None, True)),
        ([S_MIN - 10.0] * 3, [np.nan] * 3, (None, True)),
        ([S_MIN + 40.0] * 3, [-np.inf] * 3, (None, True)),
        ([S_MIN + 40.0] * 3, [0.0, 0.0, np.inf], (S_MIN + 20.0, False)),
    ],
)
def test_non_finite_states_and_rates(S, rate, expected):
    """A non-finite start is left to the table guard; a NaN or falling infinite rate is refused,
    an infinitely warming top cell arms half the distance."""
    level, refuse = _start(S, rate)
    assert refuse is expected[1] and (
        level is None if expected[0] is None else level == pytest.approx(expected[0])
    )


def test_the_scipy_event_falls_back_to_the_margin():
    """A floor-only event whose evaluation raises returns the margin, never 0, which would fire."""
    event = _phi_cap_event_factory(
        eos=None, P_stag=np.zeros(2), volume=np.ones(2), n_stag=2, phi0_global=0.0, cap=0.0,
        state_scale=np.ones(2), floor_weights=np.ones(3), floor_level=200.0, floor_margin=M,
    )  # fmt: skip
    assert event(0.0, np.array([5000.0, 5000.0])) == M and event.binding_cap is None
