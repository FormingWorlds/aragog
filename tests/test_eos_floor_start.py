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


@pytest.mark.parametrize('offset', [-10.0, 0.0])
@pytest.mark.parametrize('rate', [0.0, 1.0, -1.0])
def test_a_start_at_or_below_the_table_is_refused_at_any_rate(offset, rate):
    """A start at or below S_min is refused also when it warms."""
    assert _start(np.full(3, S_MIN + offset), [rate] * 3) == (None, True)


@pytest.mark.parametrize(('rate', 'refuse'), [(0.0, False), (1.0, False), (-1.0, True)])
def test_a_start_just_above_the_table_arms_half_its_distance_unless_it_falls(rate, refuse):
    """At S_min + 1e-6 m a warming or steady start arms half its distance; a falling one is
    refused."""
    level, r = _start(np.full(3, S_MIN + 1e-4), [rate] * 3)
    assert r is refuse and (level is None if refuse else level == pytest.approx(S_MIN + 5e-5))


def test_the_rate_of_the_lowest_component_decides():
    """With a deep cell lower than the top node, the deep cell's rate decides; the top node's does not."""
    S = np.array([S_MIN + 30.0, S_MIN + 49.0, S_MIN + 49.0])
    assert _start(S, [-10.0, 0.0, 1.0]) == (None, True)
    level, refuse = _start(S, [1.0, 0.0, -10.0])
    assert not refuse and level == pytest.approx(0.5 * (S_MIN + 30.0 + S_MIN))


@pytest.mark.parametrize(
    ('S', 'rate'),
    [([S_MIN + 40.0] * 3, [-10.0, 0.0, 1.0]), ([S_MIN + 40.0, S_MIN + 40.00005, S_MIN + 50.0], [10.0, -10.0, 10.0])],
)  # fmt: skip
def test_the_fastest_fall_among_tied_components_decides(S, rate):
    """Components within 1e-6 m of the floor entropy, the top node included, are refused by the
    fastest fall among them, even when another of them warms."""
    assert _start(S, rate) == (None, True)


def test_a_drain_that_lands_on_half_the_distance_is_refused():
    """A projected entropy equal to (s0 + S_min)/2 at the call end is refused."""
    assert _start(np.full(3, S_MIN + 40.0), [-2.0] * 3) == (None, True)


T = np.array([0.0, 0.0, 1.0])  # top node equal to the top cell, so the boundaries are exact


@pytest.mark.parametrize(
    ('s0', 'expected'),
    [
        ((S_MIN + M) + 1e-6 * M, (S_MIN + 0.5 * M, False)),
        (S_MIN + 1.5e-6 * M, (0.5 * (S_MIN + 1.5e-6 * M + S_MIN), False)),
        (S_MIN + 2e-6 * M, (0.5 * (S_MIN + 2e-6 * M + S_MIN), False)),
        (np.nextafter(S_MIN, np.inf), (None, True)),
    ],
)
def test_the_level_and_table_boundaries_are_inclusive(s0, expected):
    """A start at S_min + m + 1e-6 m arms S_min + m/2; a warming start at or below S_min + 2e-6 m
    arms half its distance, and one ulp above S_min, where no level fits between, is refused."""
    assert _floor_start_level(T, np.full(3, s0), lambda: np.ones(3), S_MIN, M, DT) == expected


@pytest.mark.parametrize('s0', [S_MIN + 1.5e-6 * M, S_MIN + 2e-6 * M])
def test_a_falling_start_in_the_table_band_is_refused_at_any_span(s0):
    """Inside S_min + 2e-6 m, including its edge, a fall is refused also over a span too short
    to reach half the distance."""
    assert _floor_start_level(T, np.full(3, s0), lambda: -np.ones(3), S_MIN, M, 1e-10) == (
        None,
        True,
    )


def test_one_ulp_above_the_table_is_refused_when_the_midpoint_rounds_up():
    """With an odd last bit of S_min the midpoint of S_min and the next float rounds up to the
    start itself, which leaves no level below it."""
    s_min = np.nextafter(S_MIN, np.inf)
    s0 = np.nextafter(s_min, np.inf)
    assert 0.5 * (s0 + s_min) == s0  # precondition: the midpoint rounds up
    assert _floor_start_level(T, np.full(3, s0), lambda: np.ones(3), s_min, M, DT) == (
        None,
        True,
    )


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


def test_a_top_node_below_the_table_is_refused_on_a_slow_fall_with_every_cell_inside():
    """A top node at S_min - 58 from a steep gradient of the top two cells, every cell inside
    the table and a slow fall of the top cell: the start counts as at the table, arms no level
    and is refused."""
    S = np.array([S_MIN + 2000.0, S_MIN + 236.0, S_MIN + 40.0])
    assert W @ S == pytest.approx(S_MIN - 58.0)
    assert _start(S, [0.0, 0.0, -1.5e-3]) == (None, True)


S_NODE_BELOW = np.array([S_MIN + 2000.0, S_MIN + 236.0, S_MIN + 40.0])  # top node at S_min - 58
S_NODE_AT = np.array([S_MIN + 2000.0, S_MIN + 120.0 - 2e-4, S_MIN + 40.0])  # S_min + 1e-4


@pytest.mark.parametrize('rate', [[-0.015, 0.069, 0.0034], [0.0, -4.0, -1.0]])
def test_a_top_node_below_the_table_is_refused_when_it_or_the_top_cell_rises(rate):
    """Top node at S_min - 58 with every cell inside: refused also when the top cell warms
    (node falling by extrapolation) or the node rises."""
    assert W @ S_NODE_BELOW == pytest.approx(S_MIN - 58.0)
    assert _start(S_NODE_BELOW, rate) == (None, True)


@pytest.mark.parametrize('rate', [[-0.015, 0.069, 0.0034], [0.0, -4.0, -1.0]])
def test_a_top_node_just_above_the_table_arms_half_its_distance_when_not_falling(rate):
    """Top node at S_min + 1e-4: a node falling only by extrapolation under a warming top cell,
    or a rising node over a falling top cell, arms half the node's distance."""
    assert W @ S_NODE_AT == pytest.approx(S_MIN + 1e-4)
    level, refuse = _start(S_NODE_AT, rate)
    assert not refuse and level == pytest.approx(S_MIN + 5e-5)
