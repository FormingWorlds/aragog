"""The top basic node: linear extrapolation, the minmod-limited node and its fallback to the top cell."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aragog.config.boundary import BoundaryConfig
from aragog.mesh import top_node_entropy
from aragog.parser import _BoundaryConditionsParameters

pytestmark = pytest.mark.unit

_BC = dict(
    outer_boundary_condition=4, outer_boundary_value=0.0, inner_boundary_condition=1,
    inner_boundary_value=0.0, emissivity=1.0, equilibrium_temperature=255.0,
    core_heat_capacity=880.0,
)  # fmt: skip
_BUILDERS = pytest.mark.parametrize(
    'build',
    [
        lambda **kw: _normalized(_BoundaryConditionsParameters(**_BC, **kw)),
        lambda **kw: BoundaryConfig(**_BC, **kw),
    ],
    ids=['_BoundaryConditionsParameters', 'BoundaryConfig'],
)


def _normalized(bc):
    bc.normalize()
    return bc


def _linear(S, r):
    return (1 + r / 2) * S[-1] - r / 2 * S[-2]


_S = st.floats(-5000.0, 25000.0, allow_nan=False)
_R = st.floats(0.97, 1.03)


@_BUILDERS
@pytest.mark.parametrize('mode', ['linear', 'limited'])
def test_the_top_node_modes_are_accepted(build, mode):
    assert build(top_node=mode).top_node == mode


@_BUILDERS
def test_the_default_top_node_is_linear(build):
    assert build().top_node == 'linear'


@_BUILDERS
@given(value=st.sampled_from([True, False, None, 1, 0.5, 'Limited', 'minmod', '', np.nan]))
def test_another_top_node_value_is_refused(build, value):
    """A TOML true, a number, None, another spelling or another word is refused by name."""
    with pytest.raises(ValueError, match='top_node'):
        build(top_node=value)


def test_a_smooth_profile_keeps_the_linear_node():
    """Equal steps (or a smaller top step) give the linear extrapolation exactly."""
    S = np.array([3000.0, 2900.0, 2800.0])
    assert top_node_entropy(S, 1.0) == _linear(S, 1.0) == 2750.0
    S = np.array([3000.0, 2800.0, 2750.0])
    assert top_node_entropy(S, 1.0) == _linear(S, 1.0)


def test_a_steep_top_step_uses_the_smaller_slope():
    """A cold top cell over a hot cell: the step below sets the half-cell change."""
    S = np.array([3100.0, 3000.0, 200.0])
    assert top_node_entropy(S, 1.0) == 200.0 - 0.5 * 100.0
    assert _linear(S, 1.0) == 200.0 - 0.5 * 2800.0


@pytest.mark.parametrize('S', [[2800.0, 3000.0, 200.0], [3000.0, 3000.0, 200.0]])
def test_opposite_or_zero_steps_give_the_top_cell(S):
    assert top_node_entropy(np.array(S), 1.0) == 200.0


def test_a_node_outside_the_table_falls_back_to_the_top_cell():
    """The limited node 106 - 25 lies below S_min = 100, so the top cell 106 is returned."""
    S = np.array([206.0, 156.0, 106.0])
    assert top_node_entropy(S, 1.0) == 81.0
    assert top_node_entropy(S, 1.0, 100.0, 25000.0) == 106.0
    S = np.array([24800.0, 24900.0, 24990.0])
    assert top_node_entropy(S, 1.0, 100.0, 25000.0) == 24990.0


@given(s=st.tuples(_S, _S, _S), r=_R)
def test_the_limited_node_lies_between_the_top_cell_and_the_linear_node(s, r):
    S = np.array(s)
    node, lin = top_node_entropy(S, r), _linear(S, r)
    assert min(S[-1], lin) - 1e-9 <= node <= max(S[-1], lin) + 1e-9


_GAP = st.floats(0.0, 1000.0)


@given(s=st.tuples(_S, _S, _S), r=_R, below=_GAP, above=_GAP)
def test_with_every_cell_inside_the_table_the_node_is_inside(s, r, below, above):
    lo, hi = min(s) - below, max(s) + above
    node = top_node_entropy(np.array(s), r, lo, hi)
    assert lo <= node <= hi


@given(s=st.tuples(_S, _S, _S), r=_R, below=_GAP)
def test_the_fallback_moves_the_node_by_at_most_the_top_cell_distance_to_the_edge(s, r, below):
    """Where the fallback acts the node jumps from a value at most S_N - S_min below the top cell
    (the limited node lies between S_N and the linear node) to the top cell itself."""
    S, lo = np.array(s), min(s) - below
    free, kept = top_node_entropy(S, r), top_node_entropy(S, r, lo, np.inf)
    if free < lo:
        assert kept == S[-1] and 0.0 <= S[-1] - lo <= S[-1] - free


@given(S=st.tuples(_S, _S, _S), r=_R)
def test_a_non_finite_cell_gives_a_non_finite_node(S, r):
    """A NaN in the top 3 cells is not hidden by the limiter."""
    for i in range(3):
        bad = np.array(S)
        bad[i] = np.nan
        assert not np.isfinite(top_node_entropy(bad, r, -6000.0, 30000.0))
