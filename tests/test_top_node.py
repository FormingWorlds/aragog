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
def test_the_gradient_core_refuses_the_limited_node(build):
    with pytest.raises(ValueError, match='top_node'):
        build(core_bc='gradient', top_node='limited')


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
    """The limited node 106 - 25 lies below S_min = 100, so the blend weight is 0 and the top cell
    106 is returned; above S_max the same."""
    S = np.array([206.0, 156.0, 106.0])
    assert top_node_entropy(S, 1.0) == 81.0
    assert top_node_entropy(S, 1.0, 100.0, 25000.0, 50.0) == 106.0
    S = np.array([24800.0, 24900.0, 24990.0])
    assert top_node_entropy(S, 1.0, 100.0, 25000.0, 50.0) == 24990.0


@given(s=st.tuples(_S, _S, _S), r=_R)
def test_the_limited_node_lies_between_the_top_cell_and_the_linear_node(s, r):
    S = np.array(s)
    node, lin = top_node_entropy(S, r), _linear(S, r)
    assert min(S[-1], lin) - 1e-9 <= node <= max(S[-1], lin) + 1e-9


_GAP = st.floats(0.0, 1000.0)


@given(s=st.tuples(_S, _S, _S), r=_R, below=_GAP, above=_GAP)
def test_with_every_cell_inside_the_table_the_node_is_inside(s, r, below, above):
    lo, hi = min(s) - below, max(s) + above
    node = top_node_entropy(np.array(s), r, lo, hi, 50.0)
    assert lo <= node <= hi


@given(s=st.tuples(_S, _S, _S), r=_R, D=st.floats(1.0, 200.0))
def test_away_from_the_band_the_node_is_the_limited_node_bit_for_bit(s, r, D):
    """With S_f at least D inside the table on both sides the blend weight is 1."""
    S = np.array(s)
    free = top_node_entropy(S, r)
    node = top_node_entropy(S, r, free - D - 1.0, free + D + 1.0, D)
    assert node == free


@given(s=st.tuples(_S, _S, _S), r=_R, D=st.floats(1.0, 200.0), x=st.floats(0.0, 1.0))
def test_inside_the_band_the_node_lies_between_the_limited_node_and_the_top_cell(s, r, D, x):
    S = np.array(s)
    free = top_node_entropy(S, r)
    for lo, hi in ((free - x * D, np.inf), (-np.inf, free + x * D)):
        node = top_node_entropy(S, r, lo, hi, D)
        assert min(free, S[-1]) - 1e-9 <= node <= max(free, S[-1]) + 1e-9


@pytest.mark.parametrize('side', ['S_min', 'S_max'])
def test_the_node_is_continuous_across_the_band(side):
    """Moving the top cell in steps of 1e-3 J/kg/K through both edges of the band moves the node
    by a small multiple of the step, where a switch to the top cell would jump by about 333."""
    D, step = 50.0, 1.0e-3
    if side == 'S_min':
        lo, hi, base, mid = 0.0, np.inf, (1700.0, 1000.0), 333.0
    else:
        lo, hi, base, mid = -np.inf, 3000.0, (1300.0, 2000.0), 2667.0
    tops = mid + step * np.arange(-60000, 60001)
    S = np.vstack([np.full_like(tops, base[0]), np.full_like(tops, base[1]), tops])
    nodes, free = top_node_entropy(S, 1.0, lo, hi, D), top_node_entropy(S, 1.0)
    assert ((free < lo) | (free > hi)).any() and ((free > lo + D) & (free < hi - D)).any()
    assert np.max(np.abs(np.diff(nodes))) < 0.05


@pytest.mark.parametrize(('x', 'w'), [(0.25, 0.15625), (0.5, 0.5), (0.75, 0.84375)])
def test_the_blend_weight_is_the_smoothstep(x, w):
    """With the limited node x D above S_min the node is S_N + w (S_f - S_N), w = 3x^2 - 2x^3."""
    S = np.array([1700.0, 1000.0, 400.0])
    free = top_node_entropy(S, 1.0)
    node = top_node_entropy(S, 1.0, free - x * 40.0, np.inf, 40.0)
    assert node == pytest.approx(S[-1] + w * (free - S[-1]), rel=1e-12)


@given(S=st.tuples(_S, _S, _S), r=_R)
def test_a_non_finite_cell_gives_a_non_finite_node(S, r):
    """A NaN in the top 3 cells is not hidden by the limiter."""
    for i in range(3):
        bad = np.array(S)
        bad[i] = np.nan
        assert not np.isfinite(top_node_entropy(bad, r, -6000.0, 30000.0))
