"""Property tests of the matprop_smooth_width config field over its whole input domain."""

from __future__ import annotations

import importlib.resources
import tomllib
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aragog.config import Config
from aragog.config.phases import MixedPhaseConfig
from aragog.parser import Parameters, _PhaseMixedParameters

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

_REQUIRED = dict(
    latent_heat_of_fusion=4e5, rheological_transition_melt_fraction=0.4,
    rheological_transition_width=0.15, solidus='solidus.dat', liquidus='liquidus.dat',
    phase='mixed', phase_transition_width=0.01, grain_size=1e-3,
)  # fmt: skip
_BUILDERS = pytest.mark.parametrize(
    'build',
    [
        lambda **kw: _PhaseMixedParameters(**_REQUIRED, **kw),
        lambda **kw: MixedPhaseConfig(**_REQUIRED, **kw),
    ],
    ids=['_PhaseMixedParameters', 'MixedPhaseConfig'],
)


@_BUILDERS
def test_the_default_width_is_0_01(build):
    """Without the key the width is 0.01, the PROTEUS default, not 0 (the hard phase switch)."""
    width = build().matprop_smooth_width
    assert width == pytest.approx(0.01, rel=1e-15)
    assert width != 0.0


@_BUILDERS
@given(
    value=st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True)
    | st.sampled_from([Fraction(1, 2), np.float64(0.2), 5e-324, 1.0 - 2.0**-53])
)
def test_a_width_inside_the_open_unit_interval_is_kept(build, value):
    """Any real number strictly between 0 and 1 is kept as given, down to the smallest
    subnormal and up to the largest float below 1."""
    width = build(matprop_smooth_width=value).matprop_smooth_width
    assert width == value
    assert 0.0 < float(width) < 1.0


@_BUILDERS
@given(
    value=st.floats(max_value=0.0)
    | st.floats(min_value=1.0)
    | st.floats(allow_nan=True).filter(lambda x: x != x)
    | st.integers(max_value=0)
    | st.integers(min_value=1)
)
def test_a_width_at_or_beyond_0_or_1_or_nan_is_refused(build, value):
    """0, a negative width, 1, a width above 1, an infinity and NaN are refused by name: 0
    is the hard switch at the solidus and liquidus, and from 1 the blend covers the whole
    melting interval."""
    with pytest.raises(ValueError, match='matprop_smooth_width must lie in'):
        build(matprop_smooth_width=value)


@_BUILDERS
@pytest.mark.parametrize(
    'value', [True, False, np.True_, '0.01', None, np.array([0.01]), [0.01], 10**400]
)
def test_a_bool_text_array_none_or_overflowing_width_is_refused(build, value):
    """A TOML true, a string, None, an array, a list and an int too large for a float are
    refused by name; none of them reaches the evaluator."""
    with pytest.raises(ValueError, match='matprop_smooth_width must lie in'):
        build(matprop_smooth_width=value)


@pytest.mark.parametrize('value', [0.0, 1.0, True, 'x'])
def test_the_dict_loader_refuses_a_bad_width_in_phase_mixed(value):
    """Config.from_dict, the path of a TOML file, refuses the width before a solver is built."""
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as path:
        good = Parameters.from_file(path)
        assert good.phase_mixed.matprop_smooth_width == pytest.approx(0.01, rel=1e-15)
        text = path.read_text(encoding='utf-8')
    assert 'matprop_smooth_width' not in text
    data = tomllib.loads(text)
    data['phase_mixed']['matprop_smooth_width'] = value
    with pytest.raises(ValueError, match='matprop_smooth_width must lie in'):
        Config.from_dict(data)
