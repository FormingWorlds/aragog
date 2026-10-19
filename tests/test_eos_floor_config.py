"""Property tests of the eos_floor_margin config field over its whole input domain."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aragog.config.energy import EnergyConfig
from aragog.parser import _EnergyParameters
from aragog.solver.entropy_solver import _resolve_step_cap

pytestmark = pytest.mark.unit

_FLAGS = dict(
    conduction=True, convection=True, gravitational_separation=False, mixing=False,
    radionuclides=False, tidal=False,
)  # fmt: skip
_BUILDERS = pytest.mark.parametrize(
    'build',
    [lambda **kw: _EnergyParameters(**_FLAGS, **kw), lambda **kw: EnergyConfig(**_FLAGS, **kw)],
    ids=['_EnergyParameters', 'EnergyConfig'],
)


@_BUILDERS
@given(value=st.floats(allow_nan=False, allow_infinity=False) | st.integers(-(10**6), 10**6))
def test_a_finite_real_margin_is_kept_and_arms_only_when_positive(build, value):
    """Any finite real number is accepted as given; the solver arms the floor only above 0."""
    margin = build(eos_floor_margin=value).eos_floor_margin
    assert margin == value and _resolve_step_cap(margin) == (float(value) if value > 0 else 0.0)


@_BUILDERS
@given(
    value=st.sampled_from(
        [True, False, np.True_, np.nan, np.inf, -np.inf, '100', np.array([1.0])]
    )
)
def test_a_bool_non_finite_text_or_array_margin_is_refused(build, value):
    """A TOML true, NaN, an infinity, a string or an array is refused by name, never armed."""
    with pytest.raises(ValueError, match='eos_floor_margin'):
        build(eos_floor_margin=value)


@_BUILDERS
def test_a_fraction_is_accepted_and_a_huge_int_refused(build):
    """Any real number that converts to a finite float is accepted; one that overflows is refused."""
    assert build(eos_floor_margin=Fraction(1, 2)).eos_floor_margin == Fraction(1, 2)
    with pytest.raises(ValueError, match='eos_floor_margin'):
        build(eos_floor_margin=10**400)


@_BUILDERS
@given(
    value=st.none()
    | st.floats(max_value=0.0, allow_nan=False, allow_infinity=False)
    | st.integers(-(10**6), 0)
)
def test_the_default_margin_is_0_and_none_or_a_non_positive_margin_is_off_like_it(build, value):
    """The default margin is 0, so the floor is off; None and any margin at or below 0 leave it
    off exactly as the default does."""
    assert build().eos_floor_margin == 0.0
    assert _resolve_step_cap(build().eos_floor_margin) == 0.0
    assert _resolve_step_cap(build(eos_floor_margin=value).eos_floor_margin) == 0.0
