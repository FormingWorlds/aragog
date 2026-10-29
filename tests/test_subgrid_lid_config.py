"""Tests of the sub-grid lid config fields and their validator."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.config.boundary import BoundaryConfig, check_subgrid_lid
from aragog.parser import _BoundaryConditionsParameters

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

BASE = dict(
    outer_boundary_condition=1,
    outer_boundary_value=0.0,
    inner_boundary_condition=1,
    inner_boundary_value=0.0,
    emissivity=1.0,
    equilibrium_temperature=255.0,
    core_heat_capacity=880.0,
)
GOOD = dict(
    subgrid_lid=True,
    lid_delta_on=1.0,
    lid_delta_lo=None,
    lid_handover_cells=3,
    lid_ra_crit=450.0,
)
BAD_NUMBERS = [0, 0.0, -1.0, -1e-300, math.nan, math.inf, -math.inf, True, False, '1.0', None]


def _both(**fields):
    """Build the attrs config and the parser dataclass (normalised) with the same fields."""
    BoundaryConfig(**BASE, **fields)
    _BoundaryConditionsParameters(**BASE, **fields).normalize()


def test_defaults_keep_the_lid_off():
    """The lid is off by default on both config layers, so existing configs do not pick it up."""
    a = BoundaryConfig(**BASE)
    b = _BoundaryConditionsParameters(**BASE)
    b.normalize()
    for cfg in (a, b):
        assert cfg.subgrid_lid is False
        assert (cfg.lid_delta_on, cfg.lid_delta_lo, cfg.lid_handover_cells) == (1.0, None, 3)
        assert cfg.lid_ra_crit == pytest.approx(450.0, abs=0)


def test_valid_settings_pass_on_both_layers():
    """Valid settings, with and without an explicit delta_lo, construct on both layers."""
    _both(**GOOD)
    _both(**{**GOOD, 'lid_delta_lo': 0.25, 'lid_handover_cells': 1})
    _both(**{**GOOD, 'subgrid_lid': False, 'lid_delta_on': 5e3})
    assert check_subgrid_lid(True, 1.0, 0.999999, 1, 1e-3, 1, 'core_module') is None


@pytest.mark.parametrize('value', [1, 0, 'true', None, 1.0])
def test_subgrid_lid_must_be_bool(value):
    """A number or a string for the switch is refused, on both layers."""
    with pytest.raises(ValueError, match='subgrid_lid must be true or false'):
        _both(**{**GOOD, 'subgrid_lid': value})
    with pytest.raises(ValueError, match='subgrid_lid must be true or false'):
        BoundaryConfig(**BASE, **{**GOOD, 'subgrid_lid': value})


@pytest.mark.parametrize('field', ['lid_delta_on', 'lid_ra_crit'])
@pytest.mark.parametrize('value', BAD_NUMBERS)
def test_positive_fields_refuse_bad_values(field, value):
    """lid_delta_on and lid_ra_crit refuse zero, negatives, NaN, inf, bools, strings and None."""
    with pytest.raises(ValueError, match=f'{field} must be a finite number > 0'):
        _both(**{**GOOD, field: value})


@pytest.mark.parametrize('value', [0, 0.0, -0.5, 1.0, 2.0, math.nan, math.inf, True, '0.5'])
def test_delta_lo_must_lie_below_delta_on(value):
    """lid_delta_lo refuses values outside (0, lid_delta_on), the edges included, and non-numbers."""
    with pytest.raises(ValueError, match='lid_delta_lo must be a finite number in'):
        _both(**{**GOOD, 'lid_delta_lo': value})


@pytest.mark.parametrize('value', [0, -3, 2.0, 3.0, True, '3', None, math.nan])
def test_handover_cells_must_be_integer_at_least_one(value):
    """lid_handover_cells refuses 0, negatives, floats (also integral ones), bools and strings."""
    with pytest.raises(ValueError, match='lid_handover_cells must be'):
        _both(**{**GOOD, 'lid_handover_cells': value})


@pytest.mark.parametrize('outer', [2, 4, 5])
def test_lid_refused_without_grey_body(outer):
    """With the lid on, an outer boundary other than the grey body is refused; off, it passes."""
    with pytest.raises(ValueError, match='needs the grey-body outer boundary'):
        BoundaryConfig(**{**BASE, 'outer_boundary_condition': outer}, **GOOD)
    BoundaryConfig(
        **{**BASE, 'outer_boundary_condition': outer}, **{**GOOD, 'subgrid_lid': False}
    )


def test_lid_refused_on_gradient_core():
    """The gradient core replaces the state layout, so the lid is refused there."""
    with pytest.raises(ValueError, match="not available on core_bc = 'gradient'"):
        _both(**GOOD, core_bc='gradient')
    _both(**{**GOOD, 'subgrid_lid': False}, core_bc='gradient')


@settings(max_examples=200, deadline=None)
@given(
    on=st.floats(allow_nan=True, allow_infinity=True),
    lo=st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True)),
    cells=st.one_of(st.integers(-5, 50), st.floats(allow_nan=True), st.booleans()),
    ra=st.floats(allow_nan=True, allow_infinity=True),
)
def test_validator_accepts_exactly_the_valid_domain(on, lo, cells, ra):
    """Over the full float domain the validator passes exactly the documented valid set."""
    valid = (
        math.isfinite(on)
        and on > 0
        and math.isfinite(ra)
        and ra > 0
        and (lo is None or (math.isfinite(lo) and 0 < lo < on))
        and isinstance(cells, int)
        and not isinstance(cells, bool)
        and cells >= 1
    )
    try:
        check_subgrid_lid(True, on, lo, cells, ra, 1, 'energy_balance')
        accepted = True
    except ValueError:
        accepted = False
    assert accepted is valid
    assert isinstance(accepted, bool)
