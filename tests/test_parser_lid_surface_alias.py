"""The name ``lid_surface_temperature`` of the boundary parameters, kept for one release as an
alias of ``surface_temperature``: a caller that still sets it holds the lid at that surface."""

from __future__ import annotations

import pytest

from aragog.config.boundary import held_lid_surface
from aragog.parser import _BoundaryConditionsParameters

pytestmark = pytest.mark.unit


def _bc() -> _BoundaryConditionsParameters:
    return _BoundaryConditionsParameters(
        outer_boundary_condition=4,
        outer_boundary_value=100.0,
        inner_boundary_condition=2,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        subgrid_lid=True,
        lid_mush_base=True,
    )


def test_the_old_name_sets_the_held_surface_with_a_warning():
    """Setting ``lid_surface_temperature`` sets ``surface_temperature``, so the lid is held
    there, and warns that the name goes."""
    bc = _bc()
    with pytest.warns(DeprecationWarning, match='surface_temperature'):
        bc.lid_surface_temperature = 1200.0
    assert bc.surface_temperature == pytest.approx(1200.0)
    assert held_lid_surface(4, True, bc.surface_temperature) == pytest.approx(1200.0)


def test_the_old_name_reads_the_surface_temperature():
    """Reading ``lid_surface_temperature`` gives ``surface_temperature``, None included, and
    the old name is no field of its own."""
    bc = _bc()
    with pytest.warns(DeprecationWarning, match='surface_temperature'):
        assert bc.lid_surface_temperature is None
    bc.surface_temperature = 1300.0
    with pytest.warns(DeprecationWarning, match='surface_temperature'):
        assert bc.lid_surface_temperature == pytest.approx(1300.0)
    assert 'lid_surface_temperature' not in _BoundaryConditionsParameters.__dataclass_fields__
