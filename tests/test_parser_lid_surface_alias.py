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


def _config(tmp_path, name: str, lines: str):
    """The bundled ``abe_solid.toml`` or ``abe_mixed.cfg`` with ``lines`` added to its
    ``[boundary_conditions]`` section."""
    from pathlib import Path

    from aragog import CFG_DATA

    src = Path(str(CFG_DATA.joinpath(name))).read_text(encoding='utf-8')
    assert src.count('[boundary_conditions]\n') == 1
    cfg = tmp_path / name
    cfg.write_text(src.replace('[boundary_conditions]\n', f'[boundary_conditions]\n{lines}', 1))
    return cfg


@pytest.mark.parametrize('name', ['abe_solid.toml', 'abe_mixed.cfg'])
def test_the_old_name_in_a_config_file_sets_the_surface_temperature(tmp_path, name):
    """A config file (TOML or INI) that still sets ``lid_surface_temperature`` loads with the
    warning and with that value as ``surface_temperature``."""
    from aragog.parser import Parameters

    cfg = _config(tmp_path, name, 'lid_surface_temperature = 1200.0\n')
    with pytest.warns(DeprecationWarning, match='surface_temperature'):
        p = Parameters.from_file(cfg)
    assert p.boundary_conditions.surface_temperature == pytest.approx(1200.0)


@pytest.mark.parametrize('name', ['abe_solid.toml', 'abe_mixed.cfg'])
def test_a_config_file_with_both_names_at_two_values_is_refused(tmp_path, name):
    """Both names at 2 values leave the held surface unclear: the load stops and names both;
    at one value it loads."""
    from aragog.parser import Parameters

    both = 'lid_surface_temperature = 1200.0\nsurface_temperature = 1300.0\n'
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match='surface_temperature 1300') as err,
    ):
        Parameters.from_file(_config(tmp_path, name, both))
    assert 'lid_surface_temperature 1200' in str(err.value)
    same = 'lid_surface_temperature = 1200.0\nsurface_temperature = 1200.0\n'
    with pytest.warns(DeprecationWarning):
        p = Parameters.from_file(_config(tmp_path, name, same))
    assert p.boundary_conditions.surface_temperature == pytest.approx(1200.0)
