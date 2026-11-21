"""The deprecated name ``lid_surface_temperature`` of ``surface_temperature`` in the boundary
parameters: a caller or a config file that still sets it holds the lid at that surface."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


def _load(how: str, cfg):
    """Parameters from ``cfg`` by one of the loaders a run uses."""
    import tomllib

    from aragog.config import Config
    from aragog.parser import Parameters

    if how == 'Parameters.from_file':
        return Parameters.from_file(cfg)
    if how == 'Config.from_file':
        return Config.from_file(str(cfg))
    with cfg.open('rb') as f:
        return Config.from_dict(tomllib.load(f), config_dir=cfg.parent)


LOADERS = [
    ('Parameters.from_file', 'abe_solid.toml'),
    ('Parameters.from_file', 'abe_mixed.cfg'),
    ('Config.from_file', 'abe_solid.toml'),
    ('Config.from_dict', 'abe_solid.toml'),
]


@pytest.mark.parametrize(('how', 'name'), LOADERS)
def test_the_old_name_in_a_config_file_sets_the_surface_temperature(tmp_path, how, name):
    """A config (TOML, INI or a dict) that still sets ``lid_surface_temperature`` loads with a
    FutureWarning, shown once per process under the default filters, and with that value as
    ``surface_temperature``."""
    cfg = _config(tmp_path, name, 'lid_surface_temperature = 1200.0\n')
    with pytest.warns(FutureWarning, match='surface_temperature'):
        try:
            p = _load(how, cfg)
        except (TypeError, ValueError) as exc:
            pytest.fail(f'the old key does not load: {exc}')
    assert p.boundary_conditions.surface_temperature == pytest.approx(1200.0)


@pytest.mark.parametrize(('how', 'name'), LOADERS)
def test_a_config_file_with_both_names_at_two_values_is_refused(tmp_path, how, name):
    """Both names at 2 values leave the held surface unclear: the load stops and names the
    section and both values; at one value it loads."""
    both = 'lid_surface_temperature = 1200.0\nsurface_temperature = 1300.0\n'
    with (
        pytest.warns(FutureWarning),
        pytest.raises(ValueError, match='surface_temperature 1300') as err,
    ):
        _load(how, _config(tmp_path, name, both))
    assert 'lid_surface_temperature 1200' in str(err.value)
    assert 'boundary_conditions' in str(err.value)
    same = 'lid_surface_temperature = 1200.0\nsurface_temperature = 1200.0\n'
    with pytest.warns(FutureWarning):
        p = _load(how, _config(tmp_path, name, same))
    assert p.boundary_conditions.surface_temperature == pytest.approx(1200.0)


def test_the_toml_refusal_names_the_section_and_the_file(tmp_path):
    """The TOML loader wraps the refusal like every other error of a section: it names
    ``[boundary_conditions]`` and the file."""
    both = 'lid_surface_temperature = 1200.0\nsurface_temperature = 1300.0\n'
    with pytest.warns(FutureWarning), pytest.raises(ValueError) as err:
        _load('Parameters.from_file', _config(tmp_path, 'abe_solid.toml', both))
    assert 'Configuration error in [boundary_conditions] of abe_solid.toml' in str(err.value)


def test_an_old_key_of_none_in_a_dict_config_counts_as_unset(tmp_path):
    """A dict config with ``lid_surface_temperature = None`` takes ``surface_temperature`` and
    still warns; without ``surface_temperature`` the held surface stays None."""
    import tomllib

    from aragog.config import Config

    cfg = _config(tmp_path, 'abe_solid.toml', '')
    with cfg.open('rb') as f:
        data = tomllib.load(f)
    data['boundary_conditions'] |= {
        'lid_surface_temperature': None,
        'surface_temperature': 1300.0,
    }
    with pytest.warns(FutureWarning, match='surface_temperature'):
        try:
            p = Config.from_dict(data, config_dir=cfg.parent)
        except (TypeError, ValueError) as exc:
            pytest.fail(f'an old key of None does not load: {exc}')
    assert p.boundary_conditions.surface_temperature == pytest.approx(1300.0)
    data['boundary_conditions'].pop('surface_temperature')
    with pytest.warns(FutureWarning):
        p = Config.from_dict(data, config_dir=cfg.parent)
    assert p.boundary_conditions.surface_temperature is None


_TEMPERATURE = st.one_of(st.none(), st.floats(300.0, 4000.0))


@settings(max_examples=60, deadline=None)
@given(old=_TEMPERATURE, new=_TEMPERATURE, same=st.booleans())
def test_a_dict_config_with_the_old_key_holds_one_surface(old, new, same):
    """For any old and new value (None included), ``Config.from_dict`` warns, holds the old value
    when only it is set or both agree, the new value when the old is None, refuses 2 values,
    and leaves the caller's dict as it was."""
    import copy
    import tomllib
    from pathlib import Path

    from aragog import CFG_DATA
    from aragog.config import Config

    if same and old is not None:
        new = old
    data = tomllib.loads(Path(str(CFG_DATA.joinpath('abe_solid.toml'))).read_text('utf-8'))
    data['boundary_conditions']['lid_surface_temperature'] = old
    if new is not None:
        data['boundary_conditions']['surface_temperature'] = new
    given_data = copy.deepcopy(data)
    with pytest.warns(FutureWarning, match='surface_temperature'):
        if old is not None and new is not None and new != old:
            with pytest.raises(ValueError, match='set surface_temperature only'):
                Config.from_dict(data, config_dir=Path(str(CFG_DATA)))
        else:
            held = Config.from_dict(data, config_dir=Path(str(CFG_DATA)))
            want = new if old is None else old
            assert held.boundary_conditions.surface_temperature == want
    assert data == given_data
