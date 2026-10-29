"""Property tests of the mesh.surface_cell_thickness and mesh.cmb_cell_thickness fields.

Module under test: the mesh config classes ``aragog.parser._MeshParameters`` (the path of
the TOML loader and of PROTEUS) and ``aragog.config.MeshConfig``. Invariants: 0 is off; a
real number in [2, (outer_radius - inner_radius) / (number_of_nodes - 1)) m is kept as
given; every other value, a bool and a string included, is refused with a ValueError that
names the field.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import aragog
from aragog.config import MeshConfig
from aragog.parser import Parameters, _MeshParameters

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

CFG = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'
R_OUT, R_IN, N = 6.371e6, 3.48e6, 100
UNIFORM = (R_OUT - R_IN) / (N - 1)
_REQUIRED = dict(
    outer_radius=R_OUT, inner_radius=R_IN, number_of_nodes=N,
    mixing_length_profile='nearest_boundary', core_density=10738.0,
)  # fmt: skip
FIELDS = pytest.mark.parametrize('name', ['surface_cell_thickness', 'cmb_cell_thickness'])
BUILDERS = pytest.mark.parametrize(
    'build',
    [
        lambda **kw: _MeshParameters(**{**_REQUIRED, **kw}),
        lambda **kw: MeshConfig(**{**_REQUIRED, **kw}),
    ],
    ids=['_MeshParameters', 'MeshConfig'],
)


def _refused(build, name, value):
    """The ValueError message for ``value``; any other outcome fails on an assertion."""
    try:
        build(**{name: value})
    except ValueError as err:
        return str(err)
    except Exception as err:  # noqa: BLE001
        pytest.fail(f'{name} = {value!r} raised {type(err).__name__}, not ValueError: {err}')
    pytest.fail(f'{name} = {value!r} was kept')


@BUILDERS
@FIELDS
def test_the_default_is_off(build, name):
    """Without the key the field is 0, the uniform grid at that end."""
    assert getattr(build(), name) == 0.0


@BUILDERS
@FIELDS
@given(
    value=st.floats(min_value=2.0, max_value=UNIFORM, exclude_max=True)
    | st.sampled_from(
        [0, 0.0, 2, 200, Fraction(5, 2), np.float64(50.0), math.nextafter(UNIFORM, 0)]
    )
)
def test_zero_and_a_value_below_the_uniform_cell_are_kept(build, name, value):
    """0 and any real number in [2 m, uniform cell) are kept as given."""
    assert getattr(build(**{name: value}), name) == value


@BUILDERS
@FIELDS
@given(
    value=st.floats(max_value=-math.ulp(0.0))
    | st.floats(min_value=math.ulp(0.0), max_value=2.0, exclude_max=True)
    | st.floats(min_value=UNIFORM)
    | st.sampled_from([math.nan, math.inf, -math.inf, -1e-300, UNIFORM, 10**400])
)
def test_a_value_outside_the_open_interval_is_refused(build, name, value):
    """A negative value, a value in (0, 2 m), NaN, inf and a value at or above the uniform
    cell are refused."""
    assert f'mesh.{name}' in _refused(build, name, value)


@BUILDERS
@FIELDS
@pytest.mark.parametrize('value', [True, False, np.bool_(True), '50', b'50', None, [50.0], 50j])
def test_a_value_that_is_not_a_real_number_is_refused(build, name, value):
    """A bool (a TOML true or false), a string, None, a list and a complex are refused."""
    assert f'mesh.{name}' in _refused(build, name, value)


def _load(tmp_path, line):
    """Parameters of abe_solid.toml with ``line`` added to its [mesh] table."""
    text = CFG.read_text()
    assert text.count('[mesh]\n') == 1
    path = tmp_path / 'abe_solid.toml'
    path.write_text(text.replace('[mesh]\n', f'[mesh]\n{line}\n'))
    return Parameters.from_file(path)


@FIELDS
def test_the_loader_refuses_a_toml_true(tmp_path, name):
    """A TOML ``true`` is refused, not read as 1 m."""
    with pytest.raises(ValueError, match=f'mesh.{name} must be 0'):
        _load(tmp_path, f'{name} = true')


@FIELDS
def test_the_loader_keeps_a_value_below_the_uniform_cell(tmp_path, name):
    """200 m reaches the parsed mesh parameters unchanged."""
    assert getattr(_load(tmp_path, f'{name} = 200.0').mesh, name) == 200.0


@BUILDERS
@FIELDS
@pytest.mark.parametrize('n', [1, 2])
def test_a_cell_thickness_with_fewer_than_3_nodes_is_refused(build, name, n):
    """A refined grid needs 3 nodes; 1 or 2 nodes with a cell thickness give a ValueError
    that names the field, not a ZeroDivisionError."""
    with pytest.raises(ValueError, match=f'mesh.{name}'):
        build(**{'number_of_nodes': n, name: 200.0})


@BUILDERS
def test_both_cells_need_4_nodes(build):
    """With both ends set the 2 cells of a 3-node grid would have to fill the shell; 3 nodes
    are refused with both fields named, 4 nodes are kept."""
    with pytest.raises(ValueError, match='number_of_nodes >= 4'):
        build(number_of_nodes=3, surface_cell_thickness=200.0, cmb_cell_thickness=200.0)
    kept = build(number_of_nodes=4, surface_cell_thickness=200.0, cmb_cell_thickness=200.0)
    assert (kept.surface_cell_thickness, kept.cmb_cell_thickness) == (200.0, 200.0)
