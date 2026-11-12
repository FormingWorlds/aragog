"""Default refined mesh and the ``mesh.surface_cell_thickness = "uniform"`` rule.

Module under test: ``aragog.config.mesh.check_cell_thickness``, the default mesh fields of
``aragog.parser._MeshParameters`` and ``aragog.config.MeshConfig``,
``aragog.mesh.stretching.stretched_unit_grid`` and ``aragog.mesh.Mesh``.
Invariants: the default mesh has a 3000 m CMB cell and an outer cell equal to the top cell
of the uniform mesh of the same coordinate mode; "uniform" with a CMB cell of 0 is the
uniform mesh, bit for bit; "uniform" is the only string accepted, and only at the surface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

import aragog
import aragog.mesh as mesh_module
from aragog.config import MeshConfig
from aragog.config.mesh import UNIFORM_CELL
from aragog.mesh import RADIUS_XTOL, Mesh
from aragog.mesh.stretching import stretched_unit_grid
from aragog.parser import Parameters, _MeshParameters
from aragog.rheology import SolidRheologyParams

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

CFG = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'
R_OUT, R_IN = 6.371e6, 3.48e6
_REQUIRED = dict(
    outer_radius=R_OUT, inner_radius=R_IN, number_of_nodes=80,
    mixing_length_profile='nearest_boundary', core_density=10738.0,
)  # fmt: skip
BUILDERS = pytest.mark.parametrize(
    'build',
    [
        lambda **kw: _MeshParameters(**{**_REQUIRED, **kw}),
        lambda **kw: MeshConfig(**{**_REQUIRED, **kw}),
    ],
    ids=['_MeshParameters', 'MeshConfig'],
)
MODES = pytest.mark.parametrize('mass_coordinates', [False, True], ids=['radius', 'mass'])


def _uniform_tol(mass_coordinates):
    """Absolute tolerance [m] of a cell of the uniform mesh: mass mode solves its radii to
    RADIUS_XTOL, so a cell holds to twice that; radius mode is exact."""
    return 2 * RADIUS_XTOL if mass_coordinates else 0.0


def _mesh(top, bottom, mass_coordinates, n=None, r_in=None):
    p = Parameters.from_file(CFG)
    p.mesh.mass_coordinates = mass_coordinates
    p.mesh.surface_cell_thickness, p.mesh.cmb_cell_thickness = top, bottom
    if n is not None:
        p.mesh.number_of_nodes = n
    if r_in is not None:
        p.mesh.inner_radius = r_in
    return Mesh(p)


def _radii(m):
    return np.asarray(m.basic.radii).ravel()


@BUILDERS
def test_the_default_mesh_fields_are_a_3000_m_cmb_cell_and_the_uniform_top_cell(build):
    """Without the keys the CMB cell is 3000 m and the surface cell is "uniform"."""
    m = build()
    assert (m.cmb_cell_thickness, m.surface_cell_thickness) == (3000.0, UNIFORM_CELL)
    assert UNIFORM_CELL == 'uniform'


def test_the_default_mlt_bottom_slope_is_0_82():
    """The viscous-branch mixing length grows as 0.82 (r - r_in) above the CMB by default."""
    assert SolidRheologyParams().mlt_bottom_slope == 0.82


def test_the_loader_reads_the_default_and_a_uniform_string(tmp_path):
    """abe_solid.toml has no cell keys: the parsed mesh holds the defaults; a TOML string
    "uniform" for the surface cell is kept."""
    p = Parameters.from_file(CFG)
    assert (p.mesh.cmb_cell_thickness, p.mesh.surface_cell_thickness) == (3000.0, 'uniform')
    text = CFG.read_text()
    assert text.count('[mesh]\n') == 1
    path = tmp_path / 'abe_solid.toml'
    path.write_text(text.replace('[mesh]\n', '[mesh]\nsurface_cell_thickness = "uniform"\n'))
    assert Parameters.from_file(path).mesh.surface_cell_thickness == 'uniform'


@BUILDERS
@pytest.mark.parametrize(
    'value', ['Uniform', 'UNIFORM', ' uniform', 'uniform ', 'auto', '', b'uniform']
)
def test_a_surface_string_other_than_uniform_is_refused(build, value):
    """Only the exact string "uniform" is a rule; other strings and bytes are refused with
    the field name."""
    with pytest.raises(ValueError, match='mesh.surface_cell_thickness'):
        build(surface_cell_thickness=value)


@BUILDERS
def test_uniform_is_refused_for_the_cmb_cell(build):
    """The rule applies to the outer cell only; at the CMB it is refused with the field name."""
    with pytest.raises(ValueError, match='mesh.cmb_cell_thickness'):
        build(cmb_cell_thickness='uniform')


@BUILDERS
@pytest.mark.parametrize('n', [1, 2, 3])
def test_uniform_with_any_cmb_cell_is_kept_at_any_node_count(build, n):
    """The config keeps the "uniform" top cell with or without a CMB cell at any node count:
    whether the CMB cell refines the mesh, and so needs 4 nodes, is known only when the mesh
    is built (test_mesh_cmb_fallback.py)."""
    for cmb in (0.0, 3000.0):
        kept = build(number_of_nodes=n, cmb_cell_thickness=cmb)
        assert (kept.surface_cell_thickness, kept.cmb_cell_thickness) == ('uniform', cmb)


@pytest.mark.parametrize('n', [4, 24, 80, 160, 1000])
def test_the_unit_grid_accepts_an_end_cell_equal_to_the_uniform_cell(n):
    """An end fraction of exactly 1 / (n - 1) is the uniform end cell: alone, or at both ends,
    it gives the uniform grid bit for bit; with a refined other end it holds both targets."""
    h = 1.0 / (n - 1)
    xi = np.linspace(0.0, 1.0, n)
    for first, last in ((0.0, h), (h, 0.0), (h, h)):
        np.testing.assert_array_equal(stretched_unit_grid(n, first, last), xi)
    first = min(1e-3, h / 2)
    s = stretched_unit_grid(n, first, h)
    d = np.diff(s)
    assert (s[0], s[-1]) == (0.0, 1.0) and np.all(d > 0.0)
    assert d[0] == pytest.approx(first, rel=1e-6)
    assert d[-1] == pytest.approx(h, rel=1e-6)


@pytest.mark.parametrize('n', [10, 100])
def test_the_unit_grid_still_refuses_an_end_cell_above_the_uniform_cell(n):
    """The next float above 1 / (n - 1) is refused at either end."""
    over = np.nextafter(1.0 / (n - 1), 1.0)
    for first, last in ((over, 0.0), (0.0, over), (1e-3, over)):
        with pytest.raises(ValueError, match='cell fraction'):
            stretched_unit_grid(n, first, last)


@pytest.mark.physics_invariant
@settings(max_examples=200, deadline=None)
@given(n=st.integers(min_value=4, max_value=1000), u=st.floats(min_value=0.0, max_value=1.0))
def test_the_two_sided_grid_with_a_uniform_last_cell_holds_both_targets(n, u):
    """For a first cell from 1e-7 to just below 1 / (n - 1) (log-uniform) and a last cell of
    exactly 1 / (n - 1): the grid runs from 0 to 1, increases strictly and holds both
    targets to 1e-6 relative. At 4 nodes a first cell up to 3.24e-7 (measured on 4000 log-spaced
    values) has no two-sided Vinokur grid, as with a last cell 1 ulp below 1 / 3, and is
    refused; from 5 nodes on none is."""
    h = 1.0 / (n - 1)
    first = min(10 ** (-7 + u * (np.log10(h) + 7)), np.nextafter(h, 0.0))
    if n == 4 and first <= 3.25e-7:
        with pytest.raises(ValueError, match='did not converge'):
            stretched_unit_grid(n, first, h)
        return
    s = stretched_unit_grid(n, first, h)
    d = np.diff(s)
    assert (s[0], s[-1]) == (0.0, 1.0)
    assert np.all(d > 0.0)
    assert d[0] == pytest.approx(first, rel=1e-6)
    assert d[-1] == pytest.approx(h, rel=1e-6)


@MODES
def test_uniform_top_without_a_cmb_cell_is_the_uniform_mesh_bit_for_bit(
    mass_coordinates, monkeypatch
):
    """CMB cell 0 with "uniform" or 0 at the surface: equal radii, pressure and density, and
    the refined grid and its radii check are never called."""

    def refined(*_):
        raise RuntimeError('refined path')

    off = _mesh(0.0, 0.0, mass_coordinates)
    monkeypatch.setattr(Mesh, '_refined_unit_grid', refined)
    monkeypatch.setattr(Mesh, '_check_refined_radii', refined)
    on = _mesh('uniform', 0.0, mass_coordinates)
    np.testing.assert_array_equal(_radii(on), _radii(off))
    np.testing.assert_array_equal(
        np.asarray(on.staggered.radii), np.asarray(off.staggered.radii)
    )
    np.testing.assert_array_equal(np.asarray(on.basic_pressure), np.asarray(off.basic_pressure))
    np.testing.assert_array_equal(
        np.asarray(on.staggered_effective_density), np.asarray(off.staggered_effective_density)
    )


@MODES
@pytest.mark.parametrize('n', [24, 80, 160])
def test_the_uniform_top_cell_equals_the_top_cell_of_the_uniform_mesh(mass_coordinates, n):
    """CMB cell 1000 m and "uniform" at the surface: the CMB cell is 1000 m and the top cell
    equals that of the uniform mesh of the same mode and node count, to 1e-6 relative or the
    2 m to which mass mode places the uniform radii; in radius mode that is
    (r_out - r_in) / (n - 1)."""
    uni = np.diff(_radii(_mesh(0.0, 0.0, mass_coordinates, n)))
    r = _radii(_mesh('uniform', 1e3, mass_coordinates, n))
    d = np.diff(r)
    assert np.all(d > 0.0)
    assert d[0] == pytest.approx(1e3, rel=1e-6)
    assert d[-1] == pytest.approx(uni[-1], rel=1e-6, abs=_uniform_tol(mass_coordinates))
    if not mass_coordinates:
        assert d[-1] == pytest.approx((r[-1] - r[0]) / (n - 1), rel=1e-6)


@MODES
def test_the_mass_mode_uniform_top_cell_is_not_the_radius_span_cell(mass_coordinates):
    """The rule follows the coordinate mode: on abe_solid the mass-mode top cell differs from
    the radius-span cell by more than 1 %, the radius-mode top cell does not."""
    p = Parameters.from_file(CFG)
    span_cell = (p.mesh.outer_radius - p.mesh.inner_radius) / (p.mesh.number_of_nodes - 1)
    top = np.diff(_radii(_mesh('uniform', 1e3, mass_coordinates)))[-1]
    assert bool(abs(top / span_cell - 1.0) > 0.01) is mass_coordinates


@MODES
def test_the_radii_check_refuses_a_top_cell_off_the_uniform_rule(mass_coordinates, monkeypatch):
    """A grid whose last cell is 1 % off the uniform cell is refused by the radii check with
    the field name, in both modes."""
    grid = stretched_unit_grid

    def off(n, first, last):
        s = grid(n, first, last)
        s[-2] = 1.0 - 1.01 * (1.0 - s[-2])
        return s

    monkeypatch.setattr(mesh_module, 'stretched_unit_grid', off)
    with pytest.raises(ValueError, match='mesh.surface_cell_thickness = uniform'):
        _mesh('uniform', 1e3, mass_coordinates)


@pytest.mark.physics_invariant
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=4, max_value=400),
    cmb=st.floats(min_value=0.0, max_value=6e4),
    r_in=st.floats(min_value=1e6, max_value=6.0e6),
    mass_coordinates=st.booleans(),
)
def test_a_cmb_cell_with_the_uniform_top_builds_or_names_the_field(
    n, cmb, r_in, mass_coordinates
):
    """Any CMB cell, node count and inner radius with the "uniform" top cell: either the mesh
    builds with strictly increasing radii, the bottom cell at the CMB cell when that is below
    the bottom cell of the uniform mesh and at the uniform bottom cell otherwise, and the top
    cell of the uniform mesh (to 1e-6 relative or the 2 m to which mass mode places the
    uniform radii), or a ValueError names mesh.cmb_cell_thickness."""
    try:
        m = _mesh('uniform', cmb, mass_coordinates, n, r_in)
    except ValueError as exc:
        assert 'mesh.cmb_cell_thickness' in str(exc)
        event('refused')
        return
    event('built')
    d = np.diff(_radii(m))
    assert np.all(d > 0.0)
    uni = np.diff(_radii(_mesh(0.0, 0.0, mass_coordinates, n, r_in)))
    if cmb and cmb < uni[0]:
        event('refined')
        assert d[0] == pytest.approx(cmb, rel=1e-6, abs=2e-3)
    else:
        event('uniform bottom')
        assert d[0] == uni[0]
    assert d[-1] == pytest.approx(uni[-1], rel=1e-6, abs=_uniform_tol(mass_coordinates))


@pytest.mark.parametrize(
    ('lines', 'want'),
    [
        ('', ('uniform', 3000.0)),
        ('surface_cell_thickness = 200\ncmb_cell_thickness = 500\n', (200.0, 500.0)),
        ('surface_cell_thickness = uniform\ncmb_cell_thickness = 0\n', ('uniform', 0.0)),
    ],
)
def test_the_ini_loader_reads_a_number_as_a_float_and_the_rule_as_a_string(
    tmp_path, lines, want
):
    """The INI loader converts a number to a float, not to the string of the float | str
    annotation, and keeps "uniform" as the rule."""
    text = (CFG.parent / 'abe_solid.cfg').read_text()
    assert text.count('[mesh]\n') == 1
    path = tmp_path / 'abe_solid.cfg'
    path.write_text(text.replace('[mesh]\n', '[mesh]\n' + lines))
    m = Parameters.from_file(path).mesh
    assert (m.surface_cell_thickness, m.cmb_cell_thickness) == want
    assert all(
        type(v) is type(w)
        for v, w in zip((m.surface_cell_thickness, m.cmb_cell_thickness), want)
    )
