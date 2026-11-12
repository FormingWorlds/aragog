"""A CMB cell at or above the uniform bottom cell gives no CMB refinement.

Module under test: ``aragog.mesh.Mesh`` with ``mesh.cmb_cell_thickness``. Invariants: a CMB
cell below the bottom cell of the uniform mesh of the same coordinate mode is the bottom
cell of the built mesh; a CMB cell at or above it gives the bottom of the uniform mesh,
with one INFO line that names both values, and no ValueError; with the "uniform" top cell
that is the uniform mesh bit for bit; a mesh refined at the CMB needs 4 nodes.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

import aragog
from aragog.mesh import Mesh
from aragog.parser import Parameters

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

CFG = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'
MODES = pytest.mark.parametrize('mass_coordinates', [False, True], ids=['radius', 'mass'])


def _mesh(cmb, mass_coordinates, n=None, r_in=None, top='uniform'):
    p = Parameters.from_file(CFG)
    p.mesh.mass_coordinates = mass_coordinates
    p.mesh.cmb_cell_thickness, p.mesh.surface_cell_thickness = cmb, top
    if n is not None:
        p.mesh.number_of_nodes = n
    if r_in is not None:
        p.mesh.inner_radius = r_in
    return Mesh(p)


def _radii(m):
    return np.asarray(m.basic.radii).ravel()


def _uniform_bottom(mass_coordinates, n=None, r_in=None):
    """Bottom cell [m] of the uniform mesh of the same mode."""
    return float(np.diff(_radii(_mesh(0.0, mass_coordinates, n, r_in)))[0])


@MODES
@pytest.mark.parametrize('n', [24, 100])
def test_a_cmb_cell_just_below_the_uniform_bottom_cell_refines(mass_coordinates, n):
    """0.999 of the uniform bottom cell is the bottom cell of the mesh, to 1e-6 relative."""
    h = _uniform_bottom(mass_coordinates, n)
    r = _radii(_mesh(0.999 * h, mass_coordinates, n))
    assert r[1] - r[0] == pytest.approx(0.999 * h, rel=1e-6)
    assert not np.array_equal(r, _radii(_mesh(0.0, mass_coordinates, n)))


@MODES
@pytest.mark.parametrize('n', [24, 100])
@pytest.mark.parametrize('factor', [1.0, 1.001, 2.0, 1e3])
def test_a_cmb_cell_at_or_above_the_uniform_bottom_cell_is_the_uniform_mesh(
    mass_coordinates, n, factor, caplog
):
    """At and above the uniform bottom cell the mesh with the "uniform" top is the uniform
    mesh bit for bit, and one INFO line names the CMB cell and the uniform bottom cell."""
    h = _uniform_bottom(mass_coordinates, n)
    with caplog.at_level(logging.INFO, logger='fwl.aragog.mesh'):
        r = _radii(_mesh(factor * h, mass_coordinates, n))
    assert np.array_equal(r, _radii(_mesh(0.0, mass_coordinates, n)))
    lines = [
        rec.getMessage() for rec in caplog.records if 'no CMB refinement' in rec.getMessage()
    ]
    assert len(lines) == 1
    assert 'mesh.cmb_cell_thickness' in lines[0]
    assert f'{factor * h:.6g} m' in lines[0] and f'{h:.6g} m' in lines[0]
    assert all(
        rec.levelno == logging.INFO for rec in caplog.records if rec.getMessage() in lines
    )


@MODES
def test_a_cmb_cell_above_the_uniform_bottom_cell_keeps_a_set_surface_cell(mass_coordinates):
    """With a set surface cell the CMB fallback leaves the surface refinement: the top cell
    holds its target and the bottom cell is that of the one-sided surface grid."""
    h = _uniform_bottom(mass_coordinates)
    r = _radii(_mesh(2.0 * h, mass_coordinates, top=2000.0))
    one_sided = _radii(_mesh(0.0, mass_coordinates, top=2000.0))
    assert np.array_equal(r, one_sided)
    assert r[-1] - r[-2] == pytest.approx(2000.0, rel=1e-6, abs=2e-3)


def test_the_3000_m_default_falls_back_on_a_thin_lid_shell():
    """A 25 km shell with 40 nodes has a uniform cell of 641 m: the 3000 m default gives the
    uniform mesh there; on the 1000 km bundled shell with 100 nodes (uniform cell 10.1 km) it
    gives a 3000 m bottom cell."""
    p = Parameters.from_file(CFG)
    assert p.mesh.cmb_cell_thickness == 3000.0
    thin = _radii(_mesh(3000.0, False, n=40, r_in=6.346e6))
    assert np.array_equal(thin, _radii(_mesh(0.0, False, n=40, r_in=6.346e6)))
    assert thin[1] - thin[0] == pytest.approx(25.0e3 / 39, rel=1e-12)
    thick = _radii(Mesh(p))
    assert thick[1] - thick[0] == pytest.approx(3000.0, rel=1e-6)


@MODES
def test_a_refining_cmb_cell_needs_4_nodes(mass_coordinates):
    """With 3 nodes a CMB cell below the uniform bottom cell is refused with the field name;
    with 4 nodes it refines, and one above the uniform bottom cell gives the uniform mesh."""
    with pytest.raises(ValueError, match='mesh.cmb_cell_thickness.*number_of_nodes >= 4'):
        _mesh(200.0, mass_coordinates, n=3)
    r = _radii(_mesh(200.0, mass_coordinates, n=4))
    assert r[1] - r[0] == pytest.approx(200.0, rel=1e-6, abs=2e-3)
    r = _radii(_mesh(2.0e6, mass_coordinates, n=4))
    assert np.array_equal(r, _radii(_mesh(0.0, mass_coordinates, n=4)))


@MODES
def test_a_cmb_cell_on_a_single_node_is_refused_by_name(mass_coordinates):
    """A 1-node mesh has no uniform bottom cell: a CMB cell is refused with the 4-node error,
    not an IndexError (the mass-mode density of the empty shell divides 0 by 0 on the way)."""
    with np.errstate(all='ignore'):
        with pytest.raises(ValueError, match='mesh.cmb_cell_thickness.*number_of_nodes >= 4'):
            _mesh(3000.0, mass_coordinates, n=1)


@MODES
@pytest.mark.parametrize('value', [math.nan, math.inf, -1.0, 1.0])
def test_a_cmb_cell_that_is_not_finite_or_below_2_m_is_still_refused(mass_coordinates, value):
    """The fallback does not cover NaN, inf, a negative value or a value in (0, 2 m)."""
    with pytest.raises(ValueError, match='mesh.cmb_cell_thickness'):
        _mesh(value, mass_coordinates)


@pytest.mark.physics_invariant
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=4, max_value=200),
    ratio=st.floats(min_value=1e-3, max_value=3.0),
    mass_coordinates=st.booleans(),
)
def test_the_bottom_cell_is_the_cmb_cell_below_the_uniform_cell_and_uniform_above(
    n, ratio, mass_coordinates
):
    """For a CMB cell of ratio times the uniform bottom cell (at least 2 m): below 1 the bottom
    cell is the CMB cell to 1e-6 relative, at or above 1 the mesh is the uniform mesh; or a
    ValueError names mesh.cmb_cell_thickness (the near-uniform limits of the stretched grid)."""
    h = _uniform_bottom(mass_coordinates, n)
    cmb = max(ratio * h, 2.0)
    try:
        r = _radii(_mesh(cmb, mass_coordinates, n))
    except ValueError as exc:
        assert 'mesh.cmb_cell_thickness' in str(exc)
        event('refused')
        return
    assert np.all(np.diff(r) > 0.0)
    if cmb < h:
        event('refined')
        assert r[1] - r[0] == pytest.approx(cmb, rel=1e-6, abs=2e-3)
    else:
        event('uniform')
        assert np.array_equal(r, _radii(_mesh(0.0, mass_coordinates, n)))
