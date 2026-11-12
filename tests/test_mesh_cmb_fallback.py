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
from aragog.mesh import CMB_FALLBACK_RTOL, Mesh
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
    """For a CMB cell of ratio times the uniform bottom cell (at least 2 m), with no ValueError:
    below 1 the bottom cell is the CMB cell to 1e-6 relative, at or above 1 the mesh is the
    uniform mesh."""
    h = _uniform_bottom(mass_coordinates, n)
    cmb = max(ratio * h, 2.0)
    r = _radii(_mesh(cmb, mass_coordinates, n))
    assert np.all(np.diff(r) > 0.0)
    event(_assert_refined_or_uniform(r, cmb, h, mass_coordinates, n))


def _assert_refined_or_uniform(r, cmb, h, mass_coordinates, n):
    """Bottom cell = cmb below h, uniform mesh at or above h. Either one within the fallback
    margin below h, and within 2 m of h in mass mode, where h comes from a node solve to 1 m.
    Return the branch checked."""
    uniform = np.array_equal(r, _radii(_mesh(0.0, mass_coordinates, n)))
    refined = r[1] - r[0] == pytest.approx(cmb, rel=1e-6, abs=2e-3)
    band = 2.0 if mass_coordinates else 0.0
    if h * (1 - CMB_FALLBACK_RTOL) - band <= cmb <= h + band:
        assert uniform or refined
        return 'boundary band'
    if cmb < h:
        assert refined
        return 'refined'
    assert uniform
    return 'uniform'


@MODES
@pytest.mark.parametrize('n', [24, 100, 400])
def test_a_cmb_cell_next_to_the_uniform_bottom_cell_is_never_refused(mass_coordinates, n):
    """CMB cells a hair below, at and above the uniform bottom cell build the refined or the
    uniform mesh; none gives a ValueError from the near-uniform limits of the stretched grid."""
    h = _uniform_bottom(mass_coordinates, n)
    cells = [h * (1 - f) for f in (1e-15, 1e-12, 1e-9, 1e-6, 1e-5, 1e-4)]
    cells += [h + d for d in (-2.0, -1.0, -0.1, -0.01, 0.0, 0.01, 1.0)]
    for cmb in cells:
        try:
            r = _radii(_mesh(cmb, mass_coordinates, n))
        except ValueError as exc:
            pytest.fail(f'CMB cell {cmb!r} m next to the uniform cell {h!r} m refused: {exc}')
        _assert_refined_or_uniform(r, cmb, h, mass_coordinates, n)


@pytest.mark.parametrize('top', ['uniform', 0.0], ids=['uniform-top', 'no-top'])
def test_a_cmb_cell_in_the_last_ppm_below_the_uniform_cell_is_never_refused(top):
    """A dense sweep of CMB cells 1e-16 to 1e-5 relative below the uniform cell, radius mode,
    4 to 120 and 300 nodes, plus mass mode at 40 and 80 nodes: every mesh builds, with the
    bottom cell at the CMB cell or the uniform mesh at that end. The stretched-grid solves do
    not converge in parts of this band (measured up to 2.2e-6), so the band falls back."""
    offsets = np.logspace(-16, -5, 23)
    cases = [(False, n) for n in [*range(4, 121), 300]] + [(True, 40), (True, 80)]
    for mass_coordinates, n in cases:
        h = _uniform_bottom(mass_coordinates, n)
        uniform_end = np.diff(_radii(_mesh(0.0, mass_coordinates, n, top=top)))[0]
        for f in offsets:
            cmb = h * (1 - f)
            try:
                r = _radii(_mesh(cmb, mass_coordinates, n, top=top))
            except (ValueError, RuntimeError) as exc:
                pytest.fail(f'n={n} mass={mass_coordinates} cmb={cmb!r} (1-{f:.1e}) h: {exc}')
            d0 = r[1] - r[0]
            assert d0 == uniform_end or d0 == pytest.approx(cmb, rel=1e-6, abs=2e-3), (n, f)


def test_a_surface_cell_next_to_the_uniform_cell_builds_or_names_the_fields():
    """A surface cell 1e-15 to 3e-14 relative below the uniform radial cell, with no CMB cell:
    the mesh builds, or a ValueError names the surface field; a solver RuntimeError never
    escapes."""
    for n in range(4, 121):
        h = _uniform_bottom(False, n)
        for f in (1e-15, 3e-15, 1e-14, 3e-14):
            try:
                _mesh(0.0, False, n, top=h * (1 - f))
            except ValueError as exc:  # the config check or the grid, by name
                assert 'mesh.surface_cell_thickness' in str(exc), exc
            except RuntimeError as exc:
                pytest.fail(f'n={n} surface (1-{f:.0e}) h: {type(exc).__name__}: {exc}')
