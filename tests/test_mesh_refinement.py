"""Basic mesh refined towards the surface and the CMB.

Module under test: ``aragog.mesh.stretching.stretched_unit_grid`` and its use in
``Mesh`` through ``mesh.surface_cell_thickness`` and ``mesh.cmb_cell_thickness``.
Invariants: 0 at both ends is the uniform grid; a positive value is the radial
thickness of that end cell, in radius and in mass-coordinate mode; the grid is
strictly increasing and keeps its end radii; the external EOS (``eos_method = 2``)
and the JAX mesh arrays follow the refined radii.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import aragog
from aragog.mesh import Mesh
from aragog.mesh.stretching import stretched_unit_grid
from aragog.parser import Parameters

pytestmark = pytest.mark.unit

D = 2.891e6  # Earth-like mantle depth [m]
CFG = Path(aragog.__file__).parent / 'cfg' / 'abe_solid.toml'


def _mesh(top=0.0, bottom=0.0, mass_coordinates=False, n=None):
    p = Parameters.from_file(CFG)
    p.mesh.mass_coordinates = mass_coordinates
    p.mesh.surface_cell_thickness, p.mesh.cmb_cell_thickness = top, bottom
    if n is not None:
        p.mesh.number_of_nodes = n
    return Mesh(p)


def _radii(m):
    return np.asarray(m.basic.radii).ravel()


def test_unit_grid_hits_both_end_cells_over_a_wide_range():
    """End cells of 0.5 to 16 km on a 2891 km span at 20 to 800 nodes, each end on
    or off: both targets to 1e-10, exact end points, strictly increasing; from 100
    nodes the neighbour-cell ratio stays below 1.2."""
    for n, f, l in itertools.product((20, 100, 400, 800), (0, 0.5, 2, 16), (0, 0.5, 2, 16)):
        first, last = f * 1e3 / D, l * 1e3 / D
        if max(first, last) >= 1.0 / (n - 1):
            continue
        s = stretched_unit_grid(n, first, last)
        d = np.diff(s)
        assert s[0] == 0.0 and s[-1] == 1.0 and np.all(d > 0.0)
        if first:
            assert d[0] == pytest.approx(first, rel=1e-10)
        if last:
            assert d[-1] == pytest.approx(last, rel=1e-10)
        if n >= 100:
            assert np.max(np.maximum(d[1:] / d[:-1], d[:-1] / d[1:])) < 1.2
    np.testing.assert_array_equal(stretched_unit_grid(50), np.linspace(0.0, 1.0, 50))


def test_unit_grid_rejects_cells_at_or_above_the_uniform_cell():
    for bad in (-1e-3, 1.0 / 99, 0.5, np.nan):
        with pytest.raises(ValueError, match='cell fraction'):
            stretched_unit_grid(100, bad, 0.0)
        with pytest.raises(ValueError, match='cell fraction'):
            stretched_unit_grid(100, 1e-4, bad)


@pytest.mark.parametrize('mass_coordinates', [False, True])
def test_mesh_end_cells_have_the_radial_thickness(mass_coordinates):
    """abe_solid (1000 km shell, 100 nodes, uniform cell 10.1 km): a 1 km top cell,
    a 2 km CMB cell, or both, to 1 mm; the end radii stay at the shell radii. In
    mass-coordinate mode the uniform grid has neither end cell at 1 or 2 km."""
    base = _radii(_mesh(mass_coordinates=mass_coordinates))
    assert np.diff(base)[-1] > 5e3 and np.diff(base)[0] > 5e3
    for top, bottom in ((1e3, 0.0), (0.0, 2e3), (1e3, 2e3)):
        r = _radii(_mesh(top, bottom, mass_coordinates))
        d = np.diff(r)
        assert (r[0], r[-1]) == (base[0], base[-1])
        assert np.all(d > 0.0)
        if top:
            assert d[-1] == pytest.approx(top, abs=1e-3)
        if bottom:
            assert d[0] == pytest.approx(bottom, abs=1e-3)
        # The other end moves away from the refined end: the cells there grow.
        if not bottom:
            assert d[0] > np.diff(base)[0]
        if not top:
            assert d[-1] > np.diff(base)[-1]


def test_mesh_rejects_a_cell_thicker_than_the_uniform_cell():
    p = Parameters.from_file(CFG)
    with pytest.raises(ValueError, match='surface_cell_thickness'):
        type(p.mesh)(**{**vars(p.mesh), 'surface_cell_thickness': 20e3})
    with pytest.raises(ValueError, match='cmb_cell_thickness'):
        type(p.mesh)(**{**vars(p.mesh), 'cmb_cell_thickness': -1.0})


@pytest.mark.physics_invariant
def test_derivative_operator_is_exact_for_a_linear_field_on_the_refined_grid():
    """d/dr of a staggered field linear in radius equals the slope at the basic
    nodes on the refined grid (top 1 km, CMB 2 km), the inner extrapolated node
    included. The outer extrapolated row is left out: it has the opposite sign on
    the uniform grid as well."""
    m = _mesh(1e3, 2e3)
    rs = np.asarray(m.staggered.radii)
    grad = (m._d_dr_transform @ (3.0e-3 * rs)).ravel()
    np.testing.assert_allclose(grad[:-1], 3.0e-3, rtol=1e-9)


def test_external_eos_and_jax_arrays_follow_the_refined_radii(tmp_path):
    """``eos_method = 2`` (a 4-column r, P, rho, g file), mass-coordinate mode, top
    1 km: the basic pressure is the PCHIP of the file at the refined radii and the
    JAX ``MeshArrays`` carry the same radii, pressure and gravity."""
    pytest.importorskip('jax')
    from aragog.jax.phase import MeshArrays
    from tests.test_entropy_solver_eos_method2_smoke import (
        _build_parameters_eos_method_2,
        _write_synthetic_external_mesh,
    )

    ef = tmp_path / 'mesh.dat'
    _write_synthetic_external_mesh(ef, n=30)
    p = _build_parameters_eos_method_2(ef, n_nodes=40)
    p.mesh.mass_coordinates = True
    p.mesh.surface_cell_thickness = 1e3
    m = Mesh(p)
    r = _radii(m)
    assert np.diff(r)[-1] == pytest.approx(1e3, abs=1e-3)
    tab = np.loadtxt(ef)
    P_ref = PchipInterpolator(tab[:, 0], tab[:, 1])(r)
    np.testing.assert_allclose(np.asarray(m.basic_pressure).ravel(), P_ref, rtol=1e-12)
    ma = MeshArrays.from_numpy_mesh(m)
    np.testing.assert_array_equal(np.asarray(ma.radii_basic), r)
    np.testing.assert_array_equal(np.asarray(ma.P_basic), np.asarray(m.basic_pressure).ravel())
    np.testing.assert_allclose(
        np.asarray(ma.gravity), PchipInterpolator(tab[:, 0], tab[:, 3])(r), rtol=1e-12
    )
