"""Tests for mesh refinement edge branches and error handling.

Exercises refined spatial coordinates, refined mass coordinates with tight root-finding
tolerance, and stretched unit grid convergence failure handling in Mesh.__init__.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aragog.mesh import Mesh
from aragog.parser import _MeshParameters


@pytest.mark.unit
def test_mesh_refined_spatial_coordinates():
    """Mesh builds refined spatial grid when mass_coordinates is False."""
    r_in = 3.480e6
    r_out = 6.371e6
    params = _MeshParameters(
        outer_radius=r_out,
        inner_radius=r_in,
        number_of_nodes=20,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
        surface_cell_thickness=1000.0,
        cmb_cell_thickness=1000.0,
        mass_coordinates=False,
    )
    mesh = Mesh(SimpleNamespace(mesh=params))
    assert mesh._refined
    assert mesh.basic.radii.shape == (20, 1)
    diffs = np.diff(mesh.basic.radii[:, 0])
    assert np.all(diffs > 0.0)
    assert np.isclose(diffs[0], 1000.0, rtol=1e-6)
    assert np.isclose(diffs[-1], 1000.0, rtol=1e-6)


@pytest.mark.unit
def test_mesh_refined_mass_coordinates():
    """Mesh builds refined mass coordinate grid with tightened root tolerance."""
    r_in = 3.480e6
    r_out = 6.371e6
    params = _MeshParameters(
        outer_radius=r_out,
        inner_radius=r_in,
        number_of_nodes=20,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
        surface_cell_thickness=1000.0,
        cmb_cell_thickness=1000.0,
        mass_coordinates=True,
    )
    mesh = Mesh(SimpleNamespace(mesh=params))
    assert mesh._refined
    assert mesh.basic.radii.shape == (20, 1)
    diffs = np.diff(mesh.basic.radii[:, 0])
    assert np.all(diffs > 0.0)
    assert np.isclose(mesh.basic.radii[0, 0], r_in)
    assert np.isclose(mesh.basic.radii[-1, 0], r_out)


@pytest.mark.unit
def test_mesh_refined_unit_grid_failure_reraises():
    """Mesh reraises ValueError when refined unit grid solve fails to converge."""
    r_in = 3.480e6
    r_out = 6.371e6
    span = r_out - r_in
    params = _MeshParameters(
        outer_radius=r_out,
        inner_radius=r_in,
        number_of_nodes=5,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
        surface_cell_thickness=0.2 * span,
        cmb_cell_thickness=1.0e-12 * span,
        mass_coordinates=False,
    )
    with pytest.raises(ValueError, match='give no refined grid of 5 nodes'):
        Mesh(SimpleNamespace(mesh=params))
