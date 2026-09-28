"""Staggered nodes of the mass-coordinate mesh sit at their own mass coordinate.

With ``mass_coordinates = true`` the basic radii solve ``xi(r) = xi_b`` exactly. The
staggered radii and the per-cell effective density must describe the same final mesh:
each staggered node inside its own basic cell at ``xi(r) = xi_s``, and the cell masses
adding up to the mantle mass between the core and the surface.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aragog.mesh import Mesh
from aragog.parser import _MeshParameters

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _mesh(n_nodes=60):
    """An Earth-like Adams-Williamson mantle on the default mass-coordinate mesh."""
    settings = _MeshParameters(
        outer_radius=6371e3,
        inner_radius=3480e3,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=10738.0,
        surface_density=4090.0,
        gravitational_acceleration=9.81,
        adiabatic_bulk_modulus=260e9,
    )
    return Mesh(SimpleNamespace(mesh=settings))


def _xi_of_r(mesh, r):
    """The mass coordinate of radius ``r``, as the basic-node solve defines it."""
    r_core = mesh.basic.radii[0, 0]
    shell = mesh.eos.get_mass_within_radii(np.array([r])).item() / (4.0 * np.pi)
    return (r_core**3 + 3.0 * shell / mesh._planet_density) ** (1.0 / 3.0)


def test_staggered_nodes_lie_in_their_cell_at_their_mass_coordinate():
    """Each staggered radius is inside its basic cell and maps back to its xi."""
    mesh = _mesh()
    rb = mesh.basic.radii[:, 0]
    rs = mesh.staggered.radii[:, 0]
    xs = mesh.staggered.mass_radii[:, 0]

    assert np.all((rs > rb[:-1]) & (rs < rb[1:]))
    xi_back = np.array([_xi_of_r(mesh, r) for r in rs])
    np.testing.assert_allclose(xi_back, xs, atol=2.0)  # metres, as the basic solve
    assert mesh.eos.staggered_pressure[-1, 0] > 0.0


def test_cell_masses_add_up_to_the_mantle_mass():
    """Effective density times cell volume gives each cell's own mass."""
    mesh = _mesh()
    rb = mesh.basic.radii[:, 0]
    cell_mass = (
        np.asarray(mesh.staggered_effective_density).ravel()
        * 4.0
        / 3.0
        * np.pi
        * np.diff(rb**3)
    )
    exact = np.diff([mesh.eos.get_mass_within_radii(np.array([r])).item() for r in rb])

    np.testing.assert_allclose(cell_mass, exact, rtol=1e-6)
