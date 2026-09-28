"""Staggered nodes of the mass-coordinate mesh sit at their own mass coordinate.

With ``mass_coordinates = true`` the basic radii solve ``xi(r) = xi_b``. The staggered
radii and the per-cell effective density must describe the same final mesh: each
staggered node inside its own basic cell at ``xi(r) = xi_s``, and each cell mass equal to
the integral of ``4 pi r^2 rho`` over the cell. Both the Adams-Williamson EOS and a
user-defined profile (``eos_method = 2``) are covered, the latter also with a profile
that starts below the core-mantle boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from aragog.mesh import Mesh
from aragog.parser import _MeshParameters

pytestmark = pytest.mark.unit

R_CORE, R_SURF = 3480e3, 6371e3
RHO_S, BETA = 4090.0, 1.5e-7


def _rho(r):
    """Density of the synthetic user-defined profile [kg/m^3]."""
    return RHO_S * np.exp(BETA * (R_SURF - r))


def _mesh(case, mass_coordinates=True, n_nodes=60):
    """An Earth-like mantle mesh; ``case`` is 'aw', 'user' or 'user_below_cmb'."""
    settings = _MeshParameters(
        outer_radius=R_SURF,
        inner_radius=R_CORE,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=10738.0,
        surface_density=RHO_S,
        gravitational_acceleration=9.81,
        adiabatic_bulk_modulus=260e9,
        mass_coordinates=mass_coordinates,
    )
    if case != 'aw':
        r_start = R_CORE - (0.02 * (R_SURF - R_CORE) if case == 'user_below_cmb' else 0.0)
        r = np.linspace(r_start, R_SURF, 4000)
        pressure = 9.81 * (_rho(r) - RHO_S) / BETA
        settings.eos_method = 2
        settings.eos_radius, settings.eos_pressure = r, pressure
        settings.eos_density, settings.eos_gravity = _rho(r), np.full_like(r, 9.81)
    return Mesh(SimpleNamespace(mesh=settings))


@pytest.fixture(scope='module', params=['aw', 'user', 'user_below_cmb'])
def mesh(request):
    return _mesh(request.param)


def _density(mesh):
    """Density function independent of the EOS mass integral."""
    if mesh.settings.eos_method == 1:
        return lambda r: float(mesh.eos.get_density_from_radii(r))
    return _rho


def _xi_back(mesh, radii):
    """Mass coordinate of each radius by quadrature of the density from the CMB."""
    rho = _density(mesh)
    shell = np.array([quad(lambda x: x * x * rho(x), R_CORE, r)[0] for r in radii])
    return np.cbrt(R_CORE**3 + 3.0 * shell / mesh._planet_density)


def test_basic_nodes_are_uniform_in_xi_from_cmb_to_surface(mesh):
    """The xi grid runs from R_cmb to R_surf and every basic node maps back to its xi."""
    xb = mesh.basic.mass_radii[:, 0]
    np.testing.assert_allclose(xb, np.linspace(R_CORE, R_SURF, xb.size), rtol=1e-14)
    np.testing.assert_allclose(_xi_back(mesh, mesh.basic.radii[:, 0]), xb, atol=5.0)


def test_staggered_nodes_lie_in_their_cell_at_their_mass_coordinate(mesh):
    """Each staggered node is inside its basic cell at the xi midpoint of that cell."""
    rb, xb = mesh.basic.radii[:, 0], mesh.basic.mass_radii[:, 0]
    rs, xs = mesh.staggered.radii[:, 0], mesh.staggered.mass_radii[:, 0]
    np.testing.assert_allclose(xs, 0.5 * (xb[:-1] + xb[1:]), rtol=1e-14)
    assert np.all((rs > rb[:-1]) & (rs < rb[1:]))
    # 1 m brentq tolerance in r, plus the trapezoid error of the user profile.
    np.testing.assert_allclose(_xi_back(mesh, rs), xs, atol=5.0)


@pytest.mark.physics_invariant
def test_staggered_pressure_positive_and_decreasing_outwards(mesh):
    """The staggered pressure stays positive up to the top cell and falls with radius."""
    pressure = np.asarray(mesh.eos.staggered_pressure).ravel()
    assert pressure[-1] > 0.0
    assert np.all(np.diff(pressure) < 0.0)


@pytest.mark.physics_invariant
def test_cell_masses_equal_the_shell_integral(mesh):
    """Effective density times cell volume is the mass of each cell."""
    rb = mesh.basic.radii[:, 0]
    cell_mass = np.asarray(mesh.staggered_effective_density).ravel() * (
        4.0 / 3.0 * np.pi * np.diff(rb**3)
    )
    rho = _density(mesh)
    exact = np.array(
        [4.0 * np.pi * quad(lambda x: x * x * rho(x), a, b)[0] for a, b in zip(rb[:-1], rb[1:])]
    )
    np.testing.assert_allclose(cell_mass, exact, rtol=1e-6)


@pytest.mark.parametrize('case', ['aw', 'user'])
def test_uniform_radius_mesh_keeps_midpoints_and_neighbour_density(case):
    """Without mass coordinates the staggered radii and densities are unchanged."""
    mesh = _mesh(case, mass_coordinates=False, n_nodes=20)
    rb = mesh.basic.radii[:, 0]
    np.testing.assert_allclose(mesh.staggered.radii[:, 0], 0.5 * (rb[:-1] + rb[1:]), rtol=1e-14)
    if case == 'user':
        rho_b = np.asarray(mesh.eos.basic_density).ravel()
        np.testing.assert_allclose(
            np.asarray(mesh.staggered_effective_density).ravel(),
            0.5 * (rho_b[:-1] + rho_b[1:]),
            rtol=1e-14,
        )
