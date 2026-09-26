"""Boundary fluxes in ``get_state`` output and the d/dr transform at the mesh ends.

The output heat flux at the CMB and surface basic nodes must be the flux the
right-hand side applied at the final state, for every boundary condition kind.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import Stefan_Boltzmann

from aragog.solver.entropy_solver import EntropySolver
from tests.test_entropy_solver_const_properties_smoke import _build_const_properties_parameters

pytestmark = pytest.mark.unit

F_TOP, F_CMB = 0.05, -0.0115  # prescribed fluxes [W/m^2], distinct from any state flux


def _solver(outer, inner, core_bc='quasi_steady', *, end_time=1.0, mass_coordinates=True):
    p = _build_const_properties_parameters(n_nodes=40, end_time=end_time)
    b = p.boundary_conditions
    b.outer_boundary_condition, b.inner_boundary_condition, b.core_bc = outer, inner, core_bc
    b.emissivity, b.equilibrium_temperature = 1.0, 300.0
    b.outer_boundary_value = {4: F_TOP, 5: 1500.0}.get(outer, 0.0)
    b.inner_boundary_value = {2: F_CMB, 3: 4000.0}.get(inner, 0.0)
    p.mesh.mass_coordinates = mass_coordinates
    s = EntropySolver(p, entropy_eos=None)
    s.initialize()
    if inner == 1 and core_bc == 'energy_balance':
        s.set_initial_dSdr_cmb(0.0)
    r = np.asarray(s._r_stag_flat)
    s.set_initial_entropy(3000.0 + 20.0 * (r[-1] - r) / (r[-1] - r[0]))
    return s


BCS = [
    (outer, inner, core_bc)
    for outer in (1, 4, 5)
    for inner, core_bc in ((1, 'quasi_steady'), (1, 'energy_balance'), (2, None), (3, None))
]


@pytest.mark.parametrize('outer,inner,core_bc', BCS)
def test_output_boundary_flux_equals_applied(outer, inner, core_bc):
    """Both end values of ``heat_flux`` are the fluxes the RHS applied at the final state."""
    s = _solver(outer, inner, core_bc or 'quasi_steady')
    s.solve()
    out = s.get_state()
    flux = np.asarray(out.heat_flux).ravel()
    s._dSdt_single(float(s._solution.t[-1]), np.asarray(s._solution.y[:, -1], dtype=float))
    applied = np.asarray(s.state._heat_flux).ravel()
    np.testing.assert_allclose(flux[[0, -1]], applied[[0, -1]], rtol=1e-12, atol=0.0)
    if outer == 1:
        T_top = float(np.asarray(out.T_basic).ravel()[-1])
        assert flux[-1] == pytest.approx(Stefan_Boltzmann * (T_top**4 - 300.0**4), rel=1e-12)
    elif outer == 4:
        assert flux[-1] == F_TOP
    if inner == 2:
        assert flux[0] == F_CMB


def test_f_cmb_without_eos_is_the_applied_flux():
    """Without an EOS the step integrals are skipped and F_cmb falls back to heat_flux[0]."""
    s = _solver(4, 2)
    s.solve()
    assert s.get_state().F_cmb == F_CMB


@pytest.mark.parametrize('core_bc', ['quasi_steady', 'energy_balance'])
def test_get_state_between_solves_changes_nothing(core_bc):
    """A get_state call between two coupled solves leaves the second solve unchanged."""
    finals = []
    for call_get_state in (False, True):
        s = _solver(1, 1, core_bc, end_time=5.0)
        s.solve()
        if call_get_state:
            s.get_state()
        s._S0 = np.asarray(s._solution.y[:, -1], dtype=float).copy()
        s.parameters.solver.start_time, s.parameters.solver.end_time = 5.0, 10.0
        s.solve()
        finals.append(np.asarray(s._solution.y[:, -1], dtype=float))
    np.testing.assert_array_equal(finals[0], finals[1])


@pytest.mark.parametrize('mass_coordinates', [True, False])
def test_d_dr_at_both_end_nodes(mass_coordinates):
    """A linear field gives its slope at both end basic nodes: exactly on a uniform mesh, to the
    second-order stencil error on the mass-coordinate mesh, where r is not linear in xi."""
    mesh = _solver(4, 2, mass_coordinates=mass_coordinates).evaluator.mesh
    r_stag = np.asarray(mesh.staggered.radii).ravel()
    r_basic = np.asarray(mesh.basic.radii).ravel()
    grad = mesh.d_dr_at_basic_nodes(np.atleast_2d(2.0 + 1e-3 * r_stag).T).ravel()
    np.testing.assert_allclose(grad[[0, -1]], 1e-3, rtol=1e-2 if mass_coordinates else 1e-9)
    if (
        not mass_coordinates
    ):  # the 3-point extrapolation is exact for a quadratic on a uniform mesh
        x = r_stag / 1e6
        grad = mesh.d_dr_at_basic_nodes(np.atleast_2d(x**2).T).ravel()
        np.testing.assert_allclose(grad[[0, -1]], 2.0 * r_basic[[0, -1]] / 1e12, rtol=1e-9)
