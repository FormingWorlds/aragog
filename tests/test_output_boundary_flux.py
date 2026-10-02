"""Boundary fluxes in ``get_state`` output and the d/dr transform at the mesh ends.

The output heat flux at the CMB and surface basic nodes must be the flux the
right-hand side applied at the final state, for every boundary condition the
right-hand side imposes (it does not impose inner BC 3 or outer BC 5).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import Stefan_Boltzmann

from aragog.solver.entropy_solver import EntropySolver
from tests.test_entropy_solver_const_properties_smoke import _build_const_properties_parameters

pytestmark = pytest.mark.unit

F_TOP, F_CMB = 0.05, -0.0115  # prescribed fluxes [W/m^2], distinct from any state flux
B_UTBL = 1.0e-7  # UTBL constant [K^-2]: T_surf well below T_top at ~1800 K


def _solver(
    outer, inner, core_bc='quasi_steady', *, end_time=1.0, mass_coordinates=True, utbl=False
):
    p = _build_const_properties_parameters(n_nodes=40, end_time=end_time)
    b = p.boundary_conditions
    b.outer_boundary_condition, b.inner_boundary_condition, b.core_bc = outer, inner, core_bc
    b.emissivity, b.equilibrium_temperature = 1.0, 300.0
    b.param_utbl, b.param_utbl_const = utbl, B_UTBL if utbl else 0.0
    b.outer_boundary_value = 1800.0 if outer == 5 else (F_TOP if outer == 4 else 0.0)
    b.inner_boundary_value = F_CMB if inner == 2 else 0.0
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
    for inner, core_bc in (
        (1, 'quasi_steady'),
        (1, 'energy_balance'),
        (1, 'gradient'),
        (2, None),
    )
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
        assert flux[-1] == pytest.approx(F_TOP, rel=1e-12)
    elif outer == 5:
        dr = float(s._surf_dr_half)
        k = float(np.asarray(s.state.phase_basic.thermal_conductivity()).ravel()[-1])
        T_cell = float(np.asarray(s.state.phase_staggered.temperature()).ravel()[-1])
        assert flux[-1] == pytest.approx(k * (T_cell - 1800.0) / dr, rel=1e-12)
    if inner == 2:
        assert flux[0] == pytest.approx(F_CMB, rel=1e-12)
    elif core_bc == 'quasi_steady':
        rho, cp = 4000.0, 1000.0  # const_rho, const_Cp; no internal heating
        alpha = s._cmb_radius_ratio_sq / (
            rho * cp * s._cmb_vol_first / (s._core_cap * s._core_tfac) + 1
        )
        assert flux[0] == pytest.approx(alpha * flux[1], rel=1e-12)


def test_f_cmb_without_eos_is_the_applied_flux():
    """Without an EOS the step integrals are skipped and F_cmb falls back to heat_flux[0]."""
    s = _solver(4, 2)
    s.solve()
    assert s.get_state().F_cmb == pytest.approx(F_CMB, rel=1e-12)


@pytest.mark.parametrize('core_bc', ['quasi_steady', 'energy_balance', 'gradient'])
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
    """The d/dr stencil is exact at both end basic nodes for a field quadratic in the mesh
    coordinate xi (mass radius or radius), which is uniformly spaced."""
    mesh = _solver(4, 2, mass_coordinates=mass_coordinates).evaluator.mesh
    xi_stag = np.asarray(mesh.staggered.mass_radii).ravel() / 1e6
    xi_basic = np.asarray(mesh.basic.mass_radii).ravel() / 1e6
    grad = mesh.d_dr_at_basic_nodes(np.atleast_2d(xi_stag**2).T).ravel()
    exact = 2.0 * xi_basic * np.asarray(mesh._dxidr).ravel() / 1e6
    np.testing.assert_allclose(grad[[0, -1]], exact[[0, -1]], rtol=1e-12)


@pytest.mark.parametrize('outer,utbl', [(1, False), (4, False), (1, True), (5, False)])
def test_jax_boundary_conditions_give_the_output_fluxes(outer, utbl):
    """The JAX BC functions, fed the numpy output state, return the output boundary fluxes."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from aragog.jax.phase import MeshArrays
    from aragog.jax.solver import BoundaryParams, _apply_cmb_bc, _apply_surface_bc

    s = _solver(outer, 1, utbl=utbl)
    s.solve()
    out = s.get_state()
    bc = BoundaryParams(
        outer_bc_type=outer,
        outer_bc_value=1800.0 if outer == 5 else F_TOP,
        emissivity=1.0,
        T_eq=300.0,
        inner_bc_type=1,
        inner_bc_value=0.0,
        core_density=s._core_density,
        core_heat_capacity=s._core_cp,
        tfac_core_avg=s._core_tfac,
        param_utbl=utbl,
        param_utbl_const=B_UTBL if utbl else 0.0,
    )
    mesh = MeshArrays.from_numpy_mesh(s.evaluator.mesh)
    rho = jnp.asarray(np.asarray(s.state.phase_staggered.density()).ravel())
    cp = jnp.asarray(np.asarray(s.state.phase_staggered.heat_capacity()).ravel())
    flux = jnp.asarray(np.asarray(out.heat_flux).ravel())
    T_basic = jnp.asarray(np.asarray(out.T_basic).ravel())
    heating_first = float(np.asarray(s.state.heating).flat[0])

    class _MockPhaseStag:
        temperature = jnp.asarray(np.asarray(s.state.phase_staggered.temperature()).ravel())
        thermal_conductivity = jnp.asarray(
            np.asarray(s.state.phase_staggered.thermal_conductivity()).ravel()
        )
        melt_fraction = jnp.asarray(np.asarray(s.state.phase_staggered.melt_fraction()).ravel())

    jax_flux = _apply_cmb_bc(
        _apply_surface_bc(flux, bc, T_basic, phase_stag=_MockPhaseStag(), mesh=mesh),
        bc,
        mesh,
        rho,
        cp,
        heating_first,
    )
    np.testing.assert_allclose(
        np.asarray(jax_flux)[[0, -1]], np.asarray(flux)[[0, -1]], rtol=1e-12
    )
    if utbl:
        T_top = float(T_basic[-1])
        assert flux[-1] < 0.9 * Stefan_Boltzmann * (T_top**4 - 300.0**4)


def test_boundary_params_unsupported_outer_bc_type_raises():
    """Verify BoundaryParams rejects unknown outer BC types."""
    pytest.importorskip('jax')
    from aragog.jax.solver import BoundaryParams

    with pytest.raises(ValueError, match='Unsupported outer_bc_type'):
        BoundaryParams(
            outer_bc_type=99,
            outer_bc_value=0.0,
            emissivity=1.0,
            T_eq=300.0,
            inner_bc_type=1,
            inner_bc_value=0.0,
            core_density=5000.0,
            core_heat_capacity=800.0,
            tfac_core_avg=1.0,
        )
