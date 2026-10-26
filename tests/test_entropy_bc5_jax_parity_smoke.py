"""Outer boundary condition 5 on the real EOS tables: the surface entropy inversion and the
NumPy-JAX parity of the right-hand side."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import EOS_DIR, entropy_eos_copy, needs_eos
from tests.test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build

pytestmark = [pytest.mark.smoke, needs_eos, pytest.mark.timeout(300)]

T_SURF = 1500.0  # [K], inside the table at the surface pressure


def _bc5_solver(core_bc, t_surf=T_SURF, profile='nearest_boundary'):
    eos = entropy_eos_copy(EOS_DIR, identity_alpha=True)
    s = _build(
        core_bc,
        eos,
        CORE_MODULE_PARAMS if core_bc == 'core_module' else None,
        s_init='driven' if core_bc == 'core_module' else None,
    )
    b = s.parameters.boundary_conditions
    b.outer_boundary_condition, b.outer_boundary_value = 5, t_surf
    s.parameters.mesh.mixing_length_profile = profile
    s.initialize()
    if core_bc != 'core_module':
        s.set_initial_entropy(2700.0)
    return s


def _jax_rhs(s, y):
    jnp = pytest.importorskip('jax.numpy')
    pytest.importorskip('equinox')
    from aragog.jax.solver import BoundaryParams, dSdt, dSdt_core_module, dSdt_energy_balance
    from tests.test_jax_dsdt_core_module import _build_jax_pieces

    if s._core_bc != 'core_module':
        s._core_module_budget, s._core_module_q_radio, s._core_module_ra_crit_cmb = None, 0, 0
    args = list(_build_jax_pieces(s))
    old = args[3]
    args[3] = BoundaryParams(
        outer_bc_type=5,
        outer_bc_value=float(s.parameters.boundary_conditions.outer_boundary_value),
        emissivity=old.emissivity,
        T_eq=old.T_eq,
        inner_bc_type=old.inner_bc_type,
        inner_bc_value=old.inner_bc_value,
        core_density=old.core_density,
        core_heat_capacity=old.core_heat_capacity,
        tfac_core_avg=old.tfac_core_avg,
        cmb_area=old.cmb_area,
        core_M=old.core_M,
        cmb_dr_cmb=old.cmb_dr_cmb,
        S_surf=float(s._outer_bc_S_surf),
        surf_dr_half=float(s._outer_bc_dr_half),
    )
    if s._core_bc == 'core_module':
        return np.asarray(dSdt_core_module(0.0, jnp.asarray(y), tuple(args))).ravel()
    fn = dSdt_energy_balance if s._core_bc == 'energy_balance' else dSdt
    return np.asarray(fn(0.0, jnp.asarray(y), tuple(args[:6]))).ravel()


@pytest.mark.physics_invariant
def test_the_surface_entropy_gives_the_prescribed_temperature_at_the_surface_pressure():
    """The setup inverts T(P_s, S) = T_surf on the tables; the forward lookup returns T_surf."""
    s = _bc5_solver('quasi_steady')
    p_s = float(s._P_basic_flat[-1])
    t_back = float(np.ravel(s.entropy_eos.temperature(p_s, s._outer_bc_S_surf))[0])
    assert t_back == pytest.approx(T_SURF, rel=1e-9)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('t_surf', [1000.0, 1500.0, 3000.0])
def test_the_surface_node_holds_the_prescribed_temperature_in_every_phase(t_surf):
    """Solid, mushy and liquid surface states: the phase evaluator at the surface node returns
    the prescribed temperature, which the update asserts on every call."""
    s = _bc5_solver('quasi_steady', t_surf=t_surf)
    s.dSdt(0.0, np.asarray(s._S0, dtype=float))
    phi = float(np.ravel(s.state.phase_basic.melt_fraction())[-1])
    assert float(np.ravel(s.state.phase_basic.temperature())[-1]) == pytest.approx(
        t_surf, rel=1e-9
    )
    assert (phi == 0.0, 0.0 < phi < 1.0, phi == 1.0)[[1000.0, 1500.0, 3000.0].index(t_surf)]


def test_the_update_stops_when_the_surface_node_leaves_the_prescribed_temperature():
    """A surface entropy that no longer matches the prescribed temperature stops the run."""
    s = _bc5_solver('quasi_steady')
    s.state.set_surface_temperature(s._outer_bc_S_surf + 50.0, T_SURF, s._outer_bc_dr_half)
    with pytest.raises(RuntimeError, match='prescribed 1500.0 K'):
        s.dSdt(0.0, np.asarray(s._S0, dtype=float))


@pytest.mark.parametrize('t_surf', [1.0e6, 1.0e-3])
def test_a_surface_temperature_beyond_the_tables_is_refused(t_surf):
    """The inversion clamps to the table edge beyond the range; the setup refuses that value
    instead of running at another temperature."""
    with pytest.raises(ValueError, match='outside the entropy tables'):
        _bc5_solver('quasi_steady', t_surf=t_surf)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('profile', ['nearest_boundary', 'constant'])
@pytest.mark.parametrize('core_bc', ['quasi_steady', 'energy_balance', 'core_module'])
def test_the_jax_rhs_matches_numpy_under_bc5(core_bc, profile):
    """The JAX right-hand side of each core mode matches the NumPy one in every mantle cell,
    and the surface face is not the prescribed value read as a flux."""
    s = _bc5_solver(core_bc, profile=profile)
    y = np.asarray(s._S0, dtype=float)
    n = s._n_stag
    f_np = np.asarray(s.dSdt(0.0, y)).ravel()
    f_jax = _jax_rhs(s, y)
    np.testing.assert_allclose(f_jax[:n], f_np[:n], rtol=1e-10)
    assert float(np.ravel(s.state.heat_flux)[-1]) != pytest.approx(T_SURF, rel=1e-3)


@pytest.mark.physics_invariant
def test_the_conductive_face_flux_is_the_half_cell_difference_on_the_tables():
    """On the tables the adiabatic term dT/dP|_S dP/dr is about a fifth of the half-cell
    gradient at the surface; the entropy gradient still gives k (T_stag - T_surf) / (dr / 2)
    from the first call, and dropping the adiabatic term would miss it by more than 1 %."""
    s = _bc5_solver('quasi_steady')
    s.dSdt(0.0, np.asarray(s._S0, dtype=float))
    st_ = s.state
    t_top_cell = float(np.ravel(st_.phase_staggered.temperature())[-1])
    k_surf = float(np.ravel(st_.phase_basic.thermal_conductivity())[-1])
    dtdr = (T_SURF - t_top_cell) / s._outer_bc_dr_half
    assert float(np.ravel(st_._jcond)[-1]) == pytest.approx(-k_surf * dtdr, rel=1e-10)
    dtdr_ad = float(np.ravel(st_.phase_basic.dTdPs())[-1]) * float(st_._dP_dr_basic[-1])
    assert abs(dtdr_ad / dtdr) > 0.01
