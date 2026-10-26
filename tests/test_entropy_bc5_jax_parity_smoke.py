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
