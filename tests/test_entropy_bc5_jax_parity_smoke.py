"""Outer boundary condition 5 on the real EOS tables: the surface entropy inversion and the
NumPy-JAX parity of the right-hand side."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.conftest import EOS_DIR, entropy_eos_copy, needs_eos
from tests.test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build

pytestmark = [pytest.mark.smoke, needs_eos, pytest.mark.timeout(300)]

T_SURF = 1500.0  # [K], inside the table at the surface pressure


def _bc5_solver(
    core_bc, t_surf=T_SURF, profile='nearest_boundary', smooth=0.0, strict=False, **build
):
    """A 10-node table solver under BC 5; outside core_module the entropy falls from 2900 to
    2600 J/kg/K, so every interior face has a nonzero gradient."""
    eos = entropy_eos_copy(EOS_DIR, strict_range=strict, identity_alpha=True)
    s = _build(
        core_bc,
        eos,
        CORE_MODULE_PARAMS if core_bc == 'core_module' else None,
        s_init='driven' if core_bc == 'core_module' else None,
        **build,
    )
    b = s.parameters.boundary_conditions
    b.outer_boundary_condition, b.outer_boundary_value = 5, t_surf
    s.parameters.mesh.mixing_length_profile = profile
    s.parameters.phase_mixed.matprop_smooth_width = smooth
    s.initialize()
    if core_bc != 'core_module':
        s.set_initial_entropy(np.linspace(2900.0, 2600.0, s._n_stag))
    return s


def _jax_rhs(s, y):
    jnp = pytest.importorskip('jax.numpy')
    pytest.importorskip('equinox')
    fn, args = _jax_rhs_and_args(s)
    return np.asarray(fn(0.0, jnp.asarray(y), args)).ravel()


def _jax_rhs_and_args(s, outer_bc_type=5):
    """The JAX right-hand side of the solver's core mode and its arguments under BC 5."""
    pytest.importorskip('equinox')
    from aragog.jax.solver import dSdt, dSdt_core_module, dSdt_energy_balance
    from tests.test_jax_dsdt_core_module import _build_jax_pieces

    if s._core_bc != 'core_module':
        s._core_module_budget, s._core_module_q_radio, s._core_module_ra_crit_cmb = None, 0, 0
    args = list(_build_jax_pieces(s))
    s_surf, _, half = s.surface_temperature_bc
    args[3] = dataclasses.replace(
        args[3],
        outer_bc_type=outer_bc_type,
        outer_bc_value=float(s.parameters.boundary_conditions.outer_boundary_value),
        S_surf=s_surf,
        surf_dr_half=half,
    )
    if s._core_bc == 'core_module':
        return dSdt_core_module, tuple(args)
    fn = dSdt_energy_balance if s._core_bc == 'energy_balance' else dSdt
    return fn, tuple(args[:6])


@pytest.mark.physics_invariant
def test_the_surface_entropy_gives_the_prescribed_temperature_at_the_surface_pressure():
    """The setup inverts T(P_s, S) = T_surf on the tables; the forward lookup returns T_surf."""
    s = _bc5_solver('quasi_steady')
    p_s = float(s._P_basic_flat[-1])
    t_back = float(np.ravel(s.entropy_eos.temperature(p_s, s.surface_temperature_bc[0]))[0])
    assert t_back == pytest.approx(T_SURF, rel=1e-9)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('t_surf', 'phi_range'), [(1000.0, (0, 0)), (1500.0, (0.01, 0.99)), (3000.0, (1, 1))]
)
def test_the_surface_node_holds_the_prescribed_temperature_in_every_phase(t_surf, phi_range):
    """Solid, mushy and liquid surface states: the phase evaluator at the surface node returns
    the prescribed temperature, which the update asserts on every call."""
    s = _bc5_solver('quasi_steady', t_surf=t_surf)
    s.dSdt(0.0, np.asarray(s._S0, dtype=float))
    phi = float(np.ravel(s.state.phase_basic.melt_fraction())[-1])
    assert float(np.ravel(s.state.phase_basic.temperature())[-1]) == pytest.approx(
        t_surf, rel=1e-9
    )
    assert phi_range[0] <= phi <= phi_range[1]


def test_the_update_stops_when_the_surface_node_leaves_the_prescribed_temperature():
    """A surface entropy that no longer matches the prescribed temperature stops the run."""
    s = _bc5_solver('quasi_steady')
    s_surf, _, half = s.surface_temperature_bc
    s.state.set_surface_temperature(s_surf + 50.0, T_SURF, half)
    with pytest.raises(RuntimeError, match='prescribed 1500.0 K'):
        s.dSdt(0.0, np.asarray(s._S0, dtype=float))


@pytest.mark.parametrize('t_surf', [1.0e6, 1.0e-3])
def test_a_surface_temperature_beyond_the_tables_is_refused(t_surf):
    """A temperature beyond the range the evaluator reaches is refused, and the message names
    the surface pressure (0 Pa here, not the pressure of the core-mantle boundary)."""
    with pytest.raises(ValueError, match='phase evaluator at the surface pressure 0 Pa'):
        _bc5_solver('quasi_steady', t_surf=t_surf)


_PARITY = [
    *[
        (core_bc, profile, {})
        for core_bc in ('quasi_steady', 'energy_balance', 'core_module')
        for profile in ('nearest_boundary',)
    ],
    ('quasi_steady', 'nearest_boundary', {'t_surf': 1000.0}),
    ('quasi_steady', 'nearest_boundary', {'t_surf': 3000.0}),
    ('quasi_steady', 'nearest_boundary', {'t_surf': 1445.0, 'smooth': 0.01}),
]


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('core_bc', 'profile', 'kw'), _PARITY)
def test_the_jax_rhs_matches_numpy_under_bc5(core_bc, profile, kw):
    """The JAX right-hand side of each core mode matches the NumPy one in every state slot, on
    a graded entropy profile, for solid, mushy and liquid surface nodes and with smoothed
    properties."""
    s = _bc5_solver(core_bc, profile=profile, **kw)
    y = np.asarray(s._S0, dtype=float)
    f_np = np.asarray(s.dSdt(0.0, y)).ravel()
    f_jax = _jax_rhs(s, y)
    assert f_jax.shape == f_np.shape
    np.testing.assert_allclose(f_jax[: s._n_stag], f_np[: s._n_stag], rtol=1e-10)
    slots = f_np[s._n_stag :]
    atol = 1e-8 * np.max(np.abs(slots), initial=0.0)
    np.testing.assert_allclose(f_jax[s._n_stag :], slots, rtol=1e-8, atol=atol)


@pytest.mark.physics_invariant
def test_the_jax_rhs_matches_numpy_with_separation_and_mixing():
    """With gravitational separation and chemical mixing on, both of which are zero at the
    surface face, the two right-hand sides still agree in every cell."""
    s = _bc5_solver('quasi_steady', t_surf=1445.0)
    s.parameters.energy.gravitational_separation = s.parameters.energy.mixing = True
    s.initialize()
    s.set_initial_entropy(np.linspace(2900.0, 2600.0, s._n_stag))
    y = np.asarray(s._S0, dtype=float)
    f_np = np.asarray(s.dSdt(0.0, y)).ravel()
    assert float(np.ravel(s.state._mass_flux)[-1]) == 0.0
    np.testing.assert_allclose(_jax_rhs(s, y), f_np, rtol=1e-10)


@pytest.mark.physics_invariant
def test_the_jax_jacobian_matches_a_finite_difference_of_the_numpy_rhs():
    """jax.jacrev of the BC 5 right-hand side equals a central difference of the NumPy one on
    the graded profile; the surface row differs from that of the prescribed-flux mode."""
    jax = pytest.importorskip('jax')
    s = _bc5_solver('quasi_steady')
    y = np.asarray(s._S0, dtype=float)
    fn, args = _jax_rhs_and_args(s)
    jac = np.asarray(jax.jacrev(lambda v: fn(0.0, v, args))(jax.numpy.asarray(y)))
    fd = np.empty_like(jac)
    for i in range(y.size):
        h = 1e-6 * abs(y[i])
        up, dn = y.copy(), y.copy()
        up[i] += h
        dn[i] -= h
        fd[:, i] = (np.ravel(s.dSdt(0.0, up)) - np.ravel(s.dSdt(0.0, dn))) / (2 * h)
    np.testing.assert_allclose(jac, fd, rtol=0, atol=1e-8 * np.max(np.abs(fd)))
    fn4, args4 = _jax_rhs_and_args(s, outer_bc_type=4)
    jac4 = np.asarray(jax.jacrev(lambda v: fn4(0.0, v, args4))(jax.numpy.asarray(y)))
    assert abs(jac4[-1, -1] - jac[-1, -1]) > 1e-3 * abs(jac[-1, -1])


@pytest.mark.parametrize('method', ['radau', 'bdf'])
def test_a_bc5_solve_finishes_with_the_surface_at_its_temperature(method):
    """A one-year solve under BC 5 finishes with status 0 and the surface node at T_surf."""
    s = _bc5_solver('quasi_steady', solver_method=method)
    s.solve()
    out = s.get_state()
    assert out.status == 0
    assert float(np.ravel(out.T_basic)[-1]) == pytest.approx(T_SURF, rel=1e-9)


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
    dtdr = (T_SURF - t_top_cell) / s.surface_temperature_bc[2]
    assert float(np.ravel(st_._jcond)[-1]) == pytest.approx(-k_surf * dtdr, rel=1e-10)
    dtdr_ad = float(np.ravel(st_.phase_basic.dTdPs())[-1]) * float(st_._dP_dr_basic[-1])
    assert abs(dtdr_ad / dtdr) > 0.01


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('t_surf', 'smooth'),
    [
        (1434.23, 0.01),
        (1445.0, 0.01),
        (1830.0, 0.01),
        (1400.0, 0.1),
        (1445.0, 0.1),
        (1825.0, 0.2),
        (1500.0, 0.2),
        (1130.0, 0.2),
    ],
)
def test_the_surface_node_holds_its_temperature_with_smoothed_properties(t_surf, smooth):
    """matprop_smooth_width blends the properties near the solidus (1439 K at the surface) and
    the liquidus (1831 K); at a width of 0.2 it moves the entropy of 1130 K by 72 J/kg/K, the
    largest shift from 330 to 2999 K. The surface entropy is found on the same evaluator, so
    the node sits at the prescribed temperature."""
    s = _bc5_solver('quasi_steady', t_surf=t_surf, smooth=smooth)
    f = np.asarray(s.dSdt(0.0, np.asarray(s._S0, dtype=float))).ravel()
    assert np.all(np.isfinite(f))
    t_node = float(np.ravel(s.state.phase_basic.temperature())[-1])
    assert t_node == pytest.approx(t_surf, rel=1e-9)


def test_the_surface_inversion_leaves_the_phase_evaluator_as_it_was():
    """Reading surface_temperature_bc after a change of outer_boundary_value inverts the
    surface entropy again, but the basic-node evaluator keeps the profile of the last update."""
    s = _bc5_solver('quasi_steady')
    s.dSdt(0.0, np.asarray(s._S0, dtype=float))
    before = np.array(np.ravel(s.state.phase_basic.temperature()))
    s.parameters.boundary_conditions.outer_boundary_value = 1600.0
    assert s.surface_temperature_bc[1] == 1600.0
    np.testing.assert_array_equal(np.ravel(s.state.phase_basic.temperature()), before)


@pytest.mark.parametrize('strict', [False, True])
def test_the_surface_entropy_stays_inside_the_table(strict):
    """At the temperature of the table minimum the surface entropy is the table minimum, not a
    point beyond it; below that temperature the setup refuses with a ValueError, also when the
    EOS raises on out-of-range lookups."""
    s = _bc5_solver('quasi_steady', strict=strict)
    p_s = float(s._P_basic_flat[-1])
    t_min = float(np.ravel(s.entropy_eos.temperature(p_s, s.entropy_eos.S_min))[0])
    edge = _bc5_solver('quasi_steady', t_surf=t_min, strict=strict)
    assert edge.surface_temperature_bc[0] == pytest.approx(edge.entropy_eos.S_min, abs=1e-9)
    with pytest.raises(ValueError, match='outside the range of the phase evaluator'):
        _bc5_solver('quasi_steady', t_surf=t_min - 20.0, strict=strict)


def test_a_temperature_below_the_smoothed_range_names_the_evaluator():
    """At a smoothing width of 0.5 the evaluator gives 363 K at the table minimum, above the
    320 K of the table; 340 K is refused with the range of the evaluator and the width."""
    with pytest.raises(
        ValueError, match=r'phase evaluator .* 363\.\d+ to .* \(matprop_smooth_width = 0\.5\)'
    ):
        _bc5_solver('quasi_steady', t_surf=340.0, smooth=0.5)


@pytest.mark.parametrize('t_surf', [1634.0, 1635.0, 1636.0])
def test_a_temperature_in_the_mid_melt_jump_is_refused(t_surf):
    """At a smoothing width of 0.2 the smoothed temperature jumps by 2.6 K where the melt
    fraction is 0.5 (1633.8 to 1636.4 K at the surface); no entropy gives a temperature inside
    the jump, and the setup says so; 1633 and 1637 K have an entropy."""
    with pytest.raises(ValueError, match='no entropy'):
        _bc5_solver('quasi_steady', t_surf=t_surf, smooth=0.2)
    for t_ok in (1633.0, 1637.0):
        s = _bc5_solver('quasi_steady', t_surf=t_ok, smooth=0.2)
        s.dSdt(0.0, np.asarray(s._S0, dtype=float))
        assert float(np.ravel(s.state.phase_basic.temperature())[-1]) == pytest.approx(
            t_ok, rel=1e-9
        )
