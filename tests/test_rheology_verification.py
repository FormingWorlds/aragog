"""Algebraic pins for the solid-state rheology reference parameters.

Module under test: ``aragog.rheology`` and ``aragog.rheology_lid`` in NumPy
and JAX (float64) modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.config.phases import MixedPhaseConfig, PhaseConfig
from aragog.rheology import (
    SolidRheologyParams,
    compute_arrhenius_viscosity,
    compute_strain_rate_local,
    compute_yield_stress,
    eta_diff,
    eta_eff,
)
from aragog.rheology_lid import compute_effective_viscosity, stress_closure

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_ETA0 = 1.0e21
_E_A = 300.0e3
_V_A = 5.0e-6
_T_REF = 1600.0
_R = 8.314462618

_TAU_C = 50.0e6
_TAU_MU = 0.6
_TAU_MAX = 500.0e6


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_arrhenius_viscosity_collapses_to_reference_at_t_ref():
    """At the reference temperature with no pressure term, eta_diff equals eta_0."""
    eta_at_ref = eta_diff(_T_REF, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R)
    assert eta_at_ref == pytest.approx(_ETA0, rel=1e-12)

    cold = 0.9 * _T_REF
    eta_cold = eta_diff(cold, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R)
    expected_cold = _ETA0 * np.exp(_E_A / _R * (1.0 / cold - 1.0 / _T_REF))
    assert eta_cold == pytest.approx(expected_cold, rel=1e-12)
    assert eta_cold > 10.0 * _ETA0

    eta_hot = eta_diff(1.1 * _T_REF, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R)
    assert eta_hot < _ETA0 < eta_cold
    assert eta_hot > 0.0

    eta_hot_limit = eta_diff(1.0e15, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R)
    floor = _ETA0 * np.exp(-_E_A / (_R * _T_REF))
    assert eta_hot_limit == pytest.approx(floor, rel=1e-9)
    assert 0.0 < eta_hot_limit < _ETA0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_arrhenius_pressure_term_uses_reference_at_t_ref():
    """At T = T_ref the temperature term cancels, leaving pure pressure activation."""
    p_test = 100.0e9
    eta_p = eta_diff(_T_REF, p_test, _ETA0, _E_A, _V_A, _T_REF, _R)
    expected = _ETA0 * np.exp(p_test * _V_A / (_R * _T_REF))
    assert eta_p == pytest.approx(expected, rel=1e-12)
    assert eta_p > 1.5 * _ETA0
    assert eta_p > 0.0
    assert eta_diff(_T_REF, 0.0, _ETA0, _E_A, _V_A, _T_REF, _R) == pytest.approx(
        _ETA0, rel=1e-12
    )


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_yield_stress_saturates_at_maximum():
    """The Byerlee yield stress rises linearly with pressure then saturates at yield_stress_max."""
    p_knee = (_TAU_MAX - _TAU_C) / _TAU_MU

    p_lin = 0.5 * p_knee
    tau_lin = compute_yield_stress(p_lin, _TAU_C, _TAU_MU, _TAU_MAX)
    assert tau_lin == pytest.approx(_TAU_C + _TAU_MU * p_lin, rel=1e-12)
    assert tau_lin < _TAU_MAX

    p_high = 1.0e12
    tau_high = compute_yield_stress(p_high, _TAU_C, _TAU_MU, _TAU_MAX)
    assert tau_high == pytest.approx(_TAU_MAX, rel=1e-12)
    unclamped = _TAU_C + _TAU_MU * p_high
    assert unclamped > 1000.0 * _TAU_MAX

    p_sweep = np.logspace(5, 13, 40)
    tau_sweep = compute_yield_stress(p_sweep, _TAU_C, _TAU_MU, _TAU_MAX)
    assert np.all(tau_sweep <= _TAU_MAX + 1.0)
    assert np.all(tau_sweep > 0.0)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_yield_stress_knee_is_continuous():
    """The linear and saturated branches meet continuously at the knee."""
    p_knee = (_TAU_MAX - _TAU_C) / _TAU_MU
    tau_below = compute_yield_stress(0.999 * p_knee, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_at = compute_yield_stress(p_knee, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_above = compute_yield_stress(1.001 * p_knee, _TAU_C, _TAU_MU, _TAU_MAX)

    assert tau_at == pytest.approx(_TAU_MAX, rel=1e-12)
    assert tau_below == pytest.approx(_TAU_C + _TAU_MU * 0.999 * p_knee, rel=1e-12)
    assert tau_above == pytest.approx(_TAU_MAX, rel=1e-12)
    assert tau_below <= tau_at <= tau_above + 1.0
    assert tau_above <= _TAU_MAX + 1.0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_numpy_jax_rheology_float64_parity():
    """The NumPy and JAX rheology kernels agree to float64 parity."""
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    jax.config.update('jax_enable_x64', True)

    t_test = 0.85 * _T_REF
    p_test = 40.0e9

    eta_np = eta_diff(t_test, p_test, _ETA0, _E_A, _V_A, _T_REF, _R)
    eta_jx = float(
        compute_arrhenius_viscosity(
            jnp.asarray(t_test),
            jnp.asarray(p_test),
            _ETA0,
            _E_A,
            _V_A,
            T_ref=_T_REF,
            R=_R,
            xp=jnp,
        )
    )
    assert eta_jx == pytest.approx(eta_np, rel=1e-12)
    assert eta_jx > 0.0

    p_clamp = 1.0e12
    tau_np = compute_yield_stress(p_clamp, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_jx = float(
        compute_yield_stress(jnp.asarray(p_clamp), _TAU_C, _TAU_MU, _TAU_MAX, xp=jnp)
    )
    assert tau_jx == pytest.approx(tau_np, rel=1e-12)
    assert tau_jx == pytest.approx(_TAU_MAX, rel=1e-12)

    p_lin = 0.5 * (_TAU_MAX - _TAU_C) / _TAU_MU
    tau_np_lin = compute_yield_stress(p_lin, _TAU_C, _TAU_MU, _TAU_MAX)
    tau_jx_lin = float(
        compute_yield_stress(jnp.asarray(p_lin), _TAU_C, _TAU_MU, _TAU_MAX, xp=jnp)
    )
    assert tau_jx_lin == pytest.approx(tau_np_lin, rel=1e-12)


def test_viscosity_max_log10_clipping():
    """Verify viscosity_max_log10 caps cold Arrhenius viscosity without overflow."""
    T = 40.0
    P = 0.0
    eta_d = eta_diff(
        temperature=T,
        pressure=P,
        viscosity_solid=1.0e21,
        activation_energy=300e3,
        activation_volume=5e-6,
        t_ref=1600.0,
        r_gas=8.314462618,
        viscosity_max_log10=40.0,
    )
    assert eta_d == pytest.approx(1e40, rel=1e-12)

    eta_e = eta_eff(
        visc_diff=eta_d,
        tau_y=50e6,
        strain_rate=1e-15,
    )
    assert eta_e == pytest.approx(50e6 / (2.0 * 1e-15), rel=1e-12)


def test_lid_base_mode_and_stress_closure_validation():
    """Verify validation of lid_base_mode, stress_closure_mode, and invalid combinations."""
    with pytest.raises(ValueError, match="Unknown stress_closure_mode 'invalid'"):
        SolidRheologyParams(stress_closure_mode='invalid')

    with pytest.raises(ValueError, match="Unknown lid_base_mode 'invalid'"):
        SolidRheologyParams(lid_base_mode='invalid')

    with pytest.raises(ValueError, match='lid_base_mode.*activation_energy'):
        SolidRheologyParams(enabled=True, activation_energy=0.0, lid_base_mode='rheological')

    p = SolidRheologyParams(enabled=False, activation_energy=0.0, lid_base_mode='rheological')
    assert p.lid_base_mode == 'rheological'

    with pytest.raises(ValueError, match='Unknown stress_closure_mode'):
        compute_effective_viscosity(eta_diff=1.0e21, stress_closure_mode='global')

    with pytest.raises(ValueError, match='mixing_length and unyielded_velocity required'):
        compute_effective_viscosity(eta_diff=1.0e21, stress_closure_mode='local')

    with pytest.raises(ValueError, match='mixing_length must be provided'):
        stress_closure('local', viscous_velocity=1.0e-9, mixing_length=None)

    with pytest.raises(ValueError, match="Unknown stress closure mode 'global'"):
        stress_closure('global', viscous_velocity=1.0e-9, mixing_length=1.0e5)

    with pytest.raises(ValueError, match="Mode 'lid' uses boundary-layer stress closure"):
        stress_closure('lid', viscous_velocity=1.0e-9, mixing_length=1.0e5)

    with pytest.raises(ValueError, match='Unknown stress closure mode'):
        stress_closure('other', viscous_velocity=1.0e-9, mixing_length=1.0e5)

    sr = stress_closure('local', viscous_velocity=2.0e-9, mixing_length=1.0e5)
    assert sr == pytest.approx(1.0e-14, rel=1e-12)
    assert compute_strain_rate_local(2.0e-9, 1.0e5) == pytest.approx(1.0e-14, rel=1e-12)


@pytest.mark.physics_invariant
def test_eta_eff_infinite_yield_stress():
    """Verify eta_eff handles infinite yield stress without NaN evaluation."""
    visc_diff = 1.0e21
    tau_y_inf = np.inf
    sr = 1.0e-15

    eta_smooth = eta_eff(visc_diff, tau_y_inf, sr, smooth=True)
    assert not np.isnan(eta_smooth)
    assert eta_smooth == pytest.approx(visc_diff, rel=1.0e-12)

    eta_sharp = eta_eff(visc_diff, tau_y_inf, sr, smooth=False)
    assert not np.isnan(eta_sharp)
    assert eta_sharp == pytest.approx(visc_diff, rel=1.0e-12)


def test_phase_config_yield_stress_max_and_viscosity_max():
    """Verify yield_stress_max and viscosity_max_log10 are stored and validated on PhaseConfig."""
    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        enabled=True,
        arrhenius_t_ref=1500.0,
        yield_stress_max=200.0e6,
        viscosity_max_log10=28.0,
    )
    assert p_cfg.yield_stress_max == 200.0e6
    assert p_cfg.viscosity_max_log10 == 28.0

    with pytest.raises(TypeError, match='unexpected keyword argument'):
        MixedPhaseConfig(
            latent_heat_of_fusion=4e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='solidus.dat',
            liquidus='liquidus.dat',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1e-3,
            enabled=True,
            yield_stress_max=150.0e6,
            viscosity_max_log10=30.0,
        )


@pytest.mark.physics_invariant
def test_numpy_jax_crossover_sweep_parity():
    """Verify numpy and JAX Arrhenius viscosity match across the viscosity_max_log10 cap."""
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    jax.config.update('jax_enable_x64', True)

    temperatures = np.linspace(400.0, 700.0, 31)
    for t_val in temperatures:
        eta_np = eta_diff(
            temperature=t_val,
            pressure=0.0,
            viscosity_solid=_ETA0,
            activation_energy=_E_A,
            activation_volume=_V_A,
            t_ref=_T_REF,
            r_gas=_R,
            viscosity_max_log10=40.0,
        )
        eta_jx = float(
            compute_arrhenius_viscosity(
                jnp.asarray(t_val),
                jnp.asarray(0.0),
                _ETA0,
                _E_A,
                _V_A,
                T_ref=_T_REF,
                R=_R,
                viscosity_max_log10=40.0,
                xp=jnp,
            )
        )
        assert eta_jx == pytest.approx(eta_np, rel=1e-11)


@pytest.mark.physics_invariant
def test_compute_effective_viscosity_infinite_yield_stress_jax():
    """Verify compute_effective_viscosity handles infinite yield stress in JAX."""
    jnp = pytest.importorskip('jax.numpy')

    visc_diff = 1.0e21
    tau_y_inf = jnp.inf
    sr = 1.0e-15

    eta_eff_val = compute_effective_viscosity(
        jnp.asarray(visc_diff), tau_y=tau_y_inf, strain_rate=sr, xp=jnp
    )
    assert jnp.isfinite(eta_eff_val)
    assert float(eta_eff_val) == pytest.approx(visc_diff, rel=1.0e-12)


def test_arrhenius_t_ref_positivity_validation():
    """Verify non-positive arrhenius_t_ref is rejected."""
    with pytest.raises(ValueError, match='t_ref must be positive'):
        eta_diff(temperature=1600.0, pressure=0.0, t_ref=0.0)

    with pytest.raises(ValueError, match='t_ref must be positive'):
        eta_diff(temperature=1600.0, pressure=0.0, t_ref=-100.0)

    with pytest.raises(ValueError):
        PhaseConfig(
            density=3000.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=2e-5,
            viscosity=1e21,
            arrhenius_t_ref=0.0,
        )


def test_phase_config_rheological_lid_requires_positive_activation_energy():
    """Verify PhaseConfig rejects zero activation energy with rheological lid."""
    with pytest.raises(ValueError, match='requires non-zero activation_energy'):
        PhaseConfig(
            density=3000.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=2e-5,
            viscosity=1e21,
            enabled=True,
            lid_base_mode='rheological',
            activation_energy=0.0,
        )


def test_water_prefactor_scalar_and_array_parity():
    """Verify water_prefactor scales viscosity linearly for scalar and array in numpy and JAX."""
    jnp = pytest.importorskip('jax.numpy')

    t = np.array([1600.0, 1800.0])
    p = np.array([0.0, 1e9])

    eta_dry = eta_diff(t, p, water_prefactor=1.0)
    eta_wet_scalar = eta_diff(t, p, water_prefactor=0.1)
    np.testing.assert_allclose(eta_wet_scalar, 0.1 * eta_dry, rtol=1e-12)

    wp_arr = np.array([0.5, 0.2])
    eta_wet_arr = eta_diff(t, p, water_prefactor=wp_arr)
    np.testing.assert_allclose(eta_wet_arr, wp_arr * eta_dry, rtol=1e-12)

    t_j = jnp.asarray(t)
    p_j = jnp.asarray(p)
    eta_j_dry = compute_arrhenius_viscosity(t_j, p_j, water_prefactor=1.0, xp=jnp)
    eta_j_wet_scalar = compute_arrhenius_viscosity(t_j, p_j, water_prefactor=0.1, xp=jnp)
    eta_j_wet_arr = compute_arrhenius_viscosity(
        t_j, p_j, water_prefactor=jnp.asarray(wp_arr), xp=jnp
    )

    np.testing.assert_allclose(np.asarray(eta_j_dry), eta_dry, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(eta_j_wet_scalar), eta_wet_scalar, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(eta_j_wet_arr), eta_wet_arr, rtol=1e-12)

    with pytest.raises(ValueError, match='water_prefactor must be positive'):
        eta_diff(t, p, water_prefactor=0.0)
    with pytest.raises(ValueError, match='water_prefactor must be positive'):
        eta_diff(t, p, water_prefactor=-0.5)


def test_phi_visc_single_parameter_control():
    """Verify phi_visc_single controls the single-phase transition threshold."""
    default_p = SolidRheologyParams()
    assert default_p.phi_visc_single == 0.5

    custom_p = SolidRheologyParams(phi_visc_single=0.7)
    assert custom_p.phi_visc_single == 0.7

    with pytest.raises(ValueError, match='phi_visc_single must be in'):
        SolidRheologyParams(enabled=True, phi_visc_single=0.0)
    with pytest.raises(ValueError, match='phi_visc_single must be in'):
        SolidRheologyParams(enabled=True, phi_visc_single=1.0)
    with pytest.raises(ValueError, match='phi_visc_single must be in'):
        SolidRheologyParams(enabled=True, phi_visc_single=-0.1)


@pytest.mark.physics_invariant
def test_effective_viscosity_unyielded_and_yielded_limits():
    """Verify effective viscosity limits in local stress closure mode with min closure."""
    eta_d = 1.0e21
    tau_y = 1.0e8

    sr_match = tau_y / (2.0 * eta_d)
    eta_match = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_match,
        stress_closure_mode='local',
    )
    assert eta_match == pytest.approx(eta_d, rel=1.0e-12)

    sr_low = 1.0e-18
    eta_below = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_low,
        stress_closure_mode='local',
    )
    assert eta_below == eta_d

    sr_high = 1.0e-10
    eta_plastic = tau_y / (2.0 * sr_high)
    eta_yielded = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_high,
        stress_closure_mode='local',
    )
    assert eta_yielded == eta_plastic

    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    jax.config.update('jax_enable_x64', True)

    eta_match_jax = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_match,
        stress_closure_mode='local',
        xp=jnp,
    )
    assert float(eta_match_jax) == pytest.approx(eta_d, rel=1.0e-12)

    eta_below_jax = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_low,
        stress_closure_mode='local',
        xp=jnp,
    )
    assert float(eta_below_jax) == pytest.approx(eta_d, rel=1.0e-12)

    eta_yielded_jax = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_y=tau_y,
        strain_rate=sr_high,
        stress_closure_mode='local',
        xp=jnp,
    )
    assert float(eta_yielded_jax) == pytest.approx(eta_plastic, rel=1.0e-12)


@pytest.mark.physics_invariant
def test_moresi_solomatov_min_closure_exact_contract():
    """Verify Moresi & Solomatov (1998) min closure exact contracts."""
    eta_d = 8.3e24
    tau_y = 1.0e6

    # 1. Below yield (tau_d < tau_y): eta_eff == eta_diff.
    td_below = 8.645e5
    eta_eff_below = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_d=td_below,
        tau_y_lid=tau_y,
    )
    assert eta_eff_below == eta_d

    # 2. Above yield (tau_d >= tau_y): eta_eff / eta_diff == tau_y / tau_d.
    td_above = 2.0e6
    eta_eff_above = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_d=td_above,
        tau_y_lid=tau_y,
    )
    expected_ratio = tau_y / td_above
    assert (eta_eff_above / eta_d) == pytest.approx(expected_ratio, rel=1.0e-14)

    # 3. Continuous at tau_d = tau_y.
    eps = 1.0e-6
    eta_left = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_d=tau_y - eps,
        tau_y_lid=tau_y,
    )
    eta_mid = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_d=tau_y,
        tau_y_lid=tau_y,
    )
    eta_right = compute_effective_viscosity(
        eta_diff=eta_d,
        tau_d=tau_y + eps,
        tau_y_lid=tau_y,
    )
    assert eta_left == eta_d
    assert eta_mid == eta_d
    assert eta_right == pytest.approx(eta_d, rel=1.0e-11)


def test_rheology_and_lid_branch_coverage():
    """Verify remaining parameter bounds, module attributes, and lid branches."""
    import aragog.rheology as rheo_mod
    from aragog.rheology_lid import compute_stagnant_lid_state

    assert 'compute_stagnant_lid_state' in dir(rheo_mod)
    with pytest.raises(AttributeError, match='has no attribute'):
        _ = rheo_mod.nonexistent_attribute

    assert compute_arrhenius_viscosity(1600.0, 0.0, t_ref=1600.0) == pytest.approx(
        1.0e21, rel=1e-12
    )
    assert np.isnan(compute_arrhenius_viscosity(np.nan, 0.0))
    assert compute_yield_stress(-200.0e6, yield_stress_c=50.0e6, yield_stress_mu=0.6) == 0.0
    assert eta_eff(np.inf, 50.0e6, 1.0e-15, smooth=True) == pytest.approx(2.5e22, rel=1e-12)

    invalid_cases = [
        {'activation_volume': -1.0e-6},
        {'activation_volume_decay_pressure': 0.0},
        {'viscosity_max_log10': 310.0},
        {'yield_stress_c': -1.0},
        {'yield_stress_mu': -0.1},
        {'yield_stress_max': 0.0},
        {'interior_flux_fraction': 0.0},
        {'lid_base_temperature': 0.0},
        {'lid_contrast_coeff': 0.0},
        {'lid_mask_width_cells': 0.0},
    ]
    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            SolidRheologyParams(**kwargs)

    n_nodes = 20
    r = np.linspace(3.48e6, 6.371e6, n_nodes)
    T = np.linspace(2600.0, 1200.0, n_nodes)
    P = np.linspace(135.0e9, 1.0e5, n_nodes)
    F_conv = np.full(n_nodes, 50.0)
    F_conv[-4:] = 0.0
    F_tot = np.full(n_nodes, 60.0)
    T_sol = np.linspace(3000.0, 1400.0, n_nodes)
    phi = np.zeros(n_nodes)
    params_fixed = SolidRheologyParams(
        enabled=True, lid_base_mode='fixed', lid_base_temperature=1350.0
    )
    st = compute_stagnant_lid_state(
        radii=r,
        temperature=T,
        pressure=P,
        convective_flux=F_conv,
        total_flux=F_tot,
        solidus_temperature=T_sol,
        melt_fraction=phi,
        params=params_fixed,
        viscosity_mixture=1.0e20,
        density=np.full(n_nodes, 3300.0),
    )
    assert st['T_lid_iso'] == 1350.0
    assert float(st['mu_i']) == pytest.approx(1.0e20, rel=1e-12)
    assert float(st['d_lid']) > 1.0e5
    assert float(st['w_lid'][-1]) > 0.5

    # Geometric-mean log10 blending at w_lid = 0.5 when tau_y / tau_d = 0.01:
    # eta_eff_val = 0.01 * 1e24 = 1e22 -> log_solid = 0.5 * 22 + 0.5 * 24 = 23 -> 1e23 Pa s.
    eta_blended = compute_effective_viscosity(
        eta_diff=np.array([1.0e24]),
        tau_d=1.0e8,
        tau_y_lid=1.0e6,
        stress_closure_mode='lid',
        w_lid=np.array([0.5]),
    )
    assert float(eta_blended[0]) == pytest.approx(1.0e23, rel=1e-12)

    eta_local_vel = compute_effective_viscosity(
        eta_diff=1.0e22,
        stress_closure_mode='local',
        unyielded_velocity=2.0e-9,
        mixing_length=1.0e5,
        tau_y_profile=1.0e7,
    )
    assert eta_local_vel == pytest.approx(1.0e7 / (2.0 * 1.0e-14), rel=1e-12)
    with pytest.raises(ValueError, match='tau_y or tau_y_profile required'):
        compute_effective_viscosity(
            eta_diff=1.0e22,
            stress_closure_mode='local',
            strain_rate=1.0e-14,
        )

    eta_lid_none = compute_effective_viscosity(
        eta_diff=1.0e22, tau_d=None, tau_y_lid=None, stress_closure_mode='lid'
    )
    assert eta_lid_none == pytest.approx(1.0e22, rel=1e-12)
