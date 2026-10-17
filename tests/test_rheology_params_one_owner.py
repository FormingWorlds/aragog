"""Verification that SolidRheologyParams is the single owner of rheology parameters.

Exercises the contract that SolidRheologyParams defines all solid rheology
parameters, and that PhaseConfig matches SolidRheologyParams field-by-field at defaults.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aragog.config.phases import MixedPhaseConfig, PhaseConfig
from aragog.rheology import (
    SolidRheologyParams,
    compute_arrhenius_viscosity,
    eta_diff,
    merge_rheology,
)

pytestmark = pytest.mark.unit

EXPECTED_FIELDS = {
    'enabled',
    'activation_energy',
    'activation_volume',
    'activation_volume_decay_pressure',
    'arrhenius_t_ref',
    'viscosity_max_log10',
    'water_prefactor',
    'yield_stress_c',
    'yield_stress_mu',
    'yield_stress_max',
    'stress_closure_mode',
    'interior_flux_fraction',
    'lid_base_mode',
    'lid_base_temperature',
    'lid_contrast_coeff',
    'lid_mask_width_cells',
    'phi_visc_single',
    'mlt_top_slope',
    'mlt_bottom_slope',
    's_top_inf',
    'c_theta',
}


def test_solid_rheology_params_has_expected_fields():
    """Verify SolidRheologyParams declares all expected fields."""
    field_names = {f.name for f in dataclasses.fields(SolidRheologyParams)}
    assert field_names == EXPECTED_FIELDS


def test_phase_config_rheology_defaults_match():
    """Verify PhaseConfig defaults match SolidRheologyParams field-by-field."""
    default_rheo = SolidRheologyParams()
    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
    )
    assert p_cfg.rheology == default_rheo
    for name in EXPECTED_FIELDS:
        assert getattr(p_cfg, name) == getattr(default_rheo, name)


def test_mixed_phase_config_has_no_rheology_fields():
    """Verify MixedPhaseConfig has no rheology fields."""
    mixed_cfg = MixedPhaseConfig(
        latent_heat_of_fusion=4e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1e-3,
    )
    for name in EXPECTED_FIELDS:
        assert not hasattr(mixed_cfg, name)


def test_custom_solid_rheology_params_propagation():
    """Verify non-default SolidRheologyParams propagates through PhaseConfig and merge_rheology."""
    custom_rheo = SolidRheologyParams(
        enabled=True,
        activation_energy=250e3,
        activation_volume=4e-6,
        water_prefactor=0.5,
        yield_stress_c=40e6,
        stress_closure_mode='local',
        lid_base_mode='fixed',
        lid_base_temperature=1350.0,
        phi_visc_single=0.6,
        s_top_inf=0.18,
        c_theta=0.20,
    )

    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        rheology=custom_rheo,
    )
    assert p_cfg.rheology == custom_rheo
    assert p_cfg.water_prefactor == 0.5
    assert p_cfg.phi_visc_single == 0.6
    assert p_cfg.s_top_inf == 0.18
    assert p_cfg.c_theta == 0.20


def test_solid_rheology_params_unconditional_bounds_validation():
    """Verify bounds validation runs unconditionally even when enabled=False."""
    with pytest.raises(ValueError, match='phi_visc_single'):
        SolidRheologyParams(enabled=False, phi_visc_single=-0.1)

    with pytest.raises(ValueError, match='activation_energy'):
        SolidRheologyParams(enabled=False, activation_energy=-100.0)

    with pytest.raises(ValueError, match='viscosity_max_log10'):
        SolidRheologyParams(enabled=False, viscosity_max_log10=15.0)


def test_phase_config_merge_precedence():
    """Verify PhaseConfig preserves custom rheology when flat kwargs are provided."""
    custom = SolidRheologyParams(
        enabled=True,
        activation_energy=250e3,
        yield_stress_c=40e6,
        water_prefactor=0.5,
    )
    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        rheology=custom,
        lid_contrast_coeff=3.0,
    )
    assert p_cfg.rheology.enabled is True
    assert p_cfg.rheology.activation_energy == 250e3
    assert p_cfg.rheology.yield_stress_c == 40e6
    assert p_cfg.rheology.water_prefactor == 0.5
    assert p_cfg.rheology.lid_contrast_coeff == 3.0
    assert p_cfg.lid_contrast_coeff == 3.0


def test_arrhenius_viscosity_water_prefactor_ceiling_numpy_and_jax():
    """Verify water_prefactor > 1 does not exceed viscosity_max_log10 cap."""
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    jax.config.update('jax_enable_x64', True)

    T = 200.0
    P = 1e9
    cap_log10 = 35.0
    expected_cap = 10.0**cap_log10

    res_np = eta_diff(
        T, P, viscosity_solid=1e21, viscosity_max_log10=cap_log10, water_prefactor=100.0
    )
    assert np.isclose(res_np, expected_cap)
    assert res_np <= expected_cap

    res_jax = compute_arrhenius_viscosity(
        jnp.asarray(T),
        jnp.asarray(P),
        viscosity_solid=1e21,
        viscosity_max_log10=cap_log10,
        water_prefactor=100.0,
        xp=jnp,
    )
    assert np.isclose(float(res_jax), expected_cap)
    assert float(res_jax) <= expected_cap


def test_solid_rheology_params_rejects_nan():
    """Verify SolidRheologyParams rejects NaN for all numeric parameters."""
    nan = float('nan')
    for f in dataclasses.fields(SolidRheologyParams):
        if f.type in ('float', float):
            with pytest.raises(ValueError):
                SolidRheologyParams(**{f.name: nan})


def test_merge_precedence_explicit_default_value():
    """Verify explicit keyword argument matching default overrides base rheology."""
    custom = SolidRheologyParams(enabled=True, activation_energy=250e3)
    merged = merge_rheology(custom, {'activation_energy': 300e3, 'enabled': False})
    assert merged.activation_energy == 300e3
    assert merged.enabled is False

    p_cfg = PhaseConfig(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        rheology=custom,
        activation_energy=300e3,
        enabled=False,
    )
    assert p_cfg.activation_energy == 300e3
    assert p_cfg.enabled is False
