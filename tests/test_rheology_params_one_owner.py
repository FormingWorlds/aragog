"""Verification that SolidRheologyParams is the single owner of rheology parameters.

Exercises the contract that SolidRheologyParams defines all 18 solid rheology
parameters, and that PhaseConfig, _PhaseParameters, EntropyPhaseEvaluator,
and PhaseParams match SolidRheologyParams field-by-field at defaults.
"""

from __future__ import annotations

import dataclasses

import pytest

from aragog.config.phases import MixedPhaseConfig, PhaseConfig
from aragog.jax.phase import PhaseParams
from aragog.parser import _PhaseMixedParameters, _PhaseParameters
from aragog.rheology import SolidRheologyParams

pytestmark = pytest.mark.unit

EXPECTED_18_FIELDS = {
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
    'yield_switch_width',
    'stress_closure_mode',
    'interior_flux_fraction',
    'lid_base_mode',
    'lid_base_temperature',
    'lid_contrast_coeff',
    'lid_mask_width_cells',
    'phi_visc_single',
}


def test_solid_rheology_params_has_exact_18_fields():
    """Verify SolidRheologyParams declares exactly the expected 18 fields."""
    field_names = {f.name for f in dataclasses.fields(SolidRheologyParams)}
    assert field_names == EXPECTED_18_FIELDS
    assert len(field_names) == 18


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
    for name in EXPECTED_18_FIELDS:
        assert getattr(p_cfg, name) == getattr(default_rheo, name)


def test_phase_parameters_rheology_defaults_match():
    """Verify _PhaseParameters defaults match SolidRheologyParams field-by-field."""
    default_rheo = SolidRheologyParams()
    p_params = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
    )
    assert p_params.rheology == default_rheo
    for name in EXPECTED_18_FIELDS:
        assert getattr(p_params, name) == getattr(default_rheo, name)


def test_phase_params_jax_rheology_defaults_match():
    """Verify JAX PhaseParams defaults match SolidRheologyParams field-by-field."""
    default_rheo = SolidRheologyParams()
    jax_params = PhaseParams()
    assert jax_params.rheology == default_rheo
    for name in EXPECTED_18_FIELDS:
        assert getattr(jax_params, name) == getattr(default_rheo, name)


def test_mixed_phase_classes_have_no_rheology_fields():
    """Verify MixedPhaseConfig and _PhaseMixedParameters have no rheology fields."""
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
    for name in EXPECTED_18_FIELDS:
        assert not hasattr(mixed_cfg, name)

    mixed_params = _PhaseMixedParameters(
        latent_heat_of_fusion=4e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1e-3,
    )
    for name in EXPECTED_18_FIELDS:
        assert not hasattr(mixed_params, name)


def test_custom_solid_rheology_params_propagation():
    """Verify non-default SolidRheologyParams propagates through layers."""
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
    )

    # PhaseConfig via rheology keyword
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

    # JAX PhaseParams from_rheology_params
    jax_p = PhaseParams.from_rheology_params(custom_rheo)
    assert jax_p.rheology == custom_rheo
    assert jax_p.water_prefactor == 0.5
    assert jax_p.phi_visc_single == 0.6


def test_solid_rheology_params_unconditional_bounds_validation():
    """Verify bounds validation runs unconditionally even when enabled=False."""
    with pytest.raises(ValueError, match='phi_visc_single'):
        SolidRheologyParams(enabled=False, phi_visc_single=-0.1)

    with pytest.raises(ValueError, match='activation_energy'):
        SolidRheologyParams(enabled=False, activation_energy=-100.0)

    with pytest.raises(ValueError, match='viscosity_max_log10'):
        SolidRheologyParams(enabled=False, viscosity_max_log10=15.0)


def test_phase_parameters_merge_precedence():
    """Verify _PhaseParameters preserves custom rheology when flat kwargs are provided."""
    custom = SolidRheologyParams(
        enabled=True,
        activation_energy=250e3,
        yield_stress_c=40e6,
        water_prefactor=0.5,
    )
    p = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=2e-5,
        viscosity=1e21,
        rheology=custom,
        lid_contrast_coeff=3.0,
    )
    assert p.rheology.enabled is True
    assert p.rheology.activation_energy == 250e3
    assert p.rheology.yield_stress_c == 40e6
    assert p.rheology.water_prefactor == 0.5
    assert p.rheology.lid_contrast_coeff == 3.0
    assert p.lid_contrast_coeff == 3.0


def test_phase_params_jax_merge_precedence():
    """Verify JAX PhaseParams preserves custom rheology when flat kwargs are provided."""
    custom = SolidRheologyParams(
        enabled=True,
        activation_energy=250e3,
        yield_stress_c=40e6,
        water_prefactor=0.5,
    )
    p = PhaseParams(rheology=custom, lid_contrast_coeff=3.0)
    assert p.rheology.enabled is True
    assert p.rheology.activation_energy == 250e3
    assert p.rheology.yield_stress_c == 40e6
    assert p.rheology.water_prefactor == 0.5
    assert p.rheology.lid_contrast_coeff == 3.0


def test_arrhenius_viscosity_water_prefactor_ceiling_numpy_and_jax():
    """Verify water_prefactor > 1 does not exceed viscosity_max_log10 cap."""
    import numpy as np

    from aragog.jax.phase import compute_arrhenius_viscosity as jax_arrhenius
    from aragog.rheology import eta_diff as np_arrhenius

    T = 200.0  # Cold temperature giving huge exponent
    P = 1e9
    cap_log10 = 35.0
    expected_cap = 10.0**cap_log10

    # NumPy
    res_np = np_arrhenius(
        T, P, viscosity_solid=1e21, viscosity_max_log10=cap_log10, water_prefactor=100.0
    )
    assert np.isclose(res_np, expected_cap)
    assert res_np <= expected_cap

    # JAX
    res_jax = jax_arrhenius(
        T, P, viscosity_solid=1e21, viscosity_max_log10=cap_log10, water_prefactor=100.0
    )
    assert np.isclose(float(res_jax), expected_cap)
    assert float(res_jax) <= expected_cap


def test_solid_rheology_params_rejects_nan():
    """Verify SolidRheologyParams rejects NaN for all numeric parameters."""
    from dataclasses import fields

    nan = float('nan')
    for f in fields(SolidRheologyParams):
        if f.type in ('float', float):
            with pytest.raises(ValueError):
                SolidRheologyParams(**{f.name: nan})


def test_merge_precedence_explicit_default_value():
    """Verify explicit keyword argument matching default overrides base rheology."""
    custom = SolidRheologyParams(enabled=True, activation_energy=250e3)

    # JAX PhaseParams
    jp = PhaseParams(rheology=custom, activation_energy=300e3, enabled=False)
    assert jp.activation_energy == 300e3
    assert jp.enabled is False

    # NumPy _PhaseParameters
    np_p = _PhaseParameters(
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
    assert np_p.activation_energy == 300e3
    assert np_p.enabled is False


def test_phase_params_jax_immediate_validation():
    """Verify JAX PhaseParams validates bounds immediately upon instantiation."""
    with pytest.raises(ValueError, match='water_prefactor'):
        PhaseParams(water_prefactor=-1.0)

    with pytest.raises(ValueError, match='activation_energy'):
        PhaseParams(activation_energy=-100.0)

    with pytest.raises(ValueError, match='phi_visc_single'):
        PhaseParams(phi_visc_single=1.5)
