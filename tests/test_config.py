"""Unit tests for the attrs-based Aragog config classes.

Each ``aragog.config.X`` subclass is a thin dataclass with attrs
validators, defaults, and (in some cases) post-init hooks. Even
though they do little arithmetic, they are the boundary between
TOML input and the legacy ``Parameters`` interior, so a regression
here can silently mis-route units or shapes into the solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.config.boundary import BoundaryConfig
from aragog.config.energy import EnergyConfig
from aragog.config.initial_condition import InitialConditionConfig
from aragog.config.mesh import MeshConfig
from aragog.config.phases import MixedPhaseConfig, PhaseConfig
from aragog.config.radionuclides import RadionuclideConfig
from aragog.config.solver import SolverConfig

pytestmark = pytest.mark.unit


# ---- BoundaryConfig --------------------------------------------------------


def test_boundary_config_defaults_match_documented_values():
    """Documented defaults: tfac_core_avg=1.147 (Bower+2018), param_utbl=False,
    param_utbl_const=1e-7, core_bc='energy_balance' (PROTEUS production).

    Discriminator: a regression that flipped the default core_bc away
    from 'energy_balance' would silently change the state-vector length
    for every TOML that omits the field, and break SPIDER bit-parity.
    """
    bc = BoundaryConfig(
        outer_boundary_condition=1,
        outer_boundary_value=2.5e3,
        inner_boundary_condition=1,
        inner_boundary_value=4000.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
    )
    assert bc.tfac_core_avg == pytest.approx(1.147, abs=1e-12)
    assert bc.param_utbl is False
    assert bc.param_utbl_const == pytest.approx(1.0e-7, rel=1e-12)
    assert bc.core_bc == 'energy_balance'


def test_boundary_config_accepts_all_documented_core_bc_modes():
    """All four core_bc modes documented in the parser/configuration
    table must be accepted by attrs construction. The validation of
    the mode value happens downstream in the solver dispatch; this
    test guards against an attrs-level rejection regression.
    """
    for mode in ('quasi_steady', 'energy_balance', 'gradient', 'bower2018'):
        bc = BoundaryConfig(
            outer_boundary_condition=1,
            outer_boundary_value=2.5e3,
            inner_boundary_condition=1,
            inner_boundary_value=4000.0,
            emissivity=1.0,
            equilibrium_temperature=255.0,
            core_heat_capacity=880.0,
            core_bc=mode,
        )
        assert bc.core_bc == mode


# ---- EnergyConfig ----------------------------------------------------------


def test_energy_config_defaults_for_eddy_chemical_and_tidal_array():
    """Default eddy_diffusivity_chemical = 1.0; default tidal_array
    is a length-1 array of zeros (so wrapper code can broadcast).

    Edge case: each EnergyConfig instance must get its own copy of
    the array (attrs.Factory). Mutating one must not leak to the
    next instance — discriminates against ``= np.array([0.0])``
    being declared as a class default rather than a Factory.
    """
    e1 = EnergyConfig(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    e2 = EnergyConfig(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    assert e1.eddy_diffusivity_chemical == pytest.approx(1.0, abs=1e-15)
    assert e1.tidal_array.shape == (1,)
    np.testing.assert_allclose(e1.tidal_array, np.zeros(1), atol=1e-30)

    e1.tidal_array[0] = 99.0
    assert e2.tidal_array[0] == pytest.approx(0.0, abs=1e-30), (
        'EnergyConfig.tidal_array is shared across instances; attrs.Factory is missing.'
    )


# ---- InitialConditionConfig ------------------------------------------------


def test_initial_condition_config_defaults():
    """Default IC is type 1 (linear), 4000 K at both surface and base.

    Discriminator: a regression that changed the default IC type
    from 1 to 2 would silently expect an init_file that doesn't
    exist for any TOML omitting initial_condition.
    """
    ic = InitialConditionConfig()
    assert ic.initial_condition == 1
    assert ic.surface_temperature == pytest.approx(4000.0, abs=1e-12)
    assert ic.basal_temperature == pytest.approx(4000.0, abs=1e-12)
    assert ic.init_file == ''
    assert ic.init_temperature is None  # init=False, default=None


# ---- MeshConfig ------------------------------------------------------------


def test_mesh_config_required_and_default_fields():
    """The 5 positional fields are required; the rest have defaults.

    Discriminator: ``surface_density`` default 4078.95095544 is the
    SPIDER ``-adams_williamson_rhos`` value used in PROTEUS production;
    ``mass_coordinates`` default True matches the SPIDER mesh layout.
    ``gravitational_acceleration`` default 9.81 is Earth.
    """
    mc = MeshConfig(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=100,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
    )
    # Documented defaults aligned with PROTEUS production.
    assert mc.eos_method == 1
    assert mc.surface_density == pytest.approx(4078.95095544, rel=1e-12)
    assert mc.gravitational_acceleration == pytest.approx(9.81, abs=1e-12)
    assert mc.adiabatic_bulk_modulus == pytest.approx(260e9, rel=1e-12)
    assert mc.adams_williamson_beta == pytest.approx(0.0, abs=1e-30)
    assert mc.surface_pressure == pytest.approx(0.0, abs=1e-30)
    assert mc.mass_coordinates is True
    assert mc.eos_file == ''


def test_mesh_config_rejects_missing_required_radius():
    """Edge case: missing required field raises TypeError at construct."""
    with pytest.raises(TypeError):
        MeshConfig(
            inner_radius=3.480e6,
            number_of_nodes=100,
            mixing_length_profile='nearest_boundary',
            core_density=10500.0,
        )  # outer_radius is missing


# ---- PhaseConfig and MixedPhaseConfig --------------------------------------


def test_phase_config_accepts_float_or_string_lookup_for_each_property():
    """Each property accepts float (constant) or str (lookup table path).

    This dual-dispatch pattern is critical for the SPIDER-parity vs
    inline-EOS routing. A regression that forced strict float would
    break every TOML using a P-S lookup, and vice versa.
    """
    pc_float = PhaseConfig(
        density=3300.0,
        heat_capacity=1200.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    assert pc_float.density == pytest.approx(3300.0, rel=1e-12)
    assert pc_float.entropy == ''  # default

    pc_str = PhaseConfig(
        density='/path/to/rho.dat',
        heat_capacity='cp.dat',
        melt_fraction=0.0,
        thermal_conductivity='k.dat',
        thermal_expansivity='alpha.dat',
        viscosity='visc.dat',
        entropy='S.dat',
    )
    assert pc_str.heat_capacity == 'cp.dat'
    assert pc_str.entropy == 'S.dat'


def test_mixed_phase_config_default_cp_blend_is_latent():
    """Default cp_blend = 'latent' = SPIDER-parity v4 convention.

    Discriminator: a regression to 'linear' would drop the latent-
    heat augmentation in the mushy band and silently divert from
    SPIDER parity in every coupled PROTEUS run.
    """
    mp = MixedPhaseConfig(
        latent_heat_of_fusion=4e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=0.001,
    )
    assert mp.cp_blend == 'latent'


def test_mixed_phase_config_default_separation_viscosity_is_melt():
    """Default separation_viscosity = 'melt' = SPIDER-parity convention.

    Discriminator: a regression to 'mixture' would silently swap the
    gravitational-separation drag viscosity to the phi_rheo-blended
    bulk viscosity, a departure from SPIDER, in every coupled PROTEUS
    run that omits the field.
    """
    mp = MixedPhaseConfig(
        latent_heat_of_fusion=4e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=0.001,
    )
    assert mp.separation_viscosity == 'melt'


def test_mixed_phase_config_rejects_invalid_separation_viscosity():
    """The ``in_`` validator must reject a value outside ('melt', 'mixture').

    Discriminator: deleting this validator would let a typo'd mode
    string reach the solver silently instead of failing at parse time.
    """
    with pytest.raises(ValueError):
        MixedPhaseConfig(
            latent_heat_of_fusion=4e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='solidus.dat',
            liquidus='liquidus.dat',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=0.001,
            separation_viscosity='bogus',
        )


def test_mixed_phase_config_const_properties_defaults_match_legacy_parser():
    """Default const_properties block reproduces the SPIDER analytic-EOS
    smoke values that the legacy ``_PhaseMixedParameters`` dataclass
    has carried for years.

    Discriminator: each default is asymmetric (rho=4000, Cp=1000,
    alpha=1e-5, cond=4, log10visc=2, T_ref=3500, S_ref=3000), so a
    single mistakenly-renamed field would make ONE assertion fail
    while the others pass. ``const_properties`` itself defaults to
    False (do NOT use the constants unless explicitly requested);
    flipping that default would break every TOML that omits the flag
    and silently route the solver onto the analytic-EOS path.
    """
    mp = MixedPhaseConfig(
        latent_heat_of_fusion=4e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=0.001,
    )
    assert mp.const_properties is False
    assert mp.matprop_smooth_width == pytest.approx(0.0, abs=1e-15)
    assert mp.const_rho == pytest.approx(4000.0, rel=1e-12)
    assert mp.const_Cp == pytest.approx(1000.0, rel=1e-12)
    assert mp.const_alpha == pytest.approx(1e-5, rel=1e-12)
    assert mp.const_cond == pytest.approx(4.0, rel=1e-12)
    assert mp.const_log10visc == pytest.approx(2.0, rel=1e-12)
    assert mp.const_T_ref == pytest.approx(3500.0, rel=1e-12)
    assert mp.const_S_ref == pytest.approx(3000.0, rel=1e-12)


def test_mixed_phase_config_const_properties_overrides_take_effect():
    """All nine constant-properties knobs must be overridable to
    arbitrary positive values without an attrs validator silently
    coercing or dropping them.

    Anti-happy-path: passes physically asymmetric values (each one
    1.5x its default) so a regression that swapped two field names
    in the attrs definition (e.g., ``const_alpha`` and ``const_cond``)
    would produce an assertion failure on whichever pair was swapped.
    """
    mp = MixedPhaseConfig(
        latent_heat_of_fusion=4e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=0.001,
        matprop_smooth_width=0.012,
        const_properties=True,
        const_rho=6000.0,
        const_Cp=1500.0,
        const_alpha=1.5e-5,
        const_cond=6.0,
        const_log10visc=3.0,
        const_T_ref=5250.0,
        const_S_ref=4500.0,
    )
    assert mp.const_properties is True
    assert mp.matprop_smooth_width == pytest.approx(0.012, rel=1e-12)
    assert mp.const_rho == pytest.approx(6000.0, rel=1e-12)
    assert mp.const_Cp == pytest.approx(1500.0, rel=1e-12)
    assert mp.const_alpha == pytest.approx(1.5e-5, rel=1e-12)
    assert mp.const_cond == pytest.approx(6.0, rel=1e-12)
    assert mp.const_log10visc == pytest.approx(3.0, rel=1e-12)
    assert mp.const_T_ref == pytest.approx(5250.0, rel=1e-12)
    assert mp.const_S_ref == pytest.approx(4500.0, rel=1e-12)


def test_config_from_dict_round_trips_eddy_diffusivity_thermal():
    """A TOML override on ``[energy].eddy_diffusivity_thermal`` must
    reach ``Parameters.energy.eddy_diffusivity_thermal``.

    Discriminator: the public ``EnergyConfig`` exposes the field, but
    ``Config.from_dict`` hydrates the legacy ``_EnergyParameters``
    parser dataclass (not ``EnergyConfig``). If ``_EnergyParameters``
    drifts and lacks the field, the dict-load path raises
    ``TypeError`` and the documented knob is dead. We pass an
    asymmetric value (0.5) so a regression cannot pass by accident
    on the default.
    """
    from aragog.config import Config

    data = {
        'solver': {
            'start_time': 0,
            'end_time': 1,
            'atol': 1e-9,
            'rtol': 1e-9,
        },
        'boundary_conditions': {
            'outer_boundary_condition': 1,
            'outer_boundary_value': 100.0,
            'inner_boundary_condition': 1,
            'inner_boundary_value': 0.0,
            'emissivity': 1.0,
            'equilibrium_temperature': 273.0,
            'core_heat_capacity': 880.0,
        },
        'mesh': {
            'outer_radius': 6.371e6,
            'inner_radius': 3.481e6,
            'number_of_nodes': 80,
            'mixing_length_profile': 'nearest_boundary',
            'core_density': 11000.0,
        },
        'energy': {
            'conduction': True,
            'convection': True,
            'gravitational_separation': False,
            'mixing': False,
            'radionuclides': False,
            'tidal': False,
            'eddy_diffusivity_thermal': 0.5,
        },
        'phase_liquid': {
            'density': 4000.0,
            'heat_capacity': 1200.0,
            'melt_fraction': 1,
            'thermal_conductivity': 4.0,
            'thermal_expansivity': 3e-5,
            'viscosity': 0.1,
        },
        'phase_solid': {
            'density': 4000.0,
            'heat_capacity': 1200.0,
            'melt_fraction': 0,
            'thermal_conductivity': 4.0,
            'thermal_expansivity': 3e-5,
            'viscosity': 1e21,
        },
        'phase_mixed': {
            'latent_heat_of_fusion': 4e5,
            'rheological_transition_melt_fraction': 0.4,
            'rheological_transition_width': 0.15,
            'solidus': '',
            'liquidus': '',
            'phase': 'mixed',
            'phase_transition_width': 0.01,
            'grain_size': 1e-3,
        },
    }
    params = Config.from_dict(data)
    assert params.energy.eddy_diffusivity_thermal == pytest.approx(0.5, rel=1e-12), (
        'eddy_diffusivity_thermal did not reach _EnergyParameters; the '
        'legacy parser dataclass has likely drifted out of sync with '
        'EnergyConfig.'
    )

    # Negative-value SPIDER convention must round-trip without coercion.
    data_pinned = {**data, 'energy': {**data['energy'], 'eddy_diffusivity_thermal': -2.5}}
    params_pinned = Config.from_dict(data_pinned)
    assert params_pinned.energy.eddy_diffusivity_thermal == pytest.approx(-2.5, rel=1e-12)


def test_energy_config_eddy_diffusivity_thermal_default_is_unity():
    """Default eddy_diffusivity_thermal = 1.0 passes the MLT-derived
    kappa_h through unchanged. Non-default negative values pin the
    diffusivity to the absolute value (SPIDER convention).

    Edge case: a regression that flipped the default to 0 would
    silently kill all convection.
    """
    e = EnergyConfig(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    assert e.eddy_diffusivity_thermal == pytest.approx(1.0, rel=1e-12)

    # Negative-value SPIDER convention must round-trip through attrs
    # without coercion or rejection.
    e_pinned = EnergyConfig(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        eddy_diffusivity_thermal=-2.5,
    )
    assert e_pinned.eddy_diffusivity_thermal == pytest.approx(-2.5, rel=1e-12)


def test_phase_boundary_entropy_margin_default_matches_across_config_layers():
    """The 200 J/kg/K proximity band default must agree in both config layers.

    ``EnergyConfig`` (attrs, public) and ``_EnergyParameters`` (legacy parser
    dataclass) carry independent literal defaults for
    ``phase_boundary_entropy_margin``. The solver reproduces the historical
    fixed band only if both read 200.0, and the coupled PROTEUS path (whose
    schema does not declare the field) relies on the parser default. A drift
    between the two layers would move ``max_step`` behaviour on one path only,
    with no other symptom, so pin both to the same value rather than to a bare
    literal that could be edited in isolation.
    """
    from aragog.parser import _EnergyParameters

    kw = dict(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    attrs_default = EnergyConfig(**kw).phase_boundary_entropy_margin
    parser_default = _EnergyParameters(**kw).phase_boundary_entropy_margin
    assert attrs_default == pytest.approx(200.0, rel=1e-12)
    assert parser_default == pytest.approx(200.0, rel=1e-12)
    # The two layers must not drift apart: same conceptual default, same value.
    assert attrs_default == pytest.approx(parser_default, rel=1e-12)


def test_config_from_dict_round_trips_phase_boundary_entropy_margin():
    """A TOML override on ``[energy].phase_boundary_entropy_margin`` must reach
    ``Parameters.energy``.

    ``Config.from_dict`` hydrates the legacy ``_EnergyParameters`` parser
    dataclass, not ``EnergyConfig``. If the parser field is dropped or
    renamed, the dict-load raises ``TypeError`` and the documented knob is
    dead. An asymmetric 400.0 (double the default) cannot pass by accident on
    the 200.0 default, and it is exactly the value at which the near-boundary
    band widens enough to change a clamp verdict.
    """
    from aragog.config import Config

    data = {
        'solver': {'start_time': 0, 'end_time': 1, 'atol': 1e-9, 'rtol': 1e-9},
        'boundary_conditions': {
            'outer_boundary_condition': 1,
            'outer_boundary_value': 100.0,
            'inner_boundary_condition': 1,
            'inner_boundary_value': 0.0,
            'emissivity': 1.0,
            'equilibrium_temperature': 273.0,
            'core_heat_capacity': 880.0,
        },
        'mesh': {
            'outer_radius': 6.371e6,
            'inner_radius': 3.481e6,
            'number_of_nodes': 80,
            'mixing_length_profile': 'nearest_boundary',
            'core_density': 11000.0,
        },
        'energy': {
            'conduction': True,
            'convection': True,
            'gravitational_separation': False,
            'mixing': False,
            'radionuclides': False,
            'tidal': False,
            'phase_boundary_entropy_margin': 400.0,
        },
        'phase_liquid': {
            'density': 4000.0,
            'heat_capacity': 1200.0,
            'melt_fraction': 1,
            'thermal_conductivity': 4.0,
            'thermal_expansivity': 3e-5,
            'viscosity': 0.1,
        },
        'phase_solid': {
            'density': 4000.0,
            'heat_capacity': 1200.0,
            'melt_fraction': 0,
            'thermal_conductivity': 4.0,
            'thermal_expansivity': 3e-5,
            'viscosity': 1e21,
        },
        'phase_mixed': {
            'latent_heat_of_fusion': 4e5,
            'rheological_transition_melt_fraction': 0.4,
            'rheological_transition_width': 0.15,
            'solidus': '',
            'liquidus': '',
            'phase': 'mixed',
            'phase_transition_width': 0.01,
            'grain_size': 1e-3,
        },
    }
    params = Config.from_dict(data)
    assert params.energy.phase_boundary_entropy_margin == pytest.approx(400.0, rel=1e-12), (
        'phase_boundary_entropy_margin did not reach _EnergyParameters; the '
        'legacy parser dataclass has likely drifted out of sync with EnergyConfig.'
    )
    # Discrimination: 400.0 must not collapse to the 200.0 default.
    assert params.energy.phase_boundary_entropy_margin > 300.0


# ---- RadionuclideConfig ----------------------------------------------------


def test_radionuclide_get_heating_at_t0_returns_full_amplitude():
    """At time = t0_years the exponential decay factor is 1.

    Use values that distinguish the correct formula
    H * abundance * concentration * exp(ln(2) * (t0 - t) / half_life)
    from plausible bugs (sign flip on the exponent, missing log(2),
    wrong product order).
    """
    r = RadionuclideConfig(
        name='K40',
        t0_years=4.6e9,
        abundance=1.17e-4,
        concentration=240.0,  # ppm
        heat_production=2.92e-5,
        half_life_years=1.25e9,
    )
    h_at_t0 = r.get_heating(time=4.6e9)
    expected = r.heat_production * r.abundance * r.concentration
    assert float(h_at_t0) == pytest.approx(expected, rel=1e-12)


def test_radionuclide_decay_one_half_life_in_the_past_doubles_heat():
    """Edge case: at time = t0 - half_life (i.e. one half-life
    earlier), the heating must be twice the t0 value because
    exp(ln(2) * (t0 - (t0 - hl)) / hl) = exp(ln(2)) = 2.

    Discriminator: catches a sign flip on (t0 - time) AND a missing
    log(2) factor, both of which would change this answer.
    """
    r = RadionuclideConfig(
        name='Th232',
        t0_years=4.6e9,
        abundance=1.0,
        concentration=1.0,
        heat_production=1.0,
        half_life_years=1.4e10,
    )
    h_t0 = r.get_heating(time=4.6e9)
    h_one_hl_earlier = r.get_heating(time=4.6e9 - 1.4e10)
    assert float(h_one_hl_earlier) == pytest.approx(
        2.0 * float(h_t0),
        rel=1e-12,
    )


def test_radionuclide_decay_array_input_broadcasts():
    """Array of times returns an array of the same shape, with a
    monotonic decay (heating now > heating in the future).
    """
    r = RadionuclideConfig(
        name='U238',
        t0_years=4.6e9,
        abundance=0.992,
        concentration=20.0,
        heat_production=9.46e-5,
        half_life_years=4.47e9,
    )
    times = np.linspace(4.0e9, 6.0e9, 5)
    h = np.asarray(r.get_heating(times))
    assert h.shape == (5,)
    diffs = np.diff(h)
    # Monotone decreasing for forward-time (later time = less heat)
    assert np.all(diffs < 0), (
        'Radiogenic heating did not decay monotonically with time; '
        'sign of the exponent is wrong.'
    )


# ---- ScalingsConfig (removed) ----------------------------------------------


def test_config_module_no_longer_exposes_scalings_config():
    """``aragog.config.scalings`` and ``ScalingsConfig`` are removed.

    Edge case: a regression that re-added the legacy attrs shim would
    silently re-introduce a config layer the parser strict-rejects at
    load time. Catch both halves: the module file must be gone, and
    the public ``__all__`` must not list ``ScalingsConfig``.
    """
    import aragog.config as cfg_pkg

    assert 'ScalingsConfig' not in cfg_pkg.__all__, (
        '__all__ should not advertise ScalingsConfig; the layer was '
        'removed when the [scalings] section was strict-rejected.'
    )
    with pytest.raises(ImportError):
        import aragog.config.scalings  # noqa: F401


@pytest.mark.parametrize('key', ['scalings', 'Scalings', 'SCALINGS'])
def test_config_from_dict_strict_rejects_scalings_key(key):
    """Config.from_dict({'scalings': ...}) must raise ValueError.

    Edge case: the dict path is the canonical entry point used by
    PROTEUS to build Parameters; silently stripping a 'scalings' key
    would mask user-intent (a TOML carrying the legacy block would
    quietly load as if the section was absent). Case variants must
    also be rejected because TOML parsers preserve key case.
    """
    from aragog.config import Config

    minimal = {
        key: {'radius': 1.0},
        'solver': {
            'start_time': 0,
            'end_time': 1,
            'atol': 1e-9,
            'rtol': 1e-6,
        },
        'boundary_conditions': {},
        'mesh': {},
        'energy': {},
        'phase_liquid': {},
        'phase_solid': {},
        'phase_mixed': {},
    }
    with pytest.raises(ValueError, match=r'scalings'):
        Config.from_dict(minimal)


# ---- SolverConfig ----------------------------------------------------------


def test_solver_config_required_fields_and_defaults():
    """The 4 tolerance/time fields are required; ``tsurf_poststep_change``
    defaults to 30 K.
    """
    s = SolverConfig(
        start_time=0.0,
        end_time=1.0e6,
        atol=1.0e-9,
        rtol=1.0e-6,
    )
    assert s.tsurf_poststep_change == pytest.approx(30.0, abs=1e-12)


def test_solver_config_rejects_missing_tolerances():
    """Edge case: omitting atol or rtol raises TypeError at construct."""
    with pytest.raises(TypeError):
        SolverConfig(start_time=0.0, end_time=1.0e6, rtol=1.0e-6)
