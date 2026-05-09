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


def test_config_from_dict_strict_rejects_scalings_key():
    """Config.from_dict({'scalings': ...}) must raise ValueError.

    Edge case: the dict path is the canonical entry point used by
    PROTEUS to build Parameters; silently stripping a 'scalings' key
    would mask user-intent (a TOML carrying the legacy block would
    quietly load as if the section was absent).
    """
    from aragog.config import Config

    minimal = {
        'scalings': {'radius': 1.0},
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
    """The 4 tolerance/time fields are required; tsurf_poststep_change
    defaults to 30 K, event_triggering defaults False.
    """
    s = SolverConfig(
        start_time=0.0,
        end_time=1.0e6,
        atol=1.0e-9,
        rtol=1.0e-6,
    )
    assert s.tsurf_poststep_change == pytest.approx(30.0, abs=1e-12)
    assert s.event_triggering is False


def test_solver_config_rejects_missing_tolerances():
    """Edge case: omitting atol or rtol raises TypeError at construct."""
    with pytest.raises(TypeError):
        SolverConfig(start_time=0.0, end_time=1.0e6, rtol=1.0e-6)
