"""Unit tests for ``aragog.parser``.

The parser layer is the dataclass-based legacy (pre-attrs) reader that
``Parameters.from_file`` relies on for INI/TOML config ingestion. It owns
the boundary-condition dispatch validators, the radionuclide decay model,
and the EOS-file shape gate. Bugs here are silent because they happen at
parse time, before any solver call, and a TOML that loaded yesterday must
still load today.

Note on scaling: ``_ScalingsParameters.__post_init__`` forces every scale
factor to 1.0. This is intentional (non-dimensionalisation has been
removed). All ``scale_attributes`` calls therefore divide by 1.0; the
value of the tests is in exercising the validators and side effects, not
in checking arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aragog.parser import (
    Parameters,
    _BoundaryConditionsParameters,
    _EnergyParameters,
    _get_dataclass_from_section_name,
    _InitialConditionParameters,
    _MeshParameters,
    _PhaseMixedParameters,
    _PhaseParameters,
    _Radionuclide,
    _ScalingsParameters,
    _SolverParameters,
)

pytestmark = pytest.mark.unit


# ---- module-level helpers --------------------------------------------------


def _scalings() -> _ScalingsParameters:
    """Return a unity-scaled _ScalingsParameters (post_init forces 1.0)."""
    return _ScalingsParameters()


def _build_bc(
    *,
    inner_boundary_condition: int = 2,
    inner_boundary_value: float = 1.0e6,
    outer_boundary_condition: int = 1,
    outer_boundary_value: float = 1500.0,
    param_utbl: bool = False,
    param_utbl_const: float = 1.0e-7,
) -> _BoundaryConditionsParameters:
    return _BoundaryConditionsParameters(
        outer_boundary_condition=outer_boundary_condition,
        outer_boundary_value=outer_boundary_value,
        inner_boundary_condition=inner_boundary_condition,
        inner_boundary_value=inner_boundary_value,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        param_utbl=param_utbl,
        param_utbl_const=param_utbl_const,
    )


# ---- _get_dataclass_from_section_name --------------------------------------


def test_section_mapping_contains_all_required_sections():
    """The mapping must cover every required TOML section.

    Discriminator: a regression that dropped 'phase_mixed' or
    renamed 'energy' would silently let the parser ignore that
    block, with the dataclass falling back to defaults.
    """
    mapping = _get_dataclass_from_section_name()
    expected = {
        'scalings',
        'solver',
        'boundary_conditions',
        'mesh',
        'energy',
        'initial_condition',
        'phase_liquid',
        'phase_solid',
        'phase_mixed',
    }
    assert set(mapping.keys()) == expected, (
        f'parser section mapping mismatch: {sorted(mapping.keys())} vs {sorted(expected)}'
    )
    # Edge case: same dataclass for phase_liquid and phase_solid by design.
    assert mapping['phase_liquid'] is mapping['phase_solid'], (
        'phase_liquid and phase_solid must share the same dataclass type; '
        'splitting them silently breaks symmetric per-phase scaling.'
    )


def test_section_mapping_excludes_radionuclides():
    """Radionuclides are dispatched separately by prefix matching, not via
    this mapping. A regression that added 'radionuclide' here would cause
    duplicate parsing.
    """
    mapping = _get_dataclass_from_section_name()
    assert not any('radionuclide' in k for k in mapping), (
        f'parser mapping must not contain radionuclide entries: {sorted(mapping.keys())}'
    )


# ---- _ScalingsParameters ---------------------------------------------------


def test_scalings_post_init_forces_all_fields_to_unity():
    """Even when the user passes non-unity values, post_init overrides
    them. This is the documented non-dim-removed behaviour.

    Discriminator: a regression that "fixed" post_init to honour user
    input would silently re-introduce dimensional drift through
    every downstream divide-by-scale call.
    """
    sc = _ScalingsParameters(radius=6.371e6, temperature=4000.0, density=4500.0, time=1e6)
    assert sc.radius == pytest.approx(1.0, abs=1e-15)
    assert sc.temperature == pytest.approx(1.0, abs=1e-15)
    assert sc.density == pytest.approx(1.0, abs=1e-15)
    assert sc.time == pytest.approx(1.0, abs=1e-15)
    # Sample derived fields (init=False, set in post_init)
    assert sc.heat_flux == pytest.approx(1.0, abs=1e-15)
    assert sc.viscosity == pytest.approx(1.0, abs=1e-15)
    assert sc.stefan_boltzmann_constant == pytest.approx(1.0, abs=1e-15)


# ---- _BoundaryConditionsParameters._scale_inner_boundary_condition --------


def test_boundary_inner_bc_type_1_zeros_value():
    """Inner BC code 1 (simple core cooling): value forced to 0
    regardless of what the TOML supplied. Use a non-zero input to
    discriminate against silent passthrough.
    """
    bc = _build_bc(inner_boundary_condition=1, inner_boundary_value=12345.0)
    bc.scale_attributes(_scalings())
    assert bc.inner_boundary_value == pytest.approx(0.0, abs=1e-30), (
        'IBC 1 must zero the inner boundary value; a non-zero leak '
        'would re-add a phantom heat sink at the CMB.'
    )


def test_boundary_inner_bc_type_2_scales_heat_flux():
    """IBC 2 (prescribed heat flux): value divided by heat_flux scale.
    With unity scales the value passes through unchanged.
    """
    bc = _build_bc(inner_boundary_condition=2, inner_boundary_value=4.5e6)
    bc.scale_attributes(_scalings())
    assert bc.inner_boundary_value == pytest.approx(4.5e6, rel=1e-12)


def test_boundary_inner_bc_type_3_scales_temperature():
    """IBC 3 (prescribed temperature): value divided by temperature scale."""
    bc = _build_bc(inner_boundary_condition=3, inner_boundary_value=4200.0)
    bc.scale_attributes(_scalings())
    assert bc.inner_boundary_value == pytest.approx(4200.0, rel=1e-12)


def test_boundary_inner_bc_unknown_code_raises_value_error():
    """Unphysical edge case: IBC 99 must raise, not silently pass.

    Discriminator: catches a regression that removed the else-raise
    after adding new BC types, which would silently ignore typos
    and run with default values.
    """
    bc = _build_bc(inner_boundary_condition=99, inner_boundary_value=0.0)
    with pytest.raises(ValueError, match='inner_boundary_condition = 99'):
        bc.scale_attributes(_scalings())


def test_boundary_outer_bc_type_4_scales_heat_flux():
    """OBC 4 (prescribed heat flux): divided by heat_flux scale."""
    bc = _build_bc(outer_boundary_condition=4, outer_boundary_value=1.0e5)
    bc.scale_attributes(_scalings())
    assert bc.outer_boundary_value == pytest.approx(1.0e5, rel=1e-12)


def test_boundary_outer_bc_type_5_scales_temperature():
    """OBC 5 (prescribed temperature): divided by temperature scale."""
    bc = _build_bc(outer_boundary_condition=5, outer_boundary_value=2000.0)
    bc.scale_attributes(_scalings())
    assert bc.outer_boundary_value == pytest.approx(2000.0, rel=1e-12)


def test_boundary_outer_bc_unknown_code_raises_value_error():
    """OBC 99 (unphysical, never documented) must raise."""
    bc = _build_bc(outer_boundary_condition=99)
    with pytest.raises(ValueError, match='outer_boundary_condition = 99'):
        bc.scale_attributes(_scalings())


def test_boundary_param_utbl_off_zeros_constant():
    """When param_utbl=False, the constant is forced to 0.0 even when
    a non-zero value is supplied. This is the production setting.
    """
    bc = _build_bc(param_utbl=False, param_utbl_const=1.0e-3)
    bc.scale_attributes(_scalings())
    assert bc.param_utbl_const == pytest.approx(0.0, abs=1e-30), (
        'param_utbl=False must zero the UTBL constant; a leak would '
        'enable the upper-thermal-boundary-layer correction by accident.'
    )


def test_boundary_param_utbl_on_scales_constant_with_temperature_squared():
    """When param_utbl=True, const *= temperature**2. Unity scales
    means the constant passes through. Use a non-zero const to
    discriminate against an accidental zero-out.
    """
    bc = _build_bc(param_utbl=True, param_utbl_const=2.5e-7)
    bc.scale_attributes(_scalings())
    # Unity scales: const *= 1.0 == const
    assert bc.param_utbl_const == pytest.approx(2.5e-7, rel=1e-12)


# ---- _EnergyParameters -----------------------------------------------------


def test_energy_scale_attributes_divides_tidal_array_by_power_per_mass():
    """``scale_attributes`` divides tidal_array elementwise by
    scalings.power_per_mass. With unity scales the array passes
    through unchanged. Pass a non-zero array to discriminate against
    a silent in-place zeroing.
    """
    e = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=True,
        tidal_array=np.array([1.5e-9, 2.0e-9, 3.0e-9]),
    )
    e.scale_attributes(_scalings())
    np.testing.assert_allclose(
        e.tidal_array,
        np.array([1.5e-9, 2.0e-9, 3.0e-9]),
        rtol=1e-12,
    )
    assert e.scalings_ is not None


# ---- _InitialConditionParameters -------------------------------------------


def test_initial_condition_scale_attributes_method_1_runs_without_file():
    """IC method 1 (linear): no init_file required."""
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3600.0, basal_temperature=4200.0
    )
    ic.scale_attributes(_scalings())
    assert ic.surface_temperature == pytest.approx(3600.0, rel=1e-12)
    assert ic.basal_temperature == pytest.approx(4200.0, rel=1e-12)


def test_initial_condition_scale_attributes_method_2_no_file_raises():
    """IC method 2 (user-defined T from file) with empty init_file
    must raise. Edge case: the legacy code path is also validated
    in Parameters.__post_init__; this test guards the duplicate
    check on the parser-side.
    """
    ic = _InitialConditionParameters(initial_condition=2, init_file='')
    with pytest.raises(ValueError, match='initial temperature file'):
        ic.scale_attributes(_scalings())


def test_initial_condition_scale_attributes_method_2_loads_file(tmp_path: Path):
    """IC method 2 with a valid 1-D temperature column file must load
    and divide by temperature scale.
    """
    f = tmp_path / 'init_T.dat'
    f.write_text('3600.0\n3700.0\n3800.0\n4000.0\n')
    ic = _InitialConditionParameters(initial_condition=2, init_file=str(f))
    ic.scale_attributes(_scalings())
    np.testing.assert_allclose(
        ic.init_temperature,
        np.array([3600.0, 3700.0, 3800.0, 4000.0]),
        rtol=1e-12,
    )


# ---- _MeshParameters -------------------------------------------------------


def test_mesh_scale_attributes_eos_method_1_runs_without_file():
    """EOS method 1 (Adams-Williamson, default): no eos_file required."""
    mp = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=100,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
    )
    mp.scale_attributes(_scalings())
    assert mp.outer_radius == pytest.approx(6.371e6, rel=1e-12)
    assert mp.inner_radius == pytest.approx(3.480e6, rel=1e-12)


def test_mesh_scale_attributes_eos_method_2_no_file_raises():
    """EOS method 2 with empty eos_file must raise."""
    mp = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=100,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=2,
        eos_file='',
    )
    with pytest.raises(ValueError, match='equation of state'):
        mp.scale_attributes(_scalings())


def test_mesh_scale_attributes_eos_method_2_rejects_radius_out_of_range(tmp_path: Path):
    """The parser-side validator rejects an EOS file whose radius
    column starts BELOW the inner_radius (i.e. inside the core).
    Discriminator: a regression that removed the radius bracket
    check would silently feed core-side EOS samples into the mantle.
    """
    f = tmp_path / 'eos_too_low.dat'
    # Columns: r, P, rho, g. r starts at 3.0e6 < 3.48e6 inner_radius.
    rows = [(3.0e6, 1.0e11, 5000.0, 9.5), (5.0e6, 5.0e10, 4500.0, 9.8)]
    f.write_text('\n'.join(f'{r} {P} {rho} {g}' for r, P, rho, g in rows))
    mp = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=100,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=2,
        eos_file=str(f),
    )
    with pytest.raises(ValueError, match='Values out of range'):
        mp.scale_attributes(_scalings())


# ---- _PhaseMixedParameters -------------------------------------------------


def test_phase_mixed_scale_attributes_runs():
    """Exercise scale_attributes on PhaseMixedParameters: latent heat
    divided by latent_heat_per_mass, grain_size divided by radius.
    Unity scales: pass-through.
    """
    mp = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    mp.scale_attributes(_scalings())
    assert mp.latent_heat_of_fusion == pytest.approx(4.0e5, rel=1e-12)
    assert mp.grain_size == pytest.approx(1.0e-3, rel=1e-12)


# ---- _PhaseParameters ------------------------------------------------------


def test_phase_parameters_scale_attributes_skips_string_lookup_paths():
    """Float fields are scaled; string fields (lookup-table paths) are
    skipped silently because the scaling helper raises TypeError on
    string division. Discriminator: a regression that removed the
    TypeError catch would crash at parse time on every TOML using a
    P-S lookup.
    """
    pp = _PhaseParameters(
        density=3300.0,
        heat_capacity='cp_lookup.dat',
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity='alpha_lookup.dat',
        viscosity=10.0,
    )
    pp.scale_attributes(_scalings())
    # Floats should be scaled (unity scale = pass-through).
    assert pp.density == pytest.approx(3300.0, rel=1e-12)
    assert pp.thermal_conductivity == pytest.approx(4.0, rel=1e-12)
    # Strings unchanged.
    assert pp.heat_capacity == 'cp_lookup.dat'
    assert pp.thermal_expansivity == 'alpha_lookup.dat'


def test_phase_parameters_handles_field_without_matching_scaling():
    """``melt_fraction`` has no matching attribute on _ScalingsParameters,
    so the AttributeError branch must be hit (logger.info path).
    Verify the value passes through unscaled.
    """
    pp = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=0.42,  # no 'melt_fraction' on ScalingsParameters
        thermal_conductivity=4.0,
        thermal_expansivity=3.0e-5,
        viscosity=100.0,
    )
    pp.scale_attributes(_scalings())
    # melt_fraction has no scale; unchanged.
    assert pp.melt_fraction == pytest.approx(0.42, rel=1e-12)


# ---- _Radionuclide.scale_attributes and get_heating ------------------------


def test_radionuclide_scale_attributes_converts_concentration_to_mass_fraction():
    """``scale_attributes`` multiplies concentration by 1e-6 (ppm to
    mass fraction). Discriminator: a regression that swapped this
    to /1e-6 would amplify radiogenic heating by 1e12.
    """
    r = _Radionuclide(
        name='U238',
        t0_years=4.6e9,
        abundance=0.992,
        concentration=20.0,  # ppm
        heat_production=9.46e-5,
        half_life_years=4.47e9,
    )
    r.scale_attributes(_scalings())
    assert r.concentration == pytest.approx(20.0e-6, rel=1e-12), (
        'ppm-to-mass-fraction conversion missing or wrong direction.'
    )


def test_radionuclide_get_heating_at_t0_returns_full_amplitude():
    """At time = t0_years the exponent is 0 and exp(0)=1.

    Use values where heat_production * abundance * concentration
    has 4 distinct factors so a wrong product order is detected.
    """
    r = _Radionuclide(
        name='K40',
        t0_years=4.55e9,
        abundance=1.17e-4,
        concentration=2.4e-4,
        heat_production=2.92e-5,
        half_life_years=1.25e9,
    )
    h_at_t0 = r.get_heating(time=4.55e9)
    expected = r.heat_production * r.abundance * r.concentration
    assert float(h_at_t0) == pytest.approx(expected, rel=1e-12)


def test_radionuclide_get_heating_one_half_life_earlier_doubles_amplitude():
    """At time = t0 - half_life, heating doubles (exp(ln 2) = 2).
    Catches sign-flip on (t0 - time) AND a missing log(2) factor.
    """
    r = _Radionuclide(
        name='Th232',
        t0_years=4.55e9,
        abundance=1.0,
        concentration=1.0,
        heat_production=1.0,
        half_life_years=1.4e10,
    )
    h_t0 = r.get_heating(time=4.55e9)
    h_one_hl_earlier = r.get_heating(time=4.55e9 - 1.4e10)
    assert float(h_one_hl_earlier) == pytest.approx(2.0 * float(h_t0), rel=1e-12)


def test_radionuclide_get_heating_array_input_returns_array():
    """Array of times returns same-shape array, monotonically
    decreasing forward in time (sign-of-exponent check).
    """
    r = _Radionuclide(
        name='U235',
        t0_years=4.55e9,
        abundance=7.2e-3,
        concentration=2.0e-8,
        heat_production=5.69e-4,
        half_life_years=7.04e8,
    )
    times = np.linspace(4.0e9, 6.0e9, 6)
    h = np.asarray(r.get_heating(times))
    assert h.shape == (6,)
    diffs = np.diff(h)
    assert np.all(diffs < 0), 'forward-time heating must decay monotonically'


# ---- _SolverParameters -----------------------------------------------------


def test_solver_scale_attributes_runs():
    """Exercise SolverParameters.scale_attributes: start_time, end_time,
    tsurf_poststep_change all divided by their scalings. Unity scales
    means pass-through.
    """
    s = _SolverParameters(start_time=0.0, end_time=1.0e6, atol=1e-9, rtol=1e-6)
    s.scale_attributes(_scalings())
    assert s.start_time == pytest.approx(0.0, abs=1e-30)
    assert s.end_time == pytest.approx(1.0e6, rel=1e-12)
    assert s.tsurf_poststep_change == pytest.approx(30.0, rel=1e-12)


# ---- Parameters.from_file (integration of all scale_attributes) ------------


def test_parameters_from_file_loads_bundled_abe_mixed_cfg():
    """Smoke-integration: read the bundled abe_mixed.cfg and verify
    every section produced the expected dataclass. This exercises
    ``from_file``, ``radionuclide_sections``, the parsing loop, and
    Parameters.__post_init__ together.
    """
    from aragog import CFG_DATA

    cfg_path = Path(str(CFG_DATA.joinpath('abe_mixed.cfg')))
    p = Parameters.from_file(cfg_path)
    assert isinstance(p, Parameters)
    assert isinstance(p.boundary_conditions, _BoundaryConditionsParameters)
    assert isinstance(p.energy, _EnergyParameters)
    assert isinstance(p.mesh, _MeshParameters)
    assert isinstance(p.solver, _SolverParameters)
    # The bundled cfg has K40, Th232, U235, U238 radionuclide sections.
    assert len(p.radionuclides) >= 1
    # Discriminator: post_init must convert ppm to mass-fraction (factor 1e-6).
    # K40 in abe_mixed.cfg has concentration=310 ppm -> 310e-6 mass fraction.
    k40 = next((r for r in p.radionuclides if r.name == 'K40'), None)
    if k40 is not None:
        assert k40.concentration == pytest.approx(310.0e-6, rel=1e-12), (
            'parser.Parameters.__post_init__ failed to convert ppm to '
            'mass fraction; downstream radiogenic heating would be 1e6x off.'
        )


def test_parameters_post_init_eos_method_2_validates_radius_range(tmp_path: Path):
    """Build Parameters by hand with eos_method=2 and a deliberately
    bad EOS file (radii below inner_radius - tol). Parameters.__post_init__
    must raise ValueError with a descriptive message.

    Discriminator: this is the secondary EOS-bracket gate inside
    Parameters; a regression that loosened the 5%-tolerance window
    would silently accept core-side samples.
    """
    f = tmp_path / 'eos_too_low.dat'
    rows = [(2.0e6, 1.0e11, 5000.0, 9.5), (3.0e6, 5.0e10, 4500.0, 9.8)]
    f.write_text('\n'.join(f'{r} {P} {rho} {g}' for r, P, rho, g in rows))
    sc = _ScalingsParameters()
    bc = _build_bc()
    en = _EnergyParameters(
        conduction=True,
        convection=False,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    ic = _InitialConditionParameters()
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=50,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=2,
        eos_file=str(f),
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    sv = _SolverParameters(start_time=0.0, end_time=1.0e6, atol=1e-9, rtol=1e-6)
    with pytest.raises(ValueError, match='Values out of range'):
        Parameters(
            boundary_conditions=bc,
            energy=en,
            initial_condition=ic,
            mesh=mesh,
            phase_solid=ps,
            phase_liquid=pl,
            phase_mixed=pm,
            radionuclides=[],
            scalings=sc,
            solver=sv,
        )


def test_parameters_post_init_initial_condition_method_2_no_file_raises():
    """When IC method 2 is set with empty init_file, Parameters.__post_init__
    must raise ValueError. This is the duplicate gate to the one tested
    on the parser dataclass; both layers must enforce.
    """
    sc = _ScalingsParameters()
    bc = _build_bc()
    en = _EnergyParameters(
        conduction=True,
        convection=False,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    ic = _InitialConditionParameters(initial_condition=2, init_file='')
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=50,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    sv = _SolverParameters(start_time=0.0, end_time=1.0e6, atol=1e-9, rtol=1e-6)
    with pytest.raises(ValueError, match='initial temperature file'):
        Parameters(
            boundary_conditions=bc,
            energy=en,
            initial_condition=ic,
            mesh=mesh,
            phase_solid=ps,
            phase_liquid=pl,
            phase_mixed=pm,
            radionuclides=[],
            scalings=sc,
            solver=sv,
        )


def test_parameters_post_init_radionuclide_concentration_converted_to_mass_fraction():
    """Parameters.__post_init__ multiplies each radionuclide concentration
    by 1e-6 (ppm → mass fraction). A regression that dropped this loop
    would leave concentrations as ppm and amplify heating by 1e6.
    """
    sc = _ScalingsParameters()
    bc = _build_bc()
    en = _EnergyParameters(
        conduction=True,
        convection=False,
        gravitational_separation=False,
        mixing=False,
        radionuclides=True,
        tidal=False,
    )
    ic = _InitialConditionParameters()
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=50,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    sv = _SolverParameters(start_time=0.0, end_time=1.0e6, atol=1e-9, rtol=1e-6)
    rad = _Radionuclide(
        name='K40',
        t0_years=4.55e9,
        abundance=1.17e-4,
        concentration=310.0,  # ppm
        heat_production=2.88e-5,
        half_life_years=1.25e9,
    )
    p = Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[rad],
        scalings=sc,
        solver=sv,
    )
    assert p.radionuclides[0].concentration == pytest.approx(310.0e-6, rel=1e-12)


def test_parameters_post_init_param_utbl_off_zeros_constant():
    """The duplicate UTBL-off gate on Parameters.__post_init__ must
    zero param_utbl_const when param_utbl is False, even if the
    user supplied a non-zero value. This is the safety belt
    behind the dataclass-level check.
    """
    sc = _ScalingsParameters()
    bc = _build_bc(param_utbl=False, param_utbl_const=2.0e-3)
    en = _EnergyParameters(
        conduction=True,
        convection=False,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    ic = _InitialConditionParameters()
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=50,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    sv = _SolverParameters(start_time=0.0, end_time=1.0e6, atol=1e-9, rtol=1e-6)
    p = Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[],
        scalings=sc,
        solver=sv,
    )
    assert p.boundary_conditions.param_utbl_const == pytest.approx(0.0, abs=1e-30)
