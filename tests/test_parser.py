"""Unit tests for ``aragog.parser``.

The parser layer is the dataclass-based legacy (pre-attrs) reader that
``Parameters.from_file`` relies on for INI/TOML config ingestion. It owns
the boundary-condition dispatch validators, the radionuclide decay model,
the EOS-file shape gate, and strict-rejection of the legacy ``[scalings]``
section. Bugs here are silent because they happen at parse time, before
any solver call, and a TOML that loaded yesterday must still load today.
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
    _SolverParameters,
)

pytestmark = pytest.mark.unit


# ---- module-level helpers --------------------------------------------------


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


def _build_minimal_parameters_kwargs(**overrides) -> dict:
    """Return constructor kwargs for a minimal Parameters instance.

    Used by integration tests that exercise __post_init__ end-to-end.
    Callers supply only the bits they care about; the rest defaults to
    a syntactically-valid baseline.
    """
    base = {
        'boundary_conditions': _build_bc(),
        'energy': _EnergyParameters(
            conduction=True,
            convection=False,
            gravitational_separation=False,
            mixing=False,
            radionuclides=False,
            tidal=False,
        ),
        'initial_condition': _InitialConditionParameters(),
        'mesh': _MeshParameters(
            outer_radius=6.371e6,
            inner_radius=3.480e6,
            number_of_nodes=50,
            mixing_length_profile='nearest_boundary',
            core_density=10500.0,
        ),
        'phase_liquid': _PhaseParameters(
            density=4000.0,
            heat_capacity=1000.0,
            melt_fraction=1.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            viscosity=10.0,
        ),
        'phase_solid': _PhaseParameters(
            density=4200.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=3e-5,
            viscosity=1e21,
        ),
        'phase_mixed': _PhaseMixedParameters(
            latent_heat_of_fusion=4.0e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='solidus.dat',
            liquidus='liquidus.dat',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1.0e-3,
        ),
        'radionuclides': [],
        'solver': _SolverParameters(start_time=0.0, end_time=1.0e6, atol=1e-9, rtol=1e-6),
    }
    base.update(overrides)
    return base


# ---- _get_dataclass_from_section_name --------------------------------------


def test_section_mapping_contains_all_required_sections():
    """The mapping must cover every required TOML section.

    Discriminator: a regression that dropped 'phase_mixed' or
    renamed 'energy' would silently let the parser ignore that
    block, with the dataclass falling back to defaults.
    """
    mapping = _get_dataclass_from_section_name()
    expected = {
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
        'splitting them silently breaks symmetric per-phase handling.'
    )
    # Discriminator: a regression that re-introduced [scalings] to the
    # mapping would silently re-enable a deprecated config layer that
    # `from_file` strict-rejects. Catch both halves of that regression.
    assert 'scalings' not in mapping, (
        'parser section mapping must not contain "scalings"; the section '
        'is strict-rejected at load time and cannot be reintroduced as '
        'a typed dataclass without breaking the strict-reject contract.'
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


# ---- _BoundaryConditionsParameters.normalize -------------------------------


def test_boundary_inner_bc_type_1_zeros_value():
    """Inner BC code 1 (simple core cooling): value forced to 0
    regardless of what the TOML supplied. Use a non-zero input to
    discriminate against silent passthrough.
    """
    bc = _build_bc(inner_boundary_condition=1, inner_boundary_value=12345.0)
    bc.normalize()
    assert bc.inner_boundary_value == pytest.approx(0.0, abs=1e-30), (
        'IBC 1 must zero the inner boundary value; a non-zero leak '
        'would re-add a phantom heat sink at the CMB.'
    )


def test_boundary_inner_bc_type_2_keeps_value():
    """IBC 2 (prescribed heat flux): value passes through unchanged
    after normalisation now that scaling has been removed.
    """
    bc = _build_bc(inner_boundary_condition=2, inner_boundary_value=4.5e6)
    bc.normalize()
    assert bc.inner_boundary_value == pytest.approx(4.5e6, rel=1e-12)


def test_boundary_inner_bc_type_3_keeps_value():
    """IBC 3 (prescribed temperature): value passes through unchanged."""
    bc = _build_bc(inner_boundary_condition=3, inner_boundary_value=4200.0)
    bc.normalize()
    assert bc.inner_boundary_value == pytest.approx(4200.0, rel=1e-12)


def test_boundary_inner_bc_unknown_code_raises_value_error():
    """Unphysical edge case: IBC 99 must raise, not silently pass.

    Discriminator: catches a regression that removed the else-raise
    after adding new BC types, which would silently ignore typos
    and run with default values.
    """
    bc = _build_bc(inner_boundary_condition=99, inner_boundary_value=0.0)
    with pytest.raises(ValueError, match='inner_boundary_condition = 99'):
        bc.normalize()


def test_boundary_outer_bc_type_4_keeps_value():
    """OBC 4 (prescribed heat flux): unchanged by normalise."""
    bc = _build_bc(outer_boundary_condition=4, outer_boundary_value=1.0e5)
    bc.normalize()
    assert bc.outer_boundary_value == pytest.approx(1.0e5, rel=1e-12)


def test_boundary_outer_bc_type_5_keeps_value():
    """OBC 5 (prescribed temperature): unchanged by normalise."""
    bc = _build_bc(outer_boundary_condition=5, outer_boundary_value=2000.0)
    bc.normalize()
    assert bc.outer_boundary_value == pytest.approx(2000.0, rel=1e-12)


def test_boundary_outer_bc_unknown_code_raises_value_error():
    """OBC 99 (unphysical, never documented) must raise."""
    bc = _build_bc(outer_boundary_condition=99)
    with pytest.raises(ValueError, match='outer_boundary_condition = 99'):
        bc.normalize()


def test_boundary_param_utbl_off_zeros_constant():
    """When param_utbl=False, the constant is forced to 0.0 even when
    a non-zero value is supplied. This is the production setting.
    """
    bc = _build_bc(param_utbl=False, param_utbl_const=1.0e-3)
    bc.normalize()
    assert bc.param_utbl_const == pytest.approx(0.0, abs=1e-30), (
        'param_utbl=False must zero the UTBL constant; a leak would '
        'enable the upper-thermal-boundary-layer correction by accident.'
    )


def test_boundary_param_utbl_on_keeps_constant():
    """When param_utbl=True, normalise leaves the constant alone (no
    longer multiplied by temperature**2 since scaling is removed).
    Use a non-zero const to discriminate against an accidental zero-out.
    """
    bc = _build_bc(param_utbl=True, param_utbl_const=2.5e-7)
    bc.normalize()
    assert bc.param_utbl_const == pytest.approx(2.5e-7, rel=1e-12)


# ---- _Radionuclide.get_heating ---------------------------------------------


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


# ---- Parameters.from_file integration --------------------------------------


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
        assert k40.concentration < 1.0, (
            f'K40 concentration {k40.concentration} not converted from ppm; '
            f'expected ~310e-6 = 3.1e-4.'
        )


def test_parameters_from_file_strict_rejects_scalings_section(tmp_path):
    """A configuration file containing a [scalings] section must
    raise ValueError at load time, not be silently ignored.

    Edge case: the section was previously parsed-but-overridden-to-1.0,
    so a regression that re-added the parser shim could silently
    re-introduce dimensional drift if any future divide-by-scale
    code path ever read non-unity values. Strict rejection
    forecloses that.
    """
    cfg = tmp_path / 'with_scalings.cfg'
    cfg.write_text(
        '[scalings]\nradius = 1.0\ntemperature = 1.0\ndensity = 1.0\ntime = 1.0\n'
        '\n[solver]\nstart_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-6\n'
    )
    with pytest.raises(ValueError, match=r'\[scalings\]'):
        Parameters.from_file(cfg)


@pytest.mark.parametrize('section_name', ['Scalings', 'SCALINGS', 'ScAlInGs'])
def test_parameters_from_file_strict_rejects_scalings_case_insensitive(tmp_path, section_name):
    """The strict-rejection of [scalings] must not depend on case.

    Edge case: typed_configparser preserves section-name case. A
    literal-string check would silently accept ``[Scalings]`` from
    a hand-edited TOML, defeating the rejection contract for the
    most common typo. The rejection must fire on any case variant.
    """
    cfg = tmp_path / 'mixed_case_scalings.cfg'
    cfg.write_text(
        f'[{section_name}]\nradius = 1.0\n'
        '\n[solver]\nstart_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-6\n'
    )
    with pytest.raises(ValueError, match=r'\[scalings\]'):
        Parameters.from_file(cfg)


def test_parameters_post_init_eos_method_2_radius_out_of_range_raises(tmp_path: Path):
    """Parameters.__post_init__ rejects an EOS file whose radius
    column starts BELOW the inner_radius (i.e. inside the core).

    Discriminator: this is the EOS-bracket gate inside
    Parameters; a regression that loosened the 5%-tolerance window
    would silently accept core-side samples.
    """
    f = tmp_path / 'eos_too_low.dat'
    rows = [(2.0e6, 1.0e11, 5000.0, 9.5), (3.0e6, 5.0e10, 4500.0, 9.8)]
    f.write_text('\n'.join(f'{r} {P} {rho} {g}' for r, P, rho, g in rows))
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=50,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=2,
        eos_file=str(f),
    )
    kwargs = _build_minimal_parameters_kwargs(mesh=mesh)
    with pytest.raises(ValueError, match='Values out of range'):
        Parameters(**kwargs)


def test_parameters_post_init_initial_condition_method_2_no_file_raises():
    """When IC method 2 is set with empty init_file, Parameters.__post_init__
    must raise ValueError.
    """
    ic = _InitialConditionParameters(initial_condition=2, init_file='')
    kwargs = _build_minimal_parameters_kwargs(initial_condition=ic)
    with pytest.raises(ValueError, match='initial temperature file'):
        Parameters(**kwargs)


def test_parameters_post_init_radionuclide_concentration_converted_to_mass_fraction():
    """Parameters.__post_init__ multiplies each radionuclide concentration
    by 1e-6 (ppm → mass fraction). A regression that dropped this loop
    would leave concentrations as ppm and amplify heating by 1e6.
    """
    rad = _Radionuclide(
        name='K40',
        t0_years=4.55e9,
        abundance=1.17e-4,
        concentration=310.0,  # ppm
        heat_production=2.88e-5,
        half_life_years=1.25e9,
    )
    energy = _EnergyParameters(
        conduction=True,
        convection=False,
        gravitational_separation=False,
        mixing=False,
        radionuclides=True,
        tidal=False,
    )
    kwargs = _build_minimal_parameters_kwargs(energy=energy, radionuclides=[rad])
    p = Parameters(**kwargs)
    assert p.radionuclides[0].concentration == pytest.approx(310.0e-6, rel=1e-12)


def test_parameters_post_init_param_utbl_off_zeros_constant():
    """The duplicate UTBL-off gate on Parameters.__post_init__ (via
    boundary_conditions.normalize()) must zero param_utbl_const when
    param_utbl is False, even if the user supplied a non-zero value.
    """
    bc = _build_bc(param_utbl=False, param_utbl_const=2.0e-3)
    kwargs = _build_minimal_parameters_kwargs(boundary_conditions=bc)
    p = Parameters(**kwargs)
    assert p.boundary_conditions.param_utbl_const == pytest.approx(0.0, abs=1e-30)
