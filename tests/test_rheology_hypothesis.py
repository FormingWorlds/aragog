"""Hypothesis property-based tests for PhaseConfig and SolidRheologyParams validators."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.config import Config
from aragog.config.phases import PhaseConfig
from aragog.parser import Parameters, _PhaseParameters
from aragog.rheology import SolidRheologyParams, merge_rheology

pytestmark = pytest.mark.unit

RHEOLOGY_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(SolidRheologyParams))

POSITIVE_FIELDS: tuple[str, ...] = (
    't_sol_0',
    'arrhenius_t_ref',
    'water_prefactor',
    'yield_stress_max',
    'lid_base_temperature',
    'lid_contrast_coeff',
    'lid_mask_width_cells',
)

NON_NEGATIVE_FIELDS: tuple[str, ...] = (
    'c_theta',
    'g_homologous',
    'dt_sol_dp',
    'yield_stress_c',
    'yield_stress_mu',
)

UNIT_HALF_OPEN_FIELDS: tuple[str, ...] = (
    'mlt_bottom_slope',
    's_top_inf',
)

BOUNDED_INTERVAL_FIELDS: tuple[str, ...] = (
    'mlt_bottom_slope',
    's_top_inf',
    'viscosity_max_log10',
    'interior_flux_fraction',
    'phi_visc_single',
)

NUMERIC_FIELD_NAMES: tuple[str, ...] = (
    POSITIVE_FIELDS + NON_NEGATIVE_FIELDS + BOUNDED_INTERVAL_FIELDS
)


def _assert_wrong_types_raise_type_error(
    field_names: tuple[str, ...], wrong_types: tuple[Any, ...]
) -> None:
    """Assert that SolidRheologyParams raises TypeError for every (field, wrong_type) pair."""
    failures: list[str] = []
    for f_name in field_names:
        for wt in wrong_types:
            try:
                SolidRheologyParams(enabled=False, **{f_name: wt})  # type: ignore[arg-type]
                failures.append(f'{f_name}={wt!r}->accepted')
            except TypeError:
                pass
            except Exception as exc:
                failures.append(f'{f_name}={wt!r}->raised {type(exc).__name__}')
    assert not failures, f'Expected TypeError on wrong types, got: {failures}'


@st.composite
def valid_rheology_dict_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a valid dictionary covering all 20 SolidRheologyParams fields."""
    enabled = draw(st.booleans())
    g_min = 1.0e-6 if enabled else 0.0
    return {
        'enabled': enabled,
        'g_homologous': draw(
            st.floats(min_value=g_min, max_value=1.0e4, allow_nan=False, allow_infinity=False)
        ),
        't_sol_0': draw(
            st.floats(min_value=1.0e-3, max_value=1.0e5, allow_nan=False, allow_infinity=False)
        ),
        'dt_sol_dp': draw(
            st.floats(min_value=0.0, max_value=1.0e-3, allow_nan=False, allow_infinity=False)
        ),
        'arrhenius_t_ref': draw(
            st.floats(min_value=1.0e-3, max_value=1.0e5, allow_nan=False, allow_infinity=False)
        ),
        'viscosity_max_log10': draw(
            st.floats(
                min_value=np.nextafter(20.0, np.inf),
                max_value=300.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        'water_prefactor': draw(
            st.floats(min_value=1.0e-6, max_value=1.0e6, allow_nan=False, allow_infinity=False)
        ),
        'yield_stress_c': draw(
            st.floats(min_value=0.0, max_value=1.0e12, allow_nan=False, allow_infinity=False)
        ),
        'yield_stress_mu': draw(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        'yield_stress_max': draw(
            st.floats(min_value=1.0e-3, max_value=1.0e12, allow_nan=False, allow_infinity=False)
        ),
        'stress_closure_mode': draw(st.sampled_from(('lid', 'local'))),
        'interior_flux_fraction': draw(
            st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False)
        ),
        'lid_base_mode': draw(st.sampled_from(('fixed', 'rheological'))),
        'lid_base_temperature': draw(
            st.floats(min_value=1.0e-3, max_value=1.0e5, allow_nan=False, allow_infinity=False)
        ),
        'lid_contrast_coeff': draw(
            st.floats(min_value=1.0e-3, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        'lid_mask_width_cells': draw(
            st.floats(min_value=1.0e-3, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        'phi_visc_single': draw(
            st.floats(
                min_value=np.nextafter(0.0, 1.0),
                max_value=np.nextafter(1.0, 0.0),
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        'mlt_bottom_slope': draw(
            st.floats(
                min_value=np.nextafter(0.0, 1.0),
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        's_top_inf': draw(
            st.floats(
                min_value=np.nextafter(0.0, 1.0),
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        'c_theta': draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
    }


def _make_phase_config(**kwargs: Any) -> PhaseConfig:
    """Construct a PhaseConfig with standard base material properties and extra kwargs."""
    base: dict[str, Any] = {
        'density': 4000.0,
        'heat_capacity': 1000.0,
        'melt_fraction': 0.0,
        'thermal_conductivity': 4.0,
        'thermal_expansivity': 1.0e-5,
        'viscosity': 1.0e21,
        'entropy': '',
    }
    base.update(kwargs)
    return PhaseConfig(**base)


def _make_phase_parameters(**kwargs: Any) -> _PhaseParameters:
    """Construct a _PhaseParameters with standard base material properties and extra kwargs."""
    base: dict[str, Any] = {
        'density': 4000.0,
        'heat_capacity': 1000.0,
        'melt_fraction': 0.0,
        'thermal_conductivity': 4.0,
        'thermal_expansivity': 1.0e-5,
        'viscosity': 1.0e21,
        'entropy': '',
    }
    base.update(kwargs)
    return _PhaseParameters(**base)


@pytest.mark.unit
@settings(max_examples=50, deadline=None)
@given(
    rheo_dict=valid_rheology_dict_strategy(),
    override_dict=valid_rheology_dict_strategy(),
    subset_keys=st.lists(st.sampled_from(RHEOLOGY_FIELD_NAMES), unique=True),
)
def test_hypothesis_phase_config_rheology_construction_and_delegation(
    rheo_dict: dict[str, Any],
    override_dict: dict[str, Any],
    subset_keys: list[str],
) -> None:
    """Verify PhaseConfig and _PhaseParameters construction from SolidRheologyParams, flat kwargs, and dict."""
    params = SolidRheologyParams(**rheo_dict)
    for builder in (_make_phase_config, _make_phase_parameters):
        cfg_obj = builder(rheology=params)
        cfg_flat = builder(**rheo_dict)
        cfg_dict = builder(rheology=rheo_dict)

        assert cfg_obj.rheology == params
        assert cfg_flat.rheology == params
        assert cfg_dict.rheology == params

        for name in RHEOLOGY_FIELD_NAMES:
            expected_val = rheo_dict[name]
            assert hasattr(cfg_obj, name) and getattr(cfg_obj, name) == expected_val
            assert hasattr(cfg_flat, name) and getattr(cfg_flat, name) == expected_val
            assert hasattr(cfg_dict, name) and getattr(cfg_dict, name) == expected_val
            with pytest.raises(AttributeError):
                setattr(cfg_obj, name, expected_val)

        partial_overrides = {k: override_dict[k] for k in subset_keys}
        if partial_overrides.get('enabled', rheo_dict['enabled']) and (
            partial_overrides.get('g_homologous', rheo_dict['g_homologous']) <= 0.0
        ):
            with pytest.raises(ValueError, match='enabled=True requires positive g_homologous'):
                builder(rheology=params, **partial_overrides)
        else:
            cfg_merged = builder(rheology=params, **partial_overrides)
            expected_merged = SolidRheologyParams(**(rheo_dict | partial_overrides))
            assert cfg_merged.rheology == expected_merged

        if builder is _make_phase_config:
            with pytest.raises(ValueError, match='viscosity must not be None'):
                builder(viscosity=None, rheology=params)
        else:
            assert builder(viscosity=None, rheology=params).viscosity is None

    # TOML reachability: tomllib parses `true`/`false` as bool, reaching PhaseConfig and
    # _PhaseParameters -> merge_rheology -> SolidRheologyParams.
    import tomllib

    assert set(NUMERIC_FIELD_NAMES) == set(RHEOLOGY_FIELD_NAMES) - {
        'enabled',
        'stress_closure_mode',
        'lid_base_mode',
    }
    toml_failures: list[str] = []
    for num_field in NUMERIC_FIELD_NAMES:
        for bool_literal in ('true', 'false'):
            toml_data = tomllib.loads(
                '[phase_solid]\n'
                'density = 4000.0\n'
                'heat_capacity = 1000.0\n'
                'melt_fraction = 0.0\n'
                'thermal_conductivity = 4.0\n'
                'thermal_expansivity = 1.0e-5\n'
                'viscosity = 1.0e21\n'
                '[phase_solid.rheology]\n'
                f'{num_field} = {bool_literal}\n'
            )
            for cls in (PhaseConfig, _PhaseParameters):
                try:
                    cls(**toml_data['phase_solid'])
                    toml_failures.append(f'{cls.__name__}:{num_field}={bool_literal}->accepted')
                except TypeError:
                    pass
                except Exception as exc:
                    toml_failures.append(
                        f'{cls.__name__}:{num_field}={bool_literal}->raised {type(exc).__name__}'
                    )
    assert not toml_failures, f'TOML bools did not raise TypeError: {toml_failures}'


@pytest.mark.unit
@settings(max_examples=50, deadline=None)
@given(
    base_dict=valid_rheology_dict_strategy(),
    override_dict=valid_rheology_dict_strategy(),
    base_keys=st.lists(st.sampled_from(RHEOLOGY_FIELD_NAMES), unique=True),
    override_keys=st.lists(st.sampled_from(RHEOLOGY_FIELD_NAMES), unique=True),
    unknown_key=st.text(
        alphabet=st.characters(whitelist_categories=('Ll', 'Lu'), min_codepoint=97),
        min_size=1,
        max_size=20,
    ).filter(
        lambda s: (
            s not in RHEOLOGY_FIELD_NAMES
            and s
            not in {
                'density',
                'heat_capacity',
                'melt_fraction',
                'thermal_conductivity',
                'thermal_expansivity',
                'viscosity',
                'entropy',
                'rheology',
            }
        )
    ),
)
def test_hypothesis_merge_rheology_dict_round_trip_and_unknown_key(
    base_dict: dict[str, Any],
    override_dict: dict[str, Any],
    base_keys: list[str],
    override_keys: list[str],
    unknown_key: str,
) -> None:
    """Verify merge_rheology round-trip from dict and SolidRheologyParams and unknown-key error."""
    defaults = asdict(SolidRheologyParams())
    sub_base = {k: base_dict[k] for k in base_keys}
    if sub_base.get('enabled', defaults['enabled']) and (
        sub_base.get('g_homologous', defaults['g_homologous']) <= 0.0
    ):
        sub_base['g_homologous'] = 25.0

    sub_overrides = {k: override_dict[k] for k in override_keys}
    combined = defaults | sub_base | sub_overrides
    if combined['enabled'] and combined['g_homologous'] <= 0.0:
        with pytest.raises(ValueError, match='enabled=True requires positive g_homologous'):
            merge_rheology(sub_base, sub_overrides)
    else:
        merged_from_dict = merge_rheology(sub_base, sub_overrides)
        merged_from_obj = merge_rheology(SolidRheologyParams(**sub_base), sub_overrides)
        merged_from_none = merge_rheology(None, sub_base | sub_overrides)
        assert merged_from_dict == merged_from_obj == merged_from_none
        read_back = asdict(merged_from_dict)
        assert read_back == combined
        for k, v in combined.items():
            assert getattr(merged_from_dict, k) == v

    with pytest.raises(TypeError):
        merge_rheology({unknown_key: 1.0}, {})
    with pytest.raises(TypeError):
        merge_rheology(None, {unknown_key: 1.0})
    with pytest.raises(TypeError):
        merge_rheology(SolidRheologyParams(), {unknown_key: 1.0})
    with pytest.raises(TypeError):
        _make_phase_config(**{unknown_key: 1.0})
    with pytest.raises(
        TypeError,
        match=r'_PhaseParameters\.__init__\(\) got an unexpected keyword argument',
    ):
        _make_phase_parameters(**{unknown_key: 1.0})


@pytest.mark.unit
@settings(max_examples=50, deadline=None)
@given(
    non_bool=st.one_of(
        st.none(),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
    ),
    bad_mode=st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text().filter(lambda s: s not in ('lid', 'local', 'fixed', 'rheological')),
    ),
    non_pos_g=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_enabled_and_mode_validators(
    non_bool: Any,
    bad_mode: Any,
    non_pos_g: float,
) -> None:
    """Verify enabled, stress_closure_mode, lid_base_mode, and coupled g_homologous validators."""
    assert SolidRheologyParams(enabled=True, g_homologous=1.0).enabled is True
    assert SolidRheologyParams(enabled=False, g_homologous=0.0).enabled is False

    with pytest.raises(ValueError, match='enabled must be a bool'):
        SolidRheologyParams(enabled=non_bool)  # type: ignore[arg-type]

    for valid_sc in ('lid', 'local'):
        assert SolidRheologyParams(stress_closure_mode=valid_sc).stress_closure_mode == valid_sc
    for invalid_sc in (bad_mode, 'global', 'fixed', 'rheological'):
        with pytest.raises(ValueError, match='Unknown stress_closure_mode'):
            SolidRheologyParams(stress_closure_mode=invalid_sc)  # type: ignore[arg-type]

    for valid_lb in ('fixed', 'rheological'):
        assert SolidRheologyParams(lid_base_mode=valid_lb).lid_base_mode == valid_lb
    for invalid_lb in (bad_mode, 'lid', 'local'):
        with pytest.raises(ValueError, match='Unknown lid_base_mode'):
            SolidRheologyParams(lid_base_mode=invalid_lb)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match='enabled=True requires positive g_homologous'):
        SolidRheologyParams(enabled=True, g_homologous=0.0)
    if non_pos_g < 0.0:
        with pytest.raises(ValueError, match='g_homologous must be non-negative and finite'):
            SolidRheologyParams(enabled=True, g_homologous=non_pos_g)


@pytest.mark.unit
@settings(max_examples=50, deadline=None)
@given(
    pos_field=st.sampled_from(POSITIVE_FIELDS),
    non_neg_field=st.sampled_from(NON_NEGATIVE_FIELDS),
    valid_pos=st.floats(
        min_value=1.0e-12, max_value=1.0e12, allow_nan=False, allow_infinity=False
    ),
    neg_val=st.floats(max_value=-1.0e-12, allow_nan=False, allow_infinity=False),
    bad_str=st.text(),
)
def test_hypothesis_positive_and_non_negative_field_validators(
    pos_field: str,
    non_neg_field: str,
    valid_pos: float,
    neg_val: float,
    bad_str: str,
) -> None:
    """Verify positive and non-negative numeric field validators across bounds and wrong types."""
    # Positive fields: accept > 0, reject 0, negative, NaN, +-inf with ValueError, and
    # reject True, False, np.bool_, None, str with TypeError.
    p_valid = SolidRheologyParams(**{pos_field: valid_pos})
    assert getattr(p_valid, pos_field) == valid_pos

    min_pos = float(np.nextafter(0.0, 1.0))
    assert getattr(SolidRheologyParams(**{pos_field: min_pos}), pos_field) == min_pos

    for rejected_num in (0.0, -0.0, neg_val, float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValueError, match=pos_field):
            SolidRheologyParams(**{pos_field: rejected_num})

    # Non-negative fields (with enabled=False): accept 0 and > 0, reject < 0, NaN, +-inf
    # with ValueError, and reject True, False, np.bool_, None, str with TypeError.
    nn_zero = SolidRheologyParams(enabled=False, **{non_neg_field: 0.0})
    assert getattr(nn_zero, non_neg_field) == 0.0
    nn_pos = SolidRheologyParams(enabled=False, **{non_neg_field: valid_pos})
    assert getattr(nn_pos, non_neg_field) == valid_pos

    for rejected_nn in (neg_val, float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValueError, match=non_neg_field):
            SolidRheologyParams(enabled=False, **{non_neg_field: rejected_nn})

    _assert_wrong_types_raise_type_error(
        POSITIVE_FIELDS + NON_NEGATIVE_FIELDS,
        (True, False, np.bool_(True), np.bool_(False), None, bad_str),
    )


@pytest.mark.unit
@settings(max_examples=50, deadline=None)
@given(
    unit_field=st.sampled_from(UNIT_HALF_OPEN_FIELDS),
    unit_open_val=st.floats(
        min_value=np.nextafter(0.0, 1.0),
        max_value=np.nextafter(1.0, 0.0),
        allow_nan=False,
        allow_infinity=False,
    ),
    visc_val=st.floats(
        min_value=np.nextafter(20.0, np.inf),
        max_value=300.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    flux_val=st.floats(
        min_value=0.05,
        max_value=0.5,
        allow_nan=False,
        allow_infinity=False,
    ),
    neg_mag=st.floats(min_value=1.0e-9, max_value=1.0e6, allow_nan=False, allow_infinity=False),
    bad_str=st.text(),
)
def test_hypothesis_bounded_interval_field_validators(
    unit_field: str,
    unit_open_val: float,
    visc_val: float,
    flux_val: float,
    neg_mag: float,
    bad_str: str,
) -> None:
    """Verify interval bounds, endpoints, non-finites, and wrong types for bounded fields."""
    wrong_types = (True, False, np.bool_(True), np.bool_(False), None, bad_str)

    # 1. mlt_bottom_slope and s_top_inf: in (0, 1]
    assert (
        getattr(SolidRheologyParams(**{unit_field: unit_open_val}), unit_field) == unit_open_val
    )
    assert getattr(SolidRheologyParams(**{unit_field: 1.0}), unit_field) == 1.0
    for bad_u in (
        0.0,
        -neg_mag,
        float(np.nextafter(1.0, np.inf)),
        1.0 + neg_mag,
        float('nan'),
        float('inf'),
        float('-inf'),
    ):
        with pytest.raises(ValueError, match=unit_field):
            SolidRheologyParams(**{unit_field: bad_u})

    # 2. viscosity_max_log10: in (20, 300]
    assert SolidRheologyParams(viscosity_max_log10=visc_val).viscosity_max_log10 == visc_val
    assert SolidRheologyParams(viscosity_max_log10=300.0).viscosity_max_log10 == 300.0
    for bad_v in (
        20.0,
        float(np.nextafter(20.0, -np.inf)),
        0.0,
        -neg_mag,
        float(np.nextafter(300.0, np.inf)),
        300.0 + neg_mag,
        float('nan'),
        float('inf'),
        float('-inf'),
    ):
        with pytest.raises(ValueError, match='viscosity_max_log10'):
            SolidRheologyParams(viscosity_max_log10=bad_v)

    # 3. interior_flux_fraction: in [0.05, 0.5]
    assert (
        SolidRheologyParams(interior_flux_fraction=flux_val).interior_flux_fraction == flux_val
    )
    assert SolidRheologyParams(interior_flux_fraction=0.05).interior_flux_fraction == 0.05
    assert SolidRheologyParams(interior_flux_fraction=0.5).interior_flux_fraction == 0.5
    for bad_f in (
        float(np.nextafter(0.05, -np.inf)),
        0.0,
        -neg_mag,
        float(np.nextafter(0.5, np.inf)),
        0.5 + neg_mag,
        float('nan'),
        float('inf'),
        float('-inf'),
    ):
        with pytest.raises(ValueError, match='interior_flux_fraction'):
            SolidRheologyParams(interior_flux_fraction=bad_f)

    # 4. phi_visc_single: in (0, 1)
    assert SolidRheologyParams(phi_visc_single=unit_open_val).phi_visc_single == unit_open_val
    for bad_p in (
        0.0,
        1.0,
        -neg_mag,
        float(np.nextafter(1.0, np.inf)),
        1.0 + neg_mag,
        float('nan'),
        float('inf'),
        float('-inf'),
    ):
        with pytest.raises(ValueError, match='phi_visc_single'):
            SolidRheologyParams(phi_visc_single=bad_p)

    _assert_wrong_types_raise_type_error(BOUNDED_INTERVAL_FIELDS, wrong_types)


def _minimal_config_dict() -> dict[str, Any]:
    """Return a minimal valid dictionary for Config.from_dict."""
    return {
        'solver': {'start_time': 0.0, 'end_time': 1.0, 'atol': 1.0e-9, 'rtol': 1.0e-9},
        'boundary_conditions': {
            'outer_boundary_condition': 1,
            'outer_boundary_value': 1500.0,
            'inner_boundary_condition': 3,
            'inner_boundary_value': 4000.0,
            'emissivity': 1.0,
            'equilibrium_temperature': 273.0,
            'core_heat_capacity': 880.0,
        },
        'mesh': {
            'outer_radius': 6.371e6,
            'inner_radius': 3.480e6,
            'number_of_nodes': 10,
            'mixing_length_profile': 'nearest_boundary',
            'core_density': 10500.0,
        },
        'energy': {
            'conduction': True,
            'convection': True,
            'gravitational_separation': False,
            'mixing': False,
            'radionuclides': False,
            'tidal': False,
        },
        'initial_condition': {
            'initial_condition': 1,
            'surface_temperature': 3500.0,
            'basal_temperature': 3500.0,
        },
        'phase_liquid': {
            'density': 4000.0,
            'heat_capacity': 1000.0,
            'melt_fraction': 1.0,
            'thermal_conductivity': 4.0,
            'thermal_expansivity': 3.0e-5,
            'viscosity': 10.0,
        },
        'phase_solid': {
            'density': 4200.0,
            'heat_capacity': 1000.0,
            'melt_fraction': 0.0,
            'thermal_conductivity': 4.0,
            'thermal_expansivity': 3.0e-5,
            'viscosity': 1.0e21,
        },
        'phase_mixed': {
            'latent_heat_of_fusion': 4.0e5,
            'rheological_transition_melt_fraction': 0.4,
            'rheological_transition_width': 0.15,
            'solidus': 'solidus.dat',
            'liquidus': 'liquidus.dat',
            'phase': 'solid',
            'phase_transition_width': 0.01,
            'grain_size': 1.0e-3,
        },
    }


def _format_rheology_section(sub_dict: dict[str, Any], *, is_toml: bool) -> str:
    """Serialize a [phase_solid.rheology] section in TOML or INI syntax."""
    lines = ['\n[phase_solid.rheology]']
    for k, v in sub_dict.items():
        if isinstance(v, bool):
            val_str = ('true' if v else 'false') if is_toml else ('True' if v else 'False')
        elif isinstance(v, str):
            val_str = f'"{v}"' if is_toml else v
        else:
            val_str = f'{v!r}'
        lines.append(f'{k} = {val_str}')
    return '\n'.join(lines) + '\n'


@pytest.mark.unit
@settings(max_examples=50, deadline=None)
@given(
    rheo_dict=valid_rheology_dict_strategy(),
    subset_keys=st.lists(st.sampled_from(RHEOLOGY_FIELD_NAMES), unique=True),
)
def test_hypothesis_phase_solid_rheology_toml_ini_dict_round_trip(
    rheo_dict: dict[str, Any],
    subset_keys: list[str],
) -> None:
    """Verify [phase_solid.rheology] round-trips every field through TOML, INI, and Config.from_dict."""
    import importlib.resources
    import tempfile
    from pathlib import Path

    defaults = asdict(SolidRheologyParams())
    sub_dict = {k: rheo_dict[k] for k in subset_keys}
    if sub_dict.get('enabled', defaults['enabled']) and (
        sub_dict.get('g_homologous', defaults['g_homologous']) <= 0.0
    ):
        sub_dict['g_homologous'] = 25.0

    expected_full = SolidRheologyParams(**rheo_dict)
    expected_sub = SolidRheologyParams(**sub_dict)

    # 1. Config.from_dict: nested dict, SolidRheologyParams object, flat keys, and dotted section key
    base_data = _minimal_config_dict()
    assert Config.from_dict(base_data).phase_solid.rheology == SolidRheologyParams()

    d_nested = _minimal_config_dict()
    d_nested['phase_solid']['rheology'] = dict(rheo_dict)
    p_nested = Config.from_dict(d_nested)
    assert p_nested.phase_solid.rheology == expected_full
    assert p_nested.phase_liquid.rheology == SolidRheologyParams()

    d_obj = _minimal_config_dict()
    d_obj['phase_solid']['rheology'] = expected_full
    assert Config.from_dict(d_obj).phase_solid.rheology == expected_full

    d_flat = _minimal_config_dict()
    d_flat['phase_solid'].update(sub_dict)
    if sub_dict:
        with pytest.raises(ValueError, match=r'\[phase_solid\.rheology\]'):
            Config.from_dict(d_flat)
    else:
        assert Config.from_dict(d_flat).phase_solid.rheology == expected_sub

    d_dotted = _minimal_config_dict()
    d_dotted['phase_solid.rheology'] = dict(sub_dict)
    p_dotted = Config.from_dict(d_dotted)
    assert p_dotted.phase_solid.rheology == expected_sub

    d_liq_dotted = _minimal_config_dict()
    d_liq_dotted['phase_liquid.rheology'] = dict(sub_dict)
    assert Config.from_dict(d_liq_dotted).phase_liquid.rheology == expected_sub

    d_conflict = _minimal_config_dict()
    d_conflict['phase_solid']['rheology'] = dict(sub_dict)
    d_conflict['phase_solid.rheology'] = dict(sub_dict)
    with pytest.raises(ValueError, match='phase_solid'):
        Config.from_dict(d_conflict)

    d_liq_conflict = _minimal_config_dict()
    d_liq_conflict['phase_liquid']['rheology'] = dict(sub_dict)
    d_liq_conflict['phase_liquid.rheology'] = dict(sub_dict)
    with pytest.raises(ValueError, match='phase_liquid'):
        Config.from_dict(d_liq_conflict)

    # 2. TOML and INI file round-trip for full 20-field dict and subset dict
    toml_base = importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml').read_text()
    ini_base = importlib.resources.files('aragog').joinpath('cfg/abe_solid.cfg').read_text()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for label, payload, expected in (
            ('full', rheo_dict, expected_full),
            ('sub', sub_dict, expected_sub),
        ):
            toml_path = tmp / f'rheo_{label}.toml'
            toml_path.write_text(
                toml_base + _format_rheology_section(payload, is_toml=True),
                encoding='utf-8',
            )
            ini_path = tmp / f'rheo_{label}.cfg'
            ini_path.write_text(
                ini_base + _format_rheology_section(payload, is_toml=False),
                encoding='utf-8',
            )

            p_toml_parser = Parameters.from_file(toml_path)
            p_toml_config = Config.from_toml(str(toml_path))
            p_toml_auto = Config.from_file(str(toml_path))
            p_ini_parser = Parameters.from_file(ini_path)
            p_ini_auto = Config.from_file(str(ini_path))

            for loaded in (
                p_toml_parser,
                p_toml_config,
                p_toml_auto,
                p_ini_parser,
                p_ini_auto,
            ):
                assert loaded.phase_solid.rheology == expected
                assert loaded.phase_liquid.rheology == SolidRheologyParams()
                for name in RHEOLOGY_FIELD_NAMES:
                    val = getattr(expected, name)
                    assert getattr(loaded.phase_solid.rheology, name) == val
                    assert getattr(loaded.phase_solid, name) == val

        # Error contract on unknown key and invalid value in [phase_solid.rheology]
        bad_toml_key = tmp / 'bad_key.toml'
        bad_toml_key.write_text(
            toml_base + '\n[phase_solid.rheology]\nunknown_rheo_field = 1.0\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match=r'\[phase_solid'):
            Parameters.from_file(bad_toml_key)

        bad_ini_key = tmp / 'bad_key.cfg'
        bad_ini_key.write_text(
            ini_base + '\n[phase_solid.rheology]\nunknown_rheo_field = 1.0\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match=r'\[phase_solid'):
            Parameters.from_file(bad_ini_key)

        bad_toml_val = tmp / 'bad_val.toml'
        bad_toml_val.write_text(
            toml_base + '\n[phase_solid.rheology]\ng_homologous = -1.0\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match=r'\[phase_solid\.rheology\]'):
            Parameters.from_file(bad_toml_val)

        bad_ini_val = tmp / 'bad_val.cfg'
        bad_ini_val.write_text(
            ini_base + '\n[phase_solid.rheology]\ng_homologous = -1.0\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match=r'\[phase_solid\.rheology\]'):
            Parameters.from_file(bad_ini_val)

        bad_ini_bool = tmp / 'bad_bool.cfg'
        bad_ini_bool.write_text(
            ini_base + '\n[phase_solid.rheology]\nenabled = 1\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='enabled'):
            Parameters.from_file(bad_ini_bool)
