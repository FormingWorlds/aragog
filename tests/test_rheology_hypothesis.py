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
from aragog.rheology import (
    SolidRheologyParams,
    compute_arrhenius_enthalpy,
    compute_arrhenius_viscosity,
    compute_mlt_top_slope,
    compute_strain_rate_local,
    compute_yield_stress,
    merge_rheology,
    viscous_mixing_length_factor,
)
from aragog.rheology_lid import compute_effective_viscosity, compute_stagnant_lid_state


def _require_jax() -> tuple[Any, Any]:
    """Import and return (jax, jnp) with x64 enabled, or skip if JAX is unavailable."""
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    try:
        jax.config.update('jax_enable_x64', True)
    except Exception:
        pass
    return jax, jnp


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
    POSITIVE_FIELDS
    + NON_NEGATIVE_FIELDS
    + BOUNDED_INTERVAL_FIELDS
    + ('activation_volume', 'activation_volume_decay_pressure')
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
    """Generate a valid dictionary covering all 23 SolidRheologyParams fields."""
    enabled = draw(st.booleans())
    g_min = 1.0e-6 if enabled else 0.0
    use_act = draw(st.booleans())
    if use_act:
        ea_min = 1.0e-3 if enabled else 0.0
        act_energy: float | None = draw(
            st.floats(min_value=ea_min, max_value=2.0e6, allow_nan=False, allow_infinity=False)
        )
        act_vol = draw(
            st.floats(min_value=0.0, max_value=1.0e-3, allow_nan=False, allow_infinity=False)
        )
        act_p_decay = draw(
            st.one_of(
                st.just(float('inf')),
                st.floats(
                    min_value=1.0e3, max_value=1.0e14, allow_nan=False, allow_infinity=False
                ),
            )
        )
    else:
        act_energy = None
        act_vol = 0.0
        act_p_decay = float('inf')
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
                min_value=np.nextafter(21.0, np.inf),
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
        'activation_energy': act_energy,
        'activation_volume': act_vol,
        'activation_volume_decay_pressure': act_p_decay,
    }


def _expected_cross_field_error(d: dict[str, Any]) -> str | None:
    """Return the expected ValueError substring for cross-field rules on d, or None if valid."""
    if d['activation_energy'] is None:
        if d['activation_volume'] != 0.0:
            return 'activation_volume requires activation_energy to be set'
        if np.isfinite(float(d['activation_volume_decay_pressure'])):
            return 'activation_volume_decay_pressure requires activation_energy to be set'
        if d['enabled'] and d['g_homologous'] <= 0.0:
            return 'enabled=True requires positive g_homologous'
    elif d['enabled'] and float(d['activation_energy']) <= 0.0:
        return 'enabled=True requires positive activation_energy'
    return None


def _normalize_sub_dict(sub: dict[str, Any], defaults: dict[str, Any]) -> None:
    """Ensure a random subset dict satisfies cross-field rules when merged with defaults."""
    eff_ea = sub.get('activation_energy', defaults['activation_energy'])
    eff_enabled = sub.get('enabled', defaults['enabled'])
    if eff_ea is None:
        if sub.get('activation_volume', defaults['activation_volume']) != 0.0 or np.isfinite(
            float(
                sub.get(
                    'activation_volume_decay_pressure',
                    defaults['activation_volume_decay_pressure'],
                )
            )
        ):
            sub['activation_energy'] = 3.0e5
        elif eff_enabled and sub.get('g_homologous', defaults['g_homologous']) <= 0.0:
            sub['g_homologous'] = 25.0
    elif eff_enabled and float(eff_ea) <= 0.0:
        sub['activation_energy'] = 3.0e5


def _assert_raises_with_match(
    exc_type: type[BaseException], match: str, kwargs: dict[str, Any]
) -> None:
    """Assert SolidRheologyParams(**kwargs) raises exc_type containing match via AssertionError."""
    caught: BaseException | None = None
    try:
        SolidRheologyParams(**kwargs)
    except BaseException as exc:  # noqa: BLE001
        caught = exc
    assert isinstance(caught, exc_type), (
        f'Expected {exc_type.__name__} matching {match!r} for {kwargs!r}, got {caught!r}'
    )
    assert match in str(caught), f'Expected {match!r} in {caught!r} message for {kwargs!r}'


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


def _scalar_wrong_types_strategy() -> st.SearchStrategy[Any]:
    """Generate non-numeric or non-scalar types: booleans, text, length-1 arrays, and lists."""
    return st.one_of(
        st.booleans(),
        st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cs',))),
        st.floats(min_value=0.1, max_value=1.0e10, allow_nan=False, allow_infinity=False).map(
            lambda x: np.array([x])
        ),
        st.sampled_from([[1.0e20], np.array([0.5]), np.array([1.0])]),
    )


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
        combined_dict = rheo_dict | partial_overrides
        cross_err = _expected_cross_field_error(combined_dict)
        if cross_err is not None:
            with pytest.raises(ValueError, match=cross_err):
                builder(rheology=params, **partial_overrides)
        else:
            cfg_merged = builder(rheology=params, **partial_overrides)
            expected_merged = SolidRheologyParams(**combined_dict)
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
        'activation_energy',
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
    _normalize_sub_dict(sub_base, defaults)

    sub_overrides = {k: override_dict[k] for k in override_keys}
    combined = defaults | sub_base | sub_overrides
    cross_err = _expected_cross_field_error(combined)
    if cross_err is not None:
        with pytest.raises(ValueError, match=cross_err):
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
    ea_pos=st.floats(min_value=1.0e-6, max_value=2.0e6, allow_nan=False, allow_infinity=False),
    v0_val=st.floats(min_value=0.0, max_value=1.0e-2, allow_nan=False, allow_infinity=False),
    v0_pos=st.floats(
        min_value=1.0e-12, max_value=1.0e-2, allow_nan=False, allow_infinity=False
    ),
    p_decay_pos=st.floats(
        min_value=1.0e-6, max_value=1.0e14, allow_nan=False, allow_infinity=False
    ),
    neg_val=st.floats(max_value=-1.0e-12, allow_nan=False, allow_infinity=False),
    bad_str=st.text(),
)
def test_hypothesis_activation_fields_and_cross_field_validators(
    ea_pos: float,
    v0_val: float,
    v0_pos: float,
    p_decay_pos: float,
    neg_val: float,
    bad_str: str,
) -> None:
    """Verify activation_energy, activation_volume, activation_volume_decay_pressure, and cross-field rules."""
    # 1. Valid ranges when activation_energy is set:
    p_valid = SolidRheologyParams(
        enabled=True,
        activation_energy=ea_pos,
        activation_volume=v0_val,
        activation_volume_decay_pressure=p_decay_pos,
    )
    assert p_valid.activation_energy == ea_pos
    assert p_valid.activation_volume == v0_val
    assert p_valid.activation_volume_decay_pressure == p_decay_pos

    p_zero_ea_inf_p = SolidRheologyParams(
        enabled=False,
        activation_energy=0.0,
        activation_volume=0.0,
        activation_volume_decay_pressure=float('inf'),
    )
    assert p_zero_ea_inf_p.activation_energy == 0.0
    assert p_zero_ea_inf_p.activation_volume == 0.0
    assert np.isinf(p_zero_ea_inf_p.activation_volume_decay_pressure)

    min_pos = float(np.nextafter(0.0, 1.0))
    p_min_p = SolidRheologyParams(
        enabled=False,
        activation_energy=ea_pos,
        activation_volume=v0_pos,
        activation_volume_decay_pressure=min_pos,
    )
    assert p_min_p.activation_volume_decay_pressure == min_pos

    # 2. Range validators when activation_energy is set:
    for bad_ea in (neg_val, float('nan'), float('inf'), float('-inf')):
        _assert_raises_with_match(
            ValueError,
            'activation_energy must be non-negative and finite',
            {'enabled': False, 'activation_energy': bad_ea},
        )
    for bad_v0 in (neg_val, float('nan'), float('inf'), float('-inf')):
        _assert_raises_with_match(
            ValueError,
            'activation_volume must be non-negative and finite',
            {'enabled': False, 'activation_energy': ea_pos, 'activation_volume': bad_v0},
        )
    for bad_p in (0.0, -0.0, neg_val, float('nan'), float('-inf')):
        _assert_raises_with_match(
            ValueError,
            'activation_volume_decay_pressure must be positive (finite or +inf)',
            {
                'enabled': False,
                'activation_energy': ea_pos,
                'activation_volume_decay_pressure': bad_p,
            },
        )

    # 3. Cross-field rules when activation_energy is None or enabled is True:
    _assert_raises_with_match(
        ValueError,
        'activation_volume requires activation_energy to be set',
        {'enabled': False, 'activation_energy': None, 'activation_volume': v0_pos},
    )
    _assert_raises_with_match(
        ValueError,
        'activation_volume_decay_pressure requires activation_energy to be set',
        {
            'enabled': False,
            'activation_energy': None,
            'activation_volume_decay_pressure': p_decay_pos,
        },
    )
    _assert_raises_with_match(
        ValueError,
        'enabled=True requires positive activation_energy',
        {'enabled': True, 'activation_energy': 0.0},
    )

    # 4. Wrong types on activation_energy, activation_volume, and activation_volume_decay_pressure:
    for wt_ea in (
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        bad_str,
        [ea_pos],
        np.array([ea_pos]),
    ):
        _assert_raises_with_match(
            TypeError,
            'activation_energy must be a real number',
            {'enabled': False, 'activation_energy': wt_ea},
        )
    for f_name in ('activation_volume', 'activation_volume_decay_pressure'):
        for wt_num in (True, False, np.bool_(True), np.bool_(False), None, bad_str):
            _assert_raises_with_match(
                TypeError,
                '',
                {'enabled': False, 'activation_energy': ea_pos, f_name: wt_num},
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
        if v is None:
            continue
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
    _normalize_sub_dict(sub_dict, defaults)

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


def _make_valid_lid_inputs(n: int = 10) -> dict[str, Any]:
    r = np.linspace(3480e3, 6371e3, n)
    t = np.linspace(3000.0, 1600.0, n)
    p = np.linspace(135e9, 0.0, n)
    f_conv = np.full(n, 100.0)
    f_tot = np.full(n, 120.0)
    phi = np.zeros(n)
    params = SolidRheologyParams(enabled=True)
    return {
        'radii': r,
        'temperature': t,
        'pressure': p,
        'convective_flux': f_conv,
        'total_flux': f_tot,
        'solidus_temperature': None,
        'melt_fraction': phi,
        'params': params,
        'viscosity_solid': 1e21,
        'phi_rheo': 0.4,
        'phi_width': 0.15,
    }


# Property 1: For p in [-1 Pa, 0), each of the 4 functions gives the output at p = 0 bitwise
@given(
    p=st.floats(min_value=-1.0, max_value=-1e-15, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=30)
def test_hypothesis_pressure_roundoff_bitwise_property(p: float) -> None:
    """Verify bitwise parity between p in [-1 Pa, 0) and p = 0 across pressure functions."""
    # 1. compute_arrhenius_enthalpy
    h_neg = compute_arrhenius_enthalpy(p, xp=np)
    h_zero = compute_arrhenius_enthalpy(0.0, xp=np)
    assert np.array_equal(h_neg, h_zero)

    # 2. compute_arrhenius_viscosity
    v_neg = compute_arrhenius_viscosity(1600.0, p, xp=np)
    v_zero = compute_arrhenius_viscosity(1600.0, 0.0, xp=np)
    assert np.array_equal(v_neg, v_zero)

    # 3. compute_yield_stress
    y_neg = compute_yield_stress(p, xp=np)
    y_zero = compute_yield_stress(0.0, xp=np)
    assert np.array_equal(y_neg, y_zero)

    # 4. compute_stagnant_lid_state
    kw = _make_valid_lid_inputs(n=10)
    kw_neg = {**kw, 'pressure': np.full(10, p)}
    kw_zero = {**kw, 'pressure': np.full(10, 0.0)}
    st_neg = compute_stagnant_lid_state(**kw_neg, xp=np)
    st_zero = compute_stagnant_lid_state(**kw_zero, xp=np)
    for k in st_zero.keys():
        assert np.array_equal(st_neg[k], st_zero[k], equal_nan=True)

    # JAX JIT bitwise parity for p in [-1, 0)
    jax, jnp = _require_jax()
    p_jnp_neg = jnp.asarray(p)
    p_jnp_zero = jnp.asarray(0.0)
    jh = jax.jit(lambda p_val: compute_arrhenius_enthalpy(p_val, xp=jnp))
    assert np.array_equal(np.asarray(jh(p_jnp_neg)), np.asarray(jh(p_jnp_zero)))

    jv = jax.jit(lambda p_val: compute_arrhenius_viscosity(1600.0, p_val, xp=jnp))
    assert np.array_equal(np.asarray(jv(p_jnp_neg)), np.asarray(jv(p_jnp_zero)))

    jy = jax.jit(lambda p_val: compute_yield_stress(p_val, xp=jnp))
    assert np.array_equal(np.asarray(jy(p_jnp_neg)), np.asarray(jy(p_jnp_zero)))


# Property 2: Underflow test reaches new lower bound and asserts eta >= tiny
@given(
    vs=st.floats(min_value=1e-30, max_value=1e-4, allow_nan=False, allow_infinity=False),
    temp=st.floats(min_value=1000.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30)
def test_hypothesis_arrhenius_underflow_bound_property(vs: float, temp: float) -> None:
    """Verify that Arrhenius viscosity under extreme cold conditions stays bounded above tiny floor."""
    tiny = np.finfo(float).tiny
    t_ref = 50.0
    g_hom = 50.0
    r_gas = 8.314462618
    t_sol_0 = 1400.0
    # Unclipped exponent
    h_p = g_hom * r_gas * t_sol_0
    h_0 = g_hom * r_gas * t_sol_0
    unclipped_exp = h_p / (r_gas * temp) - h_0 / (r_gas * t_ref)
    min_exp_bound = np.log(tiny) + 1e-5 - np.log(max(vs, 1e-300))
    min_exponent = max(-700.0, min_exp_bound)
    assert unclipped_exp < min_exponent

    eta = compute_arrhenius_viscosity(
        temp,
        0.0,
        viscosity_solid=vs,
        arrhenius_t_ref=t_ref,
        g_homologous=g_hom,
        xp=np,
    )
    assert eta >= tiny > 0.0
    assert np.isfinite(eta)


# Pressure rejection on eager NumPy and JAX JIT
@given(
    bad_p=st.one_of(
        st.floats(
            max_value=np.nextafter(-1.0, -np.inf),
            allow_nan=False,
            allow_infinity=True,
        ),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_pressure_invalid_rejection(bad_p: float) -> None:
    """Verify that pressures outside [-1 Pa, inf) are rejected on NumPy and produce NaN on JAX."""
    for fn in (
        lambda p: compute_arrhenius_enthalpy(p, xp=np),
        lambda p: compute_arrhenius_viscosity(1600.0, p, xp=np),
        lambda p: compute_yield_stress(p, xp=np),
    ):
        with pytest.raises(ValueError, match='pressure'):
            fn(bad_p)

    kw = _make_valid_lid_inputs(n=10)
    with pytest.raises(ValueError, match='pressure'):
        compute_stagnant_lid_state(**{**kw, 'pressure': np.full(10, bad_p)}, xp=np)

    # JAX JIT returns NaN
    jax, jnp = _require_jax()
    bad_p_jnp = jnp.asarray(bad_p)
    jh = jax.jit(lambda p: compute_arrhenius_enthalpy(p, xp=jnp))
    assert np.isnan(np.asarray(jh(bad_p_jnp)))

    jv = jax.jit(lambda p: compute_arrhenius_viscosity(1600.0, p, xp=jnp))
    assert np.isnan(np.asarray(jv(bad_p_jnp)))

    jy = jax.jit(lambda p: compute_yield_stress(p, xp=jnp))
    assert np.isnan(np.asarray(jy(bad_p_jnp)))


@given(
    bad_type=st.one_of(
        st.booleans(),
        st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cs',))),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_pressure_wrong_types(bad_type: Any) -> None:
    """Verify that non-numeric types for pressure raise ValueError."""
    for fn in (
        lambda p: compute_arrhenius_enthalpy(p, xp=np),
        lambda p: compute_arrhenius_viscosity(1600.0, p, xp=np),
        lambda p: compute_yield_stress(p, xp=np),
    ):
        with pytest.raises(ValueError, match='pressure'):
            fn(bad_type)


def test_hypothesis_pressure_nan_propagation() -> None:
    """Verify that NaN pressure propagates to NaN output across pressure functions."""
    assert np.isnan(compute_arrhenius_enthalpy(np.nan, xp=np))
    assert np.isnan(compute_arrhenius_viscosity(1600.0, np.nan, xp=np))
    assert np.isnan(compute_yield_stress(np.nan, xp=np))


@given(
    p_opt=st.floats(min_value=0.0, max_value=1e11, allow_nan=False, allow_infinity=False),
    t_opt=st.floats(min_value=300.0, max_value=4000.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_required_inputs_reject_none(p_opt: float, t_opt: float) -> None:
    """Verify that None for required inputs raises ValueError naming the field."""
    err = None
    try:
        compute_arrhenius_enthalpy(None, xp=np)
    except Exception as exc:
        err = exc
    assert isinstance(err, ValueError) and 'pressure' in str(err), (
        f'compute_arrhenius_enthalpy did not raise ValueError naming pressure for pressure=None, got {err}'
    )

    err = None
    try:
        compute_arrhenius_viscosity(None, p_opt, xp=np)
    except Exception as exc:
        err = exc
    assert isinstance(err, ValueError) and 'temperature' in str(err), (
        f'compute_arrhenius_viscosity did not raise ValueError naming temperature for temperature=None, got {err}'
    )

    err = None
    try:
        compute_arrhenius_viscosity(t_opt, None, xp=np)
    except Exception as exc:
        err = exc
    assert isinstance(err, ValueError) and 'pressure' in str(err), (
        f'compute_arrhenius_viscosity did not raise ValueError naming pressure for pressure=None, got {err}'
    )

    err = None
    try:
        compute_yield_stress(None, xp=np)
    except Exception as exc:
        err = exc
    assert isinstance(err, ValueError) and 'pressure' in str(err), (
        f'compute_yield_stress did not raise ValueError naming pressure for pressure=None, got {err}'
    )


# tau_d rejection on eager NumPy and JAX JIT
@given(
    bad_td=st.one_of(
        st.floats(
            max_value=np.nextafter(0.0, -np.inf),
            allow_nan=False,
            allow_infinity=True,
        ),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_tau_d_invalid_rejection(bad_td: float) -> None:
    """Verify that negative or non-finite tau_d is rejected on NumPy and produces NaN on JAX."""
    with pytest.raises(ValueError, match='tau_d'):
        compute_effective_viscosity(
            eta_diff=1e22,
            eta_i=1e20,
            tau_d=bad_td,
            tau_y_lid=1e8,
            stress_closure_mode='lid',
            xp=np,
        )

    # JAX JIT returns NaN
    jax, jnp = _require_jax()
    bad_td_jnp = jnp.asarray(bad_td)
    jit_eff = jax.jit(
        lambda td: compute_effective_viscosity(
            1e22, eta_i=1e20, tau_d=td, tau_y_lid=1e8, stress_closure_mode='lid', xp=jnp
        )
    )
    assert np.isnan(np.asarray(jit_eff(bad_td_jnp)))


@given(
    bad_type=st.one_of(
        st.booleans(),
        st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cs',))),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_tau_d_wrong_types(bad_type: Any) -> None:
    """Verify that non-numeric types for tau_d raise ValueError."""
    with pytest.raises(ValueError, match='tau_d'):
        compute_effective_viscosity(
            eta_diff=1e22,
            eta_i=1e20,
            tau_d=bad_type,
            tau_y_lid=1e8,
            stress_closure_mode='lid',
            xp=np,
        )


@given(
    eta_ratio=st.floats(min_value=0.3, max_value=3.0, allow_nan=False, allow_infinity=False),
    stress_ratio=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=30)
def test_hypothesis_tau_d_valid_finite(eta_ratio: float, stress_ratio: float) -> None:
    """Verify that effective viscosity matches independent soft-min reference formula."""
    eta_diff = 1.0e20
    eta_i = eta_diff * eta_ratio
    tau_y_lid = 1.0e8
    tau_d = tau_y_lid * stress_ratio

    eta = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=eta_i,
        tau_d=tau_d,
        tau_y_lid=tau_y_lid,
        stress_closure_mode='lid',
        xp=np,
    )
    assert np.isfinite(eta)

    w_width = 0.025
    cap_width = 0.01
    x_r = tau_d / tau_y_lid if tau_y_lid > 0.0 else 0.0
    w_b = 0.5 * (1.0 + np.tanh((x_r - 1.0) / w_width))
    if w_b <= 1.0e-15:
        expected = eta_diff
        assert np.isclose(eta, expected, rtol=1e-12, atol=1e-15)
    else:
        log10_d = np.log10(eta_diff)
        ratio = tau_y_lid / max(tau_d, 1.0e-30)
        log10_y = np.log10(eta_i * ratio)
        softmin = log10_y - cap_width * np.logaddexp(0.0, (log10_y - log10_d) / cap_width)
        expected = 10.0 ** ((1.0 - w_b) * log10_d + w_b * softmin)
        assert np.isclose(eta, expected, rtol=1e-12, atol=1e-15)

    eta_arr = compute_effective_viscosity(
        eta_diff=np.full(5, eta_diff),
        eta_i=np.full(5, eta_i),
        tau_d=np.full(5, tau_d),
        tau_y_lid=np.full(5, tau_y_lid),
        stress_closure_mode='lid',
        xp=np,
    )
    assert np.all(np.isfinite(eta_arr))
    if w_b <= 1.0e-15:
        assert np.allclose(eta_arr, expected, rtol=1e-12, atol=1e-15)
    else:
        assert np.allclose(eta_arr, expected, rtol=1e-11, atol=1e-15)


def test_hypothesis_tau_d_hard_min_reference_fails_at_yield_transition() -> None:
    """Verify that hard-min reference fails to match compute_effective_viscosity in softmin transition."""
    eta_diff = 1.0e20
    eta_i = 2.0 * eta_diff
    tau_y_lid = 1.0e8
    tau_d = 2.0 * tau_y_lid

    eta = compute_effective_viscosity(
        eta_diff=eta_diff,
        eta_i=eta_i,
        tau_d=tau_d,
        tau_y_lid=tau_y_lid,
        stress_closure_mode='lid',
        xp=np,
    )
    eta_y_hard = min(eta_diff, eta_i * (tau_y_lid / tau_d))
    w_width = 0.025
    x_r = tau_d / tau_y_lid
    w_b = 0.5 * (1.0 + np.tanh((x_r - 1.0) / w_width))
    expected_hard = (
        eta_y_hard
        if w_b >= 1.0 - 1.0e-15
        else 10.0 ** ((1.0 - w_b) * np.log10(eta_diff) + w_b * np.log10(eta_y_hard))
    )
    rel_diff = abs(eta - expected_hard) / eta
    assert rel_diff > 0.01
    assert not np.isclose(eta, expected_hard, rtol=1e-3)


# viscosity_solid validation
@given(
    bad_vs=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_viscosity_solid_invalid_rejection(bad_vs: float) -> None:
    """Verify that non-positive or infinite viscosity_solid raises ValueError."""
    with pytest.raises(ValueError, match='viscosity_solid'):
        compute_arrhenius_viscosity(1600.0, 0.0, viscosity_solid=bad_vs, xp=np)


@given(
    bad_type=_scalar_wrong_types_strategy(),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_viscosity_solid_wrong_types(bad_type: Any) -> None:
    """Verify that non-numeric types for viscosity_solid raise ValueError."""
    with pytest.raises(ValueError, match='viscosity_solid'):
        compute_arrhenius_viscosity(1600.0, 0.0, viscosity_solid=bad_type, xp=np)


def test_hypothesis_viscosity_solid_nan_propagation() -> None:
    """Verify that NaN viscosity_solid propagates to NaN output."""
    assert np.isnan(compute_arrhenius_viscosity(1600.0, 0.0, viscosity_solid=np.nan, xp=np))


# Helper 1: compute_mlt_top_slope(theta, s_top_inf, c_theta)
@given(
    bad_th=st.sampled_from([float('-inf'), float('inf')]),
)
@settings(deadline=None, max_examples=10)
def test_hypothesis_mlt_top_slope_theta_invalid(bad_th: float) -> None:
    """Verify that non-finite theta values raise ValueError on NumPy and return NaN on JAX."""
    with pytest.raises(ValueError, match='theta'):
        compute_mlt_top_slope(bad_th, xp=np)

    jax, jnp = _require_jax()
    bad_th_jnp = jnp.asarray(bad_th)
    jit_top = jax.jit(lambda th: compute_mlt_top_slope(th, xp=jnp))
    assert np.isnan(np.asarray(jit_top(bad_th_jnp)))


@given(
    bad_type=st.one_of(
        st.booleans(),
        st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cs',))),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_mlt_top_slope_theta_wrong_types(bad_type: Any) -> None:
    """Verify that non-numeric types for theta raise ValueError."""
    with pytest.raises(ValueError, match='theta'):
        compute_mlt_top_slope(bad_type, xp=np)


@given(
    bad_s=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
        st.floats(min_value=np.nextafter(1.0, np.inf), allow_nan=False, allow_infinity=True),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_mlt_top_slope_s_top_inf_invalid(bad_s: float) -> None:
    """Verify that s_top_inf outside (0, 1] raises ValueError."""
    with pytest.raises(ValueError, match='s_top_inf'):
        compute_mlt_top_slope(10.0, s_top_inf=bad_s, xp=np)


@given(
    bad_type=_scalar_wrong_types_strategy(),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_mlt_top_slope_s_top_inf_wrong_types(bad_type: Any) -> None:
    """Verify that non-numeric types for s_top_inf raise ValueError."""
    with pytest.raises(ValueError, match='s_top_inf'):
        compute_mlt_top_slope(10.0, s_top_inf=bad_type, xp=np)


@given(
    bad_c=st.one_of(
        st.floats(max_value=np.nextafter(0.0, -np.inf), allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_mlt_top_slope_c_theta_invalid(bad_c: float) -> None:
    """Verify that negative or infinite c_theta raises ValueError."""
    with pytest.raises(ValueError, match='c_theta'):
        compute_mlt_top_slope(10.0, c_theta=bad_c, xp=np)


@given(
    bad_type=_scalar_wrong_types_strategy(),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_mlt_top_slope_c_theta_wrong_types(bad_type: Any) -> None:
    """Verify that non-numeric types for c_theta raise ValueError."""
    with pytest.raises(ValueError, match='c_theta'):
        compute_mlt_top_slope(10.0, c_theta=bad_type, xp=np)


def test_hypothesis_mlt_top_slope_nan_propagation() -> None:
    """Verify that NaN inputs to compute_mlt_top_slope propagate to NaN output."""
    assert np.isnan(compute_mlt_top_slope(np.nan, xp=np))
    assert np.isnan(compute_mlt_top_slope(10.0, s_top_inf=np.nan, xp=np))
    assert np.isnan(compute_mlt_top_slope(10.0, c_theta=np.nan, xp=np))


@given(
    th=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    s=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False),
    c=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=30)
def test_hypothesis_mlt_top_slope_valid_finite(th: float, s: float, c: float) -> None:
    """Verify that MLT upper mixing-length slope matches reference formula across valid inputs."""
    res = compute_mlt_top_slope(th, s_top_inf=s, c_theta=c, xp=np)
    assert np.isfinite(res)

    expected = s + (1.0 - s) * np.exp(-c * max(th, 0.0))
    assert res == expected

    res_arr = compute_mlt_top_slope(np.full(5, th), s_top_inf=s, c_theta=c, xp=np)
    assert np.all(np.isfinite(res_arr))
    assert np.array_equal(res_arr, np.full(5, expected))


# Helper 2: compute_yield_stress(pressure, yield_stress_c, yield_stress_mu, yield_stress_max)
@given(
    bad_c=st.one_of(
        st.floats(max_value=np.nextafter(0.0, -np.inf), allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_yield_stress_c_invalid(bad_c: float) -> None:
    """Verify that negative or infinite yield_stress_c raises ValueError."""
    with pytest.raises(ValueError, match='yield_stress_c'):
        compute_yield_stress(1e9, yield_stress_c=bad_c, xp=np)


@given(
    bad_mu=st.one_of(
        st.floats(max_value=np.nextafter(0.0, -np.inf), allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_yield_stress_mu_invalid(bad_mu: float) -> None:
    """Verify that negative or infinite yield_stress_mu raises ValueError."""
    with pytest.raises(ValueError, match='yield_stress_mu'):
        compute_yield_stress(1e9, yield_stress_mu=bad_mu, xp=np)


@given(
    bad_max=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_yield_stress_max_invalid(bad_max: float) -> None:
    """Verify that non-positive or infinite yield_stress_max raises ValueError."""
    with pytest.raises(ValueError, match='yield_stress_max'):
        compute_yield_stress(1e9, yield_stress_max=bad_max, xp=np)


@given(
    bad_type=_scalar_wrong_types_strategy(),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_yield_stress_params_wrong_types(bad_type: Any) -> None:
    """Verify that non-numeric types for yield stress parameters raise ValueError."""
    with pytest.raises(ValueError, match='yield_stress_c'):
        compute_yield_stress(1e9, yield_stress_c=bad_type, xp=np)
    with pytest.raises(ValueError, match='yield_stress_mu'):
        compute_yield_stress(1e9, yield_stress_mu=bad_type, xp=np)
    with pytest.raises(ValueError, match='yield_stress_max'):
        compute_yield_stress(1e9, yield_stress_max=bad_type, xp=np)


def test_hypothesis_yield_stress_nan_propagation() -> None:
    """Verify that NaN inputs to compute_yield_stress propagate to NaN output."""
    assert np.isnan(compute_yield_stress(1e9, yield_stress_c=np.nan, xp=np))
    assert np.isnan(compute_yield_stress(1e9, yield_stress_mu=np.nan, xp=np))
    assert np.isnan(compute_yield_stress(1e9, yield_stress_max=np.nan, xp=np))


@given(
    p=st.floats(min_value=-1.0, max_value=1e12, allow_nan=False, allow_infinity=False),
    c=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    mu=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    mx=st.floats(min_value=1e-3, max_value=1e12, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=30)
def test_hypothesis_yield_stress_valid_finite(p: float, c: float, mu: float, mx: float) -> None:
    """Verify that yield stress matches independent reference formula across valid inputs."""
    res = compute_yield_stress(
        p, yield_stress_c=c, yield_stress_mu=mu, yield_stress_max=mx, xp=np
    )
    assert np.isfinite(res)

    p_eff = 0.0 if (-1.0 <= p < 0.0) else p
    expected = max(0.0, min(c + mu * p_eff, mx))
    assert res == expected

    res_arr = compute_yield_stress(
        np.full(5, p), yield_stress_c=c, yield_stress_mu=mu, yield_stress_max=mx, xp=np
    )
    assert np.all(np.isfinite(res_arr))
    assert np.array_equal(res_arr, np.full(5, expected))


# Helper 3: compute_strain_rate_local(viscous_velocity, mixing_length, eps)
@given(
    bad_v=st.sampled_from([float('-inf'), float('inf')]),
)
@settings(deadline=None, max_examples=10)
def test_hypothesis_strain_rate_velocity_invalid(bad_v: float) -> None:
    """Verify that infinite velocity raises ValueError on NumPy and returns NaN on JAX."""
    with pytest.raises(ValueError, match='viscous_velocity'):
        compute_strain_rate_local(bad_v, 1e5, xp=np)

    jax, jnp = _require_jax()
    bad_v_jnp = jnp.asarray(bad_v)
    jit_sr = jax.jit(lambda v: compute_strain_rate_local(v, 1e5, xp=jnp))
    assert np.isnan(np.asarray(jit_sr(bad_v_jnp)))


@given(
    bad_l=st.one_of(
        st.floats(max_value=np.nextafter(0.0, -np.inf), allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_strain_rate_mixing_length_invalid(bad_l: float) -> None:
    """Verify that negative or infinite mixing length raises ValueError or returns NaN on JAX."""
    with pytest.raises(ValueError, match='mixing_length'):
        compute_strain_rate_local(1e-9, bad_l, xp=np)

    jax, jnp = _require_jax()
    bad_l_jnp = jnp.asarray(bad_l)
    jit_sr = jax.jit(lambda l: compute_strain_rate_local(1e-9, l, xp=jnp))
    assert np.isnan(np.asarray(jit_sr(bad_l_jnp)))


@given(
    bad_eps=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_strain_rate_eps_invalid(bad_eps: float) -> None:
    """Verify that non-positive or infinite epsilon raises ValueError."""
    with pytest.raises(ValueError, match='eps'):
        compute_strain_rate_local(1e-9, 1e5, eps=bad_eps, xp=np)


@given(
    bad_type=st.one_of(
        st.booleans(),
        st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cs',))),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_strain_rate_wrong_types(bad_type: Any) -> None:
    """Verify that non-numeric types for strain rate arguments raise ValueError."""
    with pytest.raises(ValueError, match='viscous_velocity'):
        compute_strain_rate_local(bad_type, 1e5, xp=np)
    with pytest.raises(ValueError, match='mixing_length'):
        compute_strain_rate_local(1e-9, bad_type, xp=np)
    with pytest.raises(ValueError, match='eps'):
        compute_strain_rate_local(1e-9, 1e5, eps=bad_type, xp=np)


def test_hypothesis_strain_rate_nan_propagation() -> None:
    """Verify that NaN inputs to compute_strain_rate_local propagate to NaN output."""
    assert np.isnan(compute_strain_rate_local(np.nan, 1e5, xp=np))
    assert np.isnan(compute_strain_rate_local(1e-9, np.nan, xp=np))
    assert np.isnan(compute_strain_rate_local(1e-9, 1e5, eps=np.nan, xp=np))


@given(
    v=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    l_mix=st.floats(min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False),
    eps=st.floats(min_value=1e-20, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=30)
def test_hypothesis_strain_rate_valid_finite(v: float, l_mix: float, eps: float) -> None:
    """Verify that local strain rate matches reference formula across valid inputs."""
    res = compute_strain_rate_local(v, l_mix, eps=eps, xp=np)
    assert np.isfinite(res)

    expected = abs(v) / (2.0 * max(l_mix, eps))
    assert res == expected

    res_arr = compute_strain_rate_local(np.full(5, v), np.full(5, l_mix), eps=eps, xp=np)
    assert np.all(np.isfinite(res_arr))
    assert np.array_equal(res_arr, np.full(5, expected))


# Helper 4: viscous_mixing_length_factor(radii, r_in, r_out, mixing_length, s_top_inf, bottom_slope, theta, c_theta)
@given(
    bad_r=st.sampled_from([float('-inf'), float('inf')]),
)
@settings(deadline=None, max_examples=10)
def test_hypothesis_viscous_factor_radii_invalid(bad_r: float) -> None:
    """Verify that non-finite radius entries raise ValueError."""
    r = np.array([3480e3, bad_r, 6371e3])
    with pytest.raises(ValueError, match='radii'):
        viscous_mixing_length_factor(r, 3480e3, 6371e3, np.ones(3), xp=np)


@given(
    bad_rin=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_viscous_factor_rin_invalid(bad_rin: float) -> None:
    """Verify that non-positive or infinite r_in raises ValueError."""
    r = np.linspace(3480e3, 6371e3, 5)
    with pytest.raises(ValueError, match='r_in'):
        viscous_mixing_length_factor(r, bad_rin, 6371e3, np.ones(5), xp=np)


@given(
    bad_rout=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_viscous_factor_rout_invalid(bad_rout: float) -> None:
    """Verify that non-positive or infinite r_out raises ValueError."""
    r = np.linspace(3480e3, 6371e3, 5)
    with pytest.raises(ValueError, match='r_out'):
        viscous_mixing_length_factor(r, 3480e3, bad_rout, np.ones(5), xp=np)


@given(
    bad_l=st.one_of(
        st.floats(max_value=np.nextafter(0.0, -np.inf), allow_nan=False, allow_infinity=True),
        st.just(float('inf')),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_viscous_factor_mixing_length_invalid(bad_l: float) -> None:
    """Verify that negative or infinite mixing length raises ValueError."""
    r = np.linspace(3480e3, 6371e3, 5)
    with pytest.raises(ValueError, match='mixing_length'):
        viscous_mixing_length_factor(r, 3480e3, 6371e3, np.full(5, bad_l), xp=np)


@given(
    bad_b=st.one_of(
        st.floats(max_value=np.nextafter(0.0, -np.inf), allow_nan=False, allow_infinity=True),
        st.floats(min_value=np.nextafter(1.0, np.inf), allow_nan=False, allow_infinity=True),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_viscous_factor_bottom_slope_invalid(bad_b: float) -> None:
    """Verify that bottom_slope outside [0, 1] raises ValueError."""
    r = np.linspace(3480e3, 6371e3, 5)
    with pytest.raises(ValueError, match='bottom_slope'):
        viscous_mixing_length_factor(r, 3480e3, 6371e3, np.ones(5), bottom_slope=bad_b, xp=np)


@given(
    bad_type=st.one_of(
        st.booleans(),
        st.text(min_size=1, alphabet=st.characters(blacklist_categories=('Cs',))),
    ),
)
@settings(deadline=None, max_examples=25)
def test_hypothesis_viscous_factor_wrong_types(bad_type: Any) -> None:
    """Verify that non-numeric types for viscous factor arguments raise ValueError."""
    r = np.linspace(3480e3, 6371e3, 5)
    with pytest.raises(ValueError, match='radii'):
        viscous_mixing_length_factor(bad_type, 3480e3, 6371e3, np.ones(5), xp=np)
    with pytest.raises(ValueError, match='r_in'):
        viscous_mixing_length_factor(r, bad_type, 6371e3, np.ones(5), xp=np)
    with pytest.raises(ValueError, match='r_out'):
        viscous_mixing_length_factor(r, 3480e3, bad_type, np.ones(5), xp=np)
    with pytest.raises(ValueError, match='mixing_length'):
        viscous_mixing_length_factor(r, 3480e3, 6371e3, bad_type, xp=np)
    with pytest.raises(ValueError, match='bottom_slope'):
        viscous_mixing_length_factor(
            r, 3480e3, 6371e3, np.ones(5), bottom_slope=bad_type, xp=np
        )


def test_hypothesis_viscous_factor_nan_propagation() -> None:
    """Verify that NaN inputs to viscous_mixing_length_factor propagate to NaN output."""
    r = np.linspace(3480e3, 6371e3, 5)
    assert np.isnan(viscous_mixing_length_factor(r, np.nan, 6371e3, np.ones(5), xp=np)[0])
    assert np.isnan(viscous_mixing_length_factor(r, 3480e3, np.nan, np.ones(5), xp=np)[0])
    assert np.isnan(
        viscous_mixing_length_factor(r, 3480e3, 6371e3, np.full(5, np.nan), xp=np)[0]
    )
    assert np.isnan(
        viscous_mixing_length_factor(r, 3480e3, 6371e3, np.ones(5), bottom_slope=np.nan, xp=np)[
            0
        ]
    )


@given(
    s=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False),
    th=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    c=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=30)
def test_hypothesis_viscous_factor_valid_finite(
    s: float, b: float, th: float, c: float
) -> None:
    """Verify that viscous mixing length factor matches reference formula across valid inputs."""
    r = np.linspace(3480e3, 6371e3, 5)
    l_mix = np.linspace(0.0, 1e5, 5)
    res = viscous_mixing_length_factor(
        r,
        3480e3,
        6371e3,
        l_mix,
        s_top_inf=s,
        bottom_slope=b,
        theta=th,
        c_theta=c,
        xp=np,
    )
    assert np.all(np.isfinite(res))

    # Independent reference calculation
    s_top = s + (1.0 - s) * np.exp(-c * max(th, 0.0))
    lv = np.minimum(b * (r - 3480e3), s_top * (6371e3 - r))
    expected = np.where(l_mix > 0.0, (lv / np.where(l_mix > 0.0, l_mix, 1.0)) ** 4.0, 1.0)
    assert np.array_equal(res, expected)


@pytest.mark.unit
@settings(max_examples=40, deadline=None)
@given(
    phi_mid=st.lists(
        st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=10,
    ),
    f_scale=st.floats(min_value=0.1, max_value=0.95, allow_nan=False, allow_infinity=False),
)
def test_stagnant_lid_state_bounds(phi_mid: list[float], f_scale: float) -> None:
    """Verify general non-uniform d_lid_raw upper bound and fallback base weight bounds."""
    import aragog.rheology_lid as lid_mod
    from aragog.rheology_lid import _boundary_weights, compute_stagnant_lid_state

    melt_ramp = getattr(lid_mod, '_melt_ramp')
    n = len(phi_mid)
    r = np.linspace(3.48e6, 6.37e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t = np.linspace(2600.0, 300.0, n)
    fc = np.full(n, f_scale * 100.0)
    fc[-2:] = 0.0
    ft = np.full(n, 100.0)
    phi = np.asarray(phi_mid, dtype=float)
    params = SolidRheologyParams(enabled=True)

    st_res = compute_stagnant_lid_state(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=phi,
        params=params,
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert r[0] <= float(st_res['r_lid_base']) <= r[-1]
    assert 0.0 <= float(st_res['w_active']) <= 1.0
    assert np.all((st_res['w_lid'] >= 0.0) & (st_res['w_lid'] <= 1.0))
    assert np.isfinite(float(st_res['eta_i']))

    w_sol_phi = melt_ramp(
        phi,
        lid_mod.SOLIDUS_MELT_FRACTION_THRESHOLD,
        lid_mod.SOLIDUS_MELT_FRACTION_WIDTH,
        np,
        compact=True,
    )
    w_hot_phi = melt_ramp(phi, 0.4, 0.15, np, compact=False)
    grad_t = np.sqrt(np.gradient(t) ** 2 + 1.0e-30)
    smax2 = getattr(lid_mod, '_softmax2')
    scale = params.lid_mask_width_cells * smax2(grad_t, 1.0, lid_mod.SCALE_FLOOR_WIDTH, np)
    w_hot_iso = 0.5 * (1.0 + np.tanh((t - float(st_res['T_lid_iso'])) / scale))
    w_int = np.maximum(np.maximum(w_hot_iso, w_sol_phi), w_hot_phi)
    _, w_base_sum, w_base_weight = _boundary_weights(w_int, np)

    dr = np.diff(r)
    w_act = float(st_res['w_active'])
    d_lid_raw = w_act * max(float(r[-1] - st_res['r_lid_base']), 0.0)
    upper_bound = w_act * (
        float(w_base_weight)
        * float(np.sum((1.0 - w_sol_phi[1:]) * dr))
        / max(float(w_base_sum), 1.0e-12)
        + (1.0 - float(w_base_weight)) * float(r[-1] - r[0])
    )
    assert d_lid_raw <= upper_bound * (1.0 + 1.0e-12) + 1.0e-9

    # Cold fixed-lid column where w_interior ~ 0 requires w_base_weight = tanh(smax_w / 1e-6) -> 0
    # so r_lid_base relaxes to r[0] and d_lid_raw <= (r[-1] - r[0]) * w_active.
    st_cold = compute_stagnant_lid_state(
        radii=r,
        temperature=np.linspace(400.0, 300.0, n),
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=np.zeros(n),
        params=SolidRheologyParams(
            enabled=True, lid_base_mode='fixed', lid_base_temperature=5000.0
        ),
        viscosity_solid=1.0e21,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert float(st_cold['r_lid_base']) == pytest.approx(float(r[0]), rel=1.0e-9)
    assert float(st_cold['d_lid']) == pytest.approx(
        float(r[-1] - r[0]) * float(st_cold['w_active']), rel=1.0e-9
    )
    assert float(st_cold['d_lid']) <= float(r[-1] - r[0]) * float(st_cold['w_active']) * (
        1.0 + 1.0e-12
    )


@pytest.mark.unit
@settings(max_examples=40, deadline=None)
@given(
    x_offset=st.floats(min_value=0.55, max_value=2.0, allow_nan=False, allow_infinity=False),
    dec_offset=st.floats(min_value=1.35, max_value=5.0, allow_nan=False, allow_infinity=False),
    phi_low=st.floats(min_value=0.0, max_value=0.008, allow_nan=False, allow_infinity=False),
    phi_high=st.floats(min_value=0.012, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_smooth_rheology_blend_tails(
    x_offset: float, dec_offset: float, phi_low: float, phi_high: float
) -> None:
    """Verify blend tail agreement outside d* and compact _melt_ramp support via Hypothesis."""
    import aragog.rheology_lid as lid_mod
    from aragog.rheology import compute_arrhenius_viscosity
    from aragog.rheology_lid import compute_effective_viscosity

    melt_ramp = getattr(lid_mod, '_melt_ramp')
    assert float(melt_ramp(phi_low, 0.01, 0.002, np, compact=True)) == 0.0
    assert float(melt_ramp(0.01, 0.01, 0.002, np, compact=True)) == pytest.approx(
        0.5, rel=1.0e-14
    )
    assert float(melt_ramp(phi_high, 0.01, 0.002, np, compact=True)) == 1.0

    eta_d = 1.0e22
    eta_i = 1.0e16
    tau_y = 1.0e7
    x_above = 1.0 + x_offset
    eta_above = float(
        compute_effective_viscosity(
            eta_diff=eta_d,
            eta_i=eta_i,
            tau_d=x_above * tau_y,
            tau_y_lid=tau_y,
            w_lid=1.0,
            stress_closure_mode='lid',
        )
    )
    assert eta_above == pytest.approx(min(eta_d, eta_i / x_above), rel=1.0e-12)

    sr_cross = tau_y / (2.0 * eta_d)
    sr_high = sr_cross * (10.0**dec_offset)
    eta_loc = float(
        compute_effective_viscosity(
            eta_diff=eta_d,
            tau_y=tau_y,
            strain_rate=sr_high,
            stress_closure_mode='local',
        )
    )
    assert eta_loc == pytest.approx(tau_y / (2.0 * sr_high), rel=1.0e-12)

    vmax = 30.0
    ln_high = (vmax + dec_offset - 21.0) * np.log(10.0)
    t_cold = (25.0 * 1400.0) / (ln_high + 25.0 * 1400.0 / 1600.0)
    eta_ceil = float(
        compute_arrhenius_viscosity(
            t_cold,
            0.0,
            viscosity_solid=1.0e21,
            g_homologous=25.0,
            t_sol_0=1400.0,
            dt_sol_dp=0.0,
            arrhenius_t_ref=1600.0,
            viscosity_max_log10=vmax,
        )
    )
    assert eta_ceil == pytest.approx(10.0**vmax, rel=1.0e-12)


@pytest.mark.unit
@settings(max_examples=40, deadline=None)
@given(
    log10_vs=st.floats(min_value=15.0, max_value=35.0, allow_nan=False, allow_infinity=False),
    delta_vmax=st.floats(min_value=0.2, max_value=20.0, allow_nan=False, allow_infinity=False),
    f_thresh=st.floats(min_value=0.05, max_value=0.30, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_wp4_viscosity_max_cross_field_and_fixed_lid_convecting_gate(
    log10_vs: float, delta_vmax: float, f_thresh: float
) -> None:
    """Property test for viscosity_max_log10 > log10(viscosity_solid) and fixed-mode non-convective w_lid gate."""
    from aragog.config.phases import PhaseConfig
    from aragog.rheology import compute_arrhenius_viscosity

    vs = 10.0**log10_vs
    vmax_valid = max(20.1, min(300.0, log10_vs + delta_vmax))
    if vmax_valid > log10_vs:
        eta_ok = compute_arrhenius_viscosity(
            1600.0, 0.0, viscosity_solid=vs, viscosity_max_log10=vmax_valid
        )
        assert np.isfinite(eta_ok) and eta_ok > 0.0

    vmax_bad = max(20.1, min(300.0, log10_vs - delta_vmax))
    if vmax_bad <= log10_vs:
        with pytest.raises(ValueError, match='viscosity_max_log10'):
            compute_arrhenius_viscosity(
                1600.0, 0.0, viscosity_solid=vs, viscosity_max_log10=vmax_bad
            )
        with pytest.raises(ValueError, match='viscosity_max_log10'):
            PhaseConfig(
                density=4000.0,
                heat_capacity=1000.0,
                melt_fraction=0.0,
                thermal_conductivity=4.0,
                thermal_expansivity=3.0e-5,
                viscosity=vs,
                rheology=SolidRheologyParams(enabled=True, viscosity_max_log10=vmax_bad),
            )

    # Fixed lid-base mode: every node with f_k >= f_thresh + 0.2 * f_thresh has w_lid[k] == 0.0.
    n = 20
    r = np.linspace(3.48e6, 6.371e6, n)
    p = np.linspace(135.0e9, 1.0e5, n)
    t = np.full(n, 1400.0)
    t[-4:] = np.linspace(1400.0, 300.0, 4)
    ft = np.ones(n)
    fc = np.full(n, min(1.0, f_thresh * 1.25 + 1.0e-3))
    fc[-4:] = 0.0
    st = compute_stagnant_lid_state(
        radii=r,
        temperature=t,
        pressure=p,
        convective_flux=fc,
        total_flux=ft,
        solidus_temperature=None,
        melt_fraction=np.zeros(n),
        params=SolidRheologyParams(
            enabled=True,
            lid_base_mode='fixed',
            lid_base_temperature=2000.0,
            interior_flux_fraction=f_thresh,
        ),
        viscosity_solid=1.0e20,
        phi_rheo=0.4,
        phi_width=0.15,
        xp=np,
    )
    assert np.all(np.asarray(st['w_lid'])[:-4] == 0.0)


@pytest.mark.unit
@settings(max_examples=40, deadline=None)
@given(
    e_a=st.floats(min_value=50.0e3, max_value=1000.0e3, allow_nan=False, allow_infinity=False),
    v_0=st.floats(min_value=0.0, max_value=30.0e-6, allow_nan=False, allow_infinity=False),
    p_decay=st.floats(
        min_value=1.0e6, max_value=5000.0e9, allow_nan=False, allow_infinity=False
    ),
    p_hi=st.floats(min_value=1.0e7, max_value=1000.0e9, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_wp4_item5_saturating_activation_law(
    e_a: float, v_0: float, p_decay: float, p_hi: float
) -> None:
    """Property test for saturating integral activation enthalpy bounds, monotonicity, concavity, and JAX parity."""
    import jax
    import jax.numpy as jnp

    jax.config.update('jax_enable_x64', True)
    from aragog.rheology import compute_arrhenius_enthalpy

    p_mid = 0.5 * p_hi
    p_arr = np.array([0.0, p_mid, p_hi], dtype=float)
    h_np = np.asarray(
        compute_arrhenius_enthalpy(
            p_arr,
            activation_energy=e_a,
            activation_volume=v_0,
            activation_volume_decay_pressure=p_decay,
            xp=np,
        ),
        dtype=float,
    )
    h_jx = np.asarray(
        compute_arrhenius_enthalpy(
            jnp.asarray(p_arr),
            activation_energy=e_a,
            activation_volume=v_0,
            activation_volume_decay_pressure=p_decay,
            xp=jnp,
        ),
        dtype=float,
    )
    np.testing.assert_allclose(h_np, h_jx, rtol=1.0e-14, atol=0.0)
    # Exact zero-pressure anchor, non-decreasing monotonicity, concavity, and upper envelope bound:
    assert h_np[0] == pytest.approx(e_a, rel=1.0e-15, abs=0.0)
    assert h_np[0] <= h_np[1] * (1.0 + 1.0e-15)
    assert h_np[1] <= h_np[2] * (1.0 + 1.0e-15)
    # Concavity: midpoint enthalpy >= chord average 0.5 * (H(0) + H(p_hi))
    assert h_np[1] + 1.0e-9 >= 0.5 * (h_np[0] + h_np[2])
    upper_bound = e_a + v_0 * min(p_hi, p_decay)
    assert h_np[2] <= upper_bound * (1.0 + 1.0e-14)
