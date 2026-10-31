"""Zero-amplitude radionuclides contribute no heating; an overflowing live one is rejected."""

from __future__ import annotations

import re
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.cli import cli
from aragog.parser import Parameters, _Radionuclide, validate_radionuclide
from aragog.rheology import SolidRheologyParams

from .conftest import EOS_DIR, needs_eos

BUNDLED_LOOKUP = Path(__file__).resolve().parents[1] / 'src/aragog/cfg/abe_mixed_lookup.cfg'
AL26_ZERO = dict(
    name='Al26',
    t0_years=4.55e9,
    abundance=0.0,
    concentration=0.0,
    heat_production=0.3583,
    half_life_years=0.717e6,
)


def _lookup_cfg(tmp_path, *, end_time=None, al26_live=False) -> Path:
    """Copy of the bundled lookup cfg with a shorter end time or a live Al26."""
    text = BUNDLED_LOOKUP.read_text()
    if end_time is not None:
        text = text.replace('end_time = 1000\n', f'end_time = {end_time}\n')
        assert f'end_time = {end_time}\n' in text
    if al26_live:
        head, al26 = text.split('[radionuclide_Al26]')
        al26, fe60 = al26.split('[radionuclide_Fe60]')
        al26, n = re.subn(r'(?m)^(abundance|concentration) = 0$', r'\1 = 1', al26)
        assert n == 2
        text = head + '[radionuclide_Al26]' + al26 + '[radionuclide_Fe60]' + fe60
    path = tmp_path / 'lookup.cfg'
    path.write_text(text)
    return path


@pytest.mark.unit
@pytest.mark.parametrize('half_life', [0.0, np.nan])
def test_zero_amplitude_isotope_heats_zero_where_exp_overflows(half_life):
    """A zero-amplitude isotope gives 0 without evaluating its exponent: no overflow,
    and no division for a zero or NaN half-life (errstate raises on any of them)."""
    r = _Radionuclide(**AL26_ZERO)
    with np.errstate(all='raise'):
        try:
            values = (
                r.get_heating(0.0),
                *r.get_heating(np.array([0.0, 1e3])),
                _Radionuclide(**{**AL26_ZERO, 'half_life_years': half_life}).get_heating(0.0),
            )
        except FloatingPointError as exc:
            pytest.fail(f'the zero amplitude evaluated its exponent: {exc}')
    assert values == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.unit
def test_bundled_lookup_cfg_heating_is_finite():
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    heat = {r.name: r.get_heating(p.solver.start_time) for r in p.radionuclides}
    assert heat.pop('Al26') == heat.pop('Fe60') == 0.0
    assert all(np.isfinite(h) and h > 0.0 for h in heat.values())


@pytest.mark.unit
@pytest.mark.parametrize('half_life', ['-0.717E6', '0'])
def test_non_positive_half_life_raises(tmp_path, half_life):
    cfg = _lookup_cfg(tmp_path)
    cfg.write_text(
        cfg.read_text().replace('half_life_years = 0.717E6', f'half_life_years = {half_life}')
    )
    with pytest.raises(ValueError, match='Al26: half_life_years must be positive'):
        Parameters.from_file(str(cfg))


@pytest.mark.unit
def test_live_isotope_overflowing_at_start_raises(tmp_path):
    cfg = _lookup_cfg(tmp_path, al26_live=True)
    with pytest.raises(ValueError, match=r'Al26.*start time 0.*t0_years = 4550000000'):
        Parameters.from_file(str(cfg))


@pytest.mark.unit
def test_jax_zero_amplitude_value_and_jacobian_are_finite():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from aragog.jax.solver import make_radio_heating_fn

    rs = Parameters.from_file(str(BUNDLED_LOOKUP)).radionuclides
    keys = ('heat_production', 'abundance', 'concentration', 't0_years', 'half_life_years')
    h = make_radio_heating_fn(*(np.array([getattr(r, k) for r in rs]) for k in keys))
    value, slope = float(h(0.0)), float(jax.jacrev(h)(0.0))
    assert np.isfinite(slope) and slope < 0.0
    assert value == pytest.approx(sum(r.get_heating(0.0) for r in rs), rel=1e-12)


@pytest.mark.smoke
@needs_eos
def test_bundled_lookup_cfg_runs_with_finite_state(tmp_path):
    import netCDF4

    out = tmp_path / 'o.nc'
    args = [
        'run',
        str(_lookup_cfg(tmp_path, end_time=1)),
        '--eos-dir',
        str(EOS_DIR),
        '--out',
        str(out),
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert 'integration failed' not in result.output
    with netCDF4.Dataset(out) as d:
        assert float(d['time'][:]) == pytest.approx(1.0)
        assert np.isfinite(float(d['T_magma'][:]))


K40_PPM = dict(
    name='K40',
    t0_years=4.55e9,
    abundance=1.1668e-4,
    concentration=310.0,
    heat_production=2.8761e-5,
    half_life_years=1248e6,
)


def _with(p, *isotopes):
    """``p`` rebuilt with new ``_Radionuclide`` objects (concentrations in ppm)."""
    rs = [_Radionuclide(**r) for r in isotopes]
    return replace(p, radionuclides=rs), rs


@pytest.mark.unit
def test_rejected_parameters_leave_the_isotopes_unscaled():
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    rs = [
        _Radionuclide(**K40_PPM),
        _Radionuclide(**{**AL26_ZERO, 'abundance': 1.0, 'concentration': 1.0}),
    ]
    with pytest.raises(ValueError, match='Al26'):
        replace(p, radionuclides=rs)
    assert [r.concentration for r in rs] == [310.0, 1.0]
    rs[1].abundance, rs[1].concentration = 0.0, 0.0
    replace(p, radionuclides=rs)
    assert [r.concentration for r in rs] == [310.0 * 1e-6, 0.0]


@pytest.mark.unit
@pytest.mark.parametrize('refusal', ['liquid_rheology', 'solid_ceiling'])
def test_rheology_refusal_leaves_the_isotopes_unscaled(refusal):
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    rs = [_Radionuclide(**K40_PPM), _Radionuclide(**{**K40_PPM, 'concentration': 1.0})]
    on = SolidRheologyParams(enabled=True)
    if refusal == 'liquid_rheology':
        change = {'phase_liquid': replace(p.phase_liquid, rheology=on)}
        match = 'phase_liquid.rheology must keep default values'
    else:
        # The phase constructor checks the ceiling too, so raise the viscosity afterwards.
        change = {'phase_solid': replace(p.phase_solid, rheology=on)}
        change['phase_solid'].viscosity = 1e41
        match = r'viscosity_max_log10 \(40.0\) must be > log10\(viscosity\) \(41.0\)'
    with pytest.raises(ValueError, match=match):
        replace(p, radionuclides=rs, **change)
    assert [r.concentration for r in rs] == [310.0, 1.0]


@pytest.mark.unit
def test_negative_factors_raise():
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    two_negative = {**K40_PPM, 'heat_production': -2.8761e-5, 'concentration': -310.0}
    with pytest.raises(ValueError, match=r'K40: heat_production \(-2.8761e-05\).*-310.0 ppm'):
        _with(p, two_negative)


@pytest.mark.unit
def test_tiny_negative_concentration_raises():
    # -1e-320 ppm scales to -0.0, so the sign is checked on the ppm input.
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    with pytest.raises(ValueError, match='K40: .*must not be negative'):
        _with(p, {**K40_PPM, 'concentration': -1e-320})


@pytest.mark.unit
def test_finiteness_is_checked_on_the_mass_fraction():
    # exp(arg) ~ 1e305: 1e4 ppm overflows unconverted (1e309) but not as a mass fraction (1e303).
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    t0 = np.log(1e305) / np.log(2) * 1e6
    big = dict(
        name='X',
        t0_years=t0,
        abundance=1.0,
        concentration=1e4,
        heat_production=1.0,
        half_life_years=1e6,
    )
    try:
        q, (r,) = _with(p, big)
    except ValueError as exc:
        pytest.fail(f'the finiteness check used the ppm value: {exc}')
    assert r.concentration == pytest.approx(1e-2) and np.isfinite(
        r.get_heating(q.solver.start_time)
    )


@pytest.mark.unit
@pytest.mark.parametrize('on', [True, False])
def test_checks_run_only_with_radionuclides_on(tmp_path, on):
    cfg = _lookup_cfg(tmp_path)
    text = cfg.read_text().replace('half_life_years = 0.717E6', 'half_life_years = -1')
    if not on:
        text = text.replace('radionuclides = True', 'radionuclides = False')
    cfg.write_text(text)
    if on:
        with pytest.raises(ValueError, match='half_life_years must be positive'):
            Parameters.from_file(str(cfg))
    else:
        try:
            Parameters.from_file(str(cfg))
        except ValueError as exc:
            pytest.fail(f'an isotope was checked with radionuclides off: {exc}')


@pytest.mark.unit
def test_live_isotope_is_checked_at_the_start_time(tmp_path):
    cfg = _lookup_cfg(tmp_path, al26_live=True)
    cfg.write_text(cfg.read_text().replace('start_time = 0\n', 'start_time = 4.549e9\n'))
    p = Parameters.from_file(str(cfg))
    al26 = next(r for r in p.radionuclides if r.name == 'Al26')
    assert np.isfinite(al26.get_heating(4.549e9)) and al26.get_heating(4.549e9) > 0.0


@pytest.mark.unit
def test_live_isotope_overflowing_only_at_the_start_time_raises(tmp_path):
    # With t0 = 1e8 yr the heating is finite at t = 0 and overflows at the start time -2e9 yr.
    cfg = _lookup_cfg(tmp_path, al26_live=True)
    text = cfg.read_text()
    head, al26 = text.split('[radionuclide_Al26]')
    al26, n = re.subn(r'(?m)^t0_years = .*$', 't0_years = 1e8', al26, count=1)
    assert n == 1
    text = (head + '[radionuclide_Al26]' + al26).replace(
        'start_time = 0\n', 'start_time = -2e9\n'
    )
    assert 'start_time = -2e9\n' in text
    cfg.write_text(text)
    with pytest.raises(ValueError, match=r'Al26.*start time -2000000000.0'):
        Parameters.from_file(str(cfg))
    cfg.write_text(text.replace('start_time = -2e9\n', 'start_time = 0\n'))
    Parameters.from_file(str(cfg))


@pytest.mark.unit
def test_replace_converts_each_concentration_from_ppm_once():
    """Each concentration becomes a mass fraction once: dataclasses.replace on Parameters or
    on an isotope (other fields changed) keeps it, a new ppm value (replace or assignment)
    is converted. A missed or a second conversion is a factor 1e6."""
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    k40 = next(r for r in p.radionuclides if r.name == 'K40')
    assert k40.concentration == pytest.approx(310.0e-6, rel=1e-15)
    first = [r.concentration for r in p.radionuclides]
    q = replace(replace(p), solver=p.solver)
    assert [r.concentration for r in q.radionuclides] == first
    longer = replace(k40, half_life_years=2.0 * k40.half_life_years)
    new = replace(k40, concentration=200.0)
    replace(p, radionuclides=[longer, new])
    assert longer.concentration == pytest.approx(310.0e-6, rel=1e-15)
    assert new.concentration == pytest.approx(200.0e-6, rel=1e-15)
    k40.concentration = 150.0
    replace(p)
    assert k40.concentration == pytest.approx(150.0e-6, rel=1e-15)
    assert new.get_heating(k40.t0_years) == pytest.approx(
        new.heat_production * new.abundance * 200.0e-6, rel=1e-15
    )


@pytest.mark.unit
def test_zero_concentration_is_stable_under_replace_and_assignment():
    """A concentration of 0 (and _ppm 0) stays 0 through replace and assignment, and a
    later assigned ppm value is still converted."""
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    al26 = next(r for r in p.radionuclides if r.name == 'Al26')
    assert al26.concentration == 0.0 and al26._ppm == 0.0
    shorter = replace(al26, half_life_years=0.5 * al26.half_life_years)
    q = replace(replace(p), radionuclides=[al26, shorter])
    assert [r.concentration for r in q.radionuclides] == [0.0, 0.0]
    al26.concentration = 0.0
    replace(p)
    assert al26.concentration == 0.0 and al26._ppm == 0.0
    al26.concentration = 5.0
    replace(p)
    assert al26.concentration == pytest.approx(5.0e-6, rel=1e-15) and al26._ppm == 5.0


@pytest.mark.unit
@pytest.mark.parametrize('field', ['heat_production', 'abundance', 'concentration', 't0_years'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_parameters_reject_a_non_finite_isotope_input(field, bad):
    """A non-finite input stops Parameters before any isotope is scaled; t0_years = -inf
    gives a finite (zero) heating, so only the finite check catches it."""
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    with pytest.raises(ValueError, match='K40: .*must be finite'):
        _with(p, {**K40_PPM, field: bad})


@pytest.mark.unit
def test_a_config_section_cannot_set_the_ppm_record(tmp_path):
    """_ppm in a radionuclide section would mark the concentration as converted, so it is
    rejected; the same section without it loads and converts."""
    cfg = _lookup_cfg(tmp_path)
    text = cfg.read_text()
    cfg.write_text(text.replace('concentration = 310', 'concentration = 310\n_ppm = 310', 1))
    with pytest.raises(ValueError, match='K40: _ppm is not a configuration key'):
        Parameters.from_file(str(cfg))
    cfg.write_text(text)
    k40 = next(r for r in Parameters.from_file(str(cfg)).radionuclides if r.name == 'K40')
    assert k40.concentration == pytest.approx(310.0e-6, rel=1e-15) and k40._ppm == 310.0


@pytest.mark.unit
def test_toml_and_dict_sections_cannot_set_the_ppm_record(tmp_path):
    """The TOML file path and the dict path reject _ppm in a radionuclide section too."""
    import tomllib

    from aragog.config import Config

    text = (BUNDLED_LOOKUP.parent / 'abe_solid.toml').read_text()
    bad = text.replace('concentration = 310\n', 'concentration = 310\n_ppm = 310\n', 1)
    assert bad != text
    path = tmp_path / 'ppm.toml'
    path.write_text(bad)
    with pytest.raises(ValueError, match='K40: _ppm is not a configuration key'):
        Parameters.from_file(str(path))
    with pytest.raises(ValueError, match='K40: _ppm is not a configuration key'):
        Config.from_dict(tomllib.loads(bad), config_dir=BUNDLED_LOOKUP.parent)
    good = Config.from_dict(tomllib.loads(text), config_dir=BUNDLED_LOOKUP.parent)
    assert next(r for r in good.radionuclides if r.name == 'K40')._ppm == 310.0


@pytest.mark.unit
def test_a_rebuild_checks_a_converted_isotope_as_a_mass_fraction():
    """A rebuild of Parameters checks an isotope already converted as a mass fraction:
    its heating overflows at the earlier start time, scaled once more by 1e-6 it would not."""
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    t0 = np.log(1e300) / np.log(2) * 1e6  # exp(arg) = 1e300 at t = 0
    hot = dict(
        name='X',
        t0_years=t0,
        abundance=1.0,
        concentration=1e6,
        heat_production=1e10,
        half_life_years=1e6,
    )
    q, (r,) = _with(replace(p, solver=replace(p.solver, start_time=t0)), hot)
    assert r.concentration == 1.0  # mass fraction; heating 1e10 * 1e300 overflows at t = 0
    with pytest.raises(ValueError, match='X: heating is not finite at the start time 0'):
        replace(q, solver=replace(q.solver, start_time=0.0))


_ANY_FLOAT = st.floats(allow_nan=True, allow_infinity=True) | st.sampled_from(
    [0.0, -0.0, np.finfo(float).tiny, np.nextafter(np.finfo(float).tiny, 0.0), 5e-324]
)


@pytest.mark.unit
@settings(max_examples=300, deadline=None)
@given(st.tuples(*[_ANY_FLOAT] * 5))
def test_validate_radionuclide_accepts_exactly_the_valid_domain(values):
    """Without a start time, an isotope passes exactly when the half-life is a normal
    positive float (or inf), the other 4 inputs are finite and none of the 3 factors is
    negative; every refusal is a ValueError that names the isotope."""
    hp, ab, cn, t0, hl = values
    valid = (
        hl >= np.finfo(float).tiny and np.all(np.isfinite(values[:4])) and min(hp, ab, cn) >= 0
    )
    try:
        validate_radionuclide('X', hp, ab, cn, t0, hl, ppm=False)
    except ValueError as exc:
        assert not valid and str(exc).startswith('Radionuclide X: ')
    else:
        assert valid


@pytest.mark.unit
@pytest.mark.parametrize('field', range(5))
@pytest.mark.parametrize(
    'bad',
    [True, False, np.bool_(True), '1.0', None, 10**400, np.array('x', dtype=object), 1j],
    ids=[
        'true',
        'false',
        'np_bool',
        'str',
        'none',
        'int_above_float',
        'object_array',
        'complex',
    ],
)
def test_validate_radionuclide_refuses_a_value_that_is_not_a_number(field, bad):
    """A TOML true, a string, an integer above the float range or another value that is not
    a real float is refused with a ValueError, not read as 1 or compared."""
    values = [1.0, 1.0, 1.0, 0.0, 1.0e9]
    values[field] = bad
    with pytest.raises(Exception) as err:
        validate_radionuclide('X', *values, ppm=False)
    assert err.type is ValueError and re.search(
        'Radionuclide X: .*must be numbers', str(err.value)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    'wrap',
    [np.asarray, np.float32, lambda v: np.asarray(v, dtype=np.float64), Fraction],
    ids=['0d_array', 'float32', '0d_float64', 'fraction'],
)
def test_parameters_accept_numeric_scalars_of_any_real_type(wrap):
    """0-d NumPy arrays, NumPy scalars and fractions pass as floats do: a rebuild of Parameters
    converts the concentration once and gives the heating of the same isotope as floats."""
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    k40 = {**K40_PPM, 'concentration': 310}
    try:
        q, (r,) = _with(p, {k: v if k == 'name' else wrap(v) for k, v in k40.items()})
    except Exception as exc:
        pytest.fail(f'a numeric scalar was refused: {exc!r}')
    _, (ref,) = _with(p, k40)
    tol = 1e-6 if wrap is np.float32 else 0.0
    assert float(r.concentration) == pytest.approx(ref.concentration, rel=tol, abs=0.0)
    assert float(r.get_heating(0.0)) == pytest.approx(ref.get_heating(0.0), rel=tol, abs=0.0)
    replace(q)  # a second rebuild must not convert the concentration again
    assert float(r.concentration) == pytest.approx(ref.concentration, rel=tol, abs=0.0)


@pytest.mark.unit
def test_validate_radionuclide_accepts_jax_scalars():
    """Concrete JAX scalars pass as floats do."""
    jnp = pytest.importorskip('jax.numpy')
    values = [jnp.asarray(v) for v in (2.8761e-5, 1.1668e-4, 310.0, 4.55e9, 1248e6)]
    try:
        validate_radionuclide('K40', *values, start_time=0.0)
    except ValueError as exc:
        pytest.fail(f'a JAX scalar was refused: {exc}')


@pytest.mark.unit
def test_parameters_store_the_checked_values_as_floats():
    """Parameters keeps the settings it checked: float32 inputs whose float32 product
    overflows are stored as floats and heat as the check computed (1e24 W/kg at t0)."""
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    big = dict(
        name='X',
        t0_years=np.float32(0.0),
        abundance=np.float32(1e30),
        concentration=np.float32(1e-30),
        heat_production=np.float32(1e30),
        half_life_years=np.float32(1e9),
    )
    _, (r,) = _with(p, big)
    assert all(type(getattr(r, f)) is float for f in big if f != 'name')
    assert r.get_heating(0.0) == pytest.approx(1e24, rel=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize('wrap', [float, np.asarray, np.float64], ids=['float', '0d', 'f64'])
def test_rebuilds_keep_one_ppm_conversion(wrap):
    """A second and a third rebuild keep the concentration and the heating of the first."""
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    k40 = {**K40_PPM, 'heat_production': 0.003, 'abundance': 1.17e-4, 'concentration': 310}
    q, (r,) = _with(p, {k: v if k == 'name' else wrap(v) for k, v in k40.items()})
    first = (r.concentration, r.get_heating(0.0))
    replace(replace(q))
    assert type(r.concentration) is float and r._ppm == 310.0
    assert (r.concentration, r.get_heating(0.0)) == first
    assert r.concentration == pytest.approx(3.1e-4, rel=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize('wrap', [np.float32, np.asarray], ids=['float32', '0d'])
def test_parameters_keep_unchecked_isotopes_as_given(wrap):
    """With radionuclides off nothing is checked, so the isotope keeps its input types;
    the concentration is still converted from ppm once, also as a 0-d array."""
    p = Parameters.from_file(str(BUNDLED_LOOKUP))
    off = replace(p, energy=replace(p.energy, radionuclides=False))
    given = {k: v if k == 'name' else wrap(v) for k, v in K40_PPM.items()}
    q, (r,) = _with(off, given)
    assert type(r.heat_production) is type(wrap(1.0))
    replace(q)
    assert float(r.concentration) == pytest.approx(310e-6, rel=1e-6)


_RADIO_OK = (1e-4, 1.0, 1e-6, 0.0, 1e9)


@pytest.mark.unit
def test_jax_factory_checks_the_heating_at_the_start_time():
    """The factory refuses a live isotope whose heating overflows at its start time, as
    Parameters does; without a start time the arrays pass."""
    pytest.importorskip('jax')
    from aragog.jax.nondim import NonDimScales
    from aragog.jax.solver import check_radio_arrays
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    iso = tuple(np.array([v]) for v in (1e-4, 1.0, 1e-6, 4.55e9, 1e3))
    check_radio_arrays(*iso)
    check_radio_arrays(*iso, start_time=4.55e9)
    scales = NonDimScales(state_scale=np.full(4, 3e3), t_ref=1.0)
    with pytest.raises(ValueError, match='not finite at the start time 0.0 yr'):
        build_jax_rhs_and_jacobian(
            None,
            None,
            None,
            None,
            np.zeros(4),
            scales,
            radio_isotope_params=iso,
            start_time=0.0,
        )


@pytest.mark.unit
@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.sampled_from([(), (0,), (1,), (2,), (3,), (1, 1), (2, 1)]), min_size=5, max_size=5
    ),
    st.sampled_from([np.float64, np.float32, np.longdouble]),
)
def test_check_radio_arrays_accepts_shapes_that_broadcast_to_one_dimension(shapes, dtype):
    """5 arrays whose shapes broadcast to at most 1 dimension pass as float64 JAX arrays of
    their own shapes; any other set is refused with a ValueError that names the isotopes."""
    pytest.importorskip('jax')
    from aragog.jax.solver import check_radio_arrays

    arrays = [np.full(s, v, dtype=dtype) for s, v in zip(shapes, _RADIO_OK)]
    try:
        full = np.broadcast_shapes(*shapes)
        valid = len(full) <= 1 and (0 not in full or all(0 in s for s in shapes))
    except ValueError:
        valid = False
    try:
        out = check_radio_arrays(*arrays)
    except ValueError as exc:
        assert not valid and str(exc).startswith('Radionuclide'), exc
    else:
        assert valid and all(
            o.shape == s and o.dtype == np.float64 for o, s in zip(out, shapes)
        )
        for o, a in zip(out, arrays):
            np.testing.assert_array_equal(o, a.astype(np.float64))


@pytest.mark.unit
def test_check_radio_arrays_names_a_ragged_or_traced_list():
    """A ragged list and a list of traced values give the isotope-named ValueError."""
    jax = pytest.importorskip('jax')
    from aragog.jax.solver import check_radio_arrays

    rest = [[v] for v in _RADIO_OK[1:]]
    with pytest.raises(ValueError, match='^Radionuclide arrays'):
        check_radio_arrays([[1e-4, 1e-4], [1e-4]], *rest)
    with pytest.raises(ValueError, match='^Radionuclide arrays'):
        jax.jit(lambda x: check_radio_arrays([x], *rest)[0])(1e-4)


@pytest.mark.unit
def test_check_radio_arrays_refuses_an_empty_array_next_to_isotopes():
    """An empty array broadcast with size-1 inputs leaves no isotope to check, so a
    negative half-life next to it is refused by the shape check."""
    pytest.importorskip('jax')
    from aragog.jax.solver import check_radio_arrays

    with pytest.raises(ValueError, match='^Radionuclide arrays'):
        check_radio_arrays(np.array([]), [1.0], [1e-6], [0.0], [-5.0])


@pytest.mark.unit
def test_check_radio_arrays_skips_the_start_time_for_a_traced_input():
    """A traced input has no value, so the start-time heating is not checked with its
    stand-in; the concrete inputs still are."""
    jax = pytest.importorskip('jax')
    from aragog.jax.solver import check_radio_arrays

    def build(half_life):
        return check_radio_arrays([1e-4], [1.0], [1e-6], [4.55e9], half_life, start_time=0.0)[4]

    assert float(jax.jit(build)(jax.numpy.array([1e9]))[0]) == 1e9
    with pytest.raises(ValueError, match='#0: half_life_years'):
        jax.jit(lambda t0: check_radio_arrays([1e-4], [1.0], [1e-6], t0, [-1.0])[3])(
            jax.numpy.array([0.0])
        )


@pytest.mark.unit
def test_jax_factory_names_a_ragged_radio_list():
    """A ragged radio list reaches the factory's refusal as the radionuclide error."""
    pytest.importorskip('jax')
    from aragog.jax.nondim import NonDimScales
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    iso = ([[1e-4, 1e-4], [1e-4]], [1.0], [1e-6], [0.0], [1e9])
    scales = NonDimScales(state_scale=np.full(4, 3e3), t_ref=1.0)
    with pytest.raises(ValueError, match='^Radionuclide arrays'):
        build_jax_rhs_and_jacobian(
            None, None, None, None, np.zeros(4), scales, radio_isotope_params=iso
        )
