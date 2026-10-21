"""Thermodynamic consistency of SPIDER P-S tables.

Module under test: ``aragog.eos.consistency``. An ideal gas with constant heat capacity in closed
form in (P, S) holds every identity exactly, so its residuals are the truncation of the central
differences alone. See docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.eos import consistency as c

R_S, CP, P0, T0, S0 = 287.05, 1004.675, 1.0e5, 300.0, 7000.0
P = np.linspace(1.0e9, 1.0e11, 199)
S = np.linspace(6000.0, 8000.0, 201)


def _ideal_gas(P, S):
    """Tables of an ideal gas in the layout of ``_load_spider_ps_table``."""
    PP, SS = np.meshgrid(P, S, indexing='ij')
    T = T0 * (PP / P0) ** (R_S / CP) * np.exp((SS - S0) / CP)
    cols = {
        'temperature': T,
        'density': PP / (R_S * T),
        'heat_capacity': np.full(T.shape, CP),
        'thermal_exp': 1.0 / T,
        'adiabat_temp_grad': R_S * T / (CP * PP),
    }
    return {q: {'P': P, 'S': S, 'values': v} for q, v in cols.items()}


@pytest.fixture(scope='module')
def gas():
    return _ideal_gas(P, S)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['dTdS', 'dTdP'])
def test_the_residual_of_an_exact_table_is_the_h2_truncation(gas, name):
    """R_2h / R_h is 4, and the Richardson value R_x = (4 R_h - R_2h) / 3 keeps an h^4 rest."""
    r1, r2, r4 = (c.residuals(gas, k)[name] for k in (1, 2, 4))
    assert np.nanmedian(r2 / r1) == pytest.approx(4.0, abs=1e-2)
    x1, x2 = (4 * r1 - r2) / 3, (4 * r2 - r4) / 3
    assert np.nanmedian(x2 / x1) == pytest.approx(16.0, abs=0.2)


@pytest.mark.unit
@pytest.mark.parametrize('name, bound', [('adiabat', 1e-14), ('maxwell', 1e-12)])
def test_an_identity_without_truncation_holds_to_round_off(gas, name, bound):
    """adiabat has no derivative, and ln rho of the gas is linear in S at fixed P."""
    assert np.nanmax(np.abs(c.residuals(gas)[name])) < bound


def _plant(t, quantity, idx, size):
    """``t`` with ``quantity`` scaled by 1 + ``size`` at ``idx`` (NaN for a NaN size)."""
    v = t[quantity]['values'].copy()
    v[idx] *= 1 + size
    return {**t, quantity: {**t[quantity], 'values': v}}


@pytest.mark.unit
def test_a_missing_cell_ends_the_stencils_around_it(gas):
    t = _plant(gas, 'temperature', (100, 100), np.nan)
    r = c.residuals(t, 1)['dTdS']
    assert (
        np.isnan(r[100, 99:102]).all() and np.isfinite(r[100, 98]) and np.isfinite(r[99, 100])
    )


@pytest.mark.unit
@pytest.mark.parametrize('size', [1e-6, 1e-4, 1e-2])
def test_a_planted_temperature_error_shows_in_the_richardson_residual(gas, size):
    t = _plant(gas, 'temperature', (100, 100), size)
    rx = np.abs(4 * c.residuals(t, 1)['dTdS'] - c.residuals(t, 2)['dTdS']) / 3
    exact = np.abs(4 * c.residuals(gas, 1)['dTdS'] - c.residuals(gas, 2)['dTdS']) / 3
    assert np.nanmax(rx[100, 97:104]) > 100 * np.nanmax(exact)


@pytest.mark.unit
def test_tables_on_different_grids_raise(gas):
    t = {**gas, 'density': {**gas['density'], 'S': gas['density']['S'] + 1.0}}
    with pytest.raises(ValueError, match='density is not on the'):
        c.residuals(t)


@pytest.mark.unit
def test_values_off_the_shape_of_the_grid_raise(gas):
    t = {**gas, 'density': {**gas['density'], 'values': gas['density']['values'][:, :-1]}}
    with pytest.raises(ValueError, match='density values do not have the shape'):
        c.residuals(t)


@pytest.mark.unit
@pytest.mark.parametrize(
    'axis, grid',
    [('P', P[::-1]), ('S', S[::-1]), ('S', np.r_[S[:100], S[99], S[101:]])],
)
def test_axes_that_do_not_increase_raise(axis, grid):
    t = _ideal_gas(grid, S) if axis == 'P' else _ideal_gas(P, grid)
    with pytest.raises(ValueError, match='strictly increasing'):
        c.check(t, 5e-9)


@pytest.mark.unit
@pytest.mark.parametrize(
    'case, skipped',
    [('short', {'dTdS', 'dTdP', 'maxwell'}), ('alpha 0', {'adiabat'})],
    ids=['too short to differentiate', 'alpha 0 everywhere'],
)
def test_an_identity_that_tests_no_cell_is_skipped_and_named(gas, case, skipped):
    """An identity without a testable cell is skipped while the others run; nothing fails."""
    t = (
        _ideal_gas(P[:8], S[:8])
        if case == 'short'
        else _plant(gas, 'thermal_exp', slice(None), -1.0)
    )
    try:
        res = c.check(t, 5e-9)
    except ValueError as err:
        pytest.fail(f'a table with testable identities raised: {err}')
    assert {n for n, v in res.items() if v['skipped']} == skipped
    assert not any(v['fail'].any() for v in res.values())


@pytest.mark.unit
def test_a_check_that_tests_no_cell_raises(gas):
    t = {q: {**v, 'values': np.full((P.size, S.size), np.nan)} for q, v in gas.items()}
    with pytest.raises(ValueError, match='no cell'):
        c.check(t, 5e-9)


@pytest.mark.unit
@pytest.mark.parametrize(
    'quantity, size, failing',
    [
        ('thermal_exp', -1.0, {'adiabat'}),
        ('density', np.inf, {'maxwell', 'adiabat'}),
        ('heat_capacity', np.inf, {'dTdS', 'adiabat'}),
        ('adiabat_temp_grad', -1.0, {'dTdP', 'maxwell', 'adiabat'}),
    ],
    ids=['alpha 0', 'rho inf', 'cp inf', 'grad 0'],
)
def test_an_isolated_zero_or_infinite_factor_fails(stored, quantity, size, failing):
    """A stored 0 or inf among numbers is a table fault: its identities fail at the cell."""
    res = c.check(_plant(stored, quantity, (100, 100), size), 5e-9)
    assert {n for n, v in res.items() if v['fail'][100, 100]} == failing
    assert not any(v['skipped'] for v in res.values())
    assert all(v['fail'].sum() <= 9 for v in res.values())


@pytest.mark.unit
def test_a_missing_value_is_not_a_failure(stored):
    """NaN marks a missing value: the identities with T at the cell leave it untested, not
    failing, and maxwell, which has no T at the cell, still tests it."""
    res = c.check(_plant(stored, 'temperature', (100, 100), np.nan), 5e-9)
    assert not any(v['fail'].any() for v in res.values())
    assert {n for n, v in res.items() if v['tested'][100, 100]} == {'maxwell'}


@pytest.mark.unit
def test_a_grid_shorter_than_the_stencil_gives_no_derivative():
    assert np.isnan(c.derivative(np.ones((3, 4)), np.arange(3.0), 0, 2)).all()


def _stored(t, digits=9):
    """The tables rounded to ``digits`` significant digits, as a file stores them."""
    fmt = f'{{:.{digits - 1}e}}'
    return {
        q: {**v, 'values': np.vectorize(lambda x: float(fmt.format(x)))(v['values'])}
        for q, v in t.items()
    }


@pytest.fixture(scope='module')
def stored(gas):
    return _stored(gas)


@pytest.mark.unit
@pytest.mark.parametrize('digits', [9, None])
def test_an_exact_table_has_no_failing_cell(gas, stored, digits):
    res = c.check(stored if digits else gas, 5e-9)
    assert {n: int(v['fail'].sum()) for n, v in res.items()} == dict.fromkeys(c.IDENTITIES, 0)
    assert all(v['tested'].sum() > 30_000 for v in res.values())


@pytest.mark.unit
@pytest.mark.parametrize(
    'quantity, name, axis, size',
    [
        ('temperature', 'dTdS', 1, 1e-4),
        ('temperature', 'dTdS', 1, 1e-6),
        ('density', 'maxwell', 1, 1e-4),
        ('density', 'maxwell', 1, 1e-6),
        ('temperature', 'dTdP', 0, 1e-4),
        ('temperature', 'dTdP', 0, 1e-6),
    ],
)
def test_a_planted_value_fails_one_and_two_steps_away(stored, quantity, name, axis, size):
    t = _plant(stored, quantity, (100, 100), size)
    fail = c.check(t, 5e-9)[name]['fail']
    at = {
        o: bool(fail[(100 + o, 100) if axis == 0 else (100, 100 + o)]) for o in (-2, -1, 1, 2)
    }
    assert at == dict.fromkeys((-2, -1, 1, 2), True)


@pytest.mark.unit
def test_the_adiabat_identity_is_tested_against_its_floor(stored):
    t = _plant(stored, 'thermal_exp', (50, 60), 1e-6)
    fail = c.check(t, 5e-9)['adiabat']['fail']
    assert fail[50, 60] and int(fail.sum()) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    'quantity, expected',
    [
        (
            'temperature',
            {
                'dTdS': [(0, o) for o in (-2, -1, 0, 1, 2)],
                'dTdP': [(o, 0) for o in (-2, -1, 1, 2)],
                'adiabat': [(0, 0)],
            },
        ),
        ('density', {'maxwell': [(0, o) for o in (-2, -1, 0, 1, 2)], 'adiabat': [(0, 0)]}),
        ('heat_capacity', {'adiabat': [(0, 0)]}),
        ('thermal_exp', {'adiabat': [(0, 0)]}),
        ('adiabat_temp_grad', {'maxwell': [(0, 0)], 'adiabat': [(0, 0)]}),
    ],
)
def test_each_planted_column_fails_exactly_its_identities(stored, quantity, expected):
    res = c.check(_plant(stored, quantity, (100, 100), 1e-6), 5e-9)
    got = {
        n: sorted((int(i) - 100, int(j) - 100) for i, j in np.argwhere(v['fail']))
        for n, v in res.items()
        if v['fail'].any()
    }
    assert got == {n: sorted(x) for n, x in expected.items()}


@pytest.mark.unit
def test_the_bound_is_the_floor_plus_twice_the_h4_term(stored):
    r = {k: c.residuals(stored, k)['dTdS'] for k in (1, 2, 4)}
    f1, f2 = (c.floors(stored, 5e-9, k)['dTdS'] for k in (1, 2))
    rx = (4 * r[1] - r[2]) / 3
    est = np.abs((4 * r[2] - r[4]) / 3 - rx) / 15
    bound = c.check(stored, 0.0)['dTdS']['bound']
    np.testing.assert_allclose(bound, (4 * f1 + f2) / 3 + 2 * est, rtol=1e-12)
    assert c.EPS_FLOOR == 5e-9


@pytest.mark.unit
def test_a_missing_cell_leaves_four_untested_cells_on_each_side(gas):
    tested = c.check(_plant(gas, 'temperature', (100, 100), np.nan), 5e-9)['dTdS']['tested']
    assert not tested[100, 96:105].any() and tested[100, [95, 105]].all()


@pytest.mark.unit
def test_a_precision_below_the_floor_is_raised_to_it(stored):
    lo, at = (c.check(stored, e)['dTdS']['bound'] for e in (1e-16, 5e-9))
    np.testing.assert_array_equal(lo, at)


@pytest.mark.unit
def test_a_coarse_precision_hides_a_fault_the_9_digit_floor_finds(stored):
    t = _plant(stored, 'temperature', (100, 100), 1e-6)
    assert c.check(t, 0.0)['dTdS']['fail'][100, 101]
    assert not c.check(t, 1e-4)['dTdS']['fail'][100, 101]


# SPIDER P-S files as zalmoxis.eos_export._write_spider_2d writes them: '.18e', S slowest.
_SCALES = {
    'density': 1000.0,
    'temperature': 1000.0,
    'heat_capacity': 1000.0,
    'thermal_exp': 1e-5,
    'adiabat_temp_grad': 1e-9,
}
P_SCALE, S_SCALE = 1e9, 4824266.84604467


def _write_spider(path, P, S, values, q_scale):
    with open(path, 'w') as f:
        f.write(f'# 5 {len(P)} {len(S)}\n')
        f.write('# Pressure [nondim], Entropy [nondim], Quantity [nondim]\n')
        f.write('# column * scaling factor = SI units\n')
        f.write('# scaling factors (constant) for each column given on line below\n')
        f.write(f'# {P_SCALE} {S_SCALE} {q_scale}\n')
        for j in range(len(S)):
            for i in range(len(P)):
                f.write(
                    f'{P[i] / P_SCALE:.18e} {S[j] / S_SCALE:.18e} {values[i, j] / q_scale:.18e}\n'
                )


def _murnaghan_ps(
    P, S, rho0=4100.0, K0=250e9, Kp0=4.0, a0=2e-5, cp0=1100.0, cp1=0.05, T0=1500.0, s0=2200.0
):
    """Murnaghan solid of tests in Zalmoxis (constant alpha) on a (P, S) grid; T by Newton on s."""
    PP, SS = np.meshgrid(P, S, indexing='ij')
    phi = K0 / (Kp0 - 1.0) * ((1.0 + Kp0 * PP / K0) ** (1.0 - 1.0 / Kp0) - 1.0)

    def s_cp(T):
        v0 = np.exp(a0 * (T - T0)) / rho0
        s = s0 + cp0 * np.log(T / T0) + cp1 * (T - T0) - a0 * v0 * phi
        return s, cp0 + cp1 * T - T * a0**2 * v0 * phi

    T = np.full(PP.shape, T0)
    for _ in range(60):
        s, cp = s_cp(T)
        T = T - (s - SS) * T / cp
    s, cp = s_cp(T)
    v = np.exp(a0 * (T - T0)) / rho0 * (1.0 + Kp0 * PP / K0) ** (-1.0 / Kp0)
    assert np.max(np.abs(s - SS)) < 1e-9 * np.max(np.abs(SS))
    cols = {
        'temperature': T,
        'density': 1.0 / v,
        'heat_capacity': cp,
        'thermal_exp': np.full(T.shape, a0),
        'adiabat_temp_grad': a0 * T * v / cp,
    }
    return cols


@pytest.fixture(scope='module')
def spider_files(tmp_path_factory):
    """The exact gas and Murnaghan solid written as SPIDER files and read back by aragog."""
    from aragog.eos.entropy import _load_spider_ps_table

    out = {}
    for name in ('gas', 'murnaghan'):
        d = tmp_path_factory.mktemp(name)
        if name == 'gas':
            Pg, Sg = np.linspace(1e9, 1e11, 81), np.linspace(6000.0, 8000.0, 81)
            cols = {q: v['values'] for q, v in _ideal_gas(Pg, Sg).items()}
        else:
            Pg, Sg = np.linspace(1e9, 1e11, 81), np.linspace(2000.0, 2400.0, 81)
            cols = _murnaghan_ps(Pg, Sg)
        for q, vals in cols.items():
            _write_spider(d / f'{q}_solid.dat', Pg, Sg, vals, _SCALES[q])
        out[name] = {q: _load_spider_ps_table(d / f'{q}_solid.dat') for q in c.QUANTITIES}
    return out


@pytest.mark.unit
@pytest.mark.parametrize('name', ['gas', 'murnaghan'])
def test_an_exact_table_written_as_spider_files_has_no_failing_cell(spider_files, name):
    res = c.check(spider_files[name], np.finfo(float).eps)
    assert {n: int(v['fail'].sum()) for n, v in res.items()} == dict.fromkeys(c.IDENTITIES, 0)
    assert all(v['tested'].sum() > 5_000 for v in res.values())


@pytest.mark.unit
@pytest.mark.parametrize('name', ['gas', 'murnaghan'])
@pytest.mark.parametrize(
    'quantity, identity, axis',
    [('temperature', 'dTdS', 1), ('density', 'maxwell', 1), ('temperature', 'dTdP', 0)],
)
@pytest.mark.parametrize('size', [1e-4, 1e-6])
def test_a_planted_value_in_spider_files_fails_one_and_two_steps_away(
    spider_files, name, quantity, identity, axis, size
):
    t = _plant(spider_files[name], quantity, (40, 40), size)
    fail = c.check(t, np.finfo(float).eps)[identity]['fail']
    cells = [(40 + o, 40) if axis == 0 else (40, 40 + o) for o in (-2, -1, 1, 2)]
    assert [bool(fail[x]) for x in cells] == [True] * 4


_BAD_EPS = st.one_of(
    st.sampled_from([np.nan, np.inf, -np.inf, True, False, None, '5e-9', 1.0]),
    st.floats(max_value=0.0, exclude_max=True),
    st.floats(min_value=1.0),
    st.integers(min_value=1),
    st.integers(max_value=-1),
    st.sampled_from([10**400, -(10**400), np.int64(2), np.False_, b'1e-3']),
    st.sampled_from([bytearray(b'1e-3'), memoryview(b'1e-3'), np.array(1e-3), Decimal('1e-3')]),
)


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(eps=_BAD_EPS)
def test_a_precision_that_is_not_a_finite_fraction_is_rejected(stored, eps):
    """A NaN or infinite eps would make every bound NaN or infinite and hide every cell."""
    with pytest.raises(ValueError, match='eps'):
        c.check(stored, eps)


@pytest.mark.unit
@settings(max_examples=10, deadline=None)
@given(eps=st.one_of(st.just(0), st.floats(min_value=0.0, max_value=0.99)))
def test_an_exact_table_passes_at_every_valid_precision(stored, eps):
    res = c.check(stored, eps)
    assert all(v['tested'].sum() > 0 and not v['fail'].any() for v in res.values())


# Per identity: (quantity, axis) of its derivative ends, and the quantities of its algebraic
# factor at the cell.
_FLOOR_PARTS = {
    'dTdS': ([('temperature', 1)], ['temperature', 'heat_capacity']),
    'dTdP': ([('temperature', 0)], ['adiabat_temp_grad']),
    'maxwell': ([('density', 1)], ['density', 'adiabat_temp_grad']),
    'adiabat': (
        [],
        ['adiabat_temp_grad', 'density', 'heat_capacity', 'thermal_exp', 'temperature'],
    ),
}


@pytest.mark.unit
@pytest.mark.parametrize('name', list(_FLOOR_PARTS))
@pytest.mark.parametrize('cell', [(60, 60), (150, 120)])
@pytest.mark.parametrize('k', [1, 2])
def test_the_floor_is_the_first_order_round_off_of_each_stored_value(stored, name, cell, k):
    """Moving each stored value that enters the identity by eps alone, the sum of the changes
    of R at the cell is its floor (first order): no term of the floor is lost or doubled."""
    eps, (i, j) = 1e-9, cell
    base = c.residuals(stored, k)[name][i, j]
    ends, factors = _FLOOR_PARTS[name]
    moves = [(q, (i + o, j) if ax == 0 else (i, j + o)) for q, ax in ends for o in (-k, k)]
    moves += [(q, (i, j)) for q in factors]
    total = sum(
        abs(c.residuals(_plant(stored, q, at, eps), k)[name][i, j] - base) for q, at in moves
    )
    assert total == pytest.approx(c.floors(stored, eps, k)[name][i, j], rel=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize(
    'row, found',
    [(8, 0), (15, 2), (25, 4), (100, 4)],
    ids=['P/dP 10', 'P/dP 17', 'P/dP 27', 'P/dP 102'],
)
def test_dtdp_finds_a_1e_6_fault_only_where_p_is_large_against_the_step(stored, row, found):
    """On a linear P grid the h^4 term of dTdP grows as P/dP falls: a planted 1e-6 temperature
    is found at all 4 cells 1 and 2 steps away from P/dP of about 27 up, at none below 10."""
    res = c.check(_plant(stored, 'temperature', (row, 100), 1e-6), 5e-9)['dTdP']
    assert sum(bool(res['fail'][row + o, 100]) for o in (-2, -1, 1, 2)) == found
    assert res['fail'].sum() <= 4


@pytest.mark.unit
def test_floors_and_residuals_check_the_axes(gas):
    """floors takes the tables residuals takes: a reversed axis raises, a list axis works."""
    rev = _ideal_gas(P[::-1], S)
    with pytest.raises(ValueError, match='strictly increasing'):
        c.floors(rev, 5e-9)
    listed = {q: {**v, 'P': list(v['P']), 'S': list(v['S'])} for q, v in gas.items()}
    for name, f in c.floors(listed, 5e-9).items():
        np.testing.assert_array_equal(f, c.floors(gas, 5e-9)[name])


@pytest.mark.unit
@pytest.mark.parametrize('axis', ['P', 'S'])
@pytest.mark.parametrize('grid', ['nan', 'one point', '2-D'])
def test_an_axis_with_a_nan_one_point_or_two_dimensions_raises(axis, grid):
    """An axis needs at least 2 finite, increasing coordinates in one dimension."""
    x = (P if axis == 'P' else S).copy()
    x = x[:1] if grid == 'one point' else np.r_[x[:50], np.nan, x[51:]] if grid == 'nan' else x
    t = _ideal_gas(x, S) if axis == 'P' else _ideal_gas(P, x)
    if grid == '2-D':
        t = {q: {**v, axis: np.asarray(v[axis]).reshape(1, -1)} for q, v in t.items()}
    with pytest.raises(ValueError, match='strictly increasing'):
        c.check(t, 5e-9)


@pytest.mark.unit
def test_the_precision_floor_is_9_digits():
    """EPS_FLOOR is 0.5 x 10^(1 - 9), the precision of 9 significant digits."""
    assert c.EPS_FLOOR == pytest.approx(0.5 * 10 ** (1 - 9), rel=1e-12)
    assert c._eps(0.0) == c.EPS_FLOOR


@pytest.mark.unit
def test_an_exact_table_on_a_geometric_grid_passes():
    """A grid uniform in log P (not in P) still gives no failing cell on an exact table."""
    t = _stored(_ideal_gas(np.geomspace(1.0e9, 1.0e11, 199), S))
    res = c.check(t, 5e-9)
    assert all(v['tested'].sum() > 1000 and not v['fail'].any() for v in res.values())


@pytest.mark.unit
def test_a_missing_temperature_leaves_dtdp_untested_four_cells_each_way(gas):
    """A missing temperature blocks dTdP at the 4 cells on each side along P."""
    res = c.check(_plant(gas, 'temperature', (100, 100), np.nan), 5e-9)['dTdP']
    assert not res['tested'][96:105, 100].any()
    assert res['tested'][95, 100] and res['tested'][105, 100]


@pytest.mark.unit
def test_a_precision_that_is_not_a_number_names_the_rule():
    """The message says eps must be a real number."""
    with pytest.raises(ValueError, match='eps must be a real number in'):
        c._eps('1e-9')
