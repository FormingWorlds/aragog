"""Thermodynamic consistency of SPIDER P-S tables.

Module under test: ``aragog.eos.consistency``. An ideal gas with constant heat capacity in closed
form in (P, S) holds every identity exactly, so its residuals are the truncation of the central
differences alone. See docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.unit
def test_a_missing_cell_ends_the_stencils_around_it(gas):
    t = {
        **gas,
        'temperature': {**gas['temperature'], 'values': gas['temperature']['values'].copy()},
    }
    t['temperature']['values'][100, 100] = np.nan
    r = c.residuals(t, 1)['dTdS']
    assert (
        np.isnan(r[100, 99:102]).all() and np.isfinite(r[100, 98]) and np.isfinite(r[99, 100])
    )


@pytest.mark.unit
@pytest.mark.parametrize('size', [1e-6, 1e-4, 1e-2])
def test_a_planted_temperature_error_shows_in_the_richardson_residual(gas, size):
    t = {
        **gas,
        'temperature': {**gas['temperature'], 'values': gas['temperature']['values'].copy()},
    }
    t['temperature']['values'][100, 100] *= 1 + size
    rx = np.abs(4 * c.residuals(t, 1)['dTdS'] - c.residuals(t, 2)['dTdS']) / 3
    exact = np.abs(4 * c.residuals(gas, 1)['dTdS'] - c.residuals(gas, 2)['dTdS']) / 3
    assert np.nanmax(rx[100, 97:104]) > 100 * np.nanmax(exact)


@pytest.mark.unit
def test_tables_on_different_grids_raise(gas):
    t = {**gas, 'density': {**gas['density'], 'S': gas['density']['S'] + 1.0}}
    with pytest.raises(ValueError, match='density is not on the'):
        c.residuals(t)


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
    t = {**stored, quantity: {**stored[quantity], 'values': stored[quantity]['values'].copy()}}
    t[quantity]['values'][100, 100] *= 1 + size
    fail = c.check(t, 5e-9)[name]['fail']
    at = {
        o: bool(fail[(100 + o, 100) if axis == 0 else (100, 100 + o)]) for o in (-2, -1, 1, 2)
    }
    assert at == dict.fromkeys((-2, -1, 1, 2), True)


@pytest.mark.unit
def test_the_adiabat_identity_is_tested_against_its_floor(stored):
    t = {
        **stored,
        'thermal_exp': {
            **stored['thermal_exp'],
            'values': stored['thermal_exp']['values'].copy(),
        },
    }
    t['thermal_exp']['values'][50, 60] *= 1 + 1e-6
    fail = c.check(t, 5e-9)['adiabat']['fail']
    assert fail[50, 60] and int(fail.sum()) == 1


@pytest.mark.unit
def test_a_precision_below_the_floor_is_raised_to_it(stored):
    lo, at = (c.check(stored, e)['dTdS']['bound'] for e in (1e-16, 5e-9))
    np.testing.assert_array_equal(lo, at)
