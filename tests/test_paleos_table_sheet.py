"""The table sheet: each row can report pass and outside on a small synthetic table set."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from aragog.eos.entropy import EntropyEOS

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location(
    'paleos_table_sheet', Path(__file__).parent.parent / 'tools' / 'paleos_table_sheet.py'
)
sheet_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sheet_tool)

GPA = 1.0e9
_GOOD = {
    'p_max': 150.0 * GPA,
    'rho_melt': 3000.0,
    'rho_solid': 3300.0,
    'grad_melt': 2.0e-8,
    'grad_solid': 1.0e-8,
    'ds0': 300.0,
    'ds_slope': -1.0,  # J/kg/K per GPa
    'sol_slope': 24.0,  # J/kg/K per GPa
    'gap': 750.0,
    'pad_solid': 0,
    'pad_solid_last': 0,
}


def _tables(folder: Path, **change) -> EntropyEOS:
    """Write a linear table set; ``change`` overrides ``_GOOD``. Entropy spans 100 to 8000 J/kg/K."""
    a = {**_GOOD, **change}
    folder.mkdir(exist_ok=True)
    p = np.linspace(1.0e5, a['p_max'], 41)
    s = np.linspace(100.0, 8000.0, 81)
    s_sol = 2000.0 + a['sol_slope'] * p / GPA
    s_liq = s_sol + a['ds0'] + a['ds_slope'] * p / GPA

    def t_solid(pp, ss):
        return 1000.0 + 0.5 * (ss - 2000.0) + 1.0e-8 * pp

    def write(name, fun):
        pp, ss = np.meshgrid(p, s)
        rows = np.column_stack([pp.ravel(), ss.ravel(), fun(pp, ss).ravel()])
        head = f'5 {p.size} {s.size}\nPressure, Entropy, Quantity\nx\ny\n1.0 1.0 1.0'
        np.savetxt(folder / name, rows, header=head, fmt='%.12e')

    def solid_t(pp, ss):
        t = t_solid(pp, ss)
        if a['pad_solid']:
            t[: a['pad_solid'], :] = t[a['pad_solid'], :]
        if a['pad_solid_last']:
            t[: a['pad_solid_last'], -1] = t[a['pad_solid_last'], -1]
        return t

    write('temperature_solid.dat', solid_t)
    write('temperature_melt.dat', lambda pp, ss: t_solid(pp, ss) + a['gap'] - 150.0)
    for phase, rho, grad in (
        ('solid', a['rho_solid'], a['grad_solid']),
        ('melt', a['rho_melt'], a['grad_melt']),
    ):
        write(f'density_{phase}.dat', lambda pp, ss, v=rho: np.full_like(pp, v))
        write(f'heat_capacity_{phase}.dat', lambda pp, ss: np.full_like(pp, 1000.0))
        write(f'adiabat_temp_grad_{phase}.dat', lambda pp, ss, v=grad: np.full_like(pp, v))
    for name, curve in (('solidus_P-S.dat', s_sol), ('liquidus_P-S.dat', s_liq)):
        head = f'5 {p.size}\nPressure, Entropy\nx\ny\n1.0 1.0'
        np.savetxt(folder / name, np.column_stack([p, curve]), header=head, fmt='%.12e')
    return EntropyEOS(folder)


def _rows(eos, start):
    found = [r for r in sheet_tool.sheet(eos) if r.check.startswith(start)]
    assert found, f'no row starts with {start!r}'
    return found


def _status(eos, start):
    """Status of each row whose check starts with ``start``; at least one row must match."""
    return [r.status for r in _rows(eos, start)]


@pytest.fixture(scope='module')
def good(tmp_path_factory):
    return _tables(tmp_path_factory.mktemp('good'))


@pytest.mark.parametrize(
    ('start', 'bad'),
    [
        ('rho_melt - rho_solid', {'rho_melt': 3500.0}),
        ('alpha_melt', {'grad_melt': 0.1e-8}),
        ('alpha_melt', {'grad_solid': 0.0}),
        ('alpha_melt', {'grad_solid': -1.0e-8}),
        ('Delta S at the lowest', {'ds0': 1000.0, 'ds_slope': 0.0}),
        ('Delta S at the lowest', {'ds0': 100.0, 'ds_slope': 0.0}),
        ('Delta S at 100 GPa', {'ds_slope': 1.0}),
        ('T_liquidus', {'gap': 1500.0}),
        ('T_liquidus', {'gap': 150.0}),
        ('pressure where the isentrope', {'sol_slope': 12.0}),
        ('pressure where the isentrope', {'sol_slope': 60.0}),
        ('solid temperature rows copied above S_min, lowest', {'pad_solid': 4}),
        ('solid temperature rows copied above S_min, highest', {'pad_solid_last': 4}),
    ],
)
def test_a_row_passes_on_the_good_table_and_reports_outside_on_a_bad_one(
    good, tmp_path, start, bad
):
    assert set(_status(good, start)) == {'pass'}
    assert 'outside' in _status(_tables(tmp_path, **bad), start)


def test_the_values_are_the_measured_ones(good):
    """Printed values, not only the status: the isentrope reaches phi 0 at 0.4 x 300 / 24 GPa."""
    assert _rows(good, 'rho_melt - rho_solid at 1 GPa')[0].value == '-300.0'
    assert float(_rows(good, 'pressure where the isentrope')[0].value) == pytest.approx(
        5.05, abs=0.1
    )
    assert _rows(good, 'T_liquidus - T_solidus at 50 GPa')[0].value == f'{750 - 0.5 * 50:.0f}'
    assert _rows(good, 'alpha_melt and alpha_solid at 1 GPa')[0].value.count(',') == 1
    assert float(_rows(good, 'Delta S at 100 GPa')[0].value) == pytest.approx(
        200.0 / 300.0, abs=1.0e-2
    )


def test_the_mixed_gradient_row_reports_both_states(good, tmp_path):
    """The mixture is steeper than both phases when the liquidus falls fast with pressure; between
    the phases it is outside."""
    steep = {'sol_slope': -40.0, 'ds_slope': -1.0, 'gap': 500.0}
    assert set(_status(_tables(tmp_path / 'a', **steep), '(dT/dP)_S mixed')) == {'pass'}
    assert 'outside' in _status(good, '(dT/dP)_S mixed')
    between = _tables(tmp_path / 'b', **steep, grad_melt=5.0e-6, grad_solid=1.0e-10)
    assert 'outside' in _status(between, '(dT/dP)_S mixed')


def test_the_canary_row_fails_on_a_planted_sign_flip(tmp_path):
    """A density table with the sign of the melt contrast flipped fails the first row, and an
    exit status of 0 still reports it."""
    flipped = _tables(tmp_path, rho_melt=3300.0, rho_solid=3000.0)
    assert set(_status(flipped, 'rho_melt - rho_solid at 1 GPa')) == {'outside'}
    assert sheet_tool.main([str(tmp_path)]) == 0


def test_a_pressure_the_tables_do_not_reach_reads_na(tmp_path):
    """Edges come from the loaded tables: no row reads past them, also inside a difference step."""
    eos = _tables(tmp_path, p_max=15.0 * GPA)
    for start in (
        'rho_melt - rho_solid at 20 GPa',
        'alpha_melt and alpha_solid at 20 GPa',
        '(dT/dP)_S mixed vs pure phases at 20 GPa',
        'Delta S at 100 GPa',
        'T_liquidus - T_solidus at 50 GPa',
        'melt temperature rows copied above the liquidus at 20 GPa',
    ):
        assert _status(eos, start) == ['n/a']
    near = _tables(tmp_path / 'near', p_max=11.0 * GPA)
    assert _status(near, '(dT/dP)_S mixed vs pure phases at 10 GPa') == ['n/a']
    assert _status(near, 'rho_melt - rho_solid at 10 GPa') != ['n/a']


def test_a_degenerate_table_gives_rows_and_no_crash(tmp_path):
    """A liquidus at or below the solidus, or a solidus that never reaches the isentrope, reads n/a."""
    for name, change in (
        ('zero', {'ds0': 0.0, 'ds_slope': 0.0}),
        ('negative', {'ds0': -100.0, 'ds_slope': 0.0}),
        ('flat', {'sol_slope': 0.1}),
    ):
        eos = _tables(tmp_path / name, **change)
        assert _status(eos, 'pressure where the isentrope') == ['n/a']
    assert _status(_tables(tmp_path / 'z2', ds0=0.0, ds_slope=0.0), 'Delta S at 100 GPa') == [
        'n/a'
    ]


def test_the_melt_rows_copied_above_the_liquidus_are_measured(tmp_path):
    """A melt column flat above the liquidus reports the flat span, also flat to the table top, and
    a plateau that ends at the first node above the liquidus counts."""
    eos = _tables(tmp_path)
    table = eos._tables['temperature_melt']
    assert sheet_tool._melt_copied_above_liquidus(eos, GPA) == 0.0
    assert set(_status(eos, 'melt temperature rows copied')) == {'pass'}
    i = int(np.argmin(np.abs(table['P'] - GPA)))
    s_liq = float(eos.liquidus_entropy(float(table['P'][i])))
    j = int(np.searchsorted(table['S'], s_liq))
    original = table['values'][i].copy()
    table['values'][i, j : j + 6] = table['values'][i, j - 1]
    assert sheet_tool._melt_copied_above_liquidus(eos, GPA) == pytest.approx(
        float(table['S'][j + 5]) - s_liq
    )
    assert _status(eos, 'melt temperature rows copied above the liquidus at 1 GPa') == [
        'outside'
    ]
    table['values'][i] = original
    table['values'][i, j:] = table['values'][i, j - 1]
    assert sheet_tool._melt_copied_above_liquidus(eos, GPA) == pytest.approx(
        float(table['S'][-1]) - s_liq
    )
    table['values'][i] = original
    table['values'][i, j] = table['values'][i, j - 1]
    assert sheet_tool._melt_copied_above_liquidus(eos, GPA) == pytest.approx(
        float(table['S'][j]) - s_liq
    )
