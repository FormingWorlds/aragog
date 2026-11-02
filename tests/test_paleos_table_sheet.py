"""The table sheet: each row can report pass and outside on a small synthetic table set."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from aragog.eos.entropy import EntropyEOS

from .conftest import returns

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
    'sol_curve': 0.0,  # J/kg/K per GPa^2
    'p_max_density': None,
}


def _tables(folder: Path, **change) -> EntropyEOS:
    """Write a linear table set; ``change`` overrides ``_GOOD``. Entropy spans 100 to 8000 J/kg/K."""
    a = {**_GOOD, **change}
    folder.mkdir(exist_ok=True)
    p = np.linspace(1.0e5, a['p_max'], 41)
    s = np.linspace(100.0, 8000.0, 81)
    s_sol = 2000.0 + a['sol_slope'] * p / GPA + a['sol_curve'] * (p / GPA) ** 2
    s_liq = s_sol + a['ds0'] + a['ds_slope'] * p / GPA

    def t_solid(pp, ss):
        return 1000.0 + 0.5 * (ss - 2000.0) + 1.0e-8 * pp

    def write(name, fun, grid=p):
        pp, ss = np.meshgrid(grid, s)
        rows = np.column_stack([pp.ravel(), ss.ravel(), fun(pp, ss).ravel()])
        head = f'5 {grid.size} {s.size}\nPressure, Entropy, Quantity\nx\ny\n1.0 1.0 1.0'
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
        rho_grid = np.linspace(1.0e5, a['p_max_density'], p.size) if a['p_max_density'] else p
        write(f'density_{phase}.dat', lambda pp, ss, v=rho: np.full_like(pp, v), rho_grid)
        write(f'heat_capacity_{phase}.dat', lambda pp, ss: np.full_like(pp, 1000.0))
        write(f'adiabat_temp_grad_{phase}.dat', lambda pp, ss, v=grad: np.full_like(pp, v))
    for name, curve in (('solidus_P-S.dat', s_sol), ('liquidus_P-S.dat', s_liq)):
        head = f'5 {p.size}\nPressure, Entropy\nx\ny\n1.0 1.0'
        np.savetxt(folder / name, np.column_stack([p, curve]), header=head, fmt='%.12e')
    return EntropyEOS(folder)


def _rows(eos, start):
    found = [r for r in returns(sheet_tool.sheet, eos) if r.check.startswith(start)]
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
        ('rho_melt - rho_solid at 1 GPa', {'rho_melt': 3500.0}),
        ('rho_melt - rho_solid at 5 GPa', {'rho_melt': 3500.0}),
        ('alpha_melt and alpha_solid, ordering at 1 GPa', {'grad_melt': 0.1e-8}),
        ('alpha_melt and alpha_solid, ordering at 5 GPa', {'grad_solid': 0.0}),
        ('alpha_melt and alpha_solid, ordering at 5 GPa', {'grad_solid': -1.0e-8}),
        ('Delta S at the lowest', {'ds0': 1000.0, 'ds_slope': 0.0}),
        ('Delta S at the lowest', {'ds0': 100.0, 'ds_slope': 0.0}),
        ('Delta S at 100 GPa', {'ds_slope': 1.0}),
        ('Delta S at 100 GPa', {'ds_slope': -5.0}),
        ('pressure where the isentrope', {'sol_slope': 12.0}),
        ('pressure where the isentrope', {'sol_slope': 60.0}),
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
        5.0, abs=0.01
    )
    assert _rows(good, 'T_liquidus - T_solidus at 50 GPa')[0].value == f'{750 - 0.5 * 50:.0f}'
    # rho cp (dT/dP)_S / T: solidus 2024 and liquidus 2323 J/kg/K at 1 GPa, T of the tables linear
    t_solid = 1000.0 + 0.5 * 24.0 + 10.0
    t_melt = 1000.0 + 0.5 * 323.0 + 10.0 + 750.0 - 150.0
    value = _rows(good, 'alpha_melt and alpha_solid, ordering at 1 GPa')[0].value
    melt, solid = value.split(' (')[0].split(', ')
    assert float(melt) == pytest.approx(3000.0 * 1000.0 * 2.0e-8 / t_melt, rel=1.0e-2)
    assert float(solid) == pytest.approx(3300.0 * 1000.0 * 1.0e-8 / t_solid, rel=1.0e-2)
    assert float(_rows(good, 'Delta S at 100 GPa')[0].value) == pytest.approx(
        200.0 / 300.0, abs=1.0e-2
    )


def test_the_mixed_gradient_row_reports_both_states(good, tmp_path):
    """The mixture is steeper than both phases when the liquidus falls fast with pressure; between
    the phases it is outside."""
    steep = {'sol_slope': -40.0, 'ds_slope': -1.0, 'gap': 500.0}
    status = _status(_tables(tmp_path / 'a', **steep), '(dT/dP)_S mixed')
    assert 'pass' in status and 'outside' not in status
    assert 'outside' in _status(good, '(dT/dP)_S mixed')
    between = _tables(tmp_path / 'b', **steep, grad_melt=5.0e-6, grad_solid=1.0e-10)
    assert 'outside' in _status(between, '(dT/dP)_S mixed')


def test_the_mixed_gradient_row_prints_both_steps_and_leaves_a_step_out_of_the_mixture(
    tmp_path,
):
    """A curved solidus gives a different gradient at each step, both printed; a wide step that
    leaves the mixed region reads n/a."""
    curved = _tables(tmp_path / 'curved', sol_slope=-40.0, sol_curve=0.5, gap=500.0)
    value = _rows(curved, '(dT/dP)_S mixed vs pure phases at 20 GPa')[0].value
    wide, narrow = re.findall(r'(-?[0-9.]+) at', value)
    assert float(wide) != float(narrow)
    steep = _tables(tmp_path / 'steep', sol_slope=200.0, ds0=300.0, ds_slope=0.0)
    assert _status(steep, '(dT/dP)_S mixed vs pure phases at 20 GPa') == ['n/a']


def test_a_row_with_no_band_of_the_same_quantity_reports_a_value(good):
    """No pass or outside where the source gives another quantity, or for a table property."""
    for start in (
        'rho_melt - rho_solid at 10 GPa',
        'rho_melt - rho_solid at 20 GPa',
        'T_liquidus - T_solidus at 50 GPa',
        'T_liquidus - T_solidus at 100 GPa',
        'solid temperature rows copied above S_min, lowest',
        'solid temperature rows copied above S_min, highest',
        'pressure columns with solid temperature rows copied',
        'melt temperature rows copied above the liquidus at 1 GPa',
    ):
        assert _status(good, start) == ['report'], start


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
        'alpha_melt and alpha_solid, ordering at 20 GPa',
        '(dT/dP)_S mixed vs pure phases at 20 GPa',
        'Delta S at 100 GPa',
        'T_liquidus - T_solidus at 50 GPa',
        'melt temperature rows copied above the liquidus at 20 GPa',
    ):
        assert _status(eos, start) == ['n/a']
    near = _tables(tmp_path / 'near', p_max=11.0 * GPA)
    assert _status(near, '(dT/dP)_S mixed vs pure phases at 10 GPa') == ['n/a']
    assert _status(near, 'rho_melt - rho_solid at 10 GPa') != ['n/a']


def test_the_padded_row_counts_give_the_columns_and_their_pressure_range(good, tmp_path):
    """Each pressure column with copied rows counts: none on the good table; a padded lowest column
    gives 1 of the columns, its pressure, and the largest span."""
    row = _rows(good, 'pressure columns with solid temperature rows copied')[0]
    assert (
        row.status == 'report'
        and row.value == f'0 of {len(good._tables["temperature_solid"]["P"])}'
    )
    eos = _tables(tmp_path, pad_solid=4)
    table = eos._tables['temperature_solid']
    table['values'][1:] = table['values'][1:] + np.linspace(0.0, 1.0, table['S'].size)
    row = _rows(eos, 'pressure columns with solid temperature rows copied')[0]
    assert row.status == 'report' and row.value.startswith('1 of 41')
    assert f'{float(table["P"][0]) / GPA:.3g}-{float(table["P"][0]) / GPA:.3g} GPa' in row.value
    assert f'largest span {float(table["S"][4] - table["S"][0]):.1f}' in row.value


def test_the_sheet_names_the_paleos_commit_of_each_set(tmp_path, capsys):
    """A commit comes from a table header or from the command line; a set with neither says so."""
    for name in ('a', 'b', 'c'):
        _tables(tmp_path / name)
    (tmp_path / 'b' / 'source_table.dat').write_text('# Title\n# PALEOS commit: abc1234\n1.0\n')
    args = [str(tmp_path / n) for n in 'abc'] + ['--paleos-commit', f'{tmp_path / "c"}=def5678']
    assert sheet_tool.main(args) == 0
    out = capsys.readouterr().out
    assert 'PALEOS commit: not recorded in the tables' in out
    assert 'PALEOS commit: abc1234 (table header)' in out
    assert 'PALEOS commit: def5678 (given)' in out


def test_a_table_with_a_narrower_range_reads_na(tmp_path):
    """A lookup clamps to the range of its own table, so every table must hold the pressure."""
    eos = _tables(tmp_path, p_max_density=30.0 * GPA)
    assert _status(eos, 'rho_melt - rho_solid at 20 GPa') == ['report']
    assert _status(eos, 'T_liquidus - T_solidus at 50 GPa') == ['n/a']
    assert _status(eos, 'Delta S at 100 GPa') == ['n/a']


def test_a_non_finite_first_row_is_not_a_span_of_zero(tmp_path):
    """A NaN at the table minimum gives NaN, not a span of 0."""
    eos = _tables(tmp_path)
    table = eos._tables['temperature_solid']
    table['values'][0, 0] = np.nan
    assert np.isnan(sheet_tool._padded_span(table, 0))
    assert _rows(eos, 'solid temperature rows copied above S_min, lowest')[0].value == 'nan'


def test_the_commit_flag_refuses_a_bad_item_and_an_unused_folder(tmp_path, capsys):
    """An item with no = and a folder not given stop with a message, not a traceback."""
    _tables(tmp_path)
    for item in ('no-equal-sign', f'{tmp_path}=', f'{tmp_path / "other"}=abc'):
        with pytest.raises(SystemExit) as stop:
            sheet_tool.main([str(tmp_path), '--paleos-commit', item])
        assert stop.value.code == 2
        assert '--paleos-commit' in capsys.readouterr().err
    assert sheet_tool.main([f'{tmp_path}/', '--paleos-commit', f'{tmp_path}=abc']) == 0
    assert 'PALEOS commit: abc (given)' in capsys.readouterr().out


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
    assert (
        _rows(eos, 'melt temperature rows copied above the liquidus at 1 GPa')[0].value == '0.0'
    )
    i = int(np.argmin(np.abs(table['P'] - GPA)))
    s_liq = float(eos.liquidus_entropy(float(table['P'][i])))
    j = int(np.searchsorted(table['S'], s_liq))
    original = table['values'][i].copy()
    table['values'][i, j : j + 6] = table['values'][i, j - 1]
    assert sheet_tool._melt_copied_above_liquidus(eos, GPA) == pytest.approx(
        float(table['S'][j + 5]) - s_liq
    )
    assert (
        _rows(eos, 'melt temperature rows copied above the liquidus at 1 GPa')[0].value != '0.0'
    )
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


def _cut_entropy(table, top):
    """Cut a table to its entropy nodes at or below ``top``."""
    keep = table['S'] <= top
    table['S'], table['values'] = table['S'][keep], table['values'][:, keep]


@pytest.mark.parametrize('name', ['density_melt', 'heat_capacity_melt', 'dTdPs_melt'])
def test_a_phase_table_with_a_narrower_entropy_range_reads_na(tmp_path, name):
    """A lookup clamps in entropy too: the liquidus at 20 GPa (2760) above the table top reads n/a,
    the 1 GPa liquidus (2324) below it still reads."""
    eos = _tables(tmp_path)
    assert sheet_tool._reaches(eos, 20.0 * GPA) and sheet_tool._reaches(eos, GPA)
    _cut_entropy(eos._tables[name], 2500.0)
    assert not sheet_tool._reaches(eos, 20.0 * GPA)
    assert sheet_tool._reaches(eos, GPA)
    assert _status(eos, 'rho_melt - rho_solid at 20 GPa') == ['n/a']


def test_a_solid_table_is_held_against_the_solidus_entropy(tmp_path):
    """A solid table that stops above the solidus reaches; one that stops below it does not."""
    eos = _tables(tmp_path)
    _cut_entropy(eos._tables['density_solid'], 2600.0)
    assert sheet_tool._reaches(eos, 20.0 * GPA)
    _cut_entropy(eos._tables['density_solid'], 2400.0)
    assert not sheet_tool._reaches(eos, 20.0 * GPA)


def test_a_phase_table_that_starts_above_its_boundary_entropy_reads_na(tmp_path):
    """The lower edge counts as well: a solid table from 2600 misses the 2480 solidus at 20 GPa."""
    eos = _tables(tmp_path)
    table = eos._tables['density_solid']
    keep = table['S'] >= 2600.0
    table['S'], table['values'] = table['S'][keep], table['values'][:, keep]
    assert not sheet_tool._reaches(eos, 20.0 * GPA)
    assert _status(eos, 'rho_melt - rho_solid at 20 GPa') == ['n/a']


def test_a_liquidus_above_the_table_top_reads_na(tmp_path):
    """A liquidus above every melt table is not a value at the table edge."""
    eos = _tables(tmp_path, ds0=7000.0, ds_slope=0.0)
    assert not sheet_tool._reaches(eos, 20.0 * GPA)
    assert _status(eos, 'rho_melt - rho_solid at 20 GPa') == ['n/a']


@pytest.mark.parametrize('curve', ['_solidus', '_liquidus'])
def test_a_melting_curve_with_a_narrower_pressure_range_reads_na(tmp_path, curve):
    """A melting curve clamps at its last pressure, so it must hold the pressure of the check."""
    eos = _tables(tmp_path)
    table = getattr(eos, curve)
    setattr(eos, curve, {**table, 'P': table['P'][table['P'] <= 15.0 * GPA]})
    assert not sheet_tool._reaches(eos, 20.0 * GPA)
    assert sheet_tool._reaches(eos, GPA)


@pytest.mark.parametrize(
    ('start', 'change'),
    [
        ('rho_melt - rho_solid at 1 GPa', {'rho_melt': np.nan}),
        ('alpha_melt and alpha_solid, ordering at 1 GPa', {'grad_melt': np.nan}),
        ('(dT/dP)_S mixed vs pure phases at 5 GPa', {'grad_melt': np.nan}),
    ],
)
def test_a_measurement_that_is_not_finite_reads_na(tmp_path, start, change):
    """A NaN is not a comparison with a band."""
    assert _status(_tables(tmp_path, **change), start) == ['n/a']


def test_a_delta_s_that_is_not_finite_reads_na(tmp_path):
    """A NaN entropy of fusion reads n/a at the lowest pressure, not outside."""
    eos = _tables(tmp_path, ds0=np.nan)
    assert _status(eos, 'Delta S at the lowest') == ['n/a']


@pytest.mark.parametrize('temperature', [0.0, -1.0])
def test_a_temperature_at_or_below_zero_gives_no_expansivity(tmp_path, temperature):
    """No expansivity exists at T <= 0: the row prints nan and reads n/a."""
    eos = _tables(tmp_path)
    eos._tables['temperature_melt']['values'][:] = temperature
    row = _rows(eos, 'alpha_melt and alpha_solid, ordering at 1 GPa')[0]
    assert row.status == 'n/a' and row.value.startswith('nan')


def test_the_alpha_rows_compare_near_the_surface_only_and_print_the_prescribed_ratio(good):
    """The prescribed values hold to about 5 GPa: above it a row reports. Every row prints the ratio
    of the measured alpha to the prescribed values."""
    assert _status(good, 'alpha_melt and alpha_solid, ordering at 1 GPa') == ['pass']
    assert _status(good, 'alpha_melt and alpha_solid, ordering at 5 GPa') == ['pass']
    assert _status(good, 'alpha_melt and alpha_solid, ordering at 10 GPa') == ['report']
    assert _status(good, 'alpha_melt and alpha_solid, ordering at 20 GPa') == ['report']
    for p_gpa in (1, 20):
        row = _rows(good, f'alpha_melt and alpha_solid, ordering at {p_gpa} GPa')[0]
        melt = sheet_tool._alpha(good, p_gpa * GPA, 'melt')
        solid = sheet_tool._alpha(good, p_gpa * GPA, 'solid')
        assert f'{melt / 4.6e-5:.1f} and {solid / 3.0e-5:.1f} times' in row.value


def test_the_mixed_gradient_status_follows_the_wide_step(tmp_path, monkeypatch):
    """The band reads at the wide step; a narrow step on the other side does not change the status."""
    steep = _tables(tmp_path, sol_slope=-40.0, ds_slope=-1.0, gap=500.0)
    wide = sheet_tool.MIX_STEPS[0]
    high, low = 1.0, -1.0
    start = '(dT/dP)_S mixed vs pure phases at 5 GPa'
    monkeypatch.setattr(
        sheet_tool, '_mix_gradient', lambda e, p, st: high if st == wide else low
    )
    assert _status(steep, start) == ['pass']
    monkeypatch.setattr(
        sheet_tool, '_mix_gradient', lambda e, p, st: low if st == wide else high
    )
    assert _status(steep, start) == ['outside']


def test_a_column_that_is_not_finite_does_not_hide_the_largest_padded_span(tmp_path):
    """A NaN column does not count as padded and does not turn the largest span into NaN."""
    eos = _tables(tmp_path, pad_solid=4)
    table = eos._tables['temperature_solid']
    table['values'][1:] = table['values'][1:] + np.linspace(0.0, 1.0, table['S'].size)
    table['values'][1, 0] = np.nan
    row = _rows(eos, 'pressure columns with solid temperature rows copied')[0]
    assert row.value.startswith('1 of 41')
    assert f'largest span {float(table["S"][4] - table["S"][0]):.1f}' in row.value


def test_the_commit_flag_names_a_folder_by_its_path_in_any_spelling(
    tmp_path, capsys, monkeypatch
):
    """A relative folder and an absolute flag name one folder; two flags for one folder stop."""
    _tables(tmp_path / 'a')
    monkeypatch.chdir(tmp_path)
    for flag in (f'{tmp_path / "a"}=abc', 'a/../a=abc', './a=abc'):
        try:
            assert sheet_tool.main(['a', '--paleos-commit', flag]) == 0
        except SystemExit as stop:
            pytest.fail(f'the flag {flag} stopped the tool with status {stop.code}')
        assert 'PALEOS commit: abc (given)' in capsys.readouterr().out
    with pytest.raises(SystemExit) as stop:
        sheet_tool.main(['a', '--paleos-commit', 'a=abc', '--paleos-commit', './a=def'])
    assert stop.value.code == 2
    assert 'twice' in capsys.readouterr().err
