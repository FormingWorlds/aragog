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
    first = float(table['P'][0]) / GPA
    assert f'between {first:.3g} and {first:.3g} GPa' in row.value
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
    row = _rows(eos, 'solid temperature rows copied above S_min, lowest')[0]
    assert row.value == 'nan' and row.status == 'n/a'


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
    """A liquidus at or below the solidus has no isentrope in the mixture and no ratio: n/a."""
    for name, change in (
        ('zero', {'ds0': 0.0, 'ds_slope': 0.0}),
        ('negative', {'ds0': -100.0, 'ds_slope': 0.0}),
    ):
        eos = _tables(tmp_path / name, **change)
        assert _status(eos, 'pressure where the isentrope') == ['n/a']
        assert _status(eos, 'Delta S at 100 GPa') == ['n/a']


def test_the_melt_rows_copied_above_the_liquidus_are_measured(tmp_path):
    """A melt column flat above the liquidus reports the flat span, also flat to the table top, and
    a plateau that ends at the first node above the liquidus counts."""
    eos = _tables(tmp_path)
    table = eos._tables['temperature_melt']
    assert sheet_tool._melt_copied_above_liquidus(eos, GPA) == 0.0
    start = 'melt temperature rows copied above the liquidus at 1 GPa'
    assert _rows(eos, start)[0].value.startswith('0.0 in')
    i = int(np.argmin(np.abs(table['P'] - GPA)))
    s_liq = float(eos.liquidus_entropy(float(table['P'][i])))
    j = int(np.searchsorted(table['S'], s_liq))
    original = table['values'][i].copy()
    table['values'][i, j : j + 6] = table['values'][i, j - 1]
    assert sheet_tool._melt_copied_above_liquidus(eos, GPA) == pytest.approx(
        float(table['S'][j + 5]) - s_liq
    )
    assert not _rows(eos, start)[0].value.startswith('0.0 in')
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


def _cut_entropy(table, low=-np.inf, top=np.inf):
    """Cut a table to its entropy nodes from ``low`` to ``top``."""
    keep = (table['S'] >= low) & (table['S'] <= top)
    table['S'], table['values'] = table['S'][keep], table['values'][:, keep]


@pytest.mark.parametrize('name', ['density_melt', 'heat_capacity_melt', 'dTdPs_melt'])
def test_a_phase_table_with_a_narrower_entropy_range_reads_na(tmp_path, name):
    """A lookup clamps in entropy too: the liquidus at 20 GPa (2760) above the table top reads n/a,
    the 1 GPa liquidus (2324) below it still reads."""
    eos = _tables(tmp_path)
    assert sheet_tool._reaches(eos, 20.0 * GPA) and sheet_tool._reaches(eos, GPA)
    _cut_entropy(eos._tables[name], top=2500.0)
    assert not sheet_tool._reaches(eos, 20.0 * GPA)
    assert sheet_tool._reaches(eos, GPA)
    assert _status(eos, 'rho_melt - rho_solid at 20 GPa') == ['n/a']


def test_a_solid_table_is_held_against_the_solidus_entropy(tmp_path):
    """A solid table that stops above the solidus reaches; one that stops below it does not."""
    eos = _tables(tmp_path)
    _cut_entropy(eos._tables['density_solid'], top=2600.0)
    assert sheet_tool._reaches(eos, 20.0 * GPA)
    _cut_entropy(eos._tables['density_solid'], top=2400.0)
    assert not sheet_tool._reaches(eos, 20.0 * GPA)


@pytest.mark.parametrize(('name', 'low'), [('density_solid', 2600.0), ('density_melt', 2900.0)])
def test_a_phase_table_that_starts_above_its_boundary_entropy_reads_na(tmp_path, name, low):
    """The lower edge counts as well: 2480 solidus and 2760 liquidus at 20 GPa, below each start."""
    eos = _tables(tmp_path)
    _cut_entropy(eos._tables[name], low=low)
    assert not sheet_tool._reaches(eos, 20.0 * GPA)
    assert _status(eos, 'rho_melt - rho_solid at 20 GPa') == ['n/a']


def test_a_liquidus_above_the_table_top_reads_na(tmp_path):
    """A liquidus above every melt table is not a value at the table edge."""
    eos = _tables(tmp_path, ds0=7000.0, ds_slope=0.0)
    assert not sheet_tool._reaches(eos, 20.0 * GPA)
    assert _status(eos, 'rho_melt - rho_solid at 20 GPa') == ['n/a']


@pytest.mark.parametrize('curve', ['_solidus', '_liquidus'])
def test_a_melting_curve_with_a_narrower_pressure_range_reads_na(tmp_path, curve):
    """A melting curve clamps at its edges, so it must hold the pressure of the check."""
    eos = _tables(tmp_path)
    table = getattr(eos, curve)
    setattr(eos, curve, {**table, 'P': table['P'][table['P'] <= 15.0 * GPA]})
    assert not sheet_tool._reaches(eos, 20.0 * GPA)
    assert sheet_tool._reaches(eos, GPA)
    setattr(eos, curve, {**table, 'P': table['P'][table['P'] >= 2.0 * GPA]})
    assert not sheet_tool._reaches(eos, GPA)
    assert sheet_tool._reaches(eos, 20.0 * GPA)


@pytest.mark.parametrize(
    ('start', 'change'),
    [
        ('rho_melt - rho_solid at 1 GPa', {'rho_melt': np.nan}),
        ('alpha_melt and alpha_solid, ordering at 1 GPa', {'grad_melt': np.nan}),
        ('alpha_melt and alpha_solid, ordering at 1 GPa', {'grad_solid': np.nan}),
        ('(dT/dP)_S mixed vs pure phases at 5 GPa', {'grad_melt': np.nan}),
        ('(dT/dP)_S mixed vs pure phases at 5 GPa', {'grad_solid': np.nan}),
        ('Delta S at the lowest', {'ds0': np.nan}),
        ('rho_melt - rho_solid at 10 GPa', {'rho_melt': np.nan}),
        ('alpha_melt and alpha_solid, ordering at 10 GPa', {'grad_melt': np.nan}),
        ('alpha_melt and alpha_solid, ordering at 10 GPa', {'grad_solid': np.nan}),
    ],
)
def test_a_measurement_that_is_not_finite_reads_na(tmp_path, start, change):
    """A NaN is not a comparison with a band."""
    assert _status(_tables(tmp_path, **change), start) == ['n/a']


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


MIXED_5 = '(dT/dP)_S mixed vs pure phases at 5 GPa'


@pytest.mark.parametrize(
    ('wide', 'narrow', 'status'),
    [
        (25.0e-9, 18.0e-9, 'pass'),
        (18.0e-9, 25.0e-9, 'outside'),
        (30.0e-9, 20.0e-9, 'pass'),
        (19.0e-9, 28.0e-9, 'outside'),
        (30.2e-9, 20.0e-9, 'report'),
        (20.0e-9, 30.2e-9, 'report'),
        (25.0e-9, -25.0e-9, 'report'),
        (0.0, 0.0, 'report'),
        (np.nan, 25.0e-9, 'n/a'),
        (25.0e-9, np.nan, 'n/a'),
    ],
)
def test_the_mixed_gradient_reads_at_the_wide_step_only_where_the_2_steps_agree(
    good, monkeypatch, wide, narrow, status
):
    """Melt is the steeper phase at 20 K/GPa. The reading follows the wide step within a factor 1.5
    of the narrow one; beyond it the row reports, and a value that is not finite reads n/a."""
    wide_step = sheet_tool.MIX_STEPS[0]
    monkeypatch.setattr(
        sheet_tool, '_mix_gradient', lambda e, p, st: wide if st == wide_step else narrow
    )
    assert _status(good, MIXED_5) == [status]


def test_the_mixed_gradient_is_the_central_difference_at_the_entropy_halfway(good):
    """The printed value at each step against a difference taken here, not by the tool."""
    p = 5.0 * GPA
    s = 0.5 * float(good.solidus_entropy(p) + good.liquidus_entropy(p))
    value = _rows(good, MIXED_5)[0].value
    printed = [float(x) for x in re.findall(r'(-?[0-9.]+) at', value)]
    assert len(printed) == 2
    for step, shown in zip(sheet_tool.MIX_STEPS, printed):
        dp = step * p
        hi, lo = (
            float(good.temperature(np.array([x]), np.array([s]))[0]) for x in (p + dp, p - dp)
        )
        want = (hi - lo) / (2.0 * dp)
        assert sheet_tool._mix_gradient(good, p, step) == pytest.approx(want, rel=1.0e-6)
        assert shown == pytest.approx(GPA * want, abs=0.006)


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


def _cut_curve_files(folder, low=0.0, top=np.inf):
    """Rewrite both melting curve files to the pressures from ``low`` to ``top`` and reload."""
    for name in ('solidus_P-S.dat', 'liquidus_P-S.dat'):
        data = np.loadtxt(folder / name)
        data = data[(data[:, 0] >= low) & (data[:, 0] <= top)]
        head = f'5 {len(data)}\nPressure, Entropy\nx\ny\n1.0 1.0'
        np.savetxt(folder / name, data, header=head, fmt='%.12e')
    return EntropyEOS(folder)


LOW_P_ROWS = ('Delta S at the lowest', 'Delta S at 100 GPa', 'pressure where the isentrope')


def test_a_row_at_the_lowest_pressure_reads_na_when_a_curve_starts_above_it(tmp_path):
    """The curves clamp at 3.75 GPa; no row prints a value read at that clamp."""
    _tables(tmp_path)
    eos = _cut_curve_files(tmp_path, low=3.75 * GPA)
    assert not sheet_tool._reaches(eos, float(eos.P_min))
    for start in LOW_P_ROWS:
        row = _rows(eos, start)[0]
        assert (row.value, row.status) == ('-', 'n/a')


def test_a_row_at_the_lowest_pressure_reads_na_when_a_table_starts_above_it(tmp_path):
    """P_min comes from the temperature tables, so a density table can start above it."""
    eos = _tables(tmp_path)
    table = eos._tables['density_melt']
    keep = table['P'] >= 2.0 * GPA
    table['P'], table['values'] = table['P'][keep], table['values'][keep]
    for start in LOW_P_ROWS:
        assert _status(eos, start) == ['n/a']


def test_the_ratio_needs_the_upper_pressure_as_well(tmp_path):
    """The curves stop at 50 GPa: the ratio reads n/a while Delta S at the lowest pressure reads."""
    _tables(tmp_path)
    eos = _cut_curve_files(tmp_path, top=50.0 * GPA)
    assert _status(eos, 'Delta S at 100 GPa') == ['n/a']
    assert _status(eos, 'Delta S at the lowest') == ['report']


@pytest.mark.parametrize('start', LOW_P_ROWS)
def test_the_entropy_of_fusion_rows_and_the_isentrope_report_with_no_reading(
    good, tmp_path, start
):
    """A value far from the published number still reports, and the band text claims no band."""
    assert _status(good, start) == ['report']
    far = _tables(tmp_path, ds0=1000.0, ds_slope=1.0, sol_slope=12.0)
    assert _status(far, start) == ['report']
    assert 'chosen here' not in _rows(far, start)[0].band


def test_the_published_numbers_stand_next_to_the_reported_values(good):
    """Value and published number, in one row."""
    assert '300' in _rows(good, 'Delta S at the lowest')[0].band
    assert _rows(good, 'Delta S at the lowest')[0].value == '300.0'
    assert 'about 5' in _rows(good, 'pressure where the isentrope')[0].band


@pytest.mark.parametrize('name', ['density_melt', 'temperature_solid', 'dTdPs_solid'])
def test_the_lower_pressure_edge_of_a_table_counts(tmp_path, name):
    """A table that starts above the pressure of a check does not reach it."""
    eos = _tables(tmp_path)
    table = eos._tables[name]
    keep = table['P'] >= 2.0 * GPA
    table['P'], table['values'] = table['P'][keep], table['values'][keep]
    assert not sheet_tool._reaches(eos, GPA)
    assert sheet_tool._reaches(eos, 20.0 * GPA)


def test_the_range_edges_hold_their_own_value(tmp_path):
    """A check exactly at a pressure edge or at an entropy edge is inside the table."""
    eos = _tables(tmp_path)
    assert sheet_tool._reaches(eos, float(eos.P_min))
    assert sheet_tool._reaches(eos, float(eos.P_max))
    p = 20.0 * GPA
    s_sol = float(eos.solidus_entropy(p))
    s_liq = float(eos.liquidus_entropy(p))
    solid, melt = eos._tables['density_solid'], eos._tables['density_melt']
    solid['S'], melt['S'] = solid['S'].copy(), melt['S'].copy()
    solid['S'][0], melt['S'][-1] = s_sol, s_liq
    assert sheet_tool._reaches(eos, p)
    solid['S'][0], solid['S'][-1] = s_sol - 1.0, s_sol
    melt['S'][0], melt['S'][-1] = s_liq, s_liq + 1.0
    assert sheet_tool._reaches(eos, p)


def test_a_table_the_sheet_does_not_read_does_not_count(tmp_path):
    """An expansivity table is not read, so its range changes no row."""
    eos = _tables(tmp_path)
    before = [row.status for row in returns(sheet_tool.sheet, eos)]
    eos._tables['thermal_exp_melt'] = {
        'P': np.array([1.0, 2.0]),
        'S': np.array([1.0, 2.0]),
        'values': np.zeros((2, 2)),
    }
    assert [row.status for row in returns(sheet_tool.sheet, eos)] == before


def test_a_solidus_that_falls_back_gives_the_first_crossing(tmp_path):
    """S_sol = 2000 + 24 P - 0.5 P^2 reaches the target 2120 first near 5.7 GPa, then falls."""
    eos = _tables(tmp_path, sol_slope=24.0, sol_curve=-0.5)
    assert float(eos.solidus_entropy(float(eos.P_max))) < 2120.0
    row = _rows(eos, 'pressure where the isentrope')[0]
    assert float(row.value) == pytest.approx(24.0 - np.sqrt(24.0**2 - 2.0 * 120.0), abs=0.15)
    p = float(row.value) * GPA
    assert float(eos.solidus_entropy(p)) == pytest.approx(2120.0, abs=0.2)
    assert row.status == 'report'


def test_an_isentrope_that_stays_mixed_prints_the_top_of_the_range(tmp_path):
    """No crossing is a result: above the pressure range, with status report, and not n/a."""
    flat = _tables(tmp_path / 'flat', sol_slope=0.1)
    row = _rows(flat, 'pressure where the isentrope')[0]
    assert row.value == f'> {float(flat.P_max) / GPA:.3g}'
    assert row.status == 'report'
    peak = _tables(tmp_path / 'peak', sol_slope=24.0, sol_curve=-0.5, ds0=800.0)
    assert _rows(peak, 'pressure where the isentrope')[0].value.startswith('> ')


def test_the_search_stops_at_the_top_of_the_solidus(tmp_path):
    """A solidus that stops below the tables gives no crossing read at its clamp."""
    _tables(tmp_path / 'a', sol_slope=4.0)
    eos = _cut_curve_files(tmp_path / 'a', top=20.0 * GPA)
    row = _rows(eos, 'pressure where the isentrope')[0]
    top = float(eos._solidus['P'][-1])
    assert top < float(eos.P_max)
    assert (row.value, row.status) == (f'> {top / GPA:.3g}', 'report')


def test_a_report_row_that_is_not_finite_reads_na(tmp_path):
    """The gap at 50 GPa from melt temperatures that are not finite."""
    eos = _tables(tmp_path)
    eos._tables['temperature_melt']['values'][:] = np.nan
    assert _status(eos, 'T_liquidus - T_solidus at 50 GPa') == ['n/a']


@pytest.mark.parametrize('remove', [None, 'density_melt.dat'])
def test_a_folder_that_is_missing_or_incomplete_stops_with_a_message(tmp_path, capsys, remove):
    """No traceback: a missing folder, or a folder without one of its tables."""
    folder = tmp_path / 'absent'
    if remove:
        folder = tmp_path / 'part'
        _tables(folder)
        (folder / remove).unlink()
    try:
        sheet_tool.main([str(folder)])
    except FileNotFoundError as missing:
        pytest.fail(f'a traceback, not a message: {missing}')
    except SystemExit as stop:
        assert stop.code == 2
    else:
        pytest.fail('no stop on a folder without its tables')
    assert str(folder) in capsys.readouterr().err


def test_the_melt_row_names_the_column_it_reads(good):
    """The nearest column stands in the value, so a reader sees the pressure of the reading."""
    table = good._tables['temperature_melt']
    column = float(table['P'][int(np.argmin(np.abs(table['P'] - GPA)))])
    row = _rows(good, 'melt temperature rows copied above the liquidus at 1 GPa')[0]
    assert row.value == f'0.0 in the column at {column / GPA:.4g} GPa'


def test_the_band_texts_name_the_source_of_each_limit(good):
    """The density band is MK22A; the 5 GPa cut and the alpha limit say where they come from."""
    near = _rows(good, 'rho_melt - rho_solid at 5 GPa')[0]
    assert near.source == 'MK22A p23, p5' and 'melt layer' in near.band
    assert _rows(good, 'rho_melt - rho_solid at 20 GPa')[0].source == 'MK19b p9'
    for p_gpa in (1, 10):
        row = _rows(good, f'alpha_melt and alpha_solid, ordering at {p_gpa} GPa')[0]
        assert 'not a statement of the paper' in row.band


def test_the_count_line_counts_the_printed_rows(tmp_path):
    """Pass, outside, report and n/a all occur, with counts read from the printed table here."""
    _tables(tmp_path, rho_melt=3500.0, p_max=60.0 * GPA)
    lines = sheet_tool.describe(tmp_path).splitlines()
    head = lines.index('| check | value | band | source | status |')
    statuses = [
        line.rsplit('|', 2)[1].strip() for line in lines[head + 2 :] if line.startswith('|')
    ]
    found = {status: statuses.count(status) for status in ('pass', 'outside', 'report', 'n/a')}
    assert sum(found.values()) == len(statuses) and all(found.values())
    want = (
        f'{found["pass"]} pass, {found["outside"]} outside, '
        f'{found["report"]} report, {found["n/a"]} n/a.'
    )
    assert any(line.startswith(want) for line in lines)


def test_the_range_line_gives_the_range_of_the_tables(tmp_path):
    """Entropy and pressure edges of the loaded temperature tables, each in its place."""
    eos = _tables(tmp_path)
    solid = eos._tables['temperature_solid']
    want = (
        f'S {float(solid["S"][0]):.3f} to {float(solid["S"][-1]):.3f} J/kg/K, '
        f'P {float(solid["P"][0]):.3g} to {float(solid["P"][-1]):.3g} Pa'
    )
    assert want in sheet_tool.describe(tmp_path).splitlines()


def test_the_isentrope_row_names_both_pages_and_the_mk19b_band_its_case(good):
    """The depth is on MK22A p5 and the Katz model on p23; MK19b p9 holds for no accumulation."""
    assert _rows(good, 'pressure where the isentrope')[0].source == 'MK22A p5, p23'
    band = _rows(good, 'rho_melt - rho_solid at 20 GPa')[0].band
    assert 'without crystal accumulation' in band and 'reverses near 75 GPa' in band


def test_the_padded_range_does_not_claim_contiguous_columns(tmp_path):
    """Columns 0 and 2 are copied and column 1 is not: the range reads between them."""
    eos = _tables(tmp_path, pad_solid=4)
    table = eos._tables['temperature_solid']
    table['values'][3:] = table['values'][3:] + np.linspace(0.0, 1.0, table['S'].size)
    table['values'][1] = table['values'][1] + np.linspace(0.0, 1.0, table['S'].size)
    value = _rows(eos, 'pressure columns with solid temperature rows copied')[0].value
    first, last = (float(table['P'][i]) / GPA for i in (0, 2))
    assert value.startswith(f'2 of 41, between {first:.3g} and {last:.3g} GPa')
