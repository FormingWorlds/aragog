"""The table sheet: each row can report pass and outside on a small synthetic table set."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from aragog.eos.entropy import EntropyEOS

from .conftest import EOS_DIR, needs_eos

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
    'gap': 700.0,
    'pad_solid': 0,
}


def _tables(folder: Path, **change) -> EntropyEOS:
    """Write a linear table set; ``change`` overrides ``_GOOD``. Entropy spans 100 to 8000 J/kg/K."""
    a = {**_GOOD, **change}
    p = np.linspace(1.0e5, a['p_max'], 101)
    s = np.linspace(100.0, 8000.0, 201)
    s_sol = 2000.0 + a['sol_slope'] * p / GPA
    s_liq = s_sol + a['ds0'] + a['ds_slope'] * p / GPA
    t_solid = lambda pp, ss: 1000.0 + 0.5 * (ss - 2000.0) + 1.0e-8 * pp  # noqa: E731
    t_melt = lambda pp, ss: t_solid(pp, ss) + a['gap'] - 150.0  # noqa: E731

    def write(name, fun):
        pp, ss = np.meshgrid(p, s)
        rows = np.column_stack([pp.ravel(), ss.ravel(), fun(pp, ss).ravel()])
        head = f'5 {p.size} {s.size}\nPressure, Entropy, Quantity\nx\ny\n1.0 1.0 1.0'
        np.savetxt(folder / name, rows, header=head, fmt='%.12e')

    def solid_t(pp, ss):
        t = t_solid(pp, ss)
        if a['pad_solid']:
            t[: a['pad_solid'], :] = t[a['pad_solid'], :]
        return t

    write('temperature_solid.dat', solid_t)
    write('temperature_melt.dat', t_melt)
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


def _status(eos, start):
    """Status of each row whose check starts with ``start``; at least one row must match."""
    found = [r.status for r in sheet_tool.sheet(eos) if r.check.startswith(start)]
    assert found, f'no row starts with {start!r}'
    return found


@pytest.mark.parametrize(
    ('start', 'bad'),
    [
        ('rho_melt - rho_solid', {'rho_melt': 3500.0}),
        ('alpha_melt', {'grad_melt': 0.1e-8}),
        ('Delta S at the lowest', {'ds0': 1000.0, 'ds_slope': 0.0}),
        ('Delta S at 100 GPa', {'ds_slope': 1.0}),
        ('T_liquidus', {'gap': 1500.0}),
        ('pressure where the isentrope', {'sol_slope': 2.0}),
        ('solid temperature rows copied', {'pad_solid': 4}),
    ],
)
def test_a_row_passes_on_the_good_table_and_reports_outside_on_a_bad_one(tmp_path, start, bad):
    (tmp_path / 'good').mkdir()
    (tmp_path / 'bad').mkdir()
    assert set(_status(_tables(tmp_path / 'good'), start)) == {'pass'}
    assert 'outside' in _status(_tables(tmp_path / 'bad', **bad), start)


def test_the_mixed_gradient_row_reports_both_states(tmp_path):
    """The mixture is steeper than both phases when the liquidus falls fast with pressure."""
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    steep = _tables(tmp_path / 'a', sol_slope=-40.0, ds_slope=-1.0, gap=500.0)
    assert set(_status(steep, '(dT/dP)_S mixed')) == {'pass'}
    assert 'outside' in _status(_tables(tmp_path / 'b'), '(dT/dP)_S mixed')
    (tmp_path / 'c').mkdir()
    between = _tables(
        tmp_path / 'c',
        sol_slope=-40.0,
        ds_slope=-1.0,
        gap=500.0,
        grad_melt=5.0e-6,
        grad_solid=1.0e-10,
    )
    assert 'outside' in _status(between, '(dT/dP)_S mixed')  # above the solid, below the melt


def test_a_zero_solid_expansivity_gives_a_row_and_no_crash(tmp_path):
    """A solid gradient of 0 in a table reads as an infinite ratio, which passes the sign test."""
    eos = _tables(tmp_path, grad_solid=0.0)
    assert set(_status(eos, 'alpha_melt')) == {'pass'}


def test_the_canary_row_fails_on_a_planted_sign_flip(tmp_path):
    """A density table with the sign of the melt contrast flipped fails the first row, and an
    exit status of 0 still reports it."""
    assert sheet_tool.main([str(_write(tmp_path))]) == 0
    flipped = EntropyEOS(tmp_path)
    assert set(_status(flipped, 'rho_melt - rho_solid at 1 GPa')) == {'outside'}


def _write(folder: Path) -> Path:
    _tables(folder, rho_melt=3300.0, rho_solid=3000.0)
    return folder


def test_a_pressure_the_tables_do_not_reach_reads_na(tmp_path):
    """Edges come from the loaded tables: no row reads past them."""
    eos = _tables(tmp_path, p_max=15.0 * GPA)
    assert _status(eos, 'rho_melt - rho_solid at 20 GPa') == ['n/a']
    assert _status(eos, 'T_liquidus - T_solidus at 50 GPa') == ['n/a']
    assert _status(eos, 'melt temperature rows copied above the liquidus at 20 GPa') == ['n/a']


def test_the_melt_rows_copied_above_the_liquidus_are_measured(tmp_path):
    """A melt column that stays flat above the liquidus reports the flat span."""
    eos = _tables(tmp_path)
    table = eos._tables['temperature_melt']
    assert sheet_tool._melt_copied_above_liquidus(eos, GPA) == 0.0
    i = int(np.argmin(np.abs(table['P'] - GPA)))
    s_liq = float(eos.liquidus_entropy(GPA))
    j = int(np.searchsorted(table['S'], s_liq))
    table['values'][i, j : j + 6] = table['values'][i, j]
    assert sheet_tool._melt_copied_above_liquidus(eos, GPA) == pytest.approx(
        float(table['S'][j + 5]) - s_liq
    )


@needs_eos
def test_the_sheet_reads_a_real_table_set(capsys):
    """The test tables give a full sheet: every status is one of the 3, the md5 lines print."""
    assert sheet_tool.main([str(EOS_DIR)]) == 0
    out = capsys.readouterr().out
    assert 'temperature_solid.dat md5 ' in out and 'temperature_melt.dat md5 ' in out
    rows = sheet_tool.sheet(EntropyEOS(EOS_DIR))
    assert len(rows) == 21 and {r.status for r in rows} <= {'pass', 'outside', 'n/a'}
