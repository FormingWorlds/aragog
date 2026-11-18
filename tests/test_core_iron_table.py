"""Tests of ``aragog.core.iron_table.LiquidIronTable``, the PALEOS liquid iron table reader."""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aragog.core.iron_table import FIELDS, LiquidIronTable
from tests.conftest import PALEOS_CUT as CUT

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]


@pytest.fixture(scope='module')
def table():
    """The 40-per-decade liquid cut, 45 GPa to 3 TPa, 2000 to 2e4 K."""
    return LiquidIronTable.from_file(CUT)


def _write_dat(path, data, kind='liquid'):
    """Write ``data`` (the npz arrays) as a Zalmoxis-format text table."""
    header = str(data['header']).replace('# Kind: liquid', f'# Kind: {kind}')
    rows = []
    for i, p in enumerate(data['P_Pa']):
        for j, t in enumerate(data['T_K']):
            vals = ' '.join(f'{data[f][i, j]:.8e}' for f in FIELDS)
            rows.append(f'{p:.8e} {t:.8e} {vals} liquid')
    path.write_text(header + '\n' + '\n'.join(rows) + '\n')
    return path


def test_text_and_npz_tables_agree_at_the_nodes_and_between(tmp_path, table):
    """The .dat reader gives the npz values at every node, and both interpolate alike."""
    data = np.load(CUT)
    dat = LiquidIronTable.from_file(_write_dat(tmp_path / 't.dat', data))
    p, t = data['P_Pa'][[3, 40]], data['T_K'][[5, 30]]
    for f in FIELDS:
        np.testing.assert_allclose(dat.value(f, p, t), data[f][[3, 40], [5, 30]], rtol=1e-8)
    p_mid, t_mid = 1.234e11, 5432.1
    for f in ('rho', 's'):
        assert dat.value(f, p_mid, t_mid) == pytest.approx(table.value(f, p_mid, t_mid), 1e-8)


@pytest.mark.physics_invariant
def test_interpolant_is_exact_on_nodes_and_cubic_between(table):
    """At the nodes the interpolant returns the stored values, and its ds/dT equals the
    table's cp/T (the identity dh = T ds at fixed P) to the error of the grid."""
    data = np.load(CUT)
    i, j = 30, 20
    p, t = data['P_Pa'][i], data['T_K'][j]
    for f in FIELDS:
        assert table.value(f, p, t) == pytest.approx(data[f][i, j], rel=1e-12)
    # Thermodynamic identity on the table: (ds/dT)_P = cp / T, to the 40-per-decade grid error.
    # Interior nodes (2 from each edge): 6.2e-6 measured; a missing ln(10) gives 2.3.
    pp, tt = np.meshgrid(data['P_Pa'][2:-2], data['T_K'][2:-2], indexing='ij')
    ratio = table.ds_dT(pp, tt) * tt / table.value('cp', pp, tt)
    np.testing.assert_allclose(ratio, 1.0, atol=1e-5)


def test_outside_the_box_is_refused(table):
    """A pressure or temperature outside the table box raises and names the point."""
    with pytest.raises(ValueError, match='outside the liquid iron table'):
        table.value('rho', 3.1e12, 5000.0)
    with pytest.raises(ValueError, match='outside the liquid iron table'):
        table.value('rho', 1e11, 1500.0)
    assert table.contains(1e11, 5000.0)
    assert not table.contains(np.array([1e11, 4e12]), 5000.0)[1]


def test_kind_other_than_liquid_is_refused(tmp_path):
    """A solid, unified or mixed-kind table is refused: the core isentrope needs the liquid
    branch, and a kind word that starts with liquid is not the liquid branch."""
    data = np.load(CUT)
    with pytest.raises(ValueError, match="kind 'solid'"):
        LiquidIronTable.from_file(_write_dat(tmp_path / 's.dat', data, kind='solid'))
    for kind in ('liquid-solid', 'liquid+solid'):
        with pytest.raises(ValueError, match=f"kind '{re.escape(kind)}'"):
            LiquidIronTable.from_file(_write_dat(tmp_path / 'm.dat', data, kind=kind))
    no_kind = tmp_path / 'n.dat'
    no_kind.write_text(
        '\n'.join(
            l
            for l in _write_dat(no_kind, data).read_text().splitlines()
            if not l.startswith('# Kind')
        )
    )
    with pytest.raises(ValueError, match='no Kind line'):
        LiquidIronTable.from_file(no_kind)


def test_unknown_suffix_and_missing_file_are_refused(tmp_path):
    """Only .dat and .npz are read; a missing path raises FileNotFoundError."""
    bad = tmp_path / 't.csv'
    bad.write_text('x')
    with pytest.raises(ValueError, match='.dat or .npz'):
        LiquidIronTable.from_file(bad)
    with pytest.raises(FileNotFoundError):
        LiquidIronTable.from_file(tmp_path / 'absent.npz')


def _arrays():
    """Small valid table arrays."""
    data = np.load(CUT)
    return (
        data['P_Pa'][:6].copy(),
        data['T_K'][:6].copy(),
        {f: data[f][:6, :6].copy() for f in FIELDS},
    )


@settings(max_examples=40, deadline=None)
@given(
    field=st.sampled_from(FIELDS),
    bad=st.sampled_from([np.nan, np.inf, -np.inf]),
    i=st.integers(1, 4),
    j=st.integers(0, 5),
)
def test_a_non_finite_cell_inside_is_refused(field, bad, i, j):
    """A NaN or infinite cell in a pressure row between finite edge rows is refused."""
    p, t, fields = _arrays()
    fields[field][i, j] = bad
    with pytest.raises(ValueError, match=f'non-finite cell in {field} inside the table'):
        LiquidIronTable(p, t, fields, header='# Kind: liquid')


@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    field=st.sampled_from(FIELDS),
    bad=st.sampled_from([np.nan, np.inf, -np.inf]),
    low=st.integers(0, 2),
    high=st.integers(0, 2),
    j=st.integers(0, 5),
)
def test_non_finite_rows_at_a_pressure_edge_are_dropped(caplog, field, bad, low, high, j):
    """Pressure rows with a non-finite cell at the bottom or top of the table are dropped and
    the new range is logged; the finite rows between keep their values."""
    data = np.load(CUT)
    p, t = data['P_Pa'][:9].copy(), data['T_K'][:6].copy()
    fields = {f: data[f][:9, :6].copy() for f in FIELDS}
    for i in [*range(low), *range(9 - high, 9)]:
        fields[field][i, j] = bad
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl.aragog'):
        tab = LiquidIronTable(p, t, fields, header='# Kind: liquid')
    log = ' '.join(r.getMessage() for r in caplog.records if r.name.startswith('fwl.aragog.'))
    np.testing.assert_array_equal(tab.pressure, p[low : 9 - high])
    assert tab.value('s', p[low + 1], t[2]) == pytest.approx(data['s'][low + 1, 2], rel=1e-12)
    dropped = low + high
    assert (f'drops {dropped} pressure rows' in log) == (dropped > 0), log


def test_a_table_with_no_finite_row_is_refused():
    """Edge rows that leave fewer than 4 finite pressure rows are refused."""
    p, t, fields = _arrays()
    fields['rho'][[0, 1, 4, 5], 0] = np.nan
    with pytest.raises(ValueError, match='fewer than 4 finite pressure rows'):
        LiquidIronTable(p, t, fields, header='# Kind: liquid')


def test_the_default_grid_top_rows_load_without_their_non_finite_cells():
    """The top of the default Zalmoxis grid holds cells that did not converge (NaN, phase
    'out-of-range') above 6e13 Pa; its cut at 4000 to 5000 K, with 5 such rows at the top,
    loads up to the last finite row."""
    top = LiquidIronTable.from_file(CUT.with_name('paleos_iron_liquid_default_top.npz'))
    data = np.load(CUT.with_name('paleos_iron_liquid_default_top.npz'))
    finite = np.all([np.all(np.isfinite(data[f]), axis=1) for f in FIELDS], axis=0)
    assert top.pressure[-1] == data['P_Pa'][finite][-1] == pytest.approx(9.261e13, rel=1e-3)
    assert top.pressure.size == data['P_Pa'].size - 5


@settings(max_examples=40, deadline=None)
@given(value=st.sampled_from([0.0, -1.0, -8000.0]), i=st.integers(0, 5), j=st.integers(0, 5))
def test_a_non_positive_density_or_heat_capacity_is_refused(value, i, j):
    """Zero or negative rho, cp or cv is refused."""
    for field in ('rho', 'cp', 'cv'):
        p, t, fields = _arrays()
        fields[field][i, j] = value
        with pytest.raises(ValueError, match=f'{field} must be positive'):
            LiquidIronTable(p, t, fields, header='# Kind: liquid')


@pytest.mark.parametrize(
    ('which', 'mutate', 'msg'),
    [
        ('p', lambda a: a[::-1], 'strictly increasing'),
        ('t', lambda a: np.r_[a[:2], a[1], a[3:]], 'strictly increasing'),
        ('p', lambda a: a[:3], 'at least 4'),
        ('t', lambda a: -a, 'positive'),
        ('p', lambda a: np.r_[0.0, a[1:]], 'positive'),
    ],
)
def test_bad_axes_are_refused(which, mutate, msg):
    """Axes must be positive, strictly increasing and at least 4 long (cubic interpolation)."""
    p, t, fields = _arrays()
    if which == 'p':
        p = mutate(p)
        fields = {f: v[: p.size] for f, v in fields.items()}
    else:
        t = mutate(t)
        fields = {f: v[:, : t.size] for f, v in fields.items()}
    with pytest.raises(ValueError, match=msg):
        LiquidIronTable(p, t, fields, header='# Kind: liquid')


def test_a_field_of_the_wrong_shape_or_a_missing_field_is_refused():
    """Every field is present with shape (len P, len T)."""
    p, t, fields = _arrays()
    fields['s'] = fields['s'][:, :5]
    with pytest.raises(ValueError, match='s has shape'):
        LiquidIronTable(p, t, fields, header='# Kind: liquid')
    p, t, fields = _arrays()
    del fields['nabla_ad']
    with pytest.raises(ValueError, match='missing fields'):
        LiquidIronTable(p, t, fields, header='# Kind: liquid')


@pytest.mark.physics_invariant
def test_ds_dp_is_the_maxwell_relation(table):
    """``(ds/dP)_T = -alpha / rho`` and ``-(ds/dP)/(ds/dT) = nabla_ad T / P`` on the interior
    nodes up to 2 TPa, to the error of the grid (7.5e-6 at most measured there). Above 2.4 TPa
    the alpha column of the table departs from its entropy column (1.4e-3 to 1.5e-3 at 2.67 TPa,
    on both cuts), so those nodes are left out. A missing ln(10) or a wrong axis gives an
    error of order 1."""
    data = np.load(CUT)
    keep = data['P_Pa'][2:-2] <= 2e12
    pp, tt = np.meshgrid(data['P_Pa'][2:-2][keep], data['T_K'][2:-2], indexing='ij')
    maxwell = table.ds_dP(pp, tt) / (-table.value('alpha', pp, tt) / table.value('rho', pp, tt))
    np.testing.assert_allclose(maxwell, 1.0, atol=3e-5)
    slope = -table.ds_dP(pp, tt) / table.ds_dT(pp, tt)
    np.testing.assert_allclose(
        slope / (table.value('nabla_ad', pp, tt) * tt / pp), 1.0, atol=3e-5
    )


def test_a_text_table_that_is_not_a_full_rectangle_is_refused(tmp_path):
    """A .dat table with one row missing is refused before any interpolant is built."""
    data = np.load(CUT)
    path = _write_dat(tmp_path / 't.dat', data)
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines[:-1]) + '\n')
    with pytest.raises(ValueError, match='not a full rectangle'):
        LiquidIronTable.from_file(path)
    path.write_text('\n'.join(lines) + '\n')
    assert LiquidIronTable.from_file(path).fields['rho'].shape == data['rho'].shape


def test_a_duplicated_row_in_place_of_a_missing_one_is_refused(tmp_path):
    """A text table where one (P, T) row is missing and another appears twice has the right
    row count; the reader must still refuse it, since its values would land at wrong nodes."""
    data = np.load(CUT)
    path = _write_dat(tmp_path / 't.dat', data)
    lines = path.read_text().splitlines()
    path.write_text('\n'.join([*lines[:-1], lines[-2]]) + '\n')
    with pytest.raises(ValueError, match='not a full rectangle'):
        LiquidIronTable.from_file(path)


@pytest.mark.parametrize(
    ('body', 'msg'),
    [
        ('', 'no data rows'),
        ('1e9 1e3 1 2 3 4 5 6 liquid', 'needs 9 numbers'),
        ('1e9 1e3 1 2 3 4 5 6\n1e9 2e3 1 2 3 4 5 6', 'needs 9 numbers'),
        ('1e9 1e3 1 2 3 4 5 6 7 liquid\n1e9 2e3 1 2 3 liquid', 'needs 9 numbers'),
    ],
)
def test_a_malformed_text_table_is_refused_with_its_name(tmp_path, body, msg):
    """An empty table, a short row and a ragged table raise ValueError naming the file."""
    path = tmp_path / 'bad.dat'
    path.write_text('# Kind: liquid\n' + body + '\n')
    with pytest.raises(ValueError, match=msg) as err:
        LiquidIronTable.from_file(path)
    assert 'bad.dat' in str(err.value)


@pytest.mark.parametrize('drop', ['s', 'header', 'T_K'])
def test_an_npz_without_a_key_is_refused_with_its_name(tmp_path, drop):
    """An npz cut without a field, the axes or the header raises ValueError naming the key."""
    data = dict(np.load(CUT))
    del data[drop]
    path = tmp_path / 'cut.npz'
    np.savez(path, **data)
    with pytest.raises(ValueError, match=f"missing.*'{drop}'"):
        LiquidIronTable.from_file(path)
