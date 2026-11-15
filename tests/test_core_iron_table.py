"""Tests of ``aragog.core.iron_table.LiquidIronTable``, the PALEOS liquid iron table reader."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.core.iron_table import FIELDS, LiquidIronTable

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

CUT = pathlib.Path(__file__).parent / 'validation' / 'data' / 'paleos_iron_liquid_cut_3tpa.npz'


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
    """A solid or unified table is refused: the core isentrope needs the liquid branch."""
    data = np.load(CUT)
    with pytest.raises(ValueError, match="kind 'solid'"):
        LiquidIronTable.from_file(_write_dat(tmp_path / 's.dat', data, kind='solid'))
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
    i=st.integers(0, 5),
    j=st.integers(0, 5),
)
def test_a_non_finite_cell_is_refused(field, bad, i, j):
    """Any NaN or infinite cell in any field is refused, wherever it lies."""
    p, t, fields = _arrays()
    fields[field][i, j] = bad
    with pytest.raises(ValueError, match=f'non-finite cell in {field}'):
        LiquidIronTable(p, t, fields, header='# Kind: liquid')


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
    nodes up to 2 TPa, to the error of the grid (7.5e-6 at most measured there; the last
    nodes before the 3 TPa edge of the cut reach 1.5e-3, so they are left out). A missing
    ln(10) or a wrong axis gives an error of order 1."""
    data = np.load(CUT)
    keep = data['P_Pa'][2:-2] <= 2e12
    pp, tt = np.meshgrid(data['P_Pa'][2:-2][keep], data['T_K'][2:-2], indexing='ij')
    maxwell = table.ds_dP(pp, tt) / (-table.value('alpha', pp, tt) / table.value('rho', pp, tt))
    np.testing.assert_allclose(maxwell, 1.0, atol=3e-5)
    slope = -table.ds_dP(pp, tt) / table.ds_dT(pp, tt)
    np.testing.assert_allclose(
        slope / (table.value('nabla_ad', pp, tt) * tt / pp), 1.0, atol=3e-5
    )
