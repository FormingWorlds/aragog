"""``tools/verification/make_core_liquid_cut.py``: the cut of a liquid iron side table."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from aragog.core.iron_table import FIELDS
from tests.conftest import PALEOS_CUT as CUT

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location(
    'make_core_liquid_cut',
    Path(__file__).resolve().parents[1] / 'tools' / 'verification' / 'make_core_liquid_cut.py',
)
cut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cut)


def _source(path, rows=slice(0, 8), cols=slice(0, 6), edit=None):
    """A Zalmoxis-format liquid table from a corner of the 3 TPa cut; ``edit`` changes lines."""
    data = np.load(CUT)
    p, t = data['P_Pa'][rows], data['T_K'][cols]
    lines = [str(data['header'])]
    for i, pv in enumerate(p):
        for j, tv in enumerate(t):
            vals = ' '.join(f'{data[f][rows][i, cols][j]:.8e}' for f in FIELDS)
            lines.append(f'{pv:.8e} {tv:.8e} {vals} liquid')
    if edit is not None:
        lines = edit(lines)
    path.write_text('\n'.join(lines) + '\n')
    return path, p, t


def _run(tmp_path, src, *flags):
    """Cut the whole box of ``src``; return the npz and the provenance."""
    cut.main(['x', str(src), str(tmp_path), '1', '1e15', '1', '1e6', 'c', *flags])
    return np.load(tmp_path / 'c.npz'), json.loads((tmp_path / 'c_provenance.json').read_text())


def test_a_cut_holds_the_source_values(tmp_path):
    """The cut of a box that covers the table holds every source value and its md5."""
    src, p, t = _source(tmp_path / 's.dat')
    npz, prov = _run(tmp_path, src)
    np.testing.assert_array_equal(npz['P_Pa'], p)
    assert npz['rho'].shape == (p.size, t.size)
    assert npz['s'][2, 3] == pytest.approx(np.load(CUT)['s'][2, 3], rel=1e-8)
    assert prov['nan_edge_rows'] == 0 and prov['cut_P_Pa'][2] == p.size


def _nan_at(k):
    """Set the density of data line ``k`` (after the header line) to NaN."""

    def edit(lines):
        f = lines[k + 1].split()
        f[2] = 'nan'
        lines[k + 1] = ' '.join(f)
        return lines

    return edit


def test_nan_edge_rows_need_the_flag_and_nan_inside_is_refused(tmp_path):
    """A NaN cell in the top pressure row stops the cut without --keep-nan-edges and is kept
    with it; a NaN cell in a row between finite rows stops it in both cases."""
    src, p, t = _source(tmp_path / 's.dat', edit=_nan_at(7 * 6 + 2))
    with pytest.raises(ValueError, match='non-finite cell in the box'):
        _run(tmp_path, src)
    npz, prov = _run(tmp_path, src, '--keep-nan-edges')
    assert prov['nan_edge_rows'] == 1 and npz['P_Pa'].size == p.size
    inner, _, _ = _source(tmp_path / 'i.dat', edit=_nan_at(3 * 6 + 2))
    for flags in ((), ('--keep-nan-edges',)):
        with pytest.raises(ValueError, match='non-finite cell'):
            _run(tmp_path, inner, *flags)


@pytest.mark.parametrize(
    'edit, match',
    [
        (lambda lines: lines[:-1], 'not a full rectangle'),
        (lambda lines: [*lines[:-1], lines[-1].replace(' liquid', ' solid')], '10 columns'),
        (lambda lines: [*lines[:-1], lines[-1].rsplit(' ', 2)[0] + ' liquid'], '10 columns'),
        (lambda lines: [*lines[:-2], lines[1], lines[-1]], 'not a full rectangle|P x T grid'),
    ],
    ids=['row missing', 'solid phase word', 'column missing', 'row repeated'],
)
def test_a_source_that_is_not_a_liquid_grid_is_refused(tmp_path, edit, match):
    """A row missing, a non-liquid phase word, a short row or a repeated row is refused."""
    src, _, _ = _source(tmp_path / 's.dat', edit=edit)
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, src)
