"""Unit tests for ``aragog.mesh.derive_core_density_from_mesh``.

The helper is the Aragog side of the PROTEUS-coupled core-density
echo-back: it reads the CMB radius from the first row of a Zalmoxis
mantle mesh file and returns the self-consistent average core density
:math:`\\rho_\\mathrm{core} = M_\\mathrm{core} / (\\tfrac{4}{3}\\pi R_\\mathrm{cmb}^3)`.

The PROTEUS wrapper imports this helper to override the cached
``hf_row['core_density']`` whenever a Zalmoxis mesh file is present at
solve entry, mirroring what SPIDER's ``-rho_core`` re-derivation does
in ``proteus/interior_energetics/spider.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aragog.mesh import derive_core_density_from_mesh


def _write_mesh(target: Path, rows: list[tuple[float, ...]]) -> Path:
    """Write a 5-column ascending-r mantle mesh file.

    ``target`` may be either a directory (file is named
    ``zalmoxis_output.dat`` inside it) or a target file path.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_dir():
        f = target / 'zalmoxis_output.dat'
    else:
        f = target
    with f.open('w') as fh:
        for row in rows:
            fh.write(' '.join(f'{v:.17e}' for v in row) + '\n')
    return f


@pytest.mark.unit
def test_recovers_M_over_volume_for_earth(tmp_path):
    """At Earth-like (M_core, R_cmb), recover Stacey 2008 core density.

    Stacey (2008) Earth-model core density at the inner-most mantle
    radius is :math:`\\sim 9900` kg/m^3 averaged over the outer core.
    Inputs M_core = 1.94e24 kg, R_cmb = 3.480e6 m give
    :math:`\\rho = M / (\\tfrac{4}{3} \\pi R^3) = 11008` kg/m^3, which
    is the canonical bulk-core density used in PROTEUS coupled runs.
    """
    R_cmb = 3.480e6
    M_core = 1.94e24
    expected = M_core / (4.0 / 3.0 * math.pi * R_cmb**3)

    f = _write_mesh(tmp_path, [(R_cmb, 1.35e11, 9904.0, 10.7, 4500.0)])
    result = derive_core_density_from_mesh(str(f), M_core)

    assert result == pytest.approx(expected, rel=1e-12)
    assert 9000.0 < result < 12000.0  # plausible Earth core density


@pytest.mark.unit
def test_uses_only_first_row_r_column(tmp_path):
    """Helper reads only the first row's first column; later rows ignored.

    The Zalmoxis mantle mesh is ascending in r, so the first row is the
    bottom of the mantle (the CMB). A regression that read the last row
    would silently use ``R_int`` instead of ``R_cmb`` and produce an
    underestimate of :math:`\\rho_\\mathrm{core}` by a factor of
    :math:`(R_\\mathrm{int} / R_\\mathrm{cmb})^3 \\sim 6` for Earth.
    """
    rows = [
        (3.480e6, 1.35e11, 9904.0, 10.7, 4500.0),  # CMB
        (4.500e6, 1.10e11, 5000.0, 10.0, 4200.0),
        (6.371e6, 1.00e5, 3300.0, 9.81, 1800.0),  # surface
    ]
    f = _write_mesh(tmp_path, rows)
    M_core = 1.94e24

    result = derive_core_density_from_mesh(str(f), M_core)

    expected = M_core / (4.0 / 3.0 * math.pi * (3.480e6) ** 3)
    bad_if_last_row = M_core / (4.0 / 3.0 * math.pi * (6.371e6) ** 3)
    assert result == pytest.approx(expected, rel=1e-12)
    assert result != pytest.approx(bad_if_last_row, rel=1e-3)


@pytest.mark.unit
def test_density_scales_inversely_with_volume(tmp_path):
    """Doubling R_cmb at fixed M_core drops density by exactly 8.

    Discriminator against an off-by-one exponent: if the helper used
    R^2 (area) the ratio would be 4; if R (linear) it would be 2.
    """
    M_core = 1.94e24
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    f1 = _write_mesh(tmp_path / 'a', [(3.480e6, 1e11, 9904.0, 10.0, 4500.0)])
    f2 = _write_mesh(tmp_path / 'b', [(6.960e6, 1e11, 4952.0, 10.0, 4500.0)])

    rho1 = derive_core_density_from_mesh(str(f1), M_core)
    rho2 = derive_core_density_from_mesh(str(f2), M_core)

    assert rho1 / rho2 == pytest.approx(8.0, rel=1e-12)


@pytest.mark.unit
def test_density_scales_linearly_with_M_core(tmp_path):
    """Doubling M_core at fixed R_cmb doubles density."""
    R_cmb = 3.480e6
    f = _write_mesh(tmp_path, [(R_cmb, 1e11, 9904.0, 10.0, 4500.0)])

    rho1 = derive_core_density_from_mesh(str(f), 1.0e24)
    rho2 = derive_core_density_from_mesh(str(f), 2.0e24)

    assert rho2 / rho1 == pytest.approx(2.0, rel=1e-12)


@pytest.mark.unit
def test_handles_tab_and_multispace_separators(tmp_path):
    """Whitespace tolerance: tabs, multiple spaces, leading/trailing space."""
    R_cmb = 3.480e6
    M_core = 1.94e24
    f = tmp_path / 'wonky.dat'
    f.write_text(f'  {R_cmb:.17e}\t1.35e11   9904.0  10.7    4500.0\n')

    result = derive_core_density_from_mesh(str(f), M_core)
    expected = M_core / (4.0 / 3.0 * math.pi * R_cmb**3)
    assert result == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_rejects_zero_M_core(tmp_path):
    """M_core = 0 is unphysical and must raise ValueError."""
    f = _write_mesh(tmp_path, [(3.480e6, 1e11, 9904.0, 10.0, 4500.0)])
    with pytest.raises(ValueError, match='M_core must be positive'):
        derive_core_density_from_mesh(str(f), 0.0)


@pytest.mark.unit
def test_rejects_negative_M_core(tmp_path):
    """Negative M_core is unphysical and must raise ValueError."""
    f = _write_mesh(tmp_path, [(3.480e6, 1e11, 9904.0, 10.0, 4500.0)])
    with pytest.raises(ValueError, match='M_core must be positive'):
        derive_core_density_from_mesh(str(f), -1.94e24)


@pytest.mark.unit
def test_rejects_missing_file():
    """Missing mesh file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        derive_core_density_from_mesh('/nonexistent/zalmoxis_output.dat', 1.94e24)


@pytest.mark.unit
def test_rejects_empty_file(tmp_path):
    """Empty mesh file must raise ValueError, not return NaN."""
    f = tmp_path / 'empty.dat'
    f.write_text('')
    with pytest.raises(ValueError, match='empty'):
        derive_core_density_from_mesh(str(f), 1.94e24)


@pytest.mark.unit
def test_rejects_unparseable_first_row(tmp_path):
    """Garbled first row must raise ValueError with a helpful message."""
    f = tmp_path / 'garbled.dat'
    f.write_text('not a number\n')
    with pytest.raises(ValueError, match='Could not parse'):
        derive_core_density_from_mesh(str(f), 1.94e24)


@pytest.mark.unit
def test_rejects_non_positive_R_cmb(tmp_path):
    """Zero or negative R_cmb is unphysical and must raise ValueError."""
    f = _write_mesh(tmp_path, [(0.0, 1e11, 9904.0, 10.0, 4500.0)])
    with pytest.raises(ValueError, match='R_cmb must be positive'):
        derive_core_density_from_mesh(str(f), 1.94e24)

    neg = tmp_path / 'neg'
    neg.mkdir()
    f2 = _write_mesh(neg, [(-1.0e6, 1e11, 9904.0, 10.0, 4500.0)])
    with pytest.raises(ValueError, match='R_cmb must be positive'):
        derive_core_density_from_mesh(str(f2), 1.94e24)


@pytest.mark.unit
def test_super_earth_5me_plausible(tmp_path):
    """Self-consistency at super-Earth scale: 5 M_E core is denser than Earth core.

    For a 5 M_E planet with core_frac = 0.32 by mass and R_cmb at the
    PALEOS-equilibrium scale (~5.2e6 m), the bulk core density is
    :math:`\\sim 13000` to :math:`18000` kg/m^3 (denser than Earth's
    outer core because of compression). This test pins the order of
    magnitude to catch unit-conversion bugs that would push it to
    :math:`10^2` or :math:`10^6`.
    """
    R_cmb = 5.20e6
    f = _write_mesh(tmp_path, [(R_cmb, 4.5e11, 13000.0, 18.0, 6500.0)])
    M_core = 0.32 * 5.0 * 5.972e24  # 0.32 * 5 M_E

    result = derive_core_density_from_mesh(str(f), M_core)

    assert 12000.0 < result < 18000.0
    expected = M_core / (4.0 / 3.0 * math.pi * R_cmb**3)
    assert result == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_returns_python_float_not_numpy(tmp_path):
    """Return type must be plain ``float`` for downstream JSON-friendliness.

    The PROTEUS wrapper writes the returned value back into ``hf_row``,
    which is serialised through helpfile CSV. A returned ``np.float64``
    can survive but a returned ``np.ndarray`` of shape ``()`` would
    break the pandas helpfile writer.
    """
    f = _write_mesh(tmp_path, [(3.480e6, 1e11, 9904.0, 10.0, 4500.0)])
    result = derive_core_density_from_mesh(str(f), 1.94e24)
    assert isinstance(result, float)
    assert not isinstance(result, np.ndarray)
