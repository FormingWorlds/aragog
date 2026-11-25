"""Distances of lookup points from the valid window of the P-S tables, and the settings of the
domain check."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# 3 pressure columns, 10 entropy nodes 100 J/kg/K apart
P = np.array([0.0, 1.0e9, 2.0e9])
S = 1000.0 + 100.0 * np.arange(10)


def _mask(lo_idx, hi_idx, n_s=S.size):
    """Mask with the valid run from row lo_idx[k] to hi_idx[k] in column k."""
    m = np.zeros((n_s, len(lo_idx)))
    for k, (a, b) in enumerate(zip(lo_idx, hi_idx)):
        m[a : b + 1, k] = 1.0
    return m


def _window(lo_idx=(2, 3, 4), hi_idx=(6, 7, 8)):
    from aragog.eos.domain import PhaseWindow

    return PhaseWindow.from_mask(P, S, _mask(lo_idx, hi_idx))


class _StubEOS:
    """Phase boundaries flat in P: solidus 1300, liquidus 1500 J/kg/K."""

    S_SOL, S_LIQ = 1300.0, 1500.0

    def solidus_entropy(self, P):
        return np.full(np.shape(P), self.S_SOL)

    def liquidus_entropy(self, P):
        return np.full(np.shape(P), self.S_LIQ)

    def _melt_fraction(self, P, S):
        return np.clip((np.asarray(S) - self.S_SOL) / (self.S_LIQ - self.S_SOL), 0.0, 1.0)


def _domain(melt=((5, 5, 5), (9, 9, 9)), solid=((0, 0, 0), (3, 3, 3))):
    """Melt window [1500, 1900], solid window [1000, 1300] in every column by default."""
    from aragog.eos.domain import PhaseWindow, TableDomain

    return TableDomain(
        melt=PhaseWindow.from_mask(P, S, _mask(*melt)),
        solid=PhaseWindow.from_mask(P, S, _mask(*solid)),
    )


def test_distance_is_zero_inside_and_counts_table_cells_outside():
    """Inside the window and on its ends the distance is 0; outside it is the entropy gap over
    the S spacing of the table, with the window ends linear in P between 2 columns."""
    w = _window()
    d = w.distance(np.zeros(5), np.array([1400.0, 1200.0, 1600.0, 1650.0, 1000.0]))
    np.testing.assert_allclose(d, [0.0, 0.0, 0.0, 0.5, 2.0], rtol=0, atol=1e-12)
    # halfway between column 0 [1200, 1600] and column 1 [1300, 1700]: [1250, 1650]
    d_mid = w.distance(np.full(2, 0.5e9), np.array([1050.0, 1850.0]))
    np.testing.assert_allclose(d_mid, [2.0, 2.0], rtol=0, atol=1e-12)
    # the nearest column would give 1.5 or 2.5 cells
    assert not np.any(np.isclose(d_mid, 1.5) | np.isclose(d_mid, 2.5))


@settings(max_examples=60, deadline=None)
@given(
    cells=st.floats(min_value=0.0, max_value=40.0, allow_nan=False),
    above=st.booleans(),
    column=st.integers(min_value=0, max_value=2),
)
def test_distance_grows_by_one_per_cell_outside(cells, above, column):
    """At a pressure node the distance is the number of S cells beyond the nearer window end,
    on either side, and 1 more cell outside adds 1."""
    w = _window()
    lo, hi = S[(2, 3, 4)[column]], S[(6, 7, 8)[column]]
    s = hi + 100.0 * cells if above else lo - 100.0 * cells
    step = 100.0 if above else -100.0
    d0, d1 = w.distance(np.array([P[column]] * 2), np.array([s, s + step]))
    assert d0 == pytest.approx(cells, abs=1e-9)
    assert d1 - d0 == pytest.approx(1.0, abs=1e-9)


@settings(max_examples=40, deadline=None)
@given(
    p=st.floats(min_value=0.0, max_value=2.0e9, allow_nan=False),
    frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_distance_is_zero_everywhere_between_the_window_ends(p, frac):
    """Every entropy between the 2 window ends at its pressure lies inside: distance 0."""
    w = _window()
    lo, hi = w.ends(np.array([p]))
    s = lo + frac * (hi - lo)
    assert w.distance(np.array([p]), s)[0] == 0.0
    assert w.distance(np.array([p]), hi + 50.0)[0] == pytest.approx(0.5, abs=1e-9)


def test_distance_outside_the_pressure_axis_takes_the_end_column():
    """A pressure below the first or above the last column takes that column's window, as the
    table lookup clamps P to the axis."""
    w = _window()
    # end columns [1200, 1600] and [1400, 1800]; column 1 would give 3 and 2 cells
    d = w.distance(np.array([-1.0e8, 5.0e9]), np.array([1000.0, 1900.0]))
    np.testing.assert_allclose(d, [2.0, 1.0], rtol=0, atol=1e-12)


def test_a_non_finite_point_has_a_nan_distance():
    """A NaN or infinite entropy, or a NaN pressure, gives a NaN distance, not 0."""
    w = _window()
    d = w.distance(
        np.array([0.0, 0.0, np.nan, 0.0]), np.array([np.nan, np.inf, 1400.0, 1400.0])
    )
    assert np.isnan(d[:3]).all()
    assert d[3] == 0.0


@pytest.mark.parametrize(
    'mask, match',
    [
        (np.ones((S.size, 2)), 'shape'),
        (_mask((2, 3, 4), (6, 7, 8))[:-1], 'shape'),
        (_mask((2, 3, 4), (6, 7, 8)) * 0.5, '0 and 1'),
        (np.zeros((S.size, 3)), 'no valid'),
    ],
)
def test_from_mask_refuses_a_malformed_mask(mask, match):
    """A mask of the wrong shape, with values other than 0 and 1, or with a column that holds no
    valid node is refused."""
    from aragog.eos.domain import PhaseWindow

    with pytest.raises(ValueError, match=match):
        PhaseWindow.from_mask(P, S, mask)


def test_from_mask_refuses_a_gap_in_a_column():
    """The valid nodes of a column must be one run: a gap is refused and named by its column."""
    from aragog.eos.domain import PhaseWindow

    m = _mask((2, 3, 4), (6, 7, 8))
    m[5, 1] = 0.0
    with pytest.raises(ValueError, match='column 1'):
        PhaseWindow.from_mask(P, S, m)
    m[5, 1] = 1.0
    assert PhaseWindow.from_mask(P, S, m).hi[1] == 1700.0


def test_lookup_points_follow_the_phase_of_each_point():
    """A melt point is judged on the melt table at its entropy, a solid point on the solid table,
    and a mush point at the 2 phase-boundary entropies, which is where the EOS reads it."""
    dom = _domain()
    eos = _StubEOS()
    s = np.array([1800.0, 2100.0, 1100.0, 1400.0, 1500.0, 1300.0])
    d, melt_table, s_look = dom.lookup_distances(eos, np.zeros(s.size), s)
    # melt inside; melt 2 cells above; solid inside; mush at (1500 melt, 1300 solid): inside;
    # on the liquidus: melt only; on the solidus: solid only
    np.testing.assert_allclose(d, [0.0, 2.0, 0.0, 0.0, 0.0, 0.0], rtol=0, atol=1e-12)
    assert melt_table.tolist()[:3] == [True, True, False]
    assert s_look[3] in (1500.0, 1300.0) and s_look[1] == 2100.0
    # the mush point at its own entropy would be 1 cell below the melt window
    from aragog.eos.domain import PhaseWindow

    assert dom.melt.distance(np.zeros(1), np.array([1400.0]))[0] == pytest.approx(1.0)
    assert isinstance(dom.melt, PhaseWindow)


def test_a_mush_point_takes_the_worse_of_its_two_lookups():
    """With the solid window ending 2 cells below the solidus and the melt window holding the
    liquidus, a mush point is 2 cells outside, on the solid table at the solidus entropy."""
    dom = _domain(solid=((0, 0, 0), (1, 1, 1)))
    d, melt_table, s_look = dom.lookup_distances(
        _StubEOS(), np.zeros(2), np.array([1400.0, 1450.0])
    )
    np.testing.assert_allclose(d, [2.0, 2.0], rtol=0, atol=1e-12)
    assert not melt_table.any()
    np.testing.assert_array_equal(s_look, [1300.0, 1300.0])


def test_a_nan_entropy_is_outside_at_any_tolerance():
    """A NaN entropy point has a NaN distance, so it never passes a ``distance <= tol`` test."""
    d, _, _ = _domain().lookup_distances(_StubEOS(), np.zeros(2), np.array([np.nan, 1800.0]))
    assert np.isnan(d[0]) and d[1] == 0.0
    assert not (d[0] <= 1.0e300)


def _write_mask(path, m):
    np.savetxt(path, m, fmt='%d', header='validity mask: rows entropy nodes, columns pressure')


def _tables(phase):
    return {'P': P, 'S': S}


def test_from_dir_reads_both_masks(tmp_path):
    """The masks next to the tables give the window ends of each phase, rows the entropy nodes
    and columns the pressure nodes of the phase's temperature table."""
    from aragog.eos.domain import TableDomain

    _write_mask(tmp_path / 'valid_mask_melt.dat', _mask((5, 5, 5), (9, 9, 9)))
    _write_mask(tmp_path / 'valid_mask_solid.dat', _mask((0, 1, 2), (3, 4, 5)))
    dom = TableDomain.from_dir(tmp_path, {'melt': _tables('melt'), 'solid': _tables('solid')})
    np.testing.assert_array_equal(dom.solid.lo, [1000.0, 1100.0, 1200.0])
    np.testing.assert_array_equal(dom.solid.hi, [1300.0, 1400.0, 1500.0])
    np.testing.assert_array_equal(dom.melt.lo, [1500.0] * 3)
    assert dom.melt.dS == pytest.approx(100.0)


def test_from_dir_without_masks_turns_the_check_off(tmp_path, caplog):
    """Tables without mask files give no domain and 1 info line; a single mask file is refused."""
    from aragog.eos.domain import TableDomain

    tables = {'melt': _tables('melt'), 'solid': _tables('solid')}
    with caplog.at_level(logging.INFO, logger='fwl.aragog'):
        assert TableDomain.from_dir(tmp_path, tables) is None
    lines = [r for r in caplog.records if 'valid_mask' in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    _write_mask(tmp_path / 'valid_mask_melt.dat', _mask((5, 5, 5), (9, 9, 9)))
    with pytest.raises(FileNotFoundError, match='valid_mask_solid'):
        TableDomain.from_dir(tmp_path, tables)


def test_settings_default_to_warn_at_5_cells():
    """The check warns by default at 5 table S cells, in the parser and in the config class."""
    from aragog.config.solver import SolverConfig
    from aragog.parser import _SolverParameters

    for cls in (_SolverParameters, SolverConfig):
        c = cls(start_time=0.0, end_time=1.0)
        assert c.eos_domain_check == 'warn'
        assert c.eos_domain_tolerance_cells == 5.0
        for mode in ('off', 'warn', 'raise'):
            assert (
                cls(start_time=0.0, end_time=1.0, eos_domain_check=mode).eos_domain_check
                == mode
            )


@settings(max_examples=40, deadline=None)
@given(mode=st.one_of(st.text(max_size=8), st.none(), st.integers(), st.booleans()))
def test_settings_refuse_any_other_mode(mode):
    """Only 'off', 'warn' and 'raise' are accepted as the mode."""
    from aragog.config.solver import SolverConfig
    from aragog.parser import _SolverParameters

    if mode in ('off', 'warn', 'raise'):
        return
    for cls in (_SolverParameters, SolverConfig):
        with pytest.raises(ValueError, match='eos_domain_check must be one of'):
            cls(start_time=0.0, end_time=1.0, eos_domain_check=mode)


@pytest.mark.parametrize('bad', [0.0, -1.0, math.nan, math.inf, True, '5', None])
def test_settings_refuse_a_tolerance_that_is_not_finite_and_positive(bad):
    """The tolerance must be a finite number above 0; a bool or a string is refused."""
    from aragog.config.solver import SolverConfig
    from aragog.parser import _SolverParameters

    for cls in (_SolverParameters, SolverConfig):
        with pytest.raises((TypeError, ValueError), match='eos_domain_tolerance_cells'):
            cls(start_time=0.0, end_time=1.0, eos_domain_tolerance_cells=bad)
    assert (
        _SolverParameters(
            start_time=0.0, end_time=1.0, eos_domain_tolerance_cells=3
        ).eos_domain_tolerance_cells
        == 3.0
    )
