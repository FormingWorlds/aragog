"""Unit tests for the per-solve core-temperature excursion measure.

These tests build a bare ``EntropySolver`` via ``__new__`` and set only
the attributes the excursion path reads, so they run without the FWL data
tables. A fake entropy EOS returns the entropy unchanged, which makes the
core temperature equal to the bottom staggered entropy and keeps the
expected excursion trivial to compute by hand.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aragog.solver.entropy_solver import EntropySolver

pytestmark = pytest.mark.unit


class _FakeEOS:
    """Entropy EOS stub whose temperature equals the entropy passed in."""

    def temperature(self, pressure, entropy):
        return np.asarray(entropy, dtype=float)


class _FakeSol:
    """Solver-result stub carrying a 2-D state array ``y``."""

    def __init__(self, y):
        self.y = np.asarray(y, dtype=float)


def _make_solver(*, n_stag: int = 4, limit=None, core_bc: str = 'quasi_steady'):
    solver = EntropySolver.__new__(EntropySolver)
    solver._core_bc = core_bc
    solver._n_stag = n_stag
    solver._P_stag_flat = np.linspace(1.0, 2.0, n_stag)
    solver.entropy_eos = _FakeEOS()
    solver._tcore_change_limit = limit
    return solver


def _quasi_steady_grid(bottom_row):
    """Build a (n_stag, n_col) grid whose bottom row is ``bottom_row``."""
    bottom_row = np.asarray(bottom_row, dtype=float)
    n_col = bottom_row.size
    y = np.tile(np.array([[2000.0], [1900.0], [1800.0], [1700.0]]), (1, n_col))
    y[0, :] = bottom_row
    return _FakeSol(y)


def test_measures_max_change_smooth():
    solver = _make_solver(limit=None)
    sol = _quasi_steady_grid([1000.0, 1010.0, 1020.0, 1030.0])
    change_max, exceeded = solver._core_temperature_excursion(sol)
    assert change_max == pytest.approx(30.0)
    assert exceeded is False


def test_measures_max_change_spiky():
    solver = _make_solver(limit=None)
    sol = _quasi_steady_grid([1000.0, 1000.0, 5000.0, 1000.0])
    change_max, exceeded = solver._core_temperature_excursion(sol)
    assert change_max == pytest.approx(4000.0)
    assert exceeded is False


def test_flag_set_when_change_exceeds_limit():
    solver = _make_solver(limit=1000.0)
    sol = _quasi_steady_grid([1000.0, 1000.0, 5000.0, 1000.0])
    change_max, exceeded = solver._core_temperature_excursion(sol)
    assert change_max == pytest.approx(4000.0)
    assert exceeded is True


def test_flag_clear_when_change_below_limit():
    solver = _make_solver(limit=5000.0)
    sol = _quasi_steady_grid([1000.0, 1000.0, 5000.0, 1000.0])
    change_max, exceeded = solver._core_temperature_excursion(sol)
    assert change_max == pytest.approx(4000.0)
    assert exceeded is False


def test_smooth_grid_stays_below_limit():
    solver = _make_solver(limit=1000.0)
    sol = _quasi_steady_grid([1000.0, 1010.0, 1020.0, 1030.0])
    _, exceeded = solver._core_temperature_excursion(sol)
    assert exceeded is False


def test_empty_grid_returns_zero():
    solver = _make_solver(limit=1000.0)
    sol = _FakeSol(np.empty((4, 0)))
    change_max, exceeded = solver._core_temperature_excursion(sol)
    assert change_max == 0.0
    assert exceeded is False


def test_helper_bower_reads_core_state():
    solver = _make_solver(n_stag=4, core_bc='bower2018')
    y_col = np.array([2000.0, 1900.0, 1800.0, 1700.0, 2500.0])
    assert solver._core_temperature_from_column(y_col) == pytest.approx(2500.0)


def test_helper_energy_balance_uses_bottom_entropy():
    solver = _make_solver(n_stag=4, core_bc='energy_balance')
    y_col = np.array([1234.0, 1300.0, 1400.0, 1500.0, 0.0])
    assert solver._core_temperature_from_column(y_col) == pytest.approx(1234.0)


def test_helper_gradient_uses_reconstructed_bottom_entropy():
    solver = _make_solver(n_stag=4, core_bc='gradient')
    # gradient reconstructs the staggered entropy from the basic-node
    # state and a surface scalar, then maps it through the EOS. The fake
    # EOS returns entropy unchanged, so T_core is the reconstructed
    # bottom staggered entropy.
    solver._reconstruct_entropy = lambda dsdr, s_surf: (
        np.array([1357.0, 1400.0, 1500.0, 1600.0]),
        None,
    )
    y_col = np.array([2000.0, 1900.0, 1800.0, 1700.0, 1600.0, 100.0])
    assert solver._core_temperature_from_column(y_col) == pytest.approx(1357.0)


def test_helper_const_eos_uses_analytic_formula():
    solver = _make_solver(n_stag=4, core_bc='quasi_steady')
    # With no entropy EOS the helper uses the analytic const-property
    # relation T = T_ref * exp((S - S_ref) / Cp). The chosen references
    # make the exponent zero for the bottom cell, so T_core is T_ref.
    solver.entropy_eos = None
    solver.parameters = SimpleNamespace(
        phase_mixed=SimpleNamespace(const_T_ref=300.0, const_S_ref=1000.0, const_Cp=1000.0)
    )
    y_col = np.array([1000.0, 1100.0, 1200.0, 1300.0])
    assert solver._core_temperature_from_column(y_col) == pytest.approx(300.0)
