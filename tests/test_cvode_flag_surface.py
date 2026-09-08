"""Tests for the distinct CVODE return-flag surfaced by ``_solve_cvode``.

``result.status`` stays scipy-compatible (0 success, -1 failure), so a
single failure code cannot tell why CVODE stopped. ``_solve_cvode`` also
records the raw CVODE flag as ``result.cvode_flag`` and its enum name as
``result.cvode_flag_name`` so a caller can distinguish CV_TOO_MUCH_WORK
(step budget exhausted) from CV_CONV_FAILURE (nonlinear-solver failure).
"""

from __future__ import annotations

import numpy as np
import pytest

import aragog.solver.entropy_solver as es
from aragog.solver.entropy_solver import EntropySolver, _cvode_flag_name

pytestmark = pytest.mark.unit


def test_flag_name_maps_known_flags():
    assert _cvode_flag_name(0) == 'SUCCESS'
    assert _cvode_flag_name(-1) == 'TOO_MUCH_WORK'
    assert _cvode_flag_name(-4) == 'CONV_FAILURE'


def test_flag_name_falls_back_on_unknown_flag():
    assert _cvode_flag_name(-999) == 'FLAG_-999'


class _FlaggingCVODE:
    """Fake CVODE whose solve returns a configurable return flag.

    Lets a test drive ``_solve_cvode`` down its flag-handling branch with
    a chosen flag without running a real integration.
    """

    def __init__(self, flag: int):
        self._flag = flag

    def __call__(self, rhs, **opts):
        self._rhs = rhs
        return self

    def solve(self, tspan, y0):
        flag = self._flag

        class _V:
            t = np.asarray(tspan, dtype=float)
            y = np.tile(np.asarray(y0, dtype=float).reshape(-1, 1), (1, len(t)))

        class _Sol:
            def __init__(self):
                self.flag = flag
                self.message = f'stopped with flag {flag}'
                self.roots = None
                self.values = _V()

        return _Sol()

    def get_info(self):
        return {}


def _run_with_flag(flag: int):
    """Drive the real ``_solve_cvode`` with a fake returning ``flag``.

    Returns the ``OptimizeResult`` the real method built.
    """
    real_cvode = es._scikits_cvode
    es._scikits_cvode = _FlaggingCVODE(flag)
    try:
        solver = EntropySolver.__new__(EntropySolver)
        solver._core_bc = 'quasi_steady'
        solver._cvode_output_points = 4
        solver._max_steps = 100000

        y0 = np.array([2.0, 0.0])
        return solver._solve_cvode(
            start_time=0.0,
            end_time=1.0,
            y0=y0,
            atol=1.0e-8,
            rtol=1.0e-10,
            max_step=np.inf,
            rhs=lambda t, y: y,
        )
    finally:
        es._scikits_cvode = real_cvode


def test_too_much_work_flag_is_distinct_from_conv_failure():
    too_much_work = _run_with_flag(-1)
    conv_failure = _run_with_flag(-4)

    # scipy-compatible status collapses both to -1.
    assert too_much_work.status == -1
    assert conv_failure.status == -1

    # The raw flag keeps them distinct.
    assert too_much_work.cvode_flag == -1
    assert too_much_work.cvode_flag_name == 'TOO_MUCH_WORK'
    assert conv_failure.cvode_flag == -4
    assert conv_failure.cvode_flag_name == 'CONV_FAILURE'


def test_success_flag_surface():
    ok = _run_with_flag(0)
    assert ok.status == 0
    assert ok.cvode_flag == 0
    assert ok.cvode_flag_name == 'SUCCESS'


def test_zero_span_reports_success_sentinel():
    real_cvode = es._scikits_cvode
    es._scikits_cvode = _FlaggingCVODE(-1)  # must never be called
    try:
        solver = EntropySolver.__new__(EntropySolver)
        solver._core_bc = 'quasi_steady'
        solver._cvode_output_points = 4
        solver._max_steps = 100000

        result = solver._solve_cvode(
            start_time=0.0,
            end_time=0.0,
            y0=np.array([2.0, 0.0]),
            atol=1.0e-8,
            rtol=1.0e-10,
            max_step=np.inf,
            rhs=lambda t, y: y,
        )
    finally:
        es._scikits_cvode = real_cvode

    assert result.status == 0
    assert result.cvode_flag == 0
    assert result.cvode_flag_name == 'SUCCESS'
