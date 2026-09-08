"""Tests for the configurable CVODE ``max_steps`` solver option.

Two surfaces are covered:

* The config surface. ``max_steps`` must exist on both the live
  runtime dataclass ``aragog.parser._SolverParameters`` and its attrs
  schema mirror ``aragog.config.solver.SolverConfig``, default to
  100000, and reject non-integer or ``< 1`` values on both.
* The wiring. The configured value must reach the ``max_steps`` key of
  the options dict built inside ``EntropySolver._solve_cvode`` on both
  the finite-difference-Jacobian path and the analytic-Jacobian
  (option Z) path, since both build the same options dict.
"""

from __future__ import annotations

import numpy as np
import pytest

import aragog.solver.entropy_solver as es
from aragog.solver.entropy_solver import EntropySolver

pytestmark = pytest.mark.unit


def _solver_parameters(**overrides):
    from aragog.parser import _SolverParameters

    kwargs = dict(
        start_time=0.0,
        end_time=1.0,
        atol=1.0e-6,
        rtol=1.0e-6,
        tsurf_poststep_change=30.0,
    )
    kwargs.update(overrides)
    return _SolverParameters(**kwargs)


def _solver_config(**overrides):
    from aragog.config.solver import SolverConfig

    kwargs = dict(start_time=0.0, end_time=1.0, atol=1.0e-6, rtol=1.0e-6)
    kwargs.update(overrides)
    return SolverConfig(**kwargs)


def test_solver_parameters_default_max_steps():
    assert _solver_parameters().max_steps == 100000


def test_solver_parameters_accepts_custom_max_steps():
    assert _solver_parameters(max_steps=12345).max_steps == 12345


def test_solver_parameters_rejects_below_one():
    with pytest.raises(ValueError):
        _solver_parameters(max_steps=0)


def test_solver_parameters_rejects_non_int():
    with pytest.raises(TypeError):
        _solver_parameters(max_steps=1.5)


def test_solver_config_default_max_steps():
    assert _solver_config().max_steps == 100000


def test_solver_config_accepts_custom_max_steps():
    assert _solver_config(max_steps=12345).max_steps == 12345


def test_solver_config_rejects_below_one():
    with pytest.raises(ValueError):
        _solver_config(max_steps=0)


def test_solver_config_rejects_non_int():
    with pytest.raises(TypeError):
        _solver_config(max_steps=1.5)


class _CapturingCVODE:
    """Fake CVODE that records the options dict and returns a trivial solve.

    The recorded options let a test assert which ``max_steps`` value the
    real ``_solve_cvode`` placed in the options dict, without running an
    actual integration.
    """

    def __init__(self, captured: dict):
        self._captured = captured

    def __call__(self, rhs, **opts):
        self._captured.clear()
        self._captured.update(opts)
        self._rhs = rhs
        return self

    def solve(self, tspan, y0):
        class _V:
            t = np.asarray(tspan, dtype=float)
            y = np.tile(np.asarray(y0, dtype=float).reshape(-1, 1), (1, len(t)))

        class _Sol:
            flag = 0
            message = 'success'
            roots = None
            values = _V()

        return _Sol()

    def get_info(self):
        return {}


def _run_capture(*, max_steps, with_jacfn):
    """Drive the real ``_solve_cvode`` with a capturing CVODE fake.

    Returns the options dict the real method built. ``with_jacfn`` picks
    the option-Z path by supplying an analytic Jacobian callback.
    """
    captured: dict = {}
    real_cvode = es._scikits_cvode
    es._scikits_cvode = _CapturingCVODE(captured)
    try:
        solver = EntropySolver.__new__(EntropySolver)
        solver._core_bc = 'quasi_steady'
        solver._cvode_output_points = 4
        solver._max_steps = max_steps

        y0 = np.array([2.0, 0.0])
        kwargs = dict(
            start_time=0.0,
            end_time=1.0,
            y0=y0,
            atol=1.0e-8,
            rtol=1.0e-10,
            max_step=np.inf,
            rhs=lambda t, y: y,
        )
        if with_jacfn:
            kwargs['cvode_jacfn'] = lambda t, y, fy, J, user_data=None: 0
        solver._solve_cvode(**kwargs)
    finally:
        es._scikits_cvode = real_cvode
    return captured


def test_configured_max_steps_reaches_cvode_options_fd_path():
    opts = _run_capture(max_steps=12345, with_jacfn=False)
    assert opts['max_steps'] == 12345
    assert 'jacfn' not in opts


def test_configured_max_steps_reaches_cvode_options_option_z_path():
    opts = _run_capture(max_steps=12345, with_jacfn=True)
    assert opts['max_steps'] == 12345
    # Confirm the option-Z branch was actually taken, not the FD path.
    assert 'jacfn' in opts
