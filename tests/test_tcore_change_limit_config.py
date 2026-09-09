"""Tests for the optional ``tcore_change_limit`` solver option.

The limit must exist on both the live runtime dataclass
``aragog.parser._SolverParameters`` and its attrs schema mirror
``aragog.config.solver.SolverConfig``, default to ``None`` (flag
disabled), accept a positive number, and reject zero, negative, boolean,
and non-numeric values on both surfaces.
"""

from __future__ import annotations

import pytest

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


def test_solver_parameters_default_is_none():
    assert _solver_parameters().tcore_change_limit is None


def test_solver_parameters_accepts_positive():
    assert _solver_parameters(tcore_change_limit=1500.0).tcore_change_limit == 1500.0


def test_solver_parameters_rejects_zero():
    with pytest.raises(ValueError):
        _solver_parameters(tcore_change_limit=0.0)


def test_solver_parameters_rejects_negative():
    with pytest.raises(ValueError):
        _solver_parameters(tcore_change_limit=-5.0)


def test_solver_parameters_rejects_bool():
    with pytest.raises(TypeError):
        _solver_parameters(tcore_change_limit=True)


def test_solver_parameters_rejects_non_numeric():
    with pytest.raises(TypeError):
        _solver_parameters(tcore_change_limit='x')


def test_solver_config_default_is_none():
    assert _solver_config().tcore_change_limit is None


def test_solver_config_accepts_positive():
    assert _solver_config(tcore_change_limit=1500.0).tcore_change_limit == 1500.0


def test_solver_config_rejects_zero():
    with pytest.raises(ValueError):
        _solver_config(tcore_change_limit=0.0)


def test_solver_config_rejects_negative():
    with pytest.raises(ValueError):
        _solver_config(tcore_change_limit=-5.0)


def test_solver_config_rejects_bool():
    with pytest.raises((TypeError, ValueError)):
        _solver_config(tcore_change_limit=True)


def test_solver_config_rejects_non_numeric():
    with pytest.raises(TypeError):
        _solver_config(tcore_change_limit='x')
