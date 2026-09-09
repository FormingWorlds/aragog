"""Utilities for tests."""

from __future__ import annotations

import importlib.resources
from contextlib import AbstractContextManager
from importlib.resources.abc import Traversable
from pathlib import Path

import pytest

from aragog import CFG_DATA


class Helper:
    """Helper class for tests

    Args:
        atol: Absolute tolerance for passing tests. Defaults to 1.0e-4.
        rtol: Relative tolerance for passing tests Defaults to 1.0e-4.

    Attributes:
        atol: Absolute tolerance
        rtol: Relative tolerance
        test_data: Path to the reference test data
    """

    def __init__(self, atol: float = 1.0e-4, rtol: float = 1.0e-4):
        self.atol: float = atol
        self.rtol: float = rtol
        self.test_data: Traversable = importlib.resources.files('tests.reference')

    @staticmethod
    def get_cfg_file(filename: str) -> AbstractContextManager[Path]:
        return importlib.resources.as_file(CFG_DATA.joinpath(filename))

    def get_reference_file(self, filename: str) -> AbstractContextManager[Path]:
        return importlib.resources.as_file(self.test_data.joinpath(filename))


@pytest.fixture(scope='module')
def helper():
    return Helper()


@pytest.fixture(autouse=True)
def _reset_range_warning_counts():
    """Give each test its own count of the entropy range warning.

    ``_check_entropy_range`` throttles this warning per context string
    across the life of the process; without a reset, an earlier test
    sharing a context would suppress a later test's first occurrence.
    JAX is optional, so this is a no-op when it is not installed.
    """
    try:
        from aragog.jax.eos import reset_range_warning_counts
    except ImportError:
        yield
        return

    reset_range_warning_counts()
    yield
