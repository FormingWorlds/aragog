"""Utilities for tests."""

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
        self.test_data: Traversable = importlib.resources.files("tests.reference")

    @staticmethod
    def get_cfg_file(filename: str) -> AbstractContextManager[Path]:
        return importlib.resources.as_file(CFG_DATA.joinpath(filename))

    def get_reference_file(self, filename: str) -> AbstractContextManager[Path]:
        return importlib.resources.as_file(self.test_data.joinpath(filename))


@pytest.fixture(scope="module")
def helper():
    return Helper()
