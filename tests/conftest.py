"""Utilities for tests."""

from __future__ import annotations

import copy
import functools
import importlib.resources
import os
from contextlib import AbstractContextManager
from importlib.resources.abc import Traversable
from pathlib import Path

import pytest

from aragog import CFG_DATA
from aragog.eos.entropy import EntropyEOS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)


@functools.cache
def _parsed(eos_class: type, eos_dir: str):
    return eos_class(eos_dir)


def entropy_eos_copy(eos_dir: Path | str = EOS_DIR, strict_range: bool = False) -> EntropyEOS:
    """Return an independent copy of ``EntropyEOS(eos_dir, strict_range)``.

    The tables are parsed once per process and each call returns a deep copy,
    which starts in the state of a fresh build. Tests of the constructor itself
    build ``EntropyEOS`` directly.
    """
    eos = copy.deepcopy(_parsed(EntropyEOS, str(eos_dir)))
    eos.strict_range = strict_range
    return eos


def entropy_eos_jax(eos_dir: Path | str = EOS_DIR):
    """Return the ``EntropyEOS_JAX`` of ``eos_dir``, parsed once per process.

    The instance is a frozen equinox module, so callers share it.
    """
    from aragog.jax.eos import EntropyEOS_JAX

    return _parsed(EntropyEOS_JAX, str(eos_dir))


def pytest_collection_finish(session):
    """Parse the tables before the first test, outside every per-test timeout."""
    if not EOS_DIR.exists():
        return
    modules = {getattr(item, 'module', None) for item in session.items}
    if any(hasattr(m, 'entropy_eos_copy') for m in modules):
        entropy_eos_copy()
    if any(hasattr(m, 'entropy_eos_jax') for m in modules):
        entropy_eos_jax()


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
