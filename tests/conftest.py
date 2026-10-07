"""Utilities for tests."""

from __future__ import annotations

import copy
import functools
import importlib.resources
import os
import warnings
from contextlib import AbstractContextManager
from importlib.resources.abc import Traversable
from pathlib import Path

import numpy as np
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
def _parse(eos_class: type, eos_dir: str):
    try:
        return eos_class(eos_dir), None, None
    except Exception as exc:
        return None, exc, exc.__traceback__


def _parsed(eos_class: type, eos_dir: str):
    """Return the cached parse of ``eos_dir``; a failed parse re-raises without a retry."""
    eos, exc, traceback = _parse(eos_class, eos_dir)
    if exc is not None:
        raise exc.with_traceback(traceback)
    return eos


def entropy_eos_copy(eos_dir: Path | str = EOS_DIR, strict_range: bool = False) -> EntropyEOS:
    """Return an independent copy of ``EntropyEOS(eos_dir, strict_range)``.

    The tables are parsed once per process and each call returns a deep copy,
    which starts in the state of a fresh build; only the read-only solidus and
    liquidus interpolation functions are shared. Tests of the constructor
    itself build ``EntropyEOS`` directly.
    """
    eos = copy.deepcopy(_parsed(EntropyEOS, str(eos_dir)))
    eos.strict_range = strict_range
    return eos


def entropy_eos_jax(eos_dir: Path | str = EOS_DIR):
    """Return the ``EntropyEOS_JAX`` of ``eos_dir``, parsed once per process.

    The instance is a frozen equinox module of JAX arrays, so callers share it.
    """
    from aragog.jax.eos import EntropyEOS_JAX

    return _parsed(EntropyEOS_JAX, str(eos_dir))


def make_mesh(N: int = 10, scale_p: float = 1.0):
    """Build a canonical MeshArrays test fixture."""
    import jax.numpy as jnp

    from aragog.jax.phase import MeshArrays

    r_inner = 3.480e6
    r_outer = 6.371e6
    r_stag = np.linspace(r_inner, r_outer, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = r_inner
    r_basic[-1] = r_outer
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    area = 4.0 * np.pi * r_basic**2
    volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.maximum(np.minimum(r_basic - r_inner, r_outer - r_basic), 1.0)
    d_dr = np.zeros((N + 1, N))
    for i in range(1, N):
        d_dr[i, i - 1] = -1.0 / dr[i - 1]
        d_dr[i, i] = 1.0 / dr[i - 1]
    d_dr[0, :] = d_dr[1, :]
    d_dr[-1, :] = d_dr[-2, :]
    q_mat = np.zeros((N + 1, N))
    q_mat[0, 0] = 1.0
    q_mat[-1, -1] = 1.0
    for i in range(1, N):
        q_mat[i, i - 1] = 0.5
        q_mat[i, i] = 0.5
    p_stag = np.linspace(135e9, 1e5, N) * scale_p
    p_basic = q_mat @ p_stag
    return MeshArrays(
        d_dr_matrix=jnp.asarray(d_dr),
        quantity_matrix=jnp.asarray(q_mat),
        area=jnp.asarray(area),
        volume=jnp.asarray(volume),
        radii_basic=jnp.asarray(r_basic),
        radii_stag=jnp.asarray(r_stag),
        mixing_length=jnp.asarray(ml),
        mixing_length_sq=jnp.asarray(ml**2),
        mixing_length_cu=jnp.asarray(ml**3),
        P_stag=jnp.asarray(p_stag),
        P_basic=jnp.asarray(p_basic),
        gravity=jnp.full(N + 1, 10.0),
    )


def pytest_collection_finish(session):
    """Parse the tables before the first test, outside every per-test timeout.

    A failed parse is reported as a warning and re-raised in each test that needs the tables.
    """
    if session.config.option.collectonly or not EOS_DIR.exists():
        return
    modules = {getattr(item, 'module', None) for item in session.items}
    for helper in (entropy_eos_copy, entropy_eos_jax):
        if any(hasattr(m, helper.__name__) for m in modules):
            try:
                helper(EOS_DIR)
            except Exception as exc:
                warnings.warn(f'EOS table warm-up failed: {exc!r}', stacklevel=1)


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
