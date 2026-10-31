"""Utilities for tests."""

from __future__ import annotations

import importlib.resources
import logging
import os
import warnings
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from importlib.resources.abc import Traversable
from pathlib import Path

import numpy as np
import pytest

from aragog import CFG_DATA
from aragog import logger as _PACKAGE_LOGGER
from aragog.eos.entropy import EntropyEOS
from tools.verification import _ssc_helpers
from tools.verification._ssc_helpers import _parsed

__all__ = [
    '_parsed',
    'assert_legend_clear',
    'entropy_eos_copy',
    'entropy_eos_jax',
    'package_logger_isolation',
]

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


def entropy_eos_copy(
    eos_dir: Path | str = EOS_DIR, strict_range: bool = False, identity_alpha: bool = False
) -> EntropyEOS:
    """Return an independent copy of ``EntropyEOS(eos_dir, strict_range)``."""
    return _ssc_helpers.entropy_eos_copy(
        eos_dir, strict_range, identity_alpha, eos_class=EntropyEOS
    )


def entropy_eos_jax(eos_dir: Path | str = EOS_DIR):
    """Return the ``EntropyEOS_JAX`` of ``eos_dir``, parsed once per process."""
    return _ssc_helpers.entropy_eos_jax(eos_dir)


def assert_legend_clear(fig) -> None:
    """Assert that no legend of ``fig`` covers a marker or a line of its axes, or a text.

    Each drawn line is sampled at 50 points per segment in display space, markers at their
    points, and the legend box is padded by half the marker size. Bands (patches, filled collections) may lie under it.
    No visible text of the figure outside the legend may overlap the legend box.
    """
    from matplotlib.text import Text

    fig.canvas.draw()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        paths = [
            (
                ln.get_transform().transform(ln.get_xydata()),
                ln.get_markersize(),
                ln.get_linestyle(),
            )
            for ln in ax.get_lines()
            if ln.get_visible()
        ]
        for coll in ax.collections:
            if hasattr(coll, 'get_segments'):
                tr = coll.get_transform()
                paths += [(tr.transform(seg), 0.0, '-') for seg in coll.get_segments()]
        for xy, ms, ls in paths:
            box = legend.get_window_extent().padded(0.5 * ms * fig.dpi / 72.0)
            s = np.linspace(0.0, 1.0, 2 if ls in ('None', '', ' ') else 50)[:, None]
            dense = np.concatenate(
                [xy[:1]] + [a + s * (b - a) for a, b in zip(xy[:-1], xy[1:])]
            )
            hit = box.count_contains(dense[np.all(np.isfinite(dense), axis=1)])
            assert hit == 0, f'the legend of {ax.get_ylabel()!r} covers {hit} plotted points'
        box, own = legend.get_window_extent(), set(legend.findobj())
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text() and text not in own:
                assert not box.overlaps(text.get_window_extent()), (
                    f'a legend covers the text {text.get_text()!r}'
                )


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


@contextmanager
def package_logger_isolation():
    """Restore the level, handlers and flags of the package logger on exit.

    Each handler attached in the block and not attached before it is closed, also one that the code
    in the block detached again. Child loggers, filters and ``logging.disable`` are not covered.
    """
    lg = _PACKAGE_LOGGER
    level, propagate, disabled = lg.level, lg.propagate, lg.disabled
    handlers = list(lg.handlers)
    outer_recorder = vars(lg).get('addHandler')
    forward = outer_recorder or partial(logging.Logger.addHandler, lg)
    added = []

    def record(handler):
        added.append(handler)
        forward(handler)

    lg.addHandler = record
    try:
        yield
    finally:
        if outer_recorder is None:
            del lg.addHandler
        else:
            lg.addHandler = outer_recorder
        for handler in dict.fromkeys(added + lg.handlers):
            if handler not in handlers:
                handler.close()
        lg.handlers = handlers
        lg.propagate = propagate
        lg.disabled = disabled
        lg.setLevel(level)


@pytest.fixture(autouse=True)
def _isolated_package_logger():
    """Restore the package logger after each test, see ``package_logger_isolation``.

    ``aragog run`` sets the logger to DEBUG with a console and a file handler for the rest of the
    process; under DEBUG the mesh formats whole arrays in its debug lines on every state update.
    """
    with package_logger_isolation():
        yield


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
