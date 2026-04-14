"""Numerical helpers used across Aragog."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatOrArray = float | npt.NDArray


def is_monotonic_increasing(some_array: npt.NDArray) -> bool:
    """Return True if all consecutive differences are strictly positive."""
    return np.all(np.diff(some_array) > 0)


def tanh_weight(value: FloatOrArray, threshold: float, width: float) -> npt.NDArray:
    """Smoothed step centred on ``threshold`` with transition ``width``.

    Returns 0 for ``value`` well below ``threshold``, 1 well above, and
    a tanh-shaped transition in between. Used by the MLT eddy floor
    and other phase-dependent material properties.
    """
    arg: FloatOrArray = (value - threshold) / width
    return 0.5 * (1.0 + np.tanh(arg))
