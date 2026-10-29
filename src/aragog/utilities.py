"""Utilities"""

from __future__ import annotations

from cProfile import Profile
from functools import wraps
from pathlib import Path
from pstats import SortKey, Stats
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

MultiplyT = TypeVar('MultiplyT', float, npt.NDArray, pd.Series, pd.DataFrame)

FloatOrArray = float | npt.NDArray


def profile_decorator(func):
    """Decorator to profile a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Profile() as profile:
            result = func(*args, **kwargs)
        stats = Stats(profile).strip_dirs().sort_stats(SortKey.TIME)
        stats.print_stats()
        return result

    return wrapper


def is_file(value: Any) -> bool:
    """Checks if value is a file.

    Args:
        value: Object to be checked

    Returns:
        True if the value is a file, otherwise False
    """
    if isinstance(value, (str, Path)):
        return Path(value).is_file()

    return False


def is_monotonic_increasing(some_array: npt.NDArray) -> bool:
    """Returns True if an array is monotonically increasing, otherwise returns False."""
    return np.all(np.diff(some_array) > 0)


def is_number(value: Any) -> bool:
    """Checks if value is a number.

    Args:
        value: Object to be checked

    Returns:
        True if the value is a number, otherwise False
    """
    try:
        float(value)
        return True

    except (TypeError, ValueError):
        return False


def tanh_weight(value: FloatOrArray, threshold: float, width: float) -> npt.NDArray:
    """Computes the tanh weight for viscosity profile and smoothing.

    Args:
        value: Value
        threshold: Threshold value
        width: Width of smoothing

    Returns:
        weight
    """
    arg: FloatOrArray = (value - threshold) / width
    weight: npt.NDArray = 0.5 * (1.0 + np.tanh(arg))

    return weight


def zero_anchored_weight(
    weight_fn: Any, value: FloatOrArray, threshold: float, width: float, xp: Any = np
) -> Any:
    """Rescale ``weight_fn`` to 0 at ``value = 0`` and 1 at ``value = 1``, clipped to [0, 1].

    Parameters
    ----------
    weight_fn : Callable[[Any, float, float], Any]
        Base transition weight callable accepting ``(value, threshold, width)``.
    value : FloatOrArray
        Dimensionless phase coordinate (such as melt fraction in ``[0, 1]``).
    threshold : float
        Transition centre (dimensionless, typically in ``[0, 1]``).
    width : float
        Transition width (dimensionless, strictly positive in ``(0, +inf)``).
    xp : module, default numpy
        Array backend (``numpy`` or ``jax.numpy``).

    Returns
    -------
    FloatOrArray
        Zero-anchored transition weight (dimensionless, in ``[0, 1]``) satisfying
        ``w(0) = 0`` and ``w(1) = 1``.
    """
    w0 = weight_fn(0.0, threshold, width)
    w1 = weight_fn(1.0, threshold, width)
    scaled = (weight_fn(value, threshold, width) - w0) / xp.maximum(w1 - w0, 1.0e-30)
    return xp.clip(scaled, 0.0, 1.0)


def positive_gravity(gravity: Any, where: str) -> npt.NDArray:
    """Returns gravity as an array after checking it is finite and positive on every node.

    The separation velocity takes gravity as a positive magnitude, so its sign
    sets the direction of melt-solid separation.

    Args:
        gravity: Gravity at the mesh nodes [m/s^2]
        where: Mesh and source named in the error message

    Returns:
        gravity as a float array

    Raises:
        ValueError: if any value is not finite or not positive
    """
    g = np.asarray(gravity, dtype=float)
    bad = np.flatnonzero(~(np.isfinite(g) & (g > 0.0)))
    if bad.size:
        raise ValueError(
            f'gravity on the {where} must be finite and positive; node {bad[0]} has '
            f'{float(g.ravel()[bad[0]])!r} ({bad.size} such nodes)'
        )
    return g


def combine_properties(weight: Any, property1: Any, property2: Any) -> Any:
    """Linear weighting of two quantities.

    Args:
        weight: The weight to apply to property1
        property1: The value of the first property
        property2: The value of the second property

    Returns:
        The combined (weighted) property
    """
    out = weight * property1 + (1.0 - weight) * property2

    return out
