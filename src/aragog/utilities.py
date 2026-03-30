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

MultiplyT = TypeVar("MultiplyT", float, npt.NDArray, pd.Series, pd.DataFrame)

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
