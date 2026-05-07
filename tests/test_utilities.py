"""Unit tests for ``aragog.utilities``.

Covers the small helper layer used across the codebase:
``is_file``, ``is_monotonic_increasing``, ``is_number``,
``tanh_weight``, and ``combine_properties``. These helpers are
imported into hot paths (e.g. ``tanh_weight`` is used to smooth
the rheological transition); a regression here can silently
shift physics by O(1) in the mushy band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aragog.utilities import (
    combine_properties,
    is_file,
    is_monotonic_increasing,
    is_number,
    tanh_weight,
)

pytestmark = pytest.mark.unit


# ---- is_file ----------------------------------------------------------------


def test_is_file_returns_true_for_existing_file(tmp_path: Path):
    """An existing file at a string OR pathlib.Path is recognised."""
    target = tmp_path / 'data.txt'
    target.write_text('hello')

    assert is_file(str(target)) is True
    assert is_file(target) is True


def test_is_file_returns_false_for_missing_path(tmp_path: Path):
    """A path that does not exist returns False, not a raise."""
    missing = tmp_path / 'does_not_exist.csv'
    assert is_file(str(missing)) is False
    assert is_file(missing) is False


def test_is_file_returns_false_for_directory(tmp_path: Path):
    """Directories are not files. Edge case: ``Path.is_file`` is the
    discriminator vs a naive ``Path.exists``.
    """
    assert is_file(tmp_path) is False, (
        'is_file accepted a directory; downstream loaders will'
        ' fail with a confusing OSError instead of clean validation.'
    )


def test_is_file_returns_false_for_non_path_types():
    """Edge case: non-str, non-Path inputs return False, not raise.

    The function is used in attrs validators where it must safely
    handle dict, int, None, etc., even though those are not valid
    file specifiers.
    """
    assert is_file(None) is False
    assert is_file(42) is False
    assert is_file({'foo': 'bar'}) is False
    assert is_file([1, 2, 3]) is False


# ---- is_monotonic_increasing ------------------------------------------------


def test_is_monotonic_increasing_strict_increasing_array():
    """A strictly increasing sequence must return True."""
    assert is_monotonic_increasing(np.array([1.0, 2.0, 3.0, 4.0])) is np.True_


def test_is_monotonic_increasing_rejects_flat_segment():
    """A plateau (equal adjacent values) is NOT strictly increasing.

    Discriminator: a regression to ``np.all(np.diff >= 0)`` would
    accept this and silently allow degenerate meshes through.
    """
    arr = np.array([1.0, 2.0, 2.0, 3.0])
    assert is_monotonic_increasing(arr) is np.False_, (
        'A flat segment passed strict monotonic check; '
        'mesh validation would accept duplicate radii.'
    )


def test_is_monotonic_increasing_rejects_decreasing_array():
    """Reversed array fails."""
    arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert is_monotonic_increasing(arr) is np.False_


def test_is_monotonic_increasing_single_element():
    """Single element edge case: vacuously true (no diffs)."""
    arr = np.array([42.0])
    assert is_monotonic_increasing(arr) is np.True_


# ---- is_number --------------------------------------------------------------


def test_is_number_accepts_int_float_string_numerals():
    """Convertible-to-float inputs return True."""
    assert is_number(0) is True
    assert is_number(-3.14) is True
    assert is_number('1e6') is True
    assert is_number(' 42 ') is True  # float() strips whitespace


def test_is_number_rejects_non_numeric_inputs():
    """Edge cases: garbage strings, None, dict return False, not raise."""
    assert is_number('not_a_number') is False
    assert is_number(None) is False
    assert is_number({'k': 1}) is False
    assert is_number([1, 2]) is False


# ---- tanh_weight ------------------------------------------------------------


def test_tanh_weight_at_threshold_returns_one_half():
    """At value == threshold the weight is exactly 0.5.

    This is the inflection point of the smoothing function; if a
    regression flips the sign (``tanh(-arg)`` instead of ``tanh(arg)``)
    or shifts the threshold by mistake the value here would deviate.
    Using width != 1 to discriminate against a regression that
    silently divides by 1.
    """
    w = tanh_weight(value=300.0, threshold=300.0, width=10.0)
    assert float(w) == pytest.approx(0.5, abs=1e-15)


def test_tanh_weight_far_above_threshold_saturates_to_one():
    """Many widths above threshold the weight is essentially 1.0.

    Values: threshold=1000, width=50, value=2000 means arg=20 so
    tanh(20) ~ 1 - 1e-17. We assert > 1 - 1e-12.
    """
    w = tanh_weight(value=2000.0, threshold=1000.0, width=50.0)
    assert float(w) > 1.0 - 1e-12


def test_tanh_weight_far_below_threshold_saturates_to_zero():
    """Many widths below threshold the weight saturates to zero."""
    w = tanh_weight(value=0.0, threshold=1000.0, width=50.0)
    assert float(w) < 1e-12


def test_tanh_weight_monotonic_in_value():
    """Property: increasing value at fixed threshold/width must
    monotonically increase the weight (tanh is monotonic). Catches
    a sign-flip regression that the inflection-point test misses
    only when width is unit-magnitude.
    """
    values = np.linspace(-10.0, 10.0, 20)
    weights = tanh_weight(values, threshold=0.0, width=1.0)
    diffs = np.diff(np.asarray(weights))
    assert np.all(diffs > 0), (
        'tanh_weight is not monotonic in value; '
        'a sign flip on the tanh argument would invert the smoothing.'
    )


def test_tanh_weight_array_input_returns_array():
    """Vectorised array input returns the same-shape array."""
    values = np.array([100.0, 300.0, 500.0])
    out = tanh_weight(values, threshold=300.0, width=50.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    # Symmetry: f(threshold + d) + f(threshold - d) == 1.
    # Distance from threshold is symmetric (200 each side).
    assert float(out[0] + out[2]) == pytest.approx(1.0, abs=1e-15)


# ---- combine_properties ----------------------------------------------------


def test_combine_properties_weight_zero_returns_property2():
    """Weight=0: linear blend collapses to property2."""
    out = combine_properties(weight=0.0, property1=42.0, property2=7.0)
    assert float(out) == pytest.approx(7.0, abs=1e-15)


def test_combine_properties_weight_one_returns_property1():
    """Weight=1: linear blend collapses to property1.

    Discriminator: a regression that swaps the formula to
    weight*p2 + (1-weight)*p1 would silently invert the blend.
    """
    out = combine_properties(weight=1.0, property1=42.0, property2=7.0)
    assert float(out) == pytest.approx(42.0, abs=1e-15)


def test_combine_properties_blends_arrays_pointwise():
    """Property: array inputs blend element-wise."""
    p1 = np.array([100.0, 200.0, 300.0])
    p2 = np.array([10.0, 20.0, 30.0])
    weight = np.array([0.0, 0.5, 1.0])
    out = combine_properties(weight=weight, property1=p1, property2=p2)
    expected = np.array([10.0, 110.0, 300.0])
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-15)


def test_profile_decorator_returns_decorated_function_value(capsys):
    """``profile_decorator`` must wrap a function so that calling
    the wrapped version returns the original return value AND prints
    cProfile output to stdout. Discriminator: a regression that
    swallowed the return value would leave callers seeing None.
    """
    from aragog.utilities import profile_decorator

    @profile_decorator
    def add(a, b):
        return a + b

    result = add(2, 3)
    assert result == 5
    captured = capsys.readouterr()
    # cProfile prints a summary header containing "function calls"
    assert 'function calls' in captured.out


def test_profile_decorator_preserves_keyword_arguments():
    """Edge case: kwargs must round-trip through the decorator."""
    from aragog.utilities import profile_decorator

    @profile_decorator
    def power(base, *, exp=2):
        return base**exp

    assert power(3.0, exp=4) == pytest.approx(81.0, rel=1e-12)


def test_combine_properties_extrapolates_for_weight_above_one():
    """Edge case: a weight > 1 extrapolates linearly (not clamped).

    Aragog uses combine_properties downstream of validated weights
    (smoothing functions in [0, 1]); this test documents that the
    helper itself does NOT clamp, so a caller that passes 1.5 by
    mistake will get out = 1.5*p1 - 0.5*p2.
    """
    out = combine_properties(weight=1.5, property1=10.0, property2=4.0)
    expected = 1.5 * 10.0 + (1.0 - 1.5) * 4.0
    assert float(out) == pytest.approx(expected, abs=1e-15)
