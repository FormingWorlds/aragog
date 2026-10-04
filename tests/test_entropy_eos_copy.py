"""The cached ``EntropyEOS`` copies that the tests share are independent."""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest

from aragog.eos.entropy import EntropyEOS
from tests.conftest import EOS_DIR, _parsed, entropy_eos_copy, needs_eos


def _mutables(obj, path='eos', seen=None):
    """Yield (path, value) for each array, dict and list reachable from ``obj``."""
    seen = set() if seen is None else seen
    if id(obj) in seen or isinstance(obj, types.FunctionType):
        return
    seen.add(id(obj))
    if isinstance(obj, (np.ndarray, dict, list)):
        yield path, obj
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = enumerate(obj)
    elif isinstance(obj, np.ndarray) or not hasattr(obj, '__dict__'):
        items = ()
    else:
        items = vars(obj).items()
    for key, value in items:
        yield from _mutables(value, f'{path}[{key!r}]', seen)


@pytest.mark.smoke
@needs_eos
def test_mutating_one_copy_leaves_the_next_copy_pristine(caplog):
    P = np.array([5.0e10])
    first = entropy_eos_copy()
    S_solid = first.solidus_entropy(P) - 500.0
    T_pristine = first.temperature(P, S_solid)
    S_out = np.array([first.S_max + 1.0e5])

    first._check_entropy_range(S_out, first.S_min, first.S_max, 'probe')
    first._tables['temperature_solid']['interp'].values[:] = np.nan
    assert first._range_warning_counts == {'probe': 1}
    assert np.isnan(first.temperature(P, S_solid)).all()

    second = entropy_eos_copy()
    assert second._range_warning_counts == {}
    np.testing.assert_array_equal(second.temperature(P, S_solid), T_pristine)
    with caplog.at_level(logging.WARNING):
        second._check_entropy_range(S_out, second.S_min, second.S_max, 'probe')
    assert any('probe' in r.message for r in caplog.records)


@pytest.mark.smoke
@needs_eos
def test_a_copy_shares_no_array_dict_or_list_with_the_cached_parse():
    cached = _parsed(EntropyEOS, str(EOS_DIR))
    strict = entropy_eos_copy(strict_range=True)
    copied = dict(_mutables(strict))
    assert len(copied) > 50
    shared = [
        path
        for path, value in _mutables(cached)
        if path in copied
        and (
            copied[path] is value
            or isinstance(value, np.ndarray)
            and np.shares_memory(value, copied[path])
        )
    ]
    assert shared == []
    assert strict.strict_range and not cached.strict_range
