"""The cached ``EntropyEOS`` copies that the tests share are independent."""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest

from aragog.eos.entropy import EntropyEOS
from tests.conftest import EOS_DIR, _parsed, entropy_eos_copy, entropy_eos_jax, needs_eos


def _reachable(root):
    """Return {id: path} for each mutable object reachable from ``root``.

    The walk follows dicts, lists, tuples, instance attributes and closure cells.
    """
    found, seen, stack = {}, set(), [(root, 'eos')]
    while stack:
        obj, path = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, (np.ndarray, dict, list, set)) or hasattr(obj, '__dict__'):
            found[id(obj)] = path
        if isinstance(obj, dict):
            stack += [(v, f'{path}[{k!r}]') for k, v in obj.items()]
        elif isinstance(obj, (list, tuple)):
            stack += [(v, f'{path}[{i}]') for i, v in enumerate(obj)]
        elif isinstance(obj, types.FunctionType):
            found.pop(id(obj))
            stack += [(c.cell_contents, f'{path}.<closure>') for c in obj.__closure__ or ()]
        elif not isinstance(obj, (np.ndarray, type)) and hasattr(obj, '__dict__'):
            stack += [(v, f'{path}.{k}') for k, v in vars(obj).items()]
    return found


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
def test_a_copy_shares_only_the_phase_boundary_functions_with_the_cached_parse():
    cached = _parsed(EntropyEOS, str(EOS_DIR))
    strict = entropy_eos_copy(strict_range=True)
    in_cached, in_copy = _reachable(cached), _reachable(strict)
    boundary = {
        i
        for b in (cached._solidus, cached._liquidus)
        for f in ('interp', 'dinterp')
        for i in _reachable(b[f])
    }
    assert len(in_copy) > 50
    assert [path for i, path in in_copy.items() if i in in_cached and i not in boundary] == []
    assert strict.strict_range and not cached.strict_range


@pytest.mark.smoke
@needs_eos
def test_the_shared_jax_eos_holds_no_mutable_array():
    jax = pytest.importorskip('jax')
    leaves = jax.tree_util.tree_leaves(entropy_eos_jax())
    assert entropy_eos_jax() is entropy_eos_jax()
    assert leaves and not any(isinstance(leaf, np.ndarray) for leaf in leaves)
