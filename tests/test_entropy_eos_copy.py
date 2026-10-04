"""The cached ``EntropyEOS`` copies that the tests share are independent."""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest
from scipy.interpolate import interp1d

from aragog.eos.entropy import EntropyEOS
from tests import conftest
from tests.conftest import EOS_DIR, _parsed, entropy_eos_copy, entropy_eos_jax, needs_eos


def _reachable(root):
    """Return {id: (path, object)} for each mutable object reachable from ``root``.

    The walk follows dicts, lists, tuples, instance attributes and closure cells.
    """
    found, seen, stack = {}, set(), [(root, 'eos')]
    while stack:
        obj, path = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, types.FunctionType):
            stack += [(c.cell_contents, f'{path}.<closure>') for c in obj.__closure__ or ()]
            continue
        if isinstance(obj, (np.ndarray, dict, list, set)) or hasattr(obj, '__dict__'):
            found[id(obj)] = (path, obj)
        if isinstance(obj, dict):
            stack += [(v, f'{path}[{k!r}]') for k, v in obj.items()]
        elif isinstance(obj, (list, tuple)):
            stack += [(v, f'{path}[{i}]') for i, v in enumerate(obj)]
        elif not isinstance(obj, type) and hasattr(obj, '__dict__'):
            stack += [(v, f'{path}.{k}') for k, v in vars(obj).items()]
    return found


@pytest.mark.unit
def test_the_walk_reaches_arrays_in_tuples_closures_and_attributes():
    in_tuple, in_closure, in_attribute = np.zeros(1), np.zeros(2), np.zeros(3)
    holder = types.SimpleNamespace(
        t=(in_tuple,), f=lambda: in_closure, a=types.SimpleNamespace(x=in_attribute)
    )
    found = _reachable(holder)
    assert all(id(a) in found for a in (in_tuple, in_closure, in_attribute))


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
        i: obj
        for b in (cached._solidus, cached._liquidus)
        for f in ('interp', 'dinterp')
        for i, (_, obj) in _reachable(b[f]).items()
    }
    assert len(boundary) <= 24
    assert all(isinstance(obj, (interp1d, np.ndarray)) for obj in boundary.values())

    anchors = (
        strict._tables['density_melt']['interp']._grid[0],
        strict._h_interp,
        strict._h_grid,
    )
    assert all(id(a) in in_copy for a in anchors)
    cached_arrays = [
        o for i, (_, o) in in_cached.items() if isinstance(o, np.ndarray) and i not in boundary
    ]
    shared = [
        path
        for i, (path, obj) in in_copy.items()
        if i not in boundary
        and (
            i in in_cached
            or isinstance(obj, np.ndarray)
            and any(np.may_share_memory(obj, c) for c in cached_arrays)
        )
    ]
    assert shared == []
    assert strict.strict_range and not cached.strict_range


@pytest.mark.smoke
@needs_eos
def test_the_shared_jax_eos_holds_no_numpy_array():
    jax = pytest.importorskip('jax')
    eos = entropy_eos_jax()
    objects = [obj for _, obj in _reachable(eos).values()]
    leaves = jax.tree_util.tree_leaves(eos)
    assert eos is entropy_eos_jax()
    assert (
        sum(isinstance(o, jax.Array) for o in objects)
        >= sum(isinstance(x, jax.Array) for x in leaves)
        > 0
    )
    assert not any(isinstance(o, np.ndarray) for o in objects)


@pytest.mark.unit
def test_a_failed_parse_warns_once_and_re_raises_without_a_retry(monkeypatch, tmp_path):
    calls = []

    class Broken:
        def __init__(self, eos_dir):
            calls.append(eos_dir)
            raise FileNotFoundError(eos_dir)

    monkeypatch.setattr(conftest, 'EntropyEOS', Broken)
    monkeypatch.setattr(conftest, 'EOS_DIR', tmp_path)
    module = types.ModuleType('uses_the_tables')
    module.entropy_eos_copy = entropy_eos_copy
    session = types.SimpleNamespace(
        config=types.SimpleNamespace(option=types.SimpleNamespace(collectonly=False)),
        items=[types.SimpleNamespace(module=module)],
    )
    with pytest.warns(UserWarning, match='warm-up failed'):
        conftest.pytest_collection_finish(session)
    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            entropy_eos_copy(tmp_path)
    assert calls == [str(tmp_path)]
