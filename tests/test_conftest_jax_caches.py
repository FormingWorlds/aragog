"""The compiled JAX programs of a test file are freed when the file ends."""

from __future__ import annotations

import gc
import weakref

import jax
import pytest

from tests import conftest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


class _Cycle:
    pass


def test_the_end_of_a_file_drops_its_compiled_programs_and_cyclic_garbage():
    """The fixture teardown empties the jit cache and collects a reference cycle."""
    f = jax.jit(lambda x: x + 1.0)
    f(1.0)
    cycle = _Cycle()
    cycle.self = cycle
    ref = weakref.ref(cycle)
    del cycle
    teardown = conftest._free_jax_caches_per_file.__wrapped__()
    next(teardown)
    assert f._cache_size() == 1
    gc.disable()
    try:
        with pytest.raises(StopIteration):
            next(teardown)
    finally:
        gc.enable()
    assert f._cache_size() == 0
    assert ref() is None


def test_every_test_file_frees_its_programs_once(request):
    """The fixture is autouse and module-scoped, so it runs once per file a process leaves."""
    assert '_free_jax_caches_per_file' in request.fixturenames
    (fixturedef,) = request._fixturemanager.getfixturedefs(
        '_free_jax_caches_per_file', request.node
    )
    assert fixturedef.scope == 'module'
