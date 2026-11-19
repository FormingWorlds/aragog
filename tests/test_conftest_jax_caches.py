"""The compiled JAX programs of a test file are freed when the file ends."""

from __future__ import annotations

import gc

import pytest

from tests import conftest

jax = pytest.importorskip('jax')
pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_the_end_of_a_file_drops_its_compiled_programs_and_collects_garbage(monkeypatch):
    """The fixture teardown empties the jit cache and calls the garbage collector once."""
    f = jax.jit(lambda x: x + 1.0)
    f(1.0)
    collected = []
    monkeypatch.setattr(gc, 'collect', lambda: collected.append(True))
    teardown = conftest._free_jax_caches_per_file.__wrapped__()
    next(teardown)
    assert f._cache_size() == 1
    assert collected == []
    with pytest.raises(StopIteration):
        next(teardown)
    assert f._cache_size() == 0
    assert collected == [True]


def test_the_fixture_is_autouse_at_module_scope(request):
    """The fixture is autouse and module-scoped, so it runs once per file a process leaves."""
    assert '_free_jax_caches_per_file' in request.fixturenames
    (fixturedef,) = request._fixturemanager.getfixturedefs(
        '_free_jax_caches_per_file', request.node
    )
    assert fixturedef.scope == 'module'
