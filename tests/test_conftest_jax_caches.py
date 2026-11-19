"""The compiled JAX programs of a test file are freed when the file ends."""

from __future__ import annotations

import jax
import pytest

from tests import conftest


@pytest.mark.unit
def test_the_end_of_a_file_drops_its_compiled_programs():
    f = jax.jit(lambda x: x + 1.0)
    f(1.0)
    teardown = conftest._free_jax_caches_per_file.__wrapped__()
    next(teardown)
    assert f._cache_size() == 1
    with pytest.raises(StopIteration):
        next(teardown)
    assert f._cache_size() == 0


@pytest.mark.unit
def test_every_test_file_frees_its_programs_once(request):
    assert '_free_jax_caches_per_file' in request.fixturenames
    (fixturedef,) = request._fixturemanager.getfixturedefs(
        '_free_jax_caches_per_file', request.node
    )
    assert fixturedef.scope == 'module'
