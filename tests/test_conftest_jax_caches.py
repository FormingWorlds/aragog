"""The compiled JAX programs of a test file are freed when the file ends."""

import jax
import pytest

from tests.conftest import free_jax_caches


@pytest.mark.unit
def test_free_jax_caches_drops_a_compiled_program():
    f = jax.jit(lambda x: x + 1.0)
    f(1.0)
    assert f._cache_size() == 1
    free_jax_caches()
    assert f._cache_size() == 0


@pytest.mark.unit
def test_every_test_file_uses_the_cache_fixture(request):
    assert '_free_jax_caches_per_file' in request.fixturenames


@pytest.mark.unit
def test_the_cache_fixture_runs_once_per_file(request):
    (fixturedef,) = request._fixturemanager.getfixturedefs('_free_jax_caches_per_file', request.node)
    assert fixturedef.scope == 'module'
