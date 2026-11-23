"""The grids of the JAX EOS tables are compile-time constants compared by value.

Module under test: ``aragog.jax.eos`` (``_Table2D``, ``_PhaseBoundary1D``, ``EntropyEOS_JAX``).
Invariants: the pressure and entropy grids are static fields, so XLA folds them and merges the
searches of tables that share a grid; a new EOS object with equal grids runs the program traced
for the first; a grid that differs in one value traces a new program and gives its own result;
a grid cannot change in place.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aragog.jax.eos import EntropyEOS_JAX, _PhaseBoundary1D, _Table2D
from tests.conftest import EOS_DIR, needs_eos

pytestmark = pytest.mark.unit

P = np.linspace(0.0, 1.0e11, 6)
S = np.linspace(1000.0, 3000.0, 5)
V = np.add.outer(P / 1.0e9, S / 100.0)


def _counted(fn):
    """``fn`` under jit, with the number of traces in ``traces[0]``."""
    traces = [0]

    def body(*args):
        traces[0] += 1
        return fn(*args)

    return jax.jit(body), traces


def test_a_table_with_equal_grids_reuses_the_program():
    """A second table object with equal grids and other values traces nothing new."""
    f, traces = _counted(lambda tab, p, s: tab(p, s))
    p, s = jnp.array([2.5e10]), jnp.array([1700.0])
    f(_Table2D(P, S, V), p, s)
    out = f(_Table2D(P.copy(), S.copy(), 2.0 * V), p, s)
    assert traces[0] == 1
    np.testing.assert_allclose(np.asarray(out), np.asarray(_Table2D(P, S, 2.0 * V)(p, s)))


@pytest.mark.parametrize('axis', ['P', 'S'])
def test_a_grid_that_differs_in_one_value_traces_again_and_gives_its_result(axis):
    """A table whose grid differs in 1 knot runs a new program, not the one with the old grid."""
    f, traces = _counted(lambda tab, p, s: tab(p, s))
    p, s = jnp.array([2.5e10]), jnp.array([1700.0])
    f(_Table2D(P, S, V), p, s)
    P2, S2 = P.copy(), S.copy()
    if axis == 'P':
        P2[2] = 4.5e10
    else:
        S2[2] = 1900.0
    out = float(f(_Table2D(P2, S2, V), p, s)[0])
    assert traces[0] == 2
    with jax.disable_jit():
        want = float(_Table2D(P2, S2, V)(p, s)[0])
    assert out == want
    assert out != float(_Table2D(P, S, V)(p, s)[0])


def test_a_phase_boundary_grid_that_differs_traces_again():
    """The pressure grid of a phase boundary is static too: a changed knot traces again."""
    f, traces = _counted(lambda pb, p: pb(p))
    p = jnp.array([3.0e10])
    f(_PhaseBoundary1D(P, S[:1] + P / 1.0e8), p)
    f(_PhaseBoundary1D(P.copy(), S[:1] + P / 1.0e8), p)
    assert traces[0] == 1
    P2 = P.copy()
    P2[2] = 4.5e10
    f(_PhaseBoundary1D(P2, S[:1] + P / 1.0e8), p)
    assert traces[0] == 2


def test_a_wrapped_grid_cannot_change_in_place():
    """Writing into a static grid raises, so no program computes with a grid that changed."""
    tab = _Table2D(P, S, V)
    with pytest.raises(ValueError, match='read-only'):
        tab._P_grid.value[0] = 1.0
    with pytest.raises(ValueError, match='read-only'):
        _PhaseBoundary1D(P, S[:1] + P / 1.0e8)._P.value[0] = 1.0


@needs_eos
def test_a_new_eos_object_from_the_same_tables_reuses_the_program():
    """A second EOS read from the same table files (a structure re-solve) traces nothing new."""
    f, traces = _counted(lambda eos, p, s: eos.temperature(p, s))
    p, s = jnp.array([5.0e10, 1.0e11]), jnp.array([2600.0, 3000.0])
    first = f(EntropyEOS_JAX(EOS_DIR), p, s)
    second = f(EntropyEOS_JAX(EOS_DIR), p, s)
    assert traces[0] == 1
    np.testing.assert_array_equal(np.asarray(second), np.asarray(first))
