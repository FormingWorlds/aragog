"""The JAX CVODE programs compile once per form of their inputs, not once per input object.

Module under test: ``aragog.solver.cvode_jax.build_jax_rhs_and_jacobian``. Invariants: a
structure re-solve builds new phase parameters, a new EOS object and (core_module) a new core
budget of the same form; the RHS and the Jacobian built from them run the programs compiled for
the first set, and every input reaches those programs, so their values equal those of a build
from empty caches. Compiles are counted by the trace counters of the module and the
backend-compile events of ``jax.monitoring``.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip('jax')
eqx = pytest.importorskip('equinox')

import jax  # noqa: E402
import jax.monitoring  # noqa: E402
import numpy as np  # noqa: E402

from aragog.jax.nondim import NonDimScales  # noqa: E402
from aragog.jax.phase import PhaseParams  # noqa: E402
from aragog.solver.cvode_jax import (  # noqa: E402
    _TRACE_COUNTERS,
    build_jax_rhs_and_jacobian,
    clear_jit_cache,
)
from tests.conftest import EOS_DIR, entropy_eos_jax  # noqa: E402
from tests.conftest import make_mesh as _make_mesh  # noqa: E402
from tests.test_cvode_jax_cache import _make_bc  # noqa: E402

pytestmark = pytest.mark.unit

_COMPILE = '/jax/core/compile/backend_compile_duration'
_SEEN = []


def _listen(name, _duration, **_kw):
    if name == _COMPILE:
        _SEEN.append(name)


jax.monitoring.register_event_duration_secs_listener(_listen)

N = 10
EXTRA = {'quasi_steady': 0, 'energy_balance': 1, 'core_module': 2}
MODES = list(EXTRA)


@pytest.fixture(scope='module')
def eos():
    if not EOS_DIR.exists():
        if os.environ.get('CI'):
            pytest.fail(f'CI requires SPIDER P-S tables, but {EOS_DIR} not found.')
        pytest.skip(f'SPIDER P-S tables not found at {EOS_DIR}.')
    return entropy_eos_jax(EOS_DIR)


def _budget(rho_cen=12500.0):
    from aragog.core import build_core_module_budget

    return build_core_module_budget(
        {
            'melting_curve': 'iron',
            'light_element_fraction': 0.1,
            'depression': 1.2,
            'rho_cen': rho_cen,
        },
        r_cmb=3.48e6,
        p_cmb_fallback=135e9,
    )


def _scales(mode):
    extra = {'quasi_steady': [], 'energy_balance': [1e-3], 'core_module': [1e-3, 4.0e3]}[mode]
    return NonDimScales(state_scale=np.array([3.0e3] * N + extra), t_ref=1.0)


def _state(mode):
    extra = {'quasi_steady': [], 'energy_balance': [-1e-3], 'core_module': [-1e-3, 4100.0]}
    return (
        np.array(list(np.linspace(3100.0, 2900.0, N)) + extra[mode]) / _scales(mode).state_scale
    )


def _evaluate(mode, eos_jax, params, budget=None):
    """RHS and Jacobian of a factory built from these inputs, on one mesh and one state."""
    mesh = _make_mesh(N=N)
    rhs, jac, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh,
        boundary_params=_make_bc(mesh, outer_type=4, inner_type=1, outer_val=10.0),
        heating_array=np.zeros(N),
        scales=_scales(mode),
        core_bc_mode=mode,
        core_module_budget=budget,
        core_module_ra_crit_cmb=450.0 if mode == 'core_module' else None,
    )
    n = N + EXTRA[mode]
    f, J = np.zeros(n), np.zeros((n, n))
    assert rhs(0.0, _state(mode), f) == 0
    assert jac(0.0, _state(mode), None, J) == 0
    return f, J


def _copy(tree):
    """A new object with the same leaves, as a re-solve builds."""
    return jax.tree_util.tree_map(lambda x: x, tree)


@pytest.mark.parametrize('mode', MODES)
def test_new_input_objects_of_one_form_reuse_the_programs(eos, mode):
    """A factory built from new phase parameters, a new EOS object and a new budget of the same
    form traces and compiles nothing, and gives the values of the first build."""
    clear_jit_cache()
    core = mode == 'core_module'
    first = _evaluate(mode, eos, PhaseParams(), _budget() if core else None)
    traces = dict(_TRACE_COUNTERS)
    _SEEN.clear()
    second = _evaluate(mode, _copy(eos), PhaseParams(), _budget() if core else None)
    assert dict(_TRACE_COUNTERS) == traces
    assert len(_SEEN) == 0
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


CHANGES = {
    'eos_table': lambda e, p, b: (
        eqx.tree_at(
            lambda x: x._temperature_melt._values, e, e._temperature_melt._values * 1.001
        ),
        p,
        b,
    ),
    'phase_params': lambda e, p, b: (
        e,
        eqx.tree_at(lambda x: x.k_liquid, p, p.k_liquid * 2),
        b,
    ),
    'core_profile': lambda e, p, b: (e, p, _budget(rho_cen=12300.0)),
}


@pytest.mark.parametrize('name', list(CHANGES))
def test_each_input_reaches_the_reused_program(eos, name):
    """An input changed at the same shape reaches the program compiled for the first inputs: the
    values differ from the first ones and equal those of a build from empty caches."""
    clear_jit_cache()
    mode = 'core_module'
    first = _evaluate(mode, eos, PhaseParams(), _budget())
    changed = CHANGES[name](eos, PhaseParams(), _budget())
    got = _evaluate(mode, *changed)
    clear_jit_cache()
    jax.clear_caches()
    want = _evaluate(mode, *changed)
    assert not np.array_equal(got[0], first[0])
    assert np.max(np.abs(got[0] - want[0])) <= 1e-14 * np.max(np.abs(want[0]))
    assert np.max(np.abs(got[1] - want[1])) <= 2e-12 * np.max(np.abs(want[1]))


@pytest.mark.parametrize('mode', MODES)
def test_no_argument_of_the_programs_is_weak_typed(eos, mode):
    """Every float leaf of the program arguments (the data and the phase parameters, EOS and
    budget) is a strong float64, so builds from Python and NumPy floats pass one type."""
    mesh = _make_mesh(N=N)
    rhs, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos,
        phase_params=PhaseParams(),
        mesh_arrays=mesh,
        boundary_params=_make_bc(mesh, outer_type=4, inner_type=1, outer_val=10.0),
        heating_array=np.zeros(N),
        scales=_scales(mode),
        core_bc_mode=mode,
        core_module_budget=_budget() if mode == 'core_module' else None,
        core_module_ra_crit_cmb=450.0 if mode == 'core_module' else None,
    )
    leaves = jax.tree_util.tree_leaves(rhs.program_args)
    floats = [x for x in leaves if np.issubdtype(np.asarray(x).dtype, np.floating)]
    assert len(floats) > 20
    for x in floats:
        assert not jax.numpy.asarray(x).weak_type and np.asarray(x).dtype == np.float64


def test_clear_jit_cache_drops_the_traced_programs(eos):
    """After ``clear_jit_cache`` a build of the same form traces its programs again, so a test
    that counts traces does not depend on the tests before it."""
    _evaluate('quasi_steady', eos, PhaseParams())
    clear_jit_cache()
    _evaluate('quasi_steady', eos, PhaseParams())
    assert _TRACE_COUNTERS['rhs'] == 1
    assert _TRACE_COUNTERS['jac'] == 1
