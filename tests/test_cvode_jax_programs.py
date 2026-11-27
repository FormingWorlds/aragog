"""The JAX CVODE programs compile once per form of their inputs, not once per input object.

Module under test: ``aragog.solver.cvode_jax.build_jax_rhs_and_jacobian``. Invariants: a
structure re-solve builds new phase parameters, a new EOS object and (core_module) a new core
budget of the same form; the RHS and the Jacobian built from them run the programs compiled for
the first set, and every input reaches those programs, so their values equal those of a build
from empty caches. Compiles are counted by the trace counters of the module and the
backend-compile events of ``jax.monitoring``.
"""

from __future__ import annotations

import functools
import os

import pytest

pytest.importorskip('jax')
eqx = pytest.importorskip('equinox')

import jax  # noqa: E402
import jax.monitoring  # noqa: E402
import numpy as np  # noqa: E402

from aragog.jax.eos import StaticArray  # noqa: E402
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


def _budget(rho_cen=12500.0, r_cmb=3.48e6):
    from aragog.core import build_core_module_budget

    return build_core_module_budget(
        {
            'melting_curve': 'iron',
            'light_element_fraction': 0.1,
            'depression': 1.2,
            'rho_cen': rho_cen,
        },
        r_cmb=r_cmb,
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


def _fitted(i=0, curve='iron'):
    """A core budget of a melting curve of ``CURVES``, fitted to one core mass and centre
    pressure at the i-th CMB of a series that moves as a structure update moves it."""
    from aragog.core import build_core_module_budget

    return build_core_module_budget(
        {**CURVES[curve], 'm_core': 1.94e24, 'p_cen': 3.6e11},
        r_cmb=3.48e6 * (1 - 1e-3 * i),
        p_cmb_fallback=135e9 * (1 + 1e-3 * i),
    )


def _factory(mode, eos_jax, params, budget=None):
    """RHS and Jacobian functions of a factory built from these inputs, on one mesh."""
    mesh = _make_mesh(N=N)
    return build_jax_rhs_and_jacobian(
        eos_jax=eos_jax,
        phase_params=params,
        mesh_arrays=mesh,
        boundary_params=_make_bc(mesh, outer_type=4, inner_type=1, outer_val=10.0),
        heating_array=np.zeros(N),
        scales=_scales(mode),
        core_bc_mode=mode,
        core_module_budget=budget,
        core_module_ra_crit_cmb=450.0 if mode == 'core_module' else None,
    )[:2]


def _evaluate(mode, eos_jax, params, budget=None):
    """RHS and Jacobian of a factory built from these inputs, on one mesh and one state."""
    rhs, jac = _factory(mode, eos_jax, params, budget)
    n = N + EXTRA[mode]
    f, J = np.zeros(n), np.zeros((n, n))
    assert rhs(0.0, _state(mode), f) == 0
    assert jac(0.0, _state(mode), None, J) == 0
    return f, J


def _copy(tree):
    """A new object with the same leaves, as a re-solve builds."""
    return jax.tree_util.tree_map(lambda x: x, tree)


def _replaced(tree, changes):
    """``tree`` with the leaves at the attribute paths of ``changes`` set to their values."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    names = [tuple(k.name for k in p) for p, _ in leaves]
    assert set(changes) <= set(names), set(changes) - set(names)
    return jax.tree_util.tree_unflatten(
        treedef, [changes.get(n, x) for n, (_, x) in zip(names, leaves)]
    )


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


@pytest.mark.parametrize('source', ['copy', 'disk'])
@pytest.mark.parametrize('mode', ['energy_balance', 'core_module'])
def test_a_second_build_calls_the_programs_by_identity(eos, mode, source, monkeypatch):
    """A factory built again from equal inputs in new objects (an EOS copy, or the same tables
    parsed again from disk) runs the cached programs and matches them by identity at every
    call: no ``_ByValue`` or ``StaticArray`` is compared by value per call."""
    from aragog.jax.eos import EntropyEOS_JAX

    core = mode == 'core_module'
    clear_jit_cache()
    _evaluate(mode, eos, PhaseParams(), _budget() if core else None)
    second = _copy(eos) if source == 'copy' else EntropyEOS_JAX(EOS_DIR)
    rhs, jac = _factory(mode, second, PhaseParams(), _budget() if core else None)
    assert _compared_per_call(mode, rhs, jac, monkeypatch) == []


def _compared_per_call(mode, rhs, jac, monkeypatch):
    """The classes of the static inputs that 5 calls (after a first one, which must trace
    nothing) compare by value."""
    from aragog.solver import cvode_jax

    n = N + EXTRA[mode]
    f, J = np.zeros(n), np.zeros((n, n))
    traces = dict(_TRACE_COUNTERS)
    assert rhs(0.0, _state(mode), f) == 0 and jac(0.0, _state(mode), None, J) == 0
    assert dict(_TRACE_COUNTERS) == traces
    compared = []
    for cls in (cvode_jax._ByValue, StaticArray):
        monkeypatch.setattr(
            cls, '__eq__', lambda a, b, eq=cls.__eq__: compared.append(type(a)) or eq(a, b)
        )
    for _ in range(5):
        assert rhs(0.0, _state(mode), f) == 0 and jac(0.0, _state(mode), None, J) == 0
    return compared


@pytest.mark.parametrize('first', ['rhs', 'jac'])
def test_a_build_puts_its_leaves_on_the_device_at_its_first_call_only(eos, first, monkeypatch):
    """A build leaves its traced leaves as they are; its first call, RHS or Jacobian, puts them
    all on the device, and every later call converts only the state."""
    _evaluate('core_module', eos, PhaseParams(), _budget())
    seen, real = [], jax.numpy.asarray
    monkeypatch.setattr(
        jax.numpy, 'asarray', lambda x, *a, **k: seen.append(1) or real(x, *a, **k)
    )
    rhs, jac = _factory('core_module', _copy(eos), PhaseParams(), _budget())
    leaves = rhs.__closure__[rhs.__code__.co_freevars.index('leaves')].cell_contents
    assert not all(isinstance(x, jax.Array) for x in leaves)
    n = N + EXTRA['core_module']
    f, J = np.zeros(n), np.zeros((n, n))
    calls = {
        'rhs': lambda: rhs(0.0, _state('core_module'), f) == 0,
        'jac': lambda: jac(0.0, _state('core_module'), None, J) == 0,
    }
    second = 'jac' if first == 'rhs' else 'rhs'

    def conversions(name):
        before = len(seen)
        assert calls[name]()
        return len(seen) - before

    assert conversions(first) > 1
    assert all(isinstance(x, jax.Array) for x in leaves)
    assert [conversions(name) for name in (second, first, second)][1:] == [1, 1]


def test_every_budget_constant_names_a_leaf_of_a_budget():
    """Each attribute path of ``_BUDGET_CONSTANTS`` is a leaf path of one of the budgets of
    ``CURVES``, so a renamed field cannot leave the set without a test failing."""
    from aragog.solver import cvode_jax

    names = {
        tuple(k.name for k in path)
        for curve in CURVES
        for path, _ in jax.tree_util.tree_flatten_with_path(_fitted(0, curve))[0]
    }
    assert cvode_jax._BUDGET_CONSTANTS <= names
    assert STRUCTURE <= names


def test_a_rebuild_after_a_bound_clear_calls_the_programs_by_identity(eos, monkeypatch):
    """After a new form past ``MAX_FORMS`` clears the programs, a factory built again from equal
    inputs in new objects matches the program of that form by identity."""
    from aragog.solver import cvode_jax

    monkeypatch.setattr(cvode_jax, 'MAX_FORMS', 1)
    clear_jit_cache()
    mode = 'energy_balance'
    _evaluate(mode, eos, PhaseParams())
    _evaluate(mode, eos, PhaseParams(convection=0))
    rhs, jac = _factory(mode, _copy(eos), PhaseParams(convection=0))
    assert _compared_per_call(mode, rhs, jac, monkeypatch) == []


def test_a_refitted_core_profile_reuses_the_programs(eos):
    """Core budgets fitted to one core mass at a moving CMB radius and pressure (as a structure
    update refits them) run the programs of the first budget, with no trace and no compile,
    and give the values of a build from empty caches."""
    clear_jit_cache()
    mode = 'core_module'
    budgets = [_fitted(i) for i in range(4)]
    for name in ('rho_cen', 'length_scale', 'r_cmb', 'p_cmb'):
        assert len({float(getattr(b.profiles, name)) for b in budgets}) == 4, name
    first = _evaluate(mode, eos, PhaseParams(), budgets[0])
    traces = dict(_TRACE_COUNTERS)
    _SEEN.clear()
    got = [_evaluate(mode, _copy(eos), PhaseParams(), b) for b in budgets[1:]][-1]
    assert dict(_TRACE_COUNTERS) == traces
    assert len(_SEEN) == 0
    clear_jit_cache()
    jax.clear_caches()
    want = _evaluate(mode, eos, PhaseParams(), budgets[-1])
    assert not np.array_equal(got[0], first[0])
    assert np.max(np.abs(got[0] - want[0])) <= 1e-14 * np.max(np.abs(want[0]))
    assert np.max(np.abs(got[1] - want[1])) <= 2e-12 * np.max(np.abs(want[1]))


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
    'core_constant': lambda e, p, b: (
        e,
        p,
        _replaced(b, {('ds_fusion',): b.ds_fusion * 1.1, ('profiles', 'c_p'): 900.0}),
    ),
}


@pytest.mark.slow
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


def test_a_new_phase_or_eos_value_is_a_new_input_form(eos):
    """The phase parameters and the EOS table values are constants of the programs: a changed
    value of either is a new input form of the bound, an equal value in a new object is not."""
    from aragog.solver import cvode_jax

    clear_jit_cache()
    mode = 'energy_balance'
    _factory(mode, eos, PhaseParams())
    _factory(mode, _copy(eos), PhaseParams())
    assert len(cvode_jax._FORMS) == 1
    _factory(mode, *CHANGES['eos_table'](eos, PhaseParams(), None)[:2])
    assert len(cvode_jax._FORMS) == 2
    _factory(mode, *CHANGES['phase_params'](eos, PhaseParams(), None)[:2])
    assert len(cvode_jax._FORMS) == 3


def test_a_new_table_value_compiles_once_and_an_equal_table_reuses_it(eos):
    """A changed EOS table value compiles the RHS and the Jacobian once, a new object with the
    changed values reuses that program, and the first tables parsed again from disk reuse the
    first program."""
    from aragog.jax.eos import EntropyEOS_JAX

    clear_jit_cache()
    mode = 'energy_balance'
    changed = CHANGES['eos_table'](eos, PhaseParams(), None)[0]
    _evaluate(mode, eos, PhaseParams())
    first = dict(_TRACE_COUNTERS)
    _evaluate(mode, changed, PhaseParams())
    assert (_TRACE_COUNTERS['rhs'], _TRACE_COUNTERS['jac']) == (
        first['rhs'] + 1,
        first['jac'] + 1,
    )
    traces = dict(_TRACE_COUNTERS)
    _evaluate(mode, _copy(changed), PhaseParams())
    _evaluate(mode, EntropyEOS_JAX(EOS_DIR), PhaseParams())
    assert dict(_TRACE_COUNTERS) == traces


# Budget leaves that follow the structure (the profile fit and the CMB); all others are config.
STRUCTURE = {
    ('profiles', 'rho_cen'),
    ('profiles', 'length_scale'),
    ('profiles', 'd_scale'),
    ('profiles', 'r_cmb'),
    ('profiles', 'p_cmb'),
    ('r_convecting',),
}
CURVES = {
    'iron': {'melting_curve': 'iron', 'light_element_fraction': 0.1, 'depression': 1.2},
    'quadratic': {'melting_curve': 'quadratic', 't_m0': 5000.0, 't_m1': 1e-8, 't_m2': -1e-20},
    'latent_heat': {'melting_curve': 'iron', 'latent_heat': 7.5e5},
}


def _budget_leaves():
    for curve in CURVES:
        budget = _fitted(0, curve)
        for path, _ in jax.tree_util.tree_flatten_with_path(budget)[0]:
            names = tuple(k.name for k in path)
            yield pytest.param(budget, names, id=f'{curve}-{".".join(names)}')


@pytest.mark.parametrize(('budget', 'path'), list(_budget_leaves()))
def test_each_budget_leaf_is_a_constant_or_follows_the_structure(eos, budget, path):
    """A changed config leaf of the core budget is a new input form (a compile-time constant);
    a changed structure leaf is the same form (traced), so a refit compiles nothing."""
    from aragog.solver import cvode_jax

    value = float(functools.reduce(getattr, path, budget))
    changed = _replaced(budget, {path: value * 1.01 + 1e-3})
    clear_jit_cache()
    _factory('core_module', eos, PhaseParams(), budget)
    _factory('core_module', eos, PhaseParams(), changed)
    assert len(cvode_jax._FORMS) == (1 if path in STRUCTURE else 2)


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
    from aragog.solver import cvode_jax

    _evaluate('quasi_steady', eos, PhaseParams())
    clear_jit_cache()
    assert cvode_jax._HELD == {}
    assert cvode_jax._FORMS == set()
    _evaluate('quasi_steady', eos, PhaseParams())
    assert _TRACE_COUNTERS['rhs'] == 1
    assert _TRACE_COUNTERS['jac'] == 1


def test_a_by_value_holder_keeps_a_read_only_copy_of_its_source():
    """A holder copies its leaves: the source stays writable, a later change of the source does
    not reach the holder, the holder refuses a new attribute, and it equals itself and a new
    holder of the first values."""
    from aragog.solver import cvode_jax

    source = np.array([1.0, 2.0])
    held = cvode_jax._ByValue((source,))
    assert source.flags.writeable and not held.tree[0].flags.writeable
    source[0] = 5.0
    assert held.tree[0][0] == 1.0
    assert held == cvode_jax._ByValue((np.array([1.0, 2.0]),))
    assert held != cvode_jax._ByValue((source,))
    assert held == held
    with pytest.raises(AttributeError, match='cannot change'):
        held.tree = (source,)


def _rhs_at(eos, n):
    """RHS of a quasi_steady factory on an ``n``-node mesh (one input form per ``n``)."""
    mesh = _make_mesh(N=n)
    rhs, _, _ = build_jax_rhs_and_jacobian(
        eos_jax=eos,
        phase_params=PhaseParams(),
        mesh_arrays=mesh,
        boundary_params=_make_bc(mesh, outer_type=4, inner_type=1, outer_val=10.0),
        heating_array=np.zeros(n),
        scales=NonDimScales(state_scale=np.full(n, 3.0e3), t_ref=1.0),
        core_bc_mode='quasi_steady',
    )
    f = np.zeros(n)
    assert rhs(0.0, np.linspace(3100.0, 2900.0, n) / 3.0e3, f) == 0
    return f


def test_the_program_caches_hold_at_most_max_forms_input_forms(eos, monkeypatch):
    """A known form reuses its program; a new form past MAX_FORMS clears the programs first, so
    the caches stay bounded, and the new form still gives the values of a build from empty
    caches."""
    from aragog.solver import cvode_jax

    monkeypatch.setattr(cvode_jax, 'MAX_FORMS', 2)
    clear_jit_cache()
    _rhs_at(eos, 6)
    _rhs_at(eos, 7)
    traces = _TRACE_COUNTERS['rhs']
    _rhs_at(eos, 6)
    assert _TRACE_COUNTERS['rhs'] == traces
    third = _rhs_at(eos, 8)
    rhs_program = cvode_jax._programs()[0]
    assert rhs_program._cache_size() == 1
    assert len(cvode_jax._FORMS) == 1
    with jax.disable_jit():
        np.testing.assert_allclose(third, _rhs_at(eos, 8), rtol=1e-10, atol=0.0)


class _Grid(eqx.Module):
    """A pytree whose only field is a static grid, as the EOS tables hold theirs."""

    grid: StaticArray = eqx.field(static=True)


def test_each_part_of_the_input_form_counts():
    """The form key of the bound holds the core_bc mode, the radiogenic switch, the static fields
    (a grid by value), the shape and the dtype of each leaf: changing any one of them is a new
    form, the same inputs are not."""
    from aragog.solver import cvode_jax

    clear_jit_cache()
    data, consts = (np.zeros(3),), (np.zeros(2),)
    forms = [
        ('quasi_steady', False, data, consts),
        ('quasi_steady', False, (np.ones(3),), (np.ones(2),)),
        ('energy_balance', False, data, consts),
        ('quasi_steady', True, data, consts),
        ('quasi_steady', False, (np.zeros(4),), consts),
        ('quasi_steady', False, (np.zeros(3, np.float32),), consts),
        ('quasi_steady', False, data, (_Grid(StaticArray([0.0, 1.0])),)),
        ('quasi_steady', False, data, (_Grid(StaticArray([0.0, 2.0])),)),
        ('quasi_steady', False, data, (_Grid(StaticArray([0.0, 2.0])),)),
    ]
    counts = []
    for form in forms:
        cvode_jax._bound_forms(*form)
        counts.append(len(cvode_jax._FORMS))
    clear_jit_cache()
    assert counts == [1, 1, 2, 3, 4, 5, 6, 7, 7]


def test_a_call_at_a_new_time_reuses_the_programs(eos):
    """The time is a traced argument of the RHS and Jacobian programs: calls of one build at
    new times trace and compile nothing."""
    clear_jit_cache()
    mode = 'energy_balance'
    rhs, jac = _factory(mode, eos, PhaseParams())
    n = N + EXTRA[mode]
    f, J = np.zeros(n), np.zeros((n, n))
    assert rhs(0.0, _state(mode), f) == 0 and jac(0.0, _state(mode), None, J) == 0
    traces = dict(_TRACE_COUNTERS)
    _SEEN.clear()
    for t in (1.0, 2.5):
        assert rhs(t, _state(mode), f) == 0 and jac(t, _state(mode), None, J) == 0
    assert dict(_TRACE_COUNTERS) == traces
    assert len(_SEEN) == 0
