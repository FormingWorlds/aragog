"""The inner-core boundary solve of ``aragog.core.budget.CoreEnergyBudget`` compiles once.

Modules under test: ``CoreEnergyBudget._r_icb_solve`` through ``r_icb``,
``aragog.core.regime.refuse_unmodelled_regime`` and ``aragog.solver.entropy_solver._padded_call``.
Invariants: after the first calls, an eager call at a new CMB temperature or a new number of
states reuses the compiled programs, so a long core_module run does not add an XLA program at
every step; the padding to a length bucket changes no value; with jit off every compiled
budget program runs eagerly and gives its compiled value. Compiles are counted with the
backend-compile events of ``jax.monitoring``.
"""

from __future__ import annotations

import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np
import pytest

from aragog.core.budget import CoreEnergyBudget
from aragog.core.melting import IronMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles

pytestmark = pytest.mark.unit

_COMPILE = '/jax/core/compile/backend_compile_duration'
_SEEN = []


def _listen(name, _duration, **_kw):
    if name == _COMPILE:
        _SEEN.append(name)


jax.monitoring.register_event_duration_secs_listener(_listen)


def _compiles(calls):
    """Number of backend compiles while ``calls()`` runs."""
    _SEEN.clear()
    calls()
    return len(_SEEN)


@pytest.fixture
def budget():
    """A fresh Earth-like alloy budget in the partial-inner-core regime (onset near 4147 K), with
    no compiled program left: budgets of one form share them, so an earlier test can warm them."""
    jax.clear_caches()
    prof = GaussianCoreProfiles(
        rho_cen=12500.0,
        length_scale=7200e3,
        r_cmb=3480e3,
        p_cmb=136e9,
        alpha=1.35e-5,
        c_p=840.0,
    )
    curve = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    return CoreEnergyBudget(prof, curve, ds_fusion=170.0, icn_width=10.0)


def test_the_counter_sees_a_new_compile():
    """Canary: a jitted function of a new shape compiles once, and the counter sees it."""
    f = jax.jit(lambda x: x * 2.0 + 1.0)
    x, y = jnp.zeros(7), jnp.ones(7)
    assert _compiles(lambda: f(x)) == 1
    assert _compiles(lambda: f(y)) == 0


def test_eager_boundary_solves_at_new_temperatures_reuse_the_compiled_loop(budget):
    """After the warm-up calls, 5 calls at new CMB temperatures (array and float inputs) compile
    nothing."""
    budget.r_icb(jnp.float64(4100.0))
    budget.r_icb(4100.0)
    temperatures = (4000.0, 3990.0, 3980.0, 3970.0, 3960.0)
    assert _compiles(lambda: [budget.r_icb(jnp.float64(t)) for t in temperatures]) == 0
    assert _compiles(lambda: [budget.r_icb(t) for t in temperatures]) == 0


def test_a_second_budget_of_the_same_form_reuses_the_boundary_solve(budget):
    """After a scalar boundary solve on one budget, the same solve on a budget of the same form
    with other values (a new structure) compiles nothing."""
    budget.r_icb(4100.0)
    budget.r_icb(jnp.float64(4100.0))
    prof = GaussianCoreProfiles(
        rho_cen=12400.0,
        length_scale=7150e3,
        r_cmb=3470e3,
        p_cmb=135e9,
        alpha=1.35e-5,
        c_p=840.0,
    )
    curve = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    other = CoreEnergyBudget(prof, curve, ds_fusion=170.0, icn_width=10.0)
    assert _compiles(lambda: [other.r_icb(4050.0), other.r_icb(jnp.float64(4050.0))]) == 0


def test_the_regime_check_compiles_once_per_power_of_two_bucket(budget):
    """After a first call, the batch lengths 2 to 512 compile exactly once per new bucket: the 6
    buckets 16, 32, 64, 128, 256 and 512 (lengths 2 to 8 share the bucket of the first call)."""
    from aragog.core.regime import refuse_unmodelled_regime

    t = np.linspace(4150.0, 4400.0, 512)  # fully liquid: above the onset near 4147 K
    refuse_unmodelled_regime(budget, t[:1])
    assert (
        _compiles(lambda: [refuse_unmodelled_regime(budget, t[:n]) for n in range(2, 513)]) == 6
    )


def test_the_shell_base_check_compiles_once_per_length_bucket(budget):
    """The shell-base check of the solver runs the batched boundary solve on all states of a
    call; after calls of 101 states, calls of 102 to 128 states compile nothing."""
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver

    budget.shell = SimpleNamespace(r_base=3.0e6, r_faces=np.linspace(3.0e6, 3.48e6, 10))
    solver = SimpleNamespace()
    t, base = np.linspace(4150.0, 4400.0, 128), np.full(128, 3.4e6)
    EntropySolver._check_shell_base(solver, budget, t[:101], base[:101])
    check = EntropySolver._check_shell_base
    assert (
        _compiles(lambda: [check(solver, budget, t[:n], base[:n]) for n in (102, 110, 128)])
        == 0
    )


def test_the_stratified_core_heat_compiles_once_per_length_bucket():
    """The stratified core heat of the solver finds the layer base of every state of a call;
    after a call of 33 states, calls of 34 to 64 states (one bucket) compile nothing, and the
    effective capacity of each state follows its own layer base."""
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_core_stratified_budget import _budget, _layered_shell

    budget = _budget(stratification=True)
    cells = budget.shell.n_cells
    solver = SimpleNamespace(_n_stag=1, _check_shell_base=lambda *a: None)

    def call(n):
        y = np.zeros((3 + cells, n))
        y[2] = np.linspace(4100.0, 4200.0, n)
        y[3:] = np.asarray(_layered_shell(budget))[:, None] + np.linspace(0.0, 5.0, n)
        return y, EntropySolver._core_heat_change(solver, budget, y, np.zeros_like(y))[1]

    call(33)
    assert _compiles(lambda: [call(n) for n in (34, 40, 64)]) == 0
    # The capacity follows the layer base of each state, found by the eager vmap of main.
    y, c_eff = call(40)
    base = jax.vmap(budget.shell.layer_base)(y[3:].T, y[2])
    want = [
        float(budget.effective_capacity(t, gravitational_upper=u)) for t, u in zip(y[2], base)
    ]
    np.testing.assert_allclose(c_eff, want, rtol=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('n', [1, 5, 8, 9, 33, 100])
def test_padding_changes_no_value(budget, n):
    """For states across T_freeze and T_onset, the padding to a length bucket changes no value:
    the regime codes equal the unpadded codes, and at the bucket length the boundary radius,
    regime codes and layer base of the real states do not depend on the padding values. The
    unpadded call is another compiled program: the layer base equals it to rtol 1e-13 (from jax
    0.12 XLA rounds a batched program by its length). The radius equals it at 0 and r_cmb, and
    elsewhere within the final bracket width plus the root shift of a superheat round-off of 16
    ulp of T, min(dS / |S'|, sqrt(dS / |a|)): the root search steps by superheat values, so a
    program that rounds its steps differently ends at another point of that round-off band."""
    import contextlib

    from aragog.core.regime import refuse_unmodelled_regime
    from aragog.solver.entropy_solver import _padded_call
    from tests.test_core_stratified_budget import _budget, _layered_shell

    def across(b):
        return np.linspace(b.t_freeze - 30.0, b.t_onset + 30.0, max(n, 2))[-n:]

    def refill(a, m, row):
        """``a`` padded to length ``m`` with its row ``row`` in place of its last row."""
        return np.concatenate([a, np.repeat(a[[row]], m - len(a), axis=0)])

    t, m = across(budget), 1 << (n - 1).bit_length()
    radius = _padded_call(budget.r_icb_batch, t)
    for row in (0, n // 2):
        assert np.array_equal(radius, np.asarray(budget.r_icb_batch(refill(t, m, row)))[:n])
    unpadded, r_cmb = np.asarray(budget.r_icb_batch(t)), budget.profiles.r_cmb
    s_r = jax.grad(budget.superheat)
    slope = np.abs(np.asarray(jax.vmap(s_r)(unpadded, t)))
    a = np.abs(np.asarray(jax.vmap(jax.grad(s_r))(unpadded, t))) / 2.0
    d_s = 16.0 * np.finfo(float).eps * t
    with np.errstate(divide='ignore'):
        shift = np.minimum(d_s / slope, np.sqrt(d_s / a))
    edge = (unpadded == 0.0) | (unpadded == r_cmb)
    assert np.array_equal(radius[edge], unpadded[edge])
    bound = 2.0**-53 * r_cmb + np.spacing(r_cmb) + shift
    assert np.all(np.abs(radius - unpadded)[~edge] <= bound[~edge])

    real, seen = budget.regime_batch, []
    budget.regime_batch = lambda x: seen.append(np.asarray(real(x))) or seen[-1]
    with contextlib.suppress(ValueError):  # states between the two may be refused
        refuse_unmodelled_regime(budget, t)
    assert np.array_equal(seen[0][:n], real(t))
    assert np.array_equal(seen[0][:n], np.asarray(real(refill(t, max(8, m), 0)))[:n])

    strat = _budget(stratification=True)
    t_core = across(strat)
    t_shell = np.asarray(_layered_shell(strat))[None, :] + np.linspace(0.0, 5.0, n)[:, None]
    layer_base = jax.vmap(strat.shell.layer_base)
    base = _padded_call(layer_base, t_shell, t_core)
    other = layer_base(refill(t_shell, m, 0), refill(t_core, m, 0))
    assert np.array_equal(base, np.asarray(other)[:n])
    np.testing.assert_allclose(base, layer_base(t_shell, t_core), rtol=1e-13, atol=0)


def test_the_budget_programs_give_the_compiled_values_without_jit(budget):
    """With jit off (jax.disable_jit, JAX_DISABLE_JIT) every compiled budget program runs
    eagerly and gives its compiled value: the boundary radius, the cooling rate and the batches."""
    t_cmb, q_cmb, temps = 4000.0, 1.0e13, jnp.array([3900.0, 4000.0, 4100.0])

    def values():
        return [
            np.asarray(v, dtype=float)
            for v in (
                budget.r_icb(t_cmb),
                budget.rate(t_cmb, q_cmb),
                budget.r_icb_batch(temps),
                budget.regime_batch(temps),
                budget.capacity_batch(temps),
            )
        ]

    compiled = values()
    try:
        with jax.disable_jit():
            eager = values()
    except RecursionError as exc:
        pytest.fail(f'a budget program recursed with jit off: {exc}')
    assert 0.0 < compiled[0] < budget.profiles.r_cmb
    assert compiled[1] < 0.0
    for c, e in zip(compiled, eager, strict=True):
        np.testing.assert_allclose(e, c, rtol=1e-12, atol=0.0)


def test_the_core_heat_compiles_once_per_length_bucket(budget):
    """Without stratification, after a call of 33 states the core heat of calls of 34 to 64
    states (one bucket) compiles nothing, and each capacity is the effective capacity."""
    from types import SimpleNamespace

    from aragog.solver.entropy_solver import EntropySolver

    solver = SimpleNamespace(_n_stag=1)

    def call(n):
        y = np.zeros((3, n))
        y[2] = np.linspace(4100.0, 4200.0, n)
        return y, EntropySolver._core_heat_change(solver, budget, y, np.zeros_like(y))[1]

    call(33)
    assert _compiles(lambda: [call(n) for n in (34, 40, 64)]) == 0
    y, cap = call(40)
    want = [float(budget.effective_capacity(t)) for t in y[2]]
    np.testing.assert_allclose(cap, want, rtol=1e-12)
