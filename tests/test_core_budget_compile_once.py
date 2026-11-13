"""The inner-core boundary solve of ``aragog.core.budget.CoreEnergyBudget`` compiles once.

Modules under test: ``CoreEnergyBudget._r_icb_bisect`` through ``r_icb``, and
``aragog.core.regime.refuse_unmodelled_regime``. Invariant: after the first calls, an eager call
at a new CMB temperature or a new number of states reuses the compiled programs, so a long
core_module run does not add an XLA program at every step. Compiles are counted with the
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
_COUNT = {'on': False, 'n': 0}


def _listen(name, _duration, **_kw):
    if _COUNT['on'] and name == _COMPILE:
        _COUNT['n'] += 1


jax.monitoring.register_event_duration_secs_listener(_listen)


def _compiles(calls):
    """Number of backend compiles while ``calls()`` runs."""
    _COUNT.update(on=True, n=0)
    try:
        calls()
    finally:
        _COUNT['on'] = False
    return _COUNT['n']


@pytest.fixture
def budget():
    """A fresh Earth-like alloy budget in the partial-inner-core regime (onset near 4147 K)."""
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
    """After 2 warm-up calls, 5 calls at new CMB temperatures (array and float inputs) compile
    nothing."""
    for t in (4100.0, 4050.0):
        budget.r_icb(jnp.float64(t))
        budget.r_icb(t)
    temperatures = (4000.0, 3990.0, 3980.0, 3970.0, 3960.0)
    assert _compiles(lambda: [budget.r_icb(jnp.float64(t)) for t in temperatures]) == 0
    assert _compiles(lambda: [budget.r_icb(t) for t in temperatures]) == 0


def test_the_regime_check_compiles_once_per_length_bucket(budget):
    """After calls of 101 and 159 states, calls of 120, 150 and 200 states (inside the same
    power-of-two buckets) compile nothing."""
    from aragog.core.regime import refuse_unmodelled_regime

    t = np.linspace(4150.0, 4400.0, 200)  # fully liquid: above the onset near 4147 K
    for n in (101, 159):
        refuse_unmodelled_regime(budget, t[:n])
    assert (
        _compiles(lambda: [refuse_unmodelled_regime(budget, t[:n]) for n in (120, 150, 200)])
        == 0
    )
