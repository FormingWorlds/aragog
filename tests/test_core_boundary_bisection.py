"""The inner-core boundary bisection of ``aragog.core.budget.CoreEnergyBudget`` stops early.

Module under test: ``CoreEnergyBudget._r_icb_bisect`` through ``r_icb``, ``r_icb_batch``,
``dtcmb_dt`` and their derivatives. Invariants: the bisection stops when its bracket no longer
changes and makes no step for a liquid centre; every value equals that of the fixed 80-step
bisection to the bit, and the derivatives agree within the tolerance each test states.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from aragog.core import budget as budget_module
from aragog.core.budget import CoreEnergyBudget
from aragog.core.melting import IronMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles

pytestmark = pytest.mark.unit

EARTH = dict(
    rho_cen=12500.0, length_scale=7200e3, r_cmb=3480e3, p_cmb=136e9, alpha=1.35e-5, c_p=840.0
)


class _FixedSteps(CoreEnergyBudget):
    """The reference: the boundary bisection with all 80 steps."""

    def _r_icb_bisect(self, t_cmb):
        def body(_, carry):
            lo, hi, t = carry
            mid = (lo + hi) / 2.0
            frozen = self.superheat(mid, t) < 0.0
            return jnp.where(frozen, mid, lo), jnp.where(frozen, hi, mid), t

        t = t_cmb * 1.0
        init = (jnp.zeros_like(t), jnp.full_like(t, self.profiles.r_cmb), t)
        lo, hi, _ = jax.lax.fori_loop(0, 80, body, init)
        return jnp.where(self.superheat(0.0, t_cmb) > 0.0, 0.0, (lo + hi) / 2.0)


def _pair(cls=CoreEnergyBudget):
    curve = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    return cls(GaussianCoreProfiles(**EARTH), curve, ds_fusion=170.0, icn_width=10.0)


BUDGET, REFERENCE = _pair(), _pair(_FixedSteps)
ONSET, FREEZE = BUDGET.t_onset, BUDGET.t_freeze


def _bits(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.array_equal(a.view(np.int64), b.view(np.int64))


def test_the_reference_is_the_bisection_of_80_steps():
    """The step count of the module is 80, the length of the reference loop."""
    assert budget_module._BISECT_ITERS == 80


@pytest.mark.parametrize(
    ('t_cmb', 'most'),
    [
        (4400.0, (1, 1)),
        (3000.0, (54, 57)),
        (3915.0, (53, 56)),
        (4100.0, (54, 57)),
        (4146.0, (56, 59)),
        (ONSET - 1e-9, (70, 73)),
        (np.array([4400.0, 3915.0, 4100.0]), (55, 57)),
        (float('nan'), (81, 81)),
    ],
)
def test_the_bisection_stops_at_its_fixed_point(monkeypatch, t_cmb, most):
    """Superheat evaluations of one boundary solve (the guard and the steps): 1 for a liquid
    centre (above the onset near 4147 K), 55 to 58 for a frozen core or an inner core of 0.27e6
    m or more, more near the onset (about 53 + log2(r_cmb / r_icb) steps), and all 81 for a NaN
    temperature; the root is unchanged."""
    calls = []
    original = CoreEnergyBudget.superheat

    def counted(self, r, t):
        jax.debug.callback(lambda: calls.append(1))
        return original(self, r, t)

    monkeypatch.setattr(CoreEnergyBudget, 'superheat', counted)
    budget = _pair()
    radius = jax.jit(budget._r_icb_bisect)(t_cmb)
    jax.effects_barrier()
    assert most[0] <= len(calls) <= most[1]
    monkeypatch.setattr(CoreEnergyBudget, 'superheat', original)
    assert _bits(radius, jax.jit(_pair(_FixedSteps)._r_icb_bisect)(t_cmb))


@pytest.mark.parametrize(
    'order', [[3915.0, 4100.0, 4400.0], [4400.0, 3915.0, 4100.0, ONSET - 1e-9]]
)
@pytest.mark.parametrize('flip', [False, True])
def test_r_icb_on_an_array_equals_the_reference(order, flip):
    """r_icb on an array of temperatures (inner core, inner core, liquid centre) equals the
    80-step reference to the bit; a batch runs until its slowest entry stops."""
    t_cmb = np.array(order[::-1] if flip else order)
    assert _bits(BUDGET.r_icb(t_cmb), REFERENCE.r_icb(t_cmb))
    assert _bits(jax.jit(BUDGET.r_icb)(t_cmb), jax.jit(REFERENCE.r_icb)(t_cmb))


_TEMPERATURES = st.one_of(
    st.floats(min_value=1.0, max_value=8000.0),
    st.floats(min_value=FREEZE - 1.0, max_value=ONSET + 1.0),
    st.sampled_from([float('nan'), float('inf'), -float('inf'), 0.0, -1.0, ONSET, FREEZE]),
)


@settings(max_examples=80, deadline=None)
@given(t_cmb=_TEMPERATURES)
@example(t_cmb=4100.0)
@example(t_cmb=float(np.nextafter(ONSET, 0.0)))
@example(t_cmb=float(np.nextafter(ONSET, np.inf)))
@example(t_cmb=ONSET + 4 * float(np.spacing(ONSET)))
@example(t_cmb=ONSET - 4 * float(np.spacing(ONSET)))
@example(t_cmb=ONSET - 1e-6)
def test_the_early_stop_gives_the_bits_of_80_steps(t_cmb):
    """r_icb (eager and with a traced temperature) and its batch equal the 80-step reference to
    the bit, for liquid, partly frozen and frozen cores and for NaN and infinite temperatures.
    The gradient is a new program: near the onset the superheat crossing is tangent, a
    round-off dS moves the root by dr/r = dS/(2 a r^2), and the gradient may differ by
    1e-12 + (1 m / r_icb)^2 relative; where that bound is 1 or more (r_icb below about 1 m)
    the test checks the sign of the gradient instead."""
    for fn in (
        lambda b: b.r_icb(t_cmb),
        lambda b: jax.jit(b.r_icb)(t_cmb),
        lambda b: b.r_icb_batch(jnp.array([t_cmb, 4100.0, 4400.0])),
    ):
        assert _bits(fn(BUDGET), fn(REFERENCE))
    if np.isfinite(t_cmb):
        got, want = (
            float(jax.grad(BUDGET.r_icb)(t_cmb)),
            float(jax.grad(REFERENCE.r_icb)(t_cmb)),
        )
        radius = float(REFERENCE.r_icb(t_cmb))
        bound = 1e-12 + (1.0 / radius) ** 2 if radius > 0.0 else 0.0
        if bound < 1.0:
            assert abs(got - want) <= bound * abs(want)
        else:
            assert np.isfinite(got) and want < 0.0 and got < 0.0


@pytest.mark.parametrize('t_cmb', [3900.0, 3915.0, 4050.0, 4100.0, 4128.0, 4146.0, 4400.0])
@pytest.mark.parametrize('jac', [jax.jacrev, jax.jacfwd])
def test_the_core_rate_jacobian_equals_the_80_step_version(jac, t_cmb):
    """The derivative of dtcmb_dt in the CMB temperature and heat flow, in reverse mode (the
    solver Jacobian) and forward mode, equals the 80-step reference to 1e-11 relative, with an
    inner core (3900 to 4146 K) and with a liquid centre (4400 K); a new compiled program can
    round differently, so the test does not ask for the bits."""

    def rate(b):
        return jax.jit(jac(lambda x: b.dtcmb_dt(x[0], x[1], q_sources=1e12)))

    x = jnp.array([t_cmb, -8.0e12])
    got, want = rate(BUDGET)(x), rate(REFERENCE)(x)
    assert np.all(np.isfinite(want)) and np.any(want != 0.0)
    assert np.all(np.abs(got - want) <= 1e-11 * np.abs(want))
