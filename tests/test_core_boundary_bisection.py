"""The inner-core boundary bisection of ``aragog.core.budget.CoreEnergyBudget`` stops early.

Module under test: ``CoreEnergyBudget._r_icb_bisect`` through ``r_icb``, ``r_icb_batch``,
``dtcmb_dt`` and their derivatives. Invariants: the bisection stops when its bracket no longer
changes and makes no step for a liquid centre, and every value and derivative equals that of
the fixed 80-step bisection to the bit (reverse mode included, as the solver Jacobian uses).
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
    [(4400.0, 1), (3000.0, 60), (3915.0, 60), (4100.0, 60), (4146.0, 62)],
)
def test_the_bisection_stops_at_its_fixed_point(monkeypatch, t_cmb, most):
    """Superheat evaluations of one boundary solve: 1 for a liquid centre (above the onset near
    4147 K), at most about 60 with an inner core or a frozen core, not the 81 of 80 fixed
    steps; the root is unchanged."""
    calls = []
    original = CoreEnergyBudget.superheat

    def counted(self, r, t):
        jax.debug.callback(lambda: calls.append(1))
        return original(self, r, t)

    monkeypatch.setattr(CoreEnergyBudget, 'superheat', counted)
    budget = _pair()
    radius = jax.jit(budget._r_icb_bisect)(t_cmb)
    jax.effects_barrier()
    assert 1 <= len(calls) <= most
    monkeypatch.setattr(CoreEnergyBudget, 'superheat', original)
    assert _bits(radius, jax.jit(_pair(_FixedSteps)._r_icb_bisect)(t_cmb))


_TEMPERATURES = st.one_of(
    st.floats(min_value=1.0, max_value=8000.0),
    st.floats(min_value=FREEZE - 1.0, max_value=ONSET + 1.0),
    st.sampled_from([float('nan'), float('inf'), -float('inf'), 0.0, -1.0, ONSET, FREEZE]),
)


@settings(max_examples=80, deadline=None)
@given(t_cmb=_TEMPERATURES)
@example(t_cmb=4100.0)
@example(t_cmb=float(np.nextafter(ONSET, 0.0)))
def test_the_early_stop_gives_the_bits_of_80_steps(t_cmb):
    """r_icb (eager and with a traced temperature), its batch, and its gradient equal the
    80-step reference to the bit, for liquid, partly frozen and frozen cores and for NaN and
    infinite temperatures."""
    for fn in (
        lambda b: b.r_icb(t_cmb),
        lambda b: jax.jit(b.r_icb)(t_cmb),
        lambda b: b.r_icb_batch(jnp.array([t_cmb, 4100.0, 4400.0])),
    ):
        assert _bits(fn(BUDGET), fn(REFERENCE))
    if np.isfinite(t_cmb):
        assert _bits(jax.grad(BUDGET.r_icb)(t_cmb), jax.grad(REFERENCE.r_icb)(t_cmb))


@pytest.mark.parametrize('t_cmb', [3915.0, 4100.0, 4146.0, 4400.0])
@pytest.mark.parametrize('jac', [jax.jacrev, jax.jacfwd])
def test_the_core_rate_jacobian_equals_the_80_step_version(jac, t_cmb):
    """The derivative of dtcmb_dt in the CMB temperature and heat flow, in reverse mode (the
    solver Jacobian) and forward mode, equals the 80-step reference to the bit, with an inner
    core (3915 to 4146 K) and with a liquid centre (4400 K)."""

    def rate(b):
        return jax.jit(jac(lambda x: b.dtcmb_dt(x[0], x[1], q_sources=1e12)))

    x = jnp.array([t_cmb, -8.0e12])
    got, want = rate(BUDGET)(x), rate(REFERENCE)(x)
    assert np.all(np.isfinite(want)) and np.any(want != 0.0)
    assert _bits(got, want)
