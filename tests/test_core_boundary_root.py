"""The inner-core boundary root search of ``aragog.core.budget.CoreEnergyBudget``.

Module under test: ``CoreEnergyBudget._r_icb_bracket`` through ``r_icb``, ``r_icb_batch``,
``dtcmb_dt`` and their derivatives. Invariants: the search keeps a sign-changing bracket of the
superheat and ends with a width of at most 2 eps r_cmb in at most n_1/2 + n0 = 53 + 5 steps
(Oliveira & Takahashi 2020, Algorithm 1, eps = 2^-54 on u = r / r_cmb); the radius equals that
of a fixed 80-step bisection within the round-off of the superheat; the derivatives agree
within the tolerance each test states.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from aragog.core.budget import CoreEnergyBudget
from aragog.core.melting import IronMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles

pytestmark = pytest.mark.unit

EARTH = dict(
    rho_cen=12500.0, length_scale=7200e3, r_cmb=3480e3, p_cmb=136e9, alpha=1.35e-5, c_p=840.0
)
EPS = 2.0**-54
STEPS = math.ceil(math.log2(1.0 / (2.0 * EPS))) + 5


class _FixedSteps(CoreEnergyBudget):
    """The reference: the boundary bisection with all 80 steps."""

    def _r_icb_solve(self, t_cmb):
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
R_CMB = EARTH['r_cmb']
WIDTH = 2.0 * EPS * R_CMB + float(np.spacing(R_CMB))  # u-width in metres, product rounded
SWEEP = np.linspace(FREEZE + 0.5, ONSET - 0.5, 41)
_TEMPERATURES = st.one_of(
    st.floats(min_value=1.0, max_value=8000.0),
    st.floats(min_value=FREEZE - 1.0, max_value=ONSET + 1.0),
    st.sampled_from([float('nan'), float('inf'), -float('inf'), 0.0, -1.0, ONSET, FREEZE]),
)


def _close(got, want, t_cmb):
    """Whether ``got`` is the root ``want`` [m] within round-off: equal, or nearer than the
    final bracket width plus the root shift of a superheat round-off dS = 16 ulp of T, which is
    at most min(dS / |S'|, sqrt(dS / |a|)) for S' + a r^2 about the root (a = S'' / 2)."""
    d_s = 16.0 * np.finfo(float).eps * abs(t_cmb)
    s_r = jax.grad(REFERENCE.superheat)
    slope, a = abs(float(s_r(want, t_cmb))), abs(float(jax.grad(s_r)(want, t_cmb))) / 2.0
    return got == want or abs(got - want) <= WIDTH + min(d_s / slope, np.sqrt(d_s / a))


def _counted(monkeypatch):
    calls = []
    original = CoreEnergyBudget.superheat

    def counted(self, r, t):
        jax.debug.callback(lambda: calls.append(1))
        return original(self, r, t)

    monkeypatch.setattr(CoreEnergyBudget, 'superheat', counted)
    return calls


@settings(max_examples=80, deadline=None)
@given(t_cmb=_TEMPERATURES)
@example(t_cmb=3689.4)
@example(t_cmb=3700.0)
@example(t_cmb=float(np.nextafter(ONSET, 0.0)))
@example(t_cmb=ONSET - 1e-9)
def test_the_search_keeps_a_bracket_and_ends_within_the_bisection_bound(t_cmb):
    """The bracket [lo, hi] keeps superheat(lo) < 0 <= superheat(hi) as the search saw them and
    ends with hi - lo <= 2 eps r_cmb (plus 1 ulp of r_cmb, as each end is u r_cmb rounded) in
    at most n_1/2 + n0 steps (Theorem 2.1 of Oliveira &
    Takahashi 2020); a liquid centre gives [0, 0] and a frozen CMB [r_cmb, r_cmb] with no step;
    a NaN temperature (no sign) shrinks to a radius of at most the width above 0. The case is
    read from end superheats more than 1e-6 K from 0, so that an ulp at the onset cannot put
    the test and the search in different cases."""
    steps, lo, hi, f_lo, f_hi = map(float, jax.jit(BUDGET._r_icb_bracket)(t_cmb))
    assert 0 <= steps <= STEPS and 0.0 <= lo <= hi <= R_CMB
    assert hi - lo <= WIDTH
    centre, top = float(BUDGET.superheat(0.0, t_cmb)), float(BUDGET.superheat(R_CMB, t_cmb))
    if np.isnan(t_cmb):
        assert hi <= WIDTH
    elif centre > 1e-6:
        assert (steps, lo, hi) == (0, 0.0, 0.0)
    elif top < -1e-6:
        assert (steps, lo, hi) == (0, R_CMB, R_CMB)
    elif centre < -1e-6 and top > 1e-6:
        assert f_lo < 0.0 <= f_hi and steps > 0


def test_the_search_makes_few_calls(monkeypatch):
    """Superheat calls of one boundary solve (the 2 ends + 1 per step): 2 for a liquid centre
    and for a frozen CMB, at most 2 + n_1/2 + n0 at any temperature, and a mean of at most 25
    and a maximum of at most 45 over 41 temperatures from freeze-out to the onset (a bisection
    that stops at its fixed point needs 55 on mean, the 80-step bisection 81). The maximum holds by the Illinois halving: without it, the search keeps
    one end near freeze-out until the projection ends it at the bound."""
    calls = _counted(monkeypatch)
    solve = jax.jit(_pair().r_icb)
    counts = []
    for t_cmb in [4400.0, 3000.0, *SWEEP, float('nan')]:
        calls.clear()
        solve(t_cmb)
        jax.effects_barrier()
        counts.append(len(calls))
    assert counts[:2] == [2, 2]
    assert max(counts) <= 2 + STEPS
    assert np.mean(counts[2:-1]) <= 25.0 and max(counts[2:-1]) <= 45


@pytest.mark.parametrize(
    'order', [[3915.0, 4100.0, 4400.0], [4400.0, 3000.0, 3700.0, ONSET - 1e-9]]
)
@pytest.mark.parametrize('flip', [False, True])
def test_r_icb_on_an_array_equals_r_icb_of_each_entry(order, flip):
    """r_icb on an array of temperatures (liquid centre, frozen CMB, inner cores) equals r_icb
    of each entry, eager and compiled, within round-off (``_close``; 2 compiled programs can
    round differently); a batch runs until its slowest
    entry stops, and an entry that has stopped does not move."""
    t_cmb = np.array(order[::-1] if flip else order)
    one = [float(BUDGET.r_icb(t)) for t in t_cmb]
    for got in (BUDGET.r_icb(t_cmb), jax.jit(BUDGET.r_icb)(t_cmb)):
        for t, g, w in zip(t_cmb, np.asarray(got), one, strict=True):
            assert _close(g, w, t)


@settings(max_examples=80, deadline=None)
@given(t_cmb=_TEMPERATURES)
@example(t_cmb=4100.0)
@example(t_cmb=3700.0)
@example(t_cmb=float(np.nextafter(ONSET, 0.0)))
@example(t_cmb=float(np.nextafter(ONSET, np.inf)))
@example(t_cmb=ONSET + 4 * float(np.spacing(ONSET)))
@example(t_cmb=ONSET - 4 * float(np.spacing(ONSET)))
@example(t_cmb=ONSET - 1e-6)
def test_the_root_equals_the_80_step_root_within_round_off(t_cmb):
    """r_icb (eager and with a traced temperature) and its batch equal the 80-step bisection
    within round-off (``_close``): the final bracket width plus the root shift of a superheat
    round-off, which grows near the onset, where the crossing is tangent, and at the guards,
    where 2 programs can disagree by an ulp on a liquid centre or a frozen CMB. The gradient is a new program:
    near the onset a round-off dS moves the root by dr/r = dS/(2 a r^2), and the gradient may
    differ by 1e-12 + (1 m / r_icb)^2 relative; where that bound is 1 or more (r_icb below about
    1 m) the test checks the sign of the gradient instead. The radius comes from the
    derivative program, whose liquid guard can differ from the eager one by an ulp at the
    onset."""
    want = float(jax.jit(REFERENCE.r_icb)(t_cmb))
    for got in (
        BUDGET.r_icb(t_cmb),
        jax.jit(BUDGET.r_icb)(t_cmb),
        BUDGET.r_icb_batch(jnp.array([t_cmb, 4100.0, 4400.0]))[0],
    ):
        got = float(got)
        assert 0.0 <= got <= WIDTH if np.isnan(t_cmb) else _close(got, want, t_cmb)
    if np.isfinite(t_cmb):
        radius, want = map(float, jax.jvp(REFERENCE.r_icb, (t_cmb,), (1.0,)))
        got = float(jax.grad(BUDGET.r_icb)(t_cmb))
        bound = 1e-12 + (1.0 / radius) ** 2 if radius > 0.0 else 0.0
        if bound < 1.0:
            assert abs(got - want) <= bound * abs(want)
        else:
            assert np.isfinite(got) and want < 0.0 and got < 0.0


@pytest.mark.parametrize('t_cmb', [3900.0, 3915.0, 4050.0, 4100.0, 4128.0, 4146.0, 4400.0])
@pytest.mark.parametrize('jac', [jax.jacrev, jax.jacfwd])
def test_the_core_rate_jacobian_equals_the_80_step_version(jac, t_cmb):
    """The derivative of dtcmb_dt in the CMB temperature and heat flow, in reverse mode (the
    solver Jacobian) and forward mode, equals the 80-step reference to 1e-11 of the largest
    term J_k x_k of the linear change (an entry can pass through zero, near 4128 K), with an
    inner core (3900 to 4146 K) and with a liquid centre (4400 K); a new compiled program can
    round differently, so the test does not ask for the bits."""

    def rate(b):
        return jax.jit(jac(lambda x: b.dtcmb_dt(x[0], x[1], q_sources=1e12)))

    x = jnp.array([t_cmb, -8.0e12])
    got, want = rate(BUDGET)(x), rate(REFERENCE)(x)
    assert np.all(np.isfinite(want)) and np.any(want != 0.0)
    scale = np.max(np.abs(want * x)) / np.abs(x)
    assert np.all(np.abs(got - want) <= 1e-11 * scale)
