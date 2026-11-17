"""``aragog.core.budget.CoreEnergyBudget`` as a JAX pytree: one program per budget form.

Modules under test: ``aragog.core.budget`` (the batched functions ``r_icb_batch``,
``regime_batch``, ``_boundary``, ``capacity_batch`` and ``rate``) and the pytree registration
of the profiles, melting curves and budget. Invariants: a second budget of the same form (same
classes, settings and array shapes) runs the programs compiled for the first, so a structure
re-solve compiles nothing; each input of the budget reaches the reused program, so its values
equal those of a program built for that budget alone; no leaf is weak-typed. Compiles are
counted with the backend-compile events of ``jax.monitoring``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np
import pytest

from aragog.core.budget import CoreEnergyBudget
from aragog.core.melting import IronMeltingCurve, QuadraticMeltingCurve
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


@dataclasses.dataclass(frozen=True)
class _Form:
    """The inputs of a Gaussian core budget with an inner core (onset near 4150 K)."""

    rho_cen: float = 12500.0
    length_scale: float = 7200e3
    r_cmb: float = 3480e3
    p_cmb: float = 136e9
    alpha: float = 1.35e-5
    c_p: float = 840.0
    light_element_fraction: float = 0.1
    depression: float = 1.2
    ds_fusion: float = 170.0
    icn_width: float = 10.0
    alpha_c: float = 0.8
    c_light: float = 0.05
    latent_heat: float | None = None

    def budget(self) -> CoreEnergyBudget:
        prof = GaussianCoreProfiles(
            rho_cen=self.rho_cen,
            length_scale=self.length_scale,
            r_cmb=self.r_cmb,
            p_cmb=self.p_cmb,
            alpha=self.alpha,
            c_p=self.c_p,
        )
        curve = IronMeltingCurve(
            light_element_fraction=self.light_element_fraction, depression=self.depression
        )
        return CoreEnergyBudget(
            prof,
            curve,
            ds_fusion=self.ds_fusion,
            icn_width=self.icn_width,
            alpha_c=self.alpha_c,
            c_light=self.c_light,
            latent_heat=self.latent_heat,
        )


T = jnp.linspace(3600.0, 4400.0, 16)


def _batched(b):
    """Every batched budget function at the temperatures ``T``."""
    return (
        b.r_icb_batch(T),
        b.regime_batch(T),
        b._boundary(T),
        b.capacity_batch(T),
        b.rate(jnp.float64(4000.0), jnp.float64(3e12), jnp.float64(1e11)),
    )


def _fresh(b):
    """The same functions compiled for ``b`` alone, its values closed over as constants."""
    from aragog.core.regime import crystallization_regime

    return (
        jax.jit(jax.vmap(b.r_icb))(T),
        jax.jit(jax.vmap(lambda t: crystallization_regime(b, t)))(T),
        jax.jit(jax.vmap(lambda t: b.latent_capacity(t) + b.gravitational_capacity(t)))(T),
        jax.jit(jax.vmap(b.effective_capacity))(T),
        jax.jit(lambda t, q, s: b.dtcmb_dt(t, q, q_sources=s))(
            jnp.float64(4000.0), jnp.float64(3e12), jnp.float64(1e11)
        ),
    )


def test_a_second_budget_of_the_same_form_reuses_every_program():
    """After the calls on one budget, the same calls on a budget with other profile, melting and
    budget values (same classes and settings) compile nothing."""
    _batched(_Form().budget())
    other = _Form(rho_cen=12400.0, length_scale=7150e3, r_cmb=3470e3, p_cmb=135e9).budget()
    assert _compiles(lambda: _batched(other)) == 0


CHANGES = [
    ('rho_cen', 12300.0),
    ('length_scale', 7000e3),
    ('r_cmb', 3450e3),
    ('p_cmb', 130e9),
    ('alpha', 1.5e-5),
    ('c_p', 800.0),
    ('light_element_fraction', 0.12),
    ('depression', 1.0),
    ('ds_fusion', 150.0),
    ('icn_width', 20.0),
    ('alpha_c', 0.6),
    ('c_light', 0.08),
]


@pytest.mark.parametrize(('name', 'value'), CHANGES, ids=[c[0] for c in CHANGES])
def test_each_input_reaches_the_reused_program(name, value):
    """A budget that differs from the first in one input gives, through the programs compiled
    for the first, the values of programs built for it alone (1e-12 relative), and at least one
    value differs from the first budget's."""
    base = _batched(_Form().budget())
    changed = _Form(**{name: value}).budget()
    got, want = _batched(changed), _fresh(changed)
    for a, b in zip(got, want):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=0.0)
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(got, base))


def test_a_given_latent_heat_and_a_quadratic_curve_reach_the_program():
    """A constant latent heat and a quadratic melting curve are other forms; each gives the
    values of a program built for it alone."""
    b = _Form(latent_heat=7.0e5).budget()
    for a, w in zip(_batched(b), _fresh(b)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(w), rtol=1e-12, atol=0.0)
    f = _Form()
    prof = _Form().budget().profiles
    q = CoreEnergyBudget(
        prof,
        QuadraticMeltingCurve(t_m0=1695.0, t_m1=1.09e-11, t_m2=-8.0e-24),
        ds_fusion=f.ds_fusion,
        icn_width=f.icn_width,
    )
    for a, w in zip(_batched(q), _fresh(q)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(w), rtol=1e-12, atol=0.0)


def test_no_leaf_of_the_budget_is_weak_typed():
    """Every leaf of the budget pytree is a strong float64 or integer array, and the tree holds
    the profile and melting-curve values."""
    leaves = jax.tree_util.tree_leaves(_Form().budget())
    assert len(leaves) >= 12
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        assert not arr.weak_type and arr.dtype in (jnp.float64, jnp.int32, jnp.int64)
