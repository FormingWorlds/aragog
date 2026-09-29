"""Tests for the stratified-volume reduction of the core budgets.

With ``stratification=True`` on ``CoreEnergyBudget``, a stably
stratified sub-CMB layer at its equilibrium conductive-matching depth
reduces the convecting volume in the capacity integrals and, through
``CoreEntropyBudget``, in the entropy sources and the conduction sink.
Contract clauses exercised here:

- the switch off, the superadiabatic limit, and the missing-flow call
  all reproduce the unstratified budget exactly (the regression anchor);
- a subadiabatic flow strictly shrinks the capacity, monotonically in
  the flow, and the reduced secular integral matches an independent
  fine-grid quadrature (with the full-volume value as the
  discrimination guard);
- the conductive-matching solve carries its implicit-function
  sensitivity through the custom JVP (finite-difference cross-check in
  both arguments), so the analytic-Jacobian path sees the layer move;
- the entropy margin's volume terms shrink together with the energy
  side, and the constructor rejects the meaningless configurations.

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from aragog.core import CoreEnergyBudget, CoreEntropyBudget, GaussianCoreProfiles
from aragog.core.melting import QuadraticMeltingCurve

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

T_C = 4180.0
K_CORE = 130.0


def _profiles():
    return GaussianCoreProfiles(
        rho_cen=12500.0,
        length_scale=7272e3,
        r_cmb=3480e3,
        p_cmb=136e9,
        alpha=1.25e-5,
        c_p=840.0,
        pressure_mode='labrosse',
    )


def _budget(stratification: bool):
    return CoreEnergyBudget(
        _profiles(),
        QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25),
        ds_fusion=170.0,
        icn_width=10.0,
        stratification=stratification,
        k_core=K_CORE if stratification else None,
    )


def _qk(budget) -> float:
    """Adiabatic CMB heat flow [W], computed independently of the entropy class."""
    p = budget.profiles
    grad = 2.0 * p.r_cmb * T_C / p.d_scale**2
    return float(4.0 * np.pi * p.r_cmb**2 * K_CORE * grad)


@pytest.mark.physics_invariant
def test_switch_off_and_superadiabatic_reproduce_the_full_budget():
    """The switch off and a superadiabatic flow (zero thickness) must
    reproduce the unstratified capacity exactly: enabling the switch
    changes nothing until the physics calls for a layer. A stratified
    budget asked for its capacity WITHOUT a flow refuses (the reduction
    depends on the flow, and the full-volume answer would be silently
    wrong physics); the unstratified budget keeps the no-flow call."""
    off = _budget(stratification=False)
    on = _budget(stratification=True)
    qk = _qk(on)

    c_off = float(off.effective_capacity(T_C))
    assert float(on.effective_capacity(T_C, 1.2 * qk)) == pytest.approx(c_off, rel=1e-12)
    assert float(on.dtcmb_dt(T_C, 1.2 * qk)) == pytest.approx(
        float(off.dtcmb_dt(T_C, 1.2 * qk)), rel=1e-12
    )
    with pytest.raises(ValueError, match='q_cmb'):
        on.effective_capacity(T_C)
    assert float(off.effective_capacity(T_C)) == pytest.approx(c_off, rel=1e-12)


@pytest.mark.physics_invariant
def test_subadiabatic_flow_shrinks_the_capacity_monotonically():
    """A subadiabatic flow strictly shrinks the capacity, deeper layers
    (smaller flow) shrink it further, and the reduced secular integral
    matches an independent fine-grid trapezoid on [0, r_conv]; the
    full-volume value sits far outside that tolerance (discrimination
    guard: an implementation that ignored the upper limit would match
    the full value instead)."""
    on = _budget(stratification=True)
    off = _budget(stratification=False)
    p = on.profiles
    qk = _qk(on)

    c_full = float(off.effective_capacity(T_C))
    caps = [float(on.effective_capacity(T_C, f * qk)) for f in (0.9, 0.6, 0.3)]
    assert caps[0] < c_full
    assert caps[2] < caps[1] < caps[0]

    # Independent quadrature of the reduced secular term at f = 0.6.
    r_conv = float(on.convecting_radius(T_C, 0.6 * qk))
    assert 0.0 < r_conv < p.r_cmb
    r = np.linspace(0.0, r_conv, 20001)
    shape = np.asarray([float(p.adiabat(x, 1.0)) for x in r[:: len(r) // 200]])
    # Full-resolution vectorised evaluation via the profile callables.
    rho = np.asarray(p.density(r))
    shp = np.asarray(p.adiabat(r, 1.0))
    integrand = rho * shp * 4.0 * np.pi * r**2
    secular_ref = p.c_p * np.trapezoid(integrand, r)
    secular_got = float(on.secular_capacity(upper=r_conv))
    assert secular_got == pytest.approx(secular_ref, rel=1e-6)
    secular_full = float(off.secular_capacity())
    assert abs(secular_full - secular_ref) > 100.0 * abs(secular_got - secular_ref)
    del shape  # sampled subset only guarded the vectorised path above


@pytest.mark.physics_invariant
def test_thickness_sensitivity_matches_finite_differences():
    """The conductive-matching solve is comparison-driven, so its custom
    JVP is the only route to a nonzero derivative; both partials must
    match central finite differences (the layer deepens as the flow
    falls, and moves with temperature through the adiabat)."""
    on = _budget(stratification=True)
    qk = _qk(on)
    q0 = 0.6 * qk

    g_q = float(jax.grad(lambda q: on.convecting_radius(T_C, q))(q0))
    dq = 1e-4 * q0
    fd_q = (
        float(on.convecting_radius(T_C, q0 + dq)) - float(on.convecting_radius(T_C, q0 - dq))
    ) / (2.0 * dq)
    assert g_q == pytest.approx(fd_q, rel=1e-5)
    assert g_q > 0.0  # more flow, thinner layer, larger convecting radius

    g_t = float(jax.grad(lambda t: on.convecting_radius(t, q0))(T_C))
    dt = 1e-3 * T_C
    fd_t = (
        float(on.convecting_radius(T_C + dt, q0)) - float(on.convecting_radius(T_C - dt, q0))
    ) / (2.0 * dt)
    assert g_t == pytest.approx(fd_t, rel=1e-4)

    # Clamped regimes carry zero sensitivity: superadiabatic (no layer).
    g_clamped = float(jax.grad(lambda q: on.convecting_radius(T_C, q))(1.5 * qk))
    assert g_clamped == 0.0


@pytest.mark.physics_invariant
def test_stratified_cooling_is_faster_and_differentiable():
    """At fixed subadiabatic flow the stratified core cools faster (the
    same heat leaves a smaller thermal reservoir), the second law holds
    (positive flow cools), and the full cooling rate is differentiable
    in the flow, including the layer response through the custom JVP."""
    on = _budget(stratification=True)
    off = _budget(stratification=False)
    qk = _qk(on)
    q0 = 0.5 * qk

    rate_on = float(on.dtcmb_dt(T_C, q0))
    rate_off = float(off.dtcmb_dt(T_C, q0))
    assert rate_on < 0.0 and rate_off < 0.0
    assert abs(rate_on) > abs(rate_off)

    g = float(jax.grad(lambda q: on.dtcmb_dt(T_C, q))(q0))
    dq = 1e-4 * q0
    fd = (float(on.dtcmb_dt(T_C, q0 + dq)) - float(on.dtcmb_dt(T_C, q0 - dq))) / (2.0 * dq)
    assert g == pytest.approx(fd, rel=1e-4)


@pytest.mark.physics_invariant
def test_entropy_margin_volume_terms_shrink_together():
    """On a stratified energy budget the entropy margin's volume terms
    run over the convecting region: the conduction sink at the reduced
    radius is strictly smaller (Ek scales as upper^5), and the margin at
    a superadiabatic flow reproduces the unstratified value exactly."""
    on = _budget(stratification=True)
    off = _budget(stratification=False)
    ent_on = CoreEntropyBudget(on, k_core=K_CORE)
    ent_off = CoreEntropyBudget(off, k_core=K_CORE)
    qk = _qk(on)

    r_conv = float(on.convecting_radius(T_C, 0.5 * qk))
    ek_reduced = float(ent_on.conduction_sink(upper=r_conv))
    ek_full = float(ent_on.conduction_sink())
    assert ek_reduced == pytest.approx(ek_full * (r_conv / on.profiles.r_cmb) ** 5, rel=1e-12)
    assert ek_reduced < ek_full

    m_super_on = float(ent_on.entropy_margin(T_C, 1.2 * qk))
    m_super_off = float(ent_off.entropy_margin(T_C, 1.2 * qk))
    assert m_super_on == pytest.approx(m_super_off, rel=1e-12)
    # Subadiabatic: the two budgets genuinely diverge.
    m_sub_on = float(ent_on.entropy_margin(T_C, 0.5 * qk))
    m_sub_off = float(ent_off.entropy_margin(T_C, 0.5 * qk))
    assert m_sub_on != pytest.approx(m_sub_off, rel=1e-3)


def test_constructor_rejects_meaningless_configurations():
    """Stratification without a positive conductivity, or on the legacy
    reservoir (which has no volume to reduce), must fail at construction
    rather than surface as a runtime attribute error mid-solve."""
    prof = _profiles()
    curve = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    with pytest.raises(ValueError, match='k_core'):
        CoreEnergyBudget(prof, curve, ds_fusion=170.0, icn_width=10.0, stratification=True)
    with pytest.raises(ValueError, match='k_core'):
        CoreEnergyBudget(
            prof, curve, ds_fusion=170.0, icn_width=10.0, stratification=True, k_core=0.0
        )
    with pytest.raises(ValueError, match='legacy'):
        CoreEnergyBudget(
            prof,
            curve,
            ds_fusion=170.0,
            icn_width=10.0,
            capacity_mode='legacy',
            legacy_rho_core=10500.0,
            legacy_tfac=1.147,
            stratification=True,
            k_core=K_CORE,
        )


def test_factory_threads_the_stratification_keys():
    """``build_core_module_budget`` accepts stratification and k_core and
    sets them on the budget; the unknown-key contract still rejects a
    typo (edge case: the misspelling closest to the new key)."""
    from aragog.core import build_core_module_budget

    budget = build_core_module_budget(
        {
            'melting_curve': 'quadratic',
            't_m0': 2677.0,
            't_m1': 2.95e-12,
            't_m2': 8.37e-25,
            'stratification': True,
            'k_core': 90.0,
        },
        r_cmb=3480e3,
        p_cmb_fallback=136e9,
    )
    assert budget.stratification is True
    assert budget.k_core == pytest.approx(90.0)

    with pytest.raises(ValueError, match='unrecognised'):
        build_core_module_budget(
            {'stratifcation': True},  # the plausible typo
            r_cmb=3480e3,
            p_cmb_fallback=136e9,
        )


@pytest.mark.physics_invariant
def test_layer_below_the_inner_core_closes_the_gravitational_term():
    """A stratified layer reaching below the ICB leaves no convecting
    outer-core shell: the gravitational capacity must close to zero
    from above, never flip sign or magnitude. The compositional
    parameters are ON here (the term is identically zero without them,
    which is why an unguarded bound inversion would otherwise hide),
    and the ICB is placed by choosing a T_cmb with an inner core, then
    the upper bound is forced beneath it."""
    prof = _profiles()
    curve = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    budget = CoreEnergyBudget(
        prof,
        curve,
        ds_fusion=170.0,
        icn_width=10.0,
        alpha_c=1.0,
        c_light=0.05,
        stratification=True,
        k_core=K_CORE,
    )
    r_icb = float(budget.r_icb(T_C))
    assert r_icb > 0.25 * prof.r_cmb  # a substantial inner core (~970 km here)

    healthy = float(budget.gravitational_capacity(T_C, upper=prof.r_cmb))
    assert healthy > 0.0

    # Force the convecting top beneath the ICB: the shell vanishes.
    for upper in (0.9 * r_icb, 0.5 * r_icb, 0.11 * prof.r_cmb):
        g = float(budget.gravitational_capacity(T_C, upper=upper))
        assert g == pytest.approx(0.0, abs=1e-6 * healthy), (
            f'upper={upper:.3e} < r_icb={r_icb:.3e} must close the term, got {g:.3e}'
        )
    # Continuity from above: shrinking the shell shrinks the term.
    slightly_above = float(budget.gravitational_capacity(T_C, upper=1.05 * r_icb))
    assert 0.0 <= slightly_above < healthy


@pytest.mark.physics_invariant
def test_fully_stratified_floor_keeps_the_ode_finite():
    """Non-positive flow pins the convecting radius at the 10% floor:
    the capacity stays positive and finite, warming (q < 0) carries the
    right sign with the floored inertia, and the floored regime has zero
    flow-sensitivity so the Jacobian stays finite there. The floor is a
    numerical guard, not physics; this pins its mechanics."""
    on = _budget(stratification=True)
    p = on.profiles

    for q in (0.0, -1.0e12):
        r = float(on.convecting_radius(T_C, q))
        assert r == pytest.approx(0.1 * p.r_cmb, rel=1e-9)
        c = float(on.effective_capacity(T_C, q))
        assert 0.0 < c < float(_budget(stratification=False).effective_capacity(T_C))
        rate = float(on.dtcmb_dt(T_C, q))
        assert np.isfinite(rate)
        if q < 0.0:
            assert rate > 0.0  # heat flowing IN warms the core

    g = float(jax.grad(lambda q: on.convecting_radius(T_C, q))(-1.0e12))
    assert g == 0.0  # the floor is flat: no spurious sensitivity


def test_entropy_budget_rejects_a_second_conductivity():
    """One core, one conductivity: pairing a stratified energy budget
    with an entropy budget carrying a different k_core must fail at
    construction (the layer depth and the conduction sink would model
    the same physical quantity with different numbers); the matching
    value and the unstratified pairing both construct."""
    on = _budget(stratification=True)
    with pytest.raises(ValueError, match='one core, one conductivity'):
        CoreEntropyBudget(on, k_core=90.0)
    assert CoreEntropyBudget(on, k_core=K_CORE).k_core == pytest.approx(K_CORE)
    off = _budget(stratification=False)
    assert CoreEntropyBudget(off, k_core=90.0).k_core == pytest.approx(90.0)


def test_core_module_diagnostics_report_the_reduced_capacity():
    """CoreModule.diagnostics must report the capacity of the convecting
    volume that drove the trajectory: with a stratified budget and a
    subadiabatic flow the reported value equals the reduced capacity,
    not the full-volume one, and omitting the flow raises the budget's
    missing-flow error instead of silently answering full-volume."""
    from aragog.core import CoreModule

    on = _budget(stratification=True)
    qk = _qk(on)
    module = CoreModule(on, t_cmb=T_C)

    q_sub = 0.5 * qk
    d = module.diagnostics(q_cmb=q_sub)
    assert d['effective_capacity'] == pytest.approx(
        float(on.effective_capacity(T_C, q_sub)), rel=1e-12
    )
    full = float(_budget(stratification=False).effective_capacity(T_C))
    assert d['effective_capacity'] < full

    with pytest.raises(ValueError, match='q_cmb'):
        module.diagnostics()
