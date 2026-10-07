"""Tests for the stratified core budget: a convecting core under a resolved shell.

With ``stratification=True`` on ``CoreEnergyBudget`` the outer core above the shell base is a
:class:`aragog.core.layer.CoreShell` whose temperatures are state; ``core_rates`` advances the
convecting core and the shell together, and ``CoreEntropyBudget.entropy_margin`` adds the
shell's own sources and sinks. Contract clauses exercised here: the two rates conserve heat
exactly; the entropy margin grows with the heat flow; the one-temperature interfaces refuse a
stratified budget; the constructor and the factory reject meaningless configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.core import CoreEnergyBudget, CoreEntropyBudget, CoreModule, GaussianCoreProfiles
from aragog.core.melting import QuadraticMeltingCurve

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

T_C = 4180.0
K_CORE = 130.0


def _profiles():
    return GaussianCoreProfiles(
        rho_cen=12500.0,
        length_scale=7272e3,
        r_cmb=3480e3,
        p_cmb=139e9,
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
        alpha_c=1.0,
        c_light=0.05,
        stratification=stratification,
        k_core=K_CORE if stratification else None,
    )


def _layered_shell(budget):
    """Shell temperatures with a 300 km layer up to 20 K above the adiabat at its top."""
    shell = budget.shell
    depth = budget.profiles.r_cmb - np.asarray(shell.r_cells)
    excess = 20.0 * np.clip(1.0 - depth / 300e3, 0.0, None) ** 2
    return np.asarray(shell.adiabatic_profile(T_C)) + excess


@pytest.mark.physics_invariant
@pytest.mark.parametrize('q_radio', [0.0, 2e12])
def test_the_core_and_shell_rates_conserve_heat(q_radio):
    """C_eff dT_c/dt plus the shell's heat change equals q_radio - Q_cmb exactly, with an inner
    core (latent and gravitational terms on) and a layer in the shell."""
    budget = _budget(True)
    t_shell = _layered_shell(budget)
    d_core, d_shell = budget.core_rates(T_C, t_shell, 8e12, q_radio)
    base = budget.shell.layer_base(t_shell, T_C)
    assert float(budget.r_icb(T_C)) > 0.0
    assert float(base) < budget.profiles.r_cmb - 200e3
    conv = float(budget.effective_capacity(T_C, gravitational_upper=base)) * float(d_core)
    mass = np.asarray(budget.shell.mass)
    shell = budget.profiles.c_p * float(np.sum(mass * np.asarray(d_shell)))
    assert conv + shell == pytest.approx(q_radio - 8e12, rel=1e-10, abs=0)


@pytest.mark.physics_invariant
def test_a_stable_layer_lowers_the_entropy_margin():
    """The CMB flow acts on the top cell, whose temperature is the reference, so the margin
    follows the state: a layer conducts less out of the convecting core than a mixed shell,
    and its sink adds, so the margin drops (Greenwood et al. 2021, p. 9)."""
    budget = _budget(True)
    ent = CoreEntropyBudget(budget, k_core=K_CORE)
    mixed = np.asarray(budget.shell.adiabatic_profile(T_C))
    margins = [
        float(ent.entropy_margin(T_C, 8e12, 0.0, t_shell=t))
        for t in (mixed, _layered_shell(budget))
    ]
    assert np.all(np.isfinite(margins)) and margins[1] < margins[0]


def test_the_shell_arguments_are_keyword_only():
    """A heat flow passed by position where the layer base or the shell now goes fails loudly
    instead of being read as a radius or a temperature profile."""
    budget = _budget(True)
    ent = CoreEntropyBudget(budget, k_core=K_CORE)
    t_shell = _layered_shell(budget)
    with pytest.raises(TypeError):
        budget.effective_capacity(T_C, 8e12)
    with pytest.raises(TypeError):
        ent.entropy_margin(T_C, 8e12, 0.0, t_shell)
    for term in (ent.secular_entropy_capacity, ent.gravitational_entropy_capacity):
        with pytest.raises(TypeError):
            term(T_C, None, T_C)
    with pytest.raises(TypeError):
        ent.latent_entropy_capacity(T_C, T_C)
    with pytest.raises(TypeError):
        ent.radiogenic_entropy(T_C, 1e12, None, T_C)


def test_one_temperature_interfaces_refuse_a_stratified_budget():
    budget = _budget(True)
    with pytest.raises(ValueError, match='core_rates'):
        budget.dtcmb_dt(T_C, 8e12)
    with pytest.raises(ValueError, match='solver'):
        CoreModule(budget, t_cmb=T_C)
    off = _budget(False)
    assert off.shell is None and off.r_convecting == off.profiles.r_cmb


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
            'layer_cells': 32,
            'layer_base_fraction': 0.5,
        },
        r_cmb=3480e3,
        p_cmb_fallback=136e9,
    )
    assert budget.stratification is True
    assert budget.k_core == pytest.approx(90.0)
    assert budget.shell.n_cells == 32 and budget.shell.r_base == pytest.approx(0.5 * 3480e3)

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
