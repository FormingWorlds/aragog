"""Unit tests for ``aragog.core.module``: the pure coupling and the factory.

The standalone :class:`CoreModule` is one of the two couplings of the core
budget, so its contract is that stepping it conserves energy through the
budget's own capacity (including across inner-core nucleation) and that the
features-off configuration reproduces the isothermal-reservoir cooling law
exactly. The factory is the config surface of the solver-coupled path, so
its geometry override and unknown-key rejection are part of the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.core import (
    CoreEnergyBudget,
    CoreEntropyBudget,
    CoreModule,
    GaussianCoreProfiles,
    IronMeltingCurve,
    build_core_module_budget,
)

pytestmark = pytest.mark.unit

EARTH = dict(
    rho_cen=12500.0,
    length_scale=7200e3,
    r_cmb=3480e3,
    p_cmb=136e9,
    alpha=1.35e-5,
    c_p=840.0,
)
YEAR = 3.156e7


def _alloy_budget(**overrides):
    profiles = GaussianCoreProfiles(**EARTH)
    curve = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    kwargs = dict(ds_fusion=170.0, icn_width=10.0)
    kwargs.update(overrides)
    return CoreEnergyBudget(profiles, curve, **kwargs)


@pytest.mark.physics_invariant
def test_legacy_mode_step_reproduces_reservoir_cooling_exactly():
    """With the legacy capacity and constant heat flow the ODE is linear,
    RK4 integrates it exactly, and one step lands on T0 - Q dt / C to
    float precision: the features-off regression anchor of this coupling."""
    budget = _alloy_budget(capacity_mode='legacy', legacy_rho_core=10738.33, legacy_tfac=1.147)
    capacity = 4.0 / 3.0 * np.pi * EARTH['r_cmb'] ** 3 * 10738.33 * EARTH['c_p'] * 1.147
    module = CoreModule(budget, t_cmb=4000.0, n_substeps=4)
    q, dt = 30e12, 1e4 * YEAR
    module.step(q, dt)
    assert module.t_cmb == pytest.approx(4000.0 - q * dt / capacity, rel=1e-14)
    # A balancing source stalls the cooling exactly.
    stalled = CoreModule(budget, t_cmb=4000.0, q_radio=q, n_substeps=4)
    stalled.step(q, dt)
    assert stalled.t_cmb == pytest.approx(4000.0, rel=1e-14)


@pytest.mark.physics_invariant
def test_step_conserves_energy_through_nucleation():
    """Integrating C_eff dT along the recorded trajectory recovers the heat
    extracted, within the sub-step discretisation, across an interval that
    walks the core through inner-core onset (capacity jumps by more than
    the secular term, so the check discriminates)."""
    budget = _alloy_budget()
    module = CoreModule(budget, t_cmb=4120.0, n_substeps=64)
    q = 40e12
    dt = 1.6e8 * YEAR  # extracts ~2e29 J, cooling ~100 K past onset at ~4054 K
    module.step(q, dt)
    assert module.t_cmb < 4054.0 < 4120.0  # onset genuinely crossed
    import jax

    temps = np.asarray(module.last_t_cmb)
    caps = np.asarray(jax.jit(jax.vmap(budget.effective_capacity))(temps))
    energy_out = -np.trapezoid(caps, temps)  # integral C_eff (-dT)
    assert energy_out == pytest.approx(q * dt, rel=2e-3)
    # Discrimination: the secular capacity alone under-counts by far more
    # than the tolerance, so the latent release is genuinely resolved.
    secular_only = float(budget.secular_capacity()) * (temps[0] - temps[-1])
    assert abs(secular_only - q * dt) / (q * dt) > 0.10


def test_trajectory_shape_ramp_and_endpoint():
    """The trajectory carries n+1 samples with strictly increasing times,
    the heat-flow ramp is linear between the given endpoints, and the
    stored endpoint equals the returned state."""
    module = CoreModule(_alloy_budget(), t_cmb=4300.0, n_substeps=8)
    returned = module.step(10e12, 1e5 * YEAR, q_cmb_end=30e12)
    assert module.last_times.shape == (9,)
    assert np.all(np.diff(np.asarray(module.last_times)) > 0.0)
    q = np.asarray(module.last_q_cmb)
    np.testing.assert_allclose(q, np.linspace(10e12, 30e12, 9), rtol=1e-12)
    assert float(module.last_t_cmb[-1]) == pytest.approx(returned, rel=1e-15)
    assert returned < 4300.0  # net heat loss cools the core


def test_diagnostics_and_entropy_attachment():
    """Diagnostics report the phase state on both sides of onset, and the
    entropy attachment adds the dynamo fields only when a heat flow is
    supplied."""
    budget = _alloy_budget()
    entropy = CoreEntropyBudget(budget, k_core=130.0)
    module = CoreModule(budget, t_cmb=4300.0, entropy=entropy)
    hot = module.diagnostics()
    assert not hot['inner_core_present'] and not hot['fully_frozen']
    assert hot['r_icb'] == 0.0
    assert 'entropy_margin' not in hot  # no heat flow supplied

    module.t_cmb = 3900.0
    cold = module.diagnostics(q_cmb=30e12)
    assert cold['inner_core_present'] and not cold['fully_frozen']
    assert 0.0 < cold['r_icb'] < EARTH['r_cmb']
    assert cold['dynamo_active'] == (cold['entropy_margin'] > 0.0)
    assert cold['b_rms_core'] > 0.0
    assert cold['effective_capacity'] > hot['effective_capacity']  # latent term live


def test_factory_geometry_override_and_error_contract():
    """The factory takes its CMB radius from the caller (never the dict),
    falls back to the caller's CMB pressure, rejects unknown keys and
    unknown curve selectors, and threads every recognised group through."""
    params = dict(
        rho_cen=12500.0,
        length_scale=7200e3,
        alpha=1.35e-5,
        c_p=840.0,
        melting_curve='iron',
        light_element_fraction=0.1,
        depression=1.2,
        ds_fusion=170.0,
        icn_width=10.0,
    )
    budget = build_core_module_budget(params, r_cmb=3.3e6, p_cmb_fallback=120e9)
    assert budget.profiles.r_cmb == pytest.approx(3.3e6)
    assert budget.profiles.p_cmb == pytest.approx(120e9)
    assert budget.melting_curve.light_element_fraction == pytest.approx(0.1)

    quad = dict(params)
    quad.update(melting_curve='quadratic', t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    for key in ('light_element_fraction', 'depression'):
        quad.pop(key)
    built = build_core_module_budget(quad, r_cmb=3.48e6, p_cmb_fallback=136e9)
    assert float(built.melting_curve.t_melt(98.5e9)) > 0.0

    # A full config surface carries BOTH curves' fields; the factory must
    # accept the inactive set and consume only the active one (the PROTEUS
    # attrs block always sends every field).
    both = dict(params)
    both.update(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    iron_built = build_core_module_budget(both, r_cmb=3.48e6, p_cmb_fallback=136e9)
    assert iron_built.melting_curve.light_element_fraction == pytest.approx(0.1)

    with pytest.raises(ValueError, match='unrecognised'):
        build_core_module_budget({**params, 'q_radio': 1e12}, r_cmb=3.3e6, p_cmb_fallback=1e9)
    with pytest.raises(ValueError, match='melting_curve'):
        build_core_module_budget(
            {**params, 'melting_curve': 'cubic'}, r_cmb=3.3e6, p_cmb_fallback=1e9
        )
    with pytest.raises(ValueError, match='t_cmb'):
        CoreModule(_alloy_budget(), t_cmb=-1.0)
    with pytest.raises(ValueError, match='n_substeps'):
        CoreModule(_alloy_budget(), t_cmb=4000.0, n_substeps=0)
    with pytest.raises(ValueError, match='dt'):
        CoreModule(_alloy_budget(), t_cmb=4000.0).step(1e12, 0.0)
