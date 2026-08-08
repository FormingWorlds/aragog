"""Unit tests for ``aragog.core.budget.CoreEnergyBudget``.

The budget's contract is energetic consistency: the effective heat capacity
must equal the temperature derivative of the core's energy content. The
tests verify the secular term against independent quadrature and the
ecosystem's Earth reservoir factor, the latent term against a finite
difference of the independently integrated latent energy, the nucleation
sigmoid against its half-activation identity and hard-switch limit, and the
legacy mode against the isothermal-reservoir formula it reproduces.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy.integrate import quad

from aragog.core.budget import CoreEnergyBudget
from aragog.core.melting import IronMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles

pytestmark = pytest.mark.unit

EARTH = dict(
    rho_cen=12500.0,
    length_scale=7200e3,
    r_cmb=3480e3,
    p_cmb=136e9,
    alpha=1.35e-5,
    c_p=840.0,
)
DS_FUSION = 170.0  # J/kg/K, order of the calibrated iron value


@pytest.fixture(scope='module')
def prof():
    """Earth-like Gaussian profiles shared by the file."""
    return GaussianCoreProfiles(**EARTH)


@pytest.fixture(scope='module')
def alloy_budget(prof):
    """Alloy-curve budget in the partial-inner-core regime (onset ~4054 K)."""
    curve = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    return CoreEnergyBudget(prof, curve, ds_fusion=DS_FUSION, icn_width=10.0)


@pytest.mark.physics_invariant
def test_legacy_mode_reproduces_the_reservoir_closure(prof):
    """Legacy capacity is exactly (4/3) pi r^3 rho c_p tfac, cooling follows
    -Q/C, and internal sources offset the CMB loss term for term."""
    budget = CoreEnergyBudget(
        prof,
        IronMeltingCurve(),
        ds_fusion=DS_FUSION,
        icn_width=10.0,
        capacity_mode='legacy',
        legacy_rho_core=10738.33,
        legacy_tfac=1.147,
    )
    analytic = 4.0 / 3.0 * np.pi * EARTH['r_cmb'] ** 3 * 10738.33 * EARTH['c_p'] * 1.147
    assert float(budget.secular_capacity()) == pytest.approx(analytic, rel=1e-12)
    # Cooling algebra: -Q/C, and a source equal to the loss stalls cooling.
    q = 10e12  # 10 TW
    assert float(budget.dtcmb_dt(4000.0, q)) == pytest.approx(-q / analytic, rel=1e-12)
    assert float(budget.dtcmb_dt(4000.0, q, q_sources=q)) == pytest.approx(0.0, abs=1e-30)
    # Legacy mode carries no latent term even deep below the melting curve.
    assert float(budget.effective_capacity(3000.0)) == pytest.approx(analytic, rel=1e-12)


@pytest.mark.physics_invariant
def test_secular_capacity_matches_quadrature_and_reservoir_factor(prof, alloy_budget):
    """The profile-mode capacity equals c_p times the independently
    integrated mass-weighted adiabat shape, and the emergent temperature
    factor lands on the ecosystem's Earth value 1.147 within 2%, while
    differing from the isothermal (factor 1) closure by far more."""

    def weighted_shape(r):
        return float(prof.density(r)) * float(prof.adiabat(r, 1.0)) * 4.0 * np.pi * r**2

    integral, _ = quad(weighted_shape, 0.0, prof.r_cmb)
    assert float(alloy_budget.secular_capacity()) == pytest.approx(
        EARTH['c_p'] * integral, rel=1e-9
    )
    tfac = float(alloy_budget.effective_tfac())
    assert tfac == pytest.approx(1.147, rel=0.02)  # Bower et al. (2018) Earth value
    assert tfac - 1.0 > 0.10  # discriminates against the isothermal closure


@pytest.mark.physics_invariant
def test_latent_capacity_equals_derivative_of_latent_energy(prof, alloy_budget):
    """C_latent must equal |dE_latent/dT_cmb| with E_latent integrated
    independently (scipy quadrature over the frozen shell) and
    differentiated by central finite difference: the energetic-consistency
    contract of the implicit-function boundary sensitivity."""
    curve = alloy_budget.melting_curve

    def latent_energy(t_cmb: float) -> float:
        r_icb = float(alloy_budget.r_icb(t_cmb))
        value, _ = quad(
            lambda r: (
                float(curve.t_melt(prof.pressure(r)))
                * DS_FUSION
                * float(prof.density(r))
                * 4.0
                * np.pi
                * r**2
            ),
            0.0,
            r_icb,
            limit=200,
        )
        return value

    dt = 1.0  # K; resolves the boundary motion well above quadrature noise
    fd = abs(latent_energy(3900.0 + dt) - latent_energy(3900.0 - dt)) / (2 * dt)
    assert float(alloy_budget.latent_capacity(3900.0)) == pytest.approx(fd, rel=1e-4)
    # The latent term is comparable to (here larger than) the secular one,
    # so the consistency check discriminates: dropping it would halve C_eff.
    assert float(alloy_budget.latent_capacity(3900.0)) > float(alloy_budget.secular_capacity())


@pytest.mark.physics_invariant
def test_nucleation_onset_growth_and_freeze_out(prof, alloy_budget):
    """The inner core is absent above onset, grows monotonically as the core
    cools, and the latent term vanishes again once nothing liquid remains;
    the sigmoid sits at exactly one half at the onset temperature."""
    t_onset = float(alloy_budget.melting_curve.t_melt(prof.pressure(0.0))) / float(
        prof.adiabat(0.0, 1.0)
    )
    assert t_onset == pytest.approx(4054.17, rel=1e-4)  # emergent, alloy regime
    assert float(alloy_budget.nucleation_factor(t_onset)) == pytest.approx(0.5, abs=1e-9)
    assert float(alloy_budget.r_icb(t_onset + 50.0)) == 0.0
    assert float(alloy_budget.latent_capacity(t_onset + 50.0)) == 0.0

    radii = [float(alloy_budget.r_icb(t)) for t in (4000.0, 3900.0, 3800.0, 3700.0)]
    assert all(np.diff(radii) > 0.0)  # grows as the core cools
    assert 0.0 < radii[0] < prof.r_cmb

    # Freeze-out: at 3300 K even the CMB is subcooled; boundary pinned at
    # r_cmb and the latent release is over.
    assert float(alloy_budget.r_icb(3300.0)) == pytest.approx(prof.r_cmb, rel=1e-9)
    assert float(alloy_budget.latent_capacity(3300.0)) == 0.0

    # Hard-switch limit: a tiny width turns the sigmoid into a step.
    sharp = CoreEnergyBudget(
        prof, alloy_budget.melting_curve, ds_fusion=DS_FUSION, icn_width=1e-3
    )
    assert float(sharp.nucleation_factor(t_onset - 1.0)) == pytest.approx(1.0, abs=1e-12)
    assert float(sharp.nucleation_factor(t_onset + 1.0)) == pytest.approx(0.0, abs=1e-12)


def test_error_contract_and_jit(prof, alloy_budget):
    """Invalid constructor inputs raise eagerly with the offending name;
    the assembled cooling rate is jit-safe and matches eager evaluation."""
    curve = IronMeltingCurve()
    with pytest.raises(ValueError, match='ds_fusion'):
        CoreEnergyBudget(prof, curve, ds_fusion=0.0, icn_width=10.0)
    with pytest.raises(ValueError, match='icn_width'):
        CoreEnergyBudget(prof, curve, ds_fusion=DS_FUSION, icn_width=-1.0)
    with pytest.raises(ValueError, match='capacity_mode'):
        CoreEnergyBudget(prof, curve, ds_fusion=DS_FUSION, icn_width=1.0, capacity_mode='x')
    with pytest.raises(ValueError, match='legacy'):
        CoreEnergyBudget(
            prof, curve, ds_fusion=DS_FUSION, icn_width=1.0, capacity_mode='legacy'
        )

    eager = float(alloy_budget.dtcmb_dt(3900.0, 10e12, q_sources=2e12))
    jitted = float(jax.jit(alloy_budget.dtcmb_dt)(3900.0, 10e12, 2e12))
    assert jitted == pytest.approx(eager, rel=1e-12)
    # Cooling with a net loss, and the sign flips with a dominant source.
    assert eager < 0.0
    assert float(alloy_budget.dtcmb_dt(3900.0, 1e12, q_sources=5e12)) > 0.0
