"""Reference benchmarks for ``aragog.core`` against Nimmo (2015) Table 2.

Nimmo (2015), Treatise on Geophysics 2nd ed., ch. 9.08, Table 2 (p. 205) defines two
parameterised Earth core models (model 1: oldest-inner-core end-member;
model 2: best guess). These tests pin the profile family and the melting
curve against every quantity that is reproducible from the printed chapter
alone, and pin the three budget terms against values computed with the
independent Leeds ``thermal_history`` implementation (Greenwood,
github.com/sam-greenwood/thermal_history) evaluated on identical state.

Two documented deviations from the printed table:

* The ``T_m2`` row reads (8.37, 69) x 1e-25 for models (1, 2), but only the
  transposed assignment is consistent with each model's own ``T_i``,
  ``T_m0``, ``T_m1`` and the stated 9.4 K/GPa ICB melting gradient; the
  tests use the self-consistent assignment (model 1: 69e-25, model 2:
  8.37e-25) and verify both consistency conditions.
* The ``Q_T`` row reads (4.6, 3.3) x 1e27 J/K, but the model-2 core of
  ch. 8.02 cools at 104 K/Gyr under 15.2 TW (Table 4, p. 46), which is
  Q_T = 4.6e27 J/K; the tests pin model 2 to that value. Reading the row
  as transposed, like ``T_m2``, is an inference from the two tables.

The CMB pressure is the 139 GPa of ch. 8.02, Table 2, where the model-2
core puts the ICB at the tabulated 328 GPa.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.core.budget import CoreEnergyBudget
from aragog.core.melting import QuadraticMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles
from aragog.core.regime import refuse_unmodelled_regime

pytestmark = pytest.mark.unit

# Shared chapter constants (Table 2 caption): T_c = 4180 K, L_H = 750 kJ/kg,
# c_p = 840 J/kg/K, rho_cen = 12500 kg/m3, ICB melting gradient 9.4 K/GPa.
T_C = 4180.0
L_H = 750e3
SHARED = dict(rho_cen=12500.0, r_cmb=3480e3, p_cmb=139e9, c_p=840.0, length_scale=7272e3)
GYR = 1e9 * 365.25 * 86400.0
P_ICB = 328e9  # ICB pressure of ch. 8.02, Table 2, where the 9.4 K/GPa gradient applies

# Per-model Table 2 entries (with the self-consistent T_m2 assignment).
MODELS = {
    1: dict(
        alpha=0.9e-5,
        d_km=7310.0,
        t_i=5099.0,
        t_cen=5243.0,
        t_m0=8034.0,
        t_m1=-3.38e-12,
        t_m2=69e-25,
        drho_c=800.0,
    ),
    2: dict(
        alpha=1.25e-5,
        d_km=6203.0,
        t_i=5509.0,
        t_cen=5726.0,
        t_m0=2677.0,
        t_m1=2.95e-12,
        t_m2=8.37e-25,
        drho_c=560.0,
    ),
}


def _profiles(model: int) -> GaussianCoreProfiles:
    return GaussianCoreProfiles(
        **SHARED, alpha=MODELS[model]['alpha'], pressure_mode='labrosse'
    )


def _curve(model: int) -> QuadraticMeltingCurve:
    m = MODELS[model]
    return QuadraticMeltingCurve(t_m0=m['t_m0'], t_m1=m['t_m1'], t_m2=m['t_m2'])


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('model', [1, 2])
def test_adiabat_length_scale_matches_printed_d(model):
    """D = sqrt(3 c_p / 2 pi alpha rho_cen G) reproduces the printed Table 2
    values (7310, 6203 km) from each model's alpha alone."""
    prof = _profiles(model)
    assert prof.d_scale / 1e3 == pytest.approx(MODELS[model]['d_km'], rel=5e-4)
    # Discrimination: the two models differ by 18%, far above the tolerance.
    other = MODELS[3 - model]['d_km']
    assert abs(prof.d_scale / 1e3 - other) / other > 0.15


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('model', [1, 2])
def test_melting_curve_reproduces_printed_icb_state(model):
    """Each model's quadratic melting curve passes through the printed ICB
    temperature at the Earth ICB pressure and carries the stated 9.4 K/GPa
    gradient there; both fail under the printed (untransposed) T_m2 row."""
    curve = _curve(model)
    m = MODELS[model]
    assert float(curve.t_melt(P_ICB)) == pytest.approx(m['t_i'], rel=2e-3)
    # rel 0.03: the two-digit T_m2 = 69e-25 of model 1 alone moves the gradient by 3 %.
    assert float(curve.gradient(P_ICB)) * 1e9 == pytest.approx(9.4, rel=0.03)
    # The printed row's other value is inconsistent by construction.
    wrong = QuadraticMeltingCurve(
        t_m0=m['t_m0'], t_m1=m['t_m1'], t_m2=MODELS[3 - model]['t_m2']
    )
    assert abs(float(wrong.t_melt(P_ICB)) - m['t_i']) / m['t_i'] > 0.10


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('model', [1, 2])
def test_adiabat_consistency_with_printed_temperatures(model):
    """The printed (T_cen, T_i, D) triple sits on one Gaussian adiabat with
    the present-day inner-core radius of 1220 km (ch. 8.02, Table 2), and the printed T_cen
    is the adiabatic continuation of T_c = 4180 K to the centre."""
    prof = _profiles(model)
    m = MODELS[model]
    r_icb = prof.d_scale * np.sqrt(np.log(m['t_cen'] / m['t_i']))
    assert r_icb / 1e3 == pytest.approx(1220.0, rel=6e-3)
    assert float(prof.t_cen(T_C)) == pytest.approx(m['t_cen'], rel=2e-3)


@pytest.mark.reference_pinned
def test_model2_adiabatic_heat_flow_matches_printed_qk():
    """The CMB adiabatic heat flow 4 pi r^2 k |dT_a/dr| with k = 130 W/m/K
    reproduces the printed Q_k = 15.0 TW (model 2) from the profile's own
    adiabatic gradient."""
    prof = _profiles(2)
    grad_cmb = 2.0 * prof.r_cmb * T_C / prof.d_scale**2  # |dT_a/dr| at r_cmb
    qk = 4.0 * np.pi * prof.r_cmb**2 * 130.0 * grad_cmb
    # rel 8e-3: the printed values carry two significant digits.
    assert qk / 1e12 == pytest.approx(15.0, rel=8e-3)
    # Model 1 (k = 90, D = 7310 km) must give the other printed value, 7.5 TW.
    prof1 = _profiles(1)
    qk1 = 4.0 * np.pi * prof1.r_cmb**2 * 90.0 * (2.0 * prof1.r_cmb * T_C / prof1.d_scale**2)
    assert qk1 / 1e12 == pytest.approx(7.5, rel=8e-3)


# Leeds thermal_history values (energy.secular_cool, .latent_heat, .gravitational) on the
# model-2 state below (labrosse pressure, T_cmb 4180 K, r_icb 1209.4 km on an 8000-point
# grid with the ICB as a node), per unit CMB cooling; aragog agrees to 1e-6.
TH_SECULAR = 1.851452e27  # J/K
TH_LATENT = 1.744360e27  # J/K
TH_GRAV = 1.046510e27  # J/K


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_budget_terms_match_thermal_history_cross_check():
    """At T_c = 4180 K the model-2 core has its ICB at the 328 GPa of ch. 8.02,
    Table 2; the three capacity terms agree with the Leeds thermal_history
    values, and the total with the Q_T of ch. 8.02, Table 4."""
    prof = _profiles(2)
    budget = CoreEnergyBudget(
        prof,
        _curve(2),
        ds_fusion=170.0,
        icn_width=10.0,
        latent_heat=L_H,
        alpha_c=1.0,
        c_light=MODELS[2]['drho_c'] / 12150.0,  # alpha_c * c = drho_c / rho(r_icb)
    )
    r_icb = budget.r_icb(T_C)
    assert float(r_icb) / 1e3 == pytest.approx(1209.4, rel=1e-4)
    assert float(prof.pressure(r_icb)) / 1e9 == pytest.approx(328.0, rel=3e-3)
    assert float(budget.secular_capacity()) == pytest.approx(TH_SECULAR, rel=1e-4)
    assert float(budget.latent_capacity(T_C)) == pytest.approx(TH_LATENT, rel=1e-5)
    assert float(budget.gravitational_capacity(T_C)) == pytest.approx(TH_GRAV, rel=1e-5)
    # Q_T = Q_cmb / (dT_c/dt) of ch. 8.02, Table 4: 15.2 TW at 104 K/Gyr.
    assert float(budget.effective_capacity(T_C)) == pytest.approx(15.2e12 * GYR / 104, rel=0.02)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('model,cr_printed,rel', [(2, 10100.0, 0.05), (1, 4900.0, 0.15)])
def test_boundary_sensitivity_matches_printed_cr(model, cr_printed, rel):
    """The implicit-function |dr_icb/dT_cmb| evaluated at the present-day
    Earth state (r = 1220 km, T_c = 4180 K) reproduces the Cr values of
    ch. 9.08, Figure 3 caption, p. 211: 10100 m/K for model 2, 4900 m/K
    for model 1."""
    import jax

    prof = _profiles(model)
    budget = CoreEnergyBudget(prof, _curve(model), ds_fusion=170.0, icn_width=10.0)
    d_dr = float(jax.grad(budget.superheat, argnums=0)(1220e3, T_C))
    d_dt = float(jax.grad(budget.superheat, argnums=1)(1220e3, T_C))
    cr = abs(d_dt / d_dr)
    assert cr == pytest.approx(cr_printed, rel=rel)
    # Discrimination: the two models' printed Cr differ by a factor two,
    # far beyond either tolerance, so a model mix-up cannot pass.
    other_printed = 4900.0 if model == 2 else 10100.0
    assert abs(cr - other_printed) / other_printed > 0.4


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_baseline_scenario_reproduces_chapter_headline():
    """Nimmo's baseline (ch. 9.08, p. 212: model 2, entropy production 50 MW/K
    before inner-core formation, heat flow held constant after onset,
    backward integration) states: present-day CMB heat flow 17 TW, inner-core age
    0.5 Gy, present-day entropy production 700 MW/K. Reconstructing that
    pipeline on the module's terms with the chapter's own growth law
    (delta-T_c = 60 K, r_now = 1220 km) lands on 16.7 TW, 0.43 Gy, and
    964 MW/K; tolerances reflect that the chapter's state machinery is
    only partly printed."""
    from aragog.core.entropy import CoreEntropyBudget

    r_now, d_tc, year = 1220e3, 60.0, 3.156e7
    prof = _profiles(2)
    budget = CoreEnergyBudget(
        prof,
        _curve(2),
        ds_fusion=170.0,
        icn_width=10.0,
        latent_heat=L_H,
        alpha_c=1.0,
        c_light=MODELS[2]['drho_c'] / 12150.0,
    )
    ent = CoreEntropyBudget(budget, k_core=130.0)

    # Pre-onset: constant margin of 50 MW/K sets cooling rate and heat flow.
    cooling = (5e7 + float(ent.conduction_sink())) / float(ent.secular_entropy_capacity(T_C))
    q_cmb = float(budget.secular_capacity()) * cooling
    assert q_cmb / 1e12 == pytest.approx(17.0, rel=0.05)

    # Age: integrate the effective capacity along the chapter's quadratic
    # growth law T_c(r) = T_onset - d_tc (r/r_now)^2, heat flow constant.
    r = np.linspace(1.0, r_now, 400)
    cr_path = r_now**2 / (2.0 * d_tc * r)
    rho = np.asarray(prof.density(r))
    q_lat = L_H * 4.0 * np.pi * r**2 * rho * cr_path
    sub = r[::40]
    q_grav = []
    for r_i in sub:
        s = np.linspace(r_i, prof.r_cmb, 200)
        rho_s = np.asarray(prof.density(s))
        psi_s = np.asarray(prof.potential(s))
        m_oc = float(prof.enclosed_mass(prof.r_cmb) - prof.enclosed_mass(r_i))
        moment = np.trapezoid(rho_s * psi_s * 4.0 * np.pi * s**2, s) - m_oc * float(
            prof.potential(r_i)
        )
        cc = (
            4.0
            * np.pi
            * r_i**2
            * float(prof.density(r_i))
            * (MODELS[2]['drho_c'] / 12150.0)
            / m_oc
        )
        q_grav.append(moment * cc)
    q_grav = np.interp(r, sub, np.array(q_grav)) * cr_path
    q_total = float(budget.secular_capacity()) + q_lat + q_grav
    age = np.trapezoid(q_total * (2.0 * d_tc * r / r_now**2), r) / q_cmb / year / 1e9
    assert age == pytest.approx(0.5, rel=0.2)

    # Present-day margin at that heat flow: printed 700 MW/K, band 45%.
    capacity = (
        float(ent.secular_entropy_capacity(T_C))
        + float(ent.latent_entropy_capacity(T_C))
        + float(ent.gravitational_entropy_capacity(T_C))
    )
    cooling_now = q_cmb / float(budget.effective_capacity(T_C))
    margin = capacity * cooling_now - float(ent.conduction_sink())
    assert margin / 1e6 == pytest.approx(700.0, rel=0.45)


@pytest.mark.physics_invariant
def test_model1_printed_parameters_break_bottom_up_topology():
    """Model 1's negative T_m1 curve dips below the T_c = 4180 K adiabat at
    the CMB (melting temperature 5331 K there), a top-down/snow topology
    outside the bottom-up assumption of the budget; the budget books no
    boundary terms there, and the regime guard refuses the state."""
    prof = _profiles(1)
    curve = _curve(1)
    t_melt_cmb = float(curve.t_melt(prof.p_cmb))
    assert t_melt_cmb == pytest.approx(5330.5, rel=2e-3)
    assert t_melt_cmb > T_C  # the CMB itself sits below the melting curve
    # The curve is non-monotone over the core: its minimum lies inside.
    p_min = -curve.t_m1 / (2.0 * curve.t_m2)
    assert prof.p_cmb < p_min < float(prof.pressure(0.0))
    budget = CoreEnergyBudget(prof, curve, ds_fusion=170.0, icn_width=10.0, latent_heat=L_H)
    # With no liquid at the CMB both boundary terms are zero.
    assert float(budget.latent_capacity(T_C)) == pytest.approx(0.0, abs=1e-10)
    assert float(budget.gravitational_capacity(T_C)) == pytest.approx(0.0, abs=1e-10)
    with pytest.raises(ValueError, match='crystallizes snow at T_core = 4180.0 K'):
        refuse_unmodelled_regime(budget, T_C)
