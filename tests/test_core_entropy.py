"""Unit tests for ``aragog.core.entropy.CoreEntropyBudget``.

The entropy balance carries the dynamo criterion, so its contract is the
same energetic consistency as the budget's: every capacity term is pinned
against the independent Leeds ``thermal_history`` routines evaluated on
identical state (constants below), the conduction sink against its exact
closed form and independent quadrature, the CHR09 efficiency factors
against the printed Nature values, and the field scaling against sign,
monotonicity, and the Earth order of magnitude.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy.integrate import quad

from aragog.core.budget import CoreEnergyBudget
from aragog.core.entropy import CoreEntropyBudget
from aragog.core.melting import QuadraticMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles

pytestmark = pytest.mark.unit

T_C = 4180.0
SHARED = dict(rho_cen=12500.0, r_cmb=3480e3, p_cmb=136e9, c_p=840.0, length_scale=7272e3)

# thermal_history cross-check constants, computed 2026-08-08 on the Nimmo
# model-2 state (labrosse pressure, T_cmb = 4180 K, r_icb = 967.0 km,
# 8000-point grid; energy.secular_cool/.latent_heat/.gravitational/
# .cond_entropy/.radiogenic_heating with the numba-identity and
# scipy-trapezoid shims). Es, Ek, Er agree exactly; the two boundary terms
# carry the same 0.5% trapezoid-grid factor as the energy side.
TH_ES = 5.571710e22  # J/K^2, ratio to aragog 1.000000
TH_EL = 8.305058e22  # J/K^2, ratio 0.99493
TH_EG = 2.111343e23  # J/K^2, ratio 0.99493
TH_EK = 4.509720e08  # W/K, ratio 1.000000
TH_ER = 5.518709e07  # W/K at Q_R = Qr(h0=1e-12 W/kg), ratio 1.000000
TH_ER_QRADIO = 1.9268512e12  # Q_R = M_core * 1e-12 W/kg the TH_ER row used


@pytest.fixture(scope='module')
def ent():
    """Model-2 entropy budget on the cross-validated energy budget."""
    prof = GaussianCoreProfiles(**SHARED, alpha=1.25e-5, pressure_mode='labrosse')
    curve = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    budget = CoreEnergyBudget(
        prof,
        curve,
        ds_fusion=170.0,
        icn_width=10.0,
        latent_heat=750e3,
        alpha_c=1.0,
        c_light=560.0 / 12150.0,
    )
    return CoreEntropyBudget(budget, k_core=130.0)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_entropy_capacities_match_thermal_history(ent):
    """All five entropy terms agree with the independent Leeds
    implementation on identical state (see constants above), and each is
    positive as the second law requires of a cooling, heated core."""
    assert float(ent.secular_entropy_capacity(T_C)) == pytest.approx(TH_ES, rel=1e-4)
    assert float(ent.latent_entropy_capacity(T_C)) == pytest.approx(TH_EL, rel=0.01)
    assert float(ent.gravitational_entropy_capacity(T_C)) == pytest.approx(TH_EG, rel=0.01)
    assert float(ent.conduction_sink()) == pytest.approx(TH_EK, rel=1e-6)
    assert float(ent.radiogenic_entropy(T_C, TH_ER_QRADIO)) == pytest.approx(TH_ER, rel=1e-4)
    for value in (
        ent.secular_entropy_capacity(T_C),
        ent.latent_entropy_capacity(T_C),
        ent.gravitational_entropy_capacity(T_C),
        ent.conduction_sink(),
        ent.radiogenic_entropy(T_C, 1e12),
    ):
        assert float(value) > 0.0


@pytest.mark.physics_invariant
def test_conduction_sink_closed_form_and_quadrature(ent):
    """Ek equals both its closed form 16 pi k r^5 / (5 D^4) and an
    independent quadrature of 4 pi k (dTa/dr / Ta)^2 r^2; the adiabat's
    gradient-to-temperature ratio is exactly 2r/D^2, so the sink must not
    depend on T_cmb."""
    p = ent.budget.profiles
    closed = 16.0 * np.pi * 130.0 * p.r_cmb**5 / (5.0 * p.d_scale**4)
    assert float(ent.conduction_sink()) == pytest.approx(closed, rel=1e-12)
    numeric, _ = quad(
        lambda r: 4.0 * np.pi * 130.0 * (2.0 * r / p.d_scale**2) ** 2 * r**2, 0.0, p.r_cmb
    )
    assert float(ent.conduction_sink()) == pytest.approx(numeric, rel=1e-9)


@pytest.mark.reference_pinned
def test_chr09_efficiency_factors_reproduce_printed_values(ent):
    """The CHR09 Earth-core formula 0.88 (0.45) alpha g R / c_p gives the
    printed 0.52 (0.27) with their stated inputs (g = 10.7 m/s2); with the
    profile family's own CMB gravity (9.83 m/s2, the L = 7272 km family
    carries less mass than PREM) the factors shift to 0.484/0.247."""
    printed_inputs = 1.35e-5 * 10.7 * 3.48e6 / 840.0
    assert 0.88 * printed_inputs == pytest.approx(0.5266, rel=1e-3)
    assert 0.45 * printed_inputs == pytest.approx(0.2693, rel=1e-3)
    assert float(ent.chr09_efficiency_factor()) == pytest.approx(0.4839, rel=1e-3)
    zero_outer = CoreEntropyBudget(ent.budget, k_core=130.0, flux_geometry='zero_outer')
    assert float(zero_outer.chr09_efficiency_factor()) == pytest.approx(0.2475, rel=1e-3)


@pytest.mark.physics_invariant
def test_field_scaling_bounds_and_earth_magnitude(ent):
    """The field estimate is zero for subadiabatic heat flow, grows
    monotonically with the superadiabatic excess as its 1/3 power, and
    lands at the milliTesla order for Earth-like flow."""
    qk = float(ent.adiabatic_heat_flow(T_C))
    assert qk / 1e12 == pytest.approx(14.97, rel=1e-3)
    assert float(ent.b_rms_core(T_C, 0.5 * qk)) == 0.0
    b17 = float(ent.b_rms_core(T_C, 17e12))
    assert b17 * 1e3 == pytest.approx(1.104, rel=1e-2)  # mT, Earth order
    # 2/3-power law in the energy density: (2x excess) -> 2^(1/3) in B.
    b_double = float(ent.b_rms_core(T_C, qk + 2.0 * (17e12 - qk)))
    assert b_double / b17 == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-6)
    assert float(ent.b_dipole_cmb(T_C, 17e12, dipolarity=0.5)) == pytest.approx(
        0.5 * b17, rel=1e-12
    )


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_dynamo_threshold_and_margin(ent):
    """The heat flow where the entropy margin vanishes sits at 5.31 TW,
    consistent with (below) Nimmo's statement that model-2 flows under
    6.5 TW cannot drive a dynamo; the margin rises monotonically with heat
    flow, and radiogenic heating at FIXED flow lowers it (Nimmo 2015,
    Eq. 11 discussion: constant heat flow plus more radioactivity means
    less entropy for the dynamo, because the cooling rate drops)."""
    from scipy.optimize import brentq

    threshold = brentq(lambda q: float(ent.entropy_margin(T_C, q)), 1e12, 40e12)
    assert threshold / 1e12 == pytest.approx(5.306, rel=1e-3)
    assert threshold / 1e12 < 6.5
    margins = [float(ent.entropy_margin(T_C, q)) for q in (6e12, 10e12, 17e12)]
    assert margins[0] < margins[1] < margins[2]
    assert float(ent.entropy_margin(T_C, 17e12)) / 1e6 == pytest.approx(993.9, rel=1e-2)
    with_k = float(ent.entropy_margin(T_C, 17e12, q_radio=1e12))
    assert with_k < margins[2]
    # The loss is bounded by the cooling-term substitution alone; the
    # direct ER gain makes the actual drop strictly smaller.
    capacity = (
        float(ent.secular_entropy_capacity(T_C))
        + float(ent.latent_entropy_capacity(T_C))
        + float(ent.gravitational_entropy_capacity(T_C))
    )
    substitution = 1e12 * capacity / float(ent.budget.effective_capacity(T_C))
    assert margins[2] - with_k < substitution


def test_error_contract_and_jit(ent):
    """Constructor validation names the offending parameter; the margin and
    field evaluate identically under jit."""
    with pytest.raises(ValueError, match='k_core'):
        CoreEntropyBudget(ent.budget, k_core=0.0)
    with pytest.raises(ValueError, match='f_ohm'):
        CoreEntropyBudget(ent.budget, k_core=130.0, f_ohm=1.5)
    with pytest.raises(ValueError, match='flux_geometry'):
        CoreEntropyBudget(ent.budget, k_core=130.0, flux_geometry='spherical_cow')

    eager = float(ent.entropy_margin(T_C, 17e12, 1e12))
    jitted = float(jax.jit(ent.entropy_margin)(T_C, 17e12, 1e12))
    assert jitted == pytest.approx(eager, rel=1e-12)
    assert float(jax.jit(ent.b_rms_core)(T_C, 17e12)) == pytest.approx(
        float(ent.b_rms_core(T_C, 17e12)), rel=1e-12
    )
