"""Unit tests for ``aragog.core.regime``: crystallization-regime flags.

The classifier must place known states in their known regimes: the alloy
Earth case walks fully_liquid, bottom_up, fully_frozen as it cools, and
Nimmo (2015, ch. 9.08) Table 2 model 1's parameters, whose melting curve
dips below the adiabat at the CMB while a liquid channel survives in the
interior, must read as a snow-family topology rather than bottom-up, which
is exactly the state the budget's boundary terms shut off for.
"""

from __future__ import annotations

import jax
import pytest

from aragog.core import CoreEnergyBudget, GaussianCoreProfiles, IronMeltingCurve
from aragog.core.melting import QuadraticMeltingCurve
from aragog.core.regime import (
    REGIME_BOTTOM_UP,
    REGIME_FULLY_FROZEN,
    REGIME_FULLY_LIQUID,
    REGIME_NAMES,
    REGIME_SNOW,
    REGIME_TOP_DOWN,
    crystallization_regime,
    refuse_unmodelled_regime,
    regime_name,
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


@pytest.mark.physics_invariant
def test_alloy_earth_walks_the_bottom_up_sequence():
    """Cooling the alloy Earth core crosses fully_liquid (above the 4147 K
    onset), bottom_up (partial inner core), and fully_frozen (below the
    3689 K CMB melting temperature), in that order."""
    prof = GaussianCoreProfiles(**EARTH)
    curve = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    budget = CoreEnergyBudget(prof, curve, ds_fusion=170.0, icn_width=10.0)
    assert int(crystallization_regime(budget, 4300.0)) == REGIME_FULLY_LIQUID
    assert int(crystallization_regime(budget, 3900.0)) == REGIME_BOTTOM_UP
    assert int(crystallization_regime(budget, 3300.0)) == REGIME_FULLY_FROZEN
    # The sequence is monotone in temperature: no regime reappears.
    codes = [
        int(crystallization_regime(budget, t)) for t in (4300.0, 4100.0, 3900.0, 3500.0, 3300.0)
    ]
    assert codes == sorted(codes)


@pytest.mark.physics_invariant
def test_nimmo_model1_reads_as_snow_topology():
    """Model 1's parameters (negative T_m1: the curve dips below
    the T_c = 4180 K adiabat at the CMB while a liquid channel survives
    around the curve's interior minimum) must classify as the snow family,
    not bottom_up; this is the state whose boundary terms the budget zeroes
    and whose full energetics belongs to the multi-zone treatment."""
    prof2 = GaussianCoreProfiles(
        rho_cen=12500.0,
        length_scale=7272e3,
        r_cmb=3480e3,
        p_cmb=139e9,
        alpha=0.9e-5,
        c_p=840.0,
        pressure_mode='labrosse',
    )
    curve = QuadraticMeltingCurve(t_m0=8034.0, t_m1=-3.38e-12, t_m2=69e-25)
    budget = CoreEnergyBudget(prof2, curve, ds_fusion=170.0, icn_width=10.0)
    code = int(crystallization_regime(budget, 4180.0))
    assert code == REGIME_SNOW
    assert regime_name(code) == 'snow'
    # Discrimination: the same profiles under the model-2 curve at the same
    # temperature are plain bottom-up, so the classifier separates the two
    # printed parameter sets.
    m2 = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    b2 = CoreEnergyBudget(prof2, m2, ds_fusion=170.0, icn_width=10.0)
    assert int(crystallization_regime(b2, 4180.0)) == REGIME_BOTTOM_UP


def test_top_down_state_and_jit_and_names():
    """A curve above the adiabat only near the CMB (melting temperature
    rising outward faster than the adiabat falls) classifies top_down; the
    classifier evaluates identically under jit; every code has a name."""
    prof = GaussianCoreProfiles(**EARTH)
    # Constructed state: quadratic curve below the adiabat at the centre,
    # above it at the CMB (T_m falls with P slower than T_a rises inward).
    curve = QuadraticMeltingCurve(t_m0=5200.0, t_m1=-1.2e-12, t_m2=0.0)
    budget = CoreEnergyBudget(prof, curve, ds_fusion=170.0, icn_width=10.0)
    t = 4300.0
    p_cen = float(prof.pressure(0.0))
    assert float(curve.t_melt(p_cen)) < float(prof.adiabat(0.0, t))  # liquid centre
    assert float(curve.t_melt(prof.p_cmb)) > t  # solid top
    assert int(crystallization_regime(budget, t)) == REGIME_TOP_DOWN

    jit_code = int(jax.jit(lambda x: crystallization_regime(budget, x))(t))
    assert jit_code == REGIME_TOP_DOWN
    assert set(REGIME_NAMES) == {0, 1, 2, 3, 4}
    with pytest.raises(KeyError):
        regime_name(99)


@pytest.mark.physics_invariant
def test_only_bottom_up_freezing_passes_the_regime_guard():
    """A pure-iron core that walks bottom-up through freeze-out passes; a core whose CMB freezes
    before its centre is refused in the top-down band and when fully frozen; an alloy core is
    refused once its inner core holds more than a tenth of the core mass, each by name."""
    prof = GaussianCoreProfiles(**EARTH)
    kw = dict(ds_fusion=170.0, icn_width=10.0)
    pure = CoreEnergyBudget(prof, IronMeltingCurve(), **kw)
    refuse_unmodelled_regime(pure, [pure.t_onset + 100.0, pure.t_onset - 400.0, 2000.0])
    curve = QuadraticMeltingCurve(t_m0=5200.0, t_m1=-1.2e-12, t_m2=0.0)
    top_down = CoreEnergyBudget(prof, curve, **kw)
    assert top_down.t_freeze > top_down.t_onset
    refuse_unmodelled_regime(top_down, [5300.0, 5100.0])
    with pytest.raises(ValueError, match='crystallizes top_down at T_core = 4300.0 K'):
        refuse_unmodelled_regime(top_down, [5300.0, 4300.0])
    with pytest.raises(ValueError, match='crystallizes fully_frozen'):
        refuse_unmodelled_regime(top_down, 2000.0)
    alloy = CoreEnergyBudget(
        prof, IronMeltingCurve(light_element_fraction=0.1, depression=1.2), **kw
    )
    refuse_unmodelled_regime(alloy, [alloy.t_onset + 10.0, alloy.t_onset - 2.0])
    with pytest.raises(
        ValueError, match='outer core holds 0.591 of the core mass at T_core = 4000.0 K'
    ):
        refuse_unmodelled_regime(alloy, [alloy.t_onset - 2.0, 4000.0])


@pytest.mark.physics_invariant
def test_the_outer_core_mass_bound_applies_to_either_composition_term():
    """With a gravitational term alone, on a curve without light elements, the guard refuses
    too, and the bound sits at an outer-core mass fraction of 0.9: a kelvin above that state
    passes, a kelvin below it is refused."""
    from scipy.optimize import brentq

    from aragog.core.regime import M_OC_FRACTION_MIN

    prof = GaussianCoreProfiles(**EARTH)
    curve = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    budget = CoreEnergyBudget(
        prof, curve, ds_fusion=170.0, icn_width=10.0, alpha_c=1.0, c_light=0.046
    )
    m_core = float(prof.enclosed_mass(prof.r_cmb))

    def outer(t):
        return 1.0 - float(prof.enclosed_mass(budget.r_icb(t))) / m_core

    t_bound = brentq(
        lambda t: outer(t) - M_OC_FRACTION_MIN, budget.t_onset - 500.0, budget.t_onset - 1e-3
    )
    refuse_unmodelled_regime(budget, [budget.t_onset + 10.0, t_bound + 1.0])
    for trace in ([t_bound - 1.0], [t_bound - 1.0, budget.t_onset + 10.0]):  # coldest first
        with pytest.raises(ValueError, match='outer core holds 0.89'):
            refuse_unmodelled_regime(budget, trace)
    plain = CoreEnergyBudget(prof, curve, ds_fusion=170.0, icn_width=10.0)
    refuse_unmodelled_regime(plain, [t_bound - 1.0])  # no composition term: no bound


@pytest.mark.physics_invariant
def test_a_legacy_core_is_never_refused():
    """A legacy-mode budget, a reservoir of constant capacity, books no inner core, so the guard
    passes the outer-core mass and top-down states it refuses for the same curve in profile
    mode."""
    prof = GaussianCoreProfiles(**EARTH)
    kw = dict(ds_fusion=170.0, icn_width=10.0)
    legacy = dict(capacity_mode='legacy', legacy_rho_core=10500.0, legacy_tfac=1.147)
    alloy = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    top_down = QuadraticMeltingCurve(t_m0=5200.0, t_m1=-1.2e-12, t_m2=0.0)
    for curve, states in ((alloy, [5000.0, 4000.0]), (top_down, [5300.0, 4300.0, 2000.0])):
        with pytest.raises(ValueError, match='outer core holds|crystallizes top_down'):
            refuse_unmodelled_regime(CoreEnergyBudget(prof, curve, **kw), states)
        budget = CoreEnergyBudget(prof, curve, **kw, **legacy)
        capacity = {float(budget.effective_capacity(t)) for t in states}
        assert len(capacity) == 1
        refuse_unmodelled_regime(budget, states)
