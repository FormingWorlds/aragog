"""Rheological temperature scale and stress-closure contract of the solid rheology.

Module under test: ``aragog.rheology``. The stagnant-lid regime of
temperature-dependent-viscosity convection is governed by the
Frank-Kamenetskii rheological temperature scale

    dT_rh = R T_i^2 / (E_a + P V_a),

the temperature drop over which the Arrhenius viscosity changes by a factor
e (Solomatov & Moresi 2000). The tests check that limit of ``eta_diff`` and
the error contract of the ``stress_closure`` dispatcher.

Reference: V. S. Solomatov & L.-N. Moresi, "Scaling of time-dependent
stagnant lid convection: Application to small-scale convection on Earth and
other terrestrial planets", J. Geophys. Res. 105, B9, 21795-21817 (2000),
https://doi.org/10.1029/2000JB900197.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import (
    eta_diff,
    stress_closure,
)

pytestmark = pytest.mark.unit

_R_GAS = 8.314462618  # matches aragog.rheology default
_E_A = 300.0e3
_V_A = 0.0  # pin the pure-temperature Frank-Kamenetskii scale (P term off)
_ETA0 = 1.0e21
_T_ARR_REF = 1600.0


def _dt_rh(t_interior, e_a=_E_A, v_a=_V_A, p=0.0):
    """Frank-Kamenetskii rheological temperature scale."""
    return _R_GAS * t_interior**2 / (e_a + v_a * p)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_rheological_temperature_scale_sets_e_fold_viscosity_contrast():
    """A temperature drop of one rheological scale ``dT_rh = R T^2 / E_a``
    raises the Arrhenius viscosity by a factor e in the small-drop limit.

    This is the definition of the Solomatov & Moresi (2000) rheological
    temperature scale: it is the scale over which the viscosity varies by
    one e-fold, and it is what sets the thickness of the actively convecting
    sublayer beneath a stagnant lid.
    """
    t_i = 1800.0
    dt = _dt_rh(t_i)  # ~89.8 K

    # Small perturbation so the Frank-Kamenetskii linearisation holds: a drop
    # of a small fraction of dT_rh multiplies viscosity by exp(fraction).
    frac = 1.0e-3
    eta_hot = eta_diff(t_i, 0.0, _ETA0, _E_A, _V_A, _T_ARR_REF)
    eta_cold = eta_diff(t_i - frac * dt, 0.0, _ETA0, _E_A, _V_A, _T_ARR_REF)
    ratio = eta_cold / eta_hot
    # d(ln eta)/dT = -E_a/(R T^2) = -1/dT_rh, so a drop of frac*dT_rh gives
    # a contrast exp(frac). rtol=1e-4: second-order term is O(frac^2)~1e-6.
    assert ratio == pytest.approx(np.exp(frac), rel=1e-4)

    # Discrimination guard: a scale missing the T^2 factor (a plain E_a/R
    # constant, ~3.6e4 K) would predict an utterly different, near-unity
    # contrast; pin that the correct scale is O(100 K), not O(1e4 K).
    assert 50.0 < dt < 200.0
    # Sign guard: cooling stiffens the mantle, so the contrast exceeds one.
    assert ratio > 1.0

    # Edge case: a full one-scale drop is firmly in the nonlinear regime and
    # overshoots e, confirming dT_rh is the linearisation scale, not a literal
    # e-fold at finite amplitude.
    eta_full = eta_diff(t_i - dt, 0.0, _ETA0, _E_A, _V_A, _T_ARR_REF)
    assert eta_full / eta_hot > np.e


@pytest.mark.physics_invariant
def test_stress_closure_error_contract_and_dispatch():
    """The stress-closure dispatcher validates its mode and required inputs.

    Exercises the documented error contract: an unknown mode and each missing-
    argument path must raise ``ValueError`` and compute nothing.
    """
    v = np.full(50, 1.0e-9)

    with pytest.raises(ValueError):
        stress_closure('teleportation', viscous_velocity=v)
    with pytest.raises(ValueError):
        stress_closure('local', viscous_velocity=v)  # missing mixing_length
    with pytest.raises(ValueError, match="use 'lid'"):
        stress_closure('global', viscous_velocity=v)
    # Valid local dispatch scales as v / l: doubling the mixing length halves it.
    sr_l = stress_closure('local', viscous_velocity=1.0e-9, mixing_length=1.0e5)
    sr_l2 = stress_closure('local', viscous_velocity=1.0e-9, mixing_length=2.0e5)
    assert sr_l == pytest.approx(2.0 * sr_l2, rel=1e-9)
