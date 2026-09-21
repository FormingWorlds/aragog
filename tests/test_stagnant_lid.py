"""Stagnant-lid limits of the solid-state rheology, versus Solomatov & Moresi.

Module under test: ``aragog.rheology`` (Arrhenius viscosity, global
boundary-layer stress closure), the closure that gives ``EntropySolver`` a
spontaneous stagnant lid when a solid mantle cools (see
docs/Explanations/verification_ssc.md and solid_state_convection.md).

The stagnant-lid regime of temperature-dependent-viscosity convection is
governed by the Frank-Kamenetskii rheological temperature scale

    dT_rh = R T_i^2 / (E_a + P V_a),

the temperature drop over which the Arrhenius viscosity changes by a factor
e. Solomatov & Moresi (2000) show that in this regime the actively
convecting sublayer beneath the lid spans a fixed temperature drop of order
``a * dT_rh`` (with ``a`` ~ 2 independent of the total temperature contrast),
so the interior "forgets" the cold surface and the rigid lid thickens to
absorb the remainder. Aragog encodes this with ``lid_base_temperature =
T_i - lid_contrast_coeff * dT_rh`` and extracts the emergent lid thickness in
``rheology.compute_strain_rate_global``.

Invariants exercised: the rheological viscosity-contrast limit (analytical /
reference-pinned), boundedness of the extracted lid thickness, the strain-
rate error contract, and the Solomatov-Moresi scaling that the convecting-
sublayer temperature drop is set by the rheology, not by the surface
temperature.

Reference: V. S. Solomatov & L.-N. Moresi, "Scaling of time-dependent
stagnant lid convection: Application to small-scale convection on Earth and
other terrestrial planets", J. Geophys. Res. 105, B9, 21795-21817 (2000),
https://doi.org/10.1029/2000JB900197.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.rheology import (
    compute_strain_rate_global,
    compute_t_lid_base,
    eta_diff,
    stress_closure,
)

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

_R_GAS = 8.314  # matches aragog.rheology default
_R_SURF = 6.371e6
_R_CMB = 3.480e6
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
@pytest.mark.reference_pinned
def test_global_closure_extracts_lid_thickness_and_strain_rate():
    """The global stress closure measures the stagnant-lid thickness as the
    distance from the surface to the ``t_lid_base`` isotherm and returns the
    interior strain rate ``v_int / d_lid``.

    Pins the emergent lid geometry (the quantity Solomatov & Moresi 2000
    predict) against a synthetic hot-interior / cold-conductive-lid profile.
    """
    n = 400
    r = np.linspace(_R_CMB, _R_SURF, n)
    t_interior = 1800.0
    t_surface = 300.0
    r_lid = 6.000e6  # base of the conductive lid
    temperature = np.where(
        r <= r_lid,
        t_interior,
        t_interior + (t_surface - t_interior) * (r - r_lid) / (_R_SURF - r_lid),
    )
    v_int = 1.0e-9
    velocity = np.where(r <= r_lid, v_int, 0.0)
    t_lid_base = 1400.0

    strain = compute_strain_rate_global(r, temperature, velocity, t_lid_base=t_lid_base)
    r_base = float(np.max(r[temperature > t_lid_base]))
    d_lid = _R_SURF - r_base
    # Emergent strain rate is exactly the interior velocity over the lid depth.
    assert strain == pytest.approx(v_int / d_lid, rel=1e-6)
    # Boundedness: the lid is a thin fraction of the mantle, not the whole shell.
    assert 0.0 < d_lid < 0.2 * (_R_SURF - _R_CMB)
    # Discrimination guard: a closure that used the full mantle depth instead
    # of the lid depth would report a strain rate smaller by ~ d_lid / D.
    wrong = v_int / (_R_SURF - _R_CMB)
    assert strain > 5.0 * wrong

    # Monotonicity: a thinner conductive lid (e.g. from a hotter surface) means
    # the t_lid_base isotherm sits shallower, which raises the strain rate
    # at fixed interior velocity.
    r_lid_thin = 6.200e6  # thinner conductive layer -> lid isotherm shallower
    temp_thin = np.where(
        r <= r_lid_thin,
        t_interior,
        t_interior + (t_surface - t_interior) * (r - r_lid_thin) / (_R_SURF - r_lid_thin),
    )
    vel_thin = np.where(r <= r_lid_thin, v_int, 0.0)
    strain_thin = compute_strain_rate_global(r, temp_thin, vel_thin, t_lid_base=t_lid_base)
    assert strain_thin > strain  # thinner lid -> higher strain rate


@pytest.mark.physics_invariant
def test_stress_closure_error_contract_and_dispatch():
    """The stress-closure dispatcher validates its mode and required inputs.

    Exercises the documented error contract: an unknown mode and each missing-
    argument path must raise ``ValueError`` and compute nothing.
    """
    r = np.linspace(_R_CMB, _R_SURF, 50)
    t = np.linspace(1800.0, 300.0, 50)
    v = np.full(50, 1.0e-9)

    with pytest.raises(ValueError):
        stress_closure('teleportation', viscous_velocity=v)
    with pytest.raises(ValueError):
        stress_closure('local', viscous_velocity=v)  # missing mixing_length
    with pytest.raises(ValueError):
        stress_closure('global', viscous_velocity=v)  # missing radius/temperature

    # Valid global dispatch returns a positive, finite strain rate.
    strain = stress_closure(
        'global', viscous_velocity=v, radius=r, temperature=t, t_lid_base=1400.0
    )
    assert np.isfinite(strain) and strain > 0.0
    # Valid local dispatch scales as v / l: doubling the mixing length halves it.
    sr_l = stress_closure('local', viscous_velocity=1.0e-9, mixing_length=1.0e5)
    sr_l2 = stress_closure('local', viscous_velocity=1.0e-9, mixing_length=2.0e5)
    assert sr_l == pytest.approx(2.0 * sr_l2, rel=1e-9)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_convecting_sublayer_drop_is_set_by_rheology_not_surface():
    """Solomatov & Moresi (2000): in the stagnant-lid regime the temperature
    drop across the actively convecting sublayer is a fixed multiple of the
    rheological scale, ``T_i - T_lid_base = a * dT_rh``, independent of the
    total contrast to the surface; the lid thickens to absorb the rest.

    Verifies that the aragog lid-base temperature ``T_i - lid_contrast_coeff *
    dT_rh`` gives a sublayer drop that depends only on the interior
    temperature and the activation energy, while the emergent lid thickness
    (from the global closure) grows as the surface cools.
    """
    t_i = 1900.0
    a = 2.2  # aragog lid_contrast_coeff default; S&M report a ~ 2.2-2.4
    dt = _dt_rh(t_i)
    # The aragog source computes t_lid_base using compute_t_lid_base:
    t_lid_base = compute_t_lid_base(t_i, p_lid=0.0, e_a=_E_A, v_a=_V_A, lid_contrast_coeff=a)
    sublayer_drop = t_i - t_lid_base
    # The convecting-sublayer drop is exactly a * dT_rh, set by the rheology.
    assert sublayer_drop == pytest.approx(a * dt, rel=1e-12)
    # It is a small fraction of the total contrast to a cold surface, i.e. the
    # interior does not feel the surface temperature (the S&M invariant).
    assert sublayer_drop < 0.2 * (t_i - 300.0)

    # Emergent lid thickness grows as the surface cools, while the sublayer
    # drop above stays fixed. Build two profiles with the SAME interior T_i
    # and lid-base isotherm but different surface temperatures.
    n = 500
    r = np.linspace(_R_CMB, _R_SURF, n)
    r_conv_top = 6.100e6  # top of the well-mixed interior
    v_int = 2.0e-9

    def _d_lid(t_surface):
        temperature = np.where(
            r <= r_conv_top,
            t_i,
            t_i + (t_surface - t_i) * (r - r_conv_top) / (_R_SURF - r_conv_top),
        )
        velocity = np.where(r <= r_conv_top, v_int, 0.0)
        strain = compute_strain_rate_global(r, temperature, velocity, t_lid_base=t_lid_base)
        # Recover d_lid from the returned strain rate = v_int / d_lid.
        return v_int / strain

    d_cold = _d_lid(300.0)
    d_warm = _d_lid(1000.0)
    # Colder surface -> the t_lid_base isotherm sits deeper -> thicker lid.
    assert d_cold > d_warm > 0.0
    # Boundedness: even the cold-surface lid stays within the mantle shell.
    assert d_cold < (_R_SURF - _R_CMB)
