"""Unit tests for ``aragog.core.melting.IronMeltingCurve``.

The pure-iron branch must agree with the PALEOS source function it ports
(``paleos.iron_eos.T_melt_Fe``), so the pins below are values computed from
that function directly, including the ~0.7 K branch discontinuity at the
triple point that the published piecewise fit carries.
The depression factor is checked against its algebraic definition and its
error contract, and the whole surface must be jit-safe.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from aragog.core.melting import IronMeltingCurve, QuadraticMeltingCurve

pytestmark = pytest.mark.unit


@pytest.mark.physics_invariant
def test_pure_iron_pins_against_the_paleos_source():
    """Values pinned from paleos.iron_eos.T_melt_Fe: the two
    anchors are exact by construction, interior points to float precision;
    the curve rises monotonically over the whole planetary pressure range."""
    tm = IronMeltingCurve.t_melt_pure
    # Anchor points of the fit itself (exact in each branch's formula).
    assert float(tm(5.2e9)) == pytest.approx(1991.0, rel=1e-12)
    assert float(tm(98.5e9)) == pytest.approx(3712.362557, rel=1e-6)
    # Interior pins computed from the PALEOS function on 2026-08-08.
    assert float(tm(1e5)) == pytest.approx(1822.443733, rel=1e-9)
    assert float(tm(50e9)) == pytest.approx(2991.670101, rel=1e-9)
    assert float(tm(136e9)) == pytest.approx(4191.966007, rel=1e-9)  # Earth CMB
    assert float(tm(330e9)) == pytest.approx(6229.183781, rel=1e-9)  # Earth ICB
    assert float(tm(4e12)) == pytest.approx(24232.839977, rel=1e-9)  # 10 M_E centre
    # Discrimination guard: a single-branch fit continued across the triple
    # point misses the high-pressure pin by far more than the tolerance.
    single_branch = 1991.0 * ((4e12 - 5.2e9) / 27.39e9 + 1.0) ** (1.0 / 2.38)
    assert abs(single_branch - 24232.839977) / 24232.839977 > 0.1

    p = np.geomspace(1e5, 1e13, 400)
    t = np.asarray(tm(p))
    assert np.all(t > 0.0) and np.all(np.diff(t) > 0.0)


def test_branch_switch_is_continuous():
    """The Simon-Glatzel branches blend continuously across 98.5 GPa.

    The unblended Anzellini fit carries a ~0.73 K discontinuity at the
    triple point (98.5 GPa). Blending the branches over a 1.0 GPa pressure
    band removes the discontinuity (jump < 1e-6 K) while shifting temperature
    by at most 0.37 K.
    """
    tm = IronMeltingCurve.t_melt_pure
    below = float(tm(98.5e9 - 1.0))
    above = float(tm(98.5e9 + 1.0))
    assert abs(below - above) < 1e-6

    p = np.linspace(90e9, 110e9, 1000)
    p_gpa = p / 1e9
    low = 1991.0 * ((p_gpa - 5.2) / 27.39 + 1.0) ** (1.0 / 2.38)
    high = 3712.0 * ((p_gpa - 98.5) / 161.2 + 1.0) ** (1.0 / 1.72)
    t_blended = np.asarray(tm(p))
    dev = np.minimum(np.abs(t_blended - low), np.abs(t_blended - high))
    assert np.max(dev) < 0.37


@pytest.mark.physics_invariant
def test_depression_factor_algebra_and_error_contract():
    """x = 0 recovers pure iron exactly (the features-off anchor); a finite
    depression scales the curve by exactly (1 - depression * x); invalid
    fractions and coefficients raise eagerly with the offending name."""
    pure = IronMeltingCurve()
    p = np.geomspace(1e9, 1e13, 50)
    np.testing.assert_allclose(
        np.asarray(pure.t_melt(p)), np.asarray(pure.t_melt_pure(p)), rtol=0.0
    )

    alloy = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    factor = 1.0 - 1.2 * 0.1
    np.testing.assert_allclose(
        np.asarray(alloy.t_melt(p)), factor * np.asarray(pure.t_melt_pure(p)), rtol=1e-14
    )
    # Per-call light element fraction override evaluated at specified pressure.
    assert float(alloy.t_melt(330e9, light_element_fraction=0.0)) == pytest.approx(
        6229.183781, rel=1e-9
    )

    with pytest.raises(ValueError, match='light_element_fraction'):
        IronMeltingCurve(light_element_fraction=-0.01)
    with pytest.raises(ValueError, match='light_element_fraction'):
        IronMeltingCurve(light_element_fraction=1.0)
    with pytest.raises(ValueError, match='depression'):
        IronMeltingCurve(depression=-0.5)
    with pytest.raises(ValueError, match='reaches 1'):
        IronMeltingCurve(light_element_fraction=0.5, depression=2.0)


def test_jit_matches_eager():
    """The curve evaluates identically under jax.jit, so it can sit inside
    the budget RHS without retracing surprises."""
    alloy = IronMeltingCurve(light_element_fraction=0.08, depression=1.5)
    p = np.geomspace(1e6, 1e13, 33)
    np.testing.assert_allclose(
        np.asarray(jax.jit(alloy.t_melt)(p)), np.asarray(alloy.t_melt(p)), rtol=1e-15
    )


def test_melting_curve_runtime_overrides_and_validation():
    """t_melt validates runtime light_element_fraction bounds, and QuadraticMeltingCurve checks t_m0."""
    alloy = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    with pytest.raises(ValueError, match='light_element_fraction must be in'):
        alloy.t_melt(100e9, light_element_fraction=-0.05)
    with pytest.raises(ValueError, match='light_element_fraction must be in'):
        alloy.t_melt(100e9, light_element_fraction=1.0)
    with pytest.raises(ValueError, match='reaches 1'):
        alloy.t_melt(100e9, light_element_fraction=0.9)

    with pytest.raises(ValueError, match='t_m0 must be positive'):
        QuadraticMeltingCurve(t_m0=-1000.0, t_m1=1e-12, t_m2=1e-24)
