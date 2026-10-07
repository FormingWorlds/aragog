"""Unit tests for ``aragog.core.melting.IronMeltingCurve``.

The pure-iron branch must agree with the PALEOS source function it ports
(``paleos.iron_eos.T_melt_Fe``), so the pins below are values computed from
that function directly, including the ~0.7 K branch discontinuity at the
triple point that the published piecewise fit carries.
The depression factor is checked against its algebraic definition and its
error contract, and the whole surface must be jit-safe.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aragog.core.melting as m
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
    # Interior pins computed from the PALEOS function.
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


@pytest.mark.physics_invariant
def test_branch_switch_matches_unblended_outside_transition():
    """Outside [95.5, 101.5] GPa, t_melt_pure matches unblended branches to float precision."""
    tm = IronMeltingCurve.t_melt_pure
    # Low-pressure branch: P <= 95.5 GPa
    p_low = np.linspace(10e9, 95.5e9, 100)
    p_low_gpa = p_low / 1e9
    low_exact = 1991.0 * ((p_low_gpa - 5.2) / 27.39 + 1.0) ** (1.0 / 2.38)
    np.testing.assert_allclose(np.asarray(tm(p_low)), low_exact, rtol=1e-15)

    # High-pressure branch: P >= 101.5 GPa
    p_high = np.linspace(101.5e9, 300e9, 100)
    p_high_gpa = p_high / 1e9
    high_exact = 3712.0 * ((p_high_gpa - 98.5) / 161.2 + 1.0) ** (1.0 / 1.72)
    np.testing.assert_allclose(np.asarray(tm(p_high)), high_exact, rtol=1e-15)


def test_branch_switch_deviation_and_c1_continuity():
    """Inside [95.5, 101.5] GPa, maximum deviation from unblended branches is <= 0.4 K,
    and the derivative dT/dP is C1 continuous at 95.5, 98.5, and 101.5 GPa."""
    tm = IronMeltingCurve.t_melt_pure
    p_trans = np.linspace(95.5e9, 101.5e9, 10000)
    p_trans_gpa = p_trans / 1e9
    low = 1991.0 * ((p_trans_gpa - 5.2) / 27.39 + 1.0) ** (1.0 / 2.38)
    high = 3712.0 * ((p_trans_gpa - 98.5) / 161.2 + 1.0) ** (1.0 / 1.72)
    # The unblended fit jumps by 0.73 K at 98.5 GPa (core_bc.md).
    jump = m._T0 * ((m._PT - m._P0) / 1e9 / m._DP_LOW + 1.0) ** m._EXP_LOW - m._TT
    assert jump == pytest.approx(0.73, abs=0.01)
    t_blended = np.asarray(tm(p_trans))
    dev = np.minimum(np.abs(t_blended - low), np.abs(t_blended - high))
    assert np.max(dev) <= 0.363
    assert np.max(dev) < 0.4

    # C1 continuity of derivative at 95.5, 98.5, 101.5 GPa
    grad_tm = jax.grad(tm)
    for p_boundary in [95.5e9, 98.5e9, 101.5e9]:
        dp = 1e3
        d_left = float(grad_tm(p_boundary - dp))
        d_right = float(grad_tm(p_boundary + dp))
        assert abs(d_left - d_right) / d_left < 1e-4


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
    bad = (-0.05, np.float32(-0.05), np.array(-0.05), jnp.asarray(-0.05), np.nan)
    for concrete in (*bad, [0.1, -0.05], np.array([0.1, np.nan]), np.array([[0.1], [-0.05]])):
        with pytest.raises(ValueError, match='light_element_fraction must be in'):
            alloy.t_melt(100e9, light_element_fraction=concrete)
    for concrete in (np.array([0.1, 0.9]), jnp.array([0.9, 0.1])):
        with pytest.raises(ValueError, match='reaches 1'):
            alloy.t_melt(100e9, light_element_fraction=concrete)
    for concrete in ('0.05', b'0.05', np.True_, False, 0.05 + 0j, [[0.1], [0.1, 0.2]]):
        with pytest.raises(ValueError, match='light_element_fraction must be a real number'):
            alloy.t_melt(100e9, light_element_fraction=concrete)
        with pytest.raises(ValueError, match='light_element_fraction must be a real number'):
            IronMeltingCurve(light_element_fraction=concrete, depression=1.2)
    # Integers and other real types are taken as float64.
    t_pure = float(alloy.t_melt_pure(100e9))
    for zero in (0, np.int64(0), np.uint8(0), jnp.array(0)):
        assert float(alloy.t_melt(100e9, light_element_fraction=zero)) == t_pure
    for tenth in (Fraction(1, 10), Decimal('0.1'), np.array([0.1], dtype=object)):
        assert float(np.ravel(alloy.t_melt(100e9, light_element_fraction=tenth))[0]) == float(
            alloy.t_melt(100e9)
        )
    half = jnp.asarray(0.5, dtype=jnp.bfloat16)
    assert float(alloy.t_melt(100e9, light_element_fraction=half)) == pytest.approx(
        0.4 * t_pure
    )
    # depression * x == 1 exactly is refused; a float32 x whose float32 product rounds to 1
    # but whose float64 product stays below 1 is accepted.
    with pytest.raises(ValueError, match='reaches 1'):
        IronMeltingCurve(light_element_fraction=0.1, depression=2.0).t_melt(1e11, 0.5)
    x_edge = np.float32(0.8333333)
    assert np.float32(1.2) * x_edge == np.float32(1.0) and 1.2 * float(x_edge) < 1.0
    edge = float(alloy.t_melt(100e9, light_element_fraction=x_edge))
    assert edge == pytest.approx(t_pure * (1.0 - 1.2 * float(x_edge)), rel=1e-9)
    # A float32 override is evaluated in float64, and a traced one passes unchecked.
    p = np.geomspace(1e9, 3e11, 5)
    x32 = np.float32(0.83333325)
    near = alloy.t_melt(p, light_element_fraction=x32)
    np.testing.assert_allclose(
        near, alloy.t_melt_pure(p) * (1.0 - 1.2 * float(x32)), rtol=1e-12
    )
    assert np.all(np.asarray(near) > 0.0)
    pair = alloy.t_melt(p[:2], light_element_fraction=np.array([0.05, 0.1]))
    np.testing.assert_allclose(
        pair, [alloy.t_melt(p[0], light_element_fraction=0.05), alloy.t_melt(p[1])], rtol=1e-15
    )
    eager = alloy.t_melt(100e9, light_element_fraction=0.05)
    assert alloy.t_melt(100e9, light_element_fraction=np.float32(0.05)).dtype == jnp.float64
    traced = jax.jit(lambda x: alloy.t_melt(100e9, light_element_fraction=x))
    assert float(traced(0.05)) == pytest.approx(float(eager), rel=1e-15)
    assert float(traced(-0.05)) > float(alloy.t_melt_pure(100e9))
    with pytest.raises(ValueError, match='light_element_fraction must be in'):
        alloy.t_melt(100e9, light_element_fraction=1.0)
    with pytest.raises(ValueError, match='reaches 1'):
        alloy.t_melt(100e9, light_element_fraction=0.9)

    with pytest.raises(ValueError, match='t_m0 must be positive'):
        QuadraticMeltingCurve(t_m0=-1000.0, t_m1=1e-12, t_m2=1e-24)
