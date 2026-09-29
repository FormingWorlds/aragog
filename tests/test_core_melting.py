"""Unit tests for ``aragog.core.melting.IronMeltingCurve``.

The pure-iron branch must agree with the PALEOS source function it ports
(``paleos/iron_eos.py`` ``T_melt_Fe`` at 66ac273), so the pins below are
values computed from that function directly, including the ~0.7 K branch
discontinuity at the triple point that the published piecewise fit carries.
The depression factor is checked against its algebraic definition and its
error contract, and the whole surface must be jit-safe.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from aragog.core.melting import IronMeltingCurve

pytestmark = pytest.mark.unit


@pytest.mark.physics_invariant
def test_pure_iron_pins_against_the_paleos_source():
    """Values pinned from paleos.iron_eos.T_melt_Fe (SHA 66ac273): the two
    anchors are exact by construction, interior points to float precision;
    the curve rises monotonically over the whole planetary pressure range."""
    tm = IronMeltingCurve.t_melt_pure
    # Anchor points of the fit itself (exact in each branch's formula).
    assert float(tm(5.2e9)) == pytest.approx(1991.0, rel=1e-12)
    assert float(tm(98.5e9)) == pytest.approx(3712.0, rel=1e-12)
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


def test_branch_switch_reproduces_the_published_discontinuity():
    """The piecewise Anzellini fit as adopted by PALEOS jumps ~0.7 K at the
    triple point; the port must reproduce it, not smooth it away, so both
    codes agree bitwise within each branch."""
    tm = IronMeltingCurve.t_melt_pure
    below = float(tm(98.5e9 * (1 - 1e-9)))
    above = float(tm(98.5e9 * (1 + 1e-9)))
    assert below - above == pytest.approx(0.725, abs=0.05)
    # And the jump is small against the curve itself: below 0.03%.
    assert (below - above) / above < 3e-4


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
    # The per-call override hook used by the volatile-dissolution stage.
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
