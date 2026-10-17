"""Calibrated viscous mixing length of the solid-state MLT closure.

Module under test: ``aragog.rheology.viscous_mixing_length_factor`` and
``aragog.rheology.compute_mlt_top_slope``. With rheology enabled, the viscous
branch uses ``l_v = min(s_bot (r - r_in), s_top(theta) (r_out - r))`` in place
of the Abe length ``l``: the viscous eddy diffusivity scales by ``(l_v / l)^4``
and the inviscid branch keeps ``l``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aragog.rheology import (
    SolidRheologyParams,
    compute_mlt_top_slope,
    viscous_mixing_length_factor,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

TOP = -3  # Node near the surface where the top slope sets l_v.
BOTTOM = 3  # Node near the CMB where the bottom slope sets l_v.
RE_CRIT = 9.0 / 8.0


def _mlt_kappa_h(
    log10visc: float,
    radii: np.ndarray,
    mixing_length: np.ndarray,
    q_factor: np.ndarray | float = 1.0,
    xp=np,
):
    """Evaluate the local MLT viscous-inviscid eddy diffusivity blend with factor q."""
    rho = 4000.0
    cp = 1200.0
    alpha = 3.0e-5
    g = 9.81
    temp = 2000.0
    drive = 1.0e-6
    nu = (10.0**log10visc) / rho

    prefactor = alpha * g * temp * drive / cp
    v_visc_abe = prefactor * (mixing_length**3) / (18.0 * nu)
    v_invisc = xp.sqrt(prefactor * (mixing_length**2) / 16.0)

    v_visc = q_factor * v_visc_abe
    reynolds = q_factor * (v_visc * mixing_length / nu)
    blend_width = 0.01 * RE_CRIT
    w_invisc = 0.5 * (1.0 + xp.tanh((reynolds - RE_CRIT) / blend_width))
    v_conv = (1.0 - w_invisc) * v_visc + w_invisc * v_invisc
    return v_conv * mixing_length


@pytest.mark.physics_invariant
def test_factor_is_the_fourth_power_of_the_length_ratio():
    """Top slope 0.5 gives 0.5^4 near the surface, 1 near the CMB, and 1 at boundaries."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    q = viscous_mixing_length_factor(r, r[0], r[-1], ml, 0.5, 1.0)
    assert q[-2] == pytest.approx(0.0625, rel=1e-12)
    assert q[1] == pytest.approx(1.0, rel=1e-12)
    assert q[0] == pytest.approx(1.0, rel=0.0) and q[-1] == pytest.approx(1.0, rel=0.0)
    np.testing.assert_array_equal(
        viscous_mixing_length_factor(r, r[0], r[-1], ml, 1.0, 1.0), 1.0
    )


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_theta_dependent_top_slope_hand_values_and_limits():
    """Verify s_top(theta) = s_top_inf + (1 - s_top_inf) exp(-c_theta max(theta, 0))."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    s_inf = 0.22
    c_th = 0.15

    # 1. Isoviscous limit (theta -> 0): s_top(0) = 1.0 exactly, recovering Abe length.
    assert compute_mlt_top_slope(0.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        1.0, rel=1e-15
    )
    q_zero = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=0.0, c_theta=c_th
    )
    np.testing.assert_allclose(q_zero, 1.0, rtol=1e-15)

    # 2. Negative theta (inverted temperature difference): max(theta, 0) = 0 clamps s_top at 1.0.
    assert compute_mlt_top_slope(-5.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        1.0, rel=1e-15
    )
    q_neg = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=-5.0, c_theta=c_th
    )
    np.testing.assert_allclose(q_neg, 1.0, rtol=1e-15)

    # 3. Hand-computed intermediate value at theta = 15:
    # s_top(15) = 0.22 + 0.78 * exp(-0.15 * 15) = 0.22 + 0.78 * exp(-2.25) = 0.3022113951582542.
    expected_s15 = 0.22 + 0.78 * math.exp(-2.25)
    expected_q15 = expected_s15**4
    assert compute_mlt_top_slope(15.0) == pytest.approx(expected_s15, rel=1e-14)
    assert compute_mlt_top_slope(15.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        expected_s15, rel=1e-14
    )
    q_15 = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=15.0, c_theta=c_th
    )
    assert q_15[TOP] == pytest.approx(expected_q15, rel=1e-14)
    assert q_15[BOTTOM] == pytest.approx(1.0, rel=1e-14)

    # 4. Stagnant-lid asymptotic limit (theta >> 8, e.g. theta = 250): s_top -> s_top_inf = 0.22.
    assert compute_mlt_top_slope(250.0, s_top_inf=s_inf, c_theta=c_th) == pytest.approx(
        s_inf, rel=1e-13
    )
    q_inf = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=250.0, c_theta=c_th
    )
    assert q_inf[TOP] == pytest.approx(s_inf**4, rel=1e-13)
    assert q_inf[BOTTOM] == pytest.approx(1.0, rel=1e-14)


@pytest.mark.physics_invariant
def test_canary_theta_zero_and_exponent_mutations_fail(monkeypatch):
    """Canary: s_top(0) != 1.0 or sign flip on c_theta fails the calibration assertions."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    s_inf = 0.22
    c_th = 0.15

    # True theta=0 factor is 1.0; a mutant that ignores theta (using constant s_top_inf=0.22)
    # gives 0.22^4 = 0.00234256, failing the isoviscous recovery assertion.
    q_zero_true = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=0.0, c_theta=c_th
    )
    q_zero_mutant = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0
    )
    assert q_zero_true[TOP] == pytest.approx(1.0, rel=1e-14)
    assert abs(q_zero_mutant[TOP] - 1.0) > 0.99

    # Mutant flipping the sign of c_theta (c_theta = -0.15) causes s_top(15) to blow up > 1.
    q_15_sign_mutant = viscous_mixing_length_factor(
        r, r[0], r[-1], ml, s_top_inf=s_inf, bottom_slope=1.0, theta=15.0, c_theta=-c_th
    )
    assert q_15_sign_mutant[TOP] > 1000.0

    # Monkeypatching compute_mlt_top_slope proves viscous_mixing_length_factor delegates to it.
    monkeypatch.setattr('aragog.rheology.compute_mlt_top_slope', lambda *a, **kw: 0.5)
    q_patched = viscous_mixing_length_factor(r, r[0], r[-1], ml, theta=15.0)
    assert q_patched[TOP] == pytest.approx(0.5**4, rel=1e-14)


@pytest.mark.physics_invariant
def test_branch_switch_stays_continuous_in_kappa():
    """Sweep viscosity over 20 decades at the near-surface node for constant and theta slopes.

    Scaling the switch Reynolds number by q keeps the transition at kappa_v = kappa_i,
    so ln(kappa_h) never steps by more than the viscous slope (ln 10 * 0.02 = 0.046).
    """
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    lvs = np.arange(-2.0, 18.0, 0.02)

    cases = [
        (viscous_mixing_length_factor(r, r[0], r[-1], ml, s_top_inf=0.5), 0.5**4),
        (viscous_mixing_length_factor(r, r[0], r[-1], ml, s_top_inf=0.22), 0.22**4),
        (
            viscous_mixing_length_factor(
                r, r[0], r[-1], ml, s_top_inf=0.22, theta=15.0, c_theta=0.15
            ),
            (0.22 + 0.78 * math.exp(-2.25)) ** 4,
        ),
    ]
    for q_vec, q_expected in cases:
        k = np.array([_mlt_kappa_h(lv, r, ml, q_vec)[TOP] for lv in lvs])
        step = np.abs(np.diff(np.log(k)))
        assert step.max() < 0.05
        assert np.all(k > 0.0)
        k_ref = [_mlt_kappa_h(lv, r, ml, 1.0)[TOP] for lv in (lvs[0], lvs[-1])]
        assert k[0] / k_ref[0] == pytest.approx(1.0, rel=1e-10)
        assert k[-1] / k_ref[1] == pytest.approx(q_expected, rel=1e-10)


@pytest.mark.physics_invariant
def test_numpy_and_jax_theta_top_slope_parity():
    """Verify NumPy and JAX evaluate identical s_top(theta) and mixing-length factors."""
    pytest.importorskip('jax')
    import jax
    import jax.numpy as jnp

    jax.config.update('jax_enable_x64', True)

    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    r_j = jnp.asarray(r)
    ml_j = jnp.asarray(ml)

    for th in (-3.0, 0.0, 8.0, 15.0, 40.0):
        s_np = compute_mlt_top_slope(th, s_top_inf=0.22, c_theta=0.15, xp=np)
        s_jx = float(
            compute_mlt_top_slope(jnp.asarray(th), s_top_inf=0.22, c_theta=0.15, xp=jnp)
        )
        assert s_jx == pytest.approx(s_np, rel=1e-14)

        q_np = viscous_mixing_length_factor(
            r, r[0], r[-1], ml, s_top_inf=0.22, bottom_slope=1.0, xp=np, theta=th, c_theta=0.15
        )
        q_jx = np.asarray(
            viscous_mixing_length_factor(
                r_j,
                r_j[0],
                r_j[-1],
                ml_j,
                s_top_inf=0.22,
                bottom_slope=1.0,
                xp=jnp,
                theta=jnp.asarray(th),
                c_theta=0.15,
            )
        )
        np.testing.assert_allclose(q_jx, q_np, rtol=1e-14, atol=1e-30)


def test_solid_rheology_params_top_slope_defaults_and_bounds():
    """Verify SolidRheologyParams defaults and bounds validation for s_top_inf and c_theta."""
    params = SolidRheologyParams()
    assert not hasattr(params, 'mlt_top_slope')
    assert params.mlt_bottom_slope == pytest.approx(1.0, rel=1e-15)
    assert params.s_top_inf == pytest.approx(0.22, rel=1e-15)
    assert params.c_theta == pytest.approx(0.15, rel=1e-15)

    custom = SolidRheologyParams(s_top_inf=0.18, c_theta=0.0)
    assert custom.s_top_inf == pytest.approx(0.18, rel=1e-15)
    assert custom.c_theta == pytest.approx(0.0, abs=1e-15)

    with pytest.raises(ValueError, match='s_top_inf must be in'):
        SolidRheologyParams(s_top_inf=0.0)
    with pytest.raises(ValueError, match='s_top_inf must be in'):
        SolidRheologyParams(s_top_inf=-0.1)
    with pytest.raises(ValueError, match='s_top_inf must be in'):
        SolidRheologyParams(s_top_inf=1.1)
    with pytest.raises(ValueError, match='mlt_bottom_slope must be in'):
        SolidRheologyParams(mlt_bottom_slope=0.0)
    with pytest.raises(ValueError, match='mlt_bottom_slope must be in'):
        SolidRheologyParams(mlt_bottom_slope=1.1)
    with pytest.raises(ValueError, match='c_theta must be non-negative'):
        SolidRheologyParams(c_theta=-0.01)
