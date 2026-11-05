"""The Rayleigh-Taylor time scales of a sub-grid lid against Miyazaki and Stevenson (2024).

``aragog.surface_lid.rt_instability`` compares the growth time of a dense layer at the lid base
with the conduction time of that layer. The solver reports the result and does not act on it.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from scipy.constants import Julian_year

from aragog import surface_lid as sl
from aragog.rheology import R_GAS
from tests.conftest import assert_legend_clear, opaque_legend

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

FIG5B = (
    Path(__file__).resolve().parent
    / 'validation'
    / 'data'
    / 'miyazaki_stevenson_2024_fig5b.csv'
)
# A lid at kappa = k / (rho cp) = 1e-6 m^2/s; the paper gives no kappa near its Eq. 10.
K, RHO, CP = 3.0, 3000.0, 1000.0
KAPPA = K / (RHO * CP)
# Io crust of Fig. 5 (p. 9): E, T_s, v_z. Eq. 16 (p. 10): eta. drho, g from p. 10; Fig. 5 states neither.
E_IO, T_S_IO, V_Z_IO = 3.0e5, 150.0, 2.0e-2 / Julian_year
ETA_IO, T_ETA_IO, DRHO_IO, G_IO = 1.0e19, 1600.0, 400.0, 1.5
TAU_TH_TOL, TAU_RT_TOL = 1.2, 1.3  # factors against the mean of the 2 figure reads
GRAIN_MM = {'tau_rt_01mm_yr': 0.1, 'tau_rt_1mm_yr': 1.0, 'tau_rt_1cm_yr': 10.0}


def _call(**kw):
    """``rt_instability`` at a stable reference state, with ``kw`` replacing any input."""
    args = dict(
        drho=100.0, T_b=1400.0, alpha=0.1, H=3.0e5, eta0=1.0e20, k=K, rho=RHO, cp=CP, g=1.0
    )
    return sl.rt_instability(**(args | kw))


def _alpha_of(L, T_b=1400.0, H=3.0e5):
    """The base gradient [K/m] that gives the layer ``L`` [m] in Eq. 3 and 4 of the paper."""
    return R_GAS * T_b**2 / (H * L)


def _io(T_b, grain_mm=1.0):
    """``rt_instability`` on the Io crust: alpha of Eq. 14, eta of Eq. 16, n = 1."""
    eta = ETA_IO * grain_mm**2 * np.exp(E_IO / R_GAS * (1.0 / T_b - 1.0 / T_ETA_IO))
    return _call(
        drho=DRHO_IO,
        T_b=T_b,
        alpha=(T_b - T_S_IO) * V_Z_IO / KAPPA,
        H=E_IO,
        eta0=eta,
        g=G_IO,
    )


def _fig5b():
    """The stored reads of Fig. 5B as arrays by column."""
    with FIG5B.open() as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def test_psi0_is_the_value_of_the_paper():
    """psi0 is the coefficient that the paper gives as of the order of 0.1 (Eq. 7, p. 3)."""
    assert sl.LID_RT_PSI0 == pytest.approx(0.1, abs=0)
    assert _call().tau_rt == pytest.approx(_call(psi0=0.1).tau_rt, rel=1e-15)
    assert _call(psi0=0.2).tau_rt == pytest.approx(0.5 * _call().tau_rt, rel=1e-14)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize(('eta0', 'L_crit'), [(1.0e17, 1.0e3), (1.0e20, 1.0e4)])
def test_s1_critical_thickness_is_eq_10(eta0, L_crit):
    """S1: Eq. 10 of Miyazaki and Stevenson (2024, p. 4), L_crit = (eta0 kappa / (drho g))^(1/3),
    gives 1.00 km at 1e17 Pa s and 10.0 km at 1e20 Pa s for drho 100 kg/m^3, g 1 m/s^2 and kappa
    1e-6 m^2/s, the ends of its range of 1 to 10 km. kappa is my assumption. Inverting the ratio
    under the root moves the value by more than a factor 100."""
    out = _call(eta0=eta0)
    assert out.L_crit == pytest.approx(L_crit, rel=1e-12)
    wrong = (100.0 * 1.0 / (eta0 * KAPPA)) ** (1.0 / 3.0)
    assert abs(wrong / L_crit - 1.0) > 0.99


@pytest.mark.physics_invariant
def test_the_times_cross_above_the_critical_thickness():
    """tau_th = tau_rt lies at L = psi0^(-1/3) L_crit, 2.15 L_crit at psi0 0.1, not at L_crit
    itself; the flag is 0 below that length and 1 above it, and the thermal time grows with L
    while the growth time falls."""
    L_crit = _call().L_crit
    L_x = 0.1 ** (-1.0 / 3.0) * L_crit
    at = _call(alpha=_alpha_of(L_x))
    assert at.L == pytest.approx(L_x, rel=1e-13)
    assert at.tau_th == pytest.approx(at.tau_rt, rel=1e-12)
    below, above = _call(alpha=_alpha_of(0.99 * L_x)), _call(alpha=_alpha_of(1.01 * L_x))
    assert below.flag == 0.0
    assert above.flag == 1.0
    assert below.tau_th < above.tau_th
    assert below.tau_rt > above.tau_rt
    assert _call(alpha=_alpha_of(L_crit)).flag == 0.0


@pytest.mark.physics_invariant
def test_each_scale_follows_its_formula():
    """The layer goes as n T_b^2 / (H alpha) (Eq. 3 and 4), the growth time as eta0 / (drho g L)
    (Eq. 7) and the thermal time as L^2 / kappa, in SI units. The hand values at T_b 1400 K, H 300
    kJ/mol, alpha 0.1 K/m, kappa 1e-6, drho 100, g 1, eta0 1e20 are L = 543.2 m, 2.951e11 s and
    1.841e16 s. The layer without the square of T_b is 1400 times smaller; with h for L in Eq. 7
    the growth time depends on a thickness that this function does not take."""
    a = _call()
    assert a.L == pytest.approx(R_GAS * 1400.0**2 / (3.0e5 * 0.1), rel=1e-14)
    assert a.L == pytest.approx(543.2, rel=1e-3)
    assert a.tau_th == pytest.approx(2.951e11, rel=1e-3)
    assert a.tau_rt == pytest.approx(1.841e16, rel=1e-3)
    assert _call(T_b=2800.0).L == pytest.approx(4.0 * a.L, rel=1e-14)
    assert _call(alpha=0.2).L == pytest.approx(0.5 * a.L, rel=1e-14)
    assert _call(H=6.0e5).L == pytest.approx(0.5 * a.L, rel=1e-14)
    assert _call(n=2.0).L == pytest.approx(2.0 * a.L, rel=1e-14)
    assert _call(drho=200.0).tau_rt == pytest.approx(0.5 * a.tau_rt, rel=1e-14)
    assert _call(g=2.0).tau_rt == pytest.approx(0.5 * a.tau_rt, rel=1e-14)
    assert _call(eta0=2.0e20).tau_rt == pytest.approx(2.0 * a.tau_rt, rel=1e-14)
    assert _call(k=2.0 * K).tau_th == pytest.approx(0.5 * a.tau_th, rel=1e-14)
    assert _call(rho=2.0 * RHO).tau_th == pytest.approx(2.0 * a.tau_th, rel=1e-14)
    assert _call(cp=2.0 * CP).tau_th == pytest.approx(2.0 * a.tau_th, rel=1e-14)


def _all_nan(out):
    return all(math.isnan(v) for v in (out.L, out.tau_rt, out.tau_th, out.L_crit, out.flag))


@pytest.mark.parametrize('drho', [0.0, -1.0, -562.0, float('nan'), float('inf')])
def test_a_lid_that_is_not_denser_has_no_instability(drho):
    """A lid no denser than the fluid has no Rayleigh-Taylor growth: every value is NaN, and no
    exception runs. The flag is NaN, not 0, so the output tells 'no test' from 'test passed'."""
    assert _all_nan(_call(drho=drho))


BAD = st.one_of(
    st.sampled_from([0.0, -1.0, -1.0e300, float('nan'), float('inf'), float('-inf')]),
    st.floats(max_value=0.0, allow_nan=False),
)


@pytest.mark.parametrize(
    'name', ['T_b', 'alpha', 'H', 'eta0', 'k', 'rho', 'cp', 'g', 'n', 'psi0']
)
@given(bad=BAD)
def test_any_input_outside_its_domain_gives_nan(name, bad):
    """Zero, negative, NaN or infinite in any input gives NaN in every value, not an exception."""
    assert _all_nan(_call(**{name: bad}))


FINITE = st.floats(min_value=1.0e-300, max_value=1.0e300, allow_nan=False)


@given(
    drho=FINITE,
    T_b=FINITE,
    alpha=FINITE,
    H=FINITE,
    eta0=FINITE,
    k=FINITE,
    rho=FINITE,
    cp=FINITE,
)
@example(1e300, 1e300, 1e-300, 1e-300, 1e300, 1e-300, 1e300, 1e300)
@example(1e-300, 1e-300, 1e300, 1e300, 1e-300, 1e300, 1e-300, 1e-300)
def test_an_extreme_valid_input_gives_finite_values_or_nan_with_no_exception(
    drho, T_b, alpha, H, eta0, k, rho, cp
):
    """Inputs inside the domain whose products leave the float range give all NaN; every other
    result is finite and positive, with the flag 1 exactly when tau_th > tau_rt."""
    out = sl.rt_instability(
        drho=drho, T_b=T_b, alpha=alpha, H=H, eta0=eta0, k=k, rho=rho, cp=cp, g=1.0
    )
    if _all_nan(out):
        return
    assert min(out.L, out.tau_rt, out.tau_th, out.L_crit) > 0.0
    assert max(out.L, out.tau_rt, out.tau_th, out.L_crit) < math.inf
    assert out.flag == (1.0 if out.tau_th > out.tau_rt else 0.0)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_s2_s3_io_crust_against_figure_5b(validation_plot):
    """S2 and S3: on the Io crust of Miyazaki and Stevenson (2024, Fig. 5B, p. 9) the function
    gives L of 10 to 100 m and tau_th below 400 yr (text of p. 9), tau_th within a factor 1.2 and
    tau_rt within a factor 1.3 of the mean of 2 figure reads at T_b up to 1365 K, for grains 0.1
    mm, 1 mm and 1 cm. kappa 1e-6 m^2/s, drho 400 kg/m^3 and g 1.5 m/s^2 are my reading: Fig. 5
    states none of them. Nothing is fitted. The layer without the square of T_b, or drho 100
    instead of 400, misses the tolerance."""
    d = _fig5b()
    T = d['T_b_K']
    th = np.array([_io(t).tau_th for t in T]) / Julian_year
    assert np.all(np.abs(np.log(th / d['tau_th_yr'])) <= np.log(TAU_TH_TOL))
    rt = {}
    for key, grain in GRAIN_MM.items():
        rt[key] = np.array([_io(t, grain).tau_rt for t in T]) / Julian_year
        assert np.all(np.abs(np.log(rt[key] / d[key])) <= np.log(TAU_RT_TOL))
    for t in (1200.0, 1370.0):
        out = _io(t)
        assert 10.0 <= out.L <= 100.0
        assert out.tau_th / Julian_year < 400.0
    assert np.all(np.abs(np.log(th / T**2 / d['tau_th_yr'])) > np.log(TAU_TH_TOL))
    assert np.all(
        np.abs(np.log(4.0 * rt['tau_rt_1mm_yr'] / d['tau_rt_1mm_yr'])) > np.log(TAU_RT_TOL)
    )

    def figure():
        import matplotlib.pyplot as plt

        grid = np.linspace(1200.0, 1370.0, 12)
        fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.2), sharex=True, height_ratios=(1, 2))
        axes[0].plot(grid, [_io(t).tau_th / Julian_year for t in grid], label='this code')
        axes[0].plot(T, d['tau_th_yr'], 'o', mfc='none', label='Fig. 5B, mean of 2 reads')
        axes[0].set_ylabel(r'$\tau_{th}$ [yr]')
        opaque_legend(axes[0], loc='upper left')
        for i, (key, grain) in enumerate(GRAIN_MM.items()):
            color = f'C{i}'
            rows = [_io(t, grain).tau_rt / Julian_year for t in grid]
            axes[1].plot(grid, rows, color=color, label=f'grain {grain:g} mm')
            axes[1].plot(T, d[key], 'o', color=color, mfc='none')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('temperature at the lid base [K]')
        axes[1].set_ylabel(r'$\tau_{RT}$ [yr]')
        axes[1].set_ylim(1e7, 1e16)
        opaque_legend(axes[1], loc='upper right', ncols=3)
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_rt_io_crust')
