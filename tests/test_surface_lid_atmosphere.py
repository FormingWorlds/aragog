"""Validation of the lid surface under an atmosphere: the Robin surface and the onset weight.

Each case computes the closed form without the lid functions of ``aragog.surface_lid`` and
compares the NumPy and JAX functions with it; the onset case takes the width of its weight from
the code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erf, erfi

from aragog import surface_lid as sl
from tests.conftest import assert_legend_clear, opaque_legend

# a lid of 100 m, 3 W/m/K, base 1400 K, constant rho and cp, with the energy of a straight
# profile at a 600 K surface; the atmosphere held the surface at 500 K and gave the lid 14 W/m2
DELTA, T_B, K, RHO, CP = 100.0, 1400.0, 3.0, 3000.0, 1200.0
T_HELD, CAP = 500.0, 14.0
E_LID = -DELTA * RHO * CP * (T_B - 600.0) / 2.0
SLOPES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def _slope_and_mean(s):
    """f'(0) and the mean of f_s(x) = erf(sqrt(s) x) / erf(sqrt(s)) on [0, 1], from erf and erfi."""
    if abs(s) < 1e-8:
        return 1.0 + s / 3.0, 0.5 + s / 12.0
    if s > 0.0:
        mu = math.sqrt(s)
        e = erf(mu)
        return 2.0 * mu / (math.sqrt(math.pi) * e), 1.0 + math.expm1(-s) / (
            mu * math.sqrt(math.pi) * e
        )
    nu = math.sqrt(-s)
    e = erfi(nu)
    return 2.0 * nu / (math.sqrt(math.pi) * e), 1.0 - math.expm1(-s) / (
        nu * math.sqrt(math.pi) * e
    )


def _robin_closed_form(Fp):
    """Surface temperature and shape where the heat of the profile is E / delta and the conduction
    to the surface equals C + F' (T_s - T_held)."""

    def drop(s):
        return -E_LID / (DELTA * RHO * CP * (1.0 - _slope_and_mean(s)[1]))

    def residual(s):
        dT = drop(s)
        return K * dT * _slope_and_mean(s)[0] / DELTA - CAP - Fp * (T_B - dT - T_HELD)

    grid = np.linspace(-8.0, 8.0, 801)
    r = np.array([residual(v) for v in grid])
    i = int(np.flatnonzero(np.sign(r[:-1]) != np.sign(r[1:]))[0])
    s = brentq(residual, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-15)
    return T_B - drop(s), s


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_robin_surface_solves_the_energy_and_the_surface_balance(validation_plot):
    """Analytical limit: a lid surface holds no heat, so the conduction k (T_b - T_s) f'(0) / delta
    equals the atmosphere flux to first order, C + F' (T_s - T_held), while the profile holds the
    lid energy E. For constant heat capacity both equations have closed forms in the shape s
    (f'(0) and the profile mean from erf for s > 0 and erfi for s < 0). The NumPy and JAX Robin
    surfaces match their root to 1e-9 for F' from 0 to 1000 W/m2/K. At F' = 0.1 the root is
    the straight profile (s = 0) at the 600 K of the lid energy. The cap alone (F' = 0) is
    218.6 K warmer than the root at F' = 1000, which lies 0.02 K above the held surface."""
    from aragog.jax import surface_lid as jl

    e = jl.enthalpy(sl.LidEnthalpy.linear(RHO, CP, T_B, 0.0))
    hv = sl.LidEnthalpy.linear(RHO, CP, T_B, 0.0)
    ref, numpy_T, jax_T = [], [], []
    for Fp in (0.0,) + SLOPES:
        T_s, s = _robin_closed_form(Fp)
        got = sl.robin_surface_state(
            DELTA, E_LID, T_B, K, CAP, Fp, T_HELD, sl.LID_DELTA_FLOOR, hv
        )
        assert got.clipped == 0
        np.testing.assert_allclose(got.T_s, T_s, rtol=1e-9)
        np.testing.assert_allclose(got.s, s, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(got.F_s, CAP + Fp * (T_s - T_HELD), rtol=1e-9)
        out = jl.surface_state(
            DELTA, E_LID, T_B, K, 1.0, 0.0, sl.LID_DELTA_FLOOR, e,
            flux=1.0, F_atm=0.0, held=1.0, T_held=T_HELD, capped=1.0, cap=CAP, slope=Fp,
        )  # fmt: skip
        np.testing.assert_allclose(float(out[0]), T_s, rtol=1e-9)
        np.testing.assert_allclose(float(out[1]), CAP + Fp * (T_s - T_HELD), rtol=1e-9)
        np.testing.assert_allclose(float(out[3]), s, rtol=1e-7, atol=1e-9)
        ref.append(T_s)
        numpy_T.append(got.T_s)
        jax_T.append(float(out[0]))
    ref, numpy_T, jax_T = np.array(ref), np.array(numpy_T), np.array(jax_T)
    assert np.all(np.diff(ref) < 0.0) and np.all(ref > T_HELD)
    straight = _robin_closed_form(0.1)
    assert straight[0] == pytest.approx(600.0, rel=1e-12) and abs(straight[1]) < 1e-12
    assert ref[0] - ref[-1] == pytest.approx(218.6, abs=0.1)
    assert ref[-1] - T_HELD == pytest.approx(0.02, abs=0.005)
    Fp_axis = np.array(SLOPES)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(Fp_axis, ref[1:], color='C0', lw=1.2, label='closed form')
        ax.plot(Fp_axis, numpy_T[1:], 'o', color='C0', ms=6, label='NumPy')
        ax.plot(Fp_axis, jax_T[1:], 'x', color='C1', ms=7, label='JAX')
        ax.axhline(ref[0], color='C2', ls='--', lw=1.0, label="cap alone, F' = 0")
        ax.axhline(T_HELD, color='C3', ls=':', lw=1.2, label='held surface')
        ax.set_xscale('log')
        ax.set_xlabel("slope F' (W m$^{-2}$ K$^{-1}$)")
        ax.set_ylabel('$T_s$ (K)')
        opaque_legend(ax, loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_robin_surface')


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_onset_weight_rises_without_a_step_below_the_base(validation_plot):
    """Analytical limit: under a held surface T_s the steady crust is delta_eq = k (T_b - T_s) /
    F_b, and the weight is w = (1 + tanh(log10(delta_eq / delta_on) / W)) / 2, 0 below 1e-10, 0
    for T_s at or above T_b (also for F_b <= 0) and 1 for F_b <= 0 below it. NumPy and JAX match
    it to 1e-12 for delta_eq / delta_on from 1e-3 to 1e2 at 3 base fluxes, all with T_s at or
    above 900 K; both give w = 0.5 where T_b - T_s = F_b delta_on / k, 0 at or above T_b and 1
    for F_b <= 0, and w is 0 within the last 0.06 F_b delta_on / k below T_b. A step weight, 1
    where delta_eq exceeds delta_on, jumps from 0 to 1 at that drop."""
    from aragog.jax import surface_lid as jl

    k, delta_on, W = 4.0, 0.02, sl.LID_EXIST_WIDTH
    # delta_eq / delta_on up to 1e2 at these base fluxes keeps T_s at or above 900 K
    x_rel = np.geomspace(1e-3, 1e2, 211)
    fluxes = (10.0, 100.0, 1000.0)
    ref, numpy_w, jax_w = {}, {}, {}
    for F_b in fluxes:
        half = F_b * delta_on / k
        drops = x_rel * half
        T_s = T_B - drops
        assert np.all(T_s >= 900.0)
        # the closed form at the surface the code receives, T_b - T_s as rounded in it
        w = 0.5 * (1.0 + np.tanh(np.log10(k * (T_B - T_s) / F_b / delta_on) / W))
        ref[F_b] = np.where(w > 1e-10, w, 0.0)
        got = np.array([sl.flux_weight(F_b, T_B, T, k, 1.0, 0.0, delta_on) for T in T_s])
        got_jax = np.array(
            [float(jl.flux_weight(F_b, T_B, T, k, 1.0, 0.0, delta_on)) for T in T_s]
        )
        np.testing.assert_allclose(got, ref[F_b], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(got_jax, ref[F_b], rtol=1e-12, atol=1e-15)
        # 0.5 to the rounding of T_b - T_s at 1400 K: 1e-16 x 1400 K / 0.05 K, times dw/d ln(delta)
        for f in (sl.flux_weight, jl.flux_weight):
            assert float(f(F_b, T_B, T_B - half, k, 1.0, 0.0, delta_on)) == pytest.approx(
                0.5, abs=1e-10
            )
        assert np.all(got[x_rel < 0.06] == 0.0) and np.all(np.diff(got) >= 0.0)
        numpy_w[F_b], jax_w[F_b] = got, got_jax
    for f in (sl.flux_weight, jl.flux_weight):
        for T_s in (T_B, T_B + 1.0):
            assert float(f(100.0, T_B, T_s, k, 1.0, 0.0, delta_on)) == 0.0
        for F_b in (-1.0, 0.0):
            for drop in (0.01, 10.0):
                assert float(f(F_b, T_B, T_B - drop, k, 1.0, 0.0, delta_on)) == 1.0
            # a surface at or above the base carries no lid, also without a base flux
            for T_s in (T_B, T_B + 1.0):
                assert float(f(F_b, T_B, T_s, k, 1.0, 0.0, delta_on)) == 0.0
    win = np.flatnonzero((x_rel >= 0.2) & (x_rel <= 5.0))

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(x_rel[win], ref[fluxes[1]][win], color='C0', lw=1.2, label='closed form')
        for i, F_b in enumerate(fluxes):
            m = win[i::6]
            ax.plot(
                x_rel[m],
                numpy_w[F_b][m],
                'o',
                color=f'C{i + 1}',
                ms=5,
                label=f'NumPy, {F_b:g}',
            )
            m = win[i + 3 :: 6]
            ax.plot(
                x_rel[m], jax_w[F_b][m], 'x', color=f'C{i + 1}', ms=6, label=f'JAX, {F_b:g}'
            )
        ax.set_xscale('log')
        ax.set_xlabel(r'$\delta_\mathrm{eq} / \delta_\mathrm{on}$')
        ax.set_ylabel('lid weight $w$')
        opaque_legend(
            ax, loc='center left', bbox_to_anchor=(1.02, 0.5), title='$F_b$ (W m$^{-2}$)'
        )
        fig.tight_layout()
        assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'surface_lid_onset_weight')
