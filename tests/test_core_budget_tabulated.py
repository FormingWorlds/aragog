"""The core energy budget on a tabulated profile.

The known answer is a table of constant properties (alpha 1.35e-5 K-1, c_p 840 J/kg/K) built
from the Gaussian Earth structure: its isentrope is the exact Gaussian adiabat ``T_cmb
exp(-alpha psi / c_p)``, so every budget term must equal the Gaussian one to the
interpolation error. The PALEOS cases hold the energy identities of the budget on the real
liquid iron table.
"""

from __future__ import annotations

import pathlib

import jax
import numpy as np
import pytest

from aragog.core.budget import CoreEnergyBudget
from aragog.core.entropy import CoreEntropyBudget
from aragog.core.iron_table import LiquidIronTable
from aragog.core.melting import IronMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles
from aragog.core.profiles_tabulated import TabulatedCoreProfiles
from aragog.core.regime import refuse_unmodelled_regime

pytestmark = [pytest.mark.unit, pytest.mark.timeout(300)]

CUT = pathlib.Path(__file__).parent / 'validation' / 'data' / 'paleos_iron_liquid_cut_3tpa.npz'
EARTH = dict(
    rho_cen=12500.0, length_scale=7272e3, r_cmb=3480e3, p_cmb=136e9, alpha=1.35e-5, c_p=840.0
)
BUDGET = dict(ds_fusion=172.8, icn_width=10.0, alpha_c=0.8, c_light=0.05)
T_RANGE = (2900.0, 6100.0)
V1_RTOL = 1e-6


def core_arrays(g, n):
    """Core arrays of the profile ``g`` on ``n`` uniform nodes."""
    r = np.linspace(0.0, g.r_cmb, n)
    return {
        'r': r,
        'm': np.asarray(g.enclosed_mass(r)),
        'P': np.asarray(g.pressure(r)),
        'rho': np.asarray(g.density(r)),
        'g': np.asarray(g.gravity(r)),
    }


def constant_property_table(g, n_p=400, n_t=60):
    """Liquid table with constant alpha and c_p whose isentrope is the adiabat of ``g``.

    ``s(P, T) = c_p ln T + alpha psi(P)``, with ``psi(P) = -int_{P_cmb}^P dP'/rho`` the
    potential of the profile at the radius of pressure P (continued linearly in P outside the
    core), so that ``s = const`` gives ``T = T_cmb exp(-alpha psi / c_p)``.
    """
    r = np.linspace(0.0, g.r_cmb, 20001)
    p_r, psi_r, rho_r = (np.asarray(f(r)) for f in (g.pressure, g.potential, g.density))
    p = np.geomspace(p_r[-1] * 0.8, p_r[0] * 1.25, n_p)
    t = np.geomspace(1000.0, 30000.0, n_t)
    order = np.argsort(p_r)
    psi = np.interp(p, p_r[order], psi_r[order])
    rho = np.interp(p, p_r[order], rho_r[order])
    lo, hi = p < p_r[-1], p > p_r[0]
    psi[lo] = psi_r[-1] - (p[lo] - p_r[-1]) / rho_r[-1]
    psi[hi] = psi_r[0] - (p[hi] - p_r[0]) / rho_r[0]
    pp, tt = np.meshgrid(p, t, indexing='ij')
    a, c = EARTH['alpha'], EARTH['c_p']
    fields = {
        's': c * np.log(tt) + a * psi[:, None],
        'rho': np.broadcast_to(rho[:, None], pp.shape),
        'u': c * tt,
        'cp': np.full(pp.shape, c),
        'cv': np.full(pp.shape, c),
        'alpha': np.full(pp.shape, a),
        'nabla_ad': a * pp / (rho[:, None] * c),
    }
    return LiquidIronTable(p, t, fields, header='# Kind: liquid')


def _legend_outside(fig, ax):
    """Put the legend right of the axes and check that it covers no data."""
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')
    fig.tight_layout()
    fig.canvas.draw()
    assert legend.get_window_extent().x0 >= ax.get_window_extent().x1


@pytest.fixture(scope='module')
def v1_budgets():
    """Gaussian and constant-property tabulated budgets of one Earth core with an alloy curve."""
    g = GaussianCoreProfiles(**EARTH)
    tab = TabulatedCoreProfiles(
        core_arrays(g, 801), constant_property_table(g), t_cmb_range=T_RANGE, t_cmb_step=10.0
    )
    curve = IronMeltingCurve(light_element_fraction=0.1, depression=1.2)
    return CoreEnergyBudget(g, curve, **BUDGET), CoreEnergyBudget(tab, curve, **BUDGET)


@pytest.fixture(scope='module')
def paleos_budget():
    """Earth structure on the PALEOS liquid isentrope, pure-iron melting curve."""
    g = GaussianCoreProfiles(**EARTH)
    tab = TabulatedCoreProfiles(
        core_arrays(g, 201), LiquidIronTable.from_file(CUT), t_cmb_range=(2500.0, 7000.0)
    )
    return CoreEnergyBudget(tab, IronMeltingCurve(), **BUDGET)


def _terms(budget, ts):
    """Secular, latent and gravitational capacity [J/K] and r_icb [m] at ``ts``."""
    return {
        'secular': np.array([float(budget.secular_capacity(t_cmb=t)) for t in ts]),
        'latent': np.array([float(budget.latent_capacity(t)) for t in ts]),
        'gravitational': np.array([float(budget.gravitational_capacity(t)) for t in ts]),
        'r_icb': np.array([float(budget.r_icb(t)) for t in ts]),
    }


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a_constant_property_table_gives_the_gaussian_budget(v1_budgets, validation_plot):
    """A table with constant alpha and c_p, built from the Gaussian Earth structure, gives the
    Gaussian secular, latent and gravitational capacities, r_icb and the onset temperature to
    1e-6 relative over 3000 to 6000 K (analytic: the isentrope of that table is the exact
    Gaussian adiabat). The capacity of a Gaussian with c_p 1 % larger misses by 1e-2, and the
    errors fall about 10 times from 201 to 801 nodes (measured), so they are interpolation
    error, not a formula difference."""
    gauss, tab = v1_budgets
    window = np.linspace(gauss.t_freeze, gauss.t_onset, 23)[1:-1]
    ts = np.union1d(np.linspace(3000.0, 6000.0, 31), window)
    ref, got = _terms(gauss, ts), _terms(tab, ts)
    ref['secular'] = np.full(ts.size, float(gauss.secular_capacity()))
    rel = {}
    for key in ref:
        np.testing.assert_array_equal(ref[key] == 0.0, got[key] == 0.0, err_msg=key)
        nz = ref[key] != 0.0
        rel[key] = np.where(nz, np.abs(got[key] / np.where(nz, ref[key], 1.0) - 1.0), np.nan)
        assert np.nanmax(rel[key]) < V1_RTOL, key
    assert tab.t_onset == pytest.approx(gauss.t_onset, rel=V1_RTOL)
    assert np.any(ref['latent'] > 0.0) and np.any(ref['latent'] == 0.0)
    wrong = GaussianCoreProfiles(**{**EARTH, 'c_p': 1.01 * EARTH['c_p']})
    miss = float(CoreEnergyBudget(wrong, gauss.melting_curve, **BUDGET).secular_capacity())
    assert abs(miss / ref['secular'][0] - 1.0) > 100 * V1_RTOL

    keys = ('secular', 'latent', 'gravitational')

    def terms():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for key in keys:
            shown = np.where(ref[key] > 0.0, ref[key], np.nan)
            (line,) = ax.plot(ts, shown, label=key)
            ax.plot(
                ts[::4],
                np.where(got[key] > 0.0, got[key], np.nan)[::4],
                'o',
                ms=3,
                color=line.get_color(),
            )
        ax.set(
            xlabel='CMB temperature [K]',
            ylabel='Capacity [J/K]',
            yscale='log',
            ylim=(1e24, 1e29),
        )
        ax.set_yticks([1e24, 1e26, 1e28])
        ax.minorticks_off()
        _legend_outside(fig, ax)
        return fig

    def difference():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for key in keys:
            ax.semilogy(ts, rel[key], label=key)
        ax.axhline(V1_RTOL, color=plt.rcParams['text.color'], ls='--', lw=1, label='tolerance')
        ax.set(xlabel='CMB temperature [K]', ylabel='|Relative difference|')
        ax.minorticks_off()
        _legend_outside(fig, ax)
        return fig

    validation_plot(terms, 'core_tabulated_constant_property_terms')
    validation_plot(difference, 'core_tabulated_constant_property_difference')


@pytest.mark.physics_invariant
def test_heat_change_is_the_integral_of_the_effective_capacity(paleos_budget):
    """On the PALEOS isentrope the booked heat change equals the quadrature of the capacity
    the solver integrates, across the onset of the inner core, across freeze-out and away from
    both, to 1e-6. The largest difference, 3.2e-7 on the piece from onset to freeze-out
    (measured, with the reference converged from 200 to 1600 nodes), is the error of the
    48-point panel of ``heat_change`` that the Gaussian path also uses."""
    b = paleos_budget
    t_on = b.t_onset
    cap = jax.jit(jax.vmap(b.effective_capacity))
    x, w = np.polynomial.legendre.leggauss(400)

    def integral(t_a, t_b):
        """GL integral of the capacity; below onset in u = sqrt(t_on - T), smooth there."""
        if t_a >= t_on and t_b >= t_on:
            mid, half = 0.5 * (t_a + t_b), 0.5 * (t_b - t_a)
            return half * float(np.sum(w * np.asarray(cap(mid + half * x))))
        u_a, u_b = np.sqrt(t_on - t_a), np.sqrt(t_on - t_b)
        u = 0.5 * (u_a + u_b) + 0.5 * (u_b - u_a) * x
        return 0.5 * (u_b - u_a) * float(np.sum(w * np.asarray(cap(t_on - u**2)) * -2.0 * u))

    # The cases cross the onset, stay below it, cross freeze-out (the boundary terms jump to
    # zero there), and stay above the onset.
    for t0, dt in ((t_on + 80.0, -200.0), (t_on - 30.0, -45.0), (t_on + 400.0, -10.0)):
        t1 = t0 + dt
        cuts = [t for t in (t_on, b.t_freeze) if min(t0, t1) < t < max(t0, t1)]
        edges = [t0, *sorted(cuts, reverse=dt < 0), t1]
        ref = sum(integral(a, c) for a, c in zip(edges[:-1], edges[1:]))
        assert b.heat_change(t0, dt) == pytest.approx(ref, rel=1e-6)
    assert t_on + 80.0 - 200.0 < b.t_freeze < t_on  # the first case crosses both
    assert b.heat_content(t_on - 50.0) - b.heat_content(t_on + 50.0) == pytest.approx(
        b.heat_change(t_on + 50.0, -100.0), rel=1e-9
    )


@pytest.mark.physics_invariant
def test_onset_is_the_isentrope_that_meets_the_melting_curve_at_the_centre(paleos_budget):
    """At ``t_onset`` the centre temperature equals the pure-iron melting point at the centre
    pressure; 1 K above, the core is liquid, 1 K below it has an inner core."""
    b = paleos_budget
    p_cen = float(b.profiles.pressure(0.0))
    assert float(b.profiles.t_cen(b.t_onset)) == pytest.approx(
        float(b.melting_curve.t_melt(p_cen)), rel=1e-12
    )
    assert float(b.r_icb(b.t_onset + 1.0)) == 0.0
    assert float(b.r_icb(b.t_onset - 1.0)) > 0.0


def test_tabulated_budget_refusals(paleos_budget):
    """A tabulated profile refuses stratification, the legacy mode, the entropy budget of the
    dynamo, a secular capacity without T_cmb, and a call state outside its grid."""
    p = paleos_budget.profiles
    with pytest.raises(ValueError, match='without stratification'):
        CoreEnergyBudget(p, IronMeltingCurve(), **BUDGET, stratification=True, k_core=40.0)
    with pytest.raises(ValueError, match='profile capacity mode'):
        CoreEnergyBudget(
            p,
            IronMeltingCurve(),
            ds_fusion=172.8,
            icn_width=10.0,
            capacity_mode='legacy',
            legacy_rho_core=1e4,
            legacy_tfac=1.1,
        )
    with pytest.raises(
        ValueError, match='dynamo diagnostics .entropy margin, field strength. are not computed'
    ):
        CoreEntropyBudget(paleos_budget, k_core=40.0)
    with pytest.raises(ValueError, match='needs t_cmb'):
        paleos_budget.secular_capacity()
    with pytest.raises(ValueError, match='outside the tabulated isentropes'):
        refuse_unmodelled_regime(paleos_budget, [4000.0, 7100.0])
    refuse_unmodelled_regime(paleos_budget, [6900.0, 6950.0])


def test_effective_tfac_of_a_tabulated_profile(paleos_budget):
    """The mean of T_a/T_cmb over the mass is C_s / (840 M) with the override, between 1 and
    T_cen/T_cmb, and rising with T_cmb on the PALEOS isentrope; without an override it is
    refused, since C_s then carries the table c_p."""
    b = paleos_budget
    tfac = float(b.effective_tfac(t_cmb=4000.0))
    assert tfac == pytest.approx(
        float(b.secular_capacity(t_cmb=4000.0)) / (840.0 * b.profiles.m_core), rel=1e-15
    )
    assert 1.0 < tfac < float(b.profiles.t_cen(4000.0)) / 4000.0
    assert float(b.effective_tfac(t_cmb=6000.0)) > tfac
    table_cp = CoreEnergyBudget(
        TabulatedCoreProfiles(
            core_arrays(GaussianCoreProfiles(**EARTH), 41),
            LiquidIronTable.from_file(CUT),
            c_p=None,
            t_cmb_range=(2500.0, 7000.0),
        ),
        IronMeltingCurve(),
        **BUDGET,
    )
    with pytest.raises(ValueError, match='constant c_p of an override'):
        table_cp.effective_tfac(t_cmb=4000.0)
