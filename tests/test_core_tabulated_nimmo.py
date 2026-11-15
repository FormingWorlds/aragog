"""The Earth core budget of Nimmo (2015, ch. 8.02, Tables 2 and 4) on the PALEOS isentrope.

The structure, melting curve, latent heat and density jump are those of Nimmo's Table 2 (the
inputs of ``docs/Validation/core.md``, section 5); only the temperature profile changes, from
the small-radius Gaussian adiabat (alpha 1.25e-5, c_p 840) to the PALEOS liquid isentrope with
c_p 840 in the secular term. The terms are compared with Table 4 at the inner-core radius of
Table 2 and held only to wide bounds on the direction of the change: the isentrope changes where
and how fast the inner core grows, so Table 4 is no reference for them.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from aragog.core.budget import CoreEnergyBudget
from aragog.core.iron_table import LiquidIronTable
from aragog.core.melting import QuadraticMeltingCurve
from aragog.core.profiles import GaussianCoreProfiles
from aragog.core.profiles_tabulated import TabulatedCoreProfiles
from tests.conftest import (
    PALEOS_CUT,
    gaussian_core_arrays,
    legend_outside,
    reference_heat_change,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(300)]

GYR = 1e9 * 365.25 * 86400.0
Q_CMB = 15.2e12  # W, the first heat flow of Table 4
R_ICB = 1220e3  # m, Table 2
NIMMO_STRUCTURE = dict(
    rho_cen=12500.0, length_scale=7272e3, r_cmb=3480e3, p_cmb=139e9, alpha=1.25e-5, c_p=840.0
)
# Table 4, p. 46, 'This work (K = 0)' at 15.2 TW: TW, TW, TW, K/Gyr, km/Gyr.
TABLE4 = {'Q_s': 6.1, 'Q_L': 5.7, 'Q_g': 3.4, 'cooling': 104.0, 'growth': 1050.0}


@pytest.fixture(scope='module')
def budgets():
    """The Gaussian budget of section 5 and the same structure on the PALEOS isentrope."""
    g = GaussianCoreProfiles(
        **NIMMO_STRUCTURE, pressure_mode='labrosse', adiabat_mode='small_radius'
    )
    tab = TabulatedCoreProfiles(
        gaussian_core_arrays(g, 801), LiquidIronTable.from_file(PALEOS_CUT)
    )
    curve = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    kw = dict(
        ds_fusion=170.0, icn_width=10.0, latent_heat=750e3, alpha_c=1.0, c_light=560.0 / 12150.0
    )
    return CoreEnergyBudget(g, curve, **kw), CoreEnergyBudget(tab, curve, **kw)


def _terms(b, t):
    """Table 4 quantities at CMB temperature ``t`` [K] under ``Q_CMB``."""
    cooling = -float(b.dtcmb_dt(t, Q_CMB))  # K/s
    c_s = float(b.secular_capacity(t_cmb=t)) if b.tabulated else float(b.secular_capacity())
    return {
        'Q_s': c_s * cooling / 1e12,
        'Q_L': float(b.latent_capacity(t)) * cooling / 1e12,
        'Q_g': float(b.gravitational_capacity(t)) * cooling / 1e12,
        'cooling': cooling * GYR,
        'growth': float(b._boundary_sensitivity(t)) * cooling * GYR / 1e3,
    }


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_nimmo_earth_budget_on_the_paleos_isentrope(budgets, validation_plot):
    """At 4180 K the PALEOS isentrope holds no inner core (its centre lies above Nimmo's centre
    melting point; onset near 3874 K against 4239 K on the Gaussian adiabat). At the CMB
    temperature that gives Nimmo's 1220 km inner core, the terms are reported against Table 4:
    the PALEOS isentrope raises |dr_icb/dT_cmb| 1.67 times (measured), so the latent and
    gravitational capacities rise, the cooling rate falls to 0.69 of the table, and Q_L, Q_g
    and the growth rate end near 1.18 times the table; wide bounds hold these directions. Held
    exactly: Q_L/Q_g equals the Gaussian value
    to 1e-6 (both scale with the boundary sensitivity, on one structure), the latent heat per
    unit mass is the 750 kJ/kg of Table 2, and the Gaussian path stays within 2 % of Table 4
    on Q_s, Q_L, Q_g and the cooling rate."""
    gauss, paleos = budgets
    assert float(paleos.r_icb(4180.0)) == 0.0
    assert float(gauss.r_icb(4180.0)) > 1.1e6
    assert paleos.t_onset < 4180.0 < gauss.t_onset
    t_g, t_p = (
        brentq(
            lambda t, b=b: float(b.r_icb(t)) - R_ICB,
            b.t_freeze + 1.0,
            b.t_onset - 1e-3,
            xtol=1e-6,
        )
        for b in (gauss, paleos)
    )
    ref, got = _terms(gauss, t_g), _terms(paleos, t_p)
    assert got['Q_L'] / got['Q_g'] == pytest.approx(ref['Q_L'] / ref['Q_g'], rel=1e-6)
    radius = float(paleos.r_icb(t_p))
    area_mass = float(paleos.profiles.density(radius)) * 4.0 * np.pi * radius**2
    latent = float(paleos.latent_capacity(t_p)) / (
        area_mass * float(paleos._boundary_sensitivity(t_p))
    )
    assert latent == pytest.approx(750e3, rel=1e-12)
    for key in ('Q_s', 'Q_L', 'Q_g', 'cooling'):
        assert ref[key] == pytest.approx(TABLE4[key], rel=0.02), key
    sensitivity = float(paleos._boundary_sensitivity(t_p)) / float(
        gauss._boundary_sensitivity(t_g)
    )
    assert 1.5 < sensitivity < 1.8
    for key in ('Q_L', 'Q_g', 'growth'):
        assert got[key] / TABLE4[key] > 1.1, key
    for key in ('Q_s', 'cooling'):
        assert got[key] / TABLE4[key] < 0.8, key

    def figure():
        import matplotlib.pyplot as plt

        names = list(TABLE4)
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(x, [ref[n] / TABLE4[n] for n in names], 'o', label='Gaussian adiabat')
        ax.plot(x, [got[n] / TABLE4[n] for n in names], 's', label='PALEOS isentrope')
        ax.axhline(1.0, color=plt.rcParams['text.color'], lw=1, ls=':', label='Table 4')
        ax.set_xticks(x, ['$Q_s$', '$Q_L$', '$Q_g$', '$|\\dot T_c|$', '$\\dot r_i$'])
        ax.set(ylabel='Ratio to Table 4', ylim=(0.5, 1.5))
        legend_outside(fig, ax)
        return fig

    validation_plot(figure, 'core_tabulated_nimmo_budget')


@pytest.mark.physics_invariant
def test_the_heat_change_of_a_growing_inner_core_is_the_integral_of_its_capacity(budgets):
    """While the inner core grows on the PALEOS isentrope, the booked heat change equals a
    composite quadrature of the effective capacity to 2e-7 (the reference converges to 3e-8
    between 64 and 256 panels). The latent and gravitational capacities have kinks where the
    boundary crosses a radial node; one smooth panel across them missed by up to 3.1e-6, panels
    cut there agree with the reference within its own convergence (measured). The cases cool
    and heat, cross the onset, and cross freeze-out at 3818 K."""
    _, paleos = budgets
    for t0, dt in (
        (3870.0, -30.0),
        (3873.5, -40.0),
        (3900.0, -60.0),
        (3860.0, 40.0),
        (3900.0, -100.0),
    ):
        ref = reference_heat_change(paleos, t0, dt)
        assert paleos.heat_change(t0, dt) == pytest.approx(ref, rel=2e-7), (t0, dt)
