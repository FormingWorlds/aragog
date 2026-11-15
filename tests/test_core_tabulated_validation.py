"""Validation of the tabulated core on the PALEOS liquid iron table and Zalmoxis cores.

The cores are the Zalmoxis core arrays of 1, 3 and 5 Earth-mass planets
(``tests/validation/data/zalmoxis_core_arrays_*.npz``, provenance beside them); the table is
the 40-per-decade PALEOS liquid iron cut to 3 TPa.
"""

from __future__ import annotations

import pathlib

import jax
import numpy as np
import pytest
from scipy import constants
from scipy.integrate import solve_ivp

from aragog.core.iron_table import LiquidIronTable
from aragog.core.melting import IronMeltingCurve
from aragog.core.module import _FACTORY_DEFAULTS, load_core_arrays
from aragog.core.profiles_tabulated import TabulatedCoreProfiles

pytestmark = [pytest.mark.unit, pytest.mark.timeout(300)]

DATA = pathlib.Path(__file__).parent / 'validation' / 'data'
CUT = DATA / 'paleos_iron_liquid_cut_3tpa.npz'
CORES = {
    '1 M_E': 'zalmoxis_core_arrays_1me_3000K.npz',
    '3 M_E': 'zalmoxis_core_arrays_3me_2510K.npz',
    '5 M_E': 'zalmoxis_core_arrays_5me_2510K.npz',
}
SEAM_RTOL = 1e-5


@pytest.fixture(scope='module')
def table():
    """The PALEOS liquid iron cut, 45 GPa to 3 TPa."""
    return LiquidIronTable.from_file(CUT)


@pytest.fixture(scope='module')
def cores(table):
    """Tabulated profiles of the 3 Zalmoxis cores."""
    return {
        k: TabulatedCoreProfiles(load_core_arrays(DATA / f), table) for k, f in CORES.items()
    }


def _legend_outside(fig, ax):
    """Put the legend right of the axes and check that it covers no data."""
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')
    fig.tight_layout()
    fig.canvas.draw()
    assert legend.get_window_extent().x0 >= ax.get_window_extent().x1


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_isentrope_equals_the_integral_of_nabla_ad(cores, table, validation_plot):
    """The isentrope the budget reads (the inverted table entropy on bicubic patches) equals
    the integral of ``d ln T / d ln P = nabla_ad`` of the same table from the CMB, to 1e-5 in
    T (measured at most 1.3e-7) on the 1, 3 and 5 M_E cores, 500 and 2000 K above the pure-iron
    melting point at the CMB; the plot shows the larger of the 2 at each pressure. A 1 % error in nabla_ad moves the centre by more than 1e-3."""
    curve = IronMeltingCurve()
    rows = []
    for name, prof in cores.items():
        p_cen = float(prof.pressure(0.0))
        for above in (500.0, 2000.0):
            t_cmb = float(curve.t_melt(prof.p_cmb)) + above
            for scale in (1.0, 1.01):
                sol = solve_ivp(
                    lambda lp, lt, s=scale: [
                        s * table.value('nabla_ad', np.exp(lp), np.exp(lt[0]))
                    ],
                    (np.log(prof.p_cmb), np.log(p_cen)),
                    [np.log(t_cmb)],
                    rtol=1e-12,
                    atol=1e-14,
                    dense_output=True,
                )
                r = np.linspace(0.0, prof.r_cmb, 300)
                p = np.asarray(prof.pressure(r))
                ode = np.exp(sol.sol(np.log(p))[0])
                rel = np.abs(np.asarray(prof.adiabat(r, t_cmb)) / ode - 1.0)
                if scale == 1.0:
                    assert rel.max() < SEAM_RTOL, (name, above, rel.max())
                    rows.append((name, above, p, rel))
                else:
                    assert rel[0] > 100 * SEAM_RTOL

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for name in cores:
            p = next(r[2] for r in rows if r[0] == name)
            worst = np.max([r[3] for r in rows if r[0] == name], axis=0)
            ax.semilogy(p[::6] / 1e9, np.maximum(worst, 1e-12)[::6], label=name)
        ax.axhline(SEAM_RTOL, color=plt.rcParams['text.color'], lw=1, ls=':', label='tolerance')
        ax.set(xlabel='Pressure [GPa]', ylabel='|Relative difference in T|')
        ax.minorticks_off()
        _legend_outside(fig, ax)
        return fig

    validation_plot(figure, 'core_tabulated_isentrope_seam')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a_pure_iron_core_freezes_from_the_centre(table, validation_plot):
    """The pure-iron melting curve (Anzellini et al. 2013, the PALEOS phase line) is steeper
    than the liquid isentrope at the melting point, ``dT_m/dP > nabla_ad T_m / P``, at every
    node of the 1, 3 and 5 M_E cores, so a pure-iron core freezes from the centre. The
    smallest margin is 5.4 % (measured, liquid branch read at T_m).
    A melting slope 6 % smaller would freeze from the top on the 1 and 3 M_E cores."""
    curve = IronMeltingCurve()
    slope_m = jax.vmap(jax.grad(curve.t_melt))
    margins = {}
    for name, f in CORES.items():
        p = np.sort(load_core_arrays(DATA / f)['P'])
        t_m = np.asarray(curve.t_melt(p))
        margins[name] = (
            p,
            np.asarray(slope_m(p)) / (table.value('nabla_ad', p, t_m) * t_m / p) - 1.0,
        )
    for name, (p, m) in margins.items():
        assert m.min() > 0.05, (name, m.min())
    for name in ('1 M_E', '3 M_E'):
        assert (0.94 * (1.0 + margins[name][1]) - 1.0).min() < 0.0

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        p = np.concatenate([v[0] for v in margins.values()])
        m = np.concatenate([v[1] for v in margins.values()])
        order = np.argsort(p)
        ax.plot(p[order] / 1e9, m[order], color=plt.rcParams['text.color'], label='margin')
        for k, (name, (pc, _)) in enumerate(margins.items()):
            ax.plot(
                [pc[0] / 1e9, pc[-1] / 1e9],
                [-0.12 - 0.08 * k] * 2,
                lw=4,
                color=f'C{k + 1}',
                label=name,
            )
        ax.axhline(0.0, color=plt.rcParams['text.color'], lw=1, ls=':')
        ax.set(xlabel='Pressure [GPa]', ylabel='Slope ratio minus 1', ylim=(-0.4, 1.1))
        _legend_outside(fig, ax)
        return fig

    validation_plot(figure, 'core_tabulated_pure_iron_direction')


@pytest.mark.reference_pinned
def test_the_entropy_of_fusion_against_the_ab_initio_value(validation_plot):
    """The entropy of fusion of the core budget, 172.8 J/kg/K, lies 10.5 % above the ab initio
    1.05 k_B per atom at 300 GPa of Gonzalez-Cataldo and Militzer (2023, p. 9), 156.3 J/kg/K
    for iron; the paper gives 0.8 k_B per atom at 1000 GPa (same page). Reported, not held to
    a tolerance."""
    per_atom = constants.k / (55.845 * constants.atomic_mass)  # J/kg/K per k_B per atom
    ab_initio = 1.05 * per_atom
    assert ab_initio == pytest.approx(156.3, abs=0.05)
    ratio = _FACTORY_DEFAULTS['ds_fusion'] / ab_initio
    assert ratio == pytest.approx(1.105, abs=5e-4)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot([300.0, 1000.0], [1.05 * per_atom, 0.8 * per_atom], 'o', label='ab initio')
        ax.axhline(_FACTORY_DEFAULTS['ds_fusion'], label='aragog constant')
        ax.set(xlabel='Pressure [GPa]', ylabel='Entropy of fusion [J/kg/K]', ylim=(0.0, 200.0))
        _legend_outside(fig, ax)
        return fig

    validation_plot(figure, 'core_entropy_of_fusion')
