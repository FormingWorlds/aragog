"""Validation of the CMB closure: the core energy ledger of one call across the melt band.

Bower et al. (2018, Eqs. 19-21, https://doi.org/10.1016/j.pepi.2017.11.004) close the core
with ``m c T_hat dT_cmb/dt = -E``. With the closure factor ``k = c_p,used (dT/dS)_tab / T``
(``(dT/dS)_tab`` the slope of the tabulated blend that the node temperature follows) the
ledger of a call misses by ``r = int (1 - k) E dt``. The case runs the solver from a uniform
melt coordinate with the CMB node in the solid, the mush and above the liquidus, with the
present closure and with ``c_p/T`` of the blend, and compares the measured miss with that
integral along the trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import Julian_year

from tests.conftest import EOS_DIR, assert_figure_legend_below, entropy_eos_copy, needs_eos
from tests.test_entropy_solver_integration import _build_parameters

pytestmark = [
    pytest.mark.reference_pinned,
    pytest.mark.slow,
    needs_eos,
    pytest.mark.timeout(1800),
]

G0 = [-0.04, -0.02, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.03]
PREDICTION_TOL = 1.0e-3  # quadrature of E over the accepted steps
MUSH_TOL = 1.0e-3


def _call(eos, g0, old):
    """One 2 yr energy_balance call from melt coordinate ``g0``: measured and predicted miss."""
    from aragog.solver.entropy_solver import EntropySolver

    params = _build_parameters(
        core_bc='energy_balance', n_nodes=15, end_time=2.0, inner_boundary_condition=1
    )
    params.solver.rtol = params.solver.atol = 1.0e-10
    solver = EntropySolver(params, entropy_eos=eos)
    solver.initialize()
    P = np.asarray(solver._P_stag_flat, dtype=float)
    S_sol, S_liq = np.ravel(eos.solidus_entropy(P)), np.ravel(eos.liquidus_entropy(P))
    solver.set_initial_entropy(S_sol + g0 * (S_liq - S_sol))
    if old:

        def blend_cp():
            return float(np.asarray(solver.state.phase_basic.heat_capacity()).flat[0])

        solver._cmb_closure_cp = blend_cp
        solver._check_cmb_ledger = lambda *a, **k: None
    solver.solve()
    sol, n = solver._solution, solver._n_stag
    ev = solver._cmb_node_phase
    S_node = sol.y[0] - 0.5 * float(solver._cmb_dr_cmb) * sol.y[n]
    ev.set_pressure(np.full(S_node.size, float(solver._P_basic_flat[0])))

    def blend(S):
        ev.set_entropy(S)
        ev.update()
        return tuple(
            np.ravel(np.copy(f()))
            for f in (ev.temperature, ev.heat_capacity, ev.temperature_slope)
        )

    T, cp, slope = blend(S_node)
    s_tab = (blend(S_node + 1.0e-3)[0] - blend(S_node - 1.0e-3)[0]) / 2.0e-3
    E = np.array(
        [
            (solver._dSdt_single(float(t), y), float(solver.state._heat_flux[0]))[1]
            for t, y in zip(sol.t, np.asarray(sol.y, dtype=float).T)
        ]
    ) * float(solver._cmb_area)
    k = (cp / T if old else 1.0 / slope) * s_tab
    w = (1.0 - k) * E
    dt = np.diff(sol.t) * Julian_year
    r_pred = float(np.sum(0.5 * (w[:-1] + w[1:]) * dt))
    c_core = float(solver._core_cp * solver._core_tfac * solver._core_M)
    q = float(solver.get_state().step_dE_F_cmb_J)
    r = c_core * (T[-1] - T[0]) + q
    S_sol0 = float(np.ravel(eos.solidus_entropy(solver._P_basic_flat[:1]))[0])
    S_liq0 = float(np.ravel(eos.liquidus_entropy(solver._P_basic_flat[:1]))[0])
    return {
        'g': float((S_node[0] - S_sol0) / (S_liq0 - S_sol0)),
        'miss': r / abs(q),
        'pred': r_pred / abs(q),
        'padded': bool(np.any(s_tab <= 0.0)),
        'mush': bool(np.all((S_node > S_sol0) & (S_node < S_liq0))),
        'check': None if old else solver.cmb_ledger,
    }


def test_the_cmb_ledger_across_the_melt_band(validation_plot):
    """Each closure misses by ``int (1 - k) E dt`` within 1e-3 of the CMB heat; the present
    closure closes in the mush to 1e-3 and inside the solver bound."""
    eos = entropy_eos_copy(EOS_DIR)
    old = [_call(eos, g0, True) for g0 in G0]
    new = [_call(eos, g0, False) for g0 in G0]
    for row in old + new:
        assert abs(row['miss'] - row['pred']) < PREDICTION_TOL, row
    mush = [row for row in new if row['mush']]
    assert len(mush) >= 10
    for row in mush:
        assert (
            abs(row['miss']) < MUSH_TOL and abs(row['check']['r']) <= row['check']['bound']
        ), row
    assert max(abs(row['miss']) for row in old if row['mush']) > 0.05

    assert_figure_legend_below(_figure(old, new))
    validation_plot(lambda: _figure(old, new), 'core_cmb_closure_ledger')


def _figure(old, new):
    """Ledger miss of each call against the melt coordinate of the CMB node: solver points
    (hollow in a padded cell) and the closed-form integral (line), for both closures."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for rows, colour in ((old, 'C1'), (new, 'C0')):
        ax.plot(
            [row['g'] for row in rows],
            [row['pred'] for row in rows],
            color=colour,
            lw=2,
            alpha=0.4,
        )
        for padded in (False, True):
            sel = [row for row in rows if row['padded'] == padded]
            ax.plot(
                [row['g'] for row in sel],
                [row['miss'] for row in sel],
                'o',
                ms=4,
                color=colour,
                mfc='none' if padded else colour,
            )
    ax.set_yscale('symlog', linthresh=1.0e-4)
    ticks = [-0.1, -1.0e-3, 0.0, 1.0e-3, 0.1]
    ax.set_yticks(ticks, ['-0.1', '-0.001', '0', '0.001', '0.1'])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks([0.0, 0.5, 1.0], ['0', '0.5', '1'])
    ax.axvspan(0.0, 1.0, alpha=0.08, color='C2')
    ax.set(xlabel='g of the CMB node', ylabel='miss r/|q|')
    handles = [
        Line2D([], [], color='C1', marker='o', label='c_p/T'),
        Line2D([], [], color='C0', marker='o', label='(dS/dT)_P'),
        Line2D(
            [], [], color=plt.rcParams['axes.labelcolor'], lw=2, alpha=0.4, label='closed form'
        ),
        Line2D(
            [],
            [],
            color=plt.rcParams['axes.labelcolor'],
            marker='o',
            ls='none',
            mfc='none',
            label='padded cell',
        ),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.2, 1.0, 1.0))
    return fig
