"""Validation of the CMB closure: the core energy ledger of one call across the melt band.

Bower et al. (2018, Eqs. 19-21, https://doi.org/10.1016/j.pepi.2017.11.004) close the core
with ``m c T_hat dT_cmb/dt = -E``. With the closure factor ``k = c_p,used (dT/dS)_tab / T``
(``(dT/dS)_tab`` the slope of the tabulated blend that the node temperature follows) the
ledger of a call misses by ``r = int (1 - k) E dt``. The case runs the solver with the slope
closure from a uniform melt coordinate, with the CMB node in the solid, the mush and above the
liquidus, compares the measured miss with that integral along the trajectory, and gives the
same integral for ``c_p/T`` of the blend along that trajectory. A separate test runs the
``c_p/T`` closure itself and checks its miss against its own integral.

The plot is drawn from the rows stored in ``ROWS_FILE``, not from the fresh rows: a start
perturbation of 1e-13 moves the plotted residuals by up to 2.7e4 times the 1e-6 summary
tolerance of the plot check. The fresh rows must agree with the stored rows within ``ROWS_TOL``;
``--validation-plots`` (the default folder) writes the rows again.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytest
from proteus_mpl.pytest_plugin import DEFAULT_DIR
from scipy.constants import Julian_year

from tests.conftest import EOS_DIR, assert_figure_legend_below, entropy_eos_copy, needs_eos
from tests.test_cmb_closure_ledger import _profile
from tests.test_entropy_solver_integration import _build_parameters

pytestmark = [pytest.mark.slow, needs_eos, pytest.mark.timeout(1800)]

# the melt start stays above the flat (padded) table cell at g 1 to 1.006, which has its own test
G0 = [-0.04, -0.02, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.15]
PREDICTION_TOL = 1.0e-3  # quadrature of E over the accepted steps
MUSH_TOL = 1.0e-3
ROWS_FILE = Path(__file__).parent / 'validation' / 'data' / 'cmb_closure_ledger_rows.json'
ROW_KEYS = ('g0', 'g', 'miss', 'pred', 'pred_cp')
# 10x the largest shift of a row value under a 1e-13 start perturbation (macOS): 1.6e-7 for g0
# 10x the largest shift of a row value over start perturbations of 1e-14 to 1e-11 and solver
# tolerances x0.99 and x1.01 (macOS); weak outside the band (g0 -0.04: 48 % of its miss, 1.15: 11x)
ROWS_TOL = {'band': 2.0e-5, -0.04: 3.0e-3, -0.02: 3.0e-5, 1.15: 1.0e-3}


def _call(eos, g0, cp_over_t=False):
    """One 2 yr energy_balance call from melt coordinate ``g0``: measured and predicted miss of
    the closure used, and the predicted miss of ``c_p/T`` along the same trajectory."""
    from aragog.solver.entropy_solver import EntropySolver

    params = _build_parameters(
        core_bc='energy_balance', n_nodes=15, end_time=2.0, inner_boundary_condition=1
    )
    params.solver.rtol = params.solver.atol = 1.0e-10
    solver = EntropySolver(params, entropy_eos=eos)
    solver.initialize()
    solver.set_initial_entropy(_profile(solver, eos, g0))
    if cp_over_t:  # the counterfactual closure, outside the reference case

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
    dt = np.diff(sol.t) * Julian_year

    def integral(k):
        w = (1.0 - k) * E
        return float(np.sum(0.5 * (w[:-1] + w[1:]) * dt))

    k_cp = cp / T * s_tab
    c_core = float(solver._core_cp * solver._core_tfac * solver._core_M)
    q = float(solver.get_state().step_dE_F_cmb_J)
    r = c_core * (T[-1] - T[0]) + q
    S_sol0 = float(np.ravel(eos.solidus_entropy(solver._P_basic_flat[:1]))[0])
    S_liq0 = float(np.ravel(eos.liquidus_entropy(solver._P_basic_flat[:1]))[0])
    return {
        'g': float((S_node[0] - S_sol0) / (S_liq0 - S_sol0)),
        'miss': r / abs(q),
        'pred': integral(k_cp if cp_over_t else s_tab / slope) / abs(q),
        'pred_cp': integral(k_cp) / abs(q),
        'mush': bool(np.all((S_node > S_sol0) & (S_node < S_liq0))),
        'check': solver.cmb_ledger,
        'atol': float(solver._call_tol[0]),
        'rtol': float(solver._call_tol[1]),
    }


def _table_hashes():
    """sha256 of each table file of ``EOS_DIR``."""
    return {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(Path(EOS_DIR).glob('*.dat'))
    }


def _write_rows(rows, sha):
    """Store the plotted values and the solver tolerances of ``rows`` with their provenance,
    one top-level block per line; refuse when ``src`` or this file has local changes."""
    root = Path(__file__).parents[1]

    def git(*args):
        return subprocess.run(
            ['git', *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    if git('status', '--porcelain', '--', 'src', 'tests/test_cmb_closure_validation.py'):
        pytest.fail(
            f'{ROWS_FILE.name}: commit the changes of src and this test before the rows'
        )
    provenance = {
        'commit': git('rev-parse', 'HEAD'),
        'platform': f'{platform.system()} {platform.machine()}',
        'tables': 'SPIDER EOS test data (ARAGOG_TEST_EOS_DIR)',
        'sha256': sha,
        'builder': 'pytest tests/test_cmb_closure_validation.py::'
        'test_the_cmb_ledger_across_the_melt_band --validation-plots',
    }
    keys = (*ROW_KEYS, 'atol', 'rtol')
    lines = [f'"provenance": {json.dumps(provenance)}']
    lines += [f'"rows": {json.dumps([{k: row[k] for k in keys} for row in rows])}']
    ROWS_FILE.write_text('{\n' + ',\n'.join(lines) + '\n}\n')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_cmb_ledger_across_the_melt_band(validation_plot, request):
    """The slope closure misses by ``int (1 - k) E dt`` within 1e-3 of the CMB heat (Bower et al.
    2018, Eqs. 19-21), closes in the mush to 1e-3 and inside the solver bound, while ``c_p/T``
    along the same trajectories would miss by more than 5 % in the mush."""
    eos = entropy_eos_copy(EOS_DIR)
    rows = [dict(_call(eos, g0), g0=g0) for g0 in G0]
    for row in rows:
        assert abs(row['miss'] - row['pred']) < PREDICTION_TOL, row
    mush = [row for row in rows if row['mush']]
    assert len(mush) >= 10
    for row in mush:
        assert (
            abs(row['miss']) < MUSH_TOL and abs(row['check']['r']) <= row['check']['bound']
        ), row
    assert max(abs(row['pred_cp']) for row in mush) > 0.05

    sha = _table_hashes()
    if request.config.getoption('validation_plots') == DEFAULT_DIR:
        _write_rows(rows, sha)
    data = json.loads(ROWS_FILE.read_text())
    assert data['provenance']['sha256'] == sha, f'{ROWS_FILE.name} was made from other tables'
    stored = data['rows']
    again = f'pytest {request.node.nodeid} --validation-plots writes the rows again'
    for row, ref in zip(rows, stored, strict=True):
        tol = ROWS_TOL['band'] if 0.0 < row['g0'] < 1.0 else ROWS_TOL[row['g0']]
        assert all(abs(row[k] - ref[k]) <= tol for k in ROW_KEYS[1:]), (row, ref, again)

    assert_figure_legend_below(_figure(stored))
    validation_plot(lambda: _figure(stored), 'core_cmb_closure_ledger')


@pytest.mark.physics_invariant
def test_the_cp_over_t_closure_misses_by_its_closed_form():
    """The ``c_p/T`` closure, run in place of the slope, misses by its own ``int (1 - k) E dt``
    within 1e-3 of the CMB heat, about 10 % in the mush, so the closed form of the reference
    case describes what that closure does."""
    eos = entropy_eos_copy(EOS_DIR)
    rows = [_call(eos, g0, cp_over_t=True) for g0 in (0.1, 0.5, 0.9, 0.99)]
    for row in rows:
        assert abs(row['miss'] - row['pred']) < PREDICTION_TOL, row
    assert max(abs(row['miss']) for row in rows) > 0.05


def _figure(rows):
    """Ledger miss of each call against the melt coordinate of the CMB node: the slope closure
    (points from the solver and its closed form) and the closed form
    of ``c_p/T`` along the same trajectories."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    g = [row['g'] for row in rows]
    ax.plot(g, [row['pred_cp'] for row in rows], color='C1', lw=2, alpha=0.6)
    ax.plot(g, [row['pred'] for row in rows], color='C0', lw=2, alpha=0.4)
    ax.plot(g, [row['miss'] for row in rows], 'o', ms=4, color='C0')
    ax.set_yscale('symlog', linthresh=1.0e-4)
    ticks = [-0.1, -1.0e-3, 0.0, 1.0e-3, 0.1]
    ax.set_yticks(ticks, ['-0.1', '-0.001', '0', '0.001', '0.1'])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks([0.0, 0.5, 1.0], ['0', '0.5', '1'])
    ax.axvspan(0.0, 1.0, alpha=0.08, color='C2')
    ax.set(xlabel='g of the CMB node', ylabel='miss r/|q|')
    handles = [
        Line2D([], [], color='C1', lw=2, alpha=0.6, label='c_p/T, closed form'),
        Line2D([], [], color='C0', lw=2, alpha=0.4, label='(dS/dT)_P, closed form'),
        Line2D([], [], color='C0', marker='o', ls='none', label='(dS/dT)_P, solver'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.2, 1.0, 1.0))
    return fig
