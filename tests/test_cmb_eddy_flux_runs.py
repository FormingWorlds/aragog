"""Coupled runs of the CMB eddy-flux Validation page, read from stored data.

``tests/validation/data/cmb_eddy_flux_runs.npz`` holds the helpfile columns, the CMB ledger of each
call, the CMB power sweep and the evaluation counts of coupled PROTEUS runs under the copy rule
(the CMB node copies the floored value of node 1) and under the CMB node rule (built by
``tests/reference/build_cmb_eddy_flux_runs.py``; versions in
``cmb_eddy_flux_runs_provenance.json``). These tests pin the run numbers that the page quotes and
draw its plots. They read stored data and run no solver.

Testing standards: ../docs/How-to/build_tests.md. Physics context:
../docs/Explanations/heat_transport.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60, method='signal')]

DATA = Path(__file__).parent / 'validation' / 'data' / 'cmb_eddy_flux_runs.npz'
STATUS_TARGET_TIME, STATUS_ERROR = (
    13,
    20,
)  # PROTEUS: completed at the target time; generic error
LEDGER_LIMIT = 10.0  # the code refuses a call whose ledger miss is above 10 times the bound
COPY, RULE = 'copy rule', 'CMB node rule'


@pytest.fixture(scope='module')
def runs():
    with np.load(DATA) as d:
        return {k: d[k] for k in d.files}


def _max_rb(runs, key):
    return float(np.max(np.abs(runs[f'{key}/ledger_rB'])))


def _side_diff(runs, key, step):
    p = runs[f'{key}/sweep_P_{step}']
    return np.abs(p[:, 1] - p[:, 0]), p


def test_case_38_runs_where_the_copy_rule_stops(runs, validation_plot):
    """Case 38 resumed at 6.769704 Myr: the copy rule stops on the ledger after 3 refused calls;
    the CMB node rule runs to 6.82 Myr with every call far inside the bound and no CMB heat row
    below -1e25 J."""
    assert int(runs['v4_main/status']) == STATUS_ERROR
    assert runs['v4_main/ledger_rB'].size == 3
    assert np.all(runs['v4_main/step_dE_F_cmb_J'] == 0.0)  # no interior step after the resume
    assert np.sum(np.abs(runs['v4_main/ledger_rB']) > LEDGER_LIMIT) == 3
    assert _max_rb(runs, 'v4_main') == pytest.approx(21.06, abs=0.005)
    assert int(runs['v4_tip/status']) == STATUS_TARGET_TIME
    assert runs['v4_tip/Time'][-1] >= 6.82e6
    assert runs['v4_tip/ledger_rB'].size == 14
    assert _max_rb(runs, 'v4_tip') == pytest.approx(5.2e-9, abs=0.05e-9)
    # the acceptance row of the plan; the copy rule also has no such row: it stops before
    assert np.all(runs['v4_tip/step_dE_F_cmb_J'] > -1.0e25)

    def figure():
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
        for key, lab, m in (('v4_main', COPY, 's'), ('v4_tip', RULE, 'o')):
            t = (runs[f'{key}/Time'] - 6.7697e6) / 1.0e3
            t1 = runs[f'{key}/ledger_t1'] / 1.0e3 - 6769.7
            ax0.plot(t, runs[f'{key}/F_cmb'], m, ms=4, mfc='none', label=lab)
            ax1.plot(t1, np.abs(runs[f'{key}/ledger_rB']), m, ms=4, mfc='none', label=lab)
        ax1.axhline(LEDGER_LIMIT, ls='--', color=plt.rcParams['text.color'], lw=0.8)
        ax1.text(30.0, LEDGER_LIMIT * 2.0, 'refusal limit', fontsize=8)
        ax0.set(ylabel='$F_\\mathrm{cmb}$ [W m$^{-2}$]')
        ax1.set(yscale='log', xlabel='Time after 6.7697 Myr [kyr]', ylabel='$|r|/B$')
        ax0.legend(fontsize=8)
        ax0.text(0.02, 0.06, '(a)', transform=ax0.transAxes)
        ax1.text(0.02, 0.06, '(b)', transform=ax1.transAxes)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'cmb_eddy_flux_case38_resume')


def test_case_38_cmb_power_has_no_step_at_s1_equal_s0(runs, validation_plot):
    """At every output point of the first calls of case 38, with S1 moved to S0 -+ 1e-9 or 1e-12
    J/kg/K: the copy rule steps by 9.105e16 to 9.212e16 W at both steps; the CMB node rule
    differs at the round-off of its 8.0e11 W (1e-9) or not at all (1e-12)."""
    d9_copy, _ = _side_diff(runs, 'v4_main', '1e-09')
    d12_copy, _ = _side_diff(runs, 'v4_main', '1e-12')
    assert int(runs['v4_main/sweep_calls']) == 3 and d9_copy.size == 416
    assert d9_copy.min() == pytest.approx(9.105e16, rel=5e-4)
    assert d9_copy.max() == pytest.approx(9.212e16, rel=5e-4)
    assert d12_copy.min() == pytest.approx(9.105e16, rel=5e-4)
    d9_rule, p9_rule = _side_diff(runs, 'v4_tip', '1e-09')
    d12_rule, _ = _side_diff(runs, 'v4_tip', '1e-12')
    assert int(runs['v4_tip/sweep_calls']) == 5 and d9_rule.size == 544
    p_rule = np.abs(p9_rule)
    assert p_rule.min() == pytest.approx(7.99e11, rel=1e-3)
    assert p_rule.max() == pytest.approx(8.00e11, rel=1e-3)
    assert d9_rule.max() <= 9.8e-4 and np.max(d9_rule / p_rule[:, 0]) < 1.3e-15
    assert np.all(d12_rule == 0.0)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, d, lab, m in (
            ('v4_main', d9_copy, f'{COPY}, step 1e-9', 's'),
            ('v4_tip', np.maximum(d9_rule, 1.0e-6), f'{RULE}, step 1e-9', 'o'),
        ):
            t = (runs[f'{key}/sweep_t'] - 6.7697e6) / 1.0e3
            ax.plot(t, d, m, ms=3, mfc='none', label=lab, rasterized=True)
        ax.set(
            yscale='log',
            xlabel='Time after 6.7697 Myr [kyr]',
            ylabel='|CMB power difference| [W]',
            ylim=(1.0e-7, 1.0e18),
        )
        ax.legend(fontsize=8, loc='center right')
        fig.tight_layout()
        return fig

    validation_plot(figure, 'cmb_eddy_flux_case38_sweep')


def test_constant_mixing_length_resume_keeps_the_ledger(runs, validation_plot):
    """1 Earth mass, constant mixing length, resumed at 37854.6 yr to 38600 yr: both rules close
    the ledger in every call; the CMB node rule follows the copy rule to 1.7e-3 in F_cmb and to
    1.7e-9 in T_cmb."""
    for key in ('v6_main', 'v6_tip'):
        assert int(runs[f'{key}/status']) == STATUS_TARGET_TIME
        assert runs[f'{key}/ledger_rB'].size == 4
    assert _max_rb(runs, 'v6_main') == pytest.approx(0.28, abs=0.005)
    assert _max_rb(runs, 'v6_tip') == pytest.approx(0.30, abs=0.005)
    for key in ('v6_main', 'v6_tip'):  # the first row carries heat into the core
        assert runs[f'{key}/F_cmb'][0] == pytest.approx(-592.0, abs=0.1)
    t = runs['v6_main/Time']
    for col, rel in (('F_cmb', 1.7e-3), ('T_cmb', 1.7e-9)):
        rule = np.interp(t, runs['v6_tip/Time'], runs[f'v6_tip/{col}'])
        d = np.max(np.abs(rule / runs[f'v6_main/{col}'] - 1.0))
        assert d == pytest.approx(rel, rel=0.03)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, lab, m in (('v6_main', COPY, 's'), ('v6_tip', RULE, 'o')):
            t_kyr = runs[f'{key}/Time'] / 1.0e3
            ax.plot(t_kyr, runs[f'{key}/F_cmb'], m, ms=6, mfc='none', label=lab)
        ax.set(xlabel='Time [kyr]', ylabel='CMB heat flux [W m$^{-2}$]')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'cmb_eddy_flux_constant_resume')


def test_run_from_t0_differs_only_at_the_graph_round_off(runs):
    """1 Earth mass, constant mixing length, t = 0 to 5 kyr: the CMB node rule never differs from
    the copy rule in a CMB evaluation (0 of 88499 JAX, 0 of 454 NumPy); its run equals the copy
    rule with a value-neutral edit of the same line bit for bit (as a second run of the copy rule
    and the rule run with a debug callback equal the copy rule); against the copy rule it and the
    edit differ in 136 of 1040 helpfile columns, first at 2 yr by 9.9e-9 in F_cmb and by at most 25.6 W m^-2
    by 5 kyr."""
    assert int(runs['v7_tip/jax_evals']) == 88499 and int(runs['v7_tip/jax_differing']) == 0
    assert int(runs['v7_tip/numpy_evals']) == 454 and int(runs['v7_tip/numpy_differing']) == 0
    for col in ('Time', 'F_cmb', 'T_cmb', 'step_dE_F_cmb_J'):
        assert np.array_equal(runs[f'v7_tip/{col}'], runs[f'v7_main_graph/{col}'])
    assert int(runs['v7_columns_total']) == 1040
    for pair in ('v7_tip_vs_graph', 'v7_repeat_vs_main', 'v7_callback_vs_main'):
        assert bool(runs[f'{pair}/only_runtime']), (
            pair
        )  # equal in every column but the run time
    for pair in ('v7_tip_vs_main', 'v7_graph_vs_main'):
        assert int(runs[f'{pair}/columns_differ']) == 136, pair
    a, b = runs['v7_main/F_cmb'], runs['v7_tip/F_cmb']
    first = int(np.argmax(a != b))
    assert first == 4 and runs['v7_main/Time'][first] == 2.0
    assert abs(b[first] / a[first] - 1.0) == pytest.approx(9.9e-9, rel=0.01)
    assert np.max(np.abs(b - a)) == pytest.approx(25.6, abs=0.05)


def test_nearest_mixing_length_resume_runs_where_the_copy_rule_stops(runs, validation_plot):
    """1 Earth mass, nearest-boundary mixing length, resumed at 30169.3 yr: the copy rule cuts the
    core off (F_cmb about 1e-3 W m^-2) and stops on the ledger after its row at 30952 yr; the CMB
    node rule runs to 32500 yr with 3 of 12 calls above the bound and all below 10 B."""
    assert int(runs['v9_main/status']) == STATUS_ERROR
    assert runs['v9_main/Time'][-1] == pytest.approx(30952.1, abs=0.05)
    assert runs['v9_main/F_cmb'].min() == pytest.approx(1.4e-3, rel=0.01)
    assert _max_rb(runs, 'v9_main') == pytest.approx(787, abs=0.5)
    assert int(runs['v9_tip/status']) == STATUS_TARGET_TIME
    assert runs['v9_tip/Time'][-1] == 32500.0
    rb = np.abs(runs['v9_tip/ledger_rB'])
    assert rb.size == 12 and np.sum(rb > 1.0) == 3 and rb.max() < LEDGER_LIMIT
    np.testing.assert_allclose(np.sort(rb[rb > 1.0]), [1.160, 1.248, 4.323], atol=5e-4)
    f = runs['v9_tip/F_cmb'][runs['v9_tip/Time'] > 30200.0]
    assert f.min() == pytest.approx(128.1, abs=0.05) and f.max() == pytest.approx(6092, abs=0.5)
    # the call average against the end state of the call at 31773 yr
    k = int(np.argmin(np.abs(runs['v9_tip/call_t1'] - 31773.0)))
    assert runs['v9_tip/call_F_avg'][k] == pytest.approx(1776, abs=0.5)
    assert runs['v9_tip/call_F_end'][k] == pytest.approx(7.738e4, rel=5e-4)
    assert runs['v9_tip/call_g_cmb_end'][k] == pytest.approx(-4.0e-4, rel=0.01)
    # the calls above 0.3 B are the calls with an in-call peak of the CMB flux
    crb, peak = runs['v9_tip/call_rB'], runs['v9_tip/call_F_max']
    f_end = runs['v9_tip/call_F_end'][crb > 0.3]
    assert np.sum(crb > 0.3) == 8 and np.all(peak[crb > 0.3] > 2.0e4)
    assert peak[crb > 0.3].min() == pytest.approx(2.8e4, rel=0.01)
    assert peak[crb > 0.3].max() == pytest.approx(6.9e5, rel=0.01)
    assert np.sum((f_end >= 130.0) & (f_end <= 350.0)) == 6
    assert np.sum(peak < 3.0e3) == 4 and np.all(crb[peak < 3.0e3] < 0.004)
    assert np.max(runs['v9_tip/call_Etrap_B']) <= 0.01

    def figure():
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.8, 4.8), sharex=True)
        t_main = runs['v9_main/Time'] / 1.0e3
        ax0.plot(t_main, runs['v9_main/F_cmb'], 's', ms=5, mfc='none', label='copy, average')
        for name, m, kw, lab in (
            ('F_avg', 'o', {}, 'rule, average'),
            ('F_max', '^', {'mfc': 'none'}, 'rule, peak'),
            ('F_end', 'v', {'mfc': 'none'}, 'rule, end'),
        ):
            t1 = runs['v9_tip/call_t1'] / 1.0e3
            ax0.plot(t1, runs[f'v9_tip/call_{name}'], m, ms=5, label=lab, **kw)
        ax0.set(yscale='log', ylabel='$F_\\mathrm{cmb}$ [W m$^{-2}$]')
        ax0.legend(fontsize=7, loc='lower right', ncol=2)
        t1_main = runs['v9_main/ledger_t1'] / 1.0e3
        rb_main = np.abs(runs['v9_main/ledger_rB'])
        ax1.plot(t1_main, rb_main, 's', ms=5, mfc='none', label=COPY)
        ax1.plot(runs['v9_tip/ledger_t1'] / 1.0e3, rb, 'o', ms=5, label=RULE)
        ax1.axhline(LEDGER_LIMIT, ls='--', color=plt.rcParams['text.color'], lw=0.8)
        ax1.set(yscale='log', xlabel='Time [kyr]', ylabel='$|r|/B$')
        ax1.legend(fontsize=7, loc='lower right')
        ax0.text(0.02, 0.88, '(a)', transform=ax0.transAxes)
        ax1.text(0.02, 0.88, '(b)', transform=ax1.transAxes)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'cmb_eddy_flux_nearest_resume')


def _seal_window(runs, key):
    """Rows of a 60 kyr seal run inside the low-flux interval of the copy rule, 30.5 to 57.3 kyr."""
    t = runs[f'{key}/Time']
    return runs[f'{key}/F_cmb'][(t > 30500.0) & (t < 57300.0)]


def _quoted(value, digits):
    """``value`` rounded to ``digits`` significant digits, as the page quotes it."""
    return float(f'{value:.{digits - 1}e}')


SEAL_RUNS = {8: 'v9x_tip', 9: 'v9x_tip_r9', 10: 'v9x_tip_r10'}  # aragog rtol 1e-8, 1e-9, 1e-10


def test_seal_case_to_60_kyr_keeps_the_core_coupled(runs, validation_plot):
    """The #113 case resumed at 30169.3 yr to 60 kyr: under the copy rule of an older build F_cmb
    falls to 1e-3 to 0.1 W m^-2 from 30.5 to 57.3 kyr and the lowest cell falls 428 K below the CMB
    node; under the CMB node rule F_cmb keeps a median of 1953 W m^-2 and the cell stays 8.2 to
    9.6 K below the node."""
    copy, rule = 'v9x_copy', 'v9x_tip'
    assert int(runs[f'{copy}/status']) == STATUS_TARGET_TIME
    for key in SEAL_RUNS.values():
        assert int(runs[f'{key}/status']) == STATUS_TARGET_TIME
        assert runs[f'{key}/Time'][-1] == 60000.0
    # the rows of the #113 table, from the copy-rule run
    t = runs[f'{copy}/Time']
    k = int(np.argmin(np.abs(t - 30546.0)))
    assert runs[f'{copy}/F_cmb'][k] == pytest.approx(1.4e-3, rel=0.01)
    k = int(np.argmin(np.abs(t - 50000.0)))
    assert runs[f'{copy}/T_cmb'][k] == pytest.approx(4811.5, abs=0.05)
    assert runs[f'{copy}/T_cmb_node'][k] == pytest.approx(5239.9, abs=0.05)
    window = _seal_window(runs, copy)
    assert np.median(window) == pytest.approx(0.10, abs=0.005)
    assert window.min() == pytest.approx(1.4e-3, rel=0.01) and window.max() < 0.105
    assert np.median(_seal_window(runs, rule)) == pytest.approx(1953, abs=0.5)
    assert np.sum(runs[f'{copy}/F_cmb'] < 1.0) == 116 and runs[f'{copy}/F_cmb'].size == 130
    assert np.sum(runs[f'{rule}/F_cmb'] < 1.0) == 1 and runs[f'{rule}/F_cmb'].size == 131
    d_copy = runs[f'{copy}/T_cmb'] - runs[f'{copy}/T_cmb_node']
    d_rule = runs[f'{rule}/T_cmb'] - runs[f'{rule}/T_cmb_node']
    assert d_copy.min() == pytest.approx(-428.5, abs=0.05)
    assert d_rule.min() == pytest.approx(-9.55, abs=0.005)
    assert d_rule.max() == pytest.approx(-8.17, abs=0.005)
    rb = np.abs(runs[f'{rule}/ledger_rB'])
    assert rb.size == 131 and rb.max() < LEDGER_LIMIT
    low = runs[f'{rule}/Time'][runs[f'{rule}/F_cmb'] < 1.0]
    assert low[0] == pytest.approx(43665, abs=0.5)
    t1, crb = runs[f'{rule}/call_t1'], np.abs(runs[f'{rule}/call_rB'])
    k = int(np.argmin(np.abs(t1 - low[0])))
    top = np.sort(crb)[::-1]
    assert crb[k - 1] == top[0] and crb[k + 1] == top[2] and crb[k] < 1.0e-4
    assert [_quoted(top[0], 3), _quoted(top[2], 3)] == [7.20, 5.69]
    assert all(np.sum(runs[f'{k}/F_cmb'] < 1.0) == 0 for k in ('v9x_tip_r9', 'v9x_tip_r10'))
    t = runs[f'{copy}/Time'][runs[f'{copy}/F_cmb'] > 1.0e4]
    assert t.size == 1 and t[0] == pytest.approx(57.5e3, abs=100.0)

    def figure():
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.8, 4.8), sharex=True)
        for key, ls, lab, d in ((copy, '--', COPY, d_copy), (rule, '-', RULE, d_rule)):
            tk = runs[f'{key}/Time'] / 1.0e3
            ax0.plot(tk, runs[f'{key}/F_cmb'], ls, lw=1.2, label=lab)
            ax1.plot(tk, d, ls, lw=1.2)
        ax0.set(yscale='log', ylabel='$F_\\mathrm{cmb}$ [W m$^{-2}$]')
        ax0.legend(fontsize=7, loc='center left', bbox_to_anchor=(0.55, 0.5))
        ax1.set(xlabel='Time [kyr]', ylabel='$T_\\mathrm{cmb} - T_\\mathrm{cmb,node}$ [K]')
        ax0.text(0.02, 0.88, '(a)', transform=ax0.transAxes)
        ax1.text(0.02, 0.45, '(b)', transform=ax1.transAxes)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'cmb_eddy_flux_seal_60kyr')


def test_seal_case_ledger_misses_fall_with_rtol(runs, validation_plot):
    """The misses of the energy check in the 60 kyr seal run at aragog rtol 1e-8, 1e-9 and 1e-10
    (at most 7.20, 5.51 and 0.061 B), and single calls rerun from a stored start at rtol 1e-8 to
    1e-11. Each value the page quotes equals the stored value rounded to the quoted digits."""
    rb = {r: np.abs(runs[f'{k}/ledger_rB']) for r, k in SEAL_RUNS.items()}
    for r, (top, digits, n_above) in {
        8: (7.20, 3, 15),
        9: (5.51, 3, 13),
        10: (0.061, 2, 0),
    }.items():
        k = SEAL_RUNS[r]
        assert float(runs[f'{k}/rtol']) == 10.0**-r and rb[r].size == 131
        np.testing.assert_array_equal(runs[f'{k}/ledger_rB'], runs[f'{k}/call_rB'])
        assert _quoted(rb[r].max(), digits) == top and np.sum(rb[r] > 1.0) == n_above
    steps = [int(runs[f'{k}/cvode_steps']) for k in SEAL_RUNS.values()]
    assert steps == [61781, 66110, 34917]
    # single calls: (call end [yr], {rtol exponent: |r|/B as quoted on the page})
    table = {
        'c43151': (43407.5, {8: 7.20, 9: 6.1e-5, 10: 6.8e-5, 11: 9.2e-6}),
        'c32187': (32394.5, {8: 4.32, 9: 0.24}),
        'c31773': (31979.6, {9: 5.51, 10: 0.066, 11: 2.7e-4}),
    }
    q = {}
    for call, (end, quoted) in table.items():
        for r, value in quoted.items():
            key = f'rtol_calls/{call}_r{r}'
            assert float(runs[f'{key}/rtol']) == 10.0**-r
            assert float(runs[f'{key}/t1']) == pytest.approx(end, abs=0.05)
            assert _quoted(float(runs[f'{key}/rB']), 3 if value > 1.0 else 2) == value
            q[call, r] = float(runs[f'{key}/q'])
    tight = [q['c43151', r] for r in (9, 10, 11)]
    assert max(tight) / min(tight) - 1.0 < 1.0e-5
    assert _quoted(q['c43151', 8] / tight[0], 3) == 3.59
    assert _quoted(q['c32187', 8] / q['c32187', 9], 2) == 3.2
    assert [_quoted(q['c31773', r], 3) for r in (9, 10, 11)] == [2.84e27, 2.63e27, 2.16e27]

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        rank = np.arange(1, 132)
        for r, ls in ((8, '-'), (9, '--'), (10, ':')):
            ax.plot(rank, np.sort(rb[r])[::-1], ls, lw=1.4, label=f'rtol $10^{{-{r}}}$')
        ax.axhline(LEDGER_LIMIT, ls='-.', color=plt.rcParams['text.color'], lw=0.8)
        ax.axhline(1.0, ls=(0, (1, 3)), color=plt.rcParams['text.color'], lw=0.8)
        ax.set(yscale='log', xlabel='Calls, largest miss first', ylabel='$|r|/B$')
        ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.0, 0.78))
        fig.tight_layout()
        return fig

    validation_plot(figure, 'cmb_eddy_flux_seal_rtol')
