"""Coupled runs of the CMB eddy-flux Validation page, read from stored data.

``tests/validation/data/cmb_eddy_flux_runs.npz`` holds the helpfile columns, the CMB ledger of each
call and the CMB power sweep of coupled PROTEUS runs on aragog main (548ea357) and on the CMB node
rule (built by ``tests/reference/build_cmb_eddy_flux_runs.py``; provenance in
``cmb_eddy_flux_runs_provenance.json``). These tests check each number that the page quotes and
draw the plots of the page.

Testing standards: ../docs/How-to/build_tests.md. Physics context:
../docs/Explanations/heat_transport.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.reference_pinned,
    pytest.mark.timeout(60, method='signal'),
]

DATA = Path(__file__).parent / 'validation' / 'data' / 'cmb_eddy_flux_runs.npz'
STATUS_TARGET_TIME, STATUS_ERROR = 13, 20
LEDGER_LIMIT = 10.0  # the code refuses a call whose ledger miss is above 10 times the bound


@pytest.fixture(scope='module')
def runs():
    with np.load(DATA) as d:
        return {k: d[k] for k in d.files}


def _max_rb(runs, key):
    return float(np.max(np.abs(runs[f'{key}/ledger_rB'])))


def _side_diff(runs, key, step):
    p = runs[f'{key}/sweep_P_{step}']
    return np.abs(p[:, 1] - p[:, 0]), p


def test_case_38_runs_through_the_old_stop(runs, validation_plot):
    """Case 38 resumed at 6.769704 Myr: main stops on the ledger, the rule runs to 6.82 Myr with every
    call far inside the bound and no CMB heat row below -1e25 J."""
    assert int(runs['v4_main/status']) == STATUS_ERROR
    assert _max_rb(runs, 'v4_main') == pytest.approx(21.06, abs=0.005)
    assert np.sum(np.abs(runs['v4_main/ledger_rB']) > LEDGER_LIMIT) == 3
    assert int(runs['v4_tip/status']) == STATUS_TARGET_TIME
    assert runs['v4_tip/Time'][-1] >= 6.82e6
    assert runs['v4_tip/ledger_rB'].size == 14
    assert _max_rb(runs, 'v4_tip') == pytest.approx(5.2e-9, abs=0.05e-9)
    assert np.all(runs['v4_tip/step_dE_F_cmb_J'] > -1.0e25)

    def figure():
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
        for key, lab, m in (('v4_main', 'main', 's'), ('v4_tip', 'CMB node rule', 'o')):
            t = (runs[f'{key}/Time'] - 6.7697e6) / 1.0e3
            ax0.plot(t, runs[f'{key}/F_cmb'], m, ms=4, mfc='none', label=lab)
            ax1.plot(
                runs[f'{key}/ledger_t1'] / 1.0e3 - 6769.7,
                np.abs(runs[f'{key}/ledger_rB']),
                m,
                ms=4,
                mfc='none',
                label=lab,
            )
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
    """At every output point of the first calls of case 38, moving S1 to S0 -+ 1e-9 or 1e-12 J/kg/K:
    main jumps by 9.105e16 W at both steps; the rule differs at round-off (1e-9) or not at all (1e-12)."""
    d9_main, p9_main = _side_diff(runs, 'v4_main', '1e-09')
    d12_main, _ = _side_diff(runs, 'v4_main', '1e-12')
    assert d9_main.size == 416
    assert d9_main.min() == pytest.approx(9.105e16, rel=5e-4)
    assert d12_main.min() == pytest.approx(9.105e16, rel=5e-4)
    d9_tip, p9_tip = _side_diff(runs, 'v4_tip', '1e-09')
    d12_tip, _ = _side_diff(runs, 'v4_tip', '1e-12')
    assert d9_tip.size == 544
    assert d9_tip.max() <= 9.8e-4 and np.max(d9_tip / np.abs(p9_tip[:, 0])) < 1.3e-15
    assert np.all(d12_tip == 0.0)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, d, lab, m in (
            ('v4_main', d9_main, 'main, step 1e-9', 's'),
            ('v4_tip', np.maximum(d9_tip, 1.0e-6), 'CMB node rule, step 1e-9', 'o'),
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
    """1 Earth mass, constant mixing length, resumed at 37854.6 yr to 38600 yr: both rules close the
    ledger in every call; the CMB flux of the rule follows main to 1.7e-3 and T_cmb to 1.7e-9."""
    for key in ('v6_main', 'v6_tip'):
        assert int(runs[f'{key}/status']) == STATUS_TARGET_TIME
        assert runs[f'{key}/ledger_rB'].size == 4
    assert _max_rb(runs, 'v6_main') == pytest.approx(0.28, abs=0.005)
    assert _max_rb(runs, 'v6_tip') == pytest.approx(0.30, abs=0.005)
    t = runs['v6_main/Time']
    for col, rel in (('F_cmb', 1.7e-3), ('T_cmb', 1.7e-9)):
        tip = np.interp(t, runs['v6_tip/Time'], runs[f'v6_tip/{col}'])
        d = np.max(np.abs(tip / runs[f'v6_main/{col}'] - 1.0))
        assert d == pytest.approx(rel, rel=0.03)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for key, lab, m in (('v6_main', 'main', 's'), ('v6_tip', 'CMB node rule', 'o')):
            ax.plot(
                runs[f'{key}/Time'] / 1.0e3,
                runs[f'{key}/F_cmb'],
                m,
                ms=6,
                mfc='none',
                label=lab,
            )
        ax.set(xlabel='Time [kyr]', ylabel='CMB heat flux [W m$^{-2}$]')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'cmb_eddy_flux_constant_resume')


def test_run_from_t0_differs_only_at_the_graph_round_off(runs):
    """1 Earth mass, constant mixing length, t = 0 to 5 kyr: the rule run equals main with a
    value-neutral edit of the same line bit for bit; against main it first differs at 2 yr by
    9.9e-9 in F_cmb and by at most 25.6 W m^-2 by 5 kyr."""
    for col in ('Time', 'F_cmb', 'T_cmb', 'step_dE_F_cmb_J'):
        assert np.array_equal(runs[f'v7_tip/{col}'], runs[f'v7_main_graph/{col}'])
    a, b = runs['v7_main/F_cmb'], runs['v7_tip/F_cmb']
    first = int(np.argmax(a != b))
    assert first == 4 and runs['v7_main/Time'][first] == 2.0
    assert abs(b[first] / a[first] - 1.0) == pytest.approx(9.9e-9, rel=0.01)
    assert np.max(np.abs(b - a)) == pytest.approx(25.6, abs=0.05)


def test_nearest_mixing_length_resume_runs_where_main_stops(runs, validation_plot):
    """1 Earth mass, nearest-boundary mixing length, resumed at 30169.3 yr: main stops on the ledger
    at 30952 yr; the rule runs to 32500 yr with 3 of 12 calls above the bound and all below 10 B."""
    assert int(runs['v9_main/status']) == STATUS_ERROR
    assert runs['v9_main/Time'][-1] == pytest.approx(30952.1, abs=0.05)
    assert int(runs['v9_tip/status']) == STATUS_TARGET_TIME
    assert runs['v9_tip/Time'][-1] == 32500.0
    rb = np.abs(runs['v9_tip/ledger_rB'])
    assert rb.size == 12 and np.sum(rb > 1.0) == 3 and rb.max() < LEDGER_LIMIT
    np.testing.assert_allclose(np.sort(rb[rb > 1.0]), [1.160, 1.248, 4.323], atol=5e-4)
    sel = runs['v9_tip/Time'] > 30200.0
    f = runs['v9_tip/F_cmb'][sel]
    assert f.min() == pytest.approx(128.1, abs=0.05) and f.max() == pytest.approx(6092, abs=0.5)
    # the call average against the end state of the call at 31773 yr
    k = int(np.argmin(np.abs(runs['v9_tip/call_t1'] - 31773.0)))
    assert runs['v9_tip/call_F_avg'][k] == pytest.approx(1776, abs=0.5)
    assert runs['v9_tip/call_F_end'][k] == pytest.approx(7.738e4, rel=5e-4)
    # the calls above 0.3 B are the calls with an in-call peak of the CMB flux
    crb, peak = runs['v9_tip/call_rB'], runs['v9_tip/call_F_max']
    assert np.sum(crb > 0.3) == 8 and np.all(peak[crb > 0.3] > 2.0e4)
    assert peak[crb > 0.3].min() == pytest.approx(2.8e4, rel=0.01)
    assert peak[crb > 0.3].max() == pytest.approx(6.9e5, rel=0.01)
    assert (
        np.sum(
            (runs['v9_tip/call_F_end'][crb > 0.3] >= 130.0)
            & (runs['v9_tip/call_F_end'][crb > 0.3] <= 350.0)
        )
        == 6
    )
    assert np.sum(peak < 3.0e3) == 4 and np.all(crb[peak < 3.0e3] < 0.004)
    assert np.max(runs['v9_tip/call_Etrap_B']) <= 0.01

    def figure():
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.8, 4.8), sharex=True)
        ax0.plot(
            runs['v9_main/Time'] / 1.0e3,
            runs['v9_main/F_cmb'],
            's',
            ms=5,
            mfc='none',
            label='main, average',
        )
        ax0.plot(
            runs['v9_tip/call_t1'] / 1.0e3,
            runs['v9_tip/call_F_avg'],
            'o',
            ms=5,
            label='rule, average',
        )
        ax0.plot(
            runs['v9_tip/call_t1'] / 1.0e3,
            runs['v9_tip/call_F_max'],
            '^',
            ms=5,
            mfc='none',
            label='rule, peak',
        )
        ax0.plot(
            runs['v9_tip/call_t1'] / 1.0e3,
            runs['v9_tip/call_F_end'],
            'v',
            ms=5,
            mfc='none',
            label='rule, end',
        )
        ax0.set(yscale='log', ylabel='$F_\\mathrm{cmb}$ [W m$^{-2}$]')
        ax0.legend(fontsize=7, loc='lower right', ncol=2)
        ax1.plot(
            runs['v9_main/ledger_t1'] / 1.0e3,
            np.abs(runs['v9_main/ledger_rB']),
            's',
            ms=5,
            mfc='none',
            label='main',
        )
        ax1.plot(runs['v9_tip/ledger_t1'] / 1.0e3, rb, 'o', ms=5, label='CMB node rule')
        ax1.axhline(LEDGER_LIMIT, ls='--', color=plt.rcParams['text.color'], lw=0.8)
        ax1.set(yscale='log', xlabel='Time [kyr]', ylabel='$|r|/B$')
        ax1.legend(fontsize=7, loc='lower right')
        ax0.text(0.02, 0.88, '(a)', transform=ax0.transAxes)
        ax1.text(0.02, 0.88, '(b)', transform=ax1.transAxes)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'cmb_eddy_flux_nearest_resume')
