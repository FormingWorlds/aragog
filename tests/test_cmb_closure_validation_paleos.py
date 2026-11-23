"""Validation of the CMB closure on the PALEOS:MgSiO3 states of two coupled PROTEUS runs.

The two a115 runs (``tests/validation/data/cmb_closure_paleos_a115_provenance.json``) share the
configuration and the tables. One uses ``c_p/T`` of the blend at an aragog commit without #121 (the
lid melt and the CMB law by phase), the other ``(dS/dT)_P`` (the slope closure) on this branch, so
each run is compared with the closed form of its own closure. For each row the
core ledger misses by ``r/|dE| = sign(q) (1/k - 1)`` with ``k = c_p,used (dT/dS)_tab / T``
(Bower et al. 2018, Eqs. 19-21, https://doi.org/10.1016/j.pepi.2017.11.004): the case
evaluates ``k`` with the code's blend on the run's own tables at the stored CMB-node states and
compares it with the miss the runs measured.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import assert_figure_legend_below, write_ps_table

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
pytest.importorskip('equinox')

pytestmark = [pytest.mark.reference_pinned, pytest.mark.unit]

DATA = Path(__file__).parent / 'validation' / 'data' / 'cmb_closure_paleos_a115.npz'
WIDTH = 0.01  # matprop_smooth_width of the runs
MIN_POWER = 1.0e16  # [W] CMB power below which the solver floor dominates the row
MAX_DK = 0.02  # rows whose closure factor changes less than this inside the row
TOL = 2.0e-3


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    """The run's tables as an EOS directory (placeholders for the properties the slope does
    not read) and the stored rows."""
    from aragog.jax.eos import EntropyEOS_JAX

    d = dict(np.load(DATA))
    tmp = tmp_path_factory.mktemp('paleos_a115')
    for ph in ('solid', 'melt'):
        for name in ('temperature', 'heat_capacity'):
            write_ps_table(
                tmp / f'{name}_{ph}.dat',
                d[f'P_{name}_{ph}'],
                d[f'S_{name}_{ph}'],
                d[f'{name}_{ph}'],
            )
        P, S = d[f'P_temperature_{ph}'], d[f'S_temperature_{ph}']
        for name, value in (
            ('density', 4000.0),
            ('thermal_exp', 1.0e-5),
            ('adiabat_temp_grad', 1.0e-8),
        ):
            write_ps_table(tmp / f'{name}_{ph}.dat', P, S, np.full((P.size, S.size), value))
    for name in ('solidus', 'liquidus'):
        rows = '\n'.join(f'{p:.17g} {s:.17g}' for p, s in zip(d[f'P_{name}'], d[f'S_{name}']))
        (tmp / f'{name}_P-S.dat').write_text(
            f'# 5 {d[f"P_{name}"].size}\n#\n#\n#\n# 1 1\n{rows}\n'
        )
    return EntropyEOS_JAX(tmp), d


def _rows(case, run):
    """Measured and predicted miss of every kept row of ``run``, and the rows the test uses."""
    eos, d = case
    P, kept, q, dE, dt = (d[f'{run}_{k}'] for k in ('P_cmb', 'kept', 'q', 'dE', 'dt_s'))
    phase = jax.jit(lambda p, s: eos.compute_phase_state(p, s, 4.0, 2.0, WIDTH))
    ends = {}
    for end in ('start', 'end'):
        S = jnp.asarray(np.where(kept, d[f'{run}_S_{end}'], 8000.0))
        st = phase(jnp.asarray(P), S)
        T, cp, slope, g = (
            np.asarray(x)
            for x in (st.temperature, st.heat_capacity, st.temperature_slope, st.gphi)
        )
        s_tab = (
            np.asarray(phase(jnp.asarray(P), S + 1.0e-3).temperature)
            - np.asarray(phase(jnp.asarray(P), S - 1.0e-3).temperature)
        ) / 2.0e-3
        # instrument: the blend reproduces the run's CMB-node temperature at every kept row
        np.testing.assert_allclose(
            T[kept], d[f'{run}_T_node_{end}'][kept], rtol=0.0, atol=1.0e-6
        )
        k = (cp / T if run == 'cp_over_t' else 1.0 / slope) * s_tab
        ends[end] = (k, (g > 0.0) & (g < 1.0) & (s_tab > 0.0))
    (k0, ok0), (k1, ok1) = ends['start'], ends['end']
    pred = np.sign(q) * (2.0 / (k0 + k1) - 1.0)
    moved = (dt > 0.0) & (dE != 0.0)  # the first row has no elapsed time
    meas = np.divide(dE + q, np.abs(dE), out=np.full_like(q, np.nan), where=moved)
    power = np.divide(np.abs(q), dt, out=np.zeros_like(q), where=moved)
    strong = kept & moved & (power > MIN_POWER)
    used = strong & ok0 & ok1 & (np.abs(k1 - k0) < MAX_DK)
    return d[f'{run}_t_end'], meas, pred, strong, used


@pytest.mark.physics_invariant
def test_the_closed_form_gives_the_ledger_miss_of_both_runs(case, validation_plot):
    """Rows in the mush with a steady closure factor: the ``c_p/T`` run misses by the closed form
    (up to about 10 %) and the slope run by 0, both within 2e-3."""
    cpt, slope = _rows(case, 'cp_over_t'), _rows(case, 'slope')
    for t, meas, pred, _, used in (cpt, slope):
        assert used.sum() >= 10
        assert np.max(np.abs(meas - pred)[used]) < TOL
    assert np.max(np.abs(cpt[1][cpt[4]])) > 0.05
    assert np.max(np.abs(slope[1][slope[4]])) < TOL
    assert_figure_legend_below(_figure(cpt, slope))
    validation_plot(lambda: _figure(cpt, slope), 'core_cmb_closure_paleos')


def _figure(cpt, slope):
    """Ledger miss per row against time: the runs (points) and the closed form (line)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for (t, meas, pred, strong, used), colour in ((cpt, 'C1'), (slope, 'C0')):
        ax.plot(t[strong], pred[strong], color=colour, lw=2, alpha=0.4)
        ax.plot(t[used], meas[used], 'o', ms=4, color=colour)
        rest = strong & ~used
        ax.plot(t[rest], meas[rest], 'o', ms=4, color=colour, mfc='none')
    ax.set_xscale('log')
    ax.set_xticks([10.0, 1.0e3, 1.0e5], ['10', '1000', '100000'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yscale('symlog', linthresh=1.0e-4)
    ax.set_yticks([-0.1, -0.001, 0.0, 0.001, 0.1], ['-0.1', '-0.001', '0', '0.001', '0.1'])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set(xlabel='time [yr]', ylabel='miss r/|dE|')
    text = plt.rcParams['axes.labelcolor']
    handles = [
        Line2D([], [], color='C1', marker='o', label='c_p/T'),
        Line2D([], [], color='C0', marker='o', label='(dS/dT)_P'),
        Line2D([], [], color=text, lw=2, alpha=0.4, label='closed form'),
        Line2D([], [], color=text, marker='o', ls='none', mfc='none', label='not compared'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.2, 1.0, 1.0))
    return fig
