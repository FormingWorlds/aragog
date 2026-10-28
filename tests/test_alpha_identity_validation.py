"""Validation of the single-phase thermal expansivity on the SPIDER P-S test tables, which carry
``thermal_exp`` files: the NumPy evaluator and EOS methods take alpha from the identity
alpha = rho c_p (dT/dP)_S / T, the same source as the JAX path, at solid and melt states on the
12 node pressures (0 to 146 GPa) of the phase-boundary validation mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from tests.conftest import EOS_DIR, entropy_eos_copy, entropy_eos_jax, needs_eos
from tests.test_matprop_smooth_width_validation import _solver

jnp = pytest.importorskip('jax.numpy')

pytestmark = [pytest.mark.smoke, needs_eos, pytest.mark.timeout(120)]

_DS = (50.0, 200.0, 500.0)  # entropy offset from the boundary [J/kg/K]
_WIDTHS = (0.01, 1e-6)


@pytest.fixture(scope='module')
def states():
    """(P, S, side) of every test state: solid at S_sol - dS, melt at S_liq + dS."""
    P = np.asarray(_solver()._P_basic_flat, dtype=float)
    eos = entropy_eos_copy(EOS_DIR)
    S_sol, S_liq = np.ravel(eos.solidus_entropy(P)), np.ravel(eos.liquidus_entropy(P))
    rows = [(P, S_sol - d, 'solid') for d in _DS] + [(P, S_liq + d, 'melt') for d in _DS]
    P_all = np.concatenate([r[0] for r in rows])
    S_all = np.concatenate([r[1] for r in rows])
    side = np.concatenate([np.full(P.size, r[2]) for r in rows])
    for ph in ('solid', 'melt'):
        tbl = eos._tables[f'temperature_{ph}']
        assert np.all((S_all[side == ph] > tbl['S'][0]) & (S_all[side == ph] < tbl['S'][-1]))
    return P_all, S_all, side


def _smth(eos, P, S, width):
    """The two-phase weight of the evaluator blend at (P, S) (SPIDER get_smoothing)."""
    from aragog.utilities import tanh_weight

    S_sol, S_liq = eos.solidus_entropy(P), eos.liquidus_entropy(P)
    gphi = (S - S_sol) / np.maximum(S_liq - S_sol, 1e-10)
    return np.where(
        gphi > 0.5, 1.0 - tanh_weight(gphi, 1.0, width), tanh_weight(gphi, 0.0, width)
    )


def _numpy_evaluator(P, S, width):
    ev = EntropyPhaseEvaluator(entropy_eos_copy(EOS_DIR), 10.0, matprop_smooth_width=width)
    ev.set_pressure(P)
    ev.set_entropy(S)
    ev.update()
    return ev


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('width', _WIDTHS)
def test_numpy_and_jax_single_phase_alpha_agree_on_tables_with_thermal_exp(states, width):
    """On tables that ship ``thermal_exp`` files, the NumPy evaluator alpha equals the JAX
    ``evaluate_phase`` alpha to 1e-12 relative at every solid and melt state, at the default
    width 0.01 and at 1e-6. At width 1e-6 the two-phase weight is below 1e-15, so the alpha is
    the single-phase identity rho c_p dT/dP_S / T of the evaluator's own fields, clamped."""
    from aragog.jax.phase import PhaseParams, evaluate_phase

    P, S, _ = states
    eos = entropy_eos_copy(EOS_DIR)
    assert eos._has_alpha_tables, 'the test tables must carry thermal_exp files'
    smth = _smth(eos, P, S, width)
    print(f'width {width}: smth min {smth.min():.3e} max {smth.max():.3e}')

    ev = _numpy_evaluator(P, S, width)
    alpha_np = np.ravel(ev.thermal_expansivity())
    props = evaluate_phase(
        entropy_eos_jax(EOS_DIR),
        PhaseParams(matprop_smooth_width=width),
        jnp.asarray(P),
        jnp.asarray(S),
    )
    np.testing.assert_allclose(
        alpha_np, np.asarray(props.thermal_expansivity), rtol=1e-12, atol=0
    )

    if width == 1e-6:
        assert smth.max() < 1e-15
        T, rho = np.ravel(ev.temperature()), np.ravel(ev.density())
        a = np.ravel(ev.dTdPs()) * rho * np.ravel(ev.heat_capacity()) / np.maximum(T, 1.0)
        np.testing.assert_allclose(alpha_np, _clamp(a), rtol=1e-12, atol=0)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('method', ['thermal_expansivity', 'material_expansivity'])
def test_numpy_and_jax_eos_expansivity_methods_agree_on_tables_with_thermal_exp(states, method):
    """``EntropyEOS.thermal_expansivity`` and ``material_expansivity`` equal the JAX methods to
    1e-12 relative at every solid and melt state on tables that ship ``thermal_exp`` files."""
    P, S, _ = states
    got = np.ravel(getattr(entropy_eos_copy(EOS_DIR), method)(P, S))
    want = np.asarray(getattr(entropy_eos_jax(EOS_DIR), method)(jnp.asarray(P), jnp.asarray(S)))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


_MUSH = Path(__file__).parent / 'validation' / 'data' / 'alpha_mush_2d1af473.csv'


def _clamp(a):
    return 0.5 * (a + np.sqrt(a * a + 1e-16))


def _single_phase_alphas(eos, P, S_b, side):
    """Table alpha and identity alpha of one phase at (P, S_b), clipped as the evaluator does."""

    def table(name):
        tbl = eos._tables[f'{name}_{side}']
        P_c = np.clip(P, tbl['P'][0], tbl['P'][-1])
        S_c = np.clip(S_b, tbl['S'][0], tbl['S'][-1])
        return tbl['interp'](np.column_stack([P_c, S_c]))

    a = table('dTdPs') * table('density') * table('heat_capacity')
    return table('thermal_exp'), a / np.maximum(table('temperature'), 1.0)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_mush_alpha_changes_only_through_the_single_phase_weight():
    """At melt fractions 0.1, 0.5 and 0.9 under ``cp_blend = 'latent'``, the two-phase alpha
    (width 0) equals the value written by tools/verification/generate_alpha_mush_reference.py
    at 2d1af473 bit for bit. At width 0.01 the alpha changes by at most (1 - smth) times the
    gap between the table and the identity single-phase alpha, plus 1e-15 relative."""
    data = np.loadtxt(_MUSH, delimiter=',')
    P, S, phi, base0, base1 = data.T
    eos = entropy_eos_copy(EOS_DIR)
    np.testing.assert_array_equal(
        np.ravel(_numpy_evaluator(P, S, 0.0).thermal_expansivity()), base0
    )

    tip1 = np.ravel(_numpy_evaluator(P, S, 0.01).thermal_expansivity())
    melt = phi > 0.5
    S_b = np.where(melt, eos.liquidus_entropy(P), eos.solidus_entropy(P))
    gap = np.zeros_like(P)
    for side, sel in (('melt', melt), ('solid', ~melt)):
        tab, ident = _single_phase_alphas(eos, P[sel], S_b[sel], side)
        gap[sel] = np.abs(tab - ident)
    bound = (1.0 - _smth(eos, P, S, 0.01)) * gap + 1e-15 * np.abs(base1)
    assert np.all(np.abs(tip1 - base1) <= bound)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('boundary', ['solidus', 'liquidus'])
def test_the_linear_blend_meets_the_single_phase_alpha_at_each_boundary(states, boundary):
    """Under ``cp_blend = 'linear'`` the two-phase alpha at melt fraction 0 or 1 is the
    identity alpha of that phase on the boundary, so alpha has no step there (1e-12)."""
    P = np.unique(states[0])
    eos = entropy_eos_copy(EOS_DIR)
    side = 'solid' if boundary == 'solidus' else 'melt'
    S_b = np.ravel(getattr(eos, f'{boundary}_entropy')(P))
    ev = EntropyPhaseEvaluator(eos, 10.0, cp_blend='linear', matprop_smooth_width=0.0)
    ev.set_pressure(P)
    ev.set_entropy(S_b)
    ev.update()
    _, ident = _single_phase_alphas(eos, P, S_b, side)
    np.testing.assert_allclose(np.ravel(ev.thermal_expansivity()), _clamp(ident), rtol=1e-12)


@pytest.mark.reference_pinned
def test_the_table_alpha_leaves_the_numpy_path_with_its_gap_to_the_identity(
    states, validation_plot
):
    """At the solid and melt test states the thermal_exp table alpha differs from the identity
    alpha by 1e-7 to 1e-3 relative: the gap that NumPy runs integrated before both paths took
    the identity. The plot shows the gap per state against pressure, with the 1e-12 tolerance
    of the NumPy and JAX agreement above."""
    P, S, side = states
    eos = entropy_eos_copy(EOS_DIR)
    gap = np.empty_like(P)
    for ph in ('solid', 'melt'):
        sel = side == ph
        tab, ident = _single_phase_alphas(eos, P[sel], S[sel], ph)
        gap[sel] = np.abs(tab - ident) / np.abs(ident)
    assert gap.max() < 1e-3 and np.median(gap) > 1e-7

    def figure():
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        n = P.size // (2 * len(_DS))
        for i, d in enumerate(_DS):  # colour: entropy offset; line style: phase
            for j, ls in enumerate(('-', '--')):
                s = slice((j * len(_DS) + i) * n, (j * len(_DS) + i + 1) * n)
                ax.semilogy(
                    P[s] / 1e9, gap[s], ls=ls, color=f'C{i + 1}', label=f'{d:g}' * (j == 0)
                )
        ink = plt.rcParams['text.color']
        ax.plot([], [], ls='-', color=ink, label='solid')
        ax.plot([], [], ls='--', color=ink, label='melt')
        ax.axhline(1e-12, color=ink, ls=':', label='tolerance')
        ax.set(xlabel='Pressure [GPa]', ylabel='Relative alpha gap')
        ax.yaxis.set_major_locator(LogLocator(numticks=4))
        ax.legend(
            fontsize=7, ncol=2, frameon=False, title='S offset [J/kg/K]', title_fontsize=7
        )
        fig.tight_layout()
        return fig

    validation_plot(figure, 'alpha_identity_gap')
