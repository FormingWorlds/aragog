"""Validation of the single-phase thermal expansivity on the SPIDER P-S test tables, which carry
``thermal_exp`` files: the NumPy evaluator and EOS methods take alpha from the identity
alpha = rho c_p (dT/dP)_S / T, the same source as the JAX path, at solid and melt states on the
12 node pressures (0 to 146 GPa) of the phase-boundary validation mesh."""

from __future__ import annotations

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
        identity = 0.5 * (a + np.sqrt(a * a + 1e-16))
        np.testing.assert_allclose(alpha_np, identity, rtol=1e-12, atol=0)


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
