"""``(dT/dS)_P`` of the phase blend, the slope the CMB closure divides by.

Both evaluators return the derivative of their blended temperature in S at fixed P, with the
pure-phase slope ``T/c_p`` in the single-phase term. On tables where that equals the bilinear
table slope (``T = a(P) + k S``, ``c_p = T/k``) the slope is the exact derivative everywhere;
inside the mush it is the exact derivative on any table, since the single-phase term reads the
tables at the phase boundaries there.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from tests.conftest import EOS_DIR, entropy_eos_copy, entropy_eos_jax, needs_eos

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
pytest.importorskip('equinox')

K_SLOPE = 0.2  # [K kg K/J] dT/dS of the consistent tables
P_GRID = np.linspace(50.0e9, 150.0e9, 11)
S_GRID = np.arange(1000.0, 12001.0, 100.0)
WIDTH = st.one_of(st.just(0.0), st.floats(1.0e-3, 0.05))


def _write_table(path, values):
    """A P-S table in the SPIDER layout (P varies fastest), unit scales."""
    rows = [
        f'{p:.17g} {s:.17g} {values[i, j]:.17g}'
        for j, s in enumerate(S_GRID)
        for i, p in enumerate(P_GRID)
    ]
    head = f'# 5 {P_GRID.size} {S_GRID.size}\n#\n#\n#\n# 1 1 1\n'
    path.write_text(head + '\n'.join(rows) + '\n')


@pytest.fixture(scope='module')
def consistent_dir(tmp_path_factory):
    """Tables with ``T = 1500 K + 1e-8 P + k S`` and ``c_p = T/k`` in both phases."""
    d = tmp_path_factory.mktemp('consistent_eos')
    T = 1500.0 + 1.0e-8 * P_GRID[:, None] + K_SLOPE * S_GRID[None, :]
    tables = {
        'temperature': T,
        'heat_capacity': T / K_SLOPE,
        'density': np.full_like(T, 4000.0),
        'thermal_exp': np.full_like(T, 1.0e-5),
        'adiabat_temp_grad': np.full_like(T, 1.0e-8),
    }
    for name, values in tables.items():
        for phase in ('solid', 'melt'):
            _write_table(d / f'{name}_{phase}.dat', values)
    for name, s0 in (('solidus', 4000.0), ('liquidus', 7000.0)):
        rows = '\n'.join(f'{p:.17g} {s0 + 1.0e-8 * p:.17g}' for p in P_GRID)
        (d / f'{name}_P-S.dat').write_text(f'# 5 {P_GRID.size}\n#\n#\n#\n# 1 1\n{rows}\n')
    return d


def _jax_slope_and_jvp(eos_jax, P, S, width):
    """``temperature_slope`` of ``evaluate_phase`` and the forward derivative of its temperature."""
    from aragog.jax.phase import PhaseParams, evaluate_phase

    params = PhaseParams(matprop_smooth_width=width)
    P, S = jnp.atleast_1d(P), jnp.atleast_1d(S)

    def temperature(s):
        return evaluate_phase(eos_jax, params, P, s).temperature

    T, dT = jax.jvp(temperature, (S,), (jnp.ones_like(S),))
    slope = evaluate_phase(eos_jax, params, P, S).temperature_slope
    return float(slope[0]), float(dT[0])


def _g_to_S(eos, P, g):
    S_sol = float(np.ravel(eos.solidus_entropy(np.atleast_1d(P)))[0])
    S_liq = float(np.ravel(eos.liquidus_entropy(np.atleast_1d(P)))[0])
    return S_sol + g * (S_liq - S_sol)


@settings(max_examples=200, deadline=None)
@given(P=st.floats(60.0e9, 140.0e9), g=st.floats(-0.05, 1.05), width=WIDTH)
def test_the_slope_is_the_derivative_of_the_blend_on_consistent_tables(
    consistent_dir, P, g, width
):
    """On tables whose ``T/c_p`` equals the table slope, the slope is ``dT/dS`` of the blend."""
    from aragog.jax.eos import EntropyEOS_JAX

    assume(min(abs(g), abs(g - 0.5), abs(g - 1.0)) > 1.0e-9)
    eos_jax = EntropyEOS_JAX(consistent_dir)
    slope, dT = _jax_slope_and_jvp(eos_jax, P, _g_to_S(eos_jax, P, g), width)
    assert np.isfinite(slope) and slope > 0.0
    assert slope == pytest.approx(dT, rel=1.0e-9)


@needs_eos
@settings(max_examples=200, deadline=None)
@given(P=st.floats(20.0e9, 135.0e9), g=st.floats(1.0e-6, 1.0 - 1.0e-6), width=WIDTH)
def test_the_slope_is_the_derivative_of_the_blend_inside_the_mush(P, g, width):
    """In the mush the slope is ``dT/dS`` of the blend on the production tables."""
    assume(abs(g - 0.5) > 1.0e-9)
    eos_jax = entropy_eos_jax(EOS_DIR)
    slope, dT = _jax_slope_and_jvp(eos_jax, P, _g_to_S(eos_jax, P, g), width)
    assert np.isfinite(slope) and slope > 0.0
    assert slope == pytest.approx(dT, rel=1.0e-9)


@needs_eos
@pytest.mark.parametrize('width', [0.0, 0.005, 0.01, 0.02])
def test_the_numpy_and_jax_slopes_agree(width):
    """The NumPy evaluator and the JAX blend return the same slope from the solid to the melt."""
    eos = entropy_eos_copy(EOS_DIR)
    P = np.repeat(np.linspace(20.0e9, 135.0e9, 5), 45)
    g = np.tile(np.linspace(-0.05, 1.05, 45), 5)
    S = np.array([_g_to_S(eos, p, x) for p, x in zip(P, g)])
    ev = EntropyPhaseEvaluator(eos, 10.0, matprop_smooth_width=width)
    ev.set_pressure(P)
    ev.set_entropy(S)
    ev.update()
    from aragog.jax.phase import PhaseParams, evaluate_phase

    props = evaluate_phase(
        entropy_eos_jax(EOS_DIR),
        PhaseParams(matprop_smooth_width=width),
        jnp.asarray(P),
        jnp.asarray(S),
    )
    np.testing.assert_allclose(
        np.asarray(props.temperature_slope), ev.temperature_slope(), rtol=1.0e-10
    )
    assert np.all(ev.temperature_slope() > 0.0)


def test_the_slope_of_the_constant_property_mode_is_t_over_cp():
    """``T = T_ref exp((S - S_ref)/c_p)`` has the exact slope ``T/c_p``."""
    ev = EntropyPhaseEvaluator(
        None,
        10.0,
        const_properties=True,
        const_rho=4000.0,
        const_Cp=1200.0,
        const_alpha=1.0e-5,
        const_cond=4.0,
        const_log10visc=2.0,
        const_T_ref=3000.0,
        const_S_ref=3000.0,
    )
    ev.set_pressure(np.full(3, 1.0e10))
    ev.set_entropy(np.array([2500.0, 3000.0, 3500.0]))
    ev.update()
    np.testing.assert_allclose(ev.temperature_slope(), ev.temperature() / 1200.0, rtol=1.0e-15)


@needs_eos
def test_a_non_finite_entropy_gives_a_nan_slope():
    """The JAX blend propagates a non-finite entropy into the slope as NaN."""
    from aragog.jax.phase import PhaseParams, evaluate_phase

    props = evaluate_phase(
        entropy_eos_jax(EOS_DIR),
        PhaseParams(matprop_smooth_width=0.01),
        jnp.full(2, 1.0e11),
        jnp.array([np.nan, np.inf]),
    )
    assert np.all(np.isnan(np.asarray(props.temperature_slope)))


def test_the_consistent_tables_load_with_both_evaluators(consistent_dir):
    """Instrument check: the synthetic tables give ``T = a(P) + k S`` through the NumPy loader."""
    eos = EntropyEOS(consistent_dir)
    P, S = np.array([7.5e10]), np.array([2000.0])
    assert float(np.ravel(eos.temperature(P, S))[0]) == pytest.approx(1500.0 + 750.0 + 400.0)


@pytest.mark.parametrize('g', [0.0, 1.0])
def test_the_sharp_blend_has_the_mush_slope_at_the_phase_boundaries(consistent_dir, g):
    """With no smoothing the blend is the mush line on the closed interval, so its slope at
    the solidus and the liquidus is ``dT_fus/dS_fus`` (``k`` here), not 0."""
    from aragog.jax.eos import EntropyEOS_JAX

    eos_jax = EntropyEOS_JAX(consistent_dir)
    slope, _ = _jax_slope_and_jvp(eos_jax, 1.0e11, _g_to_S(eos_jax, 1.0e11, g), 0.0)
    assert slope == pytest.approx(K_SLOPE, rel=1.0e-12)
