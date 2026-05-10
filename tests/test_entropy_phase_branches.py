"""Edge-case unit tests for ``EntropyPhaseEvaluator``.

The bulk of ``entropy_phase.py`` is exercised through downstream
solver runs, but several public methods and edge-case branches sit
in the coverage shadow:

* ``_update_const`` scalar-input flatten branch (lines 166, 176).
* ``_update_eos`` alpha-from-dTdPs fallback (lines 254-258, 297) —
  fires when the EOS directory has no thermal_exp tables.
* ``_update_eos`` scalar-input flatten (lines 313, 322).
* ``_update_eos`` NaN detection error path (lines 353-363) — fires
  when entropy leaves the EOS table domain.
* ``dTdrs`` accessor (line 387).
* ``mass_flux_velocity`` const_properties zero-branch (line 432).
* ``delta_specific_volume`` (lines 495-499) — both const_properties
  and EOS-backed paths.

These tests build minimal evaluators and call the methods directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)


def _make_const_evaluator():
    """Build a const_properties evaluator with sane defaults."""
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    ev = EntropyPhaseEvaluator(
        entropy_eos=None,
        gravitational_acceleration=10.0,
        const_properties=True,
        const_rho=4000.0,
        const_Cp=1000.0,
        const_alpha=3.0e-5,
        const_cond=4.0,
        const_log10visc=2.0,
        const_T_ref=3000.0,
        const_S_ref=3000.0,
    )
    return ev


def test_update_const_scalar_input_flattens_arrays():
    """Scalar (P, S) input through the const_properties path must
    produce 1-D ndarrays via the post-update flatten branch (lines
    166, 176).

    Discriminator: a regression that lost the scalar-flatten branch
    would leave ``_temperature`` as a 0-D array, which then breaks
    callers that index it as ``[i]``.
    """
    ev = _make_const_evaluator()
    ev.set_pressure(50.0e9)  # scalar
    ev.entropy = 3050.0  # scalar
    ev.update()
    assert ev.temperature().ndim == 1, (
        f'_temperature ndim = {ev.temperature().ndim}, expected 1 after scalar flatten'
    )
    assert ev.density().ndim == 1
    assert ev.heat_capacity().ndim == 1


def test_const_properties_relative_velocity_returns_zero():
    """``relative_velocity`` in const_properties mode is identically
    zero (line 432). Discriminator: a regression that fell through
    to the EOS lookup would crash on ``self._eos._lookup_at_phase_boundary``
    since ``_eos = None`` in const_properties.
    """
    ev = _make_const_evaluator()
    ev.set_pressure(np.array([50.0e9]))
    ev.entropy = np.array([3050.0])
    ev.update()
    out = ev.relative_velocity()
    np.testing.assert_array_equal(out, np.zeros_like(ev.density()))


def test_const_properties_delta_specific_volume_returns_zero():
    """``delta_specific_volume`` in const_properties mode (line 495-496):
    no phase transition means no specific-volume difference.
    """
    ev = _make_const_evaluator()
    ev.set_pressure(np.array([50.0e9]))
    ev.entropy = np.array([3050.0])
    ev.update()
    out = ev.delta_specific_volume()
    np.testing.assert_array_equal(out, np.zeros_like(ev.density()))


@needs_eos
def test_eos_backed_delta_specific_volume_returns_finite_difference():
    """``delta_specific_volume`` with a real EOS (lines 497-499) must
    return finite, signed specific-volume difference between melt
    and solid.

    Discriminator: a regression that returned 1/rho_s - 1/rho_l
    (sign-flipped) would surface as the sign of the result. SPIDER
    convention has melt less dense than solid, so v_l > v_s, hence
    (1/rho_l - 1/rho_s) > 0.
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    eos = EntropyEOS(EOS_DIR)
    ev = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    P = np.array([5.0e10])
    ev.set_pressure(P)
    ev.entropy = np.array([3300.0])
    ev.update()
    dv = ev.delta_specific_volume()
    assert dv.shape == (1,)
    assert np.all(np.isfinite(dv))
    assert float(dv.item()) > 0.0, (
        f'delta_specific_volume = {float(dv.item()):.3e} <= 0; '
        'expected (1/rho_liq - 1/rho_sol) > 0 (melt less dense)'
    )


@needs_eos
def test_eos_backed_dTdrs_returns_finite_negative_in_solid():
    """``dTdrs = -g * alpha * T / Cp`` (line 387) must be negative
    in the solid mantle (T decreases outward along an adiabat).

    Discriminator: a regression that lost the negation would
    surface as a positive dT/dr along the adiabat, breaking the
    superadiabatic-temperature flux sign in conduction.
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    eos = EntropyEOS(EOS_DIR)
    ev = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    P = np.array([5.0e10])
    S_sol = float(eos.solidus_entropy(P).item())
    ev.set_pressure(P)
    ev.entropy = np.array([S_sol - 500.0])  # safely in pure solid
    ev.update()
    dTdr = ev.dTdrs()
    assert np.all(np.isfinite(dTdr))
    assert float(dTdr.item()) < 0.0, (
        f'dTdrs in pure solid = {float(dTdr.item()):.3e} >= 0; expected < 0'
    )


@needs_eos
def test_eos_update_with_scalar_input_flattens_arrays():
    """Scalar (P, S) input through ``_update_eos`` exercises the
    flatten branch at lines 313-322. ``temperature()`` must still
    be 1-D.
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    eos = EntropyEOS(EOS_DIR)
    ev = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    ev.set_pressure(5.0e10)  # scalar
    ev.entropy = 3300.0  # scalar
    ev.update()
    assert ev.temperature().ndim == 1


@needs_eos
def test_eos_update_alpha_fallback_when_thermal_exp_tables_missing(monkeypatch):
    """Lines 297: when the EOS has no thermal_exp tables,
    ``_update_eos`` must derive alpha from
    ``alpha = rho * Cp * |dTdPs| / T`` for the single-phase branch.
    Discriminator: removing the thermal_exp_solid / thermal_exp_melt
    keys via monkeypatch forces the fallback; the resulting alpha
    must be finite and positive.
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    eos = EntropyEOS(EOS_DIR)
    # Drop the thermal_exp tables so the fallback fires.
    tables_no_alpha = {k: v for k, v in eos._tables.items() if not k.startswith('thermal_exp_')}
    monkeypatch.setattr(eos, '_tables', tables_no_alpha)

    ev = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)
    P = np.array([5.0e10])
    S_sol = float(eos.solidus_entropy(P).item())
    S_liq = float(eos.liquidus_entropy(P).item())
    # One node in mushy band, one in pure solid: covers both branches.
    P_arr = np.array([5.0e10, 5.0e10])
    S_arr = np.array([0.5 * (S_sol + S_liq), S_sol - 500.0])
    ev.set_pressure(P_arr)
    ev.entropy = S_arr
    ev.update()
    alpha = ev.thermal_expansivity()
    assert np.all(np.isfinite(alpha))
    assert np.all(alpha >= 0.0), f'fallback alpha returned negative values: {alpha}'


@needs_eos
def test_eos_update_alpha_fallback_under_linear_cp_blend(monkeypatch):
    """Lines 254-258: with ``cp_blend='linear'`` AND no thermal_exp
    tables, the mixed-band alpha must come from the dTdPs * rho * Cp / T
    derivation.

    Discriminator: cp_blend='linear' switches the Cp branch, which
    is also where the linear-blend alpha fallback lives. A
    regression that lost either the cp_blend dispatch or the
    fallback would surface as a NaN or as the result silently
    coming from the cp_blend='latent' path.
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    eos = EntropyEOS(EOS_DIR)
    # Drop the thermal_exp tables to force the dTdPs-based fallback.
    tables_no_alpha = {k: v for k, v in eos._tables.items() if not k.startswith('thermal_exp_')}
    monkeypatch.setattr(eos, '_tables', tables_no_alpha)

    ev = EntropyPhaseEvaluator(
        entropy_eos=eos,
        gravitational_acceleration=10.0,
        cp_blend='linear',
    )
    P = np.array([5.0e10])
    S_sol = float(eos.solidus_entropy(P).item())
    S_liq = float(eos.liquidus_entropy(P).item())
    # Strictly mushy node (gphi between 0 and 1).
    ev.set_pressure(P)
    ev.entropy = np.array([0.4 * S_sol + 0.6 * S_liq])
    ev.update()
    alpha = ev.thermal_expansivity()
    assert np.all(np.isfinite(alpha))
    assert float(alpha.item()) > 0.0


@needs_eos
def test_eos_update_raises_runtime_error_on_out_of_table_entropy(monkeypatch):
    """Lines 353-363: when the entropy lookup yields NaN (state
    outside table domain), ``_update_eos`` must log an error and
    raise RuntimeError naming the offending entropy range.

    Discriminator: a regression that swallowed the NaN would let
    a downstream RHS call multiply NaN through the heat-flux
    pipeline, producing a silent integrator stall rather than a
    clear domain error at the EOS layer.
    """
    from aragog.eos.entropy import EntropyEOS
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    eos = EntropyEOS(EOS_DIR)
    ev = EntropyPhaseEvaluator(entropy_eos=eos, gravitational_acceleration=10.0)

    # Inject NaN at both the boundary and phase-weighted temperature
    # lookups so the resulting ``self._temperature`` is NaN even
    # after the smth-weighted blend of T_mixed and T_single.
    _orig_at_boundary = eos._lookup_at_phase_boundary
    _orig_lpw = eos._lookup_phase_weighted

    def _nan_at_boundary(prop, P, phase):
        if prop == 'temperature':
            return np.full_like(np.asarray(P, dtype=float), np.nan)
        return _orig_at_boundary(prop, P, phase)

    def _nan_lpw(prop, P, S):
        if prop == 'temperature':
            return np.full_like(np.asarray(P, dtype=float), np.nan)
        return _orig_lpw(prop, P, S)

    monkeypatch.setattr(eos, '_lookup_at_phase_boundary', _nan_at_boundary)
    monkeypatch.setattr(eos, '_lookup_phase_weighted', _nan_lpw)

    ev.set_pressure(np.array([5.0e10]))
    ev.entropy = np.array([3300.0])
    with pytest.raises(RuntimeError, match='Entropy out of EOS table domain'):
        ev.update()
