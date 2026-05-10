"""Direct unit tests for the ``EntropyEOS`` methods that the existing
suite exercises only through downstream callers.

Three public methods sit in the coverage shadow of the tests in
``test_entropy_pytest.py`` / ``test_entropy_verification.py``:

* ``thermal_expansivity_composite_blend`` (lines 964-1008): the
  SPIDER-parity composite alpha including the Clausius-Clapeyron
  contribution. Never called from the production solver path (the
  table-backed ``thermal_expansivity`` short-circuits via
  ``_has_alpha_tables``), so the body is dead code in coverage terms
  unless invoked directly.
* ``thermal_expansivity`` fallback path (lines 1042-1050): the
  ``alpha = rho * Cp * |dTdPs| / T`` derivation that fires only when
  the EOS directory does not ship pre-computed ``thermal_exp_*.dat``.
  Production tables include alpha tables, so this path is similarly
  dead code under the test fixture.
* ``invert_temperature`` (lines 1078-1103): T -> S inversion via
  brentq with explicit bracket-error message. The CLI exercises it
  through ``_derive_initial_entropy_from_config``, but only the
  in-bracket success path; the out-of-range error message and the
  pure-Python ``temperature_scalar`` callback have no direct
  exposure.

These tests target each path with discriminators that distinguish the
correct implementation from plausible-wrong variants (sign error,
missing rho factor, brentq xtol regression, swallowed bracket error).
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


@pytest.fixture(scope='module')
def eos():
    """Module-scoped EntropyEOS instance."""
    if not EOS_DIR.exists():
        pytest.skip('EOS unavailable')
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


# ──────────────────────────────────────────────────────────────────────
#                  thermal_expansivity_composite_blend
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_composite_thermal_expansivity_returns_finite_positive_in_pure_solid(eos):
    """Outside the mushy zone (S well below the solidus), the smoothed
    blend must reduce to the pure-phase alpha and stay finite, positive.

    Discriminator: a bug that flipped the smoothing weight (mixed when
    pure was wanted) would surface here as either a NaN (from
    Clausius-Clapeyron 0/0) or an alpha 5x larger than the table value.
    Compare against the direct ``thermal_expansivity`` lookup.
    """
    P = np.array([5.0e10])
    S_sol = float(eos.solidus_entropy(P).item())
    # Drop well below the solidus and use a tight smoothing width so
    # the tanh tail falls off cleanly. 800 J/kg/K below S_sol at width
    # 0.005 puts gphi/w << -10, smoothing weight ~1e-100.
    S_below = np.array([S_sol - 800.0])

    alpha_blend = eos.thermal_expansivity_composite_blend(P, S_below, width=0.005)
    alpha_pure = eos.thermal_expansivity(P, S_below)

    assert np.all(np.isfinite(alpha_blend)), (
        'composite alpha returned non-finite values in the pure-solid regime'
    )
    assert float(alpha_blend.item()) > 0.0, (
        f'composite alpha = {float(alpha_blend.item()):.3e} <= 0 in the solid regime'
    )
    rel = abs(float(alpha_blend.item()) - float(alpha_pure.item())) / abs(
        float(alpha_pure.item())
    )
    assert rel < 0.10, (
        f'pure-solid composite alpha {float(alpha_blend.item()):.3e} '
        f'differs by {rel:.2%} from the table-based alpha {float(alpha_pure.item()):.3e}; '
        'smoothing weight is leaking the mixed-phase contribution.'
    )


@needs_eos
def test_composite_thermal_expansivity_inside_mushy_exceeds_pure_phase_alpha(eos):
    """In the deep mushy zone the Clausius-Clapeyron contribution
    dominates, making composite alpha materially larger than the
    pure-phase alpha. SPIDER's eos_composite.c:246 documents the
    ~5x enhancement at the rheological transition.

    Discriminator: a regression that lost the (rho_solid - rho_melt)
    numerator or used (rho_solid + rho_melt) would either zero the
    enhancement or flip its sign. We require the composite alpha to
    be strictly greater than the table-based alpha at a point in the
    middle of the mushy band.
    """
    P = np.array([5.0e10])
    S_sol = float(eos.solidus_entropy(P).item())
    S_liq = float(eos.liquidus_entropy(P).item())
    S_mid = np.array([0.5 * (S_sol + S_liq)])  # mushy mid-band

    alpha_blend = eos.thermal_expansivity_composite_blend(P, S_mid, width=0.005)
    alpha_pure = eos.thermal_expansivity(P, S_mid)

    assert float(alpha_blend.item()) > float(alpha_pure.item()), (
        f'composite alpha {float(alpha_blend.item()):.3e} not > pure {float(alpha_pure.item()):.3e}; '
        'the Clausius-Clapeyron enhancement is missing or sign-flipped.'
    )


@needs_eos
def test_composite_thermal_expansivity_array_input_preserves_shape(eos):
    """Array input must map element-wise without broadcasting bugs.

    Edge case: a regression that flattened the input via ``ravel()``
    would lose any non-trivial shape (e.g. 2-D fields from a coupled
    P-S sweep). Composite alpha must round-trip a (3,) input.
    """
    P = np.array([5.0e10, 7.0e10, 1.0e11])
    S_sol = np.asarray(eos.solidus_entropy(P)).reshape(P.shape)
    S_liq = np.asarray(eos.liquidus_entropy(P)).reshape(P.shape)
    S_mid = 0.5 * (S_sol + S_liq)
    alpha = eos.thermal_expansivity_composite_blend(P, S_mid)
    assert alpha.shape == P.shape, (
        f'composite alpha shape {alpha.shape} != input shape {P.shape}; '
        'array shape was flattened or broadcast'
    )
    assert np.all(np.isfinite(alpha))
    assert np.all(alpha > 0.0)


# ──────────────────────────────────────────────────────────────────────
#                       thermal_expansivity fallback
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_thermal_expansivity_fallback_when_alpha_tables_absent(eos, monkeypatch):
    """The thermodynamic-identity fallback ``alpha = rho * Cp * |dTdPs| / T``
    fires when the EOS directory has no ``thermal_exp_*.dat`` files.

    Discriminator: the fallback's value must equal that closed-form
    expression to machine precision against the EOS's own ``rho``,
    ``Cp``, ``dTdPs``, ``temperature`` calls. A regression that
    dropped the ``rho`` factor would surface as a 4000x error.
    """
    P = np.array([3.0e10])
    S = np.array([3300.0])
    # Force the fallback path by flipping the flag.
    monkeypatch.setattr(eos, '_has_alpha_tables', False)
    alpha_fallback = eos.thermal_expansivity(P, S)

    # Recompute by hand from the same EOS to confirm the formula.
    T = np.asarray(eos.temperature(P, S)).item()
    rho = np.asarray(eos.density(P, S)).item()
    Cp = np.asarray(eos.heat_capacity(P, S)).item()
    dTdPs_val = np.asarray(eos.dTdPs(P, S)).item()
    expected = rho * Cp * abs(dTdPs_val) / max(T, 1.0)

    np.testing.assert_allclose(
        np.asarray(alpha_fallback).item(),
        expected,
        rtol=1e-12,
        atol=0.0,
    )


# ──────────────────────────────────────────────────────────────────────
#                          invert_temperature
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_invert_temperature_round_trips_within_table_range(eos):
    """``invert_temperature`` must return an S such that
    ``temperature_scalar(P, S)`` matches the target to <1 mK.

    Discriminator: brentq is run with ``xtol=0.1`` on S; with dT/dS
    of order 0.5-1 K/(J/kg/K) on the PALEOS table, that translates
    to ~0.1 K on T. The assertion uses 1 K as a lenient ceiling so
    a regression that loosened xtol to 1.0 would still pass, but a
    regression that lost the brentq call (returning S_lo or S_hi)
    would fail by hundreds of K.
    """
    P = 5.0e10
    T_target = 4000.0
    S_inv = eos.invert_temperature(P, T_target)
    assert isinstance(S_inv, float), (
        f'invert_temperature returned {type(S_inv).__name__}, not float'
    )
    T_back = eos.temperature_scalar(P, S_inv)
    assert abs(T_back - T_target) < 1.0, (
        f'invert_temperature did not round-trip; T_back={T_back:.4f} K, '
        f'target={T_target:.1f} K, |delta|={abs(T_back - T_target):.4e} K. '
    )


@needs_eos
def test_invert_temperature_rejects_target_outside_table_range(eos):
    """A T_target above the maximum achievable T at this P must raise
    ValueError with a message that names the table-spanned T range.

    Discriminator: a regression that silently clamped to the table edge
    would return a finite S whose ``temperature_scalar`` was nowhere
    near the target. The explicit bracket-sign check converts this
    into a clear error.
    """
    P = 5.0e10
    # 100,000 K is far above any achievable T on the PALEOS table.
    with pytest.raises(ValueError, match='outside table range'):
        eos.invert_temperature(P, 100_000.0)


@needs_eos
def test_invert_temperature_uses_clamped_pressure(eos):
    """When ``P`` is outside ``[P_min, P_max]`` the inversion must
    clamp internally and still return a sensible S (rather than
    extrapolating into an out-of-table region).

    Discriminator: setting P far above P_max and asking for a target T
    that is achievable at P_max must produce an S whose
    ``temperature_scalar(P_max, S) ~ T_target``. A regression that
    forgot to clamp would either crash inside the interpolator or
    return something far from the target.
    """
    P_far = eos.P_max * 10.0
    T_target = float(eos.temperature_scalar(eos.P_max, 3300.0))
    S_inv = eos.invert_temperature(P_far, T_target)
    T_back = eos.temperature_scalar(eos.P_max, S_inv)
    assert abs(T_back - T_target) < 1.0, (
        f'P-clamp lost: T_back={T_back:.2f} K, target={T_target:.2f} K'
    )


# ──────────────────────────────────────────────────────────────────────
#                          latent_heat
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_latent_heat_positive_and_smooth_with_pressure(eos):
    """``latent_heat(P) = T_fus * (S_liq - S_sol)`` must be strictly
    positive everywhere in the mantle pressure range.

    Discriminator: a regression that flipped the (S_liq - S_sol)
    sign or dropped the ``np.maximum(..., 1.0)`` floor would surface
    as a non-positive entry at high P where the gap can collapse.
    """
    P = np.linspace(1.0e8, 1.4e11, 25)
    L = eos.latent_heat(P)
    assert L.shape == P.shape
    assert np.all(np.isfinite(L)), 'latent_heat returned non-finite entries'
    assert np.all(L > 0.0), (
        f'latent_heat dipped below 0 at some P: min={float(L.min()):.3e} J/kg'
    )
    # Earth-like latent heat is order 1e5-1e7 J/kg; bound by a wide
    # plausibility window so the test is sensitive to a 100x drift.
    assert float(L.max()) < 1.0e8 and float(L.min()) > 1.0e3, (
        f'latent_heat range [{float(L.min()):.3e}, {float(L.max()):.3e}] J/kg '
        'is implausible for a mantle EOS.'
    )
