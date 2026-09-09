"""Unit tests for the entropy table-domain range check.

A non-finite or far-out-of-table-range entropy still produces a
finite property once ``_lookup_phase_weighted`` / ``density`` clamp
it to the table edge, or once ``_update_eos`` blends the clamped
branch values. ``_check_entropy_range`` (aragog.eos.entropy) is the
root-cause guard against that: it always logs a warning naming the
non-finite and out-of-range counts, and raises ``RuntimeError``
instead when ``strict_range`` is set on the EOS.

These tests target ``_check_entropy_range`` directly, its wiring into
``temperature``/``density`` (per-branch masked check against each
table's own range), and its wiring into
``EntropyPhaseEvaluator._update_eos`` (composite-domain check against
the union range before any table lookup). The discriminator in every
case is the exact warning context string and, for the NaN scenario
specifically, which mechanism raises: ``entropy_phase.py`` also has a
separate, pre-existing downstream NaN guard that fires whenever the
blended temperature comes out NaN, independent of ``strict_range``.
"""

from __future__ import annotations

import logging
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
    '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
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
    """Module-scoped EntropyEOS instance, non-strict (warn only)."""
    if not EOS_DIR.exists():
        pytest.skip('EOS unavailable')
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


@pytest.fixture(scope='module')
def eos_strict():
    """Module-scoped EntropyEOS instance with strict_range=True."""
    if not EOS_DIR.exists():
        pytest.skip('EOS unavailable')
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR, strict_range=True)


# ──────────────────────────────────────────────────────────────────────
#                       _check_entropy_range directly
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_check_entropy_range_in_domain_is_silent(eos, caplog):
    """A finite S well inside [S_min, S_max] must not warn.

    Discriminator: a regression that always fires the report callback
    (dropping the "not n_non_finite and not n_out_of_range" early
    return) would warn on every call, including this baseline.
    """
    S = np.array([0.5 * (eos.S_min + eos.S_max)])
    with caplog.at_level(logging.WARNING):
        eos._check_entropy_range(S, eos.S_min, eos.S_max, 'test-context')
    assert not any('test-context' in r.message for r in caplog.records)


@needs_eos
def test_check_entropy_range_warns_on_nan(eos, caplog):
    """A NaN entry must warn with the non-finite count and the context.

    Discriminator: a regression that only checked ``S < S_min`` /
    ``S > S_max`` (both False for NaN under IEEE comparison) would
    silently miss this case entirely.
    """
    S = np.array([np.nan])
    with caplog.at_level(logging.WARNING):
        eos._check_entropy_range(S, eos.S_min, eos.S_max, 'test-context')
    matches = [r.message for r in caplog.records if 'test-context' in r.message]
    assert matches, 'expected a warning naming test-context'
    assert '1 non-finite' in matches[0]
    assert '0 out-of-range' in matches[0]


@needs_eos
def test_check_entropy_range_warns_on_out_of_range(eos, caplog):
    """A finite S past S_max must warn with the out-of-range count,
    not the non-finite count.
    """
    S = np.array([eos.S_max + 1.0e5])
    with caplog.at_level(logging.WARNING):
        eos._check_entropy_range(S, eos.S_min, eos.S_max, 'test-context')
    matches = [r.message for r in caplog.records if 'test-context' in r.message]
    assert matches
    assert '0 non-finite' in matches[0]
    assert '1 out-of-range' in matches[0]


@needs_eos
def test_check_entropy_range_strict_raises_runtime_error(eos_strict, caplog):
    """With strict_range=True the same out-of-range input raises
    RuntimeError instead of only warning.

    Discriminator: a regression that checked strict_range before
    computing the counts, or that raised unconditionally regardless
    of range, would either never raise here or raise on in-domain
    input elsewhere.
    """
    S = np.array([eos_strict.S_max + 1.0e5])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match='out-of-range'):
            eos_strict._check_entropy_range(S, eos_strict.S_min, eos_strict.S_max, 'test-context')


@needs_eos
def test_check_entropy_range_strict_in_domain_does_not_raise(eos_strict):
    """strict_range=True must not raise on in-domain input.

    Discriminator: a regression that raises unconditionally whenever
    strict_range is set (ignoring the counts) would fail here.
    """
    S = np.array([0.5 * (eos_strict.S_min + eos_strict.S_max)])
    eos_strict._check_entropy_range(S, eos_strict.S_min, eos_strict.S_max, 'test-context')


# ──────────────────────────────────────────────────────────────────────
#              temperature() via _lookup_phase_weighted
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_temperature_nan_warns_on_both_branch_contexts(eos, caplog):
    """A NaN S makes both the solid_used and melt_used masks pass
    through (neither ``phi >= 1`` nor ``phi <= 0`` is well-defined for
    NaN input), so both the solid and melt table-lookup contexts warn.

    Discriminator: a regression that short-circuited after the first
    branch check would only report one of the two contexts.
    """
    P = np.array([5.0e10])
    with caplog.at_level(logging.WARNING):
        T = eos.temperature(P, np.array([np.nan]))
    assert np.all(np.isnan(np.asarray(T)))
    messages = [r.message for r in caplog.records]
    assert any('temperature (solid table lookup)' in m for m in messages)
    assert any('temperature (melt table lookup)' in m for m in messages)


@needs_eos
def test_temperature_out_of_range_low_warns_solid_context_only(eos, caplog):
    """S far below S_min only pushes the solid-branch lookup out of
    its table range; the melt context must stay silent.
    """
    P = np.array([5.0e10])
    with caplog.at_level(logging.WARNING):
        T = eos.temperature(P, np.array([eos.S_min - 1.0e5]))
    assert np.all(np.isfinite(np.asarray(T)))
    messages = [r.message for r in caplog.records]
    assert any('temperature (solid table lookup)' in m for m in messages)
    assert not any('temperature (melt table lookup)' in m for m in messages)


@needs_eos
def test_temperature_out_of_range_high_warns_melt_context_only(eos, caplog):
    """S far above S_max only pushes the melt-branch lookup out of its
    table range; the solid context must stay silent.
    """
    P = np.array([5.0e10])
    with caplog.at_level(logging.WARNING):
        T = eos.temperature(P, np.array([eos.S_max + 1.0e5]))
    assert np.all(np.isfinite(np.asarray(T)))
    messages = [r.message for r in caplog.records]
    assert any('temperature (melt table lookup)' in m for m in messages)
    assert not any('temperature (solid table lookup)' in m for m in messages)


@needs_eos
def test_temperature_deep_in_phase_does_not_warn(eos, caplog):
    """A comfortable margin below the solidus or above the liquidus
    must not trip either branch's range check.

    False-positive guard: the masked check must exclude the branch
    the blend does not actually use at this phase, not merely clamp
    the reported value there.
    """
    P = np.array([5.0e10])
    S_sol = float(eos.solidus_entropy(P).item())
    S_liq = float(eos.liquidus_entropy(P).item())
    with caplog.at_level(logging.WARNING):
        eos.temperature(P, np.array([S_sol - 500.0]))
        eos.temperature(P, np.array([S_liq + 500.0]))
    assert not any('table lookup' in r.message for r in caplog.records)


@needs_eos
def test_temperature_strict_raises_on_nan(eos_strict):
    """strict_range=True converts the NaN warning into a RuntimeError
    naming the solid table-lookup context (the first branch checked).
    """
    P = np.array([5.0e10])
    with pytest.raises(RuntimeError, match='temperature .* table lookup'):
        eos_strict.temperature(P, np.array([np.nan]))


# ──────────────────────────────────────────────────────────────────────
#                              density()
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_density_nan_warns_solid_context_only(eos, caplog):
    """Unlike ``temperature``, ``density`` selects a single branch by
    ``phi >= 0.5`` rather than masking both by phase weight; for NaN
    input this resolves to the solid branch only.

    Discriminator: assuming symmetric dual-branch warnings here (as
    for ``temperature``) would be wrong; density's branch-selection
    logic differs and must be tested against its own behavior.
    """
    P = np.array([5.0e10])
    with caplog.at_level(logging.WARNING):
        rho = eos.density(P, np.array([np.nan]))
    assert np.all(np.isnan(np.asarray(rho)))
    messages = [r.message for r in caplog.records]
    assert any('density (solid table lookup)' in m for m in messages)
    assert not any('density (melt table lookup)' in m for m in messages)


@needs_eos
def test_density_out_of_range_low_warns_solid_context_only(eos, caplog):
    """S far below S_min: only the solid-branch density lookup is
    out of range.
    """
    P = np.array([5.0e10])
    with caplog.at_level(logging.WARNING):
        rho = eos.density(P, np.array([eos.S_min - 1.0e5]))
    assert np.all(np.isfinite(np.asarray(rho)))
    messages = [r.message for r in caplog.records]
    assert any('density (solid table lookup)' in m for m in messages)
    assert not any('density (melt table lookup)' in m for m in messages)


@needs_eos
def test_density_out_of_range_high_warns_melt_context_only(eos, caplog):
    """S far above S_max: only the melt-branch density lookup is out
    of range.
    """
    P = np.array([5.0e10])
    with caplog.at_level(logging.WARNING):
        rho = eos.density(P, np.array([eos.S_max + 1.0e5]))
    assert np.all(np.isfinite(np.asarray(rho)))
    messages = [r.message for r in caplog.records]
    assert any('density (melt table lookup)' in m for m in messages)
    assert not any('density (solid table lookup)' in m for m in messages)


@needs_eos
def test_density_deep_in_phase_does_not_warn(eos, caplog):
    """A comfortable margin below the solidus or above the liquidus
    must not trip density's range check either.
    """
    P = np.array([5.0e10])
    S_sol = float(eos.solidus_entropy(P).item())
    S_liq = float(eos.liquidus_entropy(P).item())
    with caplog.at_level(logging.WARNING):
        eos.density(P, np.array([S_sol - 500.0]))
        eos.density(P, np.array([S_liq + 500.0]))
    assert not any('table lookup' in r.message for r in caplog.records)


@needs_eos
def test_density_strict_raises_on_out_of_range(eos_strict):
    """strict_range=True converts the out-of-range warning into a
    RuntimeError for density too.
    """
    P = np.array([5.0e10])
    with pytest.raises(RuntimeError, match='density .* table lookup'):
        eos_strict.density(P, np.array([eos_strict.S_min - 1.0e5]))


# ──────────────────────────────────────────────────────────────────────
#   _lookup_at_phase_boundary (density) — phase-boundary table edge
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_density_phase_boundary_narrowed_table_warns(eos, caplog, monkeypatch):
    """``density`` evaluates the solidus entropy against the solid
    table's own S range unconditionally, before any mushy-zone or
    single-phase branch selection, since the Lever Rule needs the
    phase-boundary density regardless of which branch the actual S
    falls in. Narrow the solid table's own upper S bound below the
    real solidus entropy at this P, then call ``density`` with an S
    deep in the solid single-phase branch, so that branch's own
    (unmodified) range check stays silent, and confirm the
    phase-boundary check still warns.

    Discriminator: a regression that drops the range check inside
    ``_lookup_at_phase_boundary`` would pass every other density test
    in this file, since those only narrow the caller's S, never a
    table's own bound.
    """
    P = np.array([5.0e10])
    S_sol = float(eos.solidus_entropy(P).item())
    narrowed_S = eos._tables['density_solid']['S'].copy()
    narrowed_S[-1] = S_sol - 500.0
    monkeypatch.setitem(eos._tables['density_solid'], 'S', narrowed_S)
    with caplog.at_level(logging.WARNING):
        rho = eos.density(P, np.array([S_sol - 1000.0]))
    assert np.all(np.isfinite(np.asarray(rho)))
    messages = [r.message for r in caplog.records]
    assert any('density (phase-boundary solid table lookup)' in m for m in messages)
    assert not any('density (phase-boundary melt table lookup)' in m for m in messages)
    assert not any('density (solid table lookup)' in m for m in messages)


@needs_eos
def test_density_phase_boundary_strict_raises(eos_strict, monkeypatch):
    """strict_range=True converts the phase-boundary warning above into
    a RuntimeError raised from inside ``_lookup_at_phase_boundary``,
    independent of the caller's own S.
    """
    P = np.array([5.0e10])
    S_sol = float(eos_strict.solidus_entropy(P).item())
    narrowed_S = eos_strict._tables['density_solid']['S'].copy()
    narrowed_S[-1] = S_sol - 500.0
    monkeypatch.setitem(eos_strict._tables['density_solid'], 'S', narrowed_S)
    with pytest.raises(RuntimeError, match='density \\(phase-boundary solid table lookup\\)'):
        eos_strict.density(P, np.array([S_sol - 1000.0]))


# ──────────────────────────────────────────────────────────────────────
#         EntropyPhaseEvaluator._update_eos composite-domain check
# ──────────────────────────────────────────────────────────────────────


def _make_phase_evaluator(entropy_eos):
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    return EntropyPhaseEvaluator(
        entropy_eos=entropy_eos,
        gravitational_acceleration=9.81,
        thermal_conductivity_solid=4.0,
        thermal_conductivity_liquid=2.0,
        cp_blend='latent',
        matprop_smooth_width=0.01,
    )


@needs_eos
def test_phase_evaluator_mid_domain_does_not_warn(eos, caplog):
    """A mid-domain entropy must not trip the composite-domain check
    at the top of ``_update_eos``.
    """
    phase = _make_phase_evaluator(eos)
    phase.set_pressure(np.array([5.0e10]))
    phase.set_entropy(np.array([0.5 * (eos.S_min + eos.S_max)]))
    with caplog.at_level(logging.WARNING):
        phase.update()
    assert not any('entropy_phase (composite domain)' in r.message for r in caplog.records)
    assert np.all(np.isfinite(np.asarray(phase.temperature())))


@needs_eos
def test_phase_evaluator_out_of_range_finite_warns_and_returns_clamped(eos, caplog):
    """This is the exact bug scenario the fix targets: an out-of-range
    but finite entropy must not silently clamp to a table-edge value
    without a trace. Non-strict must warn (root-cause diagnostic) and
    still return the finite clamped temperature, not raise.

    Discriminator: before the fix this path was silent; a regression
    that dropped the composite check call would pass this assertion
    on the returned value but fail the warning assertion.
    """
    phase = _make_phase_evaluator(eos)
    phase.set_pressure(np.array([5.0e10]))
    phase.set_entropy(np.array([eos.S_max + 1.0e5]))
    with caplog.at_level(logging.WARNING):
        phase.update()
    messages = [r.message for r in caplog.records]
    assert any('entropy_phase (composite domain)' in m for m in messages)
    T = np.asarray(phase.temperature())
    assert np.all(np.isfinite(T)), 'clamped out-of-range entropy must still yield a finite T'


@needs_eos
def test_phase_evaluator_out_of_range_finite_strict_raises(eos_strict):
    """strict_range=True must convert the out-of-range-but-finite case
    into a RuntimeError raised by the new composite check, before any
    table lookup happens.
    """
    phase = _make_phase_evaluator(eos_strict)
    phase.set_pressure(np.array([5.0e10]))
    phase.set_entropy(np.array([eos_strict.S_max + 1.0e5]))
    with pytest.raises(RuntimeError, match='entropy_phase \\(composite domain\\)'):
        phase.update()


@needs_eos
def test_phase_evaluator_nan_warns_via_new_check_and_raises_via_old_guard(eos, caplog):
    """NaN entropy always raises for the numpy entropy_phase path,
    strict or not, but through two different mechanisms.

    Non-strict: the new composite-domain check only warns (it does
    not raise here since strict_range is False); the pre-existing,
    unconditional downstream NaN guard at the end of ``_update_eos``
    is what actually raises, once the blended temperature comes out
    NaN. Assert both: the new check's warning is present, and a
    RuntimeError is raised regardless.
    """
    phase = _make_phase_evaluator(eos)
    phase.set_pressure(np.array([5.0e10]))
    phase.set_entropy(np.array([np.nan]))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            phase.update()
    messages = [r.message for r in caplog.records]
    assert any('entropy_phase (composite domain)' in m for m in messages), (
        'the new composite-domain check must still warn on NaN even '
        'though the old downstream guard is what raises here'
    )


@needs_eos
def test_phase_evaluator_nan_strict_raises_via_new_check(eos_strict):
    """With strict_range=True, the new composite-domain check raises
    first (before the old downstream guard is ever reached), so the
    RuntimeError carries the new check's message format.
    """
    phase = _make_phase_evaluator(eos_strict)
    phase.set_pressure(np.array([5.0e10]))
    phase.set_entropy(np.array([np.nan]))
    with pytest.raises(RuntimeError, match='entropy_phase \\(composite domain\\)'):
        phase.update()


# ──────────────────────────────────────────────────────────────────────
#   _table_lookup closure (single-phase table evaluation) — hot-path
#   edge check distinct from both the composite-domain check and the
#   phase-boundary check
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_phase_evaluator_single_phase_solid_narrowed_table_warns(eos, caplog, monkeypatch):
    """``_update_eos`` Step 4 evaluates the single-phase table at the
    ACTUAL entropy, masked to only the branch ``gphi`` selects, and
    checks it against that table's own S range. This is a distinct
    call site from the phase-boundary check (Step 2, uses S_sol/S_liq)
    and the composite-domain check (line 210, uses the cached global
    S_min/S_max), so narrowing only a table's own bound here, without
    moving S outside the global domain, must isolate this check alone.

    With ``cp_blend='latent'`` (this fixture's setting), Step 2 never
    calls ``_lookup_at_phase_boundary`` for heat_capacity, so narrowing
    ``heat_capacity_solid`` cannot trip the phase-boundary check.

    Discriminator: a regression that drops the range check inside
    ``_table_lookup`` would pass every other phase-evaluator test in
    this file, since those only move S outside the global domain,
    never narrow a single table's own bound while S stays inside it.
    """
    P = np.array([5.0e10])
    S_sol = float(eos.solidus_entropy(P).item())
    S_arr = S_sol - 1000.0  # deep solid (gphi < 0): solid_used everywhere, melt_used nowhere
    narrowed_S = eos._tables['heat_capacity_solid']['S'].copy()
    narrowed_S[0] = S_arr + 500.0
    monkeypatch.setitem(eos._tables['heat_capacity_solid'], 'S', narrowed_S)
    phase = _make_phase_evaluator(eos)
    phase.set_pressure(P)
    phase.set_entropy(np.array([S_arr]))
    with caplog.at_level(logging.WARNING):
        phase.update()
    assert np.all(np.isfinite(np.asarray(phase.temperature())))
    messages = [r.message for r in caplog.records]
    assert any('heat_capacity (single-phase solid table lookup)' in m for m in messages)
    assert not any('heat_capacity (single-phase melt table lookup)' in m for m in messages)
    assert not any('entropy_phase (composite domain)' in m for m in messages)
    assert not any('phase-boundary' in m for m in messages)


@needs_eos
def test_phase_evaluator_single_phase_strict_raises(eos_strict, monkeypatch):
    """strict_range=True converts the single-phase table-edge warning
    above into a RuntimeError raised from inside the ``_table_lookup``
    closure, distinct from the composite-domain and phase-boundary
    RuntimeErrors covered elsewhere in this file.
    """
    P = np.array([5.0e10])
    S_sol = float(eos_strict.solidus_entropy(P).item())
    S_arr = S_sol - 1000.0
    narrowed_S = eos_strict._tables['heat_capacity_solid']['S'].copy()
    narrowed_S[0] = S_arr + 500.0
    monkeypatch.setitem(eos_strict._tables['heat_capacity_solid'], 'S', narrowed_S)
    phase = _make_phase_evaluator(eos_strict)
    phase.set_pressure(P)
    phase.set_entropy(np.array([S_arr]))
    with pytest.raises(RuntimeError, match='heat_capacity \\(single-phase solid table lookup\\)'):
        phase.update()


# ──────────────────────────────────────────────────────────────────────
#   temperature_scalar() / _melt_fraction_scalar() — +-inf entropy must
#   clamp like the vectorized and JAX paths, not collapse to NaN
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_temperature_scalar_positive_inf_matches_vectorized_clamp(eos):
    """+inf S is out-of-range, not NaN: ``_melt_fraction_scalar`` must
    resolve it to phi=1.0 (all-melt) so ``temperature_scalar`` returns
    the same clamped table-edge value as the vectorized ``temperature``.

    Discriminator: a regression that reintroduces
    ``if not math.isfinite(S): return math.nan`` would return NaN here
    while ``temperature()`` still returns a finite clamped value.
    """
    P = 5.0e10
    scalar_val = eos.temperature_scalar(P, float('inf'))
    vector_val = eos.temperature(np.array([P]), np.array([np.inf]))
    assert np.isfinite(scalar_val)
    assert np.isclose(scalar_val, float(vector_val[0]))


@needs_eos
def test_temperature_scalar_negative_inf_matches_vectorized_clamp(eos):
    """-inf S must resolve to phi=0.0 (all-solid), mirroring the +inf
    case above.
    """
    P = 5.0e10
    scalar_val = eos.temperature_scalar(P, float('-inf'))
    vector_val = eos.temperature(np.array([P]), np.array([-np.inf]))
    assert np.isfinite(scalar_val)
    assert np.isclose(scalar_val, float(vector_val[0]))


@needs_eos
def test_temperature_scalar_nan_still_returns_nan(eos):
    """True NaN must still short-circuit to NaN: this is the case the
    docstring's "min/max keep the first non-NaN operand" hazard
    actually describes, distinct from +-inf.
    """
    assert np.isnan(eos.temperature_scalar(5.0e10, float('nan')))


@needs_eos
def test_temperature_scalar_strict_raises_on_inf(eos_strict):
    """strict_range=True must raise on +inf S, the same as it does for
    an out-of-range finite value, instead of silently returning NaN.
    """
    with pytest.raises(RuntimeError, match='temperature_scalar .* table lookup'):
        eos_strict.temperature_scalar(5.0e10, float('inf'))


# ──────────────────────────────────────────────────────────────────────
#   _specific_enthalpy_scalar() — must validate entropy range like
#   the vectorized specific_enthalpy()
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_specific_enthalpy_scalar_nan_warns(eos, caplog):
    """A NaN S must warn via ``_check_entropy_range``, not silently
    clamp to a table-edge enthalpy with no diagnostic.

    Discriminator: prior to the fix, ``_specific_enthalpy_scalar`` had
    no range check at all, so this warning was never emitted.
    """
    P = 5.0e10
    with caplog.at_level(logging.WARNING):
        h = eos._specific_enthalpy_scalar(P, float('nan'))
    assert np.isfinite(h)
    messages = [r.message for r in caplog.records]
    assert any('specific_enthalpy_scalar' in m for m in messages)


@needs_eos
def test_specific_enthalpy_scalar_strict_raises_on_nan(eos_strict):
    """strict_range=True must raise instead of returning a clamped
    enthalpy for NaN S.
    """
    with pytest.raises(RuntimeError, match='specific_enthalpy_scalar'):
        eos_strict._specific_enthalpy_scalar(5.0e10, float('nan'))


@needs_eos
def test_specific_enthalpy_scalar_in_domain_does_not_warn(eos, caplog):
    """An in-domain S must not warn, and must still match the
    vectorized ``specific_enthalpy`` result exactly (no regression to
    the interpolated value itself from adding the range check).
    """
    P = 5.0e10
    S_mid = 0.5 * (eos.S_min + eos.S_max)
    with caplog.at_level(logging.WARNING):
        h_scalar = eos._specific_enthalpy_scalar(P, S_mid)
    h_vector = eos.specific_enthalpy(np.array([P]), np.array([S_mid]))
    assert np.isclose(h_scalar, float(h_vector[0]))
    assert not any('specific_enthalpy_scalar' in r.message for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────
#   melt_fraction() — the one property method feeding the live CVODE
#   phi_step_cap. clip maps +-inf to a valid-looking 0/1 and passes NaN
#   through, so this entry must warn like its siblings.
# ──────────────────────────────────────────────────────────────────────


@needs_eos
def test_melt_fraction_nan_warns(eos, caplog):
    """A NaN S must warn via ``_check_entropy_range`` naming the
    ``melt_fraction`` context, and phi stays NaN (clip passes it
    through unchanged, non-strict path leaves the value alone).

    Discriminator: before the guard, ``melt_fraction`` had no range
    check, so no warning was emitted for any non-finite input.
    """
    P = np.array([5.0e10])
    with caplog.at_level(logging.WARNING):
        phi = eos.melt_fraction(P, np.array([np.nan]))
    assert np.all(np.isnan(np.asarray(phi)))
    matches = [r.message for r in caplog.records if 'melt_fraction' in r.message]
    assert matches, 'expected a warning naming melt_fraction'
    assert '1 non-finite' in matches[0]


@needs_eos
def test_melt_fraction_positive_inf_warns_and_value_unchanged(eos, caplog):
    """+inf S is non-finite: it must warn, while the non-strict return
    value stays the clamped phi=1.0 (fully molten), unchanged from the
    pre-guard behaviour.

    Discriminator: an implementation that changed the value semantics
    (e.g. propagated NaN like the JAX path) would fail the ==1.0 check;
    the guard must add only the warning.
    """
    P = np.array([5.0e10])
    with caplog.at_level(logging.WARNING):
        phi = eos.melt_fraction(P, np.array([np.inf]))
    assert float(np.asarray(phi)[0]) == 1.0
    assert any('melt_fraction' in r.message for r in caplog.records)


@needs_eos
def test_melt_fraction_negative_inf_warns_and_value_unchanged(eos, caplog):
    """-inf S must warn while the non-strict return value stays the
    clamped phi=0.0 (fully solid), mirroring the +inf case.
    """
    P = np.array([5.0e10])
    with caplog.at_level(logging.WARNING):
        phi = eos.melt_fraction(P, np.array([-np.inf]))
    assert float(np.asarray(phi)[0]) == 0.0
    assert any('melt_fraction' in r.message for r in caplog.records)


@needs_eos
def test_melt_fraction_in_domain_does_not_warn(eos, caplog):
    """A mid-domain S must not warn, and phi must stay in [0, 1].

    False-positive guard: the new check must not fire on ordinary
    in-range input, the path the live CVODE step cap hits every step.
    """
    P = np.array([5.0e10])
    S_mid = np.array([0.5 * (eos.S_min + eos.S_max)])
    with caplog.at_level(logging.WARNING):
        phi = eos.melt_fraction(P, S_mid)
    assert 0.0 <= float(np.asarray(phi)[0]) <= 1.0
    assert not any('melt_fraction' in r.message for r in caplog.records)


@needs_eos
def test_melt_fraction_strict_raises_on_nonfinite(eos_strict):
    """strict_range=True must convert the melt_fraction warning into a
    RuntimeError for a non-finite S, matching the sibling contract.
    """
    P = np.array([5.0e10])
    with pytest.raises(RuntimeError, match='melt_fraction'):
        eos_strict.melt_fraction(P, np.array([np.inf]))
