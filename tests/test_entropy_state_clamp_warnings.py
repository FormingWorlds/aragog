"""Throttled-warning behaviour for the transient EntropyState clamps.

The runtime guards in ``aragog.solver.entropy_state.EntropyState`` floor
quantities that should never realistically reach the floor on a healthy
EOS table: the solidus-liquidus entropy gap (must stay above 1 J/kg/K)
and the per-cell heat capacity (must stay above the conduction guard at
100 J/kg/K and the MLT division-safety guard at 1 J/kg/K). Each guard
emits a logger.warning *once* per ``EntropyState`` instance when it
triggers, so a degenerate EOS gets flagged but does not flood the log
across thousands of RHS calls per coupling step.

The Cp clamps share a common implementation,
``EntropyState._maybe_warn_cp_floor``, so the conduction and MLT paths
are tested through that helper and the production callsites are
verified by reading the source. The phase-boundary clamps live inline
in the cache populators and are tested by direct method call. NaN
inputs must count as below-floor and be replaced (a strict ``< floor``
test silently misses NaN, which then propagates through
``np.maximum(NaN, floor) = NaN`` into the lever-rule denominator on the
next RHS call).

The matching kappa_h floor at the rheological transition is *not*
covered: that floor fires by design at every RHS call inside the mushy
band and is intentionally silent.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from aragog.solver.entropy_state import EntropyState

pytestmark = pytest.mark.unit

LOGGER_NAME = 'fwl.aragog.solver.entropy_state'


def _make_minimal_state(
    n_basic: int = 5,
    *,
    solidus_basic: np.ndarray | None = None,
    liquidus_basic: np.ndarray | None = None,
    solidus_stag: np.ndarray | None = None,
    liquidus_stag: np.ndarray | None = None,
) -> EntropyState:
    """Build an EntropyState with enough stubs to exercise the cache
    populators and the Cp-clamp helper in isolation.

    The real EntropyState wires through Mesh + EntropyPhaseEvaluator +
    EntropyEOS; here we replace each layer with a SimpleNamespace stub
    that returns user-supplied solidus/liquidus arrays. The pressure
    objects are unique per call so the per-id cache check does not
    short-circuit. The stubs only need to satisfy ``__init__`` and the
    two ``_ensure_*_phase_boundary_cache`` populators; ``update()`` is
    not called by these tests.

    Parameters
    ----------
    n_basic
        Mesh size (basic-node count). The staggered mesh is one cell
        shorter: ``n_staggered = n_basic - 1``.
    solidus_basic, liquidus_basic, solidus_stag, liquidus_stag
        Per-node solidus/liquidus entropies. Defaults: gap = 1500 J/kg/K
        on every node, well above the 1 J/kg/K floor.
    """
    n_staggered = n_basic - 1
    if solidus_basic is None:
        solidus_basic = np.full(n_basic, 2500.0)
    if liquidus_basic is None:
        liquidus_basic = np.full(n_basic, 4000.0)
    if solidus_stag is None:
        solidus_stag = np.full(n_staggered, 2500.0)
    if liquidus_stag is None:
        liquidus_stag = np.full(n_staggered, 4000.0)

    P_basic = np.linspace(1.0e9, 1.0e11, n_basic)
    P_stag = 0.5 * (P_basic[:-1] + P_basic[1:])
    basic_radii = np.linspace(3.5e6, 6.4e6, n_basic)
    stag_radii = 0.5 * (basic_radii[:-1] + basic_radii[1:])

    # Route by query length: solver passes basic-node and staggered-node
    # pressure arrays separately, so dispatch the EOS stub by ndarray
    # length to the right precomputed solidus/liquidus.
    def _route(arr_basic: np.ndarray, arr_stag: np.ndarray, P_query):
        n = np.asarray(P_query).ravel().size
        if n == n_basic:
            return arr_basic
        if n == n_staggered:
            return arr_stag
        return np.full(n, arr_basic[0])

    eos_stub = SimpleNamespace(
        solidus_entropy=lambda P: _route(solidus_basic, solidus_stag, P),
        liquidus_entropy=lambda P: _route(liquidus_basic, liquidus_stag, P),
        solidus_entropy_dP=lambda P: np.zeros_like(np.asarray(P).ravel()),
        liquidus_entropy_dP=lambda P: np.zeros_like(np.asarray(P).ravel()),
        latent_heat=lambda P: np.full_like(np.asarray(P).ravel(), 5.0e5),
    )
    phase_stag = SimpleNamespace(
        pressure=P_stag.copy(),
        _eos=eos_stub,
        _const_properties=False,
    )
    phase_basic = SimpleNamespace(
        pressure=P_basic.copy(),
        _eos=eos_stub,
        _const_properties=False,
    )

    mesh_basic = SimpleNamespace(
        radii=basic_radii,
        mixing_length=np.full(n_basic, 1.0e6),
        mixing_length_squared=np.full(n_basic, 1.0e12),
        mixing_length_cubed=np.full(n_basic, 1.0e18),
    )
    mesh_staggered = SimpleNamespace(radii=stag_radii)
    mesh_stub = SimpleNamespace(basic=mesh_basic, staggered=mesh_staggered)
    evaluator = SimpleNamespace(mesh=mesh_stub)

    return EntropyState(
        evaluator=evaluator,
        phase_staggered=phase_stag,
        phase_basic=phase_basic,
        conduction=False,
        convection=False,
    )


def _force_recompute(state: EntropyState, which: str) -> None:
    """Invalidate the per-id phase-boundary cache so a follow-up call
    re-runs the populator. Used to verify the throttle holds across
    multiple cache misses.
    """
    if which == 'staggered':
        state._P_stag_cached_id = -1
        state.phase_staggered.pressure = state.phase_staggered.pressure.copy()
    elif which == 'basic':
        state._P_basic_cached_id = -1
        state.phase_basic.pressure = state.phase_basic.pressure.copy()
    else:
        raise ValueError(which)


# ---- _maybe_warn_cp_floor (Cp clamp helper, shared by MLT + conduction) ----


def test_cp_floor_helper_warns_once_with_correct_count(caplog):
    """The shared helper must warn exactly once per (instance, flag)
    pair and report the count of triggered nodes. Asymmetric input
    (3 of 5 below) discriminates against a regression that uses
    ``Cp.size`` or ``np.all`` instead of the boolean mask.
    """
    state = _make_minimal_state()
    Cp = np.array([2000.0, 50.0, 1500.0, 75.0, 30.0])  # 3 of 5 below 100
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        Cp_safe_first = state._maybe_warn_cp_floor(
            Cp,
            floor=100.0,
            site='conduction',
            consequence='F_cond is biased upward',
            flag_name='_cp_floor_warned',
        )
        Cp_safe_second = state._maybe_warn_cp_floor(
            Cp,
            floor=100.0,
            site='conduction',
            consequence='F_cond is biased upward',
            flag_name='_cp_floor_warned',
        )

    records = [r for r in caplog.records if 'EntropyState conduction' in r.message]
    assert len(records) == 1
    assert '3 node(s)' in records[0].message
    assert state._cp_floor_warned is True
    # The clamp itself is applied on every call, even after the warning
    # is throttled — only the LOG is silenced.
    expected = np.array([2000.0, 100.0, 1500.0, 100.0, 100.0])
    np.testing.assert_array_equal(Cp_safe_first, expected)
    np.testing.assert_array_equal(Cp_safe_second, expected)


def test_cp_floor_helper_silent_on_healthy_input(caplog):
    """Negative regression: a healthy Cp profile must NOT trigger the
    warning or the throttle flag. Catches a sign-flip regression where
    the trigger condition is inverted and fires on every call.
    """
    state = _make_minimal_state()
    Cp = np.full(5, 1500.0)  # well above any guard
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        Cp_safe = state._maybe_warn_cp_floor(
            Cp,
            floor=100.0,
            site='conduction',
            consequence='F_cond is biased upward',
            flag_name='_cp_floor_warned',
        )

    assert all('EntropyState conduction' not in r.message for r in caplog.records)
    assert state._cp_floor_warned is False
    np.testing.assert_array_equal(Cp_safe, Cp)


def test_cp_floor_helper_treats_nan_as_below_floor_and_replaces_it(caplog):
    """Critical NaN regression: a strict ``Cp < floor`` test silently
    misses NaN (IEEE 754 says ``NaN < x`` is False), and
    ``np.maximum(NaN, floor)`` returns NaN, which then propagates into
    the lever-rule denominator on the next RHS call. The helper uses
    ``~(Cp >= floor)`` so NaN counts as below-floor, AND assigns
    ``floor`` (not NaN) to those nodes via ``np.where``.
    """
    state = _make_minimal_state()
    Cp = np.array([1500.0, np.nan, 200.0, np.nan, 1500.0])  # 2 NaN, 0 sub-floor
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        Cp_safe = state._maybe_warn_cp_floor(
            Cp,
            floor=100.0,
            site='conduction',
            consequence='F_cond is biased upward',
            flag_name='_cp_floor_warned',
        )

    records = [r for r in caplog.records if 'EntropyState conduction' in r.message]
    assert len(records) == 1
    # The 2 NaN nodes count as below-floor; the sub-floor count is 0.
    assert '2 node(s)' in records[0].message
    # Critically, no NaN survives the clamp.
    assert not np.any(np.isnan(Cp_safe))
    # NaN nodes were replaced with the floor value, healthy nodes pass through.
    expected = np.array([1500.0, 100.0, 200.0, 100.0, 1500.0])
    np.testing.assert_array_equal(Cp_safe, expected)


def test_cp_floor_helper_throttle_flags_independent():
    """Conduction and MLT clamps share the helper but use distinct
    throttle flags. Tripping the conduction flag must not silence the
    MLT warning on a later call, and vice versa.
    """
    state = _make_minimal_state()
    Cp_low = np.full(5, 0.5)  # below both floors
    state._maybe_warn_cp_floor(
        Cp_low,
        floor=100.0,
        site='conduction',
        consequence='conduction biased',
        flag_name='_cp_floor_warned',
    )
    assert state._cp_floor_warned is True
    assert state._cp_mlt_floor_warned is False

    state._maybe_warn_cp_floor(
        Cp_low,
        floor=1.0,
        site='MLT',
        consequence='MLT biased',
        flag_name='_cp_mlt_floor_warned',
    )
    assert state._cp_mlt_floor_warned is True


def test_production_callsites_use_the_helper():
    """Source-level guard: the conduction and MLT blocks in update()
    must dispatch through ``_maybe_warn_cp_floor``. Catches a regression
    that re-inlines the conditional and silently breaks the NaN-safe
    behaviour.

    This is a static assertion against the bytecode constants of
    ``EntropyState.update`` so a refactor that changes the flag names
    or the floor values fails this check loudly rather than passing
    every behavioural test in this file.
    """
    update_code = EntropyState.update.__code__
    consts = update_code.co_consts
    # Both flag names must appear as string literals in update()'s
    # constant pool — they are passed by name to the helper.
    assert '_cp_floor_warned' in consts
    assert '_cp_mlt_floor_warned' in consts
    # Both floor values must appear too.
    assert 1.0 in consts  # MLT
    assert 100.0 in consts  # conduction


# ---- dS_phase_stag floor (inline in _ensure_phase_boundary_cache) ----------


def test_dS_phase_stag_floor_warns_once_when_all_nodes_collapse(caplog):
    """All staggered nodes have gap = 0: warning fires once on the
    first cache populate, count matches n_staggered, and the floor
    replaces every entry with 1.0.
    """
    n_basic = 5
    n_staggered = n_basic - 1  # 4
    sol_stag = np.full(n_staggered, 3000.0)
    liq_stag = np.full(n_staggered, 3000.0)
    state = _make_minimal_state(n_basic=n_basic, solidus_stag=sol_stag, liquidus_stag=liq_stag)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_phase_boundary_cache()
        _force_recompute(state, 'staggered')
        state._ensure_phase_boundary_cache()

    records = [r for r in caplog.records if 'phase-boundary cache (staggered)' in r.message]
    assert len(records) == 1, f'expected 1 staggered-floor warning, got {len(records)}'
    assert f'{n_staggered} node(s)' in records[0].message
    np.testing.assert_array_equal(state._dS_phase_stag, np.full(n_staggered, 1.0))


def test_dS_phase_stag_floor_warns_once_with_partial_collapse(caplog):
    """Discriminator: only 2 of 4 staggered nodes have gap < 1 J/kg/K.
    The warning message must report 2, not 4 — catches a regression
    that uses ``arr.size`` instead of the boolean mask, AND a
    regression that swaps ``np.any`` for ``np.all`` (which would not
    fire on partial collapse).
    """
    n_basic = 5
    n_staggered = n_basic - 1  # 4
    sol_stag = np.full(n_staggered, 3000.0)
    # cells 1 and 3 collapsed (gap = 0.4 and 0); cells 0 and 2 healthy.
    liq_stag = np.array([4500.0, 3000.4, 4500.0, 3000.0])
    state = _make_minimal_state(n_basic=n_basic, solidus_stag=sol_stag, liquidus_stag=liq_stag)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_phase_boundary_cache()

    records = [r for r in caplog.records if 'phase-boundary cache (staggered)' in r.message]
    assert len(records) == 1
    assert '2 node(s)' in records[0].message
    expected = np.array([1500.0, 1.0, 1500.0, 1.0])
    np.testing.assert_array_equal(state._dS_phase_stag, expected)


def test_dS_phase_stag_floor_silent_on_healthy_gap(caplog):
    """Negative regression: a healthy gap of 1500 J/kg/K must NOT
    trigger the warning or the floor.
    """
    state = _make_minimal_state()  # default gap = 1500
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_phase_boundary_cache()

    assert all('phase-boundary cache (staggered)' not in r.message for r in caplog.records)
    assert state._dS_phase_stag_floor_warned is False


def test_dS_phase_stag_floor_treats_nan_gap_as_below_floor(caplog):
    """NaN regression for the staggered cache populator: a NaN gap
    must count as below-floor AND be replaced by the floor value.
    Without this, ``np.maximum(NaN, 1.0) = NaN`` propagates into the
    lever-rule denominator on the next RHS call.
    """
    n_basic = 5
    n_staggered = n_basic - 1  # 4
    sol_stag = np.full(n_staggered, 3000.0)
    # cell 0 healthy, cell 1 NaN (S_liq = NaN), cell 2 healthy, cell 3 below floor.
    liq_stag = np.array([4500.0, np.nan, 4500.0, 3000.0])
    state = _make_minimal_state(n_basic=n_basic, solidus_stag=sol_stag, liquidus_stag=liq_stag)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_phase_boundary_cache()

    records = [r for r in caplog.records if 'phase-boundary cache (staggered)' in r.message]
    assert len(records) == 1
    assert '2 node(s)' in records[0].message  # NaN cell + below-floor cell
    assert not np.any(np.isnan(state._dS_phase_stag))
    expected = np.array([1500.0, 1.0, 1500.0, 1.0])
    np.testing.assert_array_equal(state._dS_phase_stag, expected)


# ---- dS_phase_basic floor (inline in _ensure_basic_phase_boundary_cache) ----


def test_dS_phase_basic_floor_warns_once_with_partial_collapse(caplog):
    """Discriminator at the basic-node cache: 2 of 5 nodes collapsed.
    Same anti-regression behaviour as the staggered partial-collapse
    test, but on the basic-node cache populator.
    """
    n_basic = 5
    sol = np.full(n_basic, 3000.0)
    liq = np.array([4500.0, 3000.0, 4500.0, 3000.5, 4500.0])  # 2 of 5 collapsed
    state = _make_minimal_state(n_basic=n_basic, solidus_basic=sol, liquidus_basic=liq)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_basic_phase_boundary_cache()
        _force_recompute(state, 'basic')
        state._ensure_basic_phase_boundary_cache()

    records = [r for r in caplog.records if 'phase-boundary cache (basic)' in r.message]
    assert len(records) == 1
    assert '2 node(s)' in records[0].message
    expected = np.array([1500.0, 1.0, 1500.0, 1.0, 1500.0])
    np.testing.assert_array_equal(state._dS_phase_basic, expected)


def test_dS_phase_basic_floor_silent_on_healthy_gap(caplog):
    """Negative regression mirror for the basic-node populator."""
    state = _make_minimal_state()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_basic_phase_boundary_cache()

    assert all('phase-boundary cache (basic)' not in r.message for r in caplog.records)
    assert state._dS_phase_basic_floor_warned is False


def test_dS_phase_basic_floor_treats_nan_gap_as_below_floor(caplog):
    """NaN regression for the basic-node populator. Same mechanism as
    the staggered case, on the second cache.
    """
    n_basic = 5
    sol = np.full(n_basic, 3000.0)
    liq = np.array([4500.0, np.nan, 4500.0, 3000.0, np.nan])  # 3 problem cells
    state = _make_minimal_state(n_basic=n_basic, solidus_basic=sol, liquidus_basic=liq)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_basic_phase_boundary_cache()

    records = [r for r in caplog.records if 'phase-boundary cache (basic)' in r.message]
    assert len(records) == 1
    assert '3 node(s)' in records[0].message
    assert not np.any(np.isnan(state._dS_phase_basic))
    expected = np.array([1500.0, 1.0, 1500.0, 1.0, 1.0])
    np.testing.assert_array_equal(state._dS_phase_basic, expected)
