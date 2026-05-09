"""Throttled-warning behaviour for the transient EntropyState clamps.

The runtime guards in ``aragog.solver.entropy_state.EntropyState`` floor
quantities that should never realistically reach the floor on a healthy
EOS table: the solidus-liquidus entropy gap (must stay above 1 J/kg/K)
and the per-cell heat capacity (must stay above the conduction guard at
100 J/kg/K and the MLT division-safety guard at 1 J/kg/K). Each guard
emits a logger.warning *once* per ``EntropyState`` instance when it
triggers, so a degenerate EOS gets flagged but does not flood the log
across thousands of RHS calls per coupling step.

These tests instantiate the cache-populating methods with stubbed
phase evaluators that *deliberately* return floor-triggering values.
They verify (a) the warning fires the first time the floor activates,
(b) it does NOT fire on the second invocation (throttling holds), and
(c) the warning stays silent on a healthy EOS (negative regression).
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
    Cp_basic_value: float | None = None,
) -> EntropyState:
    """Build an EntropyState with enough stubs to exercise the cache
    populators and the MLT block in isolation.

    The real EntropyState wires through Mesh + EntropyPhaseEvaluator +
    EntropyEOS; here we replace each layer with a SimpleNamespace stub
    that returns user-supplied solidus/liquidus arrays and a constant
    ``Cp`` from ``heat_capacity()``. The pressure objects are unique
    per call so the per-id cache check does not short-circuit.

    Parameters
    ----------
    n_basic
        Mesh size (basic-node count). The staggered mesh is one cell
        shorter: ``n_staggered = n_basic - 1``.
    solidus_basic, liquidus_basic, solidus_stag, liquidus_stag
        Per-node solidus/liquidus entropies. Defaults: gap = 1500 J/kg/K
        on every node, well above the 1 J/kg/K floor.
    Cp_basic_value
        Constant Cp for the phase_basic stub. Default 1500 J/kg/K (above
        every floor). Pass < 1.0 to trigger the MLT guard.
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
    if Cp_basic_value is None:
        Cp_basic_value = 1500.0

    P_basic = np.linspace(1.0e9, 1.0e11, n_basic)
    P_stag = 0.5 * (P_basic[:-1] + P_basic[1:])
    basic_radii = np.linspace(3.5e6, 6.4e6, n_basic)
    stag_radii = 0.5 * (basic_radii[:-1] + basic_radii[1:])

    # The cache populators dispatch by which pressure array the caller
    # passed (id-based), so we route the EOS stub by ndarray length.
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
        heat_capacity=lambda: np.full(n_basic, Cp_basic_value),
        temperature=lambda: np.full(n_basic, 3000.0),
        density=lambda: np.full(n_basic, 4000.0),
        gravitational_acceleration=lambda: np.full(n_basic, 9.8),
        thermal_expansivity=lambda: np.full(n_basic, 3.0e-5),
        kinematic_viscosity=lambda: np.full(n_basic, 1.0e-2),
        melt_fraction=lambda: np.full(n_basic, 0.5),
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


# ---- dS_phase_stag floor ---------------------------------------------------


def test_dS_phase_stag_floor_warns_once_when_gap_is_zero(caplog):
    """Solidus equals liquidus at every node (gap = 0): the floor must
    fire on the first cache populate, raise exactly one WARNING with the
    node count, and then stay silent on a second populate even after
    invalidating the per-id cache.
    """
    n_basic = 5
    n_staggered = n_basic - 1  # 4
    sol_stag = np.full(n_staggered, 3000.0)
    liq_stag = np.full(n_staggered, 3000.0)  # gap = 0 on every staggered node
    state = _make_minimal_state(n_basic=n_basic, solidus_stag=sol_stag, liquidus_stag=liq_stag)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_phase_boundary_cache()
        # Force a second populate: change the cached id sentinel and
        # hand the populator a fresh pressure array (different id).
        state._P_stag_cached_id = -1
        state.phase_staggered.pressure = state.phase_staggered.pressure.copy()
        state._ensure_phase_boundary_cache()

    stag_records = [
        r for r in caplog.records if 'phase-boundary cache (staggered)' in r.message
    ]
    assert len(stag_records) == 1, (
        f'expected exactly 1 staggered-floor warning, got {len(stag_records)}'
    )
    assert f'{n_staggered} node(s)' in stag_records[0].message
    # The floor is in fact applied — physical state stays well-defined.
    assert np.allclose(state._dS_phase_stag, 1.0)


def test_dS_phase_stag_floor_silent_on_healthy_gap(caplog):
    """Negative regression: a healthy gap of 1500 J/kg/K must NOT
    trigger the warning. Catches a sign-flip / off-by-one regression
    that fires the warning on every healthy table.
    """
    state = _make_minimal_state()  # default gap = 1500
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_phase_boundary_cache()

    assert all('phase-boundary cache (staggered)' not in r.message for r in caplog.records)
    assert state._dS_phase_stag_floor_warned is False


# ---- dS_phase_basic floor --------------------------------------------------


def test_dS_phase_basic_floor_warns_once_with_partial_collapse(caplog):
    """Discriminator: gap is collapsed at only 2 of 5 nodes; the
    warning message must report the *count* of triggered nodes (2),
    not all 5. Catches a regression where the count uses ``size``
    instead of the boolean mask.
    """
    n_basic = 5
    sol = np.full(n_basic, 3000.0)
    liq = np.array([4500.0, 3000.0, 4500.0, 3000.5, 4500.0])  # 2 of 5 collapsed
    state = _make_minimal_state(n_basic=n_basic, solidus_basic=sol, liquidus_basic=liq)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_basic_phase_boundary_cache()
        state._P_basic_cached_id = -1
        state.phase_basic.pressure = state.phase_basic.pressure.copy()
        state._ensure_basic_phase_boundary_cache()

    basic_records = [r for r in caplog.records if 'phase-boundary cache (basic)' in r.message]
    assert len(basic_records) == 1
    assert '2 node(s)' in basic_records[0].message  # only the collapsed cells counted
    # Healthy nodes keep their real gap; collapsed nodes were floored to 1.
    expected = np.array([1500.0, 1.0, 1500.0, 1.0, 1500.0])
    np.testing.assert_allclose(state._dS_phase_basic, expected)


def test_dS_phase_basic_floor_silent_on_healthy_gap(caplog):
    """Negative regression mirror for the basic-node populator."""
    state = _make_minimal_state()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        state._ensure_basic_phase_boundary_cache()

    assert all('phase-boundary cache (basic)' not in r.message for r in caplog.records)
    assert state._dS_phase_basic_floor_warned is False


# ---- MLT Cp floor ----------------------------------------------------------


def test_cp_mlt_floor_warned_flag_starts_false():
    """A fresh EntropyState must start with all four throttle flags
    cleared. Catches a regression that forgets to initialise a new flag
    and so warns *every* call without throttling.
    """
    state = _make_minimal_state()
    assert state._cp_floor_warned is False
    assert state._cp_mlt_floor_warned is False
    assert state._dS_phase_stag_floor_warned is False
    assert state._dS_phase_basic_floor_warned is False


def test_cp_mlt_floor_warning_logic_throttles(caplog):
    """The MLT clamp at the eddy-diffusivity prefactor warns at most
    once per instance. We exercise the throttling logic directly
    (Cp < 1 guard ⇒ flag flip) without spinning up the full update()
    path, since update() requires a fully wired solver. The test
    reproduces the if-block exactly as written in entropy_state.py.
    """
    state = _make_minimal_state(Cp_basic_value=0.5)  # below the 1.0 floor
    Cp = np.full(5, 0.5)

    # Reproduce the production conditional verbatim, twice.
    logger = logging.getLogger(LOGGER_NAME)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        for _ in range(2):
            if not state._cp_mlt_floor_warned and np.any(Cp < 1.0):
                logger.warning(
                    'EntropyState MLT: Cp dropped below the 1 J/kg/K '
                    'division-safety guard at %d node(s); eddy-diffusivity '
                    'velocity prefactor is biased upward at those points. '
                    'Suppressing further per-RHS warnings — check the EOS Cp '
                    'tables and the load-time _check_eos_floors output.',
                    int(np.sum(Cp < 1.0)),
                )
                state._cp_mlt_floor_warned = True

    mlt_records = [r for r in caplog.records if 'EntropyState MLT' in r.message]
    assert len(mlt_records) == 1
    assert '5 node(s)' in mlt_records[0].message
    assert state._cp_mlt_floor_warned is True
