"""SUNDIALS root function for the per-call ΔΦ_global cap.

The cap is implemented as a CVODE root function (and equivalent
``solve_ivp`` event for the scipy fallback). When armed, the
integrator returns at the exact time t* where the mass-weighted
|Φ_global(t*) − Φ_global(start)| crosses the configured cap. The
integrator's own trajectory decides termination; there is no
start-time rate estimate, which is required at the 1 M_E PALEOS
rheological transition where any rate extrapolated from t=0
overshoots within the call window and stalls the adaptive dt.

Tests cover:
- ``_EnergyParameters.phi_step_cap`` default and validation
- ``_PhiCapRootFunction.evaluate`` returns g[0] = cap − |ΔΦ_global|
- The rootfn fires (g[0] crosses zero) at the correct ΔΦ_global
- The event factory (scipy fallback) returns the same value as
  the CVODE rootfn at matching states
- Source-pattern regression: the rootfn class subclasses
  ``CV_RootFunction``; ``solve()`` wires it into ``cvode_options``
  with ``nr_rootfns=1``; ``_solve_cvode`` accepts ``phi_cap_rootfn``
- Default None (disabled) keeps existing behaviour: rootfn is never installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from aragog.parser import _EnergyParameters
from aragog.solver.entropy_solver import (
    _CV_ROOTFN_AVAILABLE,
    _DEFAULT_PHASE_BOUNDARY_ENTROPY_MARGIN,
    _phi_cap_event_factory,
    _PhiCapRootFunction,
    _resolve_entropy_margin,
    _resolve_step_cap,
)

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'aragog'


def _make_energy(**overrides):
    base = dict(
        conduction=True,
        convection=True,
        gravitational_separation=True,
        mixing=True,
        radionuclides=True,
        tidal=False,
    )
    base.update(overrides)
    return _EnergyParameters(**base)


def _build_synthetic_eos_and_state(n_stag=8, phi_uniform=0.6):
    """Synthetic EOS + state: every cell at melt fraction ``phi_uniform``.

    Uses uniform density and unit volume per cell so mass weighting
    reduces to simple averaging. Returns
    (eos_mock, P_stag, volume, n_stag, S_stag, phi_global).
    """
    eos = MagicMock()
    eos.density.return_value = np.full(n_stag, 4000.0)
    eos.melt_fraction.return_value = np.full(n_stag, phi_uniform)
    P_stag = np.linspace(1.0e9, 1.4e11, n_stag)
    volume = np.full(n_stag, 1.0e18)
    S_stag = np.full(n_stag, 3000.0)
    return eos, P_stag, volume, n_stag, S_stag, float(phi_uniform)


# ──────────────────────────────────────────────────────────────────────
#                       Config field defaults
# ──────────────────────────────────────────────────────────────────────


def test_phi_step_cap_default_is_none_and_disables():
    """The default phi_step_cap is the None disabled-sentinel.

    Standalone Aragog must keep the cap off unless a value is supplied: a
    positive default would arm the rootfn on every solve() call, a
    behavioural change that would break the bit-parity contract with prior
    runs. ``None`` is the self-documenting disabled sentinel, and the solver
    coerces it (and any non-positive value) to 0.0 before the arming
    inequality, so the cap stays off.
    """
    e = _make_energy()
    assert e.phi_step_cap is None
    # None, explicit zero, and negatives all resolve to the disabled 0.0.
    assert _resolve_step_cap(e.phi_step_cap) == 0.0
    assert _resolve_step_cap(0.0) == 0.0
    assert _resolve_step_cap(-5.0) == 0.0
    # A genuine positive cap survives coercion unchanged; the two verdicts
    # land on opposite sides of the arming inequality.
    assert _resolve_step_cap(0.05) == pytest.approx(0.05)
    assert _resolve_step_cap(e.phi_step_cap) <= 0.0 < _resolve_step_cap(0.05)


def test_phi_step_cap_accepts_positive_values():
    """Recommended production setting (0.05) and an extreme (0.5)."""
    assert _make_energy(phi_step_cap=0.05).phi_step_cap == pytest.approx(0.05)
    assert _make_energy(phi_step_cap=0.5).phi_step_cap == pytest.approx(0.5)


def test_phi_step_cap_accepts_zero_explicitly():
    """Explicit 0.0 (disabled) is a valid configuration and resolves to off."""
    assert _make_energy(phi_step_cap=0.0).phi_step_cap == 0.0
    assert _resolve_step_cap(_make_energy(phi_step_cap=0.0).phi_step_cap) == 0.0


def test_resolve_step_cap_maps_disabled_sentinels_to_zero():
    """``_resolve_step_cap`` collapses every disabled spelling to exactly 0.0.

    The rootfn arming inequality is ``> 0.0`` on a plain float, so None,
    zero, negatives, and non-finite values must all coerce to 0.0 (disabled)
    while a finite positive cap passes through unchanged. This is the guard
    that keeps a sentinel from being misread as a tiny positive threshold.
    """
    disabled_spellings = (
        None,
        0.0,
        -0.0,
        -1e-9,
        -100.0,
        float('nan'),
        float('inf'),
        float('-inf'),
    )
    for disabled in disabled_spellings:
        assert _resolve_step_cap(disabled) == 0.0
    # A real positive cap is preserved to full precision.
    assert _resolve_step_cap(0.05) == pytest.approx(0.05)
    assert _resolve_step_cap(200.0) == pytest.approx(200.0)
    # Boundary of the ``> 0.0`` inequality: the smallest representable positive
    # float is still an armed cap, not a disabled sentinel.
    assert _resolve_step_cap(5e-324) > 0.0
    # Discrimination: disabled sentinels and an armed cap sit on opposite
    # sides of the ``> 0.0`` inequality, so they can never be confused.
    assert _resolve_step_cap(None) <= 0.0 < _resolve_step_cap(0.05)


def test_resolve_entropy_margin_falls_back_to_default_not_disabled():
    """``_resolve_entropy_margin`` never yields a band that disables itself.

    Unlike the step caps, zero is not a valid "off" state for the
    phase-boundary band: a non-positive or non-finite margin would make every
    ``abs(margin) < entropy_margin`` proximity test false and silently stop
    arming the tighter stepping across a crossing. So ``None`` (attribute
    absent, e.g. a coupled build predating the field), non-finite, and
    non-positive inputs must fall back to the 200.0 default, while a genuine
    positive band passes through unchanged. This is the opposite fallback
    target from ``_resolve_step_cap`` (which maps disabled spellings to 0.0),
    so the two must not be confused.
    """
    default = _DEFAULT_PHASE_BOUNDARY_ENTROPY_MARGIN
    assert default == pytest.approx(200.0, rel=1e-12)
    for bad in (None, 0.0, -0.0, -1e-9, -300.0, float('nan'), float('inf'), float('-inf')):
        assert _resolve_entropy_margin(bad) == pytest.approx(default, rel=1e-12)
    # A genuine positive band survives unchanged, including a value far from
    # the default so a silent collapse to 200 would be caught.
    assert _resolve_entropy_margin(400.0) == pytest.approx(400.0, rel=1e-12)
    assert _resolve_entropy_margin(50.0) == pytest.approx(50.0, rel=1e-12)
    # Discrimination: a disabled-style input and a real band land on opposite
    # sides of the default, never both collapsing to 0.0 the way a cap would.
    assert (
        _resolve_entropy_margin(-1.0) == pytest.approx(default) != _resolve_entropy_margin(50.0)
    )


def test_temperature_and_entropy_step_caps_default_none():
    """The temperature and entropy step caps default to the None sentinel.

    Standalone Aragog must keep the caps off unless a value is supplied. The
    PROTEUS wrapper is what passes concrete positive caps for the coupled
    zalmoxis stack, so a non-None default here would silently change every
    standalone run. None resolves to the disabled 0.0; positive values
    round-trip and resolve to themselves.
    """
    e = _make_energy()
    assert e.temperature_step_cap is None
    assert e.entropy_step_cap is None
    assert _resolve_step_cap(e.temperature_step_cap) == 0.0
    assert _resolve_step_cap(e.entropy_step_cap) == 0.0
    assert _make_energy(temperature_step_cap=150.0).temperature_step_cap == pytest.approx(150.0)
    assert _make_energy(entropy_step_cap=80.0).entropy_step_cap == pytest.approx(80.0)
    assert _resolve_step_cap(150.0) == pytest.approx(150.0)


# ──────────────────────────────────────────────────────────────────────
#                   _PhiCapRootFunction.evaluate
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not _CV_ROOTFN_AVAILABLE,
    reason='scikits_odes_sundials.cvode.CV_RootFunction unavailable',
)
def test_rootfn_g_at_anchor_state_equals_cap():
    """At y=y0 the cap is fully available: g[0] == cap.

    The rootfn anchors at Φ_global(start) and reports
    g = cap − |Φ_global(t) − Φ_global(start)|. Evaluated at the
    initial state, ΔΦ_global = 0 so g equals cap exactly.
    """
    eos, P_stag, volume, n_stag, S_stag, phi0 = _build_synthetic_eos_and_state()
    cap = 0.07
    state_scale = np.ones(n_stag, dtype=float)
    rootfn = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=phi0,
        cap=cap,
        state_scale=state_scale,
    )
    g = np.zeros(1, dtype=float)
    rc = rootfn.evaluate(0.0, S_stag, g)
    assert rc == 0
    assert g[0] == pytest.approx(cap, abs=1e-12)
    assert rootfn.evals == 1


@pytest.mark.skipif(
    not _CV_ROOTFN_AVAILABLE,
    reason='scikits_odes_sundials.cvode.CV_RootFunction unavailable',
)
def test_rootfn_g_drops_with_phi_change():
    """g[0] decreases as |Φ_global| moves away from anchor.

    Mock the EOS to return a *different* phi after the first call so
    the second evaluate() sees ΔΦ_global = 0.04. With cap = 0.07
    the rootfn returns g[0] = 0.07 − 0.04 = 0.03.
    """
    eos, P_stag, volume, n_stag, S_stag, phi0 = _build_synthetic_eos_and_state(
        phi_uniform=0.60,
    )
    cap = 0.07
    rootfn = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=phi0,
        cap=cap,
        state_scale=np.ones(n_stag),
    )
    # First call at anchor (phi=0.60): g = 0.07
    g = np.zeros(1)
    rootfn.evaluate(0.0, S_stag, g)
    assert g[0] == pytest.approx(cap, abs=1e-12)
    # Second call: EOS now returns phi=0.56 (Phi dropped 0.04)
    eos.melt_fraction.return_value = np.full(n_stag, 0.56)
    g2 = np.zeros(1)
    rootfn.evaluate(0.0, S_stag, g2)
    assert g2[0] == pytest.approx(0.03, abs=1e-12)


@pytest.mark.skipif(
    not _CV_ROOTFN_AVAILABLE,
    reason='scikits_odes_sundials.cvode.CV_RootFunction unavailable',
)
def test_rootfn_fires_when_change_exceeds_cap():
    """g[0] crosses zero (i.e. goes negative) when |ΔΦ| > cap.

    CVODE detects sign changes; when g[0] crosses from positive to
    negative the integrator returns at the crossing.
    """
    eos, P_stag, volume, n_stag, S_stag, phi0 = _build_synthetic_eos_and_state(
        phi_uniform=0.60,
    )
    cap = 0.05
    rootfn = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=phi0,
        cap=cap,
        state_scale=np.ones(n_stag),
    )
    # Anchor: g = +cap
    g = np.zeros(1)
    rootfn.evaluate(0.0, S_stag, g)
    assert g[0] > 0.0
    # Move past cap: phi_global = 0.50, ΔΦ = 0.10 > cap
    eos.melt_fraction.return_value = np.full(n_stag, 0.50)
    g2 = np.zeros(1)
    rootfn.evaluate(0.0, S_stag, g2)
    assert g2[0] < 0.0
    assert g2[0] == pytest.approx(cap - 0.10, abs=1e-12)


@pytest.mark.skipif(
    not _CV_ROOTFN_AVAILABLE,
    reason='scikits_odes_sundials.cvode.CV_RootFunction unavailable',
)
def test_rootfn_uses_signed_cap_symmetric_around_anchor():
    """Cap is on |ΔΦ_global|; both directions are bounded equally.

    A magma ocean cools so Φ_global decreases with time; but a
    transient warming of the mushy zone could push Φ_global above
    the anchor. The rootfn must trip in either direction.
    """
    eos, P_stag, volume, n_stag, S_stag, phi0 = _build_synthetic_eos_and_state(
        phi_uniform=0.50,
    )
    cap = 0.05
    rootfn = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=phi0,
        cap=cap,
        state_scale=np.ones(n_stag),
    )
    # Direction 1: Φ_global drops by 0.06 → g = -0.01
    eos.melt_fraction.return_value = np.full(n_stag, 0.44)
    g_down = np.zeros(1)
    rootfn.evaluate(0.0, S_stag, g_down)
    assert g_down[0] == pytest.approx(-0.01, abs=1e-12)
    # Direction 2: Φ_global rises by 0.06 → g = -0.01 (same magnitude)
    eos.melt_fraction.return_value = np.full(n_stag, 0.56)
    g_up = np.zeros(1)
    rootfn.evaluate(0.0, S_stag, g_up)
    assert g_up[0] == pytest.approx(-0.01, abs=1e-12)


@pytest.mark.skipif(
    not _CV_ROOTFN_AVAILABLE,
    reason='scikits_odes_sundials.cvode.CV_RootFunction unavailable',
)
def test_rootfn_rescales_nondim_y():
    """The rootfn must convert nondim y → physical entropy via state_scale.

    CVODE passes nondim y; the EOS expects physical S. A state_scale
    of S_ref means y_phys = y_nd × S_ref. Verify the EOS is called
    with the scaled values.
    """
    eos, P_stag, volume, n_stag, S_stag_phys, phi0 = _build_synthetic_eos_and_state()
    cap = 0.05
    S_ref = 2993.025
    state_scale = np.full(n_stag, S_ref)
    # Nondim S is S_phys / S_ref
    S_stag_nd = S_stag_phys / S_ref
    rootfn = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=phi0,
        cap=cap,
        state_scale=state_scale,
    )
    g = np.zeros(1)
    rootfn.evaluate(0.0, S_stag_nd, g)
    # The EOS must have been called with PHYSICAL S, not nondim
    args = eos.density.call_args[0]
    np.testing.assert_allclose(args[1], S_stag_phys)


def test_rootfn_evaluate_swallows_eos_exceptions():
    """Exceptions in EOS lookup must NOT crash CVODE.

    Returning a positive g[0] keeps CVODE integrating; the cap will
    not fire if the EOS lookup fails, but the solve continues.
    """
    if not _CV_ROOTFN_AVAILABLE:
        pytest.skip('CV_RootFunction unavailable')
    eos = MagicMock()
    eos.density.side_effect = RuntimeError('synthetic EOS failure')
    cap = 0.05
    rootfn = _PhiCapRootFunction(
        eos=eos,
        P_stag=np.linspace(1e9, 1e11, 8),
        volume=np.ones(8),
        n_stag=8,
        phi0_global=0.5,
        cap=cap,
        state_scale=np.ones(8),
    )
    g = np.zeros(1)
    rc = rootfn.evaluate(0.0, np.full(8, 3000.0), g)
    assert rc == 0
    assert g[0] == pytest.approx(cap)  # safe fallback: cap stays available


# ──────────────────────────────────────────────────────────────────────
#               _phi_cap_event_factory (scipy fallback)
# ──────────────────────────────────────────────────────────────────────


def test_event_factory_returns_terminal_callable():
    """The event must have ``terminal=True`` and ``direction=-1``.

    solve_ivp uses these attributes to know that the event ends the
    integration and only fires on positive-to-negative crossings.
    """
    eos, P_stag, volume, n_stag, _, phi0 = _build_synthetic_eos_and_state()
    event = _phi_cap_event_factory(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=phi0,
        cap=0.05,
        state_scale=np.ones(n_stag),
    )
    assert event.terminal is True
    assert event.direction == pytest.approx(-1.0)


def test_event_factory_value_matches_rootfn():
    """The scipy event must return the same g[0] the CVODE rootfn does.

    Both must compute mass-weighted |ΔΦ_global| relative to the
    anchor and subtract from the cap. Verifying parity prevents the
    two paths from drifting silently.
    """
    eos, P_stag, volume, n_stag, S_stag, phi0 = _build_synthetic_eos_and_state(
        phi_uniform=0.60,
    )
    cap = 0.05
    state_scale = np.ones(n_stag)
    event = _phi_cap_event_factory(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=phi0,
        cap=cap,
        state_scale=state_scale,
    )
    # At anchor
    val_anchor = event(0.0, S_stag)
    assert val_anchor == pytest.approx(cap)
    # Move ΔΦ = -0.04
    eos.melt_fraction.return_value = np.full(n_stag, 0.56)
    val_moved = event(1.0, S_stag)
    assert val_moved == pytest.approx(0.01, abs=1e-12)


# ──────────────────────────────────────────────────────────────────────
#                        Source-pattern regressions
# ──────────────────────────────────────────────────────────────────────


def test_source_defines_rootfn_class():
    """The rootfn class must be a SUNDIALS CV_RootFunction subclass.

    Using a plain function would skip CVODE's optimised root-finding
    machinery; subclassing the typed class is the documented path.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    assert 'class _PhiCapRootFunction' in src
    assert '_CV_RootFunction' in src
    # The fallback when scikits.odes is unavailable must keep imports
    # working (subclass falls back to ``object``).
    assert '_CV_RootFunction = object' in src


def test_source_anchors_cap_to_phi_global_at_start():
    """The cap must anchor to Φ_global at solve entry (mass-weighted).

    Using cell-mean would put the cap on a metric different from
    the helpfile's Phi_global, breaking PROTEUS-side bookkeeping.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    assert 'phi0_global' in src
    assert 'self.entropy_eos.melt_fraction' in src
    assert 'self.entropy_eos.density' in src
    # The anchor is built ONCE per solve before the rootfn is wired,
    # not re-evaluated inside the rootfn.
    assert 'phi_cap_anchor' in src


def test_source_wires_rootfn_into_cvode_options():
    """``_solve_cvode`` must register the rootfn with nr_rootfns=1."""
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    assert "cvode_options['rootfn']" in src
    assert "cvode_options['nr_rootfns'] = 1" in src
    # The cap must be gated; the rootfn is only installed when armed.
    m = re.search(
        r"if phi_cap_rootfn is not None:\s*\n\s*cvode_options\['rootfn'\]",
        src,
    )
    assert m is not None, 'rootfn wiring is not gated on phi_cap_rootfn'


def test_source_passes_event_to_scipy_fallback():
    """The scipy fallback path must receive the cap event via events=."""
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # solve_ivp must be called with the events list when the cap is armed.
    assert '_phi_cap_event_factory' in src
    # Must terminate on the event (terminal=True is set by the factory).
    assert 'event.terminal = True' in src


def test_source_logs_rootfn_fire_on_flag_2():
    """When CVODE returns flag=2 (CV_ROOT_RETURN), the cap-fire is logged.

    The log line is the only operator-visible signal that the cap
    truncated the call; without it, debugging a regression where the
    cap fires too aggressively requires re-running with debug logs.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # The log message must mention 'rootfn fired' (specific phrase that
    # search-and-rescue scripts can grep for).
    assert 'rootfn fired' in src
    # The cap path must mark the result inside _solve_cvode so solve()
    # can log with PHYSICAL time (not nondim).
    assert 'flag == 2 and phi_cap_rootfn is not None' in src
    assert 'result.cap_fired = True' in src


def test_source_logs_cap_fire_in_physical_time():
    """The cap-fire log must use physical time, not nondim.

    The CVODE integrator runs in nondim units; the result wrapper
    rescales ``sol.t`` to physical years in ``solve()`` after
    ``_solve_cvode`` returns. The cap-fire log must be emitted AFTER
    that rescale, otherwise operators reading the log see a nondim
    time stamp instead of physical years.

    This regression locks the log emission to ``solve()``, gated on
    a marker attached to the result inside ``_solve_cvode``, so the
    nondim/physical translation can never drift again.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # The marker lives on the result object.
    assert "getattr(sol, 'cap_fired'" in src
    # The log uses ``sol.t[-1]`` AFTER the ``sol.t * t_ref`` rescale.
    rescale_idx = src.find('sol.t = np.asarray(sol.t, dtype=float) * t_ref')
    log_idx = src.find("'ΔΦ_global cap: CVODE rootfn fired")
    assert rescale_idx > 0, 't_ref rescale not found'
    assert log_idx > 0, 'physical-time cap-fire log not found'
    assert log_idx > rescale_idx, 'cap-fire log must come AFTER the nondim->physical rescale'
    # The scipy fallback path also rescales its event time via t_ref
    # (the event captures nondim t; physical = t × t_ref).
    assert 'sol.t_events[0][0]' in src
    assert 'event fired at' in src


def test_source_subclasses_cv_rootfunction():
    """``_PhiCapRootFunction`` must inherit from ``_CV_RootFunction``."""
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    m = re.search(
        r'class\s+_PhiCapRootFunction\s*\(\s*_CV_RootFunction\s*\)',
        src,
    )
    assert m is not None, '_PhiCapRootFunction must subclass _CV_RootFunction'


def test_source_phi_global_is_mass_weighted_in_get_state():
    """SolverOutput.Phi_global must use mass weighting, not volume.

    Volume-weighting silently froze Phi_global in mass-coordinate
    meshes because deep high-density cells have small volumes and
    contribute negligibly; the surface stays liquid longest and
    dominates the volume-weighted average. PROTEUS's helpfile
    Phi_global (used by stop criteria, structure-update triggers,
    deadlock detection) then never moved during bottom-up
    crystallisation. Verified 2026-05-02 in
    output/verify_dilon_phicap005.v3_helpfile_frozen.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # Locate the get_state() method.
    m = re.search(r'def get_state\(self\)[\s\S]+?return SolverOutput', src)
    assert m is not None, 'get_state method not found'
    block = m.group(0)
    # Phi_global must be assembled from mass_stag, not vol.
    assert 'phi_stag * mass_stag' in block
    # The volume-weighted formula must NOT remain.
    assert 'np.dot(phi_stag, vol) / np.sum(vol)' not in block


def test_source_phi_global_matches_rootfn_formula():
    """SolverOutput.Phi_global and rootfn.evaluate use the same metric.

    If they diverge, the rootfn fires at one Phi_global definition
    while PROTEUS reports a different one, leading to bookkeeping
    inconsistency. Both must use mass weighting so the cap and
    PROTEUS stop logic see the same state.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # Both compute Σ(mass * phi) / Σ(mass) over staggered cells.
    rootfn_block = re.search(
        r'class _PhiCapRootFunction[\s\S]+?(?=\n\ndef |\nclass )',
        src,
    )
    get_state_block = re.search(
        r'def get_state\(self\)[\s\S]+?return SolverOutput',
        src,
    )
    assert rootfn_block is not None and get_state_block is not None
    # Both blocks reference mass-weighting of phi.
    assert 'mass * phi' in rootfn_block.group(0).replace(
        ' ', ''
    ) or 'mass*phi' in rootfn_block.group(0).replace(' ', '')
    assert 'phi_stag*mass_stag' in get_state_block.group(0).replace(' ', '')


def test_solver_output_docstring_says_mass_weighted():
    """The SolverOutput field docstring must reflect the new semantics.

    Doc drift would mislead future contributors into computing
    Phi_global from the wrong field formula again.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # Find the SolverOutput dataclass body.
    m = re.search(r'class SolverOutput[\s\S]+?(?=\n\nclass )', src)
    assert m is not None
    body = m.group(0)
    # The Phi_global field comment must say mass-weighted.
    assert 'mass-weighted melt fraction' in body
    # And no longer say volume-weighted.
    assert 'volume-weighted melt fraction' not in body


def test_source_no_dt_safe_estimate_remaining():
    """The v1/v2 dt_safe estimate must be gone.

    Both v1 (per-cell max) and v2 (mass-weighted) wedged because the
    rate at solve entry was ~30000× higher than the achieved rate.
    The v3 implementation must not reintroduce that estimate.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # Specific names that v1/v2 used; if any remain we have not
    # fully migrated.
    assert 'dphi_dt_max' not in src
    assert 'dphi_dt_global' not in src
    assert 'dt_safe' not in src
    assert 'safety = 0.5' not in src


def test_source_solve_cvode_uses_cvode_roots_on_flag_2():
    """``_solve_cvode`` must read ``cvode_sol.roots.t`` / ``roots.y`` when
    flag==2, NOT ``cvode_sol.values.t`` (which scikits.odes reduces to
    just ``[start_time]`` on rootfn fire).

    Verified 2026-05-03 with a synthetic test (y'=1, root at y=5):
    ``flag=2``, ``values.t=[0.]``, but ``roots.t=[5.]``. Reading from
    ``values.t`` makes ``dt_actual = sol.t[-1] - sol.t[0] = 0``,
    causing PROTEUS's wrapper to fall back to ``dtswitch`` while
    Aragog's state had advanced to the root, locking the coupled run
    at a fixed point (observed v3.2 / v3.3 in
    output/verify_dilon_phicap005).
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # The fix must reference cvode_sol.roots and gate on flag == 2.
    assert "getattr(cvode_sol, 'roots'" in src or 'cvode_sol.roots' in src, (
        'cvode_sol.roots not consulted; rootfn-fire path will lose '
        't_root and dt_actual will silently zero out.'
    )
    # And the result.t must be built as [start_time, t_root] when roots
    # are available.
    assert 'np.array([float(start_time), t_root]' in src, (
        'result.t is not built as [start_time, t_root] from cvode_sol.roots; '
        "downstream get_state's dt_actual computation will be wrong."
    )


def test_solve_cvode_uses_cvode_roots_when_flag_2():
    """Behavioural regression: build a mock cvode_sol with flag=2 and
    roots.t=[t_root], and verify the returned ``result.t = [start, t_root]``,
    ``result.y`` has 2 columns (y0, y_root).
    """
    from unittest.mock import patch

    from aragog.solver.entropy_solver import EntropySolver

    n_state = 5
    # values.t reduced to [start] (the scikits.odes idiom on rootfn fire)
    mock_values = MagicMock()
    mock_values.t = np.array([0.1])
    mock_values.y = np.zeros((1, n_state))

    # roots holds the actual fire location
    mock_roots = MagicMock()
    mock_roots.t = np.array([0.4])
    y_root_row = np.linspace(0.5, 1.5, n_state)
    mock_roots.y = y_root_row.reshape(1, n_state)

    mock_cvode_sol = MagicMock()
    mock_cvode_sol.values = mock_values
    mock_cvode_sol.roots = mock_roots
    mock_cvode_sol.flag = 2
    mock_cvode_sol.message = 'root found'

    mock_solver = MagicMock()
    mock_solver.solve.return_value = mock_cvode_sol
    mock_solver._integrator.get_info.return_value = {}

    start_time = 0.1
    end_time = 1.0
    y0 = np.linspace(0.0, 1.0, n_state)
    instance = MagicMock(spec=EntropySolver)
    instance.dSdt = lambda t, y: np.zeros_like(y)
    instance._core_bc = 'energy_balance'

    fake_rootfn = MagicMock()
    fake_rootfn.evals = 7
    fake_rootfn.cap = 0.05
    fake_rootfn.phi0 = 0.88

    with patch('aragog.solver.entropy_solver._scikits_cvode', return_value=mock_solver):
        result = EntropySolver._solve_cvode(
            instance,
            start_time=start_time,
            end_time=end_time,
            y0=y0,
            atol=1e-8,
            rtol=1e-8,
            max_step=np.inf,
            phi_cap_rootfn=fake_rootfn,
        )

    # Two-point trajectory: [start, t_root] sourced from cvode_sol.roots
    assert result.t.size == 2, f'expected 2 time points, got {result.t.size}'
    assert float(result.t[0]) == pytest.approx(start_time)
    assert float(result.t[1]) == pytest.approx(0.4)
    assert result.y.shape == (n_state, 2)
    np.testing.assert_array_equal(result.y[:, 0], y0)
    np.testing.assert_array_equal(result.y[:, 1], y_root_row)
    # dt_actual must be positive (the bug symptom: this used to be 0).
    dt_actual = float(result.t[-1] - result.t[0])
    assert dt_actual > 0.0, 'dt_actual still 0; cvode_sol.roots not used correctly'
    assert dt_actual == pytest.approx(0.4 - start_time)
    assert getattr(result, 'cap_fired', False) is True
    assert int(result.cap_evals) == 7


def test_solve_cvode_falls_back_to_values_when_no_roots_on_flag_2():
    """If flag=2 but ``roots`` attribute is missing or empty, the code
    must gracefully fall back to ``values.t`` rather than crashing.

    Defensive against scikits.odes versions that may behave differently.
    """
    from unittest.mock import patch

    from aragog.solver.entropy_solver import EntropySolver

    n_state = 3
    mock_values = MagicMock()
    mock_values.t = np.array([0.1, 0.55])
    mock_values.y = np.zeros((2, n_state))

    # roots is None (older scikits.odes or different solver)
    mock_cvode_sol = MagicMock()
    mock_cvode_sol.values = mock_values
    mock_cvode_sol.roots = None
    mock_cvode_sol.flag = 2
    mock_cvode_sol.message = ''

    mock_solver = MagicMock()
    mock_solver.solve.return_value = mock_cvode_sol
    mock_solver._integrator.get_info.return_value = {}

    instance = MagicMock(spec=EntropySolver)
    instance.dSdt = lambda t, y: np.zeros_like(y)
    instance._core_bc = 'energy_balance'

    fake_rootfn = MagicMock()
    fake_rootfn.evals = 1
    fake_rootfn.cap = 0.05
    fake_rootfn.phi0 = 0.88

    with patch('aragog.solver.entropy_solver._scikits_cvode', return_value=mock_solver):
        result = EntropySolver._solve_cvode(
            instance,
            start_time=0.1,
            end_time=1.0,
            y0=np.zeros(n_state),
            atol=1e-8,
            rtol=1e-8,
            max_step=np.inf,
            phi_cap_rootfn=fake_rootfn,
        )

    # Falls back to values.t
    assert result.t.size == 2
    np.testing.assert_allclose(result.t, [0.1, 0.55])


def test_solve_cvode_uses_values_on_normal_completion_flag_0():
    """On normal flag=0 completion, ``_solve_cvode`` must read
    ``cvode_sol.values.t`` (which holds the full ``[start, end]``
    trajectory). The roots-attribute branch must NOT activate when
    flag != 2.
    """
    from unittest.mock import patch

    from aragog.solver.entropy_solver import EntropySolver

    n_state = 4
    y_full = np.array([[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]])
    mock_values = MagicMock()
    mock_values.t = np.array([0.1, 1.0])
    mock_values.y = y_full

    # roots populated but flag=0 — must be ignored.
    mock_roots = MagicMock()
    mock_roots.t = np.array([0.5])
    mock_roots.y = np.zeros((1, n_state))

    mock_cvode_sol = MagicMock()
    mock_cvode_sol.values = mock_values
    mock_cvode_sol.roots = mock_roots
    mock_cvode_sol.flag = 0
    mock_cvode_sol.message = ''

    mock_solver = MagicMock()
    mock_solver.solve.return_value = mock_cvode_sol
    mock_solver._integrator.get_info.return_value = {}

    instance = MagicMock(spec=EntropySolver)
    instance.dSdt = lambda t, y: np.zeros_like(y)
    instance._core_bc = 'energy_balance'

    with patch('aragog.solver.entropy_solver._scikits_cvode', return_value=mock_solver):
        result = EntropySolver._solve_cvode(
            instance,
            start_time=0.1,
            end_time=1.0,
            y0=np.array([0.0, 1.0, 2.0, 3.0]),
            atol=1e-8,
            rtol=1e-8,
            max_step=np.inf,
            phi_cap_rootfn=None,
        )

    # values path used, NOT roots
    assert result.t.size == 2
    assert result.y.shape == (n_state, 2)
    np.testing.assert_allclose(result.t, [0.1, 1.0])


# ──────────────────────────────────────────────────────────────────────
#               Per-cell melt-fraction guard (cliff onset)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not _CV_ROOTFN_AVAILABLE,
    reason='scikits_odes_sundials.cvode.CV_RootFunction unavailable',
)
def test_rootfn_per_cell_catches_single_deep_cell_crossing():
    """A single deep cell crossing the window fires the cap; the global
    mean alone would miss it.

    Crystallisation onset has the bottom cell go fully molten to fully
    solid in one step while every other cell stays molten. With uniform
    density and unit volume the mass-weighted global melt fraction moves
    by only ``1/N`` (here 1/32 = 0.031 vs cap 0.05), which on its own is
    below the cap, while the per-cell guard sees a unit swing. The
    discrimination guard asserts the two verdicts differ by far more than
    rounding, so the test fails if the per-cell term is dropped.
    """
    n_stag = 32
    eos, P_stag, volume, n_stag, S_stag, _ = _build_synthetic_eos_and_state(
        n_stag=n_stag, phi_uniform=1.0
    )
    cap = 0.05
    phi0_cell = np.full(n_stag, 1.0)
    phi_post = np.full(n_stag, 1.0)
    phi_post[0] = 0.0  # bottom cell crystallises completely
    eos.melt_fraction.return_value = phi_post
    global_dphi = 1.0 / n_stag
    assert global_dphi < cap  # the cliff is invisible to a global-only cap

    rootfn_pc = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=1.0,
        cap=cap,
        state_scale=np.ones(n_stag),
        phi0_per_cell=phi0_cell,
    )
    g_pc = np.zeros(1)
    rootfn_pc.evaluate(1.0, S_stag, g_pc)
    assert g_pc[0] < 0.0  # per-cell guard fires (g crosses zero)

    rootfn_global = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=1.0,
        cap=cap,
        state_scale=np.ones(n_stag),
        phi0_per_cell=None,
    )
    g_global = np.zeros(1)
    rootfn_global.evaluate(1.0, S_stag, g_global)
    assert g_global[0] > 0.0  # global-only does not fire on the same state
    # The per-cell verdict must be decisively different, not a near-tie.
    assert g_global[0] - g_pc[0] > 0.5


def test_event_factory_per_cell_mirrors_rootfn():
    """The scipy fallback event mirrors the rootfn per-cell behaviour.

    Same single-deep-cell crossing: the per-cell event value is negative
    (fires) while the global-only event stays positive (does not). The two
    must straddle zero, so a regression that drops the per-cell term in the
    event factory is caught even when CVODE is unavailable.
    """
    n_stag = 32
    eos, P_stag, volume, n_stag, S_stag, _ = _build_synthetic_eos_and_state(
        n_stag=n_stag, phi_uniform=1.0
    )
    cap = 0.05
    phi_post = np.full(n_stag, 1.0)
    phi_post[0] = 0.0
    eos.melt_fraction.return_value = phi_post

    ev_pc = _phi_cap_event_factory(
        eos,
        P_stag,
        volume,
        n_stag,
        1.0,
        cap,
        np.ones(n_stag),
        phi0_per_cell=np.full(n_stag, 1.0),
    )
    ev_global = _phi_cap_event_factory(
        eos, P_stag, volume, n_stag, 1.0, cap, np.ones(n_stag), phi0_per_cell=None
    )
    val_pc = ev_pc(1.0, S_stag)
    val_global = ev_global(1.0, S_stag)
    assert val_pc < 0.0 < val_global
    assert val_global - val_pc > 0.5


# ──────────────────────────────────────────────────────────────────────
#               Per-cell temperature and entropy step caps
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not _CV_ROOTFN_AVAILABLE,
    reason='scikits_odes_sundials.cvode.CV_RootFunction unavailable',
)
def test_rootfn_temperature_cap_catches_jump_phi_cap_misses():
    """The temperature cap fires on a per-cell |ΔT| even when melt fraction
    barely moves.

    This is the solid-adiabat blind spot: a fully solid cell (phi pinned at 0)
    keeps cooling, so the melt-fraction cap sees no change while the reported
    temperature drops sharply. With the melt-fraction cap disabled (cap=0) and
    only the temperature cap armed, a 500 K drop must drive g negative; with
    the temperature cap also disabled the same state must NOT fire. The two
    verdicts straddle zero, so dropping the temperature term is caught.
    """
    n_stag = 8
    eos = MagicMock()
    P_stag = np.linspace(1.0e9, 1.4e11, n_stag)
    volume = np.full(n_stag, 1.0e18)
    S_stag = np.full(n_stag, 3000.0)
    T0 = np.full(n_stag, 6000.0)
    eos.temperature.return_value = T0 - 500.0  # 500 K drop, no phi change

    rootfn_T = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=0.0,
        cap=0.0,
        state_scale=np.ones(n_stag),
        cap_temperature=200.0,
        T0_per_cell=T0,
    )
    gT = np.zeros(1)
    rootfn_T.evaluate(1.0, S_stag, gT)
    assert gT[0] < 0.0  # 200 - 500 = -300 -> fires

    rootfn_off = _PhiCapRootFunction(
        eos=eos,
        P_stag=P_stag,
        volume=volume,
        n_stag=n_stag,
        phi0_global=0.0,
        cap=0.0,
        state_scale=np.ones(n_stag),
        cap_temperature=0.0,
        T0_per_cell=T0,
    )
    goff = np.zeros(1)
    rootfn_off.evaluate(1.0, S_stag, goff)
    assert goff[0] > 0.0  # no cap armed -> never fires
    assert goff[0] - gT[0] > 0.5


def test_event_factory_entropy_cap_catches_native_variable_jump():
    """The entropy cap fires on a per-cell |ΔS| with no EOS lookup.

    S is the native solver variable, so the entropy cap reads y*scale directly.
    A 200 J/kg/K jump with cap_entropy=50 fires (event value negative); with the
    cap disabled it does not. The scipy event mirrors the CVODE rootfn, so this
    guards the fallback path. Straddle-zero discrimination as above.
    """
    n_stag = 6
    eos = MagicMock()
    S0 = np.full(n_stag, 2500.0)
    # y is nondim; with unit state_scale, S = y. Jump the entropy by 200.
    y_jumped = S0 + 200.0

    ev_S = _phi_cap_event_factory(
        eos,
        np.full(n_stag, 5.0e10),
        np.full(n_stag, 1.0e18),
        n_stag,
        0.0,
        0.0,
        np.ones(n_stag),
        cap_entropy=50.0,
        S0_per_cell=S0,
    )
    ev_off = _phi_cap_event_factory(
        eos,
        np.full(n_stag, 5.0e10),
        np.full(n_stag, 1.0e18),
        n_stag,
        0.0,
        0.0,
        np.ones(n_stag),
        cap_entropy=0.0,
        S0_per_cell=S0,
    )
    val_S = ev_S(1.0, y_jumped)
    val_off = ev_off(1.0, y_jumped)
    assert val_S < 0.0 < val_off
    assert val_off - val_S > 0.5
    # The entropy cap must not have called the EOS (native-variable path).
    assert not eos.temperature.called
    assert not eos.density.called
