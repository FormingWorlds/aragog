"""Strategy B v3 tests: SUNDIALS root function for ΔΦ_global cap.

The cap is implemented as a CVODE root function (and equivalent
``solve_ivp`` event for the scipy fallback). When armed, the
integrator returns at the exact time t* where the mass-weighted
|Φ_global(t*) − Φ_global(start)| crosses the configured cap. There
is no rate estimation: the integrator's own trajectory determines
termination, eliminating the start-time-rate overshoot that wedged
the v1 (per-cell max) and v2 (mass-weighted dt estimate)
formulations at the 1 M_E PALEOS-2phase rheological transition
(verified 2026-05-02 in output/verify_dilon_phicap005).

Tests cover:
- ``_EnergyParameters.phi_step_cap`` default and validation
- ``_PhiCapRootFunction.evaluate`` returns g[0] = cap − |ΔΦ_global|
- The rootfn fires (g[0] crosses zero) at the correct ΔΦ_global
- The event factory (scipy fallback) returns the same value as
  the CVODE rootfn at matching states
- Source-pattern regression: the rootfn class subclasses
  ``CV_RootFunction``; ``solve()`` wires it into ``cvode_options``
  with ``nr_rootfns=1``; ``_solve_cvode`` accepts ``phi_cap_rootfn``
- Default 0.0 keeps existing behaviour: rootfn is never installed.
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
    _phi_cap_event_factory,
    _PhiCapRootFunction,
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


def test_phi_step_cap_default_is_zero():
    """Default phi_step_cap must be 0.0 so existing tests are unchanged.

    A positive default would activate the rootfn on every solve()
    call in production, which is a behavioural change and would
    break the bit-parity contract with prior runs.
    """
    e = _make_energy()
    assert e.phi_step_cap == 0.0
    assert isinstance(e.phi_step_cap, float)


def test_phi_step_cap_accepts_positive_values():
    """Recommended production setting (0.05) and an extreme (0.5)."""
    assert _make_energy(phi_step_cap=0.05).phi_step_cap == pytest.approx(0.05)
    assert _make_energy(phi_step_cap=0.5).phi_step_cap == pytest.approx(0.5)


def test_phi_step_cap_accepts_zero_explicitly():
    """Explicit 0.0 (disabled) is a valid configuration."""
    assert _make_energy(phi_step_cap=0.0).phi_step_cap == 0.0


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
    that rescale, otherwise operators see misleading ~1e7 yr values
    (verified 2026-05-02 in the v3 validation run; nondim t × t_ref
    ≈ 3.17e-3 yr converts back to physical).

    This regression locks the log emission to ``solve()``, gated on
    a marker attached to the result inside ``_solve_cvode``, so the
    nondim/physical translation can never drift again.
    """
    src = (REPO_ROOT / 'solver' / 'entropy_solver.py').read_text()
    # The marker lives on the result object.
    assert "getattr(sol, 'cap_fired'" in src
    # The log uses ``sol.t[-1]`` AFTER the ``sol.t * t_ref`` rescale.
    rescale_idx = src.find('sol.t = np.asarray(sol.t, dtype=float) * t_ref')
    log_idx = src.find("'ΔΦ_global cap (Strategy B v3): CVODE rootfn fired")
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

    with patch('aragog.solver.entropy_solver._scikits_ode', return_value=mock_solver):
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

    with patch('aragog.solver.entropy_solver._scikits_ode', return_value=mock_solver):
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

    with patch('aragog.solver.entropy_solver._scikits_ode', return_value=mock_solver):
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
