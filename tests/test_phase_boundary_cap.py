"""Tests for ``energy.phase_boundary_cap``: the fixed 1 yr step and the rate-based step near a phase boundary."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aragog.parser import _EnergyParameters
from aragog.solver.entropy_solver import (
    EntropySolver,
    _PhaseBoundarySegmentRoot,
    _rate_phase_boundary_max_step,
)

from .test_phi_step_cap_armed_smoke import _build_mushy_parameters, _pick_mushy_S, needs_eos

S_SOL, S_LIQ, DELTA = 1000.0, 1300.0, 10.0
WIDE = (0.0, 1e9)


def _cap(S, dSdt, mass=None, **kw):
    n = len(S)
    kw.setdefault('rate_floor', False)
    return _rate_phase_boundary_max_step(
        np.asarray(S, float),
        np.asarray(dSdt, float),
        np.full(n, S_LIQ),
        np.full(n, S_SOL),
        np.ones(n) if mass is None else np.asarray(mass, float),
        DELTA,
        **kw,
    )


@pytest.mark.unit
def test_rate_cap_uses_next_boundary_in_the_direction_of_motion():
    """Cooling above the liquidus aims at the liquidus, inside the band at the solidus, heating at the liquidus."""
    assert _cap([1350.0], [-10.0], fraction=1.0, bounds=WIDE) == pytest.approx(5.0)
    assert _cap([1200.0], [-1.0], fraction=1.0, bounds=WIDE) == pytest.approx(200.0)
    assert _cap([1100.0], [2.0], fraction=1.0, bounds=WIDE) == pytest.approx(100.0)
    assert _cap([950.0], [0.5], fraction=1.0, bounds=WIDE) == pytest.approx(100.0)
    assert _cap([1350.0, 1200.0], [-10.0, -1.0], fraction=1.0, bounds=WIDE) == pytest.approx(
        5.0
    )


@pytest.mark.unit
def test_rate_cap_counts_far_cells_and_ignores_cells_without_a_boundary_ahead():
    """A far fast cell sets the cap; below the solidus cooling, above the liquidus heating, or dS/dt = 0 do not."""
    assert _cap([1510.0], [-50.0], fraction=1.0, bounds=WIDE) == pytest.approx(4.2)
    assert _cap([950.0, 1350.0, 1200.0], [-5.0, 5.0, 0.0]) == 100.0


@pytest.mark.unit
def test_rate_cap_stiff_zone_sets_the_lower_bound_in_any_direction():
    """Fast motion within delta of a boundary hits the lower bound in either direction."""
    assert _cap([1305.0], [5.0]) == 1.0
    assert _cap([995.0], [-5.0]) == 1.0
    assert _cap([1315.0], [5.0]) == 100.0


@pytest.mark.unit
def test_rate_cap_stiff_zone_scales_with_rate_for_slow_cells():
    """Inside the stiff zone a slow cell uses d_near / r_used and is not clamped to 1 yr."""
    assert _cap([1305.0], [0.01], fraction=0.1, bounds=(1.0, 100.0)) == pytest.approx(50.0)
    assert _cap([995.0], [-0.01], fraction=0.1, bounds=(1.0, 100.0)) == pytest.approx(50.0)


@pytest.mark.unit
def test_rate_cap_floor_uses_the_mass_weighted_rate_and_the_nearer_boundary():
    """A cell slower than the mass-weighted mean |dS/dt| uses that rate toward its nearer boundary."""
    S, dSdt, mass = [1200.0, 3000.0], [1e-8, -100.0], [9.0, 1.0]
    floor = (9.0 * 1e-8 + 100.0) / 10.0
    kw = dict(fraction=1.0, bounds=WIDE)
    assert _cap(S, dSdt, mass, rate_floor=True, **kw) == pytest.approx(100.0 / floor)
    assert _cap(S, dSdt, mass, **kw) == pytest.approx(17.0)
    assert _cap(S, dSdt, [9.0, 1.0], rate_floor=True, **kw) == _cap(
        S + [1200.0], dSdt + [1e-8], [9.0, 1.0, 0.0], rate_floor=True, **kw
    )


@pytest.mark.unit
def test_rate_cap_applies_fraction_and_clip():
    assert _cap([1200.0], [-1.0]) == pytest.approx(20.0)
    assert _cap([1350.0], [-10.0]) == 1.0
    assert _cap([1200.0], [-1e-3]) == 100.0


def _roots(S0, inside, cap=None):
    """Segment root on two cells (plus the CMB copy of cell 0), unit state scale."""
    return _PhaseBoundarySegmentRoot(
        np.full(3, S_LIQ),
        np.full(3, S_SOL),
        np.asarray(S0, float),
        DELTA,
        inside,
        np.ones(2),
        2,
        cap=cap,
    )


@pytest.mark.unit
def test_segment_root_is_anchored_at_the_segment_start_and_fires_on_progress():
    """At the anchor every component is positive; moving half the start distance fires 'progress'."""
    r = _roots([1200.0, 1500.0], inside=False)
    assert r.names == ['stiff', 'progress'] and r.n_roots == 2
    assert np.all(r.components(0.0, [1200.0, 1500.0]) > 0.0)
    moved = [1200.0 - 50.0, 1500.0]
    assert r.components(0.0, moved)[1] == pytest.approx(0.0)
    assert r.fired(0.0, moved) == 'progress'


@pytest.mark.unit
def test_segment_root_stiff_zone_entry_and_exit_with_hysteresis():
    """Outside, entry at delta fires 'stiff'; inside, the root sits at 2 delta and progress is off."""
    out = _roots([1350.0, 1500.0], inside=False)
    assert out.components(0.0, [1310.0, 1500.0])[0] == pytest.approx(0.0)
    assert out.fired(0.0, [1310.0, 1500.0]) == 'stiff'
    ins = _roots([1305.0, 1500.0], inside=True)
    assert ins.names == ['stiff']
    assert ins.components(0.0, [1305.0, 1500.0])[0] == pytest.approx(-15.0)
    assert ins.components(0.0, [1320.0, 1500.0])[0] == pytest.approx(0.0)


class _FakeCap:
    cap, cap_T, cap_S, phi0, evals, binding_cap = 0.05, 0.0, 0.0, 0.3, 7, 'phi'

    def __init__(self, g):
        self.g = g

    def evaluate(self, t, y, g, userdata=None):
        g[0] = self.g
        return 0


@pytest.mark.unit
def test_segment_root_puts_the_step_cap_first_and_forwards_its_attributes():
    """The step cap is component 0, names 'cap' when it is at zero, and its attributes reach _solve_cvode."""
    r = _roots([1200.0, 1500.0], inside=False, cap=_FakeCap(0.0))
    assert r.names == ['cap', 'stiff', 'progress'] and r.n_roots == 3
    assert r.fired(0.0, [1200.0, 1500.0]) == 'cap'
    assert (r.evals, r.binding_cap, r.phi0) == (7, 'phi', 0.3)
    bare = _roots([1200.0, 1500.0], inside=False)
    assert (bare.evals, bare.binding_cap, bare.cap, bare.phi0) == (0, None, 0.0, 0.0)
    with pytest.raises(AttributeError):
        bare.not_an_attribute


@pytest.mark.unit
def test_parser_rejects_unknown_phase_boundary_cap():
    kw = dict(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
    )
    assert _EnergyParameters(**kw).phase_boundary_cap == 'fixed'
    with pytest.raises(ValueError, match='phase_boundary_cap'):
        _EnergyParameters(**kw, phase_boundary_cap='adaptive')


def _solver(eos, mode, core_bc='quasi_steady', n_nodes=12, end_time=2.0, S=None, tol=None):
    p = _build_mushy_parameters(solver_method='cvode', n_nodes=n_nodes, end_time=end_time)
    p.energy = dataclasses.replace(p.energy, phase_boundary_cap=mode, phi_step_cap=None)
    p.boundary_conditions.core_bc = core_bc
    if tol is not None:
        p.solver.rtol = p.solver.atol = tol
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(_pick_mushy_S(eos) if S is None else np.full(s._n_stag, S))
    return s


@needs_eos
@pytest.mark.smoke
def test_fixed_and_gradient_keep_one_year_without_segments(shared_eos, monkeypatch):
    """'fixed' and the gradient core pass max_step 1 yr to one CVODE solve and never segment."""
    for mode, core_bc in (('fixed', 'quasi_steady'), ('rate', 'gradient')):
        s = _solver(shared_eos, mode, core_bc)
        seen, real = [], s._solve_cvode
        monkeypatch.setattr(
            s, '_solve_cvode', lambda **kw: seen.append(kw['max_step']) or real(**kw)
        )
        monkeypatch.setattr(s, '_solve_cvode_segments', None)
        s.solve()
        assert len(seen) == 1 and seen[0] * s._build_nondim_scales().t_ref == pytest.approx(1.0)


@needs_eos
@pytest.mark.smoke
def test_rate_mode_joins_segments_into_one_call_trajectory(shared_eos):
    """Segments cover the call; each segment start is a trajectory point and time strictly increases."""
    s = _solver(shared_eos, 'rate', end_time=200.0)
    s.solve()
    sol = s._solution
    starts = np.array([seg[0] for seg in sol.segments])
    assert len(starts) >= 2 and sol.t[0] == 0.0 and sol.t[-1] == pytest.approx(200.0)
    assert np.all(np.diff(sol.t) > 0.0) and sol.y.shape[1] == sol.t.size
    assert all(np.any(np.isclose(sol.t, t0, rtol=1e-12, atol=1e-9)) for t0 in starts)


def _isentropic_end(eos, mode, tol, core_bc='quasi_steady'):
    s = _solver(eos, mode, core_bc, n_nodes=24, end_time=200.0, S=8182.3, tol=tol)
    s.solve()
    S = s._solution.y[: s._n_stag, -1]
    return s, np.asarray(eos.temperature(s._P_stag_flat, S)).ravel(), S


@needs_eos
@pytest.mark.smoke
def test_rate_mode_from_an_isentropic_start_matches_a_tight_fixed_run(shared_eos):
    """Uniform start with most cells in or near the band: rate within 0.1 K of fixed at 1e-9, a stiff-zone segment, no empty segment."""
    _, T_ref, _ = _isentropic_end(shared_eos, 'fixed', 1e-9)
    s, T, _ = _isentropic_end(shared_eos, 'rate', 1e-6)
    assert np.abs(T - T_ref).max() <= 0.1
    starts = [seg[0] for seg in s._solution.segments]
    assert 'stiff' in [seg[1] for seg in s._solution.segments]
    assert np.all(np.diff(starts) > 1e-6)


@needs_eos
@pytest.mark.smoke
def test_rate_mode_keeps_the_energy_balance_state(shared_eos):
    """With the extended energy_balance state, rate mode segments on the entropy block and keeps the CMB entry."""
    s_f, T_f, _ = _isentropic_end(shared_eos, 'fixed', 1e-6, 'energy_balance')
    s_r, T_r, _ = _isentropic_end(shared_eos, 'rate', 1e-6, 'energy_balance')
    assert s_r._solution.y.shape[0] == s_f._solution.y.shape[0] == s_r._n_stag + 1
    assert len(s_r._solution.segments) >= 2 and np.isfinite(s_r._solution.y[-1, -1])
    assert np.abs(T_r - T_f).max() <= 0.1


@needs_eos
@pytest.mark.unit
def test_rate_cap_exception_fallback_returns_one_year(shared_eos, monkeypatch, caplog):
    """When dS/dt evaluation raises an exception during rate cap evaluation, fall back to 1 yr."""
    s = _solver(shared_eos, 'rate', end_time=10.0)

    calls = 0
    real_dSdt = s._dSdt_single

    def faulty_dSdt(t, y):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('simulated dS/dt failure')
        return real_dSdt(t, y)

    monkeypatch.setattr(s, '_dSdt_single', faulty_dSdt)
    with caplog.at_level('WARNING'):
        s.solve()
    assert any('rate cap: dS/dt evaluation failed' in rec.message for rec in caplog.records)
    assert s._solution.segments[0][2] == pytest.approx(1.0)


@needs_eos
@pytest.mark.unit
def test_rate_cap_segment_ceiling_falls_back_to_one_year(shared_eos, monkeypatch, caplog):
    """Reaching the segment count ceiling warns and falls back to 1 yr for the remaining call."""
    s = _solver(shared_eos, 'rate', end_time=200.0)
    orig_segments = s._solve_cvode_segments

    def capped_segments(**kw):
        return orig_segments(**{**kw, 'max_segments': 1})

    monkeypatch.setattr(s, '_solve_cvode_segments', capped_segments)
    with caplog.at_level('WARNING'):
        s.solve()
    assert any(
        'segments in one call; max_step 1 yr for the rest' in rec.message
        for rec in caplog.records
    )
    assert len(s._solution.segments) == 2
    assert s._solution.segments[-1][2] == pytest.approx(1.0)


@pytest.mark.unit
def test_rate_cap_slow_cell_moving_away_uses_nearer_boundary_when_floored():
    """A slow cell whose dS/dt moves away from boundary uses the nearer boundary when below floor."""
    res = _rate_phase_boundary_max_step(
        np.array([1350.0, 1500.0]),
        np.array([0.01, -10.0]),
        np.array([1300.0, 1300.0]),
        np.array([1000.0, 1000.0]),
        np.array([1.0, 1.0]),
        delta=10.0,
        rate_floor=True,
    )
    assert res == pytest.approx(1.0)


@needs_eos
@pytest.mark.unit
def test_rate_cap_scipy_fallback_warns_and_runs_at_one_year(shared_eos, caplog):
    """When scipy is used with rate cap, it warns and keeps max_step at 1 yr."""
    p = _build_mushy_parameters(solver_method='bdf', n_nodes=12, end_time=2.0)
    p.energy = dataclasses.replace(p.energy, phase_boundary_cap='rate', phi_step_cap=None)
    s = EntropySolver(p, entropy_eos=shared_eos)
    s.initialize()
    s.set_initial_entropy(_pick_mushy_S(shared_eos))
    with caplog.at_level('WARNING'):
        s.solve()
    assert any(
        'phase_boundary_cap="rate" needs CVODE; max_step 1 yr with scipy' in rec.message
        for rec in caplog.records
    )


@needs_eos
@pytest.mark.unit
def test_rate_mode_segments_reanchor_state_at_each_segment_start(shared_eos, monkeypatch):
    """Each CVODE segment must re-anchor its root function to the segment start state."""
    s = _solver(shared_eos, 'rate', end_time=200.0)
    anchored_states = []

    orig_solve_segments = s._solve_cvode_segments

    def wrapped_solve_segments(*args, **kw):
        real_roots = kw['roots_at']

        def tracking_roots(y_nd, inside):
            anchored_states.append(np.asarray(y_nd).copy())
            return real_roots(y_nd, inside)

        kw['roots_at'] = tracking_roots
        return orig_solve_segments(*args, **kw)

    monkeypatch.setattr(s, '_solve_cvode_segments', wrapped_solve_segments)
    s.solve()

    assert len(anchored_states) >= 2, 'Expected multiple segments in 200 yr run'
    assert not np.allclose(anchored_states[0], anchored_states[1]), (
        'Segment 1 was not re-anchored!'
    )


@pytest.fixture(scope='module')
def shared_eos():
    from aragog.eos.entropy import EntropyEOS

    from .test_phi_step_cap_armed_smoke import EOS_DIR

    return EntropyEOS(EOS_DIR)
