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
    """Outside, entry at delta fires 'stiff'; inside, exit sits at 2 delta and the other cells keep entry and progress."""
    out = _roots([1350.0, 1500.0], inside=False)
    assert out.components(0.0, [1310.0, 1500.0])[0] == pytest.approx(0.0)
    assert out.fired(0.0, [1310.0, 1500.0]) == 'stiff'
    ins = _roots([1305.0, 1500.0], inside=True)
    assert ins.names == ['stiff', 'entry', 'progress']
    assert ins.components(0.0, [1305.0, 1500.0])[0] == pytest.approx(-15.0)
    assert ins.components(0.0, [1320.0, 1500.0])[0] == pytest.approx(0.0)
    assert ins.components(0.0, [1305.0, 1400.0])[2] == pytest.approx(0.0)
    assert ins.fired(0.0, [1305.0, 1400.0]) == 'progress'
    assert ins.components(0.0, [1305.0, 1310.0])[1] == pytest.approx(0.0)
    assert ins.fired(0.0, [1305.0, 1310.0]) == 'entry'


class _FakeCap:
    cap, cap_T, cap_S, phi0, evals, binding_cap = 0.05, 0.0, 0.0, 0.3, 7, 'phi'

    def __init__(self, g):
        self.g = g

    def evaluate(self, t, y, g, userdata=None):
        g[0] = self.g
        return 0


@pytest.mark.unit
def test_segment_root_scales_the_cap_by_the_cap_that_binds():
    """A phi cap 0.015 short of firing is not taken for the stiff root at 0.01 when a 20 K cap is also armed."""

    class _TwoCaps(_FakeCap):
        cap_T = 20.0

    r = _roots([1350.0, 1500.0], inside=False, cap=_TwoCaps(0.015))
    at_stiff = [1310.01, 1500.0]
    assert r.components(0.0, at_stiff)[1] == pytest.approx(0.01)
    assert r.fired(0.0, at_stiff) == 'stiff'
    assert (
        _roots([1350.0, 1500.0], inside=False, cap=_TwoCaps(1e-6)).fired(0.0, at_stiff) == 'cap'
    )


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
    assert _EnergyParameters(**kw).phase_boundary_cap is None
    with pytest.raises(ValueError, match='phase_boundary_cap'):
        _EnergyParameters(**kw, phase_boundary_cap='adaptive')
    from aragog.config import EnergyConfig

    assert EnergyConfig(**kw).phase_boundary_cap is None
    assert EnergyConfig(**kw, phase_boundary_cap='fixed').phase_boundary_cap == 'fixed'
    with pytest.raises(ValueError, match='phase_boundary_cap'):
        EnergyConfig(**kw, phase_boundary_cap='adaptive')


@needs_eos
@pytest.mark.smoke
def test_unset_phase_boundary_cap_runs_rate_segments(shared_eos):
    """With the key unset the solver runs the rate segments, like an explicit 'rate'."""
    unset = _solver(shared_eos, None, end_time=200.0)
    rate = _solver(shared_eos, 'rate', end_time=200.0)
    unset.solve()
    rate.solve()
    assert len(unset._solution.segments) >= 2
    np.testing.assert_array_equal(unset._solution.y, rate._solution.y)


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
    """Segments cover the call with every output point; each segment start is a point; time increases."""
    s = _solver(shared_eos, 'rate', end_time=200.0)
    s.solve()
    sol = s._solution
    starts = np.array([seg[0] for seg in sol.segments])
    assert len(starts) >= 2 and sol.t[0] == 0.0 and sol.t[-1] == pytest.approx(200.0)
    assert np.all(np.diff(sol.t) > 0.0) and sol.y.shape[1] == sol.t.size
    assert all(np.any(np.isclose(sol.t, t0, rtol=1e-12, atol=1e-9)) for t0 in starts)
    assert sol.t.size >= s._cvode_output_points + len(starts) - 1


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
@pytest.mark.smoke
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
@pytest.mark.smoke
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
        bounds=(0.0, 100.0),
    )
    floor = (0.01 + 10.0) / 2.0
    expected = 0.1 * 50.0 / floor
    assert res == pytest.approx(expected)


@needs_eos
@pytest.mark.smoke
def test_rate_cap_scipy_fallback_runs_at_one_year(shared_eos, monkeypatch):
    """When scipy is used with rate cap, max_step stays at 1 yr."""
    p = _build_mushy_parameters(solver_method='bdf', n_nodes=12, end_time=2.0)
    p.energy = dataclasses.replace(p.energy, phase_boundary_cap='rate', phi_step_cap=None)
    s = EntropySolver(p, entropy_eos=shared_eos)
    s.initialize()
    s.set_initial_entropy(_pick_mushy_S(shared_eos))
    import aragog.solver.entropy_solver as es

    seen, real = [], es.solve_ivp
    monkeypatch.setattr(
        es, 'solve_ivp', lambda *a, **k: seen.append(k['max_step']) or real(*a, **k)
    )
    s.solve()
    assert seen[0] * s._build_nondim_scales().t_ref == pytest.approx(1.0)


@needs_eos
@pytest.mark.smoke
def test_rate_mode_segments_reanchor_state_at_each_segment_start(shared_eos, monkeypatch):
    """Each CVODE segment must re-anchor its root function to the segment start state."""
    s = _solver(shared_eos, 'rate', end_time=200.0)
    anchored_roots = []

    orig_solve_segments = s._solve_cvode_segments

    def wrapped_solve_segments(*args, **kw):
        real_roots = kw['roots_at']

        def tracking_roots(y_nd, inside):
            r = real_roots(y_nd, inside)
            anchored_roots.append(r.S0.copy())
            return r

        kw['roots_at'] = tracking_roots
        return orig_solve_segments(*args, **kw)

    monkeypatch.setattr(s, '_solve_cvode_segments', wrapped_solve_segments)
    s.solve()

    assert len(anchored_roots) >= 2, 'Expected multiple segments in 200 yr run'
    assert not np.allclose(anchored_roots[0], anchored_roots[1]), (
        'Segment 1 root function was not re-anchored to the segment state!'
    )


@pytest.mark.unit
def test_solve_cvode_segments_stops_at_a_root_just_below_the_call_end():
    """A root within 1e-12 of the call end ends the call; no segment gets a one-point tspan."""
    from unittest.mock import MagicMock

    from scipy.optimize import OptimizeResult

    end = 1.0e6
    s = EntropySolver.__new__(EntropySolver)
    s._output_grid = lambda t0, t1: np.linspace(t0, t1, 5)
    calls = []

    def fake_solve_cvode(**kw):
        assert len(kw['tspan']) >= 2, 'CVODE needs at least 2 output times'
        calls.append(kw['start_time'])
        res = OptimizeResult(
            t=np.array([kw['start_time'], end * (1.0 - 1e-13)]), y=np.ones((2, 2))
        )
        res.nfev = res.cvode_nst = res.cvode_nfe = res.status = 0
        res.cvode_flag = 2
        return res

    s._solve_cvode = fake_solve_cvode
    roots = MagicMock(inside=False)
    roots.fired.return_value = 'progress'
    res = s._solve_cvode_segments(
        start_time=0.0,
        end_time=end,
        y0=np.ones(2),
        roots_at=lambda y, inside: roots,
        h_at=lambda t, y: 15.0,
        t_ref=1.0,
        h_min=1.0,
    )
    assert calls == [0.0] and res.t[-1] == pytest.approx(end, rel=1e-12)
    assert res.cvode_flag == 0


@pytest.fixture(scope='module')
def shared_eos():
    from aragog.eos.entropy import EntropyEOS

    from .test_phi_step_cap_armed_smoke import EOS_DIR

    return EntropyEOS(EOS_DIR)


def _rtol_warnings(caplog):
    return [r for r in caplog.records if 'accuracy is verified for rtol' in r.getMessage()]


@needs_eos
@pytest.mark.smoke
@pytest.mark.parametrize(
    ('mode', 'tol', 'expected'), [('rate', 1e-6, 1), ('rate', 1e-8, 0), ('fixed', 1e-6, 0)]
)
def test_rate_mode_warns_once_per_solver_when_rtol_is_loose(
    shared_eos, caplog, mode, tol, expected
):
    """Rate mode warns once over two solve() calls at rtol 1e-6; never at 1e-8 or in fixed mode."""
    s = _solver(shared_eos, mode, end_time=2.0, tol=tol)
    with caplog.at_level('WARNING'):
        s.solve()
        s.solve()
    assert len(_rtol_warnings(caplog)) == expected


def _two_calls(eos, with_state):
    s = _solver(eos, 'rate', end_time=20.0)
    s.solve()
    if with_state:
        s.get_state()
    S = np.array(s._solution.y[: s._n_stag, -1])
    s.parameters.solver.start_time, s.parameters.solver.end_time = 20.0, 40.0
    s.reset()
    s.set_initial_entropy(S)
    s.solve()
    return s._solution


@needs_eos
@pytest.mark.smoke
def test_get_state_between_calls_leaves_the_next_rate_call_unchanged(shared_eos):
    """get_state() evaluates the RHS at the final state; the next rate call's segments and trajectory stay bitwise the same."""
    a, b = _two_calls(shared_eos, True), _two_calls(shared_eos, False)
    assert a.segments == b.segments and len(a.segments) >= 1
    np.testing.assert_array_equal(a.y, b.y)


@needs_eos
@pytest.mark.smoke
def test_rate_cap_includes_the_cmb_entry_at_the_cmb_pressure(shared_eos, monkeypatch):
    """The cap sees every staggered cell plus the bottom cell at the CMB-pressure boundaries, with the rate floor on."""
    import aragog.solver.entropy_solver as es

    seen, real = [], es._rate_phase_boundary_max_step
    monkeypatch.setattr(
        es,
        '_rate_phase_boundary_max_step',
        lambda *a, **k: seen.append((a, k)) or real(*a, **k),
    )
    s = _solver(shared_eos, 'rate', end_time=2.0)
    s.solve()
    (S, _, S_liq, S_sol, mass, _), kw = seen[0]
    assert kw['rate_floor'] is True
    P_cmb = np.array([s._P_basic_flat[0]])
    assert S.size == s._n_stag + 1 and S[-1] == S[0] and mass[-1] == 0.0
    assert S_liq[-1] == pytest.approx(float(shared_eos.liquidus_entropy(P_cmb).item()))
    assert S_sol[-1] == pytest.approx(float(shared_eos.solidus_entropy(P_cmb).item()))


@needs_eos
@pytest.mark.smoke
def test_rate_mode_starts_inside_the_stiff_zone_when_a_cell_is_within_delta(
    shared_eos, monkeypatch
):
    """A start with one cell 1 J/kg/K above its liquidus opens an inside segment that watches every other entry."""
    s = _solver(shared_eos, 'rate', end_time=2.0)
    S_liq = np.asarray(shared_eos.liquidus_entropy(s._P_stag_flat)).ravel()
    S0 = np.full(s._n_stag, S_liq[5] + 1.0)
    assert np.abs(S0 - S_liq).min() < 10.0
    s.set_initial_entropy(S0)
    first, real = [], s._solve_cvode_segments

    def wrapped(*args, **kw):
        roots_at = kw['roots_at']
        kw['roots_at'] = lambda y, inside: first.append(roots_at(y, inside)) or first[-1]
        return real(*args, **kw)

    monkeypatch.setattr(s, '_solve_cvode_segments', wrapped)
    s.solve()
    assert first[0].inside and first[0].names == ['stiff', 'entry', 'progress']
    assert not first[0].watch[5] and first[0].watch.sum() == s._n_stag


@needs_eos
@pytest.mark.smoke
def test_rate_mode_ends_a_segment_when_a_second_cell_approaches_inside_the_stiff_zone(
    shared_eos, monkeypatch
):
    """Cell A rests 1 J/kg/K above its liquidus; cell B falls from 150 J/kg/K above its own at 1 J/kg/K/yr.

    The call starts inside the stiff zone because of A, so only B's progress and stiff-zone entry
    can end a segment; one must fire before B crosses at 150 yr.
    """
    s = _solver(shared_eos, 'rate', end_time=200.0)
    S_liq = np.asarray(shared_eos.liquidus_entropy(s._P_stag_flat)).ravel()
    S0 = S_liq + 500.0
    S0[2], S0[8] = S_liq[2] + 1.0, S_liq[8] + 150.0
    s.set_initial_entropy(S0)
    rate = np.zeros(s._n_stag)
    rate[8] = -1.0
    monkeypatch.setattr(s, '_dSdt_single', lambda t, y: rate.copy())
    s.solve()
    starts = [seg[0] for seg in s._solution.segments]
    assert s._solution.t[-1] == pytest.approx(200.0)
    assert len(starts) >= 2 and s._solution.segments[1][1] == 'progress'
    assert starts[1] == pytest.approx(75.0, rel=1e-6)
    assert s._solution.y[8, -1] == pytest.approx(S_liq[8] - 50.0, abs=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize('bad', ['S', 'dSdt', 'mass'])
def test_rate_cap_returns_the_lower_bound_for_non_finite_input(bad):
    """A NaN entropy, rate or mass gives the 1 yr lower bound, not the 100 yr ceiling."""
    args = dict(S=[1200.0, 1350.0], dSdt=[-1.0, -2.0], mass=[1.0, 2.0])
    args[bad] = [np.nan, args[bad][1]]
    assert (
        _cap(args['S'], args['dSdt'], args['mass'], rate_floor=True, bounds=(1.0, 100.0)) == 1.0
    )
    assert _cap([1200.0, 1350.0], [-1.0, -2.0], [1.0, 2.0], rate_floor=True) > 1.0
    S_liq = np.array([np.nan, S_LIQ])
    got = _rate_phase_boundary_max_step(
        np.array([1200.0, 1350.0]),
        np.array([-1.0, -2.0]),
        S_liq,
        np.full(2, S_SOL),
        np.ones(2),
        DELTA,
    )
    assert got == 1.0


@needs_eos
@pytest.mark.smoke
@pytest.mark.parametrize('cap', ['rate', None])
@pytest.mark.parametrize(
    ('core_bc', 'method', 'message'),
    [
        ('gradient', 'cvode', 'not used by the gradient core'),
        ('quasi_steady', 'bdf', 'needs CVODE'),
    ],
)
def test_rate_mode_fallback_logs_once_and_skips_the_rtol_warning(
    shared_eos, caplog, core_bc, method, message, cap
):
    """Where 'rate' falls back to 1 yr, one line names the reason over two solves: WARNING when
    'rate' is set, INFO when it is the default; no rtol warning."""
    p = _build_mushy_parameters(solver_method=method, n_nodes=12, end_time=2.0)
    p.energy = dataclasses.replace(p.energy, phase_boundary_cap=cap, phi_step_cap=None)
    p.boundary_conditions.core_bc = core_bc
    p.solver.rtol = p.solver.atol = 1e-6
    s = EntropySolver(p, entropy_eos=shared_eos)
    s.initialize()
    s.set_initial_entropy(_pick_mushy_S(shared_eos))
    with caplog.at_level('INFO'):
        s.solve()
        s.solve()
    hits = [r for r in caplog.records if message in r.getMessage()]
    assert len(hits) == 1 and hits[0].levelname == ('WARNING' if cap else 'INFO')
    assert not _rtol_warnings(caplog)


@needs_eos
@pytest.mark.smoke
def test_rate_mode_ends_the_call_at_a_real_phi_step_cap(shared_eos):
    """With phi_step_cap armed, the rate segments stop the call at the phi cap before the call end."""
    s = _solver(shared_eos, 'rate', end_time=2000.0)
    s.parameters.energy = dataclasses.replace(s.parameters.energy, phi_step_cap=0.05)
    s.solve()
    sol = s._solution
    assert getattr(sol, 'cap_fired', False) and sol.t[-1] < 2000.0
    assert sol.cap_label == 'phi' and sol.cap_value == pytest.approx(0.05)


@pytest.mark.unit
def test_solver_tolerances_default_to_1e_8():
    """rtol and atol default to 1e-8 in both solver schemas; explicit values are kept."""
    from aragog.config.solver import SolverConfig
    from aragog.parser import _SolverParameters

    for cls in (_SolverParameters, SolverConfig):
        default = cls(start_time=0.0, end_time=1.0)
        assert (default.rtol, default.atol) == (1e-8, 1e-8)
        assert cls(start_time=0.0, end_time=1.0, rtol=1e-6, atol=1e-9).rtol == 1e-6


@needs_eos
@pytest.mark.smoke
def test_rate_mode_ends_a_segment_when_a_cell_inside_the_zone_speeds_up(
    shared_eos, monkeypatch
):
    """A cell 8 J/kg/K above its liquidus falls at 0.01 J/kg/K/yr, then at 2 from 100 yr.

    Its progress root (half of max(d0, delta) = 5 J/kg/K of motion) fires at 102 yr, before
    the crossing at 103.5 yr, and the next segment runs at the 1 yr lower bound.
    """
    s = _solver(shared_eos, 'rate', end_time=400.0)
    S_liq = np.asarray(shared_eos.liquidus_entropy(s._P_stag_flat)).ravel()
    S0 = S_liq + 3000.0
    S0[8] = S_liq[8] + 8.0
    s.set_initial_entropy(S0)

    def rate(t, y):
        r = np.zeros(s._n_stag)
        r[8] = -0.01 if t < 100.0 else -2.0
        return r

    monkeypatch.setattr(s, '_dSdt_single', rate)
    s.solve()
    seg = s._solution.segments
    assert len(seg) >= 2 and seg[1][1] == 'progress'
    assert 100.0 < seg[1][0] < 103.5 and seg[1][0] == pytest.approx(102.0, abs=0.05)
    assert seg[1][2] == pytest.approx(1.0)
