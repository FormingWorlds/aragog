"""Tests for ``energy.phase_boundary_cap``: the fixed 1 yr step and the rate-based step near a phase boundary."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import aragog.solver.entropy_solver as es
from aragog.parser import _EnergyParameters
from aragog.solver.entropy_solver import EntropySolver, _rate_phase_boundary_max_step

from .test_phi_step_cap_armed_smoke import _build_mushy_parameters, _pick_mushy_S, needs_eos

S_SOL, S_LIQ, MARGIN = 1000.0, 1300.0, 200.0


def _cap(S, dSdt, **kw):
    n = len(S)
    return _rate_phase_boundary_max_step(
        np.asarray(S, float),
        np.asarray(dSdt, float),
        np.full(n, S_LIQ),
        np.full(n, S_SOL),
        MARGIN,
        **kw,
    )


def test_rate_cap_uses_next_boundary_in_the_direction_of_motion():
    """Cooling above the liquidus aims at the liquidus, inside the band at the solidus, heating at the liquidus."""
    assert _cap([1350.0], [-10.0], fraction=1.0, bounds=(0.0, 1e9)) == pytest.approx(5.0)
    assert _cap([1200.0], [-1.0], fraction=1.0, bounds=(0.0, 1e9)) == pytest.approx(200.0)
    assert _cap([1100.0], [2.0], fraction=1.0, bounds=(0.0, 1e9)) == pytest.approx(100.0)
    assert _cap([950.0], [0.5], fraction=1.0, bounds=(0.0, 1e9)) == pytest.approx(100.0)
    assert _cap(
        [1350.0, 1200.0], [-10.0, -1.0], fraction=1.0, bounds=(0.0, 1e9)
    ) == pytest.approx(5.0)


def test_rate_cap_ignores_cells_moving_away_stationary_or_far():
    """Below the solidus and cooling, above the liquidus and heating, dS/dt = 0, or outside the margin: upper bound."""
    S = [950.0, 1350.0, 1200.0, 2000.0, 500.0]
    dSdt = [-5.0, 5.0, 0.0, -1e6, 1e6]
    assert _cap(S, dSdt) == 100.0


def test_rate_cap_applies_fraction_and_clip():
    assert _cap([1200.0], [-1.0]) == pytest.approx(20.0)
    assert _cap([1350.0], [-10.0]) == 1.0
    assert _cap([1200.0], [-1e-3]) == 100.0


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


def _max_step_yr(eos, monkeypatch, mode, core_bc='quasi_steady', rate_value=37.0):
    """Solve one short call from the mushy band; return the max_step [yr] passed to CVODE and the rate-cap calls."""
    p = _build_mushy_parameters(solver_method='cvode', n_nodes=12, end_time=2.0)
    p.energy = dataclasses.replace(p.energy, phase_boundary_cap=mode, phi_step_cap=None)
    p.boundary_conditions.core_bc = core_bc
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(_pick_mushy_S(eos))
    calls, seen = [], {}
    monkeypatch.setattr(
        es, '_rate_phase_boundary_max_step', lambda *a, **k: calls.append(a) or rate_value
    )
    real = s._solve_cvode

    def spy(**kw):
        seen['max_step'] = kw['max_step']
        return real(**kw)

    monkeypatch.setattr(s, '_solve_cvode', spy)
    s.solve()
    return seen['max_step'] * s._build_nondim_scales().t_ref, calls


@needs_eos
@pytest.mark.smoke
def test_fixed_mode_keeps_one_year_and_rate_mode_uses_the_rate_cap(shared_eos, monkeypatch):
    """'fixed' never evaluates the rate cap; 'rate' passes its value to CVODE; the gradient core keeps 1 yr."""
    step, calls = _max_step_yr(shared_eos, monkeypatch, 'fixed')
    assert step == pytest.approx(1.0) and not calls
    step, calls = _max_step_yr(shared_eos, monkeypatch, 'rate')
    assert step == pytest.approx(37.0) and len(calls) == 1
    dSdt = calls[0][1]
    assert dSdt.shape == calls[0][0].shape and np.all(np.isfinite(dSdt)) and np.any(dSdt != 0.0)
    step, calls = _max_step_yr(shared_eos, monkeypatch, 'rate', rate_value=1e4)
    assert step == pytest.approx(100.0)
    step, calls = _max_step_yr(shared_eos, monkeypatch, 'rate', core_bc='gradient')
    assert step == pytest.approx(1.0) and not calls


@pytest.fixture(scope='module')
def shared_eos():
    from aragog.eos.entropy import EntropyEOS

    from .test_phi_step_cap_armed_smoke import EOS_DIR

    return EntropyEOS(EOS_DIR)
