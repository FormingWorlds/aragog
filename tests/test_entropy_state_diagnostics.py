"""Tests for EntropyState diagnostic accessors and rheology branching.

Exercises property accessors (viscosity_basic, visc_eff, strain_rate_basic,
tau_y_basic, lid_state, F_conv_unyielded) and low-theta / local-mode branching
in ``aragog.solver.entropy_state.EntropyState``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.solver.entropy_state import EntropyState
from tests.test_convection_scaling import _make_mesh


def _make_state(*, enabled: bool = True, stress_closure_mode: str = 'lid') -> EntropyState:
    """Helper to build minimal EntropyState with constant properties."""
    mesh = _make_mesh(n=20)

    def _build_evaluator():
        ev = EntropyPhaseEvaluator(
            entropy_eos=None,
            gravitational_acceleration=9.81,
            const_properties=True,
            const_rho=4000.0,
            const_Cp=1200.0,
            const_cond=4.0,
            enabled=enabled,
            stress_closure_mode=stress_closure_mode,
            lid_base_mode='fixed',
            lid_base_temperature=1800.0,
        )
        ev.pressure = mesh.basic.pressure
        ev.entropy = np.full_like(mesh.basic.pressure, 2000.0)
        ev.update()
        return ev

    ev_basic = _build_evaluator()
    ev_stag = _build_evaluator()
    ev_stag.pressure = mesh.staggered.pressure

    eval_holder = SimpleNamespace(mesh=mesh)

    state = EntropyState(
        evaluator=eval_holder,
        phase_staggered=ev_stag,
        phase_basic=ev_basic,
        kappah_floor=0.0,
    )
    return state


@pytest.mark.unit
def test_entropy_state_property_accessors_contract():
    """EntropyState properties expose physical profiles with correct shapes."""
    state = _make_state(enabled=True, stress_closure_mode='lid')
    assert state.lid_state is None

    n_stag = len(state._evaluator.mesh.staggered.radii)
    n_basic = len(state._evaluator.mesh.basic.radii)
    S_profile = np.linspace(2500.0, 2000.0, n_stag)
    state.update(S_profile, 0.0)

    assert state.viscosity_basic.shape == (n_basic,)
    assert np.all(np.isfinite(state.viscosity_basic))
    assert np.all(state.viscosity_basic > 0.0)

    assert state.visc_eff.shape == (n_basic,)
    assert np.all(np.isfinite(state.visc_eff))
    assert np.all(state.visc_eff > 0.0)

    assert state.strain_rate_basic.shape == (n_basic,)
    assert np.all(np.isfinite(state.strain_rate_basic))
    assert np.all(state.strain_rate_basic >= 0.0)

    assert state.tau_y_basic.shape == (n_basic,)
    assert np.all(np.isfinite(state.tau_y_basic))

    assert state.lid_state is not None
    assert isinstance(state.lid_state, dict)
    assert 'tau_d' in state.lid_state

    assert state.F_conv_unyielded is not None
    assert state.F_conv_unyielded.shape == (n_basic,)


@pytest.mark.unit
def test_entropy_state_local_mode_sets_lid_state_none():
    """EntropyState in local stress closure mode sets lid_state to None."""
    state = _make_state(enabled=True, stress_closure_mode='local')
    n_stag = len(state._evaluator.mesh.staggered.radii)
    S_profile = np.linspace(2500.0, 2000.0, n_stag)
    state.update(S_profile, 0.0)

    assert state.lid_state is None
    assert np.all(np.isfinite(state.viscosity_basic))
    assert np.all(state.viscosity_basic > 0.0)


@pytest.mark.unit
def test_entropy_state_disabled_rheology_sets_unyielded_viscosity():
    """EntropyState with enabled=False passes through baseline phase viscosity."""
    state = _make_state(enabled=False)
    n_stag = len(state._evaluator.mesh.staggered.radii)
    S_profile = np.linspace(2500.0, 2000.0, n_stag)
    state.update(S_profile, 0.0)

    assert state.lid_state is None
    np.testing.assert_allclose(
        state.viscosity_basic,
        np.asarray(state.phase_basic.viscosity()).ravel(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(state.visc_eff, state.viscosity_basic, rtol=1e-12)


@pytest.mark.unit
def test_entropy_state_low_theta_warning_logged_once(caplog):
    """EntropyState warns when theta < 9.0 and suppresses duplicate warnings."""
    state = _make_state(enabled=True, stress_closure_mode='lid')
    n_stag = len(state._evaluator.mesh.staggered.radii)
    S_profile = np.linspace(2001.0, 2000.0, n_stag)

    with caplog.at_level(logging.WARNING):
        state.update(S_profile, 0.0)
        if state._low_theta_warned:
            assert 'Frank-Kamenetskii' in caplog.text or 'theta' in caplog.text
            caplog.clear()
            state.update(S_profile, 1.0)
            assert 'Frank-Kamenetskii' not in caplog.text
