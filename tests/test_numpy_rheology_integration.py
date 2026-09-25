"""Unit tests for NumPy backend solid-state rheology integration.

Validates config additions, EntropyPhaseEvaluator properties (eta_diff, tau_y),
and EntropyState MLT integration with local and global stress closure modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.config.phases import STRESS_CLOSURE_DEFAULT, MixedPhaseConfig, PhaseConfig
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.parser import _PhaseParameters
from aragog.solver.entropy_state import EntropyState


@pytest.mark.unit
def test_config_rheology_defaults():
    """Verify default values for Arrhenius and yield stress config parameters."""
    pc = PhaseConfig(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=1e-5,
        viscosity=1e21,
    )
    assert pc.activation_energy == 300e3
    assert pc.activation_volume == 5e-6
    assert pc.yield_stress_c == 50e6
    assert pc.yield_stress_mu == 0.6
    assert pc.stress_closure_mode == 'lid'

    pp = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=1e-5,
        viscosity=1e21,
    )
    assert pp.activation_energy == 300e3
    assert pp.activation_volume == 5e-6
    assert pp.yield_stress_c == 50e6
    assert pp.yield_stress_mu == 0.6
    assert pp.stress_closure_mode == STRESS_CLOSURE_DEFAULT

    mpc = MixedPhaseConfig(
        latent_heat_of_fusion=4e6,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='dummy',
        liquidus='dummy',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1e-3,
    )
    assert not hasattr(mpc, 'activation_energy')
    assert not hasattr(mpc, 'stress_closure_mode')


@pytest.mark.unit
def test_config_invalid_stress_closure_mode_raises():
    """Verify invalid stress_closure_mode raises ValueError."""
    with pytest.raises(ValueError, match='stress_closure_mode'):
        _PhaseParameters(
            density=4200.0,
            heat_capacity=1000.0,
            melt_fraction=0.0,
            thermal_conductivity=4.0,
            thermal_expansivity=1e-5,
            viscosity=1e21,
            stress_closure_mode='invalid_mode',
        )


@pytest.mark.unit
def test_entropy_phase_properties_const_mode():
    """Verify eta_diff and tau_y properties in const_properties mode."""
    ev = EntropyPhaseEvaluator(
        entropy_eos=None,
        gravitational_acceleration=10.0,
        const_properties=True,
        enabled=True,
        const_rho=4000.0,
        const_Cp=1000.0,
        const_alpha=3.0e-5,
        const_cond=4.0,
        const_log10visc=21.0,
        activation_energy=300e3,
        activation_volume=5e-6,
        yield_stress_c=50e6,
        yield_stress_mu=0.6,
        stress_closure_mode='local',
    )
    ev.set_pressure(np.array([0.0, 10e9, 50e9]))
    ev.set_entropy(np.array([3000.0, 3050.0, 3100.0]))
    ev.update()

    assert hasattr(ev, 'eta_diff')
    assert hasattr(ev, 'tau_y')
    assert ev.stress_closure_mode == 'local'
    assert len(ev.eta_diff) == 3
    assert len(ev.tau_y) == 3
    assert ev.tau_y[0] == pytest.approx(50e6)
    assert ev.tau_y[1] == pytest.approx(np.minimum(50e6 + 0.6 * 10e9, 500e6))


def _build_test_entropy_state(mode: str = 'local'):
    """Helper to build a working EntropyState with mock mesh and phase evaluators."""
    n = 20
    r_cmb, r_surf = 3480e3, 6371e3
    p_cmb, p_surf = 135e9, 1e5

    r_stag = np.linspace(r_cmb, r_surf, n)
    r_basic = np.zeros(n + 1)
    r_basic[0] = r_cmb
    r_basic[-1] = r_surf
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])
    p_stag = np.linspace(p_cmb, p_surf, n)
    p_basic = np.interp(r_basic, r_stag, p_stag)

    class _MockMesh:
        class Sub:
            pass

        def __init__(self):
            self.basic = self.Sub()
            self.staggered = self.Sub()

    mesh = _MockMesh()
    mesh.basic.radii = r_basic
    mesh.staggered.radii = r_stag
    mesh.basic.area = 4.0 * np.pi * r_basic**2
    mesh.basic.volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.minimum(r_basic - r_cmb, r_surf - r_basic)
    mesh.basic.mixing_length = np.maximum(ml, 1.0)
    mesh.basic.mixing_length_squared = mesh.basic.mixing_length**2
    mesh.basic.mixing_length_cubed = mesh.basic.mixing_length**3
    mesh.basic.pressure = p_basic
    mesh.staggered.pressure = p_stag
    mesh.basic.mass_radii = r_basic
    mesh.staggered.mass_radii = r_stag
    mesh.dxidr = np.ones_like(r_basic)

    dr = np.diff(r_stag)

    def quantity_at_basic_nodes(q):
        q_arr = np.asarray(q).flatten()
        out = np.zeros(n + 1)
        out[0], out[-1] = q_arr[0], q_arr[-1]
        out[1:-1] = 0.5 * (q_arr[:-1] + q_arr[1:])
        return out

    def d_dr_at_basic_nodes(q):
        q_arr = np.asarray(q).flatten()
        out = np.zeros(n + 1)
        out[1:-1] = np.diff(q_arr) / dr
        out[0], out[-1] = out[1], out[-2]
        return out

    mesh.quantity_at_basic_nodes = quantity_at_basic_nodes
    mesh.d_dr_at_basic_nodes = d_dr_at_basic_nodes
    mesh.dr = dr
    mesh.N = n

    class _MockEvaluator:
        def __init__(self, m):
            self.mesh = m

    evaluator = _MockEvaluator(mesh)

    phase_stag = EntropyPhaseEvaluator(
        entropy_eos=None,
        gravitational_acceleration=10.0,
        const_properties=True,
        enabled=True,
        const_log10visc=21.0,
        activation_energy=300e3,
        activation_volume=5e-6,
        yield_stress_c=50e6,
        yield_stress_mu=0.6,
        stress_closure_mode=mode,
    )
    phase_stag.set_pressure(p_stag)
    phase_stag.set_entropy(np.linspace(3100.0, 3000.0, n))
    phase_stag.update()

    phase_basic = EntropyPhaseEvaluator(
        entropy_eos=None,
        gravitational_acceleration=10.0,
        const_properties=True,
        enabled=True,
        const_log10visc=21.0,
        activation_energy=300e3,
        activation_volume=5e-6,
        yield_stress_c=50e6,
        yield_stress_mu=0.6,
        stress_closure_mode=mode,
    )
    phase_basic.set_pressure(p_basic)
    phase_basic.set_entropy(np.linspace(3100.0, 3000.0, n + 1))
    phase_basic.update()
    # Re-pin eta_diff and tau_y after update
    phase_basic._eta_diff = np.full(n + 1, 1e26)
    phase_basic._tau_y = 50e6 + 0.6 * p_basic

    state = EntropyState(
        evaluator=evaluator,
        phase_staggered=phase_stag,
        phase_basic=phase_basic,
        conduction=True,
        convection=True,
    )
    return state, phase_basic


@pytest.mark.unit
def test_entropy_state_mlt_viscosity_capping_local():
    """Verify local stress closure updates viscosity_basic and caps effective viscosity."""
    state, phase_basic = _build_test_entropy_state(mode='local')

    # Unstable entropy gradient to drive convective MLT velocity
    n_stag = len(state._entropy_staggered)
    s_stag = np.linspace(3200.0, 2800.0, n_stag)  # dSdr < 0, convective
    state.update(s_stag, 0.0)

    # Invariants
    assert hasattr(state, 'viscosity_basic')
    visc_eff = state.viscosity_basic
    assert np.all(visc_eff > 0.0)
    assert np.all(np.isfinite(visc_eff))
    # Effective viscosity must be capped at or below eta_diff
    assert np.all(visc_eff <= phase_basic.eta_diff * (1.0 + 1e-12))
    # At active convective cells, yielding caps below the extreme 1e26 Pa s
    convective_mask = state.is_convective
    if np.any(convective_mask):
        assert np.any(visc_eff[convective_mask] < 1e25)


@pytest.mark.unit
def test_entropy_state_mlt_viscosity_capping_lid():
    """Verify lid stress closure updates viscosity_basic and caps effective viscosity."""
    state, phase_basic = _build_test_entropy_state(mode='lid')

    n_stag = len(state._entropy_staggered)
    s_stag = np.linspace(3200.0, 2800.0, n_stag)
    state.update(s_stag, 0.0)

    visc_eff = state.viscosity_basic
    assert np.all(visc_eff > 0.0)
    assert np.all(np.isfinite(visc_eff))
    assert np.all(visc_eff <= phase_basic.eta_diff * (1.0 + 1e-12))
    convective_mask = state.is_convective
    if np.any(convective_mask):
        assert np.any(visc_eff[convective_mask] < 1e25)
