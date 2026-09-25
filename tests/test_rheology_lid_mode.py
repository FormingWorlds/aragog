from __future__ import annotations

import numpy as np
import pytest

from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.solver.entropy_state import EntropyState


@pytest.mark.unit
def test_rheological_lid_mode_runs_numpy_mlt():
    from tests.test_convection_scaling import _make_mesh

    mesh = _make_mesh()

    def _evaluator(pressure):
        ev = EntropyPhaseEvaluator(
            entropy_eos=None,
            gravitational_acceleration=9.8,
            const_properties=True,
            const_rho=4000.0,
            enabled=True,
            stress_closure_mode='lid',
            lid_base_mode='rheological',
        )
        ev.pressure = pressure
        ev.entropy = np.full_like(pressure, 2000.0)
        ev.update()
        return ev

    class _Eval:
        pass

    evaluator = _Eval()
    evaluator.mesh = mesh

    state = EntropyState(
        evaluator=evaluator,
        phase_staggered=_evaluator(mesh.staggered.pressure),
        phase_basic=_evaluator(mesh.basic.pressure),
    )

    S = np.linspace(3000, 2000, mesh.staggered.radii.size)

    # Verify state update with rheological lid base mode
    state.update(S, time=0.0)

    visc = state.phase_basic.viscosity()
    assert np.all(np.isfinite(visc))
    assert np.all(visc > 0.0)
    assert hasattr(state, 'viscosity_basic')
    assert np.all(state.viscosity_basic <= state.phase_basic.eta_diff * (1.0 + 1e-12))
    assert np.any(state.viscosity_basic < state.phase_basic.eta_diff)


@pytest.mark.unit
def test_strain_rate_zero_when_no_lid():
    # Test that a column hotter than t_lid_base everywhere has strain rate 0
    # and effective viscosity == eta_diff
    from aragog.rheology import compute_strain_rate_global, eta_eff

    r = np.linspace(1e6, 2e6, 20)
    T = np.full(20, 1900.0)
    v = np.full(20, 1e-9)  # Some velocity

    # 1900 K is hotter than 1400 K (default t_lid_base)
    sr = compute_strain_rate_global(r, T, v, t_lid_base=1400.0)
    assert sr == 0.0

    eta_diff = np.full(20, 1e21)
    tau_y = np.full(20, 1e6)

    # Since strain rate is 0, stress closure should return eta_diff
    eta_eff_val = eta_eff(eta_diff, tau_y, sr)
    np.testing.assert_allclose(eta_eff_val, eta_diff)


@pytest.mark.unit
def test_strain_rate_zero_when_inverted_profile():
    """Verify that a profile with a hot surface overlying a cooler interior yields zero strain rate."""
    from aragog.rheology import compute_strain_rate_global

    r = np.linspace(1e6, 2e6, 50)
    T = np.full(50, 1200.0)
    T[-1] = 1800.0  # Hot surface above t_lid_base (1400 K)
    v = np.full(50, 1e-9)

    sr = compute_strain_rate_global(r, T, v, t_lid_base=1400.0)
    assert sr == 0.0


@pytest.mark.unit
def test_strain_rate_all_cold_profile():
    """Verify that an entirely cold column uses full layer thickness as lid depth."""
    from aragog.rheology import compute_strain_rate_global

    r = np.linspace(1e6, 2e6, 50)
    T = np.full(50, 1000.0)  # All below t_lid_base (1400 K)
    v = np.full(50, 1e-9)

    sr = compute_strain_rate_global(r, T, v, t_lid_base=1400.0)
    expected_d_lid = r[-1] - r[0]
    expected_sr = 1e-9 / expected_d_lid
    np.testing.assert_allclose(sr, expected_sr)
