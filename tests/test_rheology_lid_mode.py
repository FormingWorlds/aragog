from __future__ import annotations

import numpy as np
import pytest

from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.solver.entropy_state import EntropyState


@pytest.mark.unit
def test_rheological_lid_mode_runs_numpy_mlt():
    from tests.test_convection_scaling import _make_mesh
    mesh = _make_mesh()
    n = mesh.basic.radii.size

    def _evaluator(pressure):
        ev = EntropyPhaseEvaluator(
            entropy_eos=None,
            gravitational_acceleration=9.8,
            const_properties=True,
            const_rho=4000.0,
            enabled=True,
            stress_closure_mode='global',
            lid_base_mode='rheological'
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
        phase_basic=_evaluator(mesh.basic.pressure)
    )

    T = np.linspace(3000, 300, n)
    P = np.linspace(1e10, 1e5, n)
    S = np.linspace(3000, 2000, mesh.staggered.radii.size)

    # Pre-fix this raises NameError: name 'compute_t_lid_base' is not defined
    state.update(S, time=0.0)

    # If we got here, it didn't crash
    assert np.all(np.isfinite(state.phase_basic.viscosity()))

@pytest.mark.unit
def test_strain_rate_zero_when_no_lid():
    # Test that a column hotter than t_lid_base everywhere has strain rate 0
    # and effective viscosity == eta_diff
    from aragog.rheology import compute_strain_rate_global, eta_eff
    r = np.linspace(1e6, 2e6, 20)
    T = np.full(20, 1900.0)
    v = np.full(20, 1e-9) # Some velocity

    # 1900 K is hotter than 1400 K (default t_lid_base)
    sr = compute_strain_rate_global(r, T, v, t_lid_base=1400.0)
    assert sr == 0.0

    eta_diff = np.full(20, 1e21)
    tau_y = np.full(20, 1e6)

    # Since strain rate is 0, stress closure should return eta_diff
    eta_eff = eta_eff(eta_diff, tau_y, sr)
    np.testing.assert_allclose(eta_eff, eta_diff)
