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
