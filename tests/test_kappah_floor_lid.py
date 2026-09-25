"""Verification test for stagnant lid eddy-diffusivity floor masking (Decision X).

Verifies that the PROTEUS eddy-diffusivity floor (kappah_floor * f(phi)) is masked
inside the stagnant lid by (1 - w_lid) when solid-state rheology is enabled,
preventing artificial convective heat transport in the cold conductive lid.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.solver.entropy_state import EntropyState
from tests.test_convection_scaling import _make_mesh


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_kappah_floor_lid_masking():
    """Verify kappah floor is masked in lid when rheology enabled (Decision X).

    On a conductive-lid profile with rheology enabled:
    - kappa_h < 10 * kappa at lid surface nodes.
    - Below the lid, kappa_h matches the interior convective value.
    - When rheology is disabled, kappa_h at the surface reproduces the floor
      kappah_floor * f(phi).
    """
    mesh = _make_mesh()

    def _build_evaluator(enabled: bool):
        ev = EntropyPhaseEvaluator(
            entropy_eos=None,
            gravitational_acceleration=9.8,
            const_properties=True,
            const_rho=4000.0,
            const_Cp=1000.0,
            const_cond=4.0,
            enabled=enabled,
            stress_closure_mode='lid' if enabled else 'local',
            lid_base_mode='rheological',
        )
        ev.pressure = mesh.basic.pressure
        ev.entropy = np.full_like(mesh.basic.pressure, 2000.0)
        ev.update()
        return ev

    class _EvaluatorHolder:
        pass

    eval_holder = _EvaluatorHolder()
    eval_holder.mesh = mesh

    # Configure state with rheology ON
    ev_basic_on = _build_evaluator(enabled=True)
    ev_stag_on = _build_evaluator(enabled=True)
    ev_stag_on.pressure = mesh.staggered.pressure

    state_on = EntropyState(
        evaluator=eval_holder,
        phase_staggered=ev_stag_on,
        phase_basic=ev_basic_on,
        kappah_floor=10.0,
    )
    state_on.phase_basic._phi_rheo = 0.4
    state_on.phase_basic._phi_width = 0.2
    state_on.phase_basic.melt_fraction = lambda: np.zeros(mesh.basic.radii.size)
    state_on.phase_staggered.melt_fraction = lambda: np.zeros(mesh.staggered.radii.size)

    # Configure state with rheology OFF (Decision A opt-in parity)
    ev_basic_off = _build_evaluator(enabled=False)
    ev_stag_off = _build_evaluator(enabled=False)
    ev_stag_off.pressure = mesh.staggered.pressure

    state_off = EntropyState(
        evaluator=eval_holder,
        phase_staggered=ev_stag_off,
        phase_basic=ev_basic_off,
        kappah_floor=10.0,
    )
    state_off.phase_basic._phi_rheo = 0.4
    state_off.phase_basic._phi_width = 0.2
    state_off.phase_basic.melt_fraction = lambda: np.zeros(mesh.basic.radii.size)
    state_off.phase_staggered.melt_fraction = lambda: np.zeros(mesh.staggered.radii.size)

    # Cold surface (T ~ 300 K) and hot interior (T ~ 3000 K)
    S = np.linspace(3000.0, 700.0, mesh.staggered.radii.size)

    state_on.update(S, time=0.0)
    state_off.update(S, time=0.0)

    # Thermal diffusivity kappa = k / (rho * Cp) = 4 / (4000 * 1000) = 1e-6 m^2/s
    kappa_molecular = 4.0 / (4000.0 * 1000.0)
    ten_kappa = 10.0 * kappa_molecular

    kh_on = state_on.eddy_diffusivity
    kh_off = state_off.eddy_diffusivity
    w_lid = state_on._lid_state['w_lid']

    # 1. At lid surface node, lid mask w_lid is near 1
    assert w_lid[-1] > 0.95

    # 2. At lid surface node, kappa_h with rheology enabled is below 10 * kappa
    assert kh_on[-1] < ten_kappa, f'Expected kh_on[-1] < {ten_kappa}, got {kh_on[-1]}'

    # 3. With rheology disabled, kappa_h at surface equals floor value kappah_floor * f(phi=0)
    from aragog.utilities import tanh_weight

    f_phi_0 = tanh_weight(np.array([0.0]), 0.4, 0.2)[0]
    expected_floor_off = 10.0 * f_phi_0
    assert np.isclose(kh_off[-1], expected_floor_off, rtol=1e-3)
    assert kh_off[-1] > 100.0 * ten_kappa

    # 4. In deep interior (well below lid), kh_on has full convective value
    assert kh_on[2] > 1.0e4
    assert np.isclose(kh_on[2], kh_off[2], rtol=0.2)
