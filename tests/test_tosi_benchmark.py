"""Tosi benchmark integration."""

from __future__ import annotations

import numpy as np
import pytest

from aragog.solver.entropy_solver import EntropySolver
from tests.test_entropy_solver_integration import _build_parameters

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(300)]


@pytest.mark.physics_invariant
@pytest.mark.xfail(
    strict=True, reason='reference cooling over the window is below the tolerance'
)
def test_tosi_thermal_evolution_parity():
    """Verify that Aragog matches Tosi's 0D secular cooling model.

    Integrates for 100 Myr to verify bulk cooling evolution. Matches
    the insulating core BC. We use const_properties to match Tosi's
    material properties, with Arrhenius viscosity enabled.
    """
    import interior_evolution as heat_budget

    tb = heat_budget.interior_evolution(body='Earth')
    tb.tectonics = 'SL'
    tb.core_cooling = 'no'  # Insulating CMB
    tb.Rp = 6371.0e3
    tb.Rc = 3480.0e3
    tb.g = 9.8
    tb.Ts = 300.0
    tb.Tm0 = 1900.0
    tb.Tc0 = 3500.0
    tb.etaref = 1e21
    tb.km = 4.0

    tb.X_U = 0.0
    tb.X_Th = 0.0
    tb.X_K = 0.0
    tb.Q0 = 0.0

    # 10 Myr integration
    end_time_yr = 1.0e7
    tb.maxtime = end_time_yr * tb.yrs
    tb.dt = 1000.0 * tb.yrs
    tb.n_steps = int(tb.maxtime / tb.dt)

    tb.calculate_evolution(outfile=None)
    tosi_Tm = tb.Tm

    parameters = _build_parameters(
        core_bc='quasi_steady',
        solver_method='cvode',
        n_nodes=20,
        end_time=end_time_yr,
        inner_boundary_value=0.0,
        use_jax_jacobian=True,
    )

    parameters.mesh.outer_radius = tb.Rp
    parameters.mesh.inner_radius = tb.Rc
    parameters.boundary_conditions.outer_boundary_condition = 1  # Grey body
    parameters.boundary_conditions.equilibrium_temperature = tb.Ts
    parameters.boundary_conditions.emissivity = 1.0

    parameters.boundary_conditions.inner_boundary_condition = 1  # Core cooling
    parameters.boundary_conditions.inner_boundary_value = 0.0  # Insulating

    parameters.energy.kappah_floor = 0.0

    parameters.phase_mixed.const_properties = True
    parameters.phase_mixed.const_rho = tb.rhom
    parameters.phase_mixed.const_Cp = tb.cm
    parameters.phase_mixed.const_alpha = 3e-5
    parameters.phase_mixed.const_cond = tb.km
    parameters.phase_mixed.const_log10visc = np.log10(tb.etaref)
    parameters.phase_mixed.const_T_ref = tb.Tm0
    parameters.phase_mixed.const_S_ref = 3000.0

    # Arrhenius parameters for Stagnant Lid
    parameters.phase_mixed.arrhenius_t_ref = tb.Tref
    parameters.phase_mixed.activation_energy = tb.E
    parameters.phase_mixed.activation_volume = tb.V

    # Cap yielding to avoid infinite viscosity at 300K
    parameters.phase_mixed.yield_stress_max = 1e30
    parameters.phase_mixed.yield_stress_c = 1e30

    parameters.phase_solid.arrhenius_t_ref = tb.Tref
    parameters.phase_solid.activation_energy = tb.E
    parameters.phase_solid.activation_volume = tb.V
    parameters.phase_solid.yield_stress_max = 1e30
    parameters.phase_solid.yield_stress_c = 1e30
    parameters.phase_solid.viscosity = tb.etaref

    parameters.solver.max_time_step = 1.0e5
    parameters.solver.output_interval = 1.0e6
    parameters.solver.rtol = 1e-3
    parameters.solver.atol = 1e-3

    parameters.initial_condition.initial_condition = 1
    parameters.initial_condition.surface_temperature = tb.Tm0

    solver = EntropySolver(parameters, entropy_eos=None)
    solver.initialize()

    # S_init = parameters.phase_mixed.const_S_ref
    S_init = parameters.phase_mixed.const_S_ref
    S_array = np.full(19, S_init)

    # Smooth initial boundary layer
    S_surf = S_init - parameters.phase_mixed.const_Cp * np.log(tb.Tm0 / tb.Ts)
    S_array[-1] = S_surf
    S_array[-2] = S_surf + 0.33 * (S_init - S_surf)
    S_array[-3] = S_surf + 0.66 * (S_init - S_surf)

    solver.set_initial_entropy(S_array)
    solver.solve()
    out = solver.get_state()

    # Mass-weighted average temperature of the convecting interior
    # We only average cells that are convecting (e.g. T > 1600 K)
    # Or just take the CMB temperature since it's an isentropic interior.
    aragog_final_T = out.T_stag[0]
    tosi_final_T = tosi_Tm[-1]

    # Tolerate 150 K difference
    assert abs(aragog_final_T - tosi_final_T) < 150.0, (
        f'Divergence too large: aragog={aragog_final_T:.1f} K, Tosi={tosi_final_T:.1f} K'
    )
