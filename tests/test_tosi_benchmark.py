"""Verification Tier D: Tosi Benchmark Integration."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from aragog.cli import _derive_initial_entropy_from_config
from aragog.solver.entropy_solver import EntropySolver
from tests.test_entropy_solver_integration import _build_parameters

_tosi_path = os.path.expanduser('~/git/heat_budget/Main')
if os.path.isdir(_tosi_path):
    sys.path.insert(0, _tosi_path)

try:
    import interior_evolution as heat_budget
except ImportError:
    heat_budget = None


@pytest.mark.physics_invariant
@pytest.mark.smoke
def test_tosi_thermal_evolution_parity(shared_eos):
    if heat_budget is None:
        pytest.skip("Nicola Tosi's heat_budget model not found.")

    tb = heat_budget.interior_evolution(body='Earth')
    tb.tectonics = 'SL'
    tb.core_cooling = 'yes'
    tb.Rp = 6371.0e3
    tb.Rc = 3480.0e3
    tb.g = 9.8
    tb.rhom = 4500.0
    tb.rhoc = 7190.0
    tb.Ts = 300.0
    tb.Tm0 = 1900.0
    tb.Tc0 = 3500.0
    tb.etaref = 1e21
    tb.km = 4.0

    tb.X_U = 0.0
    tb.X_Th = 0.0
    tb.X_K = 0.0
    tb.Q0 = 0.0

    # Fast test
    end_time_yr = 1000.0
    tb.maxtime = end_time_yr * tb.yrs
    tb.dt = 10.0 * tb.yrs
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
    parameters.boundary_conditions.outer_boundary_value = tb.Ts
    parameters.boundary_conditions.equilibrium_temperature = tb.Ts
    parameters.boundary_conditions.core_heat_capacity = tb.cc
    parameters.phase_solid.density = tb.rhom
    parameters.phase_solid.heat_capacity = tb.cm
    parameters.phase_solid.thermal_conductivity = tb.km
    parameters.phase_solid.viscosity = tb.etaref
    parameters.phase_solid.arrhenius_t_ref = tb.Tref
    parameters.phase_solid.activation_energy = tb.E
    parameters.phase_solid.activation_volume = tb.V

    parameters.solver.max_time_step = 10.0
    parameters.solver.output_interval = 1.0e6
    parameters.solver.rtol = 1e-4
    parameters.solver.atol = 1e-4

    parameters.initial_condition.initial_condition = 1
    parameters.initial_condition.surface_temperature = tb.Tm0

    solver = EntropySolver(parameters, entropy_eos=shared_eos)
    solver.initialize()
    S_init = _derive_initial_entropy_from_config(solver)
    solver.set_initial_entropy(S_init)
    solver.solve()
    out = solver.get_state()

    bulk_entropy = np.median(out.S_final)
    P_surf = float(parameters.mesh.surface_pressure)
    aragog_final_T = shared_eos.temperature_scalar(P_surf, bulk_entropy)
    tosi_final_T = tosi_Tm[-1]

    assert abs(aragog_final_T - tosi_final_T) < 150.0, (
        f'Divergence too large: aragog={aragog_final_T:.1f} K, Tosi={tosi_final_T:.1f} K'
    )
