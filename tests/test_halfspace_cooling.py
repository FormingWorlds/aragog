"""Verification test for thermal boundary layer half-space cooling (Step 3).

Verifies that pure heat conduction into a semi-infinite solid mantle matches the
analytical error-function solution:
    z_iso(t) = 2 * erfinv(theta) * sqrt(kappa * t)

Physical requirements from solidstate_convection_roadmap.md:
- const_properties mode with alpha = 0 (uniform temperature is isentropic).
- Convection disabled, pure conduction active.
- Fixed surface temperature boundary condition (outer_boundary_condition = 5).
- Curvature condition: 2 * sqrt(kappa * t) < 0.02 * R.
- Boundary layer resolved with at least 10 cells.
- Fitted exponent beta = 0.50 +- 0.02.
- Prefactor within tolerance derived from 3-level resolution study:
    Level 1: N = 50,  dz = 2.0 km, beta = 0.4960, A = 5.6263, error = 5.00%
    Level 2: N = 100, dz = 1.0 km, beta = 0.4979, A = 5.4921, error = 2.49%
    Level 3: N = 200, dz = 0.5 km, beta = 0.4990, A = 5.4216, error = 1.18%
  Grid convergence is first order in dz. The production test runs at N = 100
  with prefactor tolerance 3.0%.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.special import erfinv

from aragog.solver.entropy_solver import EntropySolver
from tests.test_entropy_solver_const_properties_smoke import (
    _build_const_properties_parameters,
)


@pytest.mark.smoke
def test_halfspace_cooling_boundary_layer():
    """Verify half-space cooling boundary layer against analytical error function."""
    depth_total = 100.0e3  # 100 km deep domain
    r_outer = 6.371e6  # Earth radius [m]
    r_inner = r_outer - depth_total
    n_nodes = 100
    dz = depth_total / n_nodes  # 1.0 km cell size

    # Thermal diffusivity kappa [m^2/yr]
    rho = 4000.0
    cp = 1000.0
    k_cond = 4.0
    sec_per_yr = 365.25 * 86400.0
    kappa = (k_cond / (rho * cp)) * sec_per_yr

    # Target non-dimensional isotherm theta = (T - T_surf) / (T_0 - T_surf) = 0.5
    theta_target = 0.5
    a_analytical = 2.0 * float(erfinv(theta_target)) * np.sqrt(kappa)

    # Verification times [yr]
    times = [1.0e6, 2.0e6, 4.0e6]
    z_iso_vals = []

    for t in times:
        params = _build_const_properties_parameters(n_nodes=n_nodes, end_time=t)
        params.mesh.outer_radius = r_outer
        params.mesh.inner_radius = r_inner

        # alpha = 0 ensures uniform temperature is isentropic (no adiabatic gradient)
        params.phase_mixed.const_alpha = 0.0
        params.phase_solid.thermal_expansivity = 0.0
        params.phase_liquid.thermal_expansivity = 0.0

        params.energy.convection = False
        params.energy.conduction = True

        # Surface: prescribed temperature 1000 K
        params.boundary_conditions.outer_boundary_condition = 5
        params.boundary_conditions.outer_boundary_value = 1000.0

        # Core-mantle boundary: zero flux (insulating)
        params.boundary_conditions.inner_boundary_condition = 2
        params.boundary_conditions.inner_boundary_value = 0.0

        # Initial uniform temperature 2000 K
        t_0 = 2000.0
        t_ref = params.phase_mixed.const_T_ref
        s_ref = params.phase_mixed.const_S_ref
        const_cp = params.phase_mixed.const_Cp
        s_0 = s_ref + const_cp * np.log(t_0 / t_ref)

        solver = EntropySolver(params, entropy_eos=None)
        solver.initialize()
        solver.set_initial_entropy(s_0)
        solver.solve()

        assert solver._solution.status == 0, 'Solver did not complete integration'

        state = solver.get_state()
        t_final = state.T_stag
        z_stag = r_outer - state.r_stag

        # Include surface boundary node at z = 0 with T = 1000 K
        z_all = np.concatenate([[0.0], z_stag[::-1]])
        t_all = np.concatenate([[1000.0], t_final[::-1]])

        interp = interp1d(t_all, z_all)
        z_iso = float(interp(1500.0))
        z_iso_vals.append(z_iso)

    # 1. Curvature condition: 2 * sqrt(kappa * t) < 0.02 * R
    max_diffusion_depth = 2.0 * np.sqrt(kappa * times[-1])
    assert max_diffusion_depth < 0.02 * r_outer, (
        f'Curvature limit violated: {max_diffusion_depth} >= {0.02 * r_outer}'
    )

    # 2. Boundary layer resolution: at least 10 cells across boundary layer at end time
    cells_across = z_iso_vals[-1] / dz
    assert cells_across >= 10.0, f'Expected >= 10 cells across layer, got {cells_across:.1f}'

    # 3. Fit power law z = A * t^beta
    log_t = np.log(times)
    log_z = np.log(z_iso_vals)
    slope, intercept = np.polyfit(log_t, log_z, 1)
    a_fitted = float(np.exp(intercept))

    # Fitted exponent must match 0.50 within +- 0.02
    assert abs(slope - 0.50) <= 0.02, f'Fitted exponent {slope:.4f} outside [0.48, 0.52]'

    # Prefactor within 3.0% tolerance derived from 3-level resolution study
    rel_err_a = abs(a_fitted - a_analytical) / a_analytical
    assert rel_err_a < 0.03, f'Prefactor error {rel_err_a * 100:.2f}% exceeds 3.0% limit'
