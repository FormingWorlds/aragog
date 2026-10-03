"""Tests for solver option max_step_const_mode.

Verifies:
1. _SolverParameters has max_step_const_mode option (default 100.0 years).
2. _SolverParameters accepts positive values and infinity, and validates inputs.
3. In const_properties mode, max_step_const_mode governs CVODE max_step:
   setting infinity allows large steps and fewer total integration steps.
4. Default path behaviour is bitwise unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aragog.parser import (
    Parameters,
    _BoundaryConditionsParameters,
    _EnergyParameters,
    _InitialConditionParameters,
    _MeshParameters,
    _PhaseMixedParameters,
    _PhaseParameters,
    _SolverParameters,
)
from aragog.solver.entropy_solver import EntropySolver


@pytest.mark.unit
def test_solver_parameters_max_step_const_mode_default():
    """Verify default max_step_const_mode is 100.0 years."""
    sp = _SolverParameters(start_time=0.0, end_time=1.0, atol=1e-6, rtol=1e-6)
    assert hasattr(sp, 'max_step_const_mode')
    assert sp.max_step_const_mode == 100.0


@pytest.mark.unit
def test_solver_parameters_max_step_const_mode_validation():
    """Verify validation of max_step_const_mode for infinity and invalid values."""
    sp_inf = _SolverParameters(
        start_time=0.0,
        end_time=1.0,
        atol=1e-6,
        rtol=1e-6,
        max_step_const_mode=float('inf'),
    )
    assert math.isinf(sp_inf.max_step_const_mode)

    sp_custom = _SolverParameters(
        start_time=0.0,
        end_time=1.0,
        atol=1e-6,
        rtol=1e-6,
        max_step_const_mode=500.0,
    )
    assert sp_custom.max_step_const_mode == 500.0

    sp_str = _SolverParameters(
        start_time=0.0,
        end_time=1.0,
        atol=1e-6,
        rtol=1e-6,
        max_step_const_mode='250.0',
    )
    assert sp_str.max_step_const_mode == 250.0

    with pytest.raises(TypeError, match='max_step_const_mode must be a float'):
        _SolverParameters(
            start_time=0.0,
            end_time=1.0,
            atol=1e-6,
            rtol=1e-6,
            max_step_const_mode='abc',
        )

    with pytest.raises(TypeError, match='max_step_const_mode must be a float'):
        _SolverParameters(
            start_time=0.0,
            end_time=1.0,
            atol=1e-6,
            rtol=1e-6,
            max_step_const_mode=True,
        )

    with pytest.raises(ValueError, match='max_step_const_mode must be > 0'):
        _SolverParameters(
            start_time=0.0,
            end_time=1.0,
            atol=1e-6,
            rtol=1e-6,
            max_step_const_mode=float('nan'),
        )

    with pytest.raises(ValueError, match='max_step_const_mode must be > 0'):
        _SolverParameters(
            start_time=0.0,
            end_time=1.0,
            atol=1e-6,
            rtol=1e-6,
            max_step_const_mode=0.0,
        )

    with pytest.raises(ValueError, match='max_step_const_mode must be > 0'):
        _SolverParameters(
            start_time=0.0,
            end_time=1.0,
            atol=1e-6,
            rtol=1e-6,
            max_step_const_mode=-10.0,
        )


def _build_const_mode_params(
    max_step_const_mode: float | None = None, end_time: float = 1000.0
) -> Parameters:
    """Build a minimal constant-properties simulation parameter set."""
    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,
        outer_boundary_value=288.0,
        inner_boundary_condition=2,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=288.0,
        core_heat_capacity=840.0,
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='cvode',
        use_jax_jacobian=False,
        kappah_floor=0.0,
    )
    ic = _InitialConditionParameters(
        initial_condition=1,
        surface_temperature=1700.0,
        basal_temperature=1700.0,
    )
    mesh = _MeshParameters(
        outer_radius=6370e3,
        inner_radius=3480e3,
        number_of_nodes=20,
        mixing_length_profile='nearest_boundary',
        core_density=10640.0,
        surface_density=4460.0,
        gravitational_acceleration=9.8,
        mass_coordinates=False,
    )
    common = dict(
        density=4460.0,
        heat_capacity=1200.0,
        thermal_conductivity=3.0,
        thermal_expansivity=3e-5,
    )
    ps = _PhaseParameters(
        melt_fraction=0.0,
        viscosity=1e19,
        enabled=True,
        activation_energy=3e5,
        activation_volume=2.5e-6,
        activation_volume_decay_pressure=float('inf'),
        arrhenius_t_ref=1600.0,
        yield_stress_c=1e30,
        yield_stress_max=1e30,
        **common,
    )
    pl = _PhaseParameters(melt_fraction=1.0, viscosity=1e2, **common)
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1e-3,
        matprop_smooth_width=0.01,
        const_properties=True,
        const_rho=4460.0,
        const_Cp=1200.0,
        const_alpha=3e-5,
        const_cond=3.0,
        const_log10visc=19.0,
        const_T_ref=1600.0,
        const_S_ref=3000.0,
    )
    solver_kwargs = dict(
        start_time=0.0,
        end_time=end_time,
        atol=1e-6,
        rtol=1e-6,
    )
    if max_step_const_mode is not None:
        solver_kwargs['max_step_const_mode'] = max_step_const_mode
    sv = _SolverParameters(**solver_kwargs)
    return Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        radionuclides=[],
        solver=sv,
    )


@pytest.mark.unit
def test_const_mode_respects_max_step_const_mode():
    """Verify max_step_const_mode changes maximum CVODE step in const mode."""
    # 1. With default max_step_const_mode=100.0 over 50000 years,
    # CVODE cannot take steps larger than 100 years.
    p_default = _build_const_mode_params(max_step_const_mode=100.0, end_time=50000.0)
    s_default = EntropySolver(p_default, entropy_eos=None)
    s_default.initialize()
    s_default.set_initial_entropy(3000.0)
    s_default.solve()
    steps_default = s_default.solution.cvode_nst
    last_step_default = s_default.solution.cvode_last_step

    # 2. With max_step_const_mode=inf, CVODE takes steps much larger than 100 yr
    # resulting in significantly fewer steps over the same smooth interval.
    p_inf = _build_const_mode_params(max_step_const_mode=float('inf'), end_time=50000.0)
    s_inf = EntropySolver(p_inf, entropy_eos=None)
    s_inf.initialize()
    s_inf.set_initial_entropy(3000.0)
    s_inf.solve()
    steps_inf = s_inf.solution.cvode_nst
    last_step_inf = s_inf.solution.cvode_last_step

    assert last_step_default <= 100.0 + 1e-4, f'Expected step <= 100, got {last_step_default}'
    assert last_step_inf > 100.0, f'Expected step > 100 with inf cap, got {last_step_inf}'
    assert steps_inf < steps_default, (
        f'Expected fewer steps with inf cap: {steps_inf} vs {steps_default}'
    )


@pytest.mark.unit
def test_default_path_bitwise_unchanged_with_default_max_step():
    """Verify solver output with default max_step_const_mode=100.0 is bitwise identical."""
    p1 = _build_const_mode_params(max_step_const_mode=None, end_time=100.0)
    assert p1.solver.max_step_const_mode == 100.0
    s1 = EntropySolver(p1, entropy_eos=None)
    s1.initialize()
    s1.set_initial_entropy(3000.0)
    s1.solve()
    st1 = s1.get_state()

    p2 = _build_const_mode_params(max_step_const_mode=100.0, end_time=100.0)
    s2 = EntropySolver(p2, entropy_eos=None)
    s2.initialize()
    s2.set_initial_entropy(3000.0)
    s2.solve()
    st2 = s2.get_state()

    np.testing.assert_array_equal(st1.S_final, st2.S_final)
    np.testing.assert_array_equal(st1.T_stag, st2.T_stag)
    assert s1.solution.cvode_nst == s2.solution.cvode_nst
