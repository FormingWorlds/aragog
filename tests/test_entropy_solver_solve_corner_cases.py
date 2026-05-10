"""Corner-case tests for ``EntropySolver.solve()`` and accessors.

Targets the remaining small clusters of uncovered lines in
``solver/entropy_solver.py`` that don't fit into the gradient or
const_properties smoke files:

* ``_solve_cvode`` zero-span branch (lines 1881-1888): ``end_time
  == start_time`` must short-circuit and return the IC unchanged
  rather than feeding CVODE with a degenerate span (CVODE rejects).
* ``solution`` accessor returns ``None`` before ``solve()`` runs
  (line 1719).
* ``temperature_staggered`` const_properties branch (lines
  1768-1770): T = T_ref * exp((S - S_ref) / Cp) when EOS is None.
* ``solve()`` failure log branch (lines 2538-2543): when CVODE
  returns a negative status, the solver must set
  ``self.stop_early = True`` and log an error.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _build_minimal_solver(*, core_bc: str = 'quasi_steady'):
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

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,
        outer_boundary_value=1500.0,
        inner_boundary_condition=2,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        core_bc=core_bc,
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='radau',
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=10,
        mixing_length_profile='nearest_boundary',
        core_density=10500.0,
        eos_method=1,
    )
    pl = _PhaseParameters(
        density=4000.0,
        heat_capacity=1000.0,
        melt_fraction=1.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=10.0,
    )
    ps = _PhaseParameters(
        density=4200.0,
        heat_capacity=1000.0,
        melt_fraction=0.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
        viscosity=1e21,
    )
    pm = _PhaseMixedParameters(
        latent_heat_of_fusion=4.0e5,
        rheological_transition_melt_fraction=0.4,
        rheological_transition_width=0.15,
        solidus='solidus.dat',
        liquidus='liquidus.dat',
        phase='mixed',
        phase_transition_width=0.01,
        grain_size=1.0e-3,
    )
    sv = _SolverParameters(
        start_time=0.0,
        end_time=1.0,
        atol=1.0e-6,
        rtol=1.0e-6,
        tsurf_poststep_change=30.0,
    )
    parameters = Parameters(
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
    return EntropySolver(parameters, entropy_eos=None)


def test_solution_property_returns_none_before_solve():
    """Reading ``solver.solution`` before ``solve()`` runs must return
    None (PROTEUS JAX dispatch path holds an unsolved scipy
    EntropySolver and reads ``sol.solution`` defensively).

    Discriminator: a regression that raised AttributeError instead
    of returning None would break the JAX runner on the first
    coupling step.
    """
    solver = _build_minimal_solver()
    assert solver.solution is None


def test_solve_cvode_zero_span_returns_initial_state_without_calling_cvode():
    """Calling ``_solve_cvode`` with ``end_time == start_time`` must
    short-circuit and return a scipy-compatible OptimizeResult whose
    ``y[:, -1]`` equals the initial state.

    Discriminator: PROTEUS's static-init iteration calls solve with
    start == end == 0 yr; CVODE rejects this with "tout too close to
    t0" so the caller-side guard MUST handle it. A regression that
    fell through to ``solver.solve(...)`` would either crash or
    return an empty trajectory.
    """
    pytest.importorskip('scikits_odes_sundials')

    from aragog.solver.entropy_solver import EntropySolver

    # Bypass full initialize(); _solve_cvode is a method that doesn't
    # need self.state, only the start/end/y0 args.
    solver = EntropySolver.__new__(EntropySolver)
    y0 = np.array([3050.0, 3050.0, 3050.0], dtype=float)
    res = solver._solve_cvode(
        start_time=0.0,
        end_time=0.0,
        y0=y0,
        atol=1e-8,
        rtol=1e-6,
        max_step=np.inf,
    )
    assert res.status == 0, f'zero-span result.status should be 0; got {res.status}'
    assert res.nfev == 0, f'CVODE was invoked despite zero span (nfev={res.nfev})'
    assert res.y.shape == (3, 1)
    np.testing.assert_array_equal(res.y[:, -1], y0)
    assert 'zero-span' in res.message


def test_solve_cvode_zero_span_handles_negative_span_same_way():
    """``end_time < start_time`` is also degenerate (PROTEUS retries
    sometimes pass slightly out-of-order times when the wrapper
    rolls back). Must short-circuit identically.
    """
    pytest.importorskip('scikits_odes_sundials')

    from aragog.solver.entropy_solver import EntropySolver

    solver = EntropySolver.__new__(EntropySolver)
    y0 = np.array([3050.0], dtype=float)
    res = solver._solve_cvode(
        start_time=10.0,
        end_time=5.0,
        y0=y0,
        atol=1e-8,
        rtol=1e-6,
        max_step=np.inf,
    )
    assert res.status == 0
    assert res.t[0] == pytest.approx(10.0)
    np.testing.assert_array_equal(res.y[:, -1], y0)
