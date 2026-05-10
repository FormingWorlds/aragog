"""Smoke test for the SPIDER ``-use_const_properties`` parity path.

The const_properties branch of ``EntropySolver`` and
``EntropyPhaseEvaluator`` is a structurally separate code path from the
EOS-table-backed phase evaluator: T(S) is analytic
(``T_ref * exp((S - S_ref) / Cp)``), all material properties are
constants, the phase always reports ``phi=1``, and ``entropy_eos``
remains ``None`` so the solver dispatches through the const-properties
branches in ``_initialize_internals`` (lines 899-909) and ``get_state``
(eos=None branches at 2797-2800, 2931-2932, 2963).

A single short integration here exercises that whole branch, plus the
matching ``EntropyPhaseEvaluator._update_const`` body in entropy_phase.py
(141-176) which has zero direct unit-test exposure.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _build_const_properties_parameters(*, n_nodes: int = 10, end_time: float = 5.0):
    """Construct Parameters that select the const_properties code path.

    No EOS tables; ``EntropyEOS`` is NOT supplied and the solver
    dispatches through the analytical T(S) branch end-to-end.
    """
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

    bc = _BoundaryConditionsParameters(
        outer_boundary_condition=1,
        outer_boundary_value=1500.0,
        inner_boundary_condition=2,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=880.0,
        core_bc='quasi_steady',
    )
    en = _EnergyParameters(
        conduction=True,
        convection=True,
        gravitational_separation=False,
        mixing=False,
        radionuclides=False,
        tidal=False,
        solver_method='radau',
        use_jax_jacobian=False,
    )
    ic = _InitialConditionParameters(
        initial_condition=1, surface_temperature=3500.0, basal_temperature=3500.0
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=n_nodes,
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
        const_properties=True,
        const_rho=4000.0,
        const_Cp=1000.0,
        const_alpha=3.0e-5,
        const_cond=4.0,
        const_log10visc=2.0,  # log10(viscosity Pa s)
        const_T_ref=3000.0,
        const_S_ref=3000.0,
    )
    sv = _SolverParameters(
        start_time=0.0,
        end_time=end_time,
        atol=1.0e-6,
        rtol=1.0e-6,
        tsurf_poststep_change=30.0,
    )
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


def test_const_properties_smoke_completes_without_eos():
    """End-to-end const_properties run without an EntropyEOS object.

    Exercises the analytic ``T = T_ref * exp((S - S_ref) / Cp)`` branch
    in ``EntropyPhaseEvaluator._update_const`` (entropy_phase.py:144-176)
    and the matching const_properties branches in EntropySolver:
      * ``_initialize_internals`` const-properties phase setup (899-909)
      * ``get_state`` analytic T_stag / phi=1 / rho=const fallback
        (2797-2800)
      * ``get_state`` E_state=NaN / E_state_cons=NaN when eos is None
        (2931-2932)
      * ``get_state`` Phi_global_vol=1.0 fallback (2963)

    Discriminator: the post-solve ``T_stag`` profile must satisfy the
    analytic ``T_ref * exp((S - S_ref) / Cp)`` law to machine precision
    against the recovered S_final, and ``E_state`` must be NaN (signal
    to PROTEUS that the conservation diagnostic is unavailable).
    """
    from aragog.solver.entropy_solver import EntropySolver

    parameters = _build_const_properties_parameters(n_nodes=10, end_time=5.0)
    # entropy_eos=None is the const_properties contract.
    solver = EntropySolver(parameters, entropy_eos=None)
    solver.initialize()
    solver.set_initial_entropy(3050.0)
    solver.solve()

    # The integrator produced a finite final state.
    assert solver._solution.t.size >= 2, 'const_properties solve produced <2 time points'
    final_y = solver._solution.y[:, -1] if solver._solution.y.ndim == 2 else solver._solution.y
    assert np.all(np.isfinite(final_y)), 'const_properties final state has NaN/inf'

    out = solver.get_state()

    # Discriminator: T_stag must equal T_ref * exp((S_final - S_ref) / Cp)
    # to machine precision (no EOS table interpolation involved).
    pm = parameters.phase_mixed
    T_analytic = pm.const_T_ref * np.exp((out.S_final - pm.const_S_ref) / pm.const_Cp)
    np.testing.assert_allclose(out.T_stag, T_analytic, rtol=1e-12, atol=0.0)

    # Discriminator: phi must be uniformly 1 (const_properties = "always
    # liquid" diagnostic).
    np.testing.assert_array_equal(out.phi_stag, np.ones_like(out.phi_stag))

    # Discriminator: rho must be uniformly const_rho.
    np.testing.assert_allclose(out.rho_stag, pm.const_rho, rtol=1e-12, atol=0.0)

    # Discriminator: E_state and E_state_cons must be NaN (no EOS).
    assert np.isnan(out.E_state), (
        f'const_properties E_state should be NaN; got {out.E_state}. '
        'PROTEUS reads NaN to mean "conservation diagnostic unavailable" '
        'and a finite value here would silently corrupt the helpfile.'
    )
    assert np.isnan(out.E_state_cons), (
        f'const_properties E_state_cons should be NaN; got {out.E_state_cons}.'
    )

    # Discriminator: Phi_global_vol falls back to 1.0 when EOS is None.
    assert float(out.Phi_global_vol) == pytest.approx(1.0)

    # Sanity: cooling went the expected direction (top should drop
    # below the IC, since the surface BC pulls heat out as a grey body).
    assert float(out.T_magma) < pm.const_T_ref * np.exp(
        (3050.0 - pm.const_S_ref) / pm.const_Cp
    ), 'T_magma did not drop below the IC; the grey-body BC must remove heat over 5 yr.'
