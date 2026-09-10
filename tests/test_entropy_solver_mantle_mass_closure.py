"""Mass-closure invariant for the melt/solid mantle mass split.

``EntropySolver.get_state()`` reports ``M_mantle`` from the structural
(Adams-Williamson or discrete) density field, but historically split it
into ``M_mantle_liquid`` / ``M_mantle_solid`` using a separate PALEOS
density field (``mass_stag = rho_stag * vol``). When the two density
fields disagree, the split can violate the individual bounds
``0 <= M_mantle_liquid <= M_mantle`` (equivalently ``M_mantle_solid``
goes negative), even though the sum ``M_mantle_liquid + M_mantle_solid``
always equals ``M_mantle`` by construction and cannot itself detect
this failure mode.

This test forces that density disagreement deterministically via a low
``surface_density`` (Adams-Williamson) against a higher ``const_rho``
(PALEOS), rather than relying on a solve running long enough to drift
there naturally, so the regression is fast and does not depend on
integration length.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _build_mismatched_density_parameters(*, n_nodes: int = 10, end_time: float = 5.0):
    """Const-properties Parameters with AW surface density well below
    the constant PALEOS density, so the two mass fields disagree
    everywhere in the mesh rather than only near a table edge.
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
        # Well below const_rho below: AW mass integral (M_mantle) comes
        # out smaller than the PALEOS mass sum (rho_stag * vol) at
        # every node, reproducing the field disagreement in #33.
        surface_density=2000.0,
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
        const_log10visc=2.0,
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


@pytest.mark.physics_invariant
def test_mantle_liquid_solid_split_conserves_against_reported_m_mantle():
    """The melt/solid split must stay within ``[0, M_mantle]`` and sum
    back to the SAME ``M_mantle`` the helpfile reports, even when the
    AW structural density and the PALEOS density disagree.

    Discriminator: with ``surface_density`` well below ``const_rho``,
    the PALEOS-weighted mass sum exceeds the AW ``M_mantle`` at every
    node. A split that computes ``M_mantle_liquid`` from the PALEOS
    field independently of ``M_mantle`` produces ``M_mantle_liquid >
    M_mantle`` and a negative ``M_mantle_solid``; a split derived from
    ``Phi_global * M_mantle`` cannot.
    """
    from aragog.solver.entropy_solver import EntropySolver

    parameters = _build_mismatched_density_parameters(n_nodes=10, end_time=5.0)
    solver = EntropySolver(parameters, entropy_eos=None)
    solver.initialize()
    solver.set_initial_entropy(3050.0)
    solver.solve()
    out = solver.get_state()

    assert out.M_mantle > 0.0, 'M_mantle must be positive for this check to be meaningful'

    assert 0.0 <= out.M_mantle_liquid <= out.M_mantle, (
        f'M_mantle_liquid={out.M_mantle_liquid:.6e} escaped [0, M_mantle='
        f'{out.M_mantle:.6e}]; the melt split used a mass field '
        'inconsistent with the reported M_mantle.'
    )
    assert 0.0 <= out.M_mantle_solid <= out.M_mantle, (
        f'M_mantle_solid={out.M_mantle_solid:.6e} escaped [0, M_mantle={out.M_mantle:.6e}].'
    )
    np.testing.assert_allclose(
        out.M_mantle_liquid + out.M_mantle_solid,
        out.M_mantle,
        rtol=1e-10,
        err_msg='M_mantle_liquid + M_mantle_solid must equal the reported M_mantle',
    )
