"""Tests for CMB convective flux sign gate under Ruling 35 and Ruling 36.

Ruling 35/36 requires:
1. Convective heat flux is gated by the sign of dS/dr at every node (j_conv = 0 where dS/dr >= 0).
2. A stable-CMB boundary layer where CMB convective flux vanishes and total flux equals conductive flux.
3. A convecting case bitwise unchanged in every flux column.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)

pytestmark = [pytest.mark.smoke, needs_eos]


@pytest.fixture(scope='module')
def shared_eos():
    from aragog.eos.entropy import EntropyEOS

    return EntropyEOS(EOS_DIR)


def _build_params(
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    cvode_output_points: int = 65,
    dt_yr: float = 100.0,
    n_nodes: int = 12,
):
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
        outer_boundary_condition=5,
        outer_boundary_value=1600.0,
        inner_boundary_condition=1,
        inner_boundary_value=0.0,
        emissivity=1.0,
        equilibrium_temperature=255.0,
        core_heat_capacity=840.0,
        core_bc='energy_balance',
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
    )
    ic = _InitialConditionParameters(
        initial_condition=1,
        surface_temperature=3500.0,
        basal_temperature=5500.0,
    )
    mesh = _MeshParameters(
        outer_radius=6.371e6,
        inner_radius=3.480e6,
        number_of_nodes=n_nodes,
        mass_coordinates=False,
        mixing_length_profile='nearest_boundary',
        core_density=12500.0,
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
    sol = _SolverParameters(
        start_time=0.0,
        end_time=dt_yr,
        cvode_output_points=cvode_output_points,
        atol=atol,
        rtol=rtol,
    )
    return Parameters(
        boundary_conditions=bc,
        energy=en,
        initial_condition=ic,
        mesh=mesh,
        phase_solid=ps,
        phase_liquid=pl,
        phase_mixed=pm,
        solver=sol,
        radionuclides=[],
    )


def test_stable_cmb_boundary_convective_flux_vanishes(shared_eos):
    """In a stable CMB boundary layer (dS/dr >= 0), convective heat flux at node 0 must be 0."""
    from aragog.solver import EntropySolver

    params = _build_params(rtol=1e-8, atol=1e-8, cvode_output_points=65)
    solver = EntropySolver(params, entropy_eos=shared_eos)
    solver.initialize()

    # Set initial entropy profile with convecting interior and stable CMB layer
    S_profile = np.linspace(3100.0, 3000.0, solver._n_stag)
    # Impose a positive (stable) entropy gradient at CMB basic node 0
    solver.set_initial_dSdr_cmb(1.0e-6)
    solver.set_initial_entropy(S_profile)

    # Evaluate state
    solver._dSdt_single(0.0, solver._S0)

    # Eddy diffusivity is borrowed from node 1 (which convects)
    assert solver.state.eddy_diffusivity[0] > 0.0

    # Convective flux must be gated to 0 where dS/dr >= 0
    assert solver.state.jconv[0] == 0.0, (
        f'jconv[0] must vanish when dSdr[0] >= 0; got {solver.state.jconv[0]:.6e}'
    )
    # Total heat flux at CMB must equal conductive heat flux alone
    assert solver.state.heat_flux[0] == pytest.approx(solver.state.jcond[0], abs=1e-15)


def test_convective_case_fluxes_bitwise_unchanged(shared_eos):
    """When the entire mantle convects (dS/dr < 0 everywhere), convective flux is unchanged."""
    from aragog.solver import EntropySolver

    params = _build_params(rtol=1e-8, atol=1e-8, cvode_output_points=65)
    solver = EntropySolver(params, entropy_eos=shared_eos)
    solver.initialize()

    # Superadiabatic (convective) profile: entropy decreases with radius
    S_profile = np.linspace(3200.0, 3000.0, solver._n_stag)
    solver.set_initial_dSdr_cmb(-1.0e-6)
    solver.set_initial_entropy(S_profile)

    # Evaluate state
    solver._dSdt_single(0.0, solver._S0)

    # Verify dS/dr is negative everywhere
    assert np.all(solver.state.dSdr < 0.0)

    # Convective heat flux matches the ungated formula bitwise
    ungated_jconv = (
        np.asarray(solver.state.phase_basic.density()).ravel()
        * np.asarray(solver.state.phase_basic.temperature()).ravel()
        * solver.state.eddy_diffusivity
        * (-solver.state.dSdr)
    )
    assert np.array_equal(solver.state.jconv, ungated_jconv)
