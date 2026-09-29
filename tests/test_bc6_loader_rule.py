"""Tests for the outer boundary condition 6 loader rule.

Outer boundary condition 6 (conductive surface skin) requires opt-in
surface cell refinement (``mesh.surface_cell_thickness > 0``). Evaluating
the surface skin condition on a uniform mesh chokes the conductive flux.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aragog.config import Config
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

pytestmark = pytest.mark.unit

_EXPECTED_ERROR = re.escape(
    '[boundary_conditions] outer_boundary_condition = 6 (conductive surface skin) '
    'requires [mesh] surface_cell_thickness > 0 (got 0.0); '
    'set surface_cell_thickness (e.g. 1000.0 m) to resolve the surface skin layer'
)


def _minimal_params(*, outer_bc: int = 6, surface_cell_thickness: float = 0.0) -> Parameters:
    common = dict(
        density=4000.0,
        heat_capacity=1000.0,
        thermal_conductivity=4.0,
        thermal_expansivity=3e-5,
    )
    return Parameters(
        boundary_conditions=_BoundaryConditionsParameters(
            outer_boundary_condition=outer_bc,
            outer_boundary_value=273.0,
            inner_boundary_condition=2,
            inner_boundary_value=0.0,
            emissivity=1.0,
            equilibrium_temperature=273.0,
            core_heat_capacity=880.0,
            core_bc='quasi_steady',
        ),
        energy=_EnergyParameters(
            conduction=True,
            convection=False,
            gravitational_separation=False,
            mixing=False,
            radionuclides=False,
            tidal=False,
            solver_method='cvode',
            use_jax_jacobian=False,
        ),
        initial_condition=_InitialConditionParameters(
            initial_condition=1,
            surface_temperature=1500.0,
            basal_temperature=1500.0,
        ),
        mesh=_MeshParameters(
            outer_radius=6.371e6,
            inner_radius=3.48e6,
            number_of_nodes=100,
            mixing_length_profile='nearest_boundary',
            core_density=4000.0,
            surface_density=4000.0,
            mass_coordinates=False,
            surface_cell_thickness=surface_cell_thickness,
            cmb_cell_thickness=0.0,
        ),
        phase_solid=_PhaseParameters(melt_fraction=0.0, viscosity=1e21, **common),
        phase_liquid=_PhaseParameters(melt_fraction=1.0, viscosity=1e2, **common),
        phase_mixed=_PhaseMixedParameters(
            latent_heat_of_fusion=4.0e5,
            rheological_transition_melt_fraction=0.4,
            rheological_transition_width=0.15,
            solidus='solidus.dat',
            liquidus='liquidus.dat',
            phase='mixed',
            phase_transition_width=0.01,
            grain_size=1.0e-3,
        ),
        radionuclides=[],
        solver=_SolverParameters(start_time=0.0, end_time=1.0, atol=1e-8, rtol=1e-8),
    )


_BASE_TOML = """
[boundary_conditions]
outer_boundary_condition = {outer_bc}
outer_boundary_value = 273.0
inner_boundary_condition = 2
inner_boundary_value = 0.0
emissivity = 1.0
equilibrium_temperature = 273.0
core_heat_capacity = 880.0
core_bc = "quasi_steady"

[energy]
conduction = true
convection = false
gravitational_separation = false
mixing = false
radionuclides = false
tidal = false
solver_method = "cvode"
use_jax_jacobian = false

[initial_condition]
initial_condition = 1
surface_temperature = 1500.0
basal_temperature = 1500.0

[mesh]
outer_radius = 6371e3
inner_radius = 3480e3
number_of_nodes = 100
mixing_length_profile = "nearest_boundary"
core_density = 4000.0
surface_density = 4000.0
mass_coordinates = false
surface_cell_thickness = {surface_cell_thickness}

[phase_solid]
density = 4000.0
heat_capacity = 1000.0
thermal_conductivity = 4.0
thermal_expansivity = 3e-5
melt_fraction = 0.0
viscosity = 1e21

[phase_liquid]
density = 4000.0
heat_capacity = 1000.0
thermal_conductivity = 4.0
thermal_expansivity = 3e-5
melt_fraction = 1.0
viscosity = 1e2

[phase_mixed]
latent_heat_of_fusion = 4e5
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = "solidus.dat"
liquidus = "liquidus.dat"
phase = "mixed"
phase_transition_width = 0.01
grain_size = 1e-3

[solver]
start_time = 0.0
end_time = 1.0
atol = 1e-8
rtol = 1e-8
"""

_BASE_INI = """
[boundary_conditions]
outer_boundary_condition = {outer_bc}
outer_boundary_value = 273.0
inner_boundary_condition = 2
inner_boundary_value = 0.0
emissivity = 1.0
equilibrium_temperature = 273.0
core_heat_capacity = 880.0
core_bc = quasi_steady

[energy]
conduction = true
convection = false
gravitational_separation = false
mixing = false
radionuclides = false
tidal = false
solver_method = cvode
use_jax_jacobian = false

[initial_condition]
initial_condition = 1
surface_temperature = 1500.0
basal_temperature = 1500.0

[mesh]
outer_radius = 6371e3
inner_radius = 3480e3
number_of_nodes = 100
mixing_length_profile = nearest_boundary
core_density = 4000.0
surface_density = 4000.0
mass_coordinates = false
surface_cell_thickness = {surface_cell_thickness}

[phase_solid]
density = 4000.0
heat_capacity = 1000.0
thermal_conductivity = 4.0
thermal_expansivity = 3e-5
melt_fraction = 0.0
viscosity = 1e21

[phase_liquid]
density = 4000.0
heat_capacity = 1000.0
thermal_conductivity = 4.0
thermal_expansivity = 3e-5
melt_fraction = 1.0
viscosity = 1e2

[phase_mixed]
latent_heat_of_fusion = 4e5
rheological_transition_melt_fraction = 0.4
rheological_transition_width = 0.15
solidus = solidus.dat
liquidus = liquidus.dat
phase = mixed
phase_transition_width = 0.01
grain_size = 1e-3

[solver]
start_time = 0.0
end_time = 1.0
atol = 1e-8
rtol = 1e-8
"""


def test_bc6_without_surface_cell_thickness_raises():
    """Outer BC 6 with surface_cell_thickness = 0.0 raises ValueError."""
    with pytest.raises(ValueError, match=_EXPECTED_ERROR):
        _minimal_params(outer_bc=6, surface_cell_thickness=0.0)


def test_bc6_with_surface_cell_thickness_passes():
    """Outer BC 6 with surface_cell_thickness = 1000.0 constructs cleanly."""
    p = _minimal_params(outer_bc=6, surface_cell_thickness=1000.0)
    assert p.boundary_conditions.outer_boundary_condition == 6
    assert p.mesh.surface_cell_thickness == 1000.0


@pytest.mark.parametrize('outer_bc', [1, 2, 4, 5])
def test_other_outer_bcs_with_zero_thickness_pass(outer_bc: int):
    """Outer BCs 1, 2, 4, 5 do not require surface refinement."""
    p = _minimal_params(outer_bc=outer_bc, surface_cell_thickness=0.0)
    assert p.boundary_conditions.outer_boundary_condition == outer_bc
    assert p.mesh.surface_cell_thickness == 0.0


def test_bc6_toml_rejection(tmp_path: Path):
    """TOML config specifying outer BC 6 with surface_cell_thickness = 0 raises ValueError."""
    cfg = tmp_path / 'unrefined_bc6.toml'
    cfg.write_text(_BASE_TOML.format(outer_bc=6, surface_cell_thickness=0.0))
    with pytest.raises(ValueError, match=_EXPECTED_ERROR):
        Parameters.from_file(cfg)

    # Integer 0 in TOML (parsed as int) must also raise the expected error cleanly.
    cfg_int = tmp_path / 'unrefined_bc6_int.toml'
    cfg_int.write_text(_BASE_TOML.format(outer_bc=6, surface_cell_thickness='0'))
    with pytest.raises(ValueError, match=_EXPECTED_ERROR):
        Parameters.from_file(cfg_int)

    cfg_ok = tmp_path / 'refined_bc6.toml'
    cfg_ok.write_text(_BASE_TOML.format(outer_bc=6, surface_cell_thickness=1000.0))
    p = Parameters.from_file(cfg_ok)
    assert p.boundary_conditions.outer_boundary_condition == 6
    assert p.mesh.surface_cell_thickness == 1000.0


def test_bc6_ini_rejection(tmp_path: Path):
    """INI config specifying outer BC 6 with surface_cell_thickness = 0 raises ValueError."""
    cfg = tmp_path / 'unrefined_bc6.cfg'
    cfg.write_text(_BASE_INI.format(outer_bc=6, surface_cell_thickness=0.0))
    with pytest.raises(ValueError, match=_EXPECTED_ERROR):
        Parameters.from_file(cfg)

    cfg_ok = tmp_path / 'refined_bc6.cfg'
    cfg_ok.write_text(_BASE_INI.format(outer_bc=6, surface_cell_thickness=1000.0))
    p = Parameters.from_file(cfg_ok)
    assert p.boundary_conditions.outer_boundary_condition == 6
    assert p.mesh.surface_cell_thickness == 1000.0


def test_bc6_config_facade_rejection(tmp_path: Path):
    """Config.from_file rejects unrefined BC 6."""
    cfg = tmp_path / 'facade_unrefined.toml'
    cfg.write_text(_BASE_TOML.format(outer_bc=6, surface_cell_thickness=0.0))
    with pytest.raises(ValueError, match=_EXPECTED_ERROR):
        Config.from_file(str(cfg))

    cfg_ok = tmp_path / 'facade_refined.toml'
    cfg_ok.write_text(_BASE_TOML.format(outer_bc=6, surface_cell_thickness=1000.0))
    p = Config.from_file(str(cfg_ok))
    assert p.boundary_conditions.outer_boundary_condition == 6
    assert p.mesh.surface_cell_thickness == 1000.0
