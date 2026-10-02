"""Tests for rejection of rheology fields in legacy INI / .cfg files.

Rheology fields are supported in TOML configuration only.
If an INI configuration sets any rheology field, loading raises an error
that names the TOML key to use instead.
"""

from __future__ import annotations

import pytest

from aragog.parser import Parameters

_MINIMAL_BASE_INI = """
[boundary_conditions]
outer_boundary_condition = 1
outer_boundary_value = 288.0
inner_boundary_condition = 2
inner_boundary_value = 0.0
emissivity = 1.0
equilibrium_temperature = 288.0
core_heat_capacity = 840.0

[energy]
conduction = true
convection = true
gravitational_separation = false
mixing = false
radionuclides = false
tidal = false
solver_method = cvode
use_jax_jacobian = false
kappah_floor = 0.0

[initial_condition]
initial_condition = 1
surface_temperature = 1700.0
basal_temperature = 1700.0

[mesh]
outer_radius = 6370e3
inner_radius = 3480e3
number_of_nodes = 20
mixing_length_profile = nearest_boundary
core_density = 10640.0
surface_density = 4460.0
gravitational_acceleration = 9.8
mass_coordinates = false

[phase_solid]
density = 4460.0
heat_capacity = 1200.0
thermal_conductivity = 3.0
thermal_expansivity = 3e-5
melt_fraction = 0.0
viscosity = 1e19

[phase_liquid]
density = 4460.0
heat_capacity = 1200.0
thermal_conductivity = 3.0
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
matprop_smooth_width = 0.01

[solver]
start_time = 0.0
end_time = 1.0
atol = 1e-6
rtol = 1e-6
"""


@pytest.mark.unit
@pytest.mark.parametrize(
    'field_name,field_val',
    [
        ('enabled', 'true'),
        ('activation_energy', '300e3'),
        ('activation_volume', '2.5e-6'),
        ('viscosity_max_log10', '40.0'),
        ('stress_closure_mode', 'lid'),
        ('yield_stress_c', '1e5'),
        ('mlt_top_slope', '0.22'),
    ],
)
def test_ini_rejects_rheology_field_with_clear_toml_guidance(
    tmp_path, field_name: str, field_val: str
):
    """Verify that setting any rheology field in INI raises an error naming the TOML key."""
    ini_text = _MINIMAL_BASE_INI.replace(
        '[phase_solid]\n',
        f'[phase_solid]\n{field_name} = {field_val}\n',
    )
    cfg_file = tmp_path / f'{field_name}.cfg'
    cfg_file.write_text(ini_text, encoding='utf-8')
    with pytest.raises(ValueError) as excinfo:
        Parameters.from_file(str(cfg_file))

    msg = str(excinfo.value)
    assert field_name in msg
    assert 'TOML' in msg or 'toml' in msg
    assert f'[phase_solid] {field_name}' in msg or f'phase_solid.{field_name}' in msg


@pytest.mark.unit
def test_ini_without_rheology_fields_loads_cleanly(tmp_path):
    """Verify that a valid INI config without rheology fields continues to load."""
    cfg_file = tmp_path / 'clean.cfg'
    cfg_file.write_text(_MINIMAL_BASE_INI, encoding='utf-8')
    p = Parameters.from_file(str(cfg_file))
    assert p.mesh.number_of_nodes == 20
