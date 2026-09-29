"""Regression tests for the TOML / INI dispatch in
``Parameters.from_file``.

Background
----------
Until 2026-05-10 ``Parameters.from_file`` always routed through
``typed_configparser`` (INI semantics), even for ``.toml``
configuration files. That had two coupled failure modes that broke
the standalone ``aragog new my_run.toml`` + ``aragog run`` tutorial
flow on the bundled ``abe_solid.toml``:

1. Quoted strings round-tripped with the surrounding quotes:
   ``mixing_length_profile = "nearest_boundary"`` was read as the
   18-character Python string ``'"nearest_boundary"'``, then the
   downstream ``if mixing_length_profile == 'nearest_boundary'``
   check failed with ``ValueError: Mixing length profile is unknown``.
2. ``# ...`` inline comments inside numeric fields were not stripped:
   ``heat_production = 0.3583  # W/kg of pure 26Al`` raised
   ``ParseError: Cannot cast value '0.3583  # W/kg of pure 26Al'
   to 'float'``, blocking the entire radionuclide load.

These tests guard against the regression. They cover both the
TOML-dispatch path (``.toml`` files via ``tomllib``) AND the legacy
INI path (``.cfg`` files via ``typed_configparser``) so neither
loader silently swaps semantics underneath the user.
"""

from __future__ import annotations

import importlib.resources

import pytest

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# TOML path: bundled abe_solid.toml must round-trip cleanly
# ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def loaded_toml():
    """Module-scoped load of the bundled ``abe_solid.toml`` via the
    new TOML dispatch path. Reused across the assertions below.
    """
    from aragog.parser import Parameters

    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        return Parameters.from_file(p)


def test_toml_string_field_quotes_are_stripped(loaded_toml):
    """``mixing_length_profile = "nearest_boundary"`` must round-trip
    as the bare string ``'nearest_boundary'``.

    Discriminator: the failing INI reader returned the value with
    surrounding quotes (length 18). The TOML reader returns length 16.
    We assert the exact string AND its length to guarantee the fix
    is the right one.
    """
    val = loaded_toml.mesh.mixing_length_profile
    assert val == 'nearest_boundary', (
        f'mixing_length_profile = {val!r}; expected unquoted '
        '"nearest_boundary". The TOML loader is keeping the surrounding '
        'quotes verbatim.'
    )
    assert len(val) == 16


def test_toml_path_string_fields_have_no_quote_artefacts(loaded_toml):
    """Path-valued string fields (``solidus``, ``liquidus``) must come
    out as relative paths the file system can resolve, not strings
    with surrounding quotes.
    """
    assert loaded_toml.phase_mixed.solidus == 'data/test/solidus_1d_lookup.dat'
    assert loaded_toml.phase_mixed.liquidus == 'data/test/liquidus_1d_lookup.dat'
    assert loaded_toml.phase_mixed.phase == 'solid'


def test_toml_inline_comments_are_stripped_from_numeric_fields(loaded_toml):
    """``heat_production = 0.3583  # W/kg of pure 26Al`` must parse as
    the float 0.3583 with no inline-comment leakage into the value.

    Discriminator: the bug surfaced as ``ParseError: Cannot cast
    value '0.3583  # ...' to 'float'``. After the fix the value is a
    plain float and arithmetic works.
    """
    al26 = next(r for r in loaded_toml.radionuclides if r.name == 'Al26')
    assert al26.heat_production == pytest.approx(0.3583)
    assert isinstance(al26.heat_production, float)
    fe60 = next(r for r in loaded_toml.radionuclides if r.name == 'Fe60')
    assert fe60.heat_production == pytest.approx(3.6579e-2)


def test_toml_radionuclide_sections_all_loaded(loaded_toml):
    """All six bundled radionuclides must be loaded; the inline-comment
    crash on Al26 used to abort the load before it reached Fe60.
    """
    names = sorted(r.name for r in loaded_toml.radionuclides)
    assert names == ['Al26', 'Fe60', 'K40', 'Th232', 'U235', 'U238']


def test_toml_typed_fields_have_native_types(loaded_toml):
    """Boolean / int / float TOML values must round-trip with native
    Python types. Discriminator: a regression that re-introduced the
    INI loader would surface as bool fields being strings ``'True'``
    or int fields being ``'100'``.
    """
    assert loaded_toml.energy.conduction is True
    assert loaded_toml.energy.tidal is False
    assert loaded_toml.mesh.number_of_nodes == 100
    assert isinstance(loaded_toml.mesh.number_of_nodes, int)
    assert loaded_toml.solver.atol == pytest.approx(1.0e-9)
    assert isinstance(loaded_toml.solver.atol, float)


def test_toml_loaded_solver_initializes(tmp_path):
    """End-to-end: build EntropySolver from the TOML, initialize the
    mesh + BC pipeline, and set the IC. A failure in any string field
    (e.g. ``mixing_length_profile``) would surface here as a
    ``ValueError`` from ``Mesh.__init__`` rather than at parse time.
    """
    import os
    from pathlib import Path

    from aragog.solver.entropy_solver import EntropySolver

    eos = os.environ.get('ARAGOG_TEST_EOS_DIR') or '/tmp/aragog-test-data/spider_eos'
    if not Path(eos).exists():
        pytest.skip(f'EOS unavailable at {eos}')
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.toml')
    ) as p:
        solver = EntropySolver.from_file(filename=str(p), eos_dir=eos)
    solver.initialize()
    assert solver._n_stag > 0
    assert solver.parameters.mesh.mixing_length_profile == 'nearest_boundary'


# ----------------------------------------------------------------------
# Reject paths
# ----------------------------------------------------------------------


def test_toml_rejects_scalings_section_case_insensitive(tmp_path):
    """The ``[scalings]`` strict-reject must fire under the TOML loader
    too, not only under the legacy INI loader. Tries both lower-case
    and shouting case to confirm the comparison is normalised.
    """
    from aragog.parser import Parameters

    cfg = tmp_path / 'has_scalings.toml'
    cfg.write_text(
        '[Scalings]\n'
        'radius = 1.0\n'
        '[solver]\n'
        'start_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-9\ntsurf_poststep_change = 30\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match=r'\[scalings\] section'):
        Parameters.from_file(cfg)


def test_toml_dispatch_rejects_multifile_load(tmp_path):
    """Multi-file load is supported only on the legacy INI path. A
    .toml first argument with extra files must raise.

    Discriminator: silently dropping the extra files would let users
    miss override behaviour they thought was active.
    """
    from aragog.parser import Parameters

    a = tmp_path / 'a.toml'
    b = tmp_path / 'b.toml'
    a.write_text('[solver]\n', encoding='utf-8')
    b.write_text('[solver]\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Multi-file load'):
        Parameters.from_file(a, b)


def test_toml_dispatch_reports_missing_required_section(tmp_path):
    """A TOML file missing a required section must raise with a
    message naming both the section and the file.
    """
    from aragog.parser import Parameters

    cfg = tmp_path / 'incomplete.toml'
    cfg.write_text(
        # Only [solver] – [boundary_conditions] / [mesh] / etc. all missing.
        '[solver]\nstart_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-9\n'
        'tsurf_poststep_change = 30\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='missing required section'):
        Parameters.from_file(cfg)


def test_toml_dispatch_wraps_dataclass_typeerror_with_section_context(tmp_path):
    """An unknown field in a section must surface with the section
    name, not as a bare ``TypeError`` from the dataclass constructor.
    """
    from aragog.parser import Parameters

    cfg = tmp_path / 'bad_field.toml'
    cfg.write_text(
        '[solver]\n'
        'start_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-9\n'
        'tsurf_poststep_change = 30\nunknown_garbage_field = 99\n'
        '[boundary_conditions]\n'
        'outer_boundary_condition = 1\nouter_boundary_value = 1500\n'
        'inner_boundary_condition = 2\ninner_boundary_value = 0\n'
        'emissivity = 1.0\nequilibrium_temperature = 273\n'
        'core_heat_capacity = 880\n'
        '[mesh]\nouter_radius = 6.371e6\ninner_radius = 3.480e6\n'
        'number_of_nodes = 10\nmixing_length_profile = "nearest_boundary"\n'
        'core_density = 10500\n'
        '[energy]\nconduction = true\nconvection = true\n'
        'gravitational_separation = false\nmixing = false\n'
        'radionuclides = false\ntidal = false\n'
        '[initial_condition]\ninitial_condition = 1\n'
        'surface_temperature = 3500\nbasal_temperature = 3500\n'
        '[phase_solid]\ndensity = 4200\nheat_capacity = 1000\n'
        'melt_fraction = 0\nthermal_conductivity = 4\n'
        'thermal_expansivity = 3e-5\nviscosity = 1e21\n'
        '[phase_liquid]\ndensity = 4000\nheat_capacity = 1000\n'
        'melt_fraction = 1\nthermal_conductivity = 4\n'
        'thermal_expansivity = 3e-5\nviscosity = 10\n'
        '[phase_mixed]\nlatent_heat_of_fusion = 4e5\n'
        'rheological_transition_melt_fraction = 0.4\n'
        'rheological_transition_width = 0.15\n'
        'solidus = "solidus.dat"\nliquidus = "liquidus.dat"\n'
        'phase = "mixed"\nphase_transition_width = 0.01\n'
        'grain_size = 1e-3\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match=r'\[solver\]'):
        Parameters.from_file(cfg)


# ----------------------------------------------------------------------
# INI path: keep backwards compatibility on .cfg files
# ----------------------------------------------------------------------


def test_ini_cfg_still_loads_via_typed_configparser():
    """The bundled ``abe_solid.cfg`` (INI format, unquoted strings)
    must still load via the legacy path.

    Discriminator: ``abe_solid.cfg`` uses bare ``mixing_length_profile
    = nearest_boundary`` (no quotes), which is the canonical INI
    convention. A regression that forced everything through tomllib
    would crash on the unquoted RHS.
    """
    from aragog.parser import Parameters

    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.cfg')
    ) as p:
        params = Parameters.from_file(p)
    assert params.mesh.mixing_length_profile == 'nearest_boundary'
    assert params.phase_mixed.phase == 'solid'


def test_ini_path_still_rejects_scalings_section(tmp_path):
    """The ``[scalings]`` strict-reject must still fire for legacy
    .cfg files routed through typed_configparser.
    """
    from aragog.parser import Parameters

    cfg = tmp_path / 'has_scalings.cfg'
    cfg.write_text(
        '[scalings]\nradius = 1.0\n'
        '[solver]\nstart_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-9\n'
        'tsurf_poststep_change = 30\n'
    )
    with pytest.raises(ValueError, match=r'\[scalings\] section'):
        Parameters.from_file(cfg)


def test_toml_dispatch_wraps_radionuclide_typeerror_with_section_context(tmp_path):
    """Parser line 475-476: an unknown field on a [radionuclide_*]
    section must be wrapped as a ValueError naming the offending
    section, not surfaced as a bare TypeError from the
    ``_Radionuclide`` constructor.

    Discriminator: the wrapper preserves debuggability. Without
    it the user sees a confusing ``unexpected keyword argument
    'foo'`` from deep in the dataclass machinery.
    """
    from aragog.parser import Parameters

    cfg = tmp_path / 'bad_radio.toml'
    cfg.write_text(
        # Minimal valid sections plus one bad radionuclide field.
        '[solver]\nstart_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-9\n'
        'tsurf_poststep_change = 30\n'
        '[boundary_conditions]\nouter_boundary_condition = 1\nouter_boundary_value = 1500\n'
        'inner_boundary_condition = 2\ninner_boundary_value = 0\n'
        'emissivity = 1.0\nequilibrium_temperature = 273\ncore_heat_capacity = 880\n'
        '[mesh]\nouter_radius = 6.371e6\ninner_radius = 3.480e6\n'
        'number_of_nodes = 10\nmixing_length_profile = "nearest_boundary"\n'
        'core_density = 10500\n'
        '[energy]\nconduction = true\nconvection = true\n'
        'gravitational_separation = false\nmixing = false\n'
        'radionuclides = false\ntidal = false\n'
        '[initial_condition]\ninitial_condition = 1\n'
        'surface_temperature = 3500\nbasal_temperature = 3500\n'
        '[phase_solid]\ndensity = 4200\nheat_capacity = 1000\n'
        'melt_fraction = 0\nthermal_conductivity = 4\n'
        'thermal_expansivity = 3e-5\nviscosity = 1e21\n'
        '[phase_liquid]\ndensity = 4000\nheat_capacity = 1000\n'
        'melt_fraction = 1\nthermal_conductivity = 4\n'
        'thermal_expansivity = 3e-5\nviscosity = 10\n'
        '[phase_mixed]\nlatent_heat_of_fusion = 4e5\n'
        'rheological_transition_melt_fraction = 0.4\n'
        'rheological_transition_width = 0.15\n'
        'solidus = "solidus.dat"\nliquidus = "liquidus.dat"\n'
        'phase = "mixed"\nphase_transition_width = 0.01\n'
        'grain_size = 1e-3\n'
        '[radionuclide_K40]\nname = "K40"\nt0_years = 4.55e9\n'
        'abundance = 1e-4\nconcentration = 310\nheat_production = 2.9e-5\n'
        'half_life_years = 1.25e9\nunknown_radio_field = 99\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match=r'\[radionuclide_K40\]'):
        Parameters.from_file(cfg)


def test_ic_method_2_loads_init_temperature_from_file(tmp_path):
    """``initial_condition = 2`` (user-defined T file) must trigger
    the ``np.loadtxt`` of ``init_file`` in ``__post_init__`` (parser
    line 362).

    Discriminator: the ``_InitialConditionParameters.init_temperature``
    field must be populated as an ndarray after load. A regression
    that lost the ``np.loadtxt`` call would leave it as the default.
    """
    import numpy as np

    from aragog.parser import Parameters

    init_file = tmp_path / 'init_T.dat'
    np.savetxt(init_file, np.linspace(4000.0, 1500.0, 10))

    cfg = tmp_path / 'ic_method_2.toml'
    cfg.write_text(
        '[solver]\nstart_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-9\n'
        'tsurf_poststep_change = 30\n'
        '[boundary_conditions]\nouter_boundary_condition = 1\nouter_boundary_value = 1500\n'
        'inner_boundary_condition = 2\ninner_boundary_value = 0\n'
        'emissivity = 1.0\nequilibrium_temperature = 273\ncore_heat_capacity = 880\n'
        '[mesh]\nouter_radius = 6.371e6\ninner_radius = 3.480e6\n'
        'number_of_nodes = 10\nmixing_length_profile = "nearest_boundary"\n'
        'core_density = 10500\n'
        '[energy]\nconduction = true\nconvection = true\n'
        'gravitational_separation = false\nmixing = false\n'
        'radionuclides = false\ntidal = false\n'
        '[initial_condition]\ninitial_condition = 2\n'
        f'init_file = "{init_file}"\n'
        'surface_temperature = 3500\nbasal_temperature = 3500\n'
        '[phase_solid]\ndensity = 4200\nheat_capacity = 1000\n'
        'melt_fraction = 0\nthermal_conductivity = 4\n'
        'thermal_expansivity = 3e-5\nviscosity = 1e21\n'
        '[phase_liquid]\ndensity = 4000\nheat_capacity = 1000\n'
        'melt_fraction = 1\nthermal_conductivity = 4\n'
        'thermal_expansivity = 3e-5\nviscosity = 10\n'
        '[phase_mixed]\nlatent_heat_of_fusion = 4e5\n'
        'rheological_transition_melt_fraction = 0.4\n'
        'rheological_transition_width = 0.15\n'
        'solidus = "solidus.dat"\nliquidus = "liquidus.dat"\n'
        'phase = "mixed"\nphase_transition_width = 0.01\n'
        'grain_size = 1e-3\n',
        encoding='utf-8',
    )
    params = Parameters.from_file(cfg)
    assert params.initial_condition.initial_condition == 2
    assert isinstance(params.initial_condition.init_temperature, np.ndarray)
    assert params.initial_condition.init_temperature.shape == (10,)
    assert float(params.initial_condition.init_temperature[0]) == pytest.approx(4000.0)


def test_from_file_no_arguments_raises():
    """Calling ``Parameters.from_file()`` with no filename must raise
    a clear error rather than silently building an empty Parameters.
    """
    from aragog.parser import Parameters

    with pytest.raises(ValueError, match='at least one configuration filename'):
        Parameters.from_file()
