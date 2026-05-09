"""Unit tests for the Aragog CLI entry point.

Verifies that the seven subcommands (``run``, ``inspect``,
``validate``, ``show-config``, ``new``, ``list-configs``, ``vnv``)
are registered, callable, and reject malformed input without
silent fallthrough. Also covers the ``--version`` / ``--versions``
group flags and the ``aragog run --set <key.path>=<value>``
override mechanic.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.unit


def test_cli_version_flag_prints_aragog_version():
    """`aragog --version` must print the package version and exit 0.

    Discriminator: the printed version must match
    ``aragog.__version__`` exactly. A regression that hard-coded a
    stale version in cli.py would surface here.
    """
    from aragog import __version__
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])

    assert result.exit_code == 0, result.output
    assert __version__ in result.output, (
        f'aragog --version output {result.output!r} does not match '
        f'aragog.__version__ = {__version__!r}.'
    )


def test_cli_versions_flag_lists_dependencies():
    """`aragog --versions` (plural) must print the aragog version
    plus key dependency versions (numpy, scipy, jax) so bug reports
    are reproducible.

    Edge case: in an environment with a missing optional dependency
    (e.g. JAX), the line must still appear with 'not installed' or
    'unknown' rather than crashing the CLI. We can't reliably
    uninstall jax in a test, but we can assert the multi-line
    output structure.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['--versions'])

    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0].startswith('aragog '), (
        f'first line must start with "aragog "; got {lines[0]!r}.'
    )
    # Each dependency line is indented with two spaces and contains a colon.
    dep_names = {ln.strip().split(':', 1)[0] for ln in lines[1:] if ':' in ln}
    assert {'numpy', 'scipy'} <= dep_names, (
        f'expected numpy and scipy in dependency block; got {dep_names}.'
    )


def test_cli_help_prints_usage_and_exits_zero():
    """`aragog --help` must print Click's usage block and return code 0.

    Guards against a stray top-level side-effect or import failure
    in `aragog.cli`.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])

    assert result.exit_code == 0, (
        f'aragog --help exited {result.exit_code}, output: {result.output!r}'
    )
    assert 'Usage:' in result.output
    assert 'Commands:' in result.output


def test_cli_unknown_subcommand_returns_nonzero():
    """Unknown subcommands must return a non-zero exit code.

    Silent acceptance of garbage arguments would mask user typos.
    Click's default behaviour is exit code 2 with 'No such command'
    on stderr.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['this-is-not-a-command'])

    assert result.exit_code != 0, (
        'aragog rejected nothing on garbage input, '
        'breaking discoverability of mistyped subcommands.'
    )


def test_cli_is_a_click_group():
    """The cli object must be a Click Group, not a function or Command.

    A regression that converted `@click.group()` to `@click.command()`
    would still pass --help but break future subcommand registration.
    """
    import click

    from aragog.cli import cli

    assert isinstance(cli, click.Group), (
        f'aragog.cli.cli is {type(cli).__name__}, expected click.Group.'
    )


def test_cli_registers_expected_subcommands():
    """The documented subcommands must be registered.

    Discriminator: the test fails if either an extra debug command
    leaks in (e.g. someone leaves a stray `@cli.command()` during
    development) or one of the documented commands goes missing.
    Tier 1 documented set: run, inspect, validate, show-config, new,
    list-configs, vnv.
    """
    from aragog.cli import cli

    expected = {
        'run',
        'list-configs',
        'new',
        'inspect',
        'show-config',
        'validate',
        'vnv',
    }
    actual = set(cli.commands)
    assert actual == expected, (
        f'aragog.cli subcommands: expected {sorted(expected)}, got {sorted(actual)}.'
    )


def test_list_configs_includes_toml_and_skips_python():
    """`aragog list-configs` lists bundled .toml/.cfg only.

    Edge case: a future tweak that rewrites the iteration to glob
    everything under cfg/ would also list ``__init__.py``; the
    suffix filter must reject that explicitly.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['list-configs'])

    assert result.exit_code == 0, result.output
    assert 'abe_solid.toml' in result.output
    assert '__init__.py' not in result.output, 'list-configs leaked a non-config python file.'


def test_vnv_list_lists_known_topic():
    """`aragog vnv --list` enumerates available verify_*.py topics.

    Discriminator: ``radio_decay`` is one of the canonical V&V
    figure scripts and must appear; an accidental hard-coded list
    would also be flagged by inspecting two further topics.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['vnv', '--list'])

    assert result.exit_code == 0, result.output
    for topic in ('radio_decay', 'permeability', 'rhs_parity'):
        assert topic in result.output, (
            f'`aragog vnv --list` missing expected topic {topic!r}; '
            f'output was {result.output!r}.'
        )


def test_vnv_no_args_lists_topics():
    """Calling `aragog vnv` with no topic and no --list still lists.

    Convention: omitting the topic is equivalent to `--list`,
    so users who type the bare command get a discoverable response
    rather than a silent no-op.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['vnv'])

    assert result.exit_code == 0, result.output
    assert 'Available V&V topics' in result.output


def test_vnv_rejects_unknown_topic():
    """`aragog vnv <typo>` must raise a UsageError, not silently exit 0.

    Edge case: typos like 'radiio_decay' (extra 'i') would otherwise
    silently fail to find the script and exit zero. The error path
    must point users back to ``--list``.
    """
    from aragog.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['vnv', 'radiio_decay'])

    assert result.exit_code != 0
    assert 'unknown V&V topic' in (result.output or '')
    assert '--list' in (result.output or ''), (
        'unknown-topic message should redirect users to --list.'
    )


def _write_minimal_solver_output(path):
    """Write a minimal NetCDF snapshot that mimics the schema produced
    by SolverOutput.to_netcdf.

    Includes the global attributes inspect reads, the dimensions,
    a representative scalar set, and the two profile arrays inspect
    reports min/max for. Designed to be byte-readable by the
    inspect command without pulling in the full solver path.
    """
    import netCDF4 as nc
    import numpy as np

    with nc.Dataset(path, mode='w') as ds:
        ds.description = 'unit-test snapshot'
        ds.aragog_version = 'test-build'
        ds.created_utc = '2026-05-09T00:00:00+00:00'
        ds.Conventions = 'CF-1.8'

        ds.createDimension('staggered', 5)
        ds.createDimension('basic', 6)

        scalars = {
            'time': 1.0e6,
            'dt_actual': 1.0e6,
            'T_magma': 3500.0,
            'T_core': 4200.0,
            'Phi_global': 0.42,
            'Phi_global_vol': 0.45,
            'M_mantle': 4.0e24,
            'F_heat_total': 250.0,
            'F_cmb': 50.0,
            'Q_radio_total': 1.5e15,
            'Q_tidal_total': 0.0,
            'E_th': 5.0e29,
            'Cp_eff': 1200.0,
            'RF_depth': 0.5,
        }
        for name, value in scalars.items():
            v = ds.createVariable(name, 'f8')
            v.units = 'unit-stub'
            v.long_name = name
            v[...] = value

        v = ds.createVariable('status', 'i4')
        v.units = '1'
        v.long_name = 'solver status'
        v[...] = 0

        # Profile arrays: include S_final and T_basic so inspect's
        # min/max block has something to report.
        S = ds.createVariable('S_final', 'f8', ('staggered',))
        S[:] = np.linspace(2900.0, 3000.0, 5)
        T = ds.createVariable('T_basic', 'f8', ('basic',))
        T[:] = np.linspace(1500.0, 4500.0, 6)


def test_inspect_summarises_minimal_snapshot(tmp_path):
    """`aragog inspect <file.nc>` must surface the key scalars
    (T_magma, T_core, Phi_global, status), dimensions, and the
    aragog_version global attribute in a human-readable summary.

    Discriminator: the human format must NOT be JSON. A regression
    that swapped the default to --json behaviour would still pass an
    exit-code test but break shell-side post-processing.
    """
    from aragog.cli import cli

    snap = tmp_path / 'out.nc'
    _write_minimal_solver_output(snap)

    runner = CliRunner()
    result = runner.invoke(cli, ['inspect', str(snap)])

    assert result.exit_code == 0, result.output
    out = result.output
    assert 'T_magma' in out and '3500.0000' in out, (
        f'inspect did not print T_magma in the human summary; got {out!r}.'
    )
    assert 'T_core' in out and '4200.0000' in out
    assert 'status' in out
    assert 'aragog: test-build' in out, (
        'inspect summary must include the aragog_version attribute.'
    )
    assert 'staggered=5' in out and 'basic=6' in out
    # Profile min/max block surfaces non-trivial data.
    assert 'S_final' in out and 'min=2.9000e+03' in out
    # Default output is human-readable, not JSON.
    assert not out.lstrip().startswith('{'), (
        'default inspect output unexpectedly looks like JSON; shell pipelines would break.'
    )


def test_inspect_json_emits_valid_machine_readable_document(tmp_path):
    """`aragog inspect --json <file.nc>` emits a JSON document with
    the expected top-level keys (path, dimensions, attrs, scalars,
    profiles) and parseable values.
    """
    import json

    from aragog.cli import cli

    snap = tmp_path / 'out.nc'
    _write_minimal_solver_output(snap)

    runner = CliRunner()
    result = runner.invoke(cli, ['inspect', '--json', str(snap)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {'path', 'dimensions', 'attrs', 'scalars', 'profiles'} <= set(payload)
    assert payload['scalars']['T_magma'] == pytest.approx(3500.0)
    assert payload['scalars']['status'] == pytest.approx(0)
    # Profiles min/max must be numeric, not None, on a fresh snapshot.
    assert payload['profiles']['S_final']['n_finite'] == 5


def test_inspect_rejects_non_netcdf_file(tmp_path):
    """A file that is not a NetCDF dataset must produce a clean
    ClickException, not a noisy traceback into netCDF4 internals.
    """
    from aragog.cli import cli

    fake = tmp_path / 'not-a-netcdf.nc'
    fake.write_text('not netcdf at all\n')

    runner = CliRunner()
    result = runner.invoke(cli, ['inspect', str(fake)])

    assert result.exit_code != 0
    assert 'could not open' in (result.output or '').lower()


def test_inspect_handles_nan_filled_scalar_with_fillvalue(tmp_path):
    """A NaN-filled scalar (E_state on a const_properties run is the
    canonical case) must round-trip as JSON null and as 'n/a' in the
    human format, NOT as the string 'masked' or a TypeError crash.

    Edge case: SolverOutput.to_netcdf writes E_state / E_state_cons
    with _FillValue=NaN. netCDF4 returns those reads as a masked
    array whose float() is numpy.ma.masked unless auto-masking is
    disabled. The previous implementation evaded the
    (TypeError, ValueError) guard because numpy.ma.masked is a
    valid object that float() returns silently.
    """
    import json

    import netCDF4 as nc
    import numpy as np

    from aragog.cli import cli

    snap = tmp_path / 'with_fillvalue.nc'
    with nc.Dataset(snap, mode='w') as ds:
        ds.aragog_version = 'masked-test'
        ds.createDimension('staggered', 3)
        ds.createDimension('basic', 3)

        v_status = ds.createVariable('status', 'i4')
        v_status[...] = 0
        # T_magma populated normally so the scan is not all-empty.
        v_T = ds.createVariable('T_magma', 'f8')
        v_T[...] = 3500.0
        # E_state is NaN-filled with the same _FillValue=NaN pattern
        # the production writer uses for const_properties runs.
        v_E = ds.createVariable('E_state', 'f8', fill_value=np.nan)
        v_E[...] = np.nan

        # A profile with no _FillValue (must still inspect cleanly).
        S = ds.createVariable('S_final', 'f8', ('staggered',))
        S[:] = np.linspace(2900.0, 3000.0, 3)
        T_basic = ds.createVariable('T_basic', 'f8', ('basic',))
        T_basic[:] = np.linspace(1500.0, 4500.0, 3)

    runner = CliRunner()

    # Human-readable path must not crash and must mark E_state as n/a.
    result = runner.invoke(cli, ['inspect', str(snap)])
    assert result.exit_code == 0, result.output
    assert 'T_magma' in result.output
    assert 'E_state' in result.output and 'n/a' in result.output, (
        f'NaN-filled E_state must format as n/a in the human summary; got {result.output!r}.'
    )

    # JSON path must emit null (None) for E_state, not the string 'masked'.
    result = runner.invoke(cli, ['inspect', '--json', str(snap)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['scalars']['E_state'] is None, (
        'NaN-filled E_state must serialise as JSON null, not '
        f'{payload["scalars"]["E_state"]!r}.'
    )


def test_inspect_handles_fillvalue_in_profile_array(tmp_path):
    """A profile with `_FillValue` set must report finite-count
    correctly: fill positions must NOT count as finite even when the
    fill value itself is finite.

    Edge case: netCDF4's default fill value for f8 is
    9.969209968386869e+36 (a finite number). Without
    set_auto_mask(False) + explicit fill the masked-array is sliced
    with np.isfinite, which returns True at fill positions because
    the underlying data is finite, silently corrupting the min/max.
    """
    import netCDF4 as nc
    import numpy as np

    from aragog.cli import cli

    snap = tmp_path / 'profile_fillvalue.nc'
    with nc.Dataset(snap, mode='w') as ds:
        ds.aragog_version = 'profile-fill-test'
        ds.createDimension('staggered', 5)
        ds.createDimension('basic', 5)

        v_status = ds.createVariable('status', 'i4')
        v_status[...] = 0

        # S_final with _FillValue=NaN; mark cells 2 and 4 as fill
        # by writing NaN. netCDF4 returns those as masked when
        # auto-masking is on.
        S = ds.createVariable('S_final', 'f8', ('staggered',), fill_value=np.nan)
        S[:] = [2900.0, 2950.0, np.nan, 3000.0, np.nan]
        T_basic = ds.createVariable('T_basic', 'f8', ('basic',))
        T_basic[:] = np.linspace(1500.0, 4500.0, 5)

    runner = CliRunner()
    result = runner.invoke(cli, ['inspect', '--json', str(snap)])
    assert result.exit_code == 0, result.output

    import json

    payload = json.loads(result.output)
    info = payload['profiles']['S_final']
    assert info['n_total'] == 5
    assert info['n_finite'] == 3, (
        f'expected 3 finite cells out of 5, got {info["n_finite"]}; '
        'fill-value positions are leaking into the finite count.'
    )
    assert info['min'] == pytest.approx(2900.0)
    assert info['max'] == pytest.approx(3000.0)


def test_inspect_handles_all_nan_profile(tmp_path):
    """When a profile is entirely NaN (e.g. const_properties E_state),
    inspect's profile block must say 'all NaN' rather than crashing
    with min(empty array).

    Edge case: empty-finite arrays would otherwise hit a numpy
    DeprecationWarning or ValueError on `.min()`.
    """
    import netCDF4 as nc
    import numpy as np

    from aragog.cli import cli

    snap = tmp_path / 'out.nc'
    with nc.Dataset(snap, mode='w') as ds:
        ds.aragog_version = 'test'
        ds.createDimension('staggered', 3)
        ds.createDimension('basic', 3)
        v = ds.createVariable('status', 'i4')
        v[...] = 0
        S = ds.createVariable('S_final', 'f8', ('staggered',))
        S[:] = np.array([np.nan, np.nan, np.nan])
        T = ds.createVariable('T_basic', 'f8', ('basic',))
        T[:] = np.array([np.nan, np.nan, np.nan])

    runner = CliRunner()
    result = runner.invoke(cli, ['inspect', str(snap)])

    assert result.exit_code == 0, result.output
    assert 'all NaN' in result.output


def test_coerce_value_int_then_float_then_bool_then_json_then_string():
    """The type-coercion heuristic must follow a strict order:
    int -> float -> bool -> JSON -> bare string. Anything else
    silently miscasts user input.
    """
    from aragog.cli import _coerce_value

    assert _coerce_value('20') == 20
    assert isinstance(_coerce_value('20'), int), 'plain integer must stay int.'
    assert _coerce_value('20.0') == pytest.approx(20.0)
    assert isinstance(_coerce_value('20.0'), float)
    # Scientific notation is float, not int.
    assert _coerce_value('1e-9') == pytest.approx(1.0e-9)
    # Booleans (case-insensitive).
    assert _coerce_value('true') is True
    assert _coerce_value('False') is False
    # JSON list.
    assert _coerce_value('[1.0, 2.0, 3.0]') == [1.0, 2.0, 3.0]
    # JSON dict.
    assert _coerce_value('{"a": 1}') == {'a': 1}
    # Bare strings (lookup-table paths, mode names).
    assert _coerce_value('nearest_boundary') == 'nearest_boundary'
    assert _coerce_value('lookup.dat') == 'lookup.dat'


def test_apply_overrides_walks_dotted_paths():
    """_apply_overrides must walk multi-level dotted paths and
    overwrite the leaf with the type-coerced value.
    """
    from aragog.cli import _apply_overrides

    data = {'energy': {'kappah_floor': 10.0}, 'solver': {'atol': 1e-9}}
    out = _apply_overrides(data, ('energy.kappah_floor=20.0', 'solver.atol=1e-11'))

    assert out['energy']['kappah_floor'] == pytest.approx(20.0)
    assert out['solver']['atol'] == pytest.approx(1e-11)
    # Original input must not be mutated.
    assert data['energy']['kappah_floor'] == pytest.approx(10.0), (
        '_apply_overrides mutated the caller-provided dict; should deep-copy.'
    )


def test_apply_overrides_rejects_malformed_specs():
    """Specs without '=' or with empty path segments must raise
    UsageError with a clear message; silently dropping a typo
    would mask user intent.
    """
    from aragog.cli import _apply_overrides

    data = {'energy': {'kappah_floor': 10.0}}
    with pytest.raises(click.UsageError, match='KEY=VALUE'):
        _apply_overrides(data, ('energy.kappah_floor20',))
    with pytest.raises(click.UsageError, match='malformed'):
        _apply_overrides(data, ('energy..kappah_floor=20',))


def test_apply_overrides_rejects_single_segment_path():
    """`--set energy=20.0` (no dot) must raise UsageError instead of
    silently overwriting the entire `energy` section dict with a float.

    Edge case: without this guard the override succeeds at the
    dict-walk layer, then fails downstream as
    `_EnergyParameters(**20.0)` -> TypeError 'argument of type
    'float' is not iterable'. The user sees an error from the
    dataclass constructor instead of a clear statement about the
    malformed key path. The check fires here so the message names
    the actual cause.
    """
    from aragog.cli import _apply_overrides

    data = {'energy': {'kappah_floor': 10.0}}
    with pytest.raises(click.UsageError, match='dot separator'):
        _apply_overrides(data, ('energy=20.0',))


def test_apply_overrides_rejects_missing_intermediate_section():
    """When the dotted path traverses a section that is not in the
    input, _apply_overrides must raise — silently creating new
    sections would let users mistype `enrgy.kappah_floor` and have
    it be ignored downstream.
    """
    from aragog.cli import _apply_overrides

    data = {'energy': {'kappah_floor': 10.0}}
    with pytest.raises(click.UsageError, match='not found'):
        _apply_overrides(data, ('atmos.kappah_floor=20.0',))


def test_run_set_overrides_reach_parameters(tmp_path, monkeypatch):
    """End-to-end: `aragog run cfg.toml --set energy.kappah_floor=99` causes the
    solver to receive a Parameters object with the override applied.

    Mocks EntropySolver.__init__ + from_file so the test never
    triggers a real EOS load or solve. Asserts the captured
    Parameters has the override value, not the TOML default.
    """
    from aragog.cli import cli

    cfg = tmp_path / 'cfg.toml'
    # Minimal valid TOML matching Config.from_dict's required sections.
    import importlib.resources

    cfg.write_text(
        importlib.resources.files('aragog')
        .joinpath('cfg/abe_solid.toml')
        .read_text(encoding='utf-8'),
        encoding='utf-8',
    )
    eos = tmp_path / 'eos'
    eos.mkdir()

    captured: dict = {}

    class _StubSolver:
        def __init__(self, parameters, entropy_eos):
            captured['parameters'] = parameters

        @property
        def parameters(self):
            return captured['parameters']

        def initialize(self):
            pass

        def set_initial_dSdr_cmb(self, _):
            pass

        def set_initial_entropy(self, _):
            pass

        def solve(self):
            pass

        def get_state(self):
            class _Out:
                def to_netcdf(self, *a, **kw):
                    pass

            return _Out()

    monkeypatch.setattr(
        'aragog.solver.EntropySolver',
        _StubSolver,
        raising=True,
    )

    # Stub the EntropyEOS constructor to avoid real disk I/O.
    monkeypatch.setattr(
        'aragog.eos.entropy.EntropyEOS',
        lambda *a, **kw: object(),
        raising=True,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'run',
            str(cfg),
            '--eos-dir',
            str(eos),
            '--initial-entropy',
            '2900.0',
            '--set',
            'energy.kappah_floor=99.0',
            '--set',
            'solver.atol=1e-12',
        ],
    )

    assert result.exit_code == 0, result.output
    params = captured.get('parameters')
    assert params is not None, 'EntropySolver was never constructed.'
    assert params.energy.kappah_floor == pytest.approx(99.0), (
        f'override did not reach Parameters: expected 99.0, got {params.energy.kappah_floor}.'
    )
    assert params.solver.atol == pytest.approx(1e-12), (
        f'second override did not reach Parameters: got {params.solver.atol}.'
    )


def test_run_set_overrides_rejected_for_cfg_files(tmp_path):
    """`aragog run something.cfg --set ...` must fail loud: the
    legacy INI parser does not produce a dict that Config.from_dict
    can consume, so allowing --set on .cfg would silently surface
    a different (more confusing) failure mode.
    """
    import importlib.resources

    from aragog.cli import cli

    # Bundled abe_solid.cfg doesn't have inline-comment quirks for our
    # purposes — but we don't even reach the parser; the suffix check
    # short-circuits.
    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_solid.cfg')
    ) as cfg_path:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'run',
                str(cfg_path),
                '--eos-dir',
                str(tmp_path),
                '--initial-entropy',
                '2900.0',
                '--set',
                'energy.kappah_floor=20.0',
            ],
        )

    assert result.exit_code != 0
    assert '--set requires a .toml' in (result.output or ''), (
        f'CLI must explain why .cfg + --set is rejected; got {result.output!r}.'
    )


def test_show_config_emits_valid_json_with_known_fields():
    """`aragog show-config <bundled cfg>` emits valid JSON whose
    top-level keys match the Parameters dataclass field set.

    Discriminator: the printed JSON must parse with json.loads AND
    must contain a known scalar field (energy.kappah_floor) whose
    value matches the parsed dataclass — verifies that the
    serializer walks sub-dataclasses, not just the top level.
    """
    import importlib.resources
    import json

    from aragog.cli import cli
    from aragog.parser import Parameters

    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_mixed.cfg')
    ) as cfg_path:
        runner = CliRunner()
        result = runner.invoke(cli, ['show-config', str(cfg_path)])
        ground_truth = Parameters.from_file(cfg_path)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    expected_top = {
        'boundary_conditions',
        'energy',
        'initial_condition',
        'mesh',
        'phase_solid',
        'phase_liquid',
        'phase_mixed',
        'radionuclides',
        'solver',
    }
    assert expected_top <= set(payload.keys()), (
        f'show-config payload missing top-level keys: '
        f'expected superset of {expected_top}, got {sorted(payload.keys())}.'
    )
    # Cross-check one nested scalar against the ground-truth Parameters.
    assert payload['energy']['kappah_floor'] == pytest.approx(
        ground_truth.energy.kappah_floor, rel=1e-12
    ), (
        'show-config dropped or mutated energy.kappah_floor; the serializer '
        'is not walking sub-dataclasses faithfully.'
    )


def test_show_config_indent_zero_emits_one_line(tmp_path):
    """`aragog show-config --indent 0` produces one-line JSON
    (jq-pipeable) rather than the default pretty-printed form.

    Edge case: indent=0 in json.dumps means the literal indent
    character is empty, which makes one-line output. Test discriminates
    against a regression that left default indent in place.
    """
    import importlib.resources

    from aragog.cli import cli

    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_mixed.cfg')
    ) as cfg_path:
        runner = CliRunner()
        result = runner.invoke(cli, ['show-config', '--indent', '0', str(cfg_path)])

    assert result.exit_code == 0
    body = result.output.rstrip()
    assert '\n' not in body, (
        f'--indent 0 should produce single-line JSON; got {len(body.splitlines())} lines.'
    )


def test_show_config_rejects_malformed_config(tmp_path):
    """show-config wraps parser errors in a ClickException, not a
    raw traceback, mirroring the validate command's contract.
    """
    from aragog.cli import cli

    cfg = tmp_path / 'with_scalings.cfg'
    cfg.write_text(
        '[scalings]\nradius = 1.0\n'
        '\n[solver]\nstart_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-6\n'
    )
    runner = CliRunner()
    result = runner.invoke(cli, ['show-config', str(cfg)])

    assert result.exit_code != 0
    assert 'configuration error' in (result.output or '').lower()


def test_validate_accepts_bundled_abe_mixed_cfg():
    """`aragog validate <bundled cfg>` must exit 0 and print a one-line
    summary including the parsed core_bc / IBC / OBC fields.

    Discriminator: the summary mentions concrete fields by name; a
    regression that printed only "OK" would still pass an exit-code
    test but lose the diagnostic value the command exists for.
    """
    import importlib.resources

    from aragog.cli import cli

    with importlib.resources.as_file(
        importlib.resources.files('aragog').joinpath('cfg/abe_mixed.cfg')
    ) as cfg_path:
        runner = CliRunner()
        result = runner.invoke(cli, ['validate', str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert 'OK' in result.output
    for tag in ('core_bc=', 'IBC=', 'OBC=', 'radionuclides='):
        assert tag in result.output, f'validate summary missing {tag!r}; got {result.output!r}.'


def test_validate_rejects_scalings_section_via_strict_reject(tmp_path):
    """validate must surface the [scalings] strict-reject as a
    ClickException.

    Edge case: this is the most common migration error after the
    scalings removal landed; the command's whole point is catching
    it before a solve. The ValueError from Parameters.from_file
    must be wrapped, not propagated as a raw traceback.
    """
    from aragog.cli import cli

    cfg = tmp_path / 'with_scalings.cfg'
    cfg.write_text(
        '[scalings]\nradius = 1.0\n'
        '\n[solver]\nstart_time = 0\nend_time = 1\natol = 1e-9\nrtol = 1e-6\n'
    )
    runner = CliRunner()
    result = runner.invoke(cli, ['validate', str(cfg)])

    assert result.exit_code != 0
    assert 'scalings' in (result.output or '').lower()
    assert 'configuration error' in (result.output or '').lower(), (
        'validate must wrap the parser exception in a "configuration error" '
        'frame so the user knows where the failure originated.'
    )


def test_validate_rejects_missing_required_field(tmp_path):
    """A config missing a required section must produce a non-zero
    exit AND a message that names the broken section, not a bare
    traceback.

    Discriminator: the assertion checks that the missing-section
    name (`solver`) appears in the output. A future refactor that
    catches the parser exception and prints a generic
    'configuration error: <exc>' without the section name would
    still pass an exit-code-only test, defeating the diagnostic
    value the command exists for.
    """
    from aragog.cli import cli

    cfg = tmp_path / 'no_solver.cfg'
    cfg.write_text(
        '[boundary_conditions]\nouter_boundary_condition = 1\n'
        'outer_boundary_value = 1500\n'
        'inner_boundary_condition = 2\ninner_boundary_value = 0\n'
        'emissivity = 1\nequilibrium_temperature = 273\n'
        'core_heat_capacity = 880\n'
        # No [solver] section — this is the missing required block.
    )
    runner = CliRunner()
    result = runner.invoke(cli, ['validate', str(cfg)])

    assert result.exit_code != 0
    out = (result.output or '').lower()
    assert 'configuration error' in out
    assert 'solver' in out, (
        'validate must surface the missing-section name (solver) so the '
        f'user knows what to fix; got output={result.output!r}.'
    )


def test_new_scaffolds_default_template(tmp_path, monkeypatch):
    """`aragog new my_run` writes a TOML file in the cwd by copying the
    default abe_solid template.

    Discriminator: the destination file must exist after invocation
    AND its contents must equal the template's contents byte-for-byte
    (catches a regression that wrote an empty file or a stale stub).
    """
    from aragog.cli import cli

    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ['new', 'my_run'])

    assert result.exit_code == 0, result.output
    dest = tmp_path / 'my_run.toml'
    assert dest.is_file(), f'expected {dest} to exist; output={result.output!r}'

    # Compare against bundled abe_solid.toml byte-for-byte.
    import importlib.resources

    template = (
        importlib.resources.files('aragog')
        .joinpath('cfg/abe_solid.toml')
        .read_text(encoding='utf-8')
    )
    assert dest.read_text(encoding='utf-8') == template, (
        'scaffolded file diverges from the bundled template; the copy is broken.'
    )


def test_new_appends_toml_suffix_when_missing(tmp_path, monkeypatch):
    """`aragog new foo` writes foo.toml; `aragog new foo.toml` writes
    foo.toml. Edge case: the suffix logic must be idempotent.
    """
    from aragog.cli import cli

    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(cli, ['new', 'no_suffix'])
    runner.invoke(cli, ['new', 'with_suffix.toml'])

    assert (tmp_path / 'no_suffix.toml').is_file()
    assert (tmp_path / 'with_suffix.toml').is_file()
    assert not (tmp_path / 'with_suffix.toml.toml').exists(), (
        'aragog new doubled the .toml suffix; the suffix logic is not idempotent.'
    )


def test_new_rejects_unknown_template(tmp_path, monkeypatch):
    """`aragog new foo --from nonexistent` must fail with a clear list
    of available templates, not silently produce an empty file.
    """
    from aragog.cli import cli

    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ['new', 'foo', '--from', 'nonexistent_template'])

    assert result.exit_code != 0
    assert 'unknown template' in (result.output or '')
    assert 'abe_solid.toml' in (result.output or ''), (
        'available templates list should include the canonical abe_solid.toml; '
        f'got output={result.output!r}.'
    )
    assert not (tmp_path / 'foo.toml').exists(), (
        'aragog new wrote a file even though the template was unknown.'
    )


def test_new_refuses_overwrite_without_force(tmp_path, monkeypatch):
    """When the destination exists, `aragog new` must refuse to
    overwrite unless `--force` is passed.

    Edge case: a user running `aragog new my_run` twice in a row
    must not silently lose their edits to the first file.
    """
    from aragog.cli import cli

    monkeypatch.chdir(tmp_path)
    existing = tmp_path / 'my_run.toml'
    existing.write_text('# user edits\n', encoding='utf-8')

    runner = CliRunner()
    result = runner.invoke(cli, ['new', 'my_run'])

    assert result.exit_code != 0
    assert '--force' in (result.output or ''), (
        'overwrite-refusal message should suggest the --force escape hatch.'
    )
    assert existing.read_text(encoding='utf-8') == '# user edits\n', (
        'aragog new clobbered an existing file without --force.'
    )

    # With --force the overwrite goes through.
    result = runner.invoke(cli, ['new', 'my_run', '--force'])
    assert result.exit_code == 0
    assert existing.read_text(encoding='utf-8') != '# user edits\n', (
        '--force did not actually overwrite the file.'
    )


def test_run_rejects_malformed_config(tmp_path):
    """`aragog run` rejects a malformed config before reaching the solve.

    Edge case: a near-empty TOML fails inside Parameters.from_file
    (missing required sections). The CLI must surface that as a
    non-zero exit, not catch and swallow the error.

    This test deliberately uses a stub config so the failure happens
    at parser load time. The companion test
    test_run_rejects_missing_initial_entropy_with_valid_config
    exercises the --initial-entropy guard with a parseable config.
    """
    from aragog.cli import cli

    cfg = tmp_path / 'stub.toml'
    cfg.write_text('# stub\n')

    fake_eos = tmp_path / 'no_such_eos_dir'
    fake_eos.mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, ['run', str(cfg), '--eos-dir', str(fake_eos)])

    assert result.exit_code != 0, (
        'aragog run silently accepted a malformed config; the user '
        'would never see the parser error.'
    )


def test_run_rejects_missing_initial_entropy_with_valid_config(tmp_path, monkeypatch):
    """`aragog run` raises UsageError when --initial-entropy is omitted
    on an otherwise-valid run.

    Mocks ``EntropySolver.from_file`` so the parser/EOS layer is
    bypassed and the guard at the ``initial_entropy is None`` check
    is the only failure surface. Without the mock the parser would
    reject a stub config first, masking the guard. This test
    discriminates against a future refactor that drops the
    UsageError check (e.g., silently picking a default S_0).
    """
    from aragog.cli import cli

    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# placeholder\n')
    eos = tmp_path / 'eos'
    eos.mkdir()

    class _StubSolver:
        parameters = type(
            'P',
            (),
            {'boundary_conditions': type('BC', (), {'core_bc': 'energy_balance'})()},
        )()

        def initialize(self):
            pass

        def set_initial_dSdr_cmb(self, _):
            pass

    import aragog.solver as _solver_mod

    monkeypatch.setattr(
        _solver_mod.EntropySolver,
        'from_file',
        classmethod(lambda cls, **kw: _StubSolver()),
        raising=True,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ['run', str(cfg), '--eos-dir', str(eos)])

    assert result.exit_code != 0
    assert '--initial-entropy is required' in (result.output or ''), (
        'CLI must surface the guard message verbatim so users know which '
        f'flag to pass; got output={result.output!r}.'
    )


def test_run_rejects_missing_eos_dir(tmp_path, monkeypatch):
    """`aragog run` with no --eos-dir and no FWL_DATA fails cleanly.

    Edge case: relying on the wrong env-var to be set would produce
    confusing tracebacks deep inside EntropyEOS. The CLI must
    short-circuit at the boundary with a UsageError.
    """
    from aragog.cli import cli

    cfg = tmp_path / 'stub.toml'
    cfg.write_text('# stub\n')

    monkeypatch.delenv('FWL_DATA', raising=False)

    runner = CliRunner()
    result = runner.invoke(cli, ['run', str(cfg)])

    assert result.exit_code != 0, 'aragog run should fail without eos-dir.'
    assert '--eos-dir' in (result.output or ''), 'error message should name the missing flag.'
