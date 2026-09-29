"""Relative ``eos_file`` and ``init_file`` resolve against the config directory, then the CWD."""

from __future__ import annotations

import logging
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from aragog.cli import _absolute_template_paths, _apply_overrides, cli
from aragog.config import Config
from aragog.parser import Parameters

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = REPO / 'src' / 'aragog' / 'cfg'
EOS_TEST = REPO / 'data' / 'test' / 'eos_test.dat'


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """A config directory and a separate working directory (the CWD)."""
    cfg_dir, cwd = tmp_path / 'cfg', tmp_path / 'cwd'
    cfg_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cfg_dir, cwd


def _ini_with(tmp_dir: Path, eos_file: str, extra: str = '') -> Path:
    """Copy abe_mixed_init.cfg into ``tmp_dir`` with the given eos_file line."""
    text = (CFG_DIR / 'abe_mixed_init.cfg').read_text()
    text = text.replace('eos_file = ../../../data/test/eos_test.dat', f'eos_file = {eos_file}')
    assert f'eos_file = {eos_file}' in text
    path = tmp_dir / 'run.cfg'
    path.write_text(text.replace('[initial_condition]\n', f'[initial_condition]\n{extra}'))
    return path


def _toml_with_eos(cfg_dir: Path, copy: bool = True) -> Path:
    if copy:
        shutil.copy(EOS_TEST, cfg_dir / 'eos.dat')
    toml = cfg_dir / 'run.toml'
    text = (CFG_DIR / 'abe_solid.toml').read_text()
    toml.write_text(
        text.replace('[mesh]\n', '[mesh]\neos_method = 2\neos_file = "eos.dat"\n', 1)
    )
    return toml


def _cli(toml: Path, *overrides: str):
    args = ['run', str(toml), '--eos-dir', str(Path.cwd()), '--initial-entropy', '2900']
    return CliRunner().invoke(cli, args + [a for o in overrides for a in ('--set', o)])


def _cli_parameters(monkeypatch, toml: Path, *overrides: str):
    """Run ``aragog run TOML --set ...`` in the CWD with a stub solver; return its Parameters."""
    captured = {}

    class _Solver:
        def __init__(self, parameters, entropy_eos):
            self.parameters = captured['p'] = parameters

        def solve(self):
            raise SystemExit(0)

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr('aragog.solver.EntropySolver', _Solver)
    monkeypatch.setattr('aragog.eos.entropy.EntropyEOS', lambda *a, **k: object())
    result = _cli(toml, *overrides)
    assert result.exit_code == 0, result.output
    return captured['p']


def test_bundled_mixed_init_loads_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = Parameters.from_file(str(CFG_DIR / 'abe_mixed_init.cfg'))
    assert Path(params.mesh.eos_file) == EOS_TEST
    np.testing.assert_array_equal(params.mesh.eos_radius, np.loadtxt(EOS_TEST)[:, 0])


def test_config_dir_wins_over_cwd_and_is_logged(dirs, caplog):
    cfg_dir, cwd = dirs
    shutil.copy(EOS_TEST, cfg_dir / 'eos.dat')
    shutil.copy(EOS_TEST, cwd / 'eos.dat')
    with caplog.at_level(logging.INFO, logger='fwl.aragog.parser'):
        params = Parameters.from_file(str(_ini_with(cfg_dir, 'eos.dat')))
    resolved = str(cfg_dir.resolve() / 'eos.dat')
    assert params.mesh.eos_file == resolved
    assert f'mesh.eos_file eos.dat resolved to {resolved}' in caplog.text


def test_cwd_fallback_becomes_absolute(dirs):
    cfg_dir, cwd = dirs
    shutil.copy(EOS_TEST, cwd / 'eos.dat')
    params = Parameters.from_file(str(_ini_with(cfg_dir, 'eos.dat')))
    assert params.mesh.eos_file == str(cwd.resolve() / 'eos.dat')
    assert params.mesh.eos_radius.size == np.loadtxt(EOS_TEST).shape[0]


def test_directory_named_like_the_file_is_skipped(dirs):
    cfg_dir, cwd = dirs
    (cfg_dir / 'eos.dat').mkdir()
    shutil.copy(EOS_TEST, cwd / 'eos.dat')
    params = Parameters.from_file(str(_ini_with(cfg_dir, 'eos.dat')))
    assert params.mesh.eos_file == str(cwd.resolve() / 'eos.dat')


def test_absolute_path_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = Parameters.from_file(str(_ini_with(tmp_path, str(EOS_TEST))))
    assert params.mesh.eos_file == str(EOS_TEST)


def test_init_file_resolves_against_config_dir(dirs):
    cfg_dir, _ = dirs
    np.savetxt(cfg_dir / 'T.dat', [[5.4e6, 3000.0], [6.3e6, 2000.0]])
    cfg = _ini_with(cfg_dir, 'eos.dat', extra='init_file = T.dat\n')
    shutil.copy(EOS_TEST, cfg_dir / 'eos.dat')
    cfg.write_text(cfg.read_text().replace('initial_condition = 3', 'initial_condition = 2'))
    params = Parameters.from_file(str(cfg))
    assert params.initial_condition.init_file == str(cfg_dir.resolve() / 'T.dat')
    assert params.initial_condition.init_temperature.shape == (2, 2)


@pytest.mark.parametrize('loader', ['parser', 'config'])
def test_toml_eos_file_resolves_against_config_dir(dirs, loader):
    cfg_dir, _ = dirs
    load = Parameters.from_file if loader == 'parser' else Config.from_toml
    params = load(str(_toml_with_eos(cfg_dir)))
    assert params.mesh.eos_file == str(cfg_dir.resolve() / 'eos.dat')


def test_from_dict_without_config_dir_leaves_path_as_given(tmp_path, monkeypatch):
    toml = _toml_with_eos(tmp_path)
    monkeypatch.chdir(tmp_path)
    params = Config.from_dict(tomllib.loads(toml.read_text()))
    assert params.mesh.eos_file == 'eos.dat'


def test_toml_cwd_fallback_and_missing_file(dirs):
    cfg_dir, cwd = dirs
    toml = _toml_with_eos(cfg_dir, copy=False)
    with pytest.raises(FileNotFoundError, match='eos.dat'):
        Config.from_toml(str(toml))
    shutil.copy(EOS_TEST, cwd / 'eos.dat')
    assert Config.from_toml(str(toml)).mesh.eos_file == str(cwd.resolve() / 'eos.dat')


def test_cli_set_resolves_against_config_dir(dirs, monkeypatch):
    cfg_dir, _ = dirs
    params = _cli_parameters(monkeypatch, _toml_with_eos(cfg_dir), 'solver.atol=1e-12')
    assert params.mesh.eos_file == str(cfg_dir.resolve() / 'eos.dat')


def test_cli_set_path_value_resolves_against_cwd(dirs, monkeypatch):
    cfg_dir, cwd = dirs
    toml = _toml_with_eos(cfg_dir)
    shutil.copy(EOS_TEST, cwd / 'eos.dat')
    params = _cli_parameters(monkeypatch, toml, 'mesh.eos_file=eos.dat')
    assert params.mesh.eos_file == str(cwd.resolve() / 'eos.dat')


def test_cli_set_init_file_resolves_against_cwd(dirs, monkeypatch):
    cfg_dir, cwd = dirs
    for d in (cfg_dir, cwd):
        np.savetxt(d / 'T.dat', [[5.4e6, 3000.0], [6.3e6, 2000.0]])
    ic = ('initial_condition.initial_condition=2', 'initial_condition.init_file=T.dat')
    params = _cli_parameters(monkeypatch, _toml_with_eos(cfg_dir), *ic)
    assert params.initial_condition.init_file == str(cwd.resolve() / 'T.dat')


@pytest.mark.parametrize(
    'raw, expected',
    [('"eos.dat"', 'eos.dat'), ('"a b.dat"', 'a b.dat'), ('1', '1'), ('', None)],
)
def test_set_path_values_are_strings_against_cwd(dirs, raw, expected):
    out = _apply_overrides({'mesh': {}}, (f'mesh.eos_file={raw}',))['mesh']['eos_file']
    assert out == (str(Path.cwd() / expected) if expected else '')


@pytest.mark.parametrize('overrides', [(), ('solver.atol=1e-12',)])
def test_cli_missing_data_file_is_a_usage_error(dirs, monkeypatch, overrides):
    cfg_dir, _ = dirs
    monkeypatch.setattr('aragog.eos.entropy.EntropyEOS', lambda *a, **k: object())
    result = _cli(_toml_with_eos(cfg_dir, copy=False), *overrides)
    assert result.exit_code == 2, result.output
    assert 'could not read a data file' in result.output and 'eos.dat' in result.output
    assert '--set' not in result.output and '..' not in result.output.split('eos.dat')[-1]


@pytest.mark.parametrize(
    'template, suffix', [('abe_mixed_init', '.cfg'), ('abe_solid', '.toml')]
)
def test_new_keeps_template_format_and_data_paths(tmp_path, monkeypatch, template, suffix):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['new', 'copy', '--from', template])
    assert result.exit_code == 0, result.output
    assert [p.name for p in tmp_path.iterdir()] == [f'copy{suffix}']
    params = Parameters.from_file(str(tmp_path / f'copy{suffix}'))
    if template == 'abe_mixed_init':
        assert Path(params.mesh.eos_file) == EOS_TEST
        np.testing.assert_array_equal(params.mesh.eos_radius, np.loadtxt(EOS_TEST)[:, 0])


@pytest.mark.parametrize(
    'name, template, written',
    [
        ('bar.cfg', 'abe_solid', 'bar.cfg'),
        ('bar', 'abe_solid', 'bar.toml'),
        ('x.TOML', 'abe_solid', 'x.TOML'),
        ('foo.toml', 'abe_mixed_init', None),
    ],
)
def test_new_name_suffix_selects_template_form(tmp_path, monkeypatch, name, template, written):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['new', name, '--from', template])
    assert [p.name for p in tmp_path.iterdir()] == ([written] if written else [])
    if written:
        assert result.exit_code == 0, result.output
        Parameters.from_file(str(tmp_path / written))
    else:
        assert (
            result.exit_code != 0 and "unknown template 'abe_mixed_init.toml'" in result.output
        )


def test_absolute_template_paths_rewrites_only_existing_relative_data_files(tmp_path):
    (tmp_path / 'a.dat').write_text('1')
    target = (tmp_path / 'a.dat').resolve().as_posix()
    text = (
        'eos_file = a.dat  # inline comment\n'
        '# eos_file = a.dat\n'
        'init_file = b.dat\n'
        'eos_file = "/x/a.dat"\n'
        'init_file = "a.dat"\n'
        'solidus = "a.dat"\n'
    )
    assert _absolute_template_paths(text, tmp_path).splitlines() == [
        f'eos_file = {target}  # inline comment',
        '# eos_file = a.dat',
        'init_file = b.dat',
        'eos_file = "/x/a.dat"',
        f'init_file = "{target}"',
        'solidus = "a.dat"',
    ]
