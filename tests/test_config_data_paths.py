"""Relative ``eos_file`` and ``init_file`` resolve against the config directory, then the CWD."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from aragog.config import Config
from aragog.parser import Parameters

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = REPO / 'src' / 'aragog' / 'cfg'
EOS_TEST = REPO / 'data' / 'test' / 'eos_test.dat'


def _ini_with(tmp_dir: Path, eos_file: str, extra: str = '') -> Path:
    """Copy abe_mixed_init.cfg into ``tmp_dir`` with the given eos_file line."""
    text = (CFG_DIR / 'abe_mixed_init.cfg').read_text()
    text = text.replace('eos_file = ../../../data/test/eos_test.dat', f'eos_file = {eos_file}')
    assert f'eos_file = {eos_file}' in text
    path = tmp_dir / 'run.cfg'
    path.write_text(text.replace('[initial_condition]\n', f'[initial_condition]\n{extra}'))
    return path


def test_bundled_mixed_init_loads_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = Parameters.from_file(str(CFG_DIR / 'abe_mixed_init.cfg'))
    assert Path(params.mesh.eos_file) == EOS_TEST
    np.testing.assert_array_equal(params.mesh.eos_radius, np.loadtxt(EOS_TEST)[:, 0])


def test_config_dir_wins_over_cwd(tmp_path, monkeypatch):
    cfg_dir, cwd = tmp_path / 'cfg', tmp_path / 'cwd'
    cfg_dir.mkdir()
    cwd.mkdir()
    shutil.copy(EOS_TEST, cfg_dir / 'eos.dat')
    shutil.copy(EOS_TEST, cwd / 'eos.dat')
    monkeypatch.chdir(cwd)
    params = Parameters.from_file(str(_ini_with(cfg_dir, 'eos.dat')))
    assert params.mesh.eos_file == str(cfg_dir.resolve() / 'eos.dat')


def test_cwd_fallback_becomes_absolute(tmp_path, monkeypatch):
    cfg_dir, cwd = tmp_path / 'cfg', tmp_path / 'cwd'
    cfg_dir.mkdir()
    cwd.mkdir()
    shutil.copy(EOS_TEST, cwd / 'eos.dat')
    monkeypatch.chdir(cwd)
    params = Parameters.from_file(str(_ini_with(cfg_dir, 'eos.dat')))
    assert params.mesh.eos_file == str(cwd.resolve() / 'eos.dat')
    assert params.mesh.eos_radius.size == np.loadtxt(EOS_TEST).shape[0]


def test_absolute_path_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = Parameters.from_file(str(_ini_with(tmp_path, str(EOS_TEST))))
    assert params.mesh.eos_file == str(EOS_TEST)


def test_init_file_resolves_against_config_dir(tmp_path, monkeypatch):
    cfg_dir = tmp_path / 'cfg'
    cfg_dir.mkdir()
    np.savetxt(cfg_dir / 'T.dat', [[5.4e6, 3000.0], [6.3e6, 2000.0]])
    monkeypatch.chdir(tmp_path)
    cfg = _ini_with(cfg_dir, 'eos.dat', extra='init_file = T.dat\n')
    shutil.copy(EOS_TEST, cfg_dir / 'eos.dat')
    cfg.write_text(cfg.read_text().replace('initial_condition = 3', 'initial_condition = 2'))
    params = Parameters.from_file(str(cfg))
    assert params.initial_condition.init_file == str(cfg_dir.resolve() / 'T.dat')
    assert params.initial_condition.init_temperature.shape == (2, 2)


@pytest.mark.parametrize('loader', ['parser', 'config'])
def test_toml_eos_file_resolves_against_config_dir(tmp_path, monkeypatch, loader):
    cfg_dir = tmp_path / 'cfg'
    cfg_dir.mkdir()
    toml = _toml_with_eos(cfg_dir)
    monkeypatch.chdir(tmp_path)
    load = Parameters.from_file if loader == 'parser' else Config.from_toml
    params = load(str(toml))
    assert params.mesh.eos_file == str(cfg_dir.resolve() / 'eos.dat')


def _toml_with_eos(cfg_dir: Path) -> Path:
    shutil.copy(EOS_TEST, cfg_dir / 'eos.dat')
    toml = cfg_dir / 'run.toml'
    text = (CFG_DIR / 'abe_solid.toml').read_text()
    toml.write_text(
        text.replace('[mesh]\n', '[mesh]\neos_method = 2\neos_file = "eos.dat"\n', 1)
    )
    return toml


def test_from_dict_without_config_dir_leaves_path_as_given(tmp_path, monkeypatch):
    import tomllib

    toml = _toml_with_eos(tmp_path)
    monkeypatch.chdir(tmp_path)
    params = Config.from_dict(tomllib.loads(toml.read_text()))
    assert params.mesh.eos_file == 'eos.dat'


def _cli_parameters(monkeypatch, toml: Path, cwd: Path, *overrides: str):
    """Run ``aragog run TOML --set ...`` from ``cwd`` with a stub solver; return its Parameters."""
    from click.testing import CliRunner

    from aragog.cli import cli

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
    monkeypatch.chdir(cwd)
    args = ['run', str(toml), '--eos-dir', str(cwd), '--initial-entropy', '2900']
    result = CliRunner().invoke(cli, args + [a for o in overrides for a in ('--set', o)])
    assert result.exit_code == 0, result.output
    return captured['p']


def test_cli_set_resolves_against_config_dir(tmp_path, monkeypatch):
    cfg_dir, cwd = tmp_path / 'cfg', tmp_path / 'cwd'
    cfg_dir.mkdir()
    cwd.mkdir()
    params = _cli_parameters(monkeypatch, _toml_with_eos(cfg_dir), cwd, 'solver.atol=1e-12')
    assert params.mesh.eos_file == str(cfg_dir.resolve() / 'eos.dat')


def test_cli_set_path_value_resolves_against_cwd(tmp_path, monkeypatch):
    cfg_dir, cwd = tmp_path / 'cfg', tmp_path / 'cwd'
    cfg_dir.mkdir()
    cwd.mkdir()
    toml = _toml_with_eos(cfg_dir)
    shutil.copy(EOS_TEST, cwd / 'eos.dat')
    params = _cli_parameters(monkeypatch, toml, cwd, 'mesh.eos_file=eos.dat')
    assert params.mesh.eos_file == str(cwd.resolve() / 'eos.dat')


@pytest.mark.parametrize(
    'template, suffix', [('abe_mixed_init', '.cfg'), ('abe_solid', '.toml')]
)
def test_new_keeps_template_format_and_data_paths(tmp_path, monkeypatch, template, suffix):
    from click.testing import CliRunner

    from aragog.cli import cli

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['new', 'copy', '--from', template])
    assert result.exit_code == 0, result.output
    assert [p.name for p in tmp_path.iterdir()] == [f'copy{suffix}']
    params = Parameters.from_file(str(tmp_path / f'copy{suffix}'))
    if template == 'abe_mixed_init':
        assert Path(params.mesh.eos_file) == EOS_TEST
        np.testing.assert_array_equal(params.mesh.eos_radius, np.loadtxt(EOS_TEST)[:, 0])
