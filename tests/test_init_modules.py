"""Code in the package ``__init__`` modules: logging, config dispatch and the mesh.

Covers ``aragog.debug_logger``, ``aragog.config.Config.from_file`` and the parts of
``aragog.mesh`` that the solver tests do not reach: the exact-root return of
``_radius_for_mass_coordinate``, the unknown-EOS and duplicate-radius errors of ``Mesh``,
and ``Mesh.quantity_at_staggered_nodes`` and ``Mesh.volume_average``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import aragog
from aragog import CFG_DATA
from aragog.config import Config
from aragog.mesh import Mesh, _radius_for_mass_coordinate
from aragog.parser import Parameters, _MeshParameters

pytestmark = pytest.mark.unit

R_CORE, R_SURF = 3480e3, 6371e3


def _settings(n_nodes: int = 6, mass_coordinates: bool = False) -> _MeshParameters:
    """Earth-like Adams-Williamson mantle mesh settings."""
    return _MeshParameters(
        outer_radius=R_SURF,
        inner_radius=R_CORE,
        number_of_nodes=n_nodes,
        mixing_length_profile='nearest_boundary',
        core_density=10738.0,
        surface_density=4090.0,
        gravitational_acceleration=9.81,
        adiabatic_bulk_modulus=260e9,
        mass_coordinates=mass_coordinates,
    )


def test_debug_logger_sets_debug_level_and_one_console_handler():
    """Two calls leave the package logger at DEBUG with one simple-format console handler."""
    pkg = logging.getLogger('fwl.aragog')
    saved_level, saved_handlers = pkg.level, list(pkg.handlers)
    try:
        pkg.setLevel(logging.WARNING)
        assert aragog.debug_logger() is pkg
        assert aragog.debug_logger() is pkg
        assert pkg.level == logging.DEBUG
        assert [type(h) for h in pkg.handlers] == [logging.StreamHandler]
        assert pkg.handlers[0].formatter._fmt == aragog.simple_formatter()._fmt
    finally:
        pkg.setLevel(saved_level)
        pkg.handlers = saved_handlers


def test_config_from_file_sends_a_toml_in_any_position_to_from_toml(monkeypatch):
    """A .toml among the files is read by Config.from_toml, even when it is not first."""
    calls = []
    monkeypatch.setattr(Config, 'from_toml', staticmethod(lambda f: calls.append(f) or 'toml'))
    monkeypatch.setattr(Parameters, 'from_file', classmethod(lambda cls, *f: 'ini'))
    assert Config.from_file('a.cfg', 'b.toml') == 'toml'
    assert calls == ['b.toml']


def test_config_from_file_sends_cfg_files_to_the_ini_parser(monkeypatch):
    """Without a .toml every file goes to Parameters.from_file, in order."""
    calls = []
    monkeypatch.setattr(Config, 'from_toml', staticmethod(lambda f: 'toml'))
    monkeypatch.setattr(
        Parameters, 'from_file', classmethod(lambda cls, *f: calls.append(f) or 'ini')
    )
    assert Config.from_file('a.cfg', 'b.cfg') == 'ini'
    assert calls == [('a.cfg', 'b.cfg')]


def test_config_from_file_reads_the_bundled_toml():
    """The bundled abe_solid.toml loads into Parameters with its 100 mesh nodes."""
    params = Config.from_file(str(CFG_DATA / 'abe_solid.toml'))
    assert isinstance(params, Parameters)
    assert params.mesh.number_of_nodes == 100


def test_radius_for_mass_coordinate_returns_the_upper_end_on_an_exact_root(caplog):
    """A target equal to xi at the upper bracket end returns that end without a clamp warning."""
    with caplog.at_level(logging.WARNING):
        r = _radius_for_mass_coordinate(lambda r: 2.0 * r, 10.0, 1.0, 5.0, node=3)
    assert r == 5.0
    assert not [rec for rec in caplog.records if 'bracket failed' in rec.getMessage()]


def test_mesh_rejects_an_unknown_eos_method():
    settings = _settings()
    settings.eos_method = 3
    with pytest.raises(ValueError, match='Unknown method to initialize Equation of State'):
        Mesh(SimpleNamespace(mesh=settings))


def test_mass_coordinate_mesh_rejects_duplicate_node_radii(monkeypatch):
    """Interior nodes all clamped to the lower bracket end give zero-width cells."""
    monkeypatch.setattr(
        'aragog.mesh._radius_for_mass_coordinate',
        lambda xi_of_r, xi_target, r_lo, r_hi, node=None: r_lo,
    )
    with pytest.raises(ValueError, match='Non-monotonic basic mesh'):
        Mesh(SimpleNamespace(mesh=_settings(mass_coordinates=True)))


def test_quantity_at_staggered_nodes_is_the_mean_of_the_cell_ends():
    mesh = Mesh(SimpleNamespace(mesh=_settings()))
    basic = np.array([[1.0, -2.0], [4.0, 0.5], [9.0, 3.0], [16.0, 7.5]])
    expected = np.array([[2.5, -0.75], [6.5, 1.75], [12.5, 5.25]])
    np.testing.assert_allclose(
        mesh.quantity_at_staggered_nodes(basic), expected, rtol=0, atol=0
    )


def test_volume_average_weights_each_cell_by_its_shell_volume():
    """Weights are the shell volumes 4/3 pi (r_out^3 - r_in^3) of the basic cells."""
    mesh = Mesh(SimpleNamespace(mesh=_settings(n_nodes=5)))
    r = np.ravel(mesh.basic.radii)
    shells = 4.0 / 3.0 * np.pi * (r[1:] ** 3 - r[:-1] ** 3)
    q = np.array([[1.0], [3.0], [7.0], [20.0]])
    expected = np.sum(np.ravel(q) * shells) / np.sum(shells)
    assert mesh.volume_average(q) == pytest.approx(expected, rel=1e-12)
    assert expected != pytest.approx(np.mean(q), rel=1e-3)
