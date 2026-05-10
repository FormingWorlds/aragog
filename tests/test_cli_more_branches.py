"""Round-2 coverage for ``aragog.cli`` edge branches.

The remaining 25 missing lines fall into four buckets after the
existing ``test_cli.py`` and ``test_cli_vnv_branches.py`` are
applied:

* ``_derive_initial_entropy_from_config`` early-return paths
  (lines 234, 240, 243, 255): IC method != 1/3, T_target <= 0,
  EOS missing, S_hi <= S_lo.
* ``aragog run`` ``--eos-dir`` env-var fallback (line 395) and
  ``--initial-dsdr-cmb != 0`` without ``--initial-entropy`` guard
  (lines 450-451).
* ``list_configs`` empty-bundle path (line 499) and the comment-
  scan ``OSError`` / ``continue`` / ``break`` paths (lines 512,
  517, 521).
* ``_serialize_params`` recursive handling of list / numpy-scalar
  inputs (lines 732, 736).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────────
#       _derive_initial_entropy_from_config early-return branches
# ──────────────────────────────────────────────────────────────────────


class _StubSolver:
    """Compact solver stub for the IC-derivation tests.

    Constructed with kwargs that drive each early-return branch:
    ``ic_method`` (1/2/3), ``T_target``, ``eos`` (None or an object),
    ``S_lo`` / ``S_hi`` (the table entropy axis bounds).
    """

    def __init__(
        self,
        *,
        ic_method: int,
        T_target: float,
        eos,
        S_lo: float = 500.0,
        S_hi: float = 5500.0,
        surface_pressure: float = 0.0,
    ):
        self.parameters = type(
            'P',
            (),
            {
                'initial_condition': type(
                    'IC',
                    (),
                    {
                        'initial_condition': ic_method,
                        'surface_temperature': T_target,
                    },
                )(),
                'mesh': type('M', (), {'surface_pressure': surface_pressure})(),
            },
        )()
        self.eos = eos
        if eos is not None:
            eos._tables = {
                'temperature_solid': {'S_list': [S_lo, S_hi]},
                'temperature_melt': {'S_list': [S_lo, S_hi]},
            }


def test_derive_initial_entropy_returns_none_when_ic_method_not_1_or_3():
    """IC method 2 (user T file) must short-circuit to None
    (line 234), preserving the file-based IC.
    """
    from aragog.cli import _derive_initial_entropy_from_config

    solver = _StubSolver(ic_method=2, T_target=3000.0, eos=type('E', (), {})())
    assert _derive_initial_entropy_from_config(solver) is None


def test_derive_initial_entropy_returns_none_when_target_temperature_nonpositive():
    """T_target = 0 (or negative) must return None (line 240).

    Discriminator: surface_temperature is the user-facing knob; a
    zero or negative value is invalid input but must not crash.
    """
    from aragog.cli import _derive_initial_entropy_from_config

    solver = _StubSolver(ic_method=1, T_target=0.0, eos=type('E', (), {})())
    assert _derive_initial_entropy_from_config(solver) is None
    solver_neg = _StubSolver(ic_method=1, T_target=-100.0, eos=type('E', (), {})())
    assert _derive_initial_entropy_from_config(solver_neg) is None


def test_derive_initial_entropy_returns_none_when_eos_is_missing():
    """When the solver carries no EOS (const_properties path), the
    derivation must return None (line 243), letting the caller emit
    the explicit "--initial-entropy is required" message.
    """
    from aragog.cli import _derive_initial_entropy_from_config

    solver = _StubSolver(ic_method=1, T_target=3000.0, eos=None)
    # Stub solver has no entropy_eos attr either.
    assert _derive_initial_entropy_from_config(solver) is None


def test_derive_initial_entropy_returns_none_when_table_axis_collapsed():
    """When the intersection of solid and melt entropy axes is empty
    (S_hi <= S_lo), derivation must return None (line 255). This
    guards against a malformed EOS table.
    """
    from aragog.cli import _derive_initial_entropy_from_config

    bad_eos = type('E', (), {})()
    # S_lo == S_hi triggers the S_hi > S_lo guard.
    solver = _StubSolver(ic_method=1, T_target=3000.0, eos=bad_eos, S_lo=2900.0, S_hi=2900.0)
    assert _derive_initial_entropy_from_config(solver) is None


# ──────────────────────────────────────────────────────────────────────
#                  --initial-dsdr-cmb without --initial-entropy
# ──────────────────────────────────────────────────────────────────────


def test_run_warns_when_initial_dsdr_cmb_set_without_initial_entropy(tmp_path, monkeypatch):
    """Lines 450-451: setting ``--initial-dsdr-cmb`` to a non-zero
    value without ``--initial-entropy`` (or without a derivable
    config-side IC) must surface as a UsageError naming both flags.

    Discriminator: silently ignoring the non-zero dSdr_cmb would let
    a misconfigured config pass through with the default cold-start
    dSdr_cmb=0, which in CHILI runs would lock the boundary state at
    zero from t=0 onward.
    """
    from click.testing import CliRunner

    from aragog.cli import cli

    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# minimal stub\n')
    eos = tmp_path / 'eos'
    eos.mkdir()

    class _SolverWithoutDerivableIC:
        # IC method 2 disables auto-derivation; combined with a
        # non-zero --initial-dsdr-cmb, the CLI must still raise
        # because --initial-entropy is missing.
        parameters = type(
            'P',
            (),
            {
                'boundary_conditions': type('BC', (), {'core_bc': 'energy_balance'})(),
                'initial_condition': type(
                    'IC',
                    (),
                    {'initial_condition': 2, 'surface_temperature': 0.0},
                )(),
                'mesh': type('M', (), {'surface_pressure': 0.0})(),
            },
        )()
        eos = None

        def initialize(self):
            pass

        def set_initial_dSdr_cmb(self, _):
            pass

    import aragog.solver as _solver_mod

    monkeypatch.setattr(
        _solver_mod.EntropySolver,
        'from_file',
        classmethod(lambda cls, **kw: _SolverWithoutDerivableIC()),
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
            '--initial-dsdr-cmb',
            '5.0e-9',
        ],
    )
    assert result.exit_code != 0
    assert '--initial-entropy is required' in (result.output or '')


# ──────────────────────────────────────────────────────────────────────
#                  list_configs comment-scan branches
# ──────────────────────────────────────────────────────────────────────


def test_list_configs_skips_unreadable_entry(monkeypatch, tmp_path):
    """Lines 512-513 / 521-522: when reading a bundled config's first
    comment line raises OSError / UnicodeDecodeError, the loop must
    catch and continue rather than crashing the whole listing.

    Discriminator: a regression that let the exception propagate
    would make ``aragog list-configs`` crash if a single bundled
    file was unreadable (e.g. a partial wheel install).
    """

    from click.testing import CliRunner

    import aragog.cli as cli_mod

    # Build a fake cfg dir with two files: one normal toml + one
    # binary file that triggers UnicodeDecodeError on read.
    fake_cfg = tmp_path / 'cfg'
    fake_cfg.mkdir()
    (fake_cfg / 'good.toml').write_text('# nice description\n[mesh]\n', encoding='utf-8')
    (fake_cfg / 'bad.toml').write_bytes(b'\xff\xfeBOM\x00garbage\xff\xff')

    class _FakeTraversable:
        def __init__(self, base: Path):
            self._base = base

        def iterdir(self):
            for p in self._base.iterdir():
                yield _FakeEntry(p)

    class _FakeEntry:
        def __init__(self, path: Path):
            self._path = path
            self.name = path.name

        @property
        def parts(self):
            return self._path.parts

        def is_file(self):
            return self._path.is_file()

        def open(self, mode='r', encoding=None, **kw):
            return self._path.open(mode, encoding=encoding, **kw)

        def read_text(self, encoding=None):
            return self._path.read_text(encoding=encoding)

    monkeypatch.setattr(cli_mod, '_bundled_cfg_dir', lambda: _FakeTraversable(fake_cfg))

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ['list-configs'])
    # Must list at least the good entry and exit cleanly.
    assert result.exit_code == 0, f'list-configs crashed on bad file: {result.output!r}'
    assert 'good.toml' in result.output


# ──────────────────────────────────────────────────────────────────────
#                  _serialize_params recursion (list, np scalar)
# ──────────────────────────────────────────────────────────────────────


def test_version_message_handles_missing_dependency(monkeypatch):
    """Lines 60-61 of cli.py: ``_version_message`` builds the
    ``aragog --versions`` block by importing each dependency by name;
    when one raises ImportError or other exceptions, the line must
    fall back to ``"not installed"`` rather than aborting the whole
    listing.

    Discriminator: a regression that let the exception propagate
    would make ``aragog --versions`` (the bug-report-friendly mode)
    crash on any environment that lacks one of the optional deps.
    """
    import importlib

    import aragog.cli as cli_mod

    real_import = importlib.import_module

    def _faulty_import(name, package=None):
        if name == 'jax':
            raise ImportError('intentional test fault')
        return real_import(name, package=package)

    monkeypatch.setattr(importlib, 'import_module', _faulty_import)
    msg = cli_mod._version_message()
    assert 'aragog ' in msg
    assert 'jax: not installed' in msg, (
        f'expected "jax: not installed" line on ImportError; got {msg!r}'
    )


def test_vnv_handles_module_spec_load_failure(monkeypatch, tmp_path):
    """Line 921 of cli.py: when ``importlib.util.spec_from_file_location``
    cannot build a spec / loader, ``aragog vnv <topic>`` must raise
    ClickException with a "could not load" message.

    Discriminator: forced spec=None via monkeypatch makes the loader
    branch fire. A regression that swallowed the failure or let the
    None-spec into ``module_from_spec`` would crash with AttributeError
    deeper in the stack instead of the user-facing "could not load".
    """
    import importlib.util

    from click.testing import CliRunner

    import aragog.cli as cli_mod

    figures_dir = tmp_path / 'figs'
    figures_dir.mkdir()
    (figures_dir / 'verify_demo.py').write_text('def main(): pass\n', encoding='utf-8')
    monkeypatch.setattr(cli_mod, '_vnv_figures_dir', lambda: figures_dir)

    real_spec_from_file = importlib.util.spec_from_file_location

    def _none_spec(*args, **kwargs):
        return None

    monkeypatch.setattr(importlib.util, 'spec_from_file_location', _none_spec)
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ['vnv', 'demo'])
    assert result.exit_code != 0
    assert 'could not load' in (result.output or '').lower()
    # Restore (autouse cleanup happens via monkeypatch but assert).
    monkeypatch.setattr(importlib.util, 'spec_from_file_location', real_spec_from_file)


def test_serialize_params_walks_radionuclide_list_and_unboxes_np_scalars():
    """Lines 732, 736: ``_serialize_params`` recurses into lists and
    unboxes numpy scalars via ``.item()``.

    Discriminator: ``Parameters.radionuclides`` is a list of
    ``_Radionuclide`` dataclasses, so the list branch fires. To force
    the numpy-scalar branch we shoehorn an ndarray onto a private
    field via dataclasses.replace; the asdict walk hits the array and
    must call ``.item()``. A regression that returned the np.float64
    directly would surface as ``np.float64(20.0)`` literal in the
    JSON output.
    """
    from aragog.cli import _serialize_params
    from aragog.parser import _Radionuclide

    # Build a single radionuclide with a numpy-scalar field.
    rn = _Radionuclide(
        name='U235',
        t0_years=0.0,
        abundance=np.float64(0.5),  # numpy scalar -> tests line 736
        concentration=1.0e-9,
        heat_production=1.0e-5,
        half_life_years=4.5e9,
    )

    @__import__('dataclasses').dataclass
    class _Wrapper:
        items: list

    wrapper = _Wrapper(items=[rn])
    out = _serialize_params(wrapper)
    assert isinstance(out, dict)
    assert isinstance(out['items'], list)
    inner = out['items'][0]
    # numpy scalar must be unboxed to native float
    assert not isinstance(inner['abundance'], np.generic), (
        f'numpy scalar leaked through _serialize_params: {type(inner["abundance"])}'
    )
    assert inner['abundance'] == pytest.approx(0.5)
