"""Script to generate golden regression fixtures for aragog.

Executes test simulations to completion (status 0, full dt_actual),
verifies invariants, and records .npz fixtures storing the equation of
state SHA-256 hash and package versions.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

# Ensure repository src is first on sys.path before importing aragog
_SRC_DIR = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import scikits.odes  # noqa: E402

from aragog.cli import _derive_initial_entropy_from_config  # noqa: E402
from aragog.config import Config  # noqa: E402
from aragog.eos.entropy import EntropyEOS  # noqa: E402
from aragog.solver.entropy_solver import EntropySolver  # noqa: E402


def check_clean_git_tree() -> None:
    """Refuse to record fixtures if working tree is dirty."""
    if os.environ.get('ALLOW_DIRTY_TREE') == '1':
        return
    diff = subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    if diff:
        raise RuntimeError('Refusing to generate golden fixtures with a dirty git tree')


def get_sundials_version() -> str:
    """Return SUNDIALS version string from conda environment metadata."""
    env_dir = Path(sys.executable).parents[1]
    conda_meta = env_dir / 'conda-meta'
    if conda_meta.exists():
        for f in conda_meta.glob('sundials-*.json'):
            parts = f.stem.split('-')
            if len(parts) >= 2:
                return parts[1]
    return 'unknown'


def compute_eos_hash(eos_dir: Path) -> str:
    """Compute recursive SHA-256 hash over directory files using relative paths."""
    h = hashlib.sha256()
    for p in sorted(eos_dir.rglob('*')):
        if p.is_file():
            h.update(p.relative_to(eos_dir).as_posix().encode('utf-8'))
            h.update(p.read_bytes())
    return h.hexdigest()


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        return out
    except Exception:
        return 'unknown'


def get_mixed_phase_mush_initial_entropy(mesh) -> np.ndarray:
    """Return physical non-uniform initial entropy profile for mixed_phase_mush fixture."""
    r_stag = np.asarray(mesh.staggered.radii).ravel()
    r_basic = np.asarray(mesh.basic.radii).ravel()
    r_cmb = float(r_basic[0])
    r_surf = float(r_basic[-1])
    d = r_surf - r_cmb
    return 5050.0 - 100.0 * (r_stag - r_cmb) / d


def get_partly_locked_column_initial_entropy(mesh) -> np.ndarray:
    """Return initial entropy profile for partly locked column fixture.

    Upper part has melt fraction phi < phi_rheo (0.4) forming a locked lid,
    while the interior remains in the mushy regime (phi >= 0.4).
    """
    r_stag = np.asarray(mesh.staggered.radii).ravel()
    r_basic = np.asarray(mesh.basic.radii).ravel()
    r_cmb = float(r_basic[0])
    r_surf = float(r_basic[-1])
    d = r_surf - r_cmb
    return 5100.0 - 2600.0 * (r_stag - r_cmb) / d


def record_fixture(
    name: str,
    config_file: str,
    fixture: str,
    expected_dt: float,
    eos: EntropyEOS,
    eos_hash: str,
    commit: str,
    end_time: float | None = None,
    check_yielding: bool = False,
    initial_entropy: np.ndarray | float | None = None,
) -> None:
    """Run one solve to completion, verify invariants, and record fixture."""
    print(f'Running {name}...')
    config = Config.from_file(config_file)
    if end_time is not None:
        config.solver.end_time = end_time
    solver = EntropySolver(config, entropy_eos=eos)
    solver.initialize()
    if initial_entropy is None:
        initial_entropy = _derive_initial_entropy_from_config(solver)
    if initial_entropy is not None:
        solver.set_initial_entropy(initial_entropy)
    solver.solve()
    output = solver.get_state()

    assert output.status == 0, f'Expected status 0, got {output.status}'
    assert output.dt_actual == expected_dt, f'Expected dt {expected_dt}, got {output.dt_actual}'
    if check_yielding:
        assert np.sum(output.phi_basic > 0.5) >= 1
        eta_eff = solver.state.viscosity_basic
        eta_diff = solver.state.phase_basic.eta_diff
        assert np.sum(eta_eff < 0.5 * eta_diff) >= 2

    data = {
        'S': output.S_final,
        'T': output.T_stag,
        'eos_hash': eos_hash,
        'recorded_from_commit': commit,
        'recorded_numpy_version': np.__version__,
        'recorded_jax_version': getattr(jax, '__version__', 'unknown'),
        'recorded_scikits_odes_version': getattr(scikits.odes, '__version__', 'unknown'),
        'recorded_sundials_version': get_sundials_version(),
        'recorded_platform': sys.platform,
    }
    for k, v in output.__dict__.items():
        if isinstance(v, (np.ndarray, float, int, str, bool, np.generic)):
            data[k] = v

    fixture_path = Path('tests/reference') / fixture
    np.savez(fixture_path, **data)
    print(f'Wrote {fixture_path}')


def main() -> None:
    check_clean_git_tree()
    fwl_data = os.environ.get('FWL_DATA')
    candidates = [
        os.environ.get('ARAGOG_TEST_EOS_DIR'),
        f'{fwl_data}/aragog/spider_eos' if fwl_data else None,
    ]
    eos_dir = next((Path(p) for p in candidates if p and Path(p).exists()), None)
    if eos_dir is None:
        raise RuntimeError('EOS_DIR not found')

    eos_hash = compute_eos_hash(eos_dir)
    print(f'EOS hash: {eos_hash}')
    commit = get_git_commit()
    print(f'Commit: {commit}')

    eos = EntropyEOS(eos_dir)
    record_fixture(
        name='yielding_active_probe',
        config_file='tests/configs/yielding_active_probe.toml',
        fixture='yielding_active_probe.npz',
        expected_dt=100.0,
        eos=eos,
        eos_hash=eos_hash,
        commit=commit,
        check_yielding=True,
    )
    record_fixture(
        name='abe_solid',
        config_file='src/aragog/cfg/abe_solid.toml',
        fixture='rheology_disabled_abe_solid.toml.npz',
        expected_dt=100.0,
        eos=eos,
        eos_hash=eos_hash,
        commit=commit,
        end_time=100.0,
    )
    mush_config = Config.from_file('tests/configs/mixed_phase_mush.toml')
    mush_solver = EntropySolver(mush_config, entropy_eos=eos)
    mush_solver.initialize()
    mush_s0 = get_mixed_phase_mush_initial_entropy(mush_solver.evaluator.mesh)
    record_fixture(
        name='mixed_phase_mush',
        config_file='tests/configs/mixed_phase_mush.toml',
        fixture='mixed_phase_mush.npz',
        expected_dt=1.0,
        eos=eos,
        eos_hash=eos_hash,
        commit=commit,
        initial_entropy=mush_s0,
    )
    locked_config = Config.from_file('tests/configs/partly_locked_column.toml')
    locked_solver = EntropySolver(locked_config, entropy_eos=eos)
    locked_solver.initialize()
    locked_s0 = get_partly_locked_column_initial_entropy(locked_solver.evaluator.mesh)
    record_fixture(
        name='partly_locked_column',
        config_file='tests/configs/partly_locked_column.toml',
        fixture='partly_locked_column.npz',
        expected_dt=0.10,
        eos=eos,
        eos_hash=eos_hash,
        commit=commit,
        initial_entropy=locked_s0,
    )
    print('All golden fixtures recorded successfully.')


if __name__ == '__main__':
    main()
