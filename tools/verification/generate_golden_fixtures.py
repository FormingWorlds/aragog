"""Script to generate golden regression fixtures for aragog.

Executes test simulations to completion (status 0, full dt_actual),
verifies invariants, and records .npz fixtures storing the equation of
state SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

from aragog.cli import _derive_initial_entropy_from_config
from aragog.config import Config
from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver


def compute_eos_hash(eos_dir: Path) -> str:
    """Compute recursive SHA-256 hash over directory files."""
    h = hashlib.sha256()
    for p in sorted(eos_dir.rglob('*')):
        if p.is_file():
            h.update(p.name.encode('utf-8'))
            h.update(p.read_bytes())
    return h.hexdigest()


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        return out
    except Exception:
        return 'unknown'


def record_yielding_active_probe(eos: EntropyEOS, eos_hash: str, commit: str) -> None:
    """Record yielding active probe fixture."""
    print('Running yielding_active_probe...')
    config = Config.from_file('tests/configs/yielding_active_probe.toml')
    solver = EntropySolver(config, entropy_eos=eos)
    solver.initialize()
    s0 = _derive_initial_entropy_from_config(solver)
    if s0 is not None:
        solver.set_initial_entropy(s0)
    solver.solve()
    output = solver.get_state()

    assert output.status == 0, f'Expected status 0, got {output.status}'
    assert output.dt_actual == 100.0, f'Expected dt 100.0, got {output.dt_actual}'
    assert np.sum(output.phi_basic > 0.5) >= 1
    eta_eff = solver.state.viscosity_basic
    eta_diff = solver.state.phase_basic.eta_diff
    assert np.sum(eta_eff < 0.5 * eta_diff) >= 2

    fixture_path = Path('tests/reference/yielding_active_probe.npz')
    data = {
        'S': output.S_final,
        'T': output.T_stag,
        'eos_hash': eos_hash,
        'recorded_from_commit': commit,
        'recorded_numpy_version': np.__version__,
    }
    for k, v in output.__dict__.items():
        if isinstance(v, (np.ndarray, float, int, str, bool, np.generic)):
            data[k] = v
    np.savez(fixture_path, **data)
    print(f'Wrote {fixture_path}')


def record_rheology_disabled_solid(eos: EntropyEOS, eos_hash: str, commit: str) -> None:
    """Record rheology disabled abe_solid fixture."""
    print('Running abe_solid...')
    config = Config.from_file('src/aragog/cfg/abe_solid.toml')
    config.solver.end_time = 100.0
    solver = EntropySolver(config, entropy_eos=eos)
    solver.initialize()
    s0 = _derive_initial_entropy_from_config(solver)
    if s0 is not None:
        solver.set_initial_entropy(s0)
    solver.solve()
    output = solver.get_state()

    assert output.status == 0, f'Expected status 0, got {output.status}'
    assert output.dt_actual == 100.0, f'Expected dt 100.0, got {output.dt_actual}'

    fixture_path = Path('tests/reference/rheology_disabled_abe_solid.toml.npz')
    data = {
        'S': output.S_final,
        'T': output.T_stag,
        'eos_hash': eos_hash,
        'recorded_from_commit': commit,
        'recorded_numpy_version': np.__version__,
    }
    for k, v in output.__dict__.items():
        if isinstance(v, (np.ndarray, float, int, str, bool, np.generic)):
            data[k] = v
    np.savez(fixture_path, **data)
    print(f'Wrote {fixture_path}')


def record_rheology_disabled_mixed(eos: EntropyEOS, eos_hash: str, commit: str) -> None:
    """Record rheology disabled abe_mixed fixture."""
    print('Running abe_mixed...')
    config = Config.from_file('src/aragog/cfg/abe_mixed.cfg')
    solver = EntropySolver(config, entropy_eos=eos)
    solver.initialize()
    s0 = _derive_initial_entropy_from_config(solver)
    if s0 is not None:
        solver.set_initial_entropy(s0)
    solver.solve()
    output = solver.get_state()

    assert output.status == 0, f'Expected status 0, got {output.status}'
    assert output.dt_actual == 200.0, f'Expected dt 200.0, got {output.dt_actual}'

    fixture_path = Path('tests/reference/rheology_disabled_abe_mixed.cfg.npz')
    data = {
        'S': output.S_final,
        'T': output.T_stag,
        'eos_hash': eos_hash,
        'recorded_from_commit': commit,
        'recorded_numpy_version': np.__version__,
    }
    for k, v in output.__dict__.items():
        if isinstance(v, (np.ndarray, float, int, str, bool, np.generic)):
            data[k] = v
    np.savez(fixture_path, **data)
    print(f'Wrote {fixture_path}')


def main() -> None:
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
    record_yielding_active_probe(eos, eos_hash, commit)
    record_rheology_disabled_solid(eos, eos_hash, commit)
    record_rheology_disabled_mixed(eos, eos_hash, commit)
    print('All golden fixtures recorded successfully.')


if __name__ == '__main__':
    main()
