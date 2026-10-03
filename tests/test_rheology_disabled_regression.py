from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import scikits.odes

from aragog.cli import _derive_initial_entropy_from_config
from aragog.config import Config
from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver
from tools.verification.generate_golden_fixtures import compute_eos_hash, get_sundials_version

_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
]
EOS_DIR = next((Path(p) for p in _CANDIDATES if p and Path(p).exists()), None)


@pytest.fixture(scope='module')
def shared_eos():
    if EOS_DIR is None:
        pytest.skip('EOS_DIR not found')
    return EntropyEOS(EOS_DIR)


# Reference fixtures for the rheology-disabled code path.


@pytest.mark.smoke
@pytest.mark.parametrize(
    'config_file,suffix',
    [
        ('src/aragog/cfg/abe_solid.toml', 'abe_solid.toml'),
        ('src/aragog/cfg/abe_mixed.cfg', 'abe_mixed.cfg'),
    ],
)
def test_rheology_disabled_regression(config_file, suffix, shared_eos):
    config = Config.from_file(config_file)
    if 'abe_solid' in config_file:
        config.solver.end_time = 100.0

    solver = EntropySolver(config, entropy_eos=shared_eos)
    solver.initialize()
    initial_entropy = _derive_initial_entropy_from_config(solver)
    if initial_entropy is not None:
        solver.set_initial_entropy(initial_entropy)

    solver.solve()
    output = solver.get_state()
    S = output.S_final
    T = output.T_stag

    expected_dt = 100.0 if 'abe_solid' in config_file else 200.0
    assert output.status == 0, f'Expected status 0, got {output.status}'
    assert output.dt_actual == expected_dt, f'Expected dt {expected_dt}, got {output.dt_actual}'

    fixture_path = os.path.join(
        os.path.dirname(__file__), 'reference', f'rheology_disabled_{suffix}.npz'
    )

    # Rheology-off fixtures differ from main because this branch applies inner BC 3 as half-cell conduction to the prescribed temperature.
    with np.load(fixture_path) as ref:
        assert 'recorded_from_commit' in ref.files
        assert 'eos_hash' in ref.files
        assert str(ref['eos_hash']) == compute_eos_hash(EOS_DIR)

        version_checks = {
            'recorded_numpy_version': np.__version__,
            'recorded_jax_version': getattr(jax, '__version__', None),
            'recorded_scikits_odes_version': getattr(scikits.odes, '__version__', None),
            'recorded_sundials_version': get_sundials_version(),
            'recorded_platform': sys.platform,
        }
        for ver_key, cur_ver in version_checks.items():
            if ver_key in ref.files and cur_ver is not None:
                assert str(ref[ver_key]) == str(cur_ver), (
                    f'Fixture {ver_key} mismatch: recorded {ref[ver_key]}, current {cur_ver}'
                )

        np.testing.assert_array_equal(S, ref['S'])
        np.testing.assert_array_equal(T, ref['T'])

        metadata_keys = {
            'S',
            'T',
            'eos_hash',
            'recorded_from_commit',
            'recorded_numpy_version',
            'recorded_jax_version',
            'recorded_scikits_odes_version',
            'recorded_sundials_version',
            'recorded_platform',
        }
        for k in set(ref.files) - metadata_keys:
            assert k in output.__dict__, f'Reference key {k} missing from solver output'
            np.testing.assert_array_equal(
                output.__dict__[k], ref[k], err_msg=f'Mismatch in {k}'
            )
