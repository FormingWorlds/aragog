from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from aragog.cli import _derive_initial_entropy_from_config
from aragog.config import Config
from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver

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
    config.solver.max_steps = 10

    solver = EntropySolver(config, entropy_eos=shared_eos)
    solver.initialize()
    initial_entropy = _derive_initial_entropy_from_config(solver)
    if initial_entropy is not None:
        solver.set_initial_entropy(initial_entropy)

    solver.solve()
    output = solver.get_state()
    S = output.S_final
    T = output.T_stag

    fixture_path = os.path.join(
        os.path.dirname(__file__), 'reference', f'rheology_disabled_{suffix}.npz'
    )

    with np.load(fixture_path) as ref:
        # Fixtures are recorded from aragog main, so equality here is identity with main.
        assert 'recorded_from_commit' in ref.files
        np.testing.assert_array_equal(S, ref['S'])
        np.testing.assert_array_equal(T, ref['T'])

        out_dict = output.__dict__
        for k in ref.files:
            if k in ['S', 'T']:
                continue
            if k in out_dict:
                np.testing.assert_array_equal(out_dict[k], ref[k], err_msg=f'Mismatch in {k}')
