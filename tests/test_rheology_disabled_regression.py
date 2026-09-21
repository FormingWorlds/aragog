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
    '/Users/timlichtenberg/git/PROTEUS/output/coupled_parity/spider/data/spider_eos',
    '/Users/timlichtenberg/FWL_DATA/aragog/spider_eos',
]
EOS_DIR = next((Path(p) for p in _CANDIDATES if p and Path(p).exists()), None)


@pytest.fixture(scope='module')
def shared_eos():
    if EOS_DIR is None:
        pytest.skip('EOS_DIR not found')
    return EntropyEOS(EOS_DIR)


# Fixtures recorded from aragog origin/main at aa3c94d (2026-09-21)


@pytest.mark.unit
@pytest.mark.parametrize(
    'config_file,suffix',
    [
        ('src/aragog/cfg/abe_solid.toml', 'abe_solid.toml'),
        ('src/aragog/cfg/abe_mixed.cfg', 'abe_mixed.cfg'),
    ],
)
@pytest.mark.unit
@pytest.mark.parametrize('solver_impl', ['numpy', 'jax'])
def test_rheology_disabled_regression(config_file, suffix, solver_impl, shared_eos):
    config = Config.from_file(config_file)
    config.solver.max_steps = 10

    if solver_impl == 'jax':
        config.solver.implementation = 'jax'
    else:
        config.solver.implementation = 'numpy'

    solver = EntropySolver(config, entropy_eos=shared_eos)
    solver.initialize()
    initial_entropy = _derive_initial_entropy_from_config(solver)
    if initial_entropy is not None:
        solver.set_initial_entropy(initial_entropy)

    try:
        ret = solver.solve()
    except Exception:
        ret = None

    if ret is None:
        output = solver.get_state()
        S = output.S_final
        T = output.T_stag
    else:
        S, T, output = ret

    fixture_path = os.path.join(
        os.path.dirname(__file__), 'reference', f'rheology_disabled_{suffix}_{solver_impl}.npz'
    )

    with np.load(fixture_path) as ref:
        np.testing.assert_array_equal(S, ref['S'])
        np.testing.assert_array_equal(T, ref['T'])

        out_dict = output.__dict__
        for k in ref.files:
            if k in ['S', 'T']:
                continue
            if k in out_dict:
                np.testing.assert_array_equal(out_dict[k], ref[k], err_msg=f'Mismatch in {k}')
