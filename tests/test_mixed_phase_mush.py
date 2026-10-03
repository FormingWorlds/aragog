from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from aragog.config import Config
from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver
from tools.verification.generate_golden_fixtures import (
    compute_eos_hash,
    get_mixed_phase_mush_initial_entropy,
    get_sundials_version,
)

jax = pytest.importorskip('jax')
scikits_odes = pytest.importorskip('scikits.odes')

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


@pytest.mark.smoke
def test_mixed_phase_mush_fixture(shared_eos):
    """Verify mixed-phase mush fixture against recorded reference fixture."""
    config_file = 'tests/configs/mixed_phase_mush.toml'
    config = Config.from_file(config_file)

    solver = EntropySolver(config, entropy_eos=shared_eos)
    solver.initialize()
    s0 = get_mixed_phase_mush_initial_entropy(solver.evaluator.mesh)
    solver.set_initial_entropy(s0)

    solver.solve()
    output = solver.get_state()
    S = output.S_final
    T = output.T_stag

    assert output.status == 0, f'Expected status 0, got {output.status}'
    assert output.dt_actual == 1.0, f'Expected dt 1.0, got {output.dt_actual}'

    # Check mixed-phase active invariants: mantle in mushy region
    phi = output.phi_basic
    assert np.all(phi >= 0.45), 'Expected entire mantle in mixed/mushy region'
    assert np.sum(phi >= 0.5) >= 30, 'Expected at least 30 nodes with phi >= 0.5'

    fixture_path = os.path.join(os.path.dirname(__file__), 'reference', 'mixed_phase_mush.npz')

    with np.load(fixture_path) as ref:
        assert 'recorded_from_commit' in ref.files
        assert 'eos_hash' in ref.files
        assert str(ref['eos_hash']) == compute_eos_hash(EOS_DIR)

        version_checks = {
            'recorded_numpy_version': np.__version__,
            'recorded_jax_version': getattr(jax, '__version__', None),
            'recorded_scikits_odes_version': getattr(scikits_odes, '__version__', None),
            'recorded_sundials_version': get_sundials_version(),
            'recorded_platform': sys.platform,
        }
        exact_match = True
        mismatches = []
        for ver_key, cur_ver in version_checks.items():
            if ver_key in ref.files and cur_ver is not None:
                if str(ref[ver_key]) != str(cur_ver):
                    exact_match = False
                    mismatches.append(
                        f'{ver_key}: recorded {ref[ver_key]} vs current {cur_ver}'
                    )

        if exact_match:
            np.testing.assert_array_equal(S, ref['S'])
            np.testing.assert_array_equal(T, ref['T'])
            for flux_key in ('heat_flux', 'conv_flux', 'cond_flux'):
                if flux_key in ref.files:
                    np.testing.assert_array_equal(getattr(output, flux_key), ref[flux_key])
        else:
            # Tolerance tier per Ruling 75/77. Source: run 37108391411
            # Measured Linux-Darwin spread: dS=7.89e-4 J/kg/K, dT=8.67e-5 K, flux=76.5 W/m2
            np.testing.assert_allclose(S, ref['S'], atol=0.01)
            np.testing.assert_allclose(T, ref['T'], atol=0.001)
            for flux_key in ('heat_flux', 'conv_flux', 'cond_flux'):
                if flux_key in ref.files:
                    np.testing.assert_allclose(
                        getattr(output, flux_key), ref[flux_key], atol=800.0
                    )
