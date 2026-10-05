from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scikits_odes_sundials')
pytest.importorskip('jax')

from aragog.config import Config
from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver
from tests.fixture_helpers import (
    compute_eos_hash,
    get_creep_locked_base_initial_entropy,
    get_package_version,
    get_sundials_version,
)

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
def test_creep_locked_base_fixture(shared_eos):
    """Verify creep locked base fixture against recorded reference fixture.

    Parameters
    ----------
    shared_eos : EntropyEOS
        Shared equation of state evaluator.

    Notes
    -----
    Tolerances are calibrated to 10x max(1-ulp noise floor, cross-CPU hardware spread):
    - S: atol = 1.0e-3 J kg^-1 K^-1 (physics mutant moves S by 3.93 J/kg/K, >3900x atol)
    - T: atol = 1.3e-4 K (physics mutant moves T by 0.5 K, >3800x atol)
    - Flux: atol = 170.0 W m^-2 (< 0.2% of sub-lid convective flux ~1e5 W/m2)
    - Lid stress: atol = 1.8e-3 Pa (measured 1-ulp noise floor: 1.75e-4 Pa)
    """
    config_file = 'tests/configs/creep_locked_base.toml'
    config = Config.from_file(config_file)

    solver = EntropySolver(config, entropy_eos=shared_eos)
    solver.initialize()
    s0 = get_creep_locked_base_initial_entropy(solver.evaluator.mesh)
    solver.set_initial_entropy(s0)

    solver.solve()
    output = solver.get_state()
    S = output.S_final
    T = output.T_stag

    assert output.status == 0, f'Expected status 0, got {output.status}'
    assert output.dt_actual == 0.10, f'Expected dt 0.10, got {output.dt_actual}'

    # Check creep locked base active invariants: creep locked lower cells and mushy upper cells
    phi = output.phi_basic
    assert np.sum(phi < 0.4) >= 5, 'Expected lower cells in creep locked regime (phi < 0.4)'
    assert np.sum(phi >= 0.4) >= 5, 'Expected upper cells in mushy regime (phi >= 0.4)'

    fixture_path = os.path.join(os.path.dirname(__file__), 'reference', 'creep_locked_base.npz')

    with np.load(fixture_path) as ref:
        assert 'recorded_from_commit' in ref.files
        assert 'eos_hash' in ref.files
        assert str(ref['eos_hash']) == compute_eos_hash(EOS_DIR)

        version_checks = {
            'recorded_numpy_version': np.__version__,
            'recorded_jax_version': get_package_version('jax'),
            'recorded_scikits_odes_version': get_package_version('scikits-odes-sundials'),
            'recorded_sundials_version': get_sundials_version(),
            'recorded_platform': sys.platform,
        }
        exact_match = True
        for ver_key, cur_ver in version_checks.items():
            if ver_key in ref.files and cur_ver is not None:
                if str(ref[ver_key]) != str(cur_ver):
                    exact_match = False
                    break

        if exact_match:
            np.testing.assert_array_equal(S, ref['S'])
            np.testing.assert_array_equal(T, ref['T'])
            for flux_key in ('heat_flux', 'jconv_b', 'jcond_b'):
                assert flux_key in ref.files, f'Missing flux key {flux_key!r} in fixture'
                np.testing.assert_array_equal(getattr(output, flux_key), ref[flux_key])
            if 'lid_stress' in ref.files:
                np.testing.assert_array_equal(getattr(output, 'lid_stress'), ref['lid_stress'])
        else:
            # Tolerance tier: 10x max(1-ulp noise floor, cross-CPU hardware spread).
            # Measured 1-ulp noise floor: S 8.76e-6 J/kg/K, T 1.39e-6 K, flux 141.1 W/m2, lid_stress 1.75e-4 Pa.
            np.testing.assert_allclose(S, ref['S'], atol=1.0e-3)
            np.testing.assert_allclose(T, ref['T'], atol=1.3e-4)
            for flux_key in ('heat_flux', 'jconv_b', 'jcond_b'):
                assert flux_key in ref.files, f'Missing flux key {flux_key!r} in fixture'
                np.testing.assert_allclose(getattr(output, flux_key), ref[flux_key], atol=170.0)
            if 'lid_stress' in ref.files:
                np.testing.assert_allclose(
                    getattr(output, 'lid_stress'), ref['lid_stress'], atol=1.8e-3
                )
