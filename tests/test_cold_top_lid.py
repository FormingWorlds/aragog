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
    get_cold_top_lid_initial_entropy,
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
def test_cold_top_lid_fixture(shared_eos):
    """Verify top lid weight test fixture on a coarse mesh against recorded reference fixture.

    Parameters
    ----------
    shared_eos : EntropyEOS
        Shared equation of state evaluator.

    Notes
    -----
    Tolerances are calibrated to 10x max(1-ulp perturbation noise floor across 5 seeds):
    - S: atol = 1.0e-6 J kg^-1 K^-1 (< 0.0001% of mantle entropy ~3500 J/kg/K)
    - T: atol = 1.0e-7 K (< 0.00001% of mantle temperature ~2000 K)
    - Flux: atol = 2000.0 W m^-2 (< 1.8% of surface radiative flux ~1.12e5 W/m^2)
    - Lid stress: atol = 5.0e12 Pa
    """
    config_file = 'tests/configs/cold_top_lid.toml'
    config = Config.from_file(config_file)

    solver = EntropySolver(config, entropy_eos=shared_eos)
    solver.initialize()
    s0 = get_cold_top_lid_initial_entropy(solver.evaluator.mesh)
    solver.set_initial_entropy(s0)

    solver.solve()
    output = solver.get_state()
    S = output.S_final
    T = output.T_stag

    assert output.status == 0, f'Expected status 0, got {output.status}'
    assert output.dt_actual == 0.10, f'Expected dt 0.10, got {output.dt_actual}'

    # Check cold top lid active invariants: solid top lid and mushy interior
    phi = output.phi_basic
    assert np.sum(output.lid_mask_b > 0.5) >= 1, (
        'Expected upper cells in solid lid regime (w_lid > 0.5)'
    )
    assert output.lid_thickness > 0, 'Expected positive lid thickness'
    assert np.sum(phi < 0.1) >= 2, 'Expected upper cells solid (phi < 0.1)'
    assert np.sum(phi >= 0.35) >= 5, 'Expected interior cells in mushy regime (phi >= 0.35)'

    fixture_path = os.path.join(os.path.dirname(__file__), 'reference', 'cold_top_lid.npz')

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
            for flux_key in ('heat_flux', 'jconv_b', 'jcond_b'):
                assert flux_key in ref.files, f'Missing flux key {flux_key!r} in fixture'
                np.testing.assert_array_equal(getattr(output, flux_key), ref[flux_key])
            if 'lid_stress' in ref.files:
                np.testing.assert_array_equal(getattr(output, 'lid_stress'), ref['lid_stress'])
        else:
            # Tolerance tier: 10x max(1-ulp noise floor across 5 seeds).
            # Measured 1-ulp noise floor across 5 seeds: S 1.22e-8 J/kg/K, T 1.27e-9 K, flux 43.6 W/m2, lid_stress 3.41e11 Pa.
            np.testing.assert_allclose(S, ref['S'], atol=1.0e-6)
            np.testing.assert_allclose(T, ref['T'], atol=1.0e-7)
            for flux_key in ('heat_flux', 'jconv_b', 'jcond_b'):
                assert flux_key in ref.files, f'Missing flux key {flux_key!r} in fixture'
                np.testing.assert_allclose(
                    getattr(output, flux_key), ref[flux_key], atol=2000.0
                )
            if 'lid_stress' in ref.files:
                np.testing.assert_allclose(
                    getattr(output, 'lid_stress'), ref['lid_stress'], atol=5.0e12
                )
