from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scikits_odes_sundials')
pytest.importorskip('jax')

from aragog.cli import _derive_initial_entropy_from_config
from aragog.config import Config
from aragog.eos.entropy import EntropyEOS
from aragog.solver.entropy_solver import EntropySolver
from tests.fixture_helpers import (
    compute_eos_hash,
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


# Reference fixture for yielding active regression.


@pytest.mark.smoke
def test_yielding_active_probe(shared_eos):
    """Verify yielding active probe against recorded reference fixture.

    Parameters
    ----------
    shared_eos : EntropyEOS
        Shared equation of state evaluator.

    Notes
    -----
    Tolerances use rtol = max(10 * floor / |value|, 1e-8) for integrated states,
    with atol only for quantities that can be zero (fluxes):
    - S: rtol = 1.4e-7 (1-ulp noise floor: 4.83e-5 J/kg/K)
    - T: rtol = 4.3e-8 (1-ulp noise floor: 8.51e-6 K)
    - Flux: atol = 500.0 W m^-2, rtol = 1.0e-4 (1-ulp noise floor: 9.66 W/m^2)
    - Lid stress: rtol = 2.7e-7 (measured 1-ulp noise floor: 2.99e-4 Pa)
    """
    config_file = 'tests/configs/yielding_active_probe.toml'
    config = Config.from_file(config_file)

    solver = EntropySolver(config, entropy_eos=shared_eos)
    solver.initialize()
    initial_entropy = _derive_initial_entropy_from_config(solver)
    if initial_entropy is not None:
        solver.set_initial_entropy(initial_entropy)

    solver.solve()
    output = solver.get_state()
    S = output.S_final
    T = output.T_stag

    assert output.status == 0, f'Expected status 0, got {output.status}'
    assert output.dt_actual == 100.0, f'Expected dt 100.0, got {output.dt_actual}'

    # Check yielding active invariants
    phi = output.phi_basic
    assert np.sum(phi > 0.5) >= 1, 'Expected at least one node with phi > 0.5'

    eta_eff = solver.state.viscosity_basic
    eta_diff = solver.state.phase_basic.eta_diff
    assert np.sum(eta_eff < 0.5 * eta_diff) >= 2, (
        'Expected at least 2 nodes with eta_eff < 0.5 * eta_diff'
    )

    fixture_path = os.path.join(
        os.path.dirname(__file__), 'reference', 'yielding_active_probe.npz'
    )

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
            # Tolerance tier: rtol = max(10 * floor / |value|, 1e-8) for integrated states,
            # with atol only for values that can be zero (fluxes).
            np.testing.assert_allclose(S, ref['S'], rtol=1.4e-7)
            np.testing.assert_allclose(T, ref['T'], rtol=4.3e-8)
            for flux_key in ('heat_flux', 'jconv_b', 'jcond_b'):
                assert flux_key in ref.files, f'Missing flux key {flux_key!r} in fixture'
                np.testing.assert_allclose(
                    getattr(output, flux_key), ref[flux_key], atol=500.0, rtol=1.0e-4
                )
            if 'lid_stress' in ref.files:
                np.testing.assert_allclose(
                    getattr(output, 'lid_stress'), ref['lid_stress'], rtol=2.7e-7
                )
