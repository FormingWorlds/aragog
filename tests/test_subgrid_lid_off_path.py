"""With the sub-grid lid off, the solve path equals the stored run of the code without the lid.

The reference (``tests/reference/subgrid_lid_v0_reference.npz``) is the off-path probe of
``tests/reference/subgrid_lid_v0_probe.py`` on the commit in its ``base_commit`` key: the mesh,
one right-hand side and a 2-yr solve, without and with the solid rheology ('lid' closure), on
CVODE with the finite-difference Jacobian and with a JAX factory (built with
``lid_params=None``). The mesh and the right-hand side must match to rtol 1e-10 and the solve to
rtol 1e-9, above the round-off between the macOS and Linux builds; a JAX case must have called
the JAX right-hand side. On each core mode a 5-yr NumPy solve also matches its end state and its
energy integrals to rtol 1e-8, with tidal and radiogenic heating. The solver residual and the
core energy are differences of large terms, so they match to 1e-15 and 1e-12 of the surface
energy (measured between macOS and Linux: 8.2e-17 and 2.2e-14); they act only across platforms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.reference.subgrid_lid_v0_probe import CASES, MODES, case_name, mode_probe, probe
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(600), needs_eos]

OFF_PATH_RTOL = 1e-8
REFERENCE = Path(__file__).resolve().parent / 'reference' / 'subgrid_lid_v0_reference.npz'


@pytest.fixture(scope='module')
def reference():
    with np.load(REFERENCE) as ref:
        return {k: ref[k] for k in ref.files}


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize(('rheology', 'jax'), CASES, ids=[case_name(*c) for c in CASES])
def test_lid_off_path_equals_the_base(reference, rheology, jax):
    """Off by default, the lid leaves the mesh, the right-hand side and a short solve as they
    were on the base commit, without and with the solid rheology, on both Jacobian paths."""
    from aragog.solver import entropy_solver as es

    if not es._CVODE_AVAILABLE:
        pytest.skip('scikits.odes is not installed')
    if jax:
        pytest.importorskip('jax')
        pytest.importorskip('equinox')
    got = probe(rheology, jax, EOS_DIR, {'lid_params': None} if jax else None)
    name = case_name(rheology, jax)

    def ref(key):
        return reference[f'{name}/{key}']

    for key in ('r_basic', 'r_stag', 'volume', 'rhs'):
        np.testing.assert_allclose(got[key], ref(key), rtol=1e-10, atol=0)
    assert int(got['status'][0]) == int(ref('status')[0]) == 0
    assert got['t'][-1] == pytest.approx(float(ref('t')[-1]), rel=1e-12)
    np.testing.assert_allclose(got['y_end'], ref('y_end'), rtol=1e-9, atol=0)
    if jax:
        assert int(got['jax_rhs_calls'][0]) > 0
        assert int(ref('jax_rhs_calls')[0]) > 0
    else:
        assert int(got['jax_rhs_calls'][0]) == 0
    assert got['y_end'].size == got['r_stag'].size  # no lid states with the lid off


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('mode', MODES)
def test_lid_off_core_mode_equals_the_base(reference, mode):
    """Off by default, the lid leaves the end state and every energy integral of a 5-yr solve
    as they were on the base commit, on each core mode the lid supports."""
    from aragog.solver import entropy_solver as es

    if not es._CVODE_AVAILABLE:
        pytest.skip('scikits.odes is not installed')
    got = mode_probe(mode, EOS_DIR)
    stored = {
        k.split('/', 1)[1]: reference[k] for k in reference if k.startswith(f'mode_{mode}/')
    }
    assert int(got['status'][0]) == int(stored['status'][0]) == 0
    assert got['t_end'][0] == pytest.approx(float(stored['t_end'][0]), rel=1e-12)
    np.testing.assert_allclose(got['y_end'], stored['y_end'], rtol=OFF_PATH_RTOL, atol=0)
    integrals = {k for k in stored if k.startswith('ei/')}
    assert len(integrals) > 3
    assert all(float(stored[k][0]) != 0.0 for k in ('ei/F_int', 'ei/Q_radio', 'ei/Q_tidal'))
    assert integrals <= set(got)
    heat = abs(float(stored['ei/F_int'][0]))
    limits = {'ei/solver_residual': 1e-15 * heat, 'ei/core': 1e-12 * heat}
    for key in integrals - set(limits):
        np.testing.assert_allclose(
            got[key], stored[key], rtol=OFF_PATH_RTOL, atol=0, err_msg=key
        )
    for key, limit in limits.items():
        assert abs(got[key][0] - stored[key][0]) <= limit, key
    assert not np.isclose(
        float(stored['ei/Q_radio'][0]), float(stored['ei/Q_tidal'][0]), rtol=0.1
    )
