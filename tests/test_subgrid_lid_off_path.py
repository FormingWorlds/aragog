"""With the sub-grid lid off, the solve path equals the commit the lid branches from.

The reference (``tests/reference/subgrid_lid_v0_reference.npz``) is the off-path probe of
``tests/reference/subgrid_lid_v0_probe.py`` on that commit: the mesh, one right-hand side and
a 2-yr solve, without and with the solid rheology ('lid' closure), on CVODE with the
finite-difference Jacobian and with a JAX factory. The mesh must match exactly, the
right-hand side to rtol 1e-12 and the solve to rtol 1e-9; a JAX case must have called the JAX
right-hand side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.reference.subgrid_lid_v0_probe import CASES, case_name, probe
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, needs_eos

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(600), needs_eos]

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
    got = probe(rheology, jax, EOS_DIR)
    name = case_name(rheology, jax)

    def ref(key):
        return reference[f'{name}/{key}']

    for key in ('r_basic', 'r_stag', 'volume'):
        np.testing.assert_array_equal(got[key], ref(key))
    np.testing.assert_allclose(got['rhs'], ref('rhs'), rtol=1e-12, atol=0)
    assert int(got['status'][0]) == int(ref('status')[0]) == 0
    assert got['t'][-1] == pytest.approx(float(ref('t')[-1]), rel=1e-12)
    np.testing.assert_allclose(got['y_end'], ref('y_end'), rtol=1e-9, atol=0)
    if jax:
        assert int(got['jax_rhs_calls'][0]) > 0
        assert int(ref('jax_rhs_calls')[0]) > 0
    else:
        assert int(got['jax_rhs_calls'][0]) == 0
    assert got['y_end'].size == got['r_stag'].size  # no lid states with the lid off
