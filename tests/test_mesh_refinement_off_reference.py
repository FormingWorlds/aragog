"""The mesh refinement off path against a stored main reference.

Module under test: ``aragog.mesh.Mesh`` with ``mesh.surface_cell_thickness`` and
``mesh.cmb_cell_thickness`` at 0, through the abe_solid mesh, one NumPy RHS evaluation
and a 100-year solve on the NumPy RHS and on the JAX CVODE factory. The reference
``tests/reference/mesh_refinement_off_main_reference.npz`` holds the same arrays from a
main tree, written by ``tools/verification/generate_mesh_refinement_off_reference.py``.
Invariants: the mesh arrays and the RHS equal to 1e-10 and the end states to 1e-6
relative, and the JAX solve calls the JAX RHS in both the reference and the test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'tests' / 'reference' / 'mesh_refinement_off_main_reference.npz'
GEN = ROOT / 'tools' / 'verification' / 'generate_mesh_refinement_off_reference.py'
# Linux against the macOS reference: mesh 1.6e-12, RHS 2.8e-14 of the largest value, end states 9.8e-8
MESH_RTOL, RHS_RTOL, SOLVE_RTOL = 1.0e-10, 1.0e-10, 1.0e-6


def _generator(monkeypatch):
    """The generator module, loaded from the repository root; its sys.path entries stay in
    this test."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    spec = importlib.util.spec_from_file_location('mesh_refinement_off_generator', GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def ref():
    return np.load(REF)


def test_the_reference_names_a_commit_and_a_jax_solve(ref):
    """The stored file records a 40-character commit and a JAX solve with JAX RHS calls."""
    assert len(str(ref['source_commit'])) == 40
    for mode in ('mass', 'radius'):
        assert int(ref[f'{mode}/solve/jax/rhs_calls']) > 0


@pytest.mark.parametrize('mode,mass_coordinates', [('mass', True), ('radius', False)])
def test_the_refinement_off_path_equals_main(ref, mode, mass_coordinates, monkeypatch):
    """With both cell thicknesses at 0 the mesh arrays and the NumPy RHS equal main to
    1e-10 and both end states equal main to 1e-6, relative to each value or to the largest
    value of the array; the margin over 0 is for the round-off of another platform."""
    gen = _generator(monkeypatch)
    assert gen.CFG.is_file()
    out = gen._case(mass_coordinates)
    assert out['solve/jax/rhs_calls'] > 0
    expected = {
        k.removeprefix(f'{mode}/'): ref[k] for k in ref.files if k.startswith(f'{mode}/')
    }
    assert set(out) == set(expected)
    for key, rtol in (
        *((f'mesh/{name}', MESH_RTOL) for name in gen.MESH_ARRAYS),
        ('rhs/numpy', RHS_RTOL),
        ('solve/numpy/y_end', SOLVE_RTOL),
        ('solve/jax/y_end', SOLVE_RTOL),
    ):
        e = expected[key]
        atol = rtol * float(np.max(np.abs(e)))
        np.testing.assert_allclose(out[key], e, rtol=rtol, atol=atol, err_msg=key)
