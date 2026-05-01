"""Pin the BKC -> Rumpf-Gupte -> Stokes regime-switching porosities.

The three-regime permeability model uses tanh-blended switches at two
critical porosities (Abe 1995; Soucasse Aragog formulation). Both the
numpy path (eos/entropy_phase.py) and the JAX path (jax/phase.py) must
agree on the constants byte-for-byte; otherwise gravitational separation
diverges between the analytic-Jacobian RHS and the numpy SolverOutput
post-processor that trusts the JAX trajectory.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'aragog'

_BKC_RG_THRESH = 0.0769452
_RG_STOKES_THRESH = 0.771462


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text()


@pytest.mark.unit
def test_numpy_permeability_thresholds_match_soucasse():
    """Numpy entropy_phase.py uses Soucasse permeability thresholds."""
    src = _read('eos/entropy_phase.py')
    assert f'tanh_weight(porosity, {_BKC_RG_THRESH}, 0.02)' in src, (
        'BKC -> RG threshold drifted from Soucasse spec'
    )
    assert f'tanh_weight(porosity, {_RG_STOKES_THRESH}, 0.05)' in src, (
        'RG -> Stokes threshold drifted from Soucasse spec'
    )


@pytest.mark.unit
def test_jax_permeability_thresholds_match_soucasse():
    """JAX jax/phase.py uses Soucasse permeability thresholds."""
    src = _read('jax/phase.py')
    assert f'tanh_weight(porosity, {_BKC_RG_THRESH}, 0.02)' in src, (
        'BKC -> RG threshold drifted from Soucasse spec'
    )
    assert f'tanh_weight(porosity, {_RG_STOKES_THRESH}, 0.05)' in src, (
        'RG -> Stokes threshold drifted from Soucasse spec'
    )


@pytest.mark.unit
def test_numpy_and_jax_permeability_thresholds_agree():
    """Numpy and JAX must pin the same numerical thresholds.

    Cross-path drift would silently bias gravitational separation
    between the analytic-Jacobian RHS (JAX) and the numpy
    SolverOutput post-processor.
    """
    pat = re.compile(r"tanh_weight\(porosity,\s*([0-9.]+),\s*([0-9.]+)\)")
    numpy_thresholds = pat.findall(_read('eos/entropy_phase.py'))
    jax_thresholds = pat.findall(_read('jax/phase.py'))

    assert numpy_thresholds, 'Could not parse numpy permeability thresholds'
    assert jax_thresholds, 'Could not parse JAX permeability thresholds'

    # Both files have exactly the same two regime-switch calls.
    assert set(numpy_thresholds) == set(jax_thresholds), (
        f'Numpy {numpy_thresholds} != JAX {jax_thresholds}'
    )
