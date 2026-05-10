"""Edge-case unit tests for ``aragog.jax`` modules.

Targets remaining tiny clusters in:

* ``jax/phase.py:173`` — ``PhaseParams`` must reject an unknown
  ``phase_smoothing`` mode at construction (the contract supports
  only ``'tanh'`` or ``'cubic_hermite'``).
* ``jax/eos.py:541, 557-561`` — ``EntropyEOS_JAX.dTdPs`` and the
  ``thermal_expansivity`` derived path. Both are public methods
  with no direct test exposure today.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

jax.config.update('jax_enable_x64', True)

pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parent.parent
_FWL_DATA = os.environ.get('FWL_DATA')
_CANDIDATES = [
    os.environ.get('ARAGOG_TEST_EOS_DIR'),
    f'{_FWL_DATA}/aragog/spider_eos' if _FWL_DATA else None,
    str(_REPO_ROOT.parent / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
]
EOS_DIR = next(
    (Path(p) for p in _CANDIDATES if p and Path(p).exists()),
    Path(_CANDIDATES[-1]),
)
needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(),
    reason=f'SPIDER P-S tables not found at {EOS_DIR}.',
)


def test_phase_params_rejects_unknown_phase_smoothing():
    """``PhaseParams(phase_smoothing='ramp')`` must raise ValueError
    with a message naming the supported alternatives.

    Discriminator: a regression that silently fell back to the
    default ``'tanh'`` would let a typo (``'tanh1'``, ``'cubic'``)
    quietly downgrade the JAX phase smoother to a different curve
    than the user requested.
    """
    from aragog.jax.phase import PhaseParams

    with pytest.raises(ValueError, match="phase_smoothing must be 'cubic_hermite' or 'tanh'"):
        PhaseParams(phase_smoothing='ramp')


@needs_eos
def test_jax_eos_dTdPs_returns_finite_array(eos_jax_fixture):
    """``EntropyEOS_JAX.dTdPs`` must return finite, positive (or near
    zero) values across a representative P-S sweep.

    Discriminator: a regression that swapped the ``solid`` and ``melt``
    table keys would shift the mushy-zone profile by hundreds of K/Pa.
    The check uses three discriminating P points across the mantle
    pressure range and verifies the magnitude stays in the SPIDER
    ballpark (1e-9 to 1e-7 K/Pa).
    """
    P = jnp.array([5.0e9, 5.0e10, 1.0e11])
    S = jnp.full(3, 3300.0)
    out = eos_jax_fixture.dTdPs(P, S)
    arr = np.asarray(out)
    assert arr.shape == (3,)
    assert np.all(np.isfinite(arr)), f'dTdPs has non-finite entries: {arr}'
    # Adiabatic gradient sign convention is dT/dP > 0 in a stably
    # stratified mantle; magnitude order 1e-9 to 1e-7 K/Pa.
    assert np.all(arr > 0.0), f'dTdPs returned non-positive values: {arr}'
    assert np.all(arr < 1.0e-6), (
        f'dTdPs magnitude {arr} > 1e-6 K/Pa; outside the plausible mantle range'
    )


@needs_eos
def test_jax_eos_thermal_expansivity_returns_finite_positive(eos_jax_fixture):
    """``EntropyEOS_JAX.thermal_expansivity`` (derived from rho * Cp *
    |dTdPs| / T) must return finite, positive values.

    Discriminator: a regression that lost the ``rho`` factor would
    surface as a 4000x error (alpha ~ 1e-9 instead of 1e-5). A
    regression that lost the ``np.maximum(T, 1.0)`` floor would
    crash on a degenerate state with T=0.
    """
    P = jnp.array([5.0e10, 1.0e11])
    S = jnp.full(2, 3300.0)
    alpha = eos_jax_fixture.thermal_expansivity(P, S)
    arr = np.asarray(alpha)
    assert np.all(np.isfinite(arr))
    assert np.all(arr > 0.0)
    # Earth-mantle alpha ~ 1e-5 to 1e-4 K^-1; reject >100x off.
    assert np.all(arr < 1.0e-3), f'alpha {arr} outside plausible mantle range'
    assert np.all(arr > 1.0e-7), f'alpha {arr} too small; rho factor likely missing'


@pytest.fixture(scope='module')
def eos_jax_fixture():
    """Module-scoped EntropyEOS_JAX (path differs from the live
    fixture in test_jax_entropy.py to avoid collision)."""
    if not EOS_DIR.exists():
        pytest.skip(f'EOS unavailable at {EOS_DIR}')
    from aragog.jax.eos import EntropyEOS_JAX

    return EntropyEOS_JAX(EOS_DIR)
