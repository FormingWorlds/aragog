"""Tests for the ``separation_viscosity`` config option.

``EntropyPhaseEvaluator.relative_velocity`` (numpy) and
``aragog.jax.phase.relative_velocity`` (jax) select the gravitational-
separation drag viscosity from ``separation_viscosity``: ``'melt'``
uses the fixed single-phase liquid viscosity; ``'mixture'`` uses the
rheological-transition-blended bulk viscosity. Below phi_rheo the two
diverge sharply; well above phi_rheo the blend saturates to the melt
value and the two modes agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.eos.entropy_phase import EntropyPhaseEvaluator

pytestmark = pytest.mark.unit

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
jax.config.update('jax_enable_x64', True)

from aragog.jax.phase import PhaseParams, relative_velocity  # noqa: E402


class _StubPhaseBoundaryEOS:
    """Exposes only the phase-boundary density lookup relative_velocity needs."""

    def _lookup_at_phase_boundary(self, prop, P, phase):
        assert prop == 'density'
        if phase == 'solid':
            return np.full_like(np.asarray(P, dtype=float), 4200.0)
        elif phase == 'melt':
            return np.full_like(np.asarray(P, dtype=float), 3900.0)
        raise ValueError(phase)


class _StubJaxEOS:
    """JAX counterpart of ``_StubPhaseBoundaryEOS``."""

    def _lookup_at_phase_boundary(self, field, pressure, phase):
        assert field == 'density'
        if phase == 'solid':
            return jnp.asarray(4200.0)
        elif phase == 'melt':
            return jnp.asarray(3900.0)
        raise ValueError(phase)


def _numpy_relative_velocity(mode, viscosity_val):
    ev = EntropyPhaseEvaluator(
        entropy_eos=_StubPhaseBoundaryEOS(),
        gravitational_acceleration=10.0,
        grain_size=1e-3,
        viscosity_liquid=1e-1,
        separation_viscosity=mode,
    )
    ev.pressure = np.array([1.0e9])
    ev._density = np.array([4000.0])
    ev._viscosity_val = np.array([viscosity_val])
    return float(ev.relative_velocity().item())


def test_modes_diverge_below_phi_rheo_and_agree_above_it():
    """Below phi_rheo the blended mixture viscosity is far from melt
    (near-solid); above phi_rheo it saturates to the melt value.
    """
    v_melt_below = _numpy_relative_velocity('melt', viscosity_val=1e21)
    v_mixture_below = _numpy_relative_velocity('mixture', viscosity_val=1e21)
    assert v_melt_below != pytest.approx(v_mixture_below, rel=1e-6)

    v_melt_above = _numpy_relative_velocity('melt', viscosity_val=1e-1)
    v_mixture_above = _numpy_relative_velocity('mixture', viscosity_val=1e-1)
    assert v_melt_above == pytest.approx(v_mixture_above, rel=1e-12)


@pytest.mark.parametrize('mode', ['melt', 'mixture'])
def test_numpy_jax_parity_in_both_modes(mode):
    """numpy and jax must select the same drag viscosity and agree bit-tight."""
    viscosity_val = 1e2

    ev = EntropyPhaseEvaluator(
        entropy_eos=_StubPhaseBoundaryEOS(),
        gravitational_acceleration=10.0,
        grain_size=1e-3,
        viscosity_liquid=1e-1,
        separation_viscosity=mode,
    )
    ev.pressure = np.array([1.0e9])
    ev._density = np.array([4000.0])
    ev._viscosity_val = np.array([viscosity_val])
    v_numpy = float(ev.relative_velocity().item())

    params = PhaseParams(grain_size=1e-3, viscosity_liquid=1e-1, separation_viscosity=mode)
    v_jax = float(
        relative_velocity(
            _StubJaxEOS(),
            params,
            P=jnp.asarray(1.0e9),
            density=jnp.asarray(4000.0),
            melt_fraction=jnp.asarray(0.5),
            gravity=jnp.asarray(10.0),
            viscosity=jnp.asarray(viscosity_val),
        )
    )

    assert v_jax == pytest.approx(v_numpy, rel=1e-10)
