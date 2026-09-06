"""Regression tests for the JAX settling-drag viscosity source.

``relative_velocity`` computes the melt-solid drag velocity as
``abs_drho * gravity * F / eta_l``. ``eta_l`` is selected by
``params.separation_viscosity_mixture`` via ``jnp.where``: in mixture
mode it tracks the rheological-transition-blended ``viscosity``
argument; in melt mode it is the fixed single-phase liquid viscosity
and the ``viscosity`` argument is unused.

The first test calls ``relative_velocity`` twice in mixture mode with
the same porosity state but two different ``viscosity`` values, and
checks that the output responds inversely and exactly to that change.
The second test checks that melt mode is immune to a NaN passed
through the unused ``viscosity`` argument, which an arithmetic blend
(rather than a branch select) would not be.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
jax.config.update('jax_enable_x64', True)

from aragog.jax.phase import PhaseParams, relative_velocity  # noqa: E402


class _StubEOS:
    """Exposes only the phase-boundary density lookup relative_velocity needs."""

    def _lookup_at_phase_boundary(self, field, pressure, phase):
        assert field == 'density'
        if phase == 'solid':
            return jnp.asarray(4200.0)
        elif phase == 'melt':
            return jnp.asarray(3900.0)
        raise ValueError(phase)


def test_relative_velocity_scales_with_viscosity_argument_not_fixed_field():
    """In mixture mode, the drag velocity must track the passed-in viscosity."""
    eos = _StubEOS()
    params = PhaseParams(grain_size=1e-3, viscosity_liquid=1e-1, separation_viscosity='mixture')
    P = jnp.asarray(1.0e9)
    density = jnp.asarray(4000.0)
    melt_fraction = jnp.asarray(0.5)
    gravity = jnp.asarray(10.0)

    v1 = relative_velocity(
        eos, params, P, density, melt_fraction, gravity, viscosity=jnp.asarray(1.0e2)
    )
    v2 = relative_velocity(
        eos, params, P, density, melt_fraction, gravity, viscosity=jnp.asarray(1.0e4)
    )

    assert float(v1) != float(v2)
    ratio = float(v1) / float(v2)
    assert ratio == pytest.approx(1.0e4 / 1.0e2, rel=1e-10)


def test_relative_velocity_melt_mode_immune_to_nan_viscosity_argument():
    """A NaN in the unused ``viscosity`` argument must not leak into melt mode."""
    eos = _StubEOS()
    params_melt = PhaseParams(
        grain_size=1e-3, viscosity_liquid=1e-1, separation_viscosity='melt'
    )
    params_mixture = PhaseParams(
        grain_size=1e-3, viscosity_liquid=1e-1, separation_viscosity='mixture'
    )
    P = jnp.asarray(1.0e9)
    density = jnp.asarray(4000.0)
    melt_fraction = jnp.asarray(0.5)
    gravity = jnp.asarray(10.0)

    v_melt_nan = relative_velocity(
        eos, params_melt, P, density, melt_fraction, gravity, viscosity=jnp.asarray(jnp.nan)
    )
    v_melt_finite = relative_velocity(
        eos, params_melt, P, density, melt_fraction, gravity, viscosity=jnp.asarray(1.0e2)
    )

    assert jnp.isfinite(v_melt_nan)
    assert float(v_melt_nan) == pytest.approx(float(v_melt_finite), rel=1e-12)

    v_mixture_nan = relative_velocity(
        eos, params_mixture, P, density, melt_fraction, gravity, viscosity=jnp.asarray(jnp.nan)
    )
    assert jnp.isnan(v_mixture_nan)
