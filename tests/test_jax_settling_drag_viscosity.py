"""Regression test for the JAX settling-drag viscosity source.

``relative_velocity`` computes the melt-solid drag velocity as
``abs_drho * gravity * F / eta_l``, where ``eta_l`` is meant to be the
rheological-transition-blended mixture viscosity passed in through the
``viscosity`` argument, not a fixed liquid-only value. Before the fix,
``eta_l`` was read from ``params.visc_liquid``, a constant set once at
construction, so the drag velocity never responded to the local phase
state.

This test calls ``relative_velocity`` twice with the same porosity
state but two different ``viscosity`` values, and checks that the
output responds inversely and exactly to that change. A regression to
the fixed-viscosity behavior would make both calls return the same
value, since nothing else passed in differs between them.
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
