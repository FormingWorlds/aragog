"""``entropy_at_temperature`` of the numpy and JAX EOS: the inverse of the temperature at fixed
pressure in every phase, the table edge beyond its range, and the implicit derivatives."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import entropy_eos_copy, entropy_eos_jax, needs_eos

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')

pytestmark = [pytest.mark.smoke, needs_eos]


def _states(eos, p):
    """Entropies [J/kg/K] in the solid, the mush and the melt at pressure ``p``."""
    s_sol = float(np.asarray(eos.solidus_entropy(p)).flat[0])
    s_liq = float(np.asarray(eos.liquidus_entropy(p)).flat[0])
    return s_sol - 300.0, 0.5 * (s_sol + s_liq), s_liq + 300.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('p', [1.37e9, 5.31e10, 1.312e11])
def test_the_entropy_inverts_the_temperature_in_every_phase(p):
    """At off-grid pressures the numpy inverse returns the entropy of a given temperature to
    1e-9 in the solid, the mush and the melt, and the JAX inverse equals it."""
    eos, eos_j = entropy_eos_copy(), entropy_eos_jax()
    for s in _states(eos, p):
        t = float(np.asarray(eos.temperature(p, s)).flat[0])
        s_np = eos.entropy_at_temperature(p, t)
        assert s_np == pytest.approx(s, rel=1e-9)
        assert float(
            eos_j.entropy_at_temperature(jnp.asarray(p), jnp.asarray(t))
        ) == pytest.approx(s_np, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_both_inverses_find_the_root_next_to_a_flat_phase_boundary():
    """The tables hold T(S) flat next to the solidus and the liquidus; up to 1e-6 K and one ulp
    from either boundary temperature, at pressures where 100 Brent iterations are too few and near
    the CMB pressure of the tests, both inverses return an entropy at the target temperature, the
    JAX one with |dS/dT| below 20 J kg^-1 K^-2 and |dS/dP| below 1e-5 J kg^-1 K^-1 Pa^-1, and
    both return NaN for a NaN pressure or target."""
    eos, eos_j = entropy_eos_copy(), entropy_eos_jax()
    inverse = jax.jit(lambda p, t: eos_j.entropy_at_temperature(p, t))
    grad = jax.jit(jax.grad(inverse, argnums=(0, 1)))
    for p in np.r_[np.linspace(1e9, 1.49e11, 60)[14:16], 1.457e11]:
        for edge in (eos.solidus_entropy(p), eos.liquidus_entropy(p)):
            t_edge = eos.temperature_scalar(p, float(edge))
            ulp = np.spacing(t_edge)
            for t in t_edge + np.array([-1e-11, -1e-12, -ulp, 0.0, ulp, 1e-12, 1e-9, 1e-6]):
                s_jax = inverse(jnp.asarray(p), jnp.asarray(t))
                for s in (eos.entropy_at_temperature(p, t), float(s_jax)):
                    assert eos.temperature_scalar(p, s) == pytest.approx(t, abs=1e-10)
                d_p, d_t = grad(jnp.asarray(p), jnp.asarray(t))
                assert abs(float(d_t)) < 20.0 and abs(float(d_p)) < 1e-5
    for p, t in ((1.3e11, np.nan), (np.nan, 4000.0)):
        assert np.isnan(eos.entropy_at_temperature(p, t))
        assert np.isnan(inverse(jnp.asarray(p), jnp.asarray(t)))


def test_a_temperature_beyond_the_tables_gives_the_table_edge():
    """Below and above the tables' temperature range both inverses return the entropy edge, and
    the JAX derivative there is zero."""
    eos, eos_j = entropy_eos_copy(), entropy_eos_jax()
    solid, melt = eos._tables['temperature_solid'], eos._tables['temperature_melt']
    edges = min(solid['S'][0], melt['S'][0]), max(solid['S'][-1], melt['S'][-1])
    for t, edge in ((1.0, edges[0]), (1e6, edges[1])):
        assert eos.entropy_at_temperature(1.3e11, t) == edge
        s_j = eos_j.entropy_at_temperature(jnp.asarray(1.3e11), jnp.asarray(t))
        assert float(s_j) == pytest.approx(edge, rel=1e-12)
        grad = jax.grad(lambda x: eos_j.entropy_at_temperature(jnp.asarray(1.3e11), x))(t)
        assert float(grad) == 0.0


def test_a_rising_edge_slope_keeps_the_inverse_at_the_edge(monkeypatch):
    """Where the temperature still rises at the table edge, a temperature beyond the range gives
    the edge with zero derivatives, not the implicit derivatives of that slope."""
    eos, eos_j = entropy_eos_copy(), entropy_eos_jax()
    solid, melt = eos._tables['temperature_solid'], eos._tables['temperature_melt']
    edges = min(solid['S'][0], melt['S'][0]), max(solid['S'][-1], melt['S'][-1])
    cls = type(eos_j)
    tables = cls.temperature
    monkeypatch.setattr(cls, 'temperature', lambda self, P, S: tables(self, P, S) + 1e-3 * S)
    for t, edge in ((1.0, edges[0]), (1e6, edges[1])):
        p_t = jnp.asarray(1.3e11), jnp.asarray(t)
        assert float(eos_j.entropy_at_temperature(*p_t)) == pytest.approx(edge, rel=1e-12)
        grads = jax.grad(eos_j.entropy_at_temperature, argnums=(0, 1))(*p_t)
        assert [float(g) for g in grads] == [0.0, 0.0]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('phase', [0, 1, 2])
def test_the_jax_inverse_carries_the_implicit_derivatives(phase):
    """The JAX derivatives in temperature and pressure equal central differences of the numpy
    inverse, in the solid, the mush and the melt."""
    eos, eos_j = entropy_eos_copy(), entropy_eos_jax()
    p = 5.31e10
    t = float(np.asarray(eos.temperature(p, _states(eos, p)[phase])).flat[0])
    d_t, d_p = jax.grad(eos_j.entropy_at_temperature, argnums=(1, 0))(
        jnp.asarray(p), jnp.asarray(t)
    )
    h_t, h_p = 1e-6 * t, 1e-7 * p  # within one cell of the bilinear table
    fd_t = (eos.entropy_at_temperature(p, t + h_t) - eos.entropy_at_temperature(p, t - h_t)) / (
        2 * h_t
    )
    fd_p = (eos.entropy_at_temperature(p + h_p, t) - eos.entropy_at_temperature(p - h_p, t)) / (
        2 * h_p
    )
    assert float(d_t) == pytest.approx(fd_t, rel=1e-3)
    assert float(d_p) == pytest.approx(fd_p, rel=1e-3)
