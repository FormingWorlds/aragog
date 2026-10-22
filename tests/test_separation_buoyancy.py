"""The buoyancy form of the gravitational separation velocity, ``separation_buoyancy``.

``'mixture'`` (default) takes the buoyancy against the mixture, McKenzie (1984,
Eq. 3.7): the Blake-Kozeny-Carman and Rumpf-Gupte terms of the mobility carry a
factor ``1 - phi``. ``'solid'`` takes it against the solid, Abe (1995, Eq. 40),
as SPIDER does. The Stokes term has no factor in either form.
"""

from __future__ import annotations

import importlib.resources

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from aragog.eos.entropy_phase import EntropyPhaseEvaluator, mobility_function

pytestmark = pytest.mark.unit

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
jax.config.update('jax_enable_x64', True)

from aragog.jax.phase import PhaseParams  # noqa: E402
from aragog.jax.phase import relative_velocity as jax_relative_velocity  # noqa: E402

_D, _G, _ETA, _RHO_S, _RHO_L = 1.0e-3, 10.0, 1.0e-1, 4200.0, 3900.0


def _terms(p):
    """Blend weights and the three regime mobilities at porosity ``p``."""
    w_rg = 0.5 * (1.0 + np.tanh((p - 0.0769452) / 0.02))
    w_st = 0.5 * (1.0 + np.tanh((p - 0.771462) / 0.05))
    bkc = _D**2 * p**2 / ((1.0 - p) ** 2 * 1000.0)
    rg = _D**2 * p**4.5 * (5.0 / 7.0)
    return w_rg, w_st, bkc, rg, _D**2 * 2.0 / 9.0


def _F(p, buoyancy):
    w_rg, w_st, bkc, rg, stokes = _terms(p)
    matrix = (1.0 - w_rg) * bkc + (w_rg - w_st) * rg
    return (1.0 - p if buoyancy == 'mixture' else 1.0) * matrix + w_st * stokes


class _StubEOS:
    """Constant phase-boundary densities for the numpy and the JAX path."""

    def __init__(self, rho_s=_RHO_S, rho_l=_RHO_L):
        self._rho = {'solid': rho_s, 'melt': rho_l}

    def _lookup_at_phase_boundary(self, prop, P, phase):
        return np.full(np.shape(P), self._rho[phase])


def _v(path, porosity, rho_s=_RHO_S, rho_l=_RHO_L, **kw):
    density = rho_s - porosity * (rho_s - rho_l)
    eos = _StubEOS(rho_s, rho_l)
    if path == 'numpy':
        ev = EntropyPhaseEvaluator(
            entropy_eos=eos, gravitational_acceleration=_G, grain_size=_D,
            viscosity_liquid=_ETA, separation_viscosity='melt', **kw,
        )  # fmt: skip
        ev.pressure = np.array([1.0e9])
        ev._density = np.array([density])
        return float(ev.relative_velocity().item())
    params = PhaseParams(
        grain_size=_D, viscosity_liquid=_ETA, separation_viscosity='melt', **kw
    )
    return float(
        jax_relative_velocity(
            eos, params, jnp.asarray(1.0e9), jnp.asarray(density),
            jnp.asarray(0.5), jnp.asarray(_G), viscosity=jnp.asarray(1.0e30),
        )
    )  # fmt: skip


@pytest.mark.parametrize('p', [0.02, 0.0769452, 0.1, 0.4, 0.771462, 0.9, 0.99])
def test_mobility_forms_at_each_regime_and_crossing(p):
    """'mixture' puts 1 - p on the BKC and RG terms only; 'solid' is the plain blend and
    the default of mobility_function."""
    np.testing.assert_allclose(
        mobility_function(p, _D, buoyancy='mixture'), _F(p, 'mixture'), rtol=1e-13
    )
    np.testing.assert_allclose(
        mobility_function(p, _D, buoyancy='solid'), _F(p, 'solid'), rtol=1e-13
    )
    assert mobility_function(p, _D) == mobility_function(p, _D, buoyancy='solid')


@pytest.mark.parametrize('bad', ['Mixture', '', 'melt', None, True, 1])
def test_mobility_rejects_an_unknown_buoyancy(bad):
    with pytest.raises(ValueError, match='buoyancy'):
        mobility_function(0.4, _D, buoyancy=bad)


@settings(max_examples=200, deadline=None)
@given(st.floats(0.0, 1.0))
@example(0.0)
@example(1.0)
@example(0.0769452)
@example(0.771462)
def test_mixture_mobility_lies_between_the_solid_form_and_its_matrix_reduction(p):
    """(1 - p) F_solid <= F_mixture <= F_solid on [0, 1]: the factor only reduces the
    matrix terms, and the Stokes term keeps its full weight."""
    f_solid = float(mobility_function(p, _D, buoyancy='solid'))
    f_mix = float(mobility_function(p, _D, buoyancy='mixture'))
    assert (1.0 - p) * f_solid * (1 - 1e-12) <= f_mix <= f_solid * (1 + 1e-12)


@pytest.mark.filterwarnings('ignore:invalid value:RuntimeWarning')
@pytest.mark.parametrize('mode', ['mixture', 'solid'])
@pytest.mark.parametrize('p', [float('nan'), float('inf')])
def test_mobility_of_a_nan_or_inf_porosity_is_nan(mode, p):
    """A NaN or +inf porosity never becomes a finite mobility in either form."""
    assert np.isnan(mobility_function(p, _D, buoyancy=mode))


@settings(max_examples=60, deadline=None)
@given(st.floats(-1000.0, 1000.0))
@example(0.0)
@example(1.0e-6)
@example(-1.0e-6)
@example(1.0e-300)
@example(-1.0e-300)
def test_relative_velocity_has_the_sign_of_the_contrast(contrast):
    """For rho_s - rho_l over [-1000, 1000] kg/m^3 in a mush halfway between the phase
    densities, both paths and both forms give v with the sign of the realised contrast
    (strictly for |c| >= 1e-6), 0 at 0, and |v_mixture| <= |v_solid|."""
    rho_l = 3900.0
    rho_s = rho_l + contrast
    c = rho_s - rho_l  # a contrast below ~1e-13 rounds to 0 at these densities
    for path in ('numpy', 'jax'):
        v = {
            m: _v(path, 0.5, rho_s=rho_s, rho_l=rho_l, separation_buoyancy=m)
            for m in ('mixture', 'solid')
        }
        for m in v:
            assert np.isfinite(v[m]) and v[m] * c >= 0.0
            assert abs(c) < 1.0e-6 or v[m] * c > 0.0
            assert c != 0.0 or v[m] == 0.0
        assert abs(v['mixture']) <= abs(v['solid'])


@pytest.mark.parametrize('path', ['numpy', 'jax'])
@pytest.mark.parametrize('mode', ['mixture', 'solid'])
def test_denser_melt_gives_a_negative_velocity_in_both_forms(path, mode):
    """With the melt denser than the solid by 300 kg/m^3 the velocity is negative and
    about 0 in both forms and both paths: the density denominator floors the porosity,
    so |v| is below 1e-9 of the velocity of a melt lighter by 300 kg/m^3."""
    v = _v(path, 0.5, rho_s=3900.0, rho_l=4200.0, separation_buoyancy=mode)
    v_lighter = _v(path, 0.5, rho_s=4200.0, rho_l=3900.0, separation_buoyancy=mode)
    assert v < 0.0 and abs(v) < 1.0e-9 * v_lighter


def test_jax_buoyancy_flag_is_a_threshold():
    """The JAX form follows the flag as a switch: a flag of 2.0 or 0.6 gives the
    'mixture' velocity and 0.4 the 'solid' one, never a scaled factor."""
    import equinox as eqx

    base = PhaseParams(grain_size=_D, viscosity_liquid=_ETA, separation_viscosity='melt')
    eos = _StubEOS()
    density = _RHO_S - 0.5 * (_RHO_S - _RHO_L)

    def v(flag):
        params = eqx.tree_at(lambda q: q.separation_buoyancy_mixture, base, flag)
        return float(
            jax_relative_velocity(
                eos, params, jnp.asarray(1.0e9), jnp.asarray(density), jnp.asarray(0.5),
                jnp.asarray(_G), viscosity=jnp.asarray(1.0e30),
            )
        )  # fmt: skip

    assert v(2.0) == v(1.0) and v(0.6) == v(1.0)
    assert v(0.4) == v(0.0) and v(0.0) != v(1.0)


@pytest.mark.parametrize('path', ['numpy', 'jax'])
@pytest.mark.parametrize('p', [0.02, 0.4, 0.99])
def test_relative_velocity_default_is_the_mixture_form(path, p):
    """Without the option both paths use 'mixture'; with 'solid' they give the plain blend."""
    scale = (_RHO_S - _RHO_L) * _G / _ETA
    assert _v(path, p) == pytest.approx(scale * _F(p, 'mixture'), rel=2e-3)
    assert _v(path, p, separation_buoyancy='solid') == pytest.approx(
        scale * _F(p, 'solid'), rel=2e-3
    )


@pytest.mark.parametrize('mode', ['mixture', 'solid'])
def test_relative_velocity_numpy_jax_parity_in_both_buoyancy_forms(mode):
    for p in np.linspace(0.0, 1.0, 21):
        v_np = _v('numpy', p, separation_buoyancy=mode)
        assert _v('jax', p, separation_buoyancy=mode) == pytest.approx(
            v_np, rel=1e-10, abs=1e-15
        )


@pytest.mark.parametrize('bad', ['Mixture', '', 'melt', None, True, 1])
def test_every_entry_point_rejects_an_unknown_buoyancy(bad):
    """The evaluator, the JAX parameters, the attrs config and the parser dataclass."""
    from aragog.config.phases import MixedPhaseConfig
    from aragog.parser import _PhaseMixedParameters

    common = dict(latent_heat_of_fusion=4e5, rheological_transition_melt_fraction=0.4,
                  rheological_transition_width=0.15, solidus='s.dat', liquidus='l.dat',
                  phase='mixed', phase_transition_width=0.01, grain_size=1e-3)  # fmt: skip
    with pytest.raises(ValueError, match='separation_buoyancy'):
        EntropyPhaseEvaluator(_StubEOS(), _G, separation_buoyancy=bad)
    with pytest.raises(ValueError, match='separation_buoyancy'):
        PhaseParams(grain_size=_D, viscosity_liquid=_ETA, separation_buoyancy=bad)
    with pytest.raises(ValueError, match='separation_buoyancy'):
        MixedPhaseConfig(**common, separation_buoyancy=bad)
    with pytest.raises(ValueError, match='separation_buoyancy'):
        _PhaseMixedParameters(**common, separation_buoyancy=bad)


@pytest.mark.parametrize(
    'separation_buoyancy,expected', [(None, 'mixture'), ('solid', 'solid')]
)
def test_entropy_solver_plumbs_separation_buoyancy_to_evaluators(separation_buoyancy, expected):
    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_separation_viscosity_config import _build_solver_params

    parameters = _build_solver_params(None)
    if separation_buoyancy is not None:
        parameters.phase_mixed.separation_buoyancy = separation_buoyancy
    solver = EntropySolver(parameters, entropy_eos=None)
    solver.initialize()
    assert solver.state.phase_staggered._separation_buoyancy == expected
    assert solver.state.phase_basic._separation_buoyancy == expected
    assert solver._bl_phase._separation_buoyancy == expected


def test_abe_benchmarks_with_separation_pin_the_solid_buoyancy():
    """Abe (1995) takes the buoyancy against the solid, so every bundled Abe config that
    runs gravitational separation sets 'solid' and its reference output stays valid."""
    from aragog.parser import Parameters

    cfgs = sorted(importlib.resources.files('aragog').joinpath('cfg').glob('abe_*.cfg'))
    with_sep = [c for c in cfgs if Parameters.from_file(str(c)).energy.gravitational_separation]
    assert [c.name for c in with_sep] == ['abe_mixed.cfg']
    for c in with_sep:
        assert Parameters.from_file(str(c)).phase_mixed.separation_buoyancy == 'solid'
