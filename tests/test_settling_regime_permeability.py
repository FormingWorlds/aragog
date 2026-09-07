"""Behavioural tests for the gravitational-separation relative velocity.

``relative_velocity`` computes the melt-solid drag velocity as
``abs_drho * gravity * F / eta_l``, where ``F`` is the
permeability-over-porosity mobility from the Abe (1993/1995) three-regime
model (SPIDER convention, Bower et al. 2018 Eqs. 13a-c): Blake-Kozeny-Carman
at low porosity, Rumpf-Gupte at intermediate porosity, Stokes settling at
high porosity, blended with two ``tanh`` weights at the equal-mobility
crossings 0.0769452 and 0.771462.

These tests pin the behaviour that the source-text constant pins and the
viscosity-scaling tests do not cover:

1. ``v_rel >= 0`` across the whole porosity range, with a melt-lighter
   density contrast (``delta_rho < 0``). A dropped ``abs`` on the density
   contrast, or a dropped ``max(F, 0)``, would produce a negative velocity.
2. ``v_rel`` scales linearly with the magnitude of the density contrast at
   fixed porosity. A regression that froze the contrast to a constant would
   leave the ratio at one instead of tracking the contrast ratio.
3. The correct regime is selected: deep inside each regime the velocity
   matches that regime's closed form and differs by a wide margin from the
   other two closed forms.
4. The numpy and JAX paths agree to 1e-10 across the porosity range.

Both paths take the phase-boundary densities from an EOS lookup and derive
porosity from the local density, so a stub exposing only
``_lookup_at_phase_boundary`` drives either path at a chosen porosity.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# Regime blend crossings and grain size shared by every case here.
_D = 1.0e-3
_G = 10.0
_ETA = 1.0e-1  # melt-mode drag viscosity, fixed in both paths
_RHO_S = 4200.0
_RHO_L = 3900.0
_ABS_DRHO = abs(_RHO_L - _RHO_S)  # 300 kg/m^3


def _density_for_porosity(porosity, rho_s=_RHO_S, rho_l=_RHO_L):
    """Local density that maps to ``porosity`` under the source mapping.

    The source computes ``porosity = (rho_s - rho) / (rho_s - rho_l)`` before
    a soft clip whose effect is below 1e-3 anywhere in the open interval, so
    this inverse places a test point to well within the tolerances used here.
    """
    return rho_s - porosity * (rho_s - rho_l)


def _F_bkc(porosity):
    return _D**2 * porosity**2 / ((1.0 - porosity) ** 2 * 1000.0)


def _F_rg(porosity):
    return _D**2 * porosity**4.5 * (5.0 / 7.0)


def _F_stokes():
    return _D**2 * 2.0 / 9.0


def _v_from_F(F, abs_drho=_ABS_DRHO, eta=_ETA):
    return abs_drho * _G * F / eta


# Deep-regime porosities: each is several blend widths from both crossings,
# so the single-regime closed form is the blend to better than a percent.
_REGIME_CASES = [
    ('bkc', 0.02, _F_bkc),
    ('rg', 0.40, _F_rg),
    ('stokes', 0.99, lambda p: _F_stokes()),
]


class _StubPhaseBoundaryEOS:
    """Numpy stub exposing only the phase-boundary density lookup."""

    def __init__(self, rho_s=_RHO_S, rho_l=_RHO_L):
        self._rho_s = rho_s
        self._rho_l = rho_l

    def _lookup_at_phase_boundary(self, prop, P, phase):
        assert prop == 'density'
        arr = np.asarray(P, dtype=float)
        if phase == 'solid':
            return np.full_like(arr, self._rho_s)
        elif phase == 'melt':
            return np.full_like(arr, self._rho_l)
        raise ValueError(phase)


def _numpy_evaluator(rho_s=_RHO_S, rho_l=_RHO_L):
    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    return EntropyPhaseEvaluator(
        entropy_eos=_StubPhaseBoundaryEOS(rho_s, rho_l),
        gravitational_acceleration=_G,
        grain_size=_D,
        viscosity_liquid=_ETA,
        separation_viscosity='melt',
    )


def _numpy_v_rel(ev, density):
    ev.pressure = np.array([1.0e9])
    ev._density = np.array([float(density)])
    return float(np.asarray(ev.relative_velocity()).item())


# --- numpy path -----------------------------------------------------------


def test_relative_velocity_nonnegative_across_porosity_numpy():
    """Melt-lighter contrast (delta_rho < 0) still yields v_rel >= 0.

    A regression dropping the ``abs`` on the density contrast would make
    every value here negative; dropping ``max(F, 0)`` would make the edge
    values negative.
    """
    ev = _numpy_evaluator()
    porosities = np.linspace(0.0, 1.0, 41)
    values = [_numpy_v_rel(ev, _density_for_porosity(p)) for p in porosities]
    assert all(v >= 0.0 for v in values)
    # Interior of the range is strictly positive, not identically zero.
    assert values[20] > 0.0


@pytest.mark.parametrize('name,porosity,F_expected', _REGIME_CASES)
def test_relative_velocity_selects_correct_regime_numpy(name, porosity, F_expected):
    """Deep in each regime, v_rel matches that regime's closed form and is
    far from the other two."""
    ev = _numpy_evaluator()
    v = _numpy_v_rel(ev, _density_for_porosity(porosity))

    v_here = _v_from_F(F_expected(porosity))
    others = {
        'bkc': _v_from_F(_F_bkc(porosity)),
        'rg': _v_from_F(_F_rg(porosity)),
        'stokes': _v_from_F(_F_stokes()),
    }
    assert v == pytest.approx(v_here, rel=0.02)
    for other_name, other_v in others.items():
        if other_name == name:
            continue
        assert abs(v - other_v) / v_here > 0.2


def test_relative_velocity_scales_linearly_with_density_contrast_numpy():
    """At fixed porosity, doubling |delta_rho| doubles v_rel.

    Both configurations sit at porosity 0.4; the second widens the
    solid-melt density gap symmetrically from 300 to 600 kg/m^3, so the
    porosity and hence F are unchanged and only the contrast magnitude
    differs.
    """
    porosity = 0.4
    ev_a = _numpy_evaluator(rho_s=4200.0, rho_l=3900.0)
    ev_b = _numpy_evaluator(rho_s=4350.0, rho_l=3750.0)
    v_a = _numpy_v_rel(ev_a, _density_for_porosity(porosity, 4200.0, 3900.0))
    v_b = _numpy_v_rel(ev_b, _density_for_porosity(porosity, 4350.0, 3750.0))
    assert v_b / v_a == pytest.approx(600.0 / 300.0, rel=1e-6)


# --- JAX path -------------------------------------------------------------

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
jax.config.update('jax_enable_x64', True)

from aragog.jax.phase import PhaseParams  # noqa: E402
from aragog.jax.phase import relative_velocity as jax_relative_velocity  # noqa: E402


class _StubEOS_JAX:
    """JAX stub exposing only the phase-boundary density lookup."""

    def __init__(self, rho_s=_RHO_S, rho_l=_RHO_L):
        self._rho_s = rho_s
        self._rho_l = rho_l

    def _lookup_at_phase_boundary(self, field, pressure, phase):
        assert field == 'density'
        if phase == 'solid':
            return jnp.asarray(self._rho_s)
        elif phase == 'melt':
            return jnp.asarray(self._rho_l)
        raise ValueError(phase)


def _jax_v_rel(density, rho_s=_RHO_S, rho_l=_RHO_L):
    eos = _StubEOS_JAX(rho_s, rho_l)
    params = PhaseParams(grain_size=_D, viscosity_liquid=_ETA, separation_viscosity='melt')
    out = jax_relative_velocity(
        eos,
        params,
        jnp.asarray(1.0e9),
        jnp.asarray(float(density)),
        jnp.asarray(0.5),  # melt_fraction, unused by the porosity-from-density path
        jnp.asarray(_G),
        viscosity=jnp.asarray(1.0e30),  # unused in melt mode
    )
    return float(out)


def test_relative_velocity_nonnegative_across_porosity_jax():
    """JAX path: melt-lighter contrast still yields v_rel >= 0."""
    porosities = np.linspace(0.0, 1.0, 41)
    values = [_jax_v_rel(_density_for_porosity(p)) for p in porosities]
    assert all(v >= 0.0 for v in values)
    assert values[20] > 0.0


@pytest.mark.parametrize('name,porosity,F_expected', _REGIME_CASES)
def test_relative_velocity_selects_correct_regime_jax(name, porosity, F_expected):
    """JAX path: correct regime selected deep in each regime."""
    v = _jax_v_rel(_density_for_porosity(porosity))
    v_here = _v_from_F(F_expected(porosity))
    others = {
        'bkc': _v_from_F(_F_bkc(porosity)),
        'rg': _v_from_F(_F_rg(porosity)),
        'stokes': _v_from_F(_F_stokes()),
    }
    assert v == pytest.approx(v_here, rel=0.02)
    for other_name, other_v in others.items():
        if other_name == name:
            continue
        assert abs(v - other_v) / v_here > 0.2


def test_relative_velocity_scales_linearly_with_density_contrast_jax():
    """JAX path: doubling |delta_rho| at fixed porosity doubles v_rel."""
    porosity = 0.4
    v_a = _jax_v_rel(_density_for_porosity(porosity, 4200.0, 3900.0), 4200.0, 3900.0)
    v_b = _jax_v_rel(_density_for_porosity(porosity, 4350.0, 3750.0), 4350.0, 3750.0)
    assert v_b / v_a == pytest.approx(600.0 / 300.0, rel=1e-6)


# --- cross-path parity ----------------------------------------------------


def test_relative_velocity_numpy_jax_parity_across_porosity():
    """Numpy and JAX paths agree to 1e-10 across the porosity range.

    Both run in melt mode with identical grain size, gravity, drag
    viscosity and boundary densities, so the two independent
    implementations of the porosity soft clip and the three-regime blend
    must produce the same velocity at every porosity. The relative bound
    governs the porosities where settling is active; the absolute floor of
    1e-15 covers the pure-solid endpoint, where the velocity is below
    1e-11 m/s and the soft-clip rounding residual (below 1e-17 in absolute
    terms across the whole sweep) would otherwise inflate the relative
    metric against a value that is numerically zero.
    """
    ev = _numpy_evaluator()
    porosities = np.linspace(0.0, 1.0, 41)
    for p in porosities:
        density = _density_for_porosity(p)
        v_np = _numpy_v_rel(ev, density)
        v_jx = _jax_v_rel(density)
        assert v_jx == pytest.approx(v_np, rel=1e-10, abs=1e-15)
