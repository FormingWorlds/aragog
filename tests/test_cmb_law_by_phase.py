"""CMB boundary-layer law by phase: Ra_crit 450 while the bottom cell is liquid, eq. 16 when solid.

``ra_crit_cmb = 'phase'`` blends the two critical Rayleigh numbers in the logarithm,
``ln Ra_crit = s ln 450 + (1 - s) ln Ra_crit,16``, with the liquid weight ``s`` of the
melt fraction of the bottom cell's own state at the CMB pressure (the evaluation that gives
the interior viscosity of eq. 16) around the rheological transition. The convective flux
scales as ``Ra_crit^(-1/3)``, so the law gives ``F = F_450^s F_16^(1 - s)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
pytest.importorskip('equinox')
pytest.importorskip('hypothesis')
jax.config.update('jax_enable_x64', True)

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from aragog.core import check_ra_crit, cmb_boundary_layer_flux, liquid_weight  # noqa: E402
from aragog.core.boundary_layer import RA_CRIT_PHASE, RA_CRIT_UPPER_LAYER  # noqa: E402

pytestmark = [pytest.mark.timeout(300)]

LAYER = dict(
    conductivity=4.0,
    density=5200.0,
    heat_capacity=1200.0,
    expansivity=1.5e-5,
    viscosity=0.1,
    gravity=10.5,
    dr_half=750.0,
    shell_thickness=2.9e6,
    shell_contrast=3500.0,
    interior_viscosity=0.2,
)
T_CORE, T_M = 5400.0, 4800.0
LOC, WID = 0.4, 0.2


def _flux(ra_crit=None, **kw):
    return float(cmb_boundary_layer_flux(T_CORE, T_M, ra_crit=ra_crit, **{**LAYER, **kw}))


def _phase_flux(phi, loc=LOC, wid=WID, **kw):
    return cmb_boundary_layer_flux(
        T_CORE, T_M, ra_crit=None, melt_fraction=phi, liquid_band=(loc, wid), **{**LAYER, **kw}
    )


@pytest.mark.unit
def test_the_selector_takes_phase():
    assert check_ra_crit('phase') == RA_CRIT_PHASE == 'phase'
    for bad in ('Phase', 'eq16', '', True):
        with pytest.raises(ValueError, match='ra_crit_cmb'):
            check_ra_crit(bad)


@pytest.mark.unit
@pytest.mark.physics_invariant
@settings(max_examples=200, deadline=None)
@given(phi=st.floats(0.0, 1.0), d=st.floats(0.0, 0.5))
def test_the_liquid_weight_rises_through_the_transition(phi, d):
    """``s`` lies in [0, 1], does not fall with the melt fraction, is 1/2 at the centre of the
    transition and symmetric about it."""
    s = float(liquid_weight(phi, LOC, WID))
    assert 0.0 <= s <= 1.0
    assert float(liquid_weight(min(phi + 1e-3, 1.0), LOC, WID)) >= s
    assert float(liquid_weight(LOC, LOC, WID)) == pytest.approx(0.5, abs=1e-15)
    up, down = float(liquid_weight(LOC + d, LOC, WID)), float(liquid_weight(LOC - d, LOC, WID))
    assert up + down == pytest.approx(1.0, abs=1e-14)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize('phi', [0.0, 0.25, 0.31, 0.37, 0.4, 0.43, 0.49, 0.6, 1.0])
def test_the_flux_is_the_weighted_geometric_mean_of_the_two_laws(phi):
    """``F = F_450^s F_16^(1 - s)``: the logarithmic blend of Ra_crit under the cube root."""
    s = float(liquid_weight(phi, LOC, WID))
    f450, f16 = _flux(RA_CRIT_UPPER_LAYER), _flux(None)
    assert f16 != pytest.approx(f450, rel=1e-3)  # the two laws differ at this state
    assert float(_phase_flux(phi)) == pytest.approx(f450**s * f16 ** (1.0 - s), rel=1e-13)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_centre_of_the_transition_takes_the_geometric_mean_of_ra_crit():
    """At ``phi = loc``, ``Ra_crit = sqrt(450 Ra_crit,16)``."""
    f450, f16 = _flux(RA_CRIT_UPPER_LAYER), _flux(None)
    assert float(_phase_flux(LOC)) == pytest.approx(math.sqrt(f450 * f16), rel=1e-13)


@pytest.mark.unit
@pytest.mark.parametrize('base', [None, 450.0, 1800.0])
def test_a_band_switched_off_inside_a_trace_keeps_the_base_law_bit_equal(base):
    """A NaN band centre (the traced 'off') returns the base law unchanged, with finite
    derivatives in the mantle temperature and the melt fraction."""

    def f(t_m, phi):
        return cmb_boundary_layer_flux(
            T_CORE, t_m, ra_crit=base, melt_fraction=phi, liquid_band=(jnp.nan, WID), **LAYER
        )

    assert float(f(T_M, 0.4)) == _flux(base)
    g = jax.grad(f, argnums=(0, 1))(T_M, 0.4)
    assert all(np.isfinite(float(x)) for x in g)
    assert float(g[1]) == 0.0


@pytest.mark.unit
def test_the_band_needs_a_melt_fraction_and_the_eq16_inputs():
    with pytest.raises(ValueError, match='melt_fraction'):
        cmb_boundary_layer_flux(T_CORE, T_M, ra_crit=None, liquid_band=(LOC, WID), **LAYER)
    no_shell = {k: v for k, v in LAYER.items() if k != 'shell_thickness'}
    with pytest.raises(ValueError, match='shell_thickness'):
        cmb_boundary_layer_flux(
            T_CORE, T_M, ra_crit=None, melt_fraction=0.4, liquid_band=(LOC, WID), **no_shell
        )


@pytest.mark.unit
@pytest.mark.physics_invariant
@settings(max_examples=200, deadline=None)
@given(phi=st.floats(0.0, 1.0 - 1e-6))
def test_the_flux_is_continuous_in_the_melt_fraction(phi):
    """No jump anywhere in [0, 1]: a step of 1e-7 in phi moves the flux by less than the
    largest slope of ln F in phi (here |ln(F_450/F_16)| times the largest weight slope)
    allows, and the derivative in phi is finite."""
    h = 1e-7
    f0, f1 = float(_phase_flux(phi)), float(_phase_flux(phi + h))
    ln_ratio = abs(math.log(_flux(RA_CRIT_UPPER_LAYER) / _flux(None)))
    assert abs(math.log(f1 / f0)) <= ln_ratio * (2.0 / WID) * h * (1.0 + 1e-6) + 1e-15
    g = float(jax.grad(lambda p: _phase_flux(p))(phi))
    assert np.isfinite(g)


@pytest.mark.unit
@pytest.mark.parametrize('phi', [0.33, 0.4, 0.47])
def test_the_melt_fraction_derivative_matches_a_central_difference(phi):
    h = 1e-6
    fd = (float(_phase_flux(phi + h)) - float(_phase_flux(phi - h))) / (2.0 * h)
    g = float(jax.grad(lambda p: _phase_flux(p))(phi))
    assert g == pytest.approx(fd, rel=1e-6)
    assert g > 0.0  # liquid side: smaller Ra_crit, larger flux at this state


# Solver paths (real EOS): the NumPy CMB flux and the JAX RHS take the bottom cell's melt
# fraction at the CMB pressure and agree.
from tests.conftest import entropy_eos_copy, needs_eos  # noqa: E402


def _phase_solver():
    from tests.test_jax_dsdt_core_module import _build_numpy_solver

    solver = _build_numpy_solver(entropy_eos_copy(identity_alpha=True))
    solver._core_module_ra_crit_cmb = 'phase'
    return solver


def _bottom_state(solver, phi):
    """Extended state with the bottom cell at melt fraction ``phi`` at the CMB pressure and a
    core 600 K above it (the boundary-layer branch)."""
    eos, n = solver.entropy_eos, solver._n_stag
    P = float(solver._P_basic_flat[0])
    s_sol, s_liq = float(eos.solidus_entropy(P)), float(eos.liquidus_entropy(P))
    y = np.asarray(solver._S0, dtype=float).copy()
    y[0] = s_sol + phi * (s_liq - s_sol)
    assert float(np.asarray(eos.melt_fraction(P, y[0])).flat[0]) == pytest.approx(phi, abs=1e-9)
    y[n + 1] = eos.temperature_scalar(P, y[0]) + 600.0
    return y


@pytest.mark.smoke
@pytest.mark.physics_invariant
@needs_eos
@pytest.mark.parametrize('phi', [0.32, 0.4, 0.46])
def test_the_numpy_flux_blends_the_two_laws_at_the_bottom_cell_melt_fraction(phi):
    solver = _phase_solver()
    n = solver._n_stag
    y = _bottom_state(solver, phi)
    pm = solver.parameters.phase_mixed
    s = float(
        liquid_weight(
            phi, pm.rheological_transition_melt_fraction, pm.rheological_transition_width
        )
    )
    assert 0.0 < s < 1.0
    f = {}
    for law in ('phase', RA_CRIT_UPPER_LAYER, None):
        solver._core_module_ra_crit_cmb = law
        f[law] = solver._core_module_cmb_flux(y[n + 1], y[0], y[n - 1])
    assert f[None] != pytest.approx(f[RA_CRIT_UPPER_LAYER], rel=1e-3)
    assert f['phase'] == pytest.approx(
        f[RA_CRIT_UPPER_LAYER] ** s * f[None] ** (1.0 - s), rel=1e-12
    )


def _phase_args(solver):
    from tests.test_jax_dsdt_core_module import _build_jax_pieces

    solver._core_module_ra_crit_cmb = None
    args = _build_jax_pieces(solver)
    solver._core_module_ra_crit_cmb = 'phase'
    # ra_crit NaN (eq. 16 base), no fixed surface temperature, the phase law on.
    return args[:8] + (float('nan'), float('nan'), 1.0)


@pytest.mark.smoke
@pytest.mark.physics_invariant
@needs_eos
@pytest.mark.parametrize('phi', [0.32, 0.4, 0.46])
def test_the_jax_rhs_matches_numpy_under_the_phase_law(phi):
    from aragog.jax.solver import _dSdt_core_module_parts

    solver = _phase_solver()
    n = solver._n_stag
    y = _bottom_state(solver, phi)
    args = _phase_args(solver)
    f_np = solver._core_module_cmb_flux(y[n + 1], y[0], y[n - 1])
    f_jax = float(_dSdt_core_module_parts(0.0, jnp.asarray(y), args).heat_flux[0])
    assert f_jax == pytest.approx(f_np, rel=1e-10)
    off = args[:10] + (0.0,)
    f_eq16 = float(_dSdt_core_module_parts(0.0, jnp.asarray(y), off).heat_flux[0])
    assert f_eq16 != pytest.approx(f_jax, rel=1e-3)
    rate_np = np.asarray(solver.dSdt(0.0, y)).ravel()
    rate_jax = np.asarray(_dSdt_core_module_parts(0.0, jnp.asarray(y), args).rate).ravel()
    np.testing.assert_allclose(rate_jax[n:], rate_np[n:], rtol=1e-8)


@pytest.mark.smoke
@needs_eos
def test_the_jacobian_is_finite_inside_the_transition():
    from aragog.jax.solver import dSdt_core_module

    solver = _phase_solver()
    y = _bottom_state(solver, 0.4)
    args = _phase_args(solver)
    jac = np.asarray(jax.jacrev(lambda v: dSdt_core_module(0.0, v, args))(jnp.asarray(y)))
    assert np.all(np.isfinite(jac))
    n = solver._n_stag
    h = 1e-4 * max(abs(y[0]), 1.0)
    yp, ym = y.copy(), y.copy()
    yp[0] += h
    ym[0] -= h
    fd = (
        np.asarray(dSdt_core_module(0.0, jnp.asarray(yp), args))
        - np.asarray(dSdt_core_module(0.0, jnp.asarray(ym), args))
    ) / (2.0 * h)
    assert jac[n + 1, 0] == pytest.approx(fd[n + 1], rel=1e-4)


@pytest.mark.unit
def test_the_jax_factory_takes_the_phase_law():
    """build_jax_rhs_and_jacobian accepts core_module_ra_crit_cmb='phase'."""
    from aragog.jax.nondim import NonDimScales
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
    from tests.test_jax_dsdt_core_module import _tiny_budget

    n = 4
    rhs_fn, jac_fn, _ = build_jax_rhs_and_jacobian(
        eos_jax=None,
        phase_params=None,
        mesh_arrays=None,
        boundary_params=None,
        heating_array=np.zeros(n),
        scales=NonDimScales(state_scale=np.full(n + 2, 1.0), t_ref=1.0),
        core_bc_mode='core_module',
        core_module_budget=_tiny_budget(),
        core_module_ra_crit_cmb='phase',
    )
    assert callable(rhs_fn) and callable(jac_fn)
