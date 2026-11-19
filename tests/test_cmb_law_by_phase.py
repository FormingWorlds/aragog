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
from aragog.core.boundary_layer import (  # noqa: E402
    RA_CRIT_PHASE,
    RA_CRIT_UPPER_LAYER,
    _concrete_scalar,
)

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


def _weight(phi, loc, wid):
    """Expected liquid weight: a tanh at ``loc`` that is 0.05 and 0.95 at ``loc -+ wid/2``."""
    return 0.5 * (1.0 + np.tanh((phi - loc) / (0.5 * wid / math.atanh(0.9))))


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
@pytest.mark.parametrize(
    ('ra_crit', 'loc'),
    [
        (450.0, LOC),
        (np.float64(1800.0), LOC),
        (jnp.asarray(450.0), LOC),
        (450.0, jnp.asarray(LOC)),
    ],
)
def test_a_fixed_ra_crit_and_a_band_are_refused_together(ra_crit, loc):
    """The law by phase replaces a fixed number, so a call that gives both is refused, for a
    Python, NumPy or JAX scalar outside a trace."""
    with pytest.raises(ValueError, match='exclude each other'):
        cmb_boundary_layer_flux(
            T_CORE, T_M, ra_crit=ra_crit, melt_fraction=0.4, liquid_band=(loc, WID), **LAYER
        )


@pytest.mark.unit
def test_a_saturated_weight_gives_each_law_exactly():
    """In a band of width 1e-3 the tanh saturates in float64 at phi = 0 and 1: the flux is
    then the eq. 16 flux and the 450 flux, bit for bit."""
    assert _weight(0.0, LOC, 1e-3) == 0.0 and _weight(1.0, LOC, 1e-3) == 1.0
    assert float(_phase_flux(0.0, wid=1e-3)) == _flux(None)
    assert float(_phase_flux(1.0, wid=1e-3)) == _flux(450.0)


@pytest.mark.unit
def test_a_nan_ra_crit_with_a_band_is_the_law_by_phase():
    """NaN is the eq. 16 sentinel of the traced path: with a band it gives the law by phase."""
    kw = dict(melt_fraction=0.4, liquid_band=(LOC, WID), **LAYER)
    got = cmb_boundary_layer_flux(T_CORE, T_M, ra_crit=float('nan'), **kw)
    assert float(got) == float(cmb_boundary_layer_flux(T_CORE, T_M, ra_crit=None, **kw))


@pytest.mark.unit
@pytest.mark.parametrize(
    ('ra_crit', 'loc'), [(None, LOC), (float('nan'), LOC), (450.0, None), (450.0, float('nan'))]
)
def test_a_fixed_number_or_a_band_alone_is_accepted(ra_crit, loc):
    """A call with a fixed ra_crit and no live band, or a live band and no fixed number, runs."""
    band = None if loc is None else (loc, WID)
    try:
        got = cmb_boundary_layer_flux(
            T_CORE, T_M, ra_crit=ra_crit, melt_fraction=0.4, liquid_band=band, **LAYER
        )
    except (TypeError, ValueError) as exc:
        pytest.fail(f'refused: {exc!r}')
    assert math.isfinite(float(got))


@pytest.mark.unit
@pytest.mark.parametrize(
    ('x', 'want'),
    [
        (None, None),
        (True, None),
        ('450', None),
        (np.array([1.0, 2.0]), None),
        (object(), None),
        (450, 450.0),
        (np.float64(3.0), 3.0),
        (jnp.asarray(3.0), 3.0),
    ],
)
def test_only_a_real_scalar_of_rank_0_is_a_fixed_number(x, want):
    """A bool, a string, an array of rank 1 and any other object are not a fixed number."""
    got = _concrete_scalar(x)
    if want is None:
        assert got is None
    else:
        assert type(got) is float and got == want


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


def _band_point(solver, f):
    """Melt fraction at the fraction ``f`` across the solver's transition band."""
    pm = solver.parameters.phase_mixed
    loc, wid = pm.rheological_transition_melt_fraction, pm.rheological_transition_width
    return loc - 0.5 * wid + f * wid


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
@pytest.mark.parametrize('f', [0.1, 0.5, 0.85])
def test_the_numpy_flux_blends_the_two_laws_at_the_bottom_cell_melt_fraction(f):
    solver = _phase_solver()
    n, phi = solver._n_stag, _band_point(solver, f)
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
@pytest.mark.parametrize('f', [0.1, 0.5, 0.85])
def test_the_jax_rhs_matches_numpy_under_the_phase_law(f):
    from aragog.jax.solver import _dSdt_core_module_parts

    solver = _phase_solver()
    n, phi = solver._n_stag, _band_point(solver, f)
    y = _bottom_state(solver, phi)
    args = _phase_args(solver)
    f_np = solver._core_module_cmb_flux(y[n + 1], y[0], y[n - 1])
    f_jax = float(_dSdt_core_module_parts(0.0, jnp.asarray(y), args).heat_flux[0])
    assert f_jax == pytest.approx(f_np, rel=1e-10)
    off = args[:10] + (0.0,)
    f_eq16 = float(_dSdt_core_module_parts(0.0, jnp.asarray(y), off).heat_flux[0])
    solver._core_module_ra_crit_cmb = None
    assert f_eq16 == pytest.approx(
        solver._core_module_cmb_flux(y[n + 1], y[0], y[n - 1]), rel=1e-10
    )
    solver._core_module_ra_crit_cmb = 'phase'
    assert abs(f_eq16 / f_jax - 1.0) > 1e-5
    rate_np = np.asarray(solver.dSdt(0.0, y)).ravel()
    rate_jax = np.asarray(_dSdt_core_module_parts(0.0, jnp.asarray(y), args).rate).ravel()
    np.testing.assert_allclose(rate_jax[n:], rate_np[n:], rtol=1e-8)


@pytest.mark.smoke
@needs_eos
def test_the_jacobian_is_finite_inside_the_transition():
    from aragog.jax.solver import dSdt_core_module

    solver = _phase_solver()
    y = _bottom_state(solver, _band_point(solver, 0.4))
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


# The weight is a tanh that is 0.05 and 0.95 at the band edges; the band follows the
# configuration. It is narrower than the weight of the viscosity blend, which is 0.27 and 0.73
# at the same edges.


@pytest.mark.unit
@pytest.mark.physics_invariant
@settings(max_examples=300, deadline=None)
@given(loc=st.floats(0.1, 0.9), wid=st.floats(0.01, 0.2))
def test_the_weight_is_5_and_95_percent_at_the_band_edges(loc, wid):
    lo, hi = loc - 0.5 * wid, loc + 0.5 * wid
    assert float(liquid_weight(lo, loc, wid)) == pytest.approx(0.05, rel=1e-14)
    assert float(liquid_weight(hi, loc, wid)) == pytest.approx(0.95, rel=1e-14)
    assert float(liquid_weight(loc, loc, wid)) == pytest.approx(0.5, rel=1e-15)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize(('loc', 'wid'), [(0.4, 0.2), (0.5, 0.1), (0.4, 0.15)])
def test_the_tails_follow_the_formula_and_the_weight_rises(loc, wid):
    """At phi0 = 0 and 1 the weight is (1 + tanh((phi0 - loc)/w))/2 with
    w = (wid/2)/atanh(0.9); it rises with phi0."""
    for phi in (0.0, 1.0):
        expected = 0.5 * (1.0 + math.tanh((phi - loc) / (0.5 * wid / math.atanh(0.9))))
        assert float(liquid_weight(phi, loc, wid)) == pytest.approx(expected, rel=1e-14)
    grid = np.linspace(0.0, 1.0, 2001)
    w = np.asarray(liquid_weight(jnp.asarray(grid), loc, wid))
    assert np.all(np.diff(w) > 0.0)
    f = np.array([float(_phase_flux(p, loc, wid)) for p in grid[::50]])
    f450, f16 = _flux(RA_CRIT_UPPER_LAYER), _flux(None)
    # Between the two laws everywhere, and towards the liquid law as phi0 rises.
    assert np.all((f > min(f450, f16)) & (f < max(f450, f16)))
    assert np.all(np.sign(np.diff(f)) == np.sign(f450 - f16))


@pytest.mark.unit
@pytest.mark.parametrize(('loc', 'wid'), [(0.05, 0.2), (0.95, 0.2), (0.3, 0.8), (0.4, 0.0)])
def test_a_band_that_leaves_zero_to_one_is_refused(loc, wid):
    from aragog.core import check_liquid_band

    with pytest.raises(ValueError, match='band'):
        check_liquid_band(loc, wid)


@pytest.mark.unit
@pytest.mark.parametrize(('loc', 'wid'), [(0.4, 0.2), (0.5, 0.1), (0.1, 0.2), (0.9, 0.2)])
def test_a_band_inside_zero_to_one_is_accepted(loc, wid):
    from aragog.core import check_liquid_band

    check_liquid_band(loc, wid)


def _set_band(solver, args, loc, wid):
    """The solver and its JAX arguments with the rheological transition at ``loc``, ``wid``."""
    import equinox as eqx

    pm = solver.parameters.phase_mixed
    object.__setattr__(pm, 'rheological_transition_melt_fraction', loc)
    object.__setattr__(pm, 'rheological_transition_width', wid)
    # The NumPy phase evaluators hold their own copy for the viscosity blend.
    for ev in (solver._bl_phase, solver.state.phase_staggered, solver.state.phase_basic):
        ev._phi_rheo, ev._phi_width = loc, wid
    params = eqx.tree_at(lambda p: (p.phi_rheo, p.phi_width), args[1], (loc, wid))
    return args[:1] + (params,) + args[2:]


@pytest.mark.smoke
@pytest.mark.physics_invariant
@needs_eos
@pytest.mark.parametrize('phi', [0.45, 0.5, 0.55])
def test_the_band_follows_the_configured_transition(phi):
    """With the transition at 0.5 and width 0.1 both paths weight the laws at the new band."""
    from aragog.jax.solver import _dSdt_core_module_parts

    solver = _phase_solver()
    n = solver._n_stag
    y = _bottom_state(solver, phi)
    args = _set_band(solver, _phase_args(solver), 0.5, 0.1)
    f = {}
    for law in (RA_CRIT_UPPER_LAYER, None, 'phase'):
        solver._core_module_ra_crit_cmb = law
        f[law] = solver._core_module_cmb_flux(y[n + 1], y[0], y[n - 1])
    s = float(_weight(phi, 0.5, 0.1))
    assert s != pytest.approx(float(_weight(phi, 0.4, 0.15)), rel=1e-2)
    assert f['phase'] == pytest.approx(
        f[RA_CRIT_UPPER_LAYER] ** s * f[None] ** (1.0 - s), rel=1e-12
    )
    f_jax = float(_dSdt_core_module_parts(0.0, jnp.asarray(y), args).heat_flux[0])
    assert f_jax == pytest.approx(f['phase'], rel=1e-10)


@pytest.mark.smoke
@needs_eos
def test_the_solver_refuses_the_law_by_phase_with_a_band_outside_zero_to_one():
    from tests.test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS, _build

    solver = _build(
        'core_module',
        entropy_eos_copy(),
        {**CORE_MODULE_PARAMS, 'ra_crit_cmb': 'phase'},
        s_init='driven',
    )
    assert solver._core_module_ra_crit_cmb == 'phase'
    pm = solver.parameters.phase_mixed
    object.__setattr__(pm, 'rheological_transition_melt_fraction', 0.95)
    object.__setattr__(pm, 'rheological_transition_width', 0.2)
    with pytest.raises(ValueError, match='band'):
        solver._cache_bc_constants()


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_flux_by_phase_is_the_weighted_mean_of_the_two_laws(validation_plot):
    """Across the melt fraction of the bottom cell the flux under 'phase' equals
    F_450^s F_16^(1 - s), with s the tanh weight that is 0.05 and 0.95 at the band edges, to rel
    1e-12, and lies between the two laws."""
    grid = np.linspace(0.0, 1.0, 41)
    f450, f16 = _flux(RA_CRIT_UPPER_LAYER), _flux(None)
    q = np.array([float(_phase_flux(p)) for p in grid])
    s = _weight(grid, LOC, WID)
    ref = f450**s * f16 ** (1.0 - s)
    rel = np.abs(q / ref - 1.0)
    assert rel.max() < 1e-12
    assert np.all((q > f16) & (q < f450))

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.axhline(f450 / f16, ls='--', color='C1', label='450')
        ax.axhline(1.0, ls=':', color='C2', label='eq. 16')
        ax.plot(grid, ref / f16, color='C0', lw=3, alpha=0.4, label='expected')
        ax.plot(grid, q / f16, '.', ms=3, color='C0', label="'phase'")
        ax.set(xlabel='Melt fraction of the bottom cell', ylabel='Flux / eq. 16 flux')
        ax.legend(fontsize=8, loc='lower right')
        # the round-off difference is text: its value depends on the platform
        ax.text(
            0.03,
            0.55,
            f'max |code / expected - 1|\n= {rel.max():.1e} (tolerance 1e-12)',
            transform=ax.transAxes,
            fontsize=8,
        )
        fig.tight_layout()
        return fig

    validation_plot(figure, 'core_cmb_law_by_phase')


@pytest.mark.smoke
@pytest.mark.physics_invariant
@needs_eos
@pytest.mark.parametrize('law', ['phase', 450.0, None])
def test_a_restore_inside_the_transition_continues_the_solve(law):
    """A solve stopped with the bottom cell inside the transition and restored into a new
    solver continues bit-equal to a second restore, so the law keeps no state of its own. It
    agrees with the same solver continued without a restore to 1e-11 K: that solver differs
    from a fresh one at round-off under every law (measured 4.5e-13 to 1.8e-12 K)."""
    from tests.test_entropy_solver_core_module_smoke import (
        CORE_MODULE_PARAMS,
        _build,
        needs_cvode,
    )

    if needs_cvode.args[0]:
        pytest.skip(needs_cvode.kwargs.get('reason', 'no CVODE'))
    params = {**CORE_MODULE_PARAMS, 'ra_crit_cmb': law}

    def solver():
        return _build('core_module', entropy_eos_copy(), params, solver_method='cvode')

    def call(s, start, end, state):
        s.parameters.solver.start_time, s.parameters.solver.end_time = start, end
        s.set_initial_core_temperature(state[0])
        s.set_initial_entropy(state[1])
        s.solve()
        y = np.asarray(s._solution.y)
        n = s._n_stag
        return float(y[n + 1, -1]), y[:n, -1].copy()

    first = solver()
    pm = first.parameters.phase_mixed
    loc, wid = pm.rheological_transition_melt_fraction, pm.rheological_transition_width
    y = _bottom_state(first, loc)
    start_state = (float(y[first._n_stag + 1]), y[: first._n_stag])
    mid = call(first, 0.0, 1.0, start_state)
    P = float(first._P_basic_flat[0])
    phi_mid = float(np.asarray(first.entropy_eos.melt_fraction(P, mid[1][0])).flat[0])
    assert 0.2 < float(liquid_weight(phi_mid, loc, wid)) < 0.8  # stopped inside the transition
    continued = call(first, 1.0, 2.0, mid)
    restored = call(solver(), 1.0, 2.0, mid)
    again = call(solver(), 1.0, 2.0, mid)
    assert restored[0] == again[0]
    np.testing.assert_array_equal(restored[1], again[1])
    assert restored[0] == pytest.approx(continued[0], abs=1e-11)
    np.testing.assert_allclose(restored[1], continued[1], rtol=0.0, atol=1e-11)
    assert continued[0] != mid[0]  # the second call moves the core
