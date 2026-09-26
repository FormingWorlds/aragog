"""Calibrated viscous mixing length of the solid-state MLT closure.

Module under test: ``aragog.rheology.viscous_mixing_length_factor`` and its use in
``EntropyState`` (numpy) and ``aragog.jax.phase.compute_mlt`` (JAX). With the rheology
enabled, the viscous branch uses ``l_v = min(s_bot (r - r_in), s_top (r_out - r))``
(Wagner et al. 2019, eq. 11, in slope form) in place of the Abe length ``l``: the
viscous eddy diffusivity scales by ``(l_v / l)^4`` and the inviscid branch keeps ``l``.

Invariants: slopes of 1 leave every value and the code path unchanged; the factor is
the fourth power of the length ratio; the viscous-inviscid switch stays continuous in
``kappa_h`` (it moves to ``kappa_v = kappa_i``); numpy and JAX agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.rheology import SolidRheologyParams, viscous_mixing_length_factor
from aragog.solver.entropy_state import EntropyState
from tests.test_convection_scaling import _ALPHA, _CP, _G, _K, _RHO, _S_REF, _T_REF, _make_mesh

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]

TOP = -3  # a node near the surface, where the top slope sets l_v
BOTTOM = 3  # a node near the CMB, where the bottom slope sets l_v


def _state(log10visc, top=1.0, bottom=1.0, enabled=True, profile='nearest_boundary'):
    mesh = _make_mesh()
    mesh.settings = type('Settings', (), {'mixing_length_profile': profile})()
    rheo = SolidRheologyParams(
        enabled=enabled,
        yield_stress_c=1e30,
        yield_stress_mu=0.0,
        yield_stress_max=1e40,
        stress_closure_mode='local',
        mlt_top_slope=top,
        mlt_bottom_slope=bottom,
    )

    def phase(pressure):
        ev = EntropyPhaseEvaluator(
            entropy_eos=None,
            gravitational_acceleration=_G,
            const_properties=True,
            const_rho=_RHO,
            const_Cp=_CP,
            const_alpha=_ALPHA,
            const_cond=_K,
            const_log10visc=log10visc,
            const_T_ref=_T_REF,
            const_S_ref=_S_REF,
            rheology=rheo,
        )
        ev.set_pressure(pressure)
        return ev

    evaluator = type('Evaluator', (), {})()
    evaluator.mesh = mesh
    state = EntropyState(
        evaluator=evaluator,
        phase_staggered=phase(mesh.staggered.pressure),
        phase_basic=phase(mesh.basic.pressure),
        conduction=True,
        convection=True,
    )
    rs = np.asarray(mesh.staggered.radii).ravel()
    state.update(_S_REF - 1e-6 * (rs - rs.mean()), 0.0)
    return state


def _kappa(state):
    return np.asarray(state.eddy_diffusivity).ravel()


@pytest.mark.physics_invariant
def test_factor_is_the_fourth_power_of_the_length_ratio():
    """Top slope 0.5: 0.5^4 where the surface is nearer, 1 where the CMB is nearer
    (bottom slope 1), 1 at the boundary nodes; slopes of 1 give exactly 1."""
    r = np.linspace(3.48e6, 6.371e6, 41)
    ml = np.minimum(r - r[0], r[-1] - r)
    q = viscous_mixing_length_factor(r, r[0], r[-1], ml, 0.5, 1.0)
    assert q[-2] == pytest.approx(0.0625, rel=1e-12)
    assert q[1] == pytest.approx(1.0, rel=1e-12)
    assert q[0] == pytest.approx(1.0, rel=0.0) and q[-1] == pytest.approx(1.0, rel=0.0)
    # Exponent guard: the square of the ratio (0.25) or the ratio (0.5) are wrong.
    assert abs(q[-2] - 0.25) > 0.1
    np.testing.assert_array_equal(
        viscous_mixing_length_factor(r, r[0], r[-1], ml, 1.0, 1.0), 1.0
    )


@pytest.mark.physics_invariant
def test_viscous_branch_scales_and_inviscid_branch_is_unchanged():
    """Top slope 0.5 in the viscous regime (log10 viscosity 21): kappa_h times 0.0625
    near the surface and unchanged near the CMB; in the inviscid regime unchanged."""
    ref, cal = _state(21.0), _state(21.0, top=0.5)
    assert _kappa(cal)[TOP] / _kappa(ref)[TOP] == pytest.approx(0.0625, rel=1e-10)
    assert _kappa(cal)[BOTTOM] / _kappa(ref)[BOTTOM] == pytest.approx(1.0, rel=1e-12)
    assert _kappa(ref)[TOP] > 0.0
    inv_ref, inv_cal = _state(-2.0), _state(-2.0, top=0.5)
    assert _kappa(inv_cal)[TOP] / _kappa(inv_ref)[TOP] == pytest.approx(1.0, rel=1e-12)


def test_default_slopes_and_rheology_off_keep_the_abe_path():
    """Slopes of 1, or any slopes with the rheology off, leave no factor (the code
    path of the Abe closure); calibrated slopes need the nearest-boundary profile."""
    assert _state(21.0)._visc_ml_factor is None
    assert _state(21.0, top=0.5, enabled=False)._visc_ml_factor is None
    with pytest.raises(ValueError, match='nearest_boundary'):
        _state(21.0, top=0.5, profile='constant')
    with pytest.raises(ValueError, match='mlt_top_slope must be positive'):
        SolidRheologyParams(mlt_top_slope=0.0)


@pytest.mark.physics_invariant
def test_branch_switch_stays_continuous_in_kappa():
    """Sweep the viscosity over 20 decades in steps of 0.02 at the node near the
    surface with q = 0.0625: ln kappa_h never steps by more than the viscous slope
    (ln 10 * 0.02 = 0.046). Scaling the switch Reynolds number by q instead of q^2
    would put the switch where kappa_v / kappa_i = q^(1/2) and step by ln 4 = 1.39."""
    lvs = np.arange(-2.0, 18.0, 0.02)
    k = np.array([_kappa(_state(lv, top=0.5))[TOP] for lv in lvs])
    step = np.abs(np.diff(np.log(k)))
    assert step.max() < 0.05
    assert np.all(k > 0.0)
    # The two ends are the inviscid (unchanged) and viscous (x 0.0625) limits.
    k_ref = [_kappa(_state(lv))[TOP] for lv in (lvs[0], lvs[-1])]
    assert k[0] / k_ref[0] == pytest.approx(1.0, rel=1e-10)
    assert k[-1] / k_ref[1] == pytest.approx(0.0625, rel=1e-10)


def test_jax_compute_mlt_applies_the_same_factor():
    """JAX ``compute_mlt`` in the viscous regime: kappa_h with top slope 0.5 over
    kappa_h with slopes of 1 equals the numpy factor at every interior node."""
    pytest.importorskip('jax')
    import jax.numpy as jnp

    from aragog.jax.phase import MeshArrays, PhaseParams, PhaseProperties, compute_mlt

    n_basic = 41
    r = np.linspace(3.48e6, 6.371e6, n_basic)
    ml = np.maximum(np.minimum(r - r[0], r[-1] - r), 1.0)
    mesh = MeshArrays(
        d_dr_matrix=jnp.zeros((n_basic, n_basic - 1)),
        quantity_matrix=jnp.zeros((n_basic, n_basic - 1)),
        area=jnp.ones(n_basic),
        volume=jnp.ones(n_basic),
        radii_basic=jnp.asarray(r),
        radii_stag=jnp.asarray(0.5 * (r[1:] + r[:-1])),
        mixing_length=jnp.asarray(ml),
        mixing_length_sq=jnp.asarray(ml**2),
        mixing_length_cu=jnp.asarray(ml**3),
        P_stag=jnp.zeros(n_basic - 1),
        P_basic=jnp.zeros(n_basic),
        dP_dr_basic=jnp.zeros(n_basic),
        gravity=jnp.full(n_basic, 9.81),
    )
    ones = jnp.ones(n_basic)
    phase = PhaseProperties(
        temperature=ones * 2000.0,
        density=ones * 4000.0,
        heat_capacity=ones * 1200.0,
        thermal_expansivity=ones * 3e-5,
        dTdPs=ones,
        melt_fraction=ones * 0.0,
        viscosity=ones * 1.0e21,
        kinematic_viscosity=ones * 1.0e21 / 4000.0,
        thermal_conductivity=ones * 4.0,
        latent_heat=ones,
        capacitance=ones * 4000.0 * 2000.0,
        eta_diff=ones * 1.0e21,
        tau_y=ones * 1.0e40,
        visc_solid_weight=ones * 1.0,
    )
    common = dict(
        enabled=True, kappah_floor=0.0, stress_closure_mode='local', yield_stress_c=1e30
    )
    grad = jnp.full(n_basic, -1e-6)
    k1, _ = compute_mlt(grad, phase, mesh, PhaseParams(**common))
    k2, _ = compute_mlt(grad, phase, mesh, PhaseParams(mlt_top_slope=0.5, **common))
    q = viscous_mixing_length_factor(r, r[0], r[-1], ml, 0.5, 1.0)
    interior = slice(2, -2)
    np.testing.assert_allclose(
        np.asarray(k2)[interior] / np.asarray(k1)[interior], q[interior], rtol=1e-10
    )
    assert q[TOP] == pytest.approx(0.0625, rel=1e-12)
    assert np.all(np.asarray(k1)[interior] > 0.0)
