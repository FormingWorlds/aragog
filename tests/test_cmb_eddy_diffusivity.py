"""Eddy diffusivity at the CMB basic node.

The CMB node takes ``max(kh_mlt[1], kh_floor[0])``: the mixing-length value of node 1
before the floor, or the floor of node 0 under its own mask ``dS/dr[0] < D``. These
tests check the continuity of the CMB flux where node 1 changes sign over a stable CMB
gradient, the rule on random states (NumPy and JAX), and the flux invariant that
``EntropyState.update`` checks at the CMB node.

Module under test: ``aragog.solver.entropy_state`` (``EntropyState.update``,
``check_cmb_convective_flux``) and ``aragog.jax.phase.compute_mlt``.

Testing standards: ../docs/How-to/build_tests.md. Physics context:
../docs/Explanations/heat_transport.md.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from aragog.jax.phase import PhaseParams, PhaseProperties, compute_mlt
from aragog.solver.entropy_state import CMBEddyFluxError, check_cmb_convective_flux
from aragog.utilities import tanh_weight
from tests.conftest import EOS_DIR, entropy_eos_copy
from tests.test_entropy_state_kappah_floor_mask import (
    _KAPPAH_FLOOR,
    _PHI_MUSH,
    _PHI_RHEO,
    _PHI_WIDTH,
    _build_state,
    _mesh,
)

hypothesis = pytest.importorskip('hypothesis')
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(240, method='signal')]

needs_eos = pytest.mark.skipif(
    not EOS_DIR.exists(), reason=f'SPIDER P-S tables not found at {EOS_DIR}.'
)

# Stable CMB gradient of the case 38 call start that the ledger refused [J/kg/K/m].
_CMB_STABLE = 2.36e-5


@functools.lru_cache(maxsize=1)
def _eos():
    return entropy_eos_copy(EOS_DIR)


def _state(kappah_floor=_KAPPAH_FLOOR, D=0.0, mixing_length=None):
    """A 30-cell mush state; ``mixing_length`` replaces the nearest-boundary length."""
    state, mesh = _build_state(_eos(), kappah_floor=kappah_floor, convective_threshold=D)
    if mixing_length is not None:
        ml = np.full(mesh.N + 1, float(mixing_length))
        state._mixing_length, state._mixing_length_sq, state._mixing_length_cu = (
            ml,
            ml**2,
            ml**3,
        )
    return state, mesh


def _update(state, mesh, dsdr_head, phi_stag):
    """``EntropyState.update`` with dS/dr[:len(dsdr_head)] set and -1e-6 above."""
    S_sol = np.asarray(_eos().solidus_entropy(mesh.staggered.pressure)).ravel()
    S_liq = np.asarray(_eos().liquidus_entropy(mesh.staggered.pressure)).ravel()
    dSdr = np.full(mesh.N + 1, -1.0e-6)
    dSdr[: len(dsdr_head)] = dsdr_head
    state.update(S_sol + np.asarray(phi_stag) * (S_liq - S_sol), time=0.0, dSdr=dSdr)
    return dSdr


def _basic(state, name):
    return np.asarray(getattr(state.phase_basic, name)()).ravel()


def _jax_twin(state, mesh, dSdr, kappah_floor, D, edt=1.0):
    """JAX ``compute_mlt`` on the basic-node properties of ``state``."""
    n = dSdr.size
    props = PhaseProperties(
        temperature=jnp.asarray(_basic(state, 'temperature')),
        density=jnp.asarray(_basic(state, 'density')),
        heat_capacity=jnp.asarray(_basic(state, 'heat_capacity')),
        thermal_expansivity=jnp.asarray(_basic(state, 'thermal_expansivity')),
        dTdPs=jnp.zeros(n),
        melt_fraction=jnp.asarray(_basic(state, 'melt_fraction')),
        viscosity=jnp.ones(n),
        kinematic_viscosity=jnp.asarray(_basic(state, 'kinematic_viscosity')),
        thermal_conductivity=jnp.ones(n),
        latent_heat=jnp.ones(n),
        capacitance=jnp.ones(n),
    )
    jmesh = _mesh(n, ml=jnp.asarray(state._mixing_length), gravity=10.0)

    def run(floor):
        params = PhaseParams(
            kappah_floor=floor,
            phi_rheo=_PHI_RHEO,
            phi_width=_PHI_WIDTH,
            eddy_diff_thermal=edt,
            convective_threshold=D,
        )
        return np.asarray(compute_mlt(jnp.asarray(dSdr), props, jmesh, params)[0])

    return run(kappah_floor), run(0.0)


# ---------------------------------------------------------------------------
# V1: continuity across S1 = S0 over a stable CMB gradient (node 1 on its floor)
# ---------------------------------------------------------------------------


@needs_eos
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('mixing_length', [None, 1.0e5], ids=['nearest', 'constant'])
def test_cmb_flux_is_continuous_where_node_1_changes_sign(mixing_length):
    """Over a stable CMB gradient, the CMB flux is the same on both sides of
    dS/dr[1] = 0, on the NumPy and the JAX path; jconv[0] is -rho T kh_mlt[1] dS/dr[0]
    and the total flux at node 0 is the conductive law (Bower et al. 2018 Eq. 11) plus it.

    Discriminating: if node 0 copies the floored value of node 1, node 1 on the
    S1 < S0 side carries its floor (about 5 m^2/s) and the CMB flux jumps by
    rho T kappah_floor f(phi_1) dS/dr[0], about -2.9e3 W m^-2 in this state, against a
    total CMB flux of 7e-4 W m^-2. At steps of S1 - S0 = +-1e-9 and +-1e-12 J/kg/K the
    difference between the sides must shrink with the step (the mixing-length part of
    node 1 is linear in its gradient here), and at +-1e-17 J/kg/K it is below 1e-6 of
    the flux.
    """
    d_flux, jump_main = {}, {}
    for step in (1.0e-9, 1.0e-12, 1.0e-17):
        flux = {}
        for side in (-1.0, 1.0):
            state, mesh = _state(mixing_length=mixing_length)
            g1 = side * step / mesh.dr[0]
            dSdr = _update(state, mesh, [_CMB_STABLE, g1], [_PHI_MUSH] * mesh.N)
            rho_t = _basic(state, 'density') * _basic(state, 'temperature')
            k = _basic(state, 'thermal_conductivity')
            cp = _basic(state, 'heat_capacity')
            dtdps = _basic(state, 'dTdPs')
            jcond_0 = -k[0] * (_basic(state, 'temperature')[0] / cp[0] * dSdr[0]) - k[0] * (
                dtdps[0] * state._dP_dr_basic[0]
            )
            kh_jax, kh_mlt_jax = _jax_twin(state, mesh, dSdr, _KAPPAH_FLOOR, 0.0)
            jconv_0 = -rho_t[0] * float(kh_mlt_jax[1]) * dSdr[0]
            assert state.jconv[0] == pytest.approx(jconv_0, rel=1e-12, abs=1.0e-12)
            assert state.jcond[0] == pytest.approx(jcond_0, rel=1e-12)
            assert state.heat_flux[0] == pytest.approx(jcond_0 + jconv_0, rel=1e-12)
            assert float(kh_jax[0]) == pytest.approx(
                state.eddy_diffusivity[0],
                rel=1e-10,
                abs=1.0e-25,  # numpy residual drive
            )
            if side < 0:
                assert state.eddy_diffusivity[1] > 0.1 * _KAPPAH_FLOOR  # node 1 on its floor
                jump_main[step] = rho_t[0] * state.eddy_diffusivity[1] * dSdr[0]
            flux[side] = state.heat_flux[0]
        d_flux[step] = abs(flux[-1.0] - flux[1.0])
        assert jump_main[step] > 1.0e3 and d_flux[step] < 1.0e-5 * jump_main[step]
    assert d_flux[1.0e-9] >= 100.0 * d_flux[1.0e-12]
    assert d_flux[1.0e-17] < 1.0e-6 * abs(flux[1.0]) and abs(flux[1.0]) > 1.0e-5


@needs_eos
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_cmb_flux_across_s1_equal_s0(validation_plot):
    """Over a stable CMB gradient, the CMB flux across S1 - S0 from -1e-6 to 1e-6 J/kg/K
    has no step at S1 = S0, while the flux of the copy rule (the CMB node copies the floored
    value of node 1) steps there by about 2.9e3 W m^-2."""
    steps = np.r_[-np.logspace(-6, -14, 9), np.logspace(-14, -6, 9)]
    flux, copied = [], []
    for d in steps:
        state, mesh = _state()
        dSdr = _update(state, mesh, [_CMB_STABLE, d / mesh.dr[0]], [_PHI_MUSH] * mesh.N)
        rho_t = _basic(state, 'density')[0] * _basic(state, 'temperature')[0]
        flux.append(state.heat_flux[0])
        copied.append(state.jcond[0] - rho_t * state.eddy_diffusivity[1] * dSdr[0])
    flux, copied = np.array(flux), np.array(copied)
    side = steps < 0
    jump = copied[~side][0] - copied[side][-1]
    assert jump > 1.0e3 and abs(flux[side][-1] - flux[~side][0]) < 1.0e-10 * jump

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(steps, copied, 's', ms=4, mfc='none', color='C1', label='copy rule')
        ax.plot(steps, flux, 'o', ms=3, color='C0', label='CMB node rule')
        ax.set_xscale('symlog', linthresh=1.0e-14)
        ax.set_yscale('symlog', linthresh=1.0e-3)
        ticks = [-1.0e-6, -1.0e-10, 0.0, 1.0e-10, 1.0e-6]
        labels = ['$-10^{-6}$', '$-10^{-10}$', '0', '$10^{-10}$', '$10^{-6}$']
        ax.set_xticks(ticks, labels)
        ax.set(xlabel='$S_1 - S_0$ [J kg$^{-1}$ K$^{-1}$]', ylabel='CMB heat flux [W m$^{-2}$]')
        ax.legend(fontsize=8, loc='center right')
        fig.tight_layout()
        return fig

    validation_plot(figure, 'cmb_eddy_flux_continuity')


# ---------------------------------------------------------------------------
# V3: the rule on random states, NumPy == JAX
# ---------------------------------------------------------------------------


def _signed(mag_lo=-12.0, mag_hi=-2.0):
    return st.tuples(st.sampled_from([-1.0, 1.0]), st.floats(mag_lo, mag_hi)).map(
        lambda t: t[0] * 10.0 ** t[1]
    )


@needs_eos
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@settings(max_examples=300, deadline=None, derandomize=True)
@given(
    g0=_signed(),
    g1=_signed(),
    phi0=st.floats(0.0, 1.0),
    phi1=st.floats(0.0, 1.0),
    D=st.one_of(st.just(0.0), st.floats(-9.0, -5.0).map(lambda e: 10.0**e)),
    edt=st.one_of(st.just(1.0), st.floats(-1.0, 3.0).map(lambda e: -(10.0**e))),
    kappah_floor=st.one_of(st.just(0.0), st.floats(0.1, 100.0)),
)
def test_cmb_rule_holds_on_random_states(g0, g1, phi0, phi1, D, edt, kappah_floor):
    """kh[0] = max(kh_mlt[1], kappah_floor f(phi_0) [dS/dr[0] < D]) on random CMB
    gradients, melt fractions, thresholds, diffusivity modes and floors; NumPy == JAX."""
    state, mesh = _state(kappah_floor=kappah_floor, D=D)
    state._eddy_diff_thermal = edt
    phi_stag = np.full(mesh.N, _PHI_MUSH)
    phi_stag[0], phi_stag[1] = phi0, phi1
    dSdr = _update(state, mesh, [g0, g1], phi_stag)
    kh_jax, kh_mlt_jax = _jax_twin(state, mesh, dSdr, kappah_floor, D, edt=edt)
    phi_b0 = float(_basic(state, 'melt_fraction')[0])
    floor_0 = kappah_floor * float(tanh_weight(phi_b0, _PHI_RHEO, _PHI_WIDTH)) * (g0 < D)
    expect = max(float(kh_mlt_jax[1]), floor_0)
    assert float(kh_jax[0]) == pytest.approx(expect, rel=1e-12, abs=0.0)
    assert state.eddy_diffusivity[0] == pytest.approx(float(kh_jax[0]), rel=1e-10, abs=1e-30)


# ---------------------------------------------------------------------------
# The CMB flux invariant
# ---------------------------------------------------------------------------

_RHO_T = 4500.0 * 4000.0


def _check(jconv0, dsdr0, kh_mlt_1, floor_0, threshold, rho_t=_RHO_T):
    check_cmb_convective_flux(
        jconv_0=jconv0,
        rho_t_0=rho_t,
        dsdr_0=dsdr0,
        kh_mlt_1=kh_mlt_1,
        kh_floor_0=floor_0,
        threshold=threshold,
    )


@pytest.mark.parametrize(
    ('dsdr0', 'kh0', 'kh_mlt_1', 'floor_0'),
    [
        (1.0e-4, 0.0, 0.0, 0.0),  # stable, node 1 stable
        (1.0e-4, 3.0, 3.0, 0.0),  # stable, node 1 convective (the kept downward flux)
        (-1.0e-4, 5.0, 0.0, 5.0),  # convective, own floor
        (-1.0e-4, 8.0, 8.0, 5.0),  # convective, mixing length above the floor
        (0.0, 5.0, 0.0, 5.0),  # zero gradient
    ],
)
def test_invariant_passes_the_rule(dsdr0, kh0, kh_mlt_1, floor_0):
    """The rule passes the invariant in every kind of CMB state."""
    jconv0 = -_RHO_T * kh0 * dsdr0
    _check(jconv0, dsdr0, kh_mlt_1, floor_0, 0.0)


@pytest.mark.parametrize(
    ('dsdr0', 'jconv0', 'kh_mlt_1', 'floor_0', 'part'),
    [
        # (i) the floor of node 1 (5) crosses a stable CMB where kh_mlt[1] is 0
        (1.0e-4, -_RHO_T * 5.0 * 1.0e-4, 0.0, 0.0, r'\(i\)'),
        # (ii) a convective CMB below its own floor (no floor at node 0)
        (-1.0e-4, _RHO_T * 1.0 * 1.0e-4, 1.0, 5.0, r'\(ii\)'),
        # (iii) a flux up a stable gradient
        (1.0e-4, _RHO_T * 3.0 * 1.0e-4, 3.0, 0.0, r'\(iii\)'),
        # (iii) a flux down a convective gradient
        (-1.0e-4, -_RHO_T * 6.0 * 1.0e-4, 0.0, 5.0, r'\(iii\)'),
    ],
)
def test_invariant_refuses_each_broken_part(dsdr0, jconv0, kh_mlt_1, floor_0, part):
    """Each part of the invariant refuses the flux that breaks it, and names the part."""
    with pytest.raises(CMBEddyFluxError, match=part):
        _check(jconv0, dsdr0, kh_mlt_1, floor_0, 0.0)


@pytest.mark.parametrize(
    ('dsdr0', 'jconv0', 'kh_mlt_1', 'floor_0'),
    [
        (1.0e-4, 0.0, 3.0, 0.0),  # the mixing-length part of node 1 dropped over a stable CMB
        (-1.0e-4, _RHO_T * 5.0 * 1.0e-4, 8.0, 5.0),  # the floor where node 1 is above it
        (-1.0e-4, _RHO_T * 8.0 * 1.0e-4 * (1.0 + 1.0e-6), 8.0, 5.0),  # 1e-6 off the rule
    ],
)
def test_invariant_refuses_a_flux_between_the_bounds(dsdr0, jconv0, kh_mlt_1, floor_0):
    """Part (iv) refuses a flux that parts (i) to (iii) let through but that is not the rule."""
    with pytest.raises(CMBEddyFluxError, match=r'\(iv\)'):
        _check(jconv0, dsdr0, kh_mlt_1, floor_0, 0.0)


def test_invariant_passes_round_off_of_the_rule():
    """A flux 1e-13 off the rule passes; the tolerance of the check is 1e-12."""
    _check(_RHO_T * 8.0 * 1.0e-4 * (1.0 + 1.0e-13), -1.0e-4, 8.0, 5.0, 0.0)


@pytest.mark.parametrize('bad', [np.nan, np.inf], ids=['nan', 'inf'])
@pytest.mark.parametrize('arg', ['jconv0', 'dsdr0', 'kh_mlt_1', 'floor_0', 'rho_t'])
def test_invariant_skips_a_state_that_is_not_finite(arg, bad):
    """A state with a non-finite input is not checked: a flux up a stable gradient, refused by
    part (iii) when finite, passes with any input NaN or infinite. Without the guard an infinite
    input still reaches part (iii); a NaN flux or gradient passes either way, as each comparison
    with NaN is false."""
    kw = dict(jconv0=_RHO_T * 3.0 * 1.0e-4, dsdr0=1.0e-4, kh_mlt_1=3.0, floor_0=0.0)
    with pytest.raises(CMBEddyFluxError, match=r'\(iii\)'):
        _check(**kw, threshold=0.0)
    _check(**{**kw, arg: bad}, threshold=0.0)


def test_invariant_uses_the_threshold():
    """With D > 0 a node 0 gradient in [0, D) is convective for part (ii), and a gradient at D
    is stable for part (i)."""
    D = 1.0e-6
    with pytest.raises(CMBEddyFluxError, match=r'\(ii\)'):
        _check(0.0, 5.0e-7, 0.0, 5.0, D)
    _check(-_RHO_T * 5.0 * 5.0e-7, 5.0e-7, 0.0, 5.0, D)
    _check(-_RHO_T * 3.0 * D, D, 3.0, 0.0, D)  # at D the mask of node 0 sets its floor to 0
    with pytest.raises(CMBEddyFluxError, match=r'\(i\)'):
        _check(-_RHO_T * 5.0 * D, D, 3.0, 0.0, D)


@needs_eos
@pytest.mark.parametrize('D', [0.0, 1.0e-6])
def test_update_checks_the_invariant(monkeypatch, D):
    """``EntropyState.update`` calls the invariant check with the CMB values it used."""
    import aragog.solver.entropy_state as es

    seen = []
    real = es.check_cmb_convective_flux

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(es, 'check_cmb_convective_flux', spy)
    state, mesh = _state(D=D)
    _update(state, mesh, [_CMB_STABLE, 1.0e-15], [_PHI_MUSH] * mesh.N)
    assert seen, 'update did not check the CMB invariant'
    kw = seen[-1]
    rho_t = _basic(state, 'density')[0] * _basic(state, 'temperature')[0]
    assert kw['jconv_0'] == state.jconv[0] and kw['dsdr_0'] == state.dSdr[0]
    assert kw['rho_t_0'] == rho_t and kw['threshold'] == D
    assert (kw['kh_mlt_1'], kw['kh_floor_0']) == state.cmb_eddy_diffusivity_inputs


# ---------------------------------------------------------------------------
# A CMB flux fault inside a solve
# ---------------------------------------------------------------------------


def _raise_after(monkeypatch, n_calls):
    """Make the CMB flux check fail from its call ``n_calls + 1`` on."""
    import aragog.solver.entropy_state as es

    real, calls = es.check_cmb_convective_flux, [0]

    def faulty(**kw):
        calls[0] += 1
        if calls[0] > n_calls:
            raise CMBEddyFluxError('CMB invariant (i) failed at the probe')
        return real(**kw)

    monkeypatch.setattr(es, 'check_cmb_convective_flux', faulty)


@needs_eos
@pytest.mark.smoke
def test_a_flux_fault_in_the_cvode_rhs_reaches_the_caller(monkeypatch):
    """A CMB flux fault in the NumPy RHS early in a CVODE call (where CVODE turns it into a
    failed step) stops the call with ``CMBEddyFluxError``."""
    from tests.test_cmb_closure_ledger import _solver

    _raise_after(monkeypatch, 3)
    solver = _solver(0.9, end_time=20.0, solver_method='cvode', use_jax_jacobian=False)
    with pytest.raises(CMBEddyFluxError, match='at the probe'):
        solver.solve()


@needs_eos
@pytest.mark.smoke
def test_a_flux_fault_keeps_the_last_accepted_state(monkeypatch):
    """After a CMB flux fault in the energy replay of a call, the solver holds the CMB gradient
    of its last accepted call."""
    import aragog.solver.entropy_state as es
    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_cmb_closure_ledger import _ledger

    _, _, _, solver = _ledger('energy_balance', 'radau', 0.9, end_time=2.0)
    accepted = solver.get_current_dSdr_cmb()
    replay, in_replay = EntropySolver._compute_step_energy_integrals, [False]

    def flagged(self, *a, **kw):
        in_replay[0] = True
        return replay(self, *a, **kw)

    def faulty(**kw):
        if in_replay[0]:
            raise CMBEddyFluxError('CMB invariant (i) failed at the probe')

    monkeypatch.setattr(EntropySolver, '_compute_step_energy_integrals', flagged)
    monkeypatch.setattr(es, 'check_cmb_convective_flux', faulty)
    solver.parameters.solver.end_time = 20.0
    with pytest.raises(CMBEddyFluxError, match='at the probe'):
        solver.solve()
    assert in_replay[0] and solver.get_current_dSdr_cmb() == accepted
