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
                state.eddy_diffusivity[0], rel=1e-10, abs=1e-12
            )
            if side < 0:
                assert state.eddy_diffusivity[1] > 0.1 * _KAPPAH_FLOOR  # node 1 on its floor
                jump_main[step] = rho_t[0] * state.eddy_diffusivity[1] * dSdr[0]
            flux[side] = state.heat_flux[0]
        d_flux[step] = abs(flux[-1.0] - flux[1.0])
        assert jump_main[step] > 1.0e3 and d_flux[step] < 1.0e-5 * jump_main[step]
    assert d_flux[1.0e-9] >= 100.0 * d_flux[1.0e-12]
    assert d_flux[1.0e-17] < 1.0e-6 * abs(flux[1.0]) and abs(flux[1.0]) > 1.0e-5


# ---------------------------------------------------------------------------
# V3: the rule on random states, NumPy == JAX
# ---------------------------------------------------------------------------


def _signed(mag_lo=-12.0, mag_hi=-2.0):
    return st.tuples(st.sampled_from([-1.0, 1.0]), st.floats(mag_lo, mag_hi)).map(
        lambda t: t[0] * 10.0 ** t[1]
    )


@needs_eos
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
    check_cmb_convective_flux(jconv0, _RHO_T, dsdr0, kh_mlt_1, floor_0, 0.0)


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
        check_cmb_convective_flux(jconv0, _RHO_T, dsdr0, kh_mlt_1, floor_0, 0.0)


def test_invariant_uses_the_threshold():
    """With D > 0 a node 0 gradient in [0, D) is convective for part (ii)."""
    with pytest.raises(CMBEddyFluxError, match=r'\(ii\)'):
        check_cmb_convective_flux(0.0 * _RHO_T, _RHO_T, 5.0e-7, 0.0, 5.0, 1.0e-6)
    check_cmb_convective_flux(-_RHO_T * 5.0 * 5.0e-7, _RHO_T, 5.0e-7, 0.0, 5.0, 1.0e-6)


@needs_eos
def test_update_checks_the_invariant(monkeypatch):
    """``EntropyState.update`` calls the invariant check with the CMB values it used."""
    import aragog.solver.entropy_state as es

    seen = []
    real = es.check_cmb_convective_flux

    def spy(*args):
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(es, 'check_cmb_convective_flux', spy)
    state, mesh = _state()
    _update(state, mesh, [_CMB_STABLE, 1.0e-15], [_PHI_MUSH] * mesh.N)
    assert seen, 'update did not check the CMB invariant'
    jconv0, rho_t0, dsdr0, kh_mlt_1, floor_0, D = seen[-1]
    assert jconv0 == state.jconv[0] and dsdr0 == state.dSdr[0] and D == 0.0
    assert (kh_mlt_1, floor_0) == state.cmb_eddy_diffusivity_inputs
