"""Tests of the open part of a live lid per total area at the in-step lid weight.

Under the prescribed flux boundary with a live lid the surface loses
``F_top = w F_s + A_open``, with the open part per total area
``A_open = F_atm - w L + (1 - w) G_eff(w) dT_m`` and
``G_eff(w) = G F' / (F' + w G_l + (1 - w) G)`` at the lid weight ``w`` of each evaluation
(``aragog.surface_lid.open_part_total``). So
``F_top = F_atm + w (F_s - L) + (1 - w) G_eff(w) dT_m``. ``L`` is the lid flux of the call
balance [W/m^2], ``F'`` the slope of the atmosphere flux with the surface temperature, ``G`` the
skin conductance of the open surface and ``G_l`` the lid conductance [W/m^2/K];
``dT_m = T_m - T_m0`` [K] is the rise of ``T_m = max(T_node, T_b)`` since the call start.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from aragog import surface_lid as sl
from aragog.solver.entropy_solver import SECS_PER_YEAR
from tests import test_lid_open_flux_instep as instep
from tests.conftest import entropy_eos_copy
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, needs_eos
from tests.test_lid_mush_base import _solver
from tests.test_lid_open_flux_instep import _eval, _state

pytestmark = pytest.mark.timeout(900)

FIELDS = (
    'open_lid_flux',
    'open_surface_slope',
    'open_skin_conductance',
    'open_lid_conductance',
)
# a call of the response cases: F' < G and G_l << G [W/m^2/K]; L and F_atm [W/m^2]
FP, G_SKIN, G_LID = 30.0, 200.0, 2.0
W_ROW = 0.7  # the lid weight of the row [-]
W_SET = (W_ROW, 0.5, 0.0, 1.0 - 1e-6, 1.0 - 1e-15, 1.0)


def _g_eff(w, Fp=FP, G=G_SKIN, Gl=G_LID):
    """``G_eff(w)`` written out here, apart from the code under test."""
    return 0.0 if Fp == 0.0 else G * Fp / (Fp + w * Gl + (1.0 - w) * G)


def _lid_call(w, F_open_total, F_grid=-200.0, T_s=500.0):
    """``lid_rates`` of a 1 m lid of constant properties held at ``T_s`` [K], weight ``w``."""
    T_b, k = 1400.0, 3.0
    hv = sl.LidEnthalpy.linear(3000.0, 1200.0, T_b, 0.0)
    E = hv.profile_heat(0.5, T_s, T_b)
    surf = sl.fixed_surface_state(1.0, E, T_b, k, T_s, sl.LID_DELTA_FLOOR, hv)
    rates = sl.lid_rates(
        delta=1.0,
        E_lid=E,
        surf=surf,
        F_b=500.0,
        F_grid=F_grid,
        w=w,
        rho=3000.0,
        H_f=4.0e9,
        T_b=T_b,
        H_b=3.0e9,
        hv=hv,
        h_top=10.0,
        delta_lo=0.5,
        heating=0.0,
        F_open_total=F_open_total,
    )
    return rates, float(surf.F_s)


# --- 1. the energy sum ---------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize('w', W_SET)
def test_the_surface_loss_closes_the_energy_sum_at_every_weight(w):
    """``F_top = F_atm + w (F_s - L) + (1 - w) G_eff(w) dT_m`` to 1e-12 relative at each
    weight, with ``F_s`` of the lid state; the row form ``w F_s + (1 - w) F_atm`` misses it."""
    F_atm, L, dT_m = -200.0, 150.0, 7.5
    A = sl.open_part_total(w, F_atm, L, FP, G_SKIN, G_LID, dT_m)
    rates, F_s = _lid_call(w, A, F_grid=F_atm)
    want = F_atm + w * (F_s - L) + (1.0 - w) * _g_eff(w) * dT_m
    assert rates.F_top == pytest.approx(want, rel=1e-12, abs=1e-12 * abs(F_atm))
    row, _ = _lid_call(w, None, F_grid=F_atm)
    assert abs(row.F_top - want) > 1e-6 * abs(want)
    # the fluid flux keeps the conservation of lid_rates with the new surface loss
    assert rates.F_fluid - rates.F_top == pytest.approx(row.F_fluid - row.F_top, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize('w', W_SET)
def test_a_lid_at_its_call_flux_and_no_rise_loses_the_atmosphere_flux(w):
    """With ``F_s = L`` and ``dT_m = 0`` the surface loses ``F_atm`` at every weight."""
    F_atm = 1234.5
    _, F_s = _lid_call(0.5, None)
    A = sl.open_part_total(w, F_atm, F_s, FP, G_SKIN, G_LID, 0.0)
    rates, _ = _lid_call(w, A, F_grid=F_atm)
    assert rates.F_top == pytest.approx(F_atm, rel=1e-12)
    assert abs(_lid_call(w, None, F_grid=F_atm)[0].F_top - F_atm) > 1.0 or w == 0.0


@pytest.mark.unit
def test_without_the_open_total_lid_rates_keeps_the_grid_form():
    """``F_open_total=None`` gives ``w F_s + (1 - w) F_grid``, bit for bit."""
    for w in (0.0, 0.3, 1.0):
        rates, F_s = _lid_call(w, None, F_grid=-200.0)
        assert rates.F_top == w * F_s + (1.0 - w) * -200.0
    assert _lid_call(0.3, None)[0].F_top != _lid_call(0.3, 0.0)[0].F_top


# --- 2. the response against the call balance --------------------------------------------


def _call_balance(w, dT_m, F_atm=-200.0, L=150.0, Fp=FP, G=G_SKIN, Gl=G_LID):
    """``(dT_s, F)`` of the linear call balance of the surface at weight ``w``.

    The atmosphere takes ``F_atm + F' dT_s``; the lid area gives ``w (L - G_l dT_s)``, the
    open skin ``(F_atm - w L) + (1 - w) G (dT_m - dT_s)``. Solved as a 2 x 2 system.
    """
    A = np.array([[-Fp, 1.0], [w * Gl + (1.0 - w) * G, 1.0]])
    b = np.array([F_atm, F_atm + (1.0 - w) * G * dT_m])
    return np.linalg.solve(A, b)


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.parametrize('w', (0.0, 0.5, W_ROW, 0.99))
def test_the_response_is_the_slope_of_the_call_balance(w):
    """The central difference of ``F_top`` in ``dT_m`` equals the slope of the call balance,
    ``(1 - w) G_eff(w)``, to 1e-6; the row-weight value ``(1 - w) G_eff(w_r)`` (at
    ``w != w_r``) and the denominator without ``w G_l`` (at ``w > 0``) fall outside."""
    h, dT_m = 1e-3, 4.0
    oracle = (_call_balance(w, dT_m + h)[1] - _call_balance(w, dT_m - h)[1]) / (2.0 * h)
    assert oracle == pytest.approx((1.0 - w) * _g_eff(w), rel=1e-6)

    def F_top(d):
        return _lid_call(w, sl.open_part_total(w, -200.0, 150.0, FP, G_SKIN, G_LID, d))[0].F_top

    code = (F_top(dT_m + h) - F_top(dT_m - h)) / (2.0 * h)
    assert code == pytest.approx(oracle, rel=1e-6)
    row = (1.0 - w) * _g_eff(W_ROW)
    no_lid = (1.0 - w) * G_SKIN * FP / (FP + (1.0 - w) * G_SKIN)
    if w != W_ROW:
        assert abs(row - oracle) > 1e-6 * abs(oracle)
    if w > 0.0:
        assert abs(no_lid - oracle) > 1e-6 * abs(oracle)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_surface_warming_of_the_balance_is_the_response_over_the_slope():
    """``F' dT_s = (1 - w) G_eff(w) dT_m`` at every weight: the open part tracks the atmosphere
    along its own slope, not the skin conductance."""
    for w in np.linspace(0.0, 1.0, 11):
        dT_s, F = _call_balance(w, 10.0)
        assert FP * dT_s == pytest.approx((1.0 - w) * _g_eff(w) * 10.0, rel=1e-12, abs=1e-12)
        assert F == pytest.approx(-200.0 + FP * dT_s, rel=1e-12)


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_open_part_closes_the_energy_sum_and_follows_the_call_balance(validation_plot):
    """Across the lid weight, ``F_top`` of ``lid_rates`` with ``open_part_total`` is the energy
    sum ``F_atm + w (F_s - L) + (1 - w) G_eff(w) dT_m`` (rel 1e-12), and its slope in ``dT_m``
    is the slope of the linear call balance (rel 1e-6). The row form ``w F_s + (1 - w) F_atm``
    and the response at the row weight ``(1 - w) G_eff(w_r)`` fall outside."""
    F_atm, L, dT_m, h = -200.0, 150.0, 7.5, 1e-3
    ws = np.linspace(0.0, 1.0, 11)
    sums = []
    for w in ws:
        rates, F_s = _lid_call(w, sl.open_part_total(w, F_atm, L, FP, G_SKIN, G_LID, dT_m))
        want = F_atm + w * (F_s - L) + (1.0 - w) * _g_eff(w) * dT_m
        assert abs(rates.F_top - want) <= 1e-12 * abs(F_atm)
        row = _lid_call(w, None)[0].F_top
        if w < 1.0:
            assert abs(row - want) > 1e-6 * abs(want)
        sums.append((rates.F_top, want, row))
    sums = np.array(sums)

    def F_top(w, d):
        return _lid_call(w, sl.open_part_total(w, F_atm, L, FP, G_SKIN, G_LID, d))[0].F_top

    resp = []
    for w in ws:
        oracle = (_call_balance(w, dT_m + h)[1] - _call_balance(w, dT_m - h)[1]) / (2.0 * h)
        code = (F_top(w, dT_m + h) - F_top(w, dT_m - h)) / (2.0 * h)
        assert code == pytest.approx(oracle, rel=1e-6, abs=1e-9)
        row = (1.0 - w) * _g_eff(W_ROW)
        if 0.0 < w < 1.0 and abs(w - W_ROW) > 1e-12:
            assert abs(row - oracle) > 1e-6 * abs(oracle)
        resp.append((code, oracle, row))
    resp = np.array(resp)
    err = np.max(np.abs(sums[:, 0] - sums[:, 1])) / (1e-12 * abs(F_atm))
    assert err <= 1.0

    def figure():
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
        a1.plot(ws, sums[:, 1], color='C0', label='energy sum')
        a1.plot(ws, sums[:, 0], 'o', ms=4, mfc='none', color='C1', label='code')
        a1.plot(ws, sums[:, 2], '--', color='C2', label='w F_s + (1 - w) F_atm')
        a1.set(xlabel='lid weight w', ylabel='F_top [W/m2]')
        # the round-off difference is text: its value depends on the platform
        a1.text(
            0.97,
            0.06,
            f'largest miss / tol = {err:.1e}',
            ha='right',
            transform=a1.transAxes,
        )
        a1.legend(fontsize=8)
        a2.plot(ws, resp[:, 1], color='C0', label='call balance')
        a2.plot(ws, resp[:, 0], 'o', ms=4, mfc='none', color='C1', label='code')
        a2.plot(ws, resp[:, 2], '--', color='C2', label=f'G_eff at w = {W_ROW}')
        a2.set(xlabel='lid weight w', ylabel='dF_top / dT_m [W/m2/K]')
        a2.legend(fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'surface_lid_open_part_in_step')


# --- 3. limits and bounds ------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_open_conductance_at_the_ends_of_the_weight():
    """``G_eff(0) = G F' / (F' + G)``, ``G_eff(1) = G F' / (F' + G_l)``; 0 for ``F' = 0``."""
    assert sl.open_conductance(0.0, FP, G_SKIN, G_LID) == pytest.approx(
        G_SKIN * FP / (FP + G_SKIN), rel=1e-15
    )
    assert sl.open_conductance(1.0, FP, G_SKIN, G_LID) == pytest.approx(
        G_SKIN * FP / (FP + G_LID), rel=1e-15
    )
    assert sl.open_conductance(1.0, 0.0, G_SKIN, 0.0) == 0.0
    assert sl.open_conductance(0.3, 0.0, G_SKIN, G_LID) == 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
@settings(max_examples=200, deadline=None)
@given(
    Fp=st.floats(1e-6, 1e5),
    G=st.floats(1e-3, 1e5),
    frac=st.floats(0.0, 1.0),
    flip=st.booleans(),
)
def test_the_open_conductance_is_bounded_and_monotonic_in_the_weight(Fp, G, frac, flip):
    """``0 < G_eff <= G`` for ``F' > 0``, and ``G_eff`` moves in ``w`` with the sign of
    ``G - G_l`` (rising for the physical ``G_l <= G``)."""
    Gl = G * frac if not flip else G * (1.0 + 10.0 * frac)
    w = np.linspace(0.0, 1.0, 41)
    g = np.array([sl.open_conductance(float(x), Fp, G, Gl) for x in w])
    assert np.all(g > 0.0) and np.all(g <= G * (1.0 + 1e-15))
    step = np.diff(g) * np.sign(G - Gl)
    assert np.all(step >= -1e-15 * G)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_open_part_is_continuous_at_full_lid_weight():
    """``A_open`` is ``F_atm - L`` at ``w = 1`` and tends to it as ``w -> 1``; the start
    value per open area ``(F_atm - w L) / (1 - w)`` grows without bound there."""
    F_atm, L, dT_m = -200.0, 150.0, 25.0
    at_one = sl.open_part_total(1.0, F_atm, L, FP, G_SKIN, G_LID, dT_m)
    assert at_one == F_atm - L
    for eps in (1e-3, 1e-6, 1e-9, 1e-12):
        A = sl.open_part_total(1.0 - eps, F_atm, L, FP, G_SKIN, G_LID, dT_m)
        assert abs(A - at_one) <= eps * (abs(L) + G_SKIN * abs(dT_m)) * (1.0 + 1e-6)
    assert abs((F_atm - (1.0 - 1e-12) * L) / 1e-12) > 1e12


# --- 4. JAX --------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_jax_open_part_equals_numpy_and_has_a_finite_gradient_at_full_weight():
    """The JAX ``open_part_total`` equals NumPy (rel 1e-14); its gradient in ``w`` is finite at
    ``w = 1`` and ``1 - 1e-12`` and equals ``-L - G F' / (F' + G_l) dT_m`` at ``w = 1``."""
    jax = pytest.importorskip('jax')
    from aragog.jax import surface_lid as jl

    F_atm, L, dT_m = -200.0, 150.0, 25.0

    def A(w):
        return jl.open_part_total(w, F_atm, L, FP, G_SKIN, G_LID, dT_m)

    for w in (0.0, 0.3, W_ROW, 1.0 - 1e-12, 1.0):
        assert float(A(w)) == pytest.approx(
            sl.open_part_total(w, F_atm, L, FP, G_SKIN, G_LID, dT_m), rel=1e-14
        )
    grads = [float(jax.grad(A)(w)) for w in (1.0, 1.0 - 1e-12)]
    assert all(math.isfinite(g) for g in grads)
    want = -L - G_SKIN * FP / (FP + G_LID) * dT_m
    assert grads[0] == pytest.approx(want, rel=1e-12)
    assert grads[1] == pytest.approx(want, rel=1e-9)
    assert (
        float(jax.grad(lambda w: jl.open_part_total(w, F_atm, L, 0.0, G_SKIN, 0.0, dT_m))(1.0))
        == -L
    )


@pytest.mark.unit
def test_the_nan_canary_sees_a_guarded_division_at_full_weight():
    """The hazard the product form avoids: a division by ``1 - w`` inside an untaken
    ``jnp.where`` branch gives a NaN gradient at ``w = 1``, though its value is finite."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def guarded(w):
        w = jnp.asarray(w, dtype=jnp.float64)
        return jnp.where(w < 1.0, (-200.0 - w * 150.0) / (1.0 - w), 0.0) * (1.0 - w)

    assert float(guarded(1.0)) == 0.0
    assert math.isnan(float(jax.grad(guarded)(1.0)))
    assert float(jax.grad(guarded)(0.5)) == pytest.approx(-150.0, rel=1e-12)


# --- 5. the fields -------------------------------------------------------------------------

_VALID = (150.0, 30.0, 200.0, 2.0)
_BASE = dict(
    outer_boundary_condition=4,
    outer_boundary_value=1.0,
    inner_boundary_condition=2,
    inner_boundary_value=0.0,
    emissivity=1.0,
    equilibrium_temperature=273.0,
    core_heat_capacity=880.0,
    core_bc='quasi_steady',
)
_NOT_FINITE = st.sampled_from([math.nan, math.inf, -math.inf])
_NOT_A_NUMBER = st.one_of(st.booleans(), st.text(max_size=3))
_BAD = (
    st.one_of(_NOT_FINITE, _NOT_A_NUMBER),
    st.one_of(_NOT_FINITE, _NOT_A_NUMBER, st.floats(max_value=0.0, exclude_max=True)),
    st.one_of(_NOT_FINITE, _NOT_A_NUMBER, st.floats(max_value=0.0)),
    st.one_of(_NOT_FINITE, _NOT_A_NUMBER, st.floats(max_value=0.0, exclude_max=True)),
)


@pytest.mark.unit
@settings(max_examples=200, deadline=None)
@given(data=st.data(), i=st.integers(0, 3))
def test_each_open_field_is_refused_out_of_range_by_name(data, i):
    """A field that is not finite, not a number, or out of its range is refused by name."""
    from aragog.config.boundary import check_open_part

    part = list(_VALID)
    part[i] = data.draw(_BAD[i])
    with pytest.raises(ValueError, match=FIELDS[i]):
        check_open_part(tuple(part), 4, True)


@pytest.mark.unit
@settings(max_examples=100, deadline=None)
@given(
    L=st.floats(allow_nan=False, allow_infinity=False),
    Fp=st.floats(0.0, 1e12),
    G=st.floats(1e-300, 1e12),
    Gl=st.floats(0.0, 1e12),
)
@example(L=-5.0, Fp=0.0, G=1.0, Gl=0.0)  # each bound at its edge, and a lid flux below 0
def test_finite_open_fields_in_range_are_taken(L, Fp, G, Gl):
    """Any finite ``L`` and ``F' >= 0``, ``G > 0``, ``G_l >= 0`` pass, also on the attrs layer,
    which keeps the values; a refusal of such a set fails the test."""
    from aragog.config.boundary import BoundaryConfig, check_open_part

    try:
        assert check_open_part((L, Fp, G, Gl), 4, True) is None
        bc = BoundaryConfig(
            **_BASE, subgrid_lid=True, lid_mush_base=True, **dict(zip(FIELDS, (L, Fp, G, Gl)))
        )
    except ValueError as err:
        pytest.fail(f'an in-range open-part set was refused: {err}')
    assert (bc.open_lid_flux, bc.open_surface_slope) == (L, Fp)
    assert (bc.open_skin_conductance, bc.open_lid_conductance) == (G, Gl)


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(st.sets(st.integers(0, 3), min_size=1, max_size=3))
def test_a_partial_set_of_open_fields_is_refused_naming_the_missing(given_idx):
    """1 to 3 of the 4 fields set: refused, naming each missing field."""
    from aragog.config.boundary import check_open_part

    part = tuple(_VALID[i] if i in given_idx else None for i in range(4))
    with pytest.raises(ValueError, match='missing') as err:
        check_open_part(part, 4, True)
    for i in range(4):
        assert (FIELDS[i] in str(err.value).split('missing')[1]) is (i not in given_idx)


@pytest.mark.unit
@pytest.mark.parametrize('bc, mush', [(1, True), (4, False), (1, False), (5, True)])
def test_the_open_fields_need_the_flux_boundary_and_the_mush_base(bc, mush):
    """All None passes anywhere; the 4 fields set are refused off the flux boundary with the
    mush base, naming the fields."""
    from aragog.config.boundary import BoundaryConfig, check_open_part

    assert check_open_part((None,) * 4, bc, mush) is None
    with pytest.raises(ValueError, match='open_lid_flux.*outer_boundary_condition = 4'):
        check_open_part(_VALID, bc, mush)
    base = {**_BASE, 'outer_boundary_condition': bc}
    with pytest.raises(ValueError, match='open_skin_conductance'):
        BoundaryConfig(**base, **dict(zip(FIELDS, _VALID, strict=True)))
    assert BoundaryConfig(**base).open_lid_flux is None


@pytest.mark.unit
def test_a_config_file_with_the_open_fields_restores_them(tmp_path):
    """A TOML run file with the 4 fields restores them on the parameters, through a deep copy
    and the ``show-config`` dump; without them all 4 are None."""
    from importlib.resources import files

    from aragog.parser import Parameters

    text = files('aragog').joinpath('cfg/abe_solid.toml').read_text()
    assert text.count('outer_boundary_condition = 1\n') == 1
    text = text.replace('outer_boundary_condition = 1\n', 'outer_boundary_condition = 4\n')
    head = '[boundary_conditions]\n'
    lines = ['subgrid_lid = true', 'lid_mush_base = true', 'surface_temperature = 800.0']
    values = (-3.25, 48.43, 200.0, 7.5)
    path = tmp_path / 'open.toml'
    body = ''.join(f'{v}\n' for v in lines)
    path.write_text(text.replace(head, head + body))
    assert all(
        getattr(Parameters.from_file(str(path)).boundary_conditions, n) is None for n in FIELDS
    )
    body += ''.join(f'{n} = {v!r}\n' for n, v in zip(FIELDS, values, strict=True))
    path.write_text(text.replace(head, head + body))
    p = Parameters.from_file(str(path))
    for bc in (p.boundary_conditions, copy.deepcopy(p).boundary_conditions):
        assert tuple(getattr(bc, n) for n in FIELDS) == values
    dumped = dataclasses.asdict(p)['boundary_conditions']
    assert tuple(dumped[n] for n in FIELDS) == values
    path.write_text(text.replace(head, head + body.replace('open_lid_flux = -3.25\n', '')))
    with pytest.raises(ValueError, match='missing open_lid_flux'):
        Parameters.from_file(str(path))


# --- solver level --------------------------------------------------------------------------

F_ATM, L_CALL = -200.0, 3700.0  # the boundary value and the lid flux of the call [W/m^2]
OPEN = dict(zip(FIELDS, (L_CALL, FP, G_SKIN, G_LID), strict=True))


def _solvers(eos, end=1.0):
    """A thin-shell lid solver with the 4 open fields, and the same solver without them."""
    kw = dict(end=end, outer=4, F_atm=F_ATM, thin=True, surface_temperature=500.0)
    return _solver(eos, **kw, lid_delta_on=1.0, **OPEN), _solver(eos, **kw, lid_delta_on=1.0)


def _T_m(ref, y):
    """``(w, F_s, T_m)`` of the state ``y`` from the solver without the open fields."""
    _, _, w, T_node, T_b = _eval(ref, y)
    return w, float(ref._lid_last.surf.F_s), max(T_node, T_b)


def _identity(w, F_s, dT_m):
    return F_ATM + w * (F_s - L_CALL) + (1.0 - w) * _g_eff(w) * dT_m


@pytest.fixture(scope='class')
def eos():
    """An EOS copy per test class."""
    return entropy_eos_copy(EOS_DIR)


@pytest.mark.smoke
@needs_eos
class TestOpenTotalSolver:
    @pytest.mark.physics_invariant
    def test_the_right_hand_side_loses_the_energy_sum(self, eos):
        """At states of lid weight 1, near 1 and in (0, 1), the surface loss of the right-hand
        side is ``F_atm + w (F_s - L) + (1 - w) G_eff(w) dT_m`` (rel 1e-12)."""
        s, ref = _solvers(eos)
        y0 = _state(ref, 100.0)
        T_m0 = _T_m(ref, y0)[2]
        s._set_lid_open_reference(y0)
        ws = []
        for dS, flat in (
            (10.0, False),
            (20.0, False),
            (80.0, True),
            (90.0, True),
            (120.0, True),
        ):
            y = _state(ref, dS, flat=flat)
            w, F_s, T_m = _T_m(ref, y)
            F_top = _eval(s, y)[1]
            want = _identity(w, F_s, T_m - T_m0)
            assert F_top == pytest.approx(want, rel=1e-12, abs=1e-12 * abs(F_ATM))
            ws.append(w)
        assert ws[0] == 1.0 and 1.0 - 1e-9 < ws[1] < 1.0 and 0.1 < min(ws[2:4]) < 0.95

    @pytest.mark.physics_invariant
    def test_the_booked_surface_energy_closes_the_energy_sum(self, eos):
        """The booked surface integral of a call is the trapezoid, over the accepted states,
        of ``F_atm + w (F_s - L) + (1 - w) G_eff(w) dT_m`` (rel 1e-6), with ``T_m0`` of the
        start state; the response term is not negligible."""
        s, ref = _solvers(eos, end=20.0)
        y0 = _state(ref, 100.0)
        s._S0 = y0.copy()
        s.solve()
        sol = s._solution
        assert sol.success
        t, Y = sol.get('energy_trace') or (sol.t, sol.y)
        t, Y = np.asarray(t, dtype=float), np.asarray(Y, dtype=float)
        assert t.size >= 3
        T_m0 = _T_m(ref, y0)[2]
        loss, response = [], []
        for yi in Y.T:
            w, F_s, T_m = _T_m(ref, yi)
            loss.append(_identity(w, F_s, T_m - T_m0))
            response.append((1.0 - w) * _g_eff(w) * (T_m - T_m0))
        A = 4.0 * np.pi * float(s._r_basic_flat[-1]) ** 2
        dt = np.diff(t) * SECS_PER_YEAR

        def trap(values):
            p = -np.asarray(values) * A
            return float(np.sum(0.5 * (p[:-1] + p[1:]) * dt))

        total = trap(loss)
        assert abs(sol.energy_integrals['F_int'] - total) <= 1e-6 * abs(total)
        assert abs(trap(response)) > 1e-4 * abs(total)

    @pytest.mark.physics_invariant
    def test_jax_right_hand_side_equals_numpy(self, eos):
        """At states with ``w`` in (0, 1) and ``dT_m != 0`` the JAX entropy rates equal NumPy
        (rel 1e-12) and the lid rates too (rel 1e-9), and differ from the JAX rates without the
        open fields."""
        s, ref = _solvers(eos)
        s._set_lid_open_reference(_state(ref, 100.0))
        f1, f0 = instep.TestOpenFluxJax._rhs(s)[2], instep.TestOpenFluxJax._rhs(ref)[2]
        n = s._n_stag
        for dS in (90.0, 95.0, 105.0, 120.0):
            y = _state(ref, dS)
            want, _, w, T_node, T_b = _eval(s, y)
            assert 0.0 < w < 1.0 and max(T_node, T_b) != s._lid_T_m0
            got = np.asarray(f1(y))
            np.testing.assert_allclose(got[:n], want[:n], rtol=1e-12, atol=0.0)
            np.testing.assert_allclose(got[-2:], want[-2:], rtol=1e-9)  # the lid rows
            assert got[n - 1] != np.asarray(f0(y))[n - 1]

    @staticmethod
    def _smooth(s, dS):
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[:n] = np.linspace(800.0, s._lid_S_sol + dS, n)
        y[-2:] = (1.0, float(s._lid_hv.profile_heat(0.0, 500.0, s._lid_T_sol)))
        return y

    @pytest.mark.physics_invariant
    def test_the_jacobian_is_finite_at_full_weight_and_holds_the_weight_terms(self, eos):
        """The JAX Jacobian is finite at ``w = 1`` and ``1 - 1e-12``; at a smooth state with
        ``w`` in (0, 1) its top-cell column equals the central difference (rel 1e-6), and the
        open part changes that column (the ``G_eff'(w)`` and ``(F_s - L) dw`` terms)."""
        s, ref = _solvers(eos)
        s._set_lid_open_reference(_state(ref, 100.0))
        rhs_fn, jac_fn, _ = instep.TestOpenFluxJax._rhs(s)
        sc = s._build_nondim_scales()
        for dS in (10.0, 20.0):
            y = _state(ref, dS, flat=False)
            w = _eval(ref, y)[2]
            assert w == 1.0 or 1.0 - 1e-11 < w < 1.0
            x = (y - sc.state_offset) / sc.state_scale
            J, fy = np.zeros((x.size, x.size)), np.empty(x.size)
            assert rhs_fn(0.0, x, fy) == 0 and jac_fn(0.0, x, fy, J) == 0
            assert np.all(np.isfinite(J))
        y = self._smooth(s, 100.0)
        w = _eval(ref, y)[2]
        assert 0.01 < w < 0.99
        top = s._n_stag - 1
        (J1, fd1), (J0, _) = (instep.TestOpenFluxJax()._top_column(x, y) for x in (s, ref))
        assert J1[top] == pytest.approx(fd1[top], rel=1e-6)
        assert abs(J1[top] - J0[top]) > 1e-3 * abs(J0[top])


def _open_flux_stub(T_node, part=(150.0, 1.0, 1000.0, 2.0), T_m0=1490.0):
    """A solver stand-in with the open-part fields, the top basic node and the start reference."""
    from types import SimpleNamespace

    basic = SimpleNamespace(temperature=lambda: np.array([T_node + 30.0, T_node]))
    return SimpleNamespace(
        _lid_open_part=lambda: part,
        _lid_open_T_m0=lambda: T_m0,
        state=SimpleNamespace(phase_basic=basic),
    )


@pytest.mark.unit
@pytest.mark.parametrize(('T_node', 'dT_m'), [(1540.0, 50.0), (1400.0, -50.0)])
def test_the_call_site_passes_the_fields_in_order_and_holds_the_node(T_node, dT_m):
    """The solver passes (L, F', G, G_l) to open_part_total in that order, with
    dT_m = max(T_node, T_b) - T_m0; at w = 0.9 with F' = 1 and G = 1000 W/m^2/K a swap of F' and G
    moves the open part by tens of W/m^2."""
    from aragog.solver.entropy_solver import EntropySolver

    w, F_atm, T_b = 0.9, -200.0, 1440.0
    got = EntropySolver._lid_open_flux(_open_flux_stub(T_node), w, F_atm, T_b)
    want = sl.open_part_total(w, F_atm, 150.0, 1.0, 1000.0, 2.0, dT_m)
    swapped = sl.open_part_total(w, F_atm, 150.0, 1000.0, 1.0, 2.0, dT_m)
    assert got == pytest.approx(want, rel=1e-12)
    assert abs(want - swapped) > 10.0  # the order is told apart by a wide margin
    assert (
        EntropySolver._lid_open_flux(_open_flux_stub(T_node, part=None), w, F_atm, T_b) is None
    )


@pytest.mark.unit
def test_the_parser_leaves_the_open_part_unset():
    """The 4 open-part fields of the parsed boundary default to None, so a config without them
    has no open part and no partial set."""
    from aragog.parser import _BoundaryConditionsParameters

    defaults = {f.name: f.default for f in dataclasses.fields(_BoundaryConditionsParameters)}
    assert [defaults[name] for name in FIELDS] == [None] * 4
