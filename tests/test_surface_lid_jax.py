"""JAX functions of the sub-grid lid against their NumPy and SciPy counterparts."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')

pytestmark = [pytest.mark.unit, pytest.mark.timeout(120)]


def _arguments():
    x = np.concatenate([np.geomspace(1e-8, 1.0, 2000), np.linspace(1.0, 75.0, 8000)])
    return np.concatenate([-x[::-1], [0.0], x])


def test_dawson_matches_scipy():
    """The sampling sum matches scipy.special.dawsn to relative 1e-12 for |x| <= 75, which
    covers the lid range sqrt(-LID_SHAPE_MIN) = 70.7, and is exactly 0 at 0."""
    from scipy.special import dawsn

    from aragog import surface_lid as sl
    from aragog.jax.surface_lid import dawson

    assert np.sqrt(-sl.LID_SHAPE_MIN) < 75.0
    x = _arguments()
    got, ref = np.asarray(jax.jit(dawson)(x)), dawsn(x)
    nz = ref != 0.0
    np.testing.assert_allclose(got[nz], ref[nz], rtol=1e-12, atol=0)
    assert got[x == 0.0] == 0.0


def test_dawson_derivative():
    """The JAX derivative equals F'(x) = 1 - 2 x F(x), including F'(0) = 1."""
    from scipy.special import dawsn

    from aragog.jax.surface_lid import dawson

    x = _arguments()
    got = np.asarray(jax.vmap(jax.grad(dawson))(x))
    np.testing.assert_allclose(got, 1.0 - 2.0 * x * dawsn(x), rtol=0, atol=1e-13)
    assert float(jax.grad(dawson)(0.0)) == pytest.approx(1.0, rel=1e-14)


def _shapes():
    from aragog import surface_lid as sl

    lo, hi = sl.LID_SHAPE_MIN, sl.LID_SHAPE_MAX
    s = np.concatenate([-np.geomspace(-lo, 1e-8, 400), [0.0], np.geomspace(1e-8, hi, 400)])
    return np.concatenate([s, [-1e-6, 1e-6, -1.0000001e-6, 1.0000001e-6]])


def test_shape_matches_numpy():
    """The JAX shape function equals the NumPy one over the full shape range of the lid, both
    sides of the series switch included: the slopes to relative 1e-12, the mean to absolute
    1e-14 (for s << 0 both forms lose digits in m = 1 + expm1(s) / (2 nu F(nu)), m >= 1e-4)."""
    from aragog import surface_lid as sl
    from aragog.jax.surface_lid import shape

    s = _shapes()
    got = np.array(jax.jit(jax.vmap(shape))(s)).T
    ref = np.array([sl.shape(v) for v in s])
    np.testing.assert_allclose(
        got[:, :2], ref[:, :2], rtol=1e-12, atol=1e-300
    )  # XLA flushes subnormals
    np.testing.assert_allclose(got[:, 2], ref[:, 2], rtol=0, atol=1e-14)


def test_shape_gradient_is_finite_and_matches_differences():
    """The derivative of each output in s is finite everywhere and equals a central
    difference of the NumPy shape function away from the series switch."""
    from aragog import surface_lid as sl
    from aragog.jax.surface_lid import shape

    s = _shapes()
    jac = np.array([jax.vmap(jax.grad(lambda v, i=i: shape(v)[i]))(s) for i in range(3)]).T
    assert np.all(np.isfinite(jac))
    far = (np.abs(s) > 1e-3) & (s > sl.LID_SHAPE_MIN * 0.99) & (s < sl.LID_SHAPE_MAX * 0.99)
    h = 1e-6 * np.maximum(np.abs(s[far]), 1.0)
    fd = np.array(
        [
            (np.array(sl.shape(v + d)) - np.array(sl.shape(v - d))) / (2 * d)
            for v, d in zip(s[far], h)
        ]
    )
    np.testing.assert_allclose(jac[far], fd, rtol=1e-6, atol=1e-12)


def _enthalpies():
    from aragog import surface_lid as sl
    from tests.test_surface_lid import _kinked_table

    return {'identity': sl.LidEnthalpy.linear(1.0, 1.0, 0.0, 0.0), 'kinked': _kinked_table()}


@pytest.mark.parametrize('name', ['identity', 'kinked'])
def test_enthalpy_profile_and_nodes_match_numpy(name):
    """The JAX enthalpy table, profile, quadrature nodes and profile heat equal the NumPy ones,
    outside the table knots included."""
    from aragog import surface_lid as sl
    from aragog.jax import surface_lid as jl

    hv = _enthalpies()[name]
    e = jl.enthalpy(hv)
    T = np.linspace(50.0, 1800.0, 997)
    np.testing.assert_allclose(np.asarray(jl.enthalpy_H(e, T)), hv.H(T), rtol=1e-13, atol=1e-6)
    for s_ in (-3000.0, -2.0, -1e-7, 0.0, 1e-7, 1.5, 300.0):
        x, w = sl.profile_nodes(s_)
        xj, wj = jl.profile_nodes(s_)
        np.testing.assert_allclose(np.asarray(xj), x, rtol=1e-14, atol=1e-16)
        np.testing.assert_allclose(np.asarray(wj), w, rtol=1e-13, atol=0)
        np.testing.assert_allclose(
            np.asarray(jl.profile(s_, x)), sl.profile(s_, x), rtol=1e-12, atol=1e-300
        )
        heat = float(jl.profile_heat(e, s_, 300.0, 1300.0))
        assert heat == pytest.approx(hv.profile_heat(s_, 300.0, 1300.0), rel=1e-12)
        rho = float(jl.profile_density(e, s_, 300.0, 1300.0))
        assert rho == pytest.approx(hv.profile_density(s_, 300.0, 1300.0), rel=1e-13)


def _surface_states(hv, n=200, seed=7):
    from aragog import surface_lid as sl

    rng = np.random.default_rng(seed)
    T_eq = rng.uniform(200.0, 400.0, n)
    T_b = rng.uniform(1300.0, 1800.0, n)
    frac = np.concatenate([rng.uniform(0.0, 1.0, n - 4), [0.0, 1.0, 1e-9, 1 - 1e-9]])
    delta = 10.0 ** rng.uniform(-3.0, 3.3, n)
    E = np.maximum(delta, sl.LID_DELTA_FLOOR) * hv.H(T_eq + frac * (T_b - T_eq))
    return dict(
        delta=delta,
        E=E,
        T_b=T_b,
        k=rng.uniform(2.0, 6.0, n),
        emissivity=rng.uniform(0.5, 1.0, n),
        T_eq=T_eq,
    )


KEYS = ('delta', 'E', 'T_b', 'k', 'emissivity', 'T_eq')


@pytest.mark.parametrize('name', ['identity', 'kinked'])
def test_surface_state_matches_numpy(name):
    """The JAX surface solve equals the NumPy one (Brent root) on random lid states, clipped
    ones included: same clip flag, and the surface temperature, fluxes and mean temperature to
    relative 1e-9."""
    from aragog import surface_lid as sl
    from aragog.jax import surface_lid as jl

    hv = _enthalpies()[name]
    e = jl.enthalpy(hv)
    st = _surface_states(hv)
    solve = jax.jit(jax.vmap(lambda *a: jl.surface_state(*a, sl.LID_DELTA_FLOOR, e)))
    got = solve(*(st[k] for k in KEYS))
    ref = [
        sl.surface_state(*(st[k][i] for k in KEYS), sl.LID_DELTA_FLOOR, hv) for i in range(200)
    ]
    np.testing.assert_array_equal(np.asarray(got[4]), [r.clipped for r in ref])
    assert {r.clipped for r in ref} == {-1, 0, 1}
    for j, field in ((0, 'T_s'), (1, 'F_s'), (2, 'q_b'), (5, 'T_m')):
        want = np.array([getattr(r, field) for r in ref])
        np.testing.assert_allclose(np.asarray(got[j]), want, rtol=1e-9, atol=1e-300)


@pytest.mark.parametrize('name', ['identity', 'kinked'])
def test_surface_state_derivatives_match_differences(name):
    """The implicit derivative of the surface flux in the lid thickness and the lid energy
    equals a central difference of the NumPy solve on unclipped states."""
    from aragog import surface_lid as sl
    from aragog.jax import surface_lid as jl

    hv = _enthalpies()[name]
    e = jl.enthalpy(hv)
    st = _surface_states(hv, 40, seed=11)
    grad = jax.jit(jax.grad(lambda *v: jl.surface_state(*v, 1e-5, e)[1], argnums=(0, 1)))
    n_checked = 0
    for i in range(40):
        a = [float(st[k][i]) for k in KEYS]
        if sl.surface_state(*a, 1e-5, hv).clipped:
            continue
        got = grad(*a)
        for j in (0, 1):
            h = 1e-6 * abs(a[j])
            up, dn = list(a), list(a)
            up[j], dn[j] = a[j] + h, a[j] - h
            fd = (sl.surface_state(*up, 1e-5, hv).F_s - sl.surface_state(*dn, 1e-5, hv).F_s) / (
                2 * h
            )
            assert float(got[j]) == pytest.approx(fd, rel=1e-4, abs=1e-12)
        n_checked += 1
    assert n_checked > 20


def test_rate_functions_match_numpy():
    """The JAX boundary-layer flux, Stefan rate with its stops and existence weight equal the
    NumPy ones on random states that cover each branch (no flux, growth, both stops, melt-back,
    a hidden and a present lid): the rates to relative 1e-12, the weight to absolute 1e-15."""
    from aragog import surface_lid as sl
    from aragog.jax import surface_lid as jl

    rng = np.random.default_rng(3)
    n = 300
    dT = rng.uniform(-50.0, 300.0, n)
    args = (rng.uniform(2, 6, n), rng.uniform(2e3, 4e3, n), rng.uniform(800, 1500, n))
    tail = (rng.uniform(1e-5, 5e-5, n), rng.uniform(5, 25, n), 10.0 ** rng.uniform(-1, 20, n))
    got = jax.vmap(jl.convective_base_flux)(dT, *args, *tail, np.full(n, 450.0))
    want = [
        sl.convective_base_flux(dT[i], *(a[i] for a in args + tail), 450.0) for i in range(n)
    ]
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, atol=0)
    assert min(want) == 0.0 and max(want) > 0.0

    h_top, d_lo = 2000.0, 0.5
    delta = np.concatenate(
        [rng.uniform(0.0, 2000.0, n - 6), [0.3, 0.75, 1.0, 1580, 1595, 1700]]
    )
    q_b, F_b = rng.uniform(0, 200, n), rng.uniform(0, 200, n)
    F_b[:5] = q_b[:5]
    dh = rng.uniform(-1e4, 5e5, n)
    got = jax.vmap(lambda *v: jl.stefan_rate(*v, h_top, d_lo))(
        q_b, F_b, np.full(n, 3e3), dh, delta
    )
    want = np.array(
        [sl.stefan_rate(q_b[i], F_b[i], 3e3, dh[i], delta[i], h_top, d_lo) for i in range(n)]
    )
    np.testing.assert_allclose(np.asarray(got).T, want, rtol=1e-12, atol=0)
    assert len({round(f, 3) for f in want[:, 1]} & {0.0, 1.0}) == 2 and np.any(
        (want[:, 1] > 0) & (want[:, 1] < 1)
    )

    F_b = np.concatenate([[0.0, -1.0, 1e9], 10.0 ** rng.uniform(-2, 6, n - 3)])
    T_b = rng.uniform(1200, 1800, n)
    got = jax.vmap(lambda f, t: jl.existence_weight(f, t, 4.0, 1.0, 255.0, 1.0))(F_b, T_b)
    want = [sl.existence_weight(F_b[i], T_b[i], 4.0, 1.0, 255.0, 1.0) for i in range(n)]
    # 0.5 (1 + tanh x) near the floor has an absolute error of 1 ulp of 1 in both forms
    np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-15)
    w = np.asarray(want)
    assert np.any(w == 0.0) and np.any(w == 1.0) and np.any((w > 0.01) & (w < 0.99))


@pytest.mark.parametrize('name', ['identity', 'kinked'])
def test_lid_rates_match_numpy(name):
    """The JAX lid rates equal the NumPy ones on random states that cover a full, a partial and
    a hidden lid, growth and melt-back, to relative 1e-11."""
    from aragog import surface_lid as sl
    from aragog.jax import surface_lid as jl

    hv = _enthalpies()[name]
    e = jl.enthalpy(hv)
    solve = jax.jit(lambda d, E, T_b: jl.surface_state(d, E, T_b, 4.0, 1.0, 255.0, 1e-5, e))
    rng = np.random.default_rng(5)
    for i in range(40):
        delta = 10.0 ** rng.uniform(-0.2, 3.0)
        T_b = rng.uniform(1300.0, 1700.0)
        E = delta * float(hv.H(rng.uniform(300.0, T_b)))
        surf = sl.surface_state(delta, E, T_b, 4.0, 1.0, 255.0, 1e-5, hv)
        kw = dict(
            delta=delta,
            E_lid=E,
            F_b=rng.uniform(0.0, 2.0) * surf.q_b,
            F_grid=rng.uniform(0.0, 1e4),
            w=[1.0, 0.0, 0.37][i % 3],
            rho=3100.0,
            h_f=rng.uniform(-1e5, 6e5),
            T_b=T_b,
            H_b=float(hv.H(T_b)),
            h_top=2.0e3,
            delta_lo=0.5,
            heating=1e-9,
        )
        ref = sl.lid_rates(surf=surf, hv=hv, **kw)
        js = solve(delta, E, T_b)
        got = jl.lid_rates(surf=js, e=e, **kw)
        want = (ref.ddelta, ref.dE, ref.F_top, ref.w, ref.stop, ref.H_lid)
        got = np.array(got, dtype=float)
        np.testing.assert_allclose(got[[0, 1, 2, 4, 5, 6]], want, rtol=1e-11, atol=1e-25)
        # F_fluid sums terms that can cancel to round-off: bound it by their size
        terms = max(abs(ref.F_top), abs(ref.dE), abs(kw['rho'] * kw['h_f'] * ref.ddelta))
        assert abs(got[3] - ref.F_fluid) <= 1e-11 * terms


def test_surface_solve_evaluates_the_heat_content_at_most_27_times(monkeypatch):
    """The surface solve brackets the shape root with 12 bisection and 10 Illinois steps: at
    most 27 heat-content evaluations per solve, the 2 bound checks and the 3 of the implicit
    derivative included (counted with the loops run in Python)."""
    from aragog import surface_lid as sl
    from aragog.jax import surface_lid as jl

    hv = _enthalpies()['kinked']
    e, st = jl.enthalpy(hv), _surface_states(hv)
    i = next(
        i
        for i in range(200)
        if sl.surface_state(*(st[k][i] for k in KEYS), sl.LID_DELTA_FLOOR, hv).clipped == 0
    )
    calls, heat = [0], jl.profile_heat

    def counted(*a):
        calls[0] += 1
        return heat(*a)

    monkeypatch.setattr(jl, 'profile_heat', counted)
    with jax.disable_jit():
        jl.surface_state(*(jax.numpy.asarray(st[k][i]) for k in KEYS), sl.LID_DELTA_FLOOR, e)
    assert 0 < calls[0] <= 27
