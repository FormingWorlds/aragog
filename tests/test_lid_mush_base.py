"""Tests of the mush base of the sub-grid lid: base at the solidus, floor flux, top-cell floor."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aragog import surface_lid as sl
from aragog.solver.entropy_solver import SECS_PER_YEAR
from tests.conftest import entropy_eos_copy
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, _build, needs_eos


def _solver(eos, S0=3050.0, end=1.0, mush=True, rheo=None, n_nodes=10):
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=n_nodes, end_time=end)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.lid_mush_base = mush
    p.boundary_conditions.normalize()
    p.phase_solid.rheology = rheo or SolidRheologyParams(
        enabled=True, stress_closure_mode='local', lid_base_mode='fixed'
    )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(S0)
    return s


def test_floor_weight_limits_and_smoothness():
    """The weight is 0 at and below the floor, 1 above floor + width, and C2 at both ends."""
    S_sol = 2000.0
    lo, hi = S_sol + sl.LID_FLOOR_GAP, S_sol + sl.LID_FLOOR_GAP + sl.LID_FLOOR_WIDTH
    assert sl.floor_weight(lo, S_sol) == 0.0
    assert sl.floor_weight(lo - 50.0, S_sol) == 0.0
    assert sl.floor_weight(hi, S_sol) == 1.0
    assert sl.floor_weight(hi + 50.0, S_sol) == 1.0
    assert sl.floor_weight(0.5 * (lo + hi), S_sol) == pytest.approx(0.5, abs=1e-14)
    h = 1e-4 * sl.LID_FLOOR_WIDTH
    for end, sign in ((lo, 1.0), (hi, -1.0)):
        y0 = sl.floor_weight(end, S_sol)
        y1 = sl.floor_weight(end + sign * h, S_sol)
        assert abs(y1 - y0) / h < 1e-6


@settings(max_examples=60, deadline=None)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
def test_floor_weight_stays_in_range(x):
    assert 0.0 <= sl.floor_weight(x, 0.0) <= 1.0


def test_floor_flux_constant_diffusivity_hand_value():
    """rho T kappa (S_f - S_sol) / (h_f / 2) for a constant integrand: Phi is its integral."""
    Phi = 3000.0 * 1500.0 * 2.0 * 7.0  # rho T kappa dS
    h, delta = 2000.0, 400.0
    want = 3000.0 * 1500.0 * 2.0 * 7.0 / (0.5 * (h - delta))
    assert sl.floor_flux(Phi, h, delta) == pytest.approx(want, rel=1e-14)


@pytest.mark.parametrize(
    'field, value',
    [('lid_mush_base', 1), ('lid_mush_base', 'yes'), ('lid_mush_base', None)],
)
def test_lid_mush_base_must_be_bool(field, value):
    from aragog.config.boundary import check_lid_mush_base

    with pytest.raises(ValueError, match=field):
        check_lid_mush_base(value, True)


def test_lid_mush_base_is_off_by_default():
    from aragog.parser import _BoundaryConditionsParameters

    assert _BoundaryConditionsParameters.__dataclass_fields__['lid_mush_base'].default is False


@needs_eos
class TestSolver:
    @pytest.fixture(scope='class')
    def eos(self):
        return entropy_eos_copy(EOS_DIR)

    def test_refused_without_the_rheology(self, eos):
        from aragog.parser import SolidRheologyParams

        with pytest.raises(ValueError, match='lid_mush_base'):
            _solver(eos, rheo=SolidRheologyParams(enabled=False))

    def test_refused_at_a_start_below_the_floor(self, eos):
        s = _solver(eos, S0=3050.0)
        S_cold = s._lid_S_sol - 10.0
        with pytest.raises(ValueError, match='lid_mush_base'):
            s.set_initial_entropy(S_cold)
            s.solve()

    def test_base_is_the_solidus(self, eos):
        s = _solver(eos)
        base = s._lid_base_state(1500.0, 0.01, s._lid_S_sol + 3.0)
        assert base['S_b'] == s._lid_S_sol
        assert base['T_b'] == pytest.approx(s._lid_T_sol, rel=1e-12)

    def test_phi_table_is_the_quadrature_of_the_floor(self, eos):
        """Phi(S) against a direct fine quadrature with the solver's EOS and ramp."""
        from aragog.solver.entropy_state import tanh_weight, zero_anchored_weight

        s = _solver(eos)
        ph = s.state.phase_staggered
        P, a = s._lid_P, s._lid_S_sol
        b = a + 0.3 * (s._lid_S_liq - a)
        S = np.linspace(a, b, 40001)
        Pn = np.full(S.size, P)
        phi = np.asarray(eos.melt_fraction(Pn, S), dtype=float).ravel()
        f = np.asarray(zero_anchored_weight(tanh_weight, phi, ph._phi_rheo, ph._phi_width))
        y = np.asarray(eos.density(Pn, S) * eos.temperature(Pn, S)).ravel() * f
        want = float(np.trapezoid(y, S)) * s.parameters.energy.kappah_floor
        assert float(s._lid_Phi(b)) == pytest.approx(want, rel=1e-5)
        assert float(s._lid_Phi(a)) == 0.0

    def test_blend_at_the_floor_holds_the_fluid(self, eos):
        """At g = 0 the base flux is the flux from below and the lid acts: F_fluid = F_in up
        to the Stefan floor term, so the top cell has no flux term of cooling."""
        s = _solver(eos)
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[n - 1] = s._lid_S_sol + sl.LID_FLOOR_GAP
        y[-2:] = s._lid_start(y[n - 1])
        s._dSdt_single(0.0, y)
        ev = s._lid_last
        F_in = float(s.state._heat_flux[-2])
        assert ev.rates.w == 1.0
        A_in, A = s._area_flat[-2], s._area_flat[-1]
        assert A_in < A
        assert ev.F_b == F_in * A_in / A

    def test_floor_holds_the_fluid_on_a_sphere(self, eos):
        """At g = 0 the base takes the heat that enters through the smaller lower face, so the
        fluid entropy holds; with the flux per unit area of that face it falls."""
        s = _solver(eos)
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[n - 1] = s._lid_S_sol + sl.LID_FLOOR_GAP
        y[-2:] = s._lid_start(y[n - 1])
        dS = np.asarray(s._dSdt_single(0.0, y)).ravel()[n - 1]
        ev = s._lid_last
        F_in = float(s.state._heat_flux[-2])
        assert F_in > 0.0
        ph = s.state.phase_staggered
        cap = float(np.ravel(ph.density())[-1] * np.ravel(ph.temperature())[-1]) * ev.V_fluid
        mismatch = (s._area_flat[-1] - s._area_flat[-2]) * F_in
        assert abs(dS) / SECS_PER_YEAR * cap < 1.0e-6 * mismatch

    def test_invariant_refuses_an_accepted_state_below_the_solidus(self, eos):
        s = _solver(eos)
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[-2:] = s._lid_start(3050.0)
        y[n - 1] = s._lid_S_sol - 1.0
        s._dSdt_single(0.0, y)
        with pytest.raises(RuntimeError, match='below the solidus'):
            s._check_lid_step(s._lid_delta_lo, 3, 1.0)

    def test_hand_over_at_the_fraction_of_the_cell(self, eos):
        s = _solver(eos)
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[-2:] = s._lid_start(3050.0)
        y[-2] = sl.LID_HANDOVER_FRAC * s._lid_h_top * (1.0 - 1e-6)
        assert s._lid_root_value(0.0, y) < 0.0
        y[-2] = sl.LID_HANDOVER_FRAC * s._lid_h_top * (1.0 + 1e-6)
        assert s._lid_root_value(0.0, y) > 0.0
        y[-2] = 0.0
        assert s._lid_root_value(0.0, y) == -1.0

    def test_short_run_closes_energy_and_keeps_the_floor(self, eos):
        s = _solver(eos, end=5.0)
        s.solve()
        sol = s._solution
        assert sol.status == 0
        ei = sol.energy_integrals
        budget = ei['F_int'] + ei['F_cmb'] + ei['Q_radio'] + ei['Q_tidal']
        assert abs(ei['solver_residual']) <= 1e-10 * abs(ei['F_int'])
        assert abs(ei['state_heat'] - budget) <= 1e-2 * abs(budget)
        S_f = np.asarray(sol.y)[s._n_stag - 1]
        assert np.all(S_f >= s._lid_S_sol)

    def test_jax_lid_top_matches_numpy(self, eos):
        from aragog.jax import surface_lid as jl
        from tests.test_subgrid_lid_jax_rhs import entropy_eos_jax

        s = _solver(eos)
        lp, ej = s.jax_lid_params(), entropy_eos_jax(EOS_DIR)
        assert float(lp.mush) == 1.0
        for dS in (0.5, 3.0, 40.0, 900.0):
            S_f = s._lid_S_sol + dS
            T_b = s._lid_T_sol
            top = dict(
                T=T_b + 0.1 * dS,
                rho=3000.0,
                cp=1200.0,
                k=4.0,
                alpha=2e-5,
                g=9.8,
                eta=1e20,
                phi=0.01,
            )
            y = np.array([3.0, 3.0 * float(s._lid_hv.profile_heat(0.0, 400.0, T_b))])
            out = jl.lid_top(lp, ej, y, S_f, top, 50.0, 80.0, np.nan, 0.0)
            F_b = sl.mush_base_flux(
                float(s._lid_Phi(S_f)),
                s._lid_h_top,
                3.0,
                50.0 * s._area_flat[-2] / s._area_flat[-1],
                sl.floor_weight(S_f, s._lid_S_sol),
                float(
                    sl.convective_base_flux(
                        top['T'] - T_b, 4.0, 3000.0, 1200.0, 2e-5, 9.8, 1e20, s._lid_ra_crit
                    )
                ),
            )
            assert float(
                jl.mush_base_flux(
                    lp,
                    S_f,
                    3.0,
                    50.0,
                    sl.convective_base_flux(
                        top['T'] - T_b, 4.0, 3000.0, 1200.0, 2e-5, 9.8, 1e20, s._lid_ra_crit
                    ),
                )
            ) == pytest.approx(F_b, rel=1e-10)
            assert math.isfinite(float(out[0]))


def test_flux_surface_state_carries_the_prescribed_flux():
    """A lid under a prescribed flux: T_s from the conduction, F_s = F_atm; clipped at 1 K."""
    hv = sl.LidEnthalpy.linear(3000.0, 1200.0, 1400.0, 0.0)
    delta, F_atm = 10.0, 60.0
    E = delta * hv.profile_heat(0.0, 1400.0 - F_atm * delta / 4.0, 1400.0)
    surf = sl.flux_surface_state(delta, E, 1400.0, 4.0, F_atm, sl.LID_DELTA_FLOOR, hv)
    assert surf.F_s == pytest.approx(F_atm, rel=1e-6)
    assert surf.T_s == pytest.approx(1400.0 - F_atm * delta / 4.0, rel=1e-6)
    hot = sl.flux_surface_state(delta, E, 1400.0, 4.0, 1e6, sl.LID_DELTA_FLOOR, hv)
    assert hot.T_s == 1.0
    assert hot.F_s < 1e6


@pytest.mark.parametrize(
    'bc, mush, ok', [(4, True, True), (4, False, False), (1, True, True), (5, True, False)]
)
def test_prescribed_flux_needs_the_mush_base(bc, mush, ok):
    from aragog.config.boundary import check_subgrid_lid

    args = (True, 1.0, None, 3, 450.0, bc, 'quasi_steady', False)
    if ok:
        check_subgrid_lid(*args, lid_mush_base=mush)
    else:
        with pytest.raises(ValueError, match='outer'):
            check_subgrid_lid(*args, lid_mush_base=mush)


@needs_eos
class TestFluxSolver:
    @pytest.fixture(scope='class')
    def eos(self):
        return entropy_eos_copy(EOS_DIR)

    def _s(self, eos, F_atm=200.0, end=1.0, S0=None):
        from aragog.parser import SolidRheologyParams
        from aragog.solver.entropy_solver import EntropySolver

        p = _build(
            core_bc='quasi_steady',
            outer_bc=4,
            inner_bc=2,
            n_nodes=10,
            end_time=end,
            outer_bc_value=F_atm,
        )
        p.boundary_conditions.subgrid_lid = True
        p.boundary_conditions.lid_mush_base = True
        p.boundary_conditions.normalize()
        p.phase_solid.rheology = SolidRheologyParams(
            enabled=True, stress_closure_mode='local', lid_base_mode='fixed'
        )
        s = EntropySolver(p, entropy_eos=eos)
        s.initialize()
        s.set_initial_entropy(3050.0 if S0 is None else S0)
        return s

    def test_surface_loss_is_the_prescribed_flux(self, eos):
        s = self._s(eos)
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[n - 1] = s._lid_S_sol + sl.LID_FLOOR_GAP + 0.5 * sl.LID_FLOOR_WIDTH
        y[-2:] = s._lid_start(y[n - 1])
        y[-2:] *= 20.0
        s._dSdt_single(0.0, y)
        ev = s._lid_last
        assert float(s.state._heat_flux[-1]) == pytest.approx(200.0, rel=1e-6)
        assert ev.surf.T_s < ev.T_b

    def test_short_run_closes_energy_and_reports_the_crust_surface(self, eos):
        s = self._s(eos, end=2.0)
        s.solve()
        sol = s._solution
        assert sol.status == 0
        ei = sol.energy_integrals
        assert abs(ei['solver_residual']) <= 1e-10 * abs(ei['F_int'])
        st = s.get_state()
        ev = s._lid_last
        T_node = float(np.asarray(s.state.phase_basic.temperature()).ravel()[-1])
        want = ev.rates.w * ev.T_b + (1.0 - ev.rates.w) * T_node
        assert st.T_magma == pytest.approx(want, rel=1e-9)
        assert st.lid_T_surface == ev.surf.T_s
        assert st.lid_conductance == pytest.approx(
            s._lid_k * sl.shape(ev.surf.s)[0] / ev.delta, rel=1e-12
        )

    def test_jax_flux_mode_matches_numpy(self, eos):
        from aragog.jax import surface_lid as jl

        s = self._s(eos)
        lp = s.jax_lid_params()
        assert float(lp.flux) == 1.0
        T_b = s._lid_T_sol
        y = np.array([3.0, 3.0 * float(s._lid_hv.profile_heat(0.0, 1300.0, T_b))])
        surf_j = jl.surface_state(
            y[0], y[1], T_b, s._lid_k, 1.0, 255.0, sl.LID_DELTA_FLOOR, lp.e, lp.flux, 100.0
        )
        surf_n = sl.flux_surface_state(
            y[0], y[1], T_b, s._lid_k, 100.0, sl.LID_DELTA_FLOOR, s._lid_hv
        )
        assert float(surf_j[0]) == pytest.approx(surf_n.T_s, rel=1e-9)
        assert float(surf_j[1]) == pytest.approx(surf_n.F_s, rel=1e-9)
        on = s._lid_delta_on
        for F_b, T_s in (
            (10.0, T_b - 100.0),
            (1.0e4, T_b - 1.0),
            (0.0, T_b - 5.0),
            (-2.0, T_b - 5.0),
            (7.0, T_b),
            (7.0, T_b + 3.0),
            (s._lid_k * 100.0 / on, T_b - 100.0),
        ):
            assert float(jl.flux_weight(F_b, T_b, T_s, s._lid_k, on)) == pytest.approx(
                sl.flux_weight(F_b, T_b, T_s, s._lid_k, on), rel=1e-12, abs=1e-14
            )
        for F_atm in (-5.0, 0.0):
            surf_j = jl.surface_state(
                y[0], y[1], T_b, s._lid_k, 1.0, 255.0, sl.LID_DELTA_FLOOR, lp.e, lp.flux, F_atm
            )
            assert float(surf_j[0]) == T_b
            assert float(surf_j[1]) == 0.0

    def test_a_lid_at_the_limit_of_its_profile_is_reported(self, eos, caplog):
        s = self._s(eos, F_atm=1.0e4)
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[n - 1] = s._lid_S_sol + sl.LID_FLOOR_GAP + 0.5 * sl.LID_FLOOR_WIDTH
        T_b = s._lid_T_sol
        y[-2:] = [1.0e4, 1.0e4 * float(s._lid_hv.profile_heat(0.0, 300.0, T_b))]
        s._dSdt_single(0.0, y)
        ev = s._lid_last
        assert ev.surf.T_s == sl.LID_T_S_MIN
        assert ev.surf.F_s < 1.0e4
        assert ev.rates.w > 0.5
        with caplog.at_level('WARNING', logger='fwl.aragog.solver.entropy_solver'):
            s._check_lid_step(s._lid_delta_lo, 0, 0.0)
        assert 'less or more than the prescribed flux' in caplog.text


NOT_BOOL = st.one_of(
    st.integers(),
    st.floats(allow_nan=True),
    st.text(max_size=5),
    st.none(),
    st.lists(st.booleans(), max_size=2),
)
GOOD_LID = dict(
    lid_delta_on=1.0,
    lid_delta_lo=0.5,
    lid_handover_cells=1,
    lid_ra_crit=1000.0,
    core_bc='quasi_steady',
    param_utbl=False,
)


@settings(max_examples=80, deadline=None)
@given(NOT_BOOL, st.booleans())
def test_lid_mush_base_refuses_every_other_type(value, subgrid):
    from aragog.config.boundary import check_lid_mush_base

    with pytest.raises(ValueError, match='lid_mush_base must be true or false'):
        check_lid_mush_base(value, subgrid)


@settings(max_examples=40, deadline=None)
@given(st.booleans(), st.booleans())
def test_lid_mush_base_needs_the_lid(mush, subgrid):
    from aragog.config.boundary import check_lid_mush_base

    if mush and not subgrid:
        with pytest.raises(ValueError, match='needs subgrid_lid'):
            check_lid_mush_base(mush, subgrid)
    else:
        assert check_lid_mush_base(mush, subgrid) is None


@settings(max_examples=120, deadline=None)
@given(st.booleans(), st.booleans(), st.integers(min_value=0, max_value=6))
def test_outer_boundary_rule_of_the_lid(subgrid, mush, bc):
    """The lid runs on the grey body, and on the prescribed flux only with the mush base."""
    from aragog.config.boundary import check_subgrid_lid

    allowed = not subgrid or bc == 1 or (bc == 4 and mush)
    args = dict(GOOD_LID, outer_boundary_condition=bc, lid_mush_base=mush)
    if allowed:
        assert check_subgrid_lid(subgrid, **args) is None
    else:
        with pytest.raises(ValueError, match='outer_boundary_condition'):
            check_subgrid_lid(subgrid, **args)


FLUX = st.floats(min_value=-1e7, max_value=1e7, allow_nan=False)


@settings(max_examples=120, deadline=None)
@given(
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
    st.floats(min_value=0.0, max_value=1e4, allow_nan=False),
)
def test_floor_weight_does_not_fall_with_entropy(S, dS):
    assert sl.floor_weight(S + dS, 0.0) >= sl.floor_weight(S, 0.0)


@settings(max_examples=120, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=1e12, allow_nan=False),
    st.floats(min_value=1.0, max_value=1e5, allow_nan=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    FLUX,
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=1e7, allow_nan=False),
)
def test_base_flux_lies_between_its_two_limits(Phi, h, frac, F_lo, g, q_conv):
    delta = frac * h
    F_k = sl.floor_flux(Phi, h, delta)
    assert F_k >= 0.0
    F_b = sl.mush_base_flux(Phi, h, delta, F_lo, g, q_conv)
    lo, hi = sorted((F_lo, F_k + q_conv))
    assert lo - 1e-9 * (abs(lo) + 1.0) <= F_b <= hi + 1e-9 * (abs(hi) + 1.0)
    assert sl.mush_base_flux(Phi, h, delta, F_lo, 0.0, q_conv) == F_lo
    assert sl.mush_base_flux(Phi, h, delta, F_lo, 1.0, q_conv) == pytest.approx(
        F_k + q_conv, rel=1e-12, abs=1e-6
    )


@needs_eos
class TestRestore:
    @pytest.fixture(scope='class')
    def solver(self):
        return _solver(entropy_eos_copy(EOS_DIR))

    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        st.floats(min_value=1.0, max_value=1e4),
        st.floats(min_value=1.0e3, max_value=1e12),
        st.floats(min_value=0.0, max_value=2000.0),
    )
    def test_a_resume_above_the_floor_keeps_the_lid_bitwise(self, solver, delta, E, dS):
        S_top = solver._lid_S_sol + sl.LID_FLOOR_GAP + dS
        solver.set_initial_lid(delta, E)
        try:
            assert np.array_equal(solver._lid_start(S_top), [delta, E])
        finally:
            solver.set_initial_lid(None)

    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        st.floats(min_value=1.0e-6, max_value=500.0), st.floats(min_value=1.0, max_value=1e4)
    )
    def test_a_resume_with_a_live_lid_below_the_floor_is_refused(self, solver, dS, delta):
        solver.set_initial_lid(delta, 1.0e6)
        try:
            with pytest.raises(ValueError, match='below the floor'):
                solver._lid_start(solver._lid_S_sol + sl.LID_FLOOR_GAP - dS)
        finally:
            solver.set_initial_lid(None)

    def test_a_handed_over_state_may_resume_below_the_floor(self, solver):
        solver.set_initial_lid(0.0, 0.0)
        try:
            start = solver._lid_start(solver._lid_S_sol - 50.0)
        finally:
            solver.set_initial_lid(None)
        assert np.array_equal(start, [0.0, 0.0])


def _flux_lid_run(hv, T_b, k, delta0, E0, F_atm, F_b, t_yr, H=None, w_of=None):
    """Lid states of the flux mode against time at a fixed flux from the atmosphere and a fixed
    delivery ``F_b``: the rates of :func:`aragog.surface_lid.lid_rates` in a stiff integrator."""
    from scipy.integrate import solve_ivp

    H_f, H_b, rho = H or (3.5e9, 0.0, 3000.0)
    floor = sl.LID_DELTA_FLOOR

    def rhs(_t, y):
        surf = sl.flux_surface_state(y[0], y[1], T_b, k, F_atm, floor, hv)
        r = sl.lid_rates(
            delta=y[0],
            E_lid=y[1],
            surf=surf,
            F_b=F_b,
            F_grid=F_atm,
            w=w_of(F_b, surf),
            rho=rho,
            H_f=H_f,
            T_b=T_b,
            H_b=H_b,
            hv=hv,
            h_top=1.0e6,
            delta_lo=0.5,
            heating=0.0,
        )
        return [r.ddelta, r.dE]

    yr = 3.15576e7
    sol = solve_ivp(
        rhs, (0.0, t_yr * yr), [delta0, E0], method='Radau', rtol=1e-7, atol=[1e-3, 1.0]
    )
    assert sol.status == 0
    y = sol.y[:, -1]
    return y[0], y[1], sl.flux_surface_state(y[0], y[1], T_b, k, F_atm, floor, hv)


@needs_eos
class TestFluxLidKnownAnswers:
    @pytest.fixture(scope='class')
    def s(self):
        return _solver(entropy_eos_copy(EOS_DIR))

    @staticmethod
    def _w(s):
        return lambda F_b, surf: sl.flux_weight(
            F_b, s._lid_T_sol, surf.T_s, s._lid_k, s._lid_delta_on
        )

    @pytest.mark.parametrize('F, delta0', [(1.0, 1000.0), (0.3, 3000.0), (5.0, 100.0)])
    def test_steady_crust_keeps_delta_and_the_conduction_law(self, s, F, delta0):
        """F_atm = F_b: delta holds and ``delta = k (T_b - T_s) / F``."""
        T_b, k, hv = s._lid_T_sol, s._lid_k, s._lid_hv
        T_s = T_b - F * delta0 / k
        E0 = delta0 * float(hv.profile_heat(0.0, T_s, T_b))
        delta, _E, surf = _flux_lid_run(hv, T_b, k, delta0, E0, F, F, 1.0e5, w_of=self._w(s))
        assert delta == pytest.approx(delta0, rel=1e-3)
        assert surf.T_s == pytest.approx(T_s, rel=1e-2)
        assert k * (T_b - surf.T_s) / F == pytest.approx(delta, rel=1e-2)

    def test_a_crust_off_the_linear_profile_relaxes_to_the_conduction_law(self, s):
        T_b, k, hv = s._lid_T_sol, s._lid_k, s._lid_hv
        F, delta0 = 5.0, 100.0
        E0 = delta0 * float(hv.profile_heat(1.0, T_b - 2.0 * F * delta0 / k, T_b))
        delta, _E, surf = _flux_lid_run(hv, T_b, k, delta0, E0, F, F, 1.0e5, w_of=self._w(s))
        assert surf.s == pytest.approx(0.0, abs=1e-2)
        assert k * (T_b - surf.T_s) / F == pytest.approx(delta, rel=1e-2)

    def test_growth_above_the_delivery_runs_at_the_stefan_rate(self, s):
        T_b, k, hv = s._lid_T_sol, s._lid_k, s._lid_hv
        F, F_b, delta0 = 1.5, 1.0, 1000.0
        H_f, H_b, rho = 3.5e9, 0.0, 3000.0
        E0 = delta0 * float(hv.profile_heat(0.0, T_b - F * delta0 / k, T_b))
        t = 1.0e3
        delta, _E, surf = _flux_lid_run(hv, T_b, k, delta0, E0, F, F_b, t, w_of=self._w(s))
        want = (surf.q_b - F_b) / (H_f - H_b) * t * 3.15576e7
        assert want > 1.0
        assert delta - delta0 == pytest.approx(want, rel=0.05)

    @pytest.mark.parametrize('F_atm', [0.0, -5.0, -1e4])
    def test_no_surface_flux_leaves_no_crust_above_the_base_temperature(self, s, F_atm):
        T_b, k, hv = s._lid_T_sol, s._lid_k, s._lid_hv
        surf = sl.flux_surface_state(100.0, 1.0e12, T_b, k, F_atm, sl.LID_DELTA_FLOOR, hv)
        assert surf.T_s == T_b
        assert surf.F_s == 0.0
        assert self._w(s)(10.0, surf) == 0.0

    def test_weight_turns_on_above_the_thickness_of_the_steady_crust(self, s):
        k, T_b, on = s._lid_k, s._lid_T_sol, s._lid_delta_on
        F_b = k * 100.0 / on
        assert sl.flux_weight(F_b, T_b, T_b - 100.0, k, on) == pytest.approx(0.5, abs=1e-14)
        assert sl.flux_weight(F_b, T_b, T_b - 1000.0, k, on) > 0.99999
        assert sl.flux_weight(F_b, T_b, T_b - 1.0, k, on) < 1.0e-6
        assert sl.flux_weight(0.0, T_b, T_b - 1.0, k, on) == 1.0
        assert sl.flux_weight(-3.0, T_b, T_b, k, on) == 0.0
