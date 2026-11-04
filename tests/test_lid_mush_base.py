"""Tests of the mush base of the sub-grid lid: base at the solidus, floor flux, top-cell floor."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog import surface_lid as sl
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
        assert ev.F_b == F_in

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
                50.0,
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
