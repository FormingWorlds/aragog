"""Tests of the mush base of the sub-grid lid: base at the solidus, floor flux, top-cell floor."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.optimize import brentq

from aragog import surface_lid as sl
from aragog.solver.entropy_solver import SECS_PER_YEAR
from tests.conftest import entropy_eos_copy
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, _build, needs_eos

# Pure-function and config tests are unit; tests that build a solver from the EOS tables are smoke.
pytestmark = pytest.mark.timeout(900)


def _solver(eos, S0=3050.0, end=1.0, mush=True, rheo=None, n_nodes=10, kappah_floor=None, **bc):
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(core_bc='quasi_steady', outer_bc=1, inner_bc=2, n_nodes=n_nodes, end_time=end)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.lid_mush_base = mush
    for name, value in bc.items():
        setattr(p.boundary_conditions, name, value)
    p.boundary_conditions.normalize()
    if kappah_floor is not None:
        p.energy.kappah_floor = kappah_floor
    p.phase_solid.rheology = rheo or SolidRheologyParams(
        enabled=True, stress_closure_mode='local', lid_base_mode='fixed'
    )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(S0)
    return s


@pytest.mark.unit
def test_the_hand_over_lies_below_the_growth_stop():
    """The mush-base hand-over fraction lies below the growth stop of the lid."""
    assert sl.LID_HANDOVER_FRAC < sl.LID_CAP_FRAC


@pytest.mark.unit
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


@pytest.mark.unit
def test_floor_weight_rises_from_1_to_6_j_per_kg_k_above_the_solidus():
    """The documented ramp: 0 up to S_sol + 1 J/kg/K, 0.5 at S_sol + 3.5, 1 from S_sol + 6."""
    S_sol = 2000.0
    assert sl.floor_weight(S_sol + 1.0, S_sol) == 0.0
    assert sl.floor_weight(S_sol + 1.05, S_sol) > 0.0
    assert sl.floor_weight(S_sol + 3.5, S_sol) == pytest.approx(0.5, abs=1e-14)
    assert sl.floor_weight(S_sol + 2.25, S_sol) == pytest.approx(
        0.103515625, abs=1e-14
    )  # x = 0.25
    assert sl.floor_weight(S_sol + 5.95, S_sol) < 1.0
    assert sl.floor_weight(S_sol + 6.0, S_sol) == 1.0


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
def test_floor_weight_stays_in_range(x):
    """The floor weight lies in [0, 1] for any entropy offset from the solidus."""
    assert 0.0 <= sl.floor_weight(x, 0.0) <= 1.0


@pytest.mark.unit
def test_floor_flux_constant_diffusivity_hand_value():
    """rho T kappa (S_f - S_sol) / (h_f / 2) for a constant integrand: Phi is its integral."""
    Phi = 3000.0 * 1500.0 * 2.0 * 7.0  # rho T kappa dS
    h, delta = 2000.0, 400.0
    want = 3000.0 * 1500.0 * 2.0 * 7.0 / (0.5 * (h - delta))
    assert sl.floor_flux(Phi, h, delta) == pytest.approx(want, rel=1e-14)
    # a lid that fills the cell keeps a fluid width of 1 mm, so the flux stays finite
    for d in (h + 5.0, h):
        assert sl.floor_flux(Phi, h, d) == pytest.approx(Phi / (0.5 * 1.0e-3), rel=1e-14)


@pytest.mark.unit
@pytest.mark.parametrize(
    'field, value',
    [('lid_mush_base', 1), ('lid_mush_base', 'yes'), ('lid_mush_base', None)],
)
def test_lid_mush_base_must_be_bool(field, value):
    """A lid_mush_base of 1, 'yes' or None is refused with a ValueError naming the field."""
    from aragog.config.boundary import check_lid_mush_base

    with pytest.raises(ValueError, match=field):
        check_lid_mush_base(value, True)


@pytest.mark.unit
def test_lid_mush_base_is_off_by_default():
    """The lid_mush_base field of the boundary-condition parameters defaults to False."""
    from aragog.parser import _BoundaryConditionsParameters

    assert _BoundaryConditionsParameters.__dataclass_fields__['lid_mush_base'].default is False


@pytest.mark.smoke
@needs_eos
class TestSolver:
    @pytest.fixture(scope='class')
    def eos(self):
        return entropy_eos_copy(EOS_DIR)

    def test_refused_without_the_rheology(self, eos):
        """Setup with the mush base and the solid rheology disabled raises a ValueError."""
        from aragog.parser import SolidRheologyParams

        with pytest.raises(ValueError, match='lid_mush_base'):
            _solver(eos, rheo=SolidRheologyParams(enabled=False))

    def test_refused_at_a_start_below_the_floor(self, eos):
        """A start 10 J/kg/K below the solidus entropy is refused, naming lid_mush_base."""
        s = _solver(eos, S0=3050.0)
        S_cold = s._lid_S_sol - 10.0
        with pytest.raises(ValueError, match='lid_mush_base'):
            s.set_initial_entropy(S_cold)
            s.solve()

    def test_base_is_the_solidus(self, eos):
        """The lid base state takes the solidus entropy and the solidus temperature."""
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

    @pytest.mark.parametrize('dS', [-1.0, 0.0, 0.49 * sl.LID_FLOOR_GAP])
    def test_invariant_refuses_an_accepted_state_below_the_floor(self, eos, dS):
        """The step check raises 'below the floor' for a top cell under solidus plus half the gap."""
        s = _solver(eos)
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[-2:] = s._lid_start(3050.0)
        y[n - 1] = s._lid_S_sol + dS
        s._dSdt_single(0.0, y)
        with pytest.raises(RuntimeError, match='below the floor'):
            s._check_lid_step(s._lid_delta_lo, 3, 1.0)

    def test_invariant_accepts_a_state_at_the_slack(self, eos):
        """The step check accepts a top cell 0.51 floor gaps above the solidus."""
        s = _solver(eos)
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[-2:] = s._lid_start(3050.0)
        y[n - 1] = s._lid_S_sol + 0.51 * sl.LID_FLOOR_GAP
        s._dSdt_single(0.0, y)
        s._check_lid_step(s._lid_delta_lo, 3, 1.0)

    def test_the_solve_watches_the_hand_over_without_the_lid_closure(self, eos, monkeypatch):
        """With the 'local' stress closure the mush base still installs the hand-over root."""
        s = _solver(eos)
        assert not s._lid_handover_armed()
        calls = []
        root = s._lid_root_value
        monkeypatch.setattr(s, '_lid_root_value', lambda t, y: calls.append(t) or root(t, y))
        s.solve()
        assert s._solution.status == 0
        assert len(calls) > 0

    def test_the_onset_thickness_must_lie_below_the_hand_over(self, eos):
        """With the mush base an onset thickness at or above 0.7 of the top cell is refused at
        setup with the hand-over named, so that no phantom lid hands over at the start."""
        h = _solver(eos)._lid_h_top
        for frac in (0.75, 0.7):
            with pytest.raises(
                ValueError, match=r'below the hand-over at 0\.7 x the top-cell width'
            ):
                _solver(eos, lid_delta_on=frac * h)
        s = _solver(eos, lid_delta_on=0.69 * h)
        assert s._lid_delta_on == 0.69 * h and s._lid_delta_lo == pytest.approx(
            0.345 * h, rel=1e-14
        )
        assert s._lid_delta_lo < sl.LID_HANDOVER_FRAC * h

    @pytest.mark.parametrize('floor', [0.0, -1.0])
    def test_the_mush_base_needs_a_floor(self, eos, floor):
        """The base flux of the mush base is the floor flux, so a floor of 0 or below is
        refused at setup with the setting named; a positive floor is used."""
        with pytest.raises(ValueError, match='lid_mush_base needs kappah_floor > 0'):
            _solver(eos, kappah_floor=floor)
        assert _solver(eos, kappah_floor=1.0).parameters.energy.kappah_floor == 1.0

    def test_the_mush_base_is_refused_in_the_constant_property_mode(self):
        """The constant-property column has no solidus, so the mush base is refused by name."""
        from aragog.solver.entropy_solver import EntropySolver
        from tests.test_subgrid_lid_const import _params

        p = _params()
        p.boundary_conditions.lid_mush_base = True
        p.boundary_conditions.normalize()
        with pytest.raises(ValueError, match='lid_mush_base needs the entropy EOS tables'):
            EntropySolver(p, entropy_eos=None).initialize()

    def test_hand_over_at_the_fraction_of_the_cell(self, eos):
        """The hand-over root changes sign at 0.7 of the top-cell width and is -1 at zero thickness."""
        s = _solver(eos)
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[-2:] = s._lid_start(3050.0)
        y[-2] = 0.7 * s._lid_h_top * (1.0 - 1e-6)
        assert s._lid_root_value(0.0, y) < 0.0
        y[-2] = 0.7 * s._lid_h_top * (1.0 + 1e-6)
        assert s._lid_root_value(0.0, y) > 0.0
        y[-2] = 0.0
        assert s._lid_root_value(0.0, y) == -1.0

    def test_short_run_closes_energy_and_keeps_the_floor(self, eos):
        """A short mush-base run closes its energy budget and its top cell stays above the
        solidus; the hold at the floor itself is tested by the blend tests at g = 0."""
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
        """The JAX mush base flux and the JAX lid (surface loss, fluid flux and volume, rates,
        ledger terms) equal NumPy at four top-cell entropies above the solidus."""
        from aragog.jax import surface_lid as jl
        from tests.test_subgrid_lid_jax_rhs import lid_top_against_numpy

        s = _solver(eos)
        lp = s.jax_lid_params()
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
            y_full = s._S0.copy()
            y_full[: s._n_stag] = np.linspace(3050.0, S_f, s._n_stag)
            y_full[-2:] = y
            got, want = lid_top_against_numpy(s, y_full)
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-30)
        # an onset near the hand-over gives a partial weight under a partial floor: the blend acts
        s = _solver(eos, lid_delta_on=0.65 * s._lid_h_top, lid_delta_lo=0.3 * s._lid_h_top)
        S_f = s._lid_S_sol + 4.0
        y_full = s._S0.copy()
        y_full[: s._n_stag] = np.linspace(3050.0, S_f, s._n_stag)
        y_full[-2:] = (3.0, 3.0 * float(s._lid_hv.profile_heat(0.0, 400.0, s._lid_T_sol)))
        got, want = lid_top_against_numpy(s, y_full)
        g, w = sl.floor_weight(S_f, s._lid_S_sol), float(s._lid_last.rates.w)
        assert 0.1 < g < 0.9 and 1.0 - g < w < 0.99
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-30)


@pytest.mark.smoke
@needs_eos
def test_an_intermediate_state_below_the_floor_fails_the_jax_batch_path():
    """The JAX batch path refuses an accepted state in the middle of the trace whose top cell
    lies below the slack of the floor, as the NumPy loop refuses it at every node."""
    from aragog.solver import entropy_solver as es
    from tests.test_subgrid_lid_jax_rhs import _factory_rhs

    if not es._CVODE_AVAILABLE:
        pytest.skip('scikits.odes is not installed')
    s = _solver(entropy_eos_copy(EOS_DIR), end=0.1)
    s.parameters.energy.solver_method = 'cvode'
    s.parameters.energy.use_jax_jacobian = True
    s.set_jax_cvode_factory(lambda scales, mode: _factory_rhs(s, s.jax_lid_params())[:2])
    s.solve()
    sol = s._solution
    t, y = (np.array(v, dtype=float) for v in (sol.get('energy_trace') or (sol.t, sol.y)))
    k, i = t.size // 2, s._n_stag - 1
    assert t.size >= 3 and y[-2, k] > 0.0
    target = s._lid_S_sol + 0.25 * sl.LID_FLOOR_GAP
    y[i, k] -= s._stag_entropy(y[:, k])[-1] - target
    assert s._stag_entropy(y[:, k])[-1] == pytest.approx(target, abs=1e-9)
    sol.energy_trace = (t, y)
    with pytest.raises(RuntimeError, match=f'at step {k},.*below the floor'):
        s._compute_step_energy_integrals()


@pytest.mark.unit
def test_flux_surface_state_carries_the_prescribed_flux():
    """A lid under a prescribed flux: T_s from the conduction, F_s = F_atm; clipped at 1 K and
    at 1 K below the base."""
    hv = sl.LidEnthalpy.linear(3000.0, 1200.0, 1400.0, 0.0)
    delta, F_atm = 10.0, 60.0
    E = delta * hv.profile_heat(0.0, 1400.0 - F_atm * delta / 4.0, 1400.0)
    surf = sl.flux_surface_state(delta, E, 1400.0, 4.0, F_atm, sl.LID_DELTA_FLOOR, hv)
    assert surf.F_s == pytest.approx(F_atm, rel=1e-6)
    assert surf.T_s == pytest.approx(1400.0 - F_atm * delta / 4.0, rel=1e-6)
    hot = sl.flux_surface_state(delta, E, 1400.0, 4.0, 1e6, sl.LID_DELTA_FLOOR, hv)
    assert hot.T_s == 1.0
    assert hot.F_s < 1e6
    for F_cold in (0.0, -5.0):
        assert (
            sl.flux_surface_state(delta, E, 1400.0, 4.0, F_cold, sl.LID_DELTA_FLOOR, hv).T_s
            == 1399.0
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    'bc, mush, ok', [(4, True, True), (4, False, False), (1, True, True), (2, True, False)]
)
def test_prescribed_flux_needs_the_mush_base(bc, mush, ok):
    """Outer boundary 4 needs lid_mush_base; boundary 1 passes and boundary 2 is refused."""
    from aragog.config.boundary import check_subgrid_lid

    args = (True, 1.0, None, 3, 450.0, bc, 'quasi_steady', False)
    if ok:
        check_subgrid_lid(*args, lid_mush_base=mush)
    else:
        with pytest.raises(ValueError, match='outer'):
            check_subgrid_lid(*args, lid_mush_base=mush)


@pytest.mark.smoke
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

    @pytest.mark.parametrize('F_atm', [0.0, -50.0])
    def test_the_phantom_start_keeps_a_drop_below_the_base(self, eos, F_atm):
        """Under a prescribed flux at or below 0 the phantom lid starts with its surface
        ``LID_DT_MIN`` below the base, as the flux state clamps it, not at or above the base."""
        s = self._s(eos, F_atm=F_atm)
        d, E = s._lid_start(3050.0)
        T_b = s._lid_T_sol
        assert d == s._lid_delta_lo
        assert E == pytest.approx(
            d * s._lid_hv.profile_heat(0.0, T_b - sl.LID_DT_MIN, T_b), rel=1e-14
        )

    def test_surface_loss_is_the_prescribed_flux(self, eos):
        """With a prescribed flux of 200 the top heat flux is 200 and the crust is below the base."""
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

    def test_short_run_closes_energy_and_reports_the_crust_surface(self, eos, monkeypatch):
        """A short flux-mode run closes its energy budget and reports T_magma and the crust surface."""
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
            s._lid_k * max(sl.shape(ev.surf.s)[0], 0.1) / ev.delta, rel=1e-12
        )
        # the surface slope of the conductance is at least 0.1
        rhs = s._dSdt_single
        for slope in (0.15, 0.05):
            v = brentq(lambda x: sl.shape(x)[0] - slope, sl.LID_SHAPE_MIN, -1.0)

            def thinning(t, y, v=v):
                out = rhs(t, y)
                surf = dataclasses.replace(s._lid_last.surf, s=v)
                s._lid_last = dataclasses.replace(s._lid_last, surf=surf)
                return out

            monkeypatch.setattr(s, '_dSdt_single', thinning)
            G = s.get_state().lid_conductance
            assert G == pytest.approx(s._lid_k * max(slope, 0.1) / ev.delta, rel=1e-9)
        # the blended T_magma at a partial and at a full weight, where it differs from T_node
        for w in (0.37, 1.0):

            def weighted(t, y, w=w):
                out = rhs(t, y)
                rates = dataclasses.replace(s._lid_last.rates, w=w)
                s._lid_last = dataclasses.replace(s._lid_last, rates=rates)
                return out

            monkeypatch.setattr(s, '_dSdt_single', weighted)
            T_w = s.get_state().T_magma
            T_node = float(np.asarray(s.state.phase_basic.temperature()).ravel()[-1])
            assert T_w == pytest.approx(w * s._lid_last.T_b + (1.0 - w) * T_node, rel=1e-12)
            assert abs(T_w - T_node) > 1.0

    def test_jax_flux_mode_matches_numpy(self, eos):
        """The JAX flux-mode surface state and weight match NumPy, including a flux at or below 0."""
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
            assert float(
                jl.flux_weight(F_b, T_b, T_s, s._lid_k, 1.0, 255.0, on)
            ) == pytest.approx(
                sl.flux_weight(F_b, T_b, T_s, s._lid_k, 1.0, 255.0, on), rel=1e-12, abs=1e-14
            )
        for F_atm in (-5.0, 0.0):
            surf_j = jl.surface_state(
                y[0], y[1], T_b, s._lid_k, 1.0, 255.0, sl.LID_DELTA_FLOOR, lp.e, lp.flux, F_atm
            )
            assert float(surf_j[0]) == T_b - sl.LID_DT_MIN

    def test_a_lid_at_the_limit_of_its_profile_is_reported(self, eos, caplog):
        """A lid under a flux of 1e4 reaches the minimum surface temperature and logs a warning."""
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
        assert 'at a limit of its profile' in caplog.text


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


@pytest.mark.unit
@settings(max_examples=80, deadline=None)
@given(NOT_BOOL, st.booleans())
def test_lid_mush_base_refuses_every_other_type(value, subgrid):
    """Any non-bool lid_mush_base is refused with 'must be true or false'."""
    from aragog.config.boundary import check_lid_mush_base

    with pytest.raises(ValueError, match='lid_mush_base must be true or false'):
        check_lid_mush_base(value, subgrid)


@pytest.mark.unit
@settings(max_examples=40, deadline=None)
@given(st.booleans(), st.booleans())
def test_lid_mush_base_needs_the_lid(mush, subgrid):
    """lid_mush_base true without subgrid_lid is refused; other combinations return None."""
    from aragog.config.boundary import check_lid_mush_base

    if mush and not subgrid:
        with pytest.raises(ValueError, match='needs subgrid_lid'):
            check_lid_mush_base(mush, subgrid)
    else:
        assert check_lid_mush_base(mush, subgrid) is None


@pytest.mark.unit
@settings(max_examples=120, deadline=None)
@given(st.booleans(), st.booleans(), st.integers(min_value=0, max_value=6))
def test_outer_boundary_rule_of_the_lid(subgrid, mush, bc):
    """The lid runs on the grey body and the fixed surface temperature, and on the prescribed
    flux only with the mush base."""
    from aragog.config.boundary import check_subgrid_lid

    allowed = not subgrid or bc in (1, 5) or (bc == 4 and mush)
    args = dict(GOOD_LID, outer_boundary_condition=bc, lid_mush_base=mush)
    if allowed:
        assert check_subgrid_lid(subgrid, **args) is None
    else:
        with pytest.raises(ValueError, match='outer_boundary_condition'):
            check_subgrid_lid(subgrid, **args)


FLUX = st.floats(min_value=-1e7, max_value=1e7, allow_nan=False)


@pytest.mark.unit
@settings(max_examples=120, deadline=None)
@given(
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
    st.floats(min_value=0.0, max_value=1e4, allow_nan=False),
)
def test_floor_weight_does_not_fall_with_entropy(S, dS):
    """The floor weight does not decrease when the entropy rises."""
    assert sl.floor_weight(S + dS, 0.0) >= sl.floor_weight(S, 0.0)


@pytest.mark.unit
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
    """The base flux lies between the flux from below (g = 0) and the floor flux plus convection (g = 1)."""
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


@pytest.mark.smoke
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
        """A set lid under a top cell above the floor gap starts unchanged, bit for bit."""
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
        """A set lid over a top cell below the slack of the step check is refused."""
        solver.set_initial_lid(delta, 1.0e6)
        try:
            with pytest.raises(ValueError, match='below the floor'):
                solver._lid_start(solver._lid_S_sol + 0.5 * sl.LID_FLOOR_GAP - dS)
        finally:
            solver.set_initial_lid(None)

    def test_a_kept_lid_starts_where_the_step_check_accepts_it(self, solver):
        """A set or carried lid passed the step check, which accepts a top cell down to half
        the floor gap; there it starts unchanged, while the phantom start still needs the gap."""
        S_top = solver._lid_S_sol + 0.75 * sl.LID_FLOOR_GAP
        solver.set_initial_lid(2.0, 1.0e6)
        try:
            assert np.array_equal(solver._lid_start(S_top), [2.0, 1.0e6])
        finally:
            solver.set_initial_lid(None)
        with pytest.raises(ValueError, match='below the floor'):
            solver._lid_start(S_top)
        run = _solver(entropy_eos_copy(EOS_DIR), end=0.1)
        run.solve()
        carried = run._previous_lid()
        assert carried is not None and carried[0] > 0.0
        assert np.array_equal(run._lid_start(S_top), carried)

    def test_a_handed_over_state_may_resume_below_the_floor(self, solver):
        """A handed-over lid (0, 0) restarts unchanged with a top cell below the solidus."""
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


@pytest.mark.smoke
@needs_eos
class TestFluxLidKnownAnswers:
    @pytest.fixture(scope='class')
    def s(self):
        return _solver(entropy_eos_copy(EOS_DIR))

    @staticmethod
    def _w(s):
        return lambda F_b, surf: sl.flux_weight(
            F_b, s._lid_T_sol, surf.T_s, s._lid_k, 1.0, 255.0, s._lid_delta_on
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
        """A crust off the linear profile relaxes to zero surface slope and the conduction law."""
        T_b, k, hv = s._lid_T_sol, s._lid_k, s._lid_hv
        F, delta0 = 5.0, 100.0
        E0 = delta0 * float(hv.profile_heat(1.0, T_b - 2.0 * F * delta0 / k, T_b))
        delta, _E, surf = _flux_lid_run(hv, T_b, k, delta0, E0, F, F, 1.0e5, w_of=self._w(s))
        assert surf.s == pytest.approx(0.0, abs=1e-2)
        assert k * (T_b - surf.T_s) / F == pytest.approx(delta, rel=1e-2)

    def test_growth_above_the_delivery_runs_at_the_stefan_rate(self, s):
        """With a surface flux above the delivery, the lid grows at the Stefan rate within 5 percent."""
        T_b, k, hv = s._lid_T_sol, s._lid_k, s._lid_hv
        F, F_b, delta0 = 1.5, 1.0, 1000.0
        H_f, H_b = 3.5e9, 0.0
        E0 = delta0 * float(hv.profile_heat(0.0, T_b - F * delta0 / k, T_b))
        t = 1.0e3
        delta, _E, surf = _flux_lid_run(hv, T_b, k, delta0, E0, F, F_b, t, w_of=self._w(s))
        want = (surf.q_b - F_b) / (H_f - H_b) * t * 3.15576e7
        assert want > 1.0
        assert delta - delta0 == pytest.approx(want, rel=0.05)

    @pytest.mark.parametrize('F_atm', [0.0, -5.0, -1e4])
    def test_no_surface_flux_leaves_no_crust_above_the_base_temperature(self, s, F_atm):
        """A surface flux of 0 or below keeps the surface 1 K below the base; a surface at the base has weight 0."""
        T_b, k, hv = s._lid_T_sol, s._lid_k, s._lid_hv
        surf = sl.flux_surface_state(100.0, 1.0e12, T_b, k, F_atm, sl.LID_DELTA_FLOOR, hv)
        assert surf.T_s == T_b - sl.LID_DT_MIN
        assert sl.flux_weight(10.0, T_b, T_b, k, 1.0, 255.0, s._lid_delta_on) == 0.0

    def test_weight_does_not_read_the_flux_of_the_atmosphere(self, s):
        """The surface temperature matters only at or above ``T_b``; below it the weight is the
        grey body weight, so a steady crust has ``w = 1`` at any surface flux."""
        k, T_b, on = s._lid_k, s._lid_T_sol, s._lid_delta_on
        grey = sl.existence_weight(50.0, T_b, k, 1.0, 255.0, on)
        for T_s in (1.0, 300.0, T_b - 1.0e-6):
            assert sl.flux_weight(50.0, T_b, T_s, k, 1.0, 255.0, on) == grey
        assert grey > 0.99999
        assert sl.flux_weight(50.0, T_b, T_b, k, 1.0, 255.0, on) == 0.0
        assert sl.flux_weight(1.0e7, T_b, 300.0, k, 1.0, 255.0, on) == 0.0
        assert sl.flux_weight(-3.0, T_b, T_b - 1.0, k, 1.0, 255.0, on) == 1.0


def _mush_states(s):
    """Top cells across the floor band and above it, with lids of 1 m to 1 km and a hand-over."""
    n = s._n_stag
    T_b = s._lid_T_sol
    for dS, d, T_s in (
        (sl.LID_FLOOR_GAP, 3.0, 1800.0),
        (sl.LID_FLOOR_GAP + 0.3 * sl.LID_FLOOR_WIDTH, 30.0, 1500.0),
        (sl.LID_FLOOR_GAP + 0.8 * sl.LID_FLOOR_WIDTH, 1000.0, 900.0),
        (40.0, 300.0, 1200.0),
        (700.0, 1.0, 2000.0),
        (700.0, 0.0, T_b),
    ):
        y = s._S0.copy()
        y[:n] = np.linspace(3050.0, 3000.0, n)
        y[n - 1] = s._lid_S_sol + dS
        y[-2:] = (d, d * float(s._lid_hv.profile_heat(0.0, T_s, T_b)))
        yield y


@pytest.mark.smoke
@needs_eos
@pytest.mark.parametrize('outer', [1, 4])
def test_jax_factory_rhs_matches_on_the_mush_base(outer):
    """The factory that PROTEUS uses against the NumPy right-hand side, grey body and flux."""
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver
    from tests.test_subgrid_lid_jax_rhs import _factory_rhs, _physical

    p = _build(
        core_bc='quasi_steady',
        outer_bc=outer,
        inner_bc=2,
        n_nodes=10,
        end_time=1.0,
        **({'outer_bc_value': 200.0} if outer == 4 else {}),
    )
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.lid_mush_base = True
    p.boundary_conditions.normalize()
    p.phase_solid.rheology = SolidRheologyParams(
        enabled=True, stress_closure_mode='local', lid_base_mode='fixed'
    )
    s = EntropySolver(p, entropy_eos=entropy_eos_copy(EOS_DIR))
    s.initialize()
    s.set_initial_entropy(3050.0)
    rhs_fn, _, info = _factory_rhs(s, s.jax_lid_params())
    assert info['lid'] and rhs_fn.lid
    f, n = _physical(rhs_fn, s._build_nondim_scales()), 0
    top = s._n_stag - 1
    for y in _mush_states(s):
        want = np.asarray(s._dSdt_single(0.0, y), dtype=float)
        got = f(y)
        n += 1
        np.testing.assert_allclose(got[-2:], want[-2:], rtol=1e-9, atol=1e-300)
        atol = 1e-12 * np.abs(want[: s._n_stag]).max()
        assert atol > 0.0
        np.testing.assert_allclose(got[top], want[top], rtol=1e-9, atol=atol)
        np.testing.assert_allclose(got[:top], want[:top], rtol=1e-12, atol=atol)
    assert info['rhs_calls'] == n


@pytest.mark.unit
@settings(max_examples=80, deadline=None)
@given(
    st.one_of(
        st.floats(max_value=0.0),
        st.just(math.nan),
        st.just(math.inf),
        st.booleans(),
        st.text(max_size=3),
    )
)
def test_lid_surface_temperature_refuses_every_other_value(value):
    """A lid_surface_temperature that is not a finite positive number is refused by name."""
    from aragog.config.boundary import check_lid_surface_temperature

    with pytest.raises(ValueError, match='lid_surface_temperature'):
        check_lid_surface_temperature(value, 4, True)


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(st.floats(min_value=1.0e-3, max_value=1.0e5), st.integers(0, 6), st.booleans())
def test_lid_surface_temperature_needs_the_flux_boundary_and_the_mush_base(T, bc, mush):
    """lid_surface_temperature is accepted only with outer boundary 4 and the mush base."""
    from aragog.config.boundary import check_lid_surface_temperature

    assert check_lid_surface_temperature(None, bc, mush) is None
    if bc == 4 and mush:
        assert check_lid_surface_temperature(T, bc, mush) is None
    else:
        with pytest.raises(ValueError, match='lid_surface_temperature'):
            check_lid_surface_temperature(T, bc, mush)


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(
    st.one_of(
        st.floats(max_value=0.0),
        st.just(float('nan')),
        st.just(float('inf')),
        st.booleans(),
        st.text(max_size=3),
    )
)
def test_lid_surface_flux_cap_refuses_every_other_value(value):
    """A lid_surface_flux_cap that is not a finite positive number is refused by name."""
    from aragog.config.boundary import check_lid_surface_flux_cap

    with pytest.raises(ValueError, match='lid_surface_flux_cap'):
        check_lid_surface_flux_cap(value, 4, True, 500.0)


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(
    st.floats(min_value=1.0e-3, max_value=1.0e7),
    st.integers(0, 6),
    st.booleans(),
    st.one_of(st.none(), st.just(500.0)),
)
def test_lid_surface_flux_cap_needs_the_held_surface(C, bc, mush, T_held):
    """lid_surface_flux_cap needs boundary 4, the mush base and a held surface temperature."""
    from aragog.config.boundary import check_lid_surface_flux_cap

    assert check_lid_surface_flux_cap(None, bc, mush, T_held) is None
    if bc == 4 and mush and T_held is not None:
        assert check_lid_surface_flux_cap(C, bc, mush, T_held) is None
    else:
        with pytest.raises(ValueError, match='lid_surface_flux_cap'):
            check_lid_surface_flux_cap(C, bc, mush, T_held)


def _held_solver(eos, T_held, F_atm=200.0, end=1.0, n_nodes=10, thin=False, cap=None):
    from aragog.parser import SolidRheologyParams
    from aragog.solver.entropy_solver import EntropySolver

    p = _build(
        core_bc='quasi_steady',
        outer_bc=4,
        inner_bc=2,
        n_nodes=n_nodes,
        end_time=end,
        outer_bc_value=F_atm,
    )
    if thin:  # a shell with a thin top cell, where the floor flux reaches a partial lid weight
        p.mesh.inner_radius = 6.346e6
        p.energy.convection = True
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.lid_mush_base = True
    p.boundary_conditions.lid_surface_temperature = T_held
    p.boundary_conditions.lid_surface_flux_cap = cap
    p.boundary_conditions.normalize()
    p.phase_solid.rheology = SolidRheologyParams(
        enabled=True, stress_closure_mode='local', lid_base_mode='fixed'
    )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(3050.0)
    return s


@pytest.mark.smoke
@needs_eos
class TestHeldSurface:
    @pytest.fixture(scope='class')
    def eos(self):
        return entropy_eos_copy(EOS_DIR)

    @staticmethod
    def _state(s, delta, T_s, dS=40.0):
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[n - 1] = s._lid_S_sol + dS
        y[-2:] = (delta, delta * float(s._lid_hv.profile_heat(0.0, T_s, s._lid_T_sol)))
        return y

    def test_surface_loss_is_the_conduction_to_the_held_surface(self, eos):
        """The surface flux does not read the flux value of the boundary."""
        out = []
        for F_atm in (200.0, 5.0e4):
            s = _held_solver(eos, 500.0, F_atm=F_atm)
            s._dSdt_single(0.0, self._state(s, 100.0, 500.0))
            ev = s._lid_last
            assert ev.rates.w == 1.0
            assert ev.surf.T_s == 500.0
            out.append(float(s.state._heat_flux[-1]))
        assert out[0] == out[1]
        assert out[0] == pytest.approx(s._lid_k * (s._lid_T_sol - 500.0) / 100.0, rel=1e-6)

    def test_a_cap_above_the_held_flux_leaves_the_held_state(self, eos):
        """Below the cap the lid is the held state, bit for bit."""
        y = None
        out = []
        for cap in (None, 1.0e9):
            s = _held_solver(eos, 500.0, cap=cap)
            y = self._state(s, 100.0, 500.0) if y is None else y
            dS = np.asarray(s._dSdt_single(0.0, y), dtype=float)
            out.append((dS, s._lid_last.surf.T_s, float(s.state._heat_flux[-1])))
        np.testing.assert_array_equal(out[0][0], out[1][0])
        assert out[0][1:] == out[1][1:] == (500.0, out[0][2])

    def test_a_cap_below_the_held_flux_sets_the_surface_flux(self, eos):
        """Above the cap the lid surface carries the cap, at a surface above the held one."""
        s = _held_solver(eos, 500.0)
        y = self._state(s, 100.0, 500.0)
        s._dSdt_single(0.0, y)
        F_held = float(s._lid_last.surf.F_s)
        C = 0.5 * F_held
        s = _held_solver(eos, 500.0, cap=C)
        s._dSdt_single(0.0, y)
        ev = s._lid_last
        assert ev.rates.w == 1.0
        assert ev.surf.F_s == pytest.approx(C, rel=1e-9)
        assert ev.surf.T_s > 500.0
        assert float(s.state._heat_flux[-1]) == pytest.approx(C, rel=1e-9)

    def test_change_of_the_cap_between_calls_is_read(self, eos):
        """A cap set between calls takes effect; a negative cap set later is refused at the solve."""
        s = _held_solver(eos, 500.0)
        y = self._state(s, 100.0, 500.0)
        s._dSdt_single(0.0, y)
        C = 0.25 * float(s._lid_last.surf.F_s)
        s.parameters.boundary_conditions.lid_surface_flux_cap = C
        s._dSdt_single(0.0, y)
        assert s._lid_last.surf.F_s == pytest.approx(C, rel=1e-9)
        s.parameters.boundary_conditions.lid_surface_flux_cap = -1.0
        with pytest.raises(ValueError, match='lid_surface_flux_cap'):
            s.solve()

    @pytest.mark.parametrize('frac', [0.5, 2.0])
    def test_jax_factory_matches_numpy_with_the_cap(self, eos, frac):
        """The JAX factory matches NumPy at caps of 0.5 and 2 times the held flux, Jacobian finite."""
        from tests.test_subgrid_lid_jax_rhs import _factory_rhs, _physical

        s = _held_solver(eos, 500.0)
        y0 = self._state(s, 100.0, 500.0)
        s._dSdt_single(0.0, y0)
        s = _held_solver(eos, 500.0, cap=frac * float(s._lid_last.surf.F_s))
        rhs_fn, jac, _ = _factory_rhs(s, s.jax_lid_params())
        sc = s._build_nondim_scales()
        f, top = _physical(rhs_fn, sc), s._n_stag - 1
        states = [y0, *_mush_states(s)]
        wants = [np.asarray(s._dSdt_single(0.0, y), dtype=float) for y in states]
        # y0 is the held steady lid: its dE cancels to round-off of terms near the largest dE
        atol_lid = 1e-12 * max(np.abs(w[-2:]).max() for w in wants)
        for y, want in zip(states, wants):
            s._dSdt_single(0.0, y)
            got = f(y)
            atol = 1e-12 * np.abs(want[: s._n_stag]).max()
            np.testing.assert_allclose(got[-2:], want[-2:], rtol=1e-9, atol=atol_lid)
            np.testing.assert_allclose(got[top], want[top], rtol=1e-9, atol=atol)
            x = (y - sc.state_offset) / sc.state_scale
            J = np.zeros((x.size, x.size))
            assert jac(0.0, x, np.zeros(x.size), J) == 0
            assert np.all(np.isfinite(J))

    def test_a_surface_at_or_above_the_base_hands_the_surface_to_the_flux(self, eos):
        """A held surface at or above the base falls back to the prescribed flux with weight 0."""
        s = _held_solver(eos, 1.0e4, F_atm=123.0)
        s._dSdt_single(0.0, self._state(s, 100.0, 500.0, dS=sl.LID_FLOOR_WIDTH * 3.0))
        ev = s._lid_last
        assert ev.surf.T_s == s._lid_T_sol - sl.LID_DT_MIN
        assert ev.rates.w == 0.0
        assert float(s.state._heat_flux[-1]) == 123.0

    def test_change_of_the_held_surface_between_calls_is_read(self, eos):
        """A held surface temperature set between calls is used by the next call."""
        s = _held_solver(eos, 500.0)
        y = self._state(s, 100.0, 500.0)
        s._dSdt_single(0.0, y)
        s.parameters.boundary_conditions.lid_surface_temperature = 800.0
        s._dSdt_single(0.0, y)
        assert s._lid_last.surf.T_s == 800.0

    def test_bad_value_set_between_calls_is_refused_at_the_solve(self, eos):
        """A negative held surface temperature set after setup is refused at the solve, by name."""
        s = _held_solver(eos, 500.0)
        s.parameters.boundary_conditions.lid_surface_temperature = -3.0
        with pytest.raises(ValueError, match='lid_surface_temperature'):
            s.solve()

    def test_jax_factory_matches_numpy_with_the_held_surface(self, eos):
        """The JAX factory matches NumPy for six mush states with a held surface at 500 K."""
        from tests.test_subgrid_lid_jax_rhs import _factory_rhs, _physical

        s = _held_solver(eos, 500.0)
        rhs_fn, _, info = _factory_rhs(s, s.jax_lid_params())
        f, top = _physical(rhs_fn, s._build_nondim_scales()), s._n_stag - 1
        for y in _mush_states(s):
            want = np.asarray(s._dSdt_single(0.0, y), dtype=float)
            got = f(y)
            atol = 1e-12 * np.abs(want[: s._n_stag]).max()
            np.testing.assert_allclose(got[-2:], want[-2:], rtol=1e-9, atol=1e-300)
            np.testing.assert_allclose(got[top], want[top], rtol=1e-9, atol=atol)
        assert info['rhs_calls'] == 6

    @pytest.mark.parametrize(
        'T_held, F_atm',
        [(500.0, 200.0), (1.0e4, 200.0), (None, -5.0), (None, 0.0), (None, 200.0)],
    )
    def test_jax_jacobian_is_finite_and_the_flux_goes_to_the_grid(self, eos, T_held, F_atm):
        """A surface at or above the base leaves the profile with no temperature drop."""
        from tests.test_subgrid_lid_jax_rhs import _factory_rhs

        s = _held_solver(eos, T_held, F_atm=F_atm)
        _, jac, _ = _factory_rhs(s, s.jax_lid_params())
        sc = s._build_nondim_scales()
        for y in _mush_states(s):
            x = (y - sc.state_offset) / sc.state_scale
            J = np.zeros((x.size, x.size))
            assert jac(0.0, x, np.zeros(x.size), J) == 0
            assert np.all(np.isfinite(J))

    @pytest.mark.parametrize('form', ['parallel', 'series'])
    def test_flux_of_the_interior_against_the_balance_of_the_atmosphere(self, eos, form):
        """At partial weight the interior sums the lid and the open part. A parallel balance of
        the atmosphere is that sum; a series balance differs by w (F_lid - F_atm)."""
        from scipy.optimize import brentq

        sig, T_eq, G_skin = 5.670374419e-8, 255.0, 200.0
        s = _held_solver(eos, 1000.0, F_atm=5.0e4, thin=True)
        n = s._n_stag
        y = np.empty(n + 2)
        y[:n] = np.asarray(eos.solidus_entropy(s._P_stag_flat), float).ravel() + 90.0
        y[-2:] = (0.6, 0.6 * float(s._lid_hv.profile_heat(0.0, 1000.0, s._lid_T_sol)))
        bc = s.parameters.boundary_conditions

        def balance(T_held):
            """Lid state and flux of the atmosphere balance at a held surface."""
            bc.lid_surface_temperature = T_held
            s._dSdt_single(0.0, y)
            ev = s._lid_last
            w, T_b = float(ev.rates.w), float(ev.T_b)
            G_lid = s._lid_k * max(sl.shape(ev.surf.s)[0], 0.1) / ev.delta
            T_node = float(np.asarray(s.state.phase_basic.temperature()).ravel()[-1])
            if form == 'parallel':
                F = w * G_lid * (T_b - T_held) + (1.0 - w) * G_skin * (T_node - T_held)
            else:
                T_melt = w * T_b + (1.0 - w) * T_node
                F = (T_melt - T_held) / (w / G_lid + (1.0 - w) / G_skin)
            return w, G_lid, T_b, T_node, F

        T = brentq(
            lambda v: sig * (v**4 - T_eq**4) - balance(v)[4], T_eq + 1.0, 1400.0, xtol=1e-9
        )
        w, G_lid, T_b, T_node, F_atm = balance(T)
        assert 0.1 < w < 0.9  # the balance sits at a partial weight
        assert F_atm == pytest.approx(sig * (T**4 - T_eq**4), rel=1e-6)
        bc.outer_boundary_value = G_skin * (T_node - T) if form == 'parallel' else F_atm
        s._dSdt_single(0.0, y)
        F_int = float(s.state._heat_flux[-1])
        if form == 'parallel':
            assert F_int == pytest.approx(F_atm, rel=1e-6)
        else:
            assert abs(F_int - F_atm) > 0.05 * F_atm
            assert F_int - F_atm == pytest.approx(w * (G_lid * (T_b - T) - F_atm), rel=1e-6)

    def test_coupled_budget_closes_and_reports_the_difference_from_the_atmosphere(self, eos):
        """Steps under a grey balance: the state heat of each call equals the integrated powers,
        and the mean lid flux differs from the flux of the atmosphere only by the crust change
        inside the step."""
        F_atm, T_held, y, t = None, None, None, 0.0
        sig, T_eq = 5.670374419e-8, 255.0
        k_s, d_s = 2.0, 0.01
        diffs = []
        for t_end in (
            20.0,
            22.0,
            24.2,
            26.6,
            29.3,
            32.2,
            35.4,
            38.9,
        ):  # ratio 1.1, as a coupling
            s = _held_solver(eos, T_held, F_atm=F_atm or 1.0e4, end=t_end)
            s.parameters.solver.start_time = t
            if y is None:
                s.set_initial_lid(
                    30.0, 30.0 * float(s._lid_hv.profile_heat(0.0, 500.0, s._lid_T_sol))
                )
                P = np.asarray(s._P_stag_flat, float)
                S = np.asarray(eos.solidus_entropy(P), float).ravel() + 40.0
            else:
                S = y[: s._n_stag]
                s.set_initial_lid(float(y[-2]), float(y[-1]))
            s.set_initial_entropy(S)
            s.solve()
            sol = s._solution
            assert sol.status == 0
            ei = sol.energy_integrals
            gap = ei['state_heat'] - (ei['F_int'] + ei['F_cmb'] + ei['Q_radio'] + ei['Q_tidal'])
            assert abs(gap) <= 1.0e-3 * abs(ei['F_int'])
            assert abs(ei['solver_residual']) <= 1.0e-10 * abs(ei['F_int'])
            y = np.asarray(sol.y[:, -1])
            st = s.get_state()
            F_mean = -ei['F_int'] / (s._lid_area * (t_end - t) * SECS_PER_YEAR)
            if F_atm is not None:
                diffs.append((F_mean - F_atm) / F_atm)
            w, G_lid = float(st.lid_weight), float(st.lid_conductance)
            G = 1.0 / (w / G_lid + (1.0 - w) * d_s / k_s)
            from scipy.optimize import brentq

            T_held = brentq(
                lambda T: sig * (T**4 - T_eq**4) - G * (st.T_magma - T), T_eq, st.T_magma
            )
            F_atm = sig * (T_held**4 - T_eq**4)
            t = t_end
        assert len(diffs) == 7
        assert abs(diffs[0]) < 0.25  # the first call moves the surface to the balance
        assert max(abs(v) for v in diffs[1:]) < 0.05
