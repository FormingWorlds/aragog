"""Tests of the in-step open-part flux of the sub-grid lid under the prescribed flux boundary.

With ``open_surface_conductance`` ``G`` the open part of the surface carries
``F_open(S) = F_open0 + G (T_m(S) - T_m0)``: ``F_open0`` the boundary value of the call,
``T_m = max(T_node, T_b)`` the top basic node held at least at the lid base, and ``T_m0`` its
value at the call start. The held lid surface is ``surface_temperature``.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import attrs
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aragog.solver.entropy_solver import SECS_PER_YEAR
from tests.conftest import entropy_eos_copy
from tests.test_entropy_solver_bc_dispatch_smoke import EOS_DIR, needs_eos
from tests.test_lid_mush_base import _solver

pytestmark = pytest.mark.timeout(900)

G_OPEN = 0.5  # open-part conductance of the tests [W/m^2/K]


@pytest.mark.unit
@settings(max_examples=80, deadline=None)
@given(
    st.one_of(
        st.floats(max_value=0.0, exclude_max=True),
        st.just(math.nan),
        st.just(math.inf),
        st.just(-math.inf),
        st.booleans(),
        st.text(max_size=3),
    )
)
def test_open_surface_conductance_refuses_every_other_value(value):
    """A conductance that is not None or a finite number of 0 or above is refused by name."""
    from aragog.config.boundary import check_open_surface_conductance

    with pytest.raises(ValueError, match='open_surface_conductance'):
        check_open_surface_conductance(value, 4, True)


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(st.one_of(st.none(), st.floats(min_value=0.0, allow_infinity=False), st.integers(0, 10)))
def test_open_surface_conductance_takes_none_and_finite_values_of_0_and_above(value):
    from aragog.config.boundary import check_open_surface_conductance

    assert check_open_surface_conductance(value, 4, True) is None


@pytest.mark.unit
@pytest.mark.parametrize('bc, mush', [(1, True), (4, False), (1, False), (5, True)])
def test_open_surface_conductance_needs_the_flux_boundary_and_the_mush_base(bc, mush):
    from aragog.config.boundary import check_open_surface_conductance

    assert check_open_surface_conductance(None, bc, mush) is None
    with pytest.raises(ValueError, match='open_surface_conductance'):
        check_open_surface_conductance(1.0, bc, mush)


@pytest.mark.unit
def test_the_held_lid_surface_is_the_surface_temperature_field():
    """One surface temperature field: the lid field is gone, the open conductance is new."""
    from aragog.config.boundary import BoundaryConfig
    from aragog.parser import _BoundaryConditionsParameters

    names = {f.name for f in dataclasses.fields(_BoundaryConditionsParameters)}
    config = {a.name: a for a in attrs.fields(BoundaryConfig)}
    for fields in (names, set(config)):
        assert 'lid_surface_temperature' not in fields
        assert 'open_surface_conductance' in fields
        assert 'surface_temperature' in fields
    assert config['open_surface_conductance'].default is None


@pytest.mark.unit
def test_normalize_runs_the_open_conductance_check():
    from tests.test_entropy_solver_bc_dispatch_smoke import _build

    p = _build(core_bc='quasi_steady', outer_bc=4, inner_bc=2, n_nodes=10, outer_bc_value=1.0)
    p.boundary_conditions.subgrid_lid = True
    p.boundary_conditions.lid_mush_base = True
    p.boundary_conditions.open_surface_conductance = -1.0
    with pytest.raises(ValueError, match='open_surface_conductance'):
        p.boundary_conditions.normalize()


def _open_solver(eos, G=None, F_atm=-1.0, T_s=500.0, delta_on=1.0, end=1.0):
    """A thin-shell lid solver on the flux boundary, held at ``T_s``, with conductance ``G``."""
    return _solver(
        eos,
        end=end,
        outer=4,
        F_atm=F_atm,
        thin=True,
        surface_temperature=T_s,
        open_surface_conductance=G,
        lid_delta_on=delta_on,
    )


def _state(s, dS, flat=True, d=1.0, T_s=500.0):
    """A state with the top cells (4 when ``flat``, else 1) at ``S_sol + dS`` and a lid ``d``."""
    n = s._n_stag
    y = np.full(n + 2, 3050.0)
    y[n - 4 if flat else n - 1 : n] = s._lid_S_sol + dS
    y[-2:] = (d, d * float(s._lid_hv.profile_heat(0.0, T_s, s._lid_T_sol)))
    return y


def _eval(s, y):
    """``(dS/dt, F_top, w, T_node, T_b)`` of the NumPy right-hand side at ``y``."""
    dS = np.asarray(s._dSdt_single(0.0, y), dtype=float)
    ev = s._lid_last
    T_node = float(np.ravel(s.state.phase_basic.temperature())[-1])
    return dS, float(s.state._heat_flux[-1]), float(ev.rates.w), T_node, float(ev.T_b)


@pytest.mark.smoke
@needs_eos
class TestOpenFluxNumpy:
    @pytest.fixture(scope='class')
    def eos(self):
        return entropy_eos_copy(EOS_DIR)

    def test_the_held_lid_takes_the_surface_temperature(self, eos):
        """The lid surface sits at ``surface_temperature`` and its loss is the conduction to it."""
        s = _open_solver(eos)
        y = _state(s, 20.0, flat=False, d=100.0)
        s._dSdt_single(0.0, y)
        ev = s._lid_last
        assert ev.surf.T_s == 500.0
        assert float(ev.surf.F_s) == pytest.approx(
            s._lid_k * (ev.T_b - 500.0) / 100.0, rel=1e-6
        )

    def test_start_of_step_reference(self, eos):
        """``T_m0`` is ``max(T_node, T_b)`` of the start state, and at that state the
        right-hand side with the conductance is the one without it, bit for bit."""
        s0, s1 = _open_solver(eos), _open_solver(eos, G=G_OPEN)
        y0 = _state(s0, 100.0)
        s1._set_lid_open_reference(y0)
        want, _, w, T_node, T_b = _eval(s0, y0)
        assert T_node > T_b and 0.0 < w < 1.0
        assert s1._lid_T_m0 == max(T_node, T_b)
        got = _eval(s1, y0)[0]
        np.testing.assert_array_equal(got, want)

    def test_open_flux_follows_the_node_above_the_lid_base(self, eos):
        """Away from the start the open part adds ``(1 - w) G (T_m - T_m0)`` to the loss."""
        s0, s1 = _open_solver(eos), _open_solver(eos, G=G_OPEN)
        y0, y1 = _state(s0, 100.0), _state(s0, 140.0)
        s1._set_lid_open_reference(y0)
        T_m0 = s1._lid_T_m0
        _, F0, w0, T_node, T_b = _eval(s0, y1)
        _, F1, w1, _, _ = _eval(s1, y1)
        assert w1 == w0 and 0.0 < w1 < 1.0
        assert T_node > T_b and T_node > T_m0 + 1.0
        assert F1 - F0 == pytest.approx((1.0 - w1) * G_OPEN * (T_node - T_m0), rel=1e-9)

    def test_open_flux_is_flat_below_the_lid_base(self, eos):
        """With the node below ``T_b`` at the start and now, ``T_m`` is ``T_b`` both times
        and the open part keeps its start value."""
        s0, s1 = _open_solver(eos, T_s=1200.0), _open_solver(eos, G=G_OPEN, T_s=1200.0)
        y0, y1 = _state(s0, 60.0, flat=False), _state(s0, 70.0, flat=False)
        s1._set_lid_open_reference(y0)
        _, F0, w, T_node, T_b = _eval(s0, y1)
        assert T_node < T_b and 0.0 < w < 1.0
        assert s1._lid_T_m0 == T_b
        assert _eval(s1, y1)[1] == F0

    @pytest.mark.parametrize('dS, flat', [(100.0, True), (140.0, True), (60.0, False)])
    def test_zero_or_no_conductance_is_the_flux_boundary_of_main(self, eos, dS, flat):
        """``None`` and 0 give the same right-hand side, bit for bit."""
        s0, s1 = _open_solver(eos), _open_solver(eos, G=0.0)
        y = _state(s0, dS, flat=flat)
        s1._set_lid_open_reference(_state(s0, 100.0))
        np.testing.assert_array_equal(_eval(s1, y)[0], _eval(s0, y)[0])

    def test_solve_sets_the_reference_from_its_start_state(self, eos):
        s = _open_solver(eos, G=G_OPEN, end=1.0)
        y0 = _state(s, 100.0)
        s._S0 = y0.copy()
        s.solve()
        _, _, _, T_node, T_b = _eval(_open_solver(eos), y0)
        assert s._lid_T_m0 == max(T_node, T_b)

    def test_booked_surface_energy_holds_the_open_flux_response(self, eos):
        """The booked surface integral of a call is the trapezoid, over the accepted
        states, of the loss without the conductance plus ``(1 - w) G (T_m - T_m0)``."""
        s = _open_solver(eos, G=G_OPEN, F_atm=-200.0, end=20.0)
        s._S0 = _state(s, 100.0)
        s.solve()
        sol = s._solution
        assert sol.success
        t, Y = sol.get('energy_trace') or (sol.t, sol.y)  # the nodes of the booking
        t, Y = np.asarray(t, dtype=float), np.asarray(Y, dtype=float)
        assert t.size >= 3
        ref, T_m0 = _open_solver(eos, F_atm=-200.0), s._lid_T_m0
        held, response = [], []
        for yi in Y.T:
            _, F, w, T_node, T_b = _eval(ref, yi)
            held.append(F)
            response.append((1.0 - w) * G_OPEN * (max(T_node, T_b) - T_m0))
        A = 4.0 * np.pi * float(s._r_basic_flat[-1]) ** 2
        dt = np.diff(t) * SECS_PER_YEAR

        def trap(loss):
            p = -np.asarray(loss) * A
            return float(np.sum(0.5 * (p[:-1] + p[1:]) * dt))

        total = trap(np.add(held, response))
        booked = sol.energy_integrals['F_int']
        assert abs(booked - total) <= 1e-6 * abs(total)
        assert abs(trap(response)) > 1e-3 * abs(total)

    def test_an_inward_start_flux_shrinks_as_the_node_warms(self, eos):
        """Under an inward boundary flux the response makes the open part take in less, so
        the mantle gains less energy over the call than with the start flux held."""
        gained, rise = [], []
        for G in (None, 5.0):
            s = _open_solver(eos, G=G, F_atm=-2000.0, end=50.0)
            y0 = _state(s, 100.0)
            s._S0 = y0.copy()
            s.solve()
            sol = s._solution
            assert sol.success
            gained.append(sol.energy_integrals['F_int'])
            rise.append(float(sol.y[s._n_stag - 1, -1] - y0[s._n_stag - 1]))
        assert gained[0] > 0.0 and rise[0] > 0.0
        assert 0.0 < gained[1] < gained[0]
        assert rise[1] < rise[0]

    def test_the_magma_temperature_row_takes_the_floored_node(self, eos):
        """Ruling 9: under a live lid the row ``T_magma`` is ``w T_b + (1 - w) T_m``, with the
        node below the lid base held at ``T_b``."""
        s = _open_solver(eos, end=1.0e-3)
        s._S0 = _state(s, 5.0, flat=False)
        s.solve()
        assert s._solution.success
        out = s.get_state()
        y_end = np.asarray(s._solution.y, dtype=float)[:, -1]
        _, _, w, T_node, T_b = _eval(_open_solver(eos), y_end)
        assert 0.0 < w < 1.0 and T_node < T_b
        assert out.T_magma == pytest.approx(w * T_b + (1.0 - w) * max(T_node, T_b), rel=1e-12)
        assert out.T_magma > w * T_b + (1.0 - w) * T_node + 1.0

    def test_largest_lid_weight_of_the_accepted_states(self, eos):
        """``lid_weight_max`` of a call is the largest weight over its accepted states."""
        s = _open_solver(eos, G=G_OPEN, end=200.0)
        s._S0 = _state(s, 100.0)
        s.solve()
        sol = s._solution
        t, Y = sol.get('energy_trace') or (sol.t, sol.y)  # the nodes of the booking
        ref = _open_solver(eos, G=G_OPEN)
        ref._lid_T_m0 = s._lid_T_m0
        w = [_eval(ref, yi)[2] for yi in np.asarray(Y, dtype=float).T]
        assert sol.lid_weight_max == max(w)
        assert 0.0 < sol.lid_weight_max
        assert s.get_state().lid_weight_max == sol.lid_weight_max


@pytest.mark.smoke
@needs_eos
class TestOpenFluxJax:
    @pytest.fixture(scope='class')
    def eos(self):
        return entropy_eos_copy(EOS_DIR)

    @staticmethod
    def _rhs(s):
        from tests.test_subgrid_lid_jax_rhs import _factory_rhs, _physical

        rhs_fn, jac_fn, info = _factory_rhs(s, s.jax_lid_params())
        assert info['lid']
        return rhs_fn, jac_fn, _physical(rhs_fn, s._build_nondim_scales())

    def test_lid_params_carry_the_conductance_and_the_reference(self, eos):
        s = _open_solver(eos, G=G_OPEN)
        s._set_lid_open_reference(_state(s, 100.0))
        lp = s.jax_lid_params()
        assert lp.open_G == G_OPEN
        assert lp.open_T_m0 == s._lid_T_m0
        lp0 = _open_solver(eos).jax_lid_params()
        assert lp0.open_G == 0.0

    def test_start_of_step_and_response_match_numpy(self, eos):
        """D1 on the JAX right-hand side (rel 1e-12 against itself without the conductance), and
        the response against NumPy (rel 1e-9, the parity of the lid tests)."""
        s0, s1 = _open_solver(eos), _open_solver(eos, G=G_OPEN)
        y0, y1 = _state(s0, 100.0), _state(s0, 140.0)
        s1._set_lid_open_reference(y0)
        f0, f1 = self._rhs(s0)[2], self._rhs(s1)[2]
        a, b = f1(y0), f0(y0)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max())
        top = s1._n_stag - 1
        want = _eval(s1, y1)[0]
        got = f1(y1)
        np.testing.assert_allclose(got[top], want[top], rtol=1e-9)
        np.testing.assert_allclose(got[-2:], want[-2:], rtol=1e-9, atol=1e-300)
        assert got[top] != f0(y1)[top]

    @staticmethod
    def _smooth(s, dS):
        """A smooth profile up to a top cell at ``S_sol + dS`` under a 1 m lid."""
        n = s._n_stag
        y = np.full(n + 2, 3050.0)
        y[:n] = np.linspace(800.0, s._lid_S_sol + dS, n)
        y[-2:] = (1.0, float(s._lid_hv.profile_heat(0.0, 500.0, s._lid_T_sol)))
        return y

    def _top_column(self, s, y):
        """The JAX Jacobian column of the top cell and its central difference, nondimensional."""
        rhs_fn, jac_fn, _ = self._rhs(s)
        sc = s._build_nondim_scales()
        x = (y - sc.state_offset) / sc.state_scale
        n, top = x.size, s._n_stag - 1
        J, fy, fp, fm = np.zeros((n, n)), np.empty(n), np.empty(n), np.empty(n)
        assert rhs_fn(0.0, x, fy) == 0 and jac_fn(0.0, x, fy, J) == 0
        h = 1e-6 * max(abs(x[top]), 1.0)
        rhs_fn(0.0, x + h * (np.arange(n) == top), fp)
        rhs_fn(0.0, x - h * (np.arange(n) == top), fm)
        return J[:, top], (fp - fm) / (2.0 * h)

    def test_jacobian_holds_the_response(self, eos):
        """The response term of the JAX Jacobian (with less without the conductance) against
        the same difference of central differences, on a smooth state above the lid base."""
        s0, s1 = _open_solver(eos), _open_solver(eos, G=50.0)
        y0, y1 = self._smooth(s1, 100.0), self._smooth(s1, 140.0)
        s1._set_lid_open_reference(y0)
        _, _, _, T_node, T_b = _eval(s0, y1)
        assert T_node > T_b + 1.0
        (J0, fd0), (J1, fd1) = self._top_column(s0, y1), self._top_column(s1, y1)
        top = s1._n_stag - 1
        assert J1[top] == pytest.approx(fd1[top], rel=1e-6)
        assert J1[top] - J0[top] == pytest.approx(fd1[top] - fd0[top], rel=1e-6)
        assert abs(J1[top] - J0[top]) > 1e-3 * abs(J0[top])

    def test_lid_params_refuse_a_conductance_without_a_reference(self, eos):
        """With a conductance and no start reference the JAX parameters and the NumPy
        right-hand side raise."""
        s = _open_solver(eos, G=G_OPEN)
        with pytest.raises(RuntimeError, match='reference'):
            s.jax_lid_params()
        with pytest.raises(RuntimeError, match='no start reference'):
            s._dSdt_single(0.0, _state(s, 100.0))
        assert _open_solver(eos).jax_lid_params().open_G == 0.0

    def test_the_lid_weight_batch_matches_numpy_and_falls_back(self, eos, caplog):
        """The JAX batch of lid weights equals the NumPy weight at each node (rel 1e-9); when the
        batch raises, ``_lid_weights_of`` falls back to the NumPy loop."""
        s = _open_solver(eos, G=G_OPEN)
        s._set_lid_open_reference(_state(s, 100.0))
        rhs_fn = self._rhs(s)[0]
        dS = (20.0, 60.0, 80.0, 100.0, 120.0)
        Y = np.column_stack([_state(s, d) for d in dS])
        t = np.zeros(len(dS))
        want = []
        for y in Y.T:
            s._dSdt_single(0.0, y)
            want.append(s._lid_weight_now())
        want = np.array(want)
        assert np.ptp(want) > 0.5  # the nodes span the onset
        np.testing.assert_allclose(rhs_fn.lid_weights(t, Y), want, rtol=1e-9, atol=1e-15)
        s._cvode_lid_weights_batch = rhs_fn.lid_weights
        np.testing.assert_allclose(s._lid_weights_of(t, Y), want, rtol=1e-9, atol=1e-15)

        def broken(t_pts, y_pts):
            raise RuntimeError('batch failed')

        s._cvode_lid_weights_batch = broken
        with caplog.at_level(logging.WARNING):
            got = s._lid_weights_of(t, Y)
        np.testing.assert_array_equal(got, want)
        assert 'batch lid weights failed (batch failed)' in caplog.text

    def test_zero_conductance_is_the_rhs_of_main(self, eos):
        s0, s1 = _open_solver(eos), _open_solver(eos, G=0.0)
        s1._set_lid_open_reference(_state(s0, 100.0))
        f0, f1 = self._rhs(s0)[2], self._rhs(s1)[2]
        for dS in (100.0, 140.0):
            y = _state(s0, dS)
            np.testing.assert_array_equal(f1(y), f0(y))

    def test_core_module_matches_numpy_with_open_conductance(self, eos):
        """With ``core_bc='core_module'`` and a live lid, the JAX right-hand side passes
        the top basic node for the open-part response and matches NumPy (rel 1e-9)."""
        from aragog.parser import SolidRheologyParams
        from aragog.solver.entropy_solver import EntropySolver
        from tests.test_entropy_solver_core_module_smoke import CORE_MODULE_PARAMS
        from tests.test_entropy_solver_core_module_smoke import _build as _cm_build

        def _cm_open(G):
            p = _cm_build(
                core_bc='core_module', shared_eos=eos, core_module_params=CORE_MODULE_PARAMS
            ).parameters
            p.mesh.inner_radius = 6.346e6
            p.energy.convection = True
            p.boundary_conditions.outer_boundary_condition = 4
            p.boundary_conditions.outer_boundary_value = -1.0
            p.boundary_conditions.surface_temperature = 500.0
            p.boundary_conditions.subgrid_lid = True
            p.boundary_conditions.lid_mush_base = True
            p.boundary_conditions.lid_delta_on = 1.0
            p.boundary_conditions.open_surface_conductance = G
            p.boundary_conditions.normalize()
            p.phase_solid.rheology = SolidRheologyParams(
                enabled=True, stress_closure_mode='local', lid_base_mode='fixed'
            )
            s = EntropySolver(p, entropy_eos=eos)
            s.initialize()
            s.set_initial_entropy(3050.0)
            return s

        s0, s1 = _cm_open(0.0), _cm_open(G_OPEN)
        n = s1._n_stag
        y0 = s1._S0.copy()
        y0[n - 4 : n] = s1._lid_S_sol + 100.0
        y0[-2:] = (1.0, float(s1._lid_hv.profile_heat(0.0, 500.0, s1._lid_T_sol)))
        y1 = y0.copy()
        y1[n - 4 : n] = s1._lid_S_sol + 140.0
        s1._set_lid_open_reference(y0)
        want = np.asarray(s1._dSdt_single(0.0, y1), dtype=float)
        got0 = self._rhs(s0)[2](y1)
        got1 = self._rhs(s1)[2](y1)
        top = n - 1
        np.testing.assert_allclose(got1[top], want[top], rtol=1e-9)
        assert got1[top] != got0[top]


@pytest.mark.smoke
@needs_eos
class TestAcceptedRange:
    @pytest.fixture(scope='class')
    def eos(self):
        return entropy_eos_copy(EOS_DIR)

    @staticmethod
    def _sol(s, bad):
        from scipy.optimize import OptimizeResult

        n = s._S0.size
        y = np.tile(np.asarray(s._S0, dtype=float)[:, None], (1, 3))
        if bad:
            y[2, 1] = float(s.entropy_eos.S_max) + 50.0
        t = np.array([0.0, 1.0, 2.0])
        sol = OptimizeResult(t=t, y=y, status=0, success=True)
        sol['energy_trace'] = (t, y)
        assert y.shape[0] == n
        return sol

    def _plain(self, eos, strict):
        from aragog.solver.entropy_solver import EntropySolver
        from tests.test_entropy_solver_bc_dispatch_smoke import _build

        p = _build(core_bc='quasi_steady', outer_bc=4, inner_bc=2, n_nodes=10, end_time=1.0)
        p.solver.strict_accepted_range = strict
        s = EntropySolver(p, entropy_eos=eos)
        s.initialize()
        s.set_initial_entropy(3050.0)
        return s

    def test_strict_check_raises_on_an_accepted_state_out_of_the_table(self, eos):
        """With ``strict_accepted_range`` an accepted state outside the tables raises a
        named error with the node, the entropy and the time; without it the call fails."""
        from aragog.solver.entropy_solver import AcceptedStateRangeError

        s = self._plain(eos, strict=True)
        assert s._eos_table_failure(self._sol(s, bad=False)) is None
        with pytest.raises(AcceptedStateRangeError, match=r'cell 2 .*t=1\.0+e\+00 yr'):
            s._eos_table_failure(self._sol(s, bad=True))
        s = self._plain(eos, strict=False)
        t_bad, msg = s._eos_table_failure(self._sol(s, bad=True))
        assert t_bad == 1.0 and 'cell 2' in msg

    @pytest.mark.parametrize('bad', [1, 0, 'true', None])
    def test_strict_accepted_range_is_off_by_default_and_a_bool(self, bad):
        """Off by default in the parser and in the config; anything but a bool is refused."""
        from aragog.config.solver import SolverConfig
        from aragog.parser import _SolverParameters

        assert _SolverParameters(start_time=0.0, end_time=1.0).strict_accepted_range is False
        assert SolverConfig(start_time=0.0, end_time=1.0).strict_accepted_range is False
        with pytest.raises(TypeError, match='strict_accepted_range'):
            _SolverParameters(start_time=0.0, end_time=1.0, strict_accepted_range=bad)
        with pytest.raises(TypeError, match='strict_accepted_range'):
            SolverConfig(start_time=0.0, end_time=1.0, strict_accepted_range=bad)

    def test_a_handed_over_lid_has_weight_0(self, eos):
        """After the hand-over the lid weight of the last evaluation reads 0, whatever its rates."""
        s = _open_solver(eos, G=G_OPEN)
        s._set_lid_open_reference(_state(s, 100.0))
        s._dSdt_single(0.0, _state(s, 60.0))
        w_live = s._lid_weight_now()
        assert w_live > 0.5
        s._lid_active = False
        assert s._lid_weight_now() == 0.0

    def test_the_error_is_a_runtime_error(self):
        from aragog.solver.entropy_solver import AcceptedStateRangeError

        assert issubclass(AcceptedStateRangeError, RuntimeError)


@pytest.mark.smoke
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@needs_eos
def test_the_open_part_follows_the_open_node_across_the_lid_base(validation_plot):
    """Across states from the node below the lid base to 9 K above the start, the
    change of the surface loss with the conductance is ``(1 - w) G (T_m - T_m0)``, with
    ``T_m = max(T_node, T_b)``: relative 1e-9 (absolute 1e-12 W/m^2 where ``1 - w`` rounds); a
    node far below the lid base keeps the start value exactly. The JAX top-cell rate equals the
    NumPy one to relative 1e-9 at each state."""
    eos = entropy_eos_copy(EOS_DIR)
    series = {}
    for name, T_s, flat, ref_dS, grid in (
        ('node near the lid base', 500.0, True, 100.0, range(-40, 170, 10)),
        ('node far below the lid base', 1200.0, False, 60.0, range(60, 130, 10)),
    ):
        s0, s1 = _open_solver(eos, T_s=T_s), _open_solver(eos, G=G_OPEN, T_s=T_s)
        s1._set_lid_open_reference(_state(s0, ref_dS, flat=flat))
        T_m0, top = s1._lid_T_m0, s1._n_stag - 1
        f1 = TestOpenFluxJax._rhs(s1)[2]
        rows = []
        for dS in grid:
            y = _state(s0, float(dS), flat=flat)
            _, F0, w, T_node, T_b = _eval(s0, y)
            dS1, F1, w1, _, _ = _eval(s1, y)
            assert w1 == w
            want = (1.0 - w) * G_OPEN * (max(T_node, T_b) - T_m0)
            assert abs((F1 - F0) - want) <= 1e-12 + 1e-9 * abs(want)
            assert float(f1(y)[top]) == pytest.approx(float(dS1[top]), rel=1e-9)
            rows.append((max(T_node, T_b) - T_m0, F1 - F0, want, 1.0 - w, T_node - T_b))
        series[name] = np.array(rows)
    near, far = series['node near the lid base'], series['node far below the lid base']
    assert np.any(near[:, 4] < 0.0) and np.any(near[:, 4] > 1.0)  # both sides of the base
    assert np.all(far[:, 4] < 0.0) and np.all(far[:, 1] == 0.0)  # flat below the base
    assert np.max(np.abs(near[:, 1])) > 1.0  # the response is not vacuously 0

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        x = np.linspace(near[:, 0].min(), near[:, 0].max(), 2)
        ax.plot(x, G_OPEN * x, color='C0', label='G (T_m - T_m0)')
        for (name, d), mk in zip(series.items(), ('o', 's'), strict=True):
            m = d[:, 3] > 1e-6
            ax.plot(d[m, 0], d[m, 1] / d[m, 3], mk, ms=4, mfc='none', label=name)
        ax.set(xlabel='T_m - T_m0 [K]', ylabel='dF / (1 - w) [W/m2]')
        ax.legend(fontsize=8)
        # the round-off difference is text: its value depends on the platform
        err = max(
            np.max(np.abs(d[:, 1] - d[:, 2]) / (1e-12 + 1e-9 * np.abs(d[:, 2])))
            for d in series.values()
        )
        ax.text(
            0.03,
            0.55,
            f'max difference / tolerance\n= {err:.1e}',
            transform=ax.transAxes,
            fontsize=8,
        )
        fig.tight_layout()
        return fig

    validation_plot(figure, 'surface_lid_open_part_in_step')
