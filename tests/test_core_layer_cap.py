"""Diffusion-length cap of the stratified core layer.

The quasi-static layer depth is capped by ``a sqrt(kappa t_layer)`` with ``a = 2
erfcinv(0.1)``, the depth where a conducted heat-flow deficit spreading as
``erfc(z / 2 sqrt(kappa t))`` has fallen to 10 %; ``t_layer`` is the time since the
layer's onset, which the solver carries as a scalar across calls and outputs for resume.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from scipy.special import erfc, erfcinv

from aragog.core import CoreEnergyBudget, GaussianCoreProfiles, QuadraticMeltingCurve
from tests.conftest import entropy_eos_copy, needs_eos

sys.path.insert(0, str(Path(__file__).parent))
from test_entropy_solver_core_module_smoke import (  # noqa: E402
    STRATIFIED_PARAMS,
    _build,
    _driven_s_profile,
)

pytestmark = pytest.mark.unit

MYR = 1.0e6 * 365.25 * 86400.0


@pytest.fixture(scope='module')
def budget():
    """The Nimmo (2015, ch. 8.02, Table 2) core with stratification and k = 130 W/m/K."""
    prof = GaussianCoreProfiles(
        rho_cen=12500.0,
        length_scale=7272e3,
        r_cmb=3480e3,
        p_cmb=136e9,
        alpha=1.25e-5,
        c_p=840.0,
    )
    curve = QuadraticMeltingCurve(t_m0=2677.0, t_m1=2.95e-12, t_m2=8.37e-25)
    return CoreEnergyBudget(
        prof, curve, ds_fusion=170.0, icn_width=10.0, stratification=True, k_core=130.0
    )


def test_the_prefactor_is_the_ten_percent_depth_of_the_erfc_deficit():
    from aragog.core.budget import LAYER_DIFFUSION_PREFACTOR as a

    assert erfc(a / 2.0) == pytest.approx(0.1, rel=1e-14)
    assert a == pytest.approx(2.0 * erfcinv(0.1), rel=1e-15)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('q_tw', 'age_myr', 'branch'),
    [
        (8.0, 0.0, 'cap'),
        (8.0, 10.0, 'cap'),
        (8.0, 50.0, 'cap'),
        (8.0, 200.0, 'cap'),
        (8.0, 500.0, 'quasi_static'),
        (12.0, 10.0, 'cap'),
        (12.0, 100.0, 'quasi_static'),
        (20.0, 10.0, 'none'),
    ],
)
def test_the_layer_is_the_smaller_of_the_quasi_static_and_diffusive_depths(
    budget, q_tw, age_myr, branch
):
    """At 4400 K the quasi-static depth is about 820 km at 8 TW and 370 km at 12 TW, and no
    layer forms above the adiabatic 15.8 TW; the diffusion length 2.33 sqrt(kappa t) is 163 km
    at 10 Myr, 516 km at 100 Myr, 729 km at 200 Myr and 1152 km at 500 Myr (kappa = 130 /
    (rho_cmb 840))."""
    p = budget.profiles
    t, q, age = 4400.0, q_tw * 1e12, age_myr * MYR
    kappa = 130.0 / (float(p.density(p.r_cmb)) * 840.0)
    diffusive = 2.0 * erfcinv(0.1) * np.sqrt(kappa * age)
    quasi_static = p.r_cmb - float(budget.convecting_radius(t, q))
    capped = p.r_cmb - float(budget.convecting_radius(t, q, age))
    expected = {'cap': diffusive, 'quasi_static': quasi_static, 'none': 0.0}[branch]
    assert capped == pytest.approx(expected, rel=1e-12, abs=1e-6)
    assert capped == pytest.approx(min(quasi_static, diffusive), rel=1e-12, abs=1e-6)
    # Its temperature derivative: zero on the cap, the implicit sensitivity on the quasi-static branch.
    slope = float(jax.grad(lambda x: budget.convecting_radius(x, q, age))(t))
    assert (slope == 0.0) is (branch != 'quasi_static')


def test_without_stratification_the_age_changes_nothing(budget):
    flat = CoreEnergyBudget(
        budget.profiles, budget.melting_curve, ds_fusion=170.0, icn_width=10.0
    )
    assert float(flat.convecting_radius(4400.0, 8e12, 10 * MYR)) == budget.profiles.r_cmb
    assert float(flat.effective_capacity(4400.0, 8e12, 10 * MYR)) == float(
        flat.effective_capacity(4400.0)
    )


@pytest.mark.parametrize(
    ('override', 'kept', 'stratified_end', 'expected'),
    [
        (None, None, False, 5.0),  # first solve: the layer starts with it
        (None, 1.0, True, 1.0),  # a layer stratified at the last end keeps its onset
        (None, 1.0, False, 5.0),  # an eroded layer restarts with this solve
        (-3.0, 1.0, True, -3.0),  # a resumed run sets its onset
        (float('nan'), None, True, 5.0),  # NaN from an unstratified output decides nothing
    ],
)
def test_the_layer_onset_rule(override, kept, stratified_end, expected):
    from aragog.solver.entropy_solver import EntropySolver

    fake = SimpleNamespace(
        _core_bc='core_module',
        _core_module_budget=SimpleNamespace(stratification=True),
        _layer_start_override=override,
        _core_layer_start_yr=kept,
        _core_layer_stratified_end=stratified_end,
    )
    EntropySolver._resolve_layer_start(fake, 5.0)
    assert fake._core_layer_start_yr == expected
    assert fake._layer_start_override is None


@needs_eos
@pytest.mark.physics_invariant
def test_a_resumed_run_with_the_onset_equals_the_uninterrupted_one():
    """A layer 50 Myr old (365 km by its cap) under a core 300 K below the mantle: two calls
    of 2 yr against a fresh solver started at 2 yr from the first call's end state. With the
    output onset passed back the core heats along the same path; without it the layer
    restarts at zero depth, the full core takes the heat, and the core heats more slowly."""

    def solver_at(start, t_core, s_init, dsdr=None, onset=None):
        solver = _build(
            'core_module', entropy_eos_copy(), STRATIFIED_PARAMS, end_time=start + 2.0
        )
        solver.parameters.solver.start_time = start
        solver.set_initial_core_temperature(t_core)
        if dsdr is not None:
            solver.set_initial_dSdr_cmb(dsdr)
        solver.set_initial_layer_start(onset)
        solver.set_initial_entropy(s_init)
        return solver

    probe = _build('core_module', entropy_eos_copy(), STRATIFIED_PARAMS)
    s0 = _driven_s_profile(probe._n_stag)
    t_m = float(
        np.asarray(probe.entropy_eos.temperature(probe._P_basic_flat[:1], s0[:1])).item()
    )
    whole = solver_at(0.0, t_m - 300.0, s0, onset=-50.0e6)
    whole.solve()
    first = whole.get_state()
    n = whole._n_stag
    y_mid = np.asarray(whole._solution.y[:, -1])
    whole.parameters.solver.start_time, whole.parameters.solver.end_time = 2.0, 4.0
    whole.set_initial_core_temperature(None)
    whole.reset()
    whole.set_initial_entropy(y_mid[:n])
    whole.solve()
    t_whole = float(whole._solution.y[n + 1, -1])

    def resumed(onset):
        solver = solver_at(2.0, y_mid[n + 1], y_mid[:n], dsdr=y_mid[n], onset=onset)
        solver.solve()
        return float(solver._solution.y[n + 1, -1])

    t_mid = y_mid[n + 1]
    assert first.core_layer_start_yr == -50.0e6
    assert resumed(first.core_layer_start_yr) - t_mid == pytest.approx(
        t_whole - t_mid, rel=1e-6
    )
    assert (resumed(None) - t_mid) / (t_whole - t_mid) < 0.9


@needs_eos
@pytest.mark.physics_invariant
@pytest.mark.parametrize('onset', [-1.0e7, -1.0e10], ids=['cap_active', 'cap_inactive'])
def test_jax_matches_numpy_and_its_jacobian_with_the_layer_capped(onset):
    """The core 50 K above the mantle, stratified: NumPy and JAX right-hand sides agree, and
    the JAX core column matches central differences, with a 10 Myr layer (capped) and a
    10 Gyr layer (quasi-static). The onset is an argument of one compiled function, so a new
    onset reuses it."""
    from test_jax_dsdt_core_module import _build_jax_pieces

    from aragog.jax.solver import dSdt_core_module

    solver = _build('core_module', entropy_eos_copy(), STRATIFIED_PARAMS, s_init='driven')
    solver.set_initial_layer_start(onset)
    solver._resolve_layer_start(0.0)
    n = solver._n_stag
    y = np.asarray(solver._S0, dtype=float)
    y[n + 1] += 50.0
    budget = solver._core_module_budget
    solver.dSdt(0.0, y)
    q = float(solver.state.heat_flux[0]) * solver._cmb_area
    capped = float(budget.convecting_radius(y[n + 1], q, -onset * 365.25 * 86400.0))
    assert (capped > float(budget.convecting_radius(y[n + 1], q))) is (onset == -1.0e7)

    args = _build_jax_pieces(solver)
    rhs = jax.jit(lambda t, v, start: dSdt_core_module(t, v, args + (start,)))
    f_np = np.asarray(solver.dSdt(0.0, y), dtype=float)
    f_jax = np.asarray(rhs(0.0, y, onset))
    np.testing.assert_allclose(f_jax, f_np, rtol=1e-10)
    jac = np.asarray(jax.jacrev(lambda v: dSdt_core_module(0.0, v, args + (onset,)))(y))[
        :, n + 1
    ]
    h, up, down = 0.05, y.copy(), y.copy()
    up[n + 1] += h
    down[n + 1] -= h
    fd = (np.asarray(rhs(0.0, up, onset)) - np.asarray(rhs(0.0, down, onset))) / (2.0 * h)
    assert jac[n + 1] == pytest.approx(fd[n + 1], rel=1e-6)
    rhs(0.0, y, onset - 1.0e6)
    assert rhs._cache_size() == 1


@pytest.mark.parametrize('rtol', [1e-6, 1e-9])
def test_an_implicit_solve_through_the_cap_crossover_does_not_thrash(budget, rtol):
    """The core alone under a fixed 8 TW from a 100 Myr old layer: the diffusive depth
    overtakes the quasi-static one (about 820 km) near 250 Myr, a kink in time of the cooling
    rate. SciPy's BDF crosses it with a short run of smaller steps (none at rtol 1e-6, about
    15 at 1e-9) and returns to its step size within 30 Myr; it does not thrash. This is the
    core-only rate, not a coupled CVODE solve."""
    from scipy.integrate import solve_ivp

    rate = jax.jit(lambda t, temp: budget.dtcmb_dt(temp, 8e12, t_layer=t * MYR))
    sol = solve_ivp(
        lambda t, y: [float(rate(t, y[0])) * MYR],
        (100.0, 500.0),
        [4400.0],
        method='BDF',
        rtol=rtol,
        atol=1e-6,
        dense_output=True,
    )
    assert sol.success
    gap = jax.jit(
        jax.vmap(
            lambda temp, age: (
                budget.convecting_radius(temp, 8e12, age * MYR)
                - budget.convecting_radius(temp, 8e12)
            )
        )
    )
    grid = np.linspace(100.0, 500.0, 4001)
    kink = grid[np.argmax(np.asarray(gap(sol.sol(grid)[0], grid)) <= 0.0)]
    assert 200.0 < kink < 300.0
    t, dt = sol.t[:-1], np.diff(sol.t)
    assert np.sum(np.abs(t - kink) < 20.0) <= 20
    assert dt[t > kink + 30.0].min() > 5.0
