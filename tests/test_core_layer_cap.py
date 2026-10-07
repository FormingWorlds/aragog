"""Diffusion-length cap of the stratified core layer: min(quasi-static depth,
2 erfcinv(0.1) sqrt(kappa t_layer)), with the onset time kept by the solver."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from scipy.special import erfc, erfcinv

from aragog.core import CoreEntropyBudget
from tests.conftest import entropy_eos_copy, needs_eos

sys.path.insert(0, str(Path(__file__).parent))
from test_core_stratified_budget import _budget  # noqa: E402
from test_entropy_solver_core_module_smoke import (  # noqa: E402
    STRATIFIED_PARAMS,
    _build,
    _driven_s_profile,
)

pytestmark = pytest.mark.unit
MYR = 1.0e6 * 365.25 * 86400.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('q_tw', 'age_myr', 'branch'),
    [
        (8, 10, 'cap'),
        (8, -1, 'none'),
        (8, 500, 'quasi'),
        (20, 10, 'none'),
        (8, 10, 'flat'),
    ],
)
def test_the_layer_is_the_smaller_of_the_quasi_static_and_diffusive_depths(
    q_tw, age_myr, branch
):
    """At 4400 K and 8 TW the quasi-static depth is about 820 km and 2.33 sqrt(kappa t) is 163 km
    at 10 Myr and 1156 km at 500 Myr; 20 TW is superadiabatic, and a negative age has no
    layer. The cap has no T derivative."""
    from aragog.core.budget import LAYER_DIFFUSION_PREFACTOR as a

    assert erfc(a / 2.0) == pytest.approx(0.1, rel=1e-14)
    budget, t, q, age = _budget(branch != 'flat'), 4400.0, q_tw * 1e12, age_myr * MYR
    p = budget.profiles
    kappa = 130.0 / (float(p.density(p.r_cmb)) * 840.0)
    expected = {
        'cap': 2.0 * erfcinv(0.1) * np.sqrt(kappa * max(age, 0.0)),
        'quasi': p.r_cmb - float(budget.convecting_radius(t, q)),
        'none': 0.0,
        'flat': 0.0,
    }[branch]
    depth = p.r_cmb - float(budget.convecting_radius(t, q, age))
    assert depth == pytest.approx(expected, rel=1e-12, abs=1e-6)
    ent = CoreEntropyBudget(budget, k_core=130.0)  # the dynamo margin follows the depth used
    same = float(ent.entropy_margin(t, q, t_layer=age)) == float(ent.entropy_margin(t, q))
    assert same is (depth == p.r_cmb - float(budget.convecting_radius(t, q)))
    slope = float(jax.grad(lambda x: budget.convecting_radius(x, q, age))(t))
    assert (slope == 0.0) is (branch != 'quasi')


@pytest.mark.physics_invariant
@pytest.mark.parametrize('age_myr', [10, 1e4])
def test_the_core_rate_is_continuous_through_a_zero_cmb_flow(age_myr):
    """Heat into or out of the core: the convecting core loses Q_ad(r_s) to the layer either
    way, so dT/dt has no jump at q = 0; its radiogenic share is its mass fraction."""
    budget, t, age = _budget(True), 4400.0, age_myr * MYR
    rate = [float(budget.dtcmb_dt(t, q, t_layer=age)) for q in (-1e6, 0.0, 1e6)]
    assert rate[0] == pytest.approx(rate[1], rel=1e-5, abs=0)
    assert rate[2] == pytest.approx(rate[1], rel=1e-5, abs=0)
    frac = float(budget.convecting_mass_fraction(t, 8e12, age))
    heated = float(budget.dtcmb_dt(t, 8e12, q_sources=2e12, t_layer=age))
    base = float(budget.base_heat_flow(t, 8e12, age))
    assert 0.0 < frac < 1.0
    assert heated == pytest.approx(
        (2e12 * frac - base) / float(budget.effective_capacity(t, 8e12, age)), rel=1e-12, abs=0
    )


@pytest.mark.parametrize(
    ('override', 'kept', 'stratified_start', 'expected'),
    [
        (None, None, True, 5.0),
        (None, 1.0, True, 1.0),
        (None, 1.0, False, 5.0),
        (-3.0, 1.0, True, -3.0),
        (-3.0, 1.0, False, 5.0),
        (float('nan'), None, True, 5.0),
    ],
    ids=['first_solve', 'kept', 'eroded', 'resumed', 'resumed_eroded', 'nan_from_output'],
)
def test_the_layer_onset_rule(override, kept, stratified_start, expected):
    from aragog.solver.entropy_solver import EntropySolver

    fake = SimpleNamespace(
        _core_bc='core_module',
        _core_module_budget=SimpleNamespace(stratification=True),
        _layer_start_override=override,
        _core_layer_start_yr=kept,
        _start_is_stratified=lambda t: stratified_start,
        _core_layer_stored_J=7.0,
    )
    EntropySolver._resolve_layer_start(fake, 5.0)
    assert (fake._core_layer_start_yr, fake._layer_start_override) == (expected, None)
    lost = 0.0 if stratified_start else 7.0  # a fresh start loses the stored heat once
    fake._core_layer_stored_J = 99.0
    EntropySolver._resolve_layer_start(fake, 5.0)  # a retry restores the totals first
    assert (fake._layer_loss_pending, fake._core_layer_stored_J) == (lost, 7.0 - lost)


def test_a_large_layer_loss_warns_once(caplog):
    from aragog.solver.entropy_solver import EntropySolver

    fake = SimpleNamespace(_core_heat_J=1.0e3)
    with caplog.at_level(logging.WARNING):
        for loss in (0.5, 2.0, 2.0):
            EntropySolver._warn_on_layer_loss(fake, loss)
    assert sum('eroded stratified layer' in r.message for r in caplog.records) == 1


@needs_eos
@pytest.mark.smoke
@pytest.mark.physics_invariant
def test_a_resumed_run_with_the_onset_equals_the_uninterrupted_one():
    """A 50 Myr old layer, core 300 K below the mantle: a fresh solver from the 2 yr state with
    the output onset continues that layer and heats the core as the continued run does;
    without it the layer restarts at the solve's start."""
    eos = entropy_eos_copy()

    def run(start, t_core, s_init, dsdr=None, onset=None):
        solver = _build('core_module', eos, STRATIFIED_PARAMS, end_time=start + 2.0)
        solver.parameters.solver.start_time = start
        solver.set_initial_core_temperature(t_core)
        solver.set_initial_dSdr_cmb(dsdr)
        solver.set_initial_layer_start(onset)
        solver.set_initial_entropy(s_init)
        solver.solve()
        return solver

    probe = _build('core_module', eos, STRATIFIED_PARAMS)
    s0 = _driven_s_profile(probe._n_stag)
    t_m = float(np.asarray(eos.temperature(probe._P_basic_flat[:1], s0[:1])).item())
    whole = run(0.0, t_m - 300.0, s0, onset=-50.0e6)
    onset, n = whole.get_state().core_layer_start_yr, whole._n_stag
    y_mid = np.asarray(whole._solution.y[:, -1])
    whole.parameters.solver.start_time, whole.parameters.solver.end_time = 2.0, 4.0
    whole.set_initial_core_temperature(None)
    whole.reset()
    whole.set_initial_entropy(y_mid[:n])
    whole.solve()
    resumed, fresh = (run(2.0, y_mid[n + 1], y_mid[:n], y_mid[n], o) for o in (onset, None))
    assert (onset, whole._core_layer_start_yr, resumed._core_layer_start_yr) == (-50.0e6,) * 3
    assert fresh._core_layer_start_yr == 2.0
    rounding = 8.0 * np.finfo(float).eps * y_mid[n + 1]
    end = float(whole._solution.y[n + 1, -1])
    assert float(resumed._solution.y[n + 1, -1]) == pytest.approx(end, rel=0, abs=rounding)


@needs_eos
@pytest.mark.smoke
@pytest.mark.physics_invariant
def test_an_eroded_layer_books_its_stored_heat_as_a_loss():
    """A layer stores heat under a core 300 K below the mantle; a later call that starts 300 K
    above a liquid base erodes it, books that heat as its loss, and the ledger closes with it."""
    eos = entropy_eos_copy()
    solver = _build('core_module', eos, STRATIFIED_PARAMS, end_time=2.0)
    s0 = _driven_s_profile(solver._n_stag)
    t_m = float(np.asarray(eos.temperature(solver._P_basic_flat[:1], s0[:1])).item())
    solver.set_initial_core_temperature(t_m - 300.0)
    solver.set_initial_layer_start(-50.0e6)
    solver.set_initial_entropy(s0)
    solver.solve()
    stored = solver.get_state().core_layer_stored_J
    solver.parameters.solver.start_time, solver.parameters.solver.end_time = 2.0, 2.001
    hot = np.linspace(7000.0, 6700.0, solver._n_stag)  # a liquid base carries Q > Q_k
    t_hot = float(np.asarray(eos.temperature(solver._P_basic_flat[:1], hot[:1])).item())
    solver.reset()
    solver.set_initial_core_temperature(t_hot + 300.0)
    solver.set_initial_entropy(hot)
    solver.solve()
    out = solver.get_state()
    assert stored > 0.0 and out.step_dE_layer_loss_J == stored
    assert out.core_layer_stored_J == 0.0
    closure = out.step_dE_core_J + out.step_dE_layer_loss_J
    assert closure == pytest.approx(-out.step_dE_F_cmb_J, rel=1e-4)  # the loss is 2e-3 of it


def test_an_implicit_solve_through_the_cap_crossover_does_not_thrash():
    """Core-only rate under 8 TW (not a coupled CVODE solve): the cap meets the quasi-static
    depth near 250 Myr (test above: 729 km at 200 Myr against about 800 km); SciPy's BDF at
    rtol 1e-9 crosses that kink with about 15 smaller steps and recovers within 30 Myr."""
    from scipy.integrate import solve_ivp

    budget = _budget(True)
    rate = jax.jit(lambda t, temp: budget.dtcmb_dt(temp, 8e12, t_layer=t * MYR))
    sol = solve_ivp(
        lambda t, y: [float(rate(t, y[0])) * MYR], (100.0, 500.0), [4400.0], 'BDF', rtol=1e-9
    )
    t, dt = sol.t[:-1], np.diff(sol.t)
    assert sol.success and np.sum((t > 230.0) & (t < 270.0)) <= 20 and dt[t > 280.0].min() > 5.0
