"""Diffusion-length cap of the stratified core layer: min(quasi-static depth,
2 erfcinv(0.1) sqrt(kappa t_layer)), with the onset time kept by the solver."""

from __future__ import annotations

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
    )
    EntropySolver._resolve_layer_start(fake, 5.0)
    assert (fake._core_layer_start_yr, fake._layer_start_override) == (expected, None)


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
