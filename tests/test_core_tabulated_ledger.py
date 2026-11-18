"""The per-call core ledger of a PALEOS core in CVODE solves.

The rule is that of ``docs/Validation/core.md`` (Core ledger closure), held on the Gaussian
path by ``test_entropy_solver_core_module_smoke.py``: the first 3 calls of a zero-flux start
and a call across the inner-core onset close to below 1e-5 of their CMB heat at 1000 times
tighter tolerances, falling at least 20 times; later calls close to 2e-6 at the default
tolerances. The tabulated budget books the exact integral of the capacity it integrates, so
its residual is the solver's alone.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests.conftest import PALEOS_CUT as CUT
from tests.conftest import entropy_eos_copy, legend_outside, needs_eos, write_core
from tests.test_entropy_solver_core_module_smoke import _assert_converges, _build, needs_cvode

pytestmark = [pytest.mark.smoke, needs_eos, needs_cvode, pytest.mark.timeout(900)]


@pytest.fixture(scope='module')
def shared_eos():
    return entropy_eos_copy()


@pytest.fixture(scope='module')
def params(tmp_path_factory):
    """A PALEOS Earth core (Gaussian structure on 201 nodes) on the smoke-test mesh, with the
    fixed critical Rayleigh number 450 of the CMB layer: these tests hold the ledger, not the law."""
    core, _ = write_core(tmp_path_factory.mktemp('core') / 'core.npz', n=201)
    return dict(
        profile_source='paleos',
        core_liquid_table=str(CUT),
        core_profile_file=str(core),
        ds_fusion=172.8,
        icn_width=10.0,
        q_radio=0.0,
        ra_crit_cmb=450.0,
    )


def _call(shared_eos, params, start, end, state, sf, t_core=None):
    """One CVODE call from ``state`` (T_core, S) at tolerance scale ``sf``."""
    solver = _build('core_module', shared_eos, dict(params), solver_method='cvode')
    n, p_cmb = solver._n_stag, solver._P_basic_flat[:1]
    if state is None:
        s0 = np.full(n, float(shared_eos.liquidus_entropy(p_cmb)[0]) + 300.0)
        state = (float(shared_eos.temperature(p_cmb, s0[:1])[0]), s0)
    solver.parameters.solver.start_time, solver.parameters.solver.end_time = start, end
    solver.set_initial_core_temperature(state[0] if t_core is None else t_core)
    solver.set_initial_entropy(state[1])
    solver._atol_sf = sf
    solver.parameters.solver.rtol *= sf
    solver.solve()
    y = np.asarray(solver._solution.y)
    out = solver.get_state()
    cmb = out.step_dE_F_cmb_J
    res = ((out.step_dE_core_J + cmb) / cmb, out.step_core_tol_J / cmb)
    return res, (float(y[n + 1, -1]), y[:n, -1]), solver


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_tabulated_core_ledger_converges_with_the_tolerance(
    shared_eos, params, validation_plot
):
    """A zero-flux start in 5 calls (0-1, 1-2, 2-12, 12-22, 22-122 yr). Call 1 closes to 3e-4
    at the default tolerances (measured -1.3e-4) and below 3e-6 at 1000 times tighter ones
    (-1.2e-6); calls 1 to 3 fall at least 20 times to below 1e-5 (measured more than 100
    times with SUNDIALS 7.2.1 and 7.7.0); calls 4 and 5 close to 2e-6 (-1.4e-7, 1.3e-6) and
    fall at least 20 times. The residuals depend on the CPU and the SUNDIALS build, so the
    plot shows the bound of each call, not the residual."""
    res, scale = {}, {}
    for sf in (1.0, 1.0e-3):
        state, res[sf], scale[sf] = None, [], []
        for start, end in itertools.pairwise((0.0, 1.0, 2.0, 12.0, 22.0, 122.0)):
            (r, s), state, solver = _call(shared_eos, params, start, end, state, sf)
            res[sf].append(r)
            scale[sf].append(s)
        assert solver._core_module_budget.tabulated
    assert 1e-5 < abs(res[1.0][0]) < 3e-4 and abs(res[1.0e-3][0]) < 3e-6, res
    for loose, tight in zip(res[1.0][3:], res[1.0e-3][3:]):
        assert abs(loose) < 2e-6 and abs(loose) > 20.0 * abs(tight), res
    for values in zip(res[1.0][:3], res[1.0e-3][:3], scale[1.0][:3], scale[1.0e-3][:3]):
        _assert_converges(*values)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        calls, nan = np.arange(1, 6), np.nan
        ax.semilogy(calls, [3e-4, nan, nan, 2e-6, 2e-6], 'o', label='default, passed')
        ax.semilogy(calls, [3e-6, 1e-5, 1e-5, nan, nan], 's', label='1000x tighter, passed')
        ax.set(xlabel='Call', ylabel='Residual bound / CMB heat', xticks=calls)
        ax.set_ylim(1e-6, 1e-3)
        ax.minorticks_off()
        legend_outside(fig, ax)
        return fig

    validation_plot(figure, 'core_tabulated_ledger_per_call')


@pytest.mark.physics_invariant
def test_the_tabulated_core_ledger_closes_under_the_eq16_law(shared_eos, params):
    """The first call of the zero-flux start under Thiriet et al. (2019) eq. 16. It books 9.2
    times less CMB heat than with 450 and closes 2.1 times better in J (measured -2.3e20 against
    -4.7e20 J), so its residual per CMB heat is larger: -5.7e-4 at the default tolerances and
    -4.4e-6 at 1000 times tighter ones (bounds 1e-3 and 1e-5, a fall of at least 20)."""
    eq16 = {k: v for k, v in params.items() if k != 'ra_crit_cmb'}
    res = {}
    for sf in (1.0, 1.0e-3):
        (res[sf], _), _, solver = _call(shared_eos, eq16, 0.0, 1.0, None, sf)
        assert solver._core_module_ra_crit_cmb is None
    assert 1e-5 < abs(res[1.0]) < 1e-3 and abs(res[1.0e-3]) < 1e-5, res
    assert abs(res[1.0]) > 20.0 * abs(res[1.0e-3]), res


@pytest.mark.physics_invariant
def test_a_tabulated_call_across_the_onset_converges_with_the_tolerance(shared_eos, params):
    """A 3 yr call from 0.5 K above the onset to below it: its residual falls at least 20
    times to below 1e-5 at 1000 times tighter tolerances. The melting curve is a quadratic
    fit to the isentrope through 300 K above the mantle base, 5 % below it at the CMB, so the
    onset lies near there and the core grows from the centre."""
    probe = _build('core_module', shared_eos, params)
    n, p_cmb = probe._n_stag, probe._P_basic_flat[:1]
    s = np.full(n, float(shared_eos.liquidus_entropy(p_cmb)[0]) + 300.0)
    t_m = float(shared_eos.temperature(p_cmb, s[:1])[0])
    prof = probe._core_module_budget.profiles
    r = np.linspace(0.0, prof.r_cmb, 64)
    p = np.asarray(prof.pressure(r))
    tilt = 1.0 - 0.05 * (p[0] - p) / (p[0] - p[-1])
    c2, c1, c0 = np.polyfit(p, np.asarray(prof.adiabat(r, t_m + 300.0)) * tilt, 2)
    onset_params = params | dict(melting_curve='quadratic', t_m0=c0, t_m1=c1 / c0, t_m2=c2 / c0)
    values = [[], []]
    for sf in (1.0, 1.0e-3):
        solver = _build('core_module', shared_eos, onset_params, solver_method='cvode')
        budget = solver._core_module_budget
        assert abs(budget.t_onset - (t_m + 300.0)) < 10.0
        (r, sc), state, _ = _call(
            shared_eos, onset_params, 0.0, 3.0, (None, s), sf, t_core=budget.t_onset + 0.5
        )
        assert state[0] < budget.t_onset
        values[0].append(r)
        values[1].append(sc)
    _assert_converges(*values[0], *values[1])
