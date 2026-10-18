"""Solve-level checks of accepted states, end states and the temperature tables."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aragog.eos import entropy as eos_mod
from aragog.solver import entropy_solver as es
from aragog.solver.entropy_solver import AcceptedStateError, EntropySolver

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos
from .test_entropy_solver_gradient_smoke import _build_gradient_parameters
from .test_eos_floor import _FLUX, _drained
from .test_phi_step_cap_armed_smoke import _build_mushy_parameters

pytestmark = [pytest.mark.smoke, needs_eos]


@pytest.fixture(scope='module')
def shared_eos():
    return entropy_eos_copy(EOS_DIR)


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_drain_without_the_floor_raises_instead_of_clamping(shared_eos, method):
    S_min = float(shared_eos.S_min)
    with pytest.raises(AcceptedStateError) as e:
        _drained(shared_eos, S_min + 400.0, margin=None, method=method)
    assert e.value.check == 'entropy_table' and e.value.bound == S_min
    assert e.value.value < S_min and 0.0 < e.value.time <= 10.0


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
@pytest.mark.parametrize('flux', [_FLUX, 0.0])
def test_a_call_inside_the_tables_does_not_raise(shared_eos, method, flux):
    """The floor keeps a drained call inside; a call without the drain stays inside."""
    s, f = _drained(shared_eos, float(shared_eos.S_min) + 400.0, flux=flux, method=method)
    assert f.min() >= float(shared_eos.S_min)


@pytest.mark.parametrize('name', ['temperature_melt.dat', 'temperature_solid.dat'])
@pytest.mark.parametrize('bad', [0.0, -5.0, np.nan, np.inf])
def test_a_temperature_table_with_a_non_positive_or_non_finite_value_is_refused(
    monkeypatch, bad, name
):
    load = eos_mod._load_spider_ps_table

    def spoiled(path, *a, **k):
        tbl = load(path, *a, **k)
        if path.name == name:
            tbl['values'] = np.array(tbl['values'], dtype=float)
            tbl['values'][3, 4] = bad
        return tbl

    monkeypatch.setattr(eos_mod, '_load_spider_ps_table', spoiled)
    with pytest.raises(ValueError, match=name.removesuffix('.dat')):
        eos_mod.EntropyEOS(EOS_DIR)


def test_get_state_checks_the_end_state(shared_eos, monkeypatch):
    s, _ = _drained(shared_eos, float(shared_eos.S_min) + 400.0, flux=0.0)
    s.get_state()
    monkeypatch.setattr(
        s.entropy_eos, 'melt_fraction', lambda P, S: np.full(np.size(S), np.nan)
    )
    with pytest.raises(AcceptedStateError) as e:
        s.get_state()
    assert e.value.check == 'end_melt_fraction'


def _solves(s):
    """True when the call solves without an accepted-state error, so a wrong raise fails an
    assertion."""
    try:
        s.solve()
    except AcceptedStateError:
        return False
    return True


def _energy_balance(eos, S0, dSdr_cmb, method='cvode', margin=None):
    """A 1e-3 yr energy_balance call with no surface flux from cells ``S0`` and the CMB entropy
    gradient ``dSdr_cmb``; returns the solver and the offset of the CMB node from cell 0."""
    p = _build_mushy_parameters(solver_method=method, end_time=1.0e-3)
    p.boundary_conditions.outer_boundary_condition = 4
    p.boundary_conditions.outer_boundary_value = 0.0
    p.boundary_conditions.core_bc = 'energy_balance'
    p.energy = dataclasses.replace(
        p.energy, phi_step_cap=None, eos_floor_margin=margin, phase_boundary_cap=None
    )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_dSdr_cmb(dSdr_cmb)
    s.set_initial_entropy(np.asarray(S0, dtype=float))
    r = np.asarray(s.evaluator.mesh.basic.radii).ravel()
    return s, r[0] - 0.5 * (r[0] + r[1])


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_the_cmb_node_of_energy_balance_is_the_model_value_not_the_extrapolation(
    shared_eos, method
):
    """Bottom cells S_min + [15, 150] with dSdr_cmb = 0: the model's CMB node is S_min + 15,
    inside the table, while the extrapolation 1.5 S0 - 0.5 S1 is below it; no raise."""
    S_min = float(shared_eos.S_min)
    S0 = np.full(11, S_min + 2000.0)
    S0[:2] = S_min + 15.0, S_min + 150.0
    s, _ = _energy_balance(shared_eos, S0, 0.0, method=method)
    assert _solves(s) and s._solution.status >= 0


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_cmb_node_below_the_table_raises_with_every_cell_inside(shared_eos, method):
    """All cells at S_min + 2000 and dSdr_cmb that puts the model's CMB node at S_min - 100: the
    call raises on basic node 0."""
    S_min = float(shared_eos.S_min)
    S0 = np.full(11, S_min + 2000.0)
    probe, off = _energy_balance(shared_eos, S0, 0.0, method=method)
    s, _ = _energy_balance(shared_eos, S0, (S_min - 100.0 - S0[0]) / off, method=method)
    with pytest.raises(AcceptedStateError) as e:
        s.solve()
    assert (e.value.check, e.value.node, e.value.bound) == ('entropy_table', 'basic 0', S_min)


def test_a_trace_node_outside_the_table_raises_when_the_returned_points_are_inside(
    shared_eos, monkeypatch
):
    """A stub adds one CVODE trace node with a mid cell 1 J/kg/K below S_min at the call end, the
    returned points staying inside: the check reads the trace and raises on that cell."""
    real = EntropySolver._solve_cvode

    def extra_node(self, start_time, end_time, y0, **kwargs):
        res = real(self, start_time, end_time, y0, **kwargs)
        if end_time > start_time:
            sc = self._build_nondim_scales()
            bad = np.asarray(res.y, dtype=float)[:, -1] * sc.state_scale + sc.state_offset
            bad[5] = float(self.entropy_eos.S_min) - 1.0
            tt, ty = res.energy_trace
            col = ((bad - sc.state_offset) / sc.state_scale)[:, np.newaxis]
            res.energy_trace = (np.append(tt, tt[-1]), np.hstack([ty, col]))
        return res

    monkeypatch.setattr(EntropySolver, '_solve_cvode', extra_node)
    with pytest.raises(AcceptedStateError) as e:
        _drained(shared_eos, float(shared_eos.S_min) + 400.0, flux=0.0, margin=None)
    assert (e.value.check, e.value.node) == ('entropy_table', 'cell 5')


def test_a_failed_call_is_not_checked(shared_eos, monkeypatch):
    """A call that fails (status -1) with a trace node below S_min returns its status to the
    caller's retry ladder instead of raising."""
    real = EntropySolver._solve_cvode

    def failed(self, start_time, end_time, y0, **kwargs):
        res = real(self, start_time, end_time, y0, **kwargs)
        if end_time > start_time:
            sc = self._build_nondim_scales()
            bad = np.asarray(res.y, dtype=float)[:, -1] * sc.state_scale + sc.state_offset
            bad[5] = float(self.entropy_eos.S_min) - 1.0
            res.y[:, -1] = (bad - sc.state_offset) / sc.state_scale
            res.energy_trace = None
            res.status = -1
        return res

    monkeypatch.setattr(EntropySolver, '_solve_cvode', failed)
    p = _build_mushy_parameters(solver_method='cvode', end_time=1.0)
    s = EntropySolver(p, entropy_eos=shared_eos)
    s.initialize()
    s.set_initial_entropy(np.full(s._n_stag, float(shared_eos.S_min) + 400.0))
    assert _solves(s) and s._solution.status == -1


def test_get_state_checks_the_basic_node_end_temperatures(shared_eos, monkeypatch):
    """A non-positive basic-node temperature in the end state raises on that node."""
    s, _ = _drained(shared_eos, float(shared_eos.S_min) + 400.0, flux=0.0)
    cls = type(s.state)
    real = cls.T_basic_diag.fget
    monkeypatch.setattr(
        cls,
        'T_basic_diag',
        property(lambda st: np.where(np.arange(real(st).size) == 3, 0.0, real(st))),
    )
    with pytest.raises(AcceptedStateError) as e:
        s.get_state()
    assert (e.value.check, e.value.node) == ('end_temperature', 'basic 3')


def test_the_gradient_core_is_not_checked(shared_eos, monkeypatch):
    """On the gradient core the state holds dS/dr values, not entropies: no per-call check."""
    calls = []
    monkeypatch.setattr(es, '_check_accepted_states', lambda *a, **k: calls.append(a))
    s = EntropySolver(
        _build_gradient_parameters(n_nodes=12, end_time=5.0), entropy_eos=shared_eos
    )
    s.initialize()
    s.set_initial_entropy(3050.0)
    s.solve()
    assert calls == []


def _bad_start(eos):
    S0 = np.full(11, float(eos.S_min) + 400.0)
    S0[5] = float(eos.S_min) - 10.0
    return S0


@pytest.mark.parametrize('method', ['cvode', 'bdf'])
def test_a_start_state_below_the_table_raises_on_the_first_call(shared_eos, method):
    """A start set by set_initial_entropy with a cell below S_min is a bad snapshot: the first
    call raises on it at the start time instead of a floor refusal."""
    with pytest.raises(AcceptedStateError) as e:
        _drained(shared_eos, _bad_start(shared_eos), flux=0.0, method=method)
    assert (e.value.check, e.value.node, e.value.time) == ('entropy_table', 'cell 5', 0.0)


def test_the_start_is_checked_only_once_after_set_initial_entropy(shared_eos):
    """After the first call the start is the previous end, already checked: a later call whose
    start is changed without set_initial_entropy gets the floor refusal, not the error."""
    s, _ = _drained(shared_eos, float(shared_eos.S_min) + 400.0, flux=0.0)
    s._S0 = _bad_start(shared_eos)
    assert _solves(s) and s._solution.status == 1 and s.get_state().dt_actual == 0.0
