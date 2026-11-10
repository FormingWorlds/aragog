"""The EOS table guard: with the rheology off, a call that leaves the entropy tables fails.

A call whose accepted cell entropy leaves the global entropy range of the loaded EOS by more than
the guard tolerance ends with a failed status and holds its states from before the first one
outside, so a caller can retry it with a shorter step. The guard is a safety net; it does not set
the step and it judges only the global range, not the padded rows of a pressure column.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from aragog.rheology import SolidRheologyParams
from aragog.solver import entropy_solver as es
from aragog.solver.entropy_solver import EntropySolver

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos
from .test_eos_floor import _UNSET, CVODE_BDF, _drained, needs_cvode
from .test_eos_table_guard_decision import _N_CELLS, _TOL, _sol
from .test_phi_step_cap_armed_smoke import _build_mushy_parameters

pytestmark = [pytest.mark.smoke, needs_eos]

_SHIFT = -400.0  # entropy zero of the shifted table set [J/kg/K]
_LOWT_DIR = os.environ.get('ARAGOG_TEST_LOWT_EOS_DIR')


@pytest.fixture(scope='module')
def eos():
    return entropy_eos_copy(EOS_DIR)


def _shifted_tables(dst: Path) -> Path:
    """Copy the tables with every entropy moved by ``_SHIFT``: another range, the same physics."""
    for src in Path(EOS_DIR).glob('[!.]*.dat'):  # the CI archive holds ._*.dat metadata files
        header = [ln for ln in src.read_text().splitlines() if ln.startswith('#')]
        s_scale = float(header[-1].split()[2])
        data = np.loadtxt(src)
        data[:, 1] += _SHIFT / s_scale
        np.savetxt(dst / src.name, data, header='\n'.join(h[2:] for h in header), fmt='%.18e')
    return dst


@pytest.mark.parametrize('margin', [_UNSET, 0.0, -50.0], ids=['default', 'zero', 'negative'])
@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_drain_past_the_table_minimum_fails_the_call(eos, caplog, method, margin):
    """With the floor off, a drain below S_min of the loaded table ends with status -1, a logged
    message naming the guard and the table range, zero energy integrals, and no state outside."""
    s, _ = _drained(eos, float(eos.S_min) + 400.0, margin=margin, method=method)
    sol = s._solution
    assert sol.status == -1 and s.stop_early
    assert 'EOS table guard' in sol.message and f'{float(eos.S_min):.3f}' in sol.message
    assert any(
        'EOS table guard' in r.getMessage() for r in caplog.records if r.levelname == 'ERROR'
    )
    assert 0.0 < sol.t[-1] < 10.0 and sol.t.size == sol.y.shape[1] > 1
    assert sol.y[: s._n_stag].min() >= float(eos.S_min) - _TOL
    np.testing.assert_equal(sol.energy_integrals, dict(es._ZERO_ENERGY_INTEGRALS))
    out = s.get_state()
    assert out.status == -1 and out.cvode_flag_name == 'EOS_TABLE_GUARD'
    assert out.cvode_flag in (0, 2)  # PROTEUS sends a flag of -1 to its stiffness recovery
    assert sol.success is False and 'energy_trace' not in sol and 'energy_change' not in sol
    assert out.dt_actual == pytest.approx(sol.t[-1])


@pytest.mark.parametrize('core', ['energy_balance', 'gradient', 'bower2018', 'core_module'])
@pytest.mark.parametrize('method', CVODE_BDF)
def test_the_failure_holds_on_each_core_mode(eos, core, method):
    """The core modes with their own state layout end the same drain in the same way, and the
    result holds as many times as state columns, the last one inside the tables."""
    s, _ = _drained(eos, float(eos.S_min) + 400.0, margin=0.0, method=method, core=core)
    sol = s._solution
    assert sol.status == -1 and 'EOS table guard' in sol.message and sol.t[-1] < 10.0
    assert sol.t.size == sol.y.shape[1] > 1
    cells = np.column_stack([s._stag_entropy(y) for y in sol.y.T])
    assert cells.min() >= float(eos.S_min) - _TOL
    assert s._stag_entropy(sol.y[:, -1]).min() >= float(eos.S_min) - _TOL


@needs_cvode
def test_a_call_that_failed_already_keeps_its_own_failure(eos, monkeypatch):
    """An integrator failure is not judged: its flag name and message stay the integrator's."""
    monkeypatch.setattr(
        EntropySolver, '_eos_table_failure', lambda self, sol: pytest.fail('judged')
    )
    p = _build_mushy_parameters(solver_method='cvode', end_time=10.0)
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s._max_steps = 1
    s.set_initial_entropy(float(eos.S_min) + 400.0)
    s.solve()
    assert s._solution.status == -1 and s._solution.cvode_flag_name == 'TOO_MUCH_WORK'


def test_a_strict_range_eos_stops_the_drain_before_the_guard():
    """With ``strict_range`` the lookup raises; on a scipy path the error reaches the caller."""
    strict = entropy_eos_copy(EOS_DIR, strict_range=True)
    with pytest.raises(RuntimeError, match='outside table range'):
        _drained(strict, float(strict.S_min) + 400.0, margin=0.0, method='bdf')


@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_retry_with_a_shorter_span_passes_inside_the_table(eos, method):
    """The call that fails over 10 yr passes over 2 yr from the same start."""
    S0 = float(eos.S_min) + 400.0
    assert _drained(eos, S0, margin=0.0, method=method)[0]._solution.status == -1
    sol = _drained(eos, S0, margin=0.0, method=method, end_time=2.0)[0]._solution
    assert sol.status == 0 and sol.t[-1] == pytest.approx(2.0)
    assert sol.energy_integrals['F_int'] != 0.0


@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_retry_on_the_same_solver_equals_a_fresh_call(eos, method):
    """The PROTEUS sequence after a failure: a shorter end time, reset, the entry state and the
    compression term back. The result is bit-equal to a fresh solver over the shorter span. A
    direct user who calls reset alone on this unchanged mesh gets a compression term of 0."""
    S0 = float(eos.S_min) + 400.0
    fresh = _drained(eos, S0, margin=0.0, method=method, end_time=2.0)[0]._solution
    s, _ = _drained(eos, S0, margin=0.0, method=method)
    assert s._solution.status == -1
    entry = np.full(s._n_stag, S0)
    s.parameters.solver.end_time = 2.0
    s.reset()
    assert s._last_compression_J == 0.0
    s._S0, s._last_compression_J = entry.copy(), 0.0
    s.solve()
    sol = s._solution
    assert (
        sol.status == 0 and not s.stop_early and sol.get('cvode_flag_name') != 'EOS_TABLE_GUARD'
    )
    assert np.array_equal(sol.t, fresh.t) and np.array_equal(sol.y, fresh.y)
    np.testing.assert_equal(sol.energy_integrals, fresh.energy_integrals)


@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_start_outside_the_table_names_the_call_start(eos, method):
    """A start outside fails at any span, also when the call heats back into the table: the
    message says that a shorter step does not help, and the result holds the start."""
    S0 = np.full(_N_CELLS, float(eos.S_min) + 400.0)
    S0[-1] = float(eos.S_min) - 0.5
    for t0, span in ((0.0, 10.0), (0.0, 1.0e-3), (5.0, 1.0)):
        s, _ = _drained(
            eos, S0, flux=-1.0e6, margin=0.0, end_time=span, start_time=t0, method=method
        )
        sol = s._solution
        assert sol.status == -1 and 'at the call start' in sol.message
        assert 'shorter step does not help' in sol.message
        assert sol.t.size == 1 and sol.t[0] == t0


@pytest.mark.parametrize('edge', ['S_min', 'S_max'])
@pytest.mark.parametrize('method', [*CVODE_BDF, 'radau'])
def test_a_start_at_a_table_edge_passes_a_short_call(eos, method, edge):
    """With no surface flux a state at an edge moves past it at a small rate, inside the tolerance
    over 10 yr, on all 3 integrators."""
    s, _ = _drained(eos, float(getattr(eos, edge)), flux=0.0, margin=0.0, method=method)
    assert s._solution.status == 0


@pytest.mark.parametrize('edge', ['S_min', 'S_max'])
@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_start_at_a_table_edge_fails_a_long_call(eos, method, edge):
    """The same state fails once its distance from the edge passes 1e-3 J/kg/K, before 1e5 yr."""
    S0 = float(getattr(eos, edge))
    sol = _drained(eos, S0, flux=0.0, margin=0.0, method=method, end_time=1.0e5)[0]._solution
    assert sol.status == -1 and 'EOS table guard' in sol.message


@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_drain_with_the_rheology_on_raises_its_invariant(eos, method):
    """The rheology-on path keeps its raise: the guard does not replace it."""
    p = _build_mushy_parameters(solver_method=method, end_time=10.0)
    p.boundary_conditions.outer_boundary_condition = 4
    p.boundary_conditions.outer_boundary_value = 1.0e6
    p.phase_solid = dataclasses.replace(
        p.phase_solid, rheology=SolidRheologyParams(enabled=True)
    )
    s = EntropySolver(p, entropy_eos=eos)
    s.initialize()
    s.set_initial_entropy(float(eos.S_min) + 400.0)
    with pytest.raises(RuntimeError, match='rheology invariant violated'):
        s.solve()


@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_rise_past_the_table_maximum_fails_the_call(eos, method):
    """The guard holds on the S_max side too: a surface heated from above leaves the table."""
    s, _ = _drained(eos, float(eos.S_max) - 5.0, flux=-1.0e6, margin=0.0, method=method)
    sol = s._solution
    assert sol.status == -1 and f'{float(eos.S_max):.3f}' in sol.message


@pytest.mark.parametrize('method', CVODE_BDF)
def test_the_guard_reads_its_edges_from_the_loaded_tables(eos, tmp_path, method):
    """The same drain on a table set with another range fails against that range: no number of
    the standard tables enters, so new tables need no edit."""
    shifted = entropy_eos_copy(_shifted_tables(tmp_path))
    assert float(shifted.S_min) == pytest.approx(float(eos.S_min) + _SHIFT, abs=1.0e-6)
    assert float(shifted.S_max) == pytest.approx(float(eos.S_max) + _SHIFT, abs=1.0e-6)
    s, _ = _drained(shifted, float(shifted.S_min) + 400.0, margin=0.0, method=method)
    assert s._solution.status == -1 and f'{float(shifted.S_min):.3f}' in s._solution.message
    probe = EntropySolver(_build_mushy_parameters(solver_method=method), entropy_eos=shifted)
    probe.initialize()
    # below the standard minimum but inside this range; inside the standard range but above this one
    assert probe._eos_table_failure(_sol(np.full(_N_CELLS, float(eos.S_min) - 10.0))) is None
    failure = probe._eos_table_failure(_sol(np.full(_N_CELLS, float(eos.S_max) - 10.0)))
    assert failure is not None and f'{float(shifted.S_max):.3f}' in failure[1]


@pytest.mark.skipif(not _LOWT_DIR, reason='ARAGOG_TEST_LOWT_EOS_DIR not set')
def test_a_drain_on_the_low_t_extended_tables_fails_at_their_minimum():
    """A low-temperature extended set (ARAGOG_TEST_LOWT_EOS_DIR) moves the edge the guard uses."""
    lowt = entropy_eos_copy(_LOWT_DIR)
    assert (
        0.0 < float(lowt.S_min) < 106.0
    )  # the PALEOS table without the extension ends at 106.1
    s, _ = _drained(lowt, float(lowt.S_min) + 400.0, margin=0.0, method='bdf')
    assert s._solution.status == -1 and f'{float(lowt.S_min):.3f}' in s._solution.message


@needs_cvode
def test_the_jax_factory_path_fails_the_call():
    """The production path, CVODE with the JAX factory, ends the same drain in the same way."""
    pytest.importorskip('jax')
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    from .test_rheology_solver_wiring import (
        _build_jax_bundle,
        _make_abe_solid_params,
        _make_solver,
    )

    p = _make_abe_solid_params(SolidRheologyParams(enabled=False), number_of_nodes=20)
    p = dataclasses.replace(
        p,
        energy=dataclasses.replace(p.energy, use_jax_jacobian=True),
        solver=dataclasses.replace(p.solver, end_time=1.0e6),
        boundary_conditions=dataclasses.replace(
            p.boundary_conditions, outer_boundary_condition=4, outer_boundary_value=1.0e6
        ),
    )
    s = _make_solver(p, identity_alpha=False)
    s.set_initial_entropy(float(s.entropy_eos.S_min) + 400.0)
    mesh_jax, eos_jax, pp, args = _build_jax_bundle(s)
    calls = []

    def factory(scales, core_bc_mode):
        calls.append(core_bc_mode)
        rhs, jac, _ = build_jax_rhs_and_jacobian(
            eos_jax=eos_jax,
            phase_params=pp,
            mesh_arrays=mesh_jax,
            boundary_params=args[3],
            heating_array=np.zeros(s._n_stag),
            scales=scales,
            core_bc_mode=core_bc_mode,
        )
        return rhs, jac

    s.set_jax_cvode_factory(factory, phase_params=pp)
    s.solve()
    assert calls and s._cvode_step_powers_batch is not None
    assert s._solution.status == -1 and 'EOS table guard' in s._solution.message


def test_a_refused_floor_call_keeps_its_refusal(eos):
    """With the floor on, a start below the table keeps status 1 and the floor message."""
    s, _ = _drained(eos, float(eos.S_min) - 10.0, margin=100.0, method='bdf')
    assert s._solution.status == 1 and 'EOS floor' in s._solution.message


@pytest.mark.parametrize('rheology', [False, True], ids=['rheology-off', 'rheology-on'])
@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_call_inside_the_table_is_bit_equal_with_and_without_the_guard(
    eos, monkeypatch, method, rheology
):
    """The guard writes nothing on a state inside the table: same times, states and integrals."""

    def run():
        p = _build_mushy_parameters(solver_method=method, end_time=2.0)
        if rheology:
            p.phase_solid = dataclasses.replace(
                p.phase_solid, rheology=SolidRheologyParams(enabled=True)
            )
        s = EntropySolver(p, entropy_eos=eos)
        s.initialize()
        s.set_initial_entropy(0.5 * (float(eos.S_min) + float(eos.S_max)))
        s.solve()
        return s._solution

    a = run()
    monkeypatch.setattr(EntropySolver, '_eos_table_failure', lambda self, sol: None)
    b = run()
    assert a.status == b.status and a.status in (0, 1)
    assert np.array_equal(a.t, b.t) and np.array_equal(a.y, b.y)
    np.testing.assert_equal(a.energy_integrals, b.energy_integrals)
