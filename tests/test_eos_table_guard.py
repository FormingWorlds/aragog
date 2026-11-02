"""The EOS table guard: with the rheology off, a call that leaves the entropy tables fails.

A call whose accepted cell entropy leaves the table range of the loaded EOS by more than the
guard tolerance ends with a failed status, so a caller can retry it with a shorter step. The
guard is a safety net against a state the tables do not describe; it does not set the step.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.optimize import OptimizeResult

from aragog.rheology import SolidRheologyParams
from aragog.solver import entropy_solver as es
from aragog.solver.entropy_solver import EntropySolver

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos
from .test_eos_floor import _UNSET, CVODE_BDF, _drained
from .test_phi_step_cap_armed_smoke import _build_mushy_parameters

pytestmark = [pytest.mark.smoke, needs_eos]

_SHIFT = -400.0  # entropy zero of the shifted table set [J/kg/K]
_LOWT_DIR = os.environ.get('ARAGOG_TEST_LOWT_EOS_DIR')


@pytest.fixture(scope='module')
def eos():
    return entropy_eos_copy(EOS_DIR)


@pytest.fixture(scope='module')
def solver(eos):
    s = EntropySolver(_build_mushy_parameters(solver_method='bdf'), entropy_eos=eos)
    s.initialize()
    return s


def _shifted_tables(dst: Path) -> Path:
    """Copy the tables with every entropy moved by ``_SHIFT``: another range, the same physics."""
    for src in Path(EOS_DIR).glob('*.dat'):
        header = [ln for ln in src.read_text().splitlines() if ln.startswith('#')]
        s_scale = float(header[-1].split()[2])
        data = np.loadtxt(src)
        data[:, 1] += _SHIFT / s_scale
        np.savetxt(dst / src.name, data, header='\n'.join(h[2:] for h in header), fmt='%.18e')
    return dst


def _sol(cells: np.ndarray, t=(0.0, 1.0)) -> OptimizeResult:
    """A two-point call whose accepted states both hold ``cells``."""
    return OptimizeResult(
        t=np.asarray(t), y=np.column_stack([cells, cells]), status=0, message=''
    )


@pytest.mark.parametrize('margin', [_UNSET, 0.0, -50.0], ids=['default', 'zero', 'negative'])
@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_drain_past_the_table_minimum_fails_the_call(eos, caplog, method, margin):
    """With the floor off, a drain below S_min of the loaded table ends with status -1, a logged
    message naming the guard and the table range, and zero energy integrals."""
    s, _ = _drained(eos, float(eos.S_min) + 400.0, margin=margin, method=method)
    sol = s._solution
    assert sol.status == -1 and s.stop_early
    assert 'EOS table guard' in sol.message and f'{float(eos.S_min):.3f}' in sol.message
    assert sol.cvode_flag_name == 'EOS_TABLE_GUARD'
    assert sol.energy_integrals == dict(es._ZERO_ENERGY_INTEGRALS)
    assert any(
        'EOS table guard' in r.getMessage() for r in caplog.records if r.levelname == 'ERROR'
    )


@pytest.mark.parametrize('method', CVODE_BDF)
def test_a_rise_past_the_table_maximum_fails_the_call(eos, method):
    """The guard holds on the S_max side too: a surface heated from above leaves the table."""
    s, _ = _drained(eos, float(eos.S_max) - 5.0, flux=-1.0e6, margin=0.0, method=method)
    sol = s._solution
    assert sol.status == -1 and f'{float(eos.S_max):.3f}' in sol.message


@pytest.mark.parametrize('method', CVODE_BDF)
def test_the_guard_reads_its_edges_from_the_loaded_tables(tmp_path, method):
    """The same drain on a table set with another range fails against that range: no number of
    the standard tables enters, so new PALEOS tables need no edit."""
    shifted = entropy_eos_copy(_shifted_tables(tmp_path))
    base = entropy_eos_copy(EOS_DIR)
    assert float(shifted.S_min) == pytest.approx(float(base.S_min) + _SHIFT, abs=1.0e-6)
    s, _ = _drained(shifted, float(shifted.S_min) + 400.0, margin=0.0, method=method)
    assert s._solution.status == -1 and f'{float(shifted.S_min):.3f}' in s._solution.message
    # a cell inside the standard range lies below this one: only the loaded range counts
    probe = EntropySolver(_build_mushy_parameters(solver_method=method), entropy_eos=shifted)
    probe.initialize()
    cells = np.full(probe._n_stag, float(base.S_min) - 10.0)
    assert probe._eos_table_failure(_sol(cells)) is None


@pytest.mark.skipif(not _LOWT_DIR, reason='ARAGOG_TEST_LOWT_EOS_DIR not set')
def test_a_drain_on_the_low_t_extended_tables_fails_at_their_minimum():
    """A low-temperature extended set (ARAGOG_TEST_LOWT_EOS_DIR) moves the edge the guard uses."""
    lowt = entropy_eos_copy(_LOWT_DIR)
    assert float(lowt.S_min) < 106.0  # the table without the extension ends at 106.1
    s, _ = _drained(lowt, float(lowt.S_min) + 400.0, margin=0.0, method='bdf')
    assert s._solution.status == -1 and f'{float(lowt.S_min):.3f}' in s._solution.message


def test_the_jax_factory_path_fails_the_call(eos):
    """The production path, CVODE with the JAX factory, ends the same drain in the same way."""
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian

    from .test_rheology_solver_wiring import (
        _build_jax_bundle,
        _make_abe_solid_params,
        _make_solver,
    )

    p = _make_abe_solid_params(SolidRheologyParams(enabled=False), number_of_nodes=20)
    p = dataclasses.replace(p, energy=dataclasses.replace(p.energy, use_jax_jacobian=True))
    p = dataclasses.replace(
        p,
        solver=dataclasses.replace(p.solver, end_time=1.0e6),
        boundary_conditions=dataclasses.replace(
            p.boundary_conditions, outer_boundary_condition=4, outer_boundary_value=1.0e6
        ),
    )
    s = _make_solver(p, identity_alpha=False)
    s.set_initial_entropy(float(s.entropy_eos.S_min) + 400.0)
    mesh_jax, eos_jax, pp, args = _build_jax_bundle(s)

    def factory(scales, core_bc_mode):
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
    assert s._cvode_step_powers_batch is not None or s._jax_cvode_factory is not None
    assert s._solution.status == -1 and 'EOS table guard' in s._solution.message


@settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    offset=st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.sampled_from([-2.0e-3, -1.0e-3, -5.0e-4, 0.0, 5.0e-4, 1.0e-3, 2.0e-3]),
    ),
    side=st.sampled_from(['min', 'max']),
)
def test_the_failure_follows_the_range_and_tolerance_on_every_input(solver, offset, side):
    """Any non-finite cell, or a cell past an edge by more than the tolerance, fails the call;
    a cell within the tolerance does not."""
    eos, tol = solver.entropy_eos, es._EOS_TABLE_TOL
    edge = float(eos.S_min) if side == 'min' else float(eos.S_max)
    value = edge + offset
    cells = np.full(solver._n_stag, 0.5 * (float(eos.S_min) + float(eos.S_max)))
    cells[3] = value
    inside = np.isfinite(value) and float(eos.S_min) - tol <= value <= float(eos.S_max) + tol
    assert (solver._eos_table_failure(_sol(cells)) is None) == bool(inside)


def test_a_late_step_outside_the_table_fails_even_if_the_end_state_is_inside(solver):
    """Every accepted state counts, not only the end of the call."""
    good = np.full(solver._n_stag, float(solver.entropy_eos.S_min) + 400.0)
    y = np.column_stack([good, good, good])
    y[-1, 1] = float(solver.entropy_eos.S_min) - 1.0
    sol = OptimizeResult(t=np.array([0.0, 1.0, 2.0]), y=y, status=0, message='')
    assert 'cell 10 ' in solver._eos_table_failure(sol)
    trace = OptimizeResult(
        t=np.array([0.0, 2.0]), y=y[:, [0, 2]], status=0, message='', energy_trace=(sol.t, y)
    )
    assert solver._eos_table_failure(trace) is not None


def test_no_failure_for_a_call_that_does_not_advance_or_holds_no_table(solver, eos):
    """A refused or zero-span call, the rheology-on path and a run without tables stay as they are."""
    bad = np.full(solver._n_stag, float(eos.S_min) - 50.0)
    assert solver._eos_table_failure(_sol(bad, t=(3.0, 3.0))) is None
    assert solver._eos_table_failure(OptimizeResult(t=np.array([1.0]), y=bad[:, None])) is None
    on = EntropySolver(_build_mushy_parameters(solver_method='bdf'), entropy_eos=eos)
    on.parameters.phase_solid = dataclasses.replace(
        on.parameters.phase_solid, rheology=SolidRheologyParams(enabled=True)
    )
    on.initialize()
    assert on._eos_table_failure(_sol(bad)) is None
    no_eos = EntropySolver.__new__(EntropySolver)
    no_eos.entropy_eos = None
    assert no_eos._eos_table_failure(_sol(bad)) is None


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
    assert a.energy_integrals == b.energy_integrals
