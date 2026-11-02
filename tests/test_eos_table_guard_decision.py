"""The decision of the EOS table guard on given states, with no solve."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import OptimizeResult

from aragog.rheology import SolidRheologyParams
from aragog.solver.entropy_solver import EntropySolver

from .conftest import EOS_DIR, entropy_eos_copy, needs_eos
from .test_phi_step_cap_armed_smoke import _build_mushy_parameters

pytestmark = [pytest.mark.unit, needs_eos]

_TOL = 1.0e-3  # the documented guard tolerance [J/kg/K]
_N_CELLS = 11  # cells of _build_mushy_parameters


@pytest.fixture(scope='module')
def eos():
    return entropy_eos_copy(EOS_DIR)


@pytest.fixture(scope='module')
def solver(eos):
    s = EntropySolver(_build_mushy_parameters(solver_method='bdf'), entropy_eos=eos)
    s.initialize()
    return s


def _sol(cells: np.ndarray, t=(0.0, 1.0)) -> OptimizeResult:
    """A two-point call whose accepted states both hold ``cells``."""
    return OptimizeResult(
        t=np.asarray(t), y=np.column_stack([cells, cells]), status=0, message=''
    )


@settings(max_examples=200, deadline=None)
@given(
    outward=st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.sampled_from([0.0, 5.0e-4, 9.9e-4, 1.01e-3, 2.0e-3, 1.0e6]),
    ),
    side=st.sampled_from(['min', 'max']),
    cell=st.integers(0, _N_CELLS - 1),
)
def test_the_failure_follows_the_range_and_tolerance_on_every_input(
    solver, outward, side, cell
):
    """A non-finite cell, or a cell past an edge by more than 1e-3 J/kg/K, fails the call; a cell
    within it does not, whichever cell it is. ``outward`` is the distance past the edge."""
    eos = solver.entropy_eos
    span = float(eos.S_max) - float(eos.S_min)
    value = float(eos.S_min) - outward if side == 'min' else float(eos.S_max) + outward
    cells = np.full(_N_CELLS, float(eos.S_min) + 0.5 * span)
    cells[cell] = value
    fails = not (float(eos.S_min) - _TOL <= value <= float(eos.S_max) + _TOL)
    assert (solver._eos_table_failure(_sol(cells)) is not None) == fails


def test_a_late_step_outside_the_table_fails_even_if_the_end_state_is_inside(solver):
    """Every accepted state counts, not only the end of the call."""
    good = np.full(_N_CELLS, float(solver.entropy_eos.S_min) + 400.0)
    y = np.column_stack([good, good, good])
    y[-1, 1] = float(solver.entropy_eos.S_min) - 1.0
    sol = OptimizeResult(t=np.array([0.0, 1.0, 2.0]), y=y, status=0, message='')
    t_bad, message = solver._eos_table_failure(sol)
    assert t_bad == 1.0 and 'cell 10 ' in message and 'call start' not in message
    trace = OptimizeResult(
        t=np.array([0.0, 2.0]), y=y[:, [0, 2]], status=0, message='', energy_trace=(sol.t, y)
    )
    assert solver._eos_table_failure(trace)[0] == 1.0


def test_no_failure_for_a_call_that_does_not_advance_or_holds_no_table(solver, eos):
    """A refused or zero-span call, the rheology-on path and a run without tables stay as they are."""
    bad = np.full(_N_CELLS, float(eos.S_min) - 50.0)
    assert solver._eos_table_failure(_sol(bad, t=(3.0, 3.0))) is None
    assert solver._eos_table_failure(OptimizeResult(t=np.array([1.0]), y=bad[:, None])) is None
    assert (
        solver._eos_table_failure(OptimizeResult(t=np.array([]), y=bad[:, None][:, :0])) is None
    )
    on = EntropySolver(_build_mushy_parameters(solver_method='bdf'), entropy_eos=eos)
    on.parameters.phase_solid = dataclasses.replace(
        on.parameters.phase_solid, rheology=SolidRheologyParams(enabled=True)
    )
    on.initialize()
    assert on._eos_table_failure(_sol(bad)) is None
    const = EntropySolver(_build_mushy_parameters(solver_method='bdf'), entropy_eos=eos)
    const.parameters.phase_mixed = dataclasses.replace(
        const.parameters.phase_mixed, const_properties=True
    )
    assert const._eos_table_failure(_sol(bad)) is None
    no_eos = EntropySolver.__new__(EntropySolver)
    no_eos.entropy_eos = None
    assert no_eos._eos_table_failure(_sol(bad)) is None
