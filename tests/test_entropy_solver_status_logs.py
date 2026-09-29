"""Tests for ``EntropySolver._log_solution_outcome`` (the post-status logging that
``solve()`` runs) and the ``write_netcdf`` convenience wrapper.

* status 1: the integrator stopped on a terminal step-cap event; the solver logs
  the event time and leaves ``stop_early = False``.
* status -1: the integrator failed; the solver logs an error and sets
  ``stop_early = True``.
* ``write_netcdf`` forwards ``description`` only when supplied, so the wrapper and
  ``SolverOutput.to_netcdf`` stamp the same ``description`` attribute.

The status tests use a stub ``_solution``, so no integration runs.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit

LOGGER = 'fwl.aragog.solver.entropy_solver'


def _make_solver_with_solution(*, status: int, t_end: float, n_stag: int = 5):
    """An EntropySolver with a stub ``_solution`` holding what the status logging reads."""
    from aragog.solver.entropy_solver import EntropySolver

    solver = EntropySolver.__new__(EntropySolver)
    fake_sol = MagicMock()
    fake_sol.t = np.array([0.0, t_end], dtype=float)
    fake_sol.y = np.zeros((n_stag, 2), dtype=float)
    fake_sol.status = status
    fake_sol.message = ''
    fake_sol.nfev = 1
    solver._solution = fake_sol
    return solver, fake_sol


@pytest.mark.parametrize('status', [0, 1])
def test_successful_status_logs_and_keeps_running(caplog, status):
    solver, _ = _make_solver_with_solution(status=status, t_end=2.5e3)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        solver._log_solution_outcome(end_time=1.0e4)
    assert solver.stop_early is False
    expected = {
        0: 'EntropySolver: integration completed successfully.',
        1: 'EntropySolver: step-cap event at t=2.50e+03 yr (stopped 7500.0 yr before end_time).',
    }[status]
    assert expected in [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize('status', [-1, 2])
def test_failed_status_logs_error_and_stops(caplog, status):
    solver, sol = _make_solver_with_solution(status=status, t_end=1.0e3)
    sol.message = 'CVODE failed with flag -4'
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        solver._log_solution_outcome(end_time=1.0e4)
    assert solver.stop_early is True
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [
        f'EntropySolver: integration failed (status={status}): CVODE failed with flag -4'
    ]


def test_write_netcdf_forwards_description_only_when_supplied(tmp_path):
    """``EntropySolver.write_netcdf`` is a thin wrapper around
    ``self.get_state().to_netcdf(...)``. It must forward
    ``description`` only when the caller supplied it; omitting it
    should let ``to_netcdf``'s default take over so both entry
    points stamp the same default string into the NetCDF.

    Discriminator: a regression that always forwarded
    ``description=None`` would overwrite the default with None in
    the writer, surfacing as ``ds.description == 'None'`` in the
    output — a confusing diagnostic for users.
    """
    from aragog.solver.entropy_solver import EntropySolver

    solver = EntropySolver.__new__(EntropySolver)

    captured_kwargs: dict = {}

    class _StubState:
        def to_netcdf(self, path, **kwargs):
            captured_kwargs.update(kwargs)
            captured_kwargs['_path'] = str(path)

    solver.get_state = lambda: _StubState()  # type: ignore[method-assign]

    out = tmp_path / 'a.nc'
    solver.write_netcdf(out, time=1.0e6)
    # No description passed => not forwarded.
    assert 'description' not in captured_kwargs, (
        'wrapper should not forward description when caller omitted it; '
        f'captured kwargs = {captured_kwargs}'
    )
    assert captured_kwargs['time'] == pytest.approx(1.0e6)
    assert captured_kwargs['_path'].endswith('a.nc')

    # Reset and pass description explicitly.
    captured_kwargs.clear()
    solver.write_netcdf(out, description='end-of-run snapshot')
    assert captured_kwargs.get('description') == 'end-of-run snapshot', (
        f'description was not forwarded when supplied: {captured_kwargs}'
    )
