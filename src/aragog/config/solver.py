"""Solver configuration."""

from __future__ import annotations

import logging

import attrs

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


@attrs.define
class SolverConfig:
    """ODE solver parameters.

    Parameters
    ----------
    start_time : float
        Start time [years].
    end_time : float
        End time [years].
    atol : float
        Absolute tolerance for BDF solver.
    rtol : float
        Relative tolerance for BDF solver.
    tsurf_poststep_change : float
        Maximum surface temperature change per step [K].
    """

    start_time: float
    end_time: float
    atol: float
    rtol: float
    tsurf_poststep_change: float = 30.0
