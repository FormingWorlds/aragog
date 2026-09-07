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
    cvode_output_points : int
        Number of points on the CVODE dense output grid.
    """

    start_time: float
    end_time: float
    atol: float
    rtol: float
    tsurf_poststep_change: float = 30.0
    # Not the live config path: runtime validation uses
    # aragog.parser._SolverParameters, not this attrs schema.
    cvode_output_points: int = attrs.field(
        default=65,
        validator=attrs.validators.and_(
            attrs.validators.instance_of(int),
            attrs.validators.ge(2),
        ),
    )
