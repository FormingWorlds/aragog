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
        Absolute tolerance, floored at 1e-8. Default 1e-8.
    rtol : float
        Relative tolerance for BDF solver. Default 1e-8, the value at which
        ``phase_boundary_cap = 'rate'`` is verified. Above 1e-7 a CVODE run logs one
        warning: a loose rtol can lock CVODE at the convective switch dS/dr = 0.
    tsurf_poststep_change : float
        Maximum surface temperature change per step [K].
    cvode_output_points : int
        Number of points on the CVODE dense output grid.
    max_steps : int
        Maximum number of internal CVODE steps per output interval (SUNDIALS mxstep).
    tcore_change_limit : float or None
        Optional per-solve core-temperature change limit [K]. The flag is
        also set, independent of this limit, when any sampled core
        temperature is non-finite.
    """

    start_time: float
    end_time: float
    atol: float = 1e-8
    rtol: float = 1e-8
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
    max_steps: int = attrs.field(
        default=100000,
        validator=attrs.validators.and_(
            attrs.validators.not_(attrs.validators.instance_of(bool)),
            attrs.validators.instance_of(int),
            attrs.validators.ge(1),
        ),
    )
    tcore_change_limit: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.and_(
                attrs.validators.not_(attrs.validators.instance_of(bool)),
                attrs.validators.instance_of((int, float)),
                attrs.validators.gt(0.0),
            )
        ),
    )
