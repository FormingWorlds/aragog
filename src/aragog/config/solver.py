"""Solver configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from aragog.config.scalings import ScalingsConfig

logger: logging.Logger = logging.getLogger(__name__)


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
    event_triggering : bool
        Enable event-based early stopping.
    """

    start_time: float
    end_time: float
    atol: float
    rtol: float
    tsurf_poststep_change: float = 30.0
    event_triggering: bool = False
    scalings_: ScalingsConfig | None = attrs.field(init=False, default=None)

    def scale_attributes(self, scalings: ScalingsConfig) -> None:
        """Apply non-dimensionalization."""
        self.scalings_ = scalings
        self.start_time /= scalings.time_years
        self.end_time /= scalings.time_years
        self.tsurf_poststep_change /= scalings.temperature
