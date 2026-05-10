"""Initial condition configuration."""

from __future__ import annotations

import logging

import attrs
import numpy.typing as npt

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


@attrs.define
class InitialConditionConfig:
    """Initial condition parameters.

    Parameters
    ----------
    initial_condition : int
        1: Linear profile, 2: User-defined from file, 3: Adiabatic profile.
    surface_temperature : float
        Surface temperature [K].
    basal_temperature : float
        Basal temperature [K].
    init_file : str
        Path to user-defined temperature file (for IC type 2).
    """

    initial_condition: int = 1
    surface_temperature: float = 4000.0
    basal_temperature: float = 4000.0
    init_file: str = ''

    # In-memory temperature array. Populated from ``init_file`` at parse
    # time when ``initial_condition == 2``; users do NOT set this
    # directly. ``init_file`` is the user-supplied path; ``init_temperature``
    # is the loaded numpy array used by the solver.
    init_temperature: npt.NDArray | None = attrs.field(init=False, default=None)
