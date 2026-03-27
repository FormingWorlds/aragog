"""Initial condition configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import attrs
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from aragog.config.scalings import ScalingsConfig

logger: logging.Logger = logging.getLogger(__name__)


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
    init_file: str = ""
    scalings_: ScalingsConfig | None = attrs.field(init=False, default=None)

    # Loaded temperature array (for IC type 2)
    init_temperature: npt.NDArray | None = attrs.field(init=False, default=None)

    def scale_attributes(self, scalings: ScalingsConfig) -> None:
        """Apply non-dimensionalization."""
        self.scalings_ = scalings
        self.surface_temperature /= scalings.temperature
        self.basal_temperature /= scalings.temperature

        if self.initial_condition == 2:
            if self.init_file == "":
                raise ValueError("initial_condition=2 requires an init_file")
            self.init_temperature = np.loadtxt(self.init_file)
            self.init_temperature /= scalings.temperature
