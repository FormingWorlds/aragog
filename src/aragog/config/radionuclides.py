"""Radionuclide configuration."""

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
class RadionuclideConfig:
    """Single radionuclide heating source.

    Parameters
    ----------
    name : str
        Isotope name (e.g., 'K40', 'U238').
    t0_years : float
        Reference time for initial abundance [years].
    abundance : float
        Isotopic abundance ratio (e.g., 40K/K).
    concentration : float
        Elemental concentration [ppm].
    heat_production : float
        Specific heat production [W/kg].
    half_life_years : float
        Half-life [years].
    """

    name: str
    t0_years: float
    abundance: float
    concentration: float
    heat_production: float
    half_life_years: float
    scalings_: ScalingsConfig | None = attrs.field(init=False, default=None)

    def scale_attributes(self, scalings: ScalingsConfig) -> None:
        """Apply non-dimensionalization."""
        self.scalings_ = scalings
        self.t0_years /= scalings.time_years
        self.concentration *= 1e-6  # ppm to mass fraction
        self.heat_production /= scalings.power_per_mass
        self.half_life_years /= scalings.time_years

    def get_heating(self, time: npt.NDArray | float) -> npt.NDArray | float:
        """Compute radiogenic heating at a given time.

        Parameters
        ----------
        time : float or ndarray
            Time (in non-dimensional units matching t0_years).

        Returns
        -------
        float or ndarray
            Heat production [non-dimensional power per unit mass].
        """
        arg = np.log(2) * (self.t0_years - time) / self.half_life_years
        return self.heat_production * self.abundance * self.concentration * np.exp(arg)
