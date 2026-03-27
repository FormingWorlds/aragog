"""Non-dimensionalization scaling factors.

These will be removed in a future refactoring phase when the code
switches to working in SI units throughout.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import constants
from scipy.constants import Stefan_Boltzmann

import attrs

logger: logging.Logger = logging.getLogger(__name__)


@attrs.define
class ScalingsConfig:
    """Reference scales for non-dimensionalization.

    All units are SI. Derived scales are computed from the four
    primary scales (radius, temperature, density, time).

    Parameters
    ----------
    radius : float
        Reference radius in metres.
    temperature : float
        Reference temperature in Kelvin.
    density : float
        Reference density in kg/m^3.
    time : float
        Reference time in seconds.
    """

    radius: float = 1.0
    temperature: float = 1.0
    density: float = 1.0
    time: float = 1.0

    # Derived (computed in __attrs_post_init__)
    area: float = attrs.field(init=False)
    gravitational_acceleration: float = attrs.field(init=False)
    temperature_gradient: float = attrs.field(init=False)
    thermal_expansivity: float = attrs.field(init=False)
    pressure: float = attrs.field(init=False)
    velocity: float = attrs.field(init=False)
    kinetic_energy_per_volume: float = attrs.field(init=False)
    heat_capacity: float = attrs.field(init=False)
    entropy: float = attrs.field(init=False)
    latent_heat_per_mass: float = attrs.field(init=False)
    power_per_volume: float = attrs.field(init=False)
    power_per_mass: float = attrs.field(init=False)
    heat_flux: float = attrs.field(init=False)
    thermal_conductivity: float = attrs.field(init=False)
    viscosity: float = attrs.field(init=False)
    time_years: float = attrs.field(init=False)
    stefan_boltzmann_constant: float = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self.area = np.square(self.radius)
        self.gravitational_acceleration = self.radius / np.square(self.time)
        self.temperature_gradient = self.temperature / self.radius
        self.thermal_expansivity = 1 / self.temperature
        self.pressure = self.density * self.gravitational_acceleration * self.radius
        self.velocity = self.radius / self.time
        self.kinetic_energy_per_volume = self.density * np.square(self.velocity)
        self.heat_capacity = (
            self.kinetic_energy_per_volume / self.density / self.temperature
        )
        self.entropy = self.heat_capacity
        self.latent_heat_per_mass = self.heat_capacity * self.temperature
        self.power_per_volume = self.kinetic_energy_per_volume / self.time
        self.power_per_mass = self.power_per_volume / self.density
        self.heat_flux = self.power_per_volume * self.radius
        self.thermal_conductivity = (
            self.power_per_volume * self.area / self.temperature
        )
        self.viscosity = self.pressure * self.time
        self.time_years = self.time / constants.Julian_year
        self.stefan_boltzmann_constant = Stefan_Boltzmann / (
            self.power_per_volume * self.radius / np.power(self.temperature, 4)
        )
        logger.debug("scalings = %s", self)
