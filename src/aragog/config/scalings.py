"""Vestigial scaling factors.

Non-dimensionalization has been removed. All scales are set to 1.0
so that any code dividing by a scale factor becomes an identity
operation. The config file may still contain a [scalings] section;
the values are parsed but ignored.
"""

from __future__ import annotations

import logging

import attrs

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


@attrs.define
class ScalingsConfig:
    """Vestigial scalings class retained for API compatibility.

    All scales are forced to 1.0 regardless of input values.
    """

    radius: float = 1.0
    temperature: float = 1.0
    density: float = 1.0
    time: float = 1.0

    # Derived (all 1.0)
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
        # Override whatever was parsed: all scales = 1.0
        self.radius = 1.0
        self.temperature = 1.0
        self.density = 1.0
        self.time = 1.0
        self.area = 1.0
        self.gravitational_acceleration = 1.0
        self.temperature_gradient = 1.0
        self.thermal_expansivity = 1.0
        self.pressure = 1.0
        self.velocity = 1.0
        self.kinetic_energy_per_volume = 1.0
        self.heat_capacity = 1.0
        self.entropy = 1.0
        self.latent_heat_per_mass = 1.0
        self.power_per_volume = 1.0
        self.power_per_mass = 1.0
        self.heat_flux = 1.0
        self.thermal_conductivity = 1.0
        self.viscosity = 1.0
        self.time_years = 1.0
        self.stefan_boltzmann_constant = 1.0
        logger.debug('scalings = %s (all unity, non-dimensionalization removed)', self)
