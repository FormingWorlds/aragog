"""Abstract base classes and protocols for phase evaluators."""

import logging
from abc import ABC, abstractmethod
from typing import Protocol

import numpy.typing as npt

from aragog.utilities import FloatOrArray

logger: logging.Logger = logging.getLogger(__name__)


class PropertyProtocol(Protocol):
    """Property protocol"""

    def eval(self, *args) -> FloatOrArray: ...

    def __call__(self, temperature: npt.NDArray, pressure: npt.NDArray) -> FloatOrArray: ...


class PhaseEvaluatorProtocol(Protocol):
    """Phase evaluator protocol"""

    def set_temperature(self, temperature: npt.NDArray) -> None: ...

    def set_pressure(self, pressure: npt.NDArray) -> None: ...

    def update(self) -> None: ...

    def density(self) -> FloatOrArray: ...

    def dTdPs(self) -> npt.NDArray: ...

    def dTdrs(self) -> npt.NDArray: ...

    def gravitational_acceleration(self) -> FloatOrArray: ...

    def heat_capacity(self) -> FloatOrArray: ...

    def kinematic_viscosity(self) -> FloatOrArray: ...

    def melt_fraction(self) -> FloatOrArray: ...

    def latent_heat(self) -> FloatOrArray: ...

    def thermal_conductivity(self) -> FloatOrArray: ...

    def thermal_expansivity(self) -> FloatOrArray: ...

    def viscosity(self) -> FloatOrArray: ...

    def relative_velocity(self) -> FloatOrArray: ...

    def delta_specific_volume(self) -> FloatOrArray: ...

class MixedPhaseEvaluatorProtocol(PhaseEvaluatorProtocol, Protocol):
    """Mixed phase evaluator protocol"""

    def liquidus(self) -> npt.NDArray: ...

    def liquidus_gradient(self) -> npt.NDArray: ...

    def solidus(self) -> npt.NDArray: ...

    def solidus_gradient(self) -> npt.NDArray: ...


class PhaseEvaluatorABC(ABC):
    """Phase evaluator ABC"""

    temperature: npt.NDArray
    pressure: npt.NDArray

    def set_temperature(self, temperature: npt.NDArray) -> None:
        """Sets the temperature."""
        logger.debug("set_temperature = %s", temperature)
        self.temperature = temperature

    def set_pressure(self, pressure: npt.NDArray) -> None:
        """Sets the pressure."""
        logger.debug("set_pressure = %s", pressure)
        self.pressure = pressure

    def update(self) -> None:
        """Updates quantities to avoid repeat, possibly expensive, calculations."""

    @abstractmethod
    def density(self) -> FloatOrArray: ...

    def dTdPs(self) -> npt.NDArray:
        """TODO: Update reference to sphinx: Solomatov (2007), Treatise on Geophysics, Eq. 3.2"""
        dTdPs: npt.NDArray = (
            self.thermal_expansivity() * self.temperature / (self.density() * self.heat_capacity())
        )

        return dTdPs

    def dTdrs(self) -> npt.NDArray:
        dTdrs: npt.NDArray = (
            -self.gravitational_acceleration()
            * self.thermal_expansivity()
            * self.temperature
            / self.heat_capacity()
        )

        return dTdrs

    @abstractmethod
    def gravitational_acceleration(self) -> FloatOrArray: ...

    @abstractmethod
    def heat_capacity(self) -> FloatOrArray: ...

    def kinematic_viscosity(self) -> FloatOrArray:
        viscosity: FloatOrArray = self.viscosity() / self.density()

        return viscosity

    @abstractmethod
    def melt_fraction(self) -> FloatOrArray: ...

    @abstractmethod
    def latent_heat(self) -> FloatOrArray: ...

    @abstractmethod
    def thermal_conductivity(self) -> FloatOrArray: ...

    @abstractmethod
    def thermal_expansivity(self) -> FloatOrArray: ...

    @abstractmethod
    def viscosity(self) -> FloatOrArray: ...

    @abstractmethod
    def relative_velocity(self) -> FloatOrArray: ...

    @abstractmethod
    def delta_specific_volume(self) -> FloatOrArray: ...
