"""Single-phase evaluator for EOS and transport properties."""

from __future__ import annotations

import logging
import sys
from dataclasses import Field, fields

import numpy as np
import numpy.typing as npt

from aragog.eos.base import PhaseEvaluatorABC, PropertyProtocol
from aragog.eos.properties import ConstantProperty, LookupProperty1D, LookupProperty2D
from aragog.parser import _PhaseParameters
from aragog.utilities import FloatOrArray, is_file, is_number

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class SinglePhaseEvaluator(PhaseEvaluatorABC):
    """Contains the objects to evaluate the EOS and transport properties of a phase.

    Args:
        settings: Phase parameters
        gravitational_acceleration: PropertyProtocol
    """

    # For typing
    _density: PropertyProtocol
    _gravitational_acceleration: PropertyProtocol
    _heat_capacity: PropertyProtocol
    _melt_fraction: ConstantProperty
    _thermal_conductivity: PropertyProtocol
    _thermal_expansivity: PropertyProtocol
    _viscosity: PropertyProtocol

    def __init__(self, settings: _PhaseParameters, gravitational_acceleration: PropertyProtocol):
        self._settings: _PhaseParameters = settings
        cls_fields: tuple[Field, ...] = fields(self._settings)
        for field_ in cls_fields:
            name: str = field_.name
            private_name: str = f"_{name}"
            value = getattr(self._settings, field_.name)

            if is_number(value):
                # Numbers have already been scaled by the parser
                setattr(self, private_name, ConstantProperty(name=name, value=value))

            elif is_file(value):
                with open(value, encoding="utf-8") as infile:
                    logger.debug("%s is a file = %s", name, value)
                    header = infile.readline()
                    col_names = header[1:].split()
                value_array: npt.NDArray = np.loadtxt(value, ndmin=2)
                logger.debug("loaded value_array = %s", value_array)
                ndim = value_array.shape[1]
                logger.debug("ndim = %d", ndim)
                if ndim == 2:
                    setattr(self, private_name, LookupProperty1D(name=name, value=value_array))
                elif ndim == 3:
                    setattr(self, private_name, LookupProperty2D(name=name, value=value_array))
                else:
                    raise ValueError(f"Lookup data must have 2 or 3 dimensions, not {ndim}")
            else:
                logger.info("Cannot interpret value (%s): not a number or a file", value)

        self._gravitational_acceleration = gravitational_acceleration

    @property
    def has_entropy(self) -> bool:
        """Whether this evaluator has entropy lookup data."""
        return hasattr(self, '_entropy') and isinstance(
            self._entropy, (LookupProperty1D, LookupProperty2D)
        )

    @override
    def density(self) -> FloatOrArray:
        return self._density(self.temperature, self.pressure)

    def entropy(self) -> FloatOrArray:
        """Specific entropy [J/(kg*K)].

        Only available if an entropy lookup table was provided in the
        phase parameters. Raises AttributeError otherwise.
        """
        if not self.has_entropy:
            raise AttributeError(
                "Entropy lookup not available. Provide an entropy table in the "
                "phase parameters to enable entropy-conserving adiabatic IC."
            )
        return self._entropy(self.temperature, self.pressure)

    @override
    def gravitational_acceleration(self) -> FloatOrArray:
        return self._gravitational_acceleration(self.temperature, self.pressure)

    @override
    def heat_capacity(self) -> FloatOrArray:
        return self._heat_capacity(self.temperature, self.pressure)

    @override
    def melt_fraction(self) -> float:
        return self._melt_fraction(self.temperature, self.pressure)

    @override
    def latent_heat(self) -> float:
        return 0.

    @override
    def thermal_conductivity(self) -> FloatOrArray:
        return self._thermal_conductivity(self.temperature, self.pressure)

    @override
    def thermal_expansivity(self) -> FloatOrArray:
        return self._thermal_expansivity(self.temperature, self.pressure)

    @override
    def viscosity(self) -> FloatOrArray:
        return self._viscosity(self.temperature, self.pressure)

    @override
    def relative_velocity(self) -> float:
        return 0.

    @override
    def delta_specific_volume(self) -> FloatOrArray:
        return 0.0
