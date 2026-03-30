"""EOS subpackage: equation of state and transport property evaluators.

Re-exports all public classes so that ``from aragog.eos import X`` works
for any class previously available from ``aragog.phase`` or ``aragog.interfaces``.
"""

from __future__ import annotations

import logging
from dataclasses import InitVar, dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from aragog.eos.base import (
    MixedPhaseEvaluatorProtocol,
    PhaseEvaluatorABC,
    PhaseEvaluatorProtocol,
    PropertyProtocol,
)
from aragog.eos.composite import CompositePhaseEvaluator
from aragog.eos.mixed_phase import MixedPhaseEvaluator
from aragog.eos.properties import ConstantProperty, LookupProperty1D, LookupProperty2D
from aragog.eos.single_phase import SinglePhaseEvaluator
from aragog.eos.transport import combine_properties, tanh_weight
from aragog.parser import Parameters

logger: logging.Logger = logging.getLogger(__name__)


def setup_gravitational_acceleration(parameters: Parameters):
    """Sets up the gravitational acceleration property.

    Args:
        parameters: Parameters

    Returns:
        gravitational_acceleration: PropertyProtocol
    """
    if parameters.mesh.eos_method == 1:
        # AdamsWilliamson EOS method = constant gravitational acceleration
        gravitational_acceleration = ConstantProperty(
            "gravitational_acceleration", value=parameters.mesh.gravitational_acceleration
        )
    elif parameters.mesh.eos_method == 2:
        # User defined EOS method = space dependent gravitational acceleration (1D lookup)
        # Re-interpolate EOS-data to match standard lookup pressure grid
        interp_gravity = PchipInterpolator(
            np.flip(parameters.mesh.eos_pressure),
            np.flip(parameters.mesh.eos_gravity))
        num_points: int = 138
        max_pressure: float = 1.37e11  # Pa
        lookup_data: npt.NDArray = np.zeros((num_points, 2), dtype=float)
        lookup_data[:,0] = np.linspace(0.0, max_pressure, num=num_points)
        lookup_data[:,1] = interp_gravity(lookup_data[:,0])
        gravitational_acceleration = LookupProperty1D(
            "gravitational_acceleration",
            value=lookup_data,
        )
    else:
        raise ValueError(f"EOS method = {parameters.mesh.eos_method} is not a valid selection")

    return gravitational_acceleration


@dataclass
class PhaseEvaluatorCollection:
    """A collection of phase evaluators

    Creates the phase evaluators and selects the active phase based on configuration data.

    Args:
        parameters: Parameters

    Attributes:
        liquid: Liquid evaluator
        solid: Solid evaluator
        mixed: Mixed evaluator
        composite: Composite evaluator
        active: The active evaluator, which is defined by configuration data
    """

    parameters: InitVar[Parameters]
    liquid: PhaseEvaluatorProtocol = field(init=False)
    solid: PhaseEvaluatorProtocol = field(init=False)
    mixed: MixedPhaseEvaluatorProtocol = field(init=False)
    composite: MixedPhaseEvaluatorProtocol = field(init=False)
    active: PhaseEvaluatorProtocol = field(init=False)

    def __post_init__(self, parameters: Parameters):
        gravitational_acceleration: PropertyProtocol = setup_gravitational_acceleration(parameters)
        self.liquid = SinglePhaseEvaluator(parameters.phase_liquid, gravitational_acceleration)
        self.solid = SinglePhaseEvaluator(parameters.phase_solid, gravitational_acceleration)
        self.mixed = MixedPhaseEvaluator(parameters)
        self.composite = CompositePhaseEvaluator(parameters)

        # Configuration data defines which phase to use for the model.
        phase_to_use: str = parameters.phase_mixed.phase

        if phase_to_use == "liquid":
            self.active = self.liquid
        elif phase_to_use == "solid":
            self.active = self.solid
        # Allowing selection of self.mixed doesn't really make sense because it will probably give
        # crazy results outside the mixed phase region. Hence just use composite.
        elif phase_to_use == "mixed" or phase_to_use == "composite":
            self.active = self.composite
        else:
            raise ValueError(f"Phase = {phase_to_use} is not a valid selection")


__all__ = [
    "CompositePhaseEvaluator",
    "ConstantProperty",
    "LookupProperty1D",
    "LookupProperty2D",
    "MixedPhaseEvaluator",
    "MixedPhaseEvaluatorProtocol",
    "PhaseEvaluatorABC",
    "PhaseEvaluatorCollection",
    "PhaseEvaluatorProtocol",
    "PropertyProtocol",
    "SinglePhaseEvaluator",
    "combine_properties",
    "setup_gravitational_acceleration",
    "tanh_weight",
]
