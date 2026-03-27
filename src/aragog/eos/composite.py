#
# Copyright 2024 Dan J. Bower
#
# This file is part of Aragog.
#
# Aragog is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Aragog is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Aragog. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Composite phase evaluator combining single-phase and mixed-phase regions."""

from __future__ import annotations

import logging
import sys

import numpy as np
import numpy.typing as npt

from aragog.eos.base import PhaseEvaluatorABC, PhaseEvaluatorProtocol, PropertyProtocol
from aragog.eos.mixed_phase import MixedPhaseEvaluator
from aragog.eos.single_phase import SinglePhaseEvaluator
from aragog.parser import Parameters
from aragog.utilities import FloatOrArray, combine_properties, tanh_weight

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class CompositePhaseEvaluator(PhaseEvaluatorABC):
    """Evaluates the EOS and transport properties of a composite phase.

    This combines the single phase evaluators for the liquid and solid regions with the mixed phase
    evaluator for the mixed phase region. This ensure that the phase properties are computed
    correctly for all temperatures and pressures.

    Args:
        parameters: Parameters
    """

    def __init__(self, parameters: Parameters):
        from aragog.eos import setup_gravitational_acceleration

        gravitational_acceleration: PropertyProtocol = setup_gravitational_acceleration(parameters)
        self._liquid: PhaseEvaluatorProtocol = SinglePhaseEvaluator(
            parameters.phase_liquid, gravitational_acceleration
        )
        self._solid: PhaseEvaluatorProtocol = SinglePhaseEvaluator(
            parameters.phase_solid, gravitational_acceleration
        )
        self._mixed: MixedPhaseEvaluator = MixedPhaseEvaluator(parameters)

    @override
    def set_temperature(self, temperature: npt.NDArray) -> None:
        super().set_temperature(temperature)
        self._solid.set_temperature(temperature)
        self._liquid.set_temperature(temperature)
        self._mixed.set_temperature(temperature)

    @override
    def set_pressure(self, pressure: npt.NDArray) -> None:
        """Sets pressure and updates quantities that only depend on pressure"""
        super().set_pressure(pressure)
        self._solid.set_pressure(pressure)
        self._liquid.set_pressure(pressure)
        self._mixed.set_pressure(pressure)

    @override
    def update(self) -> None:
        self._mixed.update()
        self._set_blending_and_masks()
        self._density = self._get_composite("density")
        self._heat_capacity = self._get_composite("heat_capacity")
        self._thermal_conductivity = self._get_composite("thermal_conductivity")
        self._thermal_expansivity = self._get_composite("thermal_expansivity")
        self._dTdPs = self._get_composite("dTdPs")
        self._dTdrs = self._get_composite("dTdrs")
        self._relative_velocity = self._get_composite("relative_velocity")
        self._delta_specific_volume = self._get_composite("delta_specific_volume")

        name: str = "viscosity"
        log10_mixed_phase: npt.NDArray = np.log10(getattr(self._mixed, name)())
        single_phase: npt.NDArray = np.empty_like(self._blending_factor)
        try:
            single_phase[self._liquid_mask] = getattr(self._liquid, name)()[self._liquid_mask]
        except (IndexError, TypeError):
            single_phase[self._liquid_mask] = getattr(self._liquid, name)()
        try:
            single_phase[self._solid_mask] = getattr(self._solid, name)()[self._solid_mask]
        except (IndexError, TypeError):
            single_phase[self._solid_mask] = getattr(self._solid, name)()
        log10_single_phase: npt.NDArray = np.log10(single_phase)
        self._viscosity = combine_properties(
            self._blending_factor, log10_mixed_phase, log10_single_phase
        )
        self._viscosity = 10**self._viscosity

    @property
    def has_entropy(self) -> bool:
        """Whether entropy lookups are available for entropy-conserving adiabats."""
        return self._mixed.has_entropy

    @override
    def density(self) -> npt.NDArray:
        return self._density

    @override
    def dTdPs(self) -> npt.NDArray:
        return self._dTdPs

    @override
    def dTdrs(self) -> npt.NDArray:
        return self._dTdrs

    def entropy(self) -> npt.NDArray:
        """Composite entropy dispatching to solid, mixed, or liquid evaluator.

        Raises AttributeError if entropy tables are not available.
        """
        if not self.has_entropy:
            raise AttributeError(
                "Entropy lookup not available. Provide entropy tables in "
                "the phase parameters to enable entropy evaluation."
            )
        return self._get_composite_entropy()

    def entropy_at(self, pressure: float, temperature: float) -> float:
        """Evaluate entropy at a single (P, T) point with correct phase dispatch.

        Parameters
        ----------
        pressure : float
            Pressure (in scaled units).
        temperature : float
            Temperature (in scaled units).

        Returns
        -------
        float
            Specific entropy at (P, T).
        """
        P_arr = np.atleast_1d(pressure)
        T_arr = np.atleast_1d(temperature)
        self.set_pressure(P_arr)
        self.set_temperature(T_arr)
        self.update()
        return float(np.squeeze(self.entropy()))

    @override
    def gravitational_acceleration(self) -> FloatOrArray:
        return self._mixed.gravitational_acceleration()

    @override
    def heat_capacity(self) -> npt.NDArray:
        """Heat capacity"""
        return self._heat_capacity

    def liquidus(self) -> npt.NDArray:
        return self._mixed.liquidus()

    def liquidus_gradient(self) -> npt.NDArray:
        return self._mixed.liquidus_gradient()

    @override
    def melt_fraction(self) -> npt.NDArray:
        """Melt fraction"""
        return self._mixed.melt_fraction()

    @override
    def latent_heat(self) -> FloatOrArray:
        """Latent heat of fusion"""
        return self._mixed.latent_heat()

    def solidus(self) -> npt.NDArray:
        return self._mixed.solidus()

    def solidus_gradient(self) -> npt.NDArray:
        return self._mixed.solidus_gradient()

    def delta_fusion(self) -> npt.NDArray:
        return self._mixed.delta_fusion()

    @override
    def thermal_conductivity(self) -> npt.NDArray:
        """Thermal conductivity"""
        return self._thermal_conductivity

    @override
    def thermal_expansivity(self) -> npt.NDArray:
        """Thermal expansivity"""
        return self._thermal_expansivity

    def viscosity(self) -> npt.NDArray:
        """Viscosity"""
        return self._viscosity

    @override
    def relative_velocity(self) -> npt.NDArray:
        """Relative velocity between melt and solid"""
        return self._relative_velocity

    @override
    def delta_specific_volume(self) -> npt.NDArray:
        """Difference of specific volume between melt and solid"""
        return self._delta_specific_volume

    def _set_blending_and_masks(self) -> None:
        """Sets blending and masks."""

        phase_transition_width: float = self._mixed.settings.phase_transition_width
        melt_fraction_no_clip: npt.NDArray = self._mixed.melt_fraction_no_clip()

        if phase_transition_width == 0.0:
            blending_factor: npt.NDArray = np.where(
                ((melt_fraction_no_clip < 0.0) | (melt_fraction_no_clip > 1.0)),
                0,
                1,
            )
        else:
            blending_liquid: npt.NDArray = 1.0 - tanh_weight(
                melt_fraction_no_clip, 1.0, phase_transition_width
            )
            blending_solid: npt.NDArray = tanh_weight(
                melt_fraction_no_clip, 0.0, phase_transition_width
            )
            blending_factor = np.where(
                melt_fraction_no_clip > 0.5, blending_liquid, blending_solid
            )

        self._blending_factor = blending_factor
        logger.debug("_blending_factor = %s", self._blending_factor)
        self._liquid_mask = melt_fraction_no_clip > 0.5
        logger.debug("_liquid_mask = %s", self._liquid_mask)
        self._solid_mask = ~self._liquid_mask
        logger.debug("_solid_mask = %s", self._solid_mask)

    def _get_composite(self, property_name: str) -> npt.NDArray:
        """Evaluates the composite property"""
        mixed_phase: npt.NDArray = getattr(self._mixed, property_name)()
        single_phase: npt.NDArray = np.empty_like(self._blending_factor)
        logger.debug("single_phase = %s", single_phase)
        logger.debug("_liquid_mask = %s", self._liquid_mask)
        logger.debug("_solid_mask = %s", self._solid_mask)
        test = getattr(self._liquid, property_name)()
        logger.debug("test = %s", test)

        # logger.debug(self.temperature.shape)
        # logger.debug(self.pressure.shape)
        # logger.debug(mixed_phase.shape)
        # logger.debug(single_phase.shape)

        # TODO: This is ugly.  Clean up logic.
        try:
            single_phase[self._liquid_mask] = getattr(self._liquid, property_name)()[
                self._liquid_mask
            ]
        except (IndexError, TypeError):
            single_phase[self._liquid_mask] = getattr(self._liquid, property_name)()
        try:
            single_phase[self._solid_mask] = getattr(self._solid, property_name)()[
                self._solid_mask
            ]
        except (IndexError, TypeError):
            single_phase[self._solid_mask] = getattr(self._solid, property_name)()

        combined: npt.NDArray = combine_properties(
            self._blending_factor, mixed_phase, single_phase
        )

        return combined

    def _get_composite_entropy(self) -> npt.NDArray:
        """Composite entropy with correct phase dispatch.

        In single-phase regions, uses the single-phase entropy lookup.
        In the mushy zone, uses the mixed-phase entropy (linear interpolation
        between solid entropy at solidus and liquid entropy at liquidus).
        """
        mixed_phase: npt.NDArray = self._mixed.entropy()
        single_phase: npt.NDArray = np.empty_like(self._blending_factor)
        try:
            single_phase[self._liquid_mask] = self._liquid.entropy()[self._liquid_mask]
        except (IndexError, TypeError):
            single_phase[self._liquid_mask] = self._liquid.entropy()
        try:
            single_phase[self._solid_mask] = self._solid.entropy()[self._solid_mask]
        except (IndexError, TypeError):
            single_phase[self._solid_mask] = self._solid.entropy()

        return combine_properties(
            self._blending_factor, mixed_phase, single_phase
        )
