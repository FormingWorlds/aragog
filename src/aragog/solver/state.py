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
"""State class for storing and updating the thermodynamic state."""

from __future__ import annotations

import copy
import logging
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from aragog.interfaces import PhaseEvaluatorProtocol
from aragog.parser import Parameters, _EnergyParameters
from aragog.utilities import FloatOrArray

if TYPE_CHECKING:
    from aragog.solver.evaluator import Evaluator

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class State:
    """Stores and updates the state at temperature and pressure.

    Args:
        parameters: Parameters
        evaluator: Evaluator

    Attributes:
        evaluator: Evaluator
        critical_reynolds_number: Critical Reynolds number
        gravitational_separation_flux: Gravitational separation flux at the basic nodes
        heating: Total internal heat production at the staggered nodes (power per unit mass)
        heating_radio: Radiogenic heat production at the staggered nodes (power per unit mass)
        heating_tidal: Tidal heat production at the staggered nodes (power per unit mass)
        heat_flux: Heat flux at the basic nodes (power per unit area)
        inviscid_regime: True if the flow is inviscid and otherwise False, at the basic nodes
        inviscid_velocity: Inviscid velocity at the basic nodes
        is_convective: True if the flow is convecting and otherwise False, at the basic nodes
        reynolds_number: Reynolds number at the basic nodes
        super_adiabatic_temperature_gradient: Super adiabatic temperature gradient at the basic nod
        temperature_basic: Temperature at the basic nodes
        temperature_staggered: Temperature at the staggered nodes
        bottom_temperature: Temperature at the bottom basic node
        top_temperature: Temperature at the top basic node
        viscous_regime: True if the flow is viscous and otherwise False, at the basic nodes
        viscous_velocity: Viscous velocity at the basic nodes
    """

    parameters: InitVar[Parameters]
    _evaluator: Evaluator
    _settings: _EnergyParameters = field(init=False)
    phase_basic: PhaseEvaluatorProtocol = field(init=False)
    phase_staggered: PhaseEvaluatorProtocol = field(init=False)
    _dTdr: npt.NDArray = field(init=False)
    _dphidr: npt.NDArray = field(init=False)
    _eddy_diffusivity: npt.NDArray = field(init=False)
    _mass_flux: npt.NDArray = field(init=False)
    _heat_flux: npt.NDArray = field(init=False)
    _heating: npt.NDArray = field(init=False)
    _heating_radio: npt.NDArray = field(init=False)
    _heating_dilatation: npt.NDArray = field(init=False)
    _heating_tidal: npt.NDArray = field(init=False)
    _is_convective: npt.NDArray = field(init=False)
    _reynolds_number: npt.NDArray = field(init=False)
    _super_adiabatic_temperature_gradient: npt.NDArray = field(init=False)
    _temperature_basic: npt.NDArray = field(init=False)
    _temperature_staggered: npt.NDArray = field(init=False)
    _viscous_velocity: npt.NDArray = field(init=False)
    _inviscid_velocity: npt.NDArray = field(init=False)

    def __post_init__(self, parameters: Parameters):
        self._settings = parameters.energy
        # Sets the pressure since this will not change during a model run. Must deepcopy first
        # because pressure is set as an attribute.
        self.phase_basic = copy.deepcopy(self._evaluator.phases.active)
        self.phase_basic.set_pressure(self._evaluator.mesh.basic_pressure)
        self.phase_staggered = copy.deepcopy(self._evaluator.phases.active)
        self.phase_staggered.set_pressure(self._evaluator.mesh.staggered_pressure)

    def capacitance_staggered(self) -> FloatOrArray:
        capacitance: FloatOrArray = (
            self.phase_staggered.density() * self.phase_staggered.heat_capacity()
        )

        return capacitance

    def conductive_heat_flux(self) -> npt.NDArray:
        r"""Conductive heat flux:

        $$
        q_{cond} = -k \frac{\partial T}{\partial r}
        $$

        where $k$ is thermal conductivity, $T$ is temperature, and $r$ is radius.
        """
        conductive_heat_flux: npt.NDArray = self.phase_basic.thermal_conductivity() * -self.dTdr()

        return conductive_heat_flux

    def convective_heat_flux(self) -> npt.NDArray:
        r"""Convective heat flux:

        $$
        q_{conv} = -\rho c_p \kappa_h \left( \frac{\partial T}{\partial r}
            - \left( \frac{\partial T}{\partial r} \right)_S \right)
        $$

        where $\rho$ is density, $c_p$ is heat capacity at constant pressure,
        $\kappa_h$ is eddy diffusivity, $T$ is temperature, $r$ is radius, and
        $S$ is entropy.
        """
        convective_heat_flux: npt.NDArray = (
            self.phase_basic.density()
            * self.phase_basic.heat_capacity()
            * self.eddy_diffusivity()
            * -self._super_adiabatic_temperature_gradient
        )

        return convective_heat_flux

    def chemical_eddy_diffusivity(self) -> npt.NDArray:
        r"""Chemical eddy diffusivity for compositional transport.

        Scales the thermal eddy diffusivity by the chemical diffusivity
        ratio. When eddy_diffusivity_chemical = 1.0 (default), this equals
        the thermal eddy diffusivity. Separate scaling allows independent
        control of how efficiently convection mixes composition vs heat.

        Returns
        -------
        npt.NDArray
            Chemical eddy diffusivity at basic nodes.
        """
        return self._eddy_diffusivity * self._settings.eddy_diffusivity_chemical

    def gravitational_separation_mass_flux(self) -> npt.NDArray:
        r"""Gravitational separation mass flux:

        $$
        j_{grav} = \rho \phi (1 - \phi) v_{rel}
        $$

        where $\rho$ is density, $\phi$ is melt fraction, and
        $v_{rel}$ is relative velocity.
        """
        gravitational_separation_mass_flux: npt.NDArray = (
            self.phase_basic.density()
            * self.phase_basic.melt_fraction()
            * (1.0 - self.phase_basic.melt_fraction())
            * self.phase_basic.relative_velocity()
        )
        return gravitational_separation_mass_flux

    def mixing_mass_flux(self) -> npt.NDArray:
        r"""Mixing mass flux with pressure-dependent Clapeyron slopes.

        Uses the analytical melt fraction gradient that accounts for the
        pressure-dependence of the solidus and liquidus:

        $$
        \frac{\partial \phi}{\partial r} = \frac{1}{T_{liq} - T_{sol}}
            \left[ \frac{\partial T}{\partial r}
            - \frac{\partial T_{sol}}{\partial P} \frac{\partial P}{\partial r}
            - \phi \left( \frac{\partial T_{liq}}{\partial P}
            - \frac{\partial T_{sol}}{\partial P} \right)
            \frac{\partial P}{\partial r} \right]
        $$

        The mass flux uses the chemical eddy diffusivity (kappa_c) rather
        than the thermal eddy diffusivity (kappa_h), following SPIDER's
        convention. When eddy_diffusivity_chemical = 1.0 (default), they
        are equal.

        Returns
        -------
        npt.NDArray
            Mixing mass flux at basic nodes.
        """
        phi = self.phase_basic.melt_fraction()
        delta_T = self.phase_basic.delta_fusion()
        dTdr = self.dTdr()

        # Clapeyron slopes: dT_sol/dP and dT_liq/dP
        dTsol_dP = self.phase_basic.solidus_gradient()
        dTliq_dP = self.phase_basic.liquidus_gradient()

        # Pressure gradient dP/dr at basic nodes (from finite difference of pressure)
        dPdr = self._evaluator.mesh.d_dr_at_basic_nodes(
            self._evaluator.mesh.staggered_pressure
        )

        # Analytical dphi/dr including Clapeyron slopes
        dphi_dr = (dTdr - dTsol_dP * dPdr - phi * (dTliq_dP - dTsol_dP) * dPdr) / delta_T

        # Use chemical eddy diffusivity for compositional transport
        mixing_mass_flux: npt.NDArray = (
            self.phase_basic.density()
            * self.chemical_eddy_diffusivity()
            * -dphi_dr
        )
        return mixing_mass_flux

    def radiogenic_heating(self, time: float) -> npt.NDArray:
        """Radiogenic heating (constant with radius)

        Args:
            time: Time

        Returns:
            Radiogenic heating (power per unit mass) at each layer of the staggered
                mesh, at a given point in time.
        """

        # Total heat production at a given time (power per unit mass)
        radio_heating_float: float = 0
        for radionuclide in self._evaluator.radionuclides:
            radio_heating_float += radionuclide.get_heating(time)

        # Convert to 1D array (assuming abundances are constant)
        return radio_heating_float * np.ones_like(self.temperature_staggered)

    def dilatation_heating(self) -> npt.NDArray:
        """Dilatation/compression heating (power per unit mass)

        Returns:
            Dilatation/compression heating (power per unit mass) at each layer of the staggered
                mesh, at a given point in time.
        """

        mass_flux_staggered: npt.NDArray = self._evaluator.mesh.quantity_at_staggered_nodes(self._mass_flux)

        dilatation_volume_source: npt.NDArray = (
            self.phase_staggered.gravitational_acceleration()
            * self.phase_staggered.delta_specific_volume()
            * mass_flux_staggered
        )

        return dilatation_volume_source

    def tidal_heating(self) -> npt.NDArray:
        """Tidal heating at each layer of the mantle.

        Args:
            time: Time

        Returns:
            Tidal heating (power per unit mass) at each layer of the staggered
                mesh, at a given point in time.
        """

        length = len(self._settings.tidal_array)

        # Must have correct shape
        if length == 1:
            # scalar
            out = np.ones_like(self.temperature_staggered) * self._settings.tidal_array[0]

        elif length == len(self.temperature_staggered):
            # vector with correct length
            out = np.array([self._settings.tidal_array]).T

        else:
            # vector with invalid length
            raise ValueError(f"Tidal heating array has invalid length {length}")

        return out

    @property
    def critical_reynolds_number(self) -> float:
        """Critical Reynolds number from Abe (1993)"""
        return 9 / 8

    def dTdr(self) -> npt.NDArray:
        return self._dTdr

    def dphidr(self) -> npt.NDArray:
        return self._dphidr

    def eddy_diffusivity(self) -> npt.NDArray:
        return self._eddy_diffusivity

    @property
    def heating(self) -> npt.NDArray:
        """The power generation according to the heat sources specified in the configuration."""
        return self._heating

    @property
    def heating_radio(self) -> npt.NDArray:
        """The radiogenic power generation."""
        return self._heating_radio

    @property
    def heating_dilatation(self) -> npt.NDArray:
        """The heat source through dilation/compression."""
        return self._heating_dilatation

    @property
    def heating_tidal(self) -> npt.NDArray:
        """The tidal power generation."""
        return self._heating_tidal

    @property
    def mass_flux(self) -> npt.NDArray:
        """The total melt mass flux according to the fluxes specified in the configuration."""
        return self._mass_flux

    @property
    def heat_flux(self) -> npt.NDArray:
        """The total heat flux according to the fluxes specified in the configuration."""
        return self._heat_flux

    @property
    def inviscid_regime(self) -> npt.NDArray:
        return self._reynolds_number > self.critical_reynolds_number

    @property
    def inviscid_velocity(self) -> npt.NDArray:
        return self._inviscid_velocity

    @property
    def is_convective(self) -> npt.NDArray:
        return self._is_convective

    @property
    def reynolds_number(self) -> npt.NDArray:
        return self._reynolds_number

    @property
    def super_adiabatic_temperature_gradient(self) -> npt.NDArray:
        return self._super_adiabatic_temperature_gradient

    @property
    def temperature_basic(self) -> npt.NDArray:
        return self._temperature_basic

    @property
    def temperature_staggered(self) -> npt.NDArray:
        return self._temperature_staggered

    @property
    def top_temperature(self) -> npt.NDArray:
        return self._temperature_basic[-1, :]

    @property
    def bottom_temperature(self) -> npt.NDArray:
        return self._temperature_basic[0, :]

    @property
    def viscous_regime(self) -> npt.NDArray:
        return self._reynolds_number <= self.critical_reynolds_number

    @property
    def viscous_velocity(self) -> npt.NDArray:
        return self._viscous_velocity

    def update(self, temperature: npt.NDArray, time: FloatOrArray) -> None:
        """Updates the state.

        The evaluation order matters because we want to minimise the number of evaluations.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes
            time: Time
        """
        logger.debug("Updating the state")

        logger.debug("Setting the temperature profile")
        self._temperature_staggered = temperature
        self._temperature_basic = self._evaluator.mesh.quantity_at_basic_nodes(temperature)
        logger.debug("temperature_basic = %s", self.temperature_basic)
        self._dTdr = self._evaluator.mesh.d_dr_at_basic_nodes(temperature)
        logger.debug("dTdr = %s", self.dTdr())
        self._evaluator.boundary_conditions.apply_temperature_boundary_conditions(
            temperature, self._temperature_basic, self.dTdr()
        )

        logger.debug("Setting the melt fraction profile")
        self.phase_staggered.set_temperature(temperature)
        self.phase_staggered.update()
        self.phase_basic.set_temperature(self._temperature_basic)
        self.phase_basic.update()
        phase_to_use = self._evaluator._parameters.phase_mixed.phase
        if phase_to_use == "mixed" or phase_to_use == "composite":
            self._dphidr = self._evaluator.mesh.d_dr_at_basic_nodes(
                self.phase_staggered.melt_fraction()
            )
            logger.debug("dphidr = %s", self.dphidr())
            self._evaluator.boundary_conditions.apply_temperature_boundary_conditions_melt(
                self.phase_staggered.melt_fraction(), self.phase_basic.melt_fraction(), self._dphidr
            )
        else:
            self._dphidr = np.zeros_like(self._dTdr)

        self._super_adiabatic_temperature_gradient = self.dTdr() - self.phase_basic.dTdrs()
        self._is_convective = self._super_adiabatic_temperature_gradient < 0
        velocity_prefactor: npt.NDArray = (
            self.phase_basic.gravitational_acceleration()
            * self.phase_basic.thermal_expansivity()
            * -self._super_adiabatic_temperature_gradient
        )
        # Viscous velocity
        self._viscous_velocity = (
            velocity_prefactor * self._evaluator.mesh.basic.mixing_length_cubed
        ) / (18 * self.phase_basic.kinematic_viscosity())
        self._viscous_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        # Inviscid velocity
        self._inviscid_velocity = (
            velocity_prefactor * self._evaluator.mesh.basic.mixing_length_squared
        ) / 16
        self._inviscid_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        self._inviscid_velocity[self._is_convective] = np.sqrt(
            self._inviscid_velocity[self._is_convective]
        )
        # Reynolds number
        self._reynolds_number = (
            self._viscous_velocity
            * self._evaluator.mesh.basic.mixing_length
            / self.phase_basic.kinematic_viscosity()
        )
        # Eddy diffusivity
        self._eddy_diffusivity = np.where(
            self.viscous_regime, self._viscous_velocity, self._inviscid_velocity
        )
        self._eddy_diffusivity *= self._evaluator.mesh.basic.mixing_length

        # Heat flux (power per unit area)
        self._heat_flux = np.zeros_like(self.temperature_basic)
        self._mass_flux = np.zeros_like(self.temperature_basic)
        if self._settings.conduction:
            self._heat_flux += self.conductive_heat_flux()
        if self._settings.convection:
            self._heat_flux += self.convective_heat_flux()
        if self._settings.gravitational_separation:
            self._mass_flux += self.gravitational_separation_mass_flux()
        if self._settings.mixing:
            self._mass_flux += self.mixing_mass_flux()
        self._heat_flux += self._mass_flux * self.phase_basic.latent_heat()

        # Heating (power per unit mass)
        self._heating = np.zeros_like(self.temperature_staggered)
        self._heating_radio = np.zeros_like(self.temperature_staggered)
        self._heating_dilatation = np.zeros_like(self.temperature_staggered)
        self._heating_tidal = np.zeros_like(self.temperature_staggered)

        if self._settings.radionuclides:
            self._heating_radio = self.radiogenic_heating(time)
            self._heating += self._heating_radio

        if self._settings.dilatation:
            self._heating_dilatation = self.dilatation_heating()
            self._heating += self._heating_dilatation

        if self._settings.tidal:
            self._heating_tidal = self.tidal_heating()
            self._heating += self._heating_tidal
