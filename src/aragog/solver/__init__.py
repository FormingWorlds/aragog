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
"""Solver package for the Aragog interior dynamics model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import constants as sp_constants
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

# Time unit conversion: the ODE is integrated in years, but fluxes are in SI (W/m^2).
# dT/dt from flux divergence gives K/s; multiply by SECS_PER_YEAR to get K/yr.
SECS_PER_YEAR: float = sp_constants.Julian_year  # 31557600.0 s

from aragog.parser import Parameters
from aragog.solver.boundary import BoundaryConditions
from aragog.solver.evaluator import Evaluator
from aragog.solver.initial import InitialCondition
from aragog.solver.state import State

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "BoundaryConditions",
    "Evaluator",
    "InitialCondition",
    "SECS_PER_YEAR",
    "Solver",
    "State",
]


class Solver:
    """Solves the interior dynamics

    Args:
        filename: Filename of a file with configuration settings
        root: Root path to the flename

    Attributes:
        filename: Filename of a file with configuration settings
        root: Root path to the filename. Defaults to empty
        parameters: Parameters
        evaluator: Evaluator
        state: State
    """

    def __init__(self, param: Parameters):
        logger.info("Creating an Aragog model")
        self.parameters: Parameters = param
        self.evaluator: Evaluator
        self.state: State
        self._solution: OptimizeResult

    @classmethod
    def from_file(cls, filename: str | Path, root: str | Path = Path()) -> Self:
        """Parses a configuration file

        Args:
            filename: Filename
            root: Root of the filename

        Returns:
            Parameters
        """
        configuration_file: Path = Path(root) / Path(filename)
        logger.info("Parsing configuration file = %s", configuration_file)
        parameters: Parameters = Parameters.from_file(configuration_file)

        return cls(parameters)

    def initialize(self) -> None:
        """Initializes the model."""
        logger.info("Initializing %s", self.__class__.__name__)
        self.evaluator = Evaluator(self.parameters)
        self.state = State(self.parameters, self.evaluator)

    def reset(self) -> None:
        """Reset the model for a new integration, keeping phase lookup tables.

        Re-reads the EOS file from disk if eos_method=2, so that updates
        to the Zalmoxis output file between coupling iterations are picked up.
        Phase property lookup tables (density, cp, etc.) are NOT reloaded
        to avoid expensive re-interpolation.
        """
        logger.info("Resetting %s", self.__class__.__name__)

        # Re-read the user-defined EOS file if it exists, so dynamic mesh
        # refresh from Zalmoxis is picked up (fixes stale-mesh bug).
        if self.parameters.mesh.eos_method == 2 and self.parameters.mesh.eos_file:
            import numpy as np

            arr = np.loadtxt(self.parameters.mesh.eos_file)
            self.parameters.mesh.eos_radius = arr[:, 0]
            self.parameters.mesh.eos_pressure = arr[:, 1]
            self.parameters.mesh.eos_density = arr[:, 2]
            self.parameters.mesh.eos_gravity = arr[:, 3]
            logger.debug("Re-read EOS file: %s", self.parameters.mesh.eos_file)

        # Update the Evaluator object except the phase properties
        from aragog.mesh import Mesh

        self.evaluator.mesh = Mesh(self.parameters)
        self.evaluator.boundary_conditions = BoundaryConditions(self.parameters, self.evaluator.mesh)
        self.evaluator.initial_condition = InitialCondition(self.parameters, self.evaluator.mesh, self.evaluator.phases)
        # Reinstantiate the solver state
        self.state = State(self.parameters, self.evaluator)

    @property
    def temperature_basic(self) -> npt.NDArray:
        """Temperature of the basic mesh in K"""
        return self.evaluator.mesh.quantity_at_basic_nodes(self.temperature_staggered)

    @property
    def temperature_staggered(self) -> npt.NDArray:
        """Temperature of the staggered mesh in K"""
        return self.solution.y

    @property
    def solution(self) -> OptimizeResult:
        """The solution."""
        return self._solution

    def dTdt(
        self,
        time: npt.NDArray | float,
        temperature: npt.NDArray,
    ) -> npt.NDArray:
        """dT/dt at the staggered nodes

        Args:
            time: Time
            temperature: Temperature at the staggered nodes

        Returns:
            dT/dt at the staggered nodes
        """
        logger.debug("temperature passed into dTdt = %s", temperature)
        # logger.debug("temperature.shape = %s", temperature.shape)
        self.state.update(temperature, time)
        heat_flux: npt.NDArray = self.state.heat_flux
        # logger.debug("heat_flux = %s", heat_flux)
        self.evaluator.boundary_conditions.apply_flux_boundary_conditions(self.state)
        # logger.debug("heat_flux = %s", heat_flux)
        # logger.debug("mesh.basic.area.shape = %s", self.data.mesh.basic.area.shape)

        energy_flux: npt.NDArray = heat_flux * self.evaluator.mesh.basic.area
        # logger.debug("energy_flux size = %s", energy_flux.shape)

        delta_energy_flux: npt.NDArray = np.diff(energy_flux, axis=0)
        # logger.debug("delta_energy_flux size = %s", delta_energy_flux.shape)
        # logger.debug("capacitance = %s", self.state.phase_staggered.capacitance.shape)
        # Capacitance includes latent heat: in the mixed phase, cp already contains
        # the L/(T_liq - T_sol) contribution via MixedPhaseEvaluator.heat_capacity().
        # The separate latent heat flux term (j_grav + j_mix)*L handles melt TRANSPORT,
        # not in-situ phase change. These two mechanisms are physically distinct.
        capacitance: npt.NDArray = (
            self.state.capacitance_staggered() * self.evaluator.mesh.basic.volume
        )

        # Heating rate (dT/dt) from flux divergence (power per unit area).
        # Fluxes are in SI (W/m^2), so the raw dT/dt is in K/s.
        dTdt: npt.NDArray = -delta_energy_flux / capacitance
        logger.debug("dTdt (fluxes only, K/s) = %s", dTdt)

        # Additional heating rate (dT/dt) from internal heating (power per unit mass)
        dTdt += self.state.heating * (
            self.state.phase_staggered.density() / self.state.capacitance_staggered()
        )

        # Convert K/s to K/yr since the ODE is integrated in years
        dTdt *= SECS_PER_YEAR

        logger.debug("dTdt (with internal heating, K/yr) = %s", dTdt)

        return dTdt

    def make_tsurf_event(self):
        """
        Creates a temperature event function for use with an ODE solver to monitor changes
        in the surface temperature.The event triggers when the change exceeds the
        threshold, allowing the solver to stop integration.

        Returns:
            The event has the attributes:
                - terminal = True: Integration stops when the event is triggered.
                - direction = -1: Only triggers when the function is decreasing through zero.
        """
        tsurf_initial = [None]

        def tsurf_event(time: float, temperature: npt.NDArray) -> float:
            """
            Event function to detect when surface temperature changes beyond a specified threshold.

            Args:
                time (float): Current time.
                temperature (np.ndarray): Current temperature profile.

            Returns:
                float: The difference between the threshold and the actual change in surface
                    temperature. When this value crosses zero from above, the event is triggered.
            """
            tsurf_current = temperature[-1]  # Already in K
            tsurf_threshold = self.parameters.solver.tsurf_poststep_change  # Already in K

            if tsurf_initial[0] is None:
                tsurf_initial[0] = tsurf_current
                return 1.0

            delta = abs(tsurf_current - tsurf_initial[0])

            return tsurf_threshold - delta

        tsurf_event.terminal = self.parameters.solver.event_triggering
        tsurf_event.direction = -1

        return tsurf_event

    def solve(self) -> None:
        start_time = self.parameters.solver.start_time
        end_time = self.parameters.solver.end_time
        atol = self.parameters.solver.atol
        rtol = self.parameters.solver.rtol

        tsurf_event = self.make_tsurf_event()


        self._solution = solve_ivp(
            self.dTdt,
            (start_time, end_time),
            self.evaluator.initial_condition.temperature,
            method="BDF",
            vectorized=True,
            atol=atol,
            rtol=rtol,
            events=[tsurf_event],
         )
        logger.info(self.solution)

        if self._solution.status == 1:
            logger.warning("Integration stopped early due to surface temperature jump.")
            self.stop_early = True

        elif self._solution.status == 0:
            logger.info("Integration completed successfully.")
            self.stop_early = False

        else:
            logger.error("Integration failed with status = %d", self._solution.status)
            logger.error("Message: %s", self._solution.message)
            self.stop_early = True
