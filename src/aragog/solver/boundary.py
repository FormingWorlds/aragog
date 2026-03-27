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
"""Boundary conditions for the interior solver."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import constants as sp_constants

from aragog.mesh import Mesh
from aragog.parser import Parameters, _BoundaryConditionsParameters

if TYPE_CHECKING:
    from aragog.solver.state import State

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BoundaryConditions:
    """Boundary conditions

    Args:
        parameters: Parameters
        mesh: Mesh
    """

    _parameters: Parameters
    _mesh: Mesh

    def __post_init__(self):
        self._settings: _BoundaryConditionsParameters = self._parameters.boundary_conditions

    def apply_temperature_boundary_conditions(
        self, temperature: npt.NDArray, temperature_basic: npt.NDArray, dTdr: npt.NDArray
    ) -> None:
        """Conforms the temperature and dTdr at the basic nodes to temperature boundary conditions.

        Args:
            temperature: Temperature at the staggered nodes
            temperature_basic: Temperature at the basic nodes
            dTdr: Temperature gradient at the basic nodes
        """
        # Core-mantle boundary
        if self._settings.inner_boundary_condition == 3:
            temperature_basic[0, :] = self._settings.inner_boundary_value
            dTdr[0, :] = (
                2 * (temperature[0, :] - temperature_basic[0, :])
                / self._mesh.basic.delta_mesh[0]
                * self._mesh.dxidr[0]
            )
        # Surface
        if self._settings.outer_boundary_condition == 5:
            temperature_basic[-1, :] = self._settings.outer_boundary_value
            dTdr[-1, :] = (
                2 * (temperature_basic[-1, :] - temperature[-1, :])
                / self._mesh.basic.delta_mesh[-1]
                * self._mesh.dxidr[-1]
            )

    def apply_temperature_boundary_conditions_melt(
        self, melt_fraction: npt.NDArray, melt_fraction_basic: npt.NDArray, dphidr: npt.NDArray
    ) -> None:
        """Conforms the melt fraction gradient dphidr at the basic nodes
           to temperature boundary conditions.

        Args:
            melt_fraction: Melt fraction at the staggered nodes
            melt_fraction_basic: Melt fraction at the basic nodes
            dphidr: Melt fraction gradient at the basic nodes
        """
        # Core-mantle boundary
        if self._settings.inner_boundary_condition == 3:
            dphidr[0, :] = (
                2 * (melt_fraction[0, :] - melt_fraction_basic[0, :])
                / self._mesh.basic.delta_mesh[0]
                * self._mesh.dxidr[0]
            )
        # Surface
        if self._settings.outer_boundary_condition == 5:
            dphidr[-1, :] = (
                2 * (melt_fraction_basic[-1, :] - melt_fraction[-1, :])
                / self._mesh.basic.delta_mesh[-1]
                * self._mesh.dxidr[-1]
            )

    def apply_flux_boundary_conditions(self, state: State) -> None:
        """Applies the boundary conditions to the state.

        Args:
            state: The state to apply the boundary conditions to
        """
        self.apply_flux_inner_boundary_condition(state)
        self.apply_flux_outer_boundary_condition(state)
        logger.debug("temperature = %s", state.temperature_basic)
        logger.debug("heat_flux = %s", state.heat_flux)

    def apply_flux_outer_boundary_condition(self, state: State) -> None:
        """Applies the flux boundary condition to the state at the outer boundary.

        Args:
            state: The state to apply the boundary conditions to

        Equivalent to SURFACE_BC in C code.
            1: Grey-body atmosphere
            2: Zahnle steam atmosphere
            3: Couple to atmodeller
            4: Prescribed heat flux
            5: Prescribed temperature
        """
        if self._settings.outer_boundary_condition == 1:
            self.grey_body(state)
        elif self._settings.outer_boundary_condition == 2:
            raise NotImplementedError
        elif self._settings.outer_boundary_condition == 3:
            msg: str = "Requires coupling to atmodeller"
            logger.error(msg)
            raise NotImplementedError(msg)
        elif self._settings.outer_boundary_condition == 4:
            state.heat_flux[-1, :] = self._settings.outer_boundary_value
        elif self._settings.outer_boundary_condition == 5:
            pass
        else:
            msg: str = (
                f"outer_boundary_condition = {self._settings.outer_boundary_condition} is unknown"
            )
            raise ValueError(msg)

    def grey_body(self, state: State) -> None:
        """Applies a grey body flux at the surface.

        When param_utbl is enabled, the surface radiating temperature is reduced
        to account for the ultra-thin thermal boundary layer at the magma ocean
        surface. The temperature drop across this unresolved boundary layer is
        parameterized as dT = b * T_surf^3 (Bower et al. 2018, Eq. 18), giving
        the cubic relation T_interior = T_surf + b * T_surf^3. The analytical
        solution (Cardano's formula) gives T_surf < T_interior.

        Parameters
        ----------
        state : State
            The state to apply the boundary conditions to.
        """
        t_top = state.top_temperature
        if self._settings.param_utbl:
            t_surf = self._utbl_tsurf(t_top)
        else:
            t_surf = t_top
        state.heat_flux[-1, :] = (
            self._settings.emissivity
            * sp_constants.Stefan_Boltzmann
            * (np.power(t_surf, 4) - self._settings.equilibrium_temperature**4)
        )

    def _utbl_tsurf(self, t_interior: npt.NDArray) -> npt.NDArray:
        """Compute surface radiating temperature accounting for the UTBL.

        Solves: b * x^3 + x - T = 0, where x = T_surf, T = T_interior, b = param_utbl_const.
        Standard cubic form: x^3 + (1/b)*x - T/b = 0.
        Cardano's formula gives the real root.

        Parameters
        ----------
        t_interior : npt.NDArray
            Interior temperature at the surface node [K].

        Returns
        -------
        npt.NDArray
            Surface radiating temperature [K].
        """
        b = self._settings.param_utbl_const
        p = 1.0 / b
        q = -t_interior / b
        discriminant = q**2 / 4.0 + p**3 / 27.0
        sqrt_disc = np.sqrt(discriminant)
        return np.cbrt(-q / 2.0 + sqrt_disc) + np.cbrt(-q / 2.0 - sqrt_disc)

    def apply_flux_inner_boundary_condition(self, state: State) -> None:
        """Applies the flux boundary condition to the state at the inner boundary.

        Args:
            state: The state to apply the boundary conditions to

        Equivalent to CORE_BC in C code.
            1: Simple core cooling
            2: Prescribed heat flux
            3: Prescribed temperature
        """
        if self._settings.inner_boundary_condition == 1:
            self.core_cooling(state)
        elif self._settings.inner_boundary_condition == 2:
            state.heat_flux[0, :] = self._settings.inner_boundary_value
        elif self._settings.inner_boundary_condition == 3:
            pass
            # raise NotImplementedError
        else:
            msg: str = (
                f"inner_boundary_condition = {self._settings.inner_boundary_condition} is unknown"
            )
            raise ValueError(msg)

    def core_cooling(self, state: State) -> None:
        """Applies a core cooling heat flux according to Eq. (37) of Bower et al., 2018.

        The core is modelled as a well-mixed reservoir with an effective
        temperature T_core = tfac_core_avg * T_cmb. The factor tfac_core_avg
        accounts for the adiabatic temperature gradient within the core
        (mass-weighted average core temperature / CMB temperature).
        Default 1.147 is for Earth-like parameters (Bower+2018, Table 2).

        Parameters
        ----------
        state : State
            The state to apply the boundary condition to.
        """
        # Core thermal capacity: C_core = rho_core * cp_core * V_core
        r_cmb = float(np.squeeze(self._mesh.basic.radii[0]))
        core_capacity = (
            4 / 3 * np.pi * r_cmb**3
            * self._mesh.settings.core_density
            * self._settings.core_heat_capacity
        )

        # First mantle cell thermal capacity: C_cell = rho * cp * V_cell
        cap_stag = state.capacitance_staggered()  # rho * cp, may be float or array
        cap_first = float(np.squeeze(cap_stag[0])) if hasattr(cap_stag, '__getitem__') else float(cap_stag)
        cell_capacity = float(np.squeeze(self._mesh.basic.volume[0])) * cap_first

        # Geometric correction: area ratio between first interior face and CMB
        r_above = float(np.squeeze(self._mesh.basic.radii[1]))
        radius_ratio = r_above / r_cmb

        # Core buffering factor (Bower+2018 Eq. 37):
        # alpha = (R_1/R_0)^2 / (1 + C_cell / (C_core * tfac))
        # When C_core >> C_cell: alpha -> (R_1/R_0)^2 (core tracks mantle)
        # When C_core << C_cell: alpha -> 0 (core absorbs all heat)
        tfac = self._settings.tfac_core_avg
        alpha = radius_ratio**2 / (cell_capacity / (core_capacity * tfac) + 1)

        state.heat_flux[0, :] = alpha * state.heat_flux[1, :]
