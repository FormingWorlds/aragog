"""Boundary condition configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from aragog.config.scalings import ScalingsConfig

logger: logging.Logger = logging.getLogger(__name__)


@attrs.define
class BoundaryConfig:
    """Boundary condition parameters.

    Parameters
    ----------
    outer_boundary_condition : int
        1: Grey-body, 2: Zahnle, 3: Atmodeller, 4: Prescribed flux, 5: Prescribed T
    outer_boundary_value : float
        Value for outer BC (flux in W/m^2 or T in K, depending on type).
    inner_boundary_condition : int
        1: Core cooling, 2: Prescribed flux, 3: Prescribed T
    inner_boundary_value : float
        Value for inner BC.
    emissivity : float
        Surface emissivity for grey-body BC.
    equilibrium_temperature : float
        Equilibrium temperature for grey-body BC [K].
    core_heat_capacity : float
        Core heat capacity [J/(kg K)].
    tfac_core_avg : float
        Core adiabat correction factor (Bower+2018).
    param_utbl : bool
        Enable upper thermal boundary layer parameterization.
    param_utbl_const : float
        UTBL constant.
    """

    outer_boundary_condition: int
    outer_boundary_value: float
    inner_boundary_condition: int
    inner_boundary_value: float
    emissivity: float
    equilibrium_temperature: float
    core_heat_capacity: float
    tfac_core_avg: float = 1.147
    param_utbl: bool = False
    param_utbl_const: float = 1.0e-7
    scalings_: ScalingsConfig | None = attrs.field(init=False, default=None)

    def scale_attributes(self, scalings: ScalingsConfig) -> None:
        """Apply non-dimensionalization."""
        self.scalings_ = scalings
        self.equilibrium_temperature /= scalings.temperature
        self.core_heat_capacity /= scalings.heat_capacity
        if self.param_utbl:
            self.param_utbl_const *= scalings.temperature**2
        else:
            self.param_utbl_const = 0.0
        self._scale_inner_boundary_condition()
        self._scale_outer_boundary_condition()

    def _scale_inner_boundary_condition(self) -> None:
        if self.inner_boundary_condition == 1:
            self.inner_boundary_value = 0
        elif self.inner_boundary_condition == 2:
            self.inner_boundary_value /= self.scalings_.heat_flux
        elif self.inner_boundary_condition == 3:
            self.inner_boundary_value /= self.scalings_.temperature
        else:
            raise ValueError(
                f"inner_boundary_condition = {self.inner_boundary_condition} is unknown"
            )

    def _scale_outer_boundary_condition(self) -> None:
        if self.outer_boundary_condition in (1, 2, 3):
            pass
        elif self.outer_boundary_condition == 4:
            self.outer_boundary_value /= self.scalings_.heat_flux
        elif self.outer_boundary_condition == 5:
            self.outer_boundary_value /= self.scalings_.temperature
        else:
            raise ValueError(
                f"outer_boundary_condition = {self.outer_boundary_condition} is unknown"
            )
