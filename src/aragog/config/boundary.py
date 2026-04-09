"""Boundary condition configuration."""

from __future__ import annotations

import logging

import attrs

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
    # Core BC mode (default = 'quasi_steady' v3 behaviour):
    #   'quasi_steady' = legacy v3 alpha-factor heat-flux partition
    #     between mantle bottom cell and core based on heat capacity
    #     ratio. Standard since the Aragog refactor. Gives a -19 %
    #     T_core offset against SPIDER on R8 CHILI Earth (known
    #     limitation, see memory/spider_aragog_parity_v3_v4.md).
    #   'bower2018' = EXPERIMENTAL (2026-04-09 evening): T_core as
    #     ODE state variable, F_cmb from conduction
    #     (-k_eff * (T_above - T_core) / dr_half), dT_core/dt =
    #     -F_cmb * area_cmb / (M_core * Cp_core). The conduction-only
    #     flux underestimates the actual core heat loss by ~5 orders
    #     of magnitude (the real loss is dominated by convective
    #     coupling to the bottom mantle cell, not pure conduction
    #     across the CMB). Empirically gives T_core that's TOO HIGH
    #     compared to SPIDER. Kept in the codebase for future
    #     redesign work that adds a thermal boundary layer
    #     parameterization at the CMB. NOT recommended for
    #     production runs.
    core_bc: str = 'quasi_steady'
