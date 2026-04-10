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
    # Core BC mode (default = 'quasi_steady' v3 behaviour for now,
    # to be flipped to 'energy_balance' once Path A is fully validated):
    #
    #   'quasi_steady' = legacy v3 alpha-factor heat-flux partition
    #     between mantle bottom cell and core based on heat capacity
    #     ratio. Standard since the Aragog refactor. Gives a -19 %
    #     T_core offset against SPIDER on R8 CHILI Earth (known
    #     limitation that Path A is meant to close).
    #
    #   'energy_balance' = Path A SPIDER bit-parity core BC. Adds the
    #     entropy gradient at the CMB basic node as an extra state
    #     variable (mirror of SPIDER's dSdxi[ind_cmb]) and integrates
    #     its time derivative via SPIDER's bc.c:76-131 formula:
    #         d/dt(dSdr_cmb) = (2/dr) * ((-F_cmb*area_cmb)*fac_cmb
    #                                    - dSdt_s[0])
    #     where fac_cmb = cp_cmb / (cp_core*T_cmb*tfac*M_core).
    #     This is the proper SPIDER-mirroring implementation; the
    #     state vector grows by one element. NOT yet the default
    #     until full CHILI v5 validation passes.
    #
    #   'bower2018' = EXPERIMENTAL (2026-04-09 evening): T_core as
    #     ODE state variable, F_cmb from conduction
    #     (-k_eff * (T_above - T_core) / dr_half). The conduction-
    #     only flux underestimates the actual core heat loss by ~5
    #     orders of magnitude. Failed empirical validation; kept in
    #     the codebase as a tombstone. DO NOT use for production.
    core_bc: str = 'quasi_steady'
