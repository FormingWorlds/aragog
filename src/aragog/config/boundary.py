"""Boundary condition configuration."""

from __future__ import annotations

import logging

import attrs

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


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
    # Core boundary condition mode. Selects the formulation used
    # when ``inner_boundary_condition = 1`` (core cooling).
    #
    #   'quasi_steady' = alpha-factor heat-flux partition between
    #     the bottom mantle cell and the core, weighted by heat
    #     capacity ratio. State vector length N (entropy only).
    #     Default; produces stable cooling but underestimates the
    #     true CMB heat loss relative to a SPIDER-parity reference.
    #
    #   'energy_balance' = SPIDER-parity core BC. The entropy
    #     gradient at the CMB basic node is added as an extra state
    #     variable (mirror of SPIDER's ``dSdxi[ind_cmb]``) and its
    #     time derivative is integrated via SPIDER's ``bc.c:76-131``
    #     formula:
    #         d/dt(dSdr_cmb) = (2/dr) * ((-F_cmb*area_cmb)*fac_cmb
    #                                    - dSdt_s[0])
    #     where ``fac_cmb = cp_cmb / (cp_core*T_cmb*tfac*M_core)``.
    #     State vector length N+1.
    #
    #   'gradient' = entropy gradient as the primary state field;
    #     S is reconstructed by cumulative integration from the
    #     surface. State vector length N+2.
    #
    #   'bower2018' = T_core as an ODE state variable with F_cmb
    #     from conduction across the bottom half-cell. The
    #     conduction-only flux underestimates true core heat loss
    #     by orders of magnitude; this mode is retained for parity
    #     testing only and is not recommended for production.
    #
    # Default 'energy_balance' matches the PROTEUS production path.
    # Standalone callers that want the legacy alpha-factor behaviour
    # must set core_bc='quasi_steady' explicitly.
    core_bc: str = 'energy_balance'
    # Flat parameter dict for core_bc='core_module'; keys documented in
    # aragog.core.module.build_core_module_budget (plus 'q_radio' [W]).
    core_module_params: dict | None = None
