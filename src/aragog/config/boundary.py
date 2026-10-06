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
    core_bc : str
        Core boundary formulation for ``inner_boundary_condition = 1``:
        'quasi_steady' (alpha-factor heat-flux partition between the bottom
        cell and the core by heat-capacity ratio; state length N; stable but
        underestimates the CMB heat loss against a SPIDER-parity reference),
        'energy_balance' (default, the PROTEUS production path: SPIDER-parity
        BC with the CMB entropy gradient as an extra state, integrated by
        SPIDER's ``bc.c:76-131`` formula
        d/dt(dSdr_cmb) = (2/dr) ((-F_cmb area_cmb) fac_cmb - dSdt_s[0]),
        fac_cmb = cp_cmb / (cp_core T_cmb tfac M_core); state length N+1),
        'gradient' (entropy gradient as the primary state, S rebuilt by
        integration from the surface; state length N+2),
        'bower2018' (T_core as a state with F_cmb from conduction across the
        bottom half cell, orders of magnitude below the true heat loss;
        parity testing only) or 'core_module' (the aragog.core evolution
        budget with dSdr_cmb and T_core as states; state length N+2).
        Standalone callers that want the alpha-factor behaviour set
        'quasi_steady' explicitly.
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
    # Core boundary mode for inner_boundary_condition = 1 (see the class docstring).
    core_bc: str = 'energy_balance'
    # Flat parameter dict for core_bc='core_module'; keys documented in
    # aragog.core.module.build_core_module_budget (plus 'q_radio' [W] and 'ra_crit_cmb').
    core_module_params: dict | None = None
