"""Boundary condition configuration."""

from __future__ import annotations

import logging
import math
import numbers

import attrs

logger: logging.Logger = logging.getLogger('fwl.' + __name__)

CORE_BC_MODES = ('quasi_steady', 'energy_balance', 'gradient', 'bower2018', 'core_module')
TOP_NODE_MODES = ('linear', 'limited')


def check_top_node(top_node, core_bc: str | None = None) -> None:
    """Refuse a ``top_node`` that is not one of ``TOP_NODE_MODES``, and ``'limited'`` on the
    ``'gradient'`` core, whose basic entropies come from its own state.

    Raises
    ------
    ValueError
        Naming the value and the modes.
    """
    if not isinstance(top_node, str) or top_node not in TOP_NODE_MODES:
        raise ValueError(f'unknown top_node {top_node!r}; the modes are {TOP_NODE_MODES}')
    if top_node == 'limited' and core_bc == 'gradient':
        raise ValueError("top_node = 'limited' is not available on core_bc = 'gradient'")


def _check_top_node_pair(instance, attribute, value) -> None:
    """Check ``top_node`` against ``core_bc`` at construction and on assignment of either."""
    if attribute.name == 'top_node':
        check_top_node(value, instance.core_bc)
    else:
        check_top_node(instance.top_node, value)


def _is_real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def check_subgrid_lid(
    subgrid_lid,
    lid_delta_on,
    lid_delta_lo,
    lid_handover_cells,
    lid_ra_crit,
    outer_boundary_condition: int,
    core_bc: str,
) -> None:
    """Refuse invalid sub-grid lid settings, and the lid on a boundary setup it does not support.

    ``subgrid_lid`` must be a bool; ``lid_delta_on`` [m] and ``lid_ra_crit`` finite and
    positive; ``lid_delta_lo`` [m] None or finite with ``0 < lid_delta_lo < lid_delta_on``;
    ``lid_handover_cells`` an integer of at least 1. With the lid on, the outer boundary must
    be the grey body (1) and the core boundary not ``'gradient'``. The check against the
    top-cell width needs the mesh and runs at solver setup.

    Raises
    ------
    ValueError
        Naming the field and the value.
    """
    if not isinstance(subgrid_lid, bool):
        raise ValueError(f'subgrid_lid must be true or false, got {subgrid_lid!r}')
    for name, value in (('lid_delta_on', lid_delta_on), ('lid_ra_crit', lid_ra_crit)):
        if not _is_real(value) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite number > 0, got {value!r}')
    if lid_delta_lo is not None and (
        not _is_real(lid_delta_lo)
        or not math.isfinite(lid_delta_lo)
        or not 0 < lid_delta_lo < lid_delta_on
    ):
        raise ValueError(
            f'lid_delta_lo must be a finite number in (0, lid_delta_on = {lid_delta_on!r}), '
            f'got {lid_delta_lo!r}'
        )
    if not isinstance(lid_handover_cells, numbers.Integral) or isinstance(
        lid_handover_cells, bool
    ):
        raise ValueError(f'lid_handover_cells must be an integer, got {lid_handover_cells!r}')
    if lid_handover_cells < 1:
        raise ValueError(f'lid_handover_cells must be >= 1, got {lid_handover_cells!r}')
    if subgrid_lid and outer_boundary_condition != 1:
        raise ValueError(
            'subgrid_lid needs the grey-body outer boundary (outer_boundary_condition = 1), '
            f'got {outer_boundary_condition}'
        )
    if subgrid_lid and core_bc == 'gradient':
        raise ValueError("subgrid_lid is not available on core_bc = 'gradient'")


def check_core_bc(core_bc: str, inner_boundary_condition: int, core_module_params) -> None:
    """Refuse an unknown ``core_bc``, and for ``'core_module'`` an inner boundary condition
    other than core cooling (1) or ``core_module_params`` the core module refuses.

    Raises
    ------
    ValueError
        Naming the unknown mode, the inner boundary condition, or the refused parameter.
    """
    if core_bc not in CORE_BC_MODES:
        raise ValueError(f'unknown core_bc {core_bc!r}; the modes are {CORE_BC_MODES}')
    if core_bc == 'core_module':
        if inner_boundary_condition != 1:
            raise ValueError(
                "core_bc = 'core_module' needs inner_boundary_condition = 1, got "
                f'{inner_boundary_condition}'
            )
        from aragog.core.module import split_core_module_params

        split_core_module_params(core_module_params)


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
    top_node : str
        Entropy of the top basic node: 'linear' (default) extrapolates the top 2 cells; 'limited'
        reduces that half step smoothly near a table edge, so the node rises with the top cell
        and stays in the table (``aragog.mesh.top_node_entropy``; not on the 'gradient' core,
        checked also on assignment).
    subgrid_lid : bool
        Carry a conductive lid thinner than the top cell as two extra states (thickness and
        energy per area); ``aragog.surface_lid``. Off by default.
    lid_delta_on : float
        Quasi-steady crust thickness [m] above which the lid acts; a thinner crust is left to
        the mesh. Must stay below 0.8 of the top-cell width (checked at solver setup).
    lid_delta_lo : float or None
        Thickness [m] of the hidden (phantom) lid and of the melt-back stop; None gives
        ``lid_delta_on / 2``.
    lid_handover_cells : int
        The lid hands over to the mesh when the lid base reaches the face below this many top
        cells.
    lid_ra_crit : float
        Critical Rayleigh number of the boundary-layer flux under the lid [-].
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
    core_bc: str = attrs.field(default='energy_balance', validator=_check_top_node_pair)
    # Flat parameter dict for core_bc='core_module'; keys documented in
    # aragog.core.module.build_core_module_budget (plus 'q_radio' [W] and 'ra_crit_cmb').
    core_module_params: dict | None = None
    top_node: str = attrs.field(default='linear', validator=_check_top_node_pair)
    subgrid_lid: bool = False
    lid_delta_on: float = 1.0
    lid_delta_lo: float | None = None
    lid_handover_cells: int = 3
    lid_ra_crit: float = 450.0

    def __attrs_post_init__(self) -> None:
        check_core_bc(self.core_bc, self.inner_boundary_condition, self.core_module_params)
        check_subgrid_lid(
            self.subgrid_lid,
            self.lid_delta_on,
            self.lid_delta_lo,
            self.lid_handover_cells,
            self.lid_ra_crit,
            self.outer_boundary_condition,
            self.core_bc,
        )
