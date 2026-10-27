"""Phase property configuration."""

from __future__ import annotations

import logging
import numbers
from typing import Any

import attrs

from aragog.rheology import SolidRheologyParams, delegate_rheology, merge_rheology

logger: logging.Logger = logging.getLogger('fwl.' + __name__)

# Allowed separation_viscosity modes and default, single-sourced here and
# imported by every site that parses, validates, or consumes the option.
SEPARATION_VISCOSITY_MODES: tuple[str, str] = ('melt', 'mixture')
SEPARATION_VISCOSITY_DEFAULT: str = 'melt'
# Separation buoyancy: against the mixture (McKenzie 1984) or the solid (Abe 1995).
SEPARATION_BUOYANCY_MODES: tuple[str, str] = ('mixture', 'solid')
SEPARATION_BUOYANCY_DEFAULT: str = 'mixture'
# Width of the tanh blend at the solidus and liquidus, in melt-fraction units (PROTEUS default).
MATPROP_SMOOTH_WIDTH_DEFAULT: float = 0.01
# Widths that warn (test tables): at 1e-3 Cp changes by 6e-4 across the solidus (as at 0 for
# far smaller widths); from 0.1 Cp jumps by 4.1e-5 in the middle of the melting interval.
MATPROP_SMOOTH_WIDTH_WARN_MIN: float = 1e-3
MATPROP_SMOOTH_WIDTH_WARN_MAX: float = 0.05


def check_matprop_smooth_width(value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is a real scalar strictly between 0 and 1.

    At 0 the properties switch between the two-phase and the single-phase branch exactly at
    the solidus and the liquidus; from 1 the blend spans the whole melting interval. A width
    outside [``MATPROP_SMOOTH_WIDTH_WARN_MIN``, ``MATPROP_SMOOTH_WIDTH_WARN_MAX``] logs a
    warning: below it the properties change almost as sharply as at 0, above it they jump in
    the middle of the melting interval, where the single-phase value changes tables.
    """
    # The range comparison first, so float() of a huge int never overflows; a bool is 0 or 1.
    if not (isinstance(value, numbers.Real) and 0 < value < 1 and 0.0 < float(value) < 1.0):
        raise ValueError(f'matprop_smooth_width must lie in (0, 1), got {value!r}')
    if not MATPROP_SMOOTH_WIDTH_WARN_MIN <= value <= MATPROP_SMOOTH_WIDTH_WARN_MAX:
        logger.warning(
            'matprop_smooth_width = %r lies outside [%g, %g]: below it the properties change '
            'almost as sharply as at 0 at the solidus and the liquidus, above it they jump in '
            'the middle of the melting interval',
            value,
            MATPROP_SMOOTH_WIDTH_WARN_MIN,
            MATPROP_SMOOTH_WIDTH_WARN_MAX,
        )


@delegate_rheology
@attrs.define(init=False)
class PhaseConfig:
    """Single-phase (solid or liquid) material properties.

    Each property can be a float (constant value) or a str (path to
    a lookup table file). Rheology parameters are owned by ``SolidRheologyParams``.

    Parameters
    ----------
    density : float or str
        Reference density [kg/m^3] or path to lookup table.
    heat_capacity : float or str
        Specific heat capacity [J/kg/K] or path to lookup table.
    melt_fraction : float
        Melt fraction (0 for solid, 1 for liquid).
    thermal_conductivity : float or str
        Thermal conductivity [W/m/K] or path to lookup table.
    thermal_expansivity : float or str
        Thermal expansivity [1/K] or path to lookup table.
    viscosity : float or str
        Dynamic viscosity [Pa s] or path to lookup table.
    entropy : float or str, optional
        Reference entropy [J/kg/K] or path to lookup table. Empty string means unused.
    rheology : SolidRheologyParams, optional
        Solid-state rheology parameters container. Each ``SolidRheologyParams``
        field is also exposed as a read-only property on ``PhaseConfig`` via
        ``delegate_rheology``.
    **flat_rheo : Any
        Flat rheology attributes forwarded to rheology container.
    """

    density: float | str
    heat_capacity: float | str
    melt_fraction: float
    thermal_conductivity: float | str
    thermal_expansivity: float | str
    viscosity: float | str
    entropy: float | str = ''
    rheology: SolidRheologyParams = attrs.field(factory=SolidRheologyParams)

    def __init__(
        self,
        density: float | str,
        heat_capacity: float | str,
        melt_fraction: float,
        thermal_conductivity: float | str,
        thermal_expansivity: float | str,
        viscosity: float | str,
        entropy: float | str = '',
        rheology: SolidRheologyParams | None = None,
        **flat_rheo: Any,
    ):
        if viscosity is None:
            raise ValueError('viscosity must not be None')
        self.density = density
        self.heat_capacity = heat_capacity
        self.melt_fraction = melt_fraction
        self.thermal_conductivity = thermal_conductivity
        self.thermal_expansivity = thermal_expansivity
        self.viscosity = viscosity
        self.entropy = entropy
        self.rheology = merge_rheology(rheology, flat_rheo)


@attrs.define
class MixedPhaseConfig:
    r"""Mixed-phase (mushy zone) parameters.

    Parameters
    ----------
    latent_heat_of_fusion : float
        Latent heat [J/kg].
    rheological_transition_melt_fraction : float
        Melt fraction at rheological transition.
    rheological_transition_width : float
        Width of the tanh smoothing around the transition.
    solidus : str
        Path to solidus lookup file.
    liquidus : str
        Path to liquidus lookup file.
    phase : str
        Active phase mode: 'solid', 'liquid', 'mixed', or 'composite'.
    phase_transition_width : float
        Width of smoothing at phase boundaries.
    grain_size : float
        Grain size [m] for permeability calculations.
    separation_viscosity : str
        Drag viscosity source for gravitational separation:
        ``'melt'`` (single-phase liquid viscosity, matches SPIDER's
        ``GetGravitationalHeatFlux``) or ``'mixture'`` (the
        rheological-transition-blended bulk viscosity). Default
        ``'melt'``.
    separation_buoyancy : str
        Buoyancy of the separation velocity: ``'mixture'`` (against the
        mixture, McKenzie 1984 Eq. 3.7: a factor ``1 - phi`` on the
        Blake-Kozeny-Carman and Rumpf-Gupte terms) or ``'solid'`` (against
        the solid, Abe 1995 Eq. 40, as SPIDER). Default ``'mixture'``.
    cp_blend : str
        Mushy-zone Cp blending mode: ``'latent'`` (SPIDER-parity,
        latent-heat-augmented) or ``'linear'`` (linear blend of
        pure-phase Cp without the latent term). Default ``'latent'``.
    matprop_smooth_width : float
        Width of the tanh blend between the two-phase and the
        single-phase properties at the solidus and the liquidus, in
        melt-fraction units, strictly between 0 and 1. Default 0.01,
        as in PROTEUS.
    const_properties : bool
        Replace the EOS-tabulated $(\rho, c_p, \alpha, k)$ and
        $\log_{10}\eta$ with the seven constant-property values
        below. Mirrors SPIDER's ``-use_const_properties``. Default
        False (use the EOS tables).
    const_rho : float
        Constant density [kg/m^3] when ``const_properties`` is True.
    const_Cp : float
        Constant heat capacity [J/(kg K)] when ``const_properties``
        is True.
    const_alpha : float
        Constant thermal expansivity [1/K] when ``const_properties``
        is True.
    const_cond : float
        Constant thermal conductivity [W/(m K)] when
        ``const_properties`` is True.
    const_log10visc : float
        Constant $\log_{10}$ dynamic viscosity [log10(Pa s)] when
        ``const_properties`` is True.
    const_T_ref : float
        Reference temperature [K] for the constant-properties EOS,
        used to anchor $T(P, S)$ on the analytic isentrope.
    const_S_ref : float
        Reference entropy [J/(kg K)] for the constant-properties
        EOS, paired with ``const_T_ref``.
    """

    latent_heat_of_fusion: float
    rheological_transition_melt_fraction: float
    rheological_transition_width: float
    solidus: str
    liquidus: str
    phase: str
    phase_transition_width: float
    grain_size: float
    separation_viscosity: str = attrs.field(
        default=SEPARATION_VISCOSITY_DEFAULT,
        validator=attrs.validators.in_(SEPARATION_VISCOSITY_MODES),
    )
    separation_buoyancy: str = attrs.field(
        default=SEPARATION_BUOYANCY_DEFAULT,
        validator=attrs.validators.in_(SEPARATION_BUOYANCY_MODES),
    )
    cp_blend: str = 'latent'
    matprop_smooth_width: float = attrs.field(
        default=MATPROP_SMOOTH_WIDTH_DEFAULT,
        validator=lambda _inst, _attr, value: check_matprop_smooth_width(value),
    )
    const_properties: bool = False
    const_rho: float = 4000.0
    const_Cp: float = 1000.0
    const_alpha: float = 1e-5
    const_cond: float = 4.0
    const_log10visc: float = 2.0
    const_T_ref: float = 3500.0
    const_S_ref: float = 3000.0
