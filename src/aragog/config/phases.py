"""Phase property configuration."""

from __future__ import annotations

import logging

import attrs

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


@attrs.define
class PhaseConfig:
    """Single-phase (solid or liquid) material properties.

    Each property can be a float (constant value) or a str (path to
    a lookup table file).

    Parameters
    ----------
    density : float or str
        Density [kg/m^3] or path to lookup.
    heat_capacity : float or str
        Heat capacity [J/(kg K)] or path to lookup.
    melt_fraction : float
        Melt fraction (0 for solid, 1 for liquid).
    thermal_conductivity : float or str
        Thermal conductivity [W/(m K)] or path to lookup.
    thermal_expansivity : float or str
        Thermal expansivity [1/K] or path to lookup.
    viscosity : float or str
        Dynamic viscosity [Pa s] or path to lookup.
    entropy : float or str
        Entropy [J/(kg K)] or path to lookup. Empty string means unused.
    """

    density: float | str
    heat_capacity: float | str
    melt_fraction: float
    thermal_conductivity: float | str
    thermal_expansivity: float | str
    viscosity: float | str
    entropy: float | str = ''


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
    cp_blend : str
        Mushy-zone Cp blending mode: ``'latent'`` (SPIDER-parity,
        latent-heat-augmented) or ``'linear'`` (linear blend of
        pure-phase Cp without the latent term). Default ``'latent'``.
    matprop_smooth_width : float
        Half-width of the tanh used to smooth phase-dependent
        material properties around the rheological transition.
        Default 0.0 reproduces SPIDER's convention of no smoothing
        on the property side; 0.01 is the typical JAX setting.
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
    separation_viscosity: str = 'melt'
    cp_blend: str = 'latent'
    matprop_smooth_width: float = 0.0
    const_properties: bool = False
    const_rho: float = 4000.0
    const_Cp: float = 1000.0
    const_alpha: float = 1e-5
    const_cond: float = 4.0
    const_log10visc: float = 2.0
    const_T_ref: float = 3500.0
    const_S_ref: float = 3000.0
