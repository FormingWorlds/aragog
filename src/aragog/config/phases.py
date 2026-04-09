"""Phase property configuration."""

from __future__ import annotations

import logging

import attrs

logger: logging.Logger = logging.getLogger(__name__)


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
    entropy: float | str = ""


@attrs.define
class MixedPhaseConfig:
    """Mixed-phase (mushy zone) parameters.

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
    """

    latent_heat_of_fusion: float
    rheological_transition_melt_fraction: float
    rheological_transition_width: float
    solidus: str
    liquidus: str
    phase: str
    phase_transition_width: float
    grain_size: float
    # cp_blend selects how mushy-zone Cp is computed:
    #   'latent' = SPIDER-parity v4 convention (latent-heat-augmented)
    #   'linear' = legacy v3 convention (linear blend of pure-phase Cp)
    cp_blend: str = 'latent'
