"""Phase property configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import attrs

if TYPE_CHECKING:
    from aragog.config.scalings import ScalingsConfig

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
    scalings_: ScalingsConfig | None = attrs.field(init=False, default=None)

    def scale_attributes(self, scalings: ScalingsConfig) -> None:
        """Apply non-dimensionalization to numeric properties.

        String properties (file paths) are not scaled here; they are
        scaled when loaded by the phase evaluator.
        """
        self.scalings_ = scalings
        for name in ("density", "heat_capacity", "melt_fraction",
                      "thermal_conductivity", "thermal_expansivity",
                      "viscosity", "entropy"):
            value = getattr(self, name)
            try:
                scaling = getattr(scalings, name)
                scaled_value = value / scaling
                setattr(self, name, scaled_value)
                logger.info(
                    "%s is a number (value=%s, scaling=%s, scaled=%s)",
                    name, value, scaling, scaled_value,
                )
            except AttributeError:
                logger.info("No scaling found for %s", name)
            except TypeError:
                logger.info(
                    "%s is a string (file path), will be scaled later", name
                )


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
    scalings_: ScalingsConfig | None = attrs.field(init=False, default=None)

    def scale_attributes(self, scalings: ScalingsConfig) -> None:
        """Apply non-dimensionalization."""
        self.scalings_ = scalings
        self.latent_heat_of_fusion /= scalings.latent_heat_per_mass
        self.grain_size /= scalings.radius
