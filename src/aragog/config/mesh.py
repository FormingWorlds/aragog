"""Mesh configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    from aragog.config.scalings import ScalingsConfig

logger: logging.Logger = logging.getLogger(__name__)


@attrs.define
class MeshConfig:
    """Mesh and static pressure profile parameters.

    Parameters
    ----------
    outer_radius : float
        Outer radius [m].
    inner_radius : float
        Inner radius [m].
    number_of_nodes : int
        Number of basic mesh nodes.
    mixing_length_profile : str
        'constant' or 'nearest_boundary'.
    core_density : float
        Core density [kg/m^3].
    eos_method : int
        1: Adams-Williamson, 2: User-defined.
    surface_density : float
        Surface density for Adams-Williamson EOS [kg/m^3].
    gravitational_acceleration : float
        Gravitational acceleration [m/s^2].
    adiabatic_bulk_modulus : float
        Adiabatic bulk modulus [Pa].
    surface_pressure : float
        Surface pressure [Pa].
    mass_coordinates : bool
        Use mass-coordinate spacing.
    eos_file : str
        Path to user-defined EOS file.
    """

    outer_radius: float
    inner_radius: float
    number_of_nodes: int
    mixing_length_profile: str
    core_density: float
    eos_method: int = 1
    surface_density: float = 4000.0
    gravitational_acceleration: float = 9.81
    adiabatic_bulk_modulus: float = 260e9
    surface_pressure: float = 0.0
    mass_coordinates: bool = False
    eos_file: str = ""
    scalings_: ScalingsConfig | None = attrs.field(init=False, default=None)

    # EOS data loaded during scale_attributes (if eos_method == 2)
    eos_radius: np.ndarray | None = attrs.field(init=False, default=None)
    eos_pressure: np.ndarray | None = attrs.field(init=False, default=None)
    eos_density: np.ndarray | None = attrs.field(init=False, default=None)
    eos_gravity: np.ndarray | None = attrs.field(init=False, default=None)

    def scale_attributes(self, scalings: ScalingsConfig) -> None:
        """Apply non-dimensionalization."""
        self.scalings_ = scalings
        self.outer_radius /= scalings.radius
        self.inner_radius /= scalings.radius
        self.core_density /= scalings.density
        self.surface_density /= scalings.density
        self.gravitational_acceleration /= scalings.gravitational_acceleration
        self.adiabatic_bulk_modulus /= scalings.pressure
        self.surface_pressure /= scalings.pressure

        if self.eos_method == 2:
            if self.eos_file == "":
                raise ValueError("eos_method=2 requires an eos_file")
            arr = np.loadtxt(self.eos_file)
            self.eos_radius = arr[:, 0] / scalings.radius
            self.eos_pressure = arr[:, 1] / scalings.pressure
            self.eos_density = arr[:, 2] / scalings.density
            self.eos_gravity = arr[:, 3] / scalings.gravitational_acceleration
            if (
                (self.eos_radius[0] < self.inner_radius)
                or (self.eos_radius[-1] > self.outer_radius)
                or (self.eos_radius[-1] - self.eos_radius[0])
                < 0.75 * (self.outer_radius - self.inner_radius)
            ):
                raise ValueError("Radius array in EOS file: Values out of range.")
