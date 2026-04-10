"""Mesh configuration."""

from __future__ import annotations

import logging

import attrs

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
    adams_williamson_beta: float = 0.0  # 0 = derive from K_S
    surface_pressure: float = 0.0
    mass_coordinates: bool = False
    eos_file: str = ""
