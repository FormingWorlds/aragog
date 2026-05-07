"""Fixed mesh dataclass."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from aragog.parser import _MeshParameters
from aragog.utilities import is_monotonic_increasing

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class FixedMesh:
    """A fixed mesh

    Args:
        settings: Mesh parameters
        radii: Radii of the mesh
        mass_radii: Mass coordinates of the mesh
        outer_boundary: Outer boundary for computing depth below the surface
        inner_boundary: Inner boundary for computing height above the base

    Attributes:
        settings: Mesh parameters
        radii: Radii of the mesh
        mass_radii: Mass coordinates of the mesh
        outer_boundary: Outer boundary for computing depth below the surface
        inner_boundary: Inner boundary for computing height above the base
        area: Surface area
        delta_mesh: Delta radii in mass coordinates
        depth: Depth below the outer boundary
        height: Height above the inner boundary
        mixing_length: Mixing length
        mixing_length_squared: Mixing length squared
        mixing_length_cubed: Mixing length cubed
        number_of_nodes: Number of nodes
        volume: Volume of the spherical shells defined between neighbouring radii
        total_volume: Total volume
    """

    settings: _MeshParameters
    radii: npt.NDArray
    mass_radii: npt.NDArray
    outer_boundary: float
    inner_boundary: float

    def __post_init__(self):
        if not is_monotonic_increasing(self.radii):
            msg: str = 'Mesh must be monotonically increasing'
            logger.error(msg)
            raise ValueError(msg)

    @cached_property
    def area(self) -> npt.NDArray:
        """Area"""
        return 4 * np.pi * np.square(self.radii)

    @cached_property
    def delta_mesh(self) -> npt.NDArray:
        return np.diff(self.mass_radii, axis=0)

    @cached_property
    def depth(self) -> npt.NDArray:
        return self.outer_boundary - self.radii

    @cached_property
    def height(self) -> npt.NDArray:
        return self.radii - self.inner_boundary

    @cached_property
    def _mesh_cubed(self) -> npt.NDArray:
        return np.power(self.radii, 3)

    @cached_property
    def mixing_length(self) -> npt.NDArray:
        if self.settings.mixing_length_profile == 'nearest_boundary':
            logger.debug('Set mixing length profile to nearest boundary')
            mixing_length = np.minimum(
                self.outer_boundary - self.radii, self.radii - self.inner_boundary
            )
        elif self.settings.mixing_length_profile == 'constant':
            logger.debug('Set mixing length profile to constant')
            assert self.outer_boundary is not None
            assert self.inner_boundary is not None
            mixing_length = (
                np.ones_like(self.radii) * 0.25 * (self.outer_boundary - self.inner_boundary)
            )
        else:
            msg: str = (
                f'Mixing length profile = {self.settings.mixing_length_profile} is unknown'
            )
            raise ValueError(msg)

        return mixing_length

    @cached_property
    def mixing_length_cubed(self) -> npt.NDArray:
        return np.power(self.mixing_length, 3)

    @cached_property
    def mixing_length_squared(self) -> npt.NDArray:
        return np.square(self.mixing_length)

    @cached_property
    def number_of_nodes(self) -> int:
        return self.radii.size

    @cached_property
    def volume(self) -> npt.NDArray:
        volume: npt.NDArray = 4 / 3 * np.pi * (self._mesh_cubed[1:] - self._mesh_cubed[:-1])

        return volume

    @cached_property
    def total_volume(self) -> float:
        return 4 / 3 * np.pi * (self._mesh_cubed[-1] - self._mesh_cubed[0]).item()
