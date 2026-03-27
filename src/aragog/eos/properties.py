#
# Copyright 2024 Dan J. Bower
#
# This file is part of Aragog.
#
# Aragog is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Aragog is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Aragog. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Property classes: constant, 1-D lookup, and 2-D lookup."""

from dataclasses import KW_ONLY, dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline

from aragog.eos.base import PropertyProtocol
from aragog.utilities import FloatOrArray


@dataclass
class ConstantProperty(PropertyProtocol):
    """A property with a constant value

    Args:
        name: Name of the property
        value: The constant value

    Attributes:
        name: Name of the property
        value: The constant value
        ndim: Number of dimensions, which is equal to zero for a constant property
    """

    name: str
    _: KW_ONLY
    value: float
    ndim: int = field(init=False, default=0)

    def eval(self) -> float:
        return self.value

    def __call__(self, temperature: npt.NDArray, pressure: npt.NDArray) -> float:
        """Evaluates the property.

        Args:
            temperature: Temperature
            pressure: Pressure
        """
        del temperature
        del pressure
        return self.eval()


@dataclass
class LookupProperty1D(PropertyProtocol):
    """A property from a 1-D lookup

    Args:
        name: Name of the property
        value: A 2-D array, with x values in the first column and y values in the second column.

    Attributes:
        name: Name of the property
        value: A 2-D array
        ndim: Number of dimensions, which is equal to one for a 1-D lookup
    """

    name: str
    _: KW_ONLY
    value: npt.NDArray
    ndim: int = field(init=False, default=1)
    _gradient: npt.NDArray = field(init=False)

    def __post_init__(self):
        # Sort the data to ensure x is increasing
        self.value = self.value[self.value[:, 0].argsort()]
        self._gradient = np.gradient(self.value[:, 1], self.value[:, 0])

    def eval(self, pressure: npt.NDArray) -> npt.NDArray:
        return np.interp(pressure, self.value[:, 0], self.value[:, 1])

    def gradient(self, pressure: npt.NDArray) -> npt.NDArray:
        """Computes the gradient"""
        return np.interp(pressure, self.value[:, 0], self._gradient)

    def __call__(self, temperature: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
        del temperature
        return self.eval(pressure)


@dataclass
class LookupProperty2D(PropertyProtocol):
    """A property from a 2-D lookup

    Args:
        name: Name of the property
        value: The 2-D array

    Attributes:
        name: Name of the property
        value: The 2-D array
        ndim: Number of dimensions, which is equal to two for a 2-D lookup
    """

    name: str
    _: KW_ONLY
    value: npt.NDArray
    ndim: int = field(init=False, default=2)
    _lookup: RectBivariateSpline = field(init=False)

    def __post_init__(self):
        # Prepare data for spline
        x_values, y_values, z_values = self.prepare_data_for_spline(self.value)
        self._lookup = RectBivariateSpline(x_values, y_values, z_values, kx=1, ky=1, s=0)

    def prepare_data_for_spline(self, data):
        """Ensure your data is on a regular grid for RectBivariateSpline"""
        # Extract x, y, and z values
        x_values = np.unique(data[:, 0])  # Unique pressure values
        y_values = np.unique(data[:, 1])  # Unique temperature values

        # Create a grid for z values
        z_values = np.full((x_values.size, y_values.size), np.nan)

        # Find the indices of the x and y values in the unique arrays
        x_indices = np.searchsorted(x_values, data[:, 0])
        y_indices = np.searchsorted(y_values, data[:, 1])

        # Fill the z_values grid
        z_values[x_indices, y_indices] = data[:, 2]

        return x_values, y_values, z_values

    def eval(self, temperature: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
        return self._lookup(pressure, temperature, grid=False)

    def __call__(self, temperature: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
        return self.eval(temperature, pressure)
