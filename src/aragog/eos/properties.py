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

import logging
from dataclasses import KW_ONLY, dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline

from aragog.eos.base import PropertyProtocol
from aragog.utilities import FloatOrArray

logger: logging.Logger = logging.getLogger(__name__)


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
        self._x_bounds = (self.value[0, 0], self.value[-1, 0])

    def eval(self, pressure: npt.NDArray) -> npt.NDArray:
        p_arr = np.asarray(pressure).ravel()
        if len(p_arr) > 0 and (np.any(p_arr < self._x_bounds[0]) or np.any(p_arr > self._x_bounds[1])):
            logger.warning(
                "%s: pressure outside table range [%.2e, %.2e]",
                self.name, self._x_bounds[0], self._x_bounds[1],
            )
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
        # Use cubic interpolation (kx=3, ky=3) for accuracy consistent with the
        # 2nd-order FV discretization. Falls back to bilinear if grid is too small.
        nx, ny = len(x_values), len(y_values)
        kx = min(3, nx - 1)
        ky = min(3, ny - 1)
        self._lookup = RectBivariateSpline(x_values, y_values, z_values, kx=kx, ky=ky, s=0)
        # Store bounds for out-of-range warnings
        self._x_bounds = (x_values[0], x_values[-1])
        self._y_bounds = (y_values[0], y_values[-1])

    def prepare_data_for_spline(self, data):
        """Reshape tabular data onto a regular grid for RectBivariateSpline."""
        x_values = np.unique(data[:, 0])  # Unique pressure values
        y_values = np.unique(data[:, 1])  # Unique temperature values

        z_values = np.full((x_values.size, y_values.size), np.nan)

        x_indices = np.searchsorted(x_values, data[:, 0])
        y_indices = np.searchsorted(y_values, data[:, 1])

        z_values[x_indices, y_indices] = data[:, 2]

        # Validate: check for missing grid points
        n_nan = np.sum(np.isnan(z_values))
        if n_nan > 0:
            logger.warning(
                "%s: %d of %d grid points are NaN (%.1f%%). "
                "Interpolation may be unreliable near gaps.",
                self.name, n_nan, z_values.size, 100 * n_nan / z_values.size,
            )

        return x_values, y_values, z_values

    def eval(self, temperature: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
        # Warn if queried outside table bounds (extrapolation is unreliable)
        p_arr = np.asarray(pressure).ravel()
        t_arr = np.asarray(temperature).ravel()
        if len(p_arr) > 0:
            if np.any(p_arr < self._x_bounds[0]) or np.any(p_arr > self._x_bounds[1]):
                logger.warning(
                    "%s: pressure outside table range [%.2e, %.2e]",
                    self.name, self._x_bounds[0], self._x_bounds[1],
                )
            if np.any(t_arr < self._y_bounds[0]) or np.any(t_arr > self._y_bounds[1]):
                logger.warning(
                    "%s: temperature outside table range [%.2e, %.2e]",
                    self.name, self._y_bounds[0], self._y_bounds[1],
                )
        return self._lookup(pressure, temperature, grid=False)

    def __call__(self, temperature: npt.NDArray, pressure: npt.NDArray) -> npt.NDArray:
        return self.eval(temperature, pressure)
