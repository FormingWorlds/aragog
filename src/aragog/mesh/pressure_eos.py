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
"""Pressure EOS classes for mesh density and pressure profiles."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from aragog.parser import _MeshParameters
from aragog.utilities import FloatOrArray


class EOS(ABC):
    """Generic EOS class"""

    @abstractmethod
    def staggered_effective_density(self) -> npt.NDArray: ...

    @abstractmethod
    def basic_density(self) -> npt.NDArray: ...

    @abstractmethod
    def basic_pressure(self) -> npt.NDArray: ...

    @abstractmethod
    def staggered_pressure(self) -> npt.NDArray: ...

    @abstractmethod
    def set_staggered_pressure(self, staggered_radii: npt.NDArray,) -> None: ...

class AdamsWilliamsonEOS(EOS):
    r"""Adams-Williamson equation of state (EOS).

    EOS due to adiabatic self-compression from the definition of the adiabatic bulk modulus:

    $$
    \left( \frac{{d\rho}}{{dP}} \right)_S = \frac{{\rho}}{{K_S}}
    $$

    where $\rho$ is density, $K_S$ the adiabatic bulk modulus, and $S$ is
    entropy.
    """

    def __init__(
        self,
        settings: _MeshParameters,
        basic_radii: npt.NDArray,
    ):
        self._settings: _MeshParameters = settings
        self._basic_radii: npt.NDArray = basic_radii
        self._outer_boundary = np.max(basic_radii)
        self._inner_boundary = np.min(basic_radii)
        self._surface_density: float = self._settings.surface_density
        self._gravitational_acceleration: float = self._settings.gravitational_acceleration
        self._adiabatic_bulk_modulus: float = self._settings.adiabatic_bulk_modulus
        self._surface_pressure: float = self._settings.surface_pressure
        self._basic_pressure = self.get_pressure_from_radii(basic_radii)
        self._basic_density = self.get_density_from_radii(basic_radii)
        self._staggered_effective_density = self.get_effective_density(basic_radii)

    @property
    def basic_pressure(self) -> npt.NDArray:
        """Pressure at basic nodes"""
        return self._basic_pressure

    @property
    def staggered_pressure(self) -> npt.NDArray:
        """Pressure at staggered nodes"""
        return self._staggered_pressure

    @property
    def basic_density(self) -> npt.NDArray:
        """Density at basic nodes"""
        return self._basic_density

    @property
    def staggered_effective_density(self) -> npt.NDArray:
        """Effective density at staggered nodes"""
        return self._staggered_effective_density

    def set_staggered_pressure(self, staggered_radii: npt.NDArray,) -> None:
        """Set staggered pressure based on staggered radii."""
        self._staggered_pressure = self.get_pressure_from_radii(staggered_radii)

    def get_effective_density(self, radii) -> npt.NDArray:
        r"""
        Computes effective density using density rho(r) integration
        over a spherical shell bounded by radii

        Args:
            radii: Radii array

        Returns:
            Effective Density array
        """

        mass_shell = self.get_mass_within_shell(radii)
        volume_shell = 4 / 3 * np.pi  \
            * (np.power(radii[1:],3.0) - np.power(radii[:-1],3.0))

        return mass_shell/volume_shell

    def get_density(self, pressure: FloatOrArray) -> npt.NDArray:
        r"""Computes density from pressure:

        $$
        \rho(P) = \rho_s \exp(P/K_S)
        $$

        where $\rho$ is density, $P$ is pressure, $\rho_s$ is surface density,
        and $K_S$ is adiabatic bulk modulus.

        Args:
            pressure: Pressure

        Returns:
            Density
        """
        density: npt.NDArray = self._surface_density * np.exp(
            pressure / self._adiabatic_bulk_modulus
        )

        return density

    def get_density_from_radii(self, radii: FloatOrArray) -> FloatOrArray:
        r"""Computes density from radii:

        $$
            \rho(r) = \frac{\rho_s K_S}{K_S + \rho_s g (r-r_s)}
        $$
        where $\rho$ is density, $r$ is radius, $\rho_s$ is surface density,
        $K_S$ is adiabatic bulk modulus, and $r_s$ is surface radius.

        Args:
            radii: Radii

        Returns
            Density
        """
        density: FloatOrArray = (self._surface_density * self._adiabatic_bulk_modulus) / (
            self._adiabatic_bulk_modulus
            + self._surface_density
            * self._gravitational_acceleration
            * (radii - self._outer_boundary)
        )

        return density

    def get_mass_element(self, radii: FloatOrArray) -> npt.NDArray:
        r"""Computes the mass element:

        $$
        \frac{\delta m}{\delta r} = 4 \pi r^2 \rho
        $$

        where $\delta m$ is the mass element, $r$ is radius, and $\rho$ is
        density.

        Args:
            radii: Radii

        Returns:
            The mass element at radii
        """
        mass_element: npt.NDArray = (
            4 * np.pi * np.square(radii) * self.get_density_from_radii(radii)
        )

        return mass_element

    def get_mass_within_radii(self, radii: FloatOrArray) -> npt.NDArray:
        r"""Computes mass within radii:

        $$
        m(r) = \int 4 \pi r^2 \rho dr
        $$
        where $m$ is mass, $r$ is radius, and $\rho$ is density.

        The integral was evaluated using WolframAlpha.

        Args:
            radii: Radii

        Returns:
            Mass within radii
        """
        a: float = self._surface_density
        b: float = self._adiabatic_bulk_modulus
        c: float = self._gravitational_acceleration
        d: float = self._outer_boundary
        beta: float = b / (a * c) - d

        def mass_integral(radii_: FloatOrArray) -> npt.NDArray:
            """Mass within radii including arbitrary constant of integration.

            Args:
                radii_: Radii

            Returns:
                Mass within radii
            """

            mass: npt.NDArray = (
                4
                * np.pi
                * b
                / c
                * (
                    -1.5 * beta * beta
                    - beta * radii_
                    + 0.5 * radii_ * radii_
                    + beta * beta * np.log(abs(beta + radii_))
                )
            )
            # + constant

            return mass

        mass: npt.NDArray = mass_integral(radii) - mass_integral(self._inner_boundary)

        return mass

    def get_mass_within_shell(self, radii: npt.NDArray) -> npt.NDArray:
        """Computes the mass within spherical shells bounded by radii.

        Args:
            radii: Radii

        Returns:
            Mass within the bounded spherical shells
        """
        mass: npt.NDArray = self.get_mass_within_radii(radii[1:]) - self.get_mass_within_radii(
            radii[:-1]
        )

        return mass

    def get_pressure_from_radii(self, radii: FloatOrArray) -> npt.NDArray:
        r"""Computes pressure from radii:

        $$
        P(r) = P_{surf} - K_S \ln \left( 1 + \frac{\rho_s g (r-r_s)}{K_S} \right)
        $$

        where $r$ is radius, $K_S$ is adiabatic bulk modulus, $P$ is pressure,
        $P_{surf}$ is the surface pressure (atmospheric overburden),
        $\rho_s$ is surface density, $g$ is gravitational acceleration, and
        $r_s$ is surface radius.

        Parameters
        ----------
        radii : FloatOrArray
            Radii at which to compute pressure.

        Returns
        -------
        npt.NDArray
            Pressure at the given radii.
        """
        pressure: npt.NDArray = self._surface_pressure - self._adiabatic_bulk_modulus * np.log(
            (
                self._adiabatic_bulk_modulus
                + self._surface_density
                * self._gravitational_acceleration
                * (radii - self._outer_boundary)
            )
            / self._adiabatic_bulk_modulus
        )

        return pressure

    def get_pressure_gradient(self, pressure: FloatOrArray) -> npt.NDArray:
        r"""Computes the pressure gradient:

        $$
        \frac{dP}{dr} = -g \rho
        $$

        where $\rho$ is density, $P$ is pressure, and  $g$ is gravitational
        acceleration.

        Args:
            pressure: Pressure

        Returns:
            Pressure gradient
        """
        dPdr: npt.NDArray = -self._gravitational_acceleration * self.get_density(pressure)

        return dPdr

    def get_radii_from_pressure(self, pressure: FloatOrArray) -> npt.NDArray:
        r"""Computes radii from pressure:

        $$
        P(r) = \int \frac{dP}{dr} dr = \int -g \rho_s \exp(P/K_S) dr
        $$

        And apply the boundary condition $P=0$ at $r=r_s$ to get:

        $$
        r(P) = \frac{K_s \left( \exp(-P/K_S)-1 \right)}{\rho_s g} + r_s
        $$

        where $r$ is radius, $K_S$ is adiabatic bulk modulus, $P$ is pressure,
        $\rho_s$ is surface density, $g$ is gravitational acceleration, and
        $r_s$ is surface radius.

        Args:
            pressure: Pressure

        Returns:
            Radii
        """
        radii: npt.NDArray = (
            self._adiabatic_bulk_modulus
            * (np.exp(-pressure / self._adiabatic_bulk_modulus) - 1)
            / (self._surface_density * self._gravitational_acceleration)
            + self._outer_boundary
        )

        return radii

class UserDefinedEOS(EOS):
    r"""User defined equation of state (EOS).

    Pressure field and effective density field on staggered nodes provided by the user.
    """

    def __init__(
        self,
        settings: _MeshParameters,
        basic_radii: npt.NDArray,
    ):
        self._interp_pressure = PchipInterpolator(settings.eos_radius, settings.eos_pressure)
        self._interp_density = PchipInterpolator(settings.eos_radius, settings.eos_density)
        self._basic_pressure = self._interp_pressure(basic_radii).reshape(-1,1)
        basic_effective_density = self._interp_density(basic_radii).reshape(-1,1)
        self._staggered_effective_density = 0.5*(
            basic_effective_density[:-1, :] + basic_effective_density[1:, :])
        # Assumes density and effective density are the same at basic nodes
        self._basic_density = basic_effective_density

    @property
    def basic_pressure(self) -> npt.NDArray:
        """Pressure at basic nodes"""
        return self._basic_pressure

    @property
    def staggered_pressure(self) -> npt.NDArray:
        """Pressure at staggered nodes"""
        return self._staggered_pressure

    @property
    def basic_density(self) -> npt.NDArray:
        """Effective density at basic nodes"""
        return self._basic_density

    @property
    def staggered_effective_density(self) -> npt.NDArray:
        """Effective density at staggered nodes"""
        return self._staggered_effective_density

    def set_staggered_pressure(self, staggered_radii: npt.NDArray,) -> None:
        """Set staggered pressure based on staggered radii."""
        self._staggered_pressure = self._interp_pressure(staggered_radii).reshape(-1,1)
