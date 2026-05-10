"""Pressure EOS classes for mesh density and pressure profiles."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid
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
    def set_staggered_pressure(
        self,
        staggered_radii: npt.NDArray,
    ) -> None: ...


class AdamsWilliamsonEOS(EOS):
    r"""Adams-Williamson equation of state (EOS).

    Matches SPIDER's eos_adamswilliamson.c: exponential density profile

        rho(z) = rhos * exp(beta * z)

    where z = R_surf - r is depth. The parameter beta [1/m] is passed
    directly from the config (adams_williamson_beta), NOT derived from
    the adiabatic bulk modulus K_S. This ensures the density profile
    is identical to SPIDER's when both codes use the same beta.

    The previous implementation used a rational (hyperbolic) form
    rho = rhos * K_S / (K_S - rhos*g*depth) derived from K_S,
    which differs from SPIDER's exponential by up to 6% at CMB depth,
    causing a 3% R_int mismatch for the same target planet mass.
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
        # Use beta from config if set (> 0); fall back to K_S-derived value
        beta_cfg = float(getattr(self._settings, 'adams_williamson_beta', 0.0))
        if beta_cfg > 0:
            self._beta = beta_cfg
        else:
            self._beta = (
                self._surface_density
                * self._gravitational_acceleration
                / self._adiabatic_bulk_modulus
            )
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

    def set_staggered_pressure(
        self,
        staggered_radii: npt.NDArray,
    ) -> None:
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
        volume_shell = 4 / 3 * np.pi * (np.power(radii[1:], 3.0) - np.power(radii[:-1], 3.0))

        return mass_shell / volume_shell

    def get_density(self, pressure: FloatOrArray) -> npt.NDArray:
        r"""Computes density from pressure (SPIDER parity):

        $$
        \rho(P) = \rho_s + P \beta / g
        $$

        Matches SPIDER's eos_adamswilliamson.c:GetRho (line 116):
        ``rho = rhos - P * beta / gravity`` where gravity is negative.

        Args:
            pressure: Pressure

        Returns:
            Density
        """
        density: npt.NDArray = (
            self._surface_density + pressure * self._beta / self._gravitational_acceleration
        )

        return density

    def get_density_from_radii(self, radii: FloatOrArray) -> FloatOrArray:
        r"""Computes density from radii using SPIDER-parity exponential form:

        $$
            \rho(r) = \rho_s \exp(\beta (R_s - r))
        $$

        where $\rho_s$ is surface density, $\beta$ is the Adams-Williamson
        parameter [1/m], and $R_s$ is the surface radius. Matches SPIDER's
        eos_adamswilliamson.c via the chain rho(P(r)).

        Args:
            radii: Radii

        Returns
            Density
        """
        depth = self._outer_boundary - radii
        density: FloatOrArray = self._surface_density * np.exp(self._beta * depth)

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
        r"""Computes mass within radii using SPIDER-parity exponential form:

        $$
        m(r) = 4\pi \int r^2 \rho_s \exp(\beta (R_s - r)) dr
        $$

        The antiderivative is (matching SPIDER eos_adamswilliamson.c:191):

        $$
        M(r) = 4\pi \rho(r) \left( -\frac{2}{\beta^3} - \frac{r^2}{\beta}
               - \frac{2r}{\beta^2} \right)
        $$

        where rho(r) = rhos * exp(beta*(R_s - r)).

        Args:
            radii: Radii

        Returns:
            Mass within radii (from inner_boundary to radii)
        """
        beta = self._beta

        def mass_integral(r: FloatOrArray) -> npt.NDArray:
            rho = self.get_density_from_radii(r)
            return 4.0 * np.pi * rho * (-2.0 / beta**3 - r**2 / beta - 2.0 * r / beta**2)

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
        r"""Computes pressure from radii using SPIDER-parity exponential form:

        $$
        P(r) = \frac{\rho_s g}{\beta} \left( \exp(\beta (R_s - r)) - 1 \right)
              + P_{surf}
        $$

        Matches SPIDER's eos_adamswilliamson.c:GetPressureFromRadius.
        Note: g is positive here (unlike SPIDER where gravity is negative).

        Parameters
        ----------
        radii : FloatOrArray
            Radii at which to compute pressure.

        Returns
        -------
        npt.NDArray
            Pressure at the given radii.
        """
        depth = self._outer_boundary - radii
        pressure: npt.NDArray = (
            self._surface_density
            * self._gravitational_acceleration
            / self._beta
            * (np.exp(self._beta * depth) - 1.0)
            + self._surface_pressure
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
        r"""Computes radii from pressure (SPIDER parity):

        $$
        r(P) = R_s - \frac{1}{\beta} \ln\left(1 + \frac{P \beta}{\rho_s g}\right)
        $$

        Inverse of get_pressure_from_radii.

        Args:
            pressure: Pressure

        Returns:
            Radii
        """
        radii: npt.NDArray = (
            self._outer_boundary
            - np.log(
                1.0
                + (pressure - self._surface_pressure)
                * self._beta
                / (self._surface_density * self._gravitational_acceleration)
            )
            / self._beta
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
        self._basic_pressure = self._interp_pressure(basic_radii).reshape(-1, 1)
        basic_effective_density = self._interp_density(basic_radii).reshape(-1, 1)
        self._staggered_effective_density = 0.5 * (
            basic_effective_density[:-1, :] + basic_effective_density[1:, :]
        )
        # Assumes density and effective density are the same at basic nodes
        self._basic_density = basic_effective_density

        # Cumulative mass M(r) = integral_{r_eos[0]}^{r} 4 pi r^2 rho(r) dr,
        # precomputed on the EOS radius grid so get_mass_within_radii is O(1).
        # Anchored at r_eos[0]; differences M(r2) - M(r1) cancel the anchor.
        r_eos = np.asarray(settings.eos_radius)
        rho_eos = np.asarray(settings.eos_density)
        integrand = 4.0 * np.pi * r_eos**2 * rho_eos
        cum_mass = cumulative_trapezoid(integrand, r_eos, initial=0.0)
        self._interp_mass = PchipInterpolator(r_eos, cum_mass)

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

    def set_staggered_pressure(
        self,
        staggered_radii: npt.NDArray,
    ) -> None:
        """Set staggered pressure based on staggered radii."""
        self._staggered_pressure = self._interp_pressure(staggered_radii).reshape(-1, 1)

    def get_mass_within_radii(self, radii: FloatOrArray) -> npt.NDArray:
        r"""4pi-included cumulative mass M(r), anchored at the inner EOS radius.

        Mesh.get_planet_density and the mass-coordinate brentq loop use only
        differences M(r2) - M(r1), so the anchor cancels.
        """
        return np.asarray(self._interp_mass(np.asarray(radii)))
