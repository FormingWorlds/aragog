"""Mesh subpackage: staggered mesh, fixed mesh, and pressure EOS classes."""

from __future__ import annotations

import logging
from dataclasses import field
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from aragog.mesh.fixed_mesh import FixedMesh
from aragog.mesh.pressure_eos import (
    EOS,
    AdamsWilliamsonEOS,
    UserDefinedEOS,
)
from aragog.parser import Parameters, _MeshParameters

__all__ = [
    "Mesh",
    "FixedMesh",
    "EOS",
    "AdamsWilliamsonEOS",
    "UserDefinedEOS",
]

logger: logging.Logger = logging.getLogger(__name__)


class Mesh:
    """A staggered mesh.

    The basic mesh is used for the flux calculations and the staggered mesh is used for the volume
    calculations.

    Args:
        parameters: Parameters
    """

    eos: EOS = field(init=False)

    def __init__(self, parameters: Parameters):

        # STEP 1: Set up the basic mesh
        self.settings: _MeshParameters = parameters.mesh
        # Start with uniform spatial spacing to initialize EOS/density
        initial_spatial: npt.NDArray = self.get_constant_spacing()
        if self.settings.eos_method == 1:
            self.eos = AdamsWilliamsonEOS(
                self.settings, initial_spatial
            )
        elif self.settings.eos_method == 2:
            self.eos = UserDefinedEOS(
                self.settings, initial_spatial
            )
        else:
            msg: str = (f"Unknown method to initialize Equation of State")
            raise ValueError(msg)

        if parameters.mesh.mass_coordinates:
            # Compute planet density and mass coordinates from the initial spatial grid
            self._planet_density: float = self.get_planet_density(initial_spatial)
            initial_mass_coordinates: npt.NDArray = (
                self.get_basic_mass_coordinates_from_spatial_coordinates(initial_spatial))

            # Create UNIFORM mass coordinate grid (this is the key change)
            xi_min = initial_mass_coordinates[0, 0]
            xi_max = initial_mass_coordinates[-1, 0]
            basic_mass_coordinates = np.linspace(
                xi_min, xi_max, self.settings.number_of_nodes
            ).reshape(-1, 1)

            # Derive NON-UNIFORM spatial coordinates from the uniform xi grid
            # by solving the mass-coordinate equation at each node via Newton
            # iteration, matching SPIDER's GetRadiusFromMassCoordinate.
            #
            # SPIDER's equation (eos_adamswilliamson.c:296-311):
            #   f(r) = (r_core^3 + 3*M_AW(r_core, r)/rho_avg)^(1/3) - xi = 0
            #
            # The PCHIP interpolation previously used here introduced
            # O(h^4) errors on the N-point grid that accumulated to ~3%
            # node position offsets.
            from scipy.optimize import brentq
            r_core = float(initial_spatial[0, 0])
            r_surf = float(initial_spatial[-1, 0])
            rho_avg = self._planet_density

            def _xi_of_r(r: float) -> float:
                """Mass coordinate xi(r) matching SPIDER's definition."""
                M_shell = float(self.eos.get_mass_within_radii(
                    np.array([r])
                ) - self.eos.get_mass_within_radii(
                    np.array([r_core])
                ))
                return (r_core**3 + 3.0 * M_shell / rho_avg) ** (1.0/3.0)

            basic_coordinates = np.empty_like(basic_mass_coordinates)
            basic_coordinates[0, 0] = r_core   # CMB: exact
            basic_coordinates[-1, 0] = r_surf  # surface: exact
            for j in range(1, self.settings.number_of_nodes - 1):
                xi_target = float(basic_mass_coordinates[j, 0])
                # Bracket: use neighbors from the initial grid as bounds
                r_lo = float(basic_coordinates[j-1, 0])
                r_hi = r_surf
                basic_coordinates[j, 0] = brentq(
                    lambda r: _xi_of_r(r) - xi_target,
                    r_lo, r_hi, xtol=1.0, rtol=1e-12,
                )
            logger.debug("Basic mass coordinates (uniform) = %s", basic_mass_coordinates)
            logger.debug("Basic spatial coordinates (non-uniform) = %s", basic_coordinates)
        else:
            basic_coordinates = initial_spatial
            basic_mass_coordinates = initial_spatial

        self.basic: FixedMesh = FixedMesh(
            self.settings,
            basic_coordinates,
            basic_mass_coordinates,
            np.max(basic_coordinates),
            np.min(basic_coordinates)
        )

        # STEP 2: Set up the staggered mesh
        staggered_mass_coordinates: npt.NDArray = (
            self.basic.mass_radii[:-1] + 0.5 * self.basic.delta_mesh)
        if parameters.mesh.mass_coordinates:
            staggered_coordinates: npt.NDArray = (
                self.get_staggered_spatial_coordinates_from_mass_coordinates(staggered_mass_coordinates))
        else:
            staggered_coordinates = staggered_mass_coordinates
        self.staggered: FixedMesh = FixedMesh(
            self.settings,
            staggered_coordinates,
            staggered_mass_coordinates,
            self.basic.outer_boundary,
            self.basic.inner_boundary,
        )
        self.eos.set_staggered_pressure(self.staggered.radii)

        # STEP 3: Set up the transform matrices
        if parameters.mesh.mass_coordinates:
            self._dxidr: npt.NDArray = self.get_dxidr_basic()
        else:
            self._dxidr: npt.NDArray = np.ones_like(self.basic.radii)
        self._d_dr_transform: npt.NDArray = self._get_d_dr_transform_matrix()
        self._quantity_transform: npt.NDArray = self._get_quantity_transform_matrix()

    @property
    def dxidr(self) -> npt.NDArray:
        """dxi/dr at basic nodes"""
        return self._dxidr

    @cached_property
    def staggered_effective_density(self) -> npt.NDArray:
        return self.eos.staggered_effective_density

    @cached_property
    def basic_pressure(self) -> npt.NDArray:
        return self.eos.basic_pressure

    @cached_property
    def staggered_pressure(self) -> npt.NDArray:
        return self.eos.staggered_pressure

    def get_planet_density(self, basic_coordinates: npt.NDArray) -> float:
        """Computes the mantle average density for mass-coordinate mapping.

        Matches SPIDER's ``EOSAdamsWilliamson_GetMassCoordinateAverageRho``
        (eos_adamswilliamson.c:249-267): the average density is computed
        over the mantle shell only (r_core to r_surface), NOT including
        the core. This ensures the mass-coordinate xi grid maps to the
        same physical radii as SPIDER's Newton-solved mesh.

        The previous implementation used whole-planet density (core +
        mantle) / r_surface^3, which produced ~3% node position offsets
        and cascading 12% pressure / 20% density / 17% flux differences.

        Args:
            Basic spatial coordinates

        Returns:
            Mantle average density (for mass-coordinate normalization)
        """
        r_core = basic_coordinates[0, 0]
        r_surf = basic_coordinates[-1, 0]
        basic_volumes = (np.power(basic_coordinates[1:, 0], 3.0)
            - np.power(basic_coordinates[:-1, 0], 3.0))
        mantle_mass = np.sum(
            self.staggered_effective_density[:, 0] * basic_volumes
        )
        # SPIDER convention: rho_avg = M_mantle * 3 / (r_surf^3 - r_core^3)
        # which is M_mantle / V_mantle_shell (without 4pi, but the 4pi
        # cancels in the mass-coordinate definition)
        mantle_volume = np.power(r_surf, 3.0) - np.power(r_core, 3.0)
        mantle_avg_density = mantle_mass * 3.0 / mantle_volume
        return mantle_avg_density.item()

    def get_basic_mass_coordinates_from_spatial_coordinates(self, basic_coordinates: npt.NDArray) -> npt.NDArray:
        """Computes mass coordinates matching SPIDER's definition.

        SPIDER's mass coordinate (eos_adamswilliamson.c:296-311):

            xi(r)^3 = r_core^3 + 3 * M_AW(r_core, r) / rho_avg_mantle

        where M_AW is the A-W mass integral from r_core to r (without
        4pi), and rho_avg_mantle is the mantle-only average density.
        At the CMB: xi = r_core. At the surface: xi = r_surface
        (by construction of rho_avg_mantle).

        Args:
            Basic spatial coordinates

        Returns:
            Basic mass coordinates
        """
        r_core = basic_coordinates[0, 0]

        # xi^3 at CMB = r_core^3
        basic_mass_coordinates = np.zeros_like(basic_coordinates)
        basic_mass_coordinates[:, :] = np.power(r_core, 3.0)

        # Cumulative mantle mass contribution: 3 * M_AW(r_core, r) / rho_avg
        basic_volumes = (np.power(basic_coordinates[1:, 0], 3.0)
            - np.power(basic_coordinates[:-1, 0], 3.0))
        for i in range(1, self.settings.number_of_nodes):
            basic_mass_coordinates[i:, :] += (
                self.staggered_effective_density[i-1, :] * basic_volumes[i-1]
                * 3.0 / self._planet_density
            )

        return np.power(basic_mass_coordinates, 1.0/3.0)

    def get_staggered_spatial_coordinates_from_mass_coordinates(self, staggered_mass_coordinates: npt.NDArray) -> npt.NDArray:
        """Computes the staggered spatial coordinates from staggered mass coordinates.

        Args:
            Staggered mass coordinates

        Returns:
            Staggered spatial coordinates
        """

        # Initialise the staggered spatial coordinate to the inner boundary
        staggered_coordinates = (np.ones_like(staggered_mass_coordinates)
            * np.power(self.settings.inner_radius,3.0))

        # Add first half cell contribution
        staggered_coordinates += (
            self._planet_density
             * (np.power(staggered_mass_coordinates[0,:],3.0)
            - np.power(self.basic.mass_radii[0,:], 3.0))
            / self.staggered_effective_density[0,:]
        )

        # Get spatial coordinates by adding individual cell contributions to the mantle mass
        shell_effective_density = 0.5*(
            self.staggered_effective_density[1:,:] + self.staggered_effective_density[:-1,:])
        shell_mass_volumes = (np.power(staggered_mass_coordinates[1:,:],3.0)
            - np.power(staggered_mass_coordinates[:-1,:],3.0))
        for i in range(1,self.settings.number_of_nodes-1):
            staggered_coordinates[i:,:] += (
                self._planet_density
                * shell_mass_volumes[i-1,:]
                / shell_effective_density[i-1,:]
            )

        return np.power(staggered_coordinates, 1.0/3.0)

    def get_dxidr_basic(self) -> npt.NDArray:
        """Computes dxidr at basic nodes."""

        dxidr = (
            self.eos.basic_density
            / self._planet_density
            * np.power(self.basic.radii,2.0)
            / np.power(self.basic.mass_radii,2.0)
        )

        return dxidr

    def get_constant_spacing(self) -> npt.NDArray:
        """Constant radius spacing across the mantle

        Returns:
            Radii with constant spacing as a column vector
        """
        radii: npt.NDArray = np.linspace(
            self.settings.inner_radius, self.settings.outer_radius, self.settings.number_of_nodes
        )
        radii = np.atleast_2d(radii).T

        return radii

    def _get_d_dr_transform_matrix(self) -> npt.NDArray:
        """Transform matrix for determining d/dr of a staggered quantity on the basic mesh.

        Returns:
            The transform matrix
        """
        transform: npt.NDArray = np.zeros(
            (self.basic.number_of_nodes, self.staggered.number_of_nodes)
        )
        transform[1:-1, :-1] += np.diagflat(-1 / self.staggered.delta_mesh)  # k=0 diagonal
        transform[1:-1:, 1:] += np.diagflat(1 / self.staggered.delta_mesh)  # k=1 diagonal

        # Gradient at boundaries can be extrapolated from the first two closests basic nodes
        # This only affects the estimation of indivual components of heat fluxes when working
        # with flux boundary conditions. Gradient at boundaries are overwritten when using
        # temperature boundary conditions.

        # Extrapolation of gradient at inner radius
        inner_delta_ratio = self.basic.delta_mesh[1].item() / self.basic.delta_mesh[0].item()
        transform[0, 0] = - (inner_delta_ratio + 1) / self.staggered.delta_mesh[0].item()
        transform[0, 1] = (inner_delta_ratio + 1) / self.staggered.delta_mesh[0].item()
        transform[0, 1] += inner_delta_ratio / self.staggered.delta_mesh[1].item()
        transform[0, 2] = - inner_delta_ratio / self.staggered.delta_mesh[1].item()
        # Extrapolation of gradient at outer radius
        outer_delta_ratio: float = self.basic.delta_mesh[-2].item() / self.basic.delta_mesh[-1].item()
        transform[-1, -1] = - (outer_delta_ratio + 1) / self.staggered.delta_mesh[-1].item()
        transform[-1, -2] = (outer_delta_ratio + 1) / self.staggered.delta_mesh[-1].item()
        transform[-1, -2] += outer_delta_ratio / self.staggered.delta_mesh[-2].item()
        transform[-1, -3] = - outer_delta_ratio / self.staggered.delta_mesh[-2].item()

        # Scale the transform matrix by dxi/dr at basic nodes
        for i in range(self.settings.number_of_nodes-1):
            transform[:,i] *= self._dxidr[:, 0]
        logger.debug("_d_dr_transform_matrix = %s", transform)

        return transform

    def d_dr_at_basic_nodes(self, staggered_quantity: npt.NDArray) -> npt.NDArray:
        """Determines d/dr at the basic nodes of a quantity defined at the staggered nodes.

        Args:
            staggered_quantity: A quantity defined at the staggered nodes.

        Returns:
            d/dr at the basic nodes
        """
        d_dr_at_basic_nodes: npt.NDArray = self._d_dr_transform.dot(staggered_quantity)
        logger.debug("d_dr_at_basic_nodes = %s", d_dr_at_basic_nodes)

        return d_dr_at_basic_nodes

    def _get_quantity_transform_matrix(self) -> npt.NDArray:
        """A transform matrix for mapping quantities on the staggered mesh to the basic mesh.

        Uses backward and forward differences at the inner and outer radius, respectively, to
        obtain the quantity values of the basic nodes at the innermost and outermost nodes.
        When using temperature boundary conditions, values at outer boundaries will be overwritten.
        When using flux boundary conditions, values at outer boundaries will be used to provide
        estimate of individual components of heat fluxes though the total heat flux is imposed.

        Returns:
            The transform matrix
        """
        transform: npt.NDArray = np.zeros(
            (self.basic.number_of_nodes, self.staggered.number_of_nodes)
        )
        mesh_ratio: npt.NDArray = self.basic.delta_mesh[:-1] / self.staggered.delta_mesh
        transform[1:-1, :-1] += np.diagflat(1 - 0.5 * mesh_ratio)  # k=0 diagonal
        transform[1:-1:, 1:] += np.diagflat(0.5 * mesh_ratio)  # k=1 diagonal
        # Backward difference at inner radius
        transform[0, :2] = np.array([1 + 0.5 * mesh_ratio[0], -0.5 * mesh_ratio[0]]).flatten()
        # Forward difference at outer radius
        mesh_ratio_outer: npt.NDArray = self.basic.delta_mesh[-1] / self.staggered.delta_mesh[-1]
        transform[-1, -2:] = np.array(
            [-0.5 * mesh_ratio_outer, 1 + 0.5 * mesh_ratio_outer]
        ).flatten()
        logger.debug("_quantity_transform_matrix = %s", transform)

        return transform

    def quantity_at_basic_nodes(self, staggered_quantity: npt.NDArray) -> npt.NDArray:
        """Determines a quantity at the basic nodes that is defined at the staggered nodes.

        Uses backward and forward differences at the inner and outer radius, respectively, to
        obtain the quantity values of the basic nodes at the innermost and outermost nodes.
        When using temperature boundary conditions, values at outer boundaries will be overwritten.
        When using flux boundary conditions, values at outer boundaries will be used to provide
        estimate of individual components of heat fluxes though the total heat flux is imposed.

        Args:
            staggered_quantity: A quantity defined at the staggered nodes

        Returns:
            The quantity at the basic nodes
        """
        quantity_at_basic_nodes: npt.NDArray = self._quantity_transform.dot(staggered_quantity)
        logger.debug("quantity_at_basic_nodes = %s", quantity_at_basic_nodes)

        return quantity_at_basic_nodes

    def quantity_at_staggered_nodes(self, basic_quantity: npt.NDArray) -> npt.NDArray:
        """Determines a quantity at the staggered nodes that is defined at the basic nodes.

        Staggered nodes are always located at cell centers, whatever the mesh.

        Args:
            basic_quantity: A quantity defined at the basic nodes

        Returns:
            The quantity at the staggered nodes
        """
        quantity_at_staggered_nodes: npt.NDArray = 0.5 * (
            basic_quantity[:-1, ...] + basic_quantity[1:, ...])
        logger.debug("quantity_at_staggered_nodes = %s", quantity_at_staggered_nodes)

        return quantity_at_staggered_nodes

    def volume_average(self, staggered_quantity: npt.NDArray) -> float:
        return np.dot(staggered_quantity.T, self.basic.volume).item() / self.basic.total_volume
