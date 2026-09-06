"""Mesh subpackage: staggered mesh, fixed mesh, and pressure EOS classes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import field
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from aragog.mesh.fixed_mesh import FixedMesh
from aragog.mesh.pressure_eos import (
    EOS,
    AdamsWilliamsonEOS,
    UserDefinedEOS,
)
from aragog.parser import Parameters, _MeshParameters

__all__ = [
    'Mesh',
    'FixedMesh',
    'EOS',
    'AdamsWilliamsonEOS',
    'UserDefinedEOS',
    'derive_core_density_from_mesh',
]

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


def derive_core_density_from_mesh(mesh_file: str, M_core: float) -> float:
    """Derive the self-consistent average core density from a mantle mesh file.

    Reads the CMB radius from a 5-column ascending-r mantle mesh file
    (``r``, ``P``, ``rho``, ``g``, ``T``) and returns
    :math:`\\rho_\\mathrm{core} = M_\\mathrm{core} / (\\tfrac{4}{3} \\pi R_\\mathrm{cmb}^3)`.

    The first row of the file is the bottom of the mantle (CMB), matching
    the format Zalmoxis writes to ``zalmoxis_output.dat``. This is the
    Aragog-side analogue of the SPIDER wrapper's ``-rho_core`` echo-back:
    the on-disk mesh's :math:`R_\\mathrm{cmb}` and the latest
    ``hf_row['M_core']`` give a self-consistent average density that
    survives mesh-blending fall-backs and stale-cache cases where
    ``hf_row['core_density']`` has drifted from the mesh state.

    Parameters
    ----------
    mesh_file : str
        Path to the Zalmoxis-format mantle mesh file. The first row is
        treated as the CMB; only the first column (``r``) is read.
    M_core : float
        Core mass [kg]. Must be positive.

    Returns
    -------
    float
        Self-consistent average core density [kg m^-3].

    Raises
    ------
    FileNotFoundError
        If ``mesh_file`` does not exist.
    ValueError
        If ``M_core <= 0``, the mesh file is empty, or the parsed
        :math:`R_\\mathrm{cmb}` is non-positive.
    """
    if M_core <= 0.0:
        raise ValueError(f'M_core must be positive, got {M_core}')

    with open(mesh_file) as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f'Mesh file {mesh_file} is empty')
        try:
            R_cmb = float(first_line.split()[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f'Could not parse R_cmb from first row of {mesh_file}: {first_line!r}'
            ) from exc

    if R_cmb <= 0.0:
        raise ValueError(f'Parsed R_cmb must be positive, got {R_cmb}')

    return float(M_core / (4.0 / 3.0 * np.pi * R_cmb**3))


def _radius_for_mass_coordinate(
    xi_of_r: Callable[[float], float],
    xi_target: float,
    r_lo: float,
    r_hi: float,
    node: int | None = None,
) -> float:
    """Solve ``xi_of_r(r) == xi_target`` for ``r`` in ``[r_lo, r_hi]``.

    ``xi_of_r`` is the mass coordinate, monotonically increasing in radius.
    When the bracket straddles the root the result is the ``brentq`` root, so
    the forward initial-condition build (which always brackets) is unchanged.

    When the bracket does not straddle, ``xi_target`` lies outside
    ``[xi_of_r(r_lo), xi_of_r(r_hi)]`` and the node is clamped to the nearest
    in-domain endpoint. This path is reached only on a resumed run, where a
    loaded entropy field shifts the EOS mass distribution so a near-boundary
    node on the uniform mass grid falls just outside the achievable range; a
    bare ``brentq`` raises ``ValueError`` ("f(a) and f(b) must have different
    signs") there and aborts the resume.

    Parameters
    ----------
    xi_of_r : Callable[[float], float]
        Mass coordinate as a function of radius. Must be monotonically
        increasing on ``[r_lo, r_hi]``.
    xi_target : float
        Target mass coordinate for this node.
    r_lo, r_hi : float
        Bracket endpoints [m], ``r_lo < r_hi``.
    node : int, optional
        Node index, used only in the clamp warning message.

    Returns
    -------
    float
        Radius [m] of the node, in ``[r_lo, r_hi]``.
    """

    def _f(r: float) -> float:
        return xi_of_r(r) - xi_target

    f_lo = _f(r_lo)
    f_hi = _f(r_hi)

    if f_lo == 0.0:
        return r_lo
    if f_hi == 0.0:
        return r_hi
    if f_lo * f_hi < 0.0:
        return float(brentq(_f, r_lo, r_hi, xtol=1.0, rtol=1e-12))

    clamped = r_lo if f_lo > 0.0 else r_hi
    logger.warning(
        'Mesh mass-coordinate bracket failed for node %s '
        '(xi_target=%.6e outside [%.6e, %.6e]); clamping radius to %.3f m. '
        'Expected only on resume with a loaded entropy field.',
        'unknown' if node is None else node,
        xi_target,
        xi_of_r(r_lo),
        xi_of_r(r_hi),
        clamped,
    )
    return clamped


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
            # Analytic Adams-Williamson: rho(P) = rho_top * exp(P/K_S)
            # under constant gravity; the density anchor is configured
            # via ``surface_density``. Standalone-only; PROTEUS production
            # runs go through ``eos_method = 2``.
            self.eos = AdamsWilliamsonEOS(self.settings, initial_spatial)
        elif self.settings.eos_method == 2:
            # User-defined EOS = lookup table loaded from
            # ``mesh.eos_file`` (a four-column ``r, P, rho, g`` profile).
            # In the PROTEUS-coupled path this file is written by the
            # structure solver: Zalmoxis (PALEOS-driven) writes
            # ``zalmoxis_output.dat``; the dummy-static and SPIDER paths
            # write ``spider_mesh.dat``. The four columns also drive a
            # per-node gravity profile (overrides the scalar
            # ``gravitational_acceleration``).
            self.eos = UserDefinedEOS(self.settings, initial_spatial)
        else:
            msg: str = 'Unknown method to initialize Equation of State'
            raise ValueError(msg)

        if parameters.mesh.mass_coordinates:
            # Compute planet density and mass coordinates from the initial spatial grid
            self._planet_density: float = self.get_planet_density(initial_spatial)
            initial_mass_coordinates: npt.NDArray = (
                self.get_basic_mass_coordinates_from_spatial_coordinates(initial_spatial)
            )

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
            # Newton-via-brentq avoids the O(h^4) interpolation error
            # of a PCHIP fit; on an N-point grid that error accumulated
            # to ~3% node position offsets, large enough to perturb the
            # Adams-Williamson reference state.
            r_core = float(initial_spatial[0, 0])
            r_surf = float(initial_spatial[-1, 0])
            rho_avg = self._planet_density

            def _xi_of_r(r: float) -> float:
                """Mass coordinate xi(r) matching SPIDER's definition.

                SPIDER's mass integral is WITHOUT 4pi (convention in
                SPIDER, see eos_adamswilliamson.c:185). Aragog's
                get_mass_within_radii includes 4pi. Divide by 4pi to
                match SPIDER, since rho_avg was also computed without
                4pi (from staggered_effective_density * delta_r^3).
                """
                M_shell_4pi = self.eos.get_mass_within_radii(np.array([r])).item()
                M_shell = M_shell_4pi / (4.0 * np.pi)
                return (r_core**3 + 3.0 * M_shell / rho_avg) ** (1.0 / 3.0)

            basic_coordinates = np.empty_like(basic_mass_coordinates)
            basic_coordinates[0, 0] = r_core  # CMB: exact
            basic_coordinates[-1, 0] = r_surf  # surface: exact
            # Bracket inset 1 m inside the boundaries to avoid floating-point
            # edge cases right at r_core / r_surf.
            r_lo = r_core + 1.0
            r_hi = r_surf - 1.0
            for j in range(1, self.settings.number_of_nodes - 1):
                xi_target = float(basic_mass_coordinates[j, 0])
                basic_coordinates[j, 0] = _radius_for_mass_coordinate(
                    _xi_of_r, xi_target, r_lo, r_hi, node=j
                )
            # The mesh must be strictly increasing in radius: zero-width cells
            # divide by zero in the finite-volume gradient operators. If two or
            # more near-boundary nodes clamped to the same endpoint (only
            # reachable on a resume with a badly-shifted entropy field), fail
            # loudly here rather than propagate NaNs into the entropy RHS.
            if np.any(np.diff(basic_coordinates[:, 0]) <= 0.0):
                raise ValueError(
                    'Non-monotonic basic mesh after mass-coordinate solve '
                    '(duplicate node radii). The loaded entropy field shifted '
                    'the EOS mass distribution too far for the fixed bracket; '
                    're-run the structure module to regenerate the mesh.'
                )
            logger.debug('Basic mass coordinates (uniform) = %s', basic_mass_coordinates)
            logger.debug('Basic spatial coordinates (non-uniform) = %s', basic_coordinates)
        else:
            basic_coordinates = initial_spatial
            basic_mass_coordinates = initial_spatial

        self.basic: FixedMesh = FixedMesh(
            self.settings,
            basic_coordinates,
            basic_mass_coordinates,
            np.max(basic_coordinates),
            np.min(basic_coordinates),
        )
        # Refresh basic pressure and density from the final basic radii, which
        # differ from initial_spatial once the mass-coordinate solve has run.
        # The density refresh keeps get_dxidr_basic consistent with the mesh.
        self.eos.set_basic_pressure(self.basic.radii)
        self.eos.set_basic_density(self.basic.radii)

        # STEP 2: Set up the staggered mesh
        staggered_mass_coordinates: npt.NDArray = (
            self.basic.mass_radii[:-1] + 0.5 * self.basic.delta_mesh
        )
        if parameters.mesh.mass_coordinates:
            staggered_coordinates: npt.NDArray = (
                self.get_staggered_spatial_coordinates_from_mass_coordinates(
                    staggered_mass_coordinates
                )
            )
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
        # Use the analytic A-W mass integral (without 4pi, SPIDER
        # convention) for the average density. The discrete sum
        # (rho * delta_r^3) is inconsistent because the 4pi/3
        # factors don't cancel between the volume elements and
        # the effective density definition.
        M_4pi = (
            self.eos.get_mass_within_radii(np.array([[r_surf]]))
            - self.eos.get_mass_within_radii(np.array([[r_core]]))
        ).item()
        M_no4pi = M_4pi / (4.0 * np.pi)
        mantle_volume_no4pi = (np.power(r_surf, 3.0) - np.power(r_core, 3.0)) / 3.0
        mantle_avg_density = M_no4pi / mantle_volume_no4pi
        return float(mantle_avg_density)

    def get_basic_mass_coordinates_from_spatial_coordinates(
        self, basic_coordinates: npt.NDArray
    ) -> npt.NDArray:
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

        # Cumulative mantle mass contribution.
        # staggered_effective_density * (r^3_outer - r^3_inner) gives
        # mass_shell / (4/3*pi). Dividing by rho_avg (which is
        # M_no4pi / V_no4pi = M_no4pi / ((r_s^3-r_c^3)/3)) gives
        # the correct xi^3 increment: 3 * M_shell_no4pi / rho_avg.
        basic_volumes = np.power(basic_coordinates[1:, 0], 3.0) - np.power(
            basic_coordinates[:-1, 0], 3.0
        )
        for i in range(1, self.settings.number_of_nodes):
            basic_mass_coordinates[i:, :] += (
                self.staggered_effective_density[i - 1, :]
                * basic_volumes[i - 1]
                / self._planet_density
            )

        return np.power(basic_mass_coordinates, 1.0 / 3.0)

    def get_staggered_spatial_coordinates_from_mass_coordinates(
        self, staggered_mass_coordinates: npt.NDArray
    ) -> npt.NDArray:
        """Computes the staggered spatial coordinates from staggered mass coordinates.

        Args:
            Staggered mass coordinates

        Returns:
            Staggered spatial coordinates
        """

        # Initialise the staggered spatial coordinate to the inner boundary
        staggered_coordinates = np.ones_like(staggered_mass_coordinates) * np.power(
            self.settings.inner_radius, 3.0
        )

        # Add first half cell contribution
        staggered_coordinates += (
            self._planet_density
            * (
                np.power(staggered_mass_coordinates[0, :], 3.0)
                - np.power(self.basic.mass_radii[0, :], 3.0)
            )
            / self.staggered_effective_density[0, :]
        )

        # Get spatial coordinates by adding individual cell contributions to the mantle mass
        shell_effective_density = 0.5 * (
            self.staggered_effective_density[1:, :] + self.staggered_effective_density[:-1, :]
        )
        shell_mass_volumes = np.power(staggered_mass_coordinates[1:, :], 3.0) - np.power(
            staggered_mass_coordinates[:-1, :], 3.0
        )
        for i in range(1, self.settings.number_of_nodes - 1):
            staggered_coordinates[i:, :] += (
                self._planet_density
                * shell_mass_volumes[i - 1, :]
                / shell_effective_density[i - 1, :]
            )

        return np.power(staggered_coordinates, 1.0 / 3.0)

    def get_dxidr_basic(self) -> npt.NDArray:
        """Computes dxidr at basic nodes."""

        dxidr = (
            self.eos.basic_density
            / self._planet_density
            * np.power(self.basic.radii, 2.0)
            / np.power(self.basic.mass_radii, 2.0)
        )

        return dxidr

    def get_constant_spacing(self) -> npt.NDArray:
        """Constant radius spacing across the mantle

        Returns:
            Radii with constant spacing as a column vector
        """
        radii: npt.NDArray = np.linspace(
            self.settings.inner_radius,
            self.settings.outer_radius,
            self.settings.number_of_nodes,
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
        transform[0, 0] = -(inner_delta_ratio + 1) / self.staggered.delta_mesh[0].item()
        transform[0, 1] = (inner_delta_ratio + 1) / self.staggered.delta_mesh[0].item()
        transform[0, 1] += inner_delta_ratio / self.staggered.delta_mesh[1].item()
        transform[0, 2] = -inner_delta_ratio / self.staggered.delta_mesh[1].item()
        # Extrapolation of gradient at outer radius
        outer_delta_ratio: float = (
            self.basic.delta_mesh[-2].item() / self.basic.delta_mesh[-1].item()
        )
        transform[-1, -1] = -(outer_delta_ratio + 1) / self.staggered.delta_mesh[-1].item()
        transform[-1, -2] = (outer_delta_ratio + 1) / self.staggered.delta_mesh[-1].item()
        transform[-1, -2] += outer_delta_ratio / self.staggered.delta_mesh[-2].item()
        transform[-1, -3] = -outer_delta_ratio / self.staggered.delta_mesh[-2].item()

        # Scale the transform matrix by dxi/dr at basic nodes
        for i in range(self.settings.number_of_nodes - 1):
            transform[:, i] *= self._dxidr[:, 0]
        logger.debug('_d_dr_transform_matrix = %s', transform)

        return transform

    def d_dr_at_basic_nodes(self, staggered_quantity: npt.NDArray) -> npt.NDArray:
        """Determines d/dr at the basic nodes of a quantity defined at the staggered nodes.

        Args:
            staggered_quantity: A quantity defined at the staggered nodes.

        Returns:
            d/dr at the basic nodes
        """
        d_dr_at_basic_nodes: npt.NDArray = self._d_dr_transform.dot(staggered_quantity)
        logger.debug('d_dr_at_basic_nodes = %s', d_dr_at_basic_nodes)

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
        mesh_ratio_outer: npt.NDArray = (
            self.basic.delta_mesh[-1] / self.staggered.delta_mesh[-1]
        )
        transform[-1, -2:] = np.array(
            [-0.5 * mesh_ratio_outer, 1 + 0.5 * mesh_ratio_outer]
        ).flatten()
        logger.debug('_quantity_transform_matrix = %s', transform)

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
        logger.debug('quantity_at_basic_nodes = %s', quantity_at_basic_nodes)

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
            basic_quantity[:-1, ...] + basic_quantity[1:, ...]
        )
        logger.debug('quantity_at_staggered_nodes = %s', quantity_at_staggered_nodes)

        return quantity_at_staggered_nodes

    def volume_average(self, staggered_quantity: npt.NDArray) -> float:
        return np.dot(staggered_quantity.T, self.basic.volume).item() / self.basic.total_volume
