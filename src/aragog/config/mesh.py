"""Mesh configuration."""

from __future__ import annotations

import logging
import numbers

import attrs

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


# Smallest end cell [m], a margin of 2 over the 1 m inset of the mass-mode node bracket; at
# 2 m the CMB half cell meets the 1 m floor of the bower2018 core flux.
MIN_CELL_THICKNESS = 2.0

# surface_cell_thickness value that sets the outer cell to the top cell of the uniform mesh
# of the same coordinate mode.
UNIFORM_CELL = 'uniform'


def check_cell_thickness(mesh) -> None:
    """Raise ``ValueError`` unless ``surface_cell_thickness`` and ``cmb_cell_thickness`` of
    ``mesh`` are each 0 or a real number in [``MIN_CELL_THICKNESS``, uniform cell) m, or
    ``surface_cell_thickness`` is ``UNIFORM_CELL``.

    The uniform cell is ``(outer_radius - inner_radius) / (number_of_nodes - 1)``; a refined
    mesh needs at least 4 nodes, as the end rows of the d/dr transform do. A bool (a TOML
    ``true``) is refused, not read as 1 m.
    """
    need = 4
    for name in ('surface_cell_thickness', 'cmb_cell_thickness'):
        value = getattr(mesh, name)
        real = isinstance(value, numbers.Real) and not isinstance(value, bool)
        if real and value == 0:
            continue
        if (
            name == 'surface_cell_thickness'
            and isinstance(value, str)
            and value == UNIFORM_CELL
        ):
            if mesh.cmb_cell_thickness == 0:  # the uniform mesh
                continue
            if mesh.number_of_nodes < need:
                raise ValueError(
                    f'mesh.cmb_cell_thickness with mesh.{name} = {UNIFORM_CELL!r} needs '
                    f'number_of_nodes >= {need}, got {mesh.number_of_nodes}'
                )
            continue
        if mesh.number_of_nodes < need:
            raise ValueError(
                f'mesh.{name} needs number_of_nodes >= {need}, got {mesh.number_of_nodes}'
            )
        uniform = (mesh.outer_radius - mesh.inner_radius) / (mesh.number_of_nodes - 1)
        if not (real and MIN_CELL_THICKNESS <= value < uniform):
            rule = f' or {UNIFORM_CELL!r}' if name == 'surface_cell_thickness' else ''
            raise ValueError(
                f'mesh.{name} must be 0 (off){rule} or in [{MIN_CELL_THICKNESS:g}, '
                f'{uniform:.6g}) m, the uniform radial cell; got {value!r}'
            )


@attrs.define
class MeshConfig:
    """Mesh and static pressure profile parameters.

    Parameters
    ----------
    outer_radius : float
        Outer radius of the mantle shell (planet radius minus atmosphere
        depth, in practice the same as ``planet.radius``) [m].
    inner_radius : float
        Inner radius of the mantle shell, i.e. the core-mantle boundary
        (CMB) radius [m].
    number_of_nodes : int
        Number of basic mesh nodes.
    mixing_length_profile : str
        'constant' or 'nearest_boundary'. With ``outer_boundary_condition = 5``, 'constant'
        is refused unless ``mixing_length_constant_fraction = 0``.
    mixing_length_constant_fraction : float
        Fraction of the mantle thickness to use as the mixing length
        when ``mixing_length_profile = 'constant'``. Default 0.25.
        Ignored for 'nearest_boundary'.
    core_density : float
        Core density [kg/m^3].
    eos_method : int
        1: Adams-Williamson, 2: User-defined.
    surface_density : float
        Surface-mantle density anchor [kg/m^3]. Used only by the analytic
        Adams-Williamson EOS (``eos_method = 1``) as the constant
        ``rho_top`` in ``rho(P) = rho_top * exp(P/K_S)``. NOT the
        runtime material density (which is pressure-entropy dependent and
        comes from the P-S property tables). Default 4078.95095544
        matches the SPIDER ``-adams_williamson_rhos`` value used in
        PROTEUS production runs. Ignored when ``eos_method = 2``.
    gravitational_acceleration : float
        Gravitational acceleration [m/s^2].
    adiabatic_bulk_modulus : float
        Adiabatic bulk modulus [Pa].
    surface_pressure : float
        Surface pressure [Pa].
    mass_coordinates : bool
        Use uniform spacing in mass-coordinate space instead of radius.
        Gives larger cells at the surface (lower density) and matches
        SPIDER's mesh. Default True matches the PROTEUS production path.
    eos_file : str
        Path to user-defined EOS file.
    surface_cell_thickness, cmb_cell_thickness : float or str
        Radial thickness [m] of the outermost and innermost basic cell. 0 leaves that end
        unset; with only the other end set, its cells grow. A set value lies in [2 m, uniform
        radial cell) and refines towards that boundary with a tanh stretching (both ends: the
        two-sided Vinokur form). In mass-coordinate mode the stretching acts on the mass
        coordinate and the end cells still have the given radial thickness; there a value at
        or above the uniform end cell of that mode is refused when the mesh is built.
        ``surface_cell_thickness = 'uniform'`` (default) sets the outer cell to the top cell
        of the uniform mesh of the same mode: (outer_radius - inner_radius) /
        (number_of_nodes - 1) in radius mode, the top cell of the uniform mass-coordinate
        mesh in mass-coordinate mode. Default CMB cell: 1000 m. With
        ``cmb_cell_thickness = 0`` and ``'uniform'`` the mesh is the uniform mesh.
    """

    outer_radius: float
    inner_radius: float
    number_of_nodes: int
    mixing_length_profile: str
    core_density: float
    eos_method: int = 1
    surface_density: float = 4078.95095544
    gravitational_acceleration: float = 9.81
    adiabatic_bulk_modulus: float = 260e9
    adams_williamson_beta: float = 0.0  # 0 = derive from K_S
    surface_pressure: float = 0.0
    mass_coordinates: bool = True
    eos_file: str = ''
    mixing_length_constant_fraction: float = 0.25
    surface_cell_thickness: float | str = UNIFORM_CELL
    cmb_cell_thickness: float = 1000.0

    def __attrs_post_init__(self):
        check_cell_thickness(self)
