"""Parses the configuration file and stores the parameters.

Non-dimensionalization has been removed. All quantities are in SI units,
with time in years. The _ScalingsParameters class is retained with all
scales set to 1.0 so that downstream code referencing scalings_ attributes
sees identity operations (division by 1.0).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import Field, dataclass, field, fields
from typing import Any

import numpy as np
import numpy.typing as npt
from typed_configparser import ConfigParser

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


def _get_dataclass_from_section_name() -> dict[str, Any]:
    """Maps the section names in the configuration data to the dataclasses that stores the data."""
    mapping: dict[str, Any] = {
        "scalings": _ScalingsParameters,
        "solver": _SolverParameters,
        "boundary_conditions": _BoundaryConditionsParameters,
        "mesh": _MeshParameters,
        "energy": _EnergyParameters,
        "initial_condition": _InitialConditionParameters,
        "phase_liquid": _PhaseParameters,
        "phase_solid": _PhaseParameters,
        "phase_mixed": _PhaseMixedParameters,
        # radionuclides are dealt with separately
    }

    return mapping


@dataclass
class _ScalingsParameters:
    """Vestigial scalings class retained for API compatibility.

    All scales are set to 1.0 so that any code dividing by a scale
    factor becomes an identity operation. The config file may still
    contain a [scalings] section; the values are parsed but ignored.
    """

    radius: float = 1
    temperature: float = 1
    density: float = 1
    time: float = 1
    area: float = field(init=False)
    gravitational_acceleration: float = field(init=False)
    temperature_gradient: float = field(init=False)
    thermal_expansivity: float = field(init=False)
    pressure: float = field(init=False)
    velocity: float = field(init=False)
    kinetic_energy_per_volume: float = field(init=False)
    heat_capacity: float = field(init=False)
    entropy: float = field(init=False)
    latent_heat_per_mass: float = field(init=False)
    power_per_volume: float = field(init=False)
    power_per_mass: float = field(init=False)
    heat_flux: float = field(init=False)
    thermal_conductivity: float = field(init=False)
    viscosity: float = field(init=False)
    time_years: float = field(init=False)
    stefan_boltzmann_constant: float = field(init=False)

    def __post_init__(self) -> None:
        # Override whatever was parsed from the config: all scales = 1.0
        self.radius = 1.0
        self.temperature = 1.0
        self.density = 1.0
        self.time = 1.0
        self.area = 1.0
        self.gravitational_acceleration = 1.0
        self.temperature_gradient = 1.0
        self.thermal_expansivity = 1.0
        self.pressure = 1.0
        self.velocity = 1.0
        self.kinetic_energy_per_volume = 1.0
        self.heat_capacity = 1.0
        self.entropy = 1.0
        self.latent_heat_per_mass = 1.0
        self.power_per_volume = 1.0
        self.power_per_mass = 1.0
        self.heat_flux = 1.0
        self.thermal_conductivity = 1.0
        self.viscosity = 1.0
        self.time_years = 1.0
        self.stefan_boltzmann_constant = 1.0
        logger.debug("scalings = %s (all unity, non-dimensionalization removed)", self)


@dataclass
class _BoundaryConditionsParameters:
    """Stores parameters in the boundary_conditions section in the configuration data."""

    outer_boundary_condition: int
    outer_boundary_value: float
    inner_boundary_condition: int
    inner_boundary_value: float
    emissivity: float
    equilibrium_temperature: float
    core_heat_capacity: float
    tfac_core_avg: float = 1.147
    param_utbl: bool = False
    param_utbl_const: float = 1.0e-7
    # Core BC mode selector, threaded from config/boundary.py.
    # Valid values: 'quasi_steady' (default, v3 alpha-factor),
    #               'energy_balance' (Path A SPIDER bit-parity),
    #               'bower2018' (EXPERIMENTAL tombstone, do not use).
    # See aragog/config/boundary.py docstring for details.
    core_bc: str = 'quasi_steady'
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes.

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.equilibrium_temperature /= self.scalings_.temperature
        self.core_heat_capacity /= self.scalings_.heat_capacity
        if self.param_utbl:
            self.param_utbl_const *= self.scalings_.temperature**2
        else:
            self.param_utbl_const = 0.0
        self._scale_inner_boundary_condition()
        self._scale_outer_boundary_condition()

    def _scale_inner_boundary_condition(self) -> None:
        """Scales the inner boundary value.

        Equivalent to CORE_BC in C code.
            1: Simple core cooling
            2: Prescribed heat flux
            3: Prescribed temperature
        """
        if self.inner_boundary_condition == 1:
            self.inner_boundary_value = 0
        elif self.inner_boundary_condition == 2:
            self.inner_boundary_value /= self.scalings_.heat_flux
        elif self.inner_boundary_condition == 3:
            self.inner_boundary_value /= self.scalings_.temperature
        else:
            msg: str = f"inner_boundary_condition = {self.inner_boundary_condition} is unknown"
            raise ValueError(msg)

    def _scale_outer_boundary_condition(self) -> None:
        """Scales the outer boundary value.

        Equivalent to SURFACE_BC in C code.
            1: Grey-body atmosphere
            2: Zahnle steam atmosphere
            3: Couple to atmodeller
            4: Prescribed heat flux
            5: Prescribed temperature
        """
        if self.outer_boundary_condition == 1:
            pass
        elif self.outer_boundary_condition == 2:
            pass
        elif self.outer_boundary_condition == 3:
            pass
        elif self.outer_boundary_condition == 4:
            self.outer_boundary_value /= self.scalings_.heat_flux
        elif self.outer_boundary_condition == 5:
            self.outer_boundary_value /= self.scalings_.temperature
        else:
            msg: str = f"outer_boundary_condition = {self.outer_boundary_condition} is unknown"
            raise ValueError(msg)


@dataclass
class _EnergyParameters:
    """Stores parameters in the energy section"""

    conduction: bool
    convection: bool
    gravitational_separation: bool
    mixing: bool
    radionuclides: bool
    dilatation: bool
    tidal: bool
    eddy_diffusivity_chemical: float = 1.0
    kappah_floor: float = 0.0  # m^2/s, phase-dependent eddy diffusivity floor

    # SPIDER-analogue bottom-up gate for the gravitational-separation mass
    # flux. Only allows melt/solid separation across an interface when the
    # staggered cell immediately below contains non-pure liquid/solid. This
    # is enforced via a cubic Hermite smoothing of the un-truncated
    # two-phase fraction at the cell below; see
    # EntropyState.update() and SPIDER/energy.c:523. Turning this off
    # reproduces the pre-fix CMB drain at first crystallisation and is
    # only useful for regression tests.
    bottom_up_grav_sep: bool = True
    # Phase-boundary smoothing for Jgrav and Jmix: 'cubic_hermite' or 'tanh'.
    # 'cubic_hermite' provides intermediate-phi damping that prevents the CMB
    # drain when residual EOS differences exist. 'tanh' matches SPIDER exactly
    # (full-strength fluxes across the mushy zone) but requires all material
    # properties to match SPIDER to <0.01%.
    phase_smoothing: str = 'cubic_hermite'
    # ODE solver method: 'radau' (scipy Radau, default), 'cvode' (SUNDIALS
    # CVODE via scikits.odes, same solver SPIDER uses), 'bdf' (scipy BDF).
    solver_method: str = 'radau'

    tidal_array: npt.NDArray = field(default_factory=lambda:np.array([0.0], dtype=float))

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes.

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.tidal_array /= self.scalings_.power_per_mass


@dataclass
class _InitialConditionParameters:
    """Stores the settings in the initial_condition section in the configuration data."""

    initial_condition: int = 1
    surface_temperature: float = 4000
    basal_temperature: float = 4000
    init_file: str = ""
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes.

        Initial condition method
            1: Linear profile
            2: User-defined temperature field (from file)
            3: Adiabatic profile

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.surface_temperature /= self.scalings_.temperature
        self.basal_temperature /= self.scalings_.temperature

        if self.initial_condition == 2:
            if self.init_file == "":
                msg: str = (f"you must provide an initial temperature file")
                raise ValueError(msg)
            self.init_temperature = np.loadtxt(self.init_file)
            self.init_temperature /= self.scalings_.temperature


@dataclass
class _MeshParameters:
    """Stores parameters in the mesh section in the configuration data."""

    outer_radius: float
    inner_radius: float
    number_of_nodes: int
    mixing_length_profile: str
    core_density: float
    # Static pressure profile is derived from the Adams-Williamson equation of state.
    eos_method: int = 1 # 1: Adams-Williamson / 2: User defined
    surface_density: float = 4000
    gravitational_acceleration: float = 9.81
    adiabatic_bulk_modulus: float = 260e9
    adams_williamson_beta: float = 0.0  # 0 = derive from K_S
    surface_pressure: float = 0.0
    mass_coordinates: bool = False
    eos_file: str = ""
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.outer_radius /= self.scalings_.radius
        self.inner_radius /= self.scalings_.radius
        self.core_density /= self.scalings_.density
        self.surface_density /= self.scalings_.density
        self.gravitational_acceleration /= self.scalings_.gravitational_acceleration
        self.adiabatic_bulk_modulus /= self.scalings_.pressure
        self.surface_pressure /= self.scalings_.pressure

        if self.eos_method == 2:
            if self.eos_file == "":
                msg: str = (f"you must provide a file for setting up equation of state")
                raise ValueError(msg)
            arr = np.loadtxt(self.eos_file)
            self.eos_radius = arr[:,0] / self.scalings_.radius
            self.eos_pressure = arr[:,1] / self.scalings_.pressure
            self.eos_density = arr[:,2] / self.scalings_.density
            self.eos_gravity = arr[:,3] / self.scalings_.gravitational_acceleration
            # Check that provided eos radius roughly match with Aragog mesh
            if ((self.eos_radius[0] < self.inner_radius) or
                (self.eos_radius[-1] > self.outer_radius) or
                (self.eos_radius[-1]-self.eos_radius[0]) < 0.75*(self.outer_radius-self.inner_radius)):
                msg: str = (f"Radius array in EOS file: Values out of range.")
                raise ValueError(msg)

@dataclass
class _PhaseMixedParameters:
    """Stores settings in the phase_mixed section in the configuration data."""

    latent_heat_of_fusion: float
    rheological_transition_melt_fraction: float
    rheological_transition_width: float
    solidus: str
    liquidus: str
    phase: str
    phase_transition_width: float
    grain_size: float
    matprop_smooth_width: float = 0.0
    # Constant-properties mode (matches SPIDER -use_const_properties)
    const_properties: bool = False
    const_rho: float = 4000.0
    const_Cp: float = 1000.0
    const_alpha: float = 1e-5
    const_cond: float = 4.0
    const_log10visc: float = 2.0
    const_T_ref: float = 3500.0
    const_S_ref: float = 3000.0
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.latent_heat_of_fusion /= self.scalings_.latent_heat_per_mass
        self.grain_size /= self.scalings_.radius


@dataclass
class _PhaseParameters:
    """Stores settings in a phase section in the configuration data.

    This is used to store settings from phase_liquid and phase_solid.
    """

    density: float | str
    heat_capacity: float | str
    melt_fraction: float
    thermal_conductivity: float | str
    thermal_expansivity: float | str
    viscosity: float | str
    entropy: float | str = ""
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes if they are numbers.

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        cls_fields: tuple[Field, ...] = fields(self.__class__)
        for field_ in cls_fields:
            value: Any = getattr(self, field_.name)
            try:
                scaling: float = getattr(self.scalings_, field_.name)
                scaled_value = value / scaling
                setattr(self, field_.name, scaled_value)
                logger.info(
                    "%s is a number (value = %s, scaling = %s, scaled_value = %s)",
                    field_.name,
                    value,
                    scaling,
                    scaled_value,
                )
            except AttributeError:
                logger.info("No scaling found for %s", field_.name)
            except TypeError:
                logger.info(
                    "%s is a string (path to a filename) so the data will be scaled later",
                    field_.name,
                )


@dataclass
class _Radionuclide:
    """Stores the settings in a radionuclide section in the configuration data."""

    name: str
    t0_years: float
    abundance: float
    concentration: float
    heat_production: float
    half_life_years: float
    scalings_: _ScalingsParameters = field(init=False)

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        """Scales the attributes.

        Args:
            scalings: scalings
        """
        self.scalings_ = scalings
        self.t0_years /= self.scalings_.time_years
        self.concentration *= 1e-6  # to mass fraction
        self.heat_production /= self.scalings_.power_per_mass
        self.half_life_years /= self.scalings_.time_years

    def get_heating(self, time: npt.NDArray | float) -> npt.NDArray | float:
        """Radiogenic heating

        Args:
            time: Time

        Returns:
            Radiogenic heat production (power per unit mass) as a float if time is a float,
                otherwise a numpy row array where each entry in the row is associated
                with a single time in the time array.
        """
        arg: npt.NDArray | float = np.log(2) * (self.t0_years - time) / self.half_life_years
        heating: npt.NDArray | float = (
            self.heat_production * self.abundance * self.concentration * np.exp(arg)
        )

        return heating


@dataclass
class _SolverParameters:
    """Stores settings in the solver section in the configuration data."""

    start_time: float
    end_time: float
    atol: float
    rtol: float
    scalings_: _ScalingsParameters = field(init=False)
    tsurf_poststep_change: float = 30.0
    event_triggering:bool = False

    def scale_attributes(self, scalings: _ScalingsParameters) -> None:
        self.scalings_ = scalings
        self.start_time /= self.scalings_.time_years
        self.end_time /= self.scalings_.time_years
        self.tsurf_poststep_change /= self.scalings_.temperature


@dataclass(kw_only=True)
class Parameters:
    """Assembles all the parameters.

    All quantities are stored in SI units (time in years).
    Non-dimensionalization has been removed.
    """

    boundary_conditions: _BoundaryConditionsParameters
    energy: _EnergyParameters
    initial_condition: _InitialConditionParameters
    mesh: _MeshParameters
    phase_solid: _PhaseParameters
    phase_liquid: _PhaseParameters
    phase_mixed: _PhaseMixedParameters
    radionuclides: list[_Radionuclide]
    scalings: _ScalingsParameters
    solver: _SolverParameters

    def __post_init__(self):
        # Store scalings_ reference on sub-dataclasses that need it
        # (phase.py reads scalings_ from PhaseParameters and PhaseMixedParameters).
        # With all scales = 1.0, division by scalings is a no-op.
        for sub in [
            self.boundary_conditions, self.energy, self.initial_condition,
            self.mesh, self.phase_solid, self.phase_liquid, self.phase_mixed,
            self.solver,
        ]:
            sub.scalings_ = self.scalings
        for r in self.radionuclides:
            r.scalings_ = self.scalings

        # Load initial temperature from file if IC method 2
        if self.initial_condition.initial_condition == 2:
            if self.initial_condition.init_file == "":
                raise ValueError("you must provide an initial temperature file")
            self.initial_condition.init_temperature = np.loadtxt(
                self.initial_condition.init_file
            )

        # Load EOS from file if EOS method 2
        if self.mesh.eos_method == 2:
            if self.mesh.eos_file == "":
                raise ValueError("you must provide a file for setting up equation of state")
            arr = np.loadtxt(self.mesh.eos_file)
            self.mesh.eos_radius = arr[:, 0]
            self.mesh.eos_pressure = arr[:, 1]
            self.mesh.eos_density = arr[:, 2]
            self.mesh.eos_gravity = arr[:, 3]
            # Validate EOS file radii against mesh bounds with 5% tolerance.
            # Small mismatches arise from Zalmoxis grid vs PROTEUS-passed radii.
            D = self.mesh.outer_radius - self.mesh.inner_radius
            tol = 0.05 * D if D > 0 else 1e3  # 5% of shell thickness
            if (
                (self.mesh.eos_radius[0] < self.mesh.inner_radius - tol)
                or (self.mesh.eos_radius[-1] > self.mesh.outer_radius + tol)
                or (self.mesh.eos_radius[-1] - self.mesh.eos_radius[0])
                < 0.50 * max(D, 1.0)
            ):
                raise ValueError(
                    f"Radius array in EOS file: Values out of range. "
                    f"EOS: [{self.mesh.eos_radius[0]:.3e}, {self.mesh.eos_radius[-1]:.3e}], "
                    f"Mesh: [{self.mesh.inner_radius:.3e}, {self.mesh.outer_radius:.3e}]"
                )

        # Convert radionuclide concentration from ppm to mass fraction
        for r in self.radionuclides:
            r.concentration *= 1e-6

        # UTBL constant: when param_utbl is disabled, zero it out
        if not self.boundary_conditions.param_utbl:
            self.boundary_conditions.param_utbl_const = 0.0

    @classmethod
    def from_file(cls, *filenames) -> Self:
        """Parses the parameters in a configuration file(s)

        Args:
            *filenames: Filenames of the configuration data
        """
        parser: ConfigParser = ConfigParser()
        parser.read(*filenames)

        init_dict: dict[str, Any] = {}
        for section_name, dataclass_ in _get_dataclass_from_section_name().items():
            init_dict[section_name] = parser.parse_section(
                using_dataclass=dataclass_, section_name=section_name
            )
        radionuclides: list[_Radionuclide] = []
        for radionuclide_section in cls.radionuclide_sections(parser):
            radionuclide = parser.parse_section(
                using_dataclass=_Radionuclide, section_name=radionuclide_section
            )
            radionuclides.append(radionuclide)

        init_dict["radionuclides"] = radionuclides

        return cls(**init_dict)  # Unpacking gives required arguments so pylint: disable=E1125

    @staticmethod
    def radionuclide_sections(parser: ConfigParser) -> list[str]:
        """Section names relating to radionuclides

        Sections relating to radionuclides must have the prefix radionuclide_
        """
        return [
            parser[section].name
            for section in parser.sections()
            if section.startswith("radionuclide_")
        ]
