"""Parses the configuration file into dataclasses.

All quantities are stored in SI units (time in years).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
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
    # Core BC mode selector. Valid values:
    #   'energy_balance' — SPIDER-parity (default; dSdr_cmb as ODE state)
    #   'quasi_steady'   — alpha-factor heat-flux partition
    #   'gradient'       — full dS/dr state vector
    core_bc: str = 'energy_balance'

    def __post_init__(self) -> None:
        # inner_boundary_condition=1 (simple core cooling) ignores any
        # configured inner_boundary_value; force to zero for clarity.
        if self.inner_boundary_condition == 1:
            self.inner_boundary_value = 0
        if not self.param_utbl:
            self.param_utbl_const = 0.0


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


@dataclass
class _InitialConditionParameters:
    """Stores the settings in the initial_condition section in the configuration data."""

    initial_condition: int = 1
    surface_temperature: float = 4000
    basal_temperature: float = 4000
    init_file: str = ""

    def __post_init__(self) -> None:
        if self.initial_condition == 2:
            if self.init_file == "":
                raise ValueError("you must provide an initial temperature file")
            self.init_temperature = np.loadtxt(self.init_file)


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

    def __post_init__(self) -> None:
        if self.eos_method == 2:
            if self.eos_file == "":
                raise ValueError("you must provide a file for setting up equation of state")
            arr = np.loadtxt(self.eos_file)
            self.eos_radius = arr[:, 0]
            self.eos_pressure = arr[:, 1]
            self.eos_density = arr[:, 2]
            self.eos_gravity = arr[:, 3]
            if ((self.eos_radius[0] < self.inner_radius)
                or (self.eos_radius[-1] > self.outer_radius)
                or (self.eos_radius[-1] - self.eos_radius[0])
                    < 0.75 * (self.outer_radius - self.inner_radius)):
                raise ValueError("Radius array in EOS file: values out of range.")


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


@dataclass
class _Radionuclide:
    """Stores the settings in a radionuclide section in the configuration data."""

    name: str
    t0_years: float
    abundance: float
    concentration: float
    heat_production: float
    half_life_years: float

    def __post_init__(self) -> None:
        self.concentration *= 1e-6  # to mass fraction

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
    tsurf_poststep_change: float = 30.0
    event_triggering: bool = False


@dataclass(kw_only=True)
class Parameters:
    """Assembles all the parameters.

    All quantities are stored in SI units (time in years).
    """

    boundary_conditions: _BoundaryConditionsParameters
    energy: _EnergyParameters
    initial_condition: _InitialConditionParameters
    mesh: _MeshParameters
    phase_solid: _PhaseParameters
    phase_liquid: _PhaseParameters
    phase_mixed: _PhaseMixedParameters
    radionuclides: list[_Radionuclide]
    solver: _SolverParameters

    def __post_init__(self):
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
