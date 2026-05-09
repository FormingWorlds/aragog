"""Parses the configuration file and stores the parameters.

All quantities are stored in SI units (with time in years where the
field name carries a ``_years`` suffix). Aragog's solver applies its
own internal nondimensionalisation around the integrator (see
``aragog.jax.nondim`` and ``EntropySolver._S_ref``/``_t_ref_yr``);
nothing in the config layer rescales user-supplied values.

Configuration files containing a ``[scalings]`` section are
strict-rejected at load time. Remove the section.
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

logger: logging.Logger = logging.getLogger('fwl.' + __name__)


def _get_dataclass_from_section_name() -> dict[str, Any]:
    """Maps the section names in the configuration data to the dataclasses that stores the data."""
    mapping: dict[str, Any] = {
        'solver': _SolverParameters,
        'boundary_conditions': _BoundaryConditionsParameters,
        'mesh': _MeshParameters,
        'energy': _EnergyParameters,
        'initial_condition': _InitialConditionParameters,
        'phase_liquid': _PhaseParameters,
        'phase_solid': _PhaseParameters,
        'phase_mixed': _PhaseMixedParameters,
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
    # Core BC mode selector, threaded from config/boundary.py.
    # Valid values: 'quasi_steady' (alpha-factor flux partition,
    #               state vector length N),
    #               'energy_balance' (SPIDER-parity dSdr_cmb evolution,
    #               state vector length N+1),
    #               'gradient' (entropy gradient as primary state,
    #               length N+2),
    #               'bower2018' (unrecommended; T_core as ODE state,
    #               retained for parity testing only).
    # See aragog/config/boundary.py docstring for details.
    # Default 'energy_balance' matches the PROTEUS production path.
    core_bc: str = 'energy_balance'

    def normalize(self) -> None:
        """Normalise BC values that need post-parse adjustment.

        Replaces what used to be ``scale_attributes`` once
        non-dimensionalisation was removed: the inner/outer dispatch
        still has to zero IBC=1 and zero ``param_utbl_const`` when
        the UTBL correction is disabled, and reject unknown BC codes.

        Must remain idempotent: ``Parameters.__post_init__`` calls
        this method once, but a future caller could legitimately
        invoke it on a bare ``_BoundaryConditionsParameters`` before
        passing it to ``Parameters``. Re-normalising must produce the
        same state. Any future logic added here that mutates a value
        non-idempotently (e.g. multiplying by a factor) needs an
        explicit "already normalised" sentinel to be safe.
        """
        if self.param_utbl:
            # The constant carries dimension [K^2]; with no
            # nondimensionalisation it is left at the user value.
            pass
        else:
            self.param_utbl_const = 0.0
        self._normalize_inner_boundary_condition()
        self._normalize_outer_boundary_condition()

    def _normalize_inner_boundary_condition(self) -> None:
        """Normalise the inner boundary value.

        Equivalent to CORE_BC in C code.
            1: Simple core cooling
            2: Prescribed heat flux
            3: Prescribed temperature
        """
        if self.inner_boundary_condition == 1:
            self.inner_boundary_value = 0
        elif self.inner_boundary_condition in (2, 3):
            pass
        else:
            msg: str = f'inner_boundary_condition = {self.inner_boundary_condition} is unknown'
            raise ValueError(msg)

    def _normalize_outer_boundary_condition(self) -> None:
        """Normalise the outer boundary value.

        Equivalent to SURFACE_BC in C code.
            1: Grey-body atmosphere
            2: Zahnle steam atmosphere (not implemented)
            4: Prescribed surface heat flux (atmosphere coupling)
            5: Prescribed surface temperature
        """
        if self.outer_boundary_condition in (1, 2, 4, 5):
            pass
        else:
            msg: str = f'outer_boundary_condition = {self.outer_boundary_condition} is unknown'
            raise ValueError(msg)


@dataclass
class _EnergyParameters:
    """Stores parameters in the energy section"""

    conduction: bool
    convection: bool
    gravitational_separation: bool
    mixing: bool
    radionuclides: bool
    tidal: bool
    eddy_diffusivity_chemical: float = 1.0
    # Phase-dependent eddy diffusivity floor [m^2/s]. When > 0 the
    # eddy diffusivity is clamped from below by ``floor * f(phi)``,
    # where ``f`` falls from 1 (liquid) to 0 (solid) across the
    # rheological transition. Default 10.0 matches the PROTEUS
    # production value; set to 0.0 for textbook MLT.
    kappah_floor: float = 10.0

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
    # 'tanh' is the SPIDER-parity two-branch tanh of width matprop_smooth_width
    # (default 0.01); it passes full-strength fluxes across the mushy zone and
    # is the production setting. 'cubic_hermite' applies the
    # 16*phi^2*(1-phi)^2 weight for intermediate-phi damping and is retained as
    # a fallback for cases where residual EOS mismatch would otherwise cause a
    # CMB drain; it is not the production setting.
    phase_smoothing: str = 'tanh'
    # ODE solver method: 'cvode' (SUNDIALS CVODE via scikits.odes,
    # default; same solver SPIDER uses, recommended for production
    # stiff integration), 'radau' (scipy Radau), 'bdf' (scipy BDF).
    # When scikits.odes is not installed the solver falls back to
    # scipy Radau and emits a warning at solve time.
    solver_method: str = 'cvode'

    # Option Z: when True and solver_method == 'cvode', the ODE integrator
    # uses a JAX-derived analytic Jacobian (built by PROTEUS via
    # EntropySolver.set_jax_cvode_factory) in place of CVODE's default
    # finite-difference Jacobian. Requires JAX and the aragog.jax module.
    # Default True (matches the production configuration). Has
    # no effect unless a factory is registered before solve(); silently
    # falls back to FD Jacobian when no factory is available.
    use_jax_jacobian: bool = True

    # Per-call mass-weighted |ΔΦ_global| cap. When > 0 and at least one
    # cell sits in or near the mushy band at solve() entry, register a
    # SUNDIALS root function that fires when |Φ_global(t) − Φ_global(
    # start)| reaches this value, returning early with status=2 so the
    # PROTEUS outer loop can adjust dt. Without a cap the dt adapter
    # can land on a step that straddles the rheological transition and
    # reject. 0.05 is a useful upper bound for the mushy zone in 1 M⊕
    # runs; default 0.0 disables the cap entirely.
    phi_step_cap: float = 0.0

    tidal_array: npt.NDArray = field(default_factory=lambda: np.array([0.0], dtype=float))


@dataclass
class _InitialConditionParameters:
    """Stores the settings in the initial_condition section in the configuration data."""

    initial_condition: int = 1
    surface_temperature: float = 4000
    basal_temperature: float = 4000
    init_file: str = ''


@dataclass
class _MeshParameters:
    """Stores parameters in the mesh section in the configuration data."""

    outer_radius: float
    inner_radius: float
    number_of_nodes: int
    mixing_length_profile: str
    core_density: float
    # Static pressure profile is derived from the Adams-Williamson equation of state.
    eos_method: int = 1  # 1: Adams-Williamson / 2: User defined
    # Surface mantle density [kg/m^3]. Default 4078.95095544 matches
    # the SPIDER -adams_williamson_rhos value used in PROTEUS
    # production runs.
    surface_density: float = 4078.95095544
    gravitational_acceleration: float = 9.81
    adiabatic_bulk_modulus: float = 260e9
    adams_williamson_beta: float = 0.0  # 0 = derive from K_S
    surface_pressure: float = 0.0
    # Use uniform spacing in mass coordinate space instead of radius;
    # gives larger cells at the surface where density is lower and
    # matches SPIDER's mesh layout. Default True matches the PROTEUS
    # production path.
    mass_coordinates: bool = True
    eos_file: str = ''
    # Fraction of mantle thickness used as the mixing length when
    # ``mixing_length_profile = 'constant'``. Ignored otherwise.
    mixing_length_constant_fraction: float = 0.25


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
    # Mushy-zone Cp blending mode: "latent" (SPIDER parity, default) or
    # "linear". Consumed by EntropyPhaseEvaluator via getattr fall-back,
    # but accepting it here lets the documented [phase_mixed] cp_blend
    # key in TOML / dict configs round-trip without a TypeError.
    cp_blend: str = 'latent'
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
    Float-valued fields are stored verbatim; string-valued fields are
    interpreted as paths to lookup tables by ``EntropyPhaseEvaluator``.
    """

    density: float | str
    heat_capacity: float | str
    melt_fraction: float
    thermal_conductivity: float | str
    thermal_expansivity: float | str
    viscosity: float | str
    entropy: float | str = ''


@dataclass
class _Radionuclide:
    """Stores the settings in a radionuclide section in the configuration data."""

    name: str
    t0_years: float
    abundance: float
    concentration: float
    heat_production: float
    half_life_years: float

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
        # Boundary-condition normalisation (zero param_utbl_const when
        # disabled, dispatch IBC/OBC, reject unknown codes).
        self.boundary_conditions.normalize()

        # Initial-condition file load for IC method 2 (user-defined T).
        if self.initial_condition.initial_condition == 2:
            if self.initial_condition.init_file == '':
                raise ValueError('you must provide an initial temperature file')
            self.initial_condition.init_temperature = np.loadtxt(
                self.initial_condition.init_file
            )

        # External EOS file load for mesh.eos_method = 2.
        if self.mesh.eos_method == 2:
            if self.mesh.eos_file == '':
                raise ValueError('you must provide a file for setting up equation of state')
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
                or (self.mesh.eos_radius[-1] - self.mesh.eos_radius[0]) < 0.50 * max(D, 1.0)
            ):
                raise ValueError(
                    f'Radius array in EOS file: Values out of range. '
                    f'EOS: [{self.mesh.eos_radius[0]:.3e}, {self.mesh.eos_radius[-1]:.3e}], '
                    f'Mesh: [{self.mesh.inner_radius:.3e}, {self.mesh.outer_radius:.3e}]'
                )

        # Convert radionuclide concentration from ppm to mass fraction.
        for r in self.radionuclides:
            r.concentration *= 1e-6

    @classmethod
    def from_file(cls, *filenames) -> Self:
        """Parses the parameters in a configuration file(s)

        Args:
            *filenames: Filenames of the configuration data
        """
        parser: ConfigParser = ConfigParser()
        parser.read(*filenames)

        # Strict-reject the legacy [scalings] section. Aragog no longer
        # has a config-layer non-dimensionalisation step; a [scalings]
        # block in a TOML used to be parsed and then forced to unity
        # everywhere, which silently let stale dimensional scales leak
        # back in if a downstream refactor ever toggled the override
        # off. Refusing the section at load time forecloses that path.
        # Case-insensitive: typed_configparser preserves section-name
        # case, so a stray [Scalings] / [SCALINGS] would otherwise slip
        # past a literal-string check.
        for section in parser.sections():
            if section.lower() == 'scalings':
                raise ValueError(
                    'Configuration contains a [scalings] section, which is no '
                    'longer accepted. Aragog applies its internal '
                    'nondimensionalisation around the integrator only; remove '
                    'the [scalings] block from the configuration file.'
                )

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

        init_dict['radionuclides'] = radionuclides

        return cls(**init_dict)  # Unpacking gives required arguments so pylint: disable=E1125

    @staticmethod
    def radionuclide_sections(parser: ConfigParser) -> list[str]:
        """Section names relating to radionuclides

        Sections relating to radionuclides must have the prefix radionuclide_
        """
        return [
            parser[section].name
            for section in parser.sections()
            if section.startswith('radionuclide_')
        ]
