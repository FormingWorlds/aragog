"""Aragog configuration system.

Provides attrs-based configuration classes as the modern alternative
to the legacy dataclass-based parser. Supports construction from
TOML files or dictionaries.

During the transition period, Config.to_parameters() converts to
the legacy Parameters object that the solver expects.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np

from aragog.config.boundary import BoundaryConfig
from aragog.config.energy import EnergyConfig
from aragog.config.initial_condition import InitialConditionConfig
from aragog.config.mesh import MeshConfig
from aragog.config.phases import MixedPhaseConfig, PhaseConfig
from aragog.config.radionuclides import RadionuclideConfig
from aragog.config.scalings import ScalingsConfig
from aragog.config.solver import SolverConfig

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

logger: logging.Logger = logging.getLogger(__name__)

# Re-export all config classes for convenient access
__all__ = [
    "Config",
    "BoundaryConfig",
    "EnergyConfig",
    "InitialConditionConfig",
    "MeshConfig",
    "MixedPhaseConfig",
    "PhaseConfig",
    "RadionuclideConfig",
    "ScalingsConfig",
    "SolverConfig",
]


class Config:
    """Top-level Aragog configuration.

    This is a facade that constructs a legacy Parameters object
    from modern attrs-based sub-configs or from TOML/dict input.
    During the transition, Solver still expects Parameters internally.

    Parameters
    ----------
    scalings : dict
        Scaling parameters (radius, temperature, density, time).
    solver : dict
        ODE solver parameters.
    boundary_conditions : dict
        Boundary condition parameters.
    mesh : dict
        Mesh parameters.
    energy : dict
        Energy source toggles.
    initial_condition : dict
        Initial condition parameters.
    phase_liquid : dict
        Liquid phase properties.
    phase_solid : dict
        Solid phase properties.
    phase_mixed : dict
        Mixed phase parameters.
    radionuclide_* : dict
        Radionuclide sections (any key starting with 'radionuclide_').
    """

    @staticmethod
    def from_toml(filename: str) -> "Parameters":
        """Load configuration from a TOML file and return a Parameters object.

        Parameters
        ----------
        filename : str
            Path to the TOML file.

        Returns
        -------
        Parameters
            Legacy Parameters object, ready for Solver.
        """
        from pathlib import Path

        with Path(filename).open("rb") as f:
            data = tomllib.load(f)

        return Config.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Parameters":
        """Construct a Parameters object from a nested dictionary.

        This is the primary construction path used by the PROTEUS wrapper.

        Parameters
        ----------
        data : dict
            Nested dictionary with section names as keys.

        Returns
        -------
        Parameters
            Legacy Parameters object, ready for Solver.
        """
        from aragog.parser import (
            Parameters,
            _BoundaryConditionsParameters,
            _EnergyParameters,
            _InitialConditionParameters,
            _MeshParameters,
            _PhaseMixedParameters,
            _PhaseParameters,
            _Radionuclide,
            _ScalingsParameters,
            _SolverParameters,
        )

        scalings = _ScalingsParameters(**data.get("scalings", {}))
        solver = _SolverParameters(**data["solver"])
        boundary_conditions = _BoundaryConditionsParameters(
            **data["boundary_conditions"]
        )
        mesh = _MeshParameters(**data["mesh"])
        energy = _EnergyParameters(**data["energy"])
        initial_condition = _InitialConditionParameters(
            **data.get("initial_condition", {})
        )
        phase_liquid = _PhaseParameters(**data["phase_liquid"])
        phase_solid = _PhaseParameters(**data["phase_solid"])
        phase_mixed = _PhaseMixedParameters(**data["phase_mixed"])

        radionuclides: list[_Radionuclide] = []
        for key, val in data.items():
            if key.startswith("radionuclide_"):
                radionuclides.append(_Radionuclide(**val))

        return Parameters(
            boundary_conditions=boundary_conditions,
            energy=energy,
            initial_condition=initial_condition,
            mesh=mesh,
            phase_solid=phase_solid,
            phase_liquid=phase_liquid,
            phase_mixed=phase_mixed,
            radionuclides=radionuclides,
            scalings=scalings,
            solver=solver,
        )

    @staticmethod
    def from_file(*filenames: str) -> "Parameters":
        """Load from a file, auto-detecting format (TOML or INI).

        Parameters
        ----------
        filenames : str
            One or more file paths.

        Returns
        -------
        Parameters
        """
        from pathlib import Path

        from aragog.parser import Parameters

        paths = [Path(f) for f in filenames]

        if any(p.suffix == ".toml" for p in paths):
            toml_path = next(p for p in paths if p.suffix == ".toml")
            return Config.from_toml(str(toml_path))

        # Fall back to legacy INI parser
        return Parameters.from_file(*filenames)


# Make Parameters importable from aragog.config for convenience
def _get_parameters_class():
    from aragog.parser import Parameters
    return Parameters
