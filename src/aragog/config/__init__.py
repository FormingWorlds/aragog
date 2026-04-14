"""Aragog configuration loader.

Provides ``Config.from_toml`` / ``Config.from_dict`` / ``Config.from_file``
that build a ``Parameters`` object from TOML or dict input.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["Config"]


class Config:
    """Construct a Parameters object from TOML or dict input.

    Parameters
    ----------
    solver : dict
        ODE solver parameters.
    boundary_conditions : dict
        Boundary condition parameters.
    mesh : dict
        Mesh parameters.
    energy : dict
        Energy source toggles.
    initial_condition : dict, optional
        Initial condition parameters.
    phase_liquid, phase_solid : dict
        Single-phase properties.
    phase_mixed : dict
        Mixed-phase parameters (latent heat, rheological transition, ...).
    radionuclide_* : dict
        Radionuclide sections (any key starting with 'radionuclide_').
    """

    @staticmethod
    def from_toml(filename: str) -> "Parameters":
        """Load configuration from a TOML file."""
        from pathlib import Path

        with Path(filename).open("rb") as f:
            data = tomllib.load(f)

        return Config.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Parameters":
        """Construct a Parameters object from a nested dictionary.

        Used by the PROTEUS wrapper as the primary construction path.
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
            _SolverParameters,
        )

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
            solver=solver,
        )

    @staticmethod
    def from_file(*filenames: str) -> "Parameters":
        """Load from one or more files, auto-detecting TOML vs INI."""
        from pathlib import Path

        from aragog.parser import Parameters

        paths = [Path(f) for f in filenames]

        if any(p.suffix == ".toml" for p in paths):
            toml_path = next(p for p in paths if p.suffix == ".toml")
            return Config.from_toml(str(toml_path))

        return Parameters.from_file(*filenames)
