"""Evaluator class that assembles mesh, boundary conditions, phases, and initial condition."""

from __future__ import annotations

from dataclasses import dataclass, field

from aragog.mesh import Mesh
from aragog.parser import Parameters, _Radionuclide
from aragog.eos import PhaseEvaluatorCollection
from aragog.solver.boundary import BoundaryConditions
from aragog.solver.initial import InitialCondition


@dataclass
class Evaluator:
    """Contains classes that evaluate quantities necessary to compute the interior evolution.

    Args:
        _parameters: Parameters

    Attributes:
        boundary_conditions: Boundary conditions
        initial_condition: Initial condition
        mesh: Mesh
        phases: Evaluators for all phases
        radionuclides: Radionuclides
    """

    _parameters: Parameters
    boundary_conditions: BoundaryConditions = field(init=False)
    initial_condition: InitialCondition = field(init=False)
    mesh: Mesh = field(init=False)
    phases: PhaseEvaluatorCollection = field(init=False)

    def __post_init__(self):
        self.mesh = Mesh(self._parameters)
        self.boundary_conditions = BoundaryConditions(self._parameters, self.mesh)
        self.phases = PhaseEvaluatorCollection(self._parameters)
        self.initial_condition = InitialCondition(self._parameters, self.mesh, self.phases)

    @property
    def radionuclides(self) -> list[_Radionuclide]:
        return self._parameters.radionuclides
