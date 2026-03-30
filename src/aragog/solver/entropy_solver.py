"""Entropy-formulation solver for Aragog.

Solves the entropy equation:
    rho * T * dS/dt = (1/r^2) d/dr [r^2 F_total] + rho * H

where S is specific entropy [J/kg/K], F_total is the total energy flux
[W/m^2], and H is the internal heating rate [W/kg].

This replaces the temperature-based Solver for PALEOS EOS runs.
Uses the same BDF time integrator (scipy solve_ivp) and staggered
finite-volume mesh.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.parser import Parameters
from aragog.solver import SECS_PER_YEAR
from aragog.solver.evaluator import Evaluator
from aragog.solver.entropy_state import EntropyState

logger = logging.getLogger(__name__)


class EntropySolver:
    """Entropy-based interior dynamics solver.

    Parameters
    ----------
    parameters : Parameters
        Parsed configuration.
    entropy_eos : EntropyEOS
        Loaded P-S EOS tables.
    """

    def __init__(self, parameters: Parameters, entropy_eos: EntropyEOS):
        self.parameters = parameters
        self.entropy_eos = entropy_eos
        self.evaluator: Evaluator
        self.state: EntropyState
        self._solution: OptimizeResult
        self.stop_early: bool = False

    def initialize(self) -> None:
        """Initialize mesh, boundary conditions, and entropy state."""
        logger.info('Initializing EntropySolver')
        self.evaluator = Evaluator(self.parameters)

        mesh = self.evaluator.mesh
        P_stag = mesh.staggered.pressure
        P_basic = mesh.basic.pressure
        g_basic = mesh.basic.gravitational_acceleration

        # Create entropy phase evaluators for staggered and basic nodes
        phase_kwargs = dict(
            entropy_eos=self.entropy_eos,
            rheological_transition_melt_fraction=(
                self.parameters.phase_mixed.rheological_transition_melt_fraction
            ),
            rheological_transition_width=(
                self.parameters.phase_mixed.rheological_transition_width
            ),
            grain_size=self.parameters.phase_mixed.grain_size,
            latent_heat_constant=self.parameters.phase_mixed.latent_heat_of_fusion,
        )

        # Try to get viscosity from config
        try:
            phase_kwargs['viscosity_solid'] = self.parameters.phase_solid.viscosity.eval()
        except Exception:
            phase_kwargs['viscosity_solid'] = 1e21
        try:
            phase_kwargs['viscosity_liquid'] = self.parameters.phase_liquid.viscosity.eval()
        except Exception:
            phase_kwargs['viscosity_liquid'] = 1e-1

        phase_stag = EntropyPhaseEvaluator(
            gravitational_acceleration=mesh.staggered.gravitational_acceleration,
            **phase_kwargs,
        )
        phase_stag.set_pressure(P_stag)

        phase_basic = EntropyPhaseEvaluator(
            gravitational_acceleration=g_basic,
            **phase_kwargs,
        )
        phase_basic.set_pressure(P_basic)

        # Energy settings
        energy = self.parameters.energy
        self.state = EntropyState(
            evaluator=self.evaluator,
            phase_staggered=phase_stag,
            phase_basic=phase_basic,
            conduction=energy.conduction,
            convection=energy.convection,
            gravitational_separation=energy.gravitational_separation,
            mixing=energy.mixing,
            eddy_diffusivity_chemical=energy.eddy_diffusivity_chemical,
            kappah_floor=energy.kappah_floor,
        )

    def set_initial_entropy(self, S_init: npt.NDArray | float) -> None:
        """Set the initial entropy profile.

        Parameters
        ----------
        S_init : array or float
            Entropy at staggered nodes [J/kg/K]. If scalar, sets uniform
            (isentropic) profile.
        """
        n_stag = self.evaluator.mesh.staggered.radius.shape[0]
        if np.isscalar(S_init):
            self._S0 = np.full(n_stag, float(S_init))
        else:
            self._S0 = np.asarray(S_init, dtype=float)
            if len(self._S0) != n_stag:
                raise ValueError(
                    f'S_init length {len(self._S0)} != mesh staggered nodes {n_stag}'
                )
        logger.info('Initial entropy: S_min=%.0f, S_max=%.0f J/kg/K',
                     self._S0.min(), self._S0.max())

    def dSdt(
        self,
        time: npt.NDArray | float,
        entropy: npt.NDArray,
    ) -> npt.NDArray:
        """dS/dt at the staggered nodes.

        Parameters
        ----------
        time : float
            Time [yr].
        entropy : array
            Entropy at staggered nodes [J/kg/K].

        Returns
        -------
        array
            dS/dt at staggered nodes [J/kg/K/yr].
        """
        self.state.update(entropy, time)

        # Apply flux boundary conditions
        self.evaluator.boundary_conditions.apply_flux_boundary_conditions(self.state)

        # Flux divergence: dE/dr at staggered nodes
        energy_flux = self.state.heat_flux * self.evaluator.mesh.basic.area
        delta_energy_flux = np.diff(energy_flux, axis=0)

        # Capacitance: rho * T * volume (for entropy equation)
        capacitance = (
            self.state.capacitance_staggered()
            * self.evaluator.mesh.basic.volume
        )

        # dS/dt from flux divergence [J/kg/K/s]
        dSdt = -delta_energy_flux / capacitance

        # Convert to J/kg/K/yr
        dSdt *= SECS_PER_YEAR

        return dSdt

    @property
    def solution(self) -> OptimizeResult:
        return self._solution

    @property
    def entropy_staggered(self) -> npt.NDArray:
        """Entropy at staggered nodes from the solution."""
        return self._solution.y

    @property
    def temperature_staggered(self) -> npt.NDArray:
        """Temperature at staggered nodes (derived from S via EOS)."""
        S = self._solution.y[:, -1] if self._solution.y.ndim > 1 else self._solution.y
        P = self.evaluator.mesh.staggered.pressure
        return self.entropy_eos.temperature(P, S)

    def solve(self) -> None:
        """Run the BDF time integration."""
        start_time = self.parameters.solver.start_time
        end_time = self.parameters.solver.end_time
        atol = max(self.parameters.solver.atol, 0.01)  # entropy in J/kg/K
        rtol = self.parameters.solver.rtol

        logger.info(
            'EntropySolver: integrating from %.2e to %.2e yr (atol=%.2e, rtol=%.2e)',
            start_time, end_time, atol, rtol,
        )

        self._solution = solve_ivp(
            self.dSdt,
            (start_time, end_time),
            self._S0,
            method='BDF',
            dense_output=True,
            atol=atol,
            rtol=rtol,
        )

        if self._solution.status == 0:
            logger.info('EntropySolver: integration completed successfully.')
            self.stop_early = False
        else:
            logger.error(
                'EntropySolver: integration failed (status=%d): %s',
                self._solution.status, self._solution.message,
            )
            self.stop_early = True
