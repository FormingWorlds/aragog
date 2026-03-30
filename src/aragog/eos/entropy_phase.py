"""Entropy-based phase evaluator using PALEOS P-S tables.

Implements the PhaseEvaluatorProtocol interface using entropy as the
state variable instead of temperature. All properties are looked up
from (P, S) via the EntropyEOS class.

This replaces the MixedPhaseEvaluator + SinglePhaseEvaluator stack
for entropy-formulation runs. Phase determination is trivial:
phi = (S - S_sol) / (S_liq - S_sol), no root-finding needed.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from aragog.eos.entropy import EntropyEOS
from aragog.utilities import FloatOrArray, tanh_weight

logger = logging.getLogger(__name__)


class EntropyPhaseEvaluator:
    """Phase evaluator using entropy as the state variable.

    Implements the same interface as MixedPhaseEvaluator /
    CompositePhaseEvaluator, but all lookups use (P, S) from the
    EntropyEOS tables. No solidus/liquidus root-finding is needed.

    Parameters
    ----------
    entropy_eos : EntropyEOS
        Loaded P-S EOS tables.
    gravitational_acceleration : float or array
        Gravitational acceleration profile [m/s^2].
    rheological_transition_melt_fraction : float
        Melt fraction at which viscosity transitions from solid to liquid.
    rheological_transition_width : float
        Width of the tanh viscosity transition.
    viscosity_solid : float
        Reference solid viscosity [Pa s].
    viscosity_liquid : float
        Reference liquid viscosity [Pa s].
    grain_size : float
        Grain size for permeability calculation [m].
    latent_heat_constant : float
        Latent heat of fusion [J/kg]. Used for gravitational separation flux.
    """

    def __init__(
        self,
        entropy_eos: EntropyEOS,
        gravitational_acceleration: FloatOrArray,
        rheological_transition_melt_fraction: float = 0.4,
        rheological_transition_width: float = 0.15,
        viscosity_solid: float = 1e21,
        viscosity_liquid: float = 1e-1,
        grain_size: float = 1e-3,
        latent_heat_constant: float = 4e5,
        thermal_conductivity_solid: float = 4.0,
        thermal_conductivity_liquid: float = 2.0,
    ):
        self._eos = entropy_eos
        self._g = gravitational_acceleration
        self._phi_rheo = rheological_transition_melt_fraction
        self._phi_width = rheological_transition_width
        self._visc_solid = viscosity_solid
        self._visc_liquid = viscosity_liquid
        self._grain_size = grain_size
        self._latent_heat_constant = latent_heat_constant
        self._k_solid = thermal_conductivity_solid
        self._k_liquid = thermal_conductivity_liquid

        # State arrays (set by set_entropy / set_pressure / update)
        self.entropy: npt.NDArray = np.array([])
        self.pressure: npt.NDArray = np.array([])

        # Cached properties (computed on update)
        self._temperature: npt.NDArray = np.array([])
        self._density: npt.NDArray = np.array([])
        self._heat_capacity: npt.NDArray = np.array([])
        self._thermal_expansivity: npt.NDArray = np.array([])
        self._dTdPs_val: npt.NDArray = np.array([])
        self._melt_fraction: npt.NDArray = np.array([])
        self._viscosity_val: npt.NDArray = np.array([])
        self._thermal_conductivity_val: npt.NDArray = np.array([])

    # ── State setters (match PhaseEvaluatorProtocol interface) ────────

    def set_entropy(self, entropy: npt.NDArray) -> None:
        """Set the entropy profile."""
        self.entropy = np.asarray(entropy, dtype=float)

    def set_temperature(self, temperature: npt.NDArray) -> None:
        """Not used in entropy mode. Kept for interface compatibility."""
        pass

    def set_pressure(self, pressure: npt.NDArray) -> None:
        """Set the pressure profile."""
        self.pressure = np.asarray(pressure, dtype=float)

    def update(self) -> None:
        """Recompute all cached properties from current (P, S)."""
        P = self.pressure
        S = self.entropy

        self._temperature = self._eos.temperature(P, S)
        self._density = self._eos.density(P, S)
        self._heat_capacity = self._eos.heat_capacity(P, S)
        self._dTdPs_val = self._eos.dTdPs(P, S)
        self._thermal_expansivity = self._eos.thermal_expansivity(P, S)
        self._melt_fraction = self._eos.melt_fraction(P, S)

        # Guard: clamp negative alpha to zero (can occur from EOS table edges)
        self._thermal_expansivity = np.maximum(self._thermal_expansivity, 0.0)

        # Viscosity: tanh blend between solid and liquid at phi_rheo
        phi = self._melt_fraction
        w = tanh_weight(phi, self._phi_rheo, self._phi_width)
        log_visc = (1.0 - w) * np.log10(self._visc_solid) + w * np.log10(self._visc_liquid)
        self._viscosity_val = 10.0 ** log_visc

        # Thermal conductivity from config (not hardcoded)
        self._thermal_conductivity_val = (1.0 - phi) * self._k_solid + phi * self._k_liquid

        # P-dependent latent heat from EOS
        self._latent_heat_val = self._eos.latent_heat(P)

    # ── Property accessors (PhaseEvaluatorProtocol) ──────────────────

    def density(self) -> FloatOrArray:
        return self._density

    def temperature(self) -> FloatOrArray:
        """Temperature from EOS lookup (not a state variable)."""
        return self._temperature

    def dTdPs(self) -> npt.NDArray:
        """Adiabatic temperature gradient dT/dP|_S [K/Pa]."""
        return self._dTdPs_val

    def dTdrs(self) -> npt.NDArray:
        """Adiabatic temperature gradient dT/dr|_S [K/m].

        dT/dr|_S = -g * alpha * T / Cp
        """
        return (
            -self._g * self._thermal_expansivity * self._temperature
            / self._heat_capacity
        )

    def gravitational_acceleration(self) -> FloatOrArray:
        return self._g

    def heat_capacity(self) -> FloatOrArray:
        return self._heat_capacity

    def kinematic_viscosity(self) -> FloatOrArray:
        return self._viscosity_val / self._density

    def melt_fraction(self) -> FloatOrArray:
        return self._melt_fraction

    def latent_heat(self) -> FloatOrArray:
        """Latent heat L(P) = T_fus × (S_liq - S_sol) [J/kg].

        P-dependent from EOS tables, matching SPIDER convention.
        """
        return self._latent_heat_val

    def thermal_conductivity(self) -> FloatOrArray:
        return self._thermal_conductivity_val

    def thermal_expansivity(self) -> FloatOrArray:
        return self._thermal_expansivity

    def viscosity(self) -> FloatOrArray:
        return self._viscosity_val

    def relative_velocity(self) -> FloatOrArray:
        """Melt-solid relative velocity for gravitational separation [m/s].

        Uses Abe (1993) three-regime permeability model based on porosity
        (volume fraction of melt, not mass fraction), matching SPIDER's
        GetGravitationalHeatFlux in energy.c.

        Regimes:
        1. Blake-Kozeny-Carman (low porosity): K = d^2 por^3 / (1-por)^2 / 1000
        2. Rumpf-Gupte (intermediate): K = d^2 por^4.5 / 5.6
        3. Stokes settling (high porosity): K = d^2 * 2(1-por)^2 / 9
        """
        phi = self._melt_fraction
        rho_s = self._eos._lookup_at_phase_boundary('density', self.pressure, 'solid')
        rho_l = self._eos._lookup_at_phase_boundary('density', self.pressure, 'melt')
        delta_rho = rho_l - rho_s  # typically negative (melt lighter)
        g = self._g
        d = self._grain_size
        eta_l = self._visc_liquid

        # Porosity (volume fraction of melt) from densities
        rho = self._density
        porosity = np.clip(
            (rho_s - rho) / np.maximum(rho_s - rho_l, 1.0), 0.0, 1.0
        )

        # Three-regime permeability / porosity (SPIDER convention: F = K/porosity)
        por = np.maximum(porosity, 1e-20)
        one_m_por = np.maximum(1.0 - porosity, 1e-20)

        # Blake-Kozeny-Carman (low porosity)
        F_bkc = d**2 * por**2 / (one_m_por**2 * 1000.0)
        # Rumpf-Gupte (intermediate porosity)
        F_rg = d**2 * por**3.5 / 5.6
        # Stokes settling (high porosity)
        F_stokes = d**2 * 2.0 * one_m_por**2 / 9.0

        # Regime switching at critical porosities (Abe 1993)
        # BKC -> RG transition at porosity where BKC = RG
        # RG -> Stokes transition at porosity where RG = Stokes
        w_rg = tanh_weight(porosity, 0.1, 0.05)
        w_stokes = tanh_weight(porosity, 0.4, 0.1)
        F = (1.0 - w_rg) * F_bkc + (w_rg - w_stokes) * F_rg + w_stokes * F_stokes
        F = np.maximum(F, 0.0)

        # Relative velocity: v = delta_rho * g * F / eta_liquid
        v_rel = delta_rho * g * F / np.maximum(eta_l, 1e-10)

        return v_rel

    def delta_specific_volume(self) -> FloatOrArray:
        """Specific volume difference between solid and liquid [m^3/kg]."""
        rho_s = self._eos._lookup_at_phase_boundary('density', self.pressure, 'solid')
        rho_l = self._eos._lookup_at_phase_boundary('density', self.pressure, 'melt')
        return 1.0 / np.maximum(rho_l, 1.0) - 1.0 / np.maximum(rho_s, 1.0)

    # ── Entropy-specific methods ─────────────────────────────────────

    def capacitance(self) -> npt.NDArray:
        """Capacitance for the entropy equation: rho * T [kg K / m^3].

        The entropy equation is rho * T * dS/dt = div(F).
        """
        return self._density * self._temperature
