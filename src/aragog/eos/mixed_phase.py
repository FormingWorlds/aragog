#
# Copyright 2024 Dan J. Bower
#
# This file is part of Aragog.
#
# Aragog is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Aragog is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Aragog. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Mixed-phase evaluator for EOS and transport properties."""

from __future__ import annotations

import logging
import sys

import numpy as np
import numpy.typing as npt

from aragog.eos.base import PhaseEvaluatorABC, PhaseEvaluatorProtocol, PropertyProtocol
from aragog.eos.properties import LookupProperty1D
from aragog.eos.single_phase import SinglePhaseEvaluator
from aragog.parser import Parameters, _PhaseMixedParameters
from aragog.utilities import FloatOrArray, combine_properties, tanh_weight

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class MixedPhaseEvaluator(PhaseEvaluatorABC):
    """Evaluates the EOS and transport properties of a mixed phase.

    This only computes quantities within the mixed phase region between the solidus and the
    liquidus. Computing quantities outside of this region will give incorrect results.

    Args:
        parameters: Parameters

    Attributes:
        settings: Mixed phase parameters
    """

    def __init__(self, parameters: Parameters):
        # Import here to avoid circular imports at module level
        from aragog.eos import setup_gravitational_acceleration

        gravitational_acceleration: PropertyProtocol = setup_gravitational_acceleration(parameters)
        self.settings: _PhaseMixedParameters = parameters.phase_mixed
        self._liquid: PhaseEvaluatorProtocol = SinglePhaseEvaluator(
            parameters.phase_liquid, gravitational_acceleration
        )
        self._solid: PhaseEvaluatorProtocol = SinglePhaseEvaluator(
            parameters.phase_solid, gravitational_acceleration
        )
        self._solidus: LookupProperty1D = self._get_melting_curve_lookup(
            "solidus", self.settings.solidus
        )
        self._liquidus: LookupProperty1D = self._get_melting_curve_lookup(
            "liquidus", self.settings.liquidus
        )
        self._grain_size: float = self.settings.grain_size
        self._latent_heat_constant: float = self.settings.latent_heat_of_fusion

    @override
    def set_pressure(self, pressure: npt.NDArray) -> None:
        """Sets pressure and updates quantities that only depend on pressure.

        Computes the pressure-dependent latent heat via the Clausius-Clapeyron
        relation: L(P) = T_fus * Delta_V / (dT_sol/dP). This replaces the
        constant latent_heat_of_fusion parameter for all physics that depend
        on L (mixing flux, mixed-phase Cp, gravitational separation heat flux).

        When entropy tables are available, also computes L from the entropy
        difference at the phase boundaries for verification.
        """
        super().set_pressure(pressure)
        # Sets the temperature of the solid and liquid phases to the appropriate melting curve in
        # order to evaluate mixed properties
        self._solid.set_temperature(self.solidus())
        self._solid.set_pressure(pressure)
        self._liquid.set_temperature(self.liquidus())
        self._liquid.set_pressure(pressure)
        self._delta_density = self._solid.density() - self._liquid.density()
        self._delta_specific_volume = 1.0/self._liquid.density() - 1.0/self._solid.density()
        self._delta_fusion = self.liquidus() - self.solidus()

        # Pressure-dependent effective latent heat via Clausius-Clapeyron
        # plus sensible heat across the mushy zone.
        #
        # The effective Delta_S across the mushy zone (solidus to liquidus) is:
        #   Delta_S = Delta_S_melt + Delta_S_sensible
        #
        # where Delta_S_melt = Delta_V / (dT_liq/dP) is the true latent heat
        # entropy from Clausius-Clapeyron (using the liquidus as the real
        # melting curve), and Delta_S_sensible = Cp_sol * ln(T_liq/T_sol) is
        # the sensible heat entropy of the solid from the artificial solidus
        # to the melting point.
        #
        # The effective latent heat is then L_eff = T_fus * Delta_S, which
        # matches SPIDER's T_fus * (S_liq - S_sol) formulation.
        T_fus = self.solidus() + 0.5 * self._delta_fusion
        dTliq_dP = self.liquidus_gradient()
        safe_dTliq_dP = np.where(np.abs(dTliq_dP) > 1e-30, dTliq_dP, 1e-30)

        # Compute single-phase heat capacities first (needed for both
        # the CC latent heat and the mixed-phase Cp below)
        self._heat_capacity_sensible_liquid = self._liquid.heat_capacity()
        self._heat_capacity_sensible_solid = self._solid.heat_capacity()

        # Clausius-Clapeyron latent heat entropy at the liquidus (real melting curve).
        # Delta_S_melt should be positive (melting absorbs heat). If negative,
        # the EOS density contrast and Clapeyron slope are inconsistent; warn
        # and use the absolute value as a best-effort estimate.
        delta_S_melt = self._delta_specific_volume / safe_dTliq_dP
        n_negative = np.sum(delta_S_melt < 0) if hasattr(delta_S_melt, '__len__') else (delta_S_melt < 0)
        if np.any(n_negative):
            logger.warning(
                'CC Delta_S_melt < 0 at %d nodes (Delta_V and dT_liq/dP have '
                'inconsistent signs). Using |Delta_S_melt| as fallback.',
                int(np.sum(n_negative)) if hasattr(n_negative, '__len__') else 1,
            )
        delta_S_melt = np.abs(delta_S_melt)

        # Sensible heat entropy of the solid across the mushy zone
        T_sol = self.solidus()
        T_liq = self.liquidus()
        safe_T_sol = np.where(T_sol > 0, T_sol, 1.0)
        delta_S_sensible = self._heat_capacity_sensible_solid * np.log(T_liq / safe_T_sol)
        delta_S_sensible = np.maximum(delta_S_sensible, 0.0)

        # Total effective Delta_S and latent heat
        delta_S_total = delta_S_melt + delta_S_sensible
        self._latent_heat = T_fus * delta_S_total

        # When entropy tables are available, use entropy-derived L instead
        # (more accurate since PALEOS EOS is not CC-consistent at melting curve)
        if self.has_entropy:
            S_sol = self._solid.entropy()
            S_liq = self._liquid.entropy()
            self._latent_heat_from_entropy = T_fus * (S_liq - S_sol)
            # Use entropy-derived L as the primary value
            self._latent_heat = self._latent_heat_from_entropy
        else:
            self._latent_heat_from_entropy = None

        # Heat capacity of the mixed phase (Soucasse, Aragog_notes_properties.pdf):
        #   cp = phi*cp_m + (1-phi)*cp_s + (h_m - h_s) / (T_liq - T_sol)
        # The first two terms are the sensible heat contribution from each phase,
        # the third is the latent heat contribution. Now uses pressure-dependent L.
        self._heat_capacity_latent = self.latent_heat() / self.delta_fusion()

        # Default until update() sets the phi-dependent value:
        # use phi=0.5 as initial estimate
        self._heat_capacity = (
            0.5 * self._heat_capacity_sensible_liquid
            + 0.5 * self._heat_capacity_sensible_solid
            + self._heat_capacity_latent
        )

    @override
    def update(self):
        # Melt fraction without clipping
        # phi<0 for the solid, 0<phi<1 for the mixed phase, and phi>1 for the melt
        self._melt_fraction_no_clip = (self.temperature - self.solidus()) / self.delta_fusion()
        logger.debug("_melt_fraction_no_clip = %s", self.melt_fraction_no_clip())
        self._melt_fraction = np.clip(self._melt_fraction_no_clip, 0, 1)

        # Mixed density by volume additivity
        self._density = combine_properties(
            self.melt_fraction(), 1 / self._liquid.density(), 1 / self._solid.density()
        )
        self._density = 1 / self._density

        # Porosity (volume fraction of melt)
        # Epsilon prevents division by zero when delta_density is very small (high-mass planets).
        # Clipping to [0, 1] prevents unphysical values and avoids singularities
        # in the BKC permeability formula where (1-porosity)^2 appears in the denominator.
        epsilon = 1e-10
        self._porosity = (self._solid.density() - self.density()) / (self.delta_density() + epsilon)
        self._porosity = np.clip(self._porosity, 0.0, 1.0 - epsilon)

        # Relative velocity between melt and solid
        self._relative_velocity = self._get_relative_velocity()

        # Thermal conductivity
        self._thermal_conductivity = combine_properties(
            self.melt_fraction(),
            self._liquid.thermal_conductivity(),
            self._solid.thermal_conductivity(),
        )

        # Heat capacity: sensible + latent contributions
        # cp = phi*cp_m + (1-phi)*cp_s + L / (T_liq - T_sol)
        self._heat_capacity = (
            self.melt_fraction() * self._heat_capacity_sensible_liquid
            + (1 - self.melt_fraction()) * self._heat_capacity_sensible_solid
            + self._heat_capacity_latent
        )

        # Thermal expansivity (Soucasse, Aragog_notes_properties.pdf):
        #   alpha = zeta*alpha_m + (1-zeta)*alpha_s
        #         + rho/(rho_m*rho_s) * (rho_s - rho_m) / (T_liq - T_sol)
        # where zeta = porosity = (rho_s - rho) / (rho_s - rho_m).
        # First two terms: single-phase contributions weighted by porosity.
        # Third term: mixed-phase contribution from density change on melting.
        alpha_single_phase = (
            self._porosity * self._liquid.thermal_expansivity()
            + (1 - self._porosity) * self._solid.thermal_expansivity()
        )
        alpha_mixed_phase = self.delta_density() / self.delta_fusion() / self.density()
        self._thermal_expansivity = alpha_single_phase + alpha_mixed_phase

        # Viscosity
        weight: npt.NDArray = tanh_weight(
            self.melt_fraction(),
            self.settings.rheological_transition_melt_fraction,
            self.settings.rheological_transition_width,
        )
        log10_viscosity: npt.NDArray = combine_properties(
            weight, np.log10(self._liquid.viscosity()), np.log10(self._solid.viscosity())
        )
        self._viscosity = 10**log10_viscosity

    def delta_density(self) -> FloatOrArray:
        return self._delta_density

    def delta_fusion(self) -> npt.NDArray:
        return self._delta_fusion

    @override
    def density(self) -> npt.NDArray:
        return self._density

    @override
    def gravitational_acceleration(self) -> FloatOrArray:
        return self._solid.gravitational_acceleration()

    @override
    def heat_capacity(self) -> FloatOrArray:
        """Heat capacity of the mixed phase :cite:p:`{Equation 4,}SOLO07`"""
        return self._heat_capacity

    @property
    def has_entropy(self) -> bool:
        """Whether the underlying single-phase evaluators have entropy lookups."""
        return self._solid.has_entropy and self._liquid.has_entropy

    def entropy(self) -> FloatOrArray:
        """Mixed-phase entropy: linear interpolation between solid and liquid entropies.

        S_mixed = phi * S_liquid(P, T_liquidus) + (1-phi) * S_solid(P, T_solidus)

        The single-phase entropies are evaluated at the melting curve temperatures
        (solidus for solid, liquidus for liquid), which is where each phase exists
        in thermodynamic equilibrium within the mushy zone.
        """
        return combine_properties(
            self.melt_fraction(),
            self._liquid.entropy(),   # evaluated at T=T_liquidus (set in set_pressure)
            self._solid.entropy(),    # evaluated at T=T_solidus (set in set_pressure)
        )

    def liquidus(self) -> npt.NDArray:
        """Liquidus"""
        liquidus: npt.NDArray = self._liquidus.eval(self.pressure)
        logger.debug("liquidus = %s", liquidus)

        return liquidus

    def liquidus_gradient(self) -> npt.NDArray:
        """Liquidus gradient"""
        return self._liquidus.gradient(self.pressure)

    def melt_fraction_no_clip(self) -> npt.NDArray:
        """Melt fraction without clipping"""
        return self._melt_fraction_no_clip

    @override
    def melt_fraction(self) -> npt.NDArray:
        """Melt fraction of the mixed phase

        The melt fraction is always between zero and one.
        """
        return self._melt_fraction

    @override
    def latent_heat(self) -> FloatOrArray:
        """Pressure-dependent latent heat of fusion.

        After set_pressure() has been called, returns L(P) computed from
        either entropy tables (primary) or Clausius-Clapeyron + sensible
        heat (fallback). Must call set_pressure() before using this method.
        """
        return self._latent_heat

    def latent_heat_constant(self) -> float:
        """Original constant latent heat from configuration [J/kg]."""
        return self._latent_heat_constant

    def latent_heat_from_entropy(self) -> FloatOrArray | None:
        """Latent heat computed from entropy tables (for verification).

        Returns L_S = T_fus * (S_liq - S_sol) if entropy tables are
        available, otherwise None.
        """
        return self._latent_heat_from_entropy

    def porosity(self) -> npt.NDArray:
        """Porosity of the mixed phase, that is the volume fraction occupied by the melt"""
        return self._porosity

    def solidus(self) -> npt.NDArray:
        """Solidus"""
        solidus: npt.NDArray = self._solidus.eval(self.pressure)
        logger.debug("solidus = %s", solidus)

        return solidus

    def solidus_gradient(self) -> npt.NDArray:
        """Solidus gradient"""
        return self._solidus.gradient(self.pressure)

    @override
    def thermal_conductivity(self) -> npt.NDArray:
        return self._thermal_conductivity

    @override
    def thermal_expansivity(self) -> npt.NDArray:
        return self._thermal_expansivity

    @override
    def viscosity(self) -> npt.NDArray:
        return self._viscosity

    @override
    def relative_velocity(self) -> npt.NDArray:
        """Relative velocity between melt and solid"""
        return self._relative_velocity

    def _get_relative_velocity(self) -> npt.NDArray:
        """Compute relative velocity between melt and solid.

        Positive = melt moves outward (buoyant rise).
        delta_density = rho_solid - rho_liquid > 0 for buoyant melt.
        gravitational_acceleration > 0 (magnitude; Aragog uses positive g).
        Result: dv > 0 when melt is lighter than solid (standard case).
        """
        dv = (
            self.delta_density()
            * self.gravitational_acceleration()
            * self._permeability()
            / self._liquid.viscosity()
        )
        return dv

    @override
    def delta_specific_volume(self) -> npt.NDArray:
        """Difference of specific volume between melt and solid"""
        return self._delta_specific_volume

    def _permeability(self) -> npt.NDArray:

        # Three-regime permeability model (Abe 1995):
        #   Stokes:   porosity > 0.771462 (high melt fraction, dilute suspension)
        #   Rumpf-Gupte: 0.0769618 <= porosity <= 0.771462 (intermediate)
        #   Blake-Kozeny-Carman: porosity < 0.0769618 (low melt fraction, porous flow)
        # Thresholds from Abe (1995); the BKC threshold 0.0769618 corrects the
        # truncation artifact 0.0769452 in the Soucasse formulation document.
        permeability = self._permeability_rumpf_gupte()

        permeability = np.where(
            self._porosity > 0.771462,
            self._permeability_stokes(),
            permeability
            )

        permeability = np.where(
            self._porosity < 0.0769618,
            self._permeability_blake_kozeny_carman(),
            permeability
            )

        return permeability

    def _permeability_stokes(self) -> npt.NDArray:
        """Permeability for Stokes flow in the mixed phase"""
        permeability = 2./9.*self._grain_size**2
        return permeability

    def _permeability_blake_kozeny_carman(self) -> npt.NDArray:
        """Permeability for Blake-Kozeny-Carman flow in the mixed phase"""
        permeability = (
            0.001
            * self._grain_size**2
            * self._porosity**2
            / (1-self._porosity)**2
        )
        return permeability

    def _permeability_rumpf_gupte(self) -> npt.NDArray:
        """Permeability for Rumpf-Gupte flow in the mixed phase"""
        permeability = (
            5./7.
            * self._grain_size**2
            * self._porosity**4.5
        )
        return permeability

    def _get_melting_curve_lookup(self, name: str, value: str) -> LookupProperty1D:
        with open(value, encoding="utf-8") as infile:
            header = infile.readline()
            col_names = header[1:].split()
        value_array: npt.NDArray = np.loadtxt(value, ndmin=2)
        logger.debug("loaded melting curve %s = %s", name, value_array)

        return LookupProperty1D(name=name, value=value_array)
