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
        thermal_conductivity_solid: float = 4.0,
        thermal_conductivity_liquid: float = 2.0,
        cp_blend: str = 'latent',
        matprop_smooth_width: float = 0.0,
    ):
        self._eos = entropy_eos
        self._g = gravitational_acceleration
        self._phi_rheo = rheological_transition_melt_fraction
        self._phi_width = rheological_transition_width
        self._visc_solid = viscosity_solid
        self._visc_liquid = viscosity_liquid
        self._grain_size = grain_size
        self._k_solid = thermal_conductivity_solid
        self._k_liquid = thermal_conductivity_liquid
        self._matprop_smooth_width = matprop_smooth_width
        # 'latent' = SPIDER-parity v4 convention (latent-heat-augmented Cp)
        # 'linear' = legacy v3 convention (pure-phase linear blend)
        if cp_blend not in ('latent', 'linear'):
            raise ValueError(
                f"cp_blend must be 'latent' or 'linear', got {cp_blend!r}"
            )
        self._cp_blend = cp_blend

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
        """Not used in entropy mode. Use set_entropy() instead."""
        raise NotImplementedError(
            'EntropyPhaseEvaluator uses set_entropy(), not set_temperature(). '
            'This evaluator cannot be used with the T-based solver.'
        )

    def set_pressure(self, pressure: npt.NDArray) -> None:
        """Set the pressure profile."""
        self.pressure = np.asarray(pressure, dtype=float)

    def update(self) -> None:
        """Recompute all cached properties from current (P, S)."""
        P = self.pressure
        S = self.entropy

        self._temperature = self._eos.temperature(P, S)
        self._density = self._eos.density(P, S)
        if self._cp_blend == 'latent':
            self._heat_capacity = self._eos.heat_capacity_latent_blend(P, S)
        else:
            self._heat_capacity = self._eos.heat_capacity(P, S)
        if self._cp_blend == 'latent':
            self._thermal_expansivity = (
                self._eos.thermal_expansivity_composite_blend(P, S)
            )
        else:
            self._thermal_expansivity = self._eos.thermal_expansivity(P, S)
        self._melt_fraction = self._eos.melt_fraction(P, S)

        # Guard: clamp negative alpha to zero (can occur from EOS table
        # edges). Use a smooth max (sqrt-based, C^infty) so the clip
        # doesn't introduce a kink in CVODE's BDF predictor when a cell
        # sits at alpha ~ 0. The eps=1e-8 bandwidth is much smaller than
        # physical alpha scales (~1e-5 1/K) so bulk values are untouched.
        a = self._thermal_expansivity
        eps_a = 1.0e-8
        self._thermal_expansivity = 0.5 * (a + np.sqrt(a * a + eps_a * eps_a))

        # dTdPs: full SPIDER eos_composite.c two-stage computation.
        #
        # Stage 1 (line 249): analytical from INTERMEDIATE two-phase
        # properties (lines 222-246), NOT the final combine_matprop values.
        #   T_mixed = phi*T_liq + (1-phi)*T_sol
        #   rho_mixed = harmonic(rho_liq, rho_sol, phi)
        #   alpha_mixed = (rho_sol-rho_liq)/(T_liq-T_sol)/rho_mixed
        #   Cp_mixed = (S_liq-S_sol)/(T_liq-T_sol) * T_avg
        #   dTdPs_mixed = alpha_mixed * T_mixed / (rho_mixed * Cp_mixed)
        #
        # Stage 2 (line 283): combine_matprop(smth, mixed, single)
        #   dTdPs_single from table at actual (P, S)
        #   smth = get_smoothing(matprop_smooth_width, gphi)
        P_arr = np.atleast_1d(np.asarray(P, dtype=float))
        S_arr = np.atleast_1d(np.asarray(S, dtype=float))
        phi_arr = np.atleast_1d(np.asarray(self._melt_fraction, dtype=float))

        # Phase boundary properties
        S_sol = self._eos.solidus_entropy(P_arr)
        S_liq = self._eos.liquidus_entropy(P_arr)
        T_sol = self._eos._lookup_at_phase_boundary('temperature', P_arr, 'solid')
        T_liq = self._eos._lookup_at_phase_boundary('temperature', P_arr, 'melt')
        rho_sol = self._eos._lookup_at_phase_boundary('density', P_arr, 'solid')
        rho_liq = self._eos._lookup_at_phase_boundary('density', P_arr, 'melt')

        # Intermediate two-phase properties (lines 222-249)
        T_mixed = phi_arr * T_liq + (1.0 - phi_arr) * T_sol
        inv_rho_mixed = phi_arr / np.maximum(rho_liq, 1.0) + (1.0 - phi_arr) / np.maximum(rho_sol, 1.0)
        rho_mixed = 1.0 / np.maximum(inv_rho_mixed, 1e-30)
        dT_phase = np.maximum(T_liq - T_sol, 1e-10)
        alpha_mixed = (rho_sol - rho_liq) / dT_phase / np.maximum(rho_mixed, 1.0)
        T_avg = T_sol + 0.5 * dT_phase
        Cp_mixed = (S_liq - S_sol) / dT_phase * T_avg
        Cp_mixed = np.maximum(Cp_mixed, 100.0)
        dTdPs_mixed = alpha_mixed * T_mixed / (np.maximum(rho_mixed, 1.0) * Cp_mixed)

        # Single-phase dTdPs from table at actual (P, S)
        dTdPs_single = self._eos.dTdPs(P_arr, S_arr)

        # Smoothing and blend (lines 266-283)
        dS_phase = np.maximum(S_liq - S_sol, 1e-10)
        gphi = (S_arr - S_sol) / dS_phase
        smw = self._matprop_smooth_width
        if smw > 0:
            smth = np.where(
                gphi > 0.5,
                1.0 - tanh_weight(gphi, 1.0, smw),
                tanh_weight(gphi, 0.0, smw),
            )
        else:
            smth = np.where((gphi >= 0.0) & (gphi <= 1.0), 1.0, 0.0)

        self._dTdPs_val = smth * dTdPs_mixed + (1.0 - smth) * dTdPs_single
        if np.ndim(P) == 0:
            self._dTdPs_val = self._dTdPs_val.ravel()

        # Viscosity: two-stage blend matching SPIDER eos_composite.c.
        #
        # Stage 1 (lines 255-259): tanh blend at phi_rheo between solid
        # and liquid log10visc, weighted by truncated phi.
        #
        # Stage 2 (lines 266-285): blend the Stage 1 "mixed" viscosity
        # with the single-phase viscosity using get_smoothing(smth, gphi).
        # When gphi >> 1 or gphi << 0 (deep in single-phase), smth -> 0
        # and the single-phase value dominates. Near the phase boundary,
        # smth -> 1 and the mixed-phase value dominates.
        phi = self._melt_fraction
        is_scalar = np.ndim(phi) == 0
        phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))

        # Stage 1: tanh blend (mixed-phase viscosity)
        w = tanh_weight(phi_arr, self._phi_rheo, self._phi_width)
        log_visc_mixed = (
            (1.0 - w) * np.log10(self._visc_solid)
            + w * np.log10(self._visc_liquid)
        )

        # Single-phase viscosity (constant per phase, no Arrhenius)
        log_visc_single = np.where(
            phi_arr >= 0.5,
            np.log10(self._visc_liquid),
            np.log10(self._visc_solid),
        )

        # Stage 2: matprop_smooth_width blend (SPIDER util.c:get_smoothing)
        smw = self._matprop_smooth_width
        if smw > 0:
            # Untruncated gphi from EOS
            S = self.entropy
            P = self.pressure
            S_sol = self._eos.solidus_entropy(P)
            S_liq = self._eos.liquidus_entropy(P)
            dS = np.maximum(S_liq - S_sol, 1e-10)
            gphi = np.atleast_1d((np.asarray(S) - S_sol) / dS)

            # get_smoothing: tanh transition at gphi=0 and gphi=1
            smth = np.where(
                gphi > 0.5,
                1.0 - tanh_weight(gphi, 1.0, smw),
                tanh_weight(gphi, 0.0, smw),
            )
            log_visc = smth * log_visc_mixed + (1.0 - smth) * log_visc_single
        else:
            # No smoothing (smw=0): hard switch at gphi boundaries
            S = self.entropy
            P = self.pressure
            S_sol = self._eos.solidus_entropy(P)
            S_liq = self._eos.liquidus_entropy(P)
            dS = np.maximum(S_liq - S_sol, 1e-10)
            gphi = np.atleast_1d((np.asarray(S) - S_sol) / dS)
            in_mushy = (gphi >= 0.0) & (gphi <= 1.0)
            log_visc = np.where(in_mushy, log_visc_mixed, log_visc_single)

        self._viscosity_val = 10.0 ** (log_visc.item() if is_scalar else log_visc)

        # Thermal conductivity from config (not hardcoded)
        self._thermal_conductivity_val = (1.0 - phi) * self._k_solid + phi * self._k_liquid

        # P-dependent latent heat from EOS
        self._latent_heat_val = self._eos.latent_heat(P)

        # NaN detection: catch entropy leaving the EOS table domain
        if np.any(np.isnan(self._temperature)):
            n_nan = int(np.sum(np.isnan(self._temperature)))
            logger.error(
                'NaN from EOS lookup at %d nodes. S range: [%.0f, %.0f], '
                'table domain: [%.0f, %.0f] J/kg/K',
                n_nan, float(np.nanmin(S)), float(np.nanmax(S)),
                self._eos.S_min, self._eos.S_max,
            )
            raise RuntimeError(
                f'Entropy out of EOS table domain at {n_nan} nodes. '
                f'S range [{np.nanmin(S):.0f}, {np.nanmax(S):.0f}] vs '
                f'table [{self._eos.S_min:.0f}, {self._eos.S_max:.0f}]'
            )

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

        # Porosity (volume fraction of melt) from densities. Smoothed
        # with sqrt-based soft clip + soft max so the CVODE BDF
        # predictor sees a C^infty RHS (previous np.clip + np.maximum
        # had two derivative jumps that locked the solver at order 1).
        rho = self._density
        drho = rho_s - rho_l
        eps = 1.0e-3  # kg/m^3; far below any physical density contrast
        drho_smoothmax = 0.5 * (
            drho + 1.0 + np.sqrt((drho - 1.0) ** 2 + eps * eps)
        )
        porosity_raw = (rho_s - rho) / drho_smoothmax
        # smooth_clip(porosity_raw, 0, 1) via two soft-max operations
        eps_p = 1.0e-3  # dimensionless; invisible except near [0,1] edges
        p_lo = 0.5 * (
            porosity_raw + np.sqrt(porosity_raw * porosity_raw + eps_p * eps_p)
        )
        hi_u = 1.0 - p_lo
        porosity = 1.0 - 0.5 * (hi_u + np.sqrt(hi_u * hi_u + eps_p * eps_p))

        # Three-regime permeability / porosity (Abe 1993/1995, SPIDER convention).
        # F = permeability(porosity) / porosity. The relative velocity is
        # v = |delta_rho| * g * F / eta_liquid.
        por = np.maximum(porosity, 1e-20)
        one_m_por = np.maximum(1.0 - porosity, 1e-20)

        # Blake-Kozeny-Carman (low porosity): K/por = d^2 por^2 / ((1-por)^2 * 1000)
        F_bkc = d**2 * por**2 / (one_m_por**2 * 1000.0)
        # Rumpf-Gupte (intermediate): K/por = d^2 por^4.5 * 5/7
        F_rg = d**2 * por**4.5 * (5.0 / 7.0)
        # Stokes settling (high porosity): K/por = d^2 * 2/9
        F_stokes = d**2 * 2.0 / 9.0

        # Regime switching at critical porosities (Abe 1995):
        # BKC -> RG at porosity ~0.077, RG -> Stokes at ~0.771
        w_rg = tanh_weight(porosity, 0.0769618, 0.02)
        w_stokes = tanh_weight(porosity, 0.771462, 0.05)
        F = (1.0 - w_rg) * F_bkc + (w_rg - w_stokes) * F_rg + w_stokes * F_stokes
        F = np.maximum(F, 0.0)

        # Relative velocity: v = |delta_rho| * g * F / eta_liquid
        # Sign convention: positive = outward (melt rising, solid sinking)
        # Smoothed |delta_rho| via sqrt(x^2 + eps^2); eps tiny compared
        # to physical delta_rho ~ -500 kg/m^3, so bulk result unchanged.
        abs_drho = np.sqrt(delta_rho * delta_rho + 1.0e-12)
        v_rel = abs_drho * g * F / np.maximum(eta_l, 1e-10)

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
