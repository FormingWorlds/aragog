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

from aragog.config.phases import SEPARATION_VISCOSITY_DEFAULT, SEPARATION_VISCOSITY_MODES
from aragog.eos.entropy import EntropyEOS
from aragog.utilities import FloatOrArray, tanh_weight

logger = logging.getLogger('fwl.' + __name__)


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
    separation_viscosity : str
        Drag viscosity source for gravitational separation: ``'melt'``
        (single-phase liquid viscosity, matches SPIDER's
        ``GetGravitationalHeatFlux``) or ``'mixture'`` (the
        rheological-transition-blended bulk viscosity). Default ``'melt'``.
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
        separation_viscosity: str = SEPARATION_VISCOSITY_DEFAULT,
        matprop_smooth_width: float = 0.0,
        const_properties: bool = False,
        const_rho: float = 4000.0,
        const_Cp: float = 1000.0,
        const_alpha: float = 1e-5,
        const_cond: float = 4.0,
        const_log10visc: float = 2.0,
        const_T_ref: float = 3500.0,
        const_S_ref: float = 3000.0,
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
        # Constant-properties mode (matches SPIDER -use_const_properties)
        self._const_properties = const_properties
        self._const_rho = const_rho
        self._const_Cp = const_Cp
        self._const_alpha = const_alpha
        self._const_cond = const_cond
        self._const_log10visc = const_log10visc
        self._const_T_ref = const_T_ref
        self._const_S_ref = const_S_ref
        # 'latent' = SPIDER-parity, latent-heat-augmented Cp
        # 'linear' = pure-phase linear blend (no latent term)
        if cp_blend not in ('latent', 'linear'):
            raise ValueError(f"cp_blend must be 'latent' or 'linear', got {cp_blend!r}")
        self._cp_blend = cp_blend
        if separation_viscosity not in SEPARATION_VISCOSITY_MODES:
            raise ValueError(
                f'separation_viscosity must be one of {SEPARATION_VISCOSITY_MODES}, '
                f'got {separation_viscosity!r}'
            )
        self._separation_viscosity = separation_viscosity

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
        """Recompute all cached properties from current (P, S).

        Two modes:
        - const_properties=True: analytical T(S) with constant rho, Cp,
          alpha, k, visc. Matches SPIDER's -use_const_properties. No EOS
          table lookups. phi=1 always (no phase transitions).
        - const_properties=False (default): single-pass EOS evaluation
          mirroring SPIDER's EOSEval_Composite_TwoPhase.
        """
        if self._const_properties:
            return self._update_const()
        return self._update_eos()

    def _update_const(self) -> None:
        """Constant-properties mode matching SPIDER -use_const_properties.

        T = T_ref * exp((S - S_ref) / Cp). All other properties constant.
        No EOS tables, no phase transitions (phi=1 always).
        """
        S = np.atleast_1d(np.asarray(self.entropy, dtype=float))

        self._temperature = self._const_T_ref * np.exp((S - self._const_S_ref) / self._const_Cp)
        self._density = np.full_like(S, self._const_rho)
        self._heat_capacity = np.full_like(S, self._const_Cp)
        self._thermal_expansivity = np.full_like(S, self._const_alpha)
        self._melt_fraction = np.full_like(S, 1.0)
        self._dTdPs_val = (
            self._const_alpha * self._temperature / (self._const_rho * self._const_Cp)
        )
        self._viscosity_val = np.full_like(S, 10.0**self._const_log10visc)
        self._thermal_conductivity_val = np.full_like(S, self._const_cond)
        self._latent_heat_val = np.zeros_like(S)

        # Flatten if scalar input (capture ndim before atleast_1d)
        if np.ndim(self.entropy) == 0 or np.ndim(self.pressure) == 0:
            for attr in (
                '_temperature',
                '_density',
                '_heat_capacity',
                '_thermal_expansivity',
                '_dTdPs_val',
                '_thermal_conductivity_val',
                '_melt_fraction',
                '_latent_heat_val',
            ):
                setattr(self, attr, np.asarray(getattr(self, attr)).ravel())

    def _update_eos(self) -> None:
        """EOS-based evaluation mirroring SPIDER EOSEval_Composite_TwoPhase.

        Single-pass evaluation (eos_composite.c:177-290). All properties are
        derived from a single set of cached intermediates (S_sol, S_liq, phi,
        gphi, smth, phase-boundary evaluations), eliminating redundant table
        lookups and ensuring bit-identical intermediate values across properties.
        """
        P = self.pressure
        S = self.entropy
        P_arr = np.atleast_1d(np.asarray(P, dtype=float))
        S_arr = np.atleast_1d(np.asarray(S, dtype=float))
        eos = self._eos
        smw = self._matprop_smooth_width

        # ── Step 1: phase boundaries (computed ONCE) ────────────────
        S_sol = eos.solidus_entropy(P_arr)
        S_liq = eos.liquidus_entropy(P_arr)
        dS_phase = np.maximum(S_liq - S_sol, 1e-10)

        # phi (truncated) and gphi (untruncated)
        gphi = (S_arr - S_sol) / dS_phase
        phi_arr = np.clip(gphi, 0.0, 1.0)
        self._melt_fraction = phi_arr

        # smth: matprop_smooth_width blend factor (SPIDER util.c:get_smoothing)
        if smw > 0:
            smth = np.where(
                gphi > 0.5,
                1.0 - tanh_weight(gphi, 1.0, smw),
                tanh_weight(gphi, 0.0, smw),
            )
        else:
            smth = np.where((gphi >= 0.0) & (gphi <= 1.0), 1.0, 0.0)

        # ── Step 2: phase-boundary table evaluations (ONCE each) ────
        # SPIDER eos_composite.c:216-217
        _lookup = eos._lookup_at_phase_boundary
        T_sol = _lookup('temperature', P_arr, 'solid')
        T_liq = _lookup('temperature', P_arr, 'melt')
        rho_sol = _lookup('density', P_arr, 'solid')
        rho_liq = _lookup('density', P_arr, 'melt')
        # Cp and alpha are derived from T, rho, S at boundaries (below)
        # dTdPs at boundaries not needed (computed analytically)

        # ── Step 3: intermediate two-phase properties (SPIDER 222-249) ──
        dT_phase = np.maximum(T_liq - T_sol, 1e-10)
        T_avg = T_sol + 0.5 * dT_phase

        # T: linear blend (line 222-224)
        T_mixed = phi_arr * T_liq + (1.0 - phi_arr) * T_sol

        # rho: harmonic mean (line 236-237)
        inv_rho_mixed = phi_arr / np.maximum(rho_liq, 1.0) + (1.0 - phi_arr) / np.maximum(
            rho_sol, 1.0
        )
        rho_mixed = 1.0 / np.maximum(inv_rho_mixed, 1e-30)

        if self._cp_blend == 'latent':
            # alpha: composite (SPIDER eos_composite.c:246)
            alpha_mixed = (rho_sol - rho_liq) / dT_phase / np.maximum(rho_mixed, 1.0)
            # Cp: latent-heat augmented (SPIDER lines 227-232)
            Cp_mixed = np.maximum((S_liq - S_sol) / dT_phase * T_avg, 100.0)
        else:
            # Linear-blend branch: blends pure-phase table values at the
            # phase boundaries without the latent-heat augmentation. Used
            # when the active config sets ``cp_blend = 'linear'``.
            Cp_sol = _lookup('heat_capacity', P_arr, 'solid')
            Cp_liq = _lookup('heat_capacity', P_arr, 'melt')
            Cp_mixed = phi_arr * Cp_liq + (1.0 - phi_arr) * Cp_sol
            if 'thermal_exp_solid' in eos._tables and 'thermal_exp_melt' in eos._tables:
                alpha_sol_b = _lookup('thermal_exp', P_arr, 'solid')
                alpha_liq_b = _lookup('thermal_exp', P_arr, 'melt')
                alpha_mixed = phi_arr * alpha_liq_b + (1.0 - phi_arr) * alpha_sol_b
            else:
                # Derive from thermodynamic identity
                dTdPs_sol = _lookup('dTdPs', P_arr, 'solid')
                dTdPs_liq = _lookup('dTdPs', P_arr, 'melt')
                alpha_sol_b = dTdPs_sol * rho_sol * Cp_sol / np.maximum(T_sol, 1.0)
                alpha_liq_b = dTdPs_liq * rho_liq * Cp_liq / np.maximum(T_liq, 1.0)
                alpha_mixed = phi_arr * alpha_liq_b + (1.0 - phi_arr) * alpha_sol_b

        # dTdPs: analytical from intermediates (line 249)
        dTdPs_mixed = (
            alpha_mixed * T_mixed / (np.maximum(rho_mixed, 1.0) * np.maximum(Cp_mixed, 100.0))
        )

        # cond: linear blend (line 252-253)
        cond_mixed = phi_arr * self._k_liquid + (1.0 - phi_arr) * self._k_solid

        # ── Step 4: single-phase table evaluations (SPIDER 269-276) ──
        # Evaluate the melt or solid table at the ACTUAL (P, S)
        mushy = (phi_arr > 0) & (phi_arr < 1)
        S_for_solid = np.where(mushy, S_sol, S_arr)
        S_for_melt = np.where(mushy, S_liq, S_arr)

        def _table_lookup(prop_name):
            solid_tbl = eos._tables[f'{prop_name}_solid']
            melt_tbl = eos._tables[f'{prop_name}_melt']
            S_s_c = np.clip(S_for_solid, solid_tbl['S'][0], solid_tbl['S'][-1])
            S_m_c = np.clip(S_for_melt, melt_tbl['S'][0], melt_tbl['S'][-1])
            P_s_c = np.clip(P_arr, solid_tbl['P'][0], solid_tbl['P'][-1])
            P_m_c = np.clip(P_arr, melt_tbl['P'][0], melt_tbl['P'][-1])
            v_sol = solid_tbl['interp'](
                np.column_stack([P_s_c.ravel(), S_s_c.ravel()])
            ).reshape(P_arr.shape)
            v_mel = melt_tbl['interp'](np.column_stack([P_m_c.ravel(), S_m_c.ravel()])).reshape(
                P_arr.shape
            )
            return np.where(gphi > 0.5, v_mel, v_sol)

        T_single = _table_lookup('temperature')
        rho_single = _table_lookup('density')
        Cp_single = _table_lookup('heat_capacity')
        dTdPs_single = _table_lookup('dTdPs')
        # alpha: use table if available, else derive from thermodynamic identity
        if 'thermal_exp_solid' in eos._tables and 'thermal_exp_melt' in eos._tables:
            alpha_single = _table_lookup('thermal_exp')
        else:
            alpha_single = dTdPs_single * rho_single * Cp_single / np.maximum(T_single, 1.0)
        cond_single = np.where(gphi > 0.5, self._k_liquid, self._k_solid)

        # ── Step 5: combine_matprop blend (SPIDER 278-285) ──────────
        def _blend(mixed, single):
            return smth * mixed + (1.0 - smth) * single

        self._temperature = _blend(T_mixed, T_single)
        self._density = _blend(rho_mixed, rho_single)
        self._heat_capacity = _blend(Cp_mixed, Cp_single)
        self._thermal_expansivity = _blend(alpha_mixed, alpha_single)
        self._dTdPs_val = _blend(dTdPs_mixed, dTdPs_single)
        self._thermal_conductivity_val = _blend(cond_mixed, cond_single)

        # Flatten if scalar input
        if np.ndim(P) == 0:
            for attr in (
                '_temperature',
                '_density',
                '_heat_capacity',
                '_thermal_expansivity',
                '_dTdPs_val',
                '_thermal_conductivity_val',
                '_melt_fraction',
            ):
                setattr(self, attr, np.asarray(getattr(self, attr)).ravel())

        # Guard: clamp negative alpha (EOS table edges)
        a = self._thermal_expansivity
        eps_a = 1.0e-8
        self._thermal_expansivity = 0.5 * (a + np.sqrt(a * a + eps_a * eps_a))

        # ── Step 6: viscosity (two-stage, reuses cached gphi/smth) ──
        # Stage 1: tanh blend at phi_rheo (SPIDER lines 255-259)
        w = tanh_weight(phi_arr, self._phi_rheo, self._phi_width)
        log_visc_mixed = (1.0 - w) * np.log10(self._visc_solid) + w * np.log10(
            self._visc_liquid
        )

        # Single-phase viscosity (constant per phase)
        log_visc_single = np.where(
            phi_arr > 0.5,
            np.log10(self._visc_liquid),
            np.log10(self._visc_solid),
        )

        # Stage 2: combine_matprop with cached smth
        log_visc = smth * log_visc_mixed + (1.0 - smth) * log_visc_single
        is_scalar = np.ndim(self._melt_fraction) == 0
        self._viscosity_val = 10.0 ** (log_visc.item() if is_scalar else log_visc)

        # ── Step 7: latent heat ─────────────────────────────────────
        self._latent_heat_val = eos.latent_heat(P)

        # NaN detection: catch entropy leaving the EOS table domain
        if np.any(np.isnan(self._temperature)):
            n_nan = int(np.sum(np.isnan(self._temperature)))
            logger.error(
                'NaN from EOS lookup at %d nodes. S range: [%.0f, %.0f], '
                'table domain: [%.0f, %.0f] J/kg/K',
                n_nan,
                float(np.nanmin(S)),
                float(np.nanmax(S)),
                self._eos.S_min,
                self._eos.S_max,
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
        return -self._g * self._thermal_expansivity * self._temperature / self._heat_capacity

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

        Returns zero in const_properties mode (no phase contrast).

        Uses Abe (1993) three-regime permeability model based on porosity
        (volume fraction of melt, not mass fraction), matching SPIDER's
        GetGravitationalHeatFlux in energy.c.

        Regimes (F = K/porosity, the quantity multiplying delta_rho*g/eta):
        1. Blake-Kozeny-Carman (low porosity): F = d^2 por^2 / ((1-por)^2 * 1000)
        2. Rumpf-Gupte (intermediate): F = d^2 por^4.5 * (5/7)
        3. Stokes settling (high porosity): F = d^2 * 2/9

        The drag viscosity is set by ``separation_viscosity``: ``'melt'``
        uses the fixed single-phase liquid viscosity, matching SPIDER's
        ``GetGravitationalHeatFlux``; ``'mixture'`` uses the
        rheological-transition-blended bulk viscosity, so settling locks
        up through the same phi_rheo transition that sets the bulk
        rheology.
        """
        if self._const_properties:
            return np.zeros_like(self._density)
        rho_s = self._eos._lookup_at_phase_boundary('density', self.pressure, 'solid')
        rho_l = self._eos._lookup_at_phase_boundary('density', self.pressure, 'melt')
        delta_rho = rho_l - rho_s  # typically negative (melt lighter)
        g = self._g
        d = self._grain_size
        if self._separation_viscosity == 'mixture':
            eta_l = self._viscosity_val
        else:
            eta_l = self._visc_liquid

        # Porosity (volume fraction of melt) from densities. Smoothed
        # with sqrt-based soft clip + soft max so the CVODE BDF
        # predictor sees a C^infty RHS (previous np.clip + np.maximum
        # had two derivative jumps that locked the solver at order 1).
        rho = self._density
        drho = rho_s - rho_l
        eps = 1.0e-3  # kg/m^3; far below any physical density contrast
        drho_smoothmax = 0.5 * (drho + 1.0 + np.sqrt((drho - 1.0) ** 2 + eps * eps))
        porosity_raw = (rho_s - rho) / drho_smoothmax
        # smooth_clip(porosity_raw, 0, 1) via two soft-max operations
        eps_p = 1.0e-3  # dimensionless; invisible except near [0,1] edges
        p_lo = 0.5 * (porosity_raw + np.sqrt(porosity_raw * porosity_raw + eps_p * eps_p))
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

        # Regime switching at critical porosities. These two values
        # (zeta_1 = 0.0769452, zeta_2 = 0.771462) are NOT material
        # parameters: they are the analytical equality points of the
        # three permeability laws (BKC = RG at zeta_1, RG = Stokes at
        # zeta_2) under the equal-density-ratio limit, derived in
        # Bower et al. 2018 Eqs. 13a-c. They depend only on the
        # functional forms of the three regime laws, not on the
        # underlying material; varying composition changes individual
        # F_bkc / F_rg / F_stokes prefactors but the equality points
        # remain fixed by construction. The tanh blend widths (0.02
        # and 0.05) are numerical-smoothing tunables, not physics.
        w_rg = tanh_weight(porosity, 0.0769452, 0.02)
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
        if self._const_properties:
            return np.zeros_like(self._density)
        rho_s = self._eos._lookup_at_phase_boundary('density', self.pressure, 'solid')
        rho_l = self._eos._lookup_at_phase_boundary('density', self.pressure, 'melt')
        return 1.0 / np.maximum(rho_l, 1.0) - 1.0 / np.maximum(rho_s, 1.0)

    # ── Entropy-specific methods ─────────────────────────────────────

    def capacitance(self) -> npt.NDArray:
        """Capacitance for the entropy equation: rho * T [kg K / m^3].

        The entropy equation is rho * T * dS/dt = div(F).
        """
        return self._density * self._temperature
