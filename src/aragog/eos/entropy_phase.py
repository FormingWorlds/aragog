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

from aragog.config.phases import (
    SEPARATION_BUOYANCY_DEFAULT,
    SEPARATION_BUOYANCY_MODES,
    SEPARATION_VISCOSITY_DEFAULT,
    SEPARATION_VISCOSITY_MODES,
    SolidRheologyParams,
    _compute_arrhenius_viscosity_core,
    _compute_yield_stress_core,
    check_viscosity_ceiling,
    merge_rheology,
)
from aragog.eos.entropy import EntropyEOS
from aragog.utilities import FloatOrArray, tanh_weight

logger = logging.getLogger('fwl.' + __name__)


def mobility_function(
    porosity: FloatOrArray, grain_size: float, buoyancy: str = 'solid'
) -> FloatOrArray:
    """Three-regime permeability over porosity, ``F = k(phi) / phi`` [m^2].

    The melt-solid relative velocity of gravitational separation is
    ``(rho_s - rho_l) g F / eta`` with ``g > 0`` (Abe 1995; Bower et al. 2018, section 2.1,
    Eqs. 13a-c). The regimes are blended with tanh weights centred on the
    equal-density-ratio crossings 0.0769452 (Blake-Kozeny-Carman to
    Rumpf-Gupte) and 0.771462 (Rumpf-Gupte to Stokes settling); the blend
    widths 0.02 and 0.05 are numerical smoothing, not physics. Inside the upper
    blend ``F`` exceeds the Stokes value by up to 4.4 percent (at porosity 0.805).

    Parameters
    ----------
    porosity : float or array
        Porosity [1], as :meth:`EntropyPhaseEvaluator.porosity` returns it.
    grain_size : float
        Grain size ``a`` [m]. It enters as ``a**2`` with the Stokes prefactor 2/9
        (Abe 1995, Eq. 39; Bower et al. 2018, Eqs. 12 and 13a), which is Stokes' law
        for a sphere of radius ``a``, so it acts as the grain radius.
    buoyancy : str
        ``'solid'`` (default): the buoyancy against the solid, Abe (1995, Eq. 40).
        ``'mixture'``: against the mixture, McKenzie (1984, Eq. 3.7), which puts a
        factor ``1 - phi`` on the Blake-Kozeny-Carman and Rumpf-Gupte terms;
        the Stokes term keeps the plain contrast, as under ``'solid'`` (the dilute
        limit of a grain settling alone), so it has no factor.

    Returns
    -------
    float or array
        ``F`` [m^2], non-negative: ``d^2 phi^2 / (1000 (1 - phi)^2)`` at low
        porosity, ``(5/7) d^2 phi^4.5`` at intermediate porosity and
        ``(2/9) d^2`` at high porosity, with ``d`` the grain size; under
        ``'mixture'`` the first two carry a factor ``1 - phi``.
    """
    if buoyancy not in SEPARATION_BUOYANCY_MODES:
        raise ValueError(
            f'buoyancy must be one of {SEPARATION_BUOYANCY_MODES}, got {buoyancy!r}'
        )
    d = grain_size
    por = np.maximum(porosity, 1e-20)
    one_m_por = np.maximum(1.0 - porosity, 1e-20)
    F_bkc = d**2 * por**2 / (one_m_por**2 * 1000.0)
    F_rg = d**2 * por**4.5 * (5.0 / 7.0)
    F_stokes = d**2 * 2.0 / 9.0
    # zeta_1 sits 1.7e-5 below the exact BKC-RG crossing (0.02 percent); zeta_2 is exact.
    w_rg = tanh_weight(porosity, 0.0769452, 0.02)
    w_stokes = tanh_weight(porosity, 0.771462, 0.05)
    F_matrix = (1.0 - w_rg) * F_bkc + (w_rg - w_stokes) * F_rg
    if buoyancy == 'mixture':
        F_matrix = (1.0 - porosity) * F_matrix
    return np.maximum(F_matrix + w_stokes * F_stokes, 0.0)


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
    separation_buoyancy : str
        Buoyancy of the separation velocity, passed to :func:`mobility_function`:
        ``'mixture'`` (McKenzie 1984, default) or ``'solid'`` (Abe 1995, SPIDER).
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
        separation_buoyancy: str = SEPARATION_BUOYANCY_DEFAULT,
        rheology: SolidRheologyParams | dict | None = None,
        defer_rheology_viscosity: bool = False,
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
        self._rheology: SolidRheologyParams = merge_rheology(rheology, {})
        check_viscosity_ceiling(self._visc_solid, self._rheology)
        self._defer_rheology_viscosity: bool = bool(defer_rheology_viscosity)
        self._rheology_viscosity_dirty: bool = False
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
        if separation_buoyancy not in SEPARATION_BUOYANCY_MODES:
            raise ValueError(
                f'separation_buoyancy must be one of {SEPARATION_BUOYANCY_MODES}, '
                f'got {separation_buoyancy!r}'
            )
        self._separation_buoyancy = separation_buoyancy

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
        self._solidus_temperature: npt.NDArray = np.array([])
        self._eta_diff: npt.NDArray = np.array([])
        self._tau_y: npt.NDArray = np.array([])
        self._smth_val: npt.NDArray = np.array([])

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
          alpha, k. Matches SPIDER's -use_const_properties. No EOS
          table lookups. When ``rheology.enabled`` is False, ``phi = 1``
          (liquid) with constant viscosity ``10**const_log10visc``; when
          ``rheology.enabled`` is True, ``phi = 0`` (solid) and the
          viscosity is the Arrhenius solid viscosity
          (``const_log10visc`` is not used).
        - const_properties=False (default): single-pass EOS evaluation
          mirroring SPIDER's EOSEval_Composite_TwoPhase.
        """
        if self._const_properties:
            return self._update_const()
        return self._update_eos()

    def _update_const(self) -> None:
        """Constant-properties mode matching SPIDER -use_const_properties.

        T = T_ref * exp((S - S_ref) / Cp). All other properties constant
        except when ``rheology.enabled`` is True, where ``phi = 0`` (solid)
        and viscosity is the Arrhenius solid viscosity (``const_log10visc``
        is not used); when ``rheology.enabled`` is False, ``phi = 1`` with
        constant viscosity ``10**const_log10visc``.
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
        self._thermal_conductivity_val = np.full_like(S, self._const_cond)
        self._latent_heat_val = np.zeros_like(S)
        self._solidus_temperature = None
        self._smth_val = np.zeros_like(S)
        if not self._rheology.enabled:
            self._viscosity_val = np.full_like(S, 10.0**self._const_log10visc)
            self._eta_diff = np.full_like(S, self._visc_solid, dtype=float)
            self._tau_y = np.full_like(S, np.inf)
            self._rheology_viscosity_dirty = False
        else:
            self._melt_fraction = np.zeros_like(S)
            if self._defer_rheology_viscosity:
                self._rheology_viscosity_dirty = True
            else:
                self._compute_rheology_viscosity()

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
                '_smth_val',
                '_eta_diff',
                '_tau_y',
            ):
                val = getattr(self, attr)
                if val is not None:
                    setattr(self, attr, np.asarray(val).ravel())

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

        # Root-cause check: catch entropy outside the EOS table domain
        # here, before any table lookup, instead of only detecting the
        # NaN it produces downstream (see the guard at the end of this
        # method).
        eos._check_entropy_range(
            S_arr, eos.S_min, eos.S_max, 'entropy_phase (composite domain)'
        )

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
        # Cp and alpha at the boundaries follow below: latent from T, rho, S; linear
        # from the boundary Cp and dTdPs tables through the identity

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
            # Boundary alphas from the identity, as the single-phase alpha below
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
            solid_used = ~(gphi > 0.5)
            melt_used = gphi > 0.5
            eos._check_entropy_range(
                np.where(solid_used, S_for_solid, solid_tbl['S'][0]),
                solid_tbl['S'][0],
                solid_tbl['S'][-1],
                f'{prop_name} (single-phase solid table lookup)',
            )
            eos._check_entropy_range(
                np.where(melt_used, S_for_melt, melt_tbl['S'][0]),
                melt_tbl['S'][0],
                melt_tbl['S'][-1],
                f'{prop_name} (single-phase melt table lookup)',
            )
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
        # alpha: thermodynamic identity rho Cp (dT/dP)_S / T, as the JAX evaluator
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
        self._solidus_temperature = np.asarray(T_sol, dtype=float)
        self._smth_val = np.asarray(smth, dtype=float)

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
                '_solidus_temperature',
                '_smth_val',
            ):
                setattr(self, attr, np.asarray(getattr(self, attr)).ravel())

        # Guard: clamp negative alpha (EOS table edges)
        a = self._thermal_expansivity
        eps_a = 1.0e-8
        self._thermal_expansivity = 0.5 * (a + np.sqrt(a * a + eps_a * eps_a))

        # ── Step 6: viscosity (two-stage, reuses cached gphi/smth) ──
        if not self._rheology.enabled:
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
            self._eta_diff = np.full_like(phi_arr, self._visc_solid, dtype=float)
            self._tau_y = np.full_like(phi_arr, np.inf, dtype=float)
            self._rheology_viscosity_dirty = False
        elif self._defer_rheology_viscosity:
            self._rheology_viscosity_dirty = True
        else:
            self._compute_rheology_viscosity()

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

    def blend_viscosity_with_solid(self, eta_solid: FloatOrArray) -> FloatOrArray:
        """Two-stage log10 blend of ``eta_solid`` with ``self._visc_liquid``.

        Parameters
        ----------
        eta_solid : FloatOrArray
            Solid-branch dynamic viscosity [Pa s], positive scalar or array in
            ``(0, +inf)``.

        Returns
        -------
        FloatOrArray
            Blended dynamic viscosity [Pa s] across the mushy and single-phase
            zones, positive in ``(0, +inf)``.
        """
        phi_arr = np.atleast_1d(np.asarray(self._melt_fraction, dtype=float))
        smth = np.atleast_1d(np.asarray(self._smth_val, dtype=float))
        w = tanh_weight(phi_arr, self._phi_rheo, self._phi_width)
        log_visc_solid = np.log10(
            np.maximum(np.atleast_1d(np.asarray(eta_solid, dtype=float)), 1.0e-30)
        )
        log_visc_liq = np.log10(self._visc_liquid)
        log_visc_mixed = (1.0 - w) * log_visc_solid + w * log_visc_liq
        phi_thresh = self._rheology.phi_visc_single if self._rheology.enabled else 0.5
        log_visc_single = np.where(phi_arr > phi_thresh, log_visc_liq, log_visc_solid)
        log_visc = smth * log_visc_mixed + (1.0 - smth) * log_visc_single
        is_scalar = (
            np.ndim(self._melt_fraction) == 0 and np.ndim(eta_solid) == 0 and log_visc.size == 1
        )
        return 10.0 ** (log_visc.item() if is_scalar else log_visc)

    def _compute_rheology_viscosity(self) -> None:
        """Evaluate Arrhenius diffusion creep viscosity, Byerlee yield stress, and unyielded blend."""
        T_arr = np.atleast_1d(np.asarray(self._temperature, dtype=float))
        P_raw = (
            np.atleast_1d(np.asarray(self.pressure, dtype=float))
            if self.pressure.size > 0
            else np.zeros_like(T_arr)
        )
        P_arr = np.maximum(P_raw, 0.0)
        T_sol_arr = (
            np.atleast_1d(np.asarray(self._solidus_temperature, dtype=float))
            if self._solidus_temperature is not None and np.size(self._solidus_temperature) > 0
            else None
        )
        rheo = self._rheology
        self._eta_diff = _compute_arrhenius_viscosity_core(
            temperature=T_arr,
            pressure=P_arr,
            viscosity_solid=self._visc_solid,
            g_homologous=rheo.g_homologous,
            solidus_temperature=T_sol_arr,
            t_sol_0=rheo.t_sol_0,
            dt_sol_dp=rheo.dt_sol_dp,
            arrhenius_t_ref=rheo.arrhenius_t_ref,
            viscosity_max_log10=rheo.viscosity_max_log10,
            water_prefactor=rheo.water_prefactor,
            xp=np,
            activation_energy=rheo.activation_energy,
            activation_volume=rheo.activation_volume,
            activation_volume_decay_pressure=rheo.activation_volume_decay_pressure,
        )
        self._tau_y = _compute_yield_stress_core(
            pressure=P_arr,
            yield_stress_c=rheo.yield_stress_c,
            yield_stress_mu=rheo.yield_stress_mu,
            yield_stress_max=rheo.yield_stress_max,
            xp=np,
        )
        self._viscosity_val = self.blend_viscosity_with_solid(self._eta_diff)
        self._rheology_viscosity_dirty = False

    def _ensure_rheology_viscosity(self) -> None:
        if self._rheology_viscosity_dirty:
            self._compute_rheology_viscosity()

    # ── Property accessors (PhaseEvaluatorProtocol) ──────────────────

    def density(self) -> FloatOrArray:
        return self._density

    def temperature(self) -> FloatOrArray:
        """Temperature from EOS lookup (not a state variable)."""
        return self._temperature

    def solidus_temperature(self) -> npt.NDArray | None:
        """Return cached solidus temperature ``T_sol(P)`` from the current EOS update.

        Returns
        -------
        npt.NDArray or None
            Solidus temperature [K] in ``(0, +inf)`` at each node, or ``None``
            when no solidus curve is available.
        """
        return self._solidus_temperature

    def visc_solid_weight(self) -> npt.NDArray:
        """Return two-stage solid log-viscosity blend weight ``w_solid(phi, smth)``.

        Returns
        -------
        npt.NDArray
            Dimensionless solid log-viscosity weight in ``[0, 1]`` at each node.
        """
        phi_arr = np.atleast_1d(np.asarray(self._melt_fraction, dtype=float))
        smth = np.atleast_1d(np.asarray(self._smth_val, dtype=float))
        w = tanh_weight(phi_arr, self._phi_rheo, self._phi_width)
        phi_thresh = self._rheology.phi_visc_single if self._rheology.enabled else 0.5
        return smth * (1.0 - w) + (1.0 - smth) * np.where(phi_arr > phi_thresh, 0.0, 1.0)

    def eta_diff(self) -> npt.NDArray:
        """Return cached unyielded solid diffusion-creep viscosity ``eta_diff``.

        Returns
        -------
        npt.NDArray
            Unyielded diffusion-creep dynamic viscosity [Pa s] in
            ``(0, 10**viscosity_max_log10]`` at each node.
        """
        self._ensure_rheology_viscosity()
        return self._eta_diff

    def tau_y(self) -> npt.NDArray:
        """Return cached Byerlee yield stress ``tau_y(P)``.

        Returns
        -------
        npt.NDArray
            Byerlee plastic yield stress [Pa] in ``[0, yield_stress_max]`` at
            each node.
        """
        self._ensure_rheology_viscosity()
        return self._tau_y

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
        self._ensure_rheology_viscosity()
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
        """Thermal expansivity [1/K] at each node, from the last :meth:`update`.

        The blend of the two-phase value and the single-phase identity
        ``rho cp (dT/dP)_S / T``, mapped by ``0.5 (a + sqrt(a^2 + 1e-16))`` so that it is
        positive: a negative blend gives a value near 0.
        """
        return self._thermal_expansivity

    def viscosity(self) -> FloatOrArray:
        self._ensure_rheology_viscosity()
        return self._viscosity_val

    def relative_velocity(self) -> FloatOrArray:
        """Melt-solid relative velocity for gravitational separation [m/s].

        Returns zero in const_properties mode (no phase contrast).

        Uses the Abe (1995) gravitational separation velocity with the
        three-regime permeability of Bower et al. (2018) section 2.1, based
        on the density-derived porosity (see :meth:`porosity`), matching
        SPIDER's GetGravitationalHeatFlux in energy.c.

        Regimes (F = K/porosity, the quantity multiplying (rho_s - rho_l)*g/eta);
        under ``separation_buoyancy = 'mixture'`` regimes 1 and 2 carry a factor
        1 - por (see :func:`mobility_function`):
        1. Blake-Kozeny-Carman (low porosity): F = d^2 por^2 / ((1-por)^2 * 1000)
        2. Rumpf-Gupte (intermediate): F = d^2 por^4.5 * (5/7)
        3. Stokes settling (high porosity): F = d^2 * 2/9

        The drag viscosity is set by ``separation_viscosity``: ``'melt'``
        uses the fixed single-phase liquid viscosity, matching SPIDER's
        ``GetGravitationalHeatFlux``; ``'mixture'`` uses the
        rheological-transition-blended bulk viscosity, so settling locks
        up through the same phi_rheo transition that sets the bulk
        rheology.

        The porosity denominator ``rho_s - rho_l`` is floored at about 1 kg/m^3,
        which acts where the tabulated densities cross (the melt locally denser
        or lighter by less than 1 kg/m^3). There the porosity is
        ``(rho_s - rho) / (1 kg/m^3)`` clipped to [0, 1]: about 0 where the node
        density is at or above ``rho_s``, 1 where it is more than 1 kg/m^3
        below, and intermediate in between.

        The contrast is signed, ``v_rel = (rho_s - rho_l) g F / eta`` with ``g`` the
        positive gravity magnitude (Bower et al. 2018, Eq. 12, and SPIDER, which
        write ``(rho_l - rho_s)`` with a negative ``g``): positive (melt rising) where
        the melt is lighter, negative (melt sinking) where it is denser.
        """
        if self._const_properties:
            return np.zeros_like(self._density)
        rho_s, rho_l = self.phase_boundary_densities()
        g = self._g
        # Explicit dispatch: a third mode added to SEPARATION_VISCOSITY_MODES
        # must fail here rather than silently fall back to 'melt'.
        if self._separation_viscosity == 'mixture':
            eta_l = self.viscosity()
        elif self._separation_viscosity == 'melt':
            eta_l = self._visc_liquid
        else:
            raise ValueError(f'unhandled separation_viscosity {self._separation_viscosity!r}')

        porosity = self._porosity_from(rho_s, rho_l)
        F = mobility_function(porosity, self._grain_size, self._separation_buoyancy)

        # Positive outward: melt rising, solid sinking.
        return (rho_s - rho_l) * g * F / np.maximum(eta_l, 1e-10)

    def phase_boundary_densities(self) -> tuple[FloatOrArray, FloatOrArray]:
        """Solid and melt densities at the phase boundary at the node pressure [kg/m^3].

        Under ``const_properties`` there is no phase contrast and both equal
        the node density.
        """
        if self._const_properties:
            return self._density.copy(), self._density.copy()
        rho_s = self._eos._lookup_at_phase_boundary('density', self.pressure, 'solid')
        rho_l = self._eos._lookup_at_phase_boundary('density', self.pressure, 'melt')
        return rho_s, rho_l

    def porosity(self) -> FloatOrArray:
        """Density-derived porosity ``(rho_s - rho) / (rho_s - rho_l)`` [1].

        This is the porosity :meth:`relative_velocity` uses, except under
        ``const_properties``, where there is no phase contrast, the mantle counts as
        fully liquid so this is one everywhere, and :meth:`relative_velocity` is zero.
        It is soft-clipped to [0, 1] with a width of 1e-3: its range is [-2.5e-7, 1]
        and it is 4.9975e-4 where the node density equals ``rho_s``. It is not the
        melt fraction: a solid node whose density is below ``rho_s`` can give any
        value, and where ``rho_s <= rho_l`` a mush is denser than ``rho_s``, so this
        is about 0 although the node holds melt. Do not use it for melt drainage
        where ``rho_s <= rho_l``: test the sign of the contrast first.
        """
        if self._const_properties:
            return (
                np.zeros_like(self._density)
                if self._rheology.enabled
                else np.ones_like(self._density)
            )
        return self._porosity_from(*self.phase_boundary_densities())

    def _porosity_from(self, rho_s: FloatOrArray, rho_l: FloatOrArray) -> FloatOrArray:
        """Density-derived porosity, smoothed so the CVODE predictor sees a C-infinity RHS."""
        rho = self._density
        drho = rho_s - rho_l
        eps = 1.0e-3  # kg/m^3; far below any physical density contrast
        drho_smoothmax = 0.5 * (drho + 1.0 + np.sqrt((drho - 1.0) ** 2 + eps * eps))
        porosity_raw = (rho_s - rho) / drho_smoothmax
        # smooth_clip(porosity_raw, 0, 1) via two soft-max operations
        eps_p = 1.0e-3  # dimensionless; invisible except near [0,1] edges
        p_lo = 0.5 * (porosity_raw + np.sqrt(porosity_raw * porosity_raw + eps_p * eps_p))
        hi_u = 1.0 - p_lo
        return 1.0 - 0.5 * (hi_u + np.sqrt(hi_u * hi_u + eps_p * eps_p))

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
