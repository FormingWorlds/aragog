"""Entropy-based state class for the entropy formulation solver.

Replaces State (temperature-based) with entropy as the prognostic variable.
All MLT computations use dS/dr instead of the superadiabatic T gradient.
The c_p spike at phase boundaries is eliminated because entropy changes
monotonically through the mushy zone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.utilities import FloatOrArray

if TYPE_CHECKING:
    from aragog.solver.evaluator import Evaluator

logger = logging.getLogger(__name__)

# Critical Reynolds number from Abe (1993)
RE_CRIT = 9.0 / 8.0


class EntropyState:
    """Stores and updates the thermodynamic state using entropy.

    The key difference from State: the prognostic variable is S(r,t),
    not T(r,t). All properties are looked up from (P, S) via the
    EntropyPhaseEvaluator. Convective transport is driven by dS/dr
    (entropy gradient), not by the superadiabatic T gradient.

    Parameters
    ----------
    evaluator : Evaluator
        Contains mesh and boundary conditions.
    phase : EntropyPhaseEvaluator
        Entropy-based phase evaluator with (P,S) lookups.
    settings : dict
        Energy settings (conduction, convection, mixing, etc.)
    """

    def __init__(
        self,
        evaluator: Evaluator,
        phase_staggered: EntropyPhaseEvaluator,
        phase_basic: EntropyPhaseEvaluator,
        conduction: bool = True,
        convection: bool = True,
        gravitational_separation: bool = False,
        mixing: bool = False,
        eddy_diffusivity_thermal: float = 1.0,
        eddy_diffusivity_chemical: float = 1.0,
        kappah_floor: float = 0.0,
    ):
        self._evaluator = evaluator
        self.phase_staggered = phase_staggered
        self.phase_basic = phase_basic
        self._conduction = conduction
        self._convection = convection
        self._grav_sep = gravitational_separation
        self._mixing = mixing
        self._eddy_diff_thermal = eddy_diffusivity_thermal
        self._eddy_diff_chem = eddy_diffusivity_chemical
        self._kappah_floor = kappah_floor

        mesh = evaluator.mesh
        n_basic = mesh.basic.radii.shape[0]
        n_staggered = mesh.staggered.radii.shape[0]

        # Allocate arrays. Arrays are 2D (N, 1) when used with vectorized BDF
        # solver, which passes state as (N, K) for K simultaneous evaluations.
        # BoundaryConditions indexes as heat_flux[-1, :] and heat_flux[0, :].
        self._entropy_staggered = np.zeros(n_staggered)
        self._entropy_basic = np.zeros(n_basic)
        self._dSdr = np.zeros(n_basic)
        self._dphidr = np.zeros(n_basic)
        self._eddy_diffusivity = np.zeros(n_basic)
        self._heat_flux = np.zeros(n_basic)
        self._mass_flux = np.zeros(n_basic)
        self._is_convective = np.zeros(n_basic, dtype=bool)

    def update(self, entropy: npt.NDArray, time: FloatOrArray) -> None:
        """Update the state from the entropy profile.

        Parameters
        ----------
        entropy : array
            Entropy at staggered nodes [J/kg/K].
        time : float
            Current time [yr].
        """
        mesh = self._evaluator.mesh

        # Store entropy and compute at basic nodes.
        # Mesh transforms use (N, 1) column vectors; ensure correct shape.
        S_col = np.asarray(entropy).reshape(-1, 1) if np.asarray(entropy).ndim == 1 else entropy
        self._entropy_staggered = np.asarray(entropy).flatten()
        self._entropy_basic = np.asarray(mesh.quantity_at_basic_nodes(S_col)).flatten()
        self._dSdr = np.asarray(mesh.d_dr_at_basic_nodes(S_col)).flatten()

        # Update phase evaluators with current (P, S)
        self.phase_staggered.set_entropy(entropy)
        self.phase_staggered.update()
        self.phase_basic.set_entropy(self._entropy_basic)
        self.phase_basic.update()

        # Melt fraction gradient for gravitational separation
        phi = np.asarray(self.phase_staggered.melt_fraction()).reshape(-1, 1)
        self._dphidr = np.asarray(mesh.d_dr_at_basic_nodes(phi)).flatten()

        # ── MLT from entropy gradient ────────────────────────────────
        # Convection is unstable when dS/dr < 0 (entropy decreasing outward).
        # This replaces the T-based (dT/dr - dT/dr_S) criterion.
        self._is_convective = self._dSdr < 0

        # Buoyancy: convert entropy gradient to effective thermal buoyancy
        # |superadiabatic| = alpha * T * |dS/dr| / Cp
        # All arrays must be 1D (mesh gives (N,1) column vectors)
        alpha = np.asarray(self.phase_basic.thermal_expansivity()).flatten()
        T = np.asarray(self.phase_basic.temperature()).flatten()
        Cp = np.asarray(self.phase_basic.heat_capacity()).flatten()
        g = np.asarray(self.phase_basic.gravitational_acceleration()).flatten()

        effective_superadiabatic = alpha * T * np.abs(self._dSdr) / np.maximum(Cp, 1.0)
        velocity_prefactor = g * effective_superadiabatic

        # Viscous velocity (Re <= Re_crit)
        mixing_length = np.asarray(mesh.basic.mixing_length).flatten()
        mixing_length_cubed = np.asarray(mesh.basic.mixing_length_cubed).flatten()
        mixing_length_squared = np.asarray(mesh.basic.mixing_length_squared).flatten()
        nu = np.asarray(self.phase_basic.kinematic_viscosity()).flatten()

        viscous_velocity = velocity_prefactor * mixing_length_cubed / (18.0 * nu)
        viscous_velocity[~self._is_convective] = 0.0

        # Inviscid velocity (Re > Re_crit)
        inviscid_velocity_sq = velocity_prefactor * mixing_length_squared / 16.0
        inviscid_velocity_sq[~self._is_convective] = 0.0
        inviscid_velocity = np.sqrt(np.maximum(inviscid_velocity_sq, 0.0))

        # Reynolds number
        reynolds = viscous_velocity * mixing_length / nu

        # Smooth blend between regimes (tanh transition at Re_crit)
        blend_width = 0.2 * RE_CRIT
        inviscid_weight = 0.5 * (1.0 + np.tanh(
            (reynolds - RE_CRIT) / max(blend_width, 1e-30)
        ))
        # Raw eddy diffusivity (before thermal scaling and floor)
        kh_raw = (
            (1.0 - inviscid_weight) * viscous_velocity
            + inviscid_weight * inviscid_velocity
        ) * mixing_length

        # Apply eddy_diffusivity_thermal scaling (SPIDER convention:
        # positive = scale factor, negative = fixed constant)
        if self._eddy_diff_thermal > 0:
            self._eddy_diffusivity = self._eddy_diff_thermal * kh_raw
        else:
            self._eddy_diffusivity = np.full_like(kh_raw, -self._eddy_diff_thermal)

        # Chemical eddy diffusivity uses raw kh (before floor and thermal scaling),
        # matching SPIDER's matprop.c lines 318-325
        if self._eddy_diff_chem > 0:
            self._kappac = self._eddy_diff_chem * kh_raw
        else:
            self._kappac = np.full_like(kh_raw, -self._eddy_diff_chem)

        # kappa_h floor (phase-dependent, modulated by melt fraction)
        if self._kappah_floor > 0.0:
            phi_basic = np.asarray(self.phase_basic.melt_fraction()).flatten()
            from aragog.utilities import tanh_weight
            f_floor = tanh_weight(phi_basic, 0.4, 0.15)
            kh_floor = self._kappah_floor * f_floor
            self._eddy_diffusivity = np.maximum(self._eddy_diffusivity, kh_floor)

        # ── Compute fluxes ───────────────────────────────────────────
        rho = np.asarray(self.phase_basic.density()).flatten()
        k = np.asarray(self.phase_basic.thermal_conductivity()).flatten()
        dTdPs = np.asarray(self.phase_basic.dTdPs()).flatten()

        self._heat_flux = np.zeros_like(self._entropy_basic)
        self._mass_flux = np.zeros_like(self._entropy_basic)

        if self._conduction:
            # F_cond = k * (dT/dS)_P * (-dS/dr) = k * dTdPs * (-dS/dr)
            # But dTdPs from SPIDER is dT/dP|_S, not dT/dS|_P.
            # We need (dT/dS)_P. For a regular grid:
            # (dT/dS)_P = T / (rho * Cp * dTdPs_val) ... no, let's use:
            # (dT/dr) = (dT/dS)_P * (dS/dr) + (dT/dP)_S * (dP/dr)
            # For conduction, F = -k * dT/dr. In S coordinates:
            # dT/dr = (dT/dS)_P * dS/dr  (at constant P along radial shells)
            # This is not quite right because P also varies with r.
            # The full derivative: dT/dr = (∂T/∂S)|_P * dS/dr + (∂T/∂P)|_S * dP/dr
            # SPIDER handles this by computing the total energy flux at cell edges.
            # For simplicity, use the approximation:
            # F_cond ≈ -k * (∂T/∂r) where ∂T/∂r is computed from T(P,S) profile
            T_stag = np.asarray(self.phase_staggered.temperature()).reshape(-1, 1)
            dTdr = np.asarray(mesh.d_dr_at_basic_nodes(T_stag)).flatten()
            self._heat_flux += -k * dTdr

        if self._convection:
            # F_conv = rho * T * kappa_h * (-dS/dr)
            # This is the entropy flux: positive when dS/dr < 0 (unstable)
            self._heat_flux += rho * T * self._eddy_diffusivity * (-self._dSdr)

        if self._grav_sep:
            # SPIDER formula: Jgrav = rho * phi * (1-phi) * v_rel
            # The heat contribution is Jgrav * L (latent heat)
            phi_b = np.asarray(self.phase_basic.melt_fraction()).flatten()
            v_rel = np.asarray(self.phase_basic.relative_velocity()).flatten()
            self._mass_flux += rho * phi_b * (1.0 - phi_b) * v_rel

        if self._mixing:
            # SPIDER uses kappac (from raw kh, not from floored kappah)
            self._mass_flux += rho * self._kappac * (-self._dphidr)

        self._heat_flux += self._mass_flux * self.phase_basic.latent_heat()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def entropy_staggered(self) -> npt.NDArray:
        return self._entropy_staggered

    @property
    def entropy_basic(self) -> npt.NDArray:
        return self._entropy_basic

    @property
    def heat_flux(self) -> npt.NDArray:
        return self._heat_flux

    @property
    def eddy_diffusivity(self) -> npt.NDArray:
        return self._eddy_diffusivity

    @property
    def is_convective(self) -> npt.NDArray:
        return self._is_convective

    @property
    def dSdr(self) -> npt.NDArray:
        return self._dSdr

    def capacitance_staggered(self) -> npt.NDArray:
        """Capacitance for entropy equation: rho * T [kg K / m^3].

        The entropy equation is: rho * T * dS/dt = -div(F) + sources.
        Compare T-formulation: rho * Cp * dT/dt = -div(F) + sources.
        """
        return self.phase_staggered.capacitance()

    # ── Compatibility with BoundaryConditions (expects State interface) ──

    @property
    def temperature_basic(self) -> npt.NDArray:
        """Temperature at basic nodes (derived from S via EOS)."""
        return self.phase_basic.temperature()

    @property
    def top_temperature(self) -> npt.NDArray:
        """Temperature at the outermost basic node [K]."""
        T_basic = self.phase_basic.temperature()
        return T_basic[-1:]  # keep as array for BC compatibility

    @property
    def bottom_temperature(self) -> npt.NDArray:
        """Temperature at the innermost basic node [K]."""
        T_basic = self.phase_basic.temperature()
        return T_basic[:1]

    def dTdr(self) -> npt.NDArray:
        """Temperature gradient at basic nodes (from T profile, for BCs)."""
        T_stag = self.phase_staggered.temperature()
        return self._evaluator.mesh.d_dr_at_basic_nodes(T_stag)
