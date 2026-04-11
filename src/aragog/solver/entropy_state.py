"""Entropy-based state for the Aragog interior dynamics solver.

Stores the thermodynamic state as S(r,t) and computes heat fluxes
(conduction, convection, gravitational separation, mixing) from the
entropy gradient dS/dr. Convective instability is determined by
dS/dr < 0, with eddy diffusivity from Abe (1993) viscous/inviscid
MLT. No c_p spike at phase boundaries (entropy is monotonic through
the mushy zone).
"""

from __future__ import annotations

import logging
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
        radionuclides: bool = False,
        tidal: bool = False,
        tidal_array: list | None = None,
        eddy_diffusivity_thermal: float = 1.0,
        eddy_diffusivity_chemical: float = 1.0,
        kappah_floor: float = 0.0,
        bottom_up_grav_sep: bool = True,
    ):
        self._evaluator = evaluator
        self.phase_staggered = phase_staggered
        self.phase_basic = phase_basic
        self._conduction = conduction
        self._convection = convection
        self._grav_sep = gravitational_separation
        self._mixing = mixing
        self._radionuclides = radionuclides
        self._tidal = tidal
        self._tidal_array = tidal_array if tidal_array is not None and len(tidal_array) > 0 else [0.0]
        self._eddy_diff_thermal = eddy_diffusivity_thermal
        self._eddy_diff_chem = eddy_diffusivity_chemical
        self._kappah_floor = kappah_floor
        self._bottom_up_grav_sep = bool(bottom_up_grav_sep)

        mesh = evaluator.mesh
        n_basic = mesh.basic.radii.size
        n_staggered = mesh.staggered.radii.size

        # Cache flattened (1D) mesh arrays to avoid repeated
        # np.asarray(...).flatten() calls in the hot path.
        # The mesh stores (N,1) column vectors; we flatten once here.
        self._mixing_length = np.asarray(mesh.basic.mixing_length).ravel()
        self._mixing_length_sq = np.asarray(mesh.basic.mixing_length_squared).ravel()
        self._mixing_length_cu = np.asarray(mesh.basic.mixing_length_cubed).ravel()

        # Allocate state arrays (all 1D).
        self._entropy_staggered = np.zeros(n_staggered)
        self._entropy_basic = np.zeros(n_basic)
        self._dSdr = np.zeros(n_basic)
        self._dphidr = np.zeros(n_basic)
        self._eddy_diffusivity = np.zeros(n_basic)
        self._heat_flux = np.zeros(n_basic)
        self._mass_flux = np.zeros(n_basic)
        self._is_convective = np.zeros(n_basic, dtype=bool)

    def update(
        self,
        entropy: npt.NDArray,
        time: FloatOrArray,
        dSdr_cmb: float | None = None,
        dSdr: npt.NDArray | None = None,
        entropy_basic: npt.NDArray | None = None,
    ) -> None:
        """Update the state from the entropy profile.

        Parameters
        ----------
        entropy : array
            Entropy at staggered nodes [J/kg/K].
        time : float
            Current time [yr].
        dSdr_cmb : float, optional
            Path A (energy_balance): override the CMB boundary gradient.
        dSdr : array, optional
            Gradient-mode: provide dS/dr at all basic nodes directly,
            bypassing the FD transform. Shape (N+1,).
        entropy_basic : array, optional
            Gradient-mode: provide S at all basic nodes directly,
            bypassing the quantity transform. Shape (N+1,).
        """
        mesh = self._evaluator.mesh

        S = np.asarray(entropy).ravel()
        self._entropy_staggered = S
        if entropy_basic is not None:
            self._entropy_basic = np.asarray(entropy_basic).ravel()
        else:
            self._entropy_basic = mesh.quantity_at_basic_nodes(S).ravel()
        if dSdr is not None:
            self._dSdr = np.asarray(dSdr).ravel()
        else:
            self._dSdr = mesh.d_dr_at_basic_nodes(S).ravel()

        # Path A: override the boundary entropy gradient with the
        # state-vector value. This must happen BEFORE the phase_basic
        # update so the bottom basic node uses the boundary entropy.
        if dSdr_cmb is not None:
            r_basic = np.asarray(mesh.basic.radii).ravel()
            r_stag_0 = 0.5 * (r_basic[0] + r_basic[1])
            dr_offset = r_basic[0] - r_stag_0
            self._dSdr[0] = float(dSdr_cmb)
            self._entropy_basic[0] = float(S[0]) + float(dSdr_cmb) * dr_offset

        # Update phase evaluators with current (P, S)
        self.phase_staggered.set_entropy(S)
        self.phase_staggered.update()
        self.phase_basic.set_entropy(self._entropy_basic)
        self.phase_basic.update()

        # Melt-fraction gradient for gravitational separation and mixing.
        #
        # The mixing flux (rho * kappac * (-dphi/dr) * L) must cancel
        # the convective flux (rho * T * kappah * (-dS/dr)) at every
        # node when kappac = kappah. This requires dphi/dr = dS/dr /
        # (S_liq - S_sol), which holds only when phi is the UN-TRUNCATED
        # lever-rule fraction gphi = (S - S_sol) / (S_liq - S_sol).
        # The clamped melt fraction (phi in [0,1]) truncates gphi to 1
        # in pure-liquid cells and to 0 in pure-solid cells, breaking
        # the cancellation at the crystallisation front: the clamped
        # gradient is ~50% of the un-truncated gradient when one cell
        # is liquid and the adjacent cell is mushy. The resulting
        # uncancelled convective flux drives a positive feedback that
        # solidifies the CMB cell 30x faster than SPIDER.
        P_stag = np.asarray(self.phase_staggered.pressure).ravel()
        eos = self.phase_staggered._eos
        S_sol = np.asarray(eos.solidus_entropy(P_stag)).ravel()
        S_liq = np.asarray(eos.liquidus_entropy(P_stag)).ravel()
        dS_phase = np.maximum(S_liq - S_sol, 1.0)
        gphi = (S - S_sol) / dS_phase
        self._dphidr = mesh.d_dr_at_basic_nodes(gphi).ravel()

        # ── MLT from entropy gradient ────────────────────────────────
        # Convection is unstable when dS/dr < 0 (entropy decreasing outward).
        self._is_convective = self._dSdr < 0

        # Buoyancy: convert entropy gradient to effective thermal buoyancy
        # |superadiabatic| = alpha * T * |dS/dr| / Cp
        # Phase evaluator properties are already 1D arrays.
        alpha = np.asarray(self.phase_basic.thermal_expansivity()).ravel()
        T = np.asarray(self.phase_basic.temperature()).ravel()
        Cp = np.asarray(self.phase_basic.heat_capacity()).ravel()
        g = np.asarray(self.phase_basic.gravitational_acceleration()).ravel()

        effective_superadiabatic = alpha * T * np.abs(self._dSdr) / np.maximum(Cp, 1.0)
        velocity_prefactor = g * effective_superadiabatic

        # Viscous velocity (Re <= Re_crit)
        mixing_length = self._mixing_length
        mixing_length_cubed = self._mixing_length_cu
        mixing_length_squared = self._mixing_length_sq
        nu = np.asarray(self.phase_basic.kinematic_viscosity()).ravel()

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

        # Mirror SPIDER energy.c:220-223: at the CMB basic node, use
        # the eddy diffusivity from one node above rather than the
        # boundary-extrapolated value. SPIDER does this because kappah
        # is a nonlinear function of the entropy gradient, and the
        # boundary extrapolation can over- or under-estimate it relative
        # to the interior value. Borrowing from the first interior node
        # avoids this artifact and aligns the CMB convective flux with
        # SPIDER's treatment.
        if len(self._eddy_diffusivity) >= 2:
            self._eddy_diffusivity[0] = self._eddy_diffusivity[1]

        # ── Compute fluxes ───────────────────────────────────────────
        rho = np.asarray(self.phase_basic.density()).ravel()
        k = np.asarray(self.phase_basic.thermal_conductivity()).ravel()

        self._heat_flux = np.zeros_like(self._entropy_basic)
        self._mass_flux = np.zeros_like(self._entropy_basic)

        if self._conduction:
            # SPIDER-parity conductive flux (energy.c:358-378):
            #   F_cond = -k * [(T/Cp) * dS/dr + dT/dr|_adiabat]
            #
            # The total temperature gradient decomposes into a
            # superadiabatic part (proportional to the entropy gradient)
            # and an adiabatic part (from the EOS at each node). This
            # decomposition avoids the numerical artifact that arises
            # when finite-differencing T(P,S) across staggered nodes
            # at different phases in the mushy zone: the phase-blend
            # in T(P,S) introduces a spurious gradient that is not a
            # physical conductive flux.
            #
            # Reuses T, Cp, alpha, g already fetched above (lines
            # 165-168) from phase_basic, which includes the dSdr_cmb
            # boundary override when active.
            Cp_safe = np.maximum(Cp, 100.0)  # silicate Cp floor
            superadiabatic = (T / Cp_safe) * self._dSdr
            dTdrs_ad = -g * alpha * T / Cp_safe
            self._heat_flux += -k * (superadiabatic + dTdrs_ad)

        if self._convection:
            # F_conv = rho * T * kappa_h * (-dS/dr)
            # This is the entropy flux: positive when dS/dr < 0 (unstable)
            self._heat_flux += rho * T * self._eddy_diffusivity * (-self._dSdr)

        if self._grav_sep:
            phi_b = np.asarray(self.phase_basic.melt_fraction()).ravel()
            v_rel = np.asarray(self.phase_basic.relative_velocity()).ravel()
            jgrav = rho * phi_b * (1.0 - phi_b) * v_rel

            # SPIDER-analogue phase-boundary smoothing.
            #
            # Purpose: at the first crystallisation step the raw mass
            # flux rho * phi * (1-phi) * v_rel drains the CMB cell's
            # entropy off the PALEOS P-S table in one coupling step,
            # because the Stokes-regime permeability at
            # grain_size = 0.1 m gives v_rel of several m/s and
            # phi * (1-phi) ~ 5e-3 at phi = 0.995 is not enough
            # damping. See memory/aragog_jgrav_cmb_drain.md.
            #
            # SPIDER avoids this via
            # `smth = get_smoothing(matprop_smooth_width, gphi)` where
            # gphi is the UN-truncated two-phase fraction
            #     gphi = (S - S_sol(P)) / (S_liq(P) - S_sol(P))
            # at the STAGGERED cell immediately BELOW the interface
            # (JGRAV_BOTTOM_UP, SPIDER/energy.c:523-533). gphi exceeds
            # 1 for pure liquid and goes negative for pure solid, so
            # `smth` drops cleanly to 0 on both sides, killing Jgrav
            # at any interface whose lower neighbour is in a pure
            # phase. Aragog's bookkeeping phi is clamped to [0,1] by
            # the EOS lookup so we recompute gphi here from the
            # staggered-cell entropy against the solidus/liquidus
            # entropies.
            #
            # Instead of SPIDER's tanh smoothing, we use a clipped
            # cubic Hermite `16 * gphi^2 * (1-gphi)^2`. Three reasons:
            #   1. Identical zero behaviour at pure phases (gphi = 0
            #      or gphi = 1 -> smth = 0), which is all that's
            #      needed to prevent the drain.
            #   2. Peaks at smth = 1 at gphi = 0.5, with maximum
            #      derivative |smth'| = 2 (vs ~25 for the tanh at
            #      matprop_smooth_width = 0.01). Bounded derivatives
            #      everywhere are much gentler on scipy BDF's
            #      finite-difference Jacobian approximation, which
            #      lets PROTEUS take its full adaptive coupling
            #      timestep through the mushy zone instead of
            #      grinding.
            #   3. Parameter-free: no matprop_smooth_width knob to
            #      tune. SPIDER keeps that knob for its own solver.
            if self._bottom_up_grav_sep:
                P_stag_arr = np.asarray(self.phase_staggered.pressure).ravel()
                eos_stag = self.phase_staggered._eos
                S_sol_s = np.asarray(
                    eos_stag.solidus_entropy(P_stag_arr)
                ).ravel()
                S_liq_s = np.asarray(
                    eos_stag.liquidus_entropy(P_stag_arr)
                ).ravel()
                dS_s = np.maximum(S_liq_s - S_sol_s, 1.0)
                gphi_stag = (self._entropy_staggered - S_sol_s) / dS_s

                # Clipped cubic Hermite: 16 * gphi^2 * (1-gphi)^2
                # on [0,1], zero outside. smth(0)=0, smth(0.5)=1,
                # smth(1)=0, continuous first derivative everywhere.
                gphi_clip = np.clip(gphi_stag, 0.0, 1.0)
                smth_stag = 16.0 * gphi_clip**2 * (1.0 - gphi_clip) ** 2

                # Bottom-up: basic node i (interface between staggered
                # i-1 and i) sees the smoothing of staggered i-1 (the
                # cell BELOW).
                smth_basic = np.ones_like(jgrav)
                smth_basic[1:-1] = smth_stag[:-1]
                jgrav = jgrav * smth_basic

            self._mass_flux += jgrav

        if self._mixing:
            self._mass_flux += rho * self._kappac * (-self._dphidr)

        # Zero mass fluxes at boundaries (SPIDER convention: no mass
        # transfer across CMB or surface, energy.c lines 282-285, 423-426)
        self._mass_flux[0] = 0.0
        self._mass_flux[-1] = 0.0
        self._heat_flux += self._mass_flux * self.phase_basic.latent_heat()

        # ── Internal heating (power per unit mass [W/kg]) ────────────
        n_stag = len(self._entropy_staggered)
        self._heating = np.zeros(n_stag)

        if self._radionuclides and hasattr(self._evaluator, 'radionuclides'):
            radio = 0.0
            for r in self._evaluator.radionuclides:
                radio += r.get_heating(time)
            self._heating += radio

        if self._tidal:
            if len(self._tidal_array) == 1:
                self._heating += self._tidal_array[0]
            elif len(self._tidal_array) == n_stag:
                self._heating += np.array(self._tidal_array)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def heating(self) -> npt.NDArray:
        """Total internal heating [W/kg] at staggered nodes."""
        return self._heating

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
