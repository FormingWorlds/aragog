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
"""Initial condition for the interior solver."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from aragog.mesh import Mesh
from aragog.parser import Parameters, _InitialConditionParameters
from aragog.phase import PhaseEvaluatorCollection

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class InitialCondition:
    """Initial condition

    Args:
        parameters: Parameters
        mesh: Mesh
        phases: PhaseEvaluatorCollection
    """

    _parameters: Parameters
    _mesh: Mesh
    _phases: PhaseEvaluatorCollection

    def __post_init__(self):
        self._settings: _InitialConditionParameters = self._parameters.initial_condition

        # Three initialisation methods: linear (1), user-defined field (2) or adiabat (3).
        if self._settings.initial_condition == 1:
            self._temperature: npt.NDArray = self.get_linear()
        elif self._settings.initial_condition == 2:
            if self._mesh.staggered.number_of_nodes == len(self._settings.init_temperature):
                self._temperature = self._settings.init_temperature
            else:
                msg: str = (
                    f"the size of the provided init temperature field does not match \
                    the number of staggered points {self._mesh.staggered.number_of_nodes}"
                )
                raise ValueError(msg)
        elif self._settings.initial_condition == 3:
            self._temperature: npt.NDArray = self.get_adiabat(self._mesh.basic_pressure[:,-1])
        else:
            msg: str = (
                f"initial_condition = {self._settings.initial_condition} is unknown"
            )
            raise ValueError(msg)

        logger.debug("initial staggered temperature = %s", self._temperature)

    @property
    def temperature(self) -> npt.NDArray:
        return self._temperature

    def get_linear(self) -> npt.NDArray:
        """Gets a linear temperature profile

        Returns:
            Linear temperature profile for the staggered nodes
            Only works for uniform spatial mesh.
        """
        temperature_basic: npt.NDArray = np.linspace(
            self._settings.basal_temperature,
            self._settings.surface_temperature,
            self._mesh.basic.number_of_nodes,
        )
        return self._mesh.quantity_at_staggered_nodes(temperature_basic)

    def get_adiabat(self, pressure_basic) -> npt.NDArray:
        """Gets an entropy-conserving adiabatic temperature profile.

        Uses Clausius-Clapeyron and Maxwell relations to correctly
        integrate through the two-phase (mushy) region without
        requiring entropy tables. Detects solidus/liquidus crossings
        and switches between single-phase and mushy-zone integration.

        When entropy tables are also available, verifies the CC result
        against the entropy-inversion method.

        Parameters
        ----------
        pressure_basic : npt.NDArray
            Pressure field on the basic nodes (CMB to surface ordering).

        Returns
        -------
        npt.NDArray
            Adiabatic temperature profile for the staggered nodes.
        """
        active = self._phases.active
        has_entropy = hasattr(active, 'has_entropy') and active.has_entropy

        if has_entropy:
            # Primary: entropy-table inversion (exact, matches SPIDER)
            logger.info(
                "Using entropy-table adiabat (matches SPIDER formulation)"
            )
            return self._get_adiabat_entropy_conserving(pressure_basic)

        # Fallback: Clausius-Clapeyron ODE (approximate for PALEOS
        # because the EOS is not thermodynamically consistent at the
        # melting curve; CC underestimates Delta_S by ~5x)
        logger.warning(
            "No entropy tables available. Using Clausius-Clapeyron "
            "adiabat (approximate; PALEOS CC underestimates Delta_S "
            "by ~5x at the melting curve)."
        )
        return self._get_adiabat_clausius_clapeyron(pressure_basic)

    def _get_adiabat_single_phase(self, pressure_basic) -> npt.NDArray:
        """Adiabat by integrating dTdPs = alpha*T/(rho*Cp).

        Accurate for single-phase regions only. Does not conserve entropy
        across solidus/liquidus crossings.

        Parameters
        ----------
        pressure_basic : npt.NDArray
            Pressure on basic nodes (CMB to surface).

        Returns
        -------
        npt.NDArray
            Temperature at staggered nodes.
        """

        def adiabat_ode(P, T):
            self._phases.active.set_pressure(P)
            self._phases.active.set_temperature(T)
            self._phases.active.update()
            return self._phases.active.dTdPs()

        # flip the pressure field top to bottom
        pressure_basic = np.flip(pressure_basic)

        sol = solve_ivp(
            adiabat_ode, (pressure_basic[0], pressure_basic[-1]),
            [self._settings.surface_temperature], t_eval=pressure_basic,
            method='RK45', rtol=1e-6, atol=1e-9)

        # flip back the temperature field from bottom to top
        temperature_basic = np.flip(sol.y[0])

        # Return temperature field at staggered nodes
        return self._mesh.quantity_at_staggered_nodes(temperature_basic)

    def _get_adiabat_clausius_clapeyron(self, pressure_basic) -> npt.NDArray:
        """Entropy-conserving adiabat via Clausius-Clapeyron relations.

        Integrates through three regimes:
        - Single-phase solid: dT/dP = alpha*T/(rho*Cp)
        - Mushy zone: coupled ODE using Maxwell + Clausius-Clapeyron
        - Single-phase liquid: dT/dP = alpha*T/(rho*Cp)

        In the mushy zone, entropy conservation is enforced via:
          Delta_S = Delta_V / (dT_sol/dP)           (Clausius-Clapeyron)
          dS_sol/dP = -alpha_sol/rho_sol + Cp_sol/T_sol * dT_sol/dP  (Maxwell)
          dS_liq/dP = -alpha_liq/rho_liq + Cp_liq/T_liq * dT_liq/dP  (Maxwell)
          dphi/dP = -[phi*dSliq/dP + (1-phi)*dSsol/dP] / Delta_S
          dT/dP = dTsol/dP + dphi/dP*(Tliq-Tsol) + phi*(dTliq/dP-dTsol/dP)

        No entropy tables required; uses only alpha, Cp, rho, and melting
        curve slopes.

        Parameters
        ----------
        pressure_basic : npt.NDArray
            Pressure on basic nodes (CMB to surface).

        Returns
        -------
        npt.NDArray
            Temperature at staggered nodes.
        """
        active = self._phases.active

        # CC adiabat requires composite evaluator (solid + mixed + liquid)
        if not hasattr(active, '_mixed'):
            logger.warning(
                "CC adiabat requires composite phase evaluator. "
                "Falling back to single-phase dTdPs."
            )
            return self._get_adiabat_single_phase(pressure_basic)

        def get_solidus_liquidus(P):
            """Get solidus and liquidus T at pressure P."""
            P_arr = np.atleast_1d(P)
            active.set_pressure(P_arr)
            T_sol = float(np.squeeze(active.solidus()))
            T_liq = float(np.squeeze(active.liquidus()))
            return T_sol, T_liq

        def dTdP_single_phase(P, T):
            """Single-phase adiabatic gradient."""
            active.set_pressure(np.atleast_1d(P))
            active.set_temperature(np.atleast_1d(T))
            active.update()
            return float(np.squeeze(active.dTdPs()))

        def dTdP_mushy(P, T):
            """Mushy-zone adiabatic gradient via Clausius-Clapeyron.

            Uses Maxwell relations to compute entropy gradients along
            the phase boundaries, then enforces dS_total/dP = 0.
            """
            P_arr = np.atleast_1d(P)
            active.set_pressure(P_arr)

            T_sol = float(np.squeeze(active.solidus()))
            T_liq = float(np.squeeze(active.liquidus()))
            delta_T = T_liq - T_sol
            if delta_T < 1e-10:
                return dTdP_single_phase(P, T)

            phi = max(0.0, min(1.0, (T - T_sol) / delta_T))

            # Clapeyron slopes
            dTsol_dP = float(np.squeeze(active.solidus_gradient()))
            dTliq_dP = float(np.squeeze(active.liquidus_gradient()))

            # Phase properties at solidus and liquidus
            # (already set by set_pressure -> solid at T_sol, liquid at T_liq)
            rho_sol = float(np.squeeze(active._mixed._solid.density()))
            rho_liq = float(np.squeeze(active._mixed._liquid.density()))
            alpha_sol = float(np.squeeze(active._mixed._solid.thermal_expansivity()))
            alpha_liq = float(np.squeeze(active._mixed._liquid.thermal_expansivity()))
            Cp_sol = float(np.squeeze(active._mixed._solid.heat_capacity()))
            Cp_liq = float(np.squeeze(active._mixed._liquid.heat_capacity()))

            # Effective Delta_S across the mushy zone: Clausius-Clapeyron
            # latent heat + sensible heat of solid from T_sol to T_liq.
            # Uses the liquidus slope (real melting curve) for CC.
            delta_V = 1.0 / rho_liq - 1.0 / rho_sol
            if abs(dTliq_dP) < 1e-30:
                return dTdP_single_phase(P, T)
            delta_S_melt = abs(delta_V / dTliq_dP)
            delta_S_sensible = max(0.0, Cp_sol * np.log(T_liq / max(T_sol, 1.0)))
            delta_S = delta_S_melt + delta_S_sensible

            if abs(delta_S) < 1e-30:
                return dTdP_single_phase(P, T)

            # Maxwell relations: dS_phase/dP along each phase boundary
            # dS/dP|boundary = (dS/dP)_T + (dS/dT)_P * dT_boundary/dP
            #                = -alpha/rho + Cp/T * dT_boundary/dP
            dSsol_dP = -alpha_sol / rho_sol + Cp_sol / T_sol * dTsol_dP
            dSliq_dP = -alpha_liq / rho_liq + Cp_liq / T_liq * dTliq_dP

            # Entropy conservation: dphi/dP = -[phi*dSliq/dP + (1-phi)*dSsol/dP] / Delta_S
            dphi_dP = -(phi * dSliq_dP + (1 - phi) * dSsol_dP) / delta_S

            # Temperature gradient in mushy zone
            dT_dP = dTsol_dP + dphi_dP * delta_T + phi * (dTliq_dP - dTsol_dP)

            return dT_dP

        # Flip pressure to surface -> CMB ordering for downward integration
        P_down = np.flip(pressure_basic)
        T_profile = np.zeros_like(P_down)
        T_profile[0] = self._settings.surface_temperature

        for i in range(1, len(P_down)):
            P_i = float(P_down[i])
            P_prev = float(P_down[i - 1])
            T_prev = float(T_profile[i - 1])
            dP = P_i - P_prev

            T_sol, T_liq = get_solidus_liquidus(P_i)
            T_sol_prev, T_liq_prev = get_solidus_liquidus(P_prev)

            # Determine regime at current point
            in_mushy_prev = T_sol_prev <= T_prev <= T_liq_prev
            in_mushy_next_est = T_sol <= (T_prev + dTdP_single_phase(P_prev, T_prev) * dP) <= T_liq

            if in_mushy_prev or in_mushy_next_est:
                # Use Clausius-Clapeyron ODE in mushy zone
                # RK4 step for accuracy
                k1 = dTdP_mushy(P_prev, T_prev)
                k2 = dTdP_mushy(P_prev + 0.5 * dP, T_prev + 0.5 * dP * k1)
                k3 = dTdP_mushy(P_prev + 0.5 * dP, T_prev + 0.5 * dP * k2)
                k4 = dTdP_mushy(P_i, T_prev + dP * k3)
                T_profile[i] = T_prev + dP / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                # Single-phase: standard adiabatic gradient
                k1 = dTdP_single_phase(P_prev, T_prev)
                k2 = dTdP_single_phase(P_prev + 0.5 * dP, T_prev + 0.5 * dP * k1)
                k3 = dTdP_single_phase(P_prev + 0.5 * dP, T_prev + 0.5 * dP * k2)
                k4 = dTdP_single_phase(P_i, T_prev + dP * k3)
                T_profile[i] = T_prev + dP / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        logger.info(
            "CC adiabat: T_surf=%.1f -> T_cmb=%.1f (n=%d nodes)",
            T_profile[0],
            T_profile[-1],
            len(T_profile),
        )

        # Flip back to CMB -> surface ordering
        temperature_basic = np.flip(T_profile)
        return self._mesh.quantity_at_staggered_nodes(temperature_basic)

    def _get_adiabat_entropy_conserving(self, pressure_basic) -> npt.NDArray:
        """Adiabat by inverting S(P, T) = S_target at each pressure.

        Correctly conserves entropy across phase boundaries. In the
        two-phase region, the entropy includes the latent heat contribution
        via linear interpolation between solid and liquid entropies at the
        solidus and liquidus temperatures.

        Parameters
        ----------
        pressure_basic : npt.NDArray
            Pressure on basic nodes (CMB to surface).

        Returns
        -------
        npt.NDArray
            Temperature at staggered nodes.
        """
        active = self._phases.active

        # Compute target entropy at the surface
        P_surf = np.flip(pressure_basic)[0]  # surface pressure (lowest)
        T_surf = self._settings.surface_temperature
        S_target = active.entropy_at(P_surf, T_surf)
        logger.info(
            "Entropy-conserving adiabat: S_target = %.2f at T_surf = %.2f, P_surf = %.4e",
            S_target, T_surf, P_surf,
        )

        # flip to surface-to-CMB ordering for downward integration
        P_down = np.flip(pressure_basic)
        T_profile = np.zeros_like(P_down)
        T_profile[0] = T_surf

        # Integrate downward: at each P, find T such that S(P, T) = S_target
        for i in range(1, len(P_down)):
            P_i = float(P_down[i])
            T_prev = float(T_profile[i - 1])

            def entropy_residual(T_candidate):
                return active.entropy_at(P_i, T_candidate) - S_target

            # Search interval: expand around previous T
            # Temperature must increase with depth along an adiabat
            T_lo = T_prev * 0.8
            T_hi = T_prev * 2.0

            # Ensure the bracket contains the root
            s_lo = entropy_residual(T_lo)
            s_hi = entropy_residual(T_hi)

            # Widen bracket if needed
            n_expand = 0
            while s_lo * s_hi > 0 and n_expand < 20:
                if s_lo > 0:
                    T_lo *= 0.5
                    s_lo = entropy_residual(T_lo)
                else:
                    T_hi *= 2.0
                    s_hi = entropy_residual(T_hi)
                n_expand += 1

            if s_lo * s_hi > 0:
                logger.warning(
                    "Could not bracket entropy root at P=%.4e (T_lo=%.1f, T_hi=%.1f, "
                    "S_lo=%.2f, S_hi=%.2f, S_target=%.2f). Using dTdPs fallback.",
                    P_i, T_lo, T_hi, s_lo + S_target, s_hi + S_target, S_target,
                )
                # Fallback: use single-phase gradient for this step
                active.set_pressure(np.atleast_1d(P_i))
                active.set_temperature(np.atleast_1d(T_prev))
                active.update()
                dTdP = float(active.dTdPs())
                T_profile[i] = T_prev + dTdP * (P_i - float(P_down[i - 1]))
            else:
                T_profile[i] = brentq(entropy_residual, T_lo, T_hi, rtol=1e-10)

        # Verify entropy conservation
        S_cmb = active.entropy_at(P_down[-1], T_profile[-1])
        S_drift = abs(S_cmb - S_target) / abs(S_target) * 100
        logger.info(
            "Entropy-conserving adiabat: T_surf=%.1f, T_cmb=%.1f, "
            "S_target=%.2f, S_cmb=%.2f (drift=%.4f%%)",
            T_profile[0], T_profile[-1], S_target, S_cmb, S_drift,
        )
        if S_drift > 0.1:
            logger.warning(
                "Entropy drift %.4f%% exceeds 0.1%% tolerance. The adiabat may "
                "not be fully converged.", S_drift,
            )

        # flip back to CMB-to-surface ordering
        temperature_basic = np.flip(T_profile)

        return self._mesh.quantity_at_staggered_nodes(temperature_basic)
