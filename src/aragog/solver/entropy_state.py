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


# ── Smooth-limiter helpers (RHS regularisation for implicit BDF) ─────
#
# SUNDIALS CVODE's higher-order BDF predictors use polynomial fits
# through past solution values, and its Newton iteration relies on a
# finite-difference Jacobian. Both break down where the RHS has
# kinks (|x|), step functions (x < 0), or flat tops (clip(x, 0, 1)):
# the predictor error estimate is dominated by the non-smoothness
# rather than by polynomial truncation, the order selector drops to
# 1, and Newton fails to converge because a tiny FD perturbation
# crosses the kink and flips the Jacobian sign.
#
# These helpers replace the non-smooth operations with C^1-continuous
# approximations based on `sqrt(x^2 + eps^2)`. The smoothing bandwidth
# eps is chosen small compared to the physically relevant scale so
# the regularisation only affects values in the immediate neighbourhood
# of the kink, leaving bulk physics unchanged. Both operations return
# the exact unsmoothed value to within ~eps^2 / (2 * |x|) for |x| >> eps.


def _smooth_abs_neg(x: npt.NDArray, eps: float = 1.0e-8) -> npt.NDArray:
    """Smooth approximation of ``max(-x, 0)``.

    Returns ``-x`` when ``x << -eps``, ``0`` when ``x >> eps``, and
    smoothly interpolates across the zero crossing. Folds the abs()
    and the boolean mask ``x < 0`` used for convective-cell selection
    into a single C^infinity expression.
    """
    return 0.5 * (np.sqrt(x * x + eps * eps) - x)


def _smooth_clip(
    x: npt.NDArray, lo: float = 0.0, hi: float = 1.0, eps: float = 1.0e-3,
) -> npt.NDArray:
    """Smooth approximation of ``np.clip(x, lo, hi)``.

    Built from two ``smooth_max`` operations. Bandwidth eps is in the
    same units as x; for gphi (dimensionless [0,1]) eps = 1e-3 makes
    the smoothing imperceptible except within ~0.1 % of the boundaries.
    """
    # smooth_max(a, b) = 0.5 * (a + b + sqrt((a-b)^2 + eps^2))
    # First clip at the lower bound:
    x_lo = x - lo
    x_above_lo = 0.5 * (x + lo + np.sqrt(x_lo * x_lo + eps * eps))
    # Then clip at the upper bound:
    hi_u = hi - x_above_lo
    return 0.5 * (x_above_lo + hi - np.sqrt(hi_u * hi_u + eps * eps))


def _spider_get_smoothing(
    gphi: npt.NDArray, smooth_width: float = 1.0e-2,
) -> npt.NDArray:
    """Verbatim port of SPIDER's ``get_smoothing`` (``util.c:245-270``).

    Two-branch tanh smoother used for the phase-boundary smoothing
    factor in SPIDER's mixing-flux computation. For ``gphi > 0.5``
    the curve ramps down toward zero near ``gphi = 1``; for
    ``gphi <= 0.5`` it ramps up toward one near ``gphi = 0``. The
    branches join continuously at ``gphi = 0.5`` (with a derivative
    kink that SUNDIALS CVODE's order selector handles fine; scipy
    BDF/Radau should tolerate it too because it's bounded).

    Parameters
    ----------
    gphi : array
        Un-truncated lever-rule fraction
        ``(S - S_sol) / (S_liq - S_sol)``. Values outside [0, 1]
        are allowed (pure-phase regimes); the smoothing goes to zero
        in both limits.
    smooth_width : float, optional
        tanh transition width in gphi units. SPIDER's default for
        CHILI R8 is ``matprop_smooth_width = 0.01``.

    Returns
    -------
    smth : array
        Smoothing factor, 1 across most of [0, 1], smoothly tapered
        to zero just outside.
    """
    if smooth_width == 0.0:
        # No smoothing — hard clip to [0, 1].
        out = np.ones_like(gphi)
        out[(gphi < 0.0) | (gphi > 1.0)] = 0.0
        return out
    # SPIDER's two-branch tanh (util.c:261-266). Use ``np.where`` so
    # we evaluate both branches and select per element, keeping the
    # result smooth in numpy terms (no Python conditionals in the
    # hot path).
    upper = 0.5 * (1.0 - np.tanh((gphi - 1.0) / smooth_width))
    lower = 0.5 * (1.0 + np.tanh(gphi / smooth_width))
    return np.where(gphi > 0.5, upper, lower)


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
        dilatation: bool = False,
        tidal: bool = False,
        tidal_array: list | None = None,
        eddy_diffusivity_thermal: float = 1.0,
        eddy_diffusivity_chemical: float = 1.0,
        kappah_floor: float = 0.0,
        bottom_up_grav_sep: bool = True,
        phase_smoothing: str = 'cubic_hermite',
    ):
        self._evaluator = evaluator
        self.phase_staggered = phase_staggered
        self.phase_basic = phase_basic
        self._conduction = conduction
        self._convection = convection
        self._grav_sep = gravitational_separation
        self._mixing = mixing
        self._radionuclides = radionuclides
        self._dilatation = dilatation
        self._tidal = tidal
        self._tidal_array = tidal_array if tidal_array is not None and len(tidal_array) > 0 else [0.0]
        self._eddy_diff_thermal = eddy_diffusivity_thermal
        self._eddy_diff_chem = eddy_diffusivity_chemical
        self._kappah_floor = kappah_floor
        self._bottom_up_grav_sep = bool(bottom_up_grav_sep)
        # Phase-boundary smoothing method for Jgrav and Jmix.
        # 'cubic_hermite': 16*gphi^2*(1-gphi)^2. Provides intermediate-phi
        #   damping (smth=0.32 at gphi=0.83) that prevents the CMB drain
        #   when residual EOS differences exist. Default until full bit-
        #   parity with SPIDER is achieved.
        # 'tanh': SPIDER's get_smoothing(matprop_smooth_width=0.01, gphi).
        #   Gives smth=1.0 across [0.05, 0.95]. Correct for SPIDER parity
        #   once all material properties match to <0.01%.
        if phase_smoothing not in ('cubic_hermite', 'tanh'):
            raise ValueError(
                f"phase_smoothing must be 'cubic_hermite' or 'tanh', "
                f"got {phase_smoothing!r}"
            )
        self._phase_smoothing = phase_smoothing

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

        # Phase-boundary entropy cache at staggered nodes. P_stag is
        # fixed for the lifetime of the solve, so S_sol(P_stag) and
        # S_liq(P_stag) are constants. Caching them avoids ~230k
        # scipy interpolator calls per 10-yr PROTEUS step (two per
        # RHS eval) that otherwise dominate the BDF hot path.
        self._P_stag_cached_id: int = -1
        self._S_sol_stag: npt.NDArray = np.zeros(n_staggered)
        self._S_liq_stag: npt.NDArray = np.zeros(n_staggered)
        self._dS_phase_stag: npt.NDArray = np.ones(n_staggered)
        self._pb_cache_hits: int = 0
        self._pb_cache_misses: int = 0

        # Phase-boundary cache at basic nodes for the SPIDER-parity
        # mixing flux (``_jmix_spider_heat`` in update()). Like the
        # staggered cache, these quantities are mesh-fixed because the
        # basic-node pressure profile doesn't change within a solve.
        # Needed: S_sol/S_liq and their P-derivatives, the mesh
        # pressure gradient dP/dr, and the phase-boundary mean
        # temperature T_fus = ½(T_sol + T_liq).
        self._P_basic_cached_id: int = -1
        self._S_sol_basic: npt.NDArray = np.zeros(n_basic)
        self._S_liq_basic: npt.NDArray = np.zeros(n_basic)
        self._dS_phase_basic: npt.NDArray = np.ones(n_basic)
        self._dS_sol_dP_basic: npt.NDArray = np.zeros(n_basic)
        self._dS_liq_dP_basic: npt.NDArray = np.zeros(n_basic)
        self._dP_dr_basic: npt.NDArray = np.zeros(n_basic)
        self._T_fus_basic: npt.NDArray = np.zeros(n_basic)

    def _ensure_phase_boundary_cache(self) -> None:
        """Populate or refresh the cached S_sol/S_liq at staggered nodes.

        Cheap no-op when the underlying phase_staggered.pressure ndarray
        is the same object (by Python id) as the one seen on the last
        call, or when const_properties mode is active (no phase boundaries).
        """
        if getattr(self.phase_staggered, '_const_properties', False):
            return  # no phase boundaries in const_properties mode
        pressure_obj = self.phase_staggered.pressure
        if id(pressure_obj) == self._P_stag_cached_id:
            self._pb_cache_hits += 1
            return
        self._pb_cache_misses += 1
        P_stag = np.asarray(pressure_obj).ravel()
        eos = self.phase_staggered._eos
        self._S_sol_stag = np.asarray(eos.solidus_entropy(P_stag)).ravel()
        self._S_liq_stag = np.asarray(eos.liquidus_entropy(P_stag)).ravel()
        self._dS_phase_stag = np.maximum(self._S_liq_stag - self._S_sol_stag, 1.0)
        self._P_stag_cached_id = id(pressure_obj)

    def _ensure_basic_phase_boundary_cache(self) -> None:
        """Populate or refresh the cached phase-boundary data at basic nodes.

        Caches S_sol, S_liq, their P-derivatives, the mesh dP/dr, and
        the phase-boundary mean temperature T_fus = ½(T_sol + T_liq),
        all at basic nodes. These quantities depend only on the mesh
        pressure profile, which is mesh-fixed for the lifetime of the
        solve, so the cache is a cheap no-op after the first call.

        Used by the SPIDER-parity mixing-flux formula (``Jmix_spider_heat``
        in update()) which computes Jmix via the bracket expression
        ``dS/dr − [φ dS_liq/dP + (1−φ) dS_sol/dP] dP/dr`` from
        ``energy.c::GetMixingHeatFlux`` lines 307-314.
        """
        if getattr(self.phase_basic, '_const_properties', False):
            return  # no phase boundaries in const_properties mode
        pressure_obj = self.phase_basic.pressure
        if id(pressure_obj) == self._P_basic_cached_id:
            return
        P_basic = np.asarray(pressure_obj).ravel()
        eos = self.phase_basic._eos
        # Phase-boundary entropies and their P-derivatives
        self._S_sol_basic = np.asarray(eos.solidus_entropy(P_basic)).ravel()
        self._S_liq_basic = np.asarray(eos.liquidus_entropy(P_basic)).ravel()
        self._dS_phase_basic = np.maximum(
            self._S_liq_basic - self._S_sol_basic, 1.0
        )
        self._dS_sol_dP_basic = np.asarray(eos.solidus_entropy_dP(P_basic)).ravel()
        self._dS_liq_dP_basic = np.asarray(eos.liquidus_entropy_dP(P_basic)).ravel()
        # Mesh pressure gradient at basic nodes (P decreases outward,
        # so dP/dr < 0 in the mantle).
        r_basic = np.asarray(self._evaluator.mesh.basic.radii).ravel()
        self._dP_dr_basic = np.gradient(P_basic, r_basic)
        # Phase-boundary mean temperature T_fus. Computed from
        # L = T_fus * ΔS_phase via L/ΔS_phase, where L is the
        # EOS's latent_heat(P) returning T_fus * (S_liq - S_sol).
        L_basic = np.asarray(eos.latent_heat(P_basic)).ravel()
        self._T_fus_basic = L_basic / self._dS_phase_basic
        self._P_basic_cached_id = id(pressure_obj)

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
            # SPIDER-parity entropy gradient at basic nodes.
            #
            # Replaces Aragog's ``mesh.d_dr_at_basic_nodes(S)`` which
            # used a dense transform matrix with a 3-point second-order
            # boundary extrapolation scaled by dxi/dr. That stencil
            # overshoots catastrophically at the CMB when the
            # crystallisation front creates a kink in the staggered S
            # profile: the extrapolated dS/dr[0] reached 10^7x the
            # physically correct value, producing a 2.6e10 W/m^2 Jtot
            # spike at the CMB basic node that drained the bottom
            # staggered cell's entropy to the solidus in one coupling
            # step.
            #
            # SPIDER computes dSdxi as a simple centered difference in
            # UNIFORM xi-space between adjacent staggered cells
            # (ic.c:443-446), then applies the chain rule dS/dr =
            # dSdxi * dxi/dr. At boundaries SPIDER copies the nearest
            # interior value (ic.c:450: ``arr_dSdxi_b[CMB] =
            # arr_dSdxi_b[CMB-1]``). The uniform-xi spacing makes the
            # denominator constant, bounding the gradient to the actual
            # inter-cell entropy difference regardless of the spatial
            # mesh non-uniformity.
            xi_s = np.asarray(mesh.staggered.mass_radii).ravel()
            dxi_s = xi_s[1:] - xi_s[:-1]
            n_basic = mesh.basic.radii.size
            dSdxi = np.zeros(n_basic)
            dSdxi[1:-1] = (S[1:] - S[:-1]) / dxi_s
            # Boundary values: SPIDER evolves dSdxi at the CMB and
            # surface as state variables via the core energy balance
            # ODE (bc.c:76, set_cmb_entropy_gradient_update). At the
            # CMB, this drives dSdxi toward ~0 (core acts as a thermal
            # reservoir in quasi-equilibrium). At t=33821 yr on CHILI
            # R8 Earth, SPIDER's dSdxi[CMB] = -3.94e-12 while the
            # first interior node has dSdxi = -6.99e-5 — 7 orders of
            # magnitude larger. In Aragog's quasi_steady mode (no core
            # energy balance ODE), we approximate this by setting the
            # boundary gradients to zero, which is the steady-state
            # limit of SPIDER's evolved boundary gradient.
            # SPIDER bc.c convention: boundary gradients copy from the
            # adjacent interior node. In energy_balance mode, the CMB
            # gradient is overridden later by the state-vector value.
            dSdxi[0] = dSdxi[1]    # CMB: copy from first interior
            dSdxi[-1] = dSdxi[-2]  # surface: copy from last interior
            dxidr = np.asarray(mesh.dxidr).ravel()
            self._dSdr = dSdxi * dxidr

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
        # Use the CLAMPED lever-rule fraction, matching v7.5's behaviour
        # (which scipy BDF handled cleanly). This is:
        #     phi = clip((S - S_sol) / (S_liq - S_sol), 0, 1)
        # Per-cell clipping gives dphi/dr = 0 in fully-molten cells
        # (physically correct: no phase separation in pure liquid) and
        # in fully-solid cells (physically correct: no mixing in pure
        # solid). At the crystallisation front, the mushy cell has
        # 0 < phi < 1 and the adjacent cells are clipped to 0 or 1,
        # so dphi/dr is well-defined and smooth.
        #
        # We compute the clamped phi from the cached EOS lookups rather
        # than via self.phase_staggered.melt_fraction() because that
        # call introduces tiny IEEE roundoff noise on each invocation
        # which breaks BDF's finite-difference Jacobian stability.
        # The cached S_sol/S_liq arrays are byte-identical across
        # calls within a solve, keeping the Jacobian stable.
        #
        # History: 2026-04-11 the un-truncated gphi path was tried
        # (without clipping) to preserve the Jmix+Jconv=0 cancellation
        # identity, but it made the RHS non-smooth across the
        # full mantle and caused both scipy BDF and CVODE to thrash
        # Newton convergence. A hard gate on phi_min was then tried
        # but introduced a discontinuous RHS which failed CVODE's
        # Newton entirely. Reverted to v7.5's clamped phi. The
        # resulting -12% T_core vs SPIDER gap is documented in
        # spider_aragog_parity_v3_v4.md as a formulation difference.
        self._ensure_phase_boundary_cache()
        gphi = (S - self._S_sol_stag) / self._dS_phase_stag
        phi_smoothclipped = _smooth_clip(gphi, 0.0, 1.0, eps=1.0e-3)
        # _dphidr was computed here via the dense transform matrix, but the
        # result is never read (confirmed by grep across all solver files).
        # The mixing flux now uses dSdr and dP/dr instead (SPIDER-parity
        # rewrite at lines 643-695). Removed to save one (N+1)x(N-1)
        # matrix multiply per RHS evaluation (~1000 calls per PROTEUS step).

        # ── MLT from entropy gradient ────────────────────────────────
        # Convection is unstable when dS/dr < 0 (entropy decreasing
        # outward). `_is_convective` is still maintained for downstream
        # diagnostic output but is NO LONGER used to zero the velocity
        # arrays via a boolean mask — that discontinuity broke CVODE's
        # higher-order BDF predictor. Instead, the smoothed convective
        # driver below naturally goes to zero (smoothly) for
        # stably-stratified cells and to |dS/dr| for unstable ones.
        self._is_convective = self._dSdr < 0

        # Buoyancy: convert entropy gradient to effective thermal
        # buoyancy |superadiabatic| = alpha * T * |dS/dr| / Cp.
        # Use the smoothed max(-dSdr, 0), which replaces the
        # `np.abs(dSdr)` + `mask[~convective] = 0` pair with a single
        # C^infinity expression. For dSdr << 0: reduces to -dSdr
        # (unstable profile, full convection drive). For dSdr >> 0:
        # smoothly reduces to 0 (stable profile, no convection).
        alpha = np.asarray(self.phase_basic.thermal_expansivity()).ravel()
        T = np.asarray(self.phase_basic.temperature()).ravel()
        Cp = np.asarray(self.phase_basic.heat_capacity()).ravel()
        g = np.asarray(self.phase_basic.gravitational_acceleration()).ravel()

        # eps=1e-30 is intentionally near-zero: at double precision this
        # gives a hard max(-x, 0) without the kink smoothing. The original
        # eps=1e-8 inflated |dSdr| by 15% when gradients were O(eps) at
        # isentropic ICs, causing 7% kh error. SPIDER uses a hard if/else
        # threshold, so the near-zero eps matches SPIDER's behavior. If
        # CVODE stability requires smoothing, increase to ~1e-20.
        conv_drive = _smooth_abs_neg(self._dSdr, eps=1.0e-30)
        effective_superadiabatic = alpha * T * conv_drive / np.maximum(Cp, 1.0)
        velocity_prefactor = g * effective_superadiabatic

        # Viscous velocity (Re <= Re_crit). No boolean masking needed:
        # velocity_prefactor already vanishes smoothly for stable
        # (dSdr > 0) cells via conv_drive.
        mixing_length = self._mixing_length
        mixing_length_cubed = self._mixing_length_cu
        mixing_length_squared = self._mixing_length_sq
        nu = np.asarray(self.phase_basic.kinematic_viscosity()).ravel()

        viscous_velocity = velocity_prefactor * mixing_length_cubed / (18.0 * nu)

        # Inviscid velocity (Re > Re_crit). Add a tiny eps^2 inside the
        # sqrt to avoid the sqrt-kink at velocity_sq = 0; the value is
        # negligibly different from sqrt(max(x, 0)) for any physical
        # inviscid_velocity_sq.
        inviscid_velocity_sq = velocity_prefactor * mixing_length_squared / 16.0
        inviscid_velocity = np.sqrt(inviscid_velocity_sq + 1.0e-20)

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

        # Ensure mesh dP/dr is populated (needed by conduction and mixing).
        # The cache is computed lazily from the mesh pressure profile and
        # only needs refreshing when the mesh changes.
        if self._conduction or self._mixing:
            self._ensure_basic_phase_boundary_cache()

        if self._conduction:
            # SPIDER-parity conductive flux (energy.c:358-378):
            #   F_cond = -k * [(T/Cp) * dS/dr + dT/dr|_adiabat]
            #
            # The total temperature gradient decomposes into a
            # superadiabatic part (proportional to the entropy gradient)
            # and an adiabatic part. SPIDER evaluates dT/dr|_adiabat
            # as dTdPs * dPdr, where dTdPs = (dT/dP)|_S comes from
            # the EOS table (eos_composite.c:249, energy.c:369).
            # Using the EOS table value instead of the thermodynamic
            # identity -g*alpha*T/Cp ensures exact parity with SPIDER,
            # since the table lookup and the identity can differ by
            # up to 40% due to table interpolation and composite-EOS
            # blending at phase boundaries.
            Cp_safe = np.maximum(Cp, 100.0)
            superadiabatic = (T / Cp_safe) * self._dSdr
            # Adiabatic gradient from EOS table (SPIDER parity).
            # SPIDER computes dTdxis*dxidr where dTdxis is from the
            # EOS table and dxidr is the mesh Jacobian. The equivalent
            # is dTdPs * dPdr where dPdr comes from the MESH pressure
            # profile (Adams-Williamson), NOT from -rho_material*g.
            # Using the mesh dPdr matches SPIDER exactly; using
            # -rho*g introduces up to 43% error because the mesh
            # structural density differs from the EOS material density.
            dTdPs = np.asarray(self.phase_basic.dTdPs()).ravel()
            dTdrs_ad = dTdPs * self._dP_dr_basic
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
            # Phase-boundary smoothing for gravitational separation.
            # Uses the same cubic Hermite ``16*gphi^2*(1-gphi)^2`` as
            # the Jmix term below. Both Jgrav and Jmix MUST use the
            # same smoothing function so their relative balance is
            # internally consistent.
            #
            # SPIDER uses a tanh (get_smoothing, width=0.01) for both,
            # which gives smth=1.0 across most of [0.05, 0.95]. This
            # works in SPIDER because all material properties match
            # exactly, producing a precise Jtot cancellation. In Aragog,
            # residual EOS interpolation differences create a small Jtot
            # imbalance that the full-strength tanh amplifies into a
            # CMB cell drain. The cubic Hermite provides intermediate-
            # phi damping (smth=0.32 at gphi=0.83 vs tanh=1.0) that
            # gives Aragog the margin it needs. Once all audit items
            # (1.1, 1.2) are resolved and material properties match
            # SPIDER to 0.01%, the smoothing can be switched to tanh.
            if self._bottom_up_grav_sep:
                gphi_stag = (
                    self._entropy_staggered - self._S_sol_stag
                ) / self._dS_phase_stag

                if self._phase_smoothing == 'tanh':
                    smth_stag = _spider_get_smoothing(gphi_stag, smooth_width=1.0e-2)
                else:
                    gphi_clip = _smooth_clip(gphi_stag, 0.0, 1.0, eps=1.0e-3)
                    smth_stag = 16.0 * gphi_clip**2 * (1.0 - gphi_clip) ** 2

                # Bottom-up: basic node i (interface between staggered
                # i-1 and i) sees the smoothing of staggered i-1 (the
                # cell BELOW).
                smth_basic = np.ones_like(jgrav)
                smth_basic[1:-1] = smth_stag[:-1]
                jgrav = jgrav * smth_basic

            self._mass_flux += jgrav

        # Zero mass fluxes at boundaries (SPIDER convention: no mass
        # transfer across CMB or surface, energy.c lines 282-285, 423-426)
        self._mass_flux[0] = 0.0
        self._mass_flux[-1] = 0.0
        self._heat_flux += self._mass_flux * self.phase_basic.latent_heat()

        if self._mixing:
            # SPIDER-parity mixing flux (energy.c::GetMixingHeatFlux
            # lines 307-325). Computed directly as a heat flux,
            # bypassing the mass_flux * latent_heat path used by the
            # grav_sep term above.
            #
            # Previously Aragog added ``rho * kappac * (-dphi/dr)`` to
            # mass_flux and then multiplied by L. The derivative of
            # the *clipped* phi truncates its magnitude by ~50% at the
            # crystallisation front (one cell at phi=1 next to a mushy
            # cell at phi in (0,1); the un-truncated gphi contribution
            # from the molten side is lost). This missing front-cell
            # flux integrated over solidification explains the ~-18%
            # T_core gap vs SPIDER reported in
            # ``spider_aragog_parity_v3_v4.md``.
            #
            # SPIDER's formula is equivalent to
            #     Jmix_heat = -kappac * rho * T_fus * bracket * smth
            # where
            #     bracket = dS/dr − [φ · dS_liq/dP + (1−φ) · dS_sol/dP] · dP/dr
            # and ``smth`` is a tanh (or cubic Hermite) that zeroes the
            # flux outside the mushy band 0 ≤ gphi ≤ 1. The bracket
            # is algebraically equal to ``dS_fus · dgphi/dr`` at the
            # basic node, but is computed directly from dS/dr plus
            # phase-boundary slopes so there is no un-truncated-gphi
            # arithmetic that could make the RHS non-smooth in the
            # fully-molten regime.
            self._ensure_basic_phase_boundary_cache()
            phi_basic_clipped = np.asarray(
                self.phase_basic.melt_fraction()
            ).ravel()
            bracket = self._dSdr - (
                phi_basic_clipped * self._dS_liq_dP_basic
                + (1.0 - phi_basic_clipped) * self._dS_sol_dP_basic
            ) * self._dP_dr_basic
            # Smoothing: same method as Jgrav (controlled by
            # self._phase_smoothing) for internal consistency.
            gphi_basic = (
                self._entropy_basic - self._S_sol_basic
            ) / self._dS_phase_basic
            if self._phase_smoothing == 'tanh':
                smth_basic_mix = _spider_get_smoothing(gphi_basic, smooth_width=1.0e-2)
            else:
                gphi_basic_clip = _smooth_clip(gphi_basic, 0.0, 1.0, eps=1.0e-3)
                smth_basic_mix = 16.0 * gphi_basic_clip**2 * (1.0 - gphi_basic_clip) ** 2
            jmix_spider_heat = (
                -self._kappac * rho * self._T_fus_basic * bracket * smth_basic_mix
            )
            # Zero at the actual boundaries (no mass transfer across
            # CMB or surface)
            jmix_spider_heat[0] = 0.0
            jmix_spider_heat[-1] = 0.0
            self._heat_flux += jmix_spider_heat

        # ── Internal heating (power per unit mass [W/kg]) ────────────
        n_stag = len(self._entropy_staggered)
        self._heating = np.zeros(n_stag)

        if self._radionuclides and hasattr(self._evaluator, 'radionuclides'):
            radio = 0.0
            for r in self._evaluator.radionuclides:
                radio += r.get_heating(time)
            self._heating += radio

        if self._dilatation and self._grav_sep:
            # Dilatation (PdV) heating: work done when melt of different
            # density rises through a pressure gradient. This is the
            # thermodynamically required companion to gravitational
            # separation: H = g * (1/rho_liq - 1/rho_sol) * J_mass.
            # Units: [m/s^2] * [m^3/kg] * [kg/m^2/s] = [W/kg].
            # The mass flux is on basic nodes; interpolate to staggered.
            mesh = self._evaluator.mesh
            mass_flux_stag = mesh.quantity_at_staggered_nodes(
                self._mass_flux
            ).ravel()
            delta_v = np.asarray(
                self.phase_staggered.delta_specific_volume()
            ).ravel()
            g = abs(float(self.phase_staggered._gravitational_acceleration))
            self._heating += g * delta_v * mass_flux_stag

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
