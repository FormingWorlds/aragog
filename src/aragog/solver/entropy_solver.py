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

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy
from scipy.constants import Stefan_Boltzmann
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from aragog.jax.nondim import NonDimScales

from aragog.eos.entropy import EntropyEOS
from aragog.eos.entropy_phase import EntropyPhaseEvaluator
from aragog.parser import Parameters
from aragog.solver.boundary import BoundaryConditions
from aragog.solver.entropy_state import EntropyState

# SUNDIALS CVODE via scikits-odes-sundials. Same underlying solver SPIDER
# uses (SUNDIALS CVODE BDF with modified-Newton nonlinear iteration and
# dense direct linear solver). scipy's BDF and Radau both collapse their
# step size to machine epsilon at the crystallisation front and fail with
# `status=-1: Required step size is less than spacing between numbers`
# because scipy's Newton iterator cannot converge on the stiff
# transition. CVODE's C implementation has order 1-5 BDF, a robust
# modified-Newton with cached Jacobian factorisation, and adaptive step
# control that handles phase-transition discontinuities cleanly.
#
# We import CVODE directly from scikits_odes_sundials, NOT via the
# scikits_odes metapackage's `ode('cvode', ...)` wrapper. The wrapper
# is a one-line dispatch (`ode('cvode')._integrator is CVODE` is True),
# but importing scikits_odes drags in scikits_odes_daepack as a transitive
# dependency, and daepack 3.0.0's f2py-generated C code is incompatible
# with numpy 2.3+ headers, breaking source builds in CI.
try:
    from scikits_odes_sundials.cvode import CVODE as _scikits_cvode
    from scikits_odes_sundials.cvode import CV_RootFunction as _CV_RootFunction

    _CVODE_AVAILABLE = True
    _CV_ROOTFN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CVODE_AVAILABLE = False
    _CV_ROOTFN_AVAILABLE = False
    _scikits_cvode = None  # type: ignore[assignment]
    _CV_RootFunction = object  # type: ignore[misc,assignment]

# Import SECS_PER_YEAR directly to avoid circular import with solver/__init__.py
from scipy import constants as _sp_constants

SECS_PER_YEAR: float = _sp_constants.Julian_year

logger = logging.getLogger('fwl.' + __name__)

# Directory for the per-call sub-step trajectory dump, read from
# ARAGOG_DUMP_STEP_TRAJ. Unset means no dump and no cost.
_TRAJ_DUMP_DIR = os.environ.get('ARAGOG_DUMP_STEP_TRAJ', '')
_TRAJ_DUMP_COUNT = 0


def _dump_step_trajectory(
    t_yr,
    dt_s,
    P_F_int,
    P_F_cmb,
    P_radio,
    P_resid,
    dSdr_cmb=None,
    S_bottom=None,
    y_hash=None,
    kh0=None,
    dSdr1=None,
) -> None:
    """Write the accepted sub-step power trajectory of one solver call.

    Diagnostic aid for the per-call energy integrals: the returned integrals
    are scalars, so a trajectory whose shape drives the result cannot be
    inspected after the fact. Writes one ``.npz`` per call into the directory
    named by ``ARAGOG_DUMP_STEP_TRAJ`` and is a no-op when that is unset.

    Parameters
    ----------
    t_yr : array_like
        Accepted sub-step times over the call [yr].
    dt_s : array_like
        Sub-step widths [s], length ``len(t_yr) - 1``.
    P_F_int, P_F_cmb, P_radio, P_resid : array_like
        Instantaneous surface-flux, CMB-flux, radiogenic and entropy-equation
        self-consistency powers at each accepted sub-step [W].
    dSdr_cmb, S_bottom : array_like, optional
        CMB entropy gradient state component [J/kg/K/m] and bottom-cell
        specific entropy [J/kg/K] at each accepted sub-step.
    y_hash, kh0, dSdr1 : array_like, optional
        Hash of the full state vector, the CMB-node eddy diffusivity
        [m^2/s], and the first interior node's entropy gradient
        [J/kg/K/m], at each accepted sub-step.
    """
    global _TRAJ_DUMP_COUNT
    if not _TRAJ_DUMP_DIR:
        return
    try:
        os.makedirs(_TRAJ_DUMP_DIR, exist_ok=True)
        path = os.path.join(_TRAJ_DUMP_DIR, f'traj_{_TRAJ_DUMP_COUNT:06d}.npz')
        np.savez_compressed(
            path,
            t_yr=np.asarray(t_yr, dtype=float),
            dt_s=np.asarray(dt_s, dtype=float),
            P_F_int=np.asarray(P_F_int, dtype=float),
            P_F_cmb=np.asarray(P_F_cmb, dtype=float),
            P_radio=np.asarray(P_radio, dtype=float),
            P_resid=np.asarray(P_resid, dtype=float),
            dSdr_cmb=np.asarray(dSdr_cmb if dSdr_cmb is not None else [], dtype=float),
            S_bottom=np.asarray(S_bottom if S_bottom is not None else [], dtype=float),
            y_hash=np.asarray(y_hash if y_hash is not None else [], dtype=float),
            kh0=np.asarray(kh0 if kh0 is not None else [], dtype=float),
            dSdr1=np.asarray(dSdr1 if dSdr1 is not None else [], dtype=float),
        )
    except OSError as exc:
        logger.warning('Sub-step trajectory dump failed: %s', exc)
    finally:
        _TRAJ_DUMP_COUNT += 1


def _validate_eos_radius_range(mesh_params) -> None:
    """Validate the loaded external-EOS radial grid against the mesh bounds.

    Duplicates the init-time check in ``Parameters.__post_init__`` so that
    ``EntropySolver.reset()`` catches a regenerated external EOS file
    whose radius range no longer matches the solver's current
    inner_radius / outer_radius. Without this guard the downstream
    ``PchipInterpolator`` calls in ``_initialize_internals`` and
    ``jax.phase._build_gravity_array`` would silently extrapolate gravity
    at grid boundaries, producing a subtle physics bug. Raises ValueError
    with a clear message if the file's radius range is inconsistent with
    the mesh; no-op if the file matches.
    """
    er = np.asarray(mesh_params.eos_radius).ravel()
    if er.size < 2:
        return
    inner = float(mesh_params.inner_radius)
    outer = float(mesh_params.outer_radius)
    span_mesh = outer - inner
    span_eos = float(er[-1]) - float(er[0])
    if not np.all(np.diff(er) > 0):
        raise ValueError(
            'External EOS radius array is not monotonically increasing; '
            'downstream PchipInterpolator would silently corrupt gravity / '
            'pressure / density lookups.'
        )
    # Resume from a saved mesh: Zalmoxis recomputes inner/outer R_int
    # from the live planet state, while the EOS table's radii are
    # frozen at the launch-time mesh. Single-ULP drift between the two
    # is harmless for downstream interpolation but trips a strict
    # `<`/`>` check. Use a relative tolerance scaled to the mesh span,
    # with a 1 m floor so absurd configurations (mismatched planet
    # radii) still raise.
    atol = max(1.0, 1.0e-9 * max(span_mesh, 1.0))
    if (
        float(er[0]) < inner - atol
        or float(er[-1]) > outer + atol
        or (span_mesh > 0 and span_eos < 0.75 * span_mesh)
    ):
        raise ValueError(
            f'External EOS radius range [{er[0]:.3e}, {er[-1]:.3e}] m '
            f'inconsistent with mesh bounds [{inner:.3e}, {outer:.3e}] m. '
            'A regenerated Zalmoxis mesh file may have drifted outside '
            'the solver shell; re-run Zalmoxis with the current planet '
            'radius, or rebuild the EOS table with matching bounds.'
        )


def _phase_prop_float(raw, default):
    """Convert a phase-property config value to float.

    Parameters
    ----------
    raw
        Value from _PhaseParameters: a plain float (from PROTEUS) or
        a string expression with an .eval() method (legacy .cfg parser).
    default
        Fallback if conversion fails entirely.  None means no fallback.

    Returns
    -------
    float or None
    """
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass
    try:
        return float(raw.eval())
    except Exception:
        pass
    return default


def _resolve_step_cap(raw: float | None) -> float:
    """Coerce a configured per-call step cap to the solver's float contract.

    The three per-call step caps (``phi_step_cap``, ``temperature_step_cap``,
    ``entropy_step_cap``) use a disabled sentinel at the configuration layer:
    ``None`` (unset), zero, or any non-positive or non-finite value all mean
    "cap off". The SUNDIALS root function downstream requires a plain float and
    treats a strictly-positive value as the armed cap, so this maps every
    disabled spelling to ``0.0`` and passes a genuine positive cap through
    unchanged. Centralising the coercion keeps the arming inequality
    (``> 0.0``) unambiguous: only a finite positive value can arm a cap, so a
    sentinel can never be misread as a tiny positive threshold.

    Parameters
    ----------
    raw : float or None
        The configured cap value, or ``None`` when unset.

    Returns
    -------
    float
        ``0.0`` when the cap is disabled (``None``, non-finite, or ``<= 0``),
        otherwise the positive cap.
    """
    if raw is None:
        return 0.0
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return value


# Single source of truth for the phase-boundary proximity band. The config
# layers (config/energy.py, parser.py) carry their own literal default so a
# user editing a config sees the number in place; this constant is the
# solver-side fallback and must move with them.
_DEFAULT_PHASE_BOUNDARY_ENTROPY_MARGIN = 200.0


def _resolve_entropy_margin(raw: float | None) -> float:
    """Coerce the configured phase-boundary proximity band to a usable float.

    Unlike the step caps, zero is not a valid "disabled" state for this band:
    a non-positive or non-finite margin would make every ``abs(margin) <
    entropy_margin`` test false and silently stop arming the tighter stepping
    across a phase crossing. So a missing (``None``), non-finite, or
    non-positive value falls back to the default rather than disabling the
    band; a genuine positive value passes through unchanged.

    Parameters
    ----------
    raw : float or None
        The configured margin, or ``None`` when the attribute is absent (e.g.
        a coupled build predating the field).

    Returns
    -------
    float
        The default (:data:`_DEFAULT_PHASE_BOUNDARY_ENTROPY_MARGIN`) when the
        input is ``None``, non-finite, or ``<= 0``, otherwise the positive
        margin.
    """
    if raw is None:
        return _DEFAULT_PHASE_BOUNDARY_ENTROPY_MARGIN
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        return _DEFAULT_PHASE_BOUNDARY_ENTROPY_MARGIN
    return value


def _phase_boundary_max_step_clamp(
    near_liq: bool,
    near_sol: bool,
    in_mushy: bool,
    cmb_margin_to_liq: float,
    cmb_margin_to_sol: float,
    entropy_margin: float,
) -> bool:
    """Whether ``max_step`` should be tightened to 1 yr for phase-boundary stiffness.

    The clamp resolves the stiff RHS a cell sees while crossing the
    solidus/liquidus. It fires when any staggered cell is within
    ``entropy_margin`` of either phase boundary (``near_liq``/``near_sol``), any
    cell is inside the mushy band (``in_mushy``), the CMB cell is near the
    liquidus, or the CMB cell is itself inside the mushy band. A
    fully-frozen mantle far below the solidus has no phase-boundary
    stiffness to resolve, so the clamp must NOT fire there: the CMB
    liquidus margin is tested two-sided (``abs``) so a deeply sub-liquidus
    CMB cell (large negative ``cmb_margin_to_liq``) does not trip it, which
    would otherwise pin CVODE to 1-yr steps for the entire post-solidus
    thermal history.

    Parameters
    ----------
    near_liq, near_sol : bool
        Any staggered cell within ``entropy_margin`` of the liquidus / solidus.
    in_mushy : bool
        Any staggered cell strictly inside the two-phase window.
    cmb_margin_to_liq : float
        CMB-cell entropy minus liquidus entropy at the CMB pressure
        [J kg^-1 K^-1]; negative when the CMB cell is sub-liquidus.
    cmb_margin_to_sol : float
        CMB-cell entropy minus solidus entropy at the CMB pressure
        [J kg^-1 K^-1]; positive when the CMB cell is super-solidus.
    entropy_margin : float
        Proximity band in specific-entropy units [J kg^-1 K^-1] within which a
        cell counts as near a phase boundary. Required (no default), so the
        production margin the caller resolves is never silently shadowed by a
        signature default the real path does not use.

    Returns
    -------
    bool
        True when ``max_step`` should be reduced to 1 yr.
    """
    cmb_near_liq = abs(cmb_margin_to_liq) < entropy_margin
    cmb_in_mushy = cmb_margin_to_liq < 0.0 and cmb_margin_to_sol > 0.0
    return bool(near_liq or near_sol or in_mushy or cmb_near_liq or cmb_in_mushy)


def _mantle_mass_fraction(xi: npt.NDArray) -> npt.NDArray:
    """Normalised mass coordinate for a staggered mass-radius array.

    The staggered coordinate ``xi`` is length-dimensioned with
    ``xi**3 = r_core**3 + 3 M(r_core, r) / rho_avg``, so ``xi**3`` is affine in
    the enclosed mantle mass. Normalising by the array endpoints,

        f = (xi**3 - xi[0]**3) / (xi[-1]**3 - xi[0]**3),

    cancels the average density and returns a monotone ``[0, 1]`` coordinate in
    which equal ``f`` means equal enclosed mass measured from ``xi[0]``. The
    endpoints ``xi[0]`` and ``xi[-1]`` are the innermost and outermost staggered
    cell centres, not the true core and surface radii, so ``f`` is an affine
    proxy for the enclosed mantle mass fraction: exact as a normalised mass
    coordinate spanning the staggered endpoints under the ``xi**3``-affine-in-
    mass relation, and within about one percent of the fraction referenced to
    the true ``[r_core, r_surf]`` boundaries because the endpoints sit half a
    cell inside them. Interpolating a carried entropy field on ``f`` rather than
    raw ``xi`` keeps each parcel's entropy when the mesh deforms, and conserves
    the mass-weighted mean entropy when the total mantle mass between the
    endpoints is unchanged across the re-solve, which is the SPIDER
    fixed-mass-grid parity condition.

    Parameters
    ----------
    xi : npt.NDArray
        Staggered mass-radius coordinates [m], monotone in radius.

    Returns
    -------
    npt.NDArray
        Normalised mass coordinate at each node, same shape as ``xi``. Zeros
        when every node coincides (zero mass span); the caller treats that
        degenerate case as 'no remap' rather than interpolating on it.
    """
    xi3 = np.asarray(xi, dtype=float).ravel() ** 3
    r3_core = min(xi3[0], xi3[-1])
    r3_surf = max(xi3[0], xi3[-1])
    span = r3_surf - r3_core
    if span <= 0.0:
        # Degenerate (all nodes coincident): no meaningful mass fraction.
        return np.zeros_like(xi3)
    return (xi3 - r3_core) / span


class _PhiCapRootFunction(_CV_RootFunction):
    """SUNDIALS root function for the per-call ΔΦ_global cap.

    Fires (g[0]=0) when the mass-weighted |ΔΦ_global(t, y)| relative to
    the start-of-solve value crosses the configured cap. CVODE returns
    at the exact crossing time, so the achieved |ΔΦ_global| equals
    the cap to within CVODE's root tolerance and there is no rate
    estimation involved. Per-cell weights are ρ_i(P, S) · V_i evaluated
    at the live state, matching the helpfile's mass-weighted Phi_global
    metric.

    The integrator passes nondimensional ``t`` and ``y``; this class
    re-scales ``y`` to physical entropy via ``state_scale`` before the
    EOS lookup.

    Parameters
    ----------
    eos : EntropyEOS
        Source for ρ(P, S) and Φ(P, S) lookups.
    P_stag : np.ndarray
        Pressure at staggered nodes [Pa].
    volume : np.ndarray
        Per-cell volume [m^3] at staggered nodes (same length as
        ``P_stag``).
    n_stag : int
        Number of staggered cells. The first ``n_stag`` entries of
        ``y`` are entropy; later entries are the extended-state
        boundary terms (energy_balance, gradient modes) which are not
        needed for Φ_global.
    phi0_global : float
        Φ_global at the start of solve(). Pre-computed by ``solve()``
        before the rootfn is constructed so the cap is anchored to
        the entry state.
    cap : float
        Maximum |ΔΦ_global| permitted per solve() call.
    state_scale : np.ndarray
        Element-wise nondim scaling for the state vector. The first
        ``n_stag`` entries are entropy scales (S_ref); later entries
        scale the boundary state.
    """

    def __init__(
        self,
        eos,
        P_stag,
        volume,
        n_stag,
        phi0_global,
        cap,
        state_scale,
        phi0_per_cell=None,
        cap_temperature=0.0,
        T0_per_cell=None,
        cap_entropy=0.0,
        S0_per_cell=None,
    ):
        self.eos = eos
        self.P_stag = np.asarray(P_stag, dtype=float).ravel()
        self.volume = np.asarray(volume, dtype=float).ravel()
        self.n_stag = int(n_stag)
        self.phi0 = float(phi0_global)
        self.cap = float(cap)
        self.state_scale = np.asarray(state_scale, dtype=float).ravel()
        # Per-cell melt fraction at solve entry. The cap fires on the
        # larger of the global mass-weighted |ΔΦ| and the maximum
        # single-cell |Δφ|, so a deep cell that crosses the whole
        # two-phase window in one step (a negligible move in the global
        # mean) still returns control to the integrator before the jump.
        self.phi0_cell = (
            None if phi0_per_cell is None else np.asarray(phi0_per_cell, dtype=float).ravel()
        )
        # Optional per-cell temperature and entropy step caps, anchored at
        # solve entry. The melt-fraction cap goes blind once a cell reaches
        # phi=0 or phi=1 (its derivative saturates), so it cannot bound the
        # temperature drop as a cell cools on the solid adiabat just below
        # the solidus. The temperature cap bounds max_i |T_i - T0_i| directly,
        # and the entropy cap bounds max_i |S_i - S0_i| in the native solver
        # variable. Whichever cap is tightest at a given state arrests the
        # integrator (g is the minimum of the active margins).
        self.cap_T = float(cap_temperature)
        self.T0_cell = (
            None if T0_per_cell is None else np.asarray(T0_per_cell, dtype=float).ravel()
        )
        self.cap_S = float(cap_entropy)
        self.S0_cell = (
            None if S0_per_cell is None else np.asarray(S0_per_cell, dtype=float).ravel()
        )
        # Hit counter for diagnostics; CVODE may evaluate the rootfn
        # at every internal step so this can grow large.
        self.evals = 0
        # Mass-total at start (constant within one solve); cached for
        # repeated mass-weight normalisation.
        self._mass_total_at_start = -1.0  # filled on first evaluate()

    def evaluate(self, t, y, g, userdata=None):
        try:
            y_arr = np.asarray(y, dtype=float)
            S_stag = y_arr[: self.n_stag] * self.state_scale[: self.n_stag]
            # g is the minimum margin across the active caps; CVODE fires
            # when any one is reached. Each cap's EOS lookups are computed
            # only when that cap is enabled.
            margin = np.inf
            if self.cap > 0.0:
                phi = np.asarray(self.eos.melt_fraction(self.P_stag, S_stag)).ravel()
                rho = np.asarray(self.eos.density(self.P_stag, S_stag)).ravel()
                mass = rho * self.volume
                mass_total = float(mass.sum())
                phi_global = (
                    float((mass * phi).sum() / mass_total) if mass_total > 0.0 else self.phi0
                )
                dphi = abs(phi_global - self.phi0)
                if self.phi0_cell is not None and phi.size == self.phi0_cell.size:
                    dphi = max(dphi, float(np.max(np.abs(phi - self.phi0_cell))))
                margin = min(margin, self.cap - dphi)
            if (
                self.cap_S > 0.0
                and self.S0_cell is not None
                and S_stag.size == self.S0_cell.size
            ):
                margin = min(margin, self.cap_S - float(np.max(np.abs(S_stag - self.S0_cell))))
            if self.cap_T > 0.0 and self.T0_cell is not None:
                T = np.asarray(self.eos.temperature(self.P_stag, S_stag)).ravel()
                if T.size == self.T0_cell.size:
                    margin = min(margin, self.cap_T - float(np.max(np.abs(T - self.T0_cell))))
            g[0] = margin if np.isfinite(margin) else 1.0
            self.evals += 1
            return 0
        except Exception:  # pragma: no cover - never crash CVODE on rootfn
            # Returning a positive value keeps CVODE integrating; the
            # cap simply does not fire if EOS lookup fails. The outer
            # solve() result will still be correct. The largest armed cap is
            # a safe positive sentinel (at least one cap is positive whenever
            # this rootfn was constructed).
            g[0] = max(self.cap, self.cap_T, self.cap_S)
            return 0


def _phi_cap_event_factory(
    eos,
    P_stag,
    volume,
    n_stag,
    phi0_global,
    cap,
    state_scale,
    phi0_per_cell=None,
    cap_temperature=0.0,
    T0_per_cell=None,
    cap_entropy=0.0,
    S0_per_cell=None,
):
    """Build a scipy ``solve_ivp`` event for the per-cell step caps.

    ``solve_ivp`` events are callables ``(t, y) -> float`` with sign
    crossings that trigger termination when ``terminal=True``. The
    semantics mirror :class:`_PhiCapRootFunction`: the event value is the
    minimum margin across the active caps, ``cap - max(|ΔΦ_global|,
    max_i |Δφ_i|)`` and, when enabled, ``cap_temperature - max_i |ΔT_i|``
    and ``cap_entropy - max_i |ΔS_i|``. Crossing zero from positive to
    negative terminates the integration, so a single deep cell crossing the
    two-phase window or cooling on the solid adiabat arrests the step even
    when the global mean barely moves.
    """
    P_stag_arr = np.asarray(P_stag, dtype=float).ravel()
    volume_arr = np.asarray(volume, dtype=float).ravel()
    n_stag_int = int(n_stag)
    phi0 = float(phi0_global)
    cap_float = float(cap)
    scale_arr = np.asarray(state_scale, dtype=float).ravel()
    phi0_cell = (
        None if phi0_per_cell is None else np.asarray(phi0_per_cell, dtype=float).ravel()
    )
    cap_T = float(cap_temperature)
    T0_cell = None if T0_per_cell is None else np.asarray(T0_per_cell, dtype=float).ravel()
    cap_S = float(cap_entropy)
    S0_cell = None if S0_per_cell is None else np.asarray(S0_per_cell, dtype=float).ravel()

    def _event(t_nd, y_nd):
        try:
            y_arr = np.asarray(y_nd, dtype=float)
            S_stag = y_arr[:n_stag_int] * scale_arr[:n_stag_int]
            margin = np.inf
            if cap_float > 0.0:
                phi = np.asarray(eos.melt_fraction(P_stag_arr, S_stag)).ravel()
                rho = np.asarray(eos.density(P_stag_arr, S_stag)).ravel()
                mass = rho * volume_arr
                mass_total = float(mass.sum())
                phi_global = (
                    float((mass * phi).sum() / mass_total) if mass_total > 0.0 else phi0
                )
                dphi = abs(phi_global - phi0)
                if phi0_cell is not None and phi.size == phi0_cell.size:
                    dphi = max(dphi, float(np.max(np.abs(phi - phi0_cell))))
                margin = min(margin, cap_float - dphi)
            if cap_S > 0.0 and S0_cell is not None and S_stag.size == S0_cell.size:
                margin = min(margin, cap_S - float(np.max(np.abs(S_stag - S0_cell))))
            if cap_T > 0.0 and T0_cell is not None:
                T = np.asarray(eos.temperature(P_stag_arr, S_stag)).ravel()
                if T.size == T0_cell.size:
                    margin = min(margin, cap_T - float(np.max(np.abs(T - T0_cell))))
            return margin if np.isfinite(margin) else 1.0
        except Exception:
            return max(cap_float, cap_T, cap_S)

    _event.terminal = True
    _event.direction = -1.0  # only fire on positive-to-negative crossing
    return _event


@dataclass
class SolverOutput:
    """Complete output from one EntropySolver integration step.

    This dataclass is the public contract between Aragog and PROTEUS.
    All quantities needed by the coupling wrapper are included here,
    so callers never need to reach into solver internals.
    """

    # Profiles at staggered nodes
    S_final: npt.NDArray  # entropy [J/kg/K]
    T_stag: npt.NDArray  # temperature [K]
    phi_stag: npt.NDArray  # melt fraction [-]
    rho_stag: npt.NDArray  # density [kg/m^3]
    visc_stag: npt.NDArray  # dynamic viscosity [Pa s]

    # Mesh geometry
    P_stag: npt.NDArray  # pressure at staggered nodes [Pa]
    r_basic: npt.NDArray  # radii at basic nodes [m]
    r_stag: npt.NDArray  # radii at staggered nodes [m]
    vol: npt.NDArray  # shell volumes [m^3]
    mass_stag: npt.NDArray  # mass per shell [kg]

    # Fluxes and heating (at basic / staggered nodes)
    heat_flux: npt.NDArray  # total heat flux at basic nodes [W/m^2]
    heating: npt.NDArray  # internal heating at staggered nodes [W/kg]
    eddy_diff: npt.NDArray  # eddy diffusivity at basic nodes [m^2/s]
    cap_stag: npt.NDArray  # capacitance rho*T at staggered nodes

    # Per-component flux decomposition at basic nodes for diagnostic
    # output. Populated from the final EntropyState after integration;
    # only consumers that set ``write_flux_diagnostics = true`` in the
    # PROTEUS config read these.
    jcond_b: npt.NDArray  # conductive flux [W/m^2]
    jconv_b: npt.NDArray  # convective flux [W/m^2]
    jgrav_b: npt.NDArray  # grav-sep contribution to heat flux [W/m^2]
    jmix_b: npt.NDArray  # SPIDER-parity mixing flux [W/m^2]
    dSdr_b: npt.NDArray  # entropy gradient [J/kg/K/m]
    phi_basic: npt.NDArray  # EOS melt fraction at basic nodes [-]
    T_basic: npt.NDArray  # temperature at basic nodes [K]
    cp_basic: npt.NDArray  # heat capacity at basic nodes [J/kg/K]
    rho_basic: npt.NDArray  # density at basic nodes [kg/m^3]

    # Scalar quantities
    T_magma: float  # surface temperature [K]
    T_core: float  # CMB temperature [K]
    Phi_global: float  # mass-weighted melt fraction (= M_mantle_liquid / M_mantle)
    Phi_global_vol: float  # porosity-based volumetric melt fraction
    M_mantle: float  # mantle mass [kg]
    M_mantle_liquid: float  # liquid mantle mass [kg]
    M_mantle_solid: float  # solid mantle mass [kg]
    RF_depth: float  # rheological front depth (dimensionless)
    E_th: float  # thermal energy proxy [J]: sum(m * Cp_apparent * T).
    # NOT a proper enthalpy because Cp_apparent is latent-heat-blended and
    # spikes at the rheological transition. Reported as a diagnostic
    # snapshot; use ``E_state`` for conservation work.
    E_state: float  # mantle integrated specific enthalpy [J] -- EOS-consistent
    # ``sum(m * h(P, S))`` from the precomputed ``EntropyEOS.specific_enthalpy``
    # table. Anchor at table corner is arbitrary; differences across time
    # are the conservation diagnostic. Set to NaN when no entropy_eos is
    # attached (e.g. const_properties path).
    E_state_cons: float  # conservation-grade integrated enthalpy [J].
    # Same as ``E_state`` but with the frozen STRUCTURAL mass profile
    # (``mesh.staggered_effective_density * volume``, set once at IC)
    # instead of the state-dependent ``rho(P,S) * volume``. The entropy
    # ODE evolves in fixed-volume Eulerian frame with state-dependent
    # rho, so the natural integrated thermodynamic potential against
    # which the divergence-of-flux budget closes is the one with FROZEN
    # mass weighting. Diff ``E_state_cons(t) - E_state_cons(0)`` should
    # equal the cumulative ``dE_predicted_cons`` (using frozen-mass
    # Q_*_total) to numerical precision; the residual against this is
    # the true conservation check. Set to NaN when no entropy_eos.
    Cp_eff: float  # effective heat capacity [J/kg/K]
    F_heat_total: float  # total heating flux [W/m^2]
    F_cmb: float  # heat flux at CMB (basic node 0) [W/m^2], signed
    # positive-out-of-core; equals the lower-boundary value of ``heat_flux``
    # so the closed-mantle balance dE/dt = -F_int*A_int + F_cmb*A_cmb +
    # Q_radio + Q_tidal can be checked without re-deriving from the
    # array.
    Q_radio_total: float  # mantle-integrated radiogenic power [W]
    Q_tidal_total: float  # mantle-integrated tidal power [W]

    # Per-call energy-balance contributions [J] integrated over the
    # CVODE sub-step trajectory. Replaces end-of-step instantaneous
    # capture as the conservation primitive: PROTEUS just cumulatively
    # sums these across calls instead of trapezoidal-interpolating
    # between possibly-transient end-of-step F_cmb snapshots. Sign
    # convention: positive = energy ADDED to the mantle over the call
    # (so step_dE_F_int_J is negative when the mantle is losing heat
    # to the atmosphere).
    step_dE_F_int_J: float  # = -∫ F_int * A_int dt [J]
    step_dE_F_cmb_J: float  # = +∫ F_cmb * A_cmb dt [J]
    step_dE_Q_radio_J: float  # = +∫ Q_radio_total dt [J] (state-dependent mass)
    step_dE_Q_tidal_J: float  # = +∫ Q_tidal_total dt [J] (state-dependent mass)
    # Frozen-mass variants for the conservation-grade budget. Identical
    # to ``step_dE_Q_*_J`` except mass-weighting uses ``rho_struct *
    # volume`` (set once at IC) for both ``Q_radio_total`` and
    # ``Q_tidal_total``. Surface-flux integrals (``step_dE_F_int_J``,
    # ``step_dE_F_cmb_J``) are area-only and do not need a frozen-mass
    # variant.
    step_dE_Q_radio_cons_J: float  # = +∫ Q_radio_total_cons dt [J] (frozen mass)
    step_dE_Q_tidal_cons_J: float  # = +∫ Q_tidal_total_cons dt [J] (frozen mass)

    # Per-call entropy-equation self-consistency residual [J].
    # Accumulates ``∫ (Σ capacitance (dS/dt) V - (-F_int A_int +
    # F_cmb A_cmb + Q_radio + Q_tidal)) dt`` over the substeps with the
    # LHS weighted by the same capacitance (``rho_phase * T``) the RHS
    # used. The discrete flux divergence telescopes to the boundary
    # fluxes, so this is machine-zero by construction; a non-zero value
    # flags a bug in the divergence assembly, not time-integration
    # quality (that is carried by ``E_residual_cons_frac``).
    step_solver_residual_J: float  # = ∫ (LHS - RHS) dt [J]

    # Adiabatic compression work [J] from the structure re-solve that
    # preceded this call. ``Sum mass (h(P_new, S) - h(P_old, S))`` at the
    # carried entropy and frozen mass. Zero for a static structure and for
    # the first solve (no prior mesh). Informational only, deliberately NOT
    # added to the predicted energy: the conservation residual already closes
    # against the entropy-transported heat (``step_dE_state_heat_J``,
    # Sum rho T dS), which carries the same PdV thermodynamic content, so
    # adding compression to the predicted side would double-count it. The
    # enthalpy framing also cannot close the budget on its own, because the
    # two-phase EOS specific enthalpy does not satisfy ``dh = T dS`` across
    # the mushy zone (see ``step_dE_state_heat_J``).
    step_dE_compression_J: float

    # Per-call entropy-transported heat [J]: the change in mantle heat
    # content over the call, ``Sum_i V_i integral_{S0_i}^{Sf_i}
    # rho(P_i, S) T(P_i, S) dS``, evaluated by direct EOS quadrature over
    # the entropy path of each staggered cell. This is the Lagrangian
    # heat content the entropy ODE transports (capacitance rho*T weighting),
    # so its cumulative sum is the state side of the conservation budget the
    # coupler differences against the boundary fluxes and sources. It is NOT
    # the enthalpy change: the two-phase EOS specific enthalpy does not
    # satisfy ``dh = T dS`` across the mushy zone, so an enthalpy snapshot
    # difference carries a spurious flow-work term and cannot close the
    # budget at all. The quadrature uses the hard-masked table density, which
    # differs slightly from the tanh-blended phase density in the ODE
    # capacitance, so the coupler residual closes to roughly a percent of the
    # cumulative cooling rather than to machine precision; machine-precision
    # conservation is carried by ``step_solver_residual_J``. Sign: negative
    # when the mantle is cooling.
    step_dE_state_heat_J: float

    dt_actual: float  # actual integration time [yr]
    status: int  # solver status (0 = success)

    # ── NetCDF output ──────────────────────────────────────────────
    def to_netcdf(
        self,
        path: str | Path,
        *,
        time: float | None = None,
        description: str = 'Aragog SolverOutput snapshot',
    ) -> None:
        """Write this solver snapshot to a NetCDF4 file.

        Produces a self-contained file that captures the full
        ``SolverOutput`` dataclass: scalar diagnostics, staggered-node
        profiles (entropy, temperature, melt fraction, density,
        viscosity, ...), basic-node profiles (heat flux, per-component
        flux decomposition, ...), and run-level metadata (Aragog
        version, optional simulation time, status code). All fields
        carry CF-style ``units`` and ``long_name`` attributes so the
        file is interpretable without consulting the source.

        Designed for the standalone solver path: a script that builds
        an ``EntropySolver``, runs ``solve()``, and wants a portable
        record of the final state. PROTEUS-coupled runs do *not* use
        this; they assemble their own per-iteration helpfile rows on
        the wrapper side. The two writers are independent on purpose
        so the standalone schema can evolve without breaking PROTEUS.

        Parameters
        ----------
        path : str or Path
            Destination NetCDF4 file. Overwrites if the file exists.
        time : float, optional
            Simulation time at which this snapshot was taken [yr].
            Stored as the scalar ``time`` variable; defaults to the
            integrated step duration ``dt_actual`` if not supplied
            (the standalone solver does not carry an absolute clock).
        description : str, optional
            Free-form description string written to the dataset's
            ``description`` global attribute.

        Notes
        -----
        - Uses ``netCDF4`` directly (already a hard dependency); no
          xarray import to keep startup lean.
        - All array fields are written as ``f8`` (float64). The
          integer ``status`` field is stored as ``i4``.
        - Reading back: ``netCDF4.Dataset(path)`` or
          ``xarray.open_dataset(path)`` both work; the file follows the
          CF-1.8 attribute convention.
        """
        from datetime import datetime, timezone

        import netCDF4 as nc

        from aragog import __version__

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with nc.Dataset(path, mode='w') as ds:
            ds.description = description
            ds.aragog_version = __version__
            ds.created_utc = datetime.now(timezone.utc).isoformat(timespec='seconds')
            ds.Conventions = 'CF-1.8'

            n_stag = int(np.asarray(self.S_final).size)
            n_basic = int(np.asarray(self.r_basic).size)
            ds.createDimension('staggered', n_stag)
            ds.createDimension('basic', n_basic)

            def _scalar(
                name: str,
                value: float | int,
                units: str,
                long_name: str,
                *,
                fill_value_nan: bool = False,
            ) -> None:
                dtype = 'i4' if isinstance(value, (int, np.integer)) else 'f8'
                # CF convention: ``_FillValue`` must be set at variable
                # creation time, not after, otherwise netCDF4 raises
                # AttributeError. Pass it via the ``fill_value`` kwarg of
                # createVariable.
                if fill_value_nan and dtype == 'f8':
                    v = ds.createVariable(name, dtype, fill_value=np.nan)
                else:
                    v = ds.createVariable(name, dtype)
                v.units = units
                v.long_name = long_name
                v[...] = value

            def _arr(name: str, value, dim: str, units: str, long_name: str) -> None:
                arr = np.asarray(value, dtype=np.float64).ravel()
                v = ds.createVariable(name, 'f8', (dim,))
                v.units = units
                v.long_name = long_name
                v[:] = arr

            # Time stamp: prefer caller-supplied absolute time, else
            # fall back to the per-call duration.
            t_value = float(time) if time is not None else float(self.dt_actual)
            _scalar('time', t_value, 'yr', 'Simulation time of snapshot')

            # ── Scalar diagnostics ──────────────────────────────────
            _scalar('T_magma', self.T_magma, 'K', 'Surface (magma) temperature')
            _scalar('T_core', self.T_core, 'K', 'Core-mantle boundary temperature')
            _scalar(
                'Phi_global',
                self.Phi_global,
                '1',
                'Mass-weighted global mantle melt fraction (M_liq / M_mantle)',
            )
            _scalar(
                'Phi_global_vol',
                self.Phi_global_vol,
                '1',
                'Porosity-derived volume melt fraction (V_liq / V_mantle)',
            )
            _scalar('M_mantle', self.M_mantle, 'kg', 'Total mantle mass')
            _scalar('M_mantle_liquid', self.M_mantle_liquid, 'kg', 'Liquid mantle mass')
            _scalar('M_mantle_solid', self.M_mantle_solid, 'kg', 'Solid mantle mass')
            _scalar('RF_depth', self.RF_depth, '1', 'Dimensionless rheological-front depth')
            _scalar('E_th', self.E_th, 'J', 'Thermal energy proxy (legacy sum m*Cp_apparent*T)')
            # E_state and E_state_cons are documented to be NaN on the
            # ``const_properties`` path (no entropy EOS attached). Mark
            # them with ``_FillValue = NaN`` so xarray correctly treats
            # an absent value as intentional missing data rather than a
            # silent corrupted float.
            _scalar(
                'E_state',
                self.E_state,
                'J',
                'EOS-consistent integrated mantle enthalpy',
                fill_value_nan=True,
            )
            _scalar(
                'E_state_cons',
                self.E_state_cons,
                'J',
                'Conservation-grade integrated enthalpy with frozen structural mass',
                fill_value_nan=True,
            )
            _scalar('Cp_eff', self.Cp_eff, 'J kg-1 K-1', 'Mass-weighted mean heat capacity')
            _scalar('F_heat_total', self.F_heat_total, 'W m-2', 'Total internal heating flux')
            _scalar('F_cmb', self.F_cmb, 'W m-2', 'Heat flux at the CMB (positive out of core)')
            _scalar(
                'Q_radio_total', self.Q_radio_total, 'W', 'Mantle-integrated radiogenic power'
            )
            _scalar('Q_tidal_total', self.Q_tidal_total, 'W', 'Mantle-integrated tidal power')
            _scalar(
                'step_dE_F_int_J',
                self.step_dE_F_int_J,
                'J',
                'Per-call surface heat-loss energy',
            )
            _scalar(
                'step_dE_F_cmb_J', self.step_dE_F_cmb_J, 'J', 'Per-call CMB heat-gain energy'
            )
            _scalar(
                'step_dE_Q_radio_J', self.step_dE_Q_radio_J, 'J', 'Per-call radiogenic energy'
            )
            _scalar('step_dE_Q_tidal_J', self.step_dE_Q_tidal_J, 'J', 'Per-call tidal energy')
            _scalar(
                'step_dE_Q_radio_cons_J',
                self.step_dE_Q_radio_cons_J,
                'J',
                'Per-call radiogenic energy (frozen-mass weighting)',
            )
            _scalar(
                'step_dE_Q_tidal_cons_J',
                self.step_dE_Q_tidal_cons_J,
                'J',
                'Per-call tidal energy (frozen-mass weighting)',
            )
            _scalar(
                'step_solver_residual_J',
                self.step_solver_residual_J,
                'J',
                'Per-call entropy-ODE balance residual',
            )
            _scalar(
                'step_dE_compression_J',
                self.step_dE_compression_J,
                'J',
                'Per-call compression work from the structure re-solve',
            )
            _scalar(
                'step_dE_state_heat_J',
                self.step_dE_state_heat_J,
                'J',
                'Per-call entropy-transported heat content change (EOS quadrature)',
            )
            _scalar('dt_actual', self.dt_actual, 'yr', 'Actual integration time of this step')
            _scalar('status', int(self.status), '1', 'Solver status code (0 = success)')

            # ── Staggered-node profiles ─────────────────────────────
            _arr('r_stag', self.r_stag, 'staggered', 'm', 'Radius at staggered nodes')
            _arr('P_stag', self.P_stag, 'staggered', 'Pa', 'Pressure at staggered nodes')
            _arr('S_final', self.S_final, 'staggered', 'J kg-1 K-1', 'Specific entropy')
            _arr('T_stag', self.T_stag, 'staggered', 'K', 'Temperature at staggered nodes')
            _arr(
                'phi_stag',
                self.phi_stag,
                'staggered',
                '1',
                'Melt mass fraction at staggered nodes',
            )
            _arr('rho_stag', self.rho_stag, 'staggered', 'kg m-3', 'Density at staggered nodes')
            _arr(
                'visc_stag',
                self.visc_stag,
                'staggered',
                'Pa s',
                'Dynamic viscosity at staggered nodes',
            )
            _arr('vol', self.vol, 'staggered', 'm3', 'Per-shell volume')
            _arr('mass_stag', self.mass_stag, 'staggered', 'kg', 'Per-shell mass')
            _arr('heating', self.heating, 'staggered', 'W kg-1', 'Internal heating rate')
            _arr('cap_stag', self.cap_stag, 'staggered', 'kg K m-3', 'Capacitance rho*T')

            # ── Basic-node profiles ─────────────────────────────────
            _arr('r_basic', self.r_basic, 'basic', 'm', 'Radius at basic nodes')
            _arr('heat_flux', self.heat_flux, 'basic', 'W m-2', 'Total radial heat flux')
            _arr('eddy_diff', self.eddy_diff, 'basic', 'm2 s-1', 'Eddy thermal diffusivity')
            _arr('jcond_b', self.jcond_b, 'basic', 'W m-2', 'Conductive heat flux')
            _arr('jconv_b', self.jconv_b, 'basic', 'W m-2', 'Convective (MLT) heat flux')
            _arr(
                'jgrav_b',
                self.jgrav_b,
                'basic',
                'W m-2',
                'Gravitational-separation heat-flux contribution',
            )
            _arr('jmix_b', self.jmix_b, 'basic', 'W m-2', 'SPIDER-parity phase-mixing flux')
            _arr('dSdr_b', self.dSdr_b, 'basic', 'J kg-1 K-1 m-1', 'Radial entropy gradient')
            _arr('phi_basic', self.phi_basic, 'basic', '1', 'Melt mass fraction at basic nodes')
            _arr('T_basic', self.T_basic, 'basic', 'K', 'Temperature at basic nodes')
            _arr(
                'cp_basic',
                self.cp_basic,
                'basic',
                'J kg-1 K-1',
                'Heat capacity at basic nodes',
            )
            _arr('rho_basic', self.rho_basic, 'basic', 'kg m-3', 'Density at basic nodes')


class EntropySolver:
    """Entropy-based interior dynamics solver.

    Drop-in replacement for Solver (T-based) when using PALEOS P-S tables.
    Same interface: initialize() -> set_initial_entropy() -> solve().
    PROTEUS can swap Solver for EntropySolver without changing the wrapper.

    Parameters
    ----------
    parameters : Parameters
        Parsed configuration (same as T-based Solver).
    entropy_eos : EntropyEOS
        Loaded P-S EOS tables.
    """

    def __init__(self, parameters: Parameters, entropy_eos: EntropyEOS | None = None):
        self.parameters = parameters
        self.entropy_eos = entropy_eos
        self.evaluator: object
        self.state: EntropyState
        self._solution: OptimizeResult
        self.stop_early: bool = False
        # Optional factory that builds JAX-derived CVODE callbacks.
        # Signature: factory(scales, core_bc_mode) -> (rhs_fn, jac_fn)
        # where ``scales`` is an aragog.jax.nondim.NonDimScales
        # instance built by ``_build_nondim_scales``. The factory is
        # registered by PROTEUS via ``set_jax_cvode_factory()`` when
        # ``config.interior_energetics.aragog.use_jax_jacobian`` is True.
        self._jax_cvode_factory = None
        # Number of output points CVODE returns per macro-step solve.
        # The per-call energy integrals trapezoidate the boundary fluxes
        # over this grid; the surface flux decays steeply within a long
        # call, so two endpoints under-resolve it (the F_int integral can
        # be tens of percent wrong over a multi-kyr step, which is the
        # dominant term in ``E_residual_cons_frac``). CVODE interpolates
        # these points from its own internal steps, so a finer output grid
        # sharpens the diagnostic without changing the integration, the
        # final state, or ``dt_actual``. Only the non-root path uses it;
        # when a phi-step-cap root fires the call already stops early.
        self._cvode_output_points = 65
        # Drive CVODE one internal step at a time and hand the per-call
        # energy integrals the full internal trajectory. The output-grid
        # trapezoid under-samples boundary-flux transients shorter than
        # its spacing (a freezing-onset F_cmb spike spans decades in under
        # a thousand years); the internal steps resolve whatever the
        # integrator resolved. False restores the fixed output grid.
        self._substep_integrals = True
        # Compression work [J] from the most recent structure re-solve.
        # When the planet contracts, the static pressure at each frozen
        # mass element rises, so the mantle enthalpy gains the adiabatic
        # compression term ``Sum mass (h(P_new, S) - h(P_old, S))`` at
        # fixed entropy. Exposed as an informational diagnostic only: the
        # coupler does NOT add it to the predicted energy. The conservation
        # residual differences the boundary-flux budget against the
        # entropy-transported heat (``step_dE_state_heat_J``, Sum rho T dS),
        # which is P-jump-agnostic, so no PdV energy is dropped from that
        # check and adding compression to the predicted side would
        # double-count. Zero for a static structure (P unchanged) and for
        # the first solve (no prior mesh).
        self._last_compression_J = 0.0
        # Staggered mass coordinates captured before a structure re-solve.
        # The carried entropy is interpolated from these onto the new mass
        # grid per parcel (``_remap_entropy_to_current_mesh``) so a parcel
        # keeps its entropy when the total mantle mass drifts across the
        # re-solve. One-shot: consumed by the next ``set_initial_entropy``.
        self._xi_pre_resolve = None

    def set_jax_cvode_factory(self, factory) -> None:
        """Register a factory that produces JAX-derived CVODE callbacks.

        The factory is called inside ``solve()`` when ``use_jax_jacobian``
        is enabled in the config. It is given the current nondim scaling
        spec (NonDimScales) and core-BC mode and must return the
        ``(rhs_fn, jac_fn)`` pair accepted by scikits.odes CVODE.

        Parameters
        ----------
        factory : callable
            ``factory(scales, core_bc_mode) -> (rhs_fn, jac_fn)`` where
            ``scales`` is an ``aragog.jax.nondim.NonDimScales`` instance.
            May be None to disable the Option Z path even if the flag
            is on.
        """
        self._jax_cvode_factory = factory

    def _build_nondim_scales(self) -> 'NonDimScales':
        """Construct the per-component NonDimScales for the active state.

        Single source of truth for state and RHS scaling. Mirrors the
        per-state-component scaling rules:

        - quasi_steady core_bc:
              state = [S_0, ..., S_{N-1}],         scale = S_ref each
        - energy_balance core_bc (state_is_extended):
              state = [S_0, ..., S_{N-1}, dSdr_cmb],
              scale = [S_ref, ..., S_ref, dSdr_ref]
        - bower2018 core_bc (state_is_extended):
              state = [S_0, ..., S_{N-1}, T_core],
              scale = [S_ref, ..., S_ref, T_ref]
              (T_ref matches the Bower+2018 Table 1 temperature scale.)
        - gradient core_bc:
              state = [dSdr_0, ..., dSdr_{N}, S_surf],
              scale = [dSdr_ref, ..., dSdr_ref, S_ref]

        ``rhs_scale = t_ref / state_scale`` is derived inside
        NonDimScales.__post_init__; the same dataclass instance feeds
        both the scipy/CVODE wrapper here and the JAX factory.
        """
        from aragog.jax.nondim import NonDimScales

        S_ref = self._S_ref
        t_ref = self._t_ref_yr
        dSdr_ref = self._dSdr_ref
        n_s = self._n_stag
        ss = np.full(len(self._S0), S_ref)
        if self._core_bc == 'gradient':
            nb = n_s + 1
            ss[:nb] = dSdr_ref
            ss[nb] = S_ref
        elif self._state_is_extended:
            ss[:n_s] = S_ref
            if self._core_bc == 'energy_balance':
                ss[n_s] = dSdr_ref
            else:  # bower2018: T_core in [K], not [J/kg/K]; use T_ref
                ss[n_s] = self._T_ref
        return NonDimScales(state_scale=ss, t_ref=float(t_ref))

    @classmethod
    def from_file(cls, filename: str, eos_dir: str, root: str = '') -> 'EntropySolver':
        """Create EntropySolver from a config file and EOS directory.

        Parameters
        ----------
        filename : str
            Path to TOML configuration file.
        eos_dir : str
            Path to directory with SPIDER-format P-S tables.
        root : str
            Root directory for the config file.
        """
        config_path = Path(root) / Path(filename)
        parameters = Parameters.from_file(config_path)
        entropy_eos = EntropyEOS(Path(eos_dir))
        return cls(parameters, entropy_eos)

    def initialize(self) -> None:
        """Initialize mesh, boundary conditions, and entropy state.

        Unlike the T-based Solver, we only need the mesh and BCs from
        the Evaluator. The T-based phase evaluators (which require
        solidus/liquidus files) are replaced by EntropyPhaseEvaluator.
        """
        logger.info('Initializing EntropySolver')
        self._initialize_internals()

    def _initialize_internals(self) -> None:
        """Build mesh, BCs, and entropy phase evaluators."""

        # Build mesh and BCs without the T-based phases
        from aragog.mesh import Mesh

        mesh = Mesh(self.parameters)
        bc = BoundaryConditions(self.parameters, mesh)

        # Create a lightweight evaluator-like object
        class _EntropyEvaluator:
            pass

        self.evaluator = _EntropyEvaluator()
        self.evaluator.mesh = mesh
        self.evaluator.boundary_conditions = bc

        # Cache flattened (1D) mesh arrays for the hot path.
        # Mesh stores (N,1) column vectors; we flatten once here.
        # The truth source for n_stag is mesh.staggered.radii — some
        # mesh builds give mesh.staggered_pressure a different shape
        # (e.g., one extra entry for boundary handling) so we slice
        # the pressure array to match.
        self._n_stag = int(np.asarray(mesh.staggered.radii).shape[0])
        P_basic = np.asarray(mesh.basic_pressure).ravel()
        P_stag = np.asarray(mesh.staggered_pressure).ravel()
        if P_stag.shape[0] != self._n_stag:
            logger.warning(
                'mesh.staggered_pressure length %d != mesh.staggered.radii '
                'length %d, slicing to the latter',
                P_stag.shape[0],
                self._n_stag,
            )
            P_stag = P_stag[: self._n_stag]
        self._area_flat = np.asarray(mesh.basic.area).ravel()
        self._volume_flat = np.asarray(mesh.basic.volume).ravel()
        self._r_basic_flat = np.asarray(mesh.basic.radii).ravel()
        self._r_stag_flat = np.asarray(mesh.staggered.radii).ravel()
        self._P_stag_flat = P_stag
        self._P_basic_flat = P_basic

        # CMB BC mode (set here from config so dSdt can dispatch even
        # before set_initial_entropy is called):
        #   'energy_balance' = SPIDER-parity. State =
        #     [S_0..S_{N-1}, dSdr_cmb]: the CMB entropy gradient is an
        #     ODE state variable evolved by the core energy balance
        #     (bc.c:76-131). Drives dSdr_cmb toward zero as the core
        #     tracks the mantle, preventing the CMB cell drain seen
        #     in quasi_steady mode where the FD-derived dSdr[0]
        #     overshoots at the crystallisation front.
        #   'quasi_steady' = alpha-factor BC. Faster but produces a
        #     ~18 % T_core offset versus SPIDER due to CMB cell drain.
        #   'bower2018' = T_core as ODE state with conduction-only
        #     F_cmb. Available for parity testing; not recommended.
        self._core_bc = getattr(
            self.parameters.boundary_conditions, 'core_bc', 'energy_balance'
        )
        # Per-node gravity profile when an external mesh file carries
        # eos_gravity (UserDefinedEOS / Zalmoxis path); otherwise the
        # scalar gravitational_acceleration is broadcast across the
        # mesh. Interpolation from the external eos_radius grid onto
        # aragog's basic and staggered node radii keeps gravity
        # consistent with the other node-aligned fields.
        g_scalar = abs(
            float(
                getattr(
                    mesh.eos,
                    '_gravitational_acceleration',
                    self.parameters.mesh.gravitational_acceleration,
                )
            )
        )
        eos_gravity_arr = np.asarray(
            getattr(self.parameters.mesh, 'eos_gravity', []), dtype=float
        ).ravel()
        eos_radius_arr = np.asarray(
            getattr(self.parameters.mesh, 'eos_radius', []), dtype=float
        ).ravel()
        if eos_gravity_arr.size > 1 and eos_radius_arr.size == eos_gravity_arr.size:
            if not np.all(np.diff(eos_radius_arr) > 0):
                raise ValueError(
                    'External eos_radius is not monotonically increasing; '
                    'PchipInterpolator would silently corrupt the numpy '
                    'gravity profile. Fix by sorting the external EOS '
                    'file columns 0 and 3 so radius is monotonic.'
                )
            # PchipInterpolator (C^1 monotone cubic Hermite) for parity
            # with the pressure/density interpolations in mesh.pressure_eos
            # and the JAX path in jax.phase._build_gravity_array. Gravity
            # is smooth wherever density is smooth, so the cubic Hermite
            # carries no overshoot risk like the solidus/liquidus
            # entropy curves (where PCHIP was rejected upstream).
            g_interp = PchipInterpolator(eos_radius_arr, eos_gravity_arr)
            g_basic = g_interp(self._r_basic_flat)
            g_stag = g_interp(self._r_stag_flat)
            logger.info(
                'EntropySolver gravity: per-node profile from external mesh '
                '(n=%d, basic range [%.4f, %.4f] m/s^2)',
                eos_gravity_arr.size,
                float(g_basic.min()),
                float(g_basic.max()),
            )
        else:
            g_basic = np.full(self._r_basic_flat.shape, g_scalar)
            g_stag = np.full(self._r_stag_flat.shape, g_scalar)
            logger.info(
                'EntropySolver gravity: scalar broadcast (%.4f m/s^2) — '
                'external eos_gravity not available',
                g_scalar,
            )

        # Create entropy phase evaluators for staggered and basic nodes.
        # cp_blend selects how Cp is computed in the mushy zone:
        #   'latent' = SPIDER-parity, latent-heat-augmented Cp
        #   'linear' = linear blend of pure-phase Cp (no latent term)
        cp_blend = getattr(self.parameters.phase_mixed, 'cp_blend', 'latent')

        phase_kwargs = dict(
            entropy_eos=self.entropy_eos,
            rheological_transition_melt_fraction=(
                self.parameters.phase_mixed.rheological_transition_melt_fraction
            ),
            rheological_transition_width=(
                self.parameters.phase_mixed.rheological_transition_width
            ),
            grain_size=self.parameters.phase_mixed.grain_size,
            cp_blend=cp_blend,
        )

        # Get viscosity and thermal conductivity from config.
        # Values can be plain floats (from PROTEUS) or string expressions
        # (from Aragog's legacy .cfg parser with .eval()).  Try float()
        # first; fall back to .eval() for legacy strings.
        phase_kwargs['viscosity_solid'] = _phase_prop_float(
            self.parameters.phase_solid.viscosity, 1e21
        )
        phase_kwargs['viscosity_liquid'] = _phase_prop_float(
            self.parameters.phase_liquid.viscosity, 1e-1
        )
        # matprop_smooth_width: SPIDER's phase-boundary smoothing for
        # material properties. The ``getattr`` default of 0 (no
        # smoothing) matches the SPIDER convention; the production
        # JAX setting is 0.01 and is set on the ``[phase_mixed]``
        # block of the active config.
        phase_kwargs['matprop_smooth_width'] = float(
            getattr(self.parameters.phase_mixed, 'matprop_smooth_width', 0.0)
        )
        cond_s = _phase_prop_float(self.parameters.phase_solid.thermal_conductivity, None)
        if cond_s is not None:
            phase_kwargs['thermal_conductivity_solid'] = cond_s
        cond_l = _phase_prop_float(self.parameters.phase_liquid.thermal_conductivity, None)
        if cond_l is not None:
            phase_kwargs['thermal_conductivity_liquid'] = cond_l

        # Constant-properties mode (SPIDER -use_const_properties parity)
        _const = getattr(self.parameters.phase_mixed, 'const_properties', False)
        if _const:
            phase_kwargs['const_properties'] = True
            for k in (
                'const_rho',
                'const_Cp',
                'const_alpha',
                'const_cond',
                'const_log10visc',
                'const_T_ref',
                'const_S_ref',
            ):
                phase_kwargs[k] = float(getattr(self.parameters.phase_mixed, k))

        phase_stag = EntropyPhaseEvaluator(
            gravitational_acceleration=g_stag,
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
            radionuclides=energy.radionuclides,
            tidal=energy.tidal,
            tidal_array=getattr(energy, 'tidal_array', [0.0]),
            eddy_diffusivity_thermal=getattr(energy, 'eddy_diffusivity_thermal', 1.0),
            eddy_diffusivity_chemical=energy.eddy_diffusivity_chemical,
            kappah_floor=energy.kappah_floor,
            bottom_up_grav_sep=getattr(energy, 'bottom_up_grav_sep', True),
            phase_smoothing=getattr(energy, 'phase_smoothing', 'tanh'),
        )
        # Store radionuclide data for the state's heating computation.
        # ``Parameters.radionuclides`` is a required dataclass field, so
        # the attribute is always present (an empty list when no
        # radionuclides are configured).
        self.evaluator.radionuclides = self.parameters.radionuclides

        # One-shot scan of the loaded EOS tables against the runtime
        # floors (Cp >= 100 J/kg/K, S_liq - S_sol >= 1 J/kg/K). Catches
        # a non-standard or partially-loaded EOS at load time, before
        # any RHS call where the floors could silently bias the
        # integrand.
        self._check_eos_floors()

        # Cache constant BC and mesh terms so the dSdt hot path
        # doesn't recompute them on every RHS call. None of these
        # depend on entropy.
        self._cache_bc_constants()

        # Nondimensionalisation scales. S_ref matches Bower+2018 Table 1
        # exactly (S_0 = 2993 J/kg/K). t_ref is an Aragog-specific choice
        # (1e5 s ~ 54x the SPIDER value of 1834 s = R_0 (S_0 T_0)^{-1/2});
        # both give an O(1) nondim state vector for CVODE, so the choice
        # is internally arbitrary as long as the same value is used in
        # every path that nondimensionalises (NonDimScales handles this).
        # T_ref matches Bower+2018 Table 1 (T_0 = 4034 K) and is used as
        # the nondim scale for the bower2018 core_bc T_core component;
        # using S_ref there would be a units mismatch (J/kg/K vs K).
        # Why nondim at all: in physical units S ~ 3000 J/kg/K, meaning
        # atol=1e-10 demands ~13 significant digits on a 4-digit number.
        # With nondim, atol=1e-10 needs only ~10 digits on an O(1) number.
        self._S_ref = 2993.025100070677  # entropy0 [J/kg/K] (Bower+2018 Table 1)
        self._T_ref = 4034.0  # temperature0 [K] (Bower+2018 Table 1)
        self._t_ref_yr = 1e5 / SECS_PER_YEAR  # time0 [yr]; Aragog convention, not SPIDER
        self._r_ref = 6.371e7  # radius0 [m]
        self._dSdr_ref = self._S_ref / self._r_ref  # [J/kg/K/m]

    def _check_eos_floors(self) -> None:
        """Scan loaded EOS tables for proximity to runtime floors.

        Conduction clamps Cp from below at 100 J/kg/K
        (``solver/entropy_state.py`` and ``jax/phase.py``) and the
        two-phase fraction denominator clamps ``S_liq - S_sol`` from
        below at 1 J/kg/K. Both floors guard pathological lookups; a
        production MgSiO3 P-S table stays far above either bound. If a
        loaded table sits close to the floor, the floor will silently
        bias the conduction superadiabatic term or the phi denominator
        once the integrator queries that region.

        Emits one warning per offending quantity, at table-load time,
        so a non-standard EOS is flagged before any RHS call rather
        than per call. Skipped under ``const_properties`` and when
        ``self.entropy_eos`` is None (test stubs).
        """
        eos = self.entropy_eos
        if eos is None or not hasattr(eos, '_tables'):
            return
        cp_warn_threshold = 200.0  # 2x the runtime floor (100 J/kg/K)
        ds_warn_threshold = 10.0  # 10x the runtime floor (1 J/kg/K)
        cp_min = float('inf')
        for phase in ('solid', 'melt'):
            key = f'heat_capacity_{phase}'
            table = eos._tables.get(key)
            if table is None:
                continue
            values = np.asarray(table.get('values', []), dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                cp_min = min(cp_min, float(finite.min()))
        if np.isfinite(cp_min) and cp_min < cp_warn_threshold:
            logger.warning(
                'EntropySolver EOS check: min Cp = %.1f J/kg/K in the '
                'loaded P-S tables, within 2x of the 100 J/kg/K runtime '
                'floor. Conduction superadiabatic term will be biased '
                'upward where Cp clips. Verify the EOS heat-capacity '
                'tables.',
                cp_min,
            )
        sol_dict = getattr(eos, '_solidus', None)
        liq_dict = getattr(eos, '_liquidus', None)
        if sol_dict is not None and liq_dict is not None:
            sol_P = np.asarray(sol_dict.get('P', []), dtype=float).ravel()
            sol_S = np.asarray(sol_dict.get('S', []), dtype=float).ravel()
            liq_P = np.asarray(liq_dict.get('P', []), dtype=float).ravel()
            liq_interp = liq_dict.get('interp')
            if sol_P.size and sol_S.size and liq_P.size and liq_interp is not None:
                # Restrict the scan to the intersection of the solidus
                # and liquidus P ranges. _load_spider_phase_boundary
                # builds the liquidus interpolator with bounds_error=
                # False and endpoint fill_values, so a sol_P entry
                # outside [liq_P.min(), liq_P.max()] returns the
                # liquidus endpoint S, which can be smaller than
                # S_sol(P) at the same P, producing a spurious
                # negative ds and a false phase-collapse warning.
                P_lo = max(float(sol_P.min()), float(liq_P.min()))
                P_hi = min(float(sol_P.max()), float(liq_P.max()))
                mask = (sol_P >= P_lo) & (sol_P <= P_hi)
                sol_P_in = sol_P[mask]
                sol_S_in = sol_S[mask]
                if sol_P_in.size:
                    liq_S_at_sol = np.asarray(liq_interp(sol_P_in), dtype=float).ravel()
                    ds = liq_S_at_sol - sol_S_in
                    ds = ds[np.isfinite(ds)]
                    if ds.size:
                        ds_min = float(ds.min())
                        if ds_min < ds_warn_threshold:
                            logger.warning(
                                'EntropySolver EOS check: min '
                                '(S_liq - S_sol) = %.2f J/kg/K over '
                                'the solidus-liquidus P intersection, '
                                'within 10x of the 1 J/kg/K runtime '
                                'floor. The two-phase fraction '
                                'denominator will clip where the '
                                'solidus and liquidus collapse. '
                                'Verify phase-boundary tables.',
                                ds_min,
                            )

    def _cache_bc_constants(self) -> None:
        """Pre-compute BC and mesh constants for the dSdt hot path.

        Lifts core_density, core_heat_capacity, area_cmb, M_core,
        r_cmb, r_above, dr_half, vol_first, tfac_core_avg, and the
        outer/inner BC dispatch keys out of the per-RHS-call path.
        Called from ``_initialize_internals`` after BC + mesh setup.

        Hoisting these onto the solver instance is worth a few
        microseconds per RHS evaluation, which compounds over the
        thousand-plus calls per coupling step in stiff regimes.
        """
        bc = self.evaluator.boundary_conditions._settings
        mesh = self.evaluator.mesh

        # Mesh geometry at the CMB
        r_cmb = float(self._r_basic_flat[0])
        r_above = float(self._r_basic_flat[1])
        self._cmb_r_cmb = r_cmb
        self._cmb_r_above = r_above
        self._cmb_dr_cmb = r_above - r_cmb  # basic-node spacing at CMB
        self._cmb_dr_half = 0.5 * self._cmb_dr_cmb  # basic-to-staggered half-spacing
        self._cmb_area = 4.0 * np.pi * r_cmb**2
        self._cmb_vol_first = float(self._volume_flat[0])

        # Core properties (constant in time)
        self._core_density = float(mesh.settings.core_density)
        self._core_cp = float(bc.core_heat_capacity)
        self._core_M = (4.0 / 3.0) * np.pi * r_cmb**3 * self._core_density
        self._core_cap = self._core_M * self._core_cp  # J/K
        self._core_tfac = float(getattr(bc, 'tfac_core_avg', 1.147))

        # Quasi-steady BC alpha factor uses (R_above/R_cmb)^2
        self._cmb_radius_ratio_sq = (r_above / r_cmb) ** 2

        # BC dispatch keys captured once
        self._outer_bc_kind = int(bc.outer_boundary_condition)
        self._outer_bc_value = float(bc.outer_boundary_value)
        self._outer_bc_emiss = float(bc.emissivity)
        self._outer_bc_T_eq = float(bc.equilibrium_temperature)
        self._inner_bc_kind = int(bc.inner_boundary_condition)
        self._inner_bc_value = float(bc.inner_boundary_value)

    def reset(self) -> None:
        """Reset for a new integration (PROTEUS coupling loop).

        Re-reads the EOS mesh file if eos_method=2, then rebuilds
        the mesh, BCs, and entropy state. Matches Solver.reset().
        """
        logger.info('Resetting EntropySolver')
        # Snapshot the pre-re-solve pressure, carried entropy and frozen
        # mass so the compression work across the mesh change can be
        # measured once the new mesh is built (see ``_last_compression_J``),
        # and the staggered mass coordinates so the carried entropy can be
        # remapped onto the new mass grid per parcel.
        P_old, S_old, mass_old = self._snapshot_for_compression()
        try:
            self._xi_pre_resolve = self.staggered_mass_coordinates.copy()
        except (AttributeError, TypeError):
            self._xi_pre_resolve = None
        if self.parameters.mesh.eos_method == 2 and self.parameters.mesh.eos_file:
            arr = np.loadtxt(self.parameters.mesh.eos_file)
            self.parameters.mesh.eos_radius = arr[:, 0]
            self.parameters.mesh.eos_pressure = arr[:, 1]
            self.parameters.mesh.eos_density = arr[:, 2]
            self.parameters.mesh.eos_gravity = arr[:, 3]
            _validate_eos_radius_range(self.parameters.mesh)
        self._initialize_internals()
        self._last_compression_J = self._compute_compression_work(P_old, S_old, mass_old)

    def _snapshot_for_compression(self):
        """Capture (P_stag, carried entropy, frozen mass) before a re-solve.

        Returns ``(None, None, None)`` when any piece is unavailable (the
        first reset before any solve, no EOS, or no solution yet), which
        signals a zero compression term.
        """
        if self.entropy_eos is None:
            return None, None, None
        P_old = getattr(self, '_P_stag_flat', None)
        sol = getattr(self, '_solution', None)
        if P_old is None or sol is None or getattr(sol, 'y', None) is None:
            return None, None, None
        if np.size(sol.y) == 0:
            return None, None, None
        try:
            S_block = self.entropy_staggered
            S_old = (S_block[:, -1] if S_block.ndim > 1 else S_block).ravel()
            mesh = self.evaluator.mesh
            mass_old = (
                np.asarray(mesh.staggered_effective_density).ravel()
                * np.asarray(self._volume_flat).ravel()
            )
        except (AttributeError, TypeError, IndexError):
            return None, None, None
        return (
            np.asarray(P_old, dtype=float).ravel().copy(),
            np.asarray(S_old, dtype=float).copy(),
            np.asarray(mass_old, dtype=float).copy(),
        )

    def _compute_compression_work(self, P_old, S_old, mass_old) -> float:
        """Adiabatic compression enthalpy change across a mesh re-solve [J].

        ``Sum mass (h(P_new, S) - h(P_old, S))`` at the carried entropy and
        frozen mass, using the EOS specific-enthalpy table. Returns 0.0 when
        the snapshot is unavailable or the mesh resolution changed (the cell
        count differs, so a per-cell pressure delta is undefined).
        """
        if P_old is None or S_old is None or mass_old is None:
            return 0.0
        P_new = np.asarray(self._P_stag_flat, dtype=float).ravel()
        if not (P_new.shape == P_old.shape == S_old.shape == mass_old.shape):
            return 0.0
        h_old = np.asarray(self.entropy_eos.specific_enthalpy(P_old, S_old)).ravel()
        h_new = np.asarray(self.entropy_eos.specific_enthalpy(P_new, S_old)).ravel()
        return float(np.sum(mass_old * (h_new - h_old)))

    @property
    def staggered_mass_coordinates(self) -> npt.NDArray:
        """Lagrangian mass coordinate at each staggered node.

        Labels each mass parcel independently of physical radius, which
        shifts when the structure re-solves. Used to carry the entropy
        field across a re-solve by parcel rather than by node index.
        """
        return np.asarray(self.evaluator.mesh.staggered.mass_radii).ravel()

    def _remap_entropy_to_current_mesh(self, S_arr: npt.NDArray) -> npt.NDArray:
        """Carry an entropy field onto the current mesh by mass parcel.

        When a structure re-solve changed the mesh, ``reset()`` cached the
        pre-resolve staggered mass coordinates. Interpolate the incoming
        entropy from those onto the current mesh at equal cumulative mantle
        mass fraction ``f`` so each mass parcel keeps its entropy, mirroring
        SPIDER's fixed-mass-grid state. The mapping uses ``f``, not the raw
        length coordinate ``xi``: because ``xi**3`` is affine in enclosed mass
        while ``xi`` is not, equal ``xi`` is not equal enclosed mass once the
        core radius, surface radius, or density profile shift across the
        re-solve, and a raw-``xi`` interpolation mis-places parcels and injects
        a spurious energy jump that grows with the entropy gradient. The cached
        grid is consumed (one-shot) so a later isentropic IC set is unaffected.
        Falls back to the input unchanged when no re-solve grid is cached, the
        node count changed, or the mass grid is unchanged.

        Parameters
        ----------
        S_arr : npt.NDArray
            Staggered specific entropy on the pre-resolve mesh [J/kg/K].

        Returns
        -------
        npt.NDArray
            Entropy carried onto the current mass coordinates, same shape as
            ``S_arr``; the input array unchanged when no remap applies.
        """
        xi_old = getattr(self, '_xi_pre_resolve', None)
        self._xi_pre_resolve = None
        if xi_old is None:
            return S_arr
        try:
            xi_new = self.staggered_mass_coordinates
        except (AttributeError, TypeError):
            return S_arr
        xi_old = np.asarray(xi_old, dtype=float).ravel()
        xi_new = np.asarray(xi_new, dtype=float).ravel()
        if xi_old.shape != S_arr.shape or xi_new.shape != S_arr.shape:
            # A cached pre-resolve grid whose length does not match the entropy
            # array is a coupling inconsistency, not a normal no-remap case.
            # Carry the field through unchanged, but flag it so the anomaly is
            # visible rather than silently masked.
            logger.warning(
                'Entropy remap skipped: shape mismatch between cached grid %s, '
                'current grid %s and entropy array %s; carrying entropy through '
                'unchanged.',
                xi_old.shape,
                xi_new.shape,
                S_arr.shape,
            )
            return S_arr
        if np.allclose(xi_new, xi_old, rtol=1e-12, atol=0.0):
            return S_arr
        # Interpolate on the normalised mass coordinate, not raw mass-radius,
        # so each parcel keeps its entropy when the mesh deforms.
        f_old = _mantle_mass_fraction(xi_old)
        f_new = _mantle_mass_fraction(xi_new)
        if f_old[0] == f_old[-1] or f_new[0] == f_new[-1]:
            # Degenerate mass span (coincident nodes) on the cached or current
            # grid: there is no parcel coordinate to map onto, so interpolating
            # would collapse the field to a single value. Fail safe by carrying
            # the entropy through unchanged.
            logger.warning(
                'Entropy remap skipped: degenerate mass span (coincident nodes) '
                'on the cached or current grid; carrying entropy through '
                'unchanged.'
            )
            return S_arr
        # np.interp requires ascending x; the mass coordinate is monotone in xi.
        if f_old[0] > f_old[-1]:
            return np.interp(f_new[::-1], f_old[::-1], S_arr[::-1])[::-1]
        return np.interp(f_new, f_old, S_arr)

    def set_initial_entropy(self, S_init: npt.NDArray | float) -> None:
        """Set the initial entropy profile and (if used) initial T_core.

        Parameters
        ----------
        S_init : array or float
            Entropy at staggered nodes [J/kg/K]. If scalar, sets uniform
            (isentropic) profile.

        Notes
        -----
        State vector length depends on the active core BC mode:
        ``quasi_steady`` uses length N; ``energy_balance`` and
        ``bower2018`` use length N+1 (entropy block plus one extra
        state variable: ``dSdr_cmb`` or ``T_core`` respectively);
        ``gradient`` uses length N+2. For the bower2018 mode the
        initial ``T_core`` is taken from the bottom-cell mantle
        temperature derived from ``S_init`` via the EOS unless
        ``set_initial_core_temperature`` has been called first.
        """
        # Prefer the cached _n_stag from _initialize_internals; fall
        # back to the mesh accessor for legacy callers that bypass it.
        if hasattr(self, '_n_stag') and self._n_stag is not None:
            n_stag = self._n_stag
        else:
            n_stag = self.evaluator.mesh.staggered.radii.shape[0]
            self._n_stag = n_stag

        if np.isscalar(S_init):
            S_arr = np.full(n_stag, float(S_init))
        else:
            S_arr = np.asarray(S_init, dtype=float)
            if len(S_arr) != n_stag:
                raise ValueError(f'S_init length {len(S_arr)} != mesh staggered nodes {n_stag}')

        # Carry the entropy onto the current mesh by mass parcel when a
        # structure re-solve changed the mass grid (no-op otherwise, and
        # always consumes the cached pre-resolve grid).
        S_arr = self._remap_entropy_to_current_mesh(S_arr)

        # Prefer the cached _core_bc from _initialize_internals.
        if hasattr(self, '_core_bc') and self._core_bc is not None:
            core_bc = self._core_bc
        else:
            core_bc = getattr(self.parameters.boundary_conditions, 'core_bc', 'quasi_steady')
            self._core_bc = core_bc
        logger.debug('set_initial_entropy: core_bc=%r, n_stag=%d', core_bc, n_stag)

        if core_bc == 'energy_balance':
            # SPIDER bit-parity core BC (energy_balance mode).
            # State = [S_0, ..., S_{N-1}, dSdr_cmb]
            # The boundary state dSdr_cmb is the entropy gradient at
            # the CMB basic node (mirror of SPIDER's dSdxi[ind_cmb]).
            # Its time derivative is set by the bc.c:76-131 formula
            # in _dSdt_single from the actual physical heat flux.
            #
            # Cold-start dSdr_cmb_init: use the finite-difference
            # estimate from the staggered cells (one-sided forward
            # difference, since there's no cell below the CMB).
            # The boundary state will then evolve from this to its
            # quasi-equilibrium value over the first few coupling
            # steps as the energy balance constraint kicks in.
            #
            # Hot-start dSdr_cmb_init: preserve from the previous
            # solution if available, so the integrated boundary state
            # survives PROTEUS coupling resets.
            dSdr_cmb_init = getattr(self, '_dSdr_cmb_init', None)
            if dSdr_cmb_init is None:
                # Hot start: preserve from previous solution if shape matches
                prev_sol = getattr(self, '_solution', None)
                if (
                    prev_sol is not None
                    and getattr(prev_sol, 'y', None) is not None
                    and prev_sol.y.size > 0
                    and prev_sol.y.shape[0] == n_stag + 1
                ):
                    dSdr_cmb_init = float(prev_sol.y[n_stag, -1])
                    logger.info(
                        'Preserved dSdr_cmb from previous solve: %.3e J/kg/K/m',
                        dSdr_cmb_init,
                    )
            if dSdr_cmb_init is None:
                # Cold start: one-sided FD of S_init at the bottom.
                # dSdr_cmb ≈ (S_stag[1] - S_stag[0]) / (r_stag[1] - r_stag[0])
                # For a uniform S_init this is exactly zero, which
                # is the correct neutral-buoyancy starting point.
                if n_stag >= 2:
                    r_basic = np.asarray(self.evaluator.mesh.basic.radii).ravel()
                    r_stag_0 = 0.5 * (r_basic[0] + r_basic[1])
                    r_stag_1 = 0.5 * (r_basic[1] + r_basic[2])
                    dSdr_cmb_init = (float(S_arr[1]) - float(S_arr[0])) / max(
                        r_stag_1 - r_stag_0, 1.0
                    )
                else:
                    dSdr_cmb_init = 0.0
                logger.info(
                    'Cold-start dSdr_cmb from FD: %.3e J/kg/K/m',
                    dSdr_cmb_init,
                )
            self._S0 = np.empty(n_stag + 1)
            self._S0[:n_stag] = S_arr
            self._S0[n_stag] = float(dSdr_cmb_init)
            logger.info(
                'Initial state (energy_balance): S_min=%.0f, S_max=%.0f, dSdr_cmb_init=%.3e',
                S_arr.min(),
                S_arr.max(),
                dSdr_cmb_init,
            )
        elif core_bc == 'bower2018':
            # Core temperature as ODE state variable (conduction-only
            # flux). State = [S, T_core]. Available for parity testing
            # only; not recommended for production runs.
            T_core_init = getattr(self, '_T_core_init', None)
            if T_core_init is None:
                prev_sol = getattr(self, '_solution', None)
                if (
                    prev_sol is not None
                    and getattr(prev_sol, 'y', None) is not None
                    and prev_sol.y.size > 0
                    and prev_sol.y.shape[0] == n_stag + 1
                ):
                    T_core_init = float(prev_sol.y[n_stag, -1])
            if T_core_init is None:
                P_bottom = float(self._P_stag_flat[0])
                T_core_init = float(
                    np.asarray(
                        self.entropy_eos.temperature(np.array([P_bottom]), np.array([S_arr[0]]))
                    ).item()
                )
            self._S0 = np.empty(n_stag + 1)
            self._S0[:n_stag] = S_arr
            self._S0[n_stag] = T_core_init
            logger.info(
                'Initial state (bower2018): S_min=%.0f, S_max=%.0f, T_core_init=%.0f K',
                S_arr.min(),
                S_arr.max(),
                T_core_init,
            )
        elif core_bc == 'gradient':
            # Gradient-based formulation mirroring SPIDER's dS/dxi state.
            # State = [dS/dr at N+1 basic nodes, S_surf], length N+2.
            # S at staggered nodes is reconstructed by cumulative sum
            # at each RHS evaluation, providing the global smoothing
            # that prevents BDF overshoot at the solidus.
            mesh = self.evaluator.mesh
            dSdr_init = mesh.d_dr_at_basic_nodes(S_arr).ravel()
            n_basic = len(dSdr_init)
            S_surf = float(S_arr[-1])
            self._S0 = np.empty(n_basic + 1)
            self._S0[:n_basic] = dSdr_init
            self._S0[n_basic] = S_surf
            # Verify roundtrip
            S_check, _ = self._reconstruct_entropy(dSdr_init, S_surf)
            roundtrip_err = float(np.max(np.abs(S_check - S_arr)))
            logger.info(
                'Initial state (gradient): dSdr range [%.3e, %.3e] J/kg/K/m, '
                'S_surf=%.1f, roundtrip err=%.2e J/kg/K',
                dSdr_init.min(),
                dSdr_init.max(),
                S_surf,
                roundtrip_err,
            )
        else:
            # quasi_steady BC: state = [S_0, ..., S_{N-1}]
            self._S0 = S_arr
            logger.info(
                'Initial entropy (quasi_steady BC): S_min=%.0f, S_max=%.0f J/kg/K',
                S_arr.min(),
                S_arr.max(),
            )

    def set_initial_core_temperature(self, T_core_init: float) -> None:
        """Set the initial core temperature (``bower2018`` core_bc only).

        Must be called BEFORE ``set_initial_entropy``. If not called,
        the initial T_core defaults to the bottom-cell mantle
        temperature derived from S_init via the EOS.
        """
        self._T_core_init = float(T_core_init)

    def set_initial_dSdr_cmb(self, dSdr_cmb_init: float | None) -> None:
        """Set the initial CMB entropy gradient (energy_balance mode only).

        Must be called BEFORE ``set_initial_entropy``. If not called,
        the initial ``dSdr_cmb`` is taken from the previous solution
        (if any), else from a one-sided FD of the staggered S_init at
        the bottom (which is zero for a uniform isentrope).

        Pass ``None`` to clear a previously-set override, restoring
        the hot-start behaviour on the next call to
        ``set_initial_entropy``. Used by PROTEUS's retry ladder to
        restore the pre-solve dSdr_cmb after a rejection, then release
        the override so the subsequent coupling step can hot-start
        from its own ``_solution`` normally.
        """
        self._dSdr_cmb_init = None if dSdr_cmb_init is None else float(dSdr_cmb_init)

    def get_current_dSdr_cmb(self) -> float | None:
        """Return the most recent CMB entropy gradient from the solver state.

        Reads ``self._solution.y[n_stag, -1]`` (the final dSdr_cmb from the
        last accepted solve). Returns ``None`` when no solution exists yet,
        or when the state vector lacks the dSdr_cmb slot (core_bc other
        than ``energy_balance``).

        Used by PROTEUS's retry ladder to snapshot the pre-solve
        dSdr_cmb before a sequence of retry attempts and restore it on
        each retry, breaking the positive-feedback loop where a
        rejected attempt's final dSdr_cmb would otherwise become the
        next retry's hot-start IC and drive the boundary state further
        from the pre-solve value each time.
        """
        n_stag = getattr(self, '_n_stag', None)
        prev_sol = getattr(self, '_solution', None)
        if (
            n_stag is None
            or prev_sol is None
            or getattr(prev_sol, 'y', None) is None
            or prev_sol.y.size == 0
            or prev_sol.y.shape[0] != n_stag + 1
        ):
            return None
        return float(prev_sol.y[n_stag, -1])

    def dSdt(
        self,
        time: npt.NDArray | float,
        state_vec: npt.NDArray,
    ) -> npt.NDArray:
        """Time derivative of the full state vector.

        For the ``bower2018`` core BC the state vector is
        ``[S_0, ..., S_{N-1}, T_core]`` of length N+1, and this returns
        ``[dS/dt, dT_core/dt]`` of the same length.

        For the quasi_steady BC the state vector is just
        ``[S_0, ..., S_{N-1}]`` of length N.

        The integrator passes ``vectorized=False``; this RHS handles
        the 1D path only.

        Parameters
        ----------
        time : float
            Time [yr].
        state_vec : array
            Solver state vector [J/kg/K for entropy, K for T_core].
            Shape (N,) or (N+1,) only.

        Returns
        -------
        array
            d(state_vec)/dt with the same shape as the input.
        """
        return self._dSdt_single(time, state_vec)

    def _dSdt_single(
        self,
        time: npt.NDArray | float,
        state_vec: npt.NDArray,
    ) -> npt.NDArray:
        """Time derivative of one state vector column.

        Four CMB BC modes:

        - 'quasi_steady' (default): state = [S_0, ..., S_{N-1}],
          length N. F_cmb is set by the alpha-factor partition of
          F[1].
        - 'energy_balance' (SPIDER bit-parity): state =
          [S_0, ..., S_{N-1}, dSdr_cmb], length N+1. The boundary
          state dSdr_cmb is passed into ``state.update`` so the
          convective+conductive flux at the CMB basic node uses
          the boundary entropy gradient. d/dt(dSdr_cmb) is computed
          from SPIDER's bc.c:76-131 formula.
        - 'gradient': state = [dS/dr at N+1 basic nodes, S_surf],
          length N+2. Mirrors SPIDER's dS/dxi formulation. S at
          staggered nodes is reconstructed by cumulative sum from
          the surface inward at each RHS evaluation.
        - 'bower2018': state = [S, T_core], length N+1. F_cmb from
          conduction-only Fourier law. Available for parity testing
          only; not recommended.
        """
        n_stag = self._n_stag
        gradient_mode = self._core_bc == 'gradient'
        energy_balance = self._core_bc == 'energy_balance'
        bower = self._core_bc == 'bower2018'
        is_extended = energy_balance or bower

        # ── Gradient mode: reconstruct S from the gradient state ──
        if gradient_mode:
            n_basic = n_stag + 1
            dSdr_basic = state_vec[:n_basic]
            S_surf = float(state_vec[n_basic])
            entropy, S_basic = self._reconstruct_entropy(dSdr_basic, S_surf)
            self.state.update(entropy, time, dSdr=dSdr_basic, entropy_basic=S_basic)
        elif is_extended:
            entropy = state_vec[:n_stag]
            extra = float(state_vec[n_stag])
        else:
            entropy = state_vec
            extra = None

        # In energy_balance mode the boundary state IS the entropy
        # gradient, and we pass it through to state.update() so
        # the flux operator at the CMB basic node uses the
        # boundary value rather than the FD-derived estimate.
        if gradient_mode:
            pass  # state.update already called above with dSdr
        elif energy_balance:
            self.state.update(entropy, time, dSdr_cmb=extra)
        else:
            self.state.update(entropy, time)

        # Apply flux BCs directly (not via BC module, which expects 2D arrays).
        # All BC dispatch keys and constants are pre-cached in
        # _cache_bc_constants(); cached values are reused here.

        # Surface: grey-body or prescribed flux
        if self._outer_bc_kind == 1:
            # Grey-body: F = emissivity * sigma * (T_surf^4 - T_eq^4)
            T_surf = self.state.top_temperature.item()
            self.state._heat_flux[-1] = (
                self._outer_bc_emiss * Stefan_Boltzmann * (T_surf**4 - self._outer_bc_T_eq**4)
            )
        elif self._outer_bc_kind == 4:
            # Prescribed flux (PROTEUS coupling). Note: outer_boundary_value
            # is updated in setup_or_update_solver between coupling steps,
            # so re-read from the live BC object instead of the cache.
            self.state._heat_flux[-1] = float(
                self.evaluator.boundary_conditions._settings.outer_boundary_value
            )

        # CMB boundary condition
        if self._inner_bc_kind == 1:
            if gradient_mode or energy_balance:
                # gradient/energy_balance: heat_flux[0] is the physical
                # flux computed from the state-provided dS/dr at the CMB.
                pass
            elif bower:
                # bower2018 BC: F_cmb from one-sided Fourier conduction
                # across the bottom half-cell. Available for parity tests.
                T_above = float(np.asarray(self.state.phase_staggered.temperature()).flat[0])
                k_above = (
                    float(np.asarray(self.state.phase_staggered.thermal_conductivity()).flat[0])
                    if hasattr(self.state.phase_staggered, 'thermal_conductivity')
                    else 4.0
                )
                F_cmb = -k_above * (T_above - extra) / max(self._cmb_dr_half, 1.0)
                self.state._heat_flux[0] = F_cmb
            else:
                # quasi_steady BC: alpha-factor flux partition between
                # the bottom mantle cell and the core.
                rho_first = float(np.asarray(self.state.phase_staggered.density()).flat[0])
                cp_first = float(np.asarray(self.state.phase_staggered.heat_capacity()).flat[0])
                cell_cap = self._cmb_vol_first * rho_first * cp_first  # J/K
                alpha = self._cmb_radius_ratio_sq / (
                    cell_cap / (self._core_cap * self._core_tfac) + 1.0
                )
                self.state._heat_flux[0] = alpha * self.state._heat_flux[1]
        elif self._inner_bc_kind == 2:
            self.state._heat_flux[0] = self._inner_bc_value
        elif self._inner_bc_kind == 3:
            pass  # prescribed T
        else:
            self.state._heat_flux[0] = 0.0  # insulating

        # Flux divergence: dE/dr at staggered nodes.
        energy_flux = self.state.heat_flux * self._area_flat
        delta_energy_flux = np.diff(energy_flux)

        # Capacitance: rho * T * volume (for entropy equation)
        cap = np.asarray(self.state.capacitance_staggered()).ravel()
        capacitance = cap * self._volume_flat

        # dS/dt from flux divergence [J/kg/K/s]
        dSdt = -delta_energy_flux / capacitance

        # Internal heating: dS/dt += H / T
        H = self.state.heating
        T_stag = np.asarray(self.state.phase_staggered.temperature()).ravel()
        dSdt += H / np.maximum(T_stag, 1.0)

        # Convert to J/kg/K/yr
        dSdt *= SECS_PER_YEAR

        # ── Gradient mode: build d/dt(dS/dr) at basic nodes ──
        if gradient_mode:
            r_stag = self._r_stag_flat
            dr_stag = np.diff(r_stag)  # shape (N-1,)

            # Interior basic nodes (i=1..N-1): SPIDER rhs.c:71-72
            # d/dt(dS/dr)[i] = (dSdt_stag[i] - dSdt_stag[i-1]) / dr_stag[i-1]
            rhs_interior = np.diff(dSdt) / dr_stag  # shape (N-1,)

            # CMB basic node (i=0): core energy balance (SPIDER bc.c:113-122)
            F_cmb = float(self.state._heat_flux[0])
            T_cmb = float(np.asarray(self.state.phase_basic.temperature()).flat[0])
            cp_cmb = float(np.asarray(self.state.phase_basic.heat_capacity()).flat[0])
            dSdt_s_cmb_per_s = float(dSdt[0]) / SECS_PER_YEAR
            rhs_cmb_per_s = self._energy_balance_rhs_per_s(
                F_cmb_basic=F_cmb,
                dSdt_s_cmb_per_s=dSdt_s_cmb_per_s,
                T_cmb_basic=T_cmb,
                cp_cmb_basic=cp_cmb,
            )
            rhs_cmb = rhs_cmb_per_s * SECS_PER_YEAR

            # Surface basic node (i=N): copy from adjacent interior
            # (SPIDER bc.c:63 convention)
            rhs_surf = rhs_interior[-1]

            # dS_surf/dt = dSdt at the surface staggered node
            dS_surf_dt = dSdt[-1]

            # Assemble: [rhs_cmb, rhs_interior(N-1), rhs_surf, dS_surf_dt]
            n_basic = n_stag + 1
            rhs = np.empty(n_basic + 1)
            rhs[0] = rhs_cmb
            rhs[1 : n_basic - 1] = rhs_interior
            rhs[n_basic - 1] = rhs_surf
            rhs[n_basic] = dS_surf_dt
            return rhs

        if not is_extended:
            return dSdt

        if energy_balance:
            # SPIDER bc.c boundary-state ODE for the CMB entropy
            # gradient. Extracted into the pure helper
            # `_energy_balance_rhs_per_s` for unit testing. The inputs are
            # the F_cmb heat flux computed by state.update() above
            # (which used the boundary dSdr_cmb to set the CMB basic
            # node entropy), and the dS/dt at the bottom staggered
            # cell (dSdt[0], which we just built from the flux
            # divergence).
            F_cmb_basic = float(self.state._heat_flux[0])
            T_cmb_basic = float(np.asarray(self.state.phase_basic.temperature()).flat[0])
            cp_cmb_basic = float(np.asarray(self.state.phase_basic.heat_capacity()).flat[0])
            dSdt_s_cmb_per_s = float(dSdt[0]) / SECS_PER_YEAR

            d_dSdr_cmb_dt_per_s = self._energy_balance_rhs_per_s(
                F_cmb_basic=F_cmb_basic,
                dSdt_s_cmb_per_s=dSdt_s_cmb_per_s,
                T_cmb_basic=T_cmb_basic,
                cp_cmb_basic=cp_cmb_basic,
            )
            d_dSdr_cmb_dt = d_dSdr_cmb_dt_per_s * SECS_PER_YEAR

            return np.concatenate([dSdt, [d_dSdr_cmb_dt]])

        # bower2018: T_core ODE state with conduction-only F_cmb.
        F_cmb = float(self.state._heat_flux[0])
        dT_core_dt = -F_cmb * self._cmb_area / max(self._core_cap, 1.0)
        dT_core_dt *= SECS_PER_YEAR

        return np.concatenate([dSdt, [dT_core_dt]])

    def _energy_balance_rhs_per_s(
        self,
        F_cmb_basic: float,
        dSdt_s_cmb_per_s: float,
        T_cmb_basic: float,
        cp_cmb_basic: float,
    ) -> float:
        """SPIDER bc.c:76-131 rhs for d(dSdr_cmb)/dt at the CMB.

        Pure numerical helper extracted from ``_dSdt_single`` so the
        bit-parity formula can be unit-tested without building a full
        solver state. Mirrors SPIDER's C code:

            fac_cmb = cp_cmb / (cp_core * T_cmb * tfac * M_core)
            rhs     = (-Etot_cmb + Ecore) * fac_cmb - dSdt_s_cmb
            rhs    *= 2 / dr_cmb

        where ``Etot_cmb = F_cmb_basic * area_cmb`` is the heat flow
        at the CMB basic node, ``Ecore`` is the core internal heat
        source (zero for the simple core-cooling mode used here), and
        ``dr_cmb = r_basic[1] - r_basic[0]`` is the half-cell spacing
        between the CMB basic node and the first staggered node.

        All cached constants (``_core_cp``, ``_core_tfac``, ``_core_M``,
        ``_cmb_area``, ``_cmb_dr_cmb``) are taken from
        ``_cache_bc_constants()``.

        Parameters
        ----------
        F_cmb_basic : float
            Heat flux at the CMB basic node [W/m^2]. Positive = heat
            flowing OUT of the core. This is normally the
            state.update()-derived flux, which uses the boundary
            entropy gradient to set the bottom mantle cell value.
        dSdt_s_cmb_per_s : float
            dS/dt at the bottom staggered cell (cell adjacent to the
            CMB) in J/(kg*K*s). This is ``dSdt[0]`` from the flux
            divergence, converted back from per-year to per-second.
        T_cmb_basic : float
            Temperature at the CMB basic node [K], derived from the
            boundary-entropy via the EOS.
        cp_cmb_basic : float
            Heat capacity at the CMB basic node [J/(kg*K)], derived
            from the boundary-entropy via the EOS.

        Returns
        -------
        float
            d(dSdr_cmb)/dt in J/(kg*K*m*s). The caller is responsible
            for the per-year conversion (multiply by SECS_PER_YEAR).

        Notes
        -----
        The formula is linear in F_cmb_basic and dSdt_s_cmb_per_s, so
        the zero-flux-plus-zero-dSdt limit trivially returns zero.
        This is used as a smoke-test invariant.
        """
        # Etot_cmb in J/s (= W)
        E_tot_cmb = F_cmb_basic * self._cmb_area

        # fac_cmb in 1/(kg*K). The max(1.0, .) clamps prevent divide-
        # by-zero if T_cmb or M_core is unphysically small during IC
        # wind-up. In production T_cmb ~ 4000 K and M_core ~ 2e24 kg,
        # so the clamps are never active.
        fac_cmb = cp_cmb_basic / (
            self._core_cp * max(T_cmb_basic, 1.0) * self._core_tfac * max(self._core_M, 1.0)
        )

        # Ecore = 0 for simple core cooling (no internal heat source).
        # A future release may read radioactive heating from config
        # and pass it in via a new parameter.
        E_core = 0.0

        # dS_basic_cmb/dt from energy balance [J/(kg*K*s)]
        dS_basic_cmb_dt = (-E_tot_cmb + E_core) * fac_cmb

        # d/dt(dS/dr) at the CMB basic node, from the one-sided
        # centered difference between basic and staggered:
        #
        #   d/dt(dS/dr) = (dSdt_stag - dSdt_basic) / (dr/2)
        #               = (dSdt_stag - dSdt_basic) * 2 / dr
        #
        # SPIDER bc.c writes the algebraically equivalent form
        #     rhs = (dSdt_basic - dSdt_stag) * 2 / (xi_cmb - xi_above)
        # where (xi_cmb - xi_above) < 0 because SPIDER's xi
        # coordinate decreases from surface to CMB. The negative
        # denominator flips the sign to match the formula above.
        # Aragog's radii increase from CMB to surface, so dr > 0
        # and the numerator must be (stag - basic) explicitly.
        return (dSdt_s_cmb_per_s - dS_basic_cmb_dt) * 2.0 / self._cmb_dr_cmb

    def _reconstruct_entropy(
        self,
        dSdr_basic: npt.NDArray,
        S_surf: float,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Reconstruct S at staggered and basic nodes from dS/dr gradients.

        Integrates from the surface inward, mirroring SPIDER's
        ``set_entropy_reconstruction_from_ctx`` (util.c:85-126).

        Parameters
        ----------
        dSdr_basic : array, shape (N+1,)
            dS/dr at basic nodes (index 0 = CMB, N = surface).
        S_surf : float
            Entropy at the outermost staggered node (index N-1).

        Returns
        -------
        S_stag : array, shape (N,)
            Reconstructed entropy at staggered nodes.
        S_basic : array, shape (N+1,)
            Reconstructed entropy at basic nodes.
        """
        r_basic = self._r_basic_flat
        r_stag = self._r_stag_flat
        n_stag = len(r_stag)
        n_basic = len(r_basic)

        # Staggered nodes: integrate from surface (index N-1) inward.
        # dSdr[i+1] at basic node i+1 (between stag[i] and stag[i+1])
        # gives S[i] = S[i+1] - dSdr[i+1] * (r_stag[i+1] - r_stag[i]).
        S_stag = np.empty(n_stag)
        S_stag[-1] = S_surf
        dr_stag = np.diff(r_stag)  # shape (N-1,)
        for i in range(n_stag - 2, -1, -1):
            S_stag[i] = S_stag[i + 1] - dSdr_basic[i + 1] * dr_stag[i]

        # Basic nodes: half-cell interpolation from adjacent staggered.
        S_basic = np.empty(n_basic)
        for i in range(1, n_basic - 1):
            S_basic[i] = S_stag[i - 1] + dSdr_basic[i] * (r_basic[i] - r_stag[i - 1])
        # Boundary extrapolation
        S_basic[0] = S_stag[0] + dSdr_basic[0] * (r_basic[0] - r_stag[0])
        S_basic[-1] = S_stag[-1] + dSdr_basic[-1] * (r_basic[-1] - r_stag[-1])

        return S_stag, S_basic

    @property
    def solution(self) -> OptimizeResult | None:
        """Last solve_ivp result, or ``None`` if ``solve()`` has not been
        called yet. Returning ``None`` (instead of raising
        ``AttributeError``) matters for the PROTEUS JAX dispatch path,
        where ``AragogJAXRunner`` handles the actual integration and
        the scipy ``EntropySolver`` lives only to hold Parameters / BC
        state — its ``solve()`` is never invoked, so ``_solution`` is
        never set. Callers already handle ``sol is None``.
        """
        return getattr(self, '_solution', None)

    @property
    def _state_is_extended(self) -> bool:
        """True when the state vector has more than N elements.

        - bower2018 / energy_balance: N+1 (entropy + 1 extra)
        - gradient: N+2 (N+1 gradients + S_surf)
        - quasi_steady: N (entropy only)
        """
        return self._core_bc in ('bower2018', 'energy_balance', 'gradient')

    @property
    def entropy_staggered(self) -> npt.NDArray:
        """Entropy at staggered nodes from the solution.

        For bower2018 and energy_balance modes the solver state vector is
        N+1 in length; we strip the trailing extra row and return
        only the entropy block. For gradient mode, we reconstruct S
        from the gradient state.
        """
        y = self._solution.y
        if self._core_bc == 'gradient':
            n_basic = self._n_stag + 1
            dSdr = y[:n_basic]
            S_surf_arr = y[n_basic]
            S_surf = float(S_surf_arr[-1]) if S_surf_arr.ndim > 0 else float(S_surf_arr)
            S_stag, _ = self._reconstruct_entropy(
                dSdr[:, -1] if dSdr.ndim > 1 else dSdr, S_surf
            )
            return S_stag.reshape(-1, 1) if y.ndim > 1 else S_stag
        if self._state_is_extended:
            return y[: self._n_stag]
        return y

    @property
    def temperature_staggered(self) -> npt.NDArray:
        """Temperature at staggered nodes (derived from S via EOS)."""
        y = self._solution.y
        if self._core_bc == 'gradient':
            S_stag = self.entropy_staggered
            S = S_stag[:, -1] if S_stag.ndim > 1 else S_stag
        elif self._state_is_extended:
            S = y[: self._n_stag, -1] if y.ndim > 1 else y[: self._n_stag]
        else:
            S = y[:, -1] if y.ndim > 1 else y
        if self.entropy_eos is not None:
            return self.entropy_eos.temperature(self._P_stag_flat, S)
        # const_properties: T = T_ref * exp((S - S_ref) / Cp)
        pm = self.parameters.phase_mixed
        return pm.const_T_ref * np.exp((S - pm.const_S_ref) / pm.const_Cp)

    def _build_jac_sparsity(self) -> 'scipy.sparse.spmatrix':
        """Build the Jacobian sparsity pattern for the BDF solver.

        The entropy equation couples node i to its nearest neighbours
        via the flux divergence operator. The bulk Jacobian is
        tridiagonal, extended to pentadiagonal at the boundaries
        for the 3-point d/dr extrapolation stencil.

        For the extended state modes (bower2018 or energy_balance), the
        state vector grows by one element at the end and the
        Jacobian gets an extra row and column:

          - row N (the extra state) couples to S[0] (and S[1] for
            energy_balance via the flux operator extension) and itself
          - rows 0 and 1 (S[0] and S[1]) gain couplings to the
            extra state via the boundary-flux feedback

        With this sparsity hint scipy groups finite-difference
        perturbations by graph colouring, giving ~5 RHS evaluations
        per Jacobian instead of N+1.
        """
        from scipy.sparse import lil_matrix

        # Gradient mode: the reconstruction couples all gradients to all
        # staggered nodes, making the Jacobian dense. Use no sparsity
        # hint (scipy will do full finite-difference Jacobian).
        if self._core_bc == 'gradient':
            return None

        n_stag = self._n_stag
        is_ext = self._state_is_extended
        N = n_stag + (1 if is_ext else 0)

        J = lil_matrix((N, N), dtype=float)
        # Pentadiagonal block for the entropy part
        for i in range(n_stag):
            for k in range(-2, 3):
                j = i + k
                if 0 <= j < n_stag:
                    J[i, j] = 1.0

        if is_ext:
            extra = n_stag  # index of the boundary state
            # Extra state couples to S[0] and S[1] (and S[2] for
            # energy_balance's pentadiagonal reach) and itself.
            J[extra, 0] = 1.0
            J[extra, 1] = 1.0
            if n_stag >= 3:
                J[extra, 2] = 1.0
            J[extra, extra] = 1.0
            # S[0], S[1] gain couplings to the extra state via
            # boundary-flux feedback.
            J[0, extra] = 1.0
            J[1, extra] = 1.0

        return J.tocsc()

    @staticmethod
    def _run_cvode_stepwise(solver, start_time, end_time, y0, n_keep: int = 65536):
        """Drive CVODE one internal step at a time and keep the trajectory.

        The per-call energy integrals trapezoid over whatever trajectory the
        solve returns. A fixed output grid under-samples any boundary-flux
        transient shorter than its spacing, so this driver returns the
        integrator's own accepted steps, which resolve the transient by
        construction. Requires the solver to be built with
        ``one_step_compute=True`` and ``tstop=end_time``.

        Parameters
        ----------
        solver
            A ``scikits_odes_sundials.cvode.CVODE`` instance (``old_api=False``).
        start_time, end_time : float
            Integration interval [s, nondimensional as the caller uses].
        y0 : np.ndarray
            Initial state vector.
        n_keep : int
            Ceiling on stored trajectory points. When exceeded, every other
            interior point is dropped and the keep stride doubles, so memory
            stays bounded while endpoints, root and tstop points are always
            kept exactly.

        Returns
        -------
        SimpleNamespace
            Duck-typed like the ``solve()`` return: ``flag``, ``values.t``
            (1D), ``values.y`` (shape ``(n_time, n_state)``), ``roots=None``,
            ``message``. A fired root is the last trajectory point, so the
            caller's values path reads the advance directly and the
            ``roots``-object workaround for ``solve()`` is not needed.
        """
        t_end = float(end_time)
        y0v = np.asarray(y0, dtype=float).ravel()
        solver.init_step(float(start_time), y0v)
        ts: list[float] = [float(start_time)]
        ys: list[np.ndarray] = [y0v.copy()]
        y_buf = np.empty_like(y0v)
        flag = 0
        message = ''
        stride = 1
        since_kept = 0
        # Generous ceiling: a coupling call that needs more internal steps
        # than this is not integrating, it is thrashing.
        for _ in range(2_000_000):
            ret = solver.step(t_end, y_buf)
            flag = int(ret.flag)
            if flag < 0:
                message = str(getattr(ret, 'message', '')) or 'CVODE step failure'
                break
            t_i = float(ret.values.t)
            y_i = np.asarray(ret.values.y, dtype=float).ravel()
            since_kept += 1
            terminal = flag in (1, 2) or t_i >= t_end
            if since_kept >= stride or terminal:
                ts.append(t_i)
                ys.append(y_i.copy())
                since_kept = 0
            if len(ts) > n_keep:
                ts = ts[:1] + ts[1:-1:2] + ts[-1:]
                ys = ys[:1] + ys[1:-1:2] + ys[-1:]
                stride *= 2
            if terminal:
                break
        else:
            flag = -99
            message = 'stepwise CVODE exceeded the iteration ceiling'
        if flag == 1:
            # TSTOP_RETURN: the integration reached the requested end time,
            # which is the caller's definition of success.
            flag = 0
        values = SimpleNamespace(
            t=np.asarray(ts, dtype=float),
            y=np.asarray(ys, dtype=float),
        )
        return SimpleNamespace(flag=flag, values=values, roots=None, message=message)

    def _solve_cvode(
        self,
        start_time: float,
        end_time: float,
        y0: npt.NDArray,
        atol: float | npt.NDArray,
        rtol: float,
        max_step: float,
        rhs: 'Callable | None' = None,
        cvode_rhs_fn_override: 'Callable | None' = None,
        cvode_jacfn: 'Callable | None' = None,
        phi_cap_rootfn: 'Callable | None' = None,
    ) -> 'OptimizeResult':
        """Integrate the entropy equation using SUNDIALS CVODE.

        Returns a scipy.integrate.OdeResult-compatible OptimizeResult
        so the downstream extraction code (``sol.y``, ``sol.t``,
        ``sol.status``, ``sol.nfev``, ``sol.message``) is unchanged.

        CVODE uses the same BDF family as scipy, but the C
        implementation has a modified-Newton nonlinear solver with
        cached Jacobian factorisation and adaptive order 1..5, so it
        handles the crystallisation-front phase transition without
        collapsing its step size to machine epsilon. SPIDER uses the
        same CVODE (via PETSc's TSSUNDIALS interface).

        Parameters
        ----------
        rhs : callable or None
            If provided, used as the RHS function ``rhs(t, y) -> dydt``
            instead of ``self.dSdt``. Used by ``solve()`` to pass
            the nondimensionalised RHS wrapper.
        cvode_rhs_fn_override : callable or None
            Option Z: if provided, used directly as the CVODE RHS
            callback ``rhs_fn(t, y, ydot) -> int`` (in-place signature).
            Bypasses the local rhs wrapping and the nfev counter (the
            caller is responsible for counting). Used in tandem with
            ``cvode_jacfn`` so CVODE's Newton iteration sees a
            self-consistent (RHS, Jacobian) pair.
        cvode_jacfn : callable or None
            Option Z: if provided, installed as the CVODE Jacobian
            callback so CVODE uses this analytic Jacobian instead of
            its default finite-difference approximation. Signature:
            ``jacfn(t, y, fy, J, user_data=None) -> int``.
        """
        # Zero-span edge case: scipy's solve_ivp accepts
        # `t_span = (t0, t0)` and returns the initial state trivially,
        # but CVODE rejects it with "tout too close to t0". This
        # happens during Aragog's static-init PROTEUS iterations where
        # start_time == end_time == 0. Return a scipy-compatible
        # result without invoking CVODE at all.
        if not (end_time > start_time):
            result = OptimizeResult()
            y_arr = np.asarray(y0, dtype=float).ravel().reshape(-1, 1)
            result.t = np.array([float(start_time)])
            result.y = y_arr
            result.nfev = 0
            result.status = 0
            result.message = 'zero-span solve, returned initial state'
            return result

        # Wrap the RHS into the in-place (t, y, ydot) -> int signature
        # that scikits.odes CVODE expects. Count RHS calls for parity
        # with scipy's sol.nfev.
        _rhs = rhs if rhs is not None else self.dSdt
        nfev_box = [0]

        if cvode_rhs_fn_override is not None:
            # Option Z: caller supplies a fully-formed CVODE RHS
            # callback (e.g. the JAX-backed rhs_fn from
            # build_jax_rhs_and_jacobian). Wrap it so we still count
            # RHS evaluations for the scipy-compatible nfev field.
            _cvode_rhs = cvode_rhs_fn_override

            def rhs_fn(t: float, y: npt.NDArray, ydot: npt.NDArray) -> int:
                rc = _cvode_rhs(t, y, ydot)
                nfev_box[0] += 1
                return rc
        else:

            def rhs_fn(t: float, y: npt.NDArray, ydot: npt.NDArray) -> int:
                ydot[:] = _rhs(t, y)
                nfev_box[0] += 1
                return 0

        # CVODE (via scikits.odes) accepts a per-state 1D array for
        # atol, matching the SUNDIALS interface. Pass it through
        # unchanged so each state component is controlled at its
        # physically correct tolerance. Collapsing to a scalar via
        # np.min (the previous behaviour) defeated the per-component
        # atol built in solve(), causing the tightest component's
        # atol to be applied uniformly to all other components.
        atol_cvode = (
            np.asarray(atol, dtype=float).ravel()
            if isinstance(atol, np.ndarray)
            else float(atol)
        )

        # Build the CVODE solver. lmm_type='BDF' matches SPIDER's
        # TSSundialsSetType(ts, SUNDIALS_BDF). Newton is CVODE's default
        # nonlinear solver; it uses the cached LU factorisation of the
        # iteration matrix I - gamma * J which is the critical
        # robustness advantage over scipy's simpler Newton.
        #
        # Linear solver choice: Aragog's Jacobian is pentadiagonal
        # (bandwidth 2) in quasi_steady mode because the flux divergence
        # operator only couples each staggered node to its 2 nearest
        # neighbours on each side. A dense FD Jacobian needs N=80 RHS
        # evals per Jacobian build; a banded FD Jacobian needs only
        # ~5 (bandwidth+1 per side). This is the CVODE equivalent of
        # the `jac_sparsity` graph-colouring hint scipy uses.
        #
        # Extended-state modes (energy_balance, gradient) break the
        # banded structure because the extra state row couples to
        # far-away entropy nodes, so fall back to dense for those modes.

        cvode_options = {
            'old_api': False,
            'rtol': float(rtol),
            'atol': atol_cvode,
            'lmm_type': 'BDF',
            'nonlinsolver': 'newton',
            'max_steps': 100000,  # per-solve cap; scipy used unlimited
            # Maximum BDF order. BDF orders 1-2 are A-stable
            # (unconditionally stable for stiff problems on stable
            # systems); orders 3-5 are only "stiffly stable" with
            # stability gaps near the imaginary axis. Combustion codes
            # (e.g. PelePhysics) cap at order 2 for robustness on
            # discontinuous-RHS problems. Aragog's RHS has phase-
            # boundary discontinuities that can trip the higher-order
            # stability gaps. Order 2 trades wall time for robustness.
            'order': 2,
            # Give Newton more iterations per step (default 3) and more
            # convergence-failure tolerance (default 10) before CVODE
            # demotes the order. Aragog's RHS has genuine physical
            # stiffness near the crystallisation front, and the default
            # counters trigger premature order-1 lock-in. Higher values
            # let CVODE iterate its way through the stiff transient
            # without permanently dropping order.
            'max_nonlin_iters': 10,
            'max_conv_fails': 25,
        }
        if self._core_bc == 'quasi_steady':
            cvode_options['linsolver'] = 'band'
            cvode_options['lband'] = 2
            cvode_options['uband'] = 2
        else:
            cvode_options['linsolver'] = 'dense'
        # max_step is only meaningful when < 1e100; otherwise CVODE
        # picks its own maximum internal step. This is the SAME
        # max_step value computed phase-aware in solve() at the
        # "Tighten max_step when ANY cell is near a phase boundary"
        # block, and applies uniformly to both the FD-Jacobian path
        # and the JAX-Jacobian (option Z) path, since both reach
        # this dispatch via _solve_cvode.
        if max_step is not None and np.isfinite(max_step):
            cvode_options['max_step_size'] = float(max_step)

        # Install the per-call ΔΦ_global cap as a SUNDIALS root
        # function. CVODE returns at the exact crossing time t* where
        # |Φ_global(t*) − Φ_global(start)| = cap, with flag
        # CV_ROOT_RETURN (=2). The result wrapper below already
        # treats flag==2 as success.
        if phi_cap_rootfn is not None:
            cvode_options['rootfn'] = phi_cap_rootfn
            cvode_options['nr_rootfns'] = 1

        # Option Z: install the analytic Jacobian callback. When
        # provided, CVODE's Newton iteration uses this instead of the
        # default finite-difference Jacobian. Forcing the linear solver
        # to 'dense' is required because the banded solver path assumes
        # CVODE-provided FD banded Jacobian construction; a user-
        # supplied jacfn must fill a dense matrix.
        if cvode_jacfn is not None:
            cvode_options['jacfn'] = cvode_jacfn
            cvode_options['linsolver'] = 'dense'
            cvode_options.pop('lband', None)
            cvode_options.pop('uband', None)

        # Step-wise mode: one internal CVODE step per call, so the energy
        # integrals see the integrator's own trajectory (see
        # ``_run_cvode_stepwise``). ``tstop`` keeps the final internal step
        # from crossing ``end_time``; the root function, when armed, fires
        # inside a step exactly as it does under ``solve()``.
        use_substeps = bool(getattr(self, '_substep_integrals', False)) and float(
            end_time
        ) > float(start_time)
        if use_substeps:
            cvode_options['one_step_compute'] = True
            cvode_options['tstop'] = float(end_time)

        solver = _scikits_cvode(rhs_fn, **cvode_options)
        # One-time debug print of the CVODE options actually in effect.
        # This helps verify banded linsolver dispatch vs silent fallback.
        if not hasattr(EntropySolver, '_cvode_options_logged'):
            EntropySolver._cvode_options_logged = True
            logger.info(
                'CVODE options in use: %s',
                {k: v for k, v in cvode_options.items() if k != 'old_api'},
            )
        # Request intermediate output points so the per-call energy
        # integrals resolve the within-call flux decay. CVODE interpolates
        # these from its internal steps; the integration, final state, and
        # step count are unchanged. A phi-step-cap root (when armed) stops
        # the call before these points and is handled by the root path
        # below, so the extra points are inert in that case.
        #
        # The grid is front-loaded (quadratic spacing, dense near the call
        # start) because each macro-step is a relaxation toward the new
        # boundary state, so the boundary flux changes fastest just after
        # the start and flattens later. A quadratic grid resolves the
        # F_int integral to ~0.1 percent with a few dozen points, where a
        # uniform grid of the same size leaves ~10 percent.
        if use_substeps:
            cvode_sol = self._run_cvode_stepwise(solver, start_time, end_time, y0)
        else:
            n_out = max(2, int(getattr(self, '_cvode_output_points', 2)))
            if n_out > 2 and float(end_time) > float(start_time):
                x = np.linspace(0.0, 1.0, n_out) ** 2
                tspan = float(start_time) + (float(end_time) - float(start_time)) * x
            else:
                tspan = np.array([start_time, end_time], dtype=float)
            cvode_sol = solver.solve(
                tspan,
                np.asarray(y0, dtype=float).ravel(),
            )

        # Dump CVODE's internal counters. These expose what scipy hides
        # from us and let us discriminate Newton thrash from step-size
        # control thrash from linear-solver thrash.
        #   NumSteps         = nst   (internal BDF steps taken)
        #   NumRhsEvals      = nfe   (RHS evaluations as counted by CVODE)
        #   NumLinSolvSetups = nsetups (iteration-matrix rebuilds, i.e.
        #                      Jacobian FD + LU factorisations)
        #   NumErrTestFails  = netf  (local truncation error test failures)
        #   LastOrder/CurrentOrder = BDF order actually in use (1..5)
        try:
            info = solver._integrator.get_info()
            logger.info(
                'CVODE stats: nst=%d nfe=%d nsetups=%d netf=%d '
                'order=%d last_step=%.2e cur_step=%.2e '
                'rhs_wrap=%d rhs_wrap/nst=%.1f nsetups/nst=%.2f',
                info.get('NumSteps', -1),
                info.get('NumRhsEvals', -1),
                info.get('NumLinSolvSetups', -1),
                info.get('NumErrTestFails', -1),
                info.get('CurrentOrder', info.get('LastOrder', -1)),
                info.get('LastStep', float('nan')),
                info.get('CurrentStep', float('nan')),
                nfev_box[0],
                nfev_box[0] / max(int(info.get('NumSteps', 1)), 1),
                int(info.get('NumLinSolvSetups', 0)) / max(int(info.get('NumSteps', 1)), 1),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug('CVODE get_info() failed: %s', exc)

        # Build a scipy-compatible result wrapper. scipy's
        # OptimizeResult attributes used downstream:
        #   .t      (1D array of internal time points)
        #   .y      (2D array, shape (n_state, n_time), final state at [:, -1])
        #   .status (0 = success, 1 = termination event, -1 = failure)
        #   .message (human-readable)
        #   .nfev   (RHS call count)
        result = OptimizeResult()
        # ``scikits.odes`` rootfn-fire idiosyncrasy: when CVODE's rootfn
        # fires (flag=2), ``cvode_sol.values.t`` contains ONLY the start
        # time (the integration progress to the root is dropped), while
        # the actual root location is exposed via ``cvode_sol.roots.t``
        # / ``cvode_sol.roots.y``. Reading from ``values.t`` here would
        # give ``dt_actual = sol.t[-1] - sol.t[0] = 0`` and PROTEUS's
        # wrapper would fall back to ``dtswitch`` while Aragog's state
        # had actually advanced to the root, locking the coupled run
        # at a fixed point.
        flag = int(getattr(cvode_sol, 'flag', -1))
        roots_obj = getattr(cvode_sol, 'roots', None)
        used_roots = False
        if (
            flag == 2
            and roots_obj is not None
            and getattr(roots_obj, 't', None) is not None
            and np.asarray(roots_obj.t).size > 0
        ):
            roots_t = np.asarray(roots_obj.t, dtype=float).ravel()
            roots_y = np.asarray(roots_obj.y, dtype=float)
            # ``roots.y`` is shape (n_roots_fired, n_state); take the
            # last fired root (in the rare case CVODE captured several
            # before stopping).
            if roots_y.ndim == 2:
                y_root = roots_y[-1, :]
            else:
                y_root = roots_y.ravel()
            t_root = float(roots_t[-1])
            y0_col = np.asarray(y0, dtype=float).reshape(-1, 1)
            y_root_col = np.asarray(y_root, dtype=float).reshape(-1, 1)
            result.t = np.array([float(start_time), t_root], dtype=float)
            result.y = np.concatenate((y0_col, y_root_col), axis=1)
            used_roots = True

        if not used_roots:
            if cvode_sol.values is not None and cvode_sol.values.t is not None:
                t_arr = np.asarray(cvode_sol.values.t, dtype=float).ravel()
                y_arr = np.asarray(cvode_sol.values.y, dtype=float)
                # CVODE stores y as (n_time, n_state); scipy expects
                # (n_state, n_time).
                if y_arr.ndim == 2:
                    y_arr = y_arr.T
                else:
                    y_arr = y_arr.reshape(-1, 1)
                result.t = t_arr
                result.y = y_arr
            else:
                result.t = np.array([float(start_time)])
                result.y = np.asarray(y0, dtype=float).reshape(-1, 1)

        result.nfev = nfev_box[0]
        result.message = getattr(cvode_sol, 'message', '')
        # CVODE flag 0 = success, 2 = root found (tstop), negative = failure.
        if flag == 0 or flag == 2:
            result.status = 0
            if flag == 2 and phi_cap_rootfn is not None:
                # ΔΦ_global cap fired; mark the result so ``solve()``
                # can emit a human-readable log line AFTER the nondim-
                # to-physical time restoration (the t-rescale lives
                # one frame up the stack). Logging ``result.t[-1]``
                # here would report nondim time, which is misleading
                # to operators.
                result.cap_fired = True
                result.cap_evals = int(getattr(phi_cap_rootfn, 'evals', 0))
                result.cap_value = float(phi_cap_rootfn.cap)
                result.cap_phi0 = float(phi_cap_rootfn.phi0)
        else:
            result.status = -1
            if not result.message:
                result.message = f'CVODE failed with flag {flag}'
        return result

    def solve(self) -> None:
        """Run the BDF time integration."""
        start_time = self.parameters.solver.start_time
        end_time = self.parameters.solver.end_time
        # Absolute tolerance floor (1e-8) matches SPIDER's atol=rtol.
        # Tight tolerance is necessary to resolve the crystallisation
        # front and prevent cumulative integration error during long
        # solidification runs.
        atol_base = max(self.parameters.solver.atol, 1.0e-8)
        rtol = self.parameters.solver.rtol

        # Phase-aware atol: tight during crystallization, relaxed
        # only when fully solid.
        try:
            n_stag = self._n_stag
            if self._core_bc == 'gradient':
                # Gradient mode: reconstruct S to compute phi0
                n_basic = n_stag + 1
                dSdr0 = self._S0[:n_basic]
                S_surf0 = float(self._S0[n_basic])
                S0_block, _ = self._reconstruct_entropy(dSdr0, S_surf0)
            elif self._state_is_extended:
                S0_block = self._S0[:n_stag]
            else:
                S0_block = self._S0
            phi0 = float(
                np.asarray(self.entropy_eos.melt_fraction(self._P_stag_flat, S0_block)).mean()
            )
            phi0 = max(0.0, min(1.0, phi0))
        except Exception:
            phi0 = 1.0

        if phi0 > 0.01:
            atol_scale = 1.0
            max_step = 100.0  # years
        else:
            atol_scale = 10.0
            max_step = np.inf

        # phi_cap_anchor stores the step-cap anchor dict for the SUNDIALS
        # rootfn / scipy event, or None when no cap is armed. Initialised
        # here so the consumer at the bottom of solve() never hits an
        # UnboundLocalError when the EOS branch below is skipped (e.g. the
        # const_properties path with no EOS).
        phi_cap_anchor = None

        # Tighten max_step when ANY cell is near a phase boundary and arm the
        # step caps. Runs whenever the entropy EOS is loaded, independent of
        # the mean melt fraction: the temperature and entropy caps must stay
        # active in the deep-solid regime (mean melt fraction below 0.01),
        # where a cell can still cool on the solid adiabat well below the
        # solidus and the melt-fraction cap is blind. When a cell's entropy is
        # within the configured phase-boundary entropy margin (default
        # 200 J/kg/K) of either phase boundary, OR sits inside the mushy band,
        # max_step is reduced to 1 yr to give CVODE enough resolution to
        # handle the phase-boundary stiffness gradually.
        if self.entropy_eos is not None:
            entropy_margin = _resolve_entropy_margin(
                getattr(self.parameters.energy, 'phase_boundary_entropy_margin', None)
            )
            # S0_block is staggered (length n_stag). For per-cell
            # phase-boundary check we use the staggered S directly
            # against the staggered-pressure phase boundaries.
            P_stag = self._P_stag_flat
            S_liq_stag = np.asarray(self.entropy_eos.liquidus_entropy(P_stag)).ravel()
            S_sol_stag = np.asarray(self.entropy_eos.solidus_entropy(P_stag)).ravel()
            S_arr_stag = np.asarray(S0_block).ravel()
            margin_to_liq = S_arr_stag - S_liq_stag  # > 0 means above liquidus
            margin_to_sol = S_arr_stag - S_sol_stag  # > 0 means above solidus
            near_liq = np.any(np.abs(margin_to_liq) < entropy_margin)
            near_sol = np.any(np.abs(margin_to_sol) < entropy_margin)
            in_mushy = np.any((margin_to_liq < 0.0) & (margin_to_sol > 0.0))
            # Keep the original CMB-only check too as a backstop
            P_cmb = float(self._P_basic_flat[0])
            S_liq = float(self.entropy_eos.liquidus_entropy(np.array([P_cmb])).item())
            S_sol = float(self.entropy_eos.solidus_entropy(np.array([P_cmb])).item())
            S0_block_cmb = float(S0_block[0])
            if _phase_boundary_max_step_clamp(
                near_liq,
                near_sol,
                in_mushy,
                cmb_margin_to_liq=S0_block_cmb - S_liq,
                cmb_margin_to_sol=S0_block_cmb - S_sol,
                entropy_margin=entropy_margin,
            ):
                max_step = 1.0

            # Per-call step caps as a SUNDIALS root function. Build the
            # anchor metadata here (per-cell melt fraction, temperature and
            # entropy at solve entry); the rootfn instance is constructed
            # below once the nondim state_scale is known. CVODE returns at
            # the exact time the tightest active cap is reached, enforced by
            # the integrator's own trajectory rather than a start-time rate
            # estimate. The per-cell melt-fraction term catches a deep cell
            # crossing the two-phase window in one step (a negligible move in
            # the global mean but the source of the crystallisation-onset
            # core-temperature jump). The melt-fraction derivative saturates
            # once a cell is fully solid, so it cannot bound the temperature
            # drop on the solid adiabat just below the solidus; the optional
            # temperature and entropy caps bound |ΔT| and |ΔS| per cell
            # directly. All armed caps share one rootfn (g is the minimum
            # margin). Armed as soon as any cell is inside, near, or below the
            # two-phase window so the caps stay active through the solid
            # adiabat. The caps are skipped in gradient core_bc mode, where the
            # state vector holds entropy gradients rather than entropies and
            # the per-cell EOS lookups would be dimensionally meaningless.
            phi_step_cap = _resolve_step_cap(
                getattr(self.parameters.energy, 'phi_step_cap', None)
            )
            temp_step_cap = _resolve_step_cap(
                getattr(self.parameters.energy, 'temperature_step_cap', None)
            )
            entropy_step_cap = _resolve_step_cap(
                getattr(self.parameters.energy, 'entropy_step_cap', None)
            )
            any_cap = phi_step_cap > 0.0 or temp_step_cap > 0.0 or entropy_step_cap > 0.0
            below_solidus = bool(np.any(margin_to_sol < 0.0))
            gradient_mode = getattr(self, '_core_bc', None) == 'gradient'
            if gradient_mode and any_cap:
                logger.info('step caps are not supported in gradient core_bc mode; not armed')
            if (
                any_cap
                and not gradient_mode
                and (near_liq or near_sol or in_mushy or below_solidus)
            ):
                try:
                    rho_stag_init = np.asarray(
                        self.entropy_eos.density(self._P_stag_flat, S_arr_stag)
                    ).ravel()
                    phi_stag_init = np.asarray(
                        self.entropy_eos.melt_fraction(self._P_stag_flat, S_arr_stag)
                    ).ravel()
                    mass_stag_init = rho_stag_init * self._volume_flat
                    mass_total_init = float(np.sum(mass_stag_init))
                    if mass_total_init > 0.0:
                        phi0_global = float(
                            np.sum(mass_stag_init * phi_stag_init) / mass_total_init
                        )
                        T0_stag_init = None
                        if temp_step_cap > 0.0:
                            T0_stag_init = np.asarray(
                                self.entropy_eos.temperature(self._P_stag_flat, S_arr_stag)
                            ).ravel()
                        phi_cap_anchor = {
                            'cap_phi': phi_step_cap,
                            'phi0_global': phi0_global,
                            'phi0_cell': phi_stag_init,
                            'cap_T': temp_step_cap,
                            'T0_cell': T0_stag_init,
                            'cap_S': entropy_step_cap,
                            'S0_cell': S_arr_stag.copy(),
                        }
                        logger.debug(
                            'step caps armed (CVODE rootfn): phi=%.3g, T=%.3g K, S=%.3g '
                            'J/kg/K (global Φ0=%.4f)',
                            phi_step_cap,
                            temp_step_cap,
                            entropy_step_cap,
                            phi0_global,
                        )
                except Exception as exc:
                    logger.warning(
                        'step-cap rootfn anchor failed (%s); proceeding without cap',
                        exc,
                    )

        # External per-call atol scale factor (PROTEUS retry ladder
        # opts in by setting solver._atol_sf before solve()). Default
        # 1.0 leaves behaviour unchanged for callers that don't set it.
        atol_sf_external = float(getattr(self, '_atol_sf', 1.0))
        atol = atol_base * atol_scale * atol_sf_external

        logger.info(
            'EntropySolver: integrating from %.2e to %.2e yr '
            '(Phi_init=%.3f, atol_scale=%.1fx, atol_sf=%.1fx, atol=%.2e, rtol=%.2e)',
            start_time,
            end_time,
            phi0,
            atol_scale,
            atol_sf_external,
            atol,
            rtol,
        )

        # ── Nondimensionalise state, time, and tolerances ──
        # All state components become O(1) after dividing by their
        # reference scales, so scalar atol works uniformly and the
        # BDF solver no longer needs to resolve 9+ significant digits
        # on an O(1000) state variable. The physics code in
        # _dSdt_single stays in physical units; only the solver
        # interface scales in and out.
        # NonDimScales is the single source of truth for nondim
        # scaling, consumed by both the scipy/CVODE wrapper here and
        # the JAX factory. ``__post_init__`` enforces the contract
        # ``rhs_scale = t_ref / state_scale``.
        scales = self._build_nondim_scales()
        S_ref = self._S_ref
        t_ref = scales.t_ref
        n_s = self._n_stag
        _state_scale = scales.state_scale
        _rhs_scale = scales.rhs_scale

        S0_nd = self._S0 / _state_scale
        start_nd = start_time / t_ref
        end_nd = end_time / t_ref
        max_step_nd = max_step / t_ref if np.isfinite(max_step) else max_step

        # Per-component atol in nondim units. Each state component
        # has its own nondim scale (S_ref for entropy, dSdr_ref for
        # dSdr_cmb in energy_balance mode, dSdr_ref for all gradient
        # components in gradient mode). Dividing atol by _state_scale
        # element-wise gives the same physical tolerance on every
        # component, regardless of its nondim scale. A scalar
        # atol_nd = atol/S_ref would force ~10^8x tighter physical
        # tolerance on dSdr_cmb, choking the step size and
        # suppressing the entropy cooling rate enough to prevent
        # solidification.
        atol_nd = atol / _state_scale

        def _rhs_nondim(t_nd, y_nd):
            """Nondim wrapper: scale state to physical, call physics, scale RHS back."""
            dydt_phys = self._dSdt_single(
                t_nd * t_ref,
                y_nd * _state_scale,
            )
            return dydt_phys * _rhs_scale

        logger.info(
            'Nondimensionalisation: S_ref=%.3f t_ref=%.3e yr '
            'atol_nd min/max=[%.2e, %.2e] S0_nd range [%.4f, %.4f]',
            S_ref,
            t_ref,
            float(np.min(atol_nd)),
            float(np.max(atol_nd)),
            S0_nd[:n_s].min(),
            S0_nd[:n_s].max(),
        )

        # BDF integration with phase-aware max_step constraint.
        jac_sparsity = self._build_jac_sparsity()

        # ── melt-fraction cap rootfn / event construction ──
        # Build the SUNDIALS rootfn (CVODE path) and the equivalent
        # solve_ivp event (scipy fallback) only when the cap is armed.
        # Both consume nondim ``y`` and rescale via ``_state_scale``
        # before reading the EOS. The phase-boundary max_step=1 yr
        # clamp set above is preserved; the cap is an ADDITIONAL
        # guardrail anchored at the per-cell and global melt fractions at
        # solve entry.
        events = None
        phi_cap_rootfn = None
        if phi_cap_anchor is not None:
            a = phi_cap_anchor
            try:
                cap_kw = dict(
                    eos=self.entropy_eos,
                    P_stag=self._P_stag_flat,
                    volume=self._volume_flat,
                    n_stag=self._n_stag,
                    phi0_global=a['phi0_global'],
                    cap=a['cap_phi'],
                    state_scale=_state_scale,
                    phi0_per_cell=a['phi0_cell'],
                    cap_temperature=a['cap_T'],
                    T0_per_cell=a['T0_cell'],
                    cap_entropy=a['cap_S'],
                    S0_per_cell=a['S0_cell'],
                )
                phi_cap_rootfn = _PhiCapRootFunction(**cap_kw)
                events = [_phi_cap_event_factory(**cap_kw)]
            except Exception as exc:
                logger.warning(
                    'step-cap rootfn instantiation failed (%s); integrating without cap',
                    exc,
                )
                phi_cap_rootfn = None
                events = None

        # Integrator dispatch. SUNDIALS CVODE (via scikits.odes) is
        # the same solver SPIDER uses and is the recommended choice
        # for production-grade stiff integration: scipy's BDF and
        # Radau collapse their step size to machine epsilon at the
        # crystallisation front and abort with `Required step size
        # is less than spacing between numbers` because scipy's
        # Newton iterator cannot converge through the stiff phase
        # transition. CVODE's modified-Newton with cached Jacobian
        # factorisation handles the discontinuity cleanly. Scipy
        # solve_ivp (Radau or BDF) is kept as a fallback for systems
        # without scikits.odes available.
        solver_method = getattr(self.parameters.energy, 'solver_method', 'cvode')
        # Warn when CVODE was requested but scikits.odes is not
        # importable: the fallback to scipy Radau is a substantial
        # change in solver behaviour (no modified-Newton, no cached
        # Jacobian factorisation) and a silent fallback would let a
        # broken environment masquerade as a physics regression.
        if solver_method == 'cvode' and not _CVODE_AVAILABLE:
            logger.warning(
                'EntropySolver: solver_method="cvode" requested but '
                'scikits.odes is not installed; falling back to scipy '
                'Radau. Install scikits-odes to enable the production '
                'CVODE path (same solver SPIDER uses).'
            )
        use_cvode = solver_method == 'cvode' and _CVODE_AVAILABLE
        if use_cvode:
            # Option Z: build JAX-derived CVODE callbacks when the
            # factory is registered AND the config flag is on. The
            # factory re-creates the (rhs_fn, jac_fn) pair per solve
            # call (JIT recompile cost applies; the same JAX tracing
            # cache is hit on matching signatures, so cost collapses
            # after the first call within a Python process lifetime).
            cvode_rhs_override = None
            cvode_jacfn = None
            use_jax_jac = (
                getattr(self.parameters.energy, 'use_jax_jacobian', False)
                and self._jax_cvode_factory is not None
            )
            if use_jax_jac:
                try:
                    # Pass the NonDimScales instance, which bundles
                    # the internal nondim contract.
                    cvode_rhs_override, cvode_jacfn = self._jax_cvode_factory(
                        scales,
                        self._core_bc,
                    )
                    logger.info(
                        'EntropySolver: option Z active '
                        '(JAX analytic Jacobian + JAX RHS via scikits.odes)'
                    )
                except Exception as exc:  # pragma: no cover - fallback path
                    logger.warning(
                        'EntropySolver: JAX CVODE factory failed (%s); '
                        'falling back to numpy RHS + FD Jacobian',
                        exc,
                    )
                    cvode_rhs_override = None
                    cvode_jacfn = None

            if cvode_rhs_override is None:
                logger.info('EntropySolver: using CVODE (solver_method=cvode)')
            self._solution = self._solve_cvode(
                start_time=start_nd,
                end_time=end_nd,
                y0=S0_nd,
                atol=atol_nd,
                rtol=rtol,
                max_step=max_step_nd,
                rhs=_rhs_nondim,
                cvode_rhs_fn_override=cvode_rhs_override,
                cvode_jacfn=cvode_jacfn,
                phi_cap_rootfn=phi_cap_rootfn,
            )
        else:
            method = 'Radau' if solver_method != 'bdf' else 'BDF'
            logger.info('EntropySolver: using scipy %s', method)
            self._solution = solve_ivp(
                _rhs_nondim,
                (start_nd, end_nd),
                S0_nd,
                method=method,
                vectorized=False,
                dense_output=False,
                atol=atol_nd,
                rtol=rtol,
                jac_sparsity=jac_sparsity,
                max_step=max_step_nd,
                events=events,
            )

        # ── Restore physical units ──
        sol = self._solution
        if sol.t is not None:
            sol.t = np.asarray(sol.t, dtype=float) * t_ref
        if sol.y is not None:
            sol_y = np.asarray(sol.y, dtype=float)
            if sol_y.ndim == 2:
                sol.y = sol_y * _state_scale[:, np.newaxis]
            else:
                sol.y = sol_y * _state_scale

        # ΔΦ_global cap-fire log, emitted with PHYSICAL time. The
        # CVODE path attaches ``cap_fired`` etc. to the result inside
        # ``_solve_cvode``; the scipy fallback exposes the same signal
        # via ``sol.t_events`` (non-empty when the terminal event
        # fired). Logging here (after the t_ref restoration) keeps
        # operator-facing messages in physical years rather than the
        # nondim time the rootfn sees internally.
        if getattr(sol, 'cap_fired', False) and sol.t is not None:
            t_root_phys = float(sol.t[-1])
            logger.info(
                'ΔΦ_global cap: CVODE rootfn fired at '
                't=%.3e yr after %d evals; cap=%.3g, '
                'Φ_global(start)=%.4f',
                t_root_phys,
                int(getattr(sol, 'cap_evals', 0)),
                float(getattr(sol, 'cap_value', 0.0)),
                float(getattr(sol, 'cap_phi0', 0.0)),
            )
        elif (
            getattr(sol, 't_events', None) is not None
            and len(sol.t_events) > 0
            and len(sol.t_events[0]) > 0
        ):
            t_event_phys = float(sol.t_events[0][0]) * t_ref
            logger.info(
                'ΔΦ_global cap: scipy event fired at t=%.3e yr (terminal=True)',
                t_event_phys,
            )

        # Diagnostic logging: internal BDF step statistics (in physical yr)
        if sol.t is not None and len(sol.t) > 1:
            dt_internal = np.diff(sol.t)
            logger.info(
                'EntropySolver: %d internal steps, %d RHS evals, '
                'dt_min=%.2e yr, dt_max=%.2e yr, dt_med=%.2e yr',
                len(sol.t),
                sol.nfev,
                dt_internal.min(),
                dt_internal.max(),
                np.median(dt_internal),
            )
            logger.info(
                'EntropySolver: phase-boundary cache hits=%d misses=%d',
                self.state._pb_cache_hits,
                self.state._pb_cache_misses,
            )
            self.state._pb_cache_hits = 0
            self.state._pb_cache_misses = 0

        if self._solution.status == 0:
            logger.info('EntropySolver: integration completed successfully.')
            self.stop_early = False
        elif self._solution.status == 1:
            # Termination event (liquidus crossing at CMB cell).
            # Integration succeeded up to the event time.
            t_event = self._solution.t[-1]
            logger.info(
                'EntropySolver: liquidus-crossing event at t=%.2e yr '
                '(stopped %.1f yr before end_time). Bottom cell reached '
                'onset of crystallization.',
                t_event,
                end_time - t_event,
            )
            self.stop_early = False
        else:
            logger.error(
                'EntropySolver: integration failed (status=%d): %s',
                self._solution.status,
                self._solution.message,
            )
            self.stop_early = True

    def _compute_step_energy_integrals(self) -> dict[str, float]:
        """Compute per-call energy contributions [J] over the CVODE
        sub-step trajectory, replacing end-of-step instantaneous capture.

        For each accepted internal step in the integration trajectory
        ``(sol.t[i], sol.y[:, i])``, refresh the EntropyState and read
        the instantaneous powers (F_cmb*A_cmb, F_int*A_int, mass-
        integrated radio/dil/tidal). Trapezoidal-integrate over the
        physical-time trajectory to obtain per-source energy J. This
        is correct against transient phase-boundary snapshots that
        contaminate single end-of-step values: a spike in one CVODE
        sub-step is bounded by the small dt to its neighbour rather
        than propagated as a step-mean over the whole call.

        Sign convention: positive = energy ADDED to mantle.

        Returns
        -------
        dict
            Keys ``F_int``, ``F_cmb``, ``Q_radio``, ``Q_tidal``, each
            mapping to the per-call integral in J. All zeros when no
            entropy_eos is attached or the trajectory has fewer than
            2 points (cannot integrate).
        """
        zero = {
            'F_int': 0.0,
            'F_cmb': 0.0,
            'Q_radio': 0.0,
            'Q_tidal': 0.0,
            'Q_radio_cons': 0.0,
            'Q_tidal_cons': 0.0,
            'solver_residual': 0.0,
            'state_heat': 0.0,
        }
        sol = self._solution
        eos = self.entropy_eos
        if eos is None or sol is None or sol.t is None or sol.y is None:
            return zero
        n_steps = int(sol.t.size)
        if n_steps < 2:
            return zero

        n_stag = self._n_stag
        energy_balance = self._core_bc == 'energy_balance'
        bower = self._core_bc == 'bower2018'
        gradient_mode = self._core_bc == 'gradient'
        is_ext = energy_balance or bower

        P_stag = self._P_stag_flat
        vol = self._volume_flat
        r_basic = self._r_basic_flat
        A_int = 4.0 * np.pi * float(r_basic[-1]) ** 2
        A_cmb = 4.0 * np.pi * float(r_basic[0]) ** 2

        # Frozen structural mass per shell, set once from the mesh's
        # equilibrium structure. Used for the conservation-grade
        # mass-integrated heating powers (Q_*_cons). Stays constant
        # across the CVODE trajectory so the integrated budget closes
        # against ``E_state_cons`` (also frozen-mass weighted).
        rho_struct = np.asarray(self.evaluator.mesh.staggered_effective_density).ravel()
        mass_struct = rho_struct * vol

        P_F_int = np.zeros(n_steps)
        P_F_cmb = np.zeros(n_steps)
        P_radio = np.zeros(n_steps)
        P_tidal = np.zeros(n_steps)
        P_radio_cons = np.zeros(n_steps)
        P_tidal_cons = np.zeros(n_steps)
        _dbg_dSdr_cmb = np.full(n_steps, np.nan)
        _dbg_S_bottom = np.full(n_steps, np.nan)
        _dbg_y_hash = np.full(n_steps, np.nan)
        _dbg_kh0 = np.full(n_steps, np.nan)
        _dbg_dSdr1 = np.full(n_steps, np.nan)
        # Per-substep entropy-equation self-consistency integrand
        # (LHS - RHS) [W]. LHS = Σ capacitance (dS/dt) V at the substep
        # state; RHS = -F_int A_int + F_cmb A_cmb + Q_radio + Q_tidal.
        # The discrete divergence telescopes to the boundary fluxes, so
        # this is machine-zero by construction at every state; a non-zero
        # value flags a bug in the flux-divergence assembly, not a
        # time-integration error. The time-integration quality is carried
        # by ``E_residual_cons_frac`` on the coupler side.
        P_resid_solver = np.zeros(n_steps)

        for i in range(n_steps):
            t_i = float(sol.t[i])
            y_col = sol.y[:, i] if sol.y.ndim == 2 else sol.y

            # Reconstruct S_i for EOS lookups, regardless of mode.
            if gradient_mode:
                n_basic = n_stag + 1
                dSdr_i = y_col[:n_basic]
                S_surf_i = float(y_col[n_basic])
                S_i, S_basic_i = self._reconstruct_entropy(dSdr_i, S_surf_i)
            elif is_ext:
                S_i = y_col[:n_stag]
            else:
                S_i = y_col

            if i == 0:
                S_traj_start = np.asarray(S_i, dtype=float).copy()

            if is_ext and y_col.size > n_stag:
                _dbg_dSdr_cmb[i] = float(y_col[n_stag])
            _dbg_S_bottom[i] = float(np.asarray(S_i, dtype=float).ravel()[0])
            _dbg_y_hash[i] = float(
                np.frombuffer(
                    hashlib.blake2b(
                        np.ascontiguousarray(y_col, dtype=float).tobytes(), digest_size=8
                    ).digest(),
                    dtype=np.uint64,
                )[0]
                % 9007199254740993
            )

            # Evaluate the ODE RHS at this accepted state. ``self.dSdt``
            # internally calls ``state.update`` AND applies the BC
            # dispatch (entropy_solver.py:1107-1153), populating
            # ``state._heat_flux[-1]`` and ``[0]`` with the values that
            # were actually used to compute dS/dt. Calling dSdt FIRST
            # (before reading F_int/F_cmb at lines below) ensures the
            # boundary fluxes we integrate are consistent with the
            # ones the entropy ODE saw, so the LHS-RHS check at the
            # bottom of this loop is meaningful. ``dSdt`` returns
            # derivatives in [/yr] to match the integrator's time axis
            # (entropy_solver.py:1172), so divide by SECS_PER_YEAR to
            # align with per-second flux units in the RHS.
            if not gradient_mode:
                dSdt_full = np.asarray(self.dSdt(t_i, y_col)).ravel()
                dSdt_stag_i = dSdt_full[:n_stag] / SECS_PER_YEAR  # /yr -> /s
            else:
                # Gradient mode: state.update with reconstructed
                # entropy + dSdr. Skip dSdt return value (state-vec
                # layout differs).
                self.state.update(S_i, t_i, dSdr=dSdr_i, entropy_basic=S_basic_i)
                dSdt_stag_i = None  # signals: skip solver-residual

            # Read boundary fluxes AFTER dSdt has applied the BCs.
            F_int_i = float(self.state._heat_flux[-1])
            F_cmb_i = float(self.state._heat_flux[0])
            _kh = getattr(self.state, '_eddy_diffusivity', None)
            if _kh is not None and np.size(_kh) > 0:
                _dbg_kh0[i] = float(np.asarray(_kh).ravel()[0])
            _gr = getattr(self.state, '_dSdr', None)
            if _gr is not None and np.size(_gr) > 1:
                _dbg_dSdr1[i] = float(np.asarray(_gr).ravel()[1])

            rho_i = np.asarray(eos.density(P_stag, S_i)).ravel()
            mass_i = rho_i * vol
            heating_radio_i = np.asarray(self.state.heating_radio).ravel()
            heating_tidal_i = np.asarray(self.state.heating_tidal).ravel()
            Q_radio_i = float(np.dot(heating_radio_i, mass_i))
            Q_tidal_i = float(np.dot(heating_tidal_i, mass_i))
            # Frozen-mass variants for the conservation-grade budget
            Q_radio_cons_i = float(np.dot(heating_radio_i, mass_struct))
            Q_tidal_cons_i = float(np.dot(heating_tidal_i, mass_struct))

            P_F_int[i] = -F_int_i * A_int
            P_F_cmb[i] = +F_cmb_i * A_cmb
            P_radio[i] = Q_radio_i
            P_tidal[i] = Q_tidal_i
            P_radio_cons[i] = Q_radio_cons_i
            P_tidal_cons[i] = Q_tidal_cons_i

            # Solver residual: the discrete entropy equation gives
            # ``Σ capacitance (dS/dt) V == -F_int*A + F_cmb*A + Q_radio +
            # Q_tidal`` at every accepted substep. The LHS must be
            # weighted by the SAME capacitance (``rho_phase * T``) that
            # the RHS used to form ``dS/dt``, and the source powers in the
            # RHS by the same phase density. Re-deriving the cell mass
            # from the hard-masked table density (``eos.density``) instead
            # of the tanh-blended phase density leaves a spurious residual
            # equal to the local flux divergence times the fractional
            # density mismatch, concentrated in the phase-transition band
            # (it reaches ~1e23 J for a sub-percent density difference
            # against a ~1e16 W flux divergence). With the consistent
            # weighting the interior fluxes telescope and the residual is
            # machine-zero.
            if dSdt_stag_i is not None:
                cap_i = np.asarray(self.state.capacitance_staggered()).ravel()
                T_phase_i = np.asarray(self.state.phase_staggered.temperature()).ravel()
                # The RHS adds heating as ``H / max(T, 1)`` (the floor the
                # solver applies at line ~1493), so weight the source powers
                # by the matching ``rho_phase * T / max(T, 1) * V`` to keep
                # the identity exact down to the temperature floor.
                heat_mass_i = (
                    np.asarray(self.state.phase_staggered.density()).ravel()
                    * vol
                    * (T_phase_i / np.maximum(T_phase_i, 1.0))
                )
                lhs_i = float(np.sum(cap_i * dSdt_stag_i * vol))
                Q_radio_resid = float(np.dot(heating_radio_i, heat_mass_i))
                Q_tidal_resid = float(np.dot(heating_tidal_i, heat_mass_i))
                rhs_i = P_F_int[i] + P_F_cmb[i] + Q_radio_resid + Q_tidal_resid
                P_resid_solver[i] = lhs_i - rhs_i
            else:
                P_resid_solver[i] = 0.0

        dt_s = np.diff(np.asarray(sol.t, dtype=float)) * SECS_PER_YEAR

        def trap(p):
            return float(np.sum(0.5 * (p[:-1] + p[1:]) * dt_s))

        # Entropy-transported heat content change over the call, evaluated by
        # EOS quadrature of ``rho(P,S) T(P,S) dS`` along each cell's entropy
        # path from the start-of-call state to the final state ``S_i``. This
        # is independent of the flux trajectory above, so its cumulative sum
        # provides a genuine conservation check against the boundary-flux
        # budget rather than reproducing the divergence telescoping.
        state_heat = self._step_heat_content(
            S_traj_start, np.asarray(S_i, dtype=float).ravel()[:n_stag]
        )

        _dump_step_trajectory(
            sol.t,
            dt_s,
            P_F_int,
            P_F_cmb,
            P_radio,
            P_resid_solver,
            _dbg_dSdr_cmb,
            _dbg_S_bottom,
            _dbg_y_hash,
            _dbg_kh0,
            _dbg_dSdr1,
        )

        return {
            'F_int': trap(P_F_int),
            'F_cmb': trap(P_F_cmb),
            'Q_radio': trap(P_radio),
            'Q_tidal': trap(P_tidal),
            'Q_radio_cons': trap(P_radio_cons),
            'Q_tidal_cons': trap(P_tidal_cons),
            'solver_residual': trap(P_resid_solver),
            'state_heat': state_heat,
        }

    def _step_heat_content(self, S0_stag, Sf_stag, n_quad: int = 16) -> float:
        """Entropy-transported heat content change over one solver call [J].

        Evaluates ``Sum_i V_i integral_{S0_i}^{Sf_i} rho(P_i, S) T(P_i, S) dS``
        by trapezoidal quadrature over each staggered cell's entropy path,
        reading density and temperature directly from the EOS. This is the
        Lagrangian heat content the entropy ODE transports (capacitance
        ``rho * T``), so its cumulative sum is the state side of the coupler's
        conservation budget. It is evaluated from the start- and end-of-call
        states alone, independently of the integration trajectory, so a
        mismatch against the boundary-flux and source budget is a genuine
        frame-consistency diagnostic rather than a restatement of the
        divergence telescoping. The quadrature reads the hard-masked table
        density, which differs slightly from the tanh-blended phase density in
        the capacitance, so the coupler residual closes to roughly a percent
        of the cumulative cooling, not to machine precision; the
        machine-precision check is the entropy-ODE solver residual.

        Enthalpy is deliberately not used: the two-phase EOS specific enthalpy
        does not satisfy ``dh = T dS`` across the mushy zone, so an enthalpy
        difference carries a spurious flow-work term and cannot close at all.

        Parameters
        ----------
        S0_stag, Sf_stag : np.ndarray
            Staggered specific entropy at the start and end of the call
            [J/kg/K], length ``n_stag``.
        n_quad : int
            Quadrature points along each cell's entropy path. The per-call
            entropy change is small, so a modest count resolves the integral.

        Returns
        -------
        float
            Heat content change [J], negative when the mantle cools. Zero
            when no EOS is attached.
        """
        eos = self.entropy_eos
        if eos is None:
            return 0.0
        P = np.asarray(self._P_stag_flat, dtype=float).ravel()
        vol = np.asarray(self._volume_flat, dtype=float).ravel()
        S0 = np.asarray(S0_stag, dtype=float).ravel()
        Sf = np.asarray(Sf_stag, dtype=float).ravel()
        n = min(P.size, vol.size, S0.size, Sf.size)
        P, vol, S0, Sf = P[:n], vol[:n], S0[:n], Sf[:n]
        dS = Sf - S0
        nq = int(max(2, n_quad))
        w = np.linspace(0.0, 1.0, nq)
        integrand = np.empty((nq, n), dtype=float)
        for j, wj in enumerate(w):
            S_j = S0 + wj * dS
            rho_j = np.asarray(eos.density(P, S_j)).ravel()[:n]
            T_j = np.asarray(eos.temperature(P, S_j)).ravel()[:n]
            integrand[j] = rho_j * T_j
        # Trapezoidal weights over the unit interval; integral over S is the
        # path integral over w in [0, 1] scaled by the cell entropy change.
        dw = 1.0 / (nq - 1)
        weights = np.full(nq, dw)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        cell_int = (weights[:, None] * integrand).sum(axis=0) * dS
        return float(np.sum(cell_int * vol))

    def write_netcdf(
        self,
        path: str | Path,
        *,
        time: float | None = None,
        description: str | None = None,
    ) -> None:
        """Convenience wrapper: ``self.get_state().to_netcdf(path)``.

        Designed for the standalone solver script: build the solver,
        call ``solve()``, then dump the final state as a NetCDF4 file.

        Parameters
        ----------
        path : str or Path
            Destination NetCDF4 file (overwrites if it exists).
        time : float, optional
            Simulation time at which this snapshot was taken [yr].
            Defaults to ``SolverOutput.dt_actual`` if omitted.
        description : str, optional
            Free-form description string written as the dataset's
            ``description`` global attribute.

        See Also
        --------
        SolverOutput.to_netcdf : underlying writer with the full schema.
        """
        # Forward only when the caller supplied a description; otherwise
        # let ``to_netcdf``'s default take effect so the two entry points
        # stamp the same string into ``ds.description``.
        kwargs: dict[str, str] = {}
        if description is not None:
            kwargs['description'] = description
        self.get_state().to_netcdf(path, time=time, **kwargs)

    def get_state(self) -> SolverOutput:
        """Extract the solver state as a clean output dataclass.

        This is the primary API for callers to retrieve results. It
        avoids the need to access solver internals (evaluator, mesh,
        state, phase objects).

        Returns
        -------
        SolverOutput
            Dataclass containing all quantities needed by PROTEUS.
        """
        sol = self._solution
        eos = self.entropy_eos
        mesh = self.evaluator.mesh

        n_stag = self._n_stag
        energy_balance = self._core_bc == 'energy_balance'
        bower = self._core_bc == 'bower2018'
        gradient_mode = self._core_bc == 'gradient'
        is_ext = energy_balance or bower

        # Compute per-call energy integrals BEFORE refreshing state at
        # the final entropy. ``_compute_step_energy_integrals`` walks
        # the CVODE trajectory and calls state.update() at each
        # accepted sub-step; doing it before the final refresh keeps
        # the end-of-call snapshot (heat_flux, heating arrays, etc.)
        # consistent with what callers see in the rest of get_state().
        step_integrals = self._compute_step_energy_integrals()

        # Slice the final state vector.
        if gradient_mode:
            n_basic = n_stag + 1
            dSdr_final = sol.y[:n_basic, -1]
            S_surf_final = float(sol.y[n_basic, -1])
            S_final, S_basic_final = self._reconstruct_entropy(dSdr_final, S_surf_final)
            extra_final = None
        elif is_ext:
            S_final = sol.y[:n_stag, -1]
            extra_final = float(sol.y[n_stag, -1])
        else:
            S_final = sol.y[:, -1]
            extra_final = None

        P_stag = self._P_stag_flat
        r_basic = self._r_basic_flat
        r_stag = np.asarray(mesh.staggered.radii).ravel()
        vol = self._volume_flat

        if eos is not None:
            T_stag = np.asarray(eos.temperature(P_stag, S_final)).ravel()
            phi_stag = np.asarray(eos.melt_fraction(P_stag, S_final)).ravel()
            rho_stag = np.asarray(eos.density(P_stag, S_final)).ravel()
        else:
            # const_properties: analytical T, phi=1, rho=const
            pm = self.parameters.phase_mixed
            T_stag = pm.const_T_ref * np.exp((S_final - pm.const_S_ref) / pm.const_Cp)
            phi_stag = np.ones_like(S_final)
            rho_stag = np.full_like(S_final, pm.const_rho)

        # Refresh the state at the final entropy for derived quantities.
        if gradient_mode:
            self.state.update(S_final, sol.t[-1], dSdr=dSdr_final, entropy_basic=S_basic_final)
        elif energy_balance:
            self.state.update(S_final, sol.t[-1], dSdr_cmb=extra_final)
        else:
            self.state.update(S_final, sol.t[-1])
        visc_stag = np.asarray(self.state.phase_staggered.viscosity()).ravel()
        heat_flux = self.state.heat_flux.copy()
        heating = self.state.heating.copy()
        eddy_diff = self.state.eddy_diffusivity.copy()
        cap_stag = np.asarray(self.state.capacitance_staggered()).ravel()

        # Diagnostic snapshot from the post-integration state.update().
        jcond_b = self.state.jcond.copy()
        jconv_b = self.state.jconv.copy()
        jgrav_b = self.state.jgrav_heat.copy()
        jmix_b = self.state.jmix_heat.copy()
        dSdr_b = self.state.dSdr.copy()
        phi_basic_diag = self.state.phi_basic_diag.copy()
        T_basic_diag = self.state.T_basic_diag.copy()
        cp_basic_diag = self.state.cp_basic_diag.copy()
        rho_basic_diag = self.state.rho_basic_diag.copy()

        # Scalar quantities.
        # M_mantle uses the analytic A-W mass integral (matching
        # SPIDER's EOSAdamsWilliamson_GetMassWithinShell) when
        # eos_method=1. The discrete sum (rho_stag * vol) has O(h^2)
        # quadrature error vs the analytic integral, causing the
        # structure root finder to converge to a different R_int.
        # For eos_method=2 (external mesh), fall back to discrete sum.
        mesh = self.evaluator.mesh
        if hasattr(mesh.eos, 'get_mass_within_radii'):
            r_cmb = float(self._r_basic_flat[0])
            r_surf = float(self._r_basic_flat[-1])
            M_mantle = (
                mesh.eos.get_mass_within_radii(np.array([r_surf]))
                - mesh.eos.get_mass_within_radii(np.array([r_cmb]))
            ).item()
        else:
            rho_struct_stag = np.asarray(mesh.staggered_effective_density).ravel()
            mass_stag = rho_struct_stag * vol
            M_mantle = float(np.sum(mass_stag))
        mass_stag = rho_stag * vol  # PALEOS density for per-cell output
        # T_magma = top basic-node temperature, evaluated at
        # r = outer_boundary where P = surface_pressure. This matches
        # SPIDER's `atmosphere/temperature_surface` definition.
        # PROTEUS passes T_magma to the atmosphere module as the
        # interior-side surface boundary condition.
        T_basic_final = np.asarray(self.state.phase_basic.temperature()).ravel()
        T_magma = float(T_basic_final[-1])
        # Core temperature: bottom staggered cell (T_stag[0]).
        # SPIDER reports T_core = interior_o.temp[-1], the last
        # staggered node in its surface-to-CMB ordering. Aragog
        # orders CMB-to-surface, so [0] = CMB cell. Reading T_stag[0]
        # rather than evaluating T at the CMB basic node avoids a
        # systematic ~10 K offset from the half-cell pressure
        # difference and matches SPIDER's definition.
        if bower:
            T_core = extra_final  # bower2018: T_core integrated as ODE state
        else:
            T_core = float(T_stag[0])
        # Mass-weighted melt fraction = M_mantle_liquid / M_mantle.
        # MUST be mass-weighted, not volume-weighted, when
        # ``mass_coordinates = true``: the mesh is uniform in mass
        # coordinate, so deep high-density cells span small radial
        # intervals and have small volumes while surface low-density
        # cells are large. During bottom-up crystallisation the
        # surface stays liquid longest, so volume-weighted Phi stays
        # anchored near the surface value while the actual mantle has
        # crystallised, freezing the helpfile Phi and breaking
        # PROTEUS's stop-criterion + structure-update bookkeeping.
        # Matches the rootfn formula bit-for-bit.
        mass_total_for_phi = float(np.sum(mass_stag))
        if mass_total_for_phi > 0.0:
            Phi_global = float(np.sum(phi_stag * mass_stag) / mass_total_for_phi)
        else:
            Phi_global = float(np.mean(phi_stag))

        # Rheological front depth. ``phi_basic_stag_interp`` is the
        # staggered-phi interpolated to basic nodes -- distinct from the
        # EOS-evaluated ``phi_basic_diag`` captured for the diagnostic
        # output above, which is ``phase_basic.melt_fraction()``.
        phi_rheo = self.parameters.phase_mixed.rheological_transition_melt_fraction
        phi_basic_stag_interp = mesh.quantity_at_basic_nodes(phi_stag).ravel()
        # 0.99 / 0.01 bypass thresholds: short-circuit ``argmin`` when no
        # transition exists in the mantle (essentially-fully-liquid or
        # essentially-fully-solid). These mirror ``rheological_front`` in
        # ``output/diagnostics.py`` (now exposed as ``phi_high`` /
        # ``phi_low`` kwargs there). The ``argmin``-based search assumes
        # bottom-up crystallisation — a middle-out solidification mode
        # would need a redefined RF concept (multiple crossings).
        if Phi_global > 0.99:
            rf = float(r_basic[0])
        elif Phi_global < 0.01:
            rf = float(r_basic[-1])
        else:
            idx = np.argmin(np.abs(phi_basic_stag_interp - phi_rheo))
            rf = float(r_basic[idx])
        R_outer = float(r_basic[-1])
        RF_depth = 1.0 - rf / R_outer if R_outer > 0 else 0.0

        # Thermal energy (sensible, for comparison with SPIDER).
        # Uses the real heat capacity Cp(P, S) from the EntropyEOS
        # phase evaluator so that E_th tracks the EOS internal energy
        # consistently with the solver state, rather than a fixed
        # reference Cp.
        Cp_stag = np.asarray(self.state.phase_staggered.heat_capacity()).ravel()
        E_th = float(np.sum(mass_stag * Cp_stag * T_stag))

        # Effective heat capacity (mass-weighted mean Cp)
        Cp_eff = float(np.sum(mass_stag * Cp_stag)) / max(M_mantle, 1.0)

        # EOS-consistent mantle integrated enthalpy [J]. Falls back
        # to NaN when no entropy EOS is loaded (e.g. const_properties
        # path), signalling to PROTEUS that the conservation diagnostic
        # is not available for this run.
        if eos is not None:
            from aragog.output.diagnostics import total_enthalpy

            E_state = total_enthalpy(eos, P_stag, S_final, mass_stag)
            # Conservation-grade integrated enthalpy with FROZEN
            # structural mass per shell. Same enthalpy table, but the
            # mass weighting matches the entropy ODE's frame so that
            # d/dt[E_state_cons] equals the divergence-of-flux budget
            # (with frozen-mass Q_*_cons) to numerical precision.
            mass_struct_stag = np.asarray(mesh.staggered_effective_density).ravel() * vol
            E_state_cons = total_enthalpy(eos, P_stag, S_final, mass_struct_stag)
        else:
            E_state = float('nan')
            E_state_cons = float('nan')

        # CMB heat flux: lower-boundary value of the basic-node heat
        # flux array. Sign convention follows the rest of Aragog,
        # positive-out-of-core when entering the mantle.
        F_cmb = float(heat_flux[0])

        # Mantle-integrated source powers [W] for the closed-mantle
        # energy balance dE/dt = -F_int*A_int + F_cmb*A_cmb + Q_radio +
        # Q_tidal. Each per-source heating array is in W/kg at staggered
        # nodes; mass-weighting recovers the total power.
        Q_radio_total = float(np.dot(np.asarray(self.state.heating_radio).ravel(), mass_stag))
        Q_tidal_total = float(np.dot(np.asarray(self.state.heating_tidal).ravel(), mass_stag))

        # Volumetric melt fraction (porosity-based)
        if eos is not None:
            rho_sol = np.asarray(
                eos._lookup_at_phase_boundary('density', P_stag, 'solid')
            ).ravel()
            rho_liq = np.asarray(
                eos._lookup_at_phase_boundary('density', P_stag, 'melt')
            ).ravel()
            rho_bulk = 1.0 / (
                phi_stag / np.where(rho_liq > 0, rho_liq, 1.0)
                + (1 - phi_stag) / np.where(rho_sol > 0, rho_sol, 1.0)
            )
            drho = rho_sol - rho_liq
            safe_drho = np.where(np.abs(drho) > 1e-10, drho, 1.0)
            porosity = np.clip((rho_sol - rho_bulk) / safe_drho, 0, 1)
            Phi_global_vol = float(np.sum(porosity * vol) / np.sum(vol))
        else:
            Phi_global_vol = 1.0  # const_properties: fully liquid

        # Heating flux
        area_surf = 4 * np.pi * float(r_basic[-1]) ** 2
        F_heat_total = float(np.dot(heating, mass_stag)) / area_surf

        return SolverOutput(
            S_final=S_final,
            T_stag=T_stag,
            phi_stag=phi_stag,
            rho_stag=rho_stag,
            visc_stag=visc_stag,
            P_stag=P_stag,
            r_basic=r_basic,
            r_stag=r_stag,
            vol=vol,
            mass_stag=mass_stag,
            heat_flux=heat_flux,
            heating=heating,
            eddy_diff=eddy_diff,
            cap_stag=cap_stag,
            T_magma=T_magma,
            T_core=T_core,
            Phi_global=Phi_global,
            Phi_global_vol=Phi_global_vol,
            M_mantle=M_mantle,
            M_mantle_liquid=float(np.sum(phi_stag * mass_stag)),
            M_mantle_solid=float(M_mantle - np.sum(phi_stag * mass_stag)),
            RF_depth=RF_depth,
            E_th=E_th,
            E_state=E_state,
            E_state_cons=E_state_cons,
            Cp_eff=Cp_eff,
            F_heat_total=F_heat_total,
            F_cmb=F_cmb,
            Q_radio_total=Q_radio_total,
            Q_tidal_total=Q_tidal_total,
            step_dE_F_int_J=step_integrals['F_int'],
            step_dE_F_cmb_J=step_integrals['F_cmb'],
            step_dE_Q_radio_J=step_integrals['Q_radio'],
            step_dE_Q_tidal_J=step_integrals['Q_tidal'],
            step_dE_Q_radio_cons_J=step_integrals['Q_radio_cons'],
            step_dE_Q_tidal_cons_J=step_integrals['Q_tidal_cons'],
            step_solver_residual_J=step_integrals['solver_residual'],
            # state_heat is the conservation-budget state term (Sum rho T dS);
            # compression is an informational PdV diagnostic from the preceding
            # structure re-solve, deliberately kept out of the predicted energy
            # to avoid double-counting the same work (see the SolverOutput field
            # comments and docs/Explanations/energy_diagnostics.md).
            step_dE_compression_J=float(getattr(self, '_last_compression_J', 0.0)),
            step_dE_state_heat_J=step_integrals['state_heat'],
            dt_actual=float(sol.t[-1] - sol.t[0]),
            status=sol.status,
            jcond_b=jcond_b,
            jconv_b=jconv_b,
            jgrav_b=jgrav_b,
            jmix_b=jmix_b,
            dSdr_b=dSdr_b,
            phi_basic=phi_basic_diag,
            T_basic=T_basic_diag,
            cp_basic=cp_basic_diag,
            rho_basic=rho_basic_diag,
        )
