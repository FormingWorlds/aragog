"""JAX ODE solver for the entropy equation (research-only).

Direct-JAX integration of the entropy equation via diffrax Kvaerno5
(5th-order ESDIRK, A-L stable). The RHS applies boundary conditions,
computes flux divergence, and adds internal heating, all in pure JAX.

Not production-ready: the diffrax ESDIRK solvers (kvaerno3, kvaerno5)
stall at the first crystallisation step on the cubic-Hermite J_grav
smoothing in coupled Earth-mantle runs. Kept for autodiff development. The
production JAX integration path is the CVODE Option-Z path in
``aragog/solver/cvode_jax.py``, selected by setting
``EnergyParams.use_jax_jacobian = True`` (PROTEUS-side
``backend="jax"``), which uses CVODE for time stepping and JAX only
for the analytic Jacobian.

Dependencies: jax, equinox, diffrax, lineax (transitive via diffrax).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import Stefan_Boltzmann

from aragog.jax.eos import EntropyEOS_JAX
from aragog.jax.phase import (
    MeshArrays,
    PhaseParams,
    PhaseProperties,
    compute_fluxes,
    evaluate_phase,
)

# diffrax is imported lazily inside `solve_entropy` so that the rest
# of this module (dSdt, BoundaryParams, BC helpers) can be used by
# downstream code (e.g. CVODE wrappers, parity tests) without
# requiring diffrax to be installed.

jax.config.update('jax_enable_x64', True)

logger = logging.getLogger('fwl.' + __name__)

# Seconds per Julian year (matches scipy.constants.Julian_year)
SECS_PER_YEAR: float = 31557600.0

# Stefan-Boltzmann constant [W/m^2/K^4], the value of the numpy solver
SIGMA_SB: float = Stefan_Boltzmann

# log(2) constant used by the radiogenic decay term. Cached as a
# module-level Python float so the JAX trace bakes it in cleanly.
LOG_TWO: float = 0.6931471805599453


def _no_radio(_t_yr):
    """Default radio heating callable: returns 0.0 W/kg, JAX-traceable.

    Used as the args-tuple entry by ``solve_entropy`` and other call
    sites that don't have radionuclide tables. The production
    PROTEUS path (``build_jax_rhs_and_jacobian``) replaces this with a closure
    that evaluates the per-cell radio heating at the live integrator
    time.
    """
    return jnp.asarray(0.0)


def compute_radio_heating(t_yr, radio_arrays):
    """Evaluate radiogenic heating [W/kg] at time t_yr.

    Parameters
    ----------
    t_yr : float or Array
        Integrator time in years.
    radio_arrays : tuple of 5 Arrays
        (heat_prod, abundance, concentration, t0_years, half_life_years).

    Returns
    -------
    Array
        Radiogenic heating scalar [W/kg].
    """
    hp, ab, cn, t0, hl = radio_arrays
    amp = hp * ab * cn
    # Exponential decay per isotope, weighted across isotopes.
    # The 1e-10 yr floor guards against zero denominators.
    arg = LOG_TWO * (t0 - t_yr) / jnp.maximum(hl, 1e-10)
    # Mask the argument, so a zero-amplitude isotope and its Jacobian are 0, not 0*inf.
    per_iso = amp * jnp.exp(jnp.where(amp != 0.0, arg, 0.0))
    return jnp.sum(per_iso)


def make_radio_heating_fn(heat_prod, abundance, concentration, t0_years, half_life_years):
    """Return a JAX-traceable per-cell radio heating ``H_radio(t_yr)``.

    Implements ``H_radio(t) = sum_i (heat_prod_i · abundance_i ·
    concentration_i · exp(log(2) · (t0_i − t) / half_life_i))`` from
    aragog/parser.py:_Radionuclide.get_heating, vectorised across
    isotopes. The returned scalar is broadcast across the staggered
    grid by the caller (radio is uniform per cell).

    Parameters
    ----------
    heat_prod, abundance, concentration : array_like, shape (n_iso,)
        Per-isotope power production scales [W/kg per (mass-fraction)],
        natural abundance [-], and concentration [mass fraction].
    t0_years, half_life_years : array_like, shape (n_iso,)
        Per-isotope reference time [yr] and half life [yr].

    Returns
    -------
    callable
        Function of t_yr returning radiogenic heating [W/kg].
    """
    radio = tuple(
        jnp.asarray(a, dtype=jnp.float64)
        for a in (heat_prod, abundance, concentration, t0_years, half_life_years)
    )
    return lambda t_yr: compute_radio_heating(t_yr, radio)


# ---------------------------------------------------------------------------
# Boundary condition parameters
# ---------------------------------------------------------------------------


class BoundaryParams(eqx.Module):
    """Boundary condition configuration as a JAX pytree.

    Surface BC types:
        1 = grey-body (F = emissivity * sigma * (T^4 - T_eq^4))
        4 = prescribed flux (from atmosphere module)

    CMB BC types:
        0 = insulating (F = 0)
        1 = core cooling (Bower+2018 Eq. 37)
        2 = prescribed flux
        3 = prescribed temperature (preserve conduction-derived flux)
        5 = energy_balance (SPIDER bit-parity): F_cmb derived from
            the boundary entropy gradient (state-tracked dSdr_cmb).
            Used by ``dSdt_energy_balance``.

    All float fields are stored as JAX arrays (not Python floats) to
    avoid JIT recompilation when values change between coupling steps.

    Energy-balance constants are optional (default 0); only used when
    inner_bc_type == 5 via ``dSdt_energy_balance``.
    """

    # Surface
    outer_bc_type: int = eqx.field(static=True)
    outer_bc_value: jax.Array  # prescribed flux [W/m^2] (type 4)
    emissivity: jax.Array
    T_eq: jax.Array  # equilibrium temperature [K] (type 1)
    # UTBL (ultra-thin thermal boundary layer) Cardano correction
    # (Bower+2018 Eq. 18). When param_utbl is True (static, so the
    # branch is constant-folded) the surface radiating temperature
    # is replaced with the real cubic root of
    #     param_utbl_const * T_surf^3 + T_surf - T_interior = 0.
    # Off in current production configs (param_utbl=False);
    # used by SPIDER-parity test runs.
    param_utbl: bool = eqx.field(static=True)
    param_utbl_const: jax.Array

    # CMB
    inner_bc_type: int = eqx.field(static=True)
    inner_bc_value: jax.Array  # prescribed flux [W/m^2] (type 2)
    core_density: jax.Array  # [kg/m^3]
    core_heat_capacity: jax.Array  # [J/kg/K]
    tfac_core_avg: jax.Array  # T_avg/T_cmb ratio

    # Energy_balance constants (CMB BC type 5).
    # Cached from the numpy solver's _cache_bc_constants():
    #   cmb_area: 4*pi*r_cmb^2  [m^2]
    #   core_M:   (4/3)*pi*r_cmb^3*core_density  [kg]
    #   cmb_dr_cmb: r_basic[1] - r_basic[0]      [m] (half-cell)
    cmb_area: jax.Array
    core_M: jax.Array
    cmb_dr_cmb: jax.Array

    def __init__(
        self,
        *,
        outer_bc_type,
        outer_bc_value,
        emissivity,
        T_eq,
        inner_bc_type,
        inner_bc_value,
        core_density,
        core_heat_capacity,
        tfac_core_avg,
        cmb_area=0.0,
        core_M=0.0,
        cmb_dr_cmb=0.0,
        param_utbl=False,
        param_utbl_const=0.0,
    ):
        self.outer_bc_type = outer_bc_type
        self.outer_bc_value = jnp.asarray(outer_bc_value, dtype=jnp.float64)
        self.emissivity = jnp.asarray(emissivity, dtype=jnp.float64)
        self.T_eq = jnp.asarray(T_eq, dtype=jnp.float64)
        self.inner_bc_type = inner_bc_type
        self.inner_bc_value = jnp.asarray(inner_bc_value, dtype=jnp.float64)
        self.core_density = jnp.asarray(core_density, dtype=jnp.float64)
        self.core_heat_capacity = jnp.asarray(core_heat_capacity, dtype=jnp.float64)
        self.tfac_core_avg = jnp.asarray(tfac_core_avg, dtype=jnp.float64)
        self.cmb_area = jnp.asarray(cmb_area, dtype=jnp.float64)
        self.core_M = jnp.asarray(core_M, dtype=jnp.float64)
        self.cmb_dr_cmb = jnp.asarray(cmb_dr_cmb, dtype=jnp.float64)
        self.param_utbl = bool(param_utbl)
        self.param_utbl_const = jnp.asarray(param_utbl_const, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Solver output
# ---------------------------------------------------------------------------


class SolveResult(NamedTuple):
    """Output from solve_entropy."""

    S_final: jax.Array  # entropy at staggered nodes [J/kg/K]
    t_final: float  # final time [yr]
    n_steps: int  # number of solver steps
    success: bool  # solver converged


class RhsParts(NamedTuple):
    """Components computed during JAX RHS assembly.

    Parameters
    ----------
    rate : jax.Array
        State time derivative [J/kg/K/yr] (and [J/kg/K/m/yr] if extended).
    heat_flux : jax.Array
        Heat flux after both boundary conditions [W/m^2].
    phase_stag : PhaseProperties
        Thermodynamic and phase properties at staggered nodes.
    """

    rate: jax.Array
    heat_flux: jax.Array
    phase_stag: PhaseProperties


class StepPowersAux(NamedTuple):
    """Auxiliary geometry and structural arrays for step powers.

    Parameters
    ----------
    A_int : float or jax.Array
        Surface area at basic outer node [m^2].
    A_cmb : float or jax.Array
        Core-mantle boundary area at basic inner node [m^2].
    volume : jax.Array
        Cell volume at staggered nodes [m^3].
    mass_struct : jax.Array
        Structural mass per cell [kg] (staggered_effective_density * volume).
    P_stag : jax.Array
        Pressure at staggered nodes [Pa].
    """

    A_int: float | jax.Array
    A_cmb: float | jax.Array
    volume: jax.Array
    mass_struct: jax.Array
    P_stag: jax.Array


# ---------------------------------------------------------------------------
# RHS function
# ---------------------------------------------------------------------------


def _utbl_tsurf_jax(T_interior: jax.Array, b: jax.Array) -> jax.Array:
    """JAX-traceable Cardano cubic root for the UTBL correction.

    Solves ``b * x^3 + x - T_interior = 0`` for the real root x = T_surf
    (Bower+2018 Eq. 18). Mirrors solver.boundary._utbl_tsurf, with
    np.cbrt -> jnp.cbrt and np.sqrt -> jnp.sqrt. JAX-traceable so the
    analytic Jacobian sees the closure correctly.
    """
    p = 1.0 / b
    q = -T_interior / b
    discriminant = q**2 / 4.0 + p**3 / 27.0
    sqrt_disc = jnp.sqrt(discriminant)
    return jnp.cbrt(-q / 2.0 + sqrt_disc) + jnp.cbrt(-q / 2.0 - sqrt_disc)


def _apply_surface_bc(
    heat_flux: jax.Array,
    bc: BoundaryParams,
    phase_basic_T: jax.Array,
) -> jax.Array:
    """Apply the surface boundary condition to the heat flux array.

    Uses jnp.where for JAX traceability (no Python if-statements).
    """
    T_interior = phase_basic_T[-1]

    # UTBL Cardano correction (Bower+2018 Eq. 18). Off in current
    # production configs; gated on the static `param_utbl` flag
    # so the branch is constant-folded by JIT. Mirrors the numpy path
    # in solver/boundary.py:_utbl_tsurf.
    if bc.param_utbl:
        T_surf = _utbl_tsurf_jax(T_interior, bc.param_utbl_const)
    else:
        T_surf = T_interior

    # Grey-body: F = emissivity * sigma * (T^4 - T_eq^4)
    F_grey = bc.emissivity * SIGMA_SB * (T_surf**4 - bc.T_eq**4)

    # Prescribed flux
    F_prescribed = bc.outer_bc_value

    # Select based on BC type (static, so this traces correctly)
    if bc.outer_bc_type == 1:
        F_surf = F_grey
    else:  # type 4 (prescribed)
        F_surf = F_prescribed

    return heat_flux.at[-1].set(F_surf)


def _apply_cmb_bc(
    heat_flux: jax.Array,
    bc: BoundaryParams,
    mesh: MeshArrays,
    phase_stag_rho: jax.Array,
    phase_stag_Cp: jax.Array,
    heating_first: jax.Array | float,
) -> jax.Array:
    """Apply the CMB boundary condition to the heat flux array.

    ``heating_first`` is the internal heating [W/kg] of the bottom mantle cell; the
    core-cooling BC (type 1) removes it from the power it splits with the core.
    """
    if bc.inner_bc_type == 1:
        # Core cooling (Bower+2018 Eq. 37): the bottom cell and the core share one T, so
        # the cell's net power (outflow minus its heating) is split by capacity.
        r_cmb = mesh.radii_basic[0]
        core_cap = 4.0 / 3.0 * jnp.pi * r_cmb**3 * bc.core_density * bc.core_heat_capacity
        rho_first = phase_stag_rho[0]
        cp_first = phase_stag_Cp[0]
        vol_first = mesh.volume[0]
        cell_cap = vol_first * rho_first * cp_first
        r_above = mesh.radii_basic[1]
        radius_ratio = r_above / r_cmb
        alpha = radius_ratio**2 / (cell_cap / (core_cap * bc.tfac_core_avg) + 1.0)
        F_cmb = alpha * (heat_flux[1] - heating_first * rho_first * vol_first / mesh.area[1])
    elif bc.inner_bc_type == 2:
        # Prescribed flux
        F_cmb = bc.inner_bc_value
    elif bc.inner_bc_type == 3:
        # Prescribed T: keep conduction-derived flux from compute_fluxes
        F_cmb = heat_flux[0]
    else:
        # Insulating (type 0)
        F_cmb = 0.0

    return heat_flux.at[0].set(F_cmb)


def _dSdt_parts(
    t: float,
    S: jax.Array,
    args: tuple,
) -> RhsParts:
    """Evaluate RHS rate and intermediate components at staggered nodes.

    Parameters
    ----------
    t : float
        Current time [yr].
    S : jax.Array
        Entropy at staggered nodes [J/kg/K], shape (N_stag,).
    args : tuple
        (eos, params, mesh, bc, heating_static, H_radio_fn) where:
        - eos: EntropyEOS_JAX
        - params: PhaseParams
        - mesh: MeshArrays
        - bc: BoundaryParams
        - heating_static: jax.Array, time-independent heating [W/kg]
          at staggered nodes (tidal + any other constant source).
        - H_radio_fn: callable(t_yr) -> scalar JAX array [W/kg].
          Returns the per-cell uniform radiogenic heating evaluated
          at the live integrator time. Use ``_no_radio`` as the
          callable when the run has no radionuclides.

    Returns
    -------
    RhsParts
        NamedTuple with rate, heat_flux, and phase_stag.
    """
    eos, params, mesh, bc, heating_static, H_radio_fn = args

    # Time-dependent radio heating evaluated at the live integrator time.
    heating = heating_static + H_radio_fn(t)

    # Compute fluxes (conduction, convection, grav sep, mixing).
    flux_out = compute_fluxes(S, t, eos, params, mesh, heating)
    heat_flux = flux_out.heat_flux

    # Phase properties needed for BCs
    phase_stag = evaluate_phase(eos, params, mesh.P_stag, S)
    phase_basic_T = evaluate_phase(
        eos, params, mesh.P_basic, mesh.quantity_matrix @ S
    ).temperature

    # Apply boundary conditions
    heat_flux = _apply_surface_bc(heat_flux, bc, phase_basic_T)
    heat_flux = _apply_cmb_bc(
        heat_flux,
        bc,
        mesh,
        phase_stag.density,
        phase_stag.heat_capacity,
        flux_out.heating[0],
    )

    # Flux divergence at staggered nodes
    energy_flux = heat_flux * mesh.area
    delta_energy_flux = jnp.diff(energy_flux)

    # Capacitance: rho * T * volume
    cap = phase_stag.capacitance
    capacitance = cap * mesh.volume

    # dS/dt from flux divergence [J/kg/K/s]
    dsdt = -delta_energy_flux / capacitance

    # RHS takes heating from flux_out.heating; step_powers rebuilds it from
    # heating_static and H_radio_fn(t), so its residual column flags any
    # other source that compute_fluxes adds.
    T_stag = phase_stag.temperature
    dsdt = dsdt + flux_out.heating / jnp.maximum(T_stag, 1.0)

    # Convert to J/kg/K/yr
    rate = dsdt * SECS_PER_YEAR

    return RhsParts(
        rate=rate,
        heat_flux=heat_flux,
        phase_stag=phase_stag,
    )


def dSdt(
    t: float,
    S: jax.Array,
    args: tuple,
) -> jax.Array:
    """ODE right-hand side: dS/dt at staggered nodes [J/kg/K/yr].

    Parameters
    ----------
    t : float
        Current time [yr].
    S : jax.Array
        Entropy at staggered nodes [J/kg/K], shape (N_stag,).
    args : tuple
        (eos, params, mesh, bc, heating_static, H_radio_fn) where:
        - eos: EntropyEOS_JAX
        - params: PhaseParams
        - mesh: MeshArrays
        - bc: BoundaryParams
        - heating_static: jax.Array, time-independent heating [W/kg]
          at staggered nodes (tidal + any other constant source).
        - H_radio_fn: callable(t_yr) -> scalar JAX array [W/kg].
          Returns the per-cell uniform radiogenic heating evaluated
          at the live integrator time. Use ``_no_radio`` as the
          callable when the run has no radionuclides.

    Returns
    -------
    jax.Array
        dS/dt at staggered nodes [J/kg/K/yr], shape (N_stag,).
    """
    return _dSdt_parts(t, S, args).rate


# ---------------------------------------------------------------------------
# Energy_balance core BC RHS (state vector = [S, dSdr_cmb], length N+1)
# ---------------------------------------------------------------------------


def _dSdt_energy_balance_parts(
    t: float,
    state_ext: jax.Array,
    args: tuple,
) -> RhsParts:
    """Evaluate energy-balance RHS rate and intermediate components.

    Mirrors the numpy ``EntropySolver._dSdt_single`` for
    ``core_bc='energy_balance'``. State layout:

        state_ext[0:N] = S at staggered nodes [J/kg/K]
        state_ext[N]   = dSdr_cmb at the CMB basic node [J/kg/K/m]

    Parameters
    ----------
    t : float
        Current time [yr].
    state_ext : jax.Array, shape (N+1,)
        Extended state vector (entropy + dSdr_cmb).
    args : tuple
        ``(eos, params, mesh, bc, heating_static, H_radio_fn)``: same
        as ``dSdt``. ``bc`` must have inner_bc_type = 5 and the
        energy_balance constants (cmb_area, core_M, cmb_dr_cmb) populated.

    Returns
    -------
    RhsParts
        NamedTuple with rate, heat_flux, and phase_stag.
    """
    eos, params, mesh, bc, heating_static, H_radio_fn = args
    # Live radio heating, broadcast to per-cell uniform.
    heating = heating_static + H_radio_fn(t)
    n_stag = mesh.P_stag.shape[0]
    S = state_ext[:n_stag]
    dSdr_cmb = state_ext[n_stag]

    # Reconstruct entropy at the CMB basic node using the boundary gradient.
    r_basic = mesh.radii_basic
    r_stag_0 = 0.5 * (r_basic[0] + r_basic[1])
    dr_offset = r_basic[0] - r_stag_0
    S_basic_cmb = S[0] + dSdr_cmb * dr_offset

    # Compute fluxes using compute_fluxes with the energy_balance overrides.
    flux_out = compute_fluxes(
        S,
        t,
        eos,
        params,
        mesh,
        heating,
        S_basic_cmb_override=S_basic_cmb,
        dSdr_cmb_override=dSdr_cmb,
    )
    heat_flux = flux_out.heat_flux

    # Phase properties at staggered nodes
    phase_stag = evaluate_phase(eos, params, mesh.P_stag, S)

    # Phase properties at basic nodes (with corrected CMB entropy)
    S_basic_default = mesh.quantity_matrix @ S
    S_basic = S_basic_default.at[0].set(S_basic_cmb)
    phase_basic = evaluate_phase(eos, params, mesh.P_basic, S_basic)

    # Surface BC
    heat_flux = _apply_surface_bc(heat_flux, bc, phase_basic.temperature)

    # CMB BC: heat_flux[0] is already computed by compute_fluxes.
    T_cmb = phase_basic.temperature[0]
    cp_cmb = phase_basic.heat_capacity[0]
    F_cmb_from_dSdr = heat_flux[0]

    # Flux divergence (entropy derivatives, same as dSdt).
    energy_flux = heat_flux * mesh.area
    delta_energy_flux = jnp.diff(energy_flux)
    cap = phase_stag.capacitance
    capacitance = cap * mesh.volume
    dSdt_per_s = -delta_energy_flux / capacitance
    dSdt_per_s = dSdt_per_s + flux_out.heating / jnp.maximum(phase_stag.temperature, 1.0)
    dSdt_per_yr = dSdt_per_s * SECS_PER_YEAR

    # dSdr_cmb closure equation (SPIDER bc.c:76-131)
    E_tot_cmb = F_cmb_from_dSdr * bc.cmb_area
    fac_cmb = cp_cmb / (
        bc.core_heat_capacity
        * jnp.maximum(T_cmb, 1.0)
        * bc.tfac_core_avg
        * jnp.maximum(bc.core_M, 1.0)
    )
    dSdt_basic_cmb_per_s = -E_tot_cmb * fac_cmb
    d_dSdr_cmb_dt_per_s = (dSdt_per_s[0] - dSdt_basic_cmb_per_s) * 2.0 / bc.cmb_dr_cmb
    d_dSdr_cmb_dt_per_yr = d_dSdr_cmb_dt_per_s * SECS_PER_YEAR

    # Assemble extended-state derivative
    rate = jnp.concatenate(
        [
            dSdt_per_yr,
            jnp.array([d_dSdr_cmb_dt_per_yr]),
        ]
    )

    return RhsParts(
        rate=rate,
        heat_flux=heat_flux,
        phase_stag=phase_stag,
    )


def dSdt_energy_balance(
    t: float,
    state_ext: jax.Array,
    args: tuple,
) -> jax.Array:
    """RHS for the energy_balance core BC mode (extended state, N+1).

    Mirrors the numpy ``EntropySolver._dSdt_single`` for
    ``core_bc='energy_balance'``. State layout:

        state_ext[0:N] = S at staggered nodes [J/kg/K]
        state_ext[N]   = dSdr_cmb at the CMB basic node [J/kg/K/m]

    Returns d/dt of the same layout. The dSdr_cmb evolution is the
    SPIDER bc.c:76-131 closure equation (also in numpy as
    ``EntropySolver._energy_balance_rhs_per_s``).

    Parameters
    ----------
    t : float
        Current time [yr].
    state_ext : jax.Array, shape (N+1,)
        Extended state vector (entropy + dSdr_cmb).
    args : tuple
        ``(eos, params, mesh, bc, heating_static, H_radio_fn)``: same
        as ``dSdt``. ``bc`` must have inner_bc_type = 5 and the
        energy_balance constants (cmb_area, core_M, cmb_dr_cmb) populated.

    Returns
    -------
    jax.Array
        d(state_ext)/dt at the same layout, [J/kg/K/yr] for entropy,
        [J/kg/K/m/yr] for dSdr_cmb.
    """
    return _dSdt_energy_balance_parts(t, state_ext, args).rate


def step_powers(
    t: float,
    y: jax.Array,
    args: tuple,
    mode: str,
    aux: StepPowersAux,
) -> jax.Array:
    """Powers [W] at one solver state for per-call energy integrals.

    Calls the mode's parts function once and returns the boundary, source,
    and residual powers in the order of the numpy ``_step_powers``:
    ``[-F_int*A_int, F_cmb*A_cmb, Q_radio, Q_tidal, Q_radio_cons,
    Q_tidal_cons, residual]``.

    Parameters
    ----------
    t : float
        Node time [yr].
    y : jax.Array
        Node state vector in physical units (entropy [J/kg/K], plus
        dSdr_cmb for energy_balance mode).
    args : tuple
        RHS argument tuple: (eos, params, mesh, bc, heating_static, H_radio_fn).
    mode : str
        Solver mode ('quasi_steady' or 'energy_balance').
    aux : StepPowersAux
        Per-solve geometry and structural mass arrays.

    Returns
    -------
    jax.Array, shape (7,)
        Boundary heat flows, source powers, and entropy balance residual [W].
    """
    eos, params, mesh, bc, heating_static, H_radio_fn = args
    if mode == 'quasi_steady':
        parts = _dSdt_parts(t, y, args)
        S_stag = y
        dSdt_stag = parts.rate / SECS_PER_YEAR
    elif mode == 'energy_balance':
        parts = _dSdt_energy_balance_parts(t, y, args)
        n_stag = aux.volume.shape[0]
        S_stag = y[:n_stag]
        dSdt_stag = parts.rate[:n_stag] / SECS_PER_YEAR
    else:
        raise ValueError(
            f'mode={mode!r} is not supported by step_powers; expected '
            "'quasi_steady' or 'energy_balance'."
        )

    p_int = -parts.heat_flux[-1] * aux.A_int
    p_cmb = parts.heat_flux[0] * aux.A_cmb

    H_radio = H_radio_fn(t)
    mass_i = eos.density(aux.P_stag, S_stag) * aux.volume
    Q_radio_i = H_radio * jnp.sum(mass_i)
    Q_tidal_i = jnp.dot(heating_static, mass_i)

    Q_radio_cons_i = H_radio * jnp.sum(aux.mass_struct)
    Q_tidal_cons_i = jnp.dot(heating_static, aux.mass_struct)

    phase_stag = parts.phase_stag
    cap = phase_stag.capacitance
    T_phase = phase_stag.temperature
    rho_phase = phase_stag.density
    heat_mass = rho_phase * aux.volume * (T_phase / jnp.maximum(T_phase, 1.0))
    lhs = jnp.sum(cap * dSdt_stag * aux.volume)
    Q_radio_resid = H_radio * jnp.sum(heat_mass)
    Q_tidal_resid = jnp.dot(heating_static, heat_mass)
    rhs = p_int + p_cmb + Q_radio_resid + Q_tidal_resid
    residual = lhs - rhs

    return jnp.array(
        [
            p_int,
            p_cmb,
            Q_radio_i,
            Q_tidal_i,
            Q_radio_cons_i,
            Q_tidal_cons_i,
            residual,
        ]
    )


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------


def solve_entropy(
    S0: jax.Array,
    t_start: float,
    t_end: float,
    eos: EntropyEOS_JAX,
    params: PhaseParams,
    mesh: MeshArrays,
    bc: BoundaryParams,
    heating: jax.Array,
    atol: float = 0.01,
    rtol: float = 1e-4,
    max_steps: int = 100_000,
    method: str = 'implicit_euler',
) -> SolveResult:
    """Integrate the entropy equation from t_start to t_end.

    Parameters
    ----------
    S0 : jax.Array
        Initial entropy at staggered nodes [J/kg/K].
    t_start, t_end : float
        Integration interval [yr].
    eos : EntropyEOS_JAX
        JAX EOS tables.
    params : PhaseParams
        Material parameters.
    mesh : MeshArrays
        Mesh geometry.
    bc : BoundaryParams
        Boundary conditions.
    heating : jax.Array
        Internal heating [W/kg] at staggered nodes.
    atol, rtol : float
        Solver tolerances.
    max_steps : int
        Maximum number of solver steps.

    Returns
    -------
    SolveResult
        Final entropy, time, step count, success flag.
    """
    # Lazy import: keeps the rest of this module importable when
    # diffrax is not installed. Callers using only dSdt /
    # BoundaryParams / phase helpers do not need diffrax.
    import diffrax

    # Build a closure that captures the static args (eos, params, mesh, bc).
    # diffrax traces through `args` as a pytree, but RegularGridInterpolator
    # closures inside the EOS don't survive pytree operations. By capturing
    # them in a closure and passing only the dynamic state (S) + heating
    # through args, we avoid this issue.
    def _rhs(t, S, dynamic_args):
        h = dynamic_args
        # Standalone test path: no radionuclides table is exposed,
        # so radio heating defaults to zero.
        return dSdt(t, S, (eos, params, mesh, bc, h, _no_radio))

    term = diffrax.ODETerm(_rhs)
    _solvers = {
        'tsit5': diffrax.Tsit5,  # explicit RK5, fast JIT (~5s)
        'implicit_euler': diffrax.ImplicitEuler,  # 1-stage implicit, moderate JIT (~14s)
        'kvaerno3': diffrax.Kvaerno3,  # 4-stage ESDIRK, slow JIT (~minutes)
        'kvaerno5': diffrax.Kvaerno5,  # 7-stage ESDIRK, very slow JIT
    }
    if method not in _solvers:
        raise ValueError(f'Unknown solver method: {method}. Choose from {list(_solvers)}')
    solver = _solvers[method]()
    controller = diffrax.PIDController(
        atol=atol,
        rtol=rtol,
    )

    # Initial step size: small fraction of the time interval.
    # Guard against zero or negative intervals.
    dt0 = max((t_end - t_start) * 1e-6, 1e-10)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_start,
        t1=t_end,
        dt0=dt0,
        y0=S0,
        args=heating,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=max_steps,
    )

    S_final = sol.ys[0]  # SaveAt(t1=True) saves one snapshot at t1
    t_final = float(np.asarray(sol.ts[0]).item())
    n_steps = int(np.asarray(sol.stats['num_steps']).item())
    success = bool(np.asarray(sol.result == diffrax.RESULTS.successful).item())

    return SolveResult(
        S_final=S_final,
        t_final=t_final,
        n_steps=n_steps,
        success=success,
    )
