"""JAX ODE solver for the entropy equation.

Replaces ``entropy_solver.py``'s scipy BDF integration with diffrax
Kvaerno5 (5th-order ESDIRK, A-L stable). The RHS function applies
boundary conditions, computes flux divergence, and adds internal
heating, all in pure JAX.

Dependencies: jax, equinox, diffrax (new), lineax (transitive via diffrax).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aragog.jax.eos import EntropyEOS_JAX
from aragog.jax.phase import (
    FluxOutput,
    MeshArrays,
    PhaseParams,
    compute_fluxes,
    evaluate_phase,
)

jax.config.update('jax_enable_x64', True)

logger = logging.getLogger(__name__)

# Seconds per Julian year (matches scipy.constants.Julian_year)
SECS_PER_YEAR: float = 31557600.0

# Stefan-Boltzmann constant [W/m^2/K^4]
SIGMA_SB: float = 5.670374419e-8


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

    All float fields are stored as JAX arrays (not Python floats) to
    avoid JIT recompilation when values change between coupling steps.
    """

    # Surface
    outer_bc_type: int = eqx.field(static=True)
    outer_bc_value: jax.Array      # prescribed flux [W/m^2] (type 4)
    emissivity: jax.Array
    T_eq: jax.Array                # equilibrium temperature [K] (type 1)

    # CMB
    inner_bc_type: int = eqx.field(static=True)
    inner_bc_value: jax.Array      # prescribed flux [W/m^2] (type 2)
    core_density: jax.Array        # [kg/m^3]
    core_heat_capacity: jax.Array  # [J/kg/K]
    tfac_core_avg: jax.Array       # T_avg/T_cmb ratio

    def __init__(self, *, outer_bc_type, outer_bc_value, emissivity, T_eq,
                 inner_bc_type, inner_bc_value, core_density,
                 core_heat_capacity, tfac_core_avg):
        self.outer_bc_type = outer_bc_type
        self.outer_bc_value = jnp.asarray(outer_bc_value, dtype=jnp.float64)
        self.emissivity = jnp.asarray(emissivity, dtype=jnp.float64)
        self.T_eq = jnp.asarray(T_eq, dtype=jnp.float64)
        self.inner_bc_type = inner_bc_type
        self.inner_bc_value = jnp.asarray(inner_bc_value, dtype=jnp.float64)
        self.core_density = jnp.asarray(core_density, dtype=jnp.float64)
        self.core_heat_capacity = jnp.asarray(core_heat_capacity, dtype=jnp.float64)
        self.tfac_core_avg = jnp.asarray(tfac_core_avg, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Solver output
# ---------------------------------------------------------------------------

class SolveResult(NamedTuple):
    """Output from solve_entropy."""

    S_final: jax.Array    # entropy at staggered nodes [J/kg/K]
    t_final: float        # final time [yr]
    n_steps: int          # number of solver steps
    success: bool         # solver converged


# ---------------------------------------------------------------------------
# RHS function
# ---------------------------------------------------------------------------

def _apply_surface_bc(
    heat_flux: jax.Array,
    bc: BoundaryParams,
    phase_basic_T: jax.Array,
) -> jax.Array:
    """Apply the surface boundary condition to the heat flux array.

    Uses jnp.where for JAX traceability (no Python if-statements).
    """
    T_surf = phase_basic_T[-1]

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
) -> jax.Array:
    """Apply the CMB boundary condition to the heat flux array."""
    if bc.inner_bc_type == 1:
        # Core cooling (Bower+2018 Eq. 37)
        r_cmb = mesh.radii_basic[0]
        core_cap = (
            4.0 / 3.0 * jnp.pi * r_cmb**3
            * bc.core_density * bc.core_heat_capacity
        )
        rho_first = phase_stag_rho[0]
        cp_first = phase_stag_Cp[0]
        vol_first = mesh.volume[0]
        cell_cap = vol_first * rho_first * cp_first
        r_above = mesh.radii_basic[1]
        radius_ratio = r_above / r_cmb
        alpha = radius_ratio**2 / (cell_cap / (core_cap * bc.tfac_core_avg) + 1.0)
        F_cmb = alpha * heat_flux[1]
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
        (eos, params, mesh, bc, heating) where:
        - eos: EntropyEOS_JAX
        - params: PhaseParams
        - mesh: MeshArrays
        - bc: BoundaryParams
        - heating: jax.Array, internal heating [W/kg] at staggered nodes

    Returns
    -------
    jax.Array
        dS/dt at staggered nodes [J/kg/K/yr], shape (N_stag,).
    """
    eos, params, mesh, bc, heating = args

    # Compute fluxes (conduction, convection, grav sep, mixing)
    flux_out = compute_fluxes(S, t, eos, params, mesh, heating)
    heat_flux = flux_out.heat_flux

    # Phase properties needed for BCs
    phase_stag = evaluate_phase(eos, params, mesh.P_stag, S)
    phase_basic_T = evaluate_phase(eos, params, mesh.P_basic,
                                    mesh.quantity_matrix @ S).temperature

    # Apply boundary conditions
    heat_flux = _apply_surface_bc(heat_flux, bc, phase_basic_T)
    heat_flux = _apply_cmb_bc(
        heat_flux, bc, mesh,
        phase_stag.density, phase_stag.heat_capacity,
    )

    # Flux divergence at staggered nodes
    energy_flux = heat_flux * mesh.area
    delta_energy_flux = jnp.diff(energy_flux)

    # Capacitance: rho * T * volume
    cap = phase_stag.capacitance
    capacitance = cap * mesh.volume

    # dS/dt from flux divergence [J/kg/K/s]
    dsdt = -delta_energy_flux / capacitance

    # Internal heating: dS/dt += H / T
    T_stag = phase_stag.temperature
    dsdt = dsdt + heating / jnp.maximum(T_stag, 1.0)

    # Convert to J/kg/K/yr
    return dsdt * SECS_PER_YEAR


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
    # Build a closure that captures the static args (eos, params, mesh, bc).
    # diffrax traces through `args` as a pytree, but RegularGridInterpolator
    # closures inside the EOS don't survive pytree operations. By capturing
    # them in a closure and passing only the dynamic state (S) + heating
    # through args, we avoid this issue.
    def _rhs(t, S, dynamic_args):
        h = dynamic_args
        return dSdt(t, S, (eos, params, mesh, bc, h))

    term = diffrax.ODETerm(_rhs)
    _solvers = {
        'tsit5': diffrax.Tsit5,            # explicit RK5, fast JIT (~5s)
        'implicit_euler': diffrax.ImplicitEuler,  # 1-stage implicit, moderate JIT (~14s)
        'kvaerno3': diffrax.Kvaerno3,      # 4-stage ESDIRK, slow JIT (~minutes)
        'kvaerno5': diffrax.Kvaerno5,      # 7-stage ESDIRK, very slow JIT
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
