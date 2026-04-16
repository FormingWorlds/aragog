"""JAX phase evaluator and flux computation for the entropy solver.

Pure-functional replacements for ``entropy_phase.py`` (phase properties)
and ``entropy_state.py`` (MLT convection, heat/mass fluxes). All functions
are JIT-compilable and differentiable.

The numpy versions mutate state arrays in-place. The JAX versions take
arrays in and return NamedTuples out, with no side effects.

Dependencies: jax, equinox (already in PROTEUS ecosystem via atmodeller).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from aragog.jax.eos import EntropyEOS_JAX

# Enable float64
jax.config.update('jax_enable_x64', True)

# Critical Reynolds number from Abe (1993)
RE_CRIT = 9.0 / 8.0


# ---------------------------------------------------------------------------
# Output containers (NamedTuples are JAX pytrees by default)
# ---------------------------------------------------------------------------

class PhaseProperties(NamedTuple):
    """Material properties at a set of mesh nodes."""

    temperature: jax.Array          # [K]
    density: jax.Array              # [kg/m^3]
    heat_capacity: jax.Array        # [J/kg/K]
    thermal_expansivity: jax.Array  # [1/K]
    dTdPs: jax.Array                # [K/Pa]
    melt_fraction: jax.Array        # [-]
    viscosity: jax.Array            # [Pa s]
    kinematic_viscosity: jax.Array  # [m^2/s]
    thermal_conductivity: jax.Array # [W/m/K]
    latent_heat: jax.Array          # [J/kg]
    capacitance: jax.Array          # rho * T [kg K / m^3]


class FluxOutput(NamedTuple):
    """Output from the flux computation at basic nodes."""

    heat_flux: jax.Array            # total heat flux [W/m^2]
    mass_flux: jax.Array            # mass flux [kg/m^2/s]
    eddy_diffusivity: jax.Array     # [m^2/s]
    heating: jax.Array              # internal heating at staggered nodes [W/kg]


# ---------------------------------------------------------------------------
# Static parameters (equinox Module, JAX pytree)
# ---------------------------------------------------------------------------

class PhaseParams(eqx.Module):
    """Static parameters for phase evaluation and flux computation.

    Constructed once from config, passed as ``args`` to JIT-compiled
    functions. All fields are scalars or 1D JAX arrays.
    """

    # Rheology
    phi_rheo: float
    phi_width: float
    log10_visc_solid: float
    log10_visc_liquid: float
    visc_liquid: float
    grain_size: float

    # Thermal conductivity
    k_solid: float
    k_liquid: float

    # Transport flags (stored as float for JAX tracing: 1.0 = on, 0.0 = off)
    conduction: float
    convection: float
    grav_sep: float
    mixing: float

    # Eddy diffusivity
    eddy_diff_thermal: float
    eddy_diff_chemical: float
    kappah_floor: float

    # SPIDER-analogue bottom-up gate for Jgrav. Stored as float for JAX
    # tracing (1.0 = smoothing on, 0.0 = raw un-smoothed flux). Keep ON
    # for production runs; setting this to 0.0 reproduces the pre-fix
    # CMB drain and is only useful for regression tests.
    bottom_up_grav_sep: float

    def __init__(
        self,
        phi_rheo: float = 0.4,
        phi_width: float = 0.15,
        viscosity_solid: float = 1e21,
        viscosity_liquid: float = 1e-1,
        grain_size: float = 1e-3,
        k_solid: float = 4.0,
        k_liquid: float = 2.0,
        conduction: bool = True,
        convection: bool = True,
        grav_sep: bool = False,
        mixing: bool = False,
        eddy_diff_thermal: float = 1.0,
        eddy_diff_chemical: float = 1.0,
        kappah_floor: float = 0.0,
        bottom_up_grav_sep: bool = True,
    ):
        self.phi_rheo = phi_rheo
        self.phi_width = phi_width
        self.log10_visc_solid = jnp.log10(viscosity_solid)
        self.log10_visc_liquid = jnp.log10(viscosity_liquid)
        self.visc_liquid = viscosity_liquid
        self.grain_size = grain_size
        self.k_solid = k_solid
        self.k_liquid = k_liquid
        self.conduction = float(conduction)
        self.convection = float(convection)
        self.grav_sep = float(grav_sep)
        self.mixing = float(mixing)
        self.eddy_diff_thermal = eddy_diff_thermal
        self.eddy_diff_chemical = eddy_diff_chemical
        self.kappah_floor = kappah_floor
        self.bottom_up_grav_sep = float(bottom_up_grav_sep)


class MeshArrays(eqx.Module):
    """Static mesh geometry arrays, converted from numpy Mesh once.

    All arrays are 1D JAX arrays. The transform matrices are 2D.
    """

    # Transform matrices (basic_nodes x staggered_nodes)
    d_dr_matrix: jax.Array       # d/dr transform
    quantity_matrix: jax.Array   # staggered-to-basic interpolation

    # Geometry at basic nodes (N_basic,)
    area: jax.Array              # [m^2]
    volume: jax.Array            # [m^3]
    radii_basic: jax.Array       # [m]
    mixing_length: jax.Array     # [m]
    mixing_length_sq: jax.Array  # [m^2]
    mixing_length_cu: jax.Array  # [m^3]

    # Radii at staggered nodes
    radii_stag: jax.Array        # [m]

    # Pressure profiles (1D, pre-flattened)
    P_stag: jax.Array            # [Pa] at staggered nodes
    P_basic: jax.Array           # [Pa] at basic nodes

    # Gravity at basic nodes
    gravity: jax.Array           # [m/s^2]

    @staticmethod
    def from_numpy_mesh(mesh) -> 'MeshArrays':
        """Build from a numpy Aragog Mesh object."""
        import numpy as np
        return MeshArrays(
            d_dr_matrix=jnp.asarray(mesh._d_dr_transform),
            quantity_matrix=jnp.asarray(mesh._quantity_transform),
            area=jnp.asarray(np.asarray(mesh.basic.area).ravel()),
            volume=jnp.asarray(np.asarray(mesh.basic.volume).ravel()),
            radii_basic=jnp.asarray(np.asarray(mesh.basic.radii).ravel()),
            radii_stag=jnp.asarray(np.asarray(mesh.staggered.radii).ravel()),
            mixing_length=jnp.asarray(np.asarray(mesh.basic.mixing_length).ravel()),
            mixing_length_sq=jnp.asarray(np.asarray(mesh.basic.mixing_length_squared).ravel()),
            mixing_length_cu=jnp.asarray(np.asarray(mesh.basic.mixing_length_cubed).ravel()),
            P_stag=jnp.asarray(np.asarray(mesh.staggered_pressure).ravel()),
            P_basic=jnp.asarray(np.asarray(mesh.basic_pressure).ravel()),
            gravity=jnp.asarray(
                np.full(mesh.basic.radii.size,
                        abs(float(getattr(
                            mesh.eos, '_gravitational_acceleration',
                            0.0,
                        ))))
            ),
        )


# ---------------------------------------------------------------------------
# JAX tanh_weight (replaces aragog.utilities.tanh_weight)
# ---------------------------------------------------------------------------

def tanh_weight(x: jax.Array, threshold: float, width: float) -> jax.Array:
    """Smooth step function: 0.5 * (1 + tanh((x - threshold) / width))."""
    return 0.5 * (1.0 + jnp.tanh((x - threshold) / width))


# ---------------------------------------------------------------------------
# Phase evaluation (replaces EntropyPhaseEvaluator.update)
# ---------------------------------------------------------------------------

def evaluate_phase(
    eos: EntropyEOS_JAX,
    params: PhaseParams,
    P: jax.Array,
    S: jax.Array,
) -> PhaseProperties:
    """Compute all material properties at (P, S) nodes.

    Parameters
    ----------
    eos : EntropyEOS_JAX
        JAX EOS tables.
    params : PhaseParams
        Static material parameters.
    P : jax.Array
        Pressure [Pa], 1D.
    S : jax.Array
        Entropy [J/kg/K], 1D.

    Returns
    -------
    PhaseProperties
        All material properties at the given nodes.
    """
    temperature = eos.temperature(P, S)
    density = eos.density(P, S)
    heat_capacity = eos.heat_capacity(P, S)
    dTdPs_val = eos.dTdPs(P, S)
    thermal_expansivity = jnp.maximum(eos.thermal_expansivity(P, S), 0.0)
    phi = eos.melt_fraction(P, S)
    latent_heat = eos.latent_heat(P)

    # Viscosity: tanh blend in log-space
    w = tanh_weight(phi, params.phi_rheo, params.phi_width)
    log_visc = (1.0 - w) * params.log10_visc_solid + w * params.log10_visc_liquid
    viscosity = 10.0 ** log_visc
    kinematic_viscosity = viscosity / density

    # Thermal conductivity: linear blend
    k_thermal = (1.0 - phi) * params.k_solid + phi * params.k_liquid

    # Capacitance for entropy equation
    capacitance = density * temperature

    return PhaseProperties(
        temperature=temperature,
        density=density,
        heat_capacity=heat_capacity,
        thermal_expansivity=thermal_expansivity,
        dTdPs=dTdPs_val,
        melt_fraction=phi,
        viscosity=viscosity,
        kinematic_viscosity=kinematic_viscosity,
        thermal_conductivity=k_thermal,
        latent_heat=latent_heat,
        capacitance=capacitance,
    )


# ---------------------------------------------------------------------------
# Relative velocity (gravitational separation, Abe 1993)
# ---------------------------------------------------------------------------

def relative_velocity(
    eos: EntropyEOS_JAX,
    params: PhaseParams,
    P: jax.Array,
    density: jax.Array,
    melt_fraction: jax.Array,
    gravity: jax.Array,
) -> jax.Array:
    """Melt-solid relative velocity for gravitational separation [m/s].

    Three-regime permeability model (Abe 1993/1995, SPIDER convention):
    Blake-Kozeny-Carman -> Rumpf-Gupte -> Stokes settling.
    """
    rho_s = eos._lookup_at_phase_boundary('density', P, 'solid')
    rho_l = eos._lookup_at_phase_boundary('density', P, 'melt')
    delta_rho = rho_l - rho_s
    d = params.grain_size
    eta_l = params.visc_liquid

    # Porosity (volume fraction of melt)
    porosity = jnp.clip(
        (rho_s - density) / jnp.maximum(rho_s - rho_l, 1.0), 0.0, 1.0
    )

    por = jnp.maximum(porosity, 1e-20)
    one_m_por = jnp.maximum(1.0 - porosity, 1e-20)

    # Three permeability regimes
    F_bkc = d**2 * por**2 / (one_m_por**2 * 1000.0)
    F_rg = d**2 * por**4.5 * (5.0 / 7.0)
    F_stokes = d**2 * 2.0 / 9.0

    # Smooth regime switching
    w_rg = tanh_weight(porosity, 0.0769618, 0.02)
    w_stokes = tanh_weight(porosity, 0.771462, 0.05)
    F = (1.0 - w_rg) * F_bkc + (w_rg - w_stokes) * F_rg + w_stokes * F_stokes
    F = jnp.maximum(F, 0.0)

    return jnp.abs(delta_rho) * gravity * F / jnp.maximum(eta_l, 1e-10)


# ---------------------------------------------------------------------------
# MLT eddy diffusivity (replaces the MLT block in EntropyState.update)
# ---------------------------------------------------------------------------

def compute_mlt(
    dSdr: jax.Array,
    phase_basic: PhaseProperties,
    mesh: MeshArrays,
    params: PhaseParams,
) -> tuple[jax.Array, jax.Array]:
    """Compute MLT eddy diffusivity from the entropy gradient.

    Parameters
    ----------
    dSdr : jax.Array
        Entropy gradient at basic nodes [J/kg/K/m].
    phase_basic : PhaseProperties
        Material properties at basic nodes.
    mesh : MeshArrays
        Mesh geometry.
    params : PhaseParams
        Static parameters.

    Returns
    -------
    kappa_h : jax.Array
        Thermal eddy diffusivity at basic nodes [m^2/s].
    kappa_c : jax.Array
        Chemical eddy diffusivity at basic nodes [m^2/s].
    """
    alpha = phase_basic.thermal_expansivity
    T = phase_basic.temperature
    Cp = phase_basic.heat_capacity
    nu = phase_basic.kinematic_viscosity

    # Buoyancy from entropy gradient
    effective_superadiabatic = alpha * T * jnp.abs(dSdr) / jnp.maximum(Cp, 1.0)
    velocity_prefactor = mesh.gravity * effective_superadiabatic

    # Convective mask: unstable when dS/dr < 0.
    # Hard mask matching the numpy reference. Using jnp.where (not boolean
    # indexing) for JAX traceability. Produces exactly 0 at stable nodes,
    # avoiding spurious convection from a soft sigmoid.
    conv_mask = jnp.where(dSdr < 0.0, 1.0, 0.0)

    # Viscous velocity (Re <= Re_crit)
    viscous_velocity = (
        velocity_prefactor * mesh.mixing_length_cu / (18.0 * nu)
    ) * conv_mask

    # Inviscid velocity (Re > Re_crit)
    inviscid_velocity_sq = (
        velocity_prefactor * mesh.mixing_length_sq / 16.0
    ) * conv_mask
    inviscid_velocity = jnp.sqrt(jnp.maximum(inviscid_velocity_sq, 0.0))

    # Reynolds number
    reynolds = viscous_velocity * mesh.mixing_length / nu

    # Smooth blend between viscous and inviscid regimes
    blend_width = 0.2 * RE_CRIT
    inviscid_weight = 0.5 * (1.0 + jnp.tanh(
        (reynolds - RE_CRIT) / jnp.maximum(blend_width, 1e-30)
    ))

    # Raw eddy diffusivity
    kh_raw = (
        (1.0 - inviscid_weight) * viscous_velocity
        + inviscid_weight * inviscid_velocity
    ) * mesh.mixing_length

    # Thermal eddy diffusivity (SPIDER convention: positive=scale, negative=constant)
    kappa_h = jnp.where(
        params.eddy_diff_thermal > 0,
        params.eddy_diff_thermal * kh_raw,
        jnp.full_like(kh_raw, -params.eddy_diff_thermal),
    )

    # Chemical eddy diffusivity (from raw kh, before floor)
    kappa_c = jnp.where(
        params.eddy_diff_chemical > 0,
        params.eddy_diff_chemical * kh_raw,
        jnp.full_like(kh_raw, -params.eddy_diff_chemical),
    )

    # kappa_h floor (phase-dependent, modulated by melt fraction)
    phi_basic = phase_basic.melt_fraction
    f_floor = tanh_weight(phi_basic, 0.4, 0.15)
    kh_floor = params.kappah_floor * f_floor
    kappa_h = jnp.maximum(kappa_h, kh_floor)

    return kappa_h, kappa_c


# ---------------------------------------------------------------------------
# Full flux computation (replaces EntropyState.update flux section)
# ---------------------------------------------------------------------------

def compute_fluxes(
    S_stag: jax.Array,
    time: float,
    eos: EntropyEOS_JAX,
    params: PhaseParams,
    mesh: MeshArrays,
    heating_rate: jax.Array,
    S_basic_cmb_override=None,
    dSdr_cmb_override=None,
) -> FluxOutput:
    """Compute all heat and mass fluxes from the entropy profile.

    This is the physics kernel called by the ODE RHS. It is a pure
    function: no mutation, no side effects, fully JIT-compilable.

    Parameters
    ----------
    S_stag : jax.Array
        Entropy at staggered nodes [J/kg/K].
    time : float
        Current time [yr] (used for radionuclide heating).
    eos : EntropyEOS_JAX
        JAX EOS tables.
    params : PhaseParams
        Static material parameters and transport flags.
    mesh : MeshArrays
        Mesh geometry and transform matrices.
    heating_rate : jax.Array
        Internal heating rate at staggered nodes [W/kg]
        (radionuclide + tidal, pre-computed by caller).
    S_basic_cmb_override : float or None
        Optional override for the entropy at the CMB basic node
        (basic-node index 0). Used by the energy_balance core BC
        which reconstructs S_basic[0] from the state-tracked
        dSdr_cmb via S[0] + dSdr_cmb * (r_basic[0] - r_stag[0]).
        When None, the standard quantity_matrix mapping is used.
    dSdr_cmb_override : float or None
        Optional override for the entropy gradient at the CMB
        basic node (dSdr[0]). Used by the energy_balance core BC
        where dSdr_cmb is a state-tracked variable. When None,
        the standard d_dr_matrix mapping is used.

    Returns
    -------
    FluxOutput
        Heat flux, mass flux, eddy diffusivity, heating.
    """
    # Interpolate entropy to basic nodes and compute gradient
    S_basic = mesh.quantity_matrix @ S_stag
    dSdr = mesh.d_dr_matrix @ S_stag

    # Apply energy_balance overrides at the CMB basic node.
    # Done as separate jax.lax.cond branches so the function remains
    # JIT-compatible regardless of whether the overrides are None or
    # JAX scalars. We use jnp.where with a static bool flag passed
    # through from the caller via the optional argument.
    if S_basic_cmb_override is not None:
        S_basic = S_basic.at[0].set(S_basic_cmb_override)
    if dSdr_cmb_override is not None:
        dSdr = dSdr.at[0].set(dSdr_cmb_override)

    # Phase properties at staggered and basic nodes
    phase_stag = evaluate_phase(eos, params, mesh.P_stag, S_stag)
    phase_basic = evaluate_phase(eos, params, mesh.P_basic, S_basic)

    # MLT eddy diffusivity
    kappa_h, kappa_c = compute_mlt(dSdr, phase_basic, mesh, params)

    # Melt fraction gradient (for mixing flux)
    phi_stag = phase_stag.melt_fraction
    dphidr = mesh.d_dr_matrix @ phi_stag

    # Temperature gradient (for conduction)
    T_stag = phase_stag.temperature
    dTdr = mesh.d_dr_matrix @ T_stag

    # Basic node properties
    rho = phase_basic.density
    T = phase_basic.temperature
    k = phase_basic.thermal_conductivity

    # Heat flux components (multiply by flag to enable/disable)
    heat_flux = jnp.zeros_like(S_basic)

    # Conduction: F_cond = -k * dT/dr
    heat_flux = heat_flux + params.conduction * (-k * dTdr)

    # Convection: F_conv = rho * T * kappa_h * (-dS/dr)
    heat_flux = heat_flux + params.convection * (rho * T * kappa_h * (-dSdr))

    # Mass flux for gravitational separation and mixing
    mass_flux = jnp.zeros_like(S_basic)

    # Gravitational separation.
    #
    # Raw Stokes/permeability-driven mass flux:
    #     jgrav_raw = rho * phi * (1 - phi) * v_rel
    # SPIDER analogue smoothing (SPIDER/energy.c:523-533,
    # JGRAV_BOTTOM_UP + get_smoothing): multiply jgrav_raw by a bounded
    # polynomial of an UN-truncated two-phase fraction
    #     gphi = (S - S_sol(P)) / (S_liq(P) - S_sol(P))
    # evaluated at the staggered cell immediately BELOW the interface.
    # The polynomial `16 * gphi^2 * (1 - gphi)^2` (clipped to [0, 1])
    # vanishes cleanly at both pure phases and has bounded derivatives
    # everywhere, unlike SPIDER's tanh smoothing. This is the scipy-
    # path fix from entropy_state.py mirrored here so the JAX backend
    # doesn't reproduce the pre-fix CMB drain at first crystallisation.
    phi_b = phase_basic.melt_fraction
    v_rel = relative_velocity(
        eos, params, mesh.P_basic, rho, phi_b, mesh.gravity,
    )
    jgrav_raw = rho * phi_b * (1.0 - phi_b) * v_rel

    # gphi at STAGGERED nodes (cell below each basic interface)
    S_sol_stag = eos.solidus_entropy(mesh.P_stag)
    S_liq_stag = eos.liquidus_entropy(mesh.P_stag)
    dS_stag = jnp.maximum(S_liq_stag - S_sol_stag, 1.0)
    gphi_stag = (S_stag - S_sol_stag) / dS_stag

    gphi_clip = jnp.clip(gphi_stag, 0.0, 1.0)
    smth_stag = 16.0 * gphi_clip**2 * (1.0 - gphi_clip) ** 2

    # Map staggered smoothing to basic-node interfaces: interior basic
    # node i (1..N-2) sees the smoothing of staggered node i-1 (the
    # cell BELOW). Boundary basic nodes (0 and -1) use smth = 1 as a
    # placeholder because the mass flux at those indices is zeroed a
    # few lines below anyway. Lengths: staggered has N entries, basic
    # has N+1; smth_stag[:-1] supplies the N-1 interior interfaces
    # plus the two boundaries, totalling N+1.
    smth_basic = jnp.concatenate([
        jnp.array([1.0]),
        smth_stag[:-1],
        jnp.array([1.0]),
    ])

    # `bottom_up_grav_sep = 1.0` selects the smoothed flux, 0.0 selects
    # the raw flux (for reproducing the pre-fix drain in regression
    # tests).
    jgrav_smoothed = jgrav_raw * smth_basic
    jgrav = (
        params.bottom_up_grav_sep * jgrav_smoothed
        + (1.0 - params.bottom_up_grav_sep) * jgrav_raw
    )
    mass_flux = mass_flux + params.grav_sep * jgrav

    # Mixing
    mass_flux = mass_flux + params.mixing * (rho * kappa_c * (-dphidr))

    # Zero mass fluxes at boundaries (SPIDER convention)
    mass_flux = mass_flux.at[0].set(0.0)
    mass_flux = mass_flux.at[-1].set(0.0)

    # Add latent heat transport from mass flux
    heat_flux = heat_flux + mass_flux * phase_basic.latent_heat

    return FluxOutput(
        heat_flux=heat_flux,
        mass_flux=mass_flux,
        eddy_diffusivity=kappa_h,
        heating=heating_rate,
    )
