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

    temperature: jax.Array  # [K]
    density: jax.Array  # [kg/m^3]
    heat_capacity: jax.Array  # [J/kg/K]
    thermal_expansivity: jax.Array  # [1/K]
    dTdPs: jax.Array  # [K/Pa]
    melt_fraction: jax.Array  # [-]
    viscosity: jax.Array  # [Pa s]
    kinematic_viscosity: jax.Array  # [m^2/s]
    thermal_conductivity: jax.Array  # [W/m/K]
    latent_heat: jax.Array  # [J/kg]
    capacitance: jax.Array  # rho * T [kg K / m^3]


class FluxOutput(NamedTuple):
    """Output from the flux computation at basic nodes."""

    heat_flux: jax.Array  # total heat flux [W/m^2]
    mass_flux: jax.Array  # mass flux [kg/m^2/s]
    eddy_diffusivity: jax.Array  # [m^2/s]
    heating: jax.Array  # internal heating at staggered nodes [W/kg]
    # (radio + tidal). The volumetric-work
    # piece of segregation heating is
    # implicit in the divergence of the
    # Δh-weighted mass-flux contributions
    # to ``heat_flux`` and is not added
    # separately.
    jmix_heat: jax.Array  # convective-mixing heat flux at basic
    # nodes [W/m^2]; raw (NOT gated by
    # params.mixing). Exposed for diagnostic
    # post-processing.


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

    # SPIDER -matprop_smooth_width: tanh blend width for the
    # mixed-vs-single phase blend. 0.0 = sharp (smth=1 inside [0,1]),
    # 0.01 = CHILI Earth production setting.
    matprop_smooth_width: float

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

    # Phase-boundary smoothing selection for Jgrav and Jmix.
    # phase_smoothing_tanh: 1.0 -> use SPIDER's two-branch tanh
    # (_spider_get_smoothing); 0.0 -> use cubic Hermite 16*g^2*(1-g)^2.
    # phase_smoothing_width: tanh transition width in gphi units
    # (ignored when phase_smoothing_tanh == 0.0). SPIDER default for CHILI
    # Earth is 0.01 (matprop_smooth_width).
    phase_smoothing_tanh: float
    phase_smoothing_width: float

    def __init__(
        self,
        phi_rheo: float = 0.4,
        phi_width: float = 0.15,
        viscosity_solid: float = 1e21,
        viscosity_liquid: float = 1e-1,
        grain_size: float = 1e-3,
        k_solid: float = 4.0,
        k_liquid: float = 2.0,
        matprop_smooth_width: float = 0.0,
        conduction: bool = True,
        convection: bool = True,
        grav_sep: bool = False,
        mixing: bool = False,
        eddy_diff_thermal: float = 1.0,
        eddy_diff_chemical: float = 1.0,
        kappah_floor: float = 0.0,
        bottom_up_grav_sep: bool = True,
        phase_smoothing: str = 'tanh',
        phase_smoothing_width: float = 0.01,
    ):
        self.phi_rheo = phi_rheo
        self.phi_width = phi_width
        self.log10_visc_solid = jnp.log10(viscosity_solid)
        self.log10_visc_liquid = jnp.log10(viscosity_liquid)
        self.visc_liquid = viscosity_liquid
        self.grain_size = grain_size
        self.k_solid = k_solid
        self.k_liquid = k_liquid
        self.matprop_smooth_width = matprop_smooth_width
        self.conduction = float(conduction)
        self.convection = float(convection)
        self.grav_sep = float(grav_sep)
        self.mixing = float(mixing)
        self.eddy_diff_thermal = eddy_diff_thermal
        self.eddy_diff_chemical = eddy_diff_chemical
        self.kappah_floor = kappah_floor
        self.bottom_up_grav_sep = float(bottom_up_grav_sep)
        if phase_smoothing not in ('cubic_hermite', 'tanh'):
            raise ValueError(
                f"phase_smoothing must be 'cubic_hermite' or 'tanh', got {phase_smoothing!r}"
            )
        self.phase_smoothing_tanh = 1.0 if phase_smoothing == 'tanh' else 0.0
        self.phase_smoothing_width = float(phase_smoothing_width)


class MeshArrays(eqx.Module):
    """Static mesh geometry arrays, converted from numpy Mesh once.

    All arrays are 1D JAX arrays. The transform matrices are 2D.
    """

    # Transform matrices (basic_nodes x staggered_nodes)
    d_dr_matrix: jax.Array  # d/dr transform
    quantity_matrix: jax.Array  # staggered-to-basic interpolation

    # Geometry at basic nodes (N_basic,)
    area: jax.Array  # [m^2]
    volume: jax.Array  # [m^3]
    radii_basic: jax.Array  # [m]
    mixing_length: jax.Array  # [m]
    mixing_length_sq: jax.Array  # [m^2]
    mixing_length_cu: jax.Array  # [m^3]

    # Radii at staggered nodes
    radii_stag: jax.Array  # [m]

    # Pressure profiles (1D, pre-flattened)
    P_stag: jax.Array  # [Pa] at staggered nodes
    P_basic: jax.Array  # [Pa] at basic nodes

    # dP/dr at basic nodes from Adams-Williamson EOS profile
    # (np.gradient(P_basic, r_basic) in numpy state). Stored on the mesh
    # because it only depends on the structural pressure profile, not on
    # the entropy state.
    dP_dr_basic: jax.Array  # [Pa/m] at basic nodes

    # Gravity at basic nodes
    gravity: jax.Array  # [m/s^2]

    # Gravity at staggered nodes (cell centres). Mirrors numpy's
    # ``EntropyPhaseEvaluator(gravitational_acceleration=g_stag)`` for
    # the staggered phase evaluator. Built from the external EOS
    # gravity column at staggered radii (UserDefinedEOS / Zalmoxis) or
    # scalar broadcast when the column is unavailable
    # (AdamsWilliamsonEOS).
    gravity_stag: jax.Array  # [m/s^2]

    def __init__(
        self,
        d_dr_matrix,
        quantity_matrix,
        area,
        volume,
        radii_basic,
        mixing_length,
        mixing_length_sq,
        mixing_length_cu,
        radii_stag,
        P_stag,
        P_basic,
        gravity,
        dP_dr_basic=None,
        gravity_stag=None,
    ):
        """Custom init so callers that predate the dP_dr_basic field
        keep working. When ``dP_dr_basic`` is not supplied we derive it
        from ``P_basic`` and ``radii_basic`` via numpy gradient (matches
        ``entropy_state._dP_dr_basic = np.gradient(P_basic, r_basic)``).
        """
        self.d_dr_matrix = d_dr_matrix
        self.quantity_matrix = quantity_matrix
        self.area = area
        self.volume = volume
        self.radii_basic = radii_basic
        self.mixing_length = mixing_length
        self.mixing_length_sq = mixing_length_sq
        self.mixing_length_cu = mixing_length_cu
        self.radii_stag = radii_stag
        self.P_stag = P_stag
        self.P_basic = P_basic
        self.gravity = gravity
        if dP_dr_basic is None:
            import numpy as _np

            dP_dr_basic = jnp.asarray(
                _np.gradient(_np.asarray(P_basic), _np.asarray(radii_basic))
            )
        self.dP_dr_basic = dP_dr_basic
        # Default ``gravity_stag`` to a midpoint average of ``gravity``
        # so callers that don't supply a per-staggered profile still
        # see a reasonable approximation. ``from_numpy_mesh`` overrides
        # this with the EOS-interpolated value at staggered radii
        # whenever the column is available.
        if gravity_stag is None:
            g_arr = jnp.asarray(gravity)
            gravity_stag = 0.5 * (g_arr[:-1] + g_arr[1:])
        self.gravity_stag = jnp.asarray(gravity_stag)

    @staticmethod
    def from_numpy_mesh(mesh) -> 'MeshArrays':
        """Build from a numpy Aragog Mesh object."""
        import numpy as np

        P_basic_arr = np.asarray(mesh.basic_pressure).ravel()
        r_basic_arr = np.asarray(mesh.basic.radii).ravel()
        r_stag_arr = np.asarray(mesh.staggered.radii).ravel()
        return MeshArrays(
            d_dr_matrix=jnp.asarray(mesh._d_dr_transform),
            quantity_matrix=jnp.asarray(mesh._quantity_transform),
            area=jnp.asarray(np.asarray(mesh.basic.area).ravel()),
            volume=jnp.asarray(np.asarray(mesh.basic.volume).ravel()),
            radii_basic=jnp.asarray(r_basic_arr),
            radii_stag=jnp.asarray(r_stag_arr),
            mixing_length=jnp.asarray(np.asarray(mesh.basic.mixing_length).ravel()),
            mixing_length_sq=jnp.asarray(np.asarray(mesh.basic.mixing_length_squared).ravel()),
            mixing_length_cu=jnp.asarray(np.asarray(mesh.basic.mixing_length_cubed).ravel()),
            P_stag=jnp.asarray(np.asarray(mesh.staggered_pressure).ravel()),
            P_basic=jnp.asarray(P_basic_arr),
            # SPIDER-parity dP/dr at basic nodes via numpy gradient (matches
            # entropy_state._dP_dr_basic = np.gradient(P_basic, r_basic)).
            dP_dr_basic=jnp.asarray(np.gradient(P_basic_arr, r_basic_arr)),
            # Per-node gravity profile when the external mesh file
            # carries eos_gravity (UserDefinedEOS / Zalmoxis path),
            # else scalar broadcast. The per-node array is aligned to
            # the same basic-node grid as area / volume / mixing_length,
            # so the MLT buoyancy cascade in compute_mlt picks up the
            # radial dependence without any downstream broadcasting
            # change. Scalar fallback chain:
            # mesh.eos._gravitational_acceleration (AdamsWilliamsonEOS),
            # mesh.settings.gravitational_acceleration (external EOS
            # without the private attribute), then 9.81 m/s^2.
            gravity=_build_gravity_array(mesh, r_stag=False),
            # Same construction at the staggered radii. Mirrors numpy's
            # entropy_solver.py ``g_stag = np.interp(r_stag, eos_radius,
            # eos_gravity)`` (with scalar fallback). Aligned to the
            # staggered grid.
            gravity_stag=_build_gravity_array(mesh, r_stag=True),
        )


# ---------------------------------------------------------------------------
# Per-basic-node gravity array builder (used by MeshArrays.from_numpy_mesh)
# ---------------------------------------------------------------------------


def _build_gravity_array(mesh, r_stag: bool = False) -> 'jax.Array':
    """Return the per-node gravity array for a numpy Aragog mesh.

    Per-node profile is preferred when the external mesh file (eos_method=2,
    UserDefinedEOS / Zalmoxis path) carries an eos_gravity column aligned
    with an eos_radius grid. Interpolates that column onto the aragog
    target radii using numpy linear interpolation. Falls back to a
    scalar broadcast from the pressure-EOS attribute or the mesh.settings
    default when the external profile is not available (AdamsWilliamsonEOS,
    dummy structure, or a short-circuited setup path).

    Parameters
    ----------
    r_stag : bool, default False
        Target grid: False -> basic-node radii (default), True ->
        staggered-node radii.
    """
    import numpy as np

    if r_stag:
        r_target = np.asarray(mesh.staggered.radii).ravel()
    else:
        r_target = np.asarray(mesh.basic.radii).ravel()
    # ``r_basic`` aliases the requested target grid (basic by default,
    # staggered when r_stag=True) to keep the diagnostic logging below
    # readable.
    r_basic = r_target
    # Real Aragog Mesh objects expose the loaded EOS columns on
    # mesh.settings (which aliases Parameters.mesh) after
    # Parameters.__post_init__ has run np.loadtxt on the eos_file. Test
    # stubs supply mesh.parameters directly. Tolerate both layouts.
    _eos_src = None
    for attr in ('settings', 'parameters'):
        _candidate = getattr(mesh, attr, None)
        if _candidate is not None and hasattr(_candidate, 'eos_gravity'):
            _eos_src = _candidate
            break
    if _eos_src is None:
        _eos_src = mesh
    eos_gravity_arr = np.asarray(getattr(_eos_src, 'eos_gravity', []), dtype=float).ravel()
    eos_radius_arr = np.asarray(getattr(_eos_src, 'eos_radius', []), dtype=float).ravel()
    if eos_gravity_arr.size > 1 and eos_radius_arr.size == eos_gravity_arr.size:
        # Monotonicity guard: np.interp silently produces wrong results if
        # xp is not monotonically increasing. Zalmoxis writes radius
        # CMB-outward, but an external EOS file could in principle be
        # reversed; catch that rather than silently corrupting gravity.
        if not np.all(np.diff(eos_radius_arr) > 0):
            raise ValueError(
                'eos_radius is not monotonically increasing; '
                'np.interp would silently corrupt the gravity profile. '
                'Sort columns 0 and 3 of the external EOS file so radius '
                'is monotonic.'
            )
        g_basic = np.interp(r_basic, eos_radius_arr, eos_gravity_arr)
        return jnp.asarray(g_basic)
    # Scalar fallback chain. Mirrors the numpy reference in
    # entropy_solver.py::_initialize_internals: try the pressure-EOS
    # attribute first, then mesh.settings, then mesh.parameters, then
    # the 9.81 hard default. Both ``settings`` and ``parameters`` are
    # consulted because configs populated via the attrs facade live on
    # mesh.parameters and configs built via the dataclass parser live
    # on mesh.settings; reading only one would silently fall through
    # to 9.81 m/s^2 for a non-Earth planet.
    settings_src = getattr(mesh, 'settings', None) or getattr(mesh, 'parameters', None) or mesh
    g_scalar = abs(
        float(
            getattr(
                mesh.eos,
                '_gravitational_acceleration',
                getattr(settings_src, 'gravitational_acceleration', 9.81),
            )
        )
    )
    return jnp.asarray(np.full(r_basic.size, g_scalar))


# ---------------------------------------------------------------------------
# JAX tanh_weight (replaces aragog.utilities.tanh_weight)
# ---------------------------------------------------------------------------


def tanh_weight(x: jax.Array, threshold: float, width: float) -> jax.Array:
    """Smooth step function: 0.5 * (1 + tanh((x - threshold) / width))."""
    return 0.5 * (1.0 + jnp.tanh((x - threshold) / width))


def spider_get_smoothing(gphi: jax.Array, smooth_width: float) -> jax.Array:
    """SPIDER two-branch tanh phase smoothing (port of ``util.c:245-270``).

    For ``gphi > 0.5`` ramps down to zero near gphi=1; for ``gphi <= 0.5``
    ramps up from zero near gphi=0. The branches meet continuously at
    gphi=0.5. Matches numpy ``_spider_get_smoothing`` in
    ``entropy_state.py`` used when ``phase_smoothing='tanh'``.

    ``smooth_width`` is the tanh transition width in gphi units (SPIDER
    ``matprop_smooth_width``, default 0.01 for CHILI Earth).
    """
    upper = 0.5 * (1.0 - jnp.tanh((gphi - 1.0) / smooth_width))
    lower = 0.5 * (1.0 + jnp.tanh(gphi / smooth_width))
    return jnp.where(gphi > 0.5, upper, lower)


def phase_boundary_smoothing(
    gphi: jax.Array,
    params: PhaseParams,
) -> jax.Array:
    """Blend of cubic-Hermite and SPIDER tanh smoothing selected by
    ``params.phase_smoothing_tanh`` (0.0 or 1.0). Keeps the trace static.

    Cubic Hermite: ``16 * g_clip^2 * (1 - g_clip)^2`` with hard clip to
    [0,1]. Provides intermediate-phi damping (smth=0.32 at gphi=0.83).

    SPIDER tanh: two-branch tanh with width ``params.phase_smoothing_width``
    (default 0.01). smth ≈ 1 across [0.05, 0.95].
    """
    gphi_clip = jnp.clip(gphi, 0.0, 1.0)
    smth_cubic = 16.0 * gphi_clip**2 * (1.0 - gphi_clip) ** 2
    smth_tanh = spider_get_smoothing(gphi, params.phase_smoothing_width)
    w = params.phase_smoothing_tanh
    return w * smth_tanh + (1.0 - w) * smth_cubic


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
    # SPIDER-parity single-pass evaluation: T, rho, Cp, alpha, dTdPs, k
    # all derived from one shared (S_sol, S_liq, gphi, smth, phase-boundary)
    # cache, with the smth-blend mixed<->single matching numpy
    # EntropyPhaseEvaluator._update_eos.
    state = eos.compute_phase_state(
        P,
        S,
        k_solid=params.k_solid,
        k_liquid=params.k_liquid,
        matprop_smooth_width=params.matprop_smooth_width,
    )
    phi = state.melt_fraction

    # Viscosity: two-stage blend mirroring numpy entropy_phase.py:311-327
    # (used downstream by MLT -> kappa_c -> Jmix, which is why both stages
    # matter). Stage 1: tanh blend at phi_rheo (SPIDER util.c:255-259).
    # Stage 2: combine_matprop with the cached matprop_smooth_width smth
    # that compute_phase_state also uses for T/rho/Cp/alpha/k.
    w = tanh_weight(phi, params.phi_rheo, params.phi_width)
    log_visc_mixed = (1.0 - w) * params.log10_visc_solid + w * params.log10_visc_liquid
    log_visc_single = jnp.where(
        phi > 0.5,
        params.log10_visc_liquid,
        params.log10_visc_solid,
    )
    log_visc = state.smth * log_visc_mixed + (1.0 - state.smth) * log_visc_single
    viscosity = 10.0**log_visc
    kinematic_viscosity = viscosity / state.density

    # Capacitance for entropy equation
    capacitance = state.density * state.temperature

    return PhaseProperties(
        temperature=state.temperature,
        density=state.density,
        heat_capacity=state.heat_capacity,
        thermal_expansivity=state.thermal_expansivity,
        dTdPs=state.dTdPs,
        melt_fraction=phi,
        viscosity=viscosity,
        kinematic_viscosity=kinematic_viscosity,
        thermal_conductivity=state.thermal_conductivity,
        latent_heat=state.latent_heat,
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

    # Porosity (volume fraction of melt) from densities. Smoothed with
    # sqrt-based soft clip + soft max so the CVODE BDF predictor sees a
    # C^infty RHS (mirrors entropy_phase.py:421-438 and the solver-order
    # regression it avoids).
    drho = rho_s - rho_l
    eps = 1.0e-3
    drho_smoothmax = 0.5 * (drho + 1.0 + jnp.sqrt((drho - 1.0) ** 2 + eps * eps))
    porosity_raw = (rho_s - density) / drho_smoothmax
    eps_p = 1.0e-3
    p_lo = 0.5 * (porosity_raw + jnp.sqrt(porosity_raw * porosity_raw + eps_p * eps_p))
    hi_u = 1.0 - p_lo
    porosity = 1.0 - 0.5 * (hi_u + jnp.sqrt(hi_u * hi_u + eps_p * eps_p))

    por = jnp.maximum(porosity, 1e-20)
    one_m_por = jnp.maximum(1.0 - porosity, 1e-20)

    # Three permeability regimes
    F_bkc = d**2 * por**2 / (one_m_por**2 * 1000.0)
    F_rg = d**2 * por**4.5 * (5.0 / 7.0)
    F_stokes = d**2 * 2.0 / 9.0

    # Smooth regime switching at critical porosities (Abe 1995;
    # Soucasse Aragog formulation): BKC -> RG at 0.0769452 (analytical
    # equality of the BKC and RG permeabilities), RG -> Stokes at 0.771462.
    w_rg = tanh_weight(porosity, 0.0769452, 0.02)
    w_stokes = tanh_weight(porosity, 0.771462, 0.05)
    F = (1.0 - w_rg) * F_bkc + (w_rg - w_stokes) * F_rg + w_stokes * F_stokes
    F = jnp.maximum(F, 0.0)

    # Smoothed |delta_rho|: sqrt(x^2 + eps^2) with eps tiny vs physical
    # density contrast ~ 500 kg/m^3. Matches entropy_phase.py:464.
    abs_drho = jnp.sqrt(delta_rho * delta_rho + 1.0e-12)
    return abs_drho * gravity * F / jnp.maximum(eta_l, 1e-10)


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

    # Buoyancy from entropy gradient. ``jnp.abs`` has a non-differentiable
    # kink at zero (subgradient anywhere in [-1, +1] is mathematically
    # valid, but JAX returns sign(0) = 0 in fwd and the backward pass
    # through the abs and the downstream multiplications can produce
    # NaN at exact-zero entropy gradients. Numpy uses
    # ``_smooth_abs_neg(dSdr, eps=1e-30)`` which is the smoothed
    # ``max(-x, 0)``; we use the differentiable
    # ``0.5*(|x| + sqrt(x^2 + eps^2)) ~ |x|`` instead, which keeps the
    # forward equal to ``abs(x)`` to ULP precision while delivering a
    # finite analytic gradient everywhere.
    eps_abs = 1.0e-30
    abs_dSdr_safe = 0.5 * (jnp.abs(dSdr) + jnp.sqrt(dSdr * dSdr + eps_abs * eps_abs))
    effective_superadiabatic = alpha * T * abs_dSdr_safe / jnp.maximum(Cp, 1.0)
    velocity_prefactor = mesh.gravity * effective_superadiabatic

    # Convective mask: unstable when dS/dr < 0.
    # Hard mask matching the numpy reference. Using jnp.where (not boolean
    # indexing) for JAX traceability. Produces exactly 0 at stable nodes,
    # avoiding spurious convection from a soft sigmoid.
    conv_mask = jnp.where(dSdr < 0.0, 1.0, 0.0)

    # Viscous velocity (Re <= Re_crit)
    viscous_velocity = (velocity_prefactor * mesh.mixing_length_cu / (18.0 * nu)) * conv_mask

    # Inviscid velocity (Re > Re_crit). ``jnp.sqrt(jnp.maximum(x, 0))`` is
    # the textbook NaN-safe forward, but its backward gradient at x=0 is
    # ``0.5 * 0 / sqrt(0) = 0/0 = NaN`` (the maximum's stop-at-zero
    # subgradient kills the divisor protection). Numpy avoids this with
    # ``np.sqrt(x + 1e-20)`` (always-positive argument). Mirror it here:
    # eps**2 = 1e-40 is far below any physical inviscid_velocity_sq, so
    # the forward is bit-equivalent for non-trivial x.
    eps_sqrt = 1.0e-20
    inviscid_velocity_sq = (velocity_prefactor * mesh.mixing_length_sq / 16.0) * conv_mask
    inviscid_velocity = jnp.sqrt(inviscid_velocity_sq + eps_sqrt)

    # Reynolds number
    reynolds = viscous_velocity * mesh.mixing_length / nu

    # Smooth blend between viscous and inviscid regimes. The narrow
    # blend_width (0.01 * RE_CRIT) keeps inviscid k_h confined to the
    # convecting regime; widening the blend leaks inviscid mixing
    # into the solid regime and induces T_core bistability.
    blend_width = 0.01 * RE_CRIT
    inviscid_weight = 0.5 * (
        1.0 + jnp.tanh((reynolds - RE_CRIT) / jnp.maximum(blend_width, 1e-30))
    )

    # Raw eddy diffusivity
    kh_raw = (
        (1.0 - inviscid_weight) * viscous_velocity + inviscid_weight * inviscid_velocity
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

    # kappa_h floor (phase-dependent, modulated by melt fraction).
    # Production CHILI runs use kappah_floor = 10 m^2/s; the
    # phi-modulated f_floor ramps from 0 in solid layers (no spurious
    # convective flux) to ~1 in mushy/liquid layers, where physical
    # convection is expected and MLT can otherwise numerically freeze.
    # See solver/entropy_state.py for the same comment block on the
    # numpy path.
    phi_basic = phase_basic.melt_fraction
    f_floor = tanh_weight(phi_basic, 0.4, 0.15)
    kh_floor = params.kappah_floor * f_floor
    kappa_h = jnp.maximum(kappa_h, kh_floor)

    # SPIDER energy.c:220-223 CMB fix: use kappa_h from the first interior
    # node at the CMB basic node, since kappa_h is a nonlinear function of
    # dSdr and the boundary extrapolation can over- or under-estimate it
    # relative to the interior value. Mirrors numpy entropy_state.py:533.
    kappa_h = kappa_h.at[0].set(kappa_h[1])

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

    # SPIDER ic.c:450 boundary-copy convention for dSdr at the surface
    # (mirrors numpy entropy_state.py:390 ``dSdxi[-1] = dSdxi[-2]``).
    # The d_dr_matrix gives a linear-extrapolation gradient at the top
    # basic node, but the SPIDER convention is to copy the adjacent
    # interior value. Without this, the surface dSdr can flip sign
    # relative to numpy, blowing up kappa_h and dS/dt at the second-
    # to-last staggered cell. The CMB side has its own override path
    # via ``dSdr_cmb_override``, so the surface copy here is the only
    # one needed at this layer.
    dSdr = dSdr.at[-1].set(dSdr[-2])

    # Apply energy_balance overrides at the CMB basic node.
    # Done as separate jax.lax.cond branches so the function remains
    # JIT-compatible regardless of whether the overrides are None or
    # JAX scalars. We use jnp.where with a static bool flag passed
    # through from the caller via the optional argument.
    if S_basic_cmb_override is not None:
        S_basic = S_basic.at[0].set(S_basic_cmb_override)
    if dSdr_cmb_override is not None:
        dSdr = dSdr.at[0].set(dSdr_cmb_override)
    else:
        # Mirror numpy entropy_state.py:389 ``dSdxi[0] = dSdxi[1]`` for
        # the non-energy_balance modes (quasi_steady etc.) where there
        # is no boundary-state override. In energy_balance mode the
        # explicit override above wins and this branch is skipped.
        dSdr = dSdr.at[0].set(dSdr[1])

    # Phase properties at basic nodes only. The SPIDER-bracket Jmix
    # is built entirely from basic-node quantities, so staggered-node
    # phase properties are not materialised here.
    phase_basic = evaluate_phase(eos, params, mesh.P_basic, S_basic)

    # MLT eddy diffusivity
    kappa_h, kappa_c = compute_mlt(dSdr, phase_basic, mesh, params)

    # Basic node properties
    rho = phase_basic.density
    T = phase_basic.temperature
    Cp = phase_basic.heat_capacity
    k = phase_basic.thermal_conductivity
    dTdPs_basic = phase_basic.dTdPs

    # Heat flux components (multiply by flag to enable/disable)
    heat_flux = jnp.zeros_like(S_basic)

    # Conduction (SPIDER decomposition matching numpy entropy_state):
    #   F_cond = -k * [(T/Cp) * dS/dr + dT/dr|_adiabat]
    # where the adiabatic gradient is dT/dr|_ad = dTdPs * dPdr_basic
    # (EOS-table dT/dP|_S times the structural Adams-Williamson dP/dr).
    # This avoids the noise from finite-differencing T_stag and matches
    # SPIDER's eos-consistent conduction at phase boundaries.
    Cp_safe = jnp.maximum(Cp, 100.0)
    superadiabatic = (T / Cp_safe) * dSdr
    dT_dr_adiabat = dTdPs_basic * mesh.dP_dr_basic
    heat_flux = heat_flux + params.conduction * (-k * (superadiabatic + dT_dr_adiabat))

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
        eos,
        params,
        mesh.P_basic,
        rho,
        phi_b,
        mesh.gravity,
    )
    jgrav_raw = rho * phi_b * (1.0 - phi_b) * v_rel

    # gphi at STAGGERED nodes (cell below each basic interface)
    S_sol_stag = eos.solidus_entropy(mesh.P_stag)
    S_liq_stag = eos.liquidus_entropy(mesh.P_stag)
    dS_stag = jnp.maximum(S_liq_stag - S_sol_stag, 1.0)
    gphi_stag = (S_stag - S_sol_stag) / dS_stag

    smth_stag = phase_boundary_smoothing(gphi_stag, params)

    # Map staggered smoothing to basic-node interfaces: interior basic
    # node i (1..N-2) sees the smoothing of staggered node i-1 (the
    # cell BELOW). Boundary basic nodes (0 and -1) use smth = 1 as a
    # placeholder because the mass flux at those indices is zeroed a
    # few lines below anyway. Lengths: staggered has N entries, basic
    # has N+1; smth_stag[:-1] supplies the N-1 interior interfaces
    # plus the two boundaries, totalling N+1.
    smth_basic = jnp.concatenate(
        [
            jnp.array([1.0]),
            smth_stag[:-1],
            jnp.array([1.0]),
        ]
    )

    # `bottom_up_grav_sep = 1.0` selects the smoothed flux, 0.0 selects
    # the raw flux (for reproducing the pre-fix drain in regression
    # tests).
    jgrav_smoothed = jgrav_raw * smth_basic
    jgrav = (
        params.bottom_up_grav_sep * jgrav_smoothed
        + (1.0 - params.bottom_up_grav_sep) * jgrav_raw
    )
    mass_flux = mass_flux + params.grav_sep * jgrav

    # Zero mass fluxes at boundaries (SPIDER convention)
    mass_flux = mass_flux.at[0].set(0.0)
    mass_flux = mass_flux.at[-1].set(0.0)

    # Add latent heat transport from mass flux
    heat_flux = heat_flux + mass_flux * phase_basic.latent_heat

    # Mixing flux (SPIDER-parity bracket form, heat flux only).
    #
    # Mirrors numpy ``entropy_state.update`` (entropy_state.py:656-692):
    #     Jmix_heat = -kappa_c * rho * T_fus * bracket * smth_basic_mix
    #     bracket = dS/dr - [phi·dS_liq/dP + (1-phi)·dS_sol/dP] · dP/dr
    # evaluated at basic nodes. The mass-flux form
    # ``mass_flux += mixing * rho * kappa_c * (-dphi/dr)`` (delivered
    # via the latent-heat term) is NOT equivalent and diverges from
    # the numpy production path.
    #
    # The smth polynomial ``16·gphi²·(1-gphi)²`` at basic nodes zeroes
    # the flux outside the mushy band; the bracket itself is finite in
    # pure phases because dS_sol/dP and dS_liq/dP are bounded.
    S_sol_basic = eos.solidus_entropy(mesh.P_basic)
    S_liq_basic = eos.liquidus_entropy(mesh.P_basic)
    dS_phase_basic = jnp.maximum(S_liq_basic - S_sol_basic, 1.0)
    dS_sol_dP_basic = eos.solidus_entropy_dP(mesh.P_basic)
    dS_liq_dP_basic = eos.liquidus_entropy_dP(mesh.P_basic)
    phi_clipped = phase_basic.melt_fraction
    bracket = (
        dSdr
        - (phi_clipped * dS_liq_dP_basic + (1.0 - phi_clipped) * dS_sol_dP_basic)
        * mesh.dP_dr_basic
    )

    # T_fus = latent_heat / dS_phase (mirrors numpy
    # _ensure_basic_phase_boundary_cache: T_fus_basic = L_basic / dS_phase).
    T_fus_basic = phase_basic.latent_heat / dS_phase_basic

    # Smoothing: gphi at basic nodes, same smoothing family as Jgrav
    # (cubic Hermite or SPIDER tanh, selected by params.phase_smoothing_tanh).
    gphi_basic = (S_basic - S_sol_basic) / dS_phase_basic
    smth_basic_mix = phase_boundary_smoothing(gphi_basic, params)

    jmix_heat = -kappa_c * rho * T_fus_basic * bracket * smth_basic_mix
    # Zero at CMB and surface (no mass/heat transfer across those boundaries)
    jmix_heat = jmix_heat.at[0].set(0.0)
    jmix_heat = jmix_heat.at[-1].set(0.0)
    heat_flux = heat_flux + params.mixing * jmix_heat

    return FluxOutput(
        heat_flux=heat_flux,
        mass_flux=mass_flux,
        eddy_diffusivity=kappa_h,
        heating=heating_rate,
        jmix_heat=jmix_heat,
    )
