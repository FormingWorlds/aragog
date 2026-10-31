"""JAX phase evaluator and flux computation for the entropy solver.

Pure-functional replacements for ``entropy_phase.py`` (phase properties)
and ``entropy_state.py`` (MLT convection, heat/mass fluxes). All functions
are JIT-compilable and differentiable.

The numpy versions mutate state arrays in-place. The JAX versions take
arrays in and return NamedTuples out, with no side effects.

Dependencies: jax, equinox (already in PROTEUS ecosystem via atmodeller).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator

from aragog.config.energy import CONVECTIVE_THRESHOLD_DEFAULT, check_convective_threshold
from aragog.config.phases import (
    R_GAS,
    SEPARATION_BUOYANCY_DEFAULT,
    SEPARATION_BUOYANCY_MODES,
    SEPARATION_VISCOSITY_DEFAULT,
    SEPARATION_VISCOSITY_MODES,
    SolidRheologyParams,
    _compute_arrhenius_viscosity_core,
    _compute_effective_viscosity_core,
    _compute_stagnant_lid_state_core,
    _compute_yield_stress_core,
    _viscous_mixing_length_factor_core,
    check_viscosity_ceiling,
    merge_rheology,
)
from aragog.jax.eos import EntropyEOS_JAX
from aragog.mesh import top_node_entropy
from aragog.utilities import positive_gravity, zero_anchored_weight

# Enable float64
jax.config.update('jax_enable_x64', True)

# Critical Reynolds number, Abe (1995) via Bower et al. (2018) section 2.1
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
    solidus_temperature: jax.Array | None = None  # [K]
    eta_diff: jax.Array | None = None  # [Pa s]
    tau_y: jax.Array | None = None  # [Pa]
    smth: jax.Array | None = None  # [-]
    visc_solid_weight: jax.Array | None = None  # [-]


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
    jmix_heat: jax.Array  # phase-mixing heat flux at basic nodes
    # [W/m^2] -- the SPIDER-parity convective
    # heat flux carried by the chemical mass
    # flux ``jmix`` (NOT the same as the MLT
    # convective flux F_conv, which lives in
    # ``heat_flux`` already). Returned raw,
    # NOT gated by ``params.mixing``; exposed
    # separately so diagnostic post-processing
    # can inspect the phase-mixing
    # contribution independently of whether
    # the integrator's RHS includes it.


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
    grain_size: float

    # Thermal conductivity
    k_solid: float
    k_liquid: float

    # SPIDER -matprop_smooth_width: tanh blend width for the
    # mixed-vs-single phase blend. 0.0 = sharp (smth=1 inside [0,1]),
    # 0.01 = production setting.
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
    # Minimum drive gradient on unstable nodes and floor band [J/kg/K/m].
    convective_threshold: float

    # SPIDER-analogue bottom-up gate for Jgrav. Stored as float for JAX
    # tracing (1.0 = smoothing on, 0.0 = raw un-smoothed flux). Keep ON
    # for production runs; setting this to 0.0 reproduces the pre-fix
    # CMB drain and is only useful for regression tests.
    bottom_up_grav_sep: float

    # Phase-boundary smoothing selection for Jgrav and Jmix.
    # phase_smoothing_tanh: 1.0 -> use SPIDER's two-branch tanh
    # (_spider_get_smoothing); 0.0 -> use cubic Hermite 16*g^2*(1-g)^2.
    # phase_smoothing_width: tanh transition width in gphi units
    # (ignored when phase_smoothing_tanh == 0.0). SPIDER's
    # ``matprop_smooth_width`` defaults to 0.01.
    phase_smoothing_tanh: float
    phase_smoothing_width: float

    # separation_viscosity_mixture: above 0.5 -> separation drag with the blended mixture
    # viscosity; otherwise the liquid viscosity (SPIDER's GetGravitationalHeatFlux).
    separation_viscosity_mixture: float
    # separation_buoyancy_mixture: above 0.5 -> 1 - phi on the BKC and RG mobility
    # (McKenzie 1984); otherwise buoyancy against the solid (Abe 1995, SPIDER).
    separation_buoyancy_mixture: float
    rheology: SolidRheologyParams = eqx.field(static=True, default=SolidRheologyParams())

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
        separation_viscosity: str = SEPARATION_VISCOSITY_DEFAULT,
        convective_threshold: float = CONVECTIVE_THRESHOLD_DEFAULT,
        separation_buoyancy: str = SEPARATION_BUOYANCY_DEFAULT,
        rheology: SolidRheologyParams | None = None,
    ):
        self.phi_rheo = phi_rheo
        self.phi_width = phi_width
        self.log10_visc_solid = jnp.log10(viscosity_solid)
        self.log10_visc_liquid = jnp.log10(viscosity_liquid)
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
        check_convective_threshold(convective_threshold)
        self.convective_threshold = float(convective_threshold)
        self.bottom_up_grav_sep = float(bottom_up_grav_sep)
        if phase_smoothing not in ('cubic_hermite', 'tanh'):
            raise ValueError(
                f"phase_smoothing must be 'cubic_hermite' or 'tanh', got {phase_smoothing!r}"
            )
        self.phase_smoothing_tanh = 1.0 if phase_smoothing == 'tanh' else 0.0
        self.phase_smoothing_width = float(phase_smoothing_width)
        if separation_viscosity not in SEPARATION_VISCOSITY_MODES:
            raise ValueError(
                f'separation_viscosity must be one of {SEPARATION_VISCOSITY_MODES}, '
                f'got {separation_viscosity!r}'
            )
        # Explicit dispatch: a third mode added to SEPARATION_VISCOSITY_MODES
        # must fail here rather than silently fall back to 'melt'.
        if separation_viscosity == 'mixture':
            self.separation_viscosity_mixture = 1.0
        elif separation_viscosity == 'melt':
            self.separation_viscosity_mixture = 0.0
        else:
            raise ValueError(f'unhandled separation_viscosity {separation_viscosity!r}')
        if separation_buoyancy not in SEPARATION_BUOYANCY_MODES:
            raise ValueError(
                f'separation_buoyancy must be one of {SEPARATION_BUOYANCY_MODES}, '
                f'got {separation_buoyancy!r}'
            )
        self.separation_buoyancy_mixture = {'mixture': 1.0, 'solid': 0.0}[separation_buoyancy]
        rheo_obj = merge_rheology(rheology, {})
        rheo_obj.__post_init__()
        check_viscosity_ceiling(viscosity_solid, rheo_obj)
        if rheo_obj.enabled:
            if not np.isfinite(viscosity_solid) or viscosity_solid <= 0.0:
                raise ValueError('viscosity_solid must be finite and positive')
            if not np.isfinite(phi_rheo) or not (0.0 <= float(phi_rheo) <= 1.0):
                raise ValueError('phi_rheo must lie in [0, 1]')
            if not np.isfinite(phi_width) or float(phi_width) <= 0.0:
                raise ValueError('phi_width must be finite and positive')
        self.rheology = rheo_obj

    @classmethod
    def from_parameters(cls, parameters: Any, **overrides: Any) -> 'PhaseParams':
        """Construct ``PhaseParams`` from a ``Parameters`` object, with optional keyword overrides.

        Parameters
        ----------
        parameters : Parameters
            Parsed Aragog runtime parameter container holding ``phase_solid``,
            ``phase_liquid``, ``phase_mixed``, and ``energy`` sub-containers.
        **overrides : Any
            Optional keyword arguments overriding individual :class:`PhaseParams`
            constructor fields within their valid domains.

        Returns
        -------
        PhaseParams
            Validated static material and transport parameter bundle.

        Raises
        ------
        TypeError, ValueError
            If any extracted or overridden parameter violates the domain or
            rheology ceiling checks of :class:`PhaseParams`.
        """
        kwargs: dict[str, Any] = {
            'phi_rheo': float(parameters.phase_mixed.rheological_transition_melt_fraction),
            'phi_width': float(parameters.phase_mixed.rheological_transition_width),
            'viscosity_solid': float(parameters.phase_solid.viscosity),
            'viscosity_liquid': float(parameters.phase_liquid.viscosity),
            'grain_size': float(parameters.phase_mixed.grain_size),
            'k_solid': float(parameters.phase_solid.thermal_conductivity),
            'k_liquid': float(parameters.phase_liquid.thermal_conductivity),
            'matprop_smooth_width': float(parameters.phase_mixed.matprop_smooth_width),
            'conduction': bool(parameters.energy.conduction),
            'convection': bool(parameters.energy.convection),
            'grav_sep': bool(parameters.energy.gravitational_separation),
            'mixing': bool(parameters.energy.mixing),
            'eddy_diff_thermal': float(parameters.energy.eddy_diffusivity_thermal),
            'eddy_diff_chemical': float(parameters.energy.eddy_diffusivity_chemical),
            'kappah_floor': float(parameters.energy.kappah_floor),
            'bottom_up_grav_sep': bool(parameters.energy.bottom_up_grav_sep),
            'phase_smoothing': str(parameters.energy.phase_smoothing),
            'phase_smoothing_width': (
                float(parameters.phase_mixed.matprop_smooth_width) or 1.0e-2
            ),
            'separation_viscosity': str(parameters.phase_mixed.separation_viscosity),
            'convective_threshold': float(parameters.energy.convective_threshold),
            'separation_buoyancy': str(
                getattr(
                    parameters.phase_mixed, 'separation_buoyancy', SEPARATION_BUOYANCY_DEFAULT
                )
            ),
            'rheology': getattr(parameters.phase_solid, 'rheology', None),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


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

    # Top basic node: limited (mesh.top_node == 'limited'), outer-cell spacing ratio, table range
    top_limited: bool = eqx.field(static=True)
    top_ratio: float
    top_bounds: jax.Array  # [J/kg/K], (S_min, S_max)

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
        top_limited=False,
        top_ratio=1.0,
        top_bounds=(-jnp.inf, jnp.inf),
        *,
        validate_rheology: bool = False,
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
        self.top_limited, self.top_ratio = bool(top_limited), float(top_ratio)
        self.top_bounds = jnp.asarray(top_bounds, dtype=float)
        if validate_rheology:
            self.validate_for_rheology()

    def validate_for_rheology(self) -> None:
        """Validate static mesh invariants required when solid-state rheology is enabled.

        Returns
        -------
        None
            Returns ``None`` when all mesh invariants hold.

        Raises
        ------
        ValueError
            If eager ``radii_basic`` [m] has fewer than 3 nodes or is not finite,
            strictly positive, and strictly increasing; if ``mixing_length`` [m]
            is non-finite or negative; if ``P_basic`` or ``P_stag`` [Pa] is
            non-finite or below ``-1.0`` Pa; or if ``gravity`` [m/s^2] is
            non-finite or non-positive.
        """
        if not isinstance(self.radii_basic, jax.core.Tracer):
            r_b = np.asarray(self.radii_basic, dtype=float).ravel()
            if (
                r_b.size < 3
                or not np.all(np.isfinite(r_b))
                or r_b[0] <= 0.0
                or not np.all(np.diff(r_b) > 0.0)
            ):
                raise ValueError(
                    'basic-node radii must be finite, positive, and strictly increasing with >= 3 nodes'
                )
        if not isinstance(self.mixing_length, jax.core.Tracer):
            ml = np.asarray(self.mixing_length, dtype=float).ravel()
            if not np.all(np.isfinite(ml)) or np.any(ml < 0.0):
                raise ValueError('basic-node mixing_length must be finite and non-negative')
        if not isinstance(self.P_basic, jax.core.Tracer):
            p_b = np.asarray(self.P_basic, dtype=float).ravel()
            if not np.all(np.isfinite(p_b)) or np.any(p_b < -1.0):
                raise ValueError(
                    'pressure at basic and staggered nodes must be finite and >= -1.0 Pa'
                )
        if not isinstance(self.P_stag, jax.core.Tracer):
            p_s = np.asarray(self.P_stag, dtype=float).ravel()
            if not np.all(np.isfinite(p_s)) or np.any(p_s < -1.0):
                raise ValueError(
                    'pressure at basic and staggered nodes must be finite and >= -1.0 Pa'
                )
        if not isinstance(self.gravity, jax.core.Tracer):
            g_b = np.asarray(self.gravity, dtype=float).ravel()
            if not np.all(np.isfinite(g_b)) or np.any(g_b <= 0.0):
                raise ValueError('gravity at basic nodes must be finite and positive')

    @staticmethod
    def from_numpy_mesh(mesh, *, validate_rheology: bool = False) -> 'MeshArrays':
        """Build from a numpy Aragog Mesh object."""
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
            top_limited=getattr(mesh, 'top_node', 'linear') == 'limited',
            top_ratio=getattr(mesh, 'top_ratio', 1.0),
            top_bounds=getattr(mesh, 'top_bounds', (-np.inf, np.inf)),
            validate_rheology=validate_rheology,
        )


def entropy_at_basic_nodes(mesh: MeshArrays, S: jax.Array) -> jax.Array:
    """Entropy at the basic nodes, with the limited top node of ``aragog.mesh`` when set."""
    S_basic = mesh.quantity_matrix @ S
    if mesh.top_limited:
        node = top_node_entropy(S, mesh.top_ratio, *mesh.top_bounds, xp=jnp, linear=S_basic[-1])
        return S_basic.at[-1].set(node)
    return S_basic


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
        # Monotonicity guard: PchipInterpolator silently produces
        # wrong results if x is not monotonically increasing. Zalmoxis
        # writes radius CMB-outward, but an external EOS file could in
        # principle be reversed; catch that rather than silently
        # corrupting gravity.
        if not np.all(np.diff(eos_radius_arr) > 0):
            raise ValueError(
                'eos_radius is not monotonically increasing; '
                'PchipInterpolator would silently corrupt the gravity '
                'profile. Sort columns 0 and 3 of the external EOS '
                'file so radius is monotonic.'
            )
        # PchipInterpolator (C^1 monotone cubic Hermite) for parity
        # with the pressure and density interpolations in
        # mesh.pressure_eos. Gravity is smooth wherever density is
        # smooth, so the cubic Hermite has no overshoot risk like the
        # solidus/liquidus 1-D entropy curves (where PCHIP was
        # rejected; see eos.entropy._load_spider_phase_boundary).
        if not np.all(np.isfinite(eos_gravity_arr)):
            raise ValueError('gravity profile eos_gravity must be finite on every row')
        g_basic = PchipInterpolator(eos_radius_arr, eos_gravity_arr)(r_basic)
        return jnp.asarray(positive_gravity(g_basic, 'interpolated mesh nodes'))
    # Scalar fallback chain. Mirrors the numpy reference in
    # entropy_solver.py::_initialize_internals: try the pressure-EOS
    # attribute first, then mesh.settings, then mesh.parameters. Both
    # ``settings`` and ``parameters`` are consulted because configs
    # populated via the attrs facade live on mesh.parameters and configs
    # built via the dataclass parser live on mesh.settings; reading only
    # one would silently fall through. If neither carries the field we
    # raise instead of defaulting to Earth gravity, since silently using
    # 9.81 m/s^2 on a non-Earth planet produces wrong physics.
    settings_src = getattr(mesh, 'settings', None) or getattr(mesh, 'parameters', None) or mesh
    _eos_g = getattr(mesh.eos, '_gravitational_acceleration', None)
    if _eos_g is None:
        _eos_g = getattr(settings_src, 'gravitational_acceleration', None)
    if _eos_g is None:
        raise ValueError(
            'No gravitational_acceleration available on mesh.eos, mesh.settings, '
            'or mesh.parameters and no per-node eos_gravity profile present. '
            'Set ``gravitational_acceleration`` on the mesh config (or supply '
            'an external EOS file with a gravity column) before constructing '
            'the JAX phase parameters.'
        )
    return jnp.asarray(positive_gravity(np.full(r_basic.size, float(_eos_g)), 'mesh nodes'))


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
    ``matprop_smooth_width``, default 0.01).
    """
    w_safe = jnp.maximum(smooth_width, 1.0e-30)
    upper = 0.5 * (1.0 - jnp.tanh((gphi - 1.0) / w_safe))
    lower = 0.5 * (1.0 + jnp.tanh(gphi / w_safe))
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


def blend_viscosity_with_solid_jax(
    phi: jax.Array,
    smth: jax.Array,
    log10_eta_solid: jax.Array | float,
    params: PhaseParams,
) -> jax.Array:
    """Blend solid and liquid dynamic viscosities across the mushy zone.

    Parameters
    ----------
    phi : jax.Array
        Mass melt fraction (dimensionless, in ``[0, 1]``).
    smth : jax.Array
        Two-phase boundary smoothing weight (dimensionless, in ``[0, 1]``).
    log10_eta_solid : jax.Array or float
        Base-10 logarithm of the solid-branch dynamic viscosity ``log10(eta_solid [Pa s])``,
        finite real scalar or array.
    params : PhaseParams
        Static phase parameters providing ``phi_rheo``, ``phi_width``,
        ``log10_visc_liquid``, and ``rheology.phi_visc_single``.

    Returns
    -------
    jax.Array
        Blended dynamic viscosity [Pa s], strictly positive in ``(0, +inf)``.
    """
    w = tanh_weight(phi, params.phi_rheo, params.phi_width)
    log_visc_mixed = (1.0 - w) * log10_eta_solid + w * params.log10_visc_liquid
    phi_thresh = params.rheology.phi_visc_single if params.rheology.enabled else 0.5
    log_visc_single = jnp.where(
        phi > phi_thresh,
        params.log10_visc_liquid,
        log10_eta_solid,
    )
    log_visc = smth * log_visc_mixed + (1.0 - smth) * log_visc_single
    return 10.0**log_visc


def evaluate_phase(
    eos: EntropyEOS_JAX,
    params: PhaseParams,
    P: jax.Array,
    S: jax.Array,
    *,
    include_rheology: bool = True,
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
    include_rheology : bool, default True
        When True and ``params.rheology.enabled`` is True, evaluate
        Arrhenius diffusion-creep viscosity and Byerlee yield stress
        for the solid branch.

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

    eta_diff = None
    tau_y = None
    w_blend = tanh_weight(phi, params.phi_rheo, params.phi_width)
    phi_thresh = params.rheology.phi_visc_single if params.rheology.enabled else 0.5
    visc_solid_weight = state.smth * (1.0 - w_blend) + (1.0 - state.smth) * jnp.where(
        phi > phi_thresh, 0.0, 1.0
    )
    if params.rheology.enabled and include_rheology:
        visc_solid = 10.0**params.log10_visc_solid
        p_pos = jnp.maximum(P, 0.0)
        eta_diff = _compute_arrhenius_viscosity_core(
            state.temperature,
            p_pos,
            viscosity_solid=visc_solid,
            g_homologous=params.rheology.g_homologous,
            solidus_temperature=state.solidus_temperature,
            t_sol_0=params.rheology.t_sol_0,
            dt_sol_dp=params.rheology.dt_sol_dp,
            arrhenius_t_ref=params.rheology.arrhenius_t_ref,
            r_gas=R_GAS,
            viscosity_max_log10=params.rheology.viscosity_max_log10,
            water_prefactor=params.rheology.water_prefactor,
            xp=jnp,
            activation_energy=params.rheology.activation_energy,
            activation_volume=params.rheology.activation_volume,
            activation_volume_decay_pressure=params.rheology.activation_volume_decay_pressure,
        )
        tau_y = _compute_yield_stress_core(
            p_pos,
            yield_stress_c=params.rheology.yield_stress_c,
            yield_stress_mu=params.rheology.yield_stress_mu,
            yield_stress_max=params.rheology.yield_stress_max,
            xp=jnp,
        )
        log10_eta_solid = jnp.log10(jnp.maximum(eta_diff, jnp.finfo(float).tiny))
        viscosity = blend_viscosity_with_solid_jax(phi, state.smth, log10_eta_solid, params)
    else:
        viscosity = blend_viscosity_with_solid_jax(
            phi, state.smth, params.log10_visc_solid, params
        )
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
        solidus_temperature=state.solidus_temperature,
        eta_diff=eta_diff,
        tau_y=tau_y,
        smth=state.smth,
        visc_solid_weight=visc_solid_weight,
    )


# ---------------------------------------------------------------------------
# Relative velocity (gravitational separation, Abe 1995)
# ---------------------------------------------------------------------------


def relative_velocity(
    eos: EntropyEOS_JAX,
    params: PhaseParams,
    P: jax.Array,
    density: jax.Array,
    melt_fraction: jax.Array,
    gravity: jax.Array,
    viscosity: jax.Array,
) -> jax.Array:
    """Melt-solid relative velocity for gravitational separation [m/s].

    Three-regime mobility model, the permeability over porosity that
    multiplies (rho_s - rho_l) g / eta (Abe 1995, Bower et al. 2018 section 2.1;
    SPIDER convention): Blake-Kozeny-Carman -> Rumpf-Gupte -> Stokes settling.
    The contrast is signed: the velocity is negative where the melt is denser.
    ``params.separation_buoyancy_mixture`` above 0.5 puts ``1 - phi`` on the BKC and RG
    terms (McKenzie 1984); otherwise the buoyancy is against the solid (Abe 1995).

    The drag viscosity is set by ``params.separation_viscosity_mixture``:
    0.0 uses the fixed single-phase liquid viscosity, matching SPIDER's
    ``GetGravitationalHeatFlux``; 1.0 uses the rheological-transition-
    blended mixture ``viscosity`` argument, so settling locks up through
    the same phi_rheo transition that sets the bulk rheology.
    """
    rho_s = eos._lookup_at_phase_boundary('density', P, 'solid')
    rho_l = eos._lookup_at_phase_boundary('density', P, 'melt')
    d = params.grain_size
    eta_melt = 10.0**params.log10_visc_liquid
    eta_l = jnp.where(params.separation_viscosity_mixture > 0.5, viscosity, eta_melt)

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

    # Three mobility regimes (permeability / porosity)
    F_bkc = d**2 * por**2 / (one_m_por**2 * 1000.0)
    F_rg = d**2 * por**4.5 * (5.0 / 7.0)
    F_stokes = d**2 * 2.0 / 9.0

    # Smooth regime switching at the equal-density-ratio crossings:
    # BKC -> RG at zeta_1 = 0.0769452 (1.7e-5 below the exact BKC-RG
    # crossing), RG -> Stokes at zeta_2 = 0.771462 (exact).
    w_rg = tanh_weight(porosity, 0.0769452, 0.02)
    w_stokes = tanh_weight(porosity, 0.771462, 0.05)
    F_matrix = (1.0 - w_rg) * F_bkc + (w_rg - w_stokes) * F_rg
    F_matrix = (
        jnp.where(params.separation_buoyancy_mixture > 0.5, 1.0 - porosity, 1.0) * F_matrix
    )
    F = jnp.maximum(F_matrix + w_stokes * F_stokes, 0.0)

    # Signed contrast, positive outward (melt rising), as the numpy path.
    return drho * gravity * F / jnp.maximum(eta_l, 1e-10)


# ---------------------------------------------------------------------------
# MLT eddy diffusivity (replaces the MLT block in EntropyState.update)
# ---------------------------------------------------------------------------


def _mlt_factor_powers(q_mlt: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return ``(q**0.25, q**0.75)`` with a finite derivative at ``q = 0``.

    Parameters
    ----------
    q_mlt : jax.Array
        Viscous mixing-length factor [-], ``>= 0``; it is 0 where the viscous length vanishes.

    Returns
    -------
    tuple of jax.Array
        ``q**0.25`` and ``q**0.75``; equal to the plain powers for ``q > 0`` and 0 at ``q = 0``,
        where the derivative is 0 instead of the NaN of ``inf * 0``.
    """
    q_pos = q_mlt > 0.0
    q_safe = jnp.where(q_pos, q_mlt, 1.0)
    return jnp.where(q_pos, q_safe**0.25, 0.0), jnp.where(q_pos, q_safe**0.75, 0.0)


def compute_mlt(
    dSdr: jax.Array,
    phase_basic: PhaseProperties,
    mesh: MeshArrays,
    params: PhaseParams,
    theta: jax.Array | float | None = None,
    return_aux: bool = False,
    return_q_025: bool = False,
    surface_floor: bool = True,
) -> tuple[jax.Array, ...]:
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
    theta : jax.Array or float, optional
        Frank-Kamenetskii contrast for viscous mixing-length rescaling.
    return_aux : bool, default False
        When True, also return ``(reynolds, viscous_velocity)`` as the
        third and fourth elements of the returned tuple.
    return_q_025 : bool, default False
        When True together with ``return_aux``, also append ``q_mlt**0.25``
        as the fifth element of the returned tuple.
    surface_floor : bool
        False under a prescribed surface temperature (outer BC 5): the kappa_h floor does
        not act at the surface node.

    Returns
    -------
    tuple of jax.Array
        ``(kappa_h, kappa_c)`` when ``return_aux`` is False;
        ``(kappa_h, kappa_c, reynolds, viscous_velocity)`` when True
        (or 5-tuple with ``q_025`` when ``return_q_025`` is True).
    """
    alpha = phase_basic.thermal_expansivity
    T = phase_basic.temperature
    Cp = phase_basic.heat_capacity
    nu = phase_basic.kinematic_viscosity

    # |dS/dr| as 0.5*(|x| + sqrt(x^2 + eps^2)): equal to |x| for |x| >> eps, with a
    # finite gradient at x = 0, where the backward pass of jnp.abs can give NaN.
    eps_abs = 1.0e-30
    abs_dSdr_safe = 0.5 * (jnp.abs(dSdr) + jnp.sqrt(dSdr * dSdr + eps_abs * eps_abs))
    # An unstable node is driven by at least the gradient D (convective_threshold).
    drive = jnp.maximum(abs_dSdr_safe, params.convective_threshold)
    effective_superadiabatic = alpha * T * drive / jnp.maximum(Cp, 1.0)
    velocity_prefactor = jnp.abs(mesh.gravity) * effective_superadiabatic

    # Hard masks matching the numpy reference: the mixing-length drive acts where
    # dS/dr < 0, the floor where dS/dr < D (convective_threshold).
    conv_mask = jnp.where(dSdr < 0.0, 1.0, 0.0)
    floor_mask = jnp.where(dSdr < params.convective_threshold, 1.0, 0.0)
    if not surface_floor:  # a surface held at its temperature (outer BC 5) gets no floor
        floor_mask = floor_mask.at[-1].set(0.0)

    eps_sqrt = 1.0e-20
    inviscid_velocity_sq = (velocity_prefactor * mesh.mixing_length_sq / 16.0) * conv_mask
    inviscid_velocity = jnp.sqrt(inviscid_velocity_sq + eps_sqrt)

    blend_width = 0.01 * RE_CRIT
    if params.rheology.enabled:
        q_mlt = _viscous_mixing_length_factor_core(
            radii=mesh.radii_basic,
            r_in=mesh.radii_basic[0],
            r_out=mesh.radii_basic[-1],
            mixing_length=mesh.mixing_length,
            bottom_slope=params.rheology.mlt_bottom_slope,
            theta=theta,
            s_top_inf=params.rheology.s_top_inf,
            c_theta=params.rheology.c_theta,
            xp=jnp,
        )
        q_025, q_075 = _mlt_factor_powers(q_mlt)
    else:
        q_025 = q_075 = jnp.ones_like(dSdr)
    viscous_velocity = (
        velocity_prefactor * mesh.mixing_length_cu * q_075 / (18.0 * nu)
    ) * conv_mask
    reynolds = viscous_velocity * mesh.mixing_length * q_025 / nu
    inviscid_weight = 0.5 * (
        1.0 + jnp.tanh((reynolds - RE_CRIT) / jnp.maximum(blend_width, 1e-30))
    )
    kh_raw = (
        (1.0 - inviscid_weight) * (viscous_velocity * q_025)
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

    # Melt-fraction-modulated kappa_h floor on cells with dS/dr < D only; see
    # solver/entropy_state.py:apply_kappah_floor for why cells with
    # dS/dr >= D keep their raw value.
    phi_basic = phase_basic.melt_fraction
    if params.rheology.enabled:
        f_floor = zero_anchored_weight(
            tanh_weight, phi_basic, params.phi_rheo, params.phi_width, jnp
        )
    else:
        f_floor = tanh_weight(phi_basic, params.phi_rheo, params.phi_width)
    kh_floor = params.kappah_floor * f_floor * floor_mask
    kappa_h = jnp.maximum(kappa_h, kh_floor)

    # SPIDER energy.c:220-223 CMB fix: use kappa_h from the first interior
    # node at the CMB basic node, since kappa_h is a nonlinear function of
    # dSdr and the boundary extrapolation can over- or under-estimate it
    # relative to the interior value. Mirrors numpy entropy_state.py:533.
    kappa_h = kappa_h.at[0].set(kappa_h[1])

    if return_aux:
        if return_q_025:
            return kappa_h, kappa_c, reynolds, viscous_velocity, q_025
        return kappa_h, kappa_c, reynolds, viscous_velocity
    return kappa_h, kappa_c


# ---------------------------------------------------------------------------
# Full flux computation (replaces EntropyState.update flux section)
# ---------------------------------------------------------------------------


def _matched_surface_gradient(dSdr, half_cell, t_top_cell, phase_basic, mesh):
    """dS/dr at the surface node from the half-cell temperature gradient (outer BC 5).

    Mirrors ``EntropyState._match_surface_gradient``: ``(Cp_safe / T) (dT/dr - dT/dP|_S dP/dr)``
    with ``dT/dr = (T_node - t_top_cell) / half_cell``. ``T_node``, the temperature at
    ``S_surf``, stands for the surface temperature, so a stale ``outer_bc_value`` is unused.
    """
    t_node = phase_basic.temperature[-1]
    cp = jnp.maximum(phase_basic.heat_capacity[-1], 100.0)
    dtdr = (t_node - t_top_cell) / half_cell
    dtdr_ad = phase_basic.dTdPs[-1] * mesh.dP_dr_basic[-1]
    return dSdr.at[-1].set(cp / t_node * (dtdr - dtdr_ad))


def compute_fluxes(
    S_stag: jax.Array,
    time: float,
    eos: EntropyEOS_JAX,
    params: PhaseParams,
    mesh: MeshArrays,
    heating_rate: jax.Array,
    S_basic_cmb_override=None,
    dSdr_cmb_override=None,
    surface_temperature=None,
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
        (basic-node index 0). Used by the energy_balance and
        core_module core BCs, which reconstruct S_basic[0] from the state-tracked
        dSdr_cmb via S[0] + dSdr_cmb * (r_basic[0] - r_stag[0]).
        When None, the standard quantity_matrix mapping is used.
    dSdr_cmb_override : float or None
        Optional override for the entropy gradient at the CMB
        basic node (dSdr[0]). Used by the energy_balance and
        core_module core BCs, where dSdr_cmb is a state-tracked variable. When None,
        the standard d_dr_matrix mapping is used.
    surface_temperature : tuple or None
        ``(S_surf, dr_half)`` of a prescribed surface temperature (outer BC 5): the surface
        node takes ``S_surf`` [J/kg/K], which sets its temperature, and the gradient of
        :func:`_matched_surface_gradient`. When None, the surface copy convention is used.

    Returns
    -------
    FluxOutput
        Heat flux, mass flux, eddy diffusivity, heating.
    """
    # Interpolate entropy to basic nodes and compute gradient
    S_basic = entropy_at_basic_nodes(mesh, S_stag)
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
        # Mirror numpy entropy_state.py:389 ``dSdxi[0] = dSdxi[1]`` for the modes without a
        # boundary-state override; energy_balance and core_module pass one above.
        dSdr = dSdr.at[0].set(dSdr[1])

    if surface_temperature is not None:
        S_basic = S_basic.at[-1].set(surface_temperature[0])

    # Phase properties at basic nodes only. The SPIDER-bracket Jmix
    # is built entirely from basic-node quantities, so staggered-node
    # phase properties are not materialised here.
    phase_basic = evaluate_phase(eos, params, mesh.P_basic, S_basic)
    if surface_temperature is not None:
        t_top_cell = evaluate_phase(eos, params, mesh.P_stag[-1:], S_stag[-1:]).temperature[0]
        dSdr = _matched_surface_gradient(
            dSdr, surface_temperature[1], t_top_cell, phase_basic, mesh
        )

    # MLT eddy diffusivity
    kappa_h, kappa_c, _, u_visc_1, q_025_1 = compute_mlt(
        dSdr,
        phase_basic,
        mesh,
        params,
        theta=None,
        return_aux=True,
        return_q_025=True,
        surface_floor=surface_temperature is None,
    )

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
    # Cp floor (100 J/kg/K) parity with numpy entropy_state.update. A
    # logger.warning inside this jit-compiled RHS would only fire
    # during tracing, not on actual data; the equivalent diagnostic
    # is emitted at load time by EntropySolver._check_eos_floors,
    # plus a once-per-instance runtime warning on the numpy path.
    Cp_safe = jnp.maximum(Cp, 100.0)
    superadiabatic = (T / Cp_safe) * dSdr
    dT_dr_adiabat = dTdPs_basic * mesh.dP_dr_basic
    jcond = params.conduction * (-k * (superadiabatic + dT_dr_adiabat))
    jconv = params.convection * (rho * T * kappa_h * (-dSdr))

    if params.rheology.enabled:
        kappa_th_b = k / (rho * Cp_safe)
        visc_solid = 10.0**params.log10_visc_solid
        eta_diff_b = (
            phase_basic.eta_diff
            if phase_basic.eta_diff is not None
            else jnp.full_like(T, visc_solid)
        )
        tau_y_b = (
            phase_basic.tau_y if phase_basic.tau_y is not None else jnp.full_like(T, jnp.inf)
        )
        if params.rheology.stress_closure_mode == 'local':
            lid_state = None
            eta_eff_b = _compute_effective_viscosity_core(
                eta_diff=eta_diff_b,
                stress_closure_mode='local',
                unyielded_velocity=u_visc_1,
                mixing_length=q_025_1 * mesh.mixing_length,
                tau_y=tau_y_b,
                xp=jnp,
            )
        else:
            lid_state = _compute_stagnant_lid_state_core(
                radii=mesh.radii_basic,
                temperature=T,
                pressure=jnp.maximum(mesh.P_basic, 0.0),
                convective_flux=jconv,
                total_flux=jcond + jconv,
                solidus_temperature=phase_basic.solidus_temperature,
                melt_fraction=phase_basic.melt_fraction,
                params=params.rheology,
                viscosity_solid=visc_solid,
                phi_rheo=params.phi_rheo,
                phi_width=params.phi_width,
                unyielded_velocity=u_visc_1,
                viscosity_mixture=phase_basic.viscosity,
                density=rho,
                gravity=jnp.abs(mesh.gravity),
                thermal_expansivity=phase_basic.thermal_expansivity,
                thermal_diffusivity=kappa_th_b,
                xp=jnp,
            )
            eta_eff_b = _compute_effective_viscosity_core(
                eta_diff=eta_diff_b,
                tau_d=lid_state['tau_d_lid'],
                tau_y_lid=lid_state['tau_y_lid'],
                stress_closure_mode='lid',
                w_lid=lid_state['w_lid'],
                eta_i=lid_state['eta_i'],
                xp=jnp,
            )
        log10_eta_eff = jnp.log10(jnp.maximum(eta_eff_b, jnp.finfo(float).tiny))
        mu_pass2 = blend_viscosity_with_solid_jax(
            phase_basic.melt_fraction, phase_basic.smth, log10_eta_eff, params
        )
        phase_basic = phase_basic._replace(
            viscosity=mu_pass2, kinematic_viscosity=mu_pass2 / rho
        )
        theta_pass2 = (
            None if params.rheology.stress_closure_mode == 'local' else lid_state['theta']
        )
        kappa_h, kappa_c = compute_mlt(
            dSdr,
            phase_basic,
            mesh,
            params,
            theta=theta_pass2,
            surface_floor=surface_temperature is None,
        )
        jconv = params.convection * (rho * T * kappa_h * (-dSdr))

    heat_flux = heat_flux + jcond
    heat_flux = heat_flux + jconv

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
        phase_basic.viscosity,
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
