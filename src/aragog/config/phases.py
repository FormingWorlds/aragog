"""Phase property configuration."""

from __future__ import annotations

import logging

import attrs

logger: logging.Logger = logging.getLogger('fwl.' + __name__)

# Allowed separation_viscosity modes and default, single-sourced here and
# imported by every site that parses, validates, or consumes the option
# (parser.py, eos/entropy_phase.py, jax/phase.py, solver/entropy_solver.py).
SEPARATION_VISCOSITY_MODES: tuple[str, str] = ('melt', 'mixture')
SEPARATION_VISCOSITY_DEFAULT: str = 'melt'

STRESS_CLOSURE_MODES: tuple[str, str] = ('local', 'global')
STRESS_CLOSURE_DEFAULT: str = 'local'

LID_BASE_MODES: tuple[str, str] = ('fixed', 'rheological')
LID_BASE_DEFAULT: str = 'fixed'


@attrs.define
class PhaseConfig:
    """Single-phase (solid or liquid) material properties.

    Each property can be a float (constant value) or a str (path to
    a lookup table file).

    Parameters
    ----------
    density : float or str
        Density [kg/m^3] or path to lookup.
    heat_capacity : float or str
        Heat capacity [J/(kg K)] or path to lookup.
    melt_fraction : float
        Melt fraction (0 for solid, 1 for liquid).
    thermal_conductivity : float or str
        Thermal conductivity [W/(m K)] or path to lookup.
    thermal_expansivity : float or str
        Thermal expansivity [1/K] or path to lookup.
    viscosity : float or str
        Dynamic viscosity [Pa s] or path to lookup.
    entropy : float or str
        Entropy [J/(kg K)] or path to lookup. Empty string means unused.
    enabled : bool
        If True, use Arrhenius viscosity and plastic yielding. If False, bypass them. Default False.
    activation_energy : float
        Arrhenius activation energy [J/mol]. Default 300e3.
    activation_volume : float
        Arrhenius activation volume [m^3/mol]. Default 5e-6.
    yield_stress_c : float
        Cohesion for yield stress [Pa]. Default 50e6.
    yield_stress_mu : float
        Friction coefficient for yield stress [-]. Default 0.6.
    stress_closure_mode : str
        Stress closure mode ('local' or 'global'). Default 'local'.
    """

    density: float | str
    heat_capacity: float | str
    melt_fraction: float
    thermal_conductivity: float | str
    thermal_expansivity: float | str
    viscosity: float | str
    entropy: float | str = ''
    enabled: bool = False
    activation_energy: float = attrs.field(default=300e3, validator=attrs.validators.ge(0.0))
    activation_volume: float = attrs.field(default=5e-6, validator=attrs.validators.ge(0.0))
    yield_stress_c: float = attrs.field(default=50e6, validator=attrs.validators.ge(0.0))
    yield_stress_mu: float = attrs.field(default=0.6, validator=attrs.validators.ge(0.0))
    stress_closure_mode: str = attrs.field(
        default=STRESS_CLOSURE_DEFAULT,
        validator=attrs.validators.in_(STRESS_CLOSURE_MODES),
    )
    arrhenius_t_ref: float = attrs.field(default=1600.0, validator=attrs.validators.gt(0.0))
    yield_stress_max: float = attrs.field(default=500.0e6, validator=attrs.validators.ge(0.0))
    viscosity_max_log10: float = attrs.field(default=40.0, validator=attrs.validators.gt(0.0))
    lid_base_mode: str = attrs.field(
        default=LID_BASE_DEFAULT,
        validator=attrs.validators.in_(LID_BASE_MODES),
    )
    lid_base_temperature: float = attrs.field(
        default=1400.0, validator=attrs.validators.gt(0.0)
    )
    lid_contrast_coeff: float = attrs.field(default=2.2, validator=attrs.validators.gt(0.0))

    def __attrs_post_init__(self):
        if self.enabled:
            if self.lid_base_mode == 'rheological' and self.activation_energy <= 0:
                raise ValueError(
                    f'Invalid combination: lid_base_mode={self.lid_base_mode!r} requires non-zero '
                    f'activation_energy, but activation_energy={self.activation_energy}'
                )


@attrs.define
class MixedPhaseConfig:
    r"""Mixed-phase (mushy zone) parameters.

    Parameters
    ----------
    latent_heat_of_fusion : float
        Latent heat [J/kg].
    rheological_transition_melt_fraction : float
        Melt fraction at rheological transition.
    rheological_transition_width : float
        Width of the tanh smoothing around the transition.
    solidus : str
        Path to solidus lookup file.
    liquidus : str
        Path to liquidus lookup file.
    phase : str
        Active phase mode: 'solid', 'liquid', 'mixed', or 'composite'.
    phase_transition_width : float
        Width of smoothing at phase boundaries.
    grain_size : float
        Grain size [m] for permeability calculations.
    separation_viscosity : str
        Drag viscosity source for gravitational separation:
        ``'melt'`` (single-phase liquid viscosity, matches SPIDER's
        ``GetGravitationalHeatFlux``) or ``'mixture'`` (the
        rheological-transition-blended bulk viscosity). Default
        ``'melt'``.
    cp_blend : str
        Mushy-zone Cp blending mode: ``'latent'`` (SPIDER-parity,
        latent-heat-augmented) or ``'linear'`` (linear blend of
        pure-phase Cp without the latent term). Default ``'latent'``.
    matprop_smooth_width : float
        Half-width of the tanh used to smooth phase-dependent
        material properties around the rheological transition.
        Default 0.0 reproduces SPIDER's convention of no smoothing
        on the property side; 0.01 is the typical JAX setting.
    const_properties : bool
        Replace the EOS-tabulated $(\rho, c_p, \alpha, k)$ and
        $\log_{10}\eta$ with the seven constant-property values
        below. Mirrors SPIDER's ``-use_const_properties``. Default
        False (use the EOS tables).
    const_rho : float
        Constant density [kg/m^3] when ``const_properties`` is True.
    const_Cp : float
        Constant heat capacity [J/(kg K)] when ``const_properties``
        is True.
    const_alpha : float
        Constant thermal expansivity [1/K] when ``const_properties``
        is True.
    const_cond : float
        Constant thermal conductivity [W/(m K)] when
        ``const_properties`` is True.
    const_log10visc : float
        Constant $\log_{10}$ dynamic viscosity [log10(Pa s)] when
        ``const_properties`` is True.
    const_T_ref : float
        Reference temperature [K] for the constant-properties EOS,
        used to anchor $T(P, S)$ on the analytic isentrope.
    const_S_ref : float
        Reference entropy [J/(kg K)] for the constant-properties
        EOS, paired with ``const_T_ref``.
    """

    latent_heat_of_fusion: float
    rheological_transition_melt_fraction: float
    rheological_transition_width: float
    solidus: str
    liquidus: str
    phase: str
    phase_transition_width: float
    grain_size: float
    separation_viscosity: str = attrs.field(
        default=SEPARATION_VISCOSITY_DEFAULT,
        validator=attrs.validators.in_(SEPARATION_VISCOSITY_MODES),
    )
    cp_blend: str = 'latent'
    matprop_smooth_width: float = 0.0
    const_properties: bool = False
    const_rho: float = 4000.0
    const_Cp: float = 1000.0
    const_alpha: float = 1e-5
    const_cond: float = 4.0
    const_log10visc: float = 2.0
    const_T_ref: float = 3500.0
    const_S_ref: float = 3000.0
    enabled: bool = False
    activation_energy: float = attrs.field(default=300e3, validator=attrs.validators.ge(0.0))
    activation_volume: float = attrs.field(default=5e-6, validator=attrs.validators.ge(0.0))
    yield_stress_c: float = attrs.field(default=50e6, validator=attrs.validators.ge(0.0))
    yield_stress_mu: float = attrs.field(default=0.6, validator=attrs.validators.ge(0.0))
    stress_closure_mode: str = attrs.field(
        default=STRESS_CLOSURE_DEFAULT,
        validator=attrs.validators.in_(STRESS_CLOSURE_MODES),
    )
    arrhenius_t_ref: float = attrs.field(default=1600.0, validator=attrs.validators.gt(0.0))
    yield_stress_max: float = attrs.field(default=500.0e6, validator=attrs.validators.ge(0.0))
    viscosity_max_log10: float = attrs.field(default=40.0, validator=attrs.validators.gt(0.0))
    lid_base_mode: str = attrs.field(
        default=LID_BASE_DEFAULT,
        validator=attrs.validators.in_(LID_BASE_MODES),
    )
    lid_base_temperature: float = attrs.field(
        default=1400.0, validator=attrs.validators.gt(0.0)
    )
    lid_contrast_coeff: float = attrs.field(default=2.2, validator=attrs.validators.gt(0.0))

    def __attrs_post_init__(self):
        if self.enabled:
            if self.lid_base_mode == 'rheological' and self.activation_energy <= 0:
                raise ValueError(
                    f'Invalid combination: lid_base_mode={self.lid_base_mode!r} requires non-zero '
                    f'activation_energy, but activation_energy={self.activation_energy}'
                )
