"""Phase property configuration."""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

import attrs

from aragog.rheology import SolidRheologyParams

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


@attrs.define(init=False)
class PhaseConfig:
    """Single-phase (solid or liquid) material properties.

    Each property can be a float (constant value) or a str (path to
    a lookup table file). Rheology parameters are owned by ``SolidRheologyParams``.
    """

    density: float | str
    heat_capacity: float | str
    melt_fraction: float
    thermal_conductivity: float | str
    thermal_expansivity: float | str
    viscosity: float | str
    entropy: float | str = ''
    rheology: SolidRheologyParams = attrs.field(factory=SolidRheologyParams)

    def __init__(
        self,
        density: float | str,
        heat_capacity: float | str,
        melt_fraction: float,
        thermal_conductivity: float | str,
        thermal_expansivity: float | str,
        viscosity: float | str,
        entropy: float | str = '',
        rheology: SolidRheologyParams | None = None,
        **flat_rheo: Any,
    ):
        self.density = density
        self.heat_capacity = heat_capacity
        self.melt_fraction = melt_fraction
        self.thermal_conductivity = thermal_conductivity
        self.thermal_expansivity = thermal_expansivity
        self.viscosity = viscosity
        self.entropy = entropy
        if rheology is not None:
            if flat_rheo:
                params_dict = {
                    f.name: getattr(rheology, f.name) for f in fields(SolidRheologyParams)
                }
                params_dict.update(flat_rheo)
                self.rheology = SolidRheologyParams(**params_dict)
            else:
                self.rheology = rheology
        elif flat_rheo:
            self.rheology = SolidRheologyParams(**flat_rheo)
        else:
            self.rheology = SolidRheologyParams()

    @property
    def enabled(self) -> bool:
        return self.rheology.enabled

    @property
    def activation_energy(self) -> float:
        return self.rheology.activation_energy

    @property
    def activation_volume(self) -> float:
        return self.rheology.activation_volume

    @property
    def activation_volume_decay_pressure(self) -> float:
        return self.rheology.activation_volume_decay_pressure

    @property
    def arrhenius_t_ref(self) -> float:
        return self.rheology.arrhenius_t_ref

    @property
    def viscosity_max_log10(self) -> float:
        return self.rheology.viscosity_max_log10

    @property
    def water_prefactor(self) -> float:
        return self.rheology.water_prefactor

    @property
    def yield_stress_c(self) -> float:
        return self.rheology.yield_stress_c

    @property
    def yield_stress_mu(self) -> float:
        return self.rheology.yield_stress_mu

    @property
    def yield_stress_max(self) -> float:
        return self.rheology.yield_stress_max

    @property
    def yield_switch_width(self) -> float:
        return self.rheology.yield_switch_width

    @property
    def stress_closure_mode(self) -> str:
        return self.rheology.stress_closure_mode

    @property
    def interior_flux_fraction(self) -> float:
        return self.rheology.interior_flux_fraction

    @property
    def lid_base_mode(self) -> str:
        return self.rheology.lid_base_mode

    @property
    def lid_base_temperature(self) -> float:
        return self.rheology.lid_base_temperature

    @property
    def lid_contrast_coeff(self) -> float:
        return self.rheology.lid_contrast_coeff

    @property
    def lid_mask_width_cells(self) -> float:
        return self.rheology.lid_mask_width_cells

    @property
    def phi_visc_single(self) -> float:
        return self.rheology.phi_visc_single


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
